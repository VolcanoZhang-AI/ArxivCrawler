import arxiv
import datetime
import pandas as pd
from collections import Counter
import re
import os
import jieba
import jieba.analyse
from pyecharts import options as opts
from pyecharts.charts import WordCloud, Bar
from pyecharts.globals import ThemeType
import time
import logging
import sys
from requests.exceptions import RequestException
import requests
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from openai import OpenAI
import random



# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("arxiv_crawler.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("ArxivCrawler")

# 加载配置文件和初始化全局变量
use_qwen = False
qwen_client = None

# 加载配置文件
CONFIG_PATH = "config.json"
try:
    with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
        config = json.load(f)
    qwen_api_key = config.get("qwen_api_key")
    qwen_api_base = config.get("qwen_api_base", "https://dashscope.aliyuncs.com/compatible-mode/v1")
    qwen_model_name = config.get("model_name", "qwen-max")
    qwen_max_tokens = config.get("max_tokens", 1000)
    qwen_temperature = config.get("temperature", 0.7)
    
    # 检查配置文件中是否有代理设置
    proxies = config.get("proxies", None)
    if not proxies and "_proxies" in config:
        # 用户可能按照提示将_proxies重命名为proxies，这里做个兼容
        logger.info("检测到_proxies配置，尝试使用")
        proxies = config.get("_proxies", None)
    
    # 如果有代理配置，设置环境变量
    if proxies and isinstance(proxies, dict):
        logger.info(f"使用配置的代理: {proxies}")
        if "http" in proxies:
            os.environ["http_proxy"] = proxies["http"]
            logger.info(f"设置HTTP代理环境变量: {proxies['http']}")
        if "https" in proxies:
            os.environ["https_proxy"] = proxies["https"]
            logger.info(f"设置HTTPS代理环境变量: {proxies['https']}")
    
    # 检查是否设置了有效的API密钥
    if qwen_api_key and qwen_api_key != "你的API密钥":
        use_qwen = True
        # 初始化OpenAI客户端
        try:
            # 简化初始化，仅使用必要的参数
            qwen_client = OpenAI(
                api_key=qwen_api_key,
                base_url=qwen_api_base
            )
            logger.info("成功初始化Qwen API客户端")
        except Exception as e:
            logger.error(f"初始化Qwen API客户端失败: {e}")
            import traceback
            logger.error(f"错误详情: {traceback.format_exc()}")
            use_qwen = False
    else:
        logger.warning("Qwen API密钥未设置或无效，将使用正则表达式提取论文贡献")
except Exception as e:
    logger.error(f"加载配置文件时出错: {e}")
    import traceback
    logger.error(f"错误详情: {traceback.format_exc()}")
    qwen_api_key = ""
    qwen_api_base = "https://dashscope.aliyuncs.com/compatible-mode/v1" 
    qwen_model_name = "qwen-max"
    qwen_max_tokens = 1000
    qwen_temperature = 0.7

# 加载自定义词典
jieba_dict_path = "llm_dict.txt"
if os.path.exists(jieba_dict_path):
    jieba.load_userdict(jieba_dict_path)
    logger.info(f"已加载自定义词典: {jieba_dict_path}")
else:
    logger.warning(f"自定义词典文件不存在: {jieba_dict_path}")

# 设置关键词和搜索查询
LLM_KEYWORDS = [
    "large language model", "LLM", "GPT", "ChatGPT", "transformer", "generative AI",
    "language model", "foundation model", "Llama", "Claude", "Gemini", "agent", 
    "multimodal", "instruction tuning", "prompt", "fine-tuning", "alignment"
]

# 目标论文类别
TARGET_CATEGORIES = [
    "cs.CL", "cs.AI", "cs.LG", "cs.CV", "cs.IR", "cs.HC", "cs.NE",
    "stat.ML", "cs.CR", "cs.SE", "cs.DC", "cs.RO"
]

# 创建搜索查询字符串
SEARCH_QUERY = " OR ".join([f'all:"{keyword}"' for keyword in LLM_KEYWORDS])

def get_date_range():
    """获取日期范围，用于过滤论文"""
    today = datetime.datetime.now().date()
    seven_days_ago = today - datetime.timedelta(days=7)
    
    logger.info(f"计算日期范围: {seven_days_ago} 到 {today}")
    
    return (seven_days_ago, today)

def fetch_arxiv_papers(date_range):
    """获取arXiv论文，包含重试逻辑"""
    start_date, end_date = date_range
    
    logger.info(f"正在从arXiv获取论文 (日期范围: {start_date.strftime('%Y-%m-%d')} 到 {end_date.strftime('%Y-%m-%d')})")
    
    max_retries = 3
    retry_delay = 5  # 秒
    
    for attempt in range(max_retries):
        try:
            # 构建查询
            search_query = construct_search_query()
            
            # 设置日期过滤
            # arXiv API 使用的是YYYYMMDDHHMMSS格式
            # 注意：arXiv的日期是基于美国东部时间 (ET)
            start_date_str = start_date.strftime('%Y%m%d') + '000000'
            end_date_str = end_date.strftime('%Y%m%d') + '235959'
            
            date_filter = f"submittedDate:[{start_date_str} TO {end_date_str}]"
            
            logger.info(f"arXiv查询: {search_query}, 日期过滤: {date_filter}")
            
            # 使用arXiv API获取论文
            search = arxiv.Search(
                query=search_query,
                max_results=1000,  # 设置一个较大的值以获取足够的结果
                sort_by=arxiv.SortCriterion.SubmittedDate,
                sort_order=arxiv.SortOrder.Descending
            )
            
            # 将结果转换为列表以便进一步处理
            papers = list(search.results())
            
            if not papers:
                logger.warning("未找到符合条件的论文")
            else:
                logger.info(f"成功获取 {len(papers)} 篇论文")
                
                # 按照更新日期过滤
                filtered_papers = []
                for paper in papers:
                    paper_date = paper.updated.date()
                    if start_date <= paper_date <= end_date:
                        filtered_papers.append(paper)
                
                logger.info(f"过滤后剩余 {len(filtered_papers)} 篇论文 (日期范围: {start_date} 到 {end_date})")
                return filtered_papers
            
            return []
            
        except Exception as e:
            logger.error(f"获取论文时出错 (尝试 {attempt+1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                logger.info(f"等待 {retry_delay} 秒后重试...")
                time.sleep(retry_delay)
                # 每次重试增加延迟时间
                retry_delay *= 2
            else:
                logger.error("达到最大重试次数，无法获取论文")
                return []
    
    return []

def extract_paper_info(paper):
    """提取论文的关键信息"""
    try:
        return {
            "title": paper.title,
            "authors": ", ".join(author.name for author in paper.authors),
            "summary": paper.summary,
            "published": paper.published.date(),
            "url": paper.entry_id,
            "primary_category": paper.primary_category,
            "pdf_url": paper.pdf_url,  # 添加PDF链接
            "paper_id": paper.get_short_id()  # 论文ID，用于下载
        }
    except Exception as e:
        logger.error(f"提取论文信息时出错: {e}")
        # 返回部分信息，避免整个处理过程失败
        return {
            "title": getattr(paper, "title", "无标题"),
            "authors": "未知作者",
            "summary": getattr(paper, "summary", "无摘要"),
            "published": datetime.datetime.now().date(),
            "url": getattr(paper, "entry_id", "#"),
            "primary_category": getattr(paper, "primary_category", "未分类"),
            "pdf_url": getattr(paper, "pdf_url", "#"),
            "paper_id": "unknown"
        }

def download_pdf(paper_info, output_dir):
    """下载论文PDF"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    paper_id = paper_info["paper_id"]
    pdf_url = paper_info["pdf_url"]
    output_path = os.path.join(output_dir, f"{paper_id}.pdf")
    
    # 如果文件已存在，跳过下载
    if os.path.exists(output_path):
        logger.info(f"PDF已存在，跳过下载: {paper_id}")
        return output_path
    
    try:
        logger.info(f"正在下载论文: {paper_id}")
        response = requests.get(pdf_url, stream=True, timeout=30)
        response.raise_for_status()
        
        with open(output_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        
        logger.info(f"论文下载完成: {paper_id}")
        return output_path
    except Exception as e:
        logger.error(f"下载论文PDF时出错: {paper_id}, 错误: {e}")
        return None

def download_hot_papers_pdfs(hot_topics, max_workers=5):
    """批量下载热点论文PDF"""
    pdf_dir = os.path.join("arxiv_reports", "pdfs")
    if not os.path.exists(pdf_dir):
        os.makedirs(pdf_dir)
    
    downloaded_papers = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_paper = {
            executor.submit(download_pdf, paper, pdf_dir): paper 
            for paper in hot_topics
        }
        
        for future in as_completed(future_to_paper):
            paper = future_to_paper[future]
            try:
                pdf_path = future.result()
                if pdf_path:
                    downloaded_papers.append({
                        "paper_id": paper["paper_id"],
                        "title": paper["title"],
                        "pdf_path": pdf_path
                    })
            except Exception as e:
                logger.error(f"下载论文过程中出错: {paper['paper_id']}, 错误: {e}")
    
    return downloaded_papers

def analyze_papers(papers, already_processed_titles=None):
    """分析论文内容，提取关键信息和热点"""
    if not papers:
        logger.warning("没有论文可供分析")
        return {
            "papers_df": pd.DataFrame(),
            "category_counts": {},
            "keywords": [],
            "hot_topics": []
        }
    
    already_processed_titles = already_processed_titles or set()
    
    # 加载已处理论文记录
    processed_papers = load_processed_papers()
    
    try:
        papers_data = [extract_paper_info(paper) for paper in papers]
        df = pd.DataFrame(papers_data)
        
        # 按类别统计
        category_counts = df['primary_category'].value_counts().to_dict()
        
        # 提取关键词
        all_summaries = " ".join(df['summary'].tolist())
        keywords = jieba.analyse.extract_tags(all_summaries, topK=30, withWeight=True)
        
        # 分析热点主题
        hot_topics = []
        for paper_info in papers_data:
            # 跳过已处理过的论文
            if paper_info['title'] in already_processed_titles:
                logger.info(f"跳过已分析过的论文: {paper_info['title']}")
                continue
                
            summary = paper_info['summary'].lower()
            
            # 检查是否包含LLM相关关键词
            if any(keyword.lower() in summary for keyword in LLM_KEYWORDS):
                # 使用正则表达式提取英文内容
                contributions = extract_contributions(summary)
                innovations = extract_innovations(summary)
                problems = extract_problems(summary)
                
                # 使用Qwen-Max进行中文总结
                if use_qwen:
                    paper_id = paper_info['paper_id']
                    
                    # 每次处理一种类型，这样可以单独保存
                    contributions_zh = summarize_with_qwen(paper_info['title'], summary, "contributions", paper_id, processed_papers)
                    save_processed_papers(processed_papers)  # 每次处理后保存，避免中断导致的重复处理
                    
                    problems_zh = summarize_with_qwen(paper_info['title'], summary, "problems", paper_id, processed_papers)
                    save_processed_papers(processed_papers)
                    
                    innovations_zh = summarize_with_qwen(paper_info['title'], summary, "innovations", paper_id, processed_papers)
                    save_processed_papers(processed_papers)
                else:
                    contributions_zh = []
                    problems_zh = []
                    innovations_zh = []
                
                hot_topics.append({
                    **paper_info,
                    "contributions": contributions,
                    "contributions_zh": contributions_zh,
                    "innovations": innovations,
                    "innovations_zh": innovations_zh,
                    "problems": problems,
                    "problems_zh": problems_zh
                })
                
                # 添加到已处理集合
                already_processed_titles.add(paper_info['title'])
        
        # 按相关性排序热点论文
        for paper in hot_topics:
            relevance_score = sum(1 for keyword in LLM_KEYWORDS if keyword.lower() in paper['summary'].lower())
            paper['relevance_score'] = relevance_score
        
        hot_topics.sort(key=lambda x: x['relevance_score'], reverse=True)
        
        return {
            "papers_df": df,
            "category_counts": category_counts,
            "keywords": keywords,
            "hot_topics": hot_topics[:10]  # 取前10个最相关的论文
        }
    except Exception as e:
        logger.error(f"分析论文时出错: {e}")
        return {
            "papers_df": pd.DataFrame(),
            "category_counts": {},
            "keywords": [],
            "hot_topics": []
        }

def summarize_with_qwen(title, summary, summary_type="contributions", paper_id=None, processed_papers=None):
    """使用Qwen-Max模型API对论文进行中文总结"""
    global use_qwen, qwen_client
    
    if not use_qwen or not qwen_client:
        logger.info("Qwen API未启用或客户端未初始化，跳过中文总结")
        return []
    
    # 检查是否已经处理过该论文
    if processed_papers is not None and paper_id is not None:
        cache_key = f"{summary_type}_zh"
        if is_paper_processed(paper_id, title, cache_key, processed_papers):
            # 如果已经处理过，从记录中获取结果
            if paper_id in processed_papers and "results" in processed_papers[paper_id]:
                cached_results = processed_papers[paper_id]["results"].get(cache_key, [])
                if cached_results:
                    logger.info(f"使用缓存的{summary_type}中文总结，避免重复API调用")
                    return cached_results
    
    try:
        # 限制摘要长度，防止Token过长
        summary = summary[:3000] if len(summary) > 3000 else summary
        
        if summary_type == "contributions":
            prompt = f"""
            请用中文总结这篇英文学术论文的主要贡献点，以条目形式列出不超过3点。论文标题和摘要如下：
            
            标题：{title}
            摘要：{summary}
            
            请直接输出总结，每条总结以'- '开头，不需要包含标题或其他内容。
            """
        elif summary_type == "innovations":
            prompt = f"""
            请用中文总结这篇英文学术论文的主要创新点，以条目形式列出不超过3点。论文标题和摘要如下：
            
            标题：{title}
            摘要：{summary}
            
            请直接输出总结，每条总结以'- '开头，不需要包含标题或其他内容。
            """
        elif summary_type == "problems":
            prompt = f"""
            请用中文总结这篇英文学术论文解决的主要问题，以条目形式列出不超过3点。论文标题和摘要如下：
            
            标题：{title}
            摘要：{summary}
            
            请直接输出总结，每条总结以'- '开头，不需要包含标题或其他内容。
            """
        else:
            return []
        
        logger.info(f"正在使用Qwen API进行'{summary_type}'类型的中文总结")
        
        # 调用Qwen-Max模型API
        try:
            response = qwen_client.chat.completions.create(
                model=qwen_model_name,
                messages=[
                    {"role": "system", "content": "你是一个专业的学术论文分析助手，善于提取和总结学术论文的关键信息。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=qwen_temperature,
                max_tokens=qwen_max_tokens
            )
            
            try:
                # 标准OpenAI兼容API响应格式
                content = response.choices[0].message.content
            except (AttributeError, IndexError) as e:
                logger.warning(f"使用标准OpenAI格式解析响应失败: {e}，尝试其他响应格式")
                
                # 尝试阿里云DashScope API响应格式
                if hasattr(response, 'output') and hasattr(response.output, 'text'):
                    content = response.output.text
                    logger.info("成功使用DashScope API响应格式获取内容")
                elif hasattr(response, 'result'):
                    content = response.result
                    logger.info("成功使用DashScope旧版API响应格式获取内容")
                elif isinstance(response, dict):
                    # 检查是否为字典格式的响应
                    if 'output' in response and 'text' in response['output']:
                        content = response['output']['text']
                        logger.info("成功使用DashScope字典响应格式获取内容")
                    elif 'choices' in response and len(response['choices']) > 0:
                        if 'message' in response['choices'][0] and 'content' in response['choices'][0]['message']:
                            content = response['choices'][0]['message']['content']
                            logger.info("成功使用字典格式的OpenAI兼容响应格式获取内容")
                    else:
                        logger.error(f"未知的响应字典格式: {response}")
                        return []
                else:
                    logger.error(f"无法解析的响应格式: {type(response)}")
                    return []
            
            # 解析条目
            items = []
            for line in content.strip().split('\n'):
                line = line.strip()
                if line.startswith('- '):
                    items.append(line[2:])
                elif line.startswith('•'):
                    items.append(line[1:].strip())
                elif line and len(items) < 3 and len(line) > 10:
                    # 如果没有明确的条目标记但有内容，也尝试添加
                    items.append(line)
            
            # 限制返回条数
            items = items[:3]
            
            if items:
                logger.info(f"成功获取中文总结，共{len(items)}条")
                
                # 如果提供了paper_id和processed_papers，将结果保存到处理记录中
                if processed_papers is not None and paper_id is not None:
                    cache_key = f"{summary_type}_zh"
                    # 标记为已处理
                    processed_papers = mark_paper_as_processed(paper_id, title, cache_key, processed_papers)
                    # 保存结果
                    if "results" not in processed_papers[paper_id]:
                        processed_papers[paper_id]["results"] = {}
                    processed_papers[paper_id]["results"][cache_key] = items
                    
            else:
                logger.warning(f"未能从API响应中解析出有效条目。原始响应: {content[:200]}...")
            
            return items
        except Exception as inner_e:
            logger.error(f"调用或解析API响应时出错: {inner_e}")
            import traceback
            logger.error(f"错误详情: {traceback.format_exc()}")
            return []
    
    except Exception as e:
        logger.error(f"使用Qwen模型总结论文时出错: {e}")
        # 添加更详细的错误信息用于调试
        import traceback
        logger.error(f"错误详情: {traceback.format_exc()}")
        return []

def extract_contributions(text):
    """从摘要中提取论文贡献"""
    contribution_patterns = [
        r"contribute[sd]? (to|by) (.+?)(\.|\n)",
        r"contribution[s]? (?:is|are|include[s]?) (.+?)(\.|\n)",
        r"we propose (.+?)(\.|\n)",
        r"we present (.+?)(\.|\n)",
        r"we introduce (.+?)(\.|\n)",
        r"our (approach|method|framework|system) (.+?)(\.|\n)"
    ]
    
    contributions = []
    for pattern in contribution_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            for match in matches:
                if isinstance(match, tuple):
                    # 根据模式不同，有效内容可能在不同位置
                    for part in match:
                        if len(part) > 15:  # 简单的启发式过滤短文本
                            contributions.append(part.strip())
                else:
                    contributions.append(match.strip())
    
    # 如果没有找到明确的贡献陈述，则提取可能的贡献句子
    if not contributions:
        sentences = re.split(r'(?<=[.!?])\s+', text)
        for sentence in sentences:
            if any(word in sentence.lower() for word in ["novel", "new", "propose", "present", "introduce", "approach", "method", "framework"]):
                contributions.append(sentence)
    
    return list(set(contributions))[:3]  # 去重并限制数量

def extract_innovations(text):
    """从摘要中提取创新点"""
    innovation_patterns = [
        r"novel (.+?)(\.|\n)",
        r"new (.+?)(\.|\n)",
        r"innovate[sd]? (.+?)(\.|\n)",
        r"innovation[s]? (.+?)(\.|\n)",
        r"first (.+?)(\.|\n)",
        r"outperform[s]? (.+?)(\.|\n)",
        r"state-of-the-art (.+?)(\.|\n)",
        r"sota (.+?)(\.|\n)"
    ]
    
    innovations = []
    for pattern in innovation_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            for match in matches:
                if isinstance(match, tuple):
                    for part in match:
                        if len(part) > 15:
                            innovations.append(part.strip())
                else:
                    innovations.append(match.strip())
    
    # 如果没有找到明确的创新陈述，则提取可能的创新句子
    if not innovations:
        sentences = re.split(r'(?<=[.!?])\s+', text)
        for sentence in sentences:
            if any(word in sentence.lower() for word in ["improve", "better", "advance", "enhance", "superior", "breakthrough", "unprecedented"]):
                innovations.append(sentence)
    
    return list(set(innovations))[:3]  # 去重并限制数量

def extract_problems(text):
    """从摘要中提取论文解决的问题"""
    problem_patterns = [
        r"(problem[s]? .+?)(\.|\n)",
        r"(challenge[s]? .+?)(\.|\n)",
        r"(address .+?)(\.|\n)",
        r"(solve[sd]? .+?)(\.|\n)",
        r"(overcome .+?)(\.|\n)",
        r"(limitation[s]? .+?)(\.|\n)"
    ]
    
    problems = []
    for pattern in problem_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            for match in matches:
                if isinstance(match, tuple):
                    for part in match:
                        if len(part) > 15:
                            problems.append(part.strip())
                else:
                    problems.append(match.strip())
    
    # 如果没有找到明确的问题陈述，则尝试从摘要的前几句话中提取
    if not problems:
        sentences = re.split(r'(?<=[.!?])\s+', text)
        problems = sentences[:2]  # 通常摘要的前几句会描述问题
    
    return list(set(problems))[:3]  # 去重并限制数量

def generate_report(papers_data, output_dir):
    """按日期生成分析报告"""
    if not papers_data:
        logger.warning("没有数据可供生成报告")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 按日期分组的论文
    papers_by_date = papers_data
    
    # 记录MD文件列表，用于生成索引
    md_files = []
    
    # 为每个日期生成单独的MD文件
    for date, papers in papers_by_date.items():
        if not papers:
            logger.info(f"日期 {date} 没有找到相关论文")
            continue
            
        paper_count = len(papers)
        logger.info(f"日期 {date} 找到 {paper_count} 篇相关论文")
        
        # 创建该日期的MD文件
        date_formatted = date.replace('-', '')  # 移除日期中的连字符，用于文件名
        md_filename = f"llm_papers_{date_formatted}.md"
        md_path = os.path.join(output_dir, md_filename)
        
        # 检查文件是否已存在，如果存在且有内容，则跳过
        if os.path.exists(md_path) and os.path.getsize(md_path) > 0:
            logger.info(f"日期 {date} 的报告文件已存在，跳过生成")
            md_files.append((date, md_filename))
            continue
        
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(f"# LLM相关论文日报 ({date})\n\n")
            f.write(f"## 更新日期：{date}\n\n")
            f.write(f"本日在arXiv上共发现 {paper_count} 篇与LLM相关的论文\n\n")
            
            # 为每篇论文添加详细信息
            for i, paper in enumerate(papers, 1):
                f.write(f"### {i}. {paper['title']}\n\n")
                f.write(f"**论文ID**: [{paper['paper_id']}](https://arxiv.org/abs/{paper['paper_id']})\n\n")
                f.write(f"**发布日期**: {paper['published']}\n\n")
                f.write(f"**更新日期**: {paper['updated']}\n\n")
                f.write(f"**作者**: {paper['authors']}\n\n")
                f.write(f"**类别**: {paper['primary_category']}\n\n")
                f.write("**摘要**:\n")
                f.write(f"{paper['summary']}\n\n")
                
                # 添加中英文分析
                if 'contributions' in paper and paper['contributions']:
                    f.write("**主要贡献**:\n")
                    for item in paper['contributions']:
                        f.write(f"- {item}\n")
                    f.write("\n")
                
                if 'contributions_zh' in paper and paper['contributions_zh']:
                    f.write("**主要贡献（中文）**:\n")
                    for item in paper['contributions_zh']:
                        f.write(f"- {item}\n")
                    f.write("\n")
                
                if 'innovations' in paper and paper['innovations']:
                    f.write("**创新点**:\n")
                    for item in paper['innovations']:
                        f.write(f"- {item}\n")
                    f.write("\n")
                
                if 'innovations_zh' in paper and paper['innovations_zh']:
                    f.write("**创新点（中文）**:\n")
                    for item in paper['innovations_zh']:
                        f.write(f"- {item}\n")
                    f.write("\n")
                
                if 'problems' in paper and paper['problems']:
                    f.write("**解决的问题**:\n")
                    for item in paper['problems']:
                        f.write(f"- {item}\n")
                    f.write("\n")
                
                if 'problems_zh' in paper and paper['problems_zh']:
                    f.write("**解决的问题（中文）**:\n")
                    for item in paper['problems_zh']:
                        f.write(f"- {item}\n")
                    f.write("\n")
                
                f.write("---\n\n")
        
        logger.info(f"为日期 {date} 生成了报告文件: {md_path}")
        md_files.append((date, md_filename))
    
    # 生成索引文件
    if md_files:
        index_path = os.path.join(output_dir, "index.md")
        with open(index_path, "w", encoding="utf-8") as f:
            f.write("# LLM论文分析报告索引\n\n")
            f.write("## 按日期查看报告\n\n")
            
            # 按日期倒序排列
            md_files.sort(key=lambda x: x[0], reverse=True)
            
            for date, filename in md_files:
                f.write(f"- [{date}]({filename})\n")
        
        logger.info(f"已生成索引文件: {index_path}")
    
    logger.info(f"报告生成完成，已保存到 {output_dir} 目录")
    return md_files

def generate_visualizations(five_days_analysis, report_dir, papers_data=None):
    """生成可视化图表"""
    if not five_days_analysis.get("hot_topics") and not papers_data:
        logger.warning("没有足够的数据生成可视化")
        return
    
    try:
        # 生成类别分布柱状图
        categories = five_days_analysis.get("category_counts", {})
        if categories:
            categories_sorted = dict(sorted(categories.items(), key=lambda x: x[1], reverse=True)[:10])
            
            bar = (
                Bar(init_opts=opts.InitOpts(theme=ThemeType.LIGHT))
                .add_xaxis(list(categories_sorted.keys()))
                .add_yaxis("论文数量", list(categories_sorted.values()))
                .set_global_opts(
                    title_opts=opts.TitleOpts(title="LLM相关论文类别分布"),
                    xaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(rotate=45)),
                    datazoom_opts=[opts.DataZoomOpts()],
                )
            )
            bar.render(os.path.join(report_dir, "category_distribution.html"))
            logger.info("类别分布图已生成")
        
        # 生成每日论文数量分布图
        if papers_data and "by_date" in papers_data and "sorted_dates" in papers_data:
            dates = []
            paper_counts = []
            
            # 日期从早到晚排序
            for date_str in sorted(papers_data["sorted_dates"]):
                # 将日期格式转换为更友好的显示格式
                date_obj = datetime.datetime.strptime(date_str, "%Y-%m-%d")
                display_date = date_obj.strftime("%m-%d")
                
                dates.append(display_date)
                paper_counts.append(len(papers_data["by_date"][date_str]))
            
            if dates and paper_counts:
                daily_bar = (
                    Bar(init_opts=opts.InitOpts(theme=ThemeType.LIGHT))
                    .add_xaxis(dates)
                    .add_yaxis("论文数量", paper_counts)
                    .set_global_opts(
                        title_opts=opts.TitleOpts(title="每日LLM相关论文数量"),
                        xaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(rotate=30)),
                        toolbox_opts=opts.ToolboxOpts(is_show=True),
                    )
                )
                daily_bar.render(os.path.join(report_dir, "daily_papers_count.html"))
                logger.info("每日论文数量分布图已生成")
        
        # 生成关键词词云
        word_cloud_data = [(k, v) for k, v in five_days_analysis.get("keywords", [])]
        if word_cloud_data:
            word_cloud = (
                WordCloud(init_opts=opts.InitOpts(theme=ThemeType.LIGHT))
                .add("", word_cloud_data, word_size_range=[20, 100])
                .set_global_opts(title_opts=opts.TitleOpts(title="LLM研究关键词热度"))
            )
            word_cloud.render(os.path.join(report_dir, "keywords_cloud.html"))
            logger.info("关键词词云图已生成")
    except Exception as e:
        logger.error(f"生成可视化图表时出错: {e}")
        import traceback
        logger.error(f"错误详情: {traceback.format_exc()}")

def load_processed_papers():
    """加载已处理过的论文记录"""
    processed_file = "processed_papers.json"
    if os.path.exists(processed_file):
        try:
            with open(processed_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"加载已处理论文记录失败: {e}")
            return {}
    return {}

def save_processed_papers(processed_papers):
    """保存已处理的论文记录"""
    processed_file = "processed_papers.json"
    try:
        with open(processed_file, 'w', encoding='utf-8') as f:
            json.dump(processed_papers, f, ensure_ascii=False, indent=2)
        logger.info(f"已保存处理记录到 {processed_file}")
    except Exception as e:
        logger.error(f"保存处理记录失败: {e}")

def is_paper_processed(paper_id, paper_title, process_type, processed_papers):
    """检查论文是否已处理过特定类型的总结"""
    # 通过ID和标题双重检查，避免ID变化但内容相同的情况
    if paper_id in processed_papers:
        if process_type in processed_papers[paper_id].get('process_types', []):
            logger.info(f"论文 {paper_id} 已经处理过 {process_type} 类型的总结")
            return True
    
    # 通过标题检查
    for pid, data in processed_papers.items():
        if data.get('title') == paper_title and process_type in data.get('process_types', []):
            logger.info(f"论文标题 '{paper_title}' 已经处理过 {process_type} 类型的总结")
            return True
    
    return False

def mark_paper_as_processed(paper_id, paper_title, process_type, processed_papers):
    """标记论文为已处理"""
    if paper_id not in processed_papers:
        processed_papers[paper_id] = {
            'title': paper_title,
            'process_types': [],
            'processed_date': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
    
    if process_type not in processed_papers[paper_id]['process_types']:
        processed_papers[paper_id]['process_types'].append(process_type)
        processed_papers[paper_id]['last_updated'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    return processed_papers

def group_papers_by_date(papers):
    """将论文按更新日期分组"""
    today = datetime.datetime.now().date()
    five_days_ago = today - datetime.timedelta(days=7)
    
    five_days = []
    last_day = []
    
    # 按日期分组的字典
    by_date = {}
    sorted_dates = []
    
    for paper in papers:
        updated_date = paper.updated.date()
        updated_str = updated_date.strftime("%Y-%m-%d")
        
        # 添加到日期分组
        if updated_str not in by_date:
            by_date[updated_str] = []
            sorted_dates.append(updated_str)
        
        by_date[updated_str].append(paper)
        
        # 检查是否在最近5天和最近1天内
        if five_days_ago <= updated_date <= today:
            five_days.append(paper)
            
            if updated_date == today:
                last_day.append(paper)
    
    # 对日期进行排序，最新的日期在前
    sorted_dates.sort(reverse=True)
    
    return {
        "five_days": five_days,
        "last_day": last_day,
        "by_date": by_date,
        "sorted_dates": sorted_dates
    }

def initialize_qwen_client():
    """初始化Qwen API客户端"""
    global qwen_client, use_qwen
    
    logger.info("ArxivCrawler 开始运行...")
    
    # 记录环境信息，便于调试
    try:
        import openai
        logger.info(f"OpenAI包版本: {openai.__version__}")
        
        # 记录代理环境变量
        http_proxy = os.environ.get("http_proxy", "未设置")
        https_proxy = os.environ.get("https_proxy", "未设置")
        logger.info(f"HTTP代理环境变量: {http_proxy}")
        logger.info(f"HTTPS代理环境变量: {https_proxy}")
    except Exception as e:
        logger.error(f"获取环境信息时出错: {e}")
    
    # 检查Qwen API状态
    if use_qwen and qwen_client:
        logger.info("已启用Qwen API进行论文中文总结")
        # 尝试一个简单的请求测试API连接
        try:
            test_response = qwen_client.chat.completions.create(
                model=qwen_model_name,
                messages=[
                    {"role": "system", "content": "你是一个助手。"},
                    {"role": "user", "content": "你好"}
                ],
                max_tokens=10
            )
            logger.info(f"Qwen API连接测试成功: {test_response.choices[0].message.content}")
        except Exception as e:
            logger.warning(f"Qwen API连接测试失败，但将继续尝试使用: {e}")
    else:
        logger.warning("未启用Qwen API或初始化失败，将使用正则表达式提取论文内容")
        
def configure_logging():
    """配置日志系统"""
    # 创建日志目录
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # 获取当前日期和时间作为日志文件名
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"arxiv_crawler_{current_time}.log")
    
    # 配置日志格式和处理器
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # 文件处理器
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter(log_format))
    
    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(log_format))
    
    # 设置根日志记录器
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    # 清除现有处理器
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # 添加处理器
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    logger.info(f"日志已配置，保存位置: {log_file}")

def main():
    """主函数"""
    try:
        # 设置日志
        configure_logging()
        
        # 初始化Qwen客户端
        initialize_qwen_client()
        
        # 获取日期范围
        date_range = get_date_range()
        start_date, end_date = date_range
        logger.info(f"获取日期范围: {start_date.strftime('%Y-%m-%d')} 到 {end_date.strftime('%Y-%m-%d')}")
        
        # 获取论文
        papers = fetch_arxiv_papers(date_range)
        if not papers:
            logger.warning("未找到符合条件的论文")
            return
        
        logger.info(f"从arXiv获取到 {len(papers)} 篇论文")
        
        # 按日期分组
        papers_data = group_papers_by_date(papers)
        
        # 记录已处理的论文标题，避免重复分析
        processed_titles = set()
        
        # 加载已处理论文记录
        processed_papers = load_processed_papers()
        logger.info(f"加载已处理论文记录，共 {len(processed_papers)} 篇")
        
        # 为每个日期分析论文
        for date_str in papers_data["sorted_dates"]:
            papers_on_date = papers_data["by_date"][date_str]
            logger.info(f"分析 {date_str} 日期的 {len(papers_on_date)} 篇论文")
            
            # 检查该日期的报告文件是否已经存在
            date_formatted = date_str.replace('-', '')
            md_path = os.path.join("arxiv_reports", f"llm_papers_{date_formatted}.md")
            
            if os.path.exists(md_path) and os.path.getsize(md_path) > 0:
                logger.info(f"日期 {date_str} 的报告文件已存在，跳过分析")
                continue
            
            # 过滤掉已处理过的论文
            filtered_papers = []
            for paper in papers_on_date:
                paper_id = paper.get_short_id()
                title = paper.title
                
                # 如果论文已经在过去被处理过，跳过
                if is_paper_fully_processed(paper_id, title, processed_papers):
                    logger.info(f"跳过已处理过的论文: {title}")
                    continue
                    
                filtered_papers.append(paper)
            
            if not filtered_papers:
                logger.info(f"日期 {date_str} 没有新论文需要处理")
                continue
                
            logger.info(f"日期 {date_str} 有 {len(filtered_papers)} 篇新论文需要处理")
            
            # 分析这一天的论文并提取LLM相关内容
            papers_data["by_date"][date_str] = analyze_papers(filtered_papers, processed_titles).get("hot_topics", [])
            logger.info(f"{date_str} 日期找到 {len(papers_data['by_date'][date_str])} 篇LLM相关论文")
        
        # 下载PDF
        downloaded_papers = []
        if len(papers_data["five_days"]) > 0:
            try:
                downloaded_papers = download_pdfs(papers_data["five_days"], 10)  # 最多下载10篇PDF
            except Exception as e:
                logger.error(f"下载PDF时出错: {e}")
        
        # 生成报告
        report_files = generate_report(papers_data["by_date"], "arxiv_reports")
        
        if report_files:
            # 生成可视化
            report_dir = "arxiv_reports"
            
            # 使用五天内所有论文的数据生成可视化
            five_days_analysis = analyze_papers(papers_data["five_days"])
            generate_visualizations(five_days_analysis, report_dir, papers_data)
            
            logger.info(f"论文分析完成，报告已保存至: {report_dir}")
        else:
            logger.error("报告生成失败")
        
    except Exception as e:
        logger.error(f"程序运行出错: {e}")
        traceback.print_exc()

def construct_search_query():
    """构建arXiv搜索查询语句"""
    # 关键词部分
    keywords_query = " OR ".join([f'"{keyword}"' for keyword in LLM_KEYWORDS])
    
    # 类别过滤
    category_filter = " OR ".join([f"cat:{category}" for category in TARGET_CATEGORIES])
    
    # 组合查询
    full_query = f"({keywords_query}) AND ({category_filter})"
    
    return full_query

def download_pdfs(papers, max_papers=10):
    """下载论文PDF，最多下载max_papers篇"""
    if not papers:
        logger.warning("没有论文可供下载")
        return []
    
    # 创建保存目录
    pdf_dir = "arxiv_pdfs"
    if not os.path.exists(pdf_dir):
        os.makedirs(pdf_dir)
    
    downloaded_papers = []
    count = 0
    
    for paper in papers:
        if count >= max_papers:
            break
            
        try:
            # 检查paper是否为字典或arxiv.Result对象
            if isinstance(paper, dict):
                paper_id = paper["paper_id"]
                title = paper["title"]
            else:
                # 假设它是arxiv.Result对象
                paper_id = paper.get_short_id()
                title = paper.title
                
            pdf_url = f"https://arxiv.org/pdf/{paper_id}.pdf"
            
            # 检查是否已下载
            pdf_path = os.path.join(pdf_dir, f"{paper_id}.pdf")
            if os.path.exists(pdf_path):
                logger.info(f"PDF已存在: {pdf_path}")
                
                downloaded_papers.append({
                    "paper_id": paper_id,
                    "title": title,
                    "pdf_path": pdf_path
                })
                
                count += 1
                continue
            
            # 下载PDF
            logger.info(f"正在下载: {pdf_url}")
            
            # 添加随机延迟，避免过快请求被阻止
            time.sleep(random.uniform(1, 3))
            
            response = requests.get(pdf_url, stream=True, timeout=30)
            response.raise_for_status()
            
            with open(pdf_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            
            logger.info(f"PDF下载成功: {pdf_path}")
            
            downloaded_papers.append({
                "paper_id": paper_id,
                "title": title,
                "pdf_path": pdf_path
            })
            
            count += 1
            
        except Exception as e:
            logger.error(f"下载PDF失败 (可能的标题: {getattr(paper, 'title', paper.get('title', '未知标题'))}): {e}")
            import traceback
            logger.error(f"错误详情: {traceback.format_exc()}")
    
    logger.info(f"共下载 {len(downloaded_papers)}/{min(len(papers), max_papers)} 篇PDF")
    return downloaded_papers

def is_paper_fully_processed(paper_id, paper_title, processed_papers):
    """检查论文是否已经完全处理过（所有类型的总结都已完成）"""
    # 需要检查的所有处理类型
    all_types = ["contributions_zh", "problems_zh", "innovations_zh"]
    
    # 通过ID检查
    if paper_id in processed_papers:
        paper_data = processed_papers[paper_id]
        # 检查是否所有类型都已处理
        if all(process_type in paper_data.get('process_types', []) for process_type in all_types):
            # 确认结果存在
            if "results" in paper_data and all(t in paper_data["results"] for t in all_types):
                return True
    
    # 通过标题检查
    for pid, data in processed_papers.items():
        if data.get('title') == paper_title:
            # 检查是否所有类型都已处理
            if all(process_type in data.get('process_types', []) for process_type in all_types):
                # 确认结果存在
                if "results" in data and all(t in data["results"] for t in all_types):
                    return True
    
    return False

if __name__ == "__main__":
    main() 