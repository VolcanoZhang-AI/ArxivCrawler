# ArxivCrawler

ArxivCrawler是一个自动化工具，用于从Arxiv上抓取和分析最新的LLM相关研究论文，并生成详细的研究热点报告。

## 功能特点

- 自动抓取Arxiv上最近发布的与LLM相关的论文
- 分析最近五天和最近一天的热点论文
- 总结每篇论文解决的问题、主要贡献和创新点
- 生成研究趋势可视化图表
- 输出格式化的markdown报告
- **新增**: 自动下载热点论文PDF
- **新增**: 使用自定义词典改进中文关键词提取
- **新增**: 可靠的错误处理和重试机制
- **新增**: 使用Qwen-Max大模型进行论文贡献的中文智能总结

## 安装

1. 克隆此仓库:
```
git clone https://github.com/yourusername/ArxivCrawler.git
cd ArxivCrawler
```

2. 安装依赖:
```
pip install -r requirements.txt
```

3. 配置Qwen-Max API (可选):
在`config.json`文件中填入你的Qwen API密钥和其他参数。脚本支持两种API配置:

### 标准OpenAI兼容API:
```json
{
    "qwen_api_key": "你的API密钥",
    "qwen_api_base": "https://api.qianwen-inc.com/v1",
    "max_tokens": 1000,
    "temperature": 0.7,
    "model_name": "qwen-max"
}
```

### 阿里云DashScope API:
```json
{
    "qwen_api_key": "你的API密钥",
    "qwen_api_base": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "max_tokens": 1000,
    "temperature": 0.7,
    "model_name": "qwen-max"
}
```

## 使用方法

直接运行主脚本:
```
python arxiv_crawler.py
```

该脚本将:
1. 从Arxiv获取最近发布的LLM相关论文
2. 分析这些论文的内容
3. 下载热点论文的PDF文件
4. 使用Qwen-Max大模型对论文贡献进行中文智能总结（如果配置了API密钥）
5. 生成热点研究报告
6. 在`arxiv_reports`目录下创建报告文件和可视化图表

## 输出

脚本会生成以下输出:
- Markdown格式的研究热点报告 (`arxiv_reports/LLM研究热点报告_YYYY年MM月DD日.md`)
- 研究类别分布图 (`arxiv_reports/category_distribution.html`)
- 关键词热度词云图 (`arxiv_reports/keywords_cloud.html`)
- 下载的热点论文PDF文件 (`arxiv_reports/pdfs/`)
- 日志文件 (`arxiv_crawler.log`)

## 自定义配置

- `config.json`: 配置Qwen API和模型参数
- `llm_dict.txt`: 自定义词典，用于改进中文分词和关键词提取
- 可以在源代码中修改LLM_KEYWORDS列表来自定义搜索关键词

## 故障排除

如果在使用Qwen API时遇到问题:

1. 检查API密钥是否正确
2. 确认API基础URL是否与你的服务提供商匹配
3. 查看`arxiv_crawler.log`文件中的详细错误信息
4. 如果使用阿里云DashScope，请确保你的账户已充值并有足够余额