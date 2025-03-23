"""
测试OpenAI客户端初始化
"""
import os
import sys
import inspect
from openai import OpenAI

# 打印OpenAI客户端初始化函数的参数
print("OpenAI.__init__ 参数:")
sig = inspect.signature(OpenAI.__init__)
for param_name, param in sig.parameters.items():
    print(f"  {param_name}: {param.default}")

# 测试基本初始化
try:
    client = OpenAI(
        api_key="test_key",
        base_url="https://example.com/v1"
    )
    print("基本初始化成功")
except Exception as e:
    print(f"基本初始化失败: {e}")
    import traceback
    print(traceback.format_exc())

# 测试通过环境变量设置代理
try:
    # 设置代理环境变量
    os.environ["http_proxy"] = "http://127.0.0.1:7890"
    os.environ["https_proxy"] = "http://127.0.0.1:7890"
    print(f"已设置代理环境变量 http_proxy={os.environ.get('http_proxy')}, https_proxy={os.environ.get('https_proxy')}")
    
    # 尝试初始化
    client_with_proxy = OpenAI(
        api_key="test_key", 
        base_url="https://example.com/v1"
    )
    print("使用环境变量代理初始化成功")
except Exception as e:
    print(f"使用环境变量代理初始化失败: {e}")
    import traceback
    print(traceback.format_exc())

# 检查是否存在猴子补丁
print("\n检查是否存在猴子补丁:")
OpenAI_dir = dir(OpenAI)
print(f"OpenAI类的属性和方法: {OpenAI_dir}")

# 查看模块路径
print("\n相关模块路径:")
from openai import _client
print(f"openai._client 路径: {_client.__file__}")

# 显示当前OpenAI版本
import openai
print(f"\nOpenAI 版本: {openai.__version__}")

# 检查Python路径
print("\nPython路径:")
for p in sys.path:
    print(f"  {p}")

# 打印这个检查完成的信息
print("\n检查完成，请分析以上输出信息排查问题") 