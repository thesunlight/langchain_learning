"""
LangChain 学习项目 - 模型使用示例

本文件演示了如何使用不同的语言模型，包括：
- OpenAI 模型
- SiliconFlow 模型
- 本地模型
- 聊天模型
"""

# 导入必要的库
from langchain_openai import ChatOpenAI, OpenAI
from langchain_core.messages import HumanMessage, SystemMessage
import os
from dotenv import load_dotenv

# 从工具模块导入硅基流动配置函数
from src.utils.siliconflow_utils import get_siliconflow_llm, get_siliconflow_chat_model, test_connection

# 加载环境变量
load_dotenv()

def basic_llm_example():
    """
    基础 LLM 使用示例
    """
    print("=== 基础 LLM 使用示例 ===")
    
    # 尝试使用 SiliconFlow LLM
    llm = get_siliconflow_llm(temperature=0.7)
    
    # 简单文本生成
    prompt = "请简要介绍人工智能的发展历程。"
    try:
        response = llm.invoke(prompt)
        print(f"输入: {prompt}")
        print(f"输出: {response}")
    except Exception as e:
        print(f"调用模型时出错: {e}")
    print()

def chat_model_example():
    """
    聊天模型使用示例
    """
    print("=== 聊天模型使用示例 ===")
    
    # 尝试使用 SiliconFlow 聊天模型
    chat_model = get_siliconflow_chat_model(temperature=0.3)
    
    # 创建消息列表
    messages = [
        SystemMessage(content="你是一个有帮助的AI助手。"),
        HumanMessage(content="你好，你能帮我介绍下机器学习吗？")
    ]
    
    try:
        # 获取响应
        response = chat_model.invoke(messages)
        print(f"AI响应: {response.content}")
    except Exception as e:
        print(f"调用模型时出错: {e}")
    print()

def streaming_example():
    """
    流式响应示例
    """
    print("=== 流式响应示例 ===")
    
    # 尝试使用 SiliconFlow 聊天模型
    chat_model = get_siliconflow_chat_model(temperature=0.7)
    
    try:
        # 流式获取响应
        stream = chat_model.stream([
            HumanMessage(content="请用三句话解释什么是LangChain。")
        ])
        
        print("流式响应:")
        for chunk in stream:
            print(chunk.content, end="", flush=True)
        print("\n")
    except Exception as e:
        print(f"调用模型时出错: {e}")
        print()

if __name__ == "__main__":
    print("LangChain 模型使用示例")
    print("="*50)
    
    # 测试连接
    print("测试 SiliconFlow 连接...")
    connection_ok = test_connection()
    if not connection_ok:
        print("注意: SiliconFlow 连接测试失败，可能会使用备选模型")
    print()
    
    try:
        basic_llm_example()
        chat_model_example()
        streaming_example()
    except Exception as e:
        print(f"运行时出现错误: {e}")
        print("请确保已正确设置 API 密钥")