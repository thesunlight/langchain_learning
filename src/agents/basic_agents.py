"""
LangChain 学习项目 - Agents 使用示例

本文件演示了智能体的创建和使用，包括：
- ReAct智能体
- 自定义工具智能体
- 多工具协调智能体
"""

from langchain.agents import AgentType, initialize_agent, Tool, load_tools
from langchain_openai import OpenAI, ChatOpenAI
from langchain_core.tools import tool
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
import os
import requests
from datetime import datetime
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

def get_siliconflow_llm():
    """
    获取 SiliconFlow LLM 实例
    """
    base_url = os.getenv("SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1")
    api_key = os.getenv("SILICONFLOW_API_KEY")
    
    if not api_key:
        # 如果没有 SiliconFlow API 密钥，使用 OpenAI (需要设置 OPENAI_API_KEY)
        print("警告: 未设置 SILICONFLOW_API_KEY 环境变量")
        print("使用 OpenAI 模型作为备选")
        return OpenAI(model_name="gpt-3.5-turbo-instruct", temperature=0)
    
    llm = OpenAI(
        model_name="Qwen/Qwen2.5-72B-Instruct",
        base_url=base_url,
        api_key=api_key,
        temperature=0
    )
    return llm

def get_siliconflow_chat_model():
    """
    获取 SiliconFlow 聊天模型实例
    """
    base_url = os.getenv("SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1")
    api_key = os.getenv("SILICONFLOW_API_KEY")
    
    if not api_key:
        # 如果没有 SiliconFlow API 密钥，使用 OpenAI (需要设置 OPENAI_API_KEY)
        print("警告: 未设置 SILICONFLOW_API_KEY 环境变量")
        print("使用 OpenAI 聊天模型作为备选")
        return ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    
    chat_model = ChatOpenAI(
        model_name="Qwen/Qwen2.5-72B-Instruct",
        base_url=base_url,
        api_key=api_key,
        temperature=0
    )
    return chat_model

@tool
def weather_tool(city: str) -> str:
    """
    获取指定城市的天气信息（模拟工具）
    
    Args:
        city: 城市名称
        
    Returns:
        天气信息
    """
    # 这是一个模拟工具，实际应用中可以调用真实API
    weather_data = {
        "北京": "晴天，温度15°C",
        "上海": "多云，温度18°C", 
        "广州": "雨天，温度22°C",
        "深圳": "晴天，温度24°C"
    }
    return weather_data.get(city, f"无法获取{city}的天气信息")

@tool
def calculator_tool(expression: str) -> float:
    """
    执行数学计算的工具
    
    Args:
        expression: 数学表达式字符串
        
    Returns:
        计算结果
    """
    try:
        allowed_operators = ['+', '-', '*', '/', '//', '%', '**', '(', ')', ' ', '.', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
        for char in expression:
            if char not in allowed_operators:
                raise ValueError(f"不允许的操作符: {char}")
        result = eval(expression)
        return float(result)
    except Exception as e:
        return f"计算错误: {str(e)}"

def basic_agent_example():
    """
    基础智能体示例
    """
    print("=== 基础智能体示例 ===")
    
    # 初始化 LLM
    llm = get_siliconflow_llm()
    
    # 创建工具列表
    tools = [
        Tool(
            name="Calculator",
            func=calculator_tool,
            description="用于执行数学计算，输入数学表达式字符串"
        )
    ]
    
    # 初始化智能体
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,  # 显示智能体思考过程
        handle_parsing_errors=True
    )
    
    # 运行智能体
    print("任务: 计算 25 * 4 + 10 的结果")
    try:
        result = agent.run("请帮我计算 25 * 4 + 10 的结果")
        print(f"最终结果: {result}")
    except Exception as e:
        print(f"执行出错: {e}")
    print()

def agent_with_multiple_tools_example():
    """
    多工具智能体示例
    """
    print("=== 多工具智能体示例 ===")
    
    # 初始化 LLM
    llm = get_siliconflow_llm()
    
    # 创建多个工具
    tools = [
        Tool(
            name="Weather",
            func=weather_tool,
            description="获取指定城市的天气信息，输入城市名称"
        ),
        Tool(
            name="Calculator", 
            func=calculator_tool,
            description="用于执行数学计算，输入数学表达式字符串"
        )
    ]
    
    # 初始化智能体
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        handle_parsing_errors=True
    )
    
    # 运行需要多个工具的任务
    print("任务: 获取北京的天气，然后计算温度的两倍")
    try:
        result = agent.run("请告诉我北京的天气如何，然后计算温度的两倍是多少")
        print(f"最终结果: {result}")
    except Exception as e:
        print(f"执行出错: {e}")
    print()

def react_agent_example():
    """
    ReAct智能体示例（使用Chat模型）
    """
    print("=== ReAct智能体示例 ===")
    
    # 初始化聊天模型
    chat_model = get_siliconflow_chat_model()
    
    # 创建工具
    tools = [
        Tool(
            name="Current Time",
            func=lambda x: datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            description="获取当前日期和时间"
        ),
        calculator_tool
    ]
    
    # 初始化ReAct智能体
    agent = initialize_agent(
        tools=tools,
        llm=chat_model,
        agent=AgentType.REACT_CHAT,
        verbose=True,
        handle_parsing_errors=True
    )
    
    print("任务: 获取当前时间")
    try:
        result = agent.run("请告诉我当前时间")
        print(f"最终结果: {result}")
    except Exception as e:
        print(f"执行出错: {e}")
    print()

def custom_agent_behavior_example():
    """
    自定义智能体行为示例
    """
    print("=== 自定义智能体行为示例 ===")
    
    # 初始化 LLM
    llm = get_siliconflow_llm()
    
    # 创建工具
    tools = [
        Tool(
            name="Weather Tool",
            func=weather_tool,
            description="获取指定城市的天气信息"
        ),
        Tool(
            name="Calculator Tool",
            func=calculator_tool, 
            description="用于执行数学计算"
        )
    ]
    
    # 创建自定义提示词模板
    template = """
    你是一个智能助手，可以根据需要使用以下工具：
    
    {tools}
    
    使用以下格式:
    问题: 需要回答的问题
    思考: 你应该如何解决问题
    行动: 工具名称
    输入: 工具输入
    观察: 工具输出
    ... (可以有多个思考/行动/观察)
    思考: 我现在知道最终答案
    最终答案: 原始问题的最终答案
    
    开始!
    
    问题: {input}
    {agent_scratchpad}
    """
    
    prompt = PromptTemplate.from_template(template)
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        prompt=prompt,
        verbose=True,
        handle_parsing_errors=True
    )
    
    print("任务: 比较北京和上海的温度，并计算它们的平均值")
    try:
        result = agent.run("请比较北京和上海的温度，并计算它们的平均值")
        print(f"最终结果: {result}")
    except Exception as e:
        print(f"执行出错: {e}")
    print()

def agent_with_context_example():
    """
    带上下文的智能体示例
    """
    print("=== 带上下文的智能体示例 ===")
    
    # 初始化 LLM
    llm = get_siliconflow_chat_model()
    
    # 创建工具
    tools = [weather_tool, calculator_tool]
    
    # 初始化智能体
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        handle_parsing_errors=True
    )
    
    print("任务: 基于天气建议合适的活动，并进行相关计算")
    try:
        result = agent.run("请告诉我广州的天气如何，如果下雨，计算如果雨下了3小时，每小时降水量是2毫米，总降水量是多少")
        print(f"最终结果: {result}")
    except Exception as e:
        print(f"执行出错: {e}")
    print()

if __name__ == "__main__":
    print("LangChain Agents 使用示例")
    print("="*60)
    
    try:
        basic_agent_example()
        agent_with_multiple_tools_example()
        react_agent_example()
        custom_agent_behavior_example()
        agent_with_context_example()
    except Exception as e:
        print(f"运行时出现错误: {e}")
        print("请确保已设置 API 密钥 (SILICONFLOW_API_KEY 或 OPENAI_API_KEY)")