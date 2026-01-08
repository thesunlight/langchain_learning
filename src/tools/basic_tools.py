"""
LangChain 学习项目 - Tools 使用示例

本文件演示了自定义工具的创建和使用，包括：
- 基础工具创建
- 带参数的工具
- 多工具组合
"""

from langchain.tools import tool
from langchain_core.tools import Tool
from langchain.agents import AgentType, initialize_agent, load_tools
from langchain_openai import OpenAI
from langchain_core.utils.function_calling import convert_to_openai_function
from langchain_openai import ChatOpenAI
import os
import math
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
def calculator_tool(expression: str) -> float:
    """
    执行数学计算的工具
    
    Args:
        expression: 数学表达式字符串，如 "2 + 3 * 4"
        
    Returns:
        计算结果
    """
    try:
        # 安全的数学表达式计算
        allowed_operators = ['+', '-', '*', '/', '//', '%', '**', '(', ')', ' ', '.', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
        
        # 检查表达式是否只包含允许的字符
        for char in expression:
            if char not in allowed_operators:
                raise ValueError(f"不允许的操作符: {char}")
        
        result = eval(expression)
        return float(result)
    except Exception as e:
        return f"计算错误: {str(e)}"

@tool
def date_time_tool() -> str:
    """
    获取当前日期和时间的工具
    
    Returns:
        格式化的当前日期和时间
    """
    now = datetime.now()
    return now.strftime("%Y-%m-%d %H:%M:%S")

@tool
def circle_area_tool(radius: float) -> float:
    """
    计算圆面积的工具
    
    Args:
        radius: 圆的半径
        
    Returns:
        圆的面积
    """
    if radius < 0:
        return "半径不能为负数"
    
    area = math.pi * radius ** 2
    return round(area, 2)

def basic_tool_example():
    """
    基础工具使用示例
    """
    print("=== 基础工具使用示例 ===")
    
    # 直接调用工具
    print("1. 计算器工具:")
    calc_result = calculator_tool.invoke({"expression": "15 * 4 + 10"})
    print(f"   15 * 4 + 10 = {calc_result}")
    
    print("\n2. 日期时间工具:")
    time_result = date_time_tool.invoke({})
    print(f"   当前时间: {time_result}")
    
    print("\n3. 圆面积工具:")
    circle_result = circle_area_tool.invoke({"radius": 5})
    print(f"   半径为5的圆面积: {circle_result}")
    print()

def agent_with_custom_tools_example():
    """
    在智能体中使用自定义工具的示例
    """
    print("=== 智能体中使用自定义工具示例 ===")
    
    # 初始化 LLM
    llm = get_siliconflow_llm()
    
    # 创建工具列表
    tools = [
        calculator_tool,
        date_time_tool,
        circle_area_tool
    ]
    
    # 初始化智能体
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        handle_parsing_errors=True
    )
    
    # 运行智能体任务
    print("任务1: 计算 (25 + 17) * 3 的结果")
    try:
        result1 = agent.run("请帮我计算 (25 + 17) * 3 的结果")
        print(f"结果: {result1}\n")
    except Exception as e:
        print(f"任务1出错: {e}\n")
    
    print("任务2: 获取当前时间，并计算半径为7的圆的面积")
    try:
        result2 = agent.run("请告诉我当前时间，然后帮我计算半径为7的圆的面积")
        print(f"结果: {result2}\n")
    except Exception as e:
        print(f"任务2出错: {e}\n")

def predefined_tools_example():
    """
    使用预定义工具的示例
    """
    print("=== 预定义工具示例 ===")
    
    # 获取 SiliconFlow 模型
    llm = get_siliconflow_llm()
    
    # 加载预定义工具 (需要安装额外包)
    try:
        # 这里我们展示如何加载一些常用的预定义工具
        # 注意：某些工具可能需要额外的API密钥
        predefined_tools = []
        
        # 如果需要使用 serpapi 或其他工具，可以这样加载：
        # predefined_tools = load_tools(["serpapi", "llm-math"], llm=llm)
        
        print("预定义工具通常包括:")
        print("- 搜索工具 (需要 serpapi API 密钥)")
        print("- 数学计算工具")
        print("- 终端工具")
        print("- 其他第三方API工具")
        print()
        
    except Exception as e:
        print(f"加载预定义工具时出错 (这通常是由于缺少API密钥): {e}")
        print("对于学习目的，我们使用自定义工具即可")
        print()

def tool_with_validation_example():
    """
    带参数验证的工具示例
    """
    print("=== 带参数验证的工具示例 ===")
    
    @tool
    def temperature_converter(temp: float, from_unit: str, to_unit: str) -> str:
        """
        温度转换工具
        
        Args:
            temp: 温度值
            from_unit: 原始单位 ('C', 'F', 'K')
            to_unit: 目标单位 ('C', 'F', 'K')
            
        Returns:
            转换后的温度
        """
        from_unit = from_unit.upper()
        to_unit = to_unit.upper()
        
        valid_units = ['C', 'F', 'K']
        if from_unit not in valid_units or to_unit not in valid_units:
            return "错误: 单位必须是 'C', 'F', 或 'K'"
        
        if from_unit == to_unit:
            return f"{temp} {from_unit} = {temp} {to_unit}"
        
        # 转换到摄氏度作为中间单位
        if from_unit == 'F':
            celsius = (temp - 32) * 5/9
        elif from_unit == 'K':
            celsius = temp - 273.15
        else:
            celsius = temp
        
        # 从摄氏度转换到目标单位
        if to_unit == 'F':
            result = celsius * 9/5 + 32
        elif to_unit == 'K':
            result = celsius + 273.15
        else:
            result = celsius
        
        return f"{temp}°{from_unit} = {result:.2f}°{to_unit}"
    
    # 测试温度转换工具
    print("温度转换示例:")
    conversions = [
        (100, "C", "F"),
        (32, "F", "C"),
        (273.15, "K", "C"),
    ]
    
    for temp, from_unit, to_unit in conversions:
        result = temperature_converter.invoke({
            "temp": temp,
            "from_unit": from_unit,
            "to_unit": to_unit
        })
        print(f"  {result}")
    print()

if __name__ == "__main__":
    print("LangChain Tools 使用示例")
    print("="*60)
    
    try:
        basic_tool_example()
        predefined_tools_example()
        tool_with_validation_example()
        agent_with_custom_tools_example()
    except Exception as e:
        print(f"运行时出现错误: {e}")
        print("请确保已设置 API 密钥 (SILICONFLOW_API_KEY 或 OPENAI_API_KEY)")