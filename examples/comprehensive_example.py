"""
LangChain 综合示例 - 创建一个AI助手

本示例整合了 LangChain 的多个核心组件：
- 使用聊天模型
- 自定义工具
- 对话记忆
- 提示词模板
- 智能体
"""

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.memory import ConversationBufferMemory
from langchain.agents.format_scratchpad.openai_tools import format_to_openai_tool_messages
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser
from langchain_core.utils.function_calling import convert_to_openai_function
import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

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
        return ChatOpenAI(model="gpt-3.5-turbo", temperature=0.3)
    
    chat_model = ChatOpenAI(
        model="Qwen/Qwen2.5-72B-Instruct",
        base_url=base_url,
        api_key=api_key,
        temperature=0.3
    )
    return chat_model

@tool
def product_calculator(quantity: int, unit_price: float) -> float:
    """
    计算产品总价的工具
    
    Args:
        quantity: 数量
        unit_price: 单价
        
    Returns:
        总价
    """
    if quantity < 0 or unit_price < 0:
        return "数量和单价必须为正数"
    
    total = quantity * unit_price
    return round(total, 2)

@tool
def discount_calculator(original_price: float, discount_percent: float) -> dict:
    """
    计算折扣后价格的工具
    
    Args:
        original_price: 原价
        discount_percent: 折扣百分比
        
    Returns:
        包含原价、折扣金额和折扣后价格的字典
    """
    if original_price < 0 or discount_percent < 0 or discount_percent > 100:
        return {"error": "价格不能为负数，折扣百分比应在0-100之间"}
    
    discount_amount = original_price * (discount_percent / 100)
    final_price = original_price - discount_amount
    
    return {
        "original_price": round(original_price, 2),
        "discount_amount": round(discount_amount, 2),
        "final_price": round(final_price, 2),
        "discount_percent": discount_percent
    }

def create_comprehensive_example():
    """
    创建综合示例 - AI购物助手
    """
    print("=== LangChain 综合示例 - AI购物助手 ===\n")
    
    # 初始化聊天模型
    llm = get_siliconflow_chat_model()
    
    # 定义工具
    tools = [product_calculator, discount_calculator]
    
    # 创建提示词模板
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个专业的购物助手，可以帮助用户计算价格、应用折扣等。"
                   "请使用提供的工具来完成计算任务。"
                   "始终保持友好和专业的态度。"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad")
    ])
    
    # 创建记忆组件
    memory = ConversationBufferMemory(return_messages=True, memory_key="chat_history")
    
    # 创建智能体
    agent = create_openai_tools_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        memory=memory,
        verbose=True
    )
    
    print("AI购物助手已启动！您可以询问关于价格计算、折扣等问题。")
    print("示例问题：")
    print("1. '计算10件商品，每件15.5元的总价'")
    print("2. '计算原价100元打8折后的价格'")
    print("3. '请记住我的名字是张三'")
    print()
    
    # 运行几个示例对话
    examples = [
        "你好，我需要计算购买5件商品的总价，每件商品20元",
        "现在这些商品打9折，请计算折扣后的价格",
        "请总结一下我们的对话内容"
    ]
    
    for i, example in enumerate(examples, 1):
        print(f"示例 {i}: {example}")
        try:
            response = agent_executor.invoke({"input": example})
            print(f"助手: {response['output']}")
            print()
        except Exception as e:
            print(f"处理示例 {i} 时出错: {e}")
            print()

def create_simple_chain_example():
    """
    创建简单链示例
    """
    print("=== 简单链示例 ===\n")
    
    # 初始化模型
    llm = get_siliconflow_chat_model()
    
    # 创建提示词模板
    prompt = ChatPromptTemplate.from_template(
        "请为以下产品类别生成一个营销口号: {product_category}\n"
        "口号应该简短、有吸引力，并突出产品的独特卖点。"
    )
    
    # 创建链
    chain = prompt | llm
    
    # 运行链
    try:
        result = chain.invoke({"product_category": "智能手表"})
        print(f"输入: 智能手表")
        print(f"输出: {result.content}\n")
    except Exception as e:
        print(f"调用模型时出错: {e}\n")

def create_memory_example():
    """
    创建记忆组件示例
    """
    print("=== 记忆组件示例 ===\n")
    
    # 初始化模型和记忆
    llm = get_siliconflow_chat_model()
    memory = ConversationBufferMemory(return_messages=True)
    
    # 创建带记忆的链
    from langchain.chains import ConversationChain
    
    conversation = ConversationChain(
        llm=llm,
        memory=memory,
        verbose=False
    )
    
    # 进行多轮对话
    inputs = [
        "你好，我是李四，我是一名设计师。",
        "我最喜欢的设计风格是极简主义。",
        "请根据我的信息推荐一个合适的项目管理工具。"
    ]
    
    print("多轮对话示例:")
    for i, user_input in enumerate(inputs, 1):
        print(f"用户输入 {i}: {user_input}")
        try:
            response = conversation.predict(input=user_input)
            print(f"AI响应 {i}: {response}")
        except Exception as e:
            print(f"AI响应 {i} 时出错: {e}")
        print()

def main():
    """
    主函数 - 运行所有示例
    """
    print("LangChain 综合学习示例")
    print("="*60)
    
    try:
        create_simple_chain_example()
        create_memory_example()
        create_comprehensive_example()
        
        print("所有示例运行完成！")
        print("\n要尝试自己的查询，请参考以下模式:")
        print("1. 使用自定义工具进行计算")
        print("2. 利用记忆功能进行多轮对话")
        print("3. 结合多种 LangChain 组件创建复杂应用")
        
    except Exception as e:
        print(f"运行时出现错误: {e}")
        print("请确保已正确设置 API 密钥 (SILICONFLOW_API_KEY 或 OPENAI_API_KEY)")

if __name__ == "__main__":
    main()