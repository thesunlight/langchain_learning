"""
LangChain 学习项目 - Chains 使用示例

本文件演示了 Chains 的核心概念和使用方法，包括：
- 序列链（Sequential Chains）
- 简单序列链（SimpleSequentialChain）
- LLMChain
- Router Chain
"""

from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAI, ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.chains import LLMChain, SimpleSequentialChain, SequentialChain
import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

def llm_chain_example():
    """
    LLMChain 使用示例
    """
    print("=== LLMChain 使用示例 ===")
    
    # 创建提示词模板
    template = "请为以下产品生成一个营销标语: {product}"
    prompt = PromptTemplate.from_template(template)
    
    # 初始化 LLM
    llm = OpenAI(temperature=0.7, model_name="gpt-3.5-turbo-instruct")
    
    # 创建 LLMChain
    chain = LLMChain(llm=llm, prompt=prompt)
    
    # 运行链
    result = chain.run(product="无线耳机")
    
    print(f"输入产品: 无线耳机")
    print(f"生成标语: {result}")
    print()

def simple_sequential_chain_example():
    """
    简单序列链示例
    """
    print("=== 简单序列链示例 ===")
    
    # 初始化 LLM
    llm = OpenAI(temperature=0.7, model_name="gpt-3.5-turbo-instruct")
    
    # 第一个链：生成产品描述
    template1 = "为以下产品创建一个吸引人的描述: {product}"
    prompt1 = PromptTemplate(input_variables=["product"], template=template1)
    chain1 = LLMChain(llm=llm, prompt=prompt1, output_key="description")
    
    # 第二个链：基于描述生成营销标语
    template2 = "基于以下产品描述，创建一个营销标语: {description}"
    prompt2 = PromptTemplate(input_variables=["description"], template=template2)
    chain2 = LLMChain(llm=llm, prompt=prompt2, output_key="tagline")
    
    # 组合简单序列链
    overall_chain = SimpleSequentialChain(
        chains=[chain1, chain2],
        verbose=True  # 显示中间步骤
    )
    
    # 运行链
    result = overall_chain.run("智能手表")
    
    print(f"最终结果: {result}")
    print()

def sequential_chain_example():
    """
    序列链示例
    """
    print("=== 序列链示例 ===")
    
    # 初始化 LLM
    llm = OpenAI(temperature=0.5, model_name="gpt-3.5-turbo-instruct")
    
    # 第一个链：生成产品特点
    template1 = """你是一位专业的产品经理。请为以下产品列出3个主要特点：
    产品: {product}
    
    特点列表:"""
    prompt1 = PromptTemplate(input_variables=["product"], template=template1)
    chain1 = LLMChain(llm=llm, prompt=prompt1, output_key="features")
    
    # 第二个链：根据特点生成目标用户
    template2 = """基于以下产品特点，确定目标用户群体：
    产品特点: {features}
    
    目标用户群体:"""
    prompt2 = PromptTemplate(input_variables=["features"], template=template2)
    chain2 = LLMChain(llm=llm, prompt=prompt2, output_key="target_users")
    
    # 第三个链：根据特点和目标用户生成营销策略
    template3 = """基于以下信息制定营销策略：
    产品特点: {features}
    目标用户: {target_users}
    
    营销策略:"""
    prompt3 = PromptTemplate(input_variables=["features", "target_users"], template=template3)
    chain3 = LLMChain(llm=llm, prompt=prompt3, output_key="marketing_strategy")
    
    # 组合序列链
    overall_chain = SequentialChain(
        chains=[chain1, chain2, chain3],
        input_variables=["product"],
        output_variables=["features", "target_users", "marketing_strategy"],
        verbose=True
    )
    
    # 运行链
    result = overall_chain({"product": "便携式咖啡机"})
    
    print("序列链结果:")
    print(f"产品特点: {result['features']}")
    print(f"目标用户: {result['target_users']}")
    print(f"营销策略: {result['marketing_strategy']}")
    print()

def runnable_chain_example():
    """
    使用 LangChain Expression Language (LCEL) 创建可运行链
    """
    print("=== Runnable Chain 示例 (LCEL) ===")
    
    # 初始化 LLM
    llm = ChatOpenAI(temperature=0.7, model_name="gpt-3.5-turbo")
    
    # 创建提示词模板
    template = "请为以下业务类型创建一个商业计划概要: {business_type}"
    prompt = PromptTemplate.from_template(template)
    
    # 使用 LCEL 创建链
    chain = prompt | llm | StrOutputParser()
    
    # 运行链
    result = chain.invoke({"business_type": "在线教育平台"})
    
    print(f"输入: 在线教育平台")
    print(f"商业计划概要: {result}")
    print()

def router_chain_concept():
    """
    Router Chain 概念说明（实际实现需要更复杂的设置）
    """
    print("=== Router Chain 概念说明 ===")
    print("""
    Router Chain 是 LangChain 中的高级概念，用于：
    1. 根据输入动态选择不同的处理路径
    2. 实现条件逻辑
    3. 提高系统的灵活性
    
    基本原理：
    - 输入 -> 路由器 -> 选择适当的链 -> 执行 -> 输出
    
    应用场景：
    - 根据问题类型选择不同的处理逻辑
    - 根据用户输入选择不同的工具
    - 实现多步骤决策流程
    """)
    print()

if __name__ == "__main__":
    print("LangChain Chains 使用示例")
    print("="*60)
    
    try:
        llm_chain_example()
        simple_sequential_chain_example()
        sequential_chain_example()
        runnable_chain_example()
        router_chain_concept()
    except Exception as e:
        print(f"运行时出现错误: {e}")
        print("请确保已设置 OPENAI_API_KEY 环境变量")