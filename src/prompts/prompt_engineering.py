"""
LangChain 学习项目 - 提示词工程

本文件演示了提示词工程的核心概念，包括：
- 提示词模板
- 消息模板
- 输出解析器
"""

from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

def basic_prompt_template_example():
    """
    基础提示词模板示例
    """
    print("=== 基础提示词模板示例 ===")
    
    # 创建一个简单的提示词模板
    template = "请为以下产品类别生成一个吸引人的口号: {product_category}"
    prompt_template = PromptTemplate.from_template(template)
    
    # 格式化提示词
    prompt = prompt_template.format(product_category="智能手表")
    
    print(f"生成的提示词: {prompt}")
    print()

def chat_prompt_template_example():
    """
    聊天提示词模板示例
    """
    print("=== 聊天提示词模板示例 ===")
    
    # 创建聊天提示词模板
    chat_template = ChatPromptTemplate.from_messages([
        ("system", "你是一个专业的{expertise}顾问。"),
        ("human", "我正在开发一个{project_type}项目，遇到了{issue}问题。请提供解决方案。"),
        MessagesPlaceholder(variable_name="additional_context", optional=True)
    ])
    
    # 格式化聊天提示词
    messages = chat_template.format_messages(
        expertise="软件开发",
        project_type="Web应用",
        issue="数据库性能",
        additional_context=[HumanMessage(content="我的数据库是PostgreSQL，用户量很大。")]
    )
    
    print("格式化后的消息:")
    for i, msg in enumerate(messages):
        msg_type = type(msg).__name__
        print(f"{i+1}. [{msg_type}] {msg.content}")
    print()

def complex_prompt_example():
    """
    复杂提示词示例 - 使用多个变量
    """
    print("=== 复杂提示词示例 ===")
    
    # 创建一个复杂的提示词模板
    complex_template = """
    任务: {task}
    
    背景信息:
    - 目标受众: {audience}
    - 语气: {tone}
    - 长度要求: {length}
    
    补充信息: {additional_info}
    
    请根据以上信息完成任务。
    """
    
    prompt_template = PromptTemplate.from_template(complex_template)
    
    prompt = prompt_template.format(
        task="写一篇关于环保的文章",
        audience="年轻消费者",
        tone="积极、鼓舞人心",
        length="300字左右",
        additional_info="重点强调个人行动的重要性"
    )
    
    print(f"复杂提示词:\n{prompt}")
    print()

def dynamic_prompt_example():
    """
    动态提示词示例 - 根据条件选择不同模板
    """
    print("=== 动态提示词示例 ===")
    
    # 根据不同场景选择提示词模板
    templates = {
        "technical": "请用专业术语解释{topic}的技术原理。",
        "beginner": "请用通俗易懂的语言解释{topic}。",
        "business": "请从商业角度分析{topic}的价值和应用。"
    }
    
    def generate_prompt(level, topic):
        template = templates.get(level, templates["beginner"])
        return template.format(topic=topic)
    
    # 生成不同级别的提示词
    topics = ["机器学习", "区块链"]
    levels = ["technical", "beginner", "business"]
    
    for topic in topics:
        print(f"主题: {topic}")
        for level in levels:
            prompt = generate_prompt(level, topic)
            print(f"  {level}级别: {prompt}")
        print()

if __name__ == "__main__":
    print("LangChain 提示词工程示例")
    print("="*60)
    
    try:
        basic_prompt_template_example()
        chat_prompt_template_example()
        complex_prompt_example()
        dynamic_prompt_example()
    except Exception as e:
        print(f"运行时出现错误: {e}")
        print("请确保已设置 OPENAI_API_KEY 环境变量")