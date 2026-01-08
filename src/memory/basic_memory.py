"""
LangChain 学习项目 - Memory 使用示例

本文件演示了 Memory 组件的使用，包括：
- 对话记忆
- 缓冲记忆
- 实体记忆
"""

from langchain.memory import ConversationBufferMemory, ConversationBufferWindowMemory, ConversationSummaryMemory
from langchain_openai import OpenAI, ChatOpenAI
from langchain.chains import ConversationChain
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import PromptTemplate
import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

def conversation_buffer_memory_example():
    """
    对话缓冲记忆示例
    """
    print("=== 对话缓冲记忆示例 ===")
    
    # 初始化 LLM
    llm = OpenAI(temperature=0, model_name="gpt-3.5-turbo-instruct")
    
    # 创建对话缓冲记忆
    memory = ConversationBufferMemory()
    
    # 创建对话链
    conversation = ConversationChain(
        llm=llm,
        memory=memory,
        verbose=False  # 设置为 True 可以看到提示词
    )
    
    # 进行多轮对话
    print("开始多轮对话:")
    
    response1 = conversation.predict(input="你好，我是张三，我是一名软件工程师。")
    print(f"AI: {response1}")
    
    response2 = conversation.predict(input="你能告诉我更多关于我的信息吗？")
    print(f"AI: {response2}")
    
    response3 = conversation.predict(input="请总结一下我们的对话。")
    print(f"AI: {response3}")
    
    print("\n当前记忆内容:")
    print(memory.buffer)
    print()

def conversation_buffer_window_memory_example():
    """
    对话窗口记忆示例（只记住最近的几轮对话）
    """
    print("=== 对话窗口记忆示例 ===")
    
    # 初始化 LLM
    llm = OpenAI(temperature=0, model_name="gpt-3.5-turbo-instruct")
    
    # 创建对话窗口记忆（只记住最近3次交互）
    memory = ConversationBufferWindowMemory(k=3)
    
    # 创建对话链
    conversation = ConversationChain(
        llm=llm,
        memory=memory,
        verbose=False
    )
    
    print("进行多轮对话（只保留最近3轮）:")
    
    # 进行5轮对话
    inputs = [
        "你好，我是李四。",
        "我是一名数据科学家。",
        "我喜欢分析数据。",
        "我擅长机器学习。",
        "请总结一下我们的对话。"
    ]
    
    for i, input_text in enumerate(inputs, 1):
        print(f"用户: {input_text}")
        response = conversation.predict(input=input_text)
        print(f"AI: {response}")
        print()
    
    print("当前记忆内容（只保留最近3轮）:")
    print(memory.buffer)
    print()

def conversation_summary_memory_example():
    """
    对话总结记忆示例
    """
    print("=== 对话总结记忆示例 ===")
    
    # 初始化 LLM
    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
    
    # 创建总结记忆
    memory = ConversationSummaryMemory(llm=llm, memory_key="chat_history", return_messages=True)
    
    # 创建对话链
    conversation = ConversationChain(
        llm=llm,
        memory=memory,
        verbose=False
    )
    
    print("进行长对话并自动生成总结:")
    
    inputs = [
        "你好，我是王五，我来自北京。",
        "我是一名产品经理，专注于AI产品。",
        "我在一家科技公司工作，已经5年了。",
        "我的兴趣是阅读和旅行。",
        "请告诉我你记住了什么关于我的信息。"
    ]
    
    for i, input_text in enumerate(inputs, 1):
        print(f"用户: {input_text}")
        response = conversation.predict(input=input_text)
        print(f"AI: {response}")
        print()
    
    print("对话总结:")
    print(memory.moving_summary_buffer)
    print()

def manual_memory_management_example():
    """
    手动记忆管理示例
    """
    print("=== 手动记忆管理示例 ===")
    
    # 创建记忆组件
    memory = ConversationBufferMemory(return_messages=True)
    
    # 手动添加消息到记忆
    memory.chat_memory.add_user_message("你好，我是赵六。")
    memory.chat_memory.add_ai_message("你好，赵六！有什么我可以帮你的吗？")
    
    print("手动添加的消息:")
    messages = memory.chat_memory.messages
    for msg in messages:
        msg_type = "用户" if isinstance(msg, HumanMessage) else "AI"
        print(f"  {msg_type}: {msg.content}")
    
    # 清除记忆
    memory.clear()
    print(f"\n清除后的记忆长度: {len(memory.chat_memory.messages)}")
    print()

def custom_memory_prompt_example():
    """
    自定义记忆提示词示例
    """
    print("=== 自定义记忆提示词示例 ===")
    
    # 初始化 LLM
    llm = ChatOpenAI(temperature=0.5, model_name="gpt-3.5-turbo")
    
    # 创建记忆
    memory = ConversationBufferMemory(input_key="input", memory_key="history")
    
    # 自定义提示词模板，包含记忆
    template = """你是一个有帮助的AI助手。请使用以下对话历史来回答用户的问题。

对话历史:
{history}

用户问题: {input}

AI助手回答:"""
    
    from langchain.chains import LLMChain
    from langchain_core.prompts import PromptTemplate
    
    prompt = PromptTemplate(input_variables=["history", "input"], template=template)
    conversation = LLMChain(prompt=prompt, llm=llm, memory=memory)
    
    print("使用自定义提示词的对话:")
    
    response1 = conversation.predict(input="你好，我叫钱七。")
    print(f"AI: {response1}")
    
    response2 = conversation.predict(input="你能告诉我你的名字吗？")
    print(f"AI: {response2}")
    
    response3 = conversation.predict(input="请总结一下我们的对话。")
    print(f"AI: {response3}")
    
    print("\n最终记忆内容:")
    print(memory.buffer)
    print()

if __name__ == "__main__":
    print("LangChain Memory 使用示例")
    print("="*60)
    
    try:
        conversation_buffer_memory_example()
        conversation_buffer_window_memory_example()
        conversation_summary_memory_example()
        manual_memory_management_example()
        custom_memory_prompt_example()
    except Exception as e:
        print(f"运行时出现错误: {e}")
        print("请确保已设置 OPENAI_API_KEY 环境变量")