"""
SiliconFlow 与 LangChain 集成工具函数

本文件提供了在 LangChain 项目中使用 SiliconFlow API 的工具函数，
包括模型初始化、错误处理和配置管理。
"""

from langchain_openai import ChatOpenAI, OpenAI
import os
from dotenv import load_dotenv
import logging

# 加载环境变量
load_dotenv()

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_siliconflow_config():
    """
    获取 SiliconFlow 配置信息
    
    Returns:
        dict: 包含基础URL和API密钥的配置字典
    """
    base_url = os.getenv("SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1")
    api_key = os.getenv("SILICONFLOW_API_KEY")
    
    return {
        "base_url": base_url,
        "api_key": api_key
    }

def get_default_model_name():
    """
    获取默认的 SiliconFlow 模型名称
    
    Returns:
        str: 默认模型名称
    """
    return "Qwen/Qwen2.5-72B-Instruct"

def get_siliconflow_chat_model(
    model_name: str = None,
    temperature: float = 0.3,
    timeout: int = 30
):
    """
    获取 SiliconFlow 聊天模型实例
    
    Args:
        model_name: 模型名称，默认使用 get_default_model_name() 返回的模型
        temperature: 温度参数，控制输出随机性
        timeout: 请求超时时间（秒）
        
    Returns:
        ChatOpenAI: 配置好的聊天模型实例
    """
    config = get_siliconflow_config()
    
    if not config["api_key"]:
        logger.warning(
            "未设置 SILICONFLOW_API_KEY 环境变量，将尝试使用 OpenAI 模型作为备选"
        )
        # 返回 OpenAI 模型作为备选
        return ChatOpenAI(
            model_name=os.getenv("FALLBACK_MODEL_NAME", "gpt-3.5-turbo"),
            temperature=temperature,
            timeout=timeout
        )
    
    model_name = model_name or get_default_model_name()
    
    chat_model = ChatOpenAI(
        model_name=model_name,
        base_url=config["base_url"],
        api_key=config["api_key"],
        temperature=temperature,
        timeout=timeout
    )
    
    logger.info(f"使用 SiliconFlow 模型: {model_name}")
    return chat_model


def get_siliconflow_llm(
    model_name: str = None,
    temperature: float = 0.7,
    timeout: int = 30
):
    """
    获取 SiliconFlow LLM 实例
    
    Args:
        model_name: 模型名称，默认使用 get_default_model_name() 返回的模型
        temperature: 温度参数，控制输出随机性
        timeout: 请求超时时间（秒）
        
    Returns:
        OpenAI: 配置好的LLM实例
    """
    config = get_siliconflow_config()
    
    if not config["api_key"]:
        logger.warning(
            "未设置 SILICONFLOW_API_KEY 环境变量，将尝试使用 OpenAI 模型作为备选"
        )
        # 返回 OpenAI 模型作为备选
        return OpenAI(
            model_name=os.getenv("FALLBACK_LLM_MODEL_NAME", "gpt-3.5-turbo-instruct"),
            temperature=temperature,
            timeout=timeout
        )
    
    model_name = model_name or get_default_model_name()
    
    llm = OpenAI(
        model_name=model_name,
        base_url=config["base_url"],
        api_key=config["api_key"],
        temperature=temperature,
        timeout=timeout
    )
    
    logger.info(f"使用 SiliconFlow LLM 模型: {model_name}")
    return llm


def validate_siliconflow_config():
    """
    验证 SiliconFlow 配置是否正确
    
    Returns:
        bool: 配置是否有效
    """
    config = get_siliconflow_config()
    
    if not config["api_key"]:
        logger.error("SILICONFLOW_API_KEY 未设置")
        return False
    
    if not config["base_url"]:
        logger.error("SILICONFLOW_BASE_URL 未设置")
        return False
    
    logger.info("SiliconFlow 配置验证通过")
    return True


def test_connection():
    """
    测试 SiliconFlow 连接
    
    Returns:
        bool: 连接是否成功
    """
    try:
        if not validate_siliconflow_config():
            return False
        
        # 创建一个简单的聊天模型实例
        chat_model = get_siliconflow_chat_model(temperature=0)
        
        # 尝试进行一次简单的调用
        from langchain_core.messages import HumanMessage
        response = chat_model.invoke([HumanMessage(content="你好")])
        
        logger.info("SiliconFlow 连接测试成功")
        return True
        
    except Exception as e:
        logger.error(f"SiliconFlow 连接测试失败: {e}")
        return False


# 示例使用
if __name__ == "__main__":
    print("SiliconFlow 工具函数测试")
    print("="*40)
    
    # 检查配置
    print("配置验证:", validate_siliconflow_config())
    
    # 测试连接
    print("连接测试:", test_connection())
    
    # 获取模型实例
    try:
        chat_model = get_siliconflow_chat_model()
        print(f"聊天模型类型: {type(chat_model).__name__}")
        
        llm = get_siliconflow_llm()
        print(f"LLM模型类型: {type(llm).__name__}")
        
    except Exception as e:
        print(f"获取模型实例时出错: {e}")