"""
LangChain 学习项目 - 主入口

此文件提供了一个交互式的界面来演示 LangChain 的各种功能。
运行此脚本可以选择不同的示例来学习 LangChain 的核心概念。
"""

import sys
import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

def check_api_keys():
    """
    检查API密钥配置
    """
    siliconflow_key = os.getenv("SILICONFLOW_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")
    
    print("API 密钥配置检查:")
    if siliconflow_key:
        print("✓ SiliconFlow API 密钥已配置")
    else:
        print("✗ SiliconFlow API 密钥未配置")
        
    if openai_key:
        print("✓ OpenAI API 密钥已配置 (作为备选)")
    else:
        print("? OpenAI API 密钥未配置 (可选)")
    print()

def run_example(module_path: str, example_name: str):
    """
    运行指定的示例模块
    
    Args:
        module_path: 模块路径
        example_name: 示例名称
    """
    print(f"\n{'='*20} 运行 {example_name} {'='*20}")
    try:
        # 导入并运行示例
        import importlib.util
        spec = importlib.util.spec_from_file_location("example", module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    except ImportError as e:
        print(f"导入错误: {e}")
    except Exception as e:
        print(f"运行错误: {e}")

def show_menu():
    """
    显示菜单选项
    """
    print("\nLangChain 学习项目 - 示例菜单")
    print("="*50)
    print("1. 模型使用示例")
    print("2. 提示词工程示例")
    print("3. 链(Chains)使用示例")
    print("4. 工具(Tools)使用示例")
    print("5. 记忆(Memory)使用示例")
    print("6. 智能体(Agents)使用示例")
    print("7. 综合示例")
    print("8. 运行所有示例")
    print("0. 退出")
    print("="*50)

def main():
    """
    主函数 - 提供交互式菜单
    """
    print("欢迎来到 LangChain 学习项目!")
    print("本项目支持 SiliconFlow 和 OpenAI 模型")
    
    # 检查API密钥配置
    check_api_keys()
    
    while True:
        show_menu()
        try:
            choice = input("\n请选择要运行的示例 (0-8): ").strip()
            
            if choice == "0":
                print("感谢使用 LangChain 学习项目!")
                break
            elif choice == "1":
                run_example("./src/models/basic_models.py", "模型使用示例")
            elif choice == "2":
                run_example("./src/prompts/prompt_engineering.py", "提示词工程示例")
            elif choice == "3":
                run_example("./src/chains/basic_chains.py", "链使用示例")
            elif choice == "4":
                run_example("./src/tools/basic_tools.py", "工具使用示例")
            elif choice == "5":
                run_example("./src/memory/basic_memory.py", "记忆使用示例")
            elif choice == "6":
                run_example("./src/agents/basic_agents.py", "智能体使用示例")
            elif choice == "7":
                run_example("./examples/comprehensive_example.py", "综合示例")
            elif choice == "8":
                print("\n运行所有示例...")
                
                examples = [
                    ("./src/models/basic_models.py", "模型使用示例"),
                    ("./src/prompts/prompt_engineering.py", "提示词工程示例"),
                    ("./src/chains/basic_chains.py", "链使用示例"),
                    ("./src/tools/basic_tools.py", "工具使用示例"),
                    ("./src/memory/basic_memory.py", "记忆使用示例"),
                    ("./src/agents/basic_agents.py", "智能体使用示例"),
                    ("./examples/comprehensive_example.py", "综合示例")
                ]
                
                for module_path, name in examples:
                    run_example(module_path, name)
                    
                print("\n所有示例运行完成!")
            else:
                print("无效选择，请输入 0-8 之间的数字。")
                
        except KeyboardInterrupt:
            print("\n\n程序被用户中断。")
            break
        except Exception as e:
            print(f"发生错误: {e}")

if __name__ == "__main__":
    main()