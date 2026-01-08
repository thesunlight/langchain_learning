# LangChain 学习项目

这是一个基于 LangChain 最新功能的全面学习项目，旨在帮助开发者理解和掌握 LangChain 框架的核心概念和实际应用。

## 项目特性

- 📘 **全面覆盖**: 涵盖 LangChain 的所有核心组件
- 💡 **实例丰富**: 包含大量实用示例和代码
- 📚 **文档完整**: 提供详细的学习文档和教程
- 🔧 **易于上手**: 清晰的项目结构和安装指南
- 🚀 **多平台支持**: 支持 SiliconFlow、OpenAI 等多种模型服务

## 核心组件

### 1. 模型 (Models)
- OpenAI 模型集成
- SiliconFlow 模型集成
- 聊天模型使用
- 流式响应处理

### 2. 提示词工程 (Prompt Engineering)
- 提示词模板设计
- 消息模板构建
- 动态提示词生成

### 3. 链 (Chains)
- LLMChain 基础应用
- 序列链组合
- LCEL (LangChain Expression Language)

### 4. 工具 (Tools)
- 自定义工具创建
- 工具参数验证
- 多工具协调使用

### 5. 记忆 (Memory)
- 对话记忆管理
- 窗口记忆机制
- 自动摘要功能

### 6. 智能体 (Agents)
- ReAct 智能体实现
- 工具使用决策
- 复杂任务处理

## 项目结构

```
langchain_learning/
├── src/                    # 源代码目录
│   ├── agents/            # 智能体相关代码
│   ├── chains/            # 链相关代码  
│   ├── memory/            # 记忆组件相关代码
│   ├── models/            # 模型相关代码
│   ├── prompts/           # 提示词工程相关代码
│   ├── tools/             # 工具相关代码
│   └── utils/             # 实用工具函数
├── examples/              # 完整示例应用
├── docs/                  # 学习文档
├── main.py                # 主入口文件
├── requirements.txt       # 项目依赖
├── .env                   # 环境变量配置
└── README.md              # 项目说明
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置环境变量

创建 `.env` 文件并填入 API 密钥：

```
# SiliconFlow API 配置
SILICONFLOW_API_KEY=your_siliconflow_api_key_here
SILICONFLOW_BASE_URL=https://api.siliconflow.cn/v1

# OpenAI API 配置 (可选，作为备选方案)
OPENAI_API_KEY=your_openai_api_key_here
```

#### SiliconFlow API 配置说明

根据硅基流动官方文档，您需要：

1. 访问 [硅基流动官网](https://cloud.siliconflow.cn/)
2. 登录并创建 API 密钥
3. 在 `.env` 文件中设置 `SILICONFLOW_API_KEY`
4. 使用 SiliconFlow 支持的模型名称，如 `Qwen/Qwen2.5-72B-Instruct`

### 3. 运行示例

```bash
# 交互式运行
python main.py

# 或运行特定示例
python src/models/basic_models.py
```

## 学习路径

建议按以下顺序学习：

1. **模型使用**: 了解如何使用不同的语言模型
2. **提示词工程**: 掌握提示词设计技巧
3. **链的构建**: 学习如何连接不同的组件
4. **工具创建**: 创建自定义功能工具
5. **记忆管理**: 实现上下文记忆功能
6. **智能体开发**: 构建自主决策系统

## 文档资源

- [学习指南](docs/learning_guide.md): 详细的学习路径和概念说明
- [快速入门](docs/quick_start.md): 快速上手指南
- 示例代码: 各组件的详细使用示例

## 贡献

欢迎提交 Issue 或 Pull Request 来改进此学习项目！
