# LangChain 学习项目 - 快速入门

## 项目简介

这是一个全面的 LangChain 学习项目，涵盖了 LangChain 框架的所有核心概念和实际应用。项目支持 SiliconFlow、OpenAI 等多种模型服务。

## 环境准备

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

所需的主要依赖包：
- `langchain`: LangChain 核心库
- `langchain-community`: LangChain 社区组件
- `langchain-openai`: OpenAI 模型集成
- `python-dotenv`: 环境变量管理

### 2. 配置环境变量

复制 `.env` 文件并填入您的 API 密钥：

```bash
# .env 文件
# SiliconFlow API 配置
SILICONFLOW_API_KEY=your_siliconflow_api_key_here
SILICONFLOW_BASE_URL=https://api.siliconflow.cn/v1

# OpenAI API 配置 (可选，作为备选方案)
OPENAI_API_KEY=your_openai_api_key_here
```

#### SiliconFlow 配置说明

根据硅基流动官方文档，您需要：

1. 访问 [硅基流动官网](https://cloud.siliconflow.cn/)
2. 登录并创建 API 密钥
3. 在 `.env` 文件中设置 `SILICONFLOW_API_KEY`
4. 使用 SiliconFlow 支持的模型名称，如 `Qwen/Qwen2.5-72B-Instruct`

## 运行项目

### 方法一：交互式运行

```bash
python main.py
```

这将启动一个交互式菜单，您可以选择不同的示例来运行。

### 方法二：直接运行特定示例

```bash
# 运行模型示例
python src/models/basic_models.py

# 运行提示词工程示例
python src/prompts/prompt_engineering.py

# 运行链示例
python src/chains/basic_chains.py

# 运行工具示例
python src/tools/basic_tools.py

# 运行记忆示例
python src/memory/basic_memory.py

# 运行智能体示例
python src/agents/basic_agents.py

# 运行综合示例
python examples/comprehensive_example.py
```

## SiliconFlow 与 OpenAI 兼容性

本项目中的所有示例都支持 SiliconFlow API，它兼容 OpenAI 的接口规范。当配置了 `SILICONFLOW_API_KEY` 时，示例将优先使用 SiliconFlow 服务；否则会尝试使用 OpenAI 服务作为备选。

## 核心概念学习路径

### 1. 模型 (Models)
学习如何使用不同的语言模型：
- SiliconFlow 模型
- OpenAI 模型
- 聊天模型
- 流式响应

### 2. 提示词工程 (Prompt Engineering)
掌握提示词设计技巧：
- 提示词模板
- 消息模板
- 动态提示词

### 3. 链 (Chains)
了解链的概念和应用：
- LLMChain
- 序列链
- LCEL (LangChain Expression Language)

### 4. 工具 (Tools)
创建和使用自定义工具：
- 基础工具创建
- 参数验证
- 工具集成

### 5. 记忆 (Memory)
实现上下文记忆：
- 对话记忆
- 窗口记忆
- 总结记忆

### 6. 智能体 (Agents)
构建智能决策系统：
- ReAct 智能体
- 多工具协调
- 自定义智能体行为

## 学习建议

1. **按顺序学习**：建议按照上述学习路径顺序进行学习
2. **动手实践**：修改示例代码并观察结果变化
3. **查阅文档**：参考 `docs/learning_guide.md` 和 `docs/siliconflow_integration.md` 获取详细说明
4. **扩展应用**：基于示例创建自己的 LangChain 应用

## 常见问题

### Q: 运行时出现 API 密钥错误怎么办？
A: 请确认 `.env` 文件中的 `SILICONFLOW_API_KEY` 或 `OPENAI_API_KEY` 已正确设置，并且环境变量已加载。

### Q: 如何选择合适的模型？
A: 根据任务需求选择：
- SiliconFlow 模型：适合中文场景，支持多种开源模型
- OpenAI 模型：适合通用场景
- 聊天模型：适合对话应用

### Q: 如何优化性能？
A: 
- 使用流式响应处理长输出
- 合理设置 temperature 参数
- 考虑缓存机制减少重复请求

## 进一步学习

- 查阅 [LangChain 官方文档](https://python.langchain.com/)
- 参考 [SiliconFlow 集成指南](docs/siliconflow_integration.md)
- 参与 LangChain 社区讨论
- 尝试构建自己的实际应用场景