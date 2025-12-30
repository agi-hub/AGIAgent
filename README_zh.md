# AGI Agent

[**中文**](README_zh.md) | **English**

## 🚀 项目介绍
**AGI Agent** 是一个通用的多功能平台，包括 Vibe 文档、Vibe 编程和 Vibe 计算机执行。作为 Cursor 和 Manus 的开源替代方案，它提供 GUI 和 CLI 两种模式，可以部署在云端、笔记本电脑或嵌入式设备（ARM）上。该平台包含 20+ 内置工具和许多例程文件（技能），适用于广泛的使用场景。AGI Agent 擅长创建带有丰富图表的彩色文档，您可以直接在 GUI 中预览和编辑文档。您也可以像使用 Cursor 或 Claude Code 一样用它编写程序，支持多轮交互、拖放文件支持（或 @files），以及代理模式和计划模式。

### 🤔 这款软件适合您吗？

- **正在寻找 Cursor 的替代方案？** 如果您需要适用于小团队在线代码编写的云端可部署解决方案，AGI Agent 可能非常适合。
- **需要类似 Manus 的通用工具？** 如果您正在寻找一个能够处理 Word/PDF 输入并生成分析代码、图表和报告的多功能系统，不妨试试它。
- **编写复杂的专业文档？** 如果您需要创建带有丰富插图、复杂的专业报告，如学术论文、深度研究或专利，AGI Agent 在这方面表现出色。
- **寻求可本地部署的代理？** 如果您想要一个支持本地部署且兼容各种 Anthropic/OpenAI 接口模型的代理系统，这可能是您的解决方案。
- **Vibe 爱好者？** 如果您热衷于 Vibe 工作流程，您会喜欢 AGI Agent 提供的功能。


## GUI for Vibe Everything

[![观看演示视频](./md/images/AGIAgent_GUI.png)](https://www.youtube.com/watch?v=dsRfuH3s9Kk)


**AGI Agent** 遵循基于计划的 ReAct 模型来执行复杂任务。它采用多轮迭代工作机制，大模型可以在每一轮中调用工具并接收反馈结果。它用于根据用户需求更新工作区中的文件或通过工具改变外部环境。AGIAgent 可以自主调用各种 MCP 工具和操作系统工具，具有多代理协作、多级长期记忆和具身智能感知功能。它强调代理的通用性和自主决策能力。AGIAgent 广泛的操作系统支持、大模型支持和多种操作模式使其适合构建类人通用智能系统，以实现复杂的报告研究和生成、项目级代码编写、自动计算机操作、多代理研究（如竞争、辩论、协作）等应用。


<div align="center">
      <img src="md/images/AGIAgent.png" alt="AGI Agent - L3 自主编程系统" width="800"/>
</div>

## 🚀 新闻
2025/12/30 GUI 已更新，用于高效的 Vibe 编程、Vibe 文档、Vibe 研究、Vibe 一切 <https://agiagentonline.com>。

2025/10/27 AGIAgent 在线注册现已开放！点击 <https://agiagentonline.com> 右侧的注册按钮进行注册并开始使用。

2025/10/12 提供了 AGIAgent 用于生成带丰富图片的文章的介绍，详见 [colourfuldoc/ColourfulDoc.md](colourfuldoc/ColourfulDoc.md) 和 [colourfuldoc/ColourfulDoc_zh.md](colourfuldoc/ColourfulDoc_zh.md)（中文版）。

2025/10/10 Windows 安装包（在线/离线）已就绪！请查看 [发布页面](https://github.com/agi-hub/AGIAgent/releases/)。

2025/9/15 在线网站（中文版）已可用。访问 <https://agiagentonline.com>，无需 APIKey 即可登录，您可以找到许多示例。项目介绍主页：<https://agiagentonline.com/intro>（中文版）已可用。

2025/7/21 GUI 已可用，支持 markdown/PDF/源代码预览，支持 svg 图像编辑和 mermaid 编辑功能，访问 [GUI/README_GUI_en.md](GUI/README_GUI_en.md) 了解更多信息，相同的 GUI 已部署在 <https://agiagentonline.com>。

## ✨ 核心功能

### 🤖 自主多代理协作
- **自主代理创建**：系统可以自主决定创建新的专业代理，为每个代理配置独特的提示词、模型类型和专用工具库
- **角色专业化**：构建具有不同角色和专业知识的子代理，在共享工作区中高效协作
- **通信机制**：代理具有点对点和广播消息通信能力，集成邮件查看机制以实现无缝互联

### 🔧 广泛的工具调用能力
- **内置工具库**：集成 10+ 常用开发工具，包括文件检索、网页浏览和文件修改
- **MCP 协议支持**：支持模型上下文协议（MCP），用于连接数千个扩展工具，如 GitHub 和 Slack
- **系统集成**：全面支持终端命令、Python 包管理和操作系统软件包
- **自主安装**：代理可以根据任务需求自动安装系统软件、pip 包和 MCP 工具

### 🧠 长期记忆和学习
- **持久记忆**：通过存储历史执行摘要形成长期记忆，解决了传统代理只关注当前任务的局限性
- **智能检索**：通过 RAG（检索增强生成）提取有价值的历史记忆元素用于当前工作
- **上下文管理**：集成长上下文摘要机制，确保记忆的连续性和相关性

### 👁️ 具身智能和多模态
- **多模态感知**：内置视觉、传感器和其他多模态功能，不局限于文本世界
- **物理世界交互**：可以处理丰富的物理世界信息场景
- **多通道信息处理**：通过多代理架构实现并行信息感知和交互

### 🔗 灵活的部署方式
- **独立运行**：可以作为完整的自主系统独立运行
- **嵌入式集成**：可以作为 Python 组件嵌入到其他软件进程中
- **模块化设计**：使用构建块方法构建强大的智能系统
- **轻量级部署**：仅依赖少数核心库，软件包紧凑，系统兼容性强

## 🔄 工作原理

### 输入-输出机制
AGI Agent 接收**用户提示**和**工作目录**作为输入，输出**处理后的工作目录**。所有生成的代码、文档和其他文件都统一放置在工作区文件夹中。

### 多轮迭代过程
1. **任务分析阶段**：系统将用户提示、历史聊天记录和之前的工具执行结果发送给大模型
2. **决策阶段**：大模型自主决定下一轮的工具调用策略（文件写入、代码搜索、终端命令等）
3. **工具执行阶段**：工具执行模块解析并执行大模型的指令（支持 tool_call 和 JSON 格式）
4. **结果反馈阶段**：执行结果在下一轮传递给大模型，形成闭环反馈
5. **任务完成阶段**：大模型在确定任务完成时发出结束信号，并可选择生成任务摘要

### 智能优化功能
- **上下文管理**：当聊天历史超过阈值时自动触发历史摘要，以保持高效运行
- **网络搜索优化**：可以选择性摘要搜索结果以提取关键信息
- **安全边界**：所有编辑操作都限制在用户定义的工作目录内，确保系统安全

## ⚠️ 安全提示

作为通用任务代理，AGI Agent 具有调用系统终端命令的能力。虽然它通常不会在工作目录外操作文件，但大模型可能会执行软件安装命令（如 pip、apt 等）。使用时请注意：
- 仔细审查执行的命令
- 建议在沙箱环境中运行重要任务
- 定期备份重要数据

## 🌐 平台兼容性

### 操作系统支持
- ✅ **Linux** - 完全支持
- ✅ **Windows** - 完全支持  
- ✅ **MacOS** - 完全支持

### 大模型支持
- **Anthropic Claude** - Claude 3.5 Sonnet、Claude 3 Opus 等
- **OpenAI GPT** - GPT-4、GPT-4 Turbo、GPT-3.5 等
- **Google Gemini** - Gemini Pro、Gemini Ultra 等
- **国产模型** - Kimi K2、DeepSeek、火山大模型、Qwen3（8B 及以上）

### 接口和模式
- **API 接口**：支持 Anthropic 接口和 OpenAI 兼容接口
- **输出模式**：支持流式输出和批量输出
- **调用模式**：支持工具调用模式和传统聊天模式（工具调用模式效果更好）

### 运行时接口
- **终端模式**：纯命令行界面，适用于服务器和自动化场景
- **Python 库模式**：作为组件嵌入到其他 Python 应用程序中
- **Web 界面模式**：提供可视化操作体验的现代 Web 界面

### 交互模式
- **全自动模式**：完全自主执行，无需人工干预
- **交互模式**：支持用户确认和指导，提供更多控制

<br/>

### 📦 简易安装

安装非常简单。您可以使用 `install.sh` 进行一键安装。基本功能只需要 Python 3.8+ 环境。对于文档转换和 Mermaid 图像转换，需要 Playwright 和 LaTeX。对于基本功能，您只需要配置大模型 API。您不需要配置 Embedding 模型，因为代码包含内置的向量化代码检索功能。

### 基本使用

### GUI
```bash
python GUI/app.py

# 然后通过浏览器访问 http://localhost:5001
```
Web GUI 显示文件列表。默认列出包含工作区子目录的文件夹，否则不会显示。根目录位置可以在 config/config.txt 中配置。
注意：Web GUI 目前是实验性的，仅提供单用户开发版本（不适合工业部署）。


#### CLI
```bash
#### 新任务
python agia.py "写一个笑话" 
#### 📁 指定输出目录
python agia.py "写一个笑话" --dir "my_dir"
#### 🔄 继续任务执行
python agia.py -c
#### ⚡ 设置执行轮数
python agia.py --loops 5 -r "需求描述"
#### 🔧 自定义模型配置
python agia.py --api-key YOUR_KEY --model gpt-4 --api-base https://api.openai.com/v1
```

> **注意**： 
1. 继续执行只会恢复工作目录和最后一个需求提示，不会恢复大模型的上下文。

2. 可以通过命令行直接指定 API 配置，但建议在 `config/config.txt` 中配置以便重复使用。

## 🎯 核心功能

- **🧠 智能任务分解**：AI 自动将复杂需求分解为可执行的子任务
- **🔄 多轮迭代执行**：每个任务支持多轮优化以确保质量（默认 50 轮）
- **🔍 智能代码搜索**：语义搜索 + 关键词搜索，快速定位代码
- **🌐 网络搜索集成**：实时网络搜索获取最新信息和解决方案
- **📚 代码库检索**：高级代码仓库分析和智能代码索引
- **🛠️ 丰富的工具生态系统**：完整的本地工具 + 操作系统命令调用能力，支持完整的开发流程
- **🖼️ 图像输入支持**：使用 `[img=path]` 语法在需求中包含图像，支持 Claude 和 OpenAI 视觉模型
- **🔗 MCP 集成支持**：通过模型上下文协议集成外部工具，包括第三方服务如 AI 搜索
- **🖥️ Web 界面**：直观的 Web 界面，实时执行监控
- **📊 双格式报告**：JSON 详细日志 + Markdown 可读报告
- **⚡ 实时反馈**：详细的执行进度和状态显示
- **🤝 交互式控制**：可选的用户确认模式，逐步控制
- **📁 灵活输出**：自定义输出目录，新项目自动时间戳命名



## 🤖 模型选择

AGI Agent 支持各种主流 AI 模型，包括 Claude、GPT-4、DeepSeek V3、Kimi K2 等，满足不同用户需求和预算。支持流式/非流式、工具调用或基于聊天的工具接口、Anthropic/OpenAI API 兼容性。


**🎯 [查看详细模型选择指南 →](md/MODELS.md)**

### 快速推荐

- **🏆 质量优先**：Claude Sonnet 4.5 - 最佳智能和代码质量 
- **💰 性价比**：DeepSeek V3.2 / GLM-4.7 - 出色的性价比
- **🆓 本地部署**：Qwen3-30B-A3B / GLM-4.5-air - 简单任务

> 💡 **提示**：有关详细的模型比较、配置方法和性能优化建议，请参阅 [MODELS.md](md/MODELS.md)

## ⚙️ 配置文件

AGI Agent 使用 `config/config.txt` 和 `config/config_memory.txt` 文件进行系统配置。

### 快速配置
安装后，请配置以下基本选项：

```ini
# 必需配置：API 密钥和模型
api_key=your_api_key
api_base=the_api_base
model=claude-sonnet-4-0

# 语言设置
LANG=zh
```

> 💡 **提示**：有关详细配置选项、使用建议和故障排除，请参阅 [CONFIG.md](md/CONFIG.md)

## 🔧 环境要求和安装

### 系统要求
- **Python 3.8+**
- **网络连接**：用于 API 调用和网络搜索功能

### 安装步骤
我们建议使用 install.sh 进行自动安装。
如果您希望最小化安装，请遵循以下步骤：

```bash
# 从源码安装
pip install -r requirements.txt

# 安装网页抓取工具（如果需要网页抓取）
playwright install-deps
playwright install chromium
```

安装后，不要忘记在 config/config.txt 中配置 api key、api base、model 和语言设置 LANG=en 或 LANG=zh。

## 🔗 扩展功能

### 🐍 Python 库接口
AGI Agent 现在支持在代码中直接作为 Python 库调用，提供类似于 OpenAI Chat API 的编程接口。

**📖 [查看 Python 库使用指南 →](md/README_python_lib_zh.md)**

- 🐍 纯 Python 接口，无需命令行
- 💬 OpenAI 风格 API，易于集成
- 🔧 程序化配置，灵活控制
- 📊 详细的返回信息和状态

### 🔌 MCP 协议支持
支持模型上下文协议（MCP）与外部工具服务器通信，大大扩展了系统的工具生态系统。

**📖 [查看 MCP 集成指南 →](md/README_MCP_zh.md)**

- 🌐 标准化工具调用协议
- 🔧 支持官方和第三方 MCP 服务器
- 📁 文件系统、GitHub、Slack 等服务集成
- ⚡ 动态工具发现和注册

## 🚀 快速开始

**在 Google Colab 中免费体验 AGI Agent，无需配置！**

[![在 Colab 中打开](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1JttmqQxV8Yktl4zDmls1819BCnM0_zRE)

*点击上面的徽章直接在浏览器中启动 AGI Agent，开始体验自主 AI 编程。*

