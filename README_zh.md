# AGI Bot

**中文** | [**English**](README.md)

**AGI Bot** 是一个基于AI的智能代码生成和自主任务执行系统，能够自动分解复杂任务并通过多轮迭代与工具调用完成任务。

AGI Bot 的输入为用户提示词及一个工作目录，输出为编辑过的工作目录，在工作目录内部，大模型将代码、文档等输出文件放置在workspace文件夹。AGI Bot的执行过程为多轮迭代模式，每轮是大模型与工具交互的一个过程，首先，程序会将系统提示词、用户提示词、历史聊天记录、上一轮次的工具执行结果发送给大模型，大模型自行决定下一轮的工具调用，例如编写文件、搜索代码仓库、调用终端命令等，这些调用以XML格式描述，之后工具执行模块会解析这些工具调用并执行，执行结果将在下一轮传递给大模型。大模型认为本任务已经执行完毕后，会下发任务结束信号，则程序可选的进行任务总结。大模型的所有编辑操作都在用户定义的工作目录（dir）的workspace目录下完成。此外，为了控制上下文长度，如果超过了聊天上下文阈值会触发聊天历史总结。网络搜索结果可选的会进行总结。

AGI Bot 默认在终端下执行，对于有GUI需求或网络访问需求的用户，我们也提供了Web GUI。Web GUI中提供了任务执行所需要的所有功能，此外提供了文件的上传下载功能、文件预览功能、任务执行状态监视等功能。

由于AGI Bot定位为通用任务智能体，因此可能会调用系统终端命令，一般不会操作工作目录外的文件，大模型有时候会调用软件安装命令（pip，apt等），无论如何，请谨慎使用本软件，有必要时可以采用沙盒运行。

<div align="center">
      <img src="fig/AGIBot.png" alt="AGI Bot - L3级自主编程系统" width="800"/>
  
  **🚀 自主编程与任务执行系统**
  
  *基于LLM驱动的自主代码生成，具备智能任务分解和多轮迭代执行能力*
</div>

<br/>

## 🎬 演示视频

观看 AGI Bot 实际运行效果：

[![观看演示视频](https://img.youtube.com/vi/7kW_mH18YFM/0.jpg)](https://www.youtube.com/watch?v=7kW_mH18YFM)

## 🚀 立即体验

**在Google Colab中免费体验AGI Bot，无需任何配置！**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://drive.google.com/file/d/1eFtyTz1ictFBDDJFvI0viImfNvkTFOVc/view?usp=sharing)

*点击上方徽章，直接在浏览器中启动AGI Bot并开始体验自主AI编程。*

### 基本使用

#### 🔥 单任务模式（推荐）
适合Bug修复、功能优化等单一目标任务。如果没有指定`-r`参数，程序会提示您输入任务描述，支持多行复杂提示词。

```bash
python main.py --requirement "搜索今日新闻"
python main.py -r "写一个笑话"
```

#### 📋 任务分解模式
适合复杂的多步骤任务，系统会自动将大任务分解为子任务逐步执行。

```bash
python main.py --r "搜索今日新闻" --todo --requirement "开发一个完整的博客系统"
```

#### 💬 交互模式
提供更灵活的交互体验，系统会引导您输入任务需求。

```bash
python main.py -i
python main.py --interactive --todo
```

#### 📁 指定输出目录
自定义项目输出位置。如果不指定，系统会自动创建带时间戳的`output_`目录。

```bash
python main.py --dir "my_dir"
```

#### 🔄 继续执行任务
恢复之前的任务继续执行。AGI Bot会记住最后一次使用的输出目录。

```bash
python main.py -c
python main.py --continue
```

> **注意**：继续执行不会恢复聊天历史和之前的需求提示词，但可以继续操作工作目录中的文件。

#### ⚡ 设置执行轮数
控制任务执行的最大轮数，避免无限循环。

```bash
python main.py --loops 5 -r "需求描述"
python main.py -d "my_dir" -l 10 -r "需求描述"
```

> **说明**：轮数不等于模型调用次数。每轮通常调用一次大模型，但在聊天历史过长时会额外调用一次进行总结，任务完成后也可能进行总结。

#### 🔧 自定义模型配置
直接通过命令行指定API配置，但建议在`config.txt`中配置以便重复使用。

```bash
python main.py --api-key YOUR_KEY --model gpt-4 --api-base https://api.openai.com/v1
```

## 🎯 核心特性

- **🧠 智能任务分解**：AI自动将复杂需求分解为可执行子任务
- **🔄 多轮迭代执行**：每个任务支持多轮优化，确保质量（默认25轮）
- **🔍 智能代码搜索**：语义搜索 + 关键词搜索，快速定位代码
- **🌐 网络搜索集成**：实时网络搜索获取最新信息和解决方案
- **📚 代码库检索**：高级代码仓库分析和智能代码索引
- **🛠️ 丰富工具生态**：完备的本地工具+操作系统命令调用能力，支持完整开发流程
- **🖥️ Web界面**：直观的网页界面，实时执行监控
- **📊 双格式报告**：JSON详细日志 + Markdown可读报告
- **⚡ 实时反馈**：详细的执行进度和状态显示
- **🤝 交互式控制**：可选的用户确认模式，每步骤可控
- **📁 灵活输出**：自定义输出目录，自动时间戳命名新工程

## 🌐 网络搜索功能

AGI Bot 集成了强大的网络搜索功能，可以获取实时信息：

使用方式：在需求提示词中加入"搜索网页"则会进行搜索，"不要搜索网页"则不会搜索，如不注明则大模型会自行判断。

## 📚 代码库检索系统

AGI Bot配备了实时代码库的向量化和检索功能，在每轮工具调用结束后，会搜索新修改的文件，进行动态增量入库，并支持大模型的模糊语义检索能力。此外，大模型也可以调用grep等命令观察工作空间的情况。


## 🛠️ 工具库

AGI Bot 拥有全面的工具库：

### 文件系统工具
- **文件操作**：创建、读取、更新、删除文件和目录
- **目录管理**：导航和组织项目结构
- **文件搜索**：按名称、内容或模式查找文件

### 代码分析工具
- **语法分析**：解析和理解代码结构
- **依赖分析**：映射代码关系和导入
- **代码质量**：识别问题并提出改进建议

### 网络和网络工具
- **网络搜索**：实时信息检索
- **API测试**：测试和验证API端点
- **文档获取**：检索技术文档

### 终端和执行工具
- **命令执行**：运行系统命令和脚本
- **进程管理**：监控和控制运行中的进程
- **环境设置**：配置开发环境

### 开发工具
- **代码生成**：创建样板和模板代码
- **测试工具**：生成和运行测试用例
- **构建工具**：编译和打包应用程序

## 🖥️ Web GUI 界面

AGI Bot 提供现代化、直观的网页界面，提升用户体验：

### 主要功能
- **实时执行监控**：实时观察任务执行和详细日志
- **交互式任务管理**：通过网页界面启动、停止和监控任务
- **文件管理**：直接在浏览器中上传、下载和管理项目文件
- **目录操作**：创建、重命名和组织项目目录
- **多语言支持**：包含中英文界面，请在config.txt中配置语言

### 启动 GUI
```bash
cd GUI
python app.py

# 通过浏览器访问 http://localhost:5001
```
Web GUI会显示文件列表，默认带有workspace子目录的文件夹都会被列出，否则不会被列出。根目录位置可以在config.txt中配置。
注：目前Web GUI处于实验阶段，仅提供单用户开发版本（不适合工业部署）。


## 🤖 模型选择

AGI Bot 支持多种AI模型，满足不同用户需求和预算。选择最适合您需求的模型：

### 🌟 推荐模型

#### Claude Sonnet 4 (推荐使用)
**适合：需要高准确性和详细回答的复杂任务**
- ✅ **优点**：智能程度高、准确度高、回复详细
- ❌ **缺点**：价格较贵、速度一般、有时候有幻觉
- 💰 **价格**：高端定价
- 🎯 **使用场景**：复杂代码生成、详细分析、高级问题解决

#### OpenAI GPT-4.1
**适合：需要快速可靠性能的用户**
- ✅ **优点**：较准确、速度快
- ❌ **缺点**：价格较贵（但比Claude Sonnet 4便宜）
- 💰 **价格**：高端定价（比Claude更经济）
- 🎯 **使用场景**：通用开发任务、快速迭代、平衡性能需求

#### DeepSeek V3
**适合：注重成本和准确性的用户**
- ✅ **优点**：准确、无幻觉、谨慎、便宜
- ❌ **缺点**：输出内容较为概括、解释不够详细
- 💰 **价格**：经济实惠
- 🎯 **使用场景**：代码优化、错误修复、直接的实现任务

#### Qwen2.5-7B-Instruct (SiliconFlow)
**适合：免费试用用户和简单任务**
- ✅ **优点**：非常便宜（免费）、能用来做一些简单的任务
- ❌ **缺点**：经常搜索网络、策略选择有时候不够准确、任务处理效果一般
- 💰 **价格**：免费
- 🎯 **使用场景**：学习实验、基础代码生成、简单任务处理

### 💡 模型选择指南

| 模型 | 智能程度 | 响应速度 | 成本 | 最佳用途 |
|------|---------|---------|------|----------|
| Claude Sonnet 4 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | 💰💰💰💰 | 复杂项目 |
| GPT-4.1 | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | 💰💰💰 | 通用开发 |
| DeepSeek V3 | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | 💰💰 | 预算项目 |
| Qwen2.5-7B | ⭐⭐⭐ | ⭐⭐⭐ | 免费 | 简单任务 |

### 🔧 模型配置

在 `config.txt` 文件中配置您偏好的模型，或使用命令行参数：

```bash
# 使用 Claude Sonnet 4
python main.py --model claude-3-5-sonnet-20241022 --api-key your_key -r "您的任务"

# 使用 OpenAI GPT-4.1
python main.py --model gpt-4 --api-key your_key -r "您的任务"

# 使用 DeepSeek V3
python main.py --model deepseek-chat --api-base https://api.deepseek.com --api-key your_key -r "您的任务"

# 使用 SiliconFlow (免费)
python main.py --model Qwen/Qwen2.5-7B-Instruct --api-base https://api.siliconflow.cn/v1 --api-key your_free_key -r "您的任务"
```

## ⚙️ 配置文件 (config.txt)

AGI Bot 使用 `config.txt` 文件进行系统配置。以下是主要配置选项：

### API 配置
```ini
# 选择您偏好的AI服务提供商
# OpenAI API
api_key=your_openai_api_key
api_base=https://api.openai.com/v1
model=gpt-4

# Anthropic Claude
api_key=your_anthropic_api_key
api_base=https://api.anthropic.com
model=claude-3-sonnet-20240229

# 其他支持的提供商：SiliconFlow, DeepSeek, 火山引擎豆包, Ollama
```

### 语言设置
```ini
# 界面语言：en 为英文，zh 为中文
LANG=zh
```

### 输出控制
```ini
# 流式输出：True 为实时输出，False 为批量输出
streaming=True

# 简化搜索结果显示
simplified_search_output=True

# 生成摘要报告
summary_report=False
```

### 内容截断设置
```ini
# 主要工具结果截断长度（默认：10000字符）
truncation_length=10000

# 历史记录截断长度（默认：10000字符）
history_truncation_length=10000

# 网络内容截断长度（默认：50000字符）
web_content_truncation_length=50000
```

### 历史摘要功能
```ini
# 启用AI驱动的对话历史摘要
summary_history=True

# 摘要最大长度（默认：5000字符）
summary_max_length=5000

# 当历史记录超过此长度时触发摘要
summary_trigger_length=30000


# 静默模式
# 在sudo，pip等命令执行时加入-quiet或者-y标志，默认不开启
auto_fix_interactive_commands=False
# 注：如果希望大模型成功调用sudo，请采用非静默模式并手工输入sudo密码，或采用静默模式，并在sudoers文件授权静默同意某些程序（例如your_username ALL=(ALL) NOPASSWD: /usr/bin/apt-get），
```
### GUI 配置
```ini
# GUI文件管理的默认目录
gui_default_data_directory=~
```

## 🔧 环境要求与安装

### 系统要求
- **Python 3.8+**
- **网络连接**：用于API调用和网络搜索功能

### 安装步骤

#### 方式一：pip安装（推荐）

AGI Bot可以作为Python包直接安装：

```bash
# 从源码安装
pip install .

# 或者从git仓库安装（替换为实际的仓库地址）
pip install git+https://github.com/agi-hub/AGIBot.git

# 如果已发布到PyPI，可以直接安装
pip install agibot
```

安装完成后，可以通过命令行直接使用：

```bash
agibot --requirement "你的任务描述"
```