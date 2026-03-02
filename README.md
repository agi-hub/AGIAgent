# AGI Agent

<p align="center">
  <a href="https://github.com/agi-hub/AGIAgent/releases/tag/v1.0.3"><img src="https://img.shields.io/github/v/release/agi-hub/AGIAgent?include_prereleases&style=for-the-badge&logo=github&logoColor=white&color=181717&label=Releases" alt="GitHub release"></a>
  <a href="https://agiagentonline.com"><img src="https://img.shields.io/badge/AGIAgent_Online-agiagentonline.com-1E90FF?style=for-the-badge&logo=googlechrome&logoColor=white" alt="AGIAgent Online"></a>
  <a href="http://colordoc.online"><img src="https://img.shields.io/badge/ColorDoc-colordoc.online-FF6B35?style=for-the-badge&logo=googlechrome&logoColor=white" alt="ColorDoc"></a>
  <a href="https://github.com/agi-hub/AGIAgent/wiki/AGIAgent-%E6%8A%80%E6%9C%AF%E7%99%BD%E7%9A%AE%E4%B9%A6"><img src="https://img.shields.io/badge/技术白皮书-Wiki-8A2BE2?style=for-the-badge&logo=gitbook&logoColor=white" alt="技术白皮书 Wiki"></a>
  <a href="https://github.com/user-attachments/files/25679954/AGI.Agent.intro.web.version.pdf"><img src="https://img.shields.io/badge/介绍PPT-PDF-E63946?style=for-the-badge&logo=adobeacrobatreader&logoColor=white" alt="介绍PPT"></a>
  <a href="#联系我们"><img src="https://img.shields.io/badge/微信公众号-AGI智创机器人-07C160?style=for-the-badge&logo=wechat&logoColor=white" alt="微信公众号"></a>
</p>

[**English**](README_en.md)

## 🚀 项目介绍
**AGI Agent** 是一个通用的智能体平台，可以实现氛围文档撰写（Vibe Colorful Doc）、Vibe Coding和基于自然语言的通用任务执行。
类似于 Claude cowork，AGI Agent 是一个通用化的本地智能体操作系统，能够自主操作您的计算机，通过自然语言交互处理复杂任务。平台包含 40+ 内置工具和许多例程文件（skills），适用于广泛的使用场景。AGI Agent 擅长创建带有丰富图表的专业文档，并可以直接在 GUI 中预览和编辑文档。也可以用它编写程序，支持多轮交互、拖放文件等（@files）。
提供 GUI 和 CLI 、嵌入式运行等模式，可以部署在云端、笔记本电脑或嵌入式设备（ARM）上。支持Anthropic/OpenAI大模型接口，支持开源/私有化部署。

<div align="center">
  <img src="md/images/bridview_diagram.png" alt="原理总图" width="800">
</div>

### 🤔 这款软件适合您吗？

- **正在寻找开源的 Claude cowork？** AGI Agent 提供类似的协作式 AI 体验，让您能够与智能体协作，智能体可以理解您的需求、操作本地环境并自主执行复杂任务。
- **需要通用化的本地智能体？** 如果您想要一个能够在本地机器上处理多样化任务的智能体系统——从代码编写到文档生成，从数据分析到系统操作——AGI Agent 正是为您设计的。
- **编写复杂的专业文档？** 如果您需要创建带有丰富插图、复杂的专业报告，如学术论文、深度研究或专利，AGI Agent 表现的会让你满意（[参考介绍](https://github.com/agi-hub/ColorDoc/)）;
- **寻求可本地部署的代理？** 如果您想要一个支持本地部署且兼容各种 Anthropic/OpenAI 接口模型的代理系统，这可能是您的解决方案;
- **Vibe 爱好者？** 如果您热衷于 Vibe 工作流程，您会喜欢 AGI Agent。

### 🆚 与 OpenClaw 的对比

虽然 AGI Agent 与 OpenClaw 在逻辑上都具有智能体内核、记忆系统、通用任务执行能力、长程执行能力和本地部署能力，但 AGI Agent 从零开始构建，不依赖于 OpenClaw 的任何组件，并具有以下关键优势：

- **🤖 强调智能体本身**：AGI Agent 本身是一个集成了众多工具的多智能体系统，且提示词、工具调用等各个方面都可以自己定制（位于 prompts 文件夹），自带 IDE（网页版本）。
- **🛠️ 强化生产力工具**：AGI Agent 主要用于生产力任务，如写专业文档、编写程序、整理数据等，并提供了强大的文档/图像编辑能力，包括图文文档的 Word/PDF 无损直接导出、网页小程序直接执行等。
- **🀄 强化中文处理**：对 Mermaid、SVG 图像等环节的中文进行了优化，生成的图像中文显示效果出众，界面支持中/英文切换。
- **👁️ 可实时观察智能体工作**：流式输出到网页前端，可观察大模型智能体的实时处理过程。
- **🇨🇳 强调与中国大模型的配合**：适配了多数国产大模型，支持各类大模型接口（streaming/non-streaming、tool-call/message、OpenAI/Claude）。

### 🆚 与 Claude Cowork 的对比

虽然 AGI Agent 提供与 Claude cowork 类似的协作式 AI 体验，但它具有以下关键优势：

- **🏠 完全可本地化**：AGI Agent 可以完全安装在您的本地机器上运行，让您完全控制自己的数据和环境，无需依赖云服务。
- **🔌 通用模型支持**：与 Claude cowork 仅限于 Claude 模型不同，AGI Agent 支持任何主流大语言模型，包括 Claude、GPT-4、DeepSeek V3、Kimi K2、GLM、Qwen 等，通过标准的 Anthropic/OpenAI API 接口接入。
- **💻 跨平台兼容性**：完全支持 Windows、Linux 和 macOS，让您可以在任何您喜欢的操作系统上使用 AGI Agent。
- **📖 100% 开源**：提供完整的源代码，实现透明度、可定制性和社区驱动的改进，无供应商锁定。
- **⚙️ 无需 Claude Code 作为底层**：从零开始构建的独立架构，AGI Agent 不需要 Claude Code 作为底层依赖，提供更大的灵活性和控制权。

## Vibe Demo 
<div align="center">

<a href="https://www.youtube.com/watch?v=dsRfuH3s9Kk"><img src="./md/images/AGIAgent_GUI.png" alt="观看演示视频" width="800"></a> 

鼠标单击打开Youtube视频

<a href="https://youtu.be/OfP0tCyMUFE"><img src="./md/images/AGIAgent_GUI_zh.png" alt="功能介绍（中文）" width="800"></a> 

鼠标单击打开Youtube视频 （[视频中国备用链接](https://www.bilibili.com/video/BV1ez6nBmEU3?t=2.2)）

</div>


### 📺 演示视频（单击播放）

**📝 图文写作**

<div align="center">

| [![AI行研报告写作4款软件横评](https://github.com/user-attachments/assets/2b3dc9ff-6244-4d9e-ab2e-a31ad554d35c)](https://www.bilibili.com/video/BV1eQPuzVEmJ/) | [![免费的专利AI辅助撰写神器](https://github.com/user-attachments/assets/00ecfd39-a74d-4d9e-9454-e0b03f28cf19)](https://www.bilibili.com/video/BV1RSPuzyEfB/) |
|:---:|:---:|
| AI行研报告写作4款软件横评 | 免费的专利AI辅助撰写神器 |
| [![国家项目申请书撰写](https://github.com/user-attachments/assets/9863ebfb-4d84-4d69-8fc5-0f4717d2d9c0)](https://www.bilibili.com/video/BV1iUPuzLEpM/) | [![AGI Agent：开源智能体图文写作案例](https://github.com/user-attachments/assets/babe2c1d-6306-4fc4-b0f9-52de614fb8fc)](https://www.bilibili.com/video/BV1wmUTB5EMN/) |
| 国家项目申请书撰写 | AGI Agent：开源智能体图文写作案例 |
| [![AGI Agent深度图文写作案例-房价走势分析](https://github.com/user-attachments/assets/3ec7e837-ce89-44cf-99af-a24407eb4e5f)](https://www.bilibili.com/video/BV1NzUTBSE1q/) | |
| AGI Agent深度图文写作案例-房价走势分析 | |

</div>

**🤖 通用智能体介绍**

<div align="center">

| [![适合中国宝宝体质的自主智能体AGI Agent](https://github.com/user-attachments/assets/080be6ba-0de0-4a2d-9c08-565be89e7a1d)](https://www.bilibili.com/video/BV1ez6nBmEU3/) | [![AGI Agent 登陆由纯智能体机器人组建的Moltbook社区](https://github.com/user-attachments/assets/291ed2a0-cc06-4b8c-b4d3-0c25a2841ffb)](https://www.bilibili.com/video/BV1WXFKz3EXQ/) |
|:---:|:---:|
| 适合中国宝宝体质的自主智能体AGI Agent | AGI Agent 登陆Moltbook社区 |

</div>

**🧑‍🤝‍🧑 多智能体功能**

<div align="center">

| [![多智能体辩论赛](https://github.com/user-attachments/assets/66610d97-c5af-4667-9618-c8232ca40d9c)](https://www.bilibili.com/video/BV1iUPuzLE3N/) |
|:---:|
| 多智能体辩论赛 |

</div>

**💻 编程介绍**

<div align="center">

| [![游戏编程-合金弹头](https://github.com/user-attachments/assets/6594a016-8478-4a3b-8869-bb7a1769b32b)](https://www.bilibili.com/video/BV1KJUMBpEah/) |
|:---:|
| 游戏编程-合金弹头 |

</div>

### 📹 更多的功能演示视频（鼠标单击打开演示视频）

<div align="center">

| | |
|:---:|:---:|
| <a href="https://agiagentonline.com/colordocintro/videos/专业深度图文报告.mp4"><img src="https://agiagentonline.com/colordocintro/assets/img/专业深度图文报告.png" width="380" alt="专业深度图文报告"></a> | <a href="https://agiagentonline.com/colordocintro/videos/写专利交底书.mp4"><img src="https://agiagentonline.com/colordocintro/assets/img/写专利交底书.png" width="380" alt="写专利交底书"></a> |
| <a href="https://agiagentonline.com/colordocintro/videos/写国家项目申请书.mp4"><img src="https://agiagentonline.com/colordocintro/assets/img/写国家项目申请书.png" width="380" alt="写国家项目申请书"></a> | <a href="https://agiagentonline.com/colordocintro/videos/写图文博客、小红书.mp4"><img src="https://agiagentonline.com/colordocintro/assets/img/写图文博客、小红书.png" width="380" alt="写图文博客、小红书"></a> |
| <a href="https://agiagentonline.com/colordocintro/videos/分析用户数据、绘制图表.mp4"><img src="https://agiagentonline.com/colordocintro/assets/img/分析用户数据、绘制图表.png" width="380" alt="分析用户数据、绘制图表"></a> | <a href="https://agiagentonline.com/colordocintro/videos/报告-Agent发展趋势.mp4"><img src="https://agiagentonline.com/colordocintro/assets/img/报告-Agent发展趋势.png" width="380" alt="报告-Agent发展趋势"></a> |
| <a href="https://agiagentonline.com/colordocintro/videos/矢量图像绘制及多格式图像输出.mp4"><img src="https://agiagentonline.com/colordocintro/assets/img/矢量图像绘制及多格式图像输出.png" width="380" alt="矢量图像绘制及多格式图像输出"></a> | <a href="https://agiagentonline.com/example-results-records/izhikevich_neuron_visualization.html"><img src="https://agiagentonline.com/example-results-records/python%E7%A8%8B%E5%BA%8F%E7%BB%98%E5%88%B6%E5%9B%BE%E5%83%8F.png" width="380" alt="python程序绘制图像"></a> |
| <a href="https://agiagentonline.com/example-results-records/leshan_buddha_travel_guide.html"><img src="https://agiagentonline.com/example-results-records/%E5%86%99%E4%B8%AA%E7%BD%91%E9%A1%B5%E4%BB%8B%E7%BB%8D%E4%B9%90%E5%B1%B1%E5%A4%A7%E4%BD%9B.png" width="380" alt="写个网页介绍乐山大佛"></a> | <a href="https://agiagentonline.com/example-results-records/lucky_wheel_lottery.html"><img src="https://agiagentonline.com/example-results-records/%E6%8A%BD%E5%A5%96%E8%BD%AC%E7%9B%98.png" width="380" alt="抽奖转盘"></a> |
| <a href="https://agiagentonline.com/example-results-records/maze-game.html"><img src="https://agiagentonline.com/example-results-records/%E5%86%99%E4%B8%AA%E6%89%BE%E5%A6%88%E5%A6%88%E7%9A%84%E5%B0%8F%E6%B8%B8%E6%88%8F.png" width="380" alt="写个找妈妈的小游戏"></a> | |

</div>


## 📚 作品展示

以下是使用 AGI Agent 生成的部分作品示例，展示了平台在各类文档生成任务中的能力：

### 🔬 技术研究报告（鼠标单击下载查看）
- [大模型及智能体对国内外就业的影响及未来趋势综述](https://agiagentonline.com/colordocintro/assets/examples/大模型及智能体对国内外就业的影响及未来趋势综述.docx)
- [2026年AI_Agent发展趋势报告](https://agiagentonline.com/colordocintro/assets/examples/2026年AI_Agent发展趋势报告.docx)
- [大模型推理芯片发展趋势报告](https://agiagentonline.com/colordocintro/assets/examples/大模型推理芯片发展趋势报告.docx)
- [06-大模型推理芯片未来展望与发展路径](https://agiagentonline.com/colordocintro/assets/examples/06-大模型推理芯片未来展望与发展路径.docx)
- [手机大模型推理芯片未来趋势_完整报告](https://agiagentonline.com/colordocintro/assets/examples/手机大模型推理芯片未来趋势_完整报告.docx)
- [大模型辅助金融投资的现状与未来_完整报告](https://agiagentonline.com/colordocintro/assets/examples/大模型辅助金融投资的现状与未来_完整报告.docx)
- [类脑持续学习算法综述](https://agiagentonline.com/colordocintro/assets/examples/类脑持续学习算法综述.docx)
- [浅析马斯克的脑机接口技术-完整报告](https://agiagentonline.com/colordocintro/assets/examples/浅析马斯克的脑机接口技术-完整报告.docx)
- [01_国内外发展现状](https://agiagentonline.com/colordocintro/assets/examples/01_国内外发展现状.docx)
- [色彩学完整介绍](https://agiagentonline.com/colordocintro/assets/examples/色彩学完整介绍.docx)

### 📊 社会调研报告
- [中国偏远山区儿童教育情况研究报告](https://agiagentonline.com/colordocintro/assets/examples/中国偏远山区儿童教育情况研究报告.docx)
- [中国生育率调研报告-完整版](https://agiagentonline.com/colordocintro/assets/examples/中国生育率调研报告-完整版.docx)
- [中国老龄化趋势下智能化养老研究报告](https://agiagentonline.com/colordocintro/assets/examples/中国老龄化趋势下智能化养老研究报告.docx)
- [青少年游戏成瘾情况分析报告](https://agiagentonline.com/colordocintro/assets/examples/青少年游戏成瘾情况分析报告.docx)

### 📝 项目申请书与专利
- [量子中继与星地一体化通信技术攻关项目申请书](https://agiagentonline.com/colordocintro/assets/examples/量子中继与星地一体化通信技术攻关项目申请书.docx)
- [量子通信数据稳定性增强技术专利交底书](https://agiagentonline.com/colordocintro/assets/examples/量子通信数据稳定性增强技术专利交底书.docx)

### 🗺️ 旅游攻略与介绍
- [三亚旅游攻略](https://agiagentonline.com/colordocintro/assets/examples/sanya_travel_guide.docx)
- [汕头旅游攻略](https://agiagentonline.com/colordocintro/assets/examples/汕头旅游攻略.docx)
- [乌镇历史介绍](https://agiagentonline.com/colordocintro/assets/examples/乌镇历史介绍.docx)
- [北京市鼓楼到珠市口公交路线](https://agiagentonline.com/colordocintro/assets/examples/北京市鼓楼到珠市口公交路线.docx)

### 🎬 其他作品
- [疯狂动物城2_尼克的视角](https://agiagentonline.com/colordocintro/assets/examples/疯狂动物城2_尼克的视角.docx)
- [疯狂动物城爆火的深层原因_完整报告](https://agiagentonline.com/colordocintro/assets/examples/疯狂动物城爆火的深层原因_完整报告.docx)
- [皮卡丘图片搜索报告](https://agiagentonline.com/colordocintro/assets/examples/皮卡丘图片搜索报告.docx)

## 🚀 新闻
2026/3/2 微信公众号"AGI智创机器人"已就绪，关注+留言可以联系我们。

2026/1/15 GUI支持多平台，如果您希望使用专业图文写作，请选择ColorDoc（彩文）平台，如果您想写专利，可以使用专利写作助手，各个平台的技能（功能）不一样哦～

2025/12/30 GUI 已更新，用于高效的 Vibe 编程、Vibe 文档、Vibe 研究、Vibe任何东西 <https://agiagentonline.com>。

2025/10/27 AGIAgent 在线注册现已开放！点击 <https://agiagentonline.com> 右侧的注册按钮进行注册并开始使用。

2025/10/12 提供了 AGIAgent 用于生成带丰富图片的文章的介绍，详见 [apps/colordoc/md/README.md](apps/colordoc/md/README.md) 和 [apps/colordoc/md/README_zh.md](apps/colordoc/md/README_zh.md)（中文版）。

2025/10/10 Windows 安装包（在线/离线）已就绪！请查看 [发布页面](https://github.com/agi-hub/AGIAgent/releases/)。

2025/9/15 在线网站（中文版）已可用。访问 <https://agiagentonline.com>，无需 APIKey 即可登录，您可以找到许多示例。项目介绍主页：<https://agiagentonline.com/intro>（中文版）已可用。

2025/7/21 GUI 已可用，支持 markdown/PDF/源代码预览，支持 svg 图像编辑和 mermaid 编辑功能，访问 [GUI/README_GUI_en.md](GUI/README_GUI_en.md) 了解更多信息，相同的 GUI 已部署在 <https://agiagentonline.com>。

## AGI Agent原理介绍

**AGI Agent** 遵循基于计划的 ReAct 模型来执行复杂任务。它采用多轮迭代工作机制，大模型可以在每一轮中调用工具并接收反馈结果。它用于根据用户需求更新工作区中的文件或通过工具改变外部环境。AGIAgent 可以自主调用各种 MCP 工具和操作系统工具，具有多代理协作、多级长期记忆和具身智能感知功能。它强调代理的通用性和自主决策能力。AGIAgent 广泛的操作系统支持、大模型支持和多种操作模式使其适合构建类人通用智能系统，以实现复杂的报告研究和生成、项目级代码编写、自动计算机操作、多代理研究（如竞争、辩论、协作）等应用。


<div align="center">
      <img src="md/images/AGIAgent.png" alt="AGI Agent - L3 自主编程系统" width="800"/>
</div>

## 🏗️ 核心技术架构

### 自主多智能体架构
AGIAgent 采用 **Manager + 多子 Agent** 的协作架构：Manager（老板）负责发起并管理多个子 Agent（如码工、具身机器人、艺术工作者等），每个子 Agent 在独立线程中运行，可独立完成任务，也可相互协作或竞争。

- **高度自主**：自组队、自配置、自监控、自销毁
- **协作通信**：每个智能体具备邮箱，支持点对点及广播消息通信
- **灵活配置**：每个智能体可单独配置提示词、模型、工具库（MCP）、工作区
- **智能体自主创建**：Agent 可依据任务需求自主决定创建新的子 Agent，以应对复杂任务

大模型可调用的智能体相关工具包括：启动智能体、发送消息给智能体、发送广播消息、汇报工作状态、查看活跃智能体、终止智能体。

### 基于 ReAct 的执行引擎
遵循 **Plan → Act → Observe → Reflect** 循环，大模型在每轮中调用工具并接收反馈，支持多轮迭代优化（默认 50 轮）。内置**渐进式历史压缩**机制，突破上下文长度限制，实现真正的长程任务执行。

### 双层记忆架构
- **短期记忆**：当前任务上下文与工具调用历史
- **长期记忆**：跨任务知识积累，支持语义检索 + 关键词混合检索，无需配置外部 Embedding 模型（内置向量化检索）

### 无限睡眠唤醒机制
针对长程任务设计的"节能模式"：Agent 在等待外部事件时进入休眠，条件满足后自动唤醒继续执行，支持跨会话任务恢复。

### 三层工具生态
1. **内置工具**（40+）：文件读写、代码执行、网络搜索、图像处理、文档转换等
2. **OS 工具**：直接调用系统终端命令，支持 pip/apt 等包管理
3. **MCP 工具**：通过模型上下文协议动态接入第三方服务（GitHub、Slack、文件系统等）

### 多格式文档输出
支持 Markdown / Word / PDF / LaTeX 无损转换，内置 Mermaid 图表渲染与 SVG 矢量图编辑，中文显示经过专项优化。

> 📖 详见 [AGIAgent 技术白皮书](https://github.com/agi-hub/AGIAgent/wiki/AGIAgent-%E6%8A%80%E6%9C%AF%E7%99%BD%E7%9A%AE%E4%B9%A6)

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

## 🌐 平台兼容性

### 操作系统支持
- ✅ **Linux** - 完全支持
- ✅ **Windows** - 完全支持  
- ✅ **MacOS** - 完全支持

### 运行时接口
- **终端模式**：纯命令行界面，适用于服务器和自动化场景
- **Python 库模式**：作为组件嵌入到其他 Python 应用程序中
- **Web 界面模式**：提供可视化操作体验的现代 Web 界面

### 交互模式
- **全自动模式**：完全自主执行，无需人工干预
- **交互模式**：支持用户确认和指导，提供更多控制

<br/>

## 🔧 环境要求和安装

### 系统要求
- **Python 3.8+**
- **网络连接**：用于 API 调用和网络搜索功能

### 安装步骤

推荐使用 `install.sh` 一键安装（需要 Python 3.8+）。最小化安装：

```bash
pip install -r requirements.txt

# 可选：网页抓取 / 文档转换 / Mermaid 图像
playwright install-deps && playwright install chromium
```

安装后在 `config/config.txt` 中配置 `api_key`、`api_base`、`model` 和 `LANG=en/zh`。

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

### 🛠️ 外部 Skill 文件支持

系统支持外部的 skill 文件（例程/技能）。系统默认的 skill 存放在 `routine` 文件夹中，并会显示在 GUI 界面的技能栏（中文界面显示 `routine_zh` 中的内容，英文界面显示 `routine` 中的内容）。您也可以指定自定义的 skill 目录：

```bash
#### 🧩 使用自定义 skill 目录
python agia.py "<your requirements>" --routine routine_file_dir
```

## ⚠️ 安全提示

作为通用任务代理，AGI Agent 具有调用系统终端命令的能力。虽然它通常不会在工作目录外操作文件，但大模型可能会执行软件安装命令（如 pip、apt 等）。使用时请注意：
- 仔细审查执行的命令
- 建议在沙箱环境中运行重要任务
- 定期备份重要数据

## 🎛️ 灵活的智能体自定义

### 🤖 自定义 Agent 工具集

系统支持灵活调整 Agent 所使用的工具集：

- 系统默认加载 `prompts/tool_prompts.json` 中定义的工具
- 可将需要启用的工具从 `prompts/additional_tools.json` 的对应字段移入 `prompts/tool_prompts.json`
- 不需要的工具可以从 `prompts/tool_prompts.json` 移回 `prompts/additional_tools.json`，以减少 Agent 的工具数量，降低 token 消耗

### 📝 自定义提示词

系统的各类提示词均可自由编辑定制，文件位于 `prompts/` 目录下：

| 文件 | 说明 | 生效模式 |
|------|------|----------|
| `prompts/system_prompts.txt` | 系统主提示词 | Agent 模式 |
| `prompts/rules_prompt.txt` | 工具调用规则提示词 | Agent 模式 |
| `prompts/user_rules.txt` | 额外用户需求提示词 | Agent 模式 |
| `prompts/system_plan_prompt.txt` | 计划模式系统提示词 | Plan 模式 |

在 Agent 模式下，系统会加载 `system_prompts.txt`、`rules_prompt.txt` 和 `user_rules.txt`；在 Plan 模式下，系统会加载 `system_plan_prompt.txt`。您可以直接编辑这些文件来调整 Agent 的行为和风格。

### 🔌 MCP 协议支持
支持模型上下文协议（MCP）与外部工具服务器通信，大大扩展了系统的工具生态系统。

**📖 [查看 MCP 集成指南 →](md/README_MCP_zh.md)**

- 🌐 标准化工具调用协议
- 🔧 支持官方和第三方 MCP 服务器
- 📁 文件系统、GitHub、Slack 等服务集成
- ⚡ 动态工具发现和注册

## 🔗 扩展功能

### 🐍 Python 库接口
AGI Agent 现在支持在代码中直接作为 Python 库调用，提供类似于 OpenAI Chat API 的编程接口。

**📖 [查看 Python 库使用指南 →](md/README_python_lib_zh.md)**

- 🐍 纯 Python 接口，无需命令行
- 💬 OpenAI 风格 API，易于集成
- 🔧 程序化配置，灵活控制
- 📊 详细的返回信息和状态

## 🚀 快速开始

**在 Google Colab 中体验 AGI Agent**

[![在 Colab 中打开](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1JttmqQxV8Yktl4zDmls1819BCnM0_zRE)

*点击上面的徽章直接在浏览器中启动 AGI Agent，开始体验自主 AI 编程。*

## 联系我们
您可以通过提交 Issue 来提交问题或建议。如需进一步沟通，您可以发送邮件至 bitcursor@2925.com，或关注微信公众号"AGI智创机器人"并留言联系我们。
