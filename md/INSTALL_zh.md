# AGIAgent 安装指南

## 自动安装（推荐）

我们提供了一个自动安装脚本，支持 Linux 和 macOS 系统。

### 快速开始

```bash
# 进入项目目录
cd /path/to/AGIAgent

# 运行安装脚本
./install.sh
```

### 安装脚本功能

安装脚本会自动完成以下操作：

1. **检测操作系统** - 自动识别 Linux 或 macOS
2. **检查 Python 环境** - 确保 Python 3.8+ 已安装
3. **创建虚拟环境** - 创建独立的 Python 虚拟环境 (venv)
4. **安装 Python 依赖** - 从 requirements.txt 安装所有依赖包
5. **安装 Playwright Chromium** - 安装浏览器自动化工具
6. **安装 Pandoc** - 根据系统类型安装文档转换工具
   - Linux: 使用 apt-get/yum/pacman
   - macOS: 使用 Homebrew
7. **安装 XeLaTeX（可选）** - 用于高质量 PDF 生成，支持 Unicode 和中文
   - 会询问用户是否安装
   - Linux: 安装 texlive-xetex (~500MB)
   - macOS: 安装 BasicTeX (~100MB)

### 前置要求

#### 所有系统

- Python 3.8 或更高版本
- Git（用于克隆仓库）

#### macOS 特定

- [Homebrew](https://brew.sh/) - macOS 包管理器
  
  如果未安装，请运行：
  ```bash
  /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
  ```

#### Linux 特定

- sudo 权限（用于安装系统包）
- 包管理器：apt-get (Debian/Ubuntu) / yum (RedHat/CentOS) / pacman (Arch)

## 手动安装

如果自动安装脚本不适用于你的系统，可以按以下步骤手动安装：

### 1. 创建虚拟环境

```bash
python3 -m venv venv
```

### 2. 激活虚拟环境

```bash
# Linux/macOS
source venv/bin/activate

# Windows
venv\Scripts\activate
```

### 3. 升级 pip

```bash
python -m pip install --upgrade pip
```

### 4. 安装 Python 依赖

```bash
pip install -r requirements.txt
```

### 5. 安装 Playwright Chromium

```bash
playwright install chromium
playwright install-deps chromium  # 安装系统依赖（可能需要 sudo）
```

### 6. 安装 Pandoc

#### macOS

```bash
brew install pandoc
```

#### Ubuntu/Debian

```bash
sudo apt-get update
sudo apt-get install pandoc
```

#### CentOS/RHEL

```bash
sudo yum install pandoc
```

#### Arch Linux

```bash
sudo pacman -S pandoc
```

#### Windows

从 [Pandoc 官网](https://pandoc.org/installing.html) 下载安装包

### 7. 安装 XeLaTeX（可选，用于 PDF 生成）

XeLaTeX 是一个支持 Unicode 和现代字体的 TeX 引擎，用于生成高质量 PDF 文档。

#### macOS

```bash
# 安装 BasicTeX (较小的发行版，约 100MB)
brew install --cask basictex

# 更新 TeX Live Manager
sudo tlmgr update --self

# 安装基础的中文支持包（必需）
sudo tlmgr install xetex xecjk ctex fontspec

# 将 TeX 二进制文件添加到 PATH（可能需要重启终端）
export PATH="/Library/TeX/texbin:$PATH"
```

**如果需要使用完整的 LaTeX 模板功能**（包含精美页眉页脚、文档信息等），还需安装额外的包：

```bash
# 安装模板所需的额外包
sudo tlmgr install datetime2 tracklang fvextra adjustbox lastpage fancyhdr framed seqsplit xurl
```

**一键安装所有包**（推荐）：

```bash
# 完整安装命令（包含基础包和模板包）
sudo tlmgr install xecjk ctex fontspec datetime2 tracklang fvextra adjustbox lastpage fancyhdr framed seqsplit xurl
```

**或者安装完整版 MacTeX** (~4GB)，包含所有包:
```bash
brew install --cask mactex
```

**验证安装：**
```bash
# 检查 XeLaTeX
xelatex --version

# 可选：运行包检查脚本
cd /path/to/AGIAgent
./check_latex_packages.sh
```

#### Ubuntu/Debian

```bash
# 安装基础的 LaTeX 和中文支持（必需）
sudo apt-get update
sudo apt-get install -y texlive-xetex texlive-fonts-recommended texlive-fonts-extra

# 安装 Noto CJK 字体（Linux 推荐）
sudo apt-get install -y fonts-noto-cjk

# 安装模板所需的额外包
sudo tlmgr install datetime2 tracklang fvextra adjustbox lastpage fancyhdr framed seqsplit xurl
```

**注意：** 如果 `tlmgr` 命令不可用，可能需要先安装：
```bash
sudo apt-get install -y texlive-latex-extra texlive-lang-chinese
```

#### CentOS/RHEL

```bash
# 安装基础的 LaTeX 和中文支持
sudo yum install -y texlive-xetex texlive-collection-fontsrecommended

# 如果 tlmgr 可用，安装模板所需的额外包
sudo tlmgr install datetime2 tracklang fvextra adjustbox lastpage fancyhdr framed seqsplit xurl
```

#### Arch Linux

```bash
# 安装基础的 LaTeX 支持
sudo pacman -S texlive-core texlive-fontsextra

# 安装模板所需的额外包
sudo tlmgr install datetime2 tracklang fvextra adjustbox lastpage fancyhdr framed seqsplit xurl
```

#### Windows

**注意：** Windows 系统使用不同的 PDF 生成方式（通过 Word 打印），不需要安装 XeLaTeX。如果你确实需要 XeLaTeX：

1. 下载并安装 [MiKTeX](https://miktex.org/download) 或 [TeX Live](https://tug.org/texlive/windows.html)
2. MiKTeX 会在首次使用时自动安装缺少的包

**验证安装：**
```bash
xelatex --version
```

## 验证安装

安装完成后，验证所有组件是否正确安装：

```bash
# 激活虚拟环境
source venv/bin/activate

# 检查 Python 包
python -c "import playwright; print('Playwright OK')"

# 检查 Pandoc
pandoc --version

# 检查 XeLaTeX（如果已安装）
xelatex --version

# 退出虚拟环境
deactivate
```

## 配置系统

安装完成后，需要配置系统才能正常使用。

### 1. 配置 API Key

编辑 `config/config.txt` 文件，添加你的 API 密钥：

```bash
# 使用你喜欢的编辑器打开配置文件
nano config/config.txt
# 或者
vim config/config.txt
```

找到相应的 API 配置部分，取消注释并填入你的 API 密钥。例如使用 DeepSeek：

```
# DeepSeek API configuration
api_key=your-api-key-here
api_base=https://api.deepseek.com/v1
model=deepseek-chat
max_tokens=8192
```

**支持的模型提供商包括：**
- OpenAI (GPT-4, GPT-3.5)
- Anthropic (Claude)
- DeepSeek
- 智谱 AI (GLM)
- 阿里百炼 (Qwen)
- Google Gemini
- 火山引擎豆包
- Moonshot (Kimi)
- SiliconFlow
- Ollama (本地部署)
- OpenRouter

**配置步骤：**
1. 选择你要使用的模型提供商
2. 取消注释对应的配置行（删除行首的 `#`）
3. 将 `your key` 替换为你的实际 API 密钥
4. 如需要，修改 `api_base` 和 `model` 参数
5. 注释掉或删除其他不使用的配置

### 2. 配置语言选项

在 `config/config.txt` 文件的开头，设置系统语言：

**使用中文：**
```
# Language setting: en for English, zh for Chinese
# LANG=en
LANG=zh
```

**使用英文：**
```
# Language setting: en for English, zh for Chinese
LANG=en
# LANG=zh
```

### 3. 其他重要配置项

根据需要调整以下配置：

**流式输出：**
```
streaming=True  # True 为流式输出，False 为批量输出
```

**长期记忆：**
```
enable_long_term_memory=False  # 设置为 True 启用长期记忆功能
```

**多智能体模式：**
```
multi_agent=False  # 设置为 True 启用多智能体功能
```

**调试模式：**
```
enable_debug_system=False  # 设置为 True 启用增强调试功能
```

保存配置文件后，系统就可以正常使用了。

## 使用方法

### 启动 GUI 界面

```bash
# 激活虚拟环境
source venv/bin/activate

# 运行 GUI
python GUI/app.py
```

### 使用命令行

```bash
# 激活虚拟环境
source venv/bin/activate

# 运行主程序
python agia.py "write a poem"
```

### 作为 Python 库使用

```bash
# 激活虚拟环境
source venv/bin/activate

# 运行示例
python lib_demo.py
```

## 常见问题

### 1. Python 版本要求

**要求**: Python 3.8 或更高版本

如果你的 Python 版本过低，请安装 Python 3.8 或更高版本后再运行安装脚本。

### 2. Homebrew 未安装 (macOS)

**问题**: `未找到Homebrew包管理器`

**解决方案**: 安装 Homebrew

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

### 3. XeLaTeX 未找到或 PDF 生成失败

**问题**: PDF 生成时提示找不到 xelatex 或转换失败

**解决方案**:

XeLaTeX 是可选的，用于生成高质量 PDF。如果未安装：

1. 运行安装脚本时选择安装 XeLaTeX
2. 或手动安装（见上文第 7 步）
3. 安装后可能需要重启终端或更新 PATH：
   ```bash
   # macOS
   export PATH="/Library/TeX/texbin:$PATH"
   ```

如果不需要 PDF 功能，可以跳过 XeLaTeX 安装。

### 4. TeX Live 安装时间过长

**问题**: Linux 上安装 texlive-xetex 耗时很长

**解决方案**:

- TeX Live 完整安装包较大（~500MB），下载和安装需要时间
- 这是正常现象，请耐心等待
- 如果网络较慢，可以考虑使用国内镜像源

### 5. macOS 上 tlmgr 权限错误

**问题**: 运行 `tlmgr` 时提示权限不足

**解决方案**:

```bash
# 使用 sudo 运行 tlmgr 命令
sudo tlmgr update --self
sudo tlmgr install xetex xecjk fontspec
```

## 许可证

参见 [LICENSE](LICENSE) 文件。

