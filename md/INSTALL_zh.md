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

## 验证安装

安装完成后，验证所有组件是否正确安装：

```bash
# 激活虚拟环境
source venv/bin/activate

# 检查 Python 包
python -c "import playwright; print('Playwright OK')"

# 检查 Pandoc
pandoc --version

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

## 许可证

参见 [LICENSE](LICENSE) 文件。

