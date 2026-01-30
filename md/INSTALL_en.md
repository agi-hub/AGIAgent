# AGIAgent Installation Guide

## Automatic Installation (Recommended)

We provide an automated installation script that supports both Linux and macOS systems.

### Quick Start

```bash
# Navigate to the project directory
cd /path/to/AGIAgent

# Run the installation script
./install.sh
```

### Installation Script Features

The installation script automatically performs the following tasks:

1. **Detect Operating System** - Automatically identifies Linux or macOS
2. **Check Python Environment** - Ensures Python 3.8+ is installed
3. **Create Virtual Environment** - Creates an isolated Python virtual environment (venv)
4. **Install Python Dependencies** - Installs all packages from requirements.txt
5. **Install Playwright Chromium** - Installs browser automation tools
6. **Install Pandoc** - Installs document converter based on system type
   - Linux: Uses apt-get/yum/pacman
   - macOS: Uses Homebrew
7. **Install XeLaTeX (Optional)** - For high-quality PDF generation with Unicode and CJK support
   - Prompts user for installation
   - Linux: Installs texlive-xetex (~500MB)
   - macOS: Installs BasicTeX (~100MB)

### Prerequisites

#### All Systems

- Python 3.8 or higher
- Git (for cloning the repository)

#### macOS Specific

- [Homebrew](https://brew.sh/) - macOS package manager
  
  If not installed, run:
  ```bash
  /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
  ```

#### Linux Specific

- sudo privileges (for installing system packages)
- Package manager: apt-get (Debian/Ubuntu) / yum (RedHat/CentOS) / pacman (Arch)

## Manual Installation

If the automatic installation script doesn't work for your system, you can install manually:

### 1. Create Virtual Environment

```bash
python3 -m venv venv
```

### 2. Activate Virtual Environment

```bash
# Linux/macOS
source venv/bin/activate

# Windows
venv\Scripts\activate
```

### 3. Upgrade pip

```bash
python -m pip install --upgrade pip
```

### 4. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 5. Install Playwright Chromium

```bash
playwright install chromium
playwright install-deps chromium  # Install system dependencies (may require sudo)
```

### 6. Install Pandoc

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

Download installer from [Pandoc official website](https://pandoc.org/installing.html)

### 7. Install XeLaTeX (Optional, for PDF Generation)

XeLaTeX is a TeX engine with Unicode and modern font support for generating high-quality PDF documents.

#### macOS

```bash
# Install BasicTeX (smaller distribution, ~100MB)
brew install --cask basictex

# Update TeX Live Manager
sudo tlmgr update --self

# Install basic CJK support packages (required)
sudo tlmgr install xetex xecjk ctex fontspec

# Add TeX binaries to PATH (may need to restart terminal)
export PATH="/Library/TeX/texbin:$PATH"
```

**If you need full LaTeX template functionality** (with enhanced headers, footers, document info, etc.), install additional packages:

```bash
# Install additional packages required by the template
sudo tlmgr install datetime2 tracklang fvextra adjustbox lastpage fancyhdr framed seqsplit xurl
```

**One-command installation** (recommended):

```bash
# Complete installation (includes both basic and template packages)
sudo tlmgr install xecjk ctex fontspec datetime2 tracklang fvextra adjustbox lastpage fancyhdr framed seqsplit xurl
```

**Or install the full MacTeX distribution** (~4GB), which includes all packages:
```bash
brew install --cask mactex
```

**Verify installation:**
```bash
# Check XeLaTeX
xelatex --version

# Optional: Run package check script
cd /path/to/AGIAgent
./check_latex_packages.sh
```

#### Ubuntu/Debian

```bash
# Install basic LaTeX and CJK support (required)
sudo apt-get update
sudo apt-get install -y texlive-xetex texlive-fonts-recommended texlive-fonts-extra

# Install Noto CJK fonts (recommended for Linux)
sudo apt-get install -y fonts-noto-cjk

# Install additional packages required by the template
sudo tlmgr install datetime2 tracklang fvextra adjustbox lastpage fancyhdr framed seqsplit xurl
```

**Note:** If `tlmgr` command is not available, you may need to install:
```bash
sudo apt-get install -y texlive-latex-extra texlive-lang-chinese
```

#### CentOS/RHEL

```bash
# Install basic LaTeX and CJK support
sudo yum install -y texlive-xetex texlive-collection-fontsrecommended

# If tlmgr is available, install additional packages required by the template
sudo tlmgr install datetime2 tracklang fvextra adjustbox lastpage fancyhdr framed seqsplit xurl
```

#### Arch Linux

```bash
# Install basic LaTeX support
sudo pacman -S texlive-core texlive-fontsextra

# Install additional packages required by the template
sudo tlmgr install datetime2 tracklang fvextra adjustbox lastpage fancyhdr framed seqsplit xurl
```

#### Windows

**Note:** Windows systems use a different PDF generation method (via Word printing) and do not require XeLaTeX installation. If you specifically need XeLaTeX:

1. Download and install [MiKTeX](https://miktex.org/download) or [TeX Live](https://tug.org/texlive/windows.html)
2. MiKTeX will automatically install missing packages on first use

**Verify installation:**
```bash
xelatex --version
```

## Verify Installation

After installation, verify that all components are correctly installed:

```bash
# Activate virtual environment
source venv/bin/activate

# Check Python packages
python -c "import playwright; print('Playwright OK')"

# Check Pandoc
pandoc --version

# Check XeLaTeX (if installed)
xelatex --version

# Deactivate virtual environment
deactivate
```

## Configuration

After installation, you need to configure the system before use.

### 1. Configure API Key

Edit the `config/config.txt` file to add your API key:

```bash
# Open the configuration file with your preferred editor
nano config/config.txt
# or
vim config/config.txt
```

Find the corresponding API configuration section, uncomment and fill in your API key. For example, using DeepSeek:

```
# DeepSeek API configuration
api_key=your-api-key-here
api_base=https://api.deepseek.com/v1
model=deepseek-chat
max_tokens=8192
```

**Supported model providers include:**
- OpenAI (GPT-4, GPT-3.5)
- Anthropic (Claude)
- DeepSeek
- Zhipu AI (GLM)
- Alibaba Bailian (Qwen)
- Google Gemini
- Volcengine Doubao
- Moonshot (Kimi)
- SiliconFlow
- Ollama (local deployment)
- OpenRouter

**Configuration steps:**
1. Choose your model provider
2. Uncomment the corresponding configuration lines (remove `#` at the beginning)
3. Replace `your key` with your actual API key
4. Modify `api_base` and `model` parameters if needed
5. Comment out or delete other unused configurations

### 2. Configure Language Option

At the beginning of the `config/config.txt` file, set the system language:

**Use Chinese:**
```
# Language setting: en for English, zh for Chinese
# LANG=en
LANG=zh
```

**Use English:**
```
# Language setting: en for English, zh for Chinese
LANG=en
# LANG=zh
```

### 3. Other Important Configuration Options

Adjust the following configurations as needed:

**Streaming output:**
```
streaming=True  # True for streaming, False for batch output
```

**Long-term memory:**
```
enable_long_term_memory=False  # Set to True to enable long-term memory
```

**Multi-agent mode:**
```
multi_agent=False  # Set to True to enable multi-agent functionality
```

**Debug mode:**
```
enable_debug_system=False  # Set to True to enable enhanced debugging
```

After saving the configuration file, the system is ready to use.

## Usage

### Launch GUI Interface

```bash
# Activate virtual environment
source venv/bin/activate

# Run GUI
python GUI/app.py
```

### Use Command Line

```bash
# Activate virtual environment
source venv/bin/activate

# Run main program
python agia.py "write a poem"

```

### Use as Python Library

```bash
# Activate virtual environment
source venv/bin/activate

# Run demo
python lib_demo.py
```

## Troubleshooting

### 1. Python Version Requirement

**Requirement**: Python 3.8 or higher

If your Python version is too old, please install Python 3.8 or higher before running the installation script.

### 2. Homebrew Not Installed (macOS)

**Issue**: `Homebrew not found`

**Solution**: Install Homebrew

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

### 3. XeLaTeX Not Found or PDF Generation Failed

**Issue**: PDF generation fails with xelatex not found error

**Solution**:

XeLaTeX is optional and used for high-quality PDF generation. If not installed:

1. Choose to install XeLaTeX when prompted by the installation script
2. Or install manually (see step 7 above)
3. After installation, you may need to restart your terminal or update PATH:
   ```bash
   # macOS
   export PATH="/Library/TeX/texbin:$PATH"
   ```

If you don't need PDF functionality, you can skip XeLaTeX installation.

### 4. TeX Live Installation Takes Too Long

**Issue**: Installing texlive-xetex on Linux takes a long time

**Solution**:

- TeX Live complete package is large (~500MB), download and installation takes time
- This is normal, please be patient
- If network is slow, consider using a mirror server

### 5. tlmgr Permission Error on macOS

**Issue**: Permission denied when running `tlmgr`

**Solution**:

```bash
# Run tlmgr commands with sudo
sudo tlmgr update --self
sudo tlmgr install xetex xecjk fontspec
```

### 6. Playwright Browser Download Failed

**Issue**: Network issues preventing browser download

**Solution**: Set proxy or use Playwright mirror

```bash
# Set mirror (for users in China)
export PLAYWRIGHT_DOWNLOAD_HOST=https://playwright.azureedge.net
playwright install chromium
```

## License

See the [LICENSE](LICENSE) file.

