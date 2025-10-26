#!/bin/bash

# Color definitions
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print colored messages
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Detect operating system
detect_os() {
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        OS="linux"
        print_info "Detected OS: Linux"
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        OS="mac"
        print_info "Detected OS: macOS"
    else
        print_error "Unsupported operating system: $OSTYPE"
        exit 1
    fi
}

# Check if Python is installed
check_python() {
    print_info "Checking Python installation..."
    
    if command -v python3 &> /dev/null; then
        PYTHON_CMD="python3"
        PYTHON_VERSION=$($PYTHON_CMD --version 2>&1 | awk '{print $2}')
        print_success "Found Python: $PYTHON_VERSION"
    else
        print_error "Python3 not found. Please install Python 3.8 or higher first"
        exit 1
    fi
    
    # Check Python version
    PYTHON_MAJOR=$($PYTHON_CMD -c 'import sys; print(sys.version_info[0])')
    PYTHON_MINOR=$($PYTHON_CMD -c 'import sys; print(sys.version_info[1])')
    
    if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 8 ]); then
        print_error "Python 3.8 or higher is required. Current version: $PYTHON_VERSION"
        exit 1
    fi
}

# Create virtual environment
create_venv() {
    print_info "Creating virtual environment..."
    
    if [ -d "venv" ]; then
        print_warning "Virtual environment already exists. Delete and recreate? (y/n)"
        read -r response
        if [[ "$response" =~ ^[Yy]$ ]]; then
            print_info "Removing existing virtual environment..."
            rm -rf venv
        else
            print_info "Using existing virtual environment"
            return 0
        fi
    fi
    
    $PYTHON_CMD -m venv venv
    if [ $? -eq 0 ]; then
        print_success "Virtual environment created successfully"
    else
        print_error "Failed to create virtual environment"
        exit 1
    fi
}

# Activate virtual environment
activate_venv() {
    print_info "Activating virtual environment..."
    
    if [ -f "venv/bin/activate" ]; then
        source venv/bin/activate
        print_success "Virtual environment activated"
    else
        print_error "Cannot find virtual environment activation script"
        exit 1
    fi
}

# Upgrade pip
upgrade_pip() {
    print_info "Upgrading pip..."
    python -m pip install --upgrade pip
    if [ $? -eq 0 ]; then
        print_success "pip upgraded successfully"
    else
        print_warning "pip upgrade failed, continuing with installation..."
    fi
}

# Install Python dependencies
install_python_deps() {
    print_info "Installing Python dependencies..."
    
    if [ ! -f "requirements.txt" ]; then
        print_error "requirements.txt file not found"
        exit 1
    fi
    
    pip install -r requirements.txt
    if [ $? -eq 0 ]; then
        print_success "Python dependencies installed successfully"
    else
        print_error "Failed to install Python dependencies"
        exit 1
    fi
}

# Install Playwright Chromium
install_playwright() {
    print_info "Installing Playwright Chromium browser..."
    
    playwright install chromium
    if [ $? -eq 0 ]; then
        print_success "Playwright Chromium installed successfully"
    else
        print_error "Failed to install Playwright Chromium"
        exit 1
    fi
    
    # Install system dependencies (if needed)
    print_info "Installing Playwright system dependencies..."
    playwright install-deps chromium 2>/dev/null || print_warning "Some system dependencies may need to be installed manually (may require sudo privileges)"
}

# Install Pandoc (Linux)
install_pandoc_linux() {
    print_info "Installing Pandoc on Linux..."
    
    if command -v pandoc &> /dev/null; then
        PANDOC_VERSION=$(pandoc --version | head -n 1)
        print_warning "Pandoc is already installed: $PANDOC_VERSION"
        print_info "Do you want to reinstall? (y/n)"
        read -r response
        if [[ ! "$response" =~ ^[Yy]$ ]]; then
            return 0
        fi
    fi
    
    # Check for sudo privileges
    print_info "Installing Pandoc requires sudo privileges"
    
    # Detect Linux distribution
    if [ -f /etc/debian_version ]; then
        # Debian/Ubuntu
        print_info "Detected Debian/Ubuntu system, installing with apt-get..."
        sudo apt-get update
        sudo apt-get install -y pandoc
    elif [ -f /etc/redhat-release ]; then
        # RedHat/CentOS/Fedora
        print_info "Detected RedHat/CentOS/Fedora system, installing with yum..."
        sudo yum install -y pandoc
    elif [ -f /etc/arch-release ]; then
        # Arch Linux
        print_info "Detected Arch Linux system, installing with pacman..."
        sudo pacman -S --noconfirm pandoc
    else
        print_warning "Unable to auto-detect Linux distribution, trying apt-get..."
        sudo apt-get update
        sudo apt-get install -y pandoc
    fi
    
    if command -v pandoc &> /dev/null; then
        print_success "Pandoc installed successfully: $(pandoc --version | head -n 1)"
    else
        print_error "Failed to install Pandoc, please install manually"
        print_info "Visit https://pandoc.org/installing.html for installation guide"
        exit 1
    fi
}

# Install Pandoc (Mac)
install_pandoc_mac() {
    print_info "Installing Pandoc on macOS..."
    
    if command -v pandoc &> /dev/null; then
        PANDOC_VERSION=$(pandoc --version | head -n 1)
        print_warning "Pandoc is already installed: $PANDOC_VERSION"
        print_info "Do you want to reinstall? (y/n)"
        read -r response
        if [[ ! "$response" =~ ^[Yy]$ ]]; then
            return 0
        fi
    fi
    
    # Check if Homebrew is installed
    if ! command -v brew &> /dev/null; then
        print_error "Homebrew package manager not found"
        print_info "Please install Homebrew first: https://brew.sh/"
        print_info "Or install Pandoc manually: https://pandoc.org/installing.html"
        exit 1
    fi
    
    print_info "Installing Pandoc with Homebrew..."
    brew install pandoc
    
    if command -v pandoc &> /dev/null; then
        print_success "Pandoc installed successfully: $(pandoc --version | head -n 1)"
    else
        print_error "Failed to install Pandoc, please install manually"
        print_info "Visit https://pandoc.org/installing.html for installation guide"
        exit 1
    fi
}

# Install Pandoc
install_pandoc() {
    if [ "$OS" == "linux" ]; then
        install_pandoc_linux
    elif [ "$OS" == "mac" ]; then
        install_pandoc_mac
    fi
}

# Verify installation
verify_installation() {
    print_info "Verifying installation..."
    
    # Check virtual environment
    if [ -d "venv" ]; then
        print_success "✓ Virtual environment exists"
    else
        print_error "✗ Virtual environment does not exist"
    fi
    
    # Check Python packages
    if python -c "import playwright" 2>/dev/null; then
        print_success "✓ Playwright is installed"
    else
        print_error "✗ Playwright is not installed"
    fi
    
    # Check Pandoc
    if command -v pandoc &> /dev/null; then
        print_success "✓ Pandoc is installed: $(pandoc --version | head -n 1)"
    else
        print_error "✗ Pandoc is not installed"
    fi
    
    echo ""
    print_success "Installation verification complete!"
}

# Print usage instructions
print_usage() {
    echo ""
    echo "=========================================="
    print_success "Installation Complete!"
    echo "=========================================="
    echo ""
    echo "Usage:"
    echo "1. Activate virtual environment:"
    echo -e "   ${GREEN}source venv/bin/activate${NC}"
    echo ""
    echo "2. Please config the API key in config/config.txt"
    echo ""
    echo "3. Run your application GUI:"
    echo -e "   ${GREEN}python GUI/app.py${NC}"
    echo ""
    echo "or run your application CLI:"
    echo -e "   ${GREEN}python agia.py \"write a poem\"${NC}"
    echo ""    
    echo "4. Deactivate virtual environment:"
    echo -e "   ${GREEN}deactivate${NC}"
    echo ""
}

# Main function
main() {
    echo "=========================================="
    echo "   AGIAgent Automated Installation Script"
    echo "=========================================="
    echo ""
    
    # Detect operating system
    detect_os
    
    # Check Python
    check_python
    
    # Create virtual environment
    create_venv
    
    # Activate virtual environment
    activate_venv
    
    # Upgrade pip
    upgrade_pip
    
    # Install Python dependencies
    install_python_deps
    
    # Install Playwright
    install_playwright
    
    # Deactivate virtual environment to install system packages
    deactivate 2>/dev/null || true
    
    # Install Pandoc
    install_pandoc
    
    # Reactivate virtual environment for verification
    activate_venv
    
    # Verify installation
    verify_installation
    
    # Print usage instructions
    print_usage
}

# Run main function
main

