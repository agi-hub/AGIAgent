#!/bin/bash
#
# Copyright (c) 2025 AGI Agent Research Group.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

# Cleanup Script - Clean temporary files and sensitive information
# Usage: ./package.sh

set -e  # Exit on error

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

# Show usage instructions
show_usage() {
    echo "Usage:"
    echo "  $0"
    echo ""
    echo "This script will:"
    echo "  1. Clean all __pycache__ directories"
    echo "  2. Remove all .pyc and .pyo files"
    echo "  3. Clean API keys from config/config.txt"
    echo "  4. Remove build and agia.egg-info directories"
    echo "  5. Remove .agia_last_output.json file"
    echo "  6. Remove hidden files (starting with .) except important ones"
    echo "  7. Remove other temporary files"
}

# Check parameters
if [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
    show_usage
    exit 0
fi

# Clean API keys from config/config.txt
print_info "Cleaning API keys from config/config.txt..."
if [ -f "config/config.txt" ]; then
    # 用 sed 将 api_key=xxx 替换为 api_key=your key，保留注释和行首空格
    sed -i.tmp -E 's/^([[:space:]]*#*[[:space:]]*api_key[[:space:]]*=[[:space:]]*).*/\1your key/' config/config.txt
    # 用 sed 将 mem_model_api_key=xxx 替换为 mem_model_api_key=your key，保留注释和行首空格
    sed -i.tmp -E 's/^([[:space:]]*#*[[:space:]]*mem_model_api_key[[:space:]]*=[[:space:]]*).*/\1your key/' config/config.txt
    # 用 sed 将 embedding_model_api_key=xxx 替换为 embedding_model_api_key=your key，保留注释和行首空格
    sed -i.tmp -E 's/^([[:space:]]*#*[[:space:]]*embedding_model_api_key[[:space:]]*=[[:space:]]*).*/\1your key/' config/config.txt
    rm -f config/config.txt.tmp
    print_success "API keys replaced with placeholder in config/config.txt"
    # 检查是否还有 api_key 内容残留
    if grep -qE '^([[:space:]]*#*[[:space:]]*(api_key|mem_model_api_key|embedding_model_api_key)[[:space:]]*=[[:space:]]*[^y][^o][^u][^r][[:space:]]*key)' config/config.txt; then
        print_warning "Warning: There may still be API key remnants in config/config.txt"
    else
        print_success "All API key values have been replaced with placeholder in config/config.txt"
    fi
else
    print_warning "config/config.txt file not found"
fi

# Delete all __pycache__ directories
print_info "Deleting all __pycache__ directories..."
PYCACHE_COUNT=$(find . -type d -name "__pycache__" | wc -l)
if [ "$PYCACHE_COUNT" -gt 0 ]; then
    find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
    print_success "Deleted $PYCACHE_COUNT __pycache__ directories"
else
    print_info "No __pycache__ directories found"
fi

# Delete .pyc files
print_info "Deleting all .pyc files..."
PYC_COUNT=$(find . -name "*.pyc" | wc -l)
if [ "$PYC_COUNT" -gt 0 ]; then
    find . -name "*.pyc" -delete
    print_success "Deleted $PYC_COUNT .pyc files"
else
    print_info "No .pyc files found"
fi

# Delete .pyo files
print_info "Deleting all .pyo files..."
PYO_COUNT=$(find . -name "*.pyo" | wc -l)
if [ "$PYO_COUNT" -gt 0 ]; then
    find . -name "*.pyo" -delete
    print_success "Deleted $PYO_COUNT .pyo files"
else
    print_info "No .pyo files found"
fi

# Delete other common temporary files and directories
print_info "Cleaning other temporary files..."

# Delete build directories
if [ -d "build" ]; then
    rm -rf build
    print_success "Deleted build directory"
else
    print_info "No build directory found"
fi

# Delete agia.egg-info directory
if [ -d "agia.egg-info" ]; then
    rm -rf agia.egg-info
    print_success "Deleted agia.egg-info directory"
else
    print_info "No agia.egg-info directory found"
fi

# Delete .agia_last_output.json file
if [ -f ".agia_last_output.json" ]; then
    rm -f .agia_last_output.json
    print_success "Deleted .agia_last_output.json file"
else
    print_info "No .agia_last_output.json file found"
fi

# Delete .DS_Store files (macOS)
find . -name ".DS_Store" -delete 2>/dev/null || true

# Delete Thumbs.db files (Windows)
find . -name "Thumbs.db" -delete 2>/dev/null || true

# Delete temporary files
find . -name "*.tmp" -delete 2>/dev/null || true
find . -name "*.temp" -delete 2>/dev/null || true

# Delete hidden files (files starting with .) including in subdirectories but keep important ones
print_info "Deleting hidden files (starting with .) in all subdirectories..."
# List of important hidden files/directories to preserve
PRESERVE_PATTERNS=(".git" ".gitignore" ".gitmodules" ".github" ".vscode" ".idea")

# Function to check if a file should be preserved
should_preserve() {
    local file="$1"
    local basename=$(basename "$file")
    for pattern in "${PRESERVE_PATTERNS[@]}"; do
        if [[ "$basename" == "$pattern" ]]; then
            return 0  # Should preserve
        fi
    done
    return 1  # Should not preserve
}

# Count all hidden files first
HIDDEN_FILES_TOTAL=$(find . -name ".*" -type f | wc -l)

if [ "$HIDDEN_FILES_TOTAL" -gt 0 ]; then
    # Create temporary file to store files to delete
    TEMP_FILE=$(mktemp)
    
    # Find all hidden files and categorize them
    PRESERVED_COUNT=0
    find . -name ".*" -type f | while read -r file; do
        if should_preserve "$file"; then
            # Count preserved files (this won't work in subshell, but we'll count differently)
            true
        else
            echo "$file" >> "$TEMP_FILE"
        fi
    done
    
    # Count files to be deleted
    if [ -f "$TEMP_FILE" ]; then
        DELETED_COUNT=$(wc -l < "$TEMP_FILE")
        
        # Delete files
        if [ "$DELETED_COUNT" -gt 0 ]; then
            while read -r file; do
                rm -f "$file" 2>/dev/null || true
            done < "$TEMP_FILE"
            
            PRESERVED_COUNT=$((HIDDEN_FILES_TOTAL - DELETED_COUNT))
            print_success "Deleted $DELETED_COUNT hidden files from all subdirectories (preserved $PRESERVED_COUNT important files)"
        else
            print_info "No hidden files to delete (all are preserved)"
        fi
        
        rm -f "$TEMP_FILE"
    else
        print_info "No hidden files to delete (all are preserved)"
    fi
else
    print_info "No hidden files found in any subdirectory"
fi

# Delete log files (if logs directory exists but is empty or contains only temporary logs)
if [ -d "logs" ]; then
    # Only delete .log files, preserve directory structure
    find logs -name "*.log" -delete 2>/dev/null || true
    print_info "Cleaned log files from logs directory"
fi

print_success "Temporary file cleanup completed"
print_success "All operations completed!" 