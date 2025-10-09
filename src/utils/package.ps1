# package.ps1
# Windows PowerShell version of package.sh

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

# Cleanup Script - Clean temporary files and sensitive information
# Usage: .\package.ps1

# Color definitions (PowerShell doesn't have built-in color constants like bash)
$RED = "Red"
$GREEN = "Green"
$YELLOW = "Yellow"
$BLUE = "Blue"

# Print colored messages
function Write-Info {
    param([string]$Message)
    Write-Host "[INFO] $Message" -ForegroundColor $BLUE
}

function Write-Success {
    param([string]$Message)
    Write-Host "[SUCCESS] $Message" -ForegroundColor $GREEN
}

function Write-Warning {
    param([string]$Message)
    Write-Host "[WARNING] $Message" -ForegroundColor $YELLOW
}

function Write-Error {
    param([string]$Message)
    Write-Host "[ERROR] $Message" -ForegroundColor $RED
}

# Show usage instructions
function Show-Usage {
    Write-Host "Usage:"
    Write-Host "  .\package.ps1"
    Write-Host ""
    Write-Host "This script will:"
    Write-Host "  1. Remove user files and directories (user_*, long_term_memory, guest, etc.)"
    Write-Host "  2. Clean all __pycache__ directories"
    Write-Host "  3. Remove all .pyc and .pyo files"
    Write-Host "  4. Clean API keys from config/config.txt"
    Write-Host "  5. Remove build and agia.egg-info directories"
    Write-Host "  6. Remove .agia_last_output.json file"
    Write-Host "  7. Remove hidden files (starting with .) except important ones"
    Write-Host "  8. Remove other temporary files"
}

# Check parameters
if ($args[0] -eq "-h" -or $args[0] -eq "--help") {
    Show-Usage
    exit 0
}

# remove files
Write-Info "Remove files and directories..."

# Remove user_* directories/files if they exist
$userFiles = Get-ChildItem -Path . -Filter "user_*"
if ($userFiles) {
    $userFiles | Remove-Item -Recurse -Force
    Write-Success "Removed user_* files/directories"
} else {
    Write-Info "No user_* files/directories found"
}

# Remove long_term_memory directory if it exists
if (Test-Path "long_term_memory") {
    Remove-Item "long_term_memory" -Recurse -Force
    Write-Success "Removed long_term_memory directory"
} else {
    Write-Info "No long_term_memory directory found"
}

# Remove guest directory if it exists
if (Test-Path "guest") {
    Remove-Item "guest" -Recurse -Force
    Write-Success "Removed guest directory"
} else {
    Write-Info "No guest directory found"
}

# Remove .log files if they exist
$logFiles = Get-ChildItem -Path . -Filter "*.log"
if ($logFiles) {
    $logCount = $logFiles.Count
    $logFiles | Remove-Item -Force
    Write-Success "Removed $logCount .log files"
} else {
    Write-Info "No .log files found"
}

# Remove log directory if it exists
if (Test-Path "log") {
    Remove-Item "log" -Recurse -Force
    Write-Success "Removed log directory"
} else {
    Write-Info "No log directory found"
}

# Remove test* files if they exist
$testFiles = Get-ChildItem -Path . -Filter "test*"
if ($testFiles) {
    $testCount = $testFiles.Count
    $testFiles | Remove-Item -Force
    Write-Success "Removed $testCount test* files"
} else {
    Write-Info "No test* files found"
}

# Remove .out files if they exist
$outFiles = Get-ChildItem -Path . -Filter "*.out"
if ($outFiles) {
    $outCount = $outFiles.Count
    $outFiles | Remove-Item -Force
    Write-Success "Removed $outCount .out files"
} else {
    Write-Info "No .out files found"
}

# Remove workspace directory if it exists
if (Test-Path "workspace") {
    Remove-Item "workspace" -Recurse -Force
    Write-Success "Removed workspace directory"
} else {
    Write-Info "No workspace directory found"
}

Write-Info "Cleaning API keys from config/config.txt..."
if (Test-Path "config/config.txt") {
    # Create a backup
    Copy-Item "config/config.txt" "config/config.txt.tmp"

    # Read content and replace API keys
    $content = Get-Content "config/config.txt"
    $newContent = $content | ForEach-Object {
        # Replace api_key=xxx with api_key=your key, preserving comments and leading spaces
        if ($_ -match '^\s*#*\s*api_key\s*=') {
            $_ -replace '(^\s*#*\s*api_key\s*=\s*).*', '${1}your key'
        }
        # Replace mem_model_api_key=xxx with mem_model_api_key=your key, preserving comments and leading spaces
        elseif ($_ -match '^\s*#*\s*mem_model_api_key\s*=') {
            $_ -replace '(^\s*#*\s*mem_model_api_key\s*=\s*).*', '${1}your key'
        }
        # Replace embedding_model_api_key=xxx with embedding_model_api_key=your key, preserving comments and leading spaces
        elseif ($_ -match '^\s*#*\s*embedding_model_api_key\s*=') {
            $_ -replace '(^\s*#*\s*embedding_model_api_key\s*=\s*).*', '${1}your key'
        }
        else {
            $_
        }
    }
    $newContent | Set-Content "config/config.txt"

    # Remove the backup file
    Remove-Item "config/config.txt.tmp" -Force
    Write-Success "API keys replaced with placeholder in config/config.txt"

    # Check if there are still API key remnants
    $remainingKeys = Select-String -Path "config/config.txt" -Pattern '^\s*#*\s*(api_key|mem_model_api_key|embedding_model_api_key)\s*=\s*[^y][^o][^u][^r][\s]*key'
    if ($remainingKeys) {
        Write-Warning "Warning: There may still be API key remnants in config/config.txt"
    } else {
        Write-Success "All API key values have been replaced with placeholder in config/config.txt"
    }
} else {
    Write-Warning "config/config.txt file not found"
}

# Delete all __pycache__ directories
Write-Info "Deleting all __pycache__ directories..."
$pycacheDirs = Get-ChildItem -Path . -Recurse -Directory -Filter "__pycache__" -ErrorAction SilentlyContinue
$pycacheCount = $pycacheDirs.Count
if ($pycacheDirs) {
    $pycacheDirs | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
    Write-Success "Deleted $pycacheCount __pycache__ directories"
} else {
    Write-Info "No __pycache__ directories found"
}

# Delete .pyc files
Write-Info "Deleting all .pyc files..."
$pycFiles = Get-ChildItem -Path . -Recurse -Filter "*.pyc" -ErrorAction SilentlyContinue
$pycCount = $pycFiles.Count
if ($pycFiles) {
    $pycFiles | Remove-Item -Force -ErrorAction SilentlyContinue
    Write-Success "Deleted $pycCount .pyc files"
} else {
    Write-Info "No .pyc files found"
}

# Delete .pyo files
Write-Info "Deleting all .pyo files..."
$pyoFiles = Get-ChildItem -Path . -Recurse -Filter "*.pyo" -ErrorAction SilentlyContinue
$pyoCount = $pyoFiles.Count
if ($pyoFiles) {
    $pyoFiles | Remove-Item -Force -ErrorAction SilentlyContinue
    Write-Success "Deleted $pyoCount .pyo files"
} else {
    Write-Info "No .pyo files found"
}

# Delete other common temporary files and directories
Write-Info "Cleaning other temporary files..."

# Delete build directories
if (Test-Path "build") {
    Remove-Item "build" -Recurse -Force
    Write-Success "Deleted build directory"
} else {
    Write-Info "No build directory found"
}

# Delete agia.egg-info directory
if (Test-Path "agia.egg-info") {
    Remove-Item "agia.egg-info" -Recurse -Force
    Write-Success "Deleted agia.egg-info directory"
} else {
    Write-Info "No agia.egg-info directory found"
}

# Delete .agia_last_output.json file
if (Test-Path ".agia_last_output.json") {
    Remove-Item ".agia_last_output.json" -Force
    Write-Success "Deleted .agia_last_output.json file"
} else {
    Write-Info "No .agia_last_output.json file found"
}

# Delete .DS_Store files (macOS)
Get-ChildItem -Path . -Recurse -Filter ".DS_Store" -ErrorAction SilentlyContinue | Remove-Item -Force

# Delete Thumbs.db files (Windows)
Get-ChildItem -Path . -Recurse -Filter "Thumbs.db" -ErrorAction SilentlyContinue | Remove-Item -Force

# Delete temporary files
Get-ChildItem -Path . -Recurse -Filter "*.tmp" -ErrorAction SilentlyContinue | Remove-Item -Force
Get-ChildItem -Path . -Recurse -Filter "*.temp" -ErrorAction SilentlyContinue | Remove-Item -Force

# Delete hidden files (files starting with .) including in subdirectories but keep important ones
Write-Info "Deleting hidden files (starting with .) in all subdirectories..."
# List of important hidden files/directories to preserve
$preservePatterns = @(".git", ".gitignore", ".gitmodules", ".github", ".vscode", ".idea")

# Count all hidden files first
$hiddenFilesTotal = (Get-ChildItem -Path . -Recurse -File -Force -ErrorAction SilentlyContinue | Where-Object { $_.Name.StartsWith('.') }).Count

if ($hiddenFilesTotal -gt 0) {
    # Create array to store files to delete
    $filesToDelete = @()

    # Find all hidden files and categorize them
    $hiddenFiles = Get-ChildItem -Path . -Recurse -File -Force -ErrorAction SilentlyContinue | Where-Object { $_.Name.StartsWith('.') }

    $preservedCount = 0

    foreach ($file in $hiddenFiles) {
        $shouldPreserve = $false
        foreach ($pattern in $preservePatterns) {
            if ($file.Name -eq $pattern) {
                $shouldPreserve = $true
                $preservedCount++
                break
            }
        }

        if (-not $shouldPreserve) {
            $filesToDelete += $file.FullName
        }
    }

    # Count files to be deleted
    $deletedCount = $filesToDelete.Count

    # Delete files
    if ($deletedCount -gt 0) {
        foreach ($filePath in $filesToDelete) {
            Remove-Item $filePath -Force -ErrorAction SilentlyContinue
        }
        $preservedCount = $hiddenFilesTotal - $deletedCount
        Write-Success "Deleted $deletedCount hidden files from all subdirectories (preserved $preservedCount important files)"
    } else {
        Write-Info "No hidden files to delete (all are preserved)"
    }
} else {
    Write-Info "No hidden files found in any subdirectory"
}

# Delete log files (if logs directory exists but is empty or contains only temporary logs)
if (Test-Path "logs") {
    # Only delete .log files, preserve directory structure
    Get-ChildItem -Path "logs" -Recurse -Filter "*.log" -ErrorAction SilentlyContinue | Remove-Item -Force
    Write-Info "Cleaned log files from logs directory"
}

Write-Success "Temporary file cleanup completed"
Write-Success "All operations completed!"
Write-Host "[SUCCESS] All operations completed!" -ForegroundColor Green