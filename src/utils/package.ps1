# package.ps1
# Windows PowerShell version of package.sh

Write-Host "[INFO] Cleaning API keys from config/config.txt..." -ForegroundColor Blue
if (Test-Path "config/config.txt") {
    $content = Get-Content "config/config.txt"
    $newContent = $content | ForEach-Object {
        if ($_ -match '^\s*#*\s*api_key\s*=') {
            # 保留注释和行首空格，只替换等号后的内容
            $_ -replace '(^\s*#*\s*api_key\s*=\s*).*', '${1}your key'
        } elseif ($_ -match '^\s*#*\s*mem_model_api_key\s*=') {
            # 保留注释和行首空格，只替换等号后的内容
            $_ -replace '(^\s*#*\s*mem_model_api_key\s*=\s*).*', '${1}your key'
        } elseif ($_ -match '^\s*#*\s*embedding_model_api_key\s*=') {
            # 保留注释和行首空格，只替换等号后的内容
            $_ -replace '(^\s*#*\s*embedding_model_api_key\s*=\s*).*', '${1}your key'
        } else {
            $_
        }
    }
    $newContent | Set-Content "config/config.txt"
    Write-Host "[SUCCESS] API keys replaced with placeholder in config/config.txt" -ForegroundColor Green
    if (Select-String -Path "config/config.txt" -Pattern '^\s*#*\s*(api_key|mem_model_api_key|embedding_model_api_key)\s*=\s*(?!your key)') {
        Write-Host "[WARNING] There may still be API key remnants in config/config.txt" -ForegroundColor Yellow
    } else {
        Write-Host "[SUCCESS] All API key values have been replaced with placeholder in config/config.txt" -ForegroundColor Green
    }
} else {
    Write-Host "[WARNING] config/config.txt file not found" -ForegroundColor Yellow
}

# 删除 __pycache__ 目录
Write-Host "[INFO] Deleting all __pycache__ directories..." -ForegroundColor Blue
$pycacheDirs = Get-ChildItem -Path . -Recurse -Directory -Filter "__pycache__"
if ($pycacheDirs) {
    $pycacheDirs | Remove-Item -Recurse -Force
    Write-Host "[SUCCESS] Deleted $($pycacheDirs.Count) __pycache__ directories" -ForegroundColor Green
} else {
    Write-Host "[INFO] No __pycache__ directories found" -ForegroundColor Blue
}

# 删除 .pyc 文件
Write-Host "[INFO] Deleting all .pyc files..." -ForegroundColor Blue
$pycFiles = Get-ChildItem -Path . -Recurse -Filter "*.pyc"
if ($pycFiles) {
    $pycFiles | Remove-Item -Force
    Write-Host "[SUCCESS] Deleted $($pycFiles.Count) .pyc files" -ForegroundColor Green
} else {
    Write-Host "[INFO] No .pyc files found" -ForegroundColor Blue
}

# 删除 .pyo 文件
Write-Host "[INFO] Deleting all .pyo files..." -ForegroundColor Blue
$pyoFiles = Get-ChildItem -Path . -Recurse -Filter "*.pyo"
if ($pyoFiles) {
    $pyoFiles | Remove-Item -Force
    Write-Host "[SUCCESS] Deleted $($pyoFiles.Count) .pyo files" -ForegroundColor Green
} else {
    Write-Host "[INFO] No .pyo files found" -ForegroundColor Blue
}

# 删除 build 目录
if (Test-Path "build") {
    Remove-Item "build" -Recurse -Force
    Write-Host "[SUCCESS] Deleted build directory" -ForegroundColor Green
} else {
    Write-Host "[INFO] No build directory found" -ForegroundColor Blue
}

# 删除 agia.egg-info 目录
if (Test-Path "agia.egg-info") {
    Remove-Item "agia.egg-info" -Recurse -Force
    Write-Host "[SUCCESS] Deleted agia.egg-info directory" -ForegroundColor Green
} else {
    Write-Host "[INFO] No agia.egg-info directory found" -ForegroundColor Blue
}

# 删除 .agia_last_output.json 文件
if (Test-Path ".agia_last_output.json") {
    Remove-Item ".agia_last_output.json" -Force
    Write-Host "[SUCCESS] Deleted .agia_last_output.json file" -ForegroundColor Green
} else {
    Write-Host "[INFO] No .agia_last_output.json file found" -ForegroundColor Blue
}

# 删除 .DS_Store 文件
Get-ChildItem -Path . -Recurse -Filter ".DS_Store" | Remove-Item -Force -ErrorAction SilentlyContinue

# 删除 Thumbs.db 文件
Get-ChildItem -Path . -Recurse -Filter "Thumbs.db" | Remove-Item -Force -ErrorAction SilentlyContinue

# 删除 .tmp 文件
Get-ChildItem -Path . -Recurse -Filter "*.tmp" | Remove-Item -Force -ErrorAction SilentlyContinue

# 删除 .temp 文件
Get-ChildItem -Path . -Recurse -Filter "*.temp" | Remove-Item -Force -ErrorAction SilentlyContinue

# 删除隐藏文件（以.开头的文件）包括所有子文件夹中的，但保留重要文件
Write-Host "[INFO] Deleting hidden files (starting with .) in all subdirectories..." -ForegroundColor Blue
$preservePatterns = @(".git", ".gitignore", ".gitmodules", ".github", ".vscode", ".idea")

# 获取所有隐藏文件（递归搜索所有子文件夹）
$hiddenFiles = Get-ChildItem -Path . -Recurse -File -Force | Where-Object { $_.Name.StartsWith('.') }

$deletedCount = 0
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
        try {
            Remove-Item $file.FullName -Force -ErrorAction Stop
            $deletedCount++
        } catch {
            # Silently continue if deletion fails
        }
    }
}

if ($hiddenFiles.Count -gt 0) {
    if ($deletedCount -gt 0) {
        Write-Host "[SUCCESS] Deleted $deletedCount hidden files from all subdirectories (preserved $preservedCount important files)" -ForegroundColor Green
    } else {
        Write-Host "[INFO] No hidden files to delete (all are preserved)" -ForegroundColor Blue
    }
} else {
    Write-Host "[INFO] No hidden files found in any subdirectory" -ForegroundColor Blue
}

# 清理 logs 目录下的 .log 文件
if (Test-Path "logs") {
    Get-ChildItem -Path "logs" -Recurse -Filter "*.log" | Remove-Item -Force -ErrorAction SilentlyContinue
    Write-Host "[INFO] Cleaned log files from logs directory" -ForegroundColor Blue
}

Write-Host "[SUCCESS] Temporary file cleanup completed" -ForegroundColor Green
Write-Host "[SUCCESS] All operations completed!" -ForegroundColor Green