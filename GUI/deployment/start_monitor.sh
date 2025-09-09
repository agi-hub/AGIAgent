#!/bin/bash
# AGI Agent GUI 监控程序启动脚本

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 检查Python是否存在
if ! command -v python3 &> /dev/null; then
    echo "错误: 未找到 python3 命令"
    exit 1
fi

# 检查monitor.py是否存在
if [[ ! -f "monitor.py" ]]; then
    echo "错误: 未找到 monitor.py 文件"
    exit 1
fi

# 检查GUI/app.py是否存在
if [[ ! -f "GUI/app.py" ]]; then
    echo "错误: 未找到 GUI/app.py 文件"
    exit 1
fi

echo "启动 AGI Agent GUI 监控程序..."
echo "日志文件: logs/monitor.log"
echo "按 Ctrl+C 停止监控"
echo ""

# 设置权限
chmod +x monitor.py

# 启动监控程序
python3 monitor.py

