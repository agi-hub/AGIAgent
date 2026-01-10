#!/bin/bash
# AGI Agent GUI 多应用监控程序后台运行脚本

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 设置日志文件
LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"
DAEMON_LOG="$LOG_DIR/monitor_daemon.log"
PID_FILE="$LOG_DIR/monitor.pid"

# 检查是否已经在运行
if [[ -f "$PID_FILE" ]]; then
    PID=$(cat "$PID_FILE")
    if ps -p $PID > /dev/null 2>&1; then
        echo "监控程序已在运行 (PID: $PID)"
        echo "如需停止，请运行: ./stop_monitor.sh"
        exit 1
    else
        echo "发现过期的PID文件，正在清理..."
        rm -f "$PID_FILE"
    fi
fi

# 检查依赖
if ! command -v python3 &> /dev/null; then
    echo "错误: 未找到 python3 命令"
    exit 1
fi

if [[ ! -f "monitor.py" ]]; then
    echo "错误: 未找到 monitor.py 文件"
    exit 1
fi

if [[ ! -f "monitor_config.json" ]]; then
    echo "错误: 未找到 monitor_config.json 配置文件"
    exit 1
fi

if [[ ! -f "../app.py" ]]; then
    echo "错误: 未找到 GUI/app.py 文件"
    exit 1
fi

echo "启动 AGI Agent GUI 多应用监控程序 (后台模式)..."
echo "日志文件: $DAEMON_LOG"
echo "PID文件: $PID_FILE"
echo "配置文件: monitor_config.json"

# 显示将要监控的应用
if command -v python3 &> /dev/null && [[ -f "monitor_config.json" ]]; then
    echo ""
    echo "将监控以下应用:"
    python3 -c "import json; config = json.load(open('monitor_config.json')); [print(f'  - {app[\"name\"]}: 端口 {app[\"port\"]} ({app.get(\"description\", \"\")})') for app in config.get('apps', [])]"
    echo ""
fi

# 后台启动监控程序
nohup python3 monitor.py > "$DAEMON_LOG" 2>&1 &
MONITOR_PID=$!

# 保存PID
echo $MONITOR_PID > "$PID_FILE"

echo "监控程序已启动 (PID: $MONITOR_PID)"
echo "查看日志: tail -f $DAEMON_LOG"
echo "停止监控: ./stop_monitor.sh"

# 等待几秒确认启动成功
sleep 3
if ps -p $MONITOR_PID > /dev/null 2>&1; then
    echo "监控程序运行正常"
else
    echo "警告: 监控程序可能启动失败，请检查日志"
    rm -f "$PID_FILE"
    exit 1
fi

