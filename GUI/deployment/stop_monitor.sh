#!/bin/bash
# AGI Agent GUI 多应用监控程序停止脚本

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 设置文件路径
LOG_DIR="$SCRIPT_DIR/logs"
PID_FILE="$LOG_DIR/monitor.pid"

# 检查PID文件是否存在
if [[ ! -f "$PID_FILE" ]]; then
    echo "未找到PID文件，监控程序可能未在运行"
    
    # 尝试查找运行中的monitor.py进程
    MONITOR_PIDS=$(pgrep -f "python.*monitor.py")
    if [[ -n "$MONITOR_PIDS" ]]; then
        echo "发现运行中的监控进程："
        echo "$MONITOR_PIDS" | while read pid; do
            echo "  PID: $pid"
        done
        echo ""
        read -p "是否要停止这些进程? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            echo "$MONITOR_PIDS" | while read pid; do
                echo "停止进程 PID: $pid"
                kill -TERM $pid 2>/dev/null || kill -KILL $pid 2>/dev/null
            done
            echo "已停止所有监控进程"
        fi
    else
        echo "未发现运行中的监控进程"
    fi
    exit 0
fi

# 读取PID
PID=$(cat "$PID_FILE")

# 检查进程是否存在
if ! ps -p $PID > /dev/null 2>&1; then
    echo "监控程序 (PID: $PID) 已经停止"
    rm -f "$PID_FILE"
    exit 0
fi

echo "正在停止多应用监控程序 (PID: $PID)..."
echo "这将停止所有被监控的应用实例"

# 发送TERM信号
kill -TERM $PID

# 等待进程结束
for i in {1..10}; do
    if ! ps -p $PID > /dev/null 2>&1; then
        echo "监控程序已成功停止"
        rm -f "$PID_FILE"
        exit 0
    fi
    sleep 1
done

# 如果进程仍然存在，强制杀死
echo "进程未响应TERM信号，强制终止..."
kill -KILL $PID 2>/dev/null

# 再次检查
sleep 2
if ps -p $PID > /dev/null 2>&1; then
    echo "错误: 无法停止监控程序"
    exit 1
else
    echo "监控程序已强制停止"
    rm -f "$PID_FILE"
fi

# 清理可能的子进程（所有通过monitor启动的app.py进程）
echo "清理相关应用进程..."
# 查找所有通过monitor启动的app.py进程（包含--port和--app参数）
pkill -f "python.*app.py.*--port.*--app" 2>/dev/null || true

# 等待一下让进程完全退出
sleep 2

echo "监控程序已完全停止"

