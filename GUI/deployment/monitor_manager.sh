#!/bin/bash
# AGI Agent GUI 多应用监控管理脚本
# 提供统一的管理接口，方便启动、停止、查看状态等操作

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 显示使用说明
show_usage() {
    echo "AGI Agent GUI 多应用监控管理脚本"
    echo ""
    echo "用法: $0 [命令]"
    echo ""
    echo "命令:"
    echo "  start       启动监控程序（后台运行）"
    echo "  stop        停止监控程序"
    echo "  restart     重启监控程序"
    echo "  status      查看监控程序和应用状态"
    echo "  logs        查看监控日志"
    echo "  config      查看/编辑配置文件"
    echo "  help        显示此帮助信息"
    echo ""
}

# 检查监控程序是否运行
check_monitor_running() {
    PID_FILE="$SCRIPT_DIR/logs/monitor.pid"
    if [[ -f "$PID_FILE" ]]; then
        PID=$(cat "$PID_FILE")
        if ps -p $PID > /dev/null 2>&1; then
            return 0
        fi
    fi
    return 1
}

# 启动监控程序
start_monitor() {
    if check_monitor_running; then
        echo -e "${YELLOW}监控程序已在运行${NC}"
        return 1
    fi
    
    echo -e "${BLUE}启动监控程序...${NC}"
    ./start_monitor_daemon.sh
    if [[ $? -eq 0 ]]; then
        echo -e "${GREEN}监控程序启动成功${NC}"
        return 0
    else
        echo -e "${RED}监控程序启动失败${NC}"
        return 1
    fi
}

# 停止监控程序
stop_monitor() {
    if ! check_monitor_running; then
        echo -e "${YELLOW}监控程序未运行${NC}"
        return 1
    fi
    
    echo -e "${BLUE}停止监控程序...${NC}"
    ./stop_monitor.sh
    if [[ $? -eq 0 ]]; then
        echo -e "${GREEN}监控程序已停止${NC}"
        return 0
    else
        echo -e "${RED}监控程序停止失败${NC}"
        return 1
    fi
}

# 重启监控程序
restart_monitor() {
    echo -e "${BLUE}重启监控程序...${NC}"
    stop_monitor
    sleep 2
    start_monitor
}

# 查看状态
show_status() {
    echo "============================================================"
    echo "监控程序状态"
    echo "============================================================"
    
    PID_FILE="$SCRIPT_DIR/logs/monitor.pid"
    if check_monitor_running; then
        PID=$(cat "$PID_FILE")
        echo -e "监控程序: ${GREEN}运行中${NC} (PID: $PID)"
    else
        echo -e "监控程序: ${RED}未运行${NC}"
    fi
    
    echo ""
    echo "应用实例状态:"
    echo "------------------------------------------------------------"
    
    # 读取配置文件并检查每个应用的状态
    if [[ -f "monitor_config.json" ]] && command -v python3 &> /dev/null; then
        python3 << 'EOF'
import json
import socket
import sys

try:
    with open('monitor_config.json', 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    apps = config.get('apps', [])
    if not apps:
        print("  未配置应用")
        sys.exit(0)
    
    for app in apps:
        name = app.get('name', 'unknown')
        port = app.get('port', 0)
        app_name = app.get('app_name', '')
        description = app.get('description', '')
        
        # 检查端口是否在监听
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1)
            result = sock.connect_ex(('127.0.0.1', port))
            sock.close()
            
            if result == 0:
                status = "运行中"
                status_color = "\033[0;32m"
            else:
                status = "未运行"
                status_color = "\033[0;31m"
        except:
            status = "未知"
            status_color = "\033[1;33m"
        
        print(f"  {name:15} 端口: {port:5}  状态: {status_color}{status}\033[0m  ({description})")
        
except Exception as e:
    print(f"  错误: {e}")
EOF
    else
        echo "  无法读取配置文件或python3不可用"
    fi
    
    echo ""
    echo "============================================================"
}

# 查看日志
show_logs() {
    LOG_FILE="$SCRIPT_DIR/logs/monitor.log"
    DAEMON_LOG="$SCRIPT_DIR/logs/monitor_daemon.log"
    
    if [[ ! -f "$LOG_FILE" ]] && [[ ! -f "$DAEMON_LOG" ]]; then
        echo -e "${YELLOW}日志文件不存在${NC}"
        return 1
    fi
    
    echo "选择要查看的日志:"
    echo "  1) 监控程序日志 (monitor.log)"
    echo "  2) 守护进程日志 (monitor_daemon.log)"
    echo "  3) 所有应用日志"
    read -p "请选择 [1-3]: " choice
    
    case $choice in
        1)
            if [[ -f "$LOG_FILE" ]]; then
                tail -f "$LOG_FILE"
            else
                echo -e "${YELLOW}日志文件不存在${NC}"
            fi
            ;;
        2)
            if [[ -f "$DAEMON_LOG" ]]; then
                tail -f "$DAEMON_LOG"
            else
                echo -e "${YELLOW}日志文件不存在${NC}"
            fi
            ;;
        3)
            echo "应用日志文件:"
            for log_file in "$SCRIPT_DIR/logs"/app_*_stdout.log; do
                if [[ -f "$log_file" ]]; then
                    app_name=$(basename "$log_file" | sed 's/app_\(.*\)_stdout.log/\1/')
                    echo ""
                    echo "=== $app_name ==="
                    tail -20 "$log_file"
                fi
            done
            ;;
        *)
            echo "无效选择"
            ;;
    esac
}

# 查看/编辑配置
show_config() {
    CONFIG_FILE="$SCRIPT_DIR/monitor_config.json"
    
    if [[ ! -f "$CONFIG_FILE" ]]; then
        echo -e "${RED}配置文件不存在: $CONFIG_FILE${NC}"
        return 1
    fi
    
    echo "当前配置:"
    echo "============================================================"
    if command -v python3 &> /dev/null; then
        python3 -m json.tool "$CONFIG_FILE" 2>/dev/null || cat "$CONFIG_FILE"
    else
        cat "$CONFIG_FILE"
    fi
    echo ""
    echo "============================================================"
    read -p "是否要编辑配置文件? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        ${EDITOR:-nano} "$CONFIG_FILE"
    fi
}

# 主逻辑
case "${1:-help}" in
    start)
        start_monitor
        ;;
    stop)
        stop_monitor
        ;;
    restart)
        restart_monitor
        ;;
    status)
        show_status
        ;;
    logs)
        show_logs
        ;;
    config)
        show_config
        ;;
    help|--help|-h)
        show_usage
        ;;
    *)
        echo -e "${RED}未知命令: $1${NC}"
        echo ""
        show_usage
        exit 1
        ;;
esac

exit $?

