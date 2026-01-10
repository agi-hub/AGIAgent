# AGI Agent GUI 多应用监控系统

## 概述

这是一个多应用监控系统，可以同时监控和管理多个 AGI Agent GUI 应用实例，每个实例使用不同的端口和应用名称。

## 文件说明

- `monitor_config.json` - 配置文件，定义所有需要监控的应用
- `monitor.py` - 核心监控程序，支持多应用监控
- `start_monitor_daemon.sh` - 启动监控程序（后台运行）
- `stop_monitor.sh` - 停止监控程序
- `monitor_manager.sh` - 统一管理脚本（推荐使用）

## 快速开始

### 1. 配置应用

编辑 `monitor_config.json` 文件，添加或修改需要监控的应用：

```json
{
  "apps": [
    {
      "name": "patent",
      "port": 5003,
      "app_name": "patent",
      "description": "专利应用"
    },
    {
      "name": "colordoc",
      "port": 5004,
      "app_name": "colordoc",
      "description": "彩色文档应用"
    }
  ],
  "check_interval": 1,
  "max_startup_attempts": 3,
  "startup_retry_delay": 60
}
```

### 2. 使用管理脚本（推荐）

```bash
# 启动监控
./monitor_manager.sh start

# 查看状态
./monitor_manager.sh status

# 查看日志
./monitor_manager.sh logs

# 停止监控
./monitor_manager.sh stop

# 重启监控
./monitor_manager.sh restart

# 查看/编辑配置
./monitor_manager.sh config
```

### 3. 直接使用脚本

```bash
# 启动监控（后台运行）
./start_monitor_daemon.sh

# 停止监控
./stop_monitor.sh
```

## 配置说明

### monitor_config.json

- `apps`: 应用列表
  - `name`: 应用标识名称（用于日志和进程识别）
  - `port`: 应用监听的端口
  - `app_name`: 传递给 `app.py` 的 `--app` 参数
  - `description`: 应用描述（可选）
- `check_interval`: 检查间隔（秒），默认1秒
- `max_startup_attempts`: 最大启动尝试次数，默认3次
- `startup_retry_delay`: 达到最大尝试次数后的重试延迟（秒），默认60秒

## 功能特性

1. **多应用监控**: 同时监控多个应用实例，每个实例独立管理
2. **自动重启**: 检测到应用停止后自动重启
3. **独立日志**: 每个应用有独立的日志文件
4. **优雅退出**: 支持信号处理，优雅停止所有应用
5. **状态检查**: 通过端口检测应用运行状态
6. **进程管理**: 自动清理僵尸进程

## 日志文件

所有日志文件位于 `logs/` 目录：

- `monitor.log` - 监控程序主日志
- `monitor_daemon.log` - 守护进程日志
- `app_{name}_stdout.log` - 各应用的stdout日志
- `app_{name}_stderr.log` - 各应用的stderr日志

## 工作原理

1. 监控程序读取 `monitor_config.json` 配置文件
2. 为每个应用创建一个独立的监控线程
3. 每个线程定期检查对应端口是否在监听
4. 如果检测到应用未运行，自动启动应用
5. 如果启动失败，会重试，达到最大次数后等待一段时间再重试

## 注意事项

1. 确保所有端口未被其他程序占用
2. 确保 `GUI/app.py` 文件存在且可执行
3. 确保有足够的系统资源运行多个应用实例
4. 修改配置后需要重启监控程序才能生效

## 故障排查

### 监控程序无法启动

- 检查 `monitor_config.json` 文件格式是否正确
- 检查 `GUI/app.py` 文件是否存在
- 查看 `logs/monitor_daemon.log` 日志文件

### 应用无法启动

- 检查端口是否被占用: `netstat -tuln | grep <port>`
- 查看应用日志: `logs/app_{name}_stderr.log`
- 检查应用配置是否正确

### 应用频繁重启

- 检查应用是否有错误
- 查看应用日志文件
- 检查系统资源是否充足

## 示例

假设要监控以下应用：

```bash
python GUI/app.py --port 5003 --app patent
python GUI/app.py --port 5004 --app colordoc
python GUI/app.py --port 5005 --app bit
python GUI/app.py --port 5006 --app bjou
```

只需在 `monitor_config.json` 中配置这些应用，然后运行：

```bash
./monitor_manager.sh start
```

监控程序会自动管理所有这些应用。

