# AGIBot 优先级调度器使用指南

## 📖 概述

AGIBot 现在支持基于优先级队列的智能体调度系统，能够实现更公平的资源分配和更好的并发控制。新的调度器解决了传统多线程模式下资源分配不均衡的问题。

## 🔍 问题背景

### 传统模式的问题

在传统的多线程模式下，你可能会观察到：
- 5个智能体执行轮数分别为：[3, 3, 2, 1, 4] - 严重不均衡
- 某些智能体长时间无响应
- 资源竞争导致性能不稳定
- LLM API 调用时间差异影响公平性

### 优先级调度器的优势

使用新的调度器后，你将看到：
- 执行轮数更均衡：[3, 3, 3, 3, 3] 
- 公平的资源分配
- 动态优先级调整
- 更好的并发控制
- 实时性能监控

## 🚀 快速开始

### 1. 基本使用

```python
from tools.multiagents import MultiAgentTools

# 启用优先级调度器（默认启用，懒加载模式）
multi_agent = MultiAgentTools(
    workspace_root="./workspace",
    use_priority_scheduler=True,    # 启用优先级调度器
    max_concurrent_agents=5,        # 最大并发智能体数量（默认5）
    # lazy_scheduler_start=True,    # 默认懒加载，可省略
    debug_mode=True
)

# 启动智能体
result = multi_agent.spawn_agibot(
    task_description="创建一个Python计算器程序",
    agent_id="agent_001",
    max_loops=5
)

print(f"调度模式: {result['scheduler_mode']}")
print(f"执行说明: {result['execution_note']}")
```

### 2. 监控调度器状态

```python
# 获取调度器状态
status = multi_agent.get_scheduler_status()
print(f"调度器状态: {status['scheduler_enabled']}")

# 打印详细状态
if status['scheduler_enabled']:
    multi_agent.priority_scheduler.print_status()
```

### 3. 模式切换

```python
# 切换到传统线程模式（仅在无活跃任务时）
result = multi_agent.toggle_scheduler_mode(enable_scheduler=False)
print(f"切换结果: {result['message']}")

# 重新启用调度器
result = multi_agent.toggle_scheduler_mode(enable_scheduler=True)
```

## ⚙️ 配置选项

### MultiAgentTools 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `use_priority_scheduler` | bool | True | 是否启用优先级调度器 |
| `max_concurrent_agents` | int | 5 | 最大并发智能体数量 |
| `lazy_scheduler_start` | bool | True | 是否使用懒加载模式 |
| `debug_mode` | bool | False | 是否启用调试模式 |

### spawn_agibot 参数变化

新增的状态信息：
- `scheduler_mode`: "priority_queue" 或 "traditional_threading"
- `execution_note`: 执行模式说明
- `scheduler_status`: 调度器详细状态（仅调度器模式）

## 📊 性能监控

### 智能体指标

调度器为每个智能体跟踪以下指标：

```python
{
    "total_executions": 5,          # 总执行次数
    "avg_execution_time": 45.2,     # 平均执行时间（秒）
    "success_rate": 0.8,            # 成功率 (80%)
    "fairness_score": 1.2,          # 公平性分数
    "last_execution_time": 1234567890  # 最后执行时间戳
}
```

### 系统状态

```python
{
    "scheduler_active": True,        # 调度器是否活跃
    "queue_size": 2,                # 队列中待处理任务数
    "active_agents": 3,             # 当前活跃智能体数
    "max_workers": 3,               # 最大工作线程数
    "total_submitted": 15,          # 总提交任务数
    "total_completed": 12,          # 总完成任务数
    "total_failed": 1               # 总失败任务数
}
```

## 🎯 公平性机制

### 动态优先级计算

调度器使用多因素算法计算动态优先级：

1. **基础优先级**: 默认为 5.0
2. **公平性调整**: 执行次数少的智能体优先级更高
3. **等待时间补偿**: 等待时间长的智能体提高优先级
4. **成功率奖励**: 成功率高的智能体获得轻微优势

### 公平性调整

每30秒自动调整一次公平性分数：
- 执行次数低于平均值80%：提高公平性分数
- 执行次数高于平均值120%：降低公平性分数

## 📈 使用示例

### 示例1：基本多智能体任务

```python
# 运行演示脚本
python example_priority_scheduler.py
```

### 示例2：对比模式

```python
# 对比传统模式和调度器模式
python example_priority_scheduler.py compare
```

### 示例3：自定义配置

```python
multi_agent = MultiAgentTools(
    workspace_root="./my_workspace",
    use_priority_scheduler=True,
    max_concurrent_agents=5,  # 增加并发数
    debug_mode=True
)

# 提交不同优先级的任务
for i in range(10):
    multi_agent.spawn_agibot(
        task_description=f"任务 {i+1}: 创建示例程序",
        agent_id=f"agent_{i+1:03d}",
        max_loops=3
    )

# 监控30秒
import time
time.sleep(30)

# 查看最终状态
multi_agent.get_scheduler_status()
```

## 🔧 故障排除

### 常见问题

1. **调度器无法启动**
   ```python
   # 检查是否有导入错误
   from tools.priority_scheduler import get_priority_scheduler
   scheduler = get_priority_scheduler()
   ```

2. **任务被卡在队列中**
   ```python
   # 检查资源限制
   status = multi_agent.get_scheduler_status()
   print(f"活跃智能体: {status['scheduler_details']['active_agents']}")
   print(f"队列大小: {status['scheduler_details']['queue_size']}")
   ```

3. **模式切换失败**
   ```python
   # 确保没有活跃任务
   session_info = multi_agent.get_agent_session_info()
   print(f"活跃智能体数量: {session_info['active_agents']}")
   ```

### 调试技巧

1. **启用详细日志**
   ```python
   multi_agent = MultiAgentTools(debug_mode=True)
   ```

2. **定期检查状态**
   ```python
   import time
   while True:
       multi_agent.get_scheduler_status()
       time.sleep(30)
   ```

3. **手动清理资源**
   ```python
   multi_agent.cleanup()
   ```

## 📋 最佳实践

### 1. 合理设置并发数

```python
# 根据系统资源调整
import multiprocessing
max_agents = min(multiprocessing.cpu_count(), 5)

multi_agent = MultiAgentTools(
    max_concurrent_agents=max_agents
)
```

### 2. 任务分解

```python
# 将大任务分解为小任务
large_task = "开发一个完整的Web应用"

# 分解为：
subtasks = [
    "设计数据库模式",
    "创建后端API",
    "开发前端界面",
    "编写测试用例"
]
```

### 3. 监控和调优

```python
# 定期监控性能
def monitor_performance():
    status = multi_agent.get_scheduler_status()
    if status['scheduler_enabled']:
        details = status['scheduler_details']
        if details['total_failed'] / max(details['total_submitted'], 1) > 0.2:
            print("⚠️ 失败率过高，考虑降低并发数")
```

## 🔄 迁移指南

### 从传统模式迁移

如果你之前使用传统的`spawn_agibot`：

```python
# 旧代码
multi_agent = MultiAgentTools()
result = multi_agent.spawn_agibot(task_description="任务")

# 新代码（无需修改，默认启用调度器 + 懒加载）
multi_agent = MultiAgentTools()  # 自动启用优先级调度器，懒加载模式
result = multi_agent.spawn_agibot(task_description="任务")
print(f"调度模式: {result['scheduler_mode']}")
```

### 兼容性和默认行为

- ✅ 所有现有API保持兼容
- ✅ 返回格式向后兼容，只是新增字段
- ✅ 可以随时切换回传统模式
- ✅ 支持渐进式迁移
- 🆕 **默认懒加载模式**: 只在首次调用`spawn_agibot`时启动调度器，节省资源
- 🆕 **默认支持20个并发智能体**: 提供更好的并行处理能力

### 资源优化

新的默认配置带来以下好处：
- 🚀 **零启动开销**: 创建`MultiAgentTools`不再立即创建worker线程
- 💾 **内存友好**: 只在需要时分配资源
- ⚡ **更快的应用启动**: 减少初始化时间
- 🔄 **更高并发**: 默认支持20个并发智能体，适应更大规模的任务

## 🚀 高级功能

### 自定义优先级策略

未来版本将支持：
- 自定义优先级算法
- 基于任务类型的优先级
- 动态资源配额调整
- 负载均衡策略

---

## 📞 支持

如有问题或建议，请：
1. 查看调试日志
2. 运行示例脚本验证环境
3. 提交issue或联系技术支持

享受更公平、更高效的多智能体体验！🎉 