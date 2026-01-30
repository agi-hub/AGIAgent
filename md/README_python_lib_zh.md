# AGI Agent Python Library Interface

> **⚠️ 实验性功能**: Python库接口目前为实验性功能，API可能会在未来版本中发生变化。建议在生产环境中谨慎使用。

AGI Agent现在支持作为Python库使用，提供类似OpenAI Chat API的编程接口。你可以直接在Python代码中调用AGI Agent，而不需要通过命令行。

## 特性

- 🐍 **纯Python接口**: 无需命令行，直接在代码中调用
- 🔧 **编程式配置**: 所有配置通过参数传递，不依赖config.txt文件
- 💬 **OpenAI风格API**: 熟悉的chat接口，易于集成
- 📁 **灵活输出目录**: 支持自定义输出目录参数
- 🔄 **继续模式**: 可以基于之前的工作继续开发
- 📊 **详细返回信息**: 包含执行状态、输出路径、执行时间等详细信息

## 安装和设置

### 方式一：pip安装（推荐）

AGI Agent可以作为Python包直接安装：

```bash
# 从源码安装
pip install .


```

安装完成后，你可以直接在Python代码中导入使用：

```python
from agia import AGIAgentClient, create_client
```

### 方式二：依赖安装

如果选择不安装为系统包，确保你已经安装了AGI Agent的所有依赖：

```bash
pip install -r requirements.txt
```

## 基本用法

### 1. 简单示例

```python
# 如果通过pip安装，使用：
from agia import AGIAgentClient
# 如果使用源码，使用：
# from main import AGIAgentClient

# 初始化客户端
client = AGIAgentClient(
    api_key="your_api_key_here",
    model="claude-3-sonnet-20240229",  # 或 "gpt-4", "gpt-3.5-turbo"等
    api_base="https://api.anthropic.com"  # 可选
)

# 发送任务请求
response = client.chat(
    messages=[
        {"role": "user", "content": "创建一个Python计算器应用"}
    ],
    dir="my_calculator",  # 输出目录
    loops=10  # 最大执行轮数
)

# 检查结果
if response["success"]:
    print(f"任务完成! 输出目录: {response['output_dir']}")
else:
    print(f"任务失败: {response['message']}")
```

### 2. 使用便捷函数

```python
# 如果通过pip安装，使用：
from agia import create_client
# 如果使用源码，使用：
# from main import create_client

# 使用便捷函数创建客户端
client = create_client(
    api_key="your_api_key_here",
    model="gpt-4",
    debug_mode=True
)

response = client.chat(
    messages=[{"role": "user", "content": "构建一个Web应用"}],
    dir="web_project"
)
```

## API 参考

### AGIAgentClient

#### 初始化参数

```python
AGIAgentClient(
    api_key: str,              # 必需: API密钥
    model: str,                # 必需: 模型名称
    api_base: str = None,      # 可选: API基础URL
    debug_mode: bool = False,  # 是否启用调试模式
    detailed_summary: bool = True,     # 是否生成详细摘要
    single_task_mode: bool = True,     # 是否使用单任务模式
    interactive_mode: bool = False     # 是否启用交互模式
)
```

#### chat方法

```python
client.chat(
    messages: list,            # 必需: 消息列表
    dir: str = None,          # 可选: 输出目录 
    loops: int = 25,          # 最大执行轮数
    continue_mode: bool = False,  # 是否继续之前的工作
    **kwargs                  # 其他参数
) -> dict
```

**消息格式：**
```python
messages = [
    {"role": "user", "content": "你的任务描述"}
]
```

**返回值：**
```python
{
    "success": bool,           # 是否成功
    "message": str,            # 结果消息
    "output_dir": str,         # 输出目录路径
    "workspace_dir": str,      # 工作空间目录路径
    "execution_time": float,   # 执行时间（秒）
    "details": dict           # 详细信息
}
```

### 支持的模型

通过`client.get_models()`获取支持的模型列表：

- `gpt-4`
- `gpt-4-turbo` 
- `gpt-3.5-turbo`
- `claude-3-sonnet-20240229`
- `claude-3-haiku-20240307`
- `claude-3-opus-20240229`
- `claude-3-5-sonnet-20241022`

## 使用场景

### 1. 单个任务执行

```python
client = AGIAgentClient(api_key="xxx", model="gpt-4")

response = client.chat(
    messages=[{"role": "user", "content": "创建一个待办事项应用"}],
    dir="todo_app"
)
```

### 2. 继续之前的工作

```python
# 第一次：创建基础项目
response1 = client.chat(
    messages=[{"role": "user", "content": "创建一个Flask应用"}],
    dir="my_app"
)

# 第二次：在现有项目基础上添加功能
response2 = client.chat(
    messages=[{"role": "user", "content": "添加用户认证功能"}],
    dir="my_app",
    continue_mode=True  # 继续之前的工作
)
```

### 3. 批处理多个任务

```python
tasks = [
    "创建Python爬虫脚本",
    "构建数据分析工具", 
    "编写自动化测试"
]

results = []
for task in tasks:
    response = client.chat(
        messages=[{"role": "user", "content": task}],
        dir=f"project_{len(results)+1}"
    )
    results.append(response)
```

### 4. 多任务模式（复杂项目）

```python
client = AGIAgentClient(
    api_key="xxx",
    model="gpt-4",
    single_task_mode=False  # 启用多任务模式
)

response = client.chat(
    messages=[{"role": "user", "content": "创建完整的电商网站，包含用户系统、商品管理、订单处理等"}],
    dir="ecommerce_site",
    loops=20
)
```

## 配置选项

### 调试模式

```python
client = AGIAgentClient(
    api_key="xxx",
    model="gpt-4",
    debug_mode=True  # 启用详细日志
)
```

### 自定义配置

```python
client = AGIAgentClient(
    api_key="xxx",
    model="claude-3-haiku-20240307",
    api_base="https://custom-api.com",
    detailed_summary=True,
    interactive_mode=False
)

# 查看当前配置
config = client.get_config()
print(config)
```

## 错误处理

```python
try:
    client = AGIAgentClient(api_key="", model="gpt-4")  # 空API密钥
except ValueError as e:
    print(f"配置错误: {e}")

# 检查执行结果
response = client.chat(messages=[{"role": "user", "content": "任务"}])
if not response["success"]:
    print(f"执行失败: {response['message']}")
    print(f"错误详情: {response['details']}")
```

## 与命令行模式的对比

| 特性 | 命令行模式 | Python库模式 |
|------|-----------|-------------|
| 配置方式 | config.txt文件 | 代码参数 |
| 调用方式 | `python main.py` | `client.chat()` |
| 集成性 | 独立运行 | 嵌入Python程序 |
| 返回信息 | 终端输出 | 结构化字典 |
| 批处理 | 脚本循环 | 原生Python循环 |

## 注意事项

1. **API密钥安全**: 不要在代码中硬编码API密钥，建议使用环境变量
2. **输出目录**: 如果不指定`dir`参数，会自动生成时间戳目录
3. **执行时间**: 复杂任务可能需要较长时间，请耐心等待
4. **模型选择**: 根据任务复杂度选择合适的模型
5. **继续模式**: 使用`continue_mode=True`时确保目录存在且包含之前的工作

## 完整示例

查看`example_usage.py`文件获取更多详细示例，包括：

- 基本用法
- 继续模式
- 多任务模式  
- 自定义配置
- 错误处理
- 批处理

## 环境变量配置

推荐使用环境变量管理API密钥：

```python
import os
# 如果通过pip安装，使用：
from agia import AGIAgentClient
# 如果使用源码，使用：
# from main import AGIAgentClient

client = AGIAgentClient(
    api_key=os.environ.get("OPENAI_API_KEY"),  # 或 ANTHROPIC_API_KEY
    model=os.environ.get("MODEL_NAME", "gpt-4")
)
```

## 总结

AGI Agent的Python库接口提供了强大而灵活的编程访问方式，让你可以：

- 🔧 在Python应用中直接集成AGI Agent功能
- 📊 获得结构化的执行结果和详细信息
- 🔄 轻松实现批处理和工作流自动化
- ⚙️ 通过代码进行精确的配置控制

开始使用AGI Agent Python库，让AI驱动的任务执行成为你Python项目的一部分！ 