# AGI Bot Python Library Interface

> **⚠️ Experimental Feature**: The Python library interface is currently an experimental feature, and the API may change in future versions. Use with caution in production environments.

AGI Bot now supports usage as a Python library, providing an OpenAI Chat API-like programming interface. You can call AGI Bot directly in Python code without using the command line.

## Features

- 🐍 **Pure Python Interface**: Call directly in code without command line
- 🔧 **Programmatic Configuration**: All configuration passed through parameters, no dependency on config.txt file
- 💬 **OpenAI-style API**: Familiar chat interface, easy to integrate
- 📁 **Flexible Output Directory**: Support custom output directory parameter
- 🔄 **Continue Mode**: Can continue development based on previous work
- 📊 **Detailed Return Information**: Contains execution status, output paths, execution time and other detailed information

## Installation and Setup

### Method 1: pip Installation (Recommended)

AGI Bot can be installed directly as a Python package:

```bash
# Install from source
pip install .

# Or install from git repository
pip install git+https://github.com/agi-hub/AGIBot.git

```

After installation, you can import and use directly in Python code:

```python
from agibot import AGIBotClient, create_client
```

### Method 2: Dependencies Installation

If you choose not to install as a system package, make sure you have installed all AGI Bot dependencies:

```bash
pip install -r requirements.txt
```

## Basic Usage

### 1. Simple Example

```python
# If installed via pip, use:
from agibot import AGIBotClient
# If using source code, use:
# from main import AGIBotClient

# Initialize client
client = AGIBotClient(
    api_key="your_api_key_here",
    model="claude-3-sonnet-20240229",  # or "gpt-4", "gpt-3.5-turbo", etc.
    api_base="https://api.anthropic.com"  # Optional
)

# Send task request
response = client.chat(
    messages=[
        {"role": "user", "content": "Create a Python calculator application"}
    ],
    dir="my_calculator",  # Output directory
    loops=10  # Maximum execution rounds
)

# Check result
if response["success"]:
    print(f"Task completed! Output directory: {response['output_dir']}")
else:
    print(f"Task failed: {response['message']}")
```

### 2. Using Convenience Function

```python
# If installed via pip, use:
from agibot import create_client
# If using source code, use:
# from main import create_client

# Use convenience function to create client
client = create_client(
    api_key="your_api_key_here",
    model="gpt-4",
    debug_mode=True
)

response = client.chat(
    messages=[{"role": "user", "content": "Build a web application"}],
    dir="web_project"
)
```

## API Reference

### AGIBotClient

#### Initialization Parameters

```python
AGIBotClient(
    api_key: str,              # Required: API key
    model: str,                # Required: Model name
    api_base: str = None,      # Optional: API base URL
    debug_mode: bool = False,  # Whether to enable debug mode
    detailed_summary: bool = True,     # Whether to generate detailed summary
    single_task_mode: bool = True,     # Whether to use single task mode
    interactive_mode: bool = False     # Whether to enable interactive mode
)
```

#### chat Method

```python
client.chat(
    messages: list,            # Required: Message list
    dir: str = None,          # Optional: Output directory
    loops: int = 25,          # Maximum execution rounds
    continue_mode: bool = False,  # Whether to continue previous work
    **kwargs                  # Other parameters
) -> dict
```

**Message Format:**
```python
messages = [
    {"role": "user", "content": "Your task description"}
]
```

**Return Value:**
```python
{
    "success": bool,           # Whether successful
    "message": str,            # Result message
    "output_dir": str,         # Output directory path
    "workspace_dir": str,      # Workspace directory path
    "execution_time": float,   # Execution time (seconds)
    "details": dict           # Detailed information
}
```

### Supported Models

Get supported model list through `client.get_models()`:

- `gpt-4`
- `gpt-4-turbo`
- `gpt-3.5-turbo`
- `claude-3-sonnet-20240229`
- `claude-3-haiku-20240307`
- `claude-3-opus-20240229`
- `claude-3-5-sonnet-20241022`

## Use Cases

### 1. Single Task Execution

```python
client = AGIBotClient(api_key="xxx", model="gpt-4")

response = client.chat(
    messages=[{"role": "user", "content": "Create a todo list application"}],
    dir="todo_app"
)
```

### 2. Continue Previous Work

```python
# First: Create basic project
response1 = client.chat(
    messages=[{"role": "user", "content": "Create a Flask application"}],
    dir="my_app"
)

# Second: Add features to existing project
response2 = client.chat(
    messages=[{"role": "user", "content": "Add user authentication feature"}],
    dir="my_app",
    continue_mode=True  # Continue previous work
)
```

### 3. Batch Processing Multiple Tasks

```python
tasks = [
    "Create Python web scraper script",
    "Build data analysis tool",
    "Write automated tests"
]

results = []
for task in tasks:
    response = client.chat(
        messages=[{"role": "user", "content": task}],
        dir=f"project_{len(results)+1}"
    )
    results.append(response)
```

### 4. Multi-Task Mode (Complex Projects)

```python
client = AGIBotClient(
    api_key="xxx",
    model="gpt-4",
    single_task_mode=False  # Enable multi-task mode
)

response = client.chat(
    messages=[{"role": "user", "content": "Create complete e-commerce website with user system, product management, order processing, etc."}],
    dir="ecommerce_site",
    loops=20
)
```

## Configuration Options

### Debug Mode

```python
client = AGIBotClient(
    api_key="xxx",
    model="gpt-4",
    debug_mode=True  # Enable detailed logging
)
```

### Custom Configuration

```python
client = AGIBotClient(
    api_key="xxx",
    model="claude-3-haiku-20240307",
    api_base="https://custom-api.com",
    detailed_summary=True,
    interactive_mode=False
)

# Check current configuration
config = client.get_config()
print(config)
```

## Error Handling

```python
try:
    client = AGIBotClient(api_key="", model="gpt-4")  # Empty API key
except ValueError as e:
    print(f"Configuration error: {e}")

# Check execution result
response = client.chat(messages=[{"role": "user", "content": "task"}])
if not response["success"]:
    print(f"Execution failed: {response['message']}")
    print(f"Error details: {response['details']}")
```

## Comparison with Command Line Mode

| Feature | Command Line Mode | Python Library Mode |
|---------|-------------------|---------------------|
| Configuration | config.txt file | Code parameters |
| Invocation | `python main.py` | `client.chat()` |
| Integration | Independent run | Embedded in Python programs |
| Return Info | Terminal output | Structured dictionary |
| Batch Processing | Script loops | Native Python loops |

## Important Notes

1. **API Key Security**: Don't hardcode API keys in code, recommend using environment variables
2. **Output Directory**: If `dir` parameter is not specified, timestamp directory will be auto-generated
3. **Execution Time**: Complex tasks may take longer time, please be patient
4. **Model Selection**: Choose appropriate model based on task complexity
5. **Continue Mode**: When using `continue_mode=True`, ensure directory exists and contains previous work

## Complete Examples

See `example_usage.py` file for more detailed examples, including:

- Basic usage
- Continue mode
- Multi-task mode
- Custom configuration
- Error handling
- Batch processing

## Environment Variable Configuration

Recommended to use environment variables for managing API keys:

```python
import os
# If installed via pip, use:
from agibot import AGIBotClient
# If using source code, use:
# from main import AGIBotClient

client = AGIBotClient(
    api_key=os.environ.get("OPENAI_API_KEY"),  # or ANTHROPIC_API_KEY
    model=os.environ.get("MODEL_NAME", "gpt-4")
)
```

## Summary

AGI Bot's Python library interface provides powerful and flexible programmatic access, allowing you to:

- 🔧 Directly integrate AGI Bot functionality in Python applications
- 📊 Get structured execution results and detailed information
- 🔄 Easily implement batch processing and workflow automation
- ⚙️ Precise configuration control through code

Start using AGI Bot Python library and make AI-driven task execution part of your Python projects! 