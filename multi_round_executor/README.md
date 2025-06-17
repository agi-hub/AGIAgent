# Multi-Round Task Executor

## Project Structure

This package refactors the original large `multi_round_task_executor.py` file into multiple modular files:

```
multi_round_executor/
├── __init__.py              # Package initialization file
├── config.py               # Configuration constants and settings
├── task_loader.py          # CSV task loader
├── debug_recorder.py       # Debug recorder
├── task_checker.py         # Task completion checker
├── summary_generator.py    # Intelligent summary generator
├── report_generator.py     # Execution report generator
├── executor.py             # Main executor class
├── main.py                # Main function and command line interface
└── README.md               # Documentation
```

## Module Description

### 1. config.py
- Defines all configuration constants and default values
- Provides configuration reading utility functions  
- Contains timestamp, formatting and other configurations

### 2. task_loader.py
- Responsible for parsing and loading todo.csv files
- Detects CSV format and header rows
- Provides file validation functionality

### 3. debug_recorder.py
- Manages LLM call records in debug mode
- Provides call statistics and export functionality
- Memory-efficient record storage

### 4. task_checker.py
- Checks task completion status
- Analyzes execution history
- Extracts key achievements and error information

### 5. summary_generator.py
- Uses large models to generate intelligent summaries
- Supports both detailed and simplified modes
- Extracts key task information

### 6. report_generator.py
- Generates Markdown format execution reports
- Supports both Chinese and English languages
- Generates both detailed and concise reports

### 7. executor.py
- Main MultiRoundTaskExecutor class
- Integrates all module functionality
- Provides complete task execution workflow

### 8. main.py
- Command line interface entry point
- Argument parsing and configuration
- Error handling and exit code management

## Usage

### Using as a Package

```python
from multi_round_executor import MultiRoundTaskExecutor

executor = MultiRoundTaskExecutor(
    subtask_loops=3,
    logs_dir="logs",
    workspace_dir="workspace",
    debug_mode=False,
    detailed_summary=True
)

result = executor.execute_all_tasks("todo.csv")
```

### Command Line Usage

```bash
python -m multi_round_executor.main todo.csv --loops 3 --workspace-dir workspace --debug
```

## Refactoring Advantages

1. **Modularity**: Each file has a single responsibility, easy to understand and maintain
2. **Testability**: Each module can be tested independently
3. **Extensibility**: Easy to add new features or modify existing functionality
4. **Code Reuse**: Each module can be reused in other projects
5. **Clear Structure**: Code organization is clearer, reducing maintenance costs

## Compatibility

The refactored code is fully compatible with the original functionality, all APIs and configuration options remain unchanged.