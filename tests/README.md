# ToolExecutor 测试套件

本目录包含 ToolExecutor 的所有测试代码，已从主文件 `tool_executor.py` 中分离出来以提高代码组织性。

## 测试文件结构

```
tests/
├── __init__.py                 # 测试包初始化文件
├── README.md                   # 本说明文件
├── run_all_tests.py           # 测试运行器
├── test_tool_calling.py       # 工具调用功能测试
├── test_statistics.py         # LLM统计功能测试  
├── test_cache.py              # 缓存机制测试
├── test_history.py            # 历史记录功能测试
└── test_search.py             # 搜索结果格式化测试
```

## 运行测试

### 运行所有测试

```bash
# 从项目根目录运行
python tests/run_all_tests.py

# 或者
cd tests
python run_all_tests.py
```

### 运行特定测试

```bash
# 运行工具调用测试
python tests/run_all_tests.py --test chat_tools

# 运行JSON解析测试
python tests/run_all_tests.py --test json_parsing

# 运行统计功能测试
python tests/run_all_tests.py --test statistics

# 运行缓存机制测试
python tests/run_all_tests.py --test cache

# 运行历史记录测试
python tests/run_all_tests.py --test history

# 运行搜索格式化测试
python tests/run_all_tests.py --test search
```

### 查看可用测试

```bash
python tests/run_all_tests.py --list
```

### 直接运行单个测试文件

```bash
# 直接运行某个测试文件
python tests/test_tool_calling.py
python tests/test_statistics.py
python tests/test_cache.py
python tests/test_history.py
python tests/test_search.py
```

## 测试内容说明

### test_tool_calling.py
- **test_chat_based_tool_calling**: 测试基于聊天的工具调用功能
- **test_json_tool_calling_parsing**: 测试JSON格式工具调用解析

### test_statistics.py
- **test_llm_statistics**: 测试LLM统计计算功能，包括token估算和缓存分析

### test_cache.py
- **test_cache_mechanisms**: 测试增强的缓存机制和文本格式化

### test_history.py
- **test_history_summarization**: 测试改进的历史记录汇总功能

### test_search.py
- **test_search_result_formatting**: 测试搜索结果格式化功能

## 注意事项

1. 所有测试都使用模拟的 ToolExecutor 实例，不会进行实际的API调用
2. 测试主要验证功能逻辑和数据处理的正确性
3. 如果某些测试需要特定的配置或API密钥，请确保已正确设置环境

## 添加新测试

如需添加新测试：

1. 在相应的测试文件中添加新的测试函数
2. 或创建新的测试文件（按功能模块分类）
3. 在 `__init__.py` 中导入新的测试函数
4. 在 `run_all_tests.py` 中添加新测试到测试列表

## 故障排除

如果测试失败，请检查：

1. 确保所有依赖项已安装
2. 确保从正确的目录运行测试
3. 检查是否需要特定的配置文件或环境变量
4. 查看详细的错误信息以确定问题所在 