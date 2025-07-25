# AGIBot 精简测试框架使用指南

## 📋 概述

AGIBot精简测试框架专注于核心功能验证，将测试执行时间从1-2小时优化到15-45分钟，同时保持对关键功能的充分测试覆盖。

## 🎯 测试架构

### 测试分层
```
tests/
├── unit/                    # 单元测试（第一优先级）
│   ├── test_agibot_client.py      # AGIBot客户端核心功能
│   ├── test_task_decomposer.py    # 任务分解器
│   ├── test_config/               # 配置管理测试
│   └── test_tools/                # 工具组件测试
├── integration/             # 集成测试（第一/二优先级）
│   ├── test_single_task_workflow.py  # 单任务完整流程
│   └── test_tool_integration.py      # 工具集成测试
├── performance/             # 性能测试（第二优先级）
│   └── test_basic_performance.py     # 基础性能指标
├── security/                # 安全测试（第二优先级）
│   └── test_workspace_isolation.py   # 工作空间隔离
└── utils/                   # 测试工具
    ├── test_helpers.py              # 测试辅助工具
    └── performance_monitor.py       # 性能监控工具
```

## 🚀 快速开始

### 安装测试依赖
```bash
pip install -r requirements-test.txt
```

### 运行测试

#### 1. 快速验证（开发阶段）
```bash
# 只运行核心功能测试 - 5-10分钟
pytest src/tests/unit/ src/tests/integration/test_single_task_workflow.py -v --tb=short
```

#### 2. 标准验证（提交前）
```bash
# 运行核心和重要功能测试 - 15-25分钟
pytest src/tests/unit/ src/tests/integration/ src/tests/performance/test_basic_performance.py -v
```

#### 3. 完整验证（发布前）
```bash
# 运行所有测试 - 30-45分钟
pytest src/tests/ -v --cov=src
```

#### 4. 并行执行（推荐）
```bash
# 使用多进程加速测试
pip install pytest-xdist
pytest -n auto src/tests/unit/
```

## 📊 测试优先级

### 🎯 第一优先级：核心功能（必须）
**目标：确保AGIBot基本可用**

| 测试文件 | 功能 | 预计时间 |
|---------|------|----------|
| `test_agibot_client.py` | AGIBot客户端接口 | 2-3分钟 |
| `test_task_decomposer.py` | 任务分解逻辑 | 2-3分钟 |
| `test_single_task_workflow.py` | 单任务完整流程 | 3-5分钟 |
| `test_config_loader.py` | 配置管理 | 1-2分钟 |

### 🔧 第二优先级：重要功能（推荐）
**目标：验证扩展功能和异常处理**

| 测试文件 | 功能 | 预计时间 |
|---------|------|----------|
| `test_basic_performance.py` | 基础性能指标 | 3-5分钟 |
| `test_workspace_isolation.py` | 工作空间隔离 | 2-3分钟 |
| `test_file_system_tools.py` | 文件系统工具 | 2-3分钟 |

### 📊 第三优先级：可选验证（CI/CD中可选）
**目标：深度验证和边界条件**

| 测试类型 | 说明 | 预计时间 |
|---------|------|----------|
| 复杂场景测试 | 多工具协作、错误恢复 | 5-10分钟 |
| 扩展性能测试 | 并发、负载测试 | 5-10分钟 |
| 详细安全测试 | 注入攻击、权限验证 | 3-5分钟 |

## 🛠️ 配置选项

### pytest.ini 配置
```ini
[tool:pytest]
testpaths = src/tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = 
    -v
    --tb=short
    --durations=10
markers =
    unit: 单元测试
    integration: 集成测试
    performance: 性能测试
    slow: 慢速测试
    expensive: 资源密集测试
```

### 环境变量控制
```bash
# 跳过资源密集测试
export SKIP_EXPENSIVE_TESTS=1

# 只运行快速测试
export QUICK_TESTS_ONLY=1

# 启用详细输出
export PYTEST_VERBOSE=1
```

## 📈 性能基准

### 当前性能目标
```python
PERFORMANCE_BASELINES = {
    "client_init_time": 1.0,           # 客户端初始化时间（秒）
    "simple_task_time": 5.0,           # 简单任务执行时间（秒）
    "memory_increase_mb": 50,          # 内存增长限制（MB）
    "max_cpu_percent": 80,             # 最大CPU使用率（%）
    "concurrent_tasks_time": 15.0,     # 并发任务总时间（秒）
}
```

### 性能回归检测
```bash
# 运行性能回归测试
pytest src/tests/performance/test_basic_performance.py::TestBasicPerformance::test_performance_regression_check -v
```

## 🔧 测试工具使用

### 使用TestHelper创建测试数据
```python
from src.tests.utils.test_helpers import TestHelper

# 生成测试需求
requirement = TestHelper.generate_test_requirement("simple")

# 创建模拟LLM响应
response = TestHelper.create_mock_llm_response("任务完成！")

# 创建模拟工具调用
tool_call = TestHelper.create_mock_tool_call("edit_file", {
    "target_file": "test.py",
    "code_edit": "print('hello')"
})
```

### 使用PerformanceMonitor监控性能
```python
from src.tests.utils.test_helpers import PerformanceMonitor

with PerformanceMonitor() as monitor:
    # 执行需要监控的代码
    result = agibot_client.chat(messages)
    
# 获取性能指标
metrics = monitor.get_metrics()
print(f"执行时间: {metrics['execution_time']:.3f}s")
```

### 使用TestValidator验证结果
```python
from src.tests.utils.test_helpers import TestValidator

# 验证AGIBot结果格式
assert TestValidator.validate_agibot_result(result)

# 验证Python语法
is_valid, error = TestValidator.validate_python_syntax("test.py")
assert is_valid, f"语法错误: {error}"
```

## 📝 编写新测试

### 单元测试模板
```python
import pytest
from unittest.mock import patch
from src.tests.utils.test_helpers import TestHelper

class TestNewComponent:
    @pytest.fixture
    def component(self):
        return NewComponent()
    
    def test_basic_functionality(self, component):
        """测试基础功能"""
        result = component.basic_method()
        assert result is not None
    
    def test_error_handling(self, component):
        """测试错误处理"""
        with pytest.raises(ValueError):
            component.invalid_operation()
```

### 集成测试模板
```python
@pytest.mark.integration
class TestNewIntegration:
    def test_component_integration(self, test_workspace):
        """测试组件集成"""
        # 模拟外部依赖
        with patch('external_service.call') as mock_service:
            mock_service.return_value = {"status": "success"}
            
            # 执行集成测试
            result = integrated_function()
            assert result["success"] == True
```

## 🎯 测试策略

### 什么时候运行哪些测试

| 场景 | 推荐测试 | 时间 |
|------|----------|------|
| 本地开发 | 第一优先级 | 5-10分钟 |
| Pull Request | 第一+二优先级 | 15-25分钟 |
| 发布前验证 | 全部测试 | 30-45分钟 |
| 夜间CI | 全部测试+压力测试 | 1小时+ |

### 跳过特定测试
```bash
# 跳过慢速测试
pytest -m "not slow"

# 跳过性能测试
pytest -m "not performance"

# 跳过资源密集测试
pytest -m "not expensive"

# 只运行单元测试
pytest -m "unit"
```

## 🚨 故障排除

### 常见问题

#### 1. 测试运行缓慢
```bash
# 使用并行执行
pytest -n auto

# 跳过慢速测试
pytest -m "not slow"

# 只运行核心测试
pytest src/tests/unit/test_agibot_client.py
```

#### 2. 内存不足
```bash
# 跳过内存密集测试
pytest -m "not expensive"

# 增加进程内存限制
ulimit -v 4194304  # 4GB
```

#### 3. 测试超时
```bash
# 设置更长的超时时间
pytest --timeout=300

# 跳过超时测试
pytest -m "not slow"
```

### 调试测试
```bash
# 详细输出
pytest -v -s

# 显示完整堆栈跟踪
pytest --tb=long

# 进入调试器
pytest --pdb

# 只显示失败的测试
pytest --tb=short -q
```

## 📋 测试检查清单

### 新功能开发检查清单
- [ ] 为新功能编写单元测试
- [ ] 更新集成测试（如需要）
- [ ] 运行核心测试确保没有回归
- [ ] 检查测试覆盖率
- [ ] 更新性能基准（如需要）

### 发布前检查清单
- [ ] 运行完整测试套件
- [ ] 检查性能回归
- [ ] 验证安全测试通过
- [ ] 确认测试覆盖率达标（85%+）
- [ ] 检查测试执行时间在预期范围内

## 📊 测试报告

### 生成覆盖率报告
```bash
pytest --cov=src --cov-report=html
# 报告生成在 htmlcov/index.html
```

### 生成性能报告
```bash
pytest src/tests/performance/ --durations=0
```

### 生成测试摘要
```bash
pytest --tb=no -q --disable-warnings
```

这个精简测试框架帮助您在保持测试质量的同时显著减少测试时间和资源消耗！ 