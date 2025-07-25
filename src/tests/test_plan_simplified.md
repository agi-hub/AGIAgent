# AGIBot 精简测试计划

## 测试目标
构建针对性强、资源消耗合理的测试框架，重点验证核心功能的正确性和稳定性。

## 测试分层策略

### 🎯 **第一优先级：核心功能测试**
**目标：确保AGIBot基本可用**

#### 1. 单元测试 - 核心组件
- `test_agibot_client.py` - AGIBotClient接口测试
- `test_task_decomposer.py` - 任务分解核心逻辑
- `test_tool_executor.py` - 工具执行器基础功能
- `test_config_loader.py` - 配置管理

#### 2. 集成测试 - 关键流程  
- `test_single_task_workflow.py` - 单任务完整执行流程
- `test_tool_integration.py` - 核心工具集成测试
- `test_file_operations.py` - 文件操作集成测试

**估计时间：10-15分钟运行**

### 🔧 **第二优先级：重要功能测试**
**目标：验证扩展功能和异常处理**

#### 3. 功能测试
- `test_multi_agent_basic.py` - 基础多代理功能
- `test_mcp_integration.py` - MCP集成基础测试
- `test_error_handling.py` - 基本错误处理

#### 4. 基础性能测试
- `test_basic_performance.py` - 基础性能指标
- `test_workspace_isolation.py` - 工作空间隔离

**估计时间：5-10分钟运行**

### 📊 **第三优先级：可选验证测试**
**目标：深度验证和边界条件（CI/CD中可选）**

#### 5. 扩展测试
- `test_complex_scenarios.py` - 复杂场景（简化版）
- `test_load_basic.py` - 基础负载测试
- `test_security_basic.py` - 基础安全验证

**估计时间：10-20分钟运行**

## 测试执行策略

### 快速验证（开发阶段）
```bash
# 只运行第一优先级测试
pytest src/tests/unit/ src/tests/integration/test_single_task_workflow.py -v --tb=short

# 预期运行时间：5-10分钟
```

### 标准验证（提交前）
```bash
# 运行第一、二优先级测试
pytest src/tests/unit/ src/tests/integration/ src/tests/performance/test_basic_performance.py -v

# 预期运行时间：15-25分钟
```

### 完整验证（发布前）
```bash
# 运行所有测试
pytest src/tests/ -v --cov=src

# 预期运行时间：30-45分钟
```

## 资源优化措施

### 1. 减少重复测试
- 合并相似的测试场景
- 使用参数化测试减少重复代码
- 共享测试夹具和模拟数据

### 2. 并行执行
```bash
# 使用pytest-xdist并行运行
pytest -n auto src/tests/unit/
```

### 3. 智能跳过
```python
@pytest.mark.skipif(not EXPENSIVE_TESTS, reason="跳过资源密集测试")
def test_complex_scenario():
    pass
```

### 4. 分环境测试
- **开发环境**：只运行核心功能测试
- **CI环境**：运行核心+重要功能测试  
- **发布环境**：运行完整测试套件

## 测试覆盖率目标

- **核心模块**：90%+ 覆盖率
- **工具模块**：80%+ 覆盖率
- **扩展功能**：70%+ 覆盖率
- **总体目标**：85%+ 覆盖率

## 维护策略

### 测试更新原则
1. 新功能必须有对应的核心测试
2. Bug修复必须有回归测试
3. 性能改进需要对应的性能基准测试

### 定期评估
- 每月评估测试执行时间和资源消耗
- 每季度评估测试价值和维护成本
- 及时移除过时或低价值的测试

## 工具推荐

```toml
# pyproject.toml 测试配置
[tool.pytest.ini_options]
testpaths = ["src/tests"]
python_files = ["test_*.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]
addopts = [
    "-v",
    "--tb=short", 
    "--cov=src",
    "--cov-report=term-missing",
    "--durations=10"  # 显示最慢的10个测试
]
markers = [
    "unit: 单元测试",
    "integration: 集成测试", 
    "e2e: 端到端测试",
    "performance: 性能测试",
    "slow: 慢速测试",
    "expensive: 资源密集测试"
]
```

这个精简方案将测试执行时间从可能的1-2小时减少到15-45分钟，同时保持对核心功能的充分验证。 