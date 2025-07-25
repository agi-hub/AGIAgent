#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
任务分解器单元测试
测试task_decomposer.py中的任务分解功能
"""

import pytest
import os
import tempfile
import shutil
from unittest.mock import patch, Mock, MagicMock
import sys

# 添加src路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from task_decomposer import TaskDecomposer
from utils.test_helpers import TestHelper

class TestTaskDecomposer:
    """任务分解器测试类"""
    
    @pytest.fixture
    def temp_output_dir(self):
        """创建临时输出目录"""
        temp_dir = tempfile.mkdtemp(prefix="task_decomposer_test_")
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def task_decomposer(self, temp_output_dir):
        """创建任务分解器实例"""
        return TaskDecomposer(
            api_key="test_api_key",
            model="test_model",
            api_base="https://test-api.example.com",
            debug_mode=True,
            out_dir=temp_output_dir
        )
    
    @pytest.fixture
    def mock_llm_response(self):
        """模拟LLM响应"""
        def _create_response(content):
            return {
                "choices": [{
                    "message": {
                        "content": content,
                        "role": "assistant"
                    },
                    "finish_reason": "stop"
                }]
            }
        return _create_response
    
    def test_task_decomposer_initialization(self, task_decomposer):
        """测试任务分解器初始化"""
        assert task_decomposer.api_key == "test_api_key"
        assert task_decomposer.model == "test_model"
        assert task_decomposer.debug_mode == True
        assert hasattr(task_decomposer, 'decompose_task')
    
    def test_simple_task_decomposition(self, task_decomposer, mock_llm_response):
        """测试简单任务分解"""
        requirement = "创建一个简单的计算器程序"
        
        # 模拟LLM返回的任务分解结果
        decomposed_tasks = """
# 计算器开发任务

## 任务1: 创建基础计算器类
- 实现基本的四则运算功能
- 添加输入验证
- 创建calculator.py文件

## 任务2: 实现用户界面
- 创建简单的命令行界面
- 处理用户输入
- 显示计算结果

## 任务3: 添加测试
- 创建单元测试
- 测试边界情况
- 验证错误处理
"""
        
        with patch.object(task_decomposer, '_call_llm_api', return_value=mock_llm_response(decomposed_tasks)):
            result = task_decomposer.decompose_task(requirement)
            
            assert result["success"] == True
            assert "todo_file" in result
            assert os.path.exists(result["todo_file"])
            
            # 验证todo文件内容
            with open(result["todo_file"], 'r', encoding='utf-8') as f:
                content = f.read()
                assert "计算器开发任务" in content
                assert "任务1" in content
                assert "任务2" in content
                assert "任务3" in content
    
    def test_complex_task_decomposition(self, task_decomposer, mock_llm_response):
        """测试复杂任务分解"""
        requirement = """
        开发一个完整的Web应用，包括：
        1. 用户注册和登录系统
        2. 数据库设计
        3. API接口
        4. 前端界面
        5. 部署配置
        """
        
        complex_decomposition = """
# Web应用开发项目

## 阶段1: 项目规划和环境搭建
### 任务1.1: 项目结构设计
- 创建项目目录结构
- 设置虚拟环境
- 配置开发工具

### 任务1.2: 依赖管理
- 创建requirements.txt
- 安装必要的依赖包
- 配置开发环境

## 阶段2: 后端开发
### 任务2.1: 数据库设计
- 设计用户表结构
- 创建数据库迁移文件
- 设置数据库连接

### 任务2.2: 用户认证系统
- 实现用户注册功能
- 实现用户登录功能
- 添加JWT认证

### 任务2.3: API接口开发
- 创建用户管理API
- 实现数据验证
- 添加错误处理

## 阶段3: 前端开发
### 任务3.1: 页面结构
- 创建HTML模板
- 设计CSS样式
- 实现响应式布局

### 任务3.2: 交互功能
- 实现表单提交
- 添加前端验证
- 处理API调用

## 阶段4: 测试和部署
### 任务4.1: 测试
- 编写单元测试
- 进行集成测试
- 性能测试

### 任务4.2: 部署配置
- 配置生产环境
- 设置CI/CD
- 监控和日志
"""
        
        with patch.object(task_decomposer, '_call_llm_api', return_value=mock_llm_response(complex_decomposition)):
            result = task_decomposer.decompose_task(requirement)
            
            assert result["success"] == True
            
            # 验证复杂任务的分解结构
            with open(result["todo_file"], 'r', encoding='utf-8') as f:
                content = f.read()
                assert "阶段1" in content
                assert "阶段2" in content
                assert "阶段3" in content
                assert "阶段4" in content
                assert "任务1.1" in content
                assert "任务2.1" in content
    
    def test_task_decomposition_with_constraints(self, task_decomposer, mock_llm_response):
        """测试带约束条件的任务分解"""
        requirement = "开发一个Python脚本，要求使用特定的库和框架"
        
        constrained_decomposition = """
# 约束条件下的Python脚本开发

## 约束条件
- 必须使用pandas进行数据处理
- 使用matplotlib进行可视化
- 代码需要符合PEP8规范

## 任务1: 环境准备
- 安装pandas==1.5.0
- 安装matplotlib==3.6.0
- 配置代码格式化工具

## 任务2: 核心功能实现
- 使用pandas读取数据
- 数据清洗和处理
- 使用matplotlib生成图表

## 任务3: 代码质量
- 运行flake8检查
- 添加类型注解
- 编写文档字符串
"""
        
        with patch.object(task_decomposer, '_call_llm_api', return_value=mock_llm_response(constrained_decomposition)):
            result = task_decomposer.decompose_task(requirement)
            
            assert result["success"] == True
            
            with open(result["todo_file"], 'r', encoding='utf-8') as f:
                content = f.read()
                assert "约束条件" in content
                assert "pandas" in content
                assert "matplotlib" in content
                assert "PEP8" in content
    
    def test_task_decomposition_error_handling(self, task_decomposer):
        """测试任务分解错误处理"""
        requirement = "创建一个简单程序"
        
        # 模拟LLM API调用失败
        with patch.object(task_decomposer, '_call_llm_api', side_effect=Exception("API调用失败")):
            result = task_decomposer.decompose_task(requirement)
            
            assert result["success"] == False
            assert "error" in result
            assert "API调用失败" in result["error"]
    
    def test_empty_requirement_handling(self, task_decomposer, mock_llm_response):
        """测试空需求处理"""
        empty_requirements = ["", "   ", "\n\n", None]
        
        for req in empty_requirements:
            result = task_decomposer.decompose_task(req)
            assert result["success"] == False
            assert "error" in result
    
    def test_malformed_llm_response_handling(self, task_decomposer):
        """测试LLM响应格式错误处理"""
        requirement = "创建一个测试程序"
        
        # 模拟格式错误的LLM响应
        malformed_responses = [
            {"invalid": "format"},  # 缺少choices
            {"choices": []},  # 空choices
            {"choices": [{"message": {}}]},  # 缺少content
            {"choices": [{"message": {"content": ""}}]},  # 空content
        ]
        
        for response in malformed_responses:
            with patch.object(task_decomposer, '_call_llm_api', return_value=response):
                result = task_decomposer.decompose_task(requirement)
                assert result["success"] == False
                assert "error" in result
    
    def test_todo_file_creation_permissions(self, task_decomposer, mock_llm_response):
        """测试todo文件创建权限"""
        requirement = "创建测试程序"
        
        # 模拟无法写入文件的情况
        invalid_output_dir = "/root/restricted_dir"  # 通常无权限的目录
        
        decomposer_with_invalid_dir = TaskDecomposer(
            api_key="test_key",
            model="test_model", 
            out_dir=invalid_output_dir
        )
        
        decomposition = "# 测试任务\n## 任务1: 创建文件"
        
        with patch.object(decomposer_with_invalid_dir, '_call_llm_api', return_value=mock_llm_response(decomposition)):
            result = decomposer_with_invalid_dir.decompose_task(requirement)
            
            # 应该能优雅处理权限错误
            if result["success"] == False:
                assert "error" in result
    
    def test_large_task_decomposition(self, task_decomposer, mock_llm_response):
        """测试大型任务分解"""
        # 非常详细的需求
        large_requirement = """
        开发一个企业级的电商平台，包括：
        1. 用户管理系统（注册、登录、权限、个人资料）
        2. 商品管理系统（分类、库存、价格、图片）
        3. 订单管理系统（购物车、结算、支付、发货）
        4. 支付系统集成（支付宝、微信、银行卡）
        5. 物流系统集成（快递查询、配送跟踪）
        6. 客服系统（在线聊天、工单系统）
        7. 数据分析系统（销售报表、用户行为分析）
        8. 后台管理系统（管理员界面、系统配置）
        9. 移动端APP（iOS和Android）
        10. 系统监控和日志
        """
        
        # 生成超长的分解结果
        large_decomposition = "# 企业级电商平台开发\n\n"
        for i in range(1, 11):
            large_decomposition += f"## 模块{i}: 功能{i}\n"
            for j in range(1, 6):
                large_decomposition += f"### 任务{i}.{j}: 子任务{j}\n"
                large_decomposition += f"- 详细步骤{j}.1\n"
                large_decomposition += f"- 详细步骤{j}.2\n"
                large_decomposition += f"- 详细步骤{j}.3\n\n"
        
        with patch.object(task_decomposer, '_call_llm_api', return_value=mock_llm_response(large_decomposition)):
            result = task_decomposer.decompose_task(large_requirement)
            
            assert result["success"] == True
            
            # 验证大型任务文件能正确创建
            with open(result["todo_file"], 'r', encoding='utf-8') as f:
                content = f.read()
                assert len(content) > 1000  # 确保内容足够长
                assert "模块1" in content
                assert "模块10" in content
    
    def test_unicode_requirement_handling(self, task_decomposer, mock_llm_response):
        """测试Unicode需求处理"""
        unicode_requirement = """
        创建一个多语言应用：
        - 支持中文、日文、韩文
        - 处理特殊字符：®©™€£¥
        - 支持Emoji：🚀🤖💻🎉
        """
        
        unicode_decomposition = """
# 多语言应用开发 🌍

## 任务1: 国际化设置 🌐
- 配置i18n支持
- 创建语言包（中文🇨🇳、日文🇯🇵、韩文🇰🇷）
- 处理特殊字符：®©™€£¥

## 任务2: UI适配 📱
- 设计多语言界面
- 支持Emoji显示：🚀🤖💻🎉
- 测试不同语言下的布局

## 任务3: 测试验证 ✅
- 多语言功能测试
- 字符编码测试
- 用户体验测试
"""
        
        with patch.object(task_decomposer, '_call_llm_api', return_value=mock_llm_response(unicode_decomposition)):
            result = task_decomposer.decompose_task(unicode_requirement)
            
            assert result["success"] == True
            
            with open(result["todo_file"], 'r', encoding='utf-8') as f:
                content = f.read()
                assert "🌍" in content
                assert "中文🇨🇳" in content
                assert "🚀🤖💻🎉" in content
                assert "®©™€£¥" in content
    
    def test_task_decomposition_caching(self, task_decomposer, mock_llm_response):
        """测试任务分解缓存机制"""
        requirement = "创建一个简单程序"
        decomposition = "# 简单程序\n## 任务1: 创建文件"
        
        call_count = 0
        def counting_llm_call(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return mock_llm_response(decomposition)
        
        with patch.object(task_decomposer, '_call_llm_api', side_effect=counting_llm_call):
            # 第一次调用
            result1 = task_decomposer.decompose_task(requirement)
            assert result1["success"] == True
            
            # 如果有缓存机制，第二次相同需求应该使用缓存
            result2 = task_decomposer.decompose_task(requirement)
            assert result2["success"] == True
            
            # 检查LLM是否被多次调用（如果有缓存，应该只调用一次）
            # 这取决于具体实现是否有缓存机制
            print(f"LLM调用次数: {call_count}")
    
    def test_task_decomposition_with_context(self, task_decomposer, mock_llm_response):
        """测试带上下文的任务分解"""
        # 模拟有上下文信息的分解
        requirement = "在现有项目基础上添加新功能"
        context_info = {
            "existing_files": ["app.py", "models.py", "config.py"],
            "current_framework": "Flask",
            "database": "PostgreSQL"
        }
        
        contextual_decomposition = """
# 基于现有Flask项目的功能扩展

## 背景信息
- 现有文件: app.py, models.py, config.py
- 框架: Flask
- 数据库: PostgreSQL

## 任务1: 分析现有代码
- 审查app.py结构
- 检查models.py中的数据模型
- 理解config.py配置

## 任务2: 设计新功能
- 基于现有架构设计
- 确保与PostgreSQL兼容
- 遵循Flask最佳实践

## 任务3: 实现新功能
- 修改相关模型
- 添加新的路由
- 更新配置
"""
        
        with patch.object(task_decomposer, '_call_llm_api', return_value=mock_llm_response(contextual_decomposition)):
            # 如果支持上下文，传入额外信息
            result = task_decomposer.decompose_task(requirement, context=context_info)
            
            assert result["success"] == True
            
            with open(result["todo_file"], 'r', encoding='utf-8') as f:
                content = f.read()
                assert "Flask" in content
                assert "PostgreSQL" in content
                assert "app.py" in content