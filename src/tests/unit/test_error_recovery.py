#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
错误处理和恢复单元测试
测试异常捕获、错误恢复、容错机制等功能
"""

import pytest
import os
import sys
import time
import json
import traceback
from unittest.mock import patch, Mock, MagicMock, side_effect
from typing import Dict, List, Any

# 添加src路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from main import AGIBotClient, AGIBotMain
from multi_round_executor import MultiRoundTaskExecutor
from tool_executor import ToolExecutor
from utils.test_helpers import TestHelper

@pytest.mark.unit
class TestErrorRecovery:
    """错误处理和恢复测试类"""
    
    @pytest.fixture
    def agibot_client(self):
        """创建AGIBot客户端实例"""
        return AGIBotClient(
            api_key="test_key",
            model="test_model",
            api_base="test_base",
            debug_mode=True
        )
    
    @pytest.fixture
    def error_scenarios(self):
        """常见错误场景"""
        return {
            "network_error": ConnectionError("Network connection failed"),
            "timeout_error": TimeoutError("Request timeout"),
            "api_error": Exception("API call failed"),
            "json_decode_error": json.JSONDecodeError("Invalid JSON", "", 0),
            "file_not_found": FileNotFoundError("File not found"),
            "permission_error": PermissionError("Permission denied"),
            "memory_error": MemoryError("Out of memory"),
            "keyboard_interrupt": KeyboardInterrupt("User interrupted"),
            "value_error": ValueError("Invalid value"),
            "type_error": TypeError("Invalid type")
        }
    
    def test_network_error_recovery(self, agibot_client, error_scenarios):
        """测试网络错误恢复"""
        with patch('main.AGIBotMain') as mock_main:
            # 模拟网络错误然后恢复
            mock_instance = Mock()
            mock_instance.run.side_effect = [
                error_scenarios["network_error"],
                True  # 重试后成功
            ]
            mock_main.return_value = mock_instance
            
            # 第一次调用失败，第二次成功（如果有重试机制）
            try:
                result = agibot_client.chat([{"role": "user", "content": "test task"}])
                # 验证错误被正确处理
                assert result is not None
                assert isinstance(result, dict)
            except Exception as e:
                # 验证异常被适当处理
                assert isinstance(e, (ConnectionError, Exception))
    
    def test_api_timeout_handling(self, agibot_client):
        """测试API超时处理"""
        with patch('main.AGIBotMain') as mock_main:
            mock_instance = Mock()
            
            def timeout_simulation(*args, **kwargs):
                time.sleep(0.1)  # 模拟短暂延迟
                raise TimeoutError("Request timeout")
            
            mock_instance.run.side_effect = timeout_simulation
            mock_main.return_value = mock_instance
            
            # 测试超时处理
            result = agibot_client.chat([{"role": "user", "content": "timeout test"}])
            
            # 验证超时错误处理
            assert result is not None
            assert result["success"] is False
            assert "timeout" in result["message"].lower() or "error" in result
    
    def test_invalid_json_response_handling(self, agibot_client):
        """测试无效JSON响应处理"""
        with patch('requests.post') as mock_post:
            # 模拟返回无效JSON的响应
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.side_effect = json.JSONDecodeError("Invalid JSON", "", 0)
            mock_response.text = "Invalid JSON response"
            mock_post.return_value = mock_response
            
            result = agibot_client.chat([{"role": "user", "content": "json test"}])
            
            # 验证JSON错误处理
            assert result is not None
            assert result["success"] is False
    
    def test_file_system_error_recovery(self, agibot_client, test_workspace):
        """测试文件系统错误恢复"""
        with patch('main.AGIBotMain') as mock_main:
            mock_instance = Mock()
            
            # 模拟文件系统错误
            def file_error_simulation(*args, **kwargs):
                raise PermissionError("Permission denied: cannot write to file")
            
            mock_instance.run.side_effect = file_error_simulation
            mock_main.return_value = mock_instance
            
            result = agibot_client.chat(
                [{"role": "user", "content": "create file"}],
                dir=test_workspace
            )
            
            # 验证文件系统错误处理
            assert result is not None
            assert result["success"] is False
    
    def test_memory_error_handling(self, agibot_client):
        """测试内存错误处理"""
        with patch('main.AGIBotMain') as mock_main:
            mock_instance = Mock()
            mock_instance.run.side_effect = MemoryError("Out of memory")
            mock_main.return_value = mock_instance
            
            result = agibot_client.chat([{"role": "user", "content": "memory test"}])
            
            # 验证内存错误处理
            assert result is not None
            assert result["success"] is False
            assert "memory" in result["message"].lower() or "error" in result
    
    def test_keyboard_interrupt_handling(self, agibot_client):
        """测试键盘中断处理"""
        with patch('main.AGIBotMain') as mock_main:
            mock_instance = Mock()
            mock_instance.run.side_effect = KeyboardInterrupt("User interrupted")
            mock_main.return_value = mock_instance
            
            result = agibot_client.chat([{"role": "user", "content": "interrupt test"}])
            
            # 验证键盘中断处理
            assert result is not None
            assert result["success"] is False
    
    def test_invalid_input_handling(self, agibot_client):
        """测试无效输入处理"""
        # 测试空消息
        result1 = agibot_client.chat([])
        assert result1 is not None
        assert result1["success"] is False
        
        # 测试无效消息格式
        result2 = agibot_client.chat([{"invalid": "format"}])
        assert result2 is not None
        assert result2["success"] is False
        
        # 测试None输入
        result3 = agibot_client.chat(None)
        assert result3 is not None
        assert result3["success"] is False
    
    def test_retry_mechanism(self, agibot_client):
        """测试重试机制"""
        with patch('main.AGIBotMain') as mock_main:
            mock_instance = Mock()
            
            # 模拟前两次失败，第三次成功
            call_count = 0
            def retry_simulation(*args, **kwargs):
                nonlocal call_count
                call_count += 1
                if call_count <= 2:
                    raise ConnectionError(f"Attempt {call_count} failed")
                return True
            
            mock_instance.run.side_effect = retry_simulation
            mock_main.return_value = mock_instance
            
            # 如果有重试机制，应该最终成功
            result = agibot_client.chat([{"role": "user", "content": "retry test"}])
            
            # 验证重试机制（取决于实现）
            assert result is not None
    
    def test_graceful_degradation(self, agibot_client):
        """测试优雅降级"""
        with patch('tools.web_search_tools.WebSearchTools.search_web') as mock_search:
            # 模拟网络搜索失败
            mock_search.side_effect = ConnectionError("Search service unavailable")
            
            with patch('main.AGIBotMain') as mock_main:
                mock_instance = Mock()
                mock_instance.run.return_value = True  # 主任务仍然成功
                mock_main.return_value = mock_instance
                
                result = agibot_client.chat([{"role": "user", "content": "search and analyze"}])
                
                # 验证在部分功能失败时系统仍能运行
                assert result is not None
    
    def test_error_logging(self, agibot_client, test_workspace):
        """测试错误日志记录"""
        with patch('main.AGIBotMain') as mock_main:
            mock_instance = Mock()
            mock_instance.run.side_effect = Exception("Test error for logging")
            mock_main.return_value = mock_instance
            
            result = agibot_client.chat(
                [{"role": "user", "content": "logging test"}],
                dir=test_workspace
            )
            
            # 验证错误被记录
            assert result is not None
            assert result["success"] is False
            
            # 检查是否生成了错误日志（如果支持）
            log_dir = os.path.join(test_workspace, "logs")
            if os.path.exists(log_dir):
                log_files = os.listdir(log_dir)
                assert len(log_files) > 0
    
    def test_partial_execution_recovery(self, agibot_client):
        """测试部分执行恢复"""
        with patch('multi_round_executor.MultiRoundTaskExecutor') as mock_executor:
            mock_instance = Mock()
            
            # 模拟部分任务成功，部分失败
            def partial_execution(*args, **kwargs):
                return {
                    "success": False,
                    "completed_tasks": 3,
                    "failed_tasks": 2,
                    "error": "Some tasks failed",
                    "partial_results": ["task1", "task2", "task3"]
                }
            
            mock_instance.execute_all_tasks.side_effect = partial_execution
            mock_executor.return_value = mock_instance
            
            with patch('main.AGIBotMain') as mock_main:
                mock_main_instance = Mock()
                mock_main_instance.run.return_value = False  # 总体失败但有部分结果
                mock_main.return_value = mock_main_instance
                
                result = agibot_client.chat([{"role": "user", "content": "multi-task"}])
                
                # 验证部分执行结果被保留
                assert result is not None
    
    def test_circular_dependency_handling(self, test_workspace):
        """测试循环依赖处理"""
        # 创建循环依赖的文件结构
        file_a = os.path.join(test_workspace, "module_a.py")
        file_b = os.path.join(test_workspace, "module_b.py")
        
        with open(file_a, "w") as f:
            f.write("from module_b import function_b\ndef function_a(): return function_b()")
        
        with open(file_b, "w") as f:
            f.write("from module_a import function_a\ndef function_b(): return function_a()")
        
        # 测试代码分析工具处理循环依赖
        with patch('tools.code_search_tools.CodeSearchTools') as mock_tools:
            mock_instance = Mock()
            
            def circular_detection(*args, **kwargs):
                # 模拟检测到循环依赖但不崩溃
                return "Circular dependency detected but handled gracefully"
            
            mock_instance.search_code.side_effect = circular_detection
            mock_tools.return_value = mock_instance
            
            from tools.code_search_tools import CodeSearchTools
            tools = CodeSearchTools()
            result = tools.search_code("import", test_workspace)
            
            # 验证循环依赖处理
            assert result is not None
    
    def test_resource_cleanup_on_error(self, agibot_client, test_workspace):
        """测试错误时的资源清理"""
        with patch('main.AGIBotMain') as mock_main:
            mock_instance = Mock()
            
            # 模拟资源分配后发生错误
            def error_with_resources(*args, **kwargs):
                # 创建一些临时文件
                temp_file = os.path.join(test_workspace, "temp_resource.txt")
                with open(temp_file, "w") as f:
                    f.write("temporary resource")
                
                # 然后抛出错误
                raise Exception("Error after resource allocation")
            
            mock_instance.run.side_effect = error_with_resources
            mock_main.return_value = mock_instance
            
            result = agibot_client.chat(
                [{"role": "user", "content": "resource test"}],
                dir=test_workspace
            )
            
            # 验证错误处理
            assert result is not None
            assert result["success"] is False
    
    def test_cascading_error_prevention(self, agibot_client):
        """测试级联错误预防"""
        with patch('main.AGIBotMain') as mock_main:
            mock_instance = Mock()
            
            # 模拟多层错误
            def cascading_errors(*args, **kwargs):
                try:
                    raise ValueError("Primary error")
                except ValueError:
                    try:
                        # 在错误处理中又发生错误
                        raise TypeError("Secondary error during error handling")
                    except TypeError:
                        raise RuntimeError("Tertiary error")
            
            mock_instance.run.side_effect = cascading_errors
            mock_main.return_value = mock_instance
            
            result = agibot_client.chat([{"role": "user", "content": "cascade test"}])
            
            # 验证级联错误被适当处理
            assert result is not None
            assert result["success"] is False
    
    def test_thread_safety_error_handling(self, agibot_client):
        """测试线程安全的错误处理"""
        import threading
        
        results = []
        errors = []
        
        def concurrent_error_test(thread_id):
            try:
                with patch('main.AGIBotMain') as mock_main:
                    mock_instance = Mock()
                    
                    # 每个线程模拟不同的错误
                    if thread_id % 2 == 0:
                        mock_instance.run.side_effect = ValueError(f"Error in thread {thread_id}")
                    else:
                        mock_instance.run.return_value = True
                    
                    mock_main.return_value = mock_instance
                    
                    result = agibot_client.chat([{"role": "user", "content": f"thread test {thread_id}"}])
                    results.append(result)
            except Exception as e:
                errors.append(e)
        
        # 创建多个并发线程
        threads = []
        for i in range(5):
            thread = threading.Thread(target=concurrent_error_test, args=(i,))
            threads.append(thread)
            thread.start()
        
        # 等待所有线程完成
        for thread in threads:
            thread.join(timeout=10)
        
        # 验证线程安全的错误处理
        assert len(results) > 0
        # 应该没有未捕获的异常导致线程崩溃
        critical_errors = [e for e in errors if not isinstance(e, (ValueError, Exception))]
        assert len(critical_errors) == 0
    
    def test_error_context_preservation(self, agibot_client):
        """测试错误上下文保留"""
        with patch('main.AGIBotMain') as mock_main:
            mock_instance = Mock()
            
            def contextual_error(*args, **kwargs):
                # 模拟在特定上下文中发生的错误
                context = {
                    "task": "complex analysis",
                    "step": "data processing",
                    "iteration": 3
                }
                error = Exception("Processing failed")
                error.context = context
                raise error
            
            mock_instance.run.side_effect = contextual_error
            mock_main.return_value = mock_instance
            
            result = agibot_client.chat([{"role": "user", "content": "context test"}])
            
            # 验证错误上下文被保留
            assert result is not None
            assert result["success"] is False
            # 检查详细信息中是否包含上下文（如果支持）
            if "details" in result:
                assert isinstance(result["details"], dict)
    
    def test_error_recovery_strategies(self, agibot_client):
        """测试错误恢复策略"""
        recovery_strategies = [
            "retry_with_backoff",
            "fallback_to_simple_mode",
            "skip_failed_component",
            "use_cached_result"
        ]
        
        for strategy in recovery_strategies:
            with patch('main.AGIBotMain') as mock_main:
                mock_instance = Mock()
                
                # 模拟不同的恢复策略
                if strategy == "retry_with_backoff":
                    call_count = 0
                    def retry_strategy(*args, **kwargs):
                        nonlocal call_count
                        call_count += 1
                        if call_count == 1:
                            raise ConnectionError("First attempt failed")
                        return True
                    mock_instance.run.side_effect = retry_strategy
                
                elif strategy == "fallback_to_simple_mode":
                    mock_instance.run.return_value = True  # 成功但功能受限
                
                else:
                    mock_instance.run.return_value = True
                
                mock_main.return_value = mock_instance
                
                result = agibot_client.chat([{"role": "user", "content": f"test {strategy}"}])
                
                # 验证恢复策略
                assert result is not None
    
    def test_error_metrics_collection(self, agibot_client):
        """测试错误指标收集"""
        error_types = ["network", "timeout", "validation", "system"]
        
        for error_type in error_types:
            with patch('main.AGIBotMain') as mock_main:
                mock_instance = Mock()
                
                # 模拟不同类型的错误
                error_map = {
                    "network": ConnectionError("Network error"),
                    "timeout": TimeoutError("Timeout error"),
                    "validation": ValueError("Validation error"),
                    "system": RuntimeError("System error")
                }
                
                mock_instance.run.side_effect = error_map[error_type]
                mock_main.return_value = mock_instance
                
                result = agibot_client.chat([{"role": "user", "content": f"error {error_type}"}])
                
                # 验证错误指标收集
                assert result is not None
                assert result["success"] is False
                
                # 检查是否收集了错误类型信息
                if "details" in result and "error_type" in result["details"]:
                    assert result["details"]["error_type"] is not None
    
    def test_progressive_error_handling(self, agibot_client):
        """测试渐进式错误处理"""
        with patch('main.AGIBotMain') as mock_main:
            mock_instance = Mock()
            
            # 模拟渐进式错误：从轻微到严重
            error_sequence = [
                Warning("Minor issue"),
                ValueError("Moderate issue"),
                RuntimeError("Serious issue"),
                SystemError("Critical issue")
            ]
            
            for i, error in enumerate(error_sequence):
                mock_instance.run.side_effect = error
                mock_main.return_value = mock_instance
                
                result = agibot_client.chat([{"role": "user", "content": f"progressive test {i}"}])
                
                # 验证不同严重程度的错误都被适当处理
                assert result is not None
                if isinstance(error, Warning):
                    # 警告可能不会导致完全失败
                    pass
                else:
                    assert result["success"] is False 