#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基础性能测试
测试AGIBot的基础性能指标，重点关注核心功能的响应时间和资源使用
"""

import pytest
import time
import psutil
import os
import tempfile
import shutil
from unittest.mock import patch, Mock
import sys

# 添加src路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from main import AGIBotClient

@pytest.mark.performance
class TestBasicPerformance:
    """基础性能测试类"""
    
    @pytest.fixture
    def temp_workspace(self):
        """创建临时工作空间"""
        temp_dir = tempfile.mkdtemp(prefix="perf_test_")
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def agibot_client(self):
        """创建性能测试客户端"""
        return AGIBotClient(
            api_key="test_key",
            model="test_model",
            debug_mode=False,  # 关闭调试模式以获得更好的性能
            single_task_mode=True
        )
    
    def test_client_initialization_performance(self):
        """测试客户端初始化性能"""
        start_time = time.time()
        
        client = AGIBotClient(
            api_key="test_key",
            model="test_model"
        )
        
        init_time = time.time() - start_time
        
        assert client is not None
        assert init_time < 1.0, f"Client initialization took too long: {init_time:.3f}s"
        print(f"Client initialization time: {init_time:.3f}s")
    
    def test_simple_task_response_time(self, agibot_client, temp_workspace):
        """测试简单任务的响应时间"""
        messages = [{"role": "user", "content": "创建一个Hello World文件"}]
        
        # 模拟快速LLM响应
        mock_response = {
            "choices": [{
                "message": {
                    "content": "任务完成！",
                    "tool_calls": [],
                    "role": "assistant"
                },
                "finish_reason": "stop"
            }]
        }
        
        with patch('tool_executor.ToolExecutor._call_llm_api', return_value=mock_response):
            with patch.object(agibot_client, '_execute_tool_call') as mock_tool:
                mock_tool.return_value = {"status": "success", "message": "File created"}
                
                start_time = time.time()
                result = agibot_client.chat(messages=messages, dir=temp_workspace, loops=3)
                response_time = time.time() - start_time
        
        assert result["success"] == True
        assert response_time < 5.0, f"Simple task took too long: {response_time:.3f}s"
        print(f"Simple task response time: {response_time:.3f}s")
    
    def test_memory_usage_baseline(self, agibot_client, temp_workspace):
        """测试基础内存使用情况"""
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        messages = [{"role": "user", "content": "执行内存使用测试"}]
        
        mock_response = {
            "choices": [{
                "message": {
                    "content": "内存测试完成！",
                    "tool_calls": [],
                    "role": "assistant"
                },
                "finish_reason": "stop"
            }]
        }
        
        with patch('tool_executor.ToolExecutor._call_llm_api', return_value=mock_response):
            with patch.object(agibot_client, '_execute_tool_call') as mock_tool:
                mock_tool.return_value = {"status": "success"}
                
                result = agibot_client.chat(messages=messages, dir=temp_workspace)
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        assert result["success"] == True
        assert memory_increase < 50, f"Memory increase too high: {memory_increase:.1f}MB"
        print(f"Memory usage - Initial: {initial_memory:.1f}MB, Final: {final_memory:.1f}MB, Increase: {memory_increase:.1f}MB")
    
    def test_concurrent_tasks_basic(self, temp_workspace):
        """测试基础并发任务性能"""
        import threading
        
        num_tasks = 3  # 保持在较小的数量
        results = []
        errors = []
        
        def concurrent_task(task_id):
            try:
                client = AGIBotClient(
                    api_key="test_key",
                    model="test_model",
                    debug_mode=False
                )
                
                mock_response = {
                    "choices": [{
                        "message": {
                            "content": f"Task {task_id} completed",
                            "tool_calls": [],
                            "role": "assistant"
                        },
                        "finish_reason": "stop"
                    }]
                }
                
                with patch('tool_executor.ToolExecutor._call_llm_api', return_value=mock_response):
                    with patch.object(client, '_execute_tool_call') as mock_tool:
                        mock_tool.return_value = {"status": "success"}
                        
                        start_time = time.time()
                        result = client.chat(
                            messages=[{"role": "user", "content": f"Concurrent task {task_id}"}],
                            dir=os.path.join(temp_workspace, f"task_{task_id}")
                        )
                        end_time = time.time()
                        
                        results.append({
                            "task_id": task_id,
                            "success": result["success"],
                            "time": end_time - start_time
                        })
            except Exception as e:
                errors.append({"task_id": task_id, "error": str(e)})
        
        # 启动并发任务
        threads = []
        overall_start = time.time()
        
        for i in range(num_tasks):
            thread = threading.Thread(target=concurrent_task, args=(i,))
            threads.append(thread)
            thread.start()
        
        # 等待所有任务完成
        for thread in threads:
            thread.join(timeout=30)
        
        overall_time = time.time() - overall_start
        
        # 验证并发性能
        assert len(errors) == 0, f"Concurrent tasks had errors: {errors}"
        assert len(results) == num_tasks, f"Expected {num_tasks} results, got {len(results)}"
        assert overall_time < 15, f"Concurrent tasks took too long: {overall_time:.3f}s"
        
        # 计算平均响应时间
        avg_time = sum(r["time"] for r in results) / len(results)
        assert avg_time < 5, f"Average response time too high: {avg_time:.3f}s"
        
        print(f"Concurrent tasks - Total: {overall_time:.3f}s, Average: {avg_time:.3f}s")
    
    def test_repeated_tasks_performance(self, agibot_client, temp_workspace):
        """测试重复任务的性能稳定性"""
        num_iterations = 5
        response_times = []
        
        mock_response = {
            "choices": [{
                "message": {
                    "content": "重复任务完成！",
                    "tool_calls": [],
                    "role": "assistant"
                },
                "finish_reason": "stop"
            }]
        }
        
        with patch('tool_executor.ToolExecutor._call_llm_api', return_value=mock_response):
            with patch.object(agibot_client, '_execute_tool_call') as mock_tool:
                mock_tool.return_value = {"status": "success"}
                
                for i in range(num_iterations):
                    start_time = time.time()
                    
                    result = agibot_client.chat(
                        messages=[{"role": "user", "content": f"重复任务 {i+1}"}],
                        dir=os.path.join(temp_workspace, f"iteration_{i}")
                    )
                    
                    response_time = time.time() - start_time
                    response_times.append(response_time)
                    
                    assert result["success"] == True
                    assert response_time < 5, f"Iteration {i+1} took too long: {response_time:.3f}s"
        
        # 分析性能稳定性
        avg_time = sum(response_times) / len(response_times)
        max_time = max(response_times)
        min_time = min(response_times)
        variance = max_time - min_time
        
        assert avg_time < 3, f"Average response time too high: {avg_time:.3f}s"
        assert variance < 2, f"Response time variance too high: {variance:.3f}s"
        
        print(f"Repeated tasks - Avg: {avg_time:.3f}s, Min: {min_time:.3f}s, Max: {max_time:.3f}s, Variance: {variance:.3f}s")
    
    def test_cpu_usage_monitoring(self, agibot_client, temp_workspace):
        """测试CPU使用情况监控"""
        process = psutil.Process()
        
        # 记录CPU使用情况
        cpu_samples = []
        
        def monitor_cpu():
            for _ in range(10):
                cpu_samples.append(process.cpu_percent(interval=0.1))
        
        import threading
        monitor_thread = threading.Thread(target=monitor_cpu)
        
        mock_response = {
            "choices": [{
                "message": {
                    "content": "CPU监控任务完成！",
                    "tool_calls": [],
                    "role": "assistant"
                },
                "finish_reason": "stop"
            }]
        }
        
        # 启动CPU监控
        monitor_thread.start()
        
        with patch('tool_executor.ToolExecutor._call_llm_api', return_value=mock_response):
            with patch.object(agibot_client, '_execute_tool_call') as mock_tool:
                mock_tool.return_value = {"status": "success"}
                
                result = agibot_client.chat(
                    messages=[{"role": "user", "content": "CPU监控测试"}],
                    dir=temp_workspace
                )
        
        monitor_thread.join()
        
        # 分析CPU使用情况
        if cpu_samples:
            avg_cpu = sum(cpu_samples) / len(cpu_samples)
            max_cpu = max(cpu_samples)
            
            assert result["success"] == True
            assert max_cpu < 80, f"CPU usage too high: {max_cpu:.1f}%"
            
            print(f"CPU usage - Average: {avg_cpu:.1f}%, Peak: {max_cpu:.1f}%")
        else:
            print("Warning: No CPU samples collected")
    
    def test_file_operations_performance(self, agibot_client, temp_workspace):
        """测试文件操作性能"""
        messages = [{"role": "user", "content": "执行文件操作性能测试"}]
        
        # 模拟多个文件操作
        mock_responses = [
            {
                "choices": [{
                    "message": {
                        "content": "创建文件...",
                        "tool_calls": [{
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "edit_file",
                                "arguments": '{"target_file": "test1.txt", "code_edit": "File 1 content"}'
                            }
                        }],
                        "role": "assistant"
                    },
                    "finish_reason": "tool_calls"
                }]
            },
            {
                "choices": [{
                    "message": {
                        "content": "读取文件...",
                        "tool_calls": [{
                            "id": "call_2",
                            "type": "function",
                            "function": {
                                "name": "read_file",
                                "arguments": '{"relative_workspace_path": "test1.txt"}'
                            }
                        }],
                        "role": "assistant"
                    },
                    "finish_reason": "tool_calls"
                }]
            },
            {
                "choices": [{
                    "message": {
                        "content": "文件操作完成！",
                        "tool_calls": [],
                        "role": "assistant"
                    },
                    "finish_reason": "stop"
                }]
            }
        ]
        
        call_count = 0
        def mock_sequence(*args, **kwargs):
            nonlocal call_count
            if call_count < len(mock_responses):
                response = mock_responses[call_count]
                call_count += 1
                return response
            return mock_responses[-1]
        
        with patch('tool_executor.ToolExecutor._call_llm_api', side_effect=mock_sequence):
            with patch.object(agibot_client, '_execute_tool_call') as mock_tool:
                mock_tool.return_value = {"status": "success", "content": "File content"}
                
                start_time = time.time()
                result = agibot_client.chat(messages=messages, dir=temp_workspace, loops=5)
                file_ops_time = time.time() - start_time
        
        assert result["success"] == True
        assert file_ops_time < 10, f"File operations took too long: {file_ops_time:.3f}s"
        assert mock_tool.call_count >= 2, "Expected multiple file operations"
        
        print(f"File operations performance: {file_ops_time:.3f}s for {mock_tool.call_count} operations")
    
    def test_performance_regression_check(self, agibot_client, temp_workspace):
        """基础性能回归检查"""
        # 定义性能基准（这些数值应该基于实际的性能基准测试）
        PERFORMANCE_BASELINES = {
            "max_response_time": 5.0,  # 最大响应时间（秒）
            "max_memory_increase": 30,  # 最大内存增长（MB）
            "max_cpu_usage": 70,       # 最大CPU使用率（%）
        }
        
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024
        
        mock_response = {
            "choices": [{
                "message": {
                    "content": "性能回归测试完成！",
                    "tool_calls": [],
                    "role": "assistant"
                },
                "finish_reason": "stop"
            }]
        }
        
        # 执行基准测试
        with patch('tool_executor.ToolExecutor._call_llm_api', return_value=mock_response):
            with patch.object(agibot_client, '_execute_tool_call') as mock_tool:
                mock_tool.return_value = {"status": "success"}
                
                start_time = time.time()
                result = agibot_client.chat(
                    messages=[{"role": "user", "content": "性能回归检查"}],
                    dir=temp_workspace
                )
                response_time = time.time() - start_time
        
        final_memory = process.memory_info().rss / 1024 / 1024
        memory_increase = final_memory - initial_memory
        
        # 验证性能指标
        assert result["success"] == True
        assert response_time <= PERFORMANCE_BASELINES["max_response_time"], \
            f"Response time regression: {response_time:.3f}s > {PERFORMANCE_BASELINES['max_response_time']}s"
        assert memory_increase <= PERFORMANCE_BASELINES["max_memory_increase"], \
            f"Memory usage regression: {memory_increase:.1f}MB > {PERFORMANCE_BASELINES['max_memory_increase']}MB"
        
        print(f"Performance regression check - Response: {response_time:.3f}s, Memory: {memory_increase:.1f}MB")
        print("✅ All performance indicators within acceptable ranges") 