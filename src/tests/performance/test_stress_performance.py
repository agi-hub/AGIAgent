#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AGIBot性能压力测试
测试系统在高负载下的性能表现
"""

import pytest
import os
import time
import threading
import psutil
import gc
from unittest.mock import patch, Mock
import sys

# 添加src路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from main import AGIBotClient, AGIBotMain
from utils.test_helpers import TestHelper
from utils.performance_monitor import PerformanceMonitor, ResourceTracker

@pytest.mark.performance
@pytest.mark.slow
class TestStressPerformance:
    """压力性能测试类"""
    
    def test_concurrent_clients_stress(self, test_workspace):
        """测试并发客户端压力"""
        num_clients = 10
        num_requests_per_client = 5
        
        results = []
        errors = []
        
        def client_worker(client_id):
            try:
                client = AGIBotClient(debug_mode=False)
                
                # 模拟快速响应
                def fast_response(*args, **kwargs):
                    return {
                        "choices": [{
                            "message": {
                                "content": f"Client {client_id} task completed",
                                "tool_calls": [],
                                "role": "assistant"
                            },
                            "finish_reason": "stop"
                        }]
                    }
                
                with patch('tool_executor.ToolExecutor._call_llm_api', side_effect=fast_response):
                    start_time = time.time()
                    
                    for i in range(num_requests_per_client):
                        result = client.chat(
                            messages=[{"role": "user", "content": f"Task {i} for client {client_id}"}],
                            dir=os.path.join(test_workspace, f"client_{client_id}"),
                            loops=3
                        )
                        results.append((client_id, i, result["success"], time.time() - start_time))
                    
            except Exception as e:
                errors.append((client_id, str(e)))
        
        # 启动并发客户端
        threads = []
        overall_start = time.time()
        
        for client_id in range(num_clients):
            thread = threading.Thread(target=client_worker, args=(client_id,))
            threads.append(thread)
            thread.start()
        
        # 等待所有线程完成
        for thread in threads:
            thread.join(timeout=60)  # 60秒超时
        
        overall_end = time.time()
        
        # 验证结果
        assert len(errors) == 0, f"Concurrent stress test had errors: {errors}"
        assert len(results) == num_clients * num_requests_per_client
        
        # 性能指标
        total_time = overall_end - overall_start
        throughput = len(results) / total_time
        
        print(f"Concurrent stress test completed:")
        print(f"  Total requests: {len(results)}")
        print(f"  Total time: {total_time:.2f}s")
        print(f"  Throughput: {throughput:.2f} requests/second")
        
        # 验证性能要求
        assert throughput > 1.0, f"Throughput too low: {throughput} requests/second"
        assert total_time < 120, f"Total time too long: {total_time} seconds"

    def test_memory_stress_large_operations(self, test_workspace):
        """测试大操作的内存压力"""
        monitor = PerformanceMonitor()
        
        client = AGIBotClient(debug_mode=False)
        
        # 模拟大内容操作
        large_content = "x" * (1024 * 1024)  # 1MB内容
        
        def memory_intensive_response(*args, **kwargs):
            return {
                "choices": [{
                    "message": {
                        "content": "Processing large content operation...",
                        "tool_calls": [{
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "edit_file",
                                "arguments": f'{{"target_file": "large_file.txt", "code_edit": "{large_content}"}}'
                            }
                        }],
                        "role": "assistant"
                    },
                    "finish_reason": "tool_calls"
                }]
            }
        
        with monitor:
            with patch('tool_executor.ToolExecutor._call_llm_api', side_effect=memory_intensive_response):
                with patch.object(client, '_execute_tool_call') as mock_tool:
                    mock_tool.return_value = {"status": "success", "message": "Large file processed"}
                    
                    # 执行多个大操作
                    for i in range(10):
                        result = client.chat(
                            messages=[{"role": "user", "content": f"Process large operation {i}"}],
                            dir=test_workspace,
                            loops=3
                        )
                        assert result["success"] == True
                        
                        # 手动垃圾回收
                        gc.collect()
        
        # 验证内存使用
        max_memory = monitor.get_max_memory_usage()
        memory_growth = monitor.get_memory_growth()
        
        print(f"Memory stress test results:")
        print(f"  Max memory usage: {max_memory / 1024 / 1024:.2f} MB")
        print(f"  Memory growth: {memory_growth / 1024 / 1024:.2f} MB")
        
        # 验证内存没有无限增长
        assert memory_growth < 100 * 1024 * 1024, f"Memory growth too high: {memory_growth} bytes"

    def test_long_running_stability(self, test_workspace):
        """测试长时间运行的稳定性"""
        client = AGIBotClient(debug_mode=False)
        
        # 运行时间（秒）
        run_duration = 30
        request_interval = 1  # 每秒一个请求
        
        start_time = time.time()
        request_count = 0
        success_count = 0
        error_count = 0
        
        def stable_response(*args, **kwargs):
            return {
                "choices": [{
                    "message": {
                        "content": "Stable operation completed",
                        "tool_calls": [],
                        "role": "assistant"
                    },
                    "finish_reason": "stop"
                }]
            }
        
        with patch('tool_executor.ToolExecutor._call_llm_api', side_effect=stable_response):
            while time.time() - start_time < run_duration:
                try:
                    result = client.chat(
                        messages=[{"role": "user", "content": f"Stability test request {request_count}"}],
                        dir=test_workspace,
                        loops=2
                    )
                    
                    request_count += 1
                    if result["success"]:
                        success_count += 1
                    else:
                        error_count += 1
                        
                    time.sleep(request_interval)
                    
                except Exception as e:
                    error_count += 1
                    print(f"Request {request_count} failed: {e}")
        
        # 验证稳定性
        success_rate = success_count / request_count if request_count > 0 else 0
        
        print(f"Long-running stability test results:")
        print(f"  Total requests: {request_count}")
        print(f"  Successful requests: {success_count}")
        print(f"  Error requests: {error_count}")
        print(f"  Success rate: {success_rate:.2%}")
        
        assert success_rate >= 0.95, f"Success rate too low: {success_rate:.2%}"
        assert request_count >= run_duration * 0.8, f"Too few requests processed: {request_count}"

    def test_resource_usage_monitoring(self, test_workspace):
        """测试资源使用监控"""
        tracker = ResourceTracker()
        
        client = AGIBotClient(debug_mode=False)
        
        def resource_intensive_response(*args, **kwargs):
            # 模拟CPU密集型操作
            time.sleep(0.1)
            return {
                "choices": [{
                    "message": {
                        "content": "Resource intensive operation completed",
                        "tool_calls": [],
                        "role": "assistant"
                    },
                    "finish_reason": "stop"
                }]
            }
        
        with tracker:
            with patch('tool_executor.ToolExecutor._call_llm_api', side_effect=resource_intensive_response):
                # 执行多个操作
                for i in range(20):
                    result = client.chat(
                        messages=[{"role": "user", "content": f"Resource test {i}"}],
                        dir=test_workspace,
                        loops=2
                    )
                    assert result["success"] == True
        
        # 获取资源使用统计
        stats = tracker.get_statistics()
        
        print("Resource usage statistics:")
        print(f"  Peak CPU usage: {stats['peak_cpu_percent']:.1f}%")
        print(f"  Peak memory usage: {stats['peak_memory_mb']:.1f} MB")
        print(f"  Average CPU usage: {stats['avg_cpu_percent']:.1f}%")
        print(f"  Average memory usage: {stats['avg_memory_mb']:.1f} MB")
        
        # 验证资源使用在合理范围内
        assert stats['peak_cpu_percent'] < 90, f"CPU usage too high: {stats['peak_cpu_percent']}%"
        assert stats['peak_memory_mb'] < 1000, f"Memory usage too high: {stats['peak_memory_mb']} MB"

    def test_high_frequency_requests(self, test_workspace):
        """测试高频请求处理"""
        client = AGIBotClient(debug_mode=False)
        
        num_requests = 100
        max_time = 10  # 10秒内完成100个请求
        
        def fast_response(*args, **kwargs):
            return {
                "choices": [{
                    "message": {
                        "content": "Fast response",
                        "tool_calls": [],
                        "role": "assistant"
                    },
                    "finish_reason": "stop"
                }]
            }
        
        start_time = time.time()
        completed_requests = 0
        
        with patch('tool_executor.ToolExecutor._call_llm_api', side_effect=fast_response):
            for i in range(num_requests):
                try:
                    result = client.chat(
                        messages=[{"role": "user", "content": f"Fast request {i}"}],
                        dir=test_workspace,
                        loops=1
                    )
                    
                    if result["success"]:
                        completed_requests += 1
                        
                    # 检查是否超时
                    if time.time() - start_time > max_time:
                        break
                        
                except Exception as e:
                    print(f"Request {i} failed: {e}")
        
        total_time = time.time() - start_time
        requests_per_second = completed_requests / total_time
        
        print(f"High frequency test results:")
        print(f"  Completed requests: {completed_requests}/{num_requests}")
        print(f"  Total time: {total_time:.2f}s")
        print(f"  Requests per second: {requests_per_second:.2f}")
        
        # 验证高频处理能力
        assert completed_requests >= num_requests * 0.9, f"Too few requests completed: {completed_requests}"
        assert requests_per_second >= 8, f"Request rate too low: {requests_per_second} req/s"

    def test_scalability_with_increasing_load(self, test_workspace):
        """测试随负载增加的扩展性"""
        client = AGIBotClient(debug_mode=False)
        
        load_levels = [1, 5, 10, 20]  # 不同的并发级别
        results = {}
        
        def scalable_response(*args, **kwargs):
            return {
                "choices": [{
                    "message": {
                        "content": "Scalable operation completed",
                        "tool_calls": [],
                        "role": "assistant"
                    },
                    "finish_reason": "stop"
                }]
            }
        
        with patch('tool_executor.ToolExecutor._call_llm_api', side_effect=scalable_response):
            for load_level in load_levels:
                print(f"Testing load level: {load_level} concurrent requests")
                
                completed = []
                errors = []
                
                def worker(worker_id):
                    try:
                        start = time.time()
                        result = client.chat(
                            messages=[{"role": "user", "content": f"Load test {worker_id}"}],
                            dir=os.path.join(test_workspace, f"load_{load_level}"),
                            loops=2
                        )
                        end = time.time()
                        completed.append((worker_id, result["success"], end - start))
                    except Exception as e:
                        errors.append((worker_id, str(e)))
                
                # 启动并发请求
                threads = []
                start_time = time.time()
                
                for i in range(load_level):
                    thread = threading.Thread(target=worker, args=(i,))
                    threads.append(thread)
                    thread.start()
                
                # 等待完成
                for thread in threads:
                    thread.join(timeout=30)
                
                total_time = time.time() - start_time
                success_count = sum(1 for _, success, _ in completed if success)
                avg_response_time = sum(duration for _, _, duration in completed) / len(completed) if completed else 0
                
                results[load_level] = {
                    'total_time': total_time,
                    'success_count': success_count,
                    'error_count': len(errors),
                    'avg_response_time': avg_response_time,
                    'throughput': success_count / total_time if total_time > 0 else 0
                }
                
                print(f"  Success: {success_count}/{load_level}")
                print(f"  Avg response time: {avg_response_time:.3f}s")
                print(f"  Throughput: {results[load_level]['throughput']:.2f} req/s")
        
        # 验证扩展性
        print("\nScalability analysis:")
        for load_level in load_levels:
            stats = results[load_level]
            print(f"Load {load_level}: {stats['success_count']} success, "
                  f"{stats['avg_response_time']:.3f}s avg, "
                  f"{stats['throughput']:.2f} req/s")
        
        # 验证系统能处理增加的负载
        for load_level in load_levels:
            success_rate = results[load_level]['success_count'] / load_level
            assert success_rate >= 0.8, f"Success rate too low at load {load_level}: {success_rate:.2%}"
            assert results[load_level]['avg_response_time'] < 5.0, f"Response time too high at load {load_level}"

    def test_performance_regression_detection(self, test_workspace):
        """测试性能回归检测"""
        client = AGIBotClient(debug_mode=False)
        
        # 基准性能测试
        baseline_iterations = 10
        regression_iterations = 10
        
        def measure_performance(iterations, label):
            times = []
            
            def benchmark_response(*args, **kwargs):
                return {
                    "choices": [{
                        "message": {
                            "content": "Benchmark operation completed",
                            "tool_calls": [],
                            "role": "assistant"
                        },
                        "finish_reason": "stop"
                    }]
                }
            
            with patch('tool_executor.ToolExecutor._call_llm_api', side_effect=benchmark_response):
                for i in range(iterations):
                    start_time = time.time()
                    
                    result = client.chat(
                        messages=[{"role": "user", "content": f"{label} benchmark {i}"}],
                        dir=test_workspace,
                        loops=3
                    )
                    
                    end_time = time.time()
                    
                    assert result["success"] == True
                    times.append(end_time - start_time)
            
            return {
                'avg_time': sum(times) / len(times),
                'min_time': min(times),
                'max_time': max(times),
                'times': times
            }
        
        # 测量基准性能
        baseline_perf = measure_performance(baseline_iterations, "Baseline")
        
        # 模拟一些操作后再次测量
        time.sleep(1)  # 模拟系统状态变化
        
        # 测量当前性能
        current_perf = measure_performance(regression_iterations, "Current")
        
        # 性能比较
        avg_degradation = (current_perf['avg_time'] - baseline_perf['avg_time']) / baseline_perf['avg_time']
        max_degradation = (current_perf['max_time'] - baseline_perf['max_time']) / baseline_perf['max_time']
        
        print(f"Performance regression test results:")
        print(f"  Baseline avg time: {baseline_perf['avg_time']:.3f}s")
        print(f"  Current avg time: {current_perf['avg_time']:.3f}s")
        print(f"  Average degradation: {avg_degradation:.1%}")
        print(f"  Max degradation: {max_degradation:.1%}")
        
        # 验证没有显著性能回归
        assert avg_degradation < 0.2, f"Average performance degraded by {avg_degradation:.1%}"
        assert max_degradation < 0.5, f"Max performance degraded by {max_degradation:.1%}" 