#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AGIBot负载和性能测试
测试系统在高负载下的表现和性能指标
"""

import pytest
import time
import threading
import queue
import psutil
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from unittest.mock import patch
import sys

# 添加src路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from main import AGIBotClient
from utils.test_helpers import TestHelper
from utils.performance_monitor import PerformanceMonitor, ResourceTracker

@pytest.mark.performance
@pytest.mark.slow
class TestLoadTesting:
    """负载测试类"""
    
    @pytest.fixture
    def performance_monitor(self):
        """性能监控器"""
        return PerformanceMonitor()
    
    @pytest.fixture
    def resource_tracker(self):
        """资源跟踪器"""
        return ResourceTracker()
    
    def test_concurrent_task_execution(self, test_workspace, performance_monitor):
        """测试并发任务执行性能"""
        num_concurrent_tasks = 10
        task_duration_limit = 30  # 秒
        
        results = queue.Queue()
        errors = queue.Queue()
        
        def execute_concurrent_task(task_id):
            """执行并发任务"""
            try:
                client = AGIBotClient(debug_mode=False)
                
                # 模拟快速响应
                def mock_quick_response(*args, **kwargs):
                    time.sleep(0.1)  # 模拟处理时间
                    return {
                        "choices": [{
                            "message": {
                                "content": f"Task {task_id} completed successfully",
                                "role": "assistant",
                                "tool_calls": []
                            },
                            "finish_reason": "stop"
                        }]
                    }
                
                with patch('tool_executor.ToolExecutor._call_llm_api', side_effect=mock_quick_response):
                    start_time = time.time()
                    
                    result = client.chat(
                        messages=[{"role": "user", "content": f"Simple task {task_id}"}],
                        dir=os.path.join(test_workspace, f"task_{task_id}"),
                        loops=3
                    )
                    
                    end_time = time.time()
                    execution_time = end_time - start_time
                    
                    results.put({
                        "task_id": task_id,
                        "success": result["success"],
                        "execution_time": execution_time,
                        "result": result
                    })
                    
            except Exception as e:
                errors.put({"task_id": task_id, "error": str(e)})
        
        # 启动性能监控
        with performance_monitor.monitor_execution():
            # 使用线程池执行并发任务
            with ThreadPoolExecutor(max_workers=num_concurrent_tasks) as executor:
                start_time = time.time()
                
                # 提交所有任务
                futures = [
                    executor.submit(execute_concurrent_task, i) 
                    for i in range(num_concurrent_tasks)
                ]
                
                # 等待所有任务完成
                for future in as_completed(futures, timeout=task_duration_limit):
                    try:
                        future.result()
                    except Exception as e:
                        errors.put({"error": str(e)})
                
                total_time = time.time() - start_time
        
        # 收集结果
        task_results = []
        task_errors = []
        
        while not results.empty():
            task_results.append(results.get())
        
        while not errors.empty():
            task_errors.append(errors.get())
        
        # 验证性能要求
        assert len(task_errors) == 0, f"Concurrent tasks had errors: {task_errors}"
        assert len(task_results) == num_concurrent_tasks, f"Expected {num_concurrent_tasks} results"
        assert total_time < task_duration_limit, f"Concurrent execution exceeded time limit: {total_time}s"
        
        # 验证所有任务成功
        successful_tasks = [r for r in task_results if r["success"]]
        assert len(successful_tasks) == num_concurrent_tasks, "Not all tasks completed successfully"
        
        # 分析性能指标
        execution_times = [r["execution_time"] for r in task_results]
        avg_execution_time = sum(execution_times) / len(execution_times)
        max_execution_time = max(execution_times)
        
        performance_metrics = performance_monitor.get_metrics()
        
        print(f"\n=== Concurrent Task Performance ===")
        print(f"Tasks: {num_concurrent_tasks}")
        print(f"Total time: {total_time:.2f}s")
        print(f"Average task time: {avg_execution_time:.2f}s")
        print(f"Max task time: {max_execution_time:.2f}s")
        print(f"Tasks per second: {num_concurrent_tasks / total_time:.2f}")
        
        # 性能断言
        assert avg_execution_time < 10, f"Average execution time too high: {avg_execution_time}s"
        assert max_execution_time < 15, f"Max execution time too high: {max_execution_time}s"

    def test_memory_usage_under_load(self, test_workspace, resource_tracker):
        """测试负载下的内存使用"""
        initial_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        # 执行内存密集型任务
        def memory_intensive_task():
            client = AGIBotClient(debug_mode=False)
            
            # 模拟生成大量内容的响应
            def mock_large_response(*args, **kwargs):
                large_content = "x" * (1024 * 100)  # 100KB content
                return {
                    "choices": [{
                        "message": {
                            "content": f"Generated large content: {large_content}",
                            "role": "assistant",
                            "tool_calls": []
                        },
                        "finish_reason": "stop"
                    }]
                }
            
            with patch('tool_executor.ToolExecutor._call_llm_api', side_effect=mock_large_response):
                return client.chat(
                    messages=[{"role": "user", "content": "Generate large content"}],
                    dir=test_workspace,
                    loops=2
                )
        
        # 监控内存使用
        with resource_tracker.track_resources():
            # 执行多个内存密集型任务
            for i in range(5):
                result = memory_intensive_task()
                assert result["success"] == True
        
        final_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # 获取资源使用报告
        resource_report = resource_tracker.get_report()
        
        print(f"\n=== Memory Usage Analysis ===")
        print(f"Initial memory: {initial_memory:.1f} MB")
        print(f"Final memory: {final_memory:.1f} MB")
        print(f"Memory increase: {memory_increase:.1f} MB")
        print(f"Peak memory: {resource_report['peak_memory_mb']:.1f} MB")
        
        # 内存使用断言
        assert memory_increase < 100, f"Memory increase too high: {memory_increase:.1f} MB"
        assert resource_report['peak_memory_mb'] < initial_memory + 200, "Peak memory usage too high"

    def test_throughput_performance(self, test_workspace):
        """测试吞吐量性能"""
        num_tasks = 20
        time_limit = 60  # 60秒内完成
        
        def quick_task(task_id):
            """快速任务执行"""
            client = AGIBotClient(debug_mode=False)
            
            def mock_instant_response(*args, **kwargs):
                return {
                    "choices": [{
                        "message": {
                            "content": f"Quick task {task_id} done",
                            "role": "assistant",
                            "tool_calls": []
                        },
                        "finish_reason": "stop"
                    }]
                }
            
            with patch('tool_executor.ToolExecutor._call_llm_api', side_effect=mock_instant_response):
                start = time.time()
                result = client.chat(
                    messages=[{"role": "user", "content": f"Quick task {task_id}"}],
                    dir=os.path.join(test_workspace, f"quick_{task_id}"),
                    loops=1
                )
                end = time.time()
                return {"success": result["success"], "time": end - start}
        
        # 串行执行测试
        start_time = time.time()
        
        serial_results = []
        for i in range(num_tasks):
            result = quick_task(i)
            serial_results.append(result)
        
        serial_time = time.time() - start_time
        
        # 并行执行测试
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=5) as executor:
            parallel_futures = [executor.submit(quick_task, i) for i in range(num_tasks)]
            parallel_results = [future.result() for future in as_completed(parallel_futures)]
        
        parallel_time = time.time() - start_time
        
        # 分析吞吐量
        serial_throughput = num_tasks / serial_time
        parallel_throughput = num_tasks / parallel_time
        speedup = serial_time / parallel_time
        
        print(f"\n=== Throughput Performance ===")
        print(f"Tasks: {num_tasks}")
        print(f"Serial time: {serial_time:.2f}s (throughput: {serial_throughput:.2f} tasks/s)")
        print(f"Parallel time: {parallel_time:.2f}s (throughput: {parallel_throughput:.2f} tasks/s)")
        print(f"Speedup: {speedup:.2f}x")
        
        # 性能断言
        assert parallel_time < time_limit, f"Parallel execution exceeded time limit: {parallel_time}s"
        assert speedup > 1.5, f"Insufficient parallelization speedup: {speedup}x"
        assert all(r["success"] for r in serial_results), "Serial execution had failures"
        assert all(r["success"] for r in parallel_results), "Parallel execution had failures"

    def test_cpu_usage_optimization(self, test_workspace):
        """测试CPU使用优化"""
        def cpu_intensive_task():
            """CPU密集型任务"""
            client = AGIBotClient(debug_mode=False)
            
            def mock_cpu_heavy_response(*args, **kwargs):
                # 模拟CPU密集型处理
                result = sum(i * i for i in range(10000))  # 简单的CPU工作
                return {
                    "choices": [{
                        "message": {
                            "content": f"CPU intensive calculation result: {result}",
                            "role": "assistant",
                            "tool_calls": []
                        },
                        "finish_reason": "stop"
                    }]
                }
            
            with patch('tool_executor.ToolExecutor._call_llm_api', side_effect=mock_cpu_heavy_response):
                return client.chat(
                    messages=[{"role": "user", "content": "Perform CPU intensive task"}],
                    dir=test_workspace,
                    loops=2
                )
        
        # 监控CPU使用
        process = psutil.Process()
        cpu_samples = []
        
        def monitor_cpu():
            for _ in range(10):
                cpu_samples.append(process.cpu_percent(interval=0.1))
        
        # 启动CPU监控
        monitor_thread = threading.Thread(target=monitor_cpu)
        monitor_thread.start()
        
        # 执行CPU密集型任务
        start_time = time.time()
        result = cpu_intensive_task()
        execution_time = time.time() - start_time
        
        monitor_thread.join()
        
        # 分析CPU使用
        avg_cpu = sum(cpu_samples) / len(cpu_samples) if cpu_samples else 0
        max_cpu = max(cpu_samples) if cpu_samples else 0
        
        print(f"\n=== CPU Usage Analysis ===")
        print(f"Execution time: {execution_time:.2f}s")
        print(f"Average CPU: {avg_cpu:.1f}%")
        print(f"Peak CPU: {max_cpu:.1f}%")
        
        # CPU使用断言
        assert result["success"] == True, "CPU intensive task failed"
        assert max_cpu < 90, f"CPU usage too high: {max_cpu}%"
        assert execution_time < 10, f"CPU task took too long: {execution_time}s"

    def test_stress_testing(self, test_workspace):
        """压力测试：极限负载"""
        stress_duration = 30  # 30秒压力测试
        max_concurrent = 15
        
        def stress_task(task_id):
            """压力测试任务"""
            try:
                client = AGIBotClient(debug_mode=False)
                
                def mock_variable_response(*args, **kwargs):
                    # 随机延迟模拟真实环境
                    import random
                    time.sleep(random.uniform(0.05, 0.2))
                    return {
                        "choices": [{
                            "message": {
                                "content": f"Stress task {task_id} completed",
                                "role": "assistant",
                                "tool_calls": []
                            },
                            "finish_reason": "stop"
                        }]
                    }
                
                with patch('tool_executor.ToolExecutor._call_llm_api', side_effect=mock_variable_response):
                    return client.chat(
                        messages=[{"role": "user", "content": f"Stress test task {task_id}"}],
                        dir=os.path.join(test_workspace, f"stress_{task_id}"),
                        loops=2
                    )
            except Exception as e:
                return {"success": False, "error": str(e)}
        
        # 收集压力测试结果
        results = []
        errors = []
        
        start_time = time.time()
        task_counter = 0
        
        # 持续压力测试
        with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
            futures = []
            
            while time.time() - start_time < stress_duration:
                if len(futures) < max_concurrent:
                    future = executor.submit(stress_task, task_counter)
                    futures.append((future, task_counter))
                    task_counter += 1
                
                # 检查完成的任务
                completed_futures = []
                for future, tid in futures:
                    if future.done():
                        try:
                            result = future.result()
                            if result and result.get("success"):
                                results.append(result)
                            else:
                                errors.append({"task_id": tid, "error": result.get("error", "Unknown")})
                        except Exception as e:
                            errors.append({"task_id": tid, "error": str(e)})
                        completed_futures.append((future, tid))
                
                # 移除完成的任务
                for completed in completed_futures:
                    futures.remove(completed)
                
                time.sleep(0.1)  # 短暂休息
        
        total_stress_time = time.time() - start_time
        
        # 等待剩余任务完成
        for future, tid in futures:
            try:
                result = future.result(timeout=5)
                if result and result.get("success"):
                    results.append(result)
                else:
                    errors.append({"task_id": tid, "error": result.get("error", "Unknown")})
            except Exception as e:
                errors.append({"task_id": tid, "error": str(e)})
        
        # 分析压力测试结果
        total_tasks = len(results) + len(errors)
        success_rate = len(results) / total_tasks if total_tasks > 0 else 0
        throughput = total_tasks / total_stress_time
        
        print(f"\n=== Stress Test Results ===")
        print(f"Duration: {total_stress_time:.1f}s")
        print(f"Total tasks: {total_tasks}")
        print(f"Successful: {len(results)}")
        print(f"Failed: {len(errors)}")
        print(f"Success rate: {success_rate:.1%}")
        print(f"Throughput: {throughput:.2f} tasks/s")
        
        if errors:
            print(f"Sample errors: {errors[:3]}")
        
        # 压力测试断言
        assert success_rate >= 0.90, f"Success rate too low: {success_rate:.1%}"
        assert throughput >= 5, f"Throughput too low: {throughput:.2f} tasks/s"
        assert total_tasks >= 50, f"Not enough tasks processed: {total_tasks}"

    def test_resource_cleanup(self, test_workspace):
        """测试资源清理效果"""
        initial_memory = psutil.Process().memory_info().rss / 1024 / 1024
        initial_threads = threading.active_count()
        
        def resource_heavy_task():
            """资源密集型任务"""
            client = AGIBotClient(debug_mode=False)
            
            def mock_resource_response(*args, **kwargs):
                # 模拟创建临时资源
                temp_data = ["x" * 1024 for _ in range(100)]  # 100KB per call
                return {
                    "choices": [{
                        "message": {
                            "content": f"Resource task completed, created {len(temp_data)} items",
                            "role": "assistant",
                            "tool_calls": []
                        },
                        "finish_reason": "stop"
                    }]
                }
            
            with patch('tool_executor.ToolExecutor._call_llm_api', side_effect=mock_resource_response):
                result = client.chat(
                    messages=[{"role": "user", "content": "Resource heavy task"}],
                    dir=test_workspace,
                    loops=3
                )
                return result
        
        # 执行多个资源密集型任务
        for i in range(5):
            result = resource_heavy_task()
            assert result["success"] == True
        
        # 强制垃圾回收
        import gc
        gc.collect()
        
        # 等待资源清理
        time.sleep(2)
        
        final_memory = psutil.Process().memory_info().rss / 1024 / 1024
        final_threads = threading.active_count()
        
        memory_increase = final_memory - initial_memory
        thread_increase = final_threads - initial_threads
        
        print(f"\n=== Resource Cleanup Analysis ===")
        print(f"Memory increase: {memory_increase:.1f} MB")
        print(f"Thread increase: {thread_increase}")
        print(f"Initial threads: {initial_threads}")
        print(f"Final threads: {final_threads}")
        
        # 资源清理断言
        assert memory_increase < 50, f"Memory not properly cleaned up: +{memory_increase:.1f}MB"
        assert thread_increase <= 2, f"Too many threads created: +{thread_increase}"

    @pytest.mark.slow
    def test_endurance_testing(self, test_workspace):
        """耐久性测试：长时间运行"""
        endurance_duration = 120  # 2分钟耐久测试
        check_interval = 10  # 每10秒检查一次
        
        def endurance_task(iteration):
            """耐久性任务"""
            client = AGIBotClient(debug_mode=False)
            
            def mock_steady_response(*args, **kwargs):
                return {
                    "choices": [{
                        "message": {
                            "content": f"Endurance iteration {iteration} completed",
                            "role": "assistant",
                            "tool_calls": []
                        },
                        "finish_reason": "stop"
                    }]
                }
            
            with patch('tool_executor.ToolExecutor._call_llm_api', side_effect=mock_steady_response):
                return client.chat(
                    messages=[{"role": "user", "content": f"Endurance task {iteration}"}],
                    dir=os.path.join(test_workspace, f"endurance_{iteration}"),
                    loops=2
                )
        
        start_time = time.time()
        iteration = 0
        results = []
        errors = []
        memory_samples = []
        
        # 长时间运行测试
        while time.time() - start_time < endurance_duration:
            try:
                # 执行任务
                result = endurance_task(iteration)
                if result["success"]:
                    results.append(iteration)
                else:
                    errors.append({"iteration": iteration, "error": "Task failed"})
                
                # 记录内存使用
                current_memory = psutil.Process().memory_info().rss / 1024 / 1024
                memory_samples.append(current_memory)
                
                iteration += 1
                
                # 短暂休息
                time.sleep(1)
                
            except Exception as e:
                errors.append({"iteration": iteration, "error": str(e)})
                iteration += 1
        
        total_duration = time.time() - start_time
        
        # 分析耐久性结果
        success_rate = len(results) / iteration if iteration > 0 else 0
        avg_memory = sum(memory_samples) / len(memory_samples) if memory_samples else 0
        memory_trend = memory_samples[-1] - memory_samples[0] if len(memory_samples) >= 2 else 0
        
        print(f"\n=== Endurance Test Results ===")
        print(f"Duration: {total_duration:.1f}s")
        print(f"Iterations: {iteration}")
        print(f"Successful: {len(results)}")
        print(f"Failed: {len(errors)}")
        print(f"Success rate: {success_rate:.1%}")
        print(f"Average memory: {avg_memory:.1f} MB")
        print(f"Memory trend: {memory_trend:+.1f} MB")
        
        # 耐久性断言
        assert success_rate >= 0.95, f"Endurance success rate too low: {success_rate:.1%}"
        assert abs(memory_trend) < 20, f"Memory leak detected: {memory_trend:+.1f}MB trend"
        assert iteration >= 60, f"Not enough iterations completed: {iteration}" 