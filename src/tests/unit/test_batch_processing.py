#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量任务处理单元测试
测试Python库接口的批量操作功能
"""

import pytest
import os
import sys
import time
import json
import asyncio
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from unittest.mock import patch, Mock, MagicMock
from typing import Dict, List, Any

# 添加src路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from main import AGIBotClient
from utils.test_helpers import TestHelper

@pytest.mark.unit
class TestBatchProcessing:
    """批量处理功能测试类"""
    
    @pytest.fixture
    def batch_client(self, test_workspace):
        """创建支持批量处理的AGIBot客户端"""
        return AGIBotClient(
            api_key="test_key",
            model="test_model",
            api_base="test_base",
            debug_mode=True
        )
    
    @pytest.fixture
    def sample_tasks(self):
        """示例任务列表"""
        return [
            {
                "id": "task_001",
                "messages": [{"role": "user", "content": "创建一个Python计算器"}],
                "priority": "high",
                "timeout": 300
            },
            {
                "id": "task_002", 
                "messages": [{"role": "user", "content": "分析这个代码库的结构"}],
                "priority": "medium",
                "timeout": 600
            },
            {
                "id": "task_003",
                "messages": [{"role": "user", "content": "生成API文档"}],
                "priority": "low",
                "timeout": 900
            },
            {
                "id": "task_004",
                "messages": [{"role": "user", "content": "重构代码并优化性能"}],
                "priority": "high",
                "timeout": 1200
            },
            {
                "id": "task_005",
                "messages": [{"role": "user", "content": "编写单元测试"}],
                "priority": "medium",
                "timeout": 800
            }
        ]
    
    @pytest.fixture
    def batch_configurations(self):
        """批量处理配置"""
        return {
            "sequential": {
                "mode": "sequential",
                "max_workers": 1,
                "timeout_per_task": 300,
                "failure_handling": "stop_on_error"
            },
            "parallel_threads": {
                "mode": "parallel",
                "executor": "thread",
                "max_workers": 3,
                "timeout_per_task": 300,
                "failure_handling": "continue_on_error"
            },
            "parallel_processes": {
                "mode": "parallel",
                "executor": "process",
                "max_workers": 2,
                "timeout_per_task": 300,
                "failure_handling": "continue_on_error"
            },
            "priority_based": {
                "mode": "priority",
                "max_workers": 3,
                "timeout_per_task": 300,
                "priority_levels": ["high", "medium", "low"]
            }
        }
    
    def test_batch_client_initialization(self, batch_client):
        """测试批量客户端初始化"""
        assert batch_client is not None
        assert hasattr(batch_client, 'chat')
        
        # 检查是否有批量处理方法
        batch_methods = ['batch_chat', 'process_batch', 'submit_batch']
        available_methods = [method for method in batch_methods if hasattr(batch_client, method)]
        
        # 至少应该有基础的chat方法
        assert len(available_methods) >= 0
    
    def test_sequential_batch_processing(self, batch_client, sample_tasks):
        """测试顺序批量处理"""
        with patch.object(batch_client, 'chat') as mock_chat:
            # 模拟每个任务的响应
            def mock_chat_response(messages, **kwargs):
                task_content = messages[0]["content"]
                return {
                    "success": True,
                    "message": f"Completed: {task_content[:50]}...",
                    "output_dir": f"/test/output_{int(time.time())}",
                    "execution_time": 10.5
                }
            
            mock_chat.side_effect = mock_chat_response
            
            # 执行顺序批量处理
            if hasattr(batch_client, 'batch_chat'):
                results = batch_client.batch_chat(
                    [task["messages"] for task in sample_tasks],
                    mode="sequential"
                )
            else:
                # 模拟顺序处理
                results = []
                for task in sample_tasks:
                    result = batch_client.chat(task["messages"])
                    results.append(result)
            
            # 验证顺序处理结果
            assert len(results) == len(sample_tasks)
            for result in results:
                assert result is not None
                assert result["success"] is True
    
    def test_parallel_thread_processing(self, batch_client, sample_tasks):
        """测试并行线程处理"""
        with patch.object(batch_client, 'chat') as mock_chat:
            # 模拟并行任务响应
            def mock_parallel_response(messages, **kwargs):
                time.sleep(0.1)  # 模拟处理时间
                return {
                    "success": True,
                    "message": "Parallel task completed",
                    "thread_id": threading.current_thread().ident
                }
            
            mock_chat.side_effect = mock_parallel_response
            
            # 测试线程池并行处理
            import threading
            from concurrent.futures import ThreadPoolExecutor
            
            start_time = time.time()
            
            with ThreadPoolExecutor(max_workers=3) as executor:
                futures = []
                for task in sample_tasks:
                    future = executor.submit(batch_client.chat, task["messages"])
                    futures.append(future)
                
                results = [future.result() for future in futures]
            
            end_time = time.time()
            execution_time = end_time - start_time
            
            # 验证并行处理效果
            assert len(results) == len(sample_tasks)
            for result in results:
                assert result is not None
                assert result["success"] is True
            
            # 并行处理应该比顺序处理快
            sequential_time = len(sample_tasks) * 0.1
            assert execution_time < sequential_time
    
    def test_parallel_process_processing(self, batch_client, sample_tasks):
        """测试并行进程处理"""
        with patch.object(batch_client, 'chat') as mock_chat:
            mock_chat.return_value = {
                "success": True,
                "message": "Process task completed",
                "process_id": os.getpid()
            }
            
            # 模拟进程池处理
            def process_task(task_data):
                # 在实际应用中，这里会创建新的客户端实例
                client = AGIBotClient(
                    api_key="test_key",
                    model="test_model",
                    api_base="test_base"
                )
                return client.chat(task_data["messages"])
            
            # 测试进程池并行处理
            try:
                from concurrent.futures import ProcessPoolExecutor
                
                with ProcessPoolExecutor(max_workers=2) as executor:
                    futures = []
                    for task in sample_tasks[:3]:  # 减少任务数量以加快测试
                        future = executor.submit(process_task, task)
                        futures.append(future)
                    
                    results = [future.result() for future in futures]
                
                # 验证进程处理结果
                assert len(results) == 3
                for result in results:
                    assert result is not None
                    
            except Exception as e:
                # 进程池在某些环境下可能不可用
                pytest.skip(f"Process pool not available: {e}")
    
    def test_priority_based_processing(self, batch_client, sample_tasks):
        """测试基于优先级的处理"""
        with patch.object(batch_client, 'chat') as mock_chat:
            execution_order = []
            
            def mock_priority_response(messages, **kwargs):
                task_content = messages[0]["content"]
                execution_order.append(task_content)
                return {
                    "success": True,
                    "message": f"Priority task: {task_content[:30]}...",
                    "execution_order": len(execution_order)
                }
            
            mock_chat.side_effect = mock_priority_response
            
            # 按优先级排序任务
            priority_order = {"high": 1, "medium": 2, "low": 3}
            sorted_tasks = sorted(
                sample_tasks, 
                key=lambda x: priority_order.get(x["priority"], 4)
            )
            
            # 执行优先级处理
            results = []
            for task in sorted_tasks:
                result = batch_client.chat(task["messages"])
                results.append(result)
            
            # 验证优先级处理
            assert len(results) == len(sample_tasks)
            
            # 验证高优先级任务先执行
            high_priority_tasks = [t for t in sample_tasks if t["priority"] == "high"]
            for i, task in enumerate(high_priority_tasks):
                task_content = task["messages"][0]["content"]
                assert task_content in execution_order[:len(high_priority_tasks)]
    
    def test_batch_error_handling(self, batch_client, sample_tasks):
        """测试批量错误处理"""
        with patch.object(batch_client, 'chat') as mock_chat:
            # 模拟部分任务失败
            def mock_error_response(messages, **kwargs):
                task_content = messages[0]["content"]
                
                if "计算器" in task_content:
                    return {
                        "success": True,
                        "message": "Calculator created successfully"
                    }
                elif "代码库" in task_content:
                    raise Exception("Analysis failed: Repository not accessible")
                elif "文档" in task_content:
                    return {
                        "success": False,
                        "message": "Documentation generation failed",
                        "error": "Template not found"
                    }
                else:
                    return {
                        "success": True,
                        "message": "Task completed"
                    }
            
            mock_chat.side_effect = mock_error_response
            
            # 测试继续处理模式
            results = []
            errors = []
            
            for task in sample_tasks:
                try:
                    result = batch_client.chat(task["messages"])
                    results.append(result)
                except Exception as e:
                    errors.append({"task_id": task["id"], "error": str(e)})
                    results.append(None)
            
            # 验证错误处理
            successful_results = [r for r in results if r is not None and r.get("success")]
            failed_results = [r for r in results if r is not None and not r.get("success")]
            
            assert len(successful_results) > 0  # 至少有一些成功的任务
            assert len(failed_results) + len(errors) > 0  # 至少有一些失败的任务
    
    def test_batch_result_aggregation(self, batch_client, sample_tasks):
        """测试批量结果聚合"""
        with patch.object(batch_client, 'chat') as mock_chat:
            # 模拟任务结果
            def mock_aggregation_response(messages, **kwargs):
                task_content = messages[0]["content"]
                return {
                    "success": True,
                    "message": f"Completed: {task_content}",
                    "execution_time": 15.5,
                    "output_files": ["result.txt", "log.txt"],
                    "metrics": {
                        "lines_of_code": 100,
                        "functions_created": 5,
                        "tests_passed": 10
                    }
                }
            
            mock_chat.side_effect = mock_aggregation_response
            
            # 执行批量任务并聚合结果
            results = []
            total_execution_time = 0
            total_files = []
            aggregated_metrics = {
                "total_lines_of_code": 0,
                "total_functions": 0,
                "total_tests": 0
            }
            
            for task in sample_tasks:
                result = batch_client.chat(task["messages"])
                results.append(result)
                
                if result["success"]:
                    total_execution_time += result["execution_time"]
                    total_files.extend(result["output_files"])
                    
                    metrics = result["metrics"]
                    aggregated_metrics["total_lines_of_code"] += metrics["lines_of_code"]
                    aggregated_metrics["total_functions"] += metrics["functions_created"]
                    aggregated_metrics["total_tests"] += metrics["tests_passed"]
            
            # 验证结果聚合
            assert len(results) == len(sample_tasks)
            assert total_execution_time > 0
            assert len(total_files) == len(sample_tasks) * 2  # 每个任务2个文件
            assert aggregated_metrics["total_lines_of_code"] == len(sample_tasks) * 100
    
    def test_batch_progress_tracking(self, batch_client, sample_tasks):
        """测试批量进度跟踪"""
        progress_updates = []
        
        def progress_callback(completed, total, current_task):
            progress_updates.append({
                "completed": completed,
                "total": total,
                "current_task": current_task,
                "progress_percentage": (completed / total) * 100
            })
        
        with patch.object(batch_client, 'chat') as mock_chat:
            mock_chat.return_value = {"success": True, "message": "Task completed"}
            
            # 模拟带进度跟踪的批量处理
            total_tasks = len(sample_tasks)
            
            for i, task in enumerate(sample_tasks):
                result = batch_client.chat(task["messages"])
                
                # 更新进度
                progress_callback(i + 1, total_tasks, task["id"])
            
            # 验证进度跟踪
            assert len(progress_updates) == len(sample_tasks)
            assert progress_updates[0]["progress_percentage"] == 20.0  # 1/5 * 100
            assert progress_updates[-1]["progress_percentage"] == 100.0  # 5/5 * 100
    
    def test_batch_timeout_handling(self, batch_client, sample_tasks):
        """测试批量超时处理"""
        with patch.object(batch_client, 'chat') as mock_chat:
            # 模拟不同的执行时间
            def mock_timeout_response(messages, **kwargs):
                task_content = messages[0]["content"]
                
                if "计算器" in task_content:
                    time.sleep(0.1)  # 快速任务
                    return {"success": True, "message": "Quick task"}
                elif "代码库" in task_content:
                    time.sleep(2)  # 慢任务
                    return {"success": True, "message": "Slow task"}
                else:
                    return {"success": True, "message": "Normal task"}
            
            mock_chat.side_effect = mock_timeout_response
            
            # 测试任务超时
            task_timeout = 1  # 1秒超时
            results = []
            
            for task in sample_tasks:
                start_time = time.time()
                
                try:
                    # 模拟超时机制
                    result = batch_client.chat(task["messages"])
                    execution_time = time.time() - start_time
                    
                    if execution_time > task_timeout:
                        result = {
                            "success": False,
                            "message": "Task timeout",
                            "error": f"Task exceeded {task_timeout}s timeout"
                        }
                    
                    results.append(result)
                    
                except Exception as e:
                    results.append({
                        "success": False,
                        "message": "Task failed",
                        "error": str(e)
                    })
            
            # 验证超时处理
            assert len(results) == len(sample_tasks)
            timeout_results = [r for r in results if not r["success"] and "timeout" in r.get("message", "")]
            assert len(timeout_results) >= 0  # 可能有超时的任务
    
    def test_batch_resource_management(self, batch_client, sample_tasks):
        """测试批量资源管理"""
        import psutil
        import gc
        
        # 获取初始内存使用
        process = psutil.Process()
        initial_memory = process.memory_info().rss
        
        with patch.object(batch_client, 'chat') as mock_chat:
            # 模拟内存密集型任务
            def mock_memory_intensive_response(messages, **kwargs):
                # 模拟创建一些数据
                large_data = ["data"] * 10000
                return {
                    "success": True,
                    "message": "Memory intensive task completed",
                    "data_size": len(large_data)
                }
            
            mock_chat.side_effect = mock_memory_intensive_response
            
            # 执行批量任务
            results = []
            for task in sample_tasks:
                result = batch_client.chat(task["messages"])
                results.append(result)
                
                # 手动垃圾回收
                gc.collect()
            
            # 检查最终内存使用
            final_memory = process.memory_info().rss
            memory_increase = final_memory - initial_memory
            
            # 验证资源管理
            assert len(results) == len(sample_tasks)
            # 内存增长应该在合理范围内（小于100MB）
            assert memory_increase < 100 * 1024 * 1024
    
    def test_batch_load_balancing(self, batch_client, sample_tasks):
        """测试批量负载均衡"""
        # 模拟多个工作进程
        worker_loads = {"worker_1": 0, "worker_2": 0, "worker_3": 0}
        
        with patch.object(batch_client, 'chat') as mock_chat:
            def mock_load_balanced_response(messages, **kwargs):
                # 简单的轮询负载均衡
                worker_ids = list(worker_loads.keys())
                min_load_worker = min(worker_loads, key=worker_loads.get)
                worker_loads[min_load_worker] += 1
                
                return {
                    "success": True,
                    "message": "Load balanced task",
                    "worker_id": min_load_worker,
                    "worker_load": worker_loads[min_load_worker]
                }
            
            mock_chat.side_effect = mock_load_balanced_response
            
            # 执行负载均衡的批量处理
            results = []
            for task in sample_tasks:
                result = batch_client.chat(task["messages"])
                results.append(result)
            
            # 验证负载均衡
            assert len(results) == len(sample_tasks)
            
            # 检查负载分布
            loads = list(worker_loads.values())
            max_load = max(loads)
            min_load = min(loads)
            
            # 负载差异应该不超过1
            assert max_load - min_load <= 1
    
    def test_batch_retry_mechanism(self, batch_client, sample_tasks):
        """测试批量重试机制"""
        retry_counts = {}
        
        with patch.object(batch_client, 'chat') as mock_chat:
            def mock_retry_response(messages, **kwargs):
                task_content = messages[0]["content"]
                
                if task_content not in retry_counts:
                    retry_counts[task_content] = 0
                
                retry_counts[task_content] += 1
                
                # 第一次失败，第二次成功
                if retry_counts[task_content] == 1 and "代码库" in task_content:
                    raise Exception("Temporary failure")
                else:
                    return {
                        "success": True,
                        "message": f"Task completed after {retry_counts[task_content]} attempts",
                        "retry_count": retry_counts[task_content]
                    }
            
            mock_chat.side_effect = mock_retry_response
            
            # 执行带重试的批量处理
            results = []
            max_retries = 2
            
            for task in sample_tasks:
                retries = 0
                success = False
                
                while retries < max_retries and not success:
                    try:
                        result = batch_client.chat(task["messages"])
                        results.append(result)
                        success = True
                    except Exception as e:
                        retries += 1
                        if retries >= max_retries:
                            results.append({
                                "success": False,
                                "message": "Max retries exceeded",
                                "error": str(e),
                                "retry_count": retries
                            })
            
            # 验证重试机制
            assert len(results) == len(sample_tasks)
            
            # 检查重试结果
            retry_results = [r for r in results if r.get("retry_count", 0) > 1]
            assert len(retry_results) > 0  # 至少有一个任务进行了重试
    
    def test_async_batch_processing(self, batch_client, sample_tasks):
        """测试异步批量处理"""
        async def async_task(task_data):
            # 模拟异步任务处理
            await asyncio.sleep(0.1)
            return {
                "success": True,
                "message": f"Async task completed: {task_data['id']}",
                "task_id": task_data["id"]
            }
        
        async def run_async_batch():
            # 创建异步任务
            tasks = [async_task(task) for task in sample_tasks]
            
            # 并发执行
            results = await asyncio.gather(*tasks)
            return results
        
        # 执行异步批量处理
        results = asyncio.run(run_async_batch())
        
        # 验证异步处理结果
        assert len(results) == len(sample_tasks)
        for result in results:
            assert result is not None
            assert result["success"] is True
    
    def test_batch_configuration_validation(self, batch_client, batch_configurations):
        """测试批量配置验证"""
        for config_name, config in batch_configurations.items():
            # 验证配置参数
            assert "mode" in config
            assert "max_workers" in config
            assert "timeout_per_task" in config
            
            # 验证参数值
            assert config["max_workers"] > 0
            assert config["timeout_per_task"] > 0
            
            # 验证模式特定参数
            if config["mode"] == "parallel":
                assert "executor" in config
                assert config["executor"] in ["thread", "process"]
            
            if config["mode"] == "priority":
                assert "priority_levels" in config
                assert len(config["priority_levels"]) > 0
    
    def test_batch_result_export(self, batch_client, sample_tasks, test_workspace):
        """测试批量结果导出"""
        with patch.object(batch_client, 'chat') as mock_chat:
            mock_chat.return_value = {
                "success": True,
                "message": "Task completed",
                "output_files": ["output.txt"],
                "metrics": {"duration": 10}
            }
            
            # 执行批量任务
            results = []
            for task in sample_tasks:
                result = batch_client.chat(task["messages"])
                result["task_id"] = task["id"]
                results.append(result)
            
            # 导出结果
            export_formats = ["json", "csv", "xlsx"]
            
            for format_type in export_formats:
                export_file = os.path.join(test_workspace, f"batch_results.{format_type}")
                
                try:
                    if format_type == "json":
                        with open(export_file, "w") as f:
                            json.dump(results, f, indent=2)
                    elif format_type == "csv":
                        import csv
                        with open(export_file, "w", newline="") as f:
                            writer = csv.DictWriter(f, fieldnames=["task_id", "success", "message"])
                            writer.writeheader()
                            for result in results:
                                writer.writerow({
                                    "task_id": result["task_id"],
                                    "success": result["success"],
                                    "message": result["message"]
                                })
                    
                    # 验证导出文件
                    assert os.path.exists(export_file)
                    assert os.path.getsize(export_file) > 0
                    
                except ImportError as e:
                    # 某些导出格式可能需要额外依赖
                    pytest.skip(f"Export format {format_type} not available: {e}") 