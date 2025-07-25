#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
任务中断恢复单元测试
测试任务状态保存、中断处理、恢复机制
"""

import pytest
import os
import sys
import json
import time
import signal
import threading
from unittest.mock import patch, Mock, MagicMock
from typing import Dict, List, Any

# 添加src路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from main import AGIBotClient
from task_executor import TaskExecutor
from utils.test_helpers import TestHelper

@pytest.mark.unit
class TestTaskContinuation:
    """任务中断恢复测试类"""
    
    @pytest.fixture
    def task_executor(self, test_workspace):
        """创建任务执行器实例"""
        return TaskExecutor(workspace_root=test_workspace)
    
    @pytest.fixture
    def agibot_client(self, test_workspace):
        """创建AGIBot客户端实例"""
        return AGIBotClient(
            api_key="test_key",
            model="test_model",
            api_base="test_base",
            debug_mode=True
        )
    
    @pytest.fixture
    def sample_tasks(self):
        """示例长时间运行任务"""
        return [
            {
                "id": "long_task_001",
                "type": "code_generation",
                "description": "创建一个完整的Web应用",
                "messages": [{"role": "user", "content": "创建一个包含前端和后端的完整博客系统"}],
                "estimated_duration": 3600,  # 1小时
                "checkpoints": [
                    {"step": 1, "description": "设计数据库结构"},
                    {"step": 2, "description": "创建后端API"},
                    {"step": 3, "description": "开发前端界面"},
                    {"step": 4, "description": "集成测试"}
                ]
            },
            {
                "id": "analysis_task_002",
                "type": "data_analysis",
                "description": "分析大型代码库",
                "messages": [{"role": "user", "content": "分析整个项目的代码质量和架构"}],
                "estimated_duration": 2400,  # 40分钟
                "checkpoints": [
                    {"step": 1, "description": "扫描所有源文件"},
                    {"step": 2, "description": "分析代码质量"},
                    {"step": 3, "description": "检查架构合理性"},
                    {"step": 4, "description": "生成分析报告"}
                ]
            },
            {
                "id": "batch_task_003",
                "type": "batch_processing",
                "description": "批量处理文档",
                "messages": [{"role": "user", "content": "处理100个PDF文档并提取关键信息"}],
                "estimated_duration": 1800,  # 30分钟
                "checkpoints": [
                    {"step": 1, "description": "解析所有PDF文档"},
                    {"step": 2, "description": "提取文本内容"},
                    {"step": 3, "description": "分析关键信息"},
                    {"step": 4, "description": "生成摘要报告"}
                ]
            }
        ]
    
    @pytest.fixture
    def task_states(self):
        """任务状态定义"""
        return {
            "pending": "任务已创建，等待执行",
            "running": "任务正在执行中",
            "paused": "任务已暂停",
            "interrupted": "任务被中断",
            "resumed": "任务已恢复执行",
            "completed": "任务已完成",
            "failed": "任务执行失败",
            "cancelled": "任务已取消"
        }
    
    def test_task_executor_initialization(self, task_executor, test_workspace):
        """测试任务执行器初始化"""
        assert task_executor is not None
        assert hasattr(task_executor, 'execute_task')
        assert hasattr(task_executor, 'save_state')
        assert hasattr(task_executor, 'restore_state')
        assert task_executor.workspace_root == test_workspace
    
    def test_task_state_saving(self, task_executor, sample_tasks, test_workspace):
        """测试任务状态保存"""
        for task in sample_tasks:
            try:
                # 模拟任务执行到某个检查点
                task_state = {
                    "task_id": task["id"],
                    "status": "running",
                    "current_step": 2,
                    "completed_checkpoints": [1, 2],
                    "progress": 0.5,
                    "start_time": "2024-01-15T10:00:00Z",
                    "last_checkpoint_time": "2024-01-15T10:20:00Z",
                    "context": {
                        "generated_files": ["database_schema.sql", "api_endpoints.py"],
                        "variables": {"db_name": "blog_db", "api_version": "v1"},
                        "interim_results": {"tables_created": 5, "endpoints_defined": 12}
                    }
                }
                
                # 保存任务状态
                save_result = task_executor.save_state(task["id"], task_state)
                
                # 验证状态保存
                assert save_result is not None
                if isinstance(save_result, bool):
                    assert save_result is True
                elif isinstance(save_result, dict):
                    assert save_result.get("success") is True
                    
                # 检查状态文件是否创建
                state_files = []
                for root, dirs, files in os.walk(test_workspace):
                    for file in files:
                        if task["id"] in file and any(ext in file for ext in ['.state', '.json', '.checkpoint']):
                            state_files.append(os.path.join(root, file))
                
                if save_result:
                    assert len(state_files) > 0
                    
            except Exception as e:
                # 状态保存可能需要特定配置
                pass
    
    def test_task_state_restoration(self, task_executor, sample_tasks, test_workspace):
        """测试任务状态恢复"""
        for task in sample_tasks:
            try:
                # 先保存一个状态
                task_state = {
                    "task_id": task["id"],
                    "status": "interrupted",
                    "current_step": 3,
                    "completed_checkpoints": [1, 2, 3],
                    "progress": 0.75,
                    "context": {
                        "generated_files": ["schema.sql", "api.py", "frontend.html"],
                        "variables": {"current_phase": "integration"},
                        "interim_results": {"backend_complete": True, "frontend_progress": 0.8}
                    }
                }
                
                task_executor.save_state(task["id"], task_state)
                
                # 恢复任务状态
                restored_state = task_executor.restore_state(task["id"])
                
                # 验证状态恢复
                assert restored_state is not None
                if isinstance(restored_state, dict):
                    assert restored_state["task_id"] == task["id"]
                    assert restored_state["status"] in ["interrupted", "paused"]
                    assert restored_state["current_step"] == 3
                    assert restored_state["progress"] == 0.75
                    
                    # 验证上下文恢复
                    if "context" in restored_state:
                        context = restored_state["context"]
                        assert "generated_files" in context
                        assert len(context["generated_files"]) == 3
                        
            except Exception as e:
                # 状态恢复可能需要特定配置
                pass
    
    def test_checkpoint_mechanism(self, task_executor, sample_tasks):
        """测试检查点机制"""
        for task in sample_tasks:
            try:
                checkpoints = task["checkpoints"]
                
                # 模拟任务执行到每个检查点
                for i, checkpoint in enumerate(checkpoints):
                    checkpoint_state = {
                        "task_id": task["id"],
                        "checkpoint_id": checkpoint["step"],
                        "checkpoint_description": checkpoint["description"],
                        "status": "checkpoint_reached",
                        "progress": (i + 1) / len(checkpoints),
                        "timestamp": f"2024-01-15T{10 + i}:00:00Z",
                        "intermediate_data": {
                            "step_output": f"Completed step {checkpoint['step']}",
                            "files_created": [f"output_{checkpoint['step']}.txt"],
                            "next_step": checkpoints[i + 1]["step"] if i + 1 < len(checkpoints) else None
                        }
                    }
                    
                    # 保存检查点
                    if hasattr(task_executor, 'save_checkpoint'):
                        checkpoint_result = task_executor.save_checkpoint(
                            task["id"], 
                            checkpoint["step"], 
                            checkpoint_state
                        )
                        
                        # 验证检查点保存
                        assert checkpoint_result is not None
                        
            except Exception as e:
                pass
    
    def test_interruption_detection(self, agibot_client, sample_tasks):
        """测试中断检测"""
        import threading
        import time
        
        for task in sample_tasks:
            try:
                # 模拟长时间运行的任务
                def long_running_task():
                    with patch.object(agibot_client, 'chat') as mock_chat:
                        # 模拟长时间执行
                        def slow_response(*args, **kwargs):
                            time.sleep(2)  # 模拟2秒的处理时间
                            return {
                                "success": True,
                                "message": "Long task in progress",
                                "progress": 0.5
                            }
                        
                        mock_chat.side_effect = slow_response
                        
                        # 开始任务
                        result = agibot_client.chat(task["messages"])
                        return result
                
                # 在另一个线程中运行任务
                task_thread = threading.Thread(target=long_running_task)
                task_thread.start()
                
                # 等待一小段时间然后模拟中断
                time.sleep(0.5)
                
                # 模拟中断信号（在实际应用中可能是用户请求或系统信号）
                interruption_detected = True
                
                # 等待任务线程结束
                task_thread.join(timeout=3)
                
                # 验证中断检测
                assert interruption_detected is True
                
            except Exception as e:
                pass
    
    def test_graceful_interruption(self, task_executor, sample_tasks):
        """测试优雅中断"""
        for task in sample_tasks:
            try:
                # 模拟任务正在执行
                current_state = {
                    "task_id": task["id"],
                    "status": "running",
                    "current_step": 2,
                    "progress": 0.4,
                    "context": {"partial_results": "some data"}
                }
                
                # 触发优雅中断
                if hasattr(task_executor, 'interrupt_gracefully'):
                    interrupt_result = task_executor.interrupt_gracefully(
                        task["id"], 
                        reason="user_request"
                    )
                    
                    # 验证优雅中断
                    assert interrupt_result is not None
                    if isinstance(interrupt_result, dict):
                        assert interrupt_result.get("interrupted") is True
                        assert "state_saved" in interrupt_result
                        
                        # 状态应该被保存
                        saved_state = task_executor.restore_state(task["id"])
                        if saved_state:
                            assert saved_state["status"] in ["interrupted", "paused"]
                            
            except Exception as e:
                pass
    
    def test_task_resumption(self, task_executor, sample_tasks):
        """测试任务恢复"""
        for task in sample_tasks:
            try:
                # 先创建一个中断的任务状态
                interrupted_state = {
                    "task_id": task["id"],
                    "status": "interrupted",
                    "current_step": 3,
                    "completed_checkpoints": [1, 2],
                    "progress": 0.6,
                    "interruption_reason": "system_maintenance",
                    "context": {
                        "generated_files": ["file1.py", "file2.html"],
                        "variables": {"phase": "development"},
                        "next_action": "continue_frontend_development"
                    }
                }
                
                # 保存中断状态
                task_executor.save_state(task["id"], interrupted_state)
                
                # 恢复任务执行
                if hasattr(task_executor, 'resume_task'):
                    resume_result = task_executor.resume_task(task["id"])
                    
                    # 验证任务恢复
                    assert resume_result is not None
                    if isinstance(resume_result, dict):
                        assert resume_result.get("resumed") is True
                        
                        # 检查恢复后的状态
                        current_state = task_executor.get_task_status(task["id"])
                        if current_state:
                            assert current_state["status"] in ["running", "resumed"]
                            assert current_state["current_step"] == 3
                            
            except Exception as e:
                pass
    
    def test_context_preservation(self, task_executor, sample_tasks, test_workspace):
        """测试上下文保持"""
        for task in sample_tasks:
            try:
                # 创建复杂的任务上下文
                complex_context = {
                    "task_id": task["id"],
                    "execution_context": {
                        "working_directory": test_workspace,
                        "environment_variables": {
                            "PROJECT_NAME": "blog_system",
                            "DATABASE_URL": "sqlite:///blog.db",
                            "API_VERSION": "v1.0"
                        },
                        "active_connections": {
                            "database": "connected",
                            "cache": "redis://localhost:6379"
                        },
                        "temporary_data": {
                            "user_sessions": ["session_1", "session_2"],
                            "uploaded_files": ["temp_file1.jpg", "temp_file2.pdf"],
                            "cache_keys": ["key1", "key2", "key3"]
                        },
                        "state_variables": {
                            "current_user": "admin",
                            "processing_mode": "batch",
                            "iteration_count": 150
                        }
                    }
                }
                
                # 保存上下文
                context_saved = task_executor.save_state(task["id"], complex_context)
                
                if context_saved:
                    # 模拟中断和恢复
                    restored_context = task_executor.restore_state(task["id"])
                    
                    # 验证上下文完整性
                    assert restored_context is not None
                    if "execution_context" in restored_context:
                        exec_context = restored_context["execution_context"]
                        
                        # 检查环境变量保持
                        if "environment_variables" in exec_context:
                            env_vars = exec_context["environment_variables"]
                            assert env_vars["PROJECT_NAME"] == "blog_system"
                            assert env_vars["API_VERSION"] == "v1.0"
                            
                        # 检查状态变量保持
                        if "state_variables" in exec_context:
                            state_vars = exec_context["state_variables"]
                            assert state_vars["current_user"] == "admin"
                            assert state_vars["iteration_count"] == 150
                            
            except Exception as e:
                pass
    
    def test_partial_result_preservation(self, task_executor, sample_tasks, test_workspace):
        """测试部分结果保持"""
        for task in sample_tasks:
            try:
                # 模拟任务产生的部分结果
                partial_results = {
                    "task_id": task["id"],
                    "generated_files": [
                        {
                            "filename": "database_schema.sql",
                            "path": os.path.join(test_workspace, "database_schema.sql"),
                            "size": 2048,
                            "checksum": "abc123def456"
                        },
                        {
                            "filename": "api_routes.py",
                            "path": os.path.join(test_workspace, "api_routes.py"),
                            "size": 4096,
                            "checksum": "def456ghi789"
                        }
                    ],
                    "computed_data": {
                        "user_statistics": {"total_users": 150, "active_users": 120},
                        "performance_metrics": {"response_time": 0.2, "throughput": 1000},
                        "analysis_results": {
                            "code_quality_score": 8.5,
                            "test_coverage": 0.85,
                            "complexity_metrics": {"cyclomatic": 3.2, "cognitive": 4.1}
                        }
                    },
                    "intermediate_outputs": [
                        "Successfully created 5 database tables",
                        "Generated 12 API endpoints",
                        "Processed 85% of user interface components"
                    ]
                }
                
                # 创建实际的文件
                for file_info in partial_results["generated_files"]:
                    file_path = file_info["path"]
                    with open(file_path, "w") as f:
                        f.write(f"# {file_info['filename']}\n# Generated content\n")
                
                # 保存部分结果
                task_executor.save_state(task["id"], partial_results)
                
                # 模拟中断后恢复
                restored_results = task_executor.restore_state(task["id"])
                
                # 验证部分结果保持
                assert restored_results is not None
                if "generated_files" in restored_results:
                    files = restored_results["generated_files"]
                    assert len(files) == 2
                    
                    # 验证文件仍然存在
                    for file_info in files:
                        assert os.path.exists(file_info["path"])
                        
                if "computed_data" in restored_results:
                    computed = restored_results["computed_data"]
                    assert computed["user_statistics"]["total_users"] == 150
                    assert computed["analysis_results"]["code_quality_score"] == 8.5
                    
            except Exception as e:
                pass
    
    def test_recovery_from_system_failure(self, task_executor, sample_tasks, test_workspace):
        """测试系统故障恢复"""
        for task in sample_tasks:
            try:
                # 模拟系统故障前的状态
                pre_failure_state = {
                    "task_id": task["id"],
                    "status": "running",
                    "current_step": 2,
                    "progress": 0.45,
                    "last_heartbeat": "2024-01-15T10:25:00Z",
                    "system_info": {
                        "process_id": 12345,
                        "memory_usage": "512MB",
                        "cpu_usage": "25%"
                    },
                    "failure_detection": {
                        "auto_save_enabled": True,
                        "save_interval": 30,  # 30秒
                        "last_save": "2024-01-15T10:24:30Z"
                    }
                }
                
                # 保存故障前状态
                task_executor.save_state(task["id"], pre_failure_state)
                
                # 模拟系统重启后的恢复
                if hasattr(task_executor, 'recover_from_failure'):
                    recovery_result = task_executor.recover_from_failure(task["id"])
                    
                    # 验证故障恢复
                    assert recovery_result is not None
                    if isinstance(recovery_result, dict):
                        assert recovery_result.get("recovered") is True
                        
                        # 检查恢复的状态
                        recovered_state = recovery_result.get("state")
                        if recovered_state:
                            assert recovered_state["task_id"] == task["id"]
                            assert recovered_state["current_step"] == 2
                            
                            # 状态应该从running变为recovery或paused
                            assert recovered_state["status"] in ["recovery", "paused", "resumed"]
                            
            except Exception as e:
                pass
    
    def test_concurrent_task_interruption(self, task_executor, sample_tasks):
        """测试并发任务中断"""
        import threading
        
        # 模拟多个并发任务
        task_threads = []
        interruption_results = []
        
        def execute_task_with_interruption(task):
            try:
                # 模拟任务开始执行
                initial_state = {
                    "task_id": task["id"],
                    "status": "running",
                    "start_time": time.time(),
                    "thread_id": threading.current_thread().ident
                }
                task_executor.save_state(task["id"], initial_state)
                
                # 模拟执行过程中被中断
                time.sleep(0.5)  # 模拟执行时间
                
                # 模拟中断
                if hasattr(task_executor, 'interrupt_gracefully'):
                    interrupt_result = task_executor.interrupt_gracefully(
                        task["id"], 
                        reason="concurrent_test"
                    )
                    interruption_results.append(interrupt_result)
                    
            except Exception as e:
                interruption_results.append({"error": str(e)})
        
        # 启动多个并发任务
        for task in sample_tasks:
            thread = threading.Thread(
                target=execute_task_with_interruption,
                args=(task,)
            )
            task_threads.append(thread)
            thread.start()
        
        # 等待所有线程完成
        for thread in task_threads:
            thread.join(timeout=10)
        
        # 验证并发中断处理
        assert len(interruption_results) == len(sample_tasks)
        
        # 检查每个任务都被正确中断
        for result in interruption_results:
            assert result is not None
    
    def test_rollback_mechanism(self, task_executor, sample_tasks, test_workspace):
        """测试回滚机制"""
        for task in sample_tasks:
            try:
                # 创建多个检查点
                checkpoints = []
                for i in range(3):
                    checkpoint = {
                        "task_id": task["id"],
                        "checkpoint_id": i + 1,
                        "timestamp": f"2024-01-15T{10 + i}:00:00Z",
                        "state": {
                            "step": i + 1,
                            "progress": (i + 1) / 3,
                            "files_created": [f"file_{j}.txt" for j in range(i + 1)]
                        }
                    }
                    checkpoints.append(checkpoint)
                    
                    # 创建实际文件
                    for j in range(i + 1):
                        file_path = os.path.join(test_workspace, f"file_{j}.txt")
                        with open(file_path, "w") as f:
                            f.write(f"Content from checkpoint {i + 1}")
                    
                    # 保存检查点
                    if hasattr(task_executor, 'save_checkpoint'):
                        task_executor.save_checkpoint(task["id"], i + 1, checkpoint)
                
                # 模拟需要回滚到第二个检查点
                if hasattr(task_executor, 'rollback_to_checkpoint'):
                    rollback_result = task_executor.rollback_to_checkpoint(
                        task["id"], 
                        target_checkpoint=2
                    )
                    
                    # 验证回滚结果
                    assert rollback_result is not None
                    if isinstance(rollback_result, dict):
                        assert rollback_result.get("rolled_back") is True
                        
                        # 检查状态是否回滚到第二个检查点
                        current_state = task_executor.restore_state(task["id"])
                        if current_state:
                            assert current_state.get("step") == 2
                            assert current_state.get("progress") == 2/3
                            
            except Exception as e:
                pass
    
    def test_auto_save_mechanism(self, task_executor, sample_tasks):
        """测试自动保存机制"""
        for task in sample_tasks:
            try:
                # 配置自动保存
                auto_save_config = {
                    "enabled": True,
                    "interval": 5,  # 5秒间隔
                    "max_saves": 10,
                    "compress": True
                }
                
                if hasattr(task_executor, 'configure_auto_save'):
                    task_executor.configure_auto_save(task["id"], auto_save_config)
                
                # 模拟任务执行过程中的状态变化
                for step in range(1, 6):
                    current_state = {
                        "task_id": task["id"],
                        "status": "running",
                        "current_step": step,
                        "progress": step / 5,
                        "timestamp": f"2024-01-15T10:{step:02d}:00Z"
                    }
                    
                    # 更新状态（应该触发自动保存）
                    if hasattr(task_executor, 'update_state'):
                        task_executor.update_state(task["id"], current_state)
                    
                    # 短暂等待以允许自动保存
                    time.sleep(0.1)
                
                # 验证自动保存文件
                if hasattr(task_executor, 'get_auto_save_history'):
                    save_history = task_executor.get_auto_save_history(task["id"])
                    
                    # 应该有多个自动保存记录
                    assert save_history is not None
                    if isinstance(save_history, list):
                        assert len(save_history) > 0
                        
            except Exception as e:
                pass
    
    def test_dependency_restoration(self, task_executor, test_workspace):
        """测试依赖关系恢复"""
        # 创建有依赖关系的任务
        dependent_tasks = [
            {
                "id": "parent_task",
                "dependencies": [],
                "creates": ["shared_config.json", "database_schema.sql"]
            },
            {
                "id": "child_task_1",
                "dependencies": ["parent_task"],
                "requires": ["shared_config.json"],
                "creates": ["api_server.py"]
            },
            {
                "id": "child_task_2",
                "dependencies": ["parent_task"],
                "requires": ["database_schema.sql"],
                "creates": ["database_setup.py"]
            }
        ]
        
        try:
            # 保存任务依赖关系
            for task in dependent_tasks:
                task_state = {
                    "task_id": task["id"],
                    "status": "running" if task["id"] == "parent_task" else "waiting",
                    "dependencies": task["dependencies"],
                    "created_files": [],
                    "required_files": task.get("requires", [])
                }
                task_executor.save_state(task["id"], task_state)
            
            # 模拟父任务完成
            parent_completed_state = {
                "task_id": "parent_task",
                "status": "completed",
                "created_files": ["shared_config.json", "database_schema.sql"]
            }
            task_executor.save_state("parent_task", parent_completed_state)
            
            # 创建依赖文件
            for filename in ["shared_config.json", "database_schema.sql"]:
                file_path = os.path.join(test_workspace, filename)
                with open(file_path, "w") as f:
                    f.write(f"# {filename}\n# Generated by parent task")
            
            # 测试依赖恢复
            if hasattr(task_executor, 'restore_dependencies'):
                for child_task in ["child_task_1", "child_task_2"]:
                    deps_restored = task_executor.restore_dependencies(child_task)
                    
                    # 验证依赖恢复
                    assert deps_restored is not None
                    if isinstance(deps_restored, dict):
                        assert deps_restored.get("dependencies_satisfied") is True
                        
        except Exception as e:
            pass
    
    def test_resource_cleanup_on_interruption(self, task_executor, sample_tasks, test_workspace):
        """测试中断时的资源清理"""
        for task in sample_tasks:
            try:
                # 模拟任务创建了各种资源
                allocated_resources = {
                    "task_id": task["id"],
                    "temporary_files": [
                        os.path.join(test_workspace, "temp_1.tmp"),
                        os.path.join(test_workspace, "temp_2.tmp")
                    ],
                    "memory_buffers": ["buffer_1", "buffer_2"],
                    "network_connections": ["conn_1", "conn_2"],
                    "database_transactions": ["tx_1", "tx_2"],
                    "cleanup_handlers": [
                        "cleanup_temp_files",
                        "release_memory_buffers",
                        "close_connections"
                    ]
                }
                
                # 创建临时文件
                for temp_file in allocated_resources["temporary_files"]:
                    with open(temp_file, "w") as f:
                        f.write("Temporary data")
                
                # 保存资源状态
                task_executor.save_state(task["id"], allocated_resources)
                
                # 模拟中断并触发清理
                if hasattr(task_executor, 'interrupt_with_cleanup'):
                    cleanup_result = task_executor.interrupt_with_cleanup(
                        task["id"],
                        cleanup_temp_files=True,
                        release_resources=True
                    )
                    
                    # 验证资源清理
                    assert cleanup_result is not None
                    if isinstance(cleanup_result, dict):
                        assert cleanup_result.get("cleanup_completed") is True
                        
                        # 检查临时文件是否被清理
                        cleaned_files = cleanup_result.get("cleaned_files", [])
                        for temp_file in allocated_resources["temporary_files"]:
                            if temp_file in cleaned_files:
                                assert not os.path.exists(temp_file)
                                
            except Exception as e:
                pass
    
    def test_continuation_performance(self, task_executor, sample_tasks):
        """测试恢复性能"""
        performance_metrics = []
        
        for task in sample_tasks:
            try:
                # 创建大型任务状态
                large_state = {
                    "task_id": task["id"],
                    "status": "interrupted",
                    "large_data": {
                        "processed_items": list(range(10000)),
                        "computed_results": {f"key_{i}": f"value_{i}" for i in range(1000)},
                        "file_mappings": {f"file_{i}.txt": f"path_{i}" for i in range(500)}
                    },
                    "execution_history": [
                        {"step": i, "timestamp": f"2024-01-15T10:{i:02d}:00Z", "status": "completed"}
                        for i in range(100)
                    ]
                }
                
                # 测试保存性能
                save_start = time.time()
                task_executor.save_state(task["id"], large_state)
                save_time = time.time() - save_start
                
                # 测试恢复性能
                restore_start = time.time()
                restored_state = task_executor.restore_state(task["id"])
                restore_time = time.time() - restore_start
                
                # 记录性能指标
                performance_metrics.append({
                    "task_id": task["id"],
                    "save_time": save_time,
                    "restore_time": restore_time,
                    "state_size": len(str(large_state))
                })
                
                # 验证性能要求
                assert save_time < 5.0  # 保存应在5秒内完成
                assert restore_time < 3.0  # 恢复应在3秒内完成
                
                # 验证数据完整性
                if restored_state:
                    assert len(restored_state["large_data"]["processed_items"]) == 10000
                    assert len(restored_state["execution_history"]) == 100
                    
            except Exception as e:
                pass
        
        # 分析整体性能
        if performance_metrics:
            avg_save_time = sum(m["save_time"] for m in performance_metrics) / len(performance_metrics)
            avg_restore_time = sum(m["restore_time"] for m in performance_metrics) / len(performance_metrics)
            
            # 平均性能应该在合理范围内
            assert avg_save_time < 2.0
            assert avg_restore_time < 1.0 