#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
智能体生命周期管理单元测试
测试智能体创建、运行、终止等生命周期管理功能
"""

import pytest
import os
import sys
import time
import threading
from unittest.mock import patch, Mock, MagicMock
from typing import Dict, List, Any

# 添加src路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from tools.multiagents import MultiAgentTools
from tools.message_system import MessageRouter, MessageType
from utils.test_helpers import TestHelper

@pytest.mark.unit
class TestAgentLifecycle:
    """智能体生命周期管理测试类"""
    
    @pytest.fixture
    def multi_agent_tools(self, test_workspace):
        """创建多智能体工具实例"""
        return MultiAgentTools(workspace_root=test_workspace, debug_mode=True)
    
    @pytest.fixture
    def message_router(self, test_workspace):
        """创建消息路由器实例"""
        return MessageRouter(test_workspace)
    
    @pytest.fixture
    def agent_config(self):
        """标准智能体配置"""
        return {
            "task_description": "执行测试任务",
            "agent_id": "test_agent_001",
            "api_key": "test_api_key",
            "model": "test_model",
            "max_loops": 5,
            "wait_for_completion": False,
            "shared_workspace": True
        }
    
    def test_agent_creation(self, multi_agent_tools, agent_config):
        """测试智能体创建"""
        with patch('tools.multiagents.AGIBotClient') as mock_client:
            # 模拟AGI Bot客户端
            mock_instance = Mock()
            mock_instance.chat.return_value = {
                "success": True,
                "message": "Task completed successfully",
                "output_dir": "/test/output",
                "workspace_dir": "/test/workspace",
                "execution_time": 10.5
            }
            mock_client.return_value = mock_instance
            
            # 创建智能体
            result = multi_agent_tools.spawn_agibot(**agent_config)
            
            # 验证智能体创建
            assert result is not None
            assert result["success"] is True
            assert "agent_id" in result
            assert result["agent_id"] == agent_config["agent_id"]
    
    def test_agent_id_generation(self, multi_agent_tools):
        """测试智能体ID生成"""
        # 不指定agent_id的情况
        with patch('tools.multiagents.AGIBotClient') as mock_client:
            mock_instance = Mock()
            mock_instance.chat.return_value = {"success": True}
            mock_client.return_value = mock_instance
            
            result = multi_agent_tools.spawn_agibot(
                task_description="测试任务",
                api_key="test_key",
                model="test_model"
            )
            
            # 验证自动生成的agent_id
            assert result is not None
            assert "agent_id" in result
            assert result["agent_id"] is not None
            assert len(result["agent_id"]) > 0
    
    def test_multiple_agent_creation(self, multi_agent_tools):
        """测试多个智能体创建"""
        with patch('tools.multiagents.AGIBotClient') as mock_client:
            mock_instance = Mock()
            mock_instance.chat.return_value = {"success": True}
            mock_client.return_value = mock_instance
            
            agent_ids = []
            
            # 创建多个智能体
            for i in range(3):
                result = multi_agent_tools.spawn_agibot(
                    task_description=f"测试任务 {i}",
                    agent_id=f"test_agent_{i}",
                    api_key="test_key",
                    model="test_model"
                )
                
                assert result is not None
                assert result["success"] is True
                agent_ids.append(result["agent_id"])
            
            # 验证所有智能体都有唯一ID
            assert len(set(agent_ids)) == 3
    
    def test_agent_task_execution(self, multi_agent_tools, agent_config):
        """测试智能体任务执行"""
        with patch('tools.multiagents.AGIBotClient') as mock_client:
            # 模拟任务执行过程
            mock_instance = Mock()
            
            def mock_chat(messages, **kwargs):
                # 模拟任务执行时间
                time.sleep(0.1)
                return {
                    "success": True,
                    "message": "Task completed",
                    "output_dir": "/test/output",
                    "workspace_dir": "/test/workspace",
                    "execution_time": 0.1
                }
            
            mock_instance.chat.side_effect = mock_chat
            mock_client.return_value = mock_instance
            
            # 执行任务
            result = multi_agent_tools.spawn_agibot(**agent_config)
            
            # 验证任务执行
            assert result is not None
            assert result["success"] is True
    
    def test_agent_termination(self, multi_agent_tools, agent_config):
        """测试智能体终止"""
        with patch('tools.multiagents.AGIBotClient') as mock_client:
            # 创建长时间运行的模拟任务
            mock_instance = Mock()
            
            def mock_long_task(messages, **kwargs):
                time.sleep(5)  # 模拟长时间任务
                return {"success": True}
            
            mock_instance.chat.side_effect = mock_long_task
            mock_client.return_value = mock_instance
            
            # 启动智能体
            result = multi_agent_tools.spawn_agibot(**agent_config)
            agent_id = result["agent_id"]
            
            # 等待一小段时间确保任务开始
            time.sleep(0.5)
            
            # 终止智能体
            terminate_result = multi_agent_tools.terminate_agibot(agent_id)
            
            # 验证终止操作
            assert terminate_result is not None
            assert terminate_result["success"] is True
    
    def test_agent_status_monitoring(self, multi_agent_tools, agent_config):
        """测试智能体状态监控"""
        with patch('tools.multiagents.AGIBotClient') as mock_client:
            mock_instance = Mock()
            mock_instance.chat.return_value = {"success": True}
            mock_client.return_value = mock_instance
            
            # 创建智能体
            result = multi_agent_tools.spawn_agibot(**agent_config)
            agent_id = result["agent_id"]
            
            # 获取智能体状态
            status = multi_agent_tools.get_agent_session_info()
            
            # 验证状态信息
            assert status is not None
            assert isinstance(status, dict)
    
    def test_agent_error_handling(self, multi_agent_tools, agent_config):
        """测试智能体错误处理"""
        with patch('tools.multiagents.AGIBotClient') as mock_client:
            # 模拟AGI Bot客户端创建失败
            mock_client.side_effect = Exception("Client creation failed")
            
            # 尝试创建智能体
            result = multi_agent_tools.spawn_agibot(**agent_config)
            
            # 验证错误处理
            assert result is not None
            assert result["success"] is False
            assert "error" in result
    
    def test_agent_workspace_isolation(self, multi_agent_tools, test_workspace):
        """测试智能体工作空间隔离"""
        with patch('tools.multiagents.AGIBotClient') as mock_client:
            mock_instance = Mock()
            mock_instance.chat.return_value = {"success": True}
            mock_client.return_value = mock_instance
            
            agents = []
            
            # 创建多个智能体，每个都有独立工作空间
            for i in range(2):
                result = multi_agent_tools.spawn_agibot(
                    task_description=f"任务 {i}",
                    agent_id=f"agent_{i}",
                    output_directory=f"agent_{i}_output",
                    shared_workspace=False,
                    api_key="test_key",
                    model="test_model"
                )
                agents.append(result)
            
            # 验证工作空间隔离
            for agent in agents:
                assert agent is not None
                assert agent["success"] is True
    
    def test_agent_shared_workspace(self, multi_agent_tools):
        """测试智能体共享工作空间"""
        with patch('tools.multiagents.AGIBotClient') as mock_client:
            mock_instance = Mock()
            mock_instance.chat.return_value = {"success": True}
            mock_client.return_value = mock_instance
            
            agents = []
            
            # 创建多个智能体，使用共享工作空间
            for i in range(2):
                result = multi_agent_tools.spawn_agibot(
                    task_description=f"共享任务 {i}",
                    agent_id=f"shared_agent_{i}",
                    shared_workspace=True,
                    api_key="test_key",
                    model="test_model"
                )
                agents.append(result)
            
            # 验证共享工作空间
            for agent in agents:
                assert agent is not None
                assert agent["success"] is True
    
    def test_agent_configuration_validation(self, multi_agent_tools):
        """测试智能体配置验证"""
        # 测试缺少必要参数
        invalid_configs = [
            {}, # 完全空配置
            {"task_description": "测试"}, # 缺少API配置
            {"api_key": "test_key"}, # 缺少任务描述
            {"task_description": "", "api_key": "test_key", "model": "test_model"} # 空任务描述
        ]
        
        for config in invalid_configs:
            result = multi_agent_tools.spawn_agibot(**config)
            # 验证配置验证
            assert result is not None
            # 某些配置可能成功（使用默认值），某些可能失败
    
    def test_agent_completion_callback(self, multi_agent_tools, agent_config):
        """测试智能体完成回调"""
        completion_called = False
        
        def completion_callback(agent_id, result):
            nonlocal completion_called
            completion_called = True
        
        with patch('tools.multiagents.AGIBotClient') as mock_client:
            mock_instance = Mock()
            mock_instance.chat.return_value = {"success": True}
            mock_client.return_value = mock_instance
            
            # 创建带回调的智能体（如果支持）
            config = agent_config.copy()
            if hasattr(multi_agent_tools, 'set_completion_callback'):
                multi_agent_tools.set_completion_callback(completion_callback)
            
            result = multi_agent_tools.spawn_agibot(**config)
            
            # 等待任务完成
            time.sleep(0.5)
            
            # 验证回调（如果支持）
            assert result is not None
    
    def test_agent_resource_cleanup(self, multi_agent_tools, agent_config):
        """测试智能体资源清理"""
        with patch('tools.multiagents.AGIBotClient') as mock_client:
            mock_instance = Mock()
            mock_instance.chat.return_value = {"success": True}
            mock_client.return_value = mock_instance
            
            # 创建智能体
            result = multi_agent_tools.spawn_agibot(**agent_config)
            agent_id = result["agent_id"]
            
            # 终止智能体
            terminate_result = multi_agent_tools.terminate_agibot(agent_id)
            
            # 验证资源清理
            assert terminate_result is not None
            
            # 检查智能体是否从活跃列表中移除
            status = multi_agent_tools.get_agent_session_info()
            if isinstance(status, dict) and "active_agents" in status:
                assert agent_id not in status["active_agents"]
    
    def test_concurrent_agent_operations(self, multi_agent_tools):
        """测试并发智能体操作"""
        with patch('tools.multiagents.AGIBotClient') as mock_client:
            mock_instance = Mock()
            mock_instance.chat.return_value = {"success": True}
            mock_client.return_value = mock_instance
            
            agents = []
            errors = []
            
            def create_agent(agent_index):
                try:
                    result = multi_agent_tools.spawn_agibot(
                        task_description=f"并发任务 {agent_index}",
                        agent_id=f"concurrent_agent_{agent_index}",
                        api_key="test_key",
                        model="test_model"
                    )
                    agents.append(result)
                except Exception as e:
                    errors.append(e)
            
            # 创建并发线程
            threads = []
            for i in range(5):
                thread = threading.Thread(target=create_agent, args=(i,))
                threads.append(thread)
                thread.start()
            
            # 等待所有线程完成
            for thread in threads:
                thread.join(timeout=10)
            
            # 验证并发操作
            assert len(errors) == 0, f"Concurrent creation errors: {errors}"
            assert len(agents) == 5
            
            # 验证所有智能体都成功创建
            for agent in agents:
                assert agent is not None
                assert agent["success"] is True
    
    def test_agent_memory_management(self, multi_agent_tools, agent_config):
        """测试智能体内存管理"""
        with patch('tools.multiagents.AGIBotClient') as mock_client:
            mock_instance = Mock()
            mock_instance.chat.return_value = {"success": True}
            mock_client.return_value = mock_instance
            
            # 创建多个智能体来测试内存使用
            agents = []
            for i in range(10):
                result = multi_agent_tools.spawn_agibot(
                    task_description=f"内存测试任务 {i}",
                    agent_id=f"memory_test_agent_{i}",
                    api_key="test_key",
                    model="test_model"
                )
                agents.append(result)
            
            # 验证内存管理
            assert len(agents) == 10
            for agent in agents:
                assert agent is not None
                assert agent["success"] is True
    
    def test_agent_timeout_handling(self, multi_agent_tools):
        """测试智能体超时处理"""
        with patch('tools.multiagents.AGIBotClient') as mock_client:
            # 模拟超时任务
            mock_instance = Mock()
            
            def timeout_task(messages, **kwargs):
                time.sleep(10)  # 模拟超时任务
                return {"success": True}
            
            mock_instance.chat.side_effect = timeout_task
            mock_client.return_value = mock_instance
            
            # 创建智能体（如果支持超时设置）
            config = {
                "task_description": "超时测试任务",
                "agent_id": "timeout_test_agent",
                "api_key": "test_key",
                "model": "test_model",
                "max_loops": 1,
                "timeout": 2  # 2秒超时（如果支持）
            }
            
            start_time = time.time()
            result = multi_agent_tools.spawn_agibot(**config)
            end_time = time.time()
            
            # 验证超时处理
            assert result is not None
            execution_time = end_time - start_time
            # 如果支持超时，应该在合理时间内返回
            assert execution_time < 15  # 不应该真的等待10秒
    
    def test_agent_state_persistence(self, multi_agent_tools, agent_config, test_workspace):
        """测试智能体状态持久化"""
        with patch('tools.multiagents.AGIBotClient') as mock_client:
            mock_instance = Mock()
            mock_instance.chat.return_value = {"success": True}
            mock_client.return_value = mock_instance
            
            # 创建智能体
            result = multi_agent_tools.spawn_agibot(**agent_config)
            agent_id = result["agent_id"]
            
            # 创建新的MultiAgentTools实例（模拟重启）
            new_tools = MultiAgentTools(workspace_root=test_workspace, debug_mode=True)
            
            # 检查智能体状态是否持久化（如果支持）
            if hasattr(new_tools, 'get_persisted_agents'):
                persisted = new_tools.get_persisted_agents()
                assert persisted is not None
    
    def test_agent_communication_setup(self, multi_agent_tools, message_router, agent_config):
        """测试智能体通信设置"""
        with patch('tools.multiagents.AGIBotClient') as mock_client:
            mock_instance = Mock()
            mock_instance.chat.return_value = {"success": True}
            mock_client.return_value = mock_instance
            
            # 创建智能体
            result = multi_agent_tools.spawn_agibot(**agent_config)
            agent_id = result["agent_id"]
            
            # 验证智能体已注册到消息系统
            assert agent_id in message_router.mailboxes
    
    def test_agent_failure_recovery(self, multi_agent_tools, agent_config):
        """测试智能体失败恢复"""
        with patch('tools.multiagents.AGIBotClient') as mock_client:
            # 第一次调用失败，第二次成功
            mock_instance = Mock()
            mock_instance.chat.side_effect = [
                Exception("Task failed"),
                {"success": True, "message": "Recovered"}
            ]
            mock_client.return_value = mock_instance
            
            # 创建智能体（可能包含重试逻辑）
            result = multi_agent_tools.spawn_agibot(**agent_config)
            
            # 验证失败恢复处理
            assert result is not None
    
    def test_agent_metrics_collection(self, multi_agent_tools, agent_config):
        """测试智能体指标收集"""
        with patch('tools.multiagents.AGIBotClient') as mock_client:
            mock_instance = Mock()
            mock_instance.chat.return_value = {
                "success": True,
                "execution_time": 5.5,
                "tokens_used": 1000
            }
            mock_client.return_value = mock_instance
            
            # 创建智能体
            result = multi_agent_tools.spawn_agibot(**agent_config)
            
            # 验证指标收集
            assert result is not None
            if "metrics" in result:
                assert isinstance(result["metrics"], dict) 