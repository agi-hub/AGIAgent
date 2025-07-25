#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多代理协作单元测试
测试spawn_agibot、agent通信、消息传递等多代理功能
"""

import pytest
import os
import json
import threading
import time
from unittest.mock import patch, Mock, MagicMock
import sys

# 添加src路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from tools.multiagents import MultiAgentTools
from tools.message_system import MessageSystem
from main import AGIBotClient

class TestMultiAgentCollaboration:
    """多代理协作测试类"""
    
    @pytest.fixture
    def multi_agent_tools(self, test_workspace):
        """创建多代理工具实例"""
        return MultiAgentTools(workspace_root=test_workspace)
    
    @pytest.fixture
    def message_system(self, test_workspace):
        """创建消息系统实例"""
        mailbox_dir = os.path.join(test_workspace, "mailboxes")
        os.makedirs(mailbox_dir, exist_ok=True)
        return MessageSystem(mailbox_dir=mailbox_dir)
    
    def test_spawn_agibot_basic(self, multi_agent_tools, test_workspace):
        """测试基础的AGIBot生成功能"""
        
        # 模拟spawn_agibot调用
        with patch('subprocess.Popen') as mock_popen:
            mock_process = Mock()
            mock_process.pid = 12345
            mock_process.poll.return_value = None  # 进程仍在运行
            mock_popen.return_value = mock_process
            
            result = multi_agent_tools.spawn_agibot(
                requirement="创建一个简单的计算器",
                agent_id="agent_001",
                workspace_dir=os.path.join(test_workspace, "agent_001")
            )
            
            # 验证返回结果
            assert result["status"] == "success"
            assert "agent_001" in result["message"]
            assert result["agent_id"] == "agent_001"
            assert result["pid"] == 12345
            
            # 验证subprocess.Popen被正确调用
            mock_popen.assert_called_once()
            call_args = mock_popen.call_args
            assert "python" in call_args[0][0][0].lower()
            assert "agibot.py" in call_args[0][0][-2]
            assert "创建一个简单的计算器" in call_args[0][0][-1]
    
    def test_spawn_agibot_with_parameters(self, multi_agent_tools, test_workspace):
        """测试带参数的AGIBot生成"""
        
        with patch('subprocess.Popen') as mock_popen:
            mock_process = Mock()
            mock_process.pid = 12346
            mock_process.poll.return_value = None
            mock_popen.return_value = mock_process
            
            result = multi_agent_tools.spawn_agibot(
                requirement="开发Web应用",
                agent_id="web_agent",
                workspace_dir=os.path.join(test_workspace, "web_agent"),
                loops=10,
                debug=True,
                model="gpt-4"
            )
            
            assert result["status"] == "success"
            assert result["agent_id"] == "web_agent"
            
            # 验证命令行参数
            call_args = mock_popen.call_args[0][0]
            assert "--loops" in call_args
            assert "10" in call_args
            assert "--debug" in call_args
            assert "--model" in call_args
            assert "gpt-4" in call_args
    
    def test_spawn_agibot_error_handling(self, multi_agent_tools, test_workspace):
        """测试AGIBot生成错误处理"""
        
        # 模拟进程启动失败
        with patch('subprocess.Popen', side_effect=Exception("Failed to start process")):
            result = multi_agent_tools.spawn_agibot(
                requirement="测试任务",
                agent_id="failed_agent",
                workspace_dir=os.path.join(test_workspace, "failed_agent")
            )
            
            assert result["status"] == "error"
            assert "Failed to start process" in result["message"]
    
    def test_terminate_agibot(self, multi_agent_tools):
        """测试终止AGIBot"""
        
        # 模拟运行中的进程
        with patch('psutil.process_iter') as mock_process_iter:
            mock_process = Mock()
            mock_process.pid = 12345
            mock_process.cmdline.return_value = ["python", "agibot.py", "--agent-id", "test_agent"]
            mock_process.terminate.return_value = None
            mock_process_iter.return_value = [mock_process]
            
            result = multi_agent_tools.terminate_agibot(agent_id="test_agent")
            
            assert result["status"] == "success"
            assert "test_agent" in result["message"]
            mock_process.terminate.assert_called_once()
    
    def test_terminate_agibot_not_found(self, multi_agent_tools):
        """测试终止不存在的AGIBot"""
        
        with patch('psutil.process_iter', return_value=[]):
            result = multi_agent_tools.terminate_agibot(agent_id="nonexistent_agent")
            
            assert result["status"] == "error"
            assert "not found" in result["message"].lower()
    
    def test_send_message_to_agent(self, multi_agent_tools, message_system):
        """测试向agent发送消息"""
        
        # 设置消息系统
        multi_agent_tools.message_system = message_system
        
        result = multi_agent_tools.send_message_to_agent_or_manager(
            recipient_id="agent_002",
            sender_id="agent_001",
            message_content="请帮我处理数据分析任务",
            message_type="task_request"
        )
        
        assert result["status"] == "success"
        assert "agent_002" in result["message"]
        
        # 验证消息已发送
        messages = message_system.get_messages("agent_002")
        assert len(messages) > 0
        assert messages[0]["sender_id"] == "agent_001"
        assert messages[0]["content"] == "请帮我处理数据分析任务"
        assert messages[0]["message_type"] == "task_request"
    
    def test_get_agent_messages(self, multi_agent_tools, message_system):
        """测试获取agent消息"""
        
        multi_agent_tools.message_system = message_system
        
        # 先发送一些消息
        message_system.send_message(
            sender_id="manager",
            recipient_id="agent_003",
            content="开始执行任务",
            message_type="command"
        )
        
        message_system.send_message(
            sender_id="agent_001",
            recipient_id="agent_003", 
            content="协作完成项目",
            message_type="collaboration"
        )
        
        result = multi_agent_tools.get_agent_messages(agent_id="agent_003")
        
        assert result["status"] == "success"
        assert len(result["messages"]) == 2
        
        # 验证消息内容
        messages = result["messages"]
        assert any(msg["content"] == "开始执行任务" for msg in messages)
        assert any(msg["content"] == "协作完成项目" for msg in messages)
    
    def test_broadcast_message_to_agents(self, multi_agent_tools, message_system):
        """测试向所有agent广播消息"""
        
        multi_agent_tools.message_system = message_system
        
        # 创建一些agent邮箱
        agent_ids = ["agent_001", "agent_002", "agent_003"]
        for agent_id in agent_ids:
            os.makedirs(os.path.join(message_system.mailbox_dir, agent_id), exist_ok=True)
        
        result = multi_agent_tools.broadcast_message_to_agents(
            sender_id="manager",
            message_content="所有agent请注意：项目截止日期是明天",
            message_type="announcement"
        )
        
        assert result["status"] == "success"
        assert str(len(agent_ids)) in result["message"]
        
        # 验证所有agent都收到了广播消息
        for agent_id in agent_ids:
            messages = message_system.get_messages(agent_id)
            assert len(messages) > 0
            broadcast_msg = messages[-1]  # 最新消息
            assert broadcast_msg["sender_id"] == "manager"
            assert broadcast_msg["content"] == "所有agent请注意：项目截止日期是明天"
            assert broadcast_msg["message_type"] == "announcement"
    
    def test_multi_agent_coordination_scenario(self, multi_agent_tools, message_system, test_workspace):
        """测试多agent协调场景"""
        
        multi_agent_tools.message_system = message_system
        
        # 模拟复杂的多agent协作场景
        with patch('subprocess.Popen') as mock_popen:
            # 创建多个模拟进程
            processes = []
            for i in range(3):
                mock_process = Mock()
                mock_process.pid = 10000 + i
                mock_process.poll.return_value = None
                processes.append(mock_process)
            
            mock_popen.side_effect = processes
            
            # 步骤1: 创建多个专门的agent
            agents = [
                {"id": "data_analyst", "task": "分析销售数据", "role": "数据分析师"},
                {"id": "web_developer", "task": "创建数据展示网站", "role": "前端开发"},
                {"id": "report_generator", "task": "生成分析报告", "role": "报告生成器"}
            ]
            
            spawned_agents = []
            for agent in agents:
                result = multi_agent_tools.spawn_agibot(
                    requirement=f"作为{agent['role']}，{agent['task']}",
                    agent_id=agent["id"],
                    workspace_dir=os.path.join(test_workspace, agent["id"])
                )
                assert result["status"] == "success"
                spawned_agents.append(agent["id"])
            
            # 步骤2: Manager向所有agent发布项目启动消息
            broadcast_result = multi_agent_tools.broadcast_message_to_agents(
                sender_id="project_manager",
                message_content="项目开始：电商销售数据分析项目启动",
                message_type="project_start"
            )
            assert broadcast_result["status"] == "success"
            
            # 步骤3: 设置agent间的协作关系
            # Data analyst完成后通知web developer
            collab_result1 = multi_agent_tools.send_message_to_agent_or_manager(
                recipient_id="web_developer",
                sender_id="data_analyst", 
                message_content="数据分析完成，结果文件已准备好，可以开始前端开发",
                message_type="task_complete"
            )
            assert collab_result1["status"] == "success"
            
            # Web developer完成后通知report generator
            collab_result2 = multi_agent_tools.send_message_to_agent_or_manager(
                recipient_id="report_generator",
                sender_id="web_developer",
                message_content="网站开发完成，可以生成最终报告",
                message_type="ready_for_report"
            )
            assert collab_result2["status"] == "success"
            
            # 步骤4: 验证消息流
            # 验证web_developer收到了data_analyst的消息
            web_messages = multi_agent_tools.get_agent_messages("web_developer")
            assert web_messages["status"] == "success"
            assert len(web_messages["messages"]) >= 2  # 广播消息 + 协作消息
            
            # 验证report_generator收到了消息
            report_messages = multi_agent_tools.get_agent_messages("report_generator")
            assert report_messages["status"] == "success"
            assert len(report_messages["messages"]) >= 2
            
            # 步骤5: 项目完成，清理agent
            for agent_id in spawned_agents:
                with patch('psutil.process_iter') as mock_iter:
                    mock_process = Mock()
                    mock_process.pid = 10000
                    mock_process.cmdline.return_value = ["python", "agibot.py", "--agent-id", agent_id]
                    mock_process.terminate.return_value = None
                    mock_iter.return_value = [mock_process]
                    
                    terminate_result = multi_agent_tools.terminate_agibot(agent_id=agent_id)
                    assert terminate_result["status"] == "success"
    
    def test_message_system_reliability(self, message_system):
        """测试消息系统可靠性"""
        
        # 测试大量消息的处理
        num_messages = 100
        sender_id = "stress_test_sender"
        recipient_id = "stress_test_recipient"
        
        # 发送大量消息
        start_time = time.time()
        for i in range(num_messages):
            result = message_system.send_message(
                sender_id=sender_id,
                recipient_id=recipient_id,
                content=f"消息 {i}: 这是压力测试消息",
                message_type="stress_test"
            )
            assert result["status"] == "success"
        
        send_time = time.time() - start_time
        
        # 接收所有消息
        start_time = time.time()
        messages = message_system.get_messages(recipient_id)
        receive_time = time.time() - start_time
        
        # 验证消息完整性
        assert len(messages) == num_messages
        
        # 验证消息顺序和内容
        for i, message in enumerate(messages):
            assert message["sender_id"] == sender_id
            assert f"消息 {i}" in message["content"]
            assert message["message_type"] == "stress_test"
            assert "timestamp" in message
        
        # 性能验证
        print(f"Message system performance:")
        print(f"  Send time: {send_time:.3f}s ({num_messages/send_time:.1f} msg/s)")
        print(f"  Receive time: {receive_time:.3f}s")
        
        assert send_time < 5.0, f"Sending {num_messages} messages took too long: {send_time}s"
        assert receive_time < 1.0, f"Receiving {num_messages} messages took too long: {receive_time}s"
    
    def test_concurrent_message_operations(self, message_system):
        """测试并发消息操作"""
        
        num_threads = 5
        messages_per_thread = 20
        
        results = []
        errors = []
        
        def message_worker(thread_id):
            try:
                thread_results = []
                
                # 每个线程发送消息给其他线程
                for i in range(messages_per_thread):
                    for target_thread in range(num_threads):
                        if target_thread != thread_id:
                            result = message_system.send_message(
                                sender_id=f"thread_{thread_id}",
                                recipient_id=f"thread_{target_thread}",
                                content=f"来自线程{thread_id}的消息{i}",
                                message_type="concurrent_test"
                            )
                            thread_results.append(result)
                
                results.extend(thread_results)
                
            except Exception as e:
                errors.append((thread_id, str(e)))
        
        # 启动并发线程
        threads = []
        for thread_id in range(num_threads):
            thread = threading.Thread(target=message_worker, args=(thread_id,))
            threads.append(thread)
            thread.start()
        
        # 等待所有线程完成
        for thread in threads:
            thread.join()
        
        # 验证并发操作结果
        assert len(errors) == 0, f"Concurrent message operations had errors: {errors}"
        
        expected_messages = num_threads * messages_per_thread * (num_threads - 1)
        assert len(results) == expected_messages
        
        # 验证所有消息都成功发送
        success_count = sum(1 for result in results if result["status"] == "success")
        assert success_count == expected_messages
        
        # 验证每个线程都收到了正确数量的消息
        for thread_id in range(num_threads):
            messages = message_system.get_messages(f"thread_{thread_id}")
            expected_received = (num_threads - 1) * messages_per_thread
            assert len(messages) == expected_received, f"Thread {thread_id} received {len(messages)}, expected {expected_received}"
    
    def test_agent_lifecycle_management(self, multi_agent_tools, test_workspace):
        """测试agent生命周期管理"""
        
        agent_id = "lifecycle_test_agent"
        
        # 第1步: 创建agent
        with patch('subprocess.Popen') as mock_popen:
            mock_process = Mock()
            mock_process.pid = 99999
            mock_process.poll.return_value = None
            mock_popen.return_value = mock_process
            
            spawn_result = multi_agent_tools.spawn_agibot(
                requirement="生命周期测试任务",
                agent_id=agent_id,
                workspace_dir=os.path.join(test_workspace, agent_id)
            )
            
            assert spawn_result["status"] == "success"
            assert spawn_result["agent_id"] == agent_id
        
        # 第2步: 验证agent状态
        with patch('psutil.process_iter') as mock_iter:
            mock_process = Mock()
            mock_process.pid = 99999
            mock_process.cmdline.return_value = ["python", "agibot.py", "--agent-id", agent_id]
            mock_process.is_running.return_value = True
            mock_iter.return_value = [mock_process]
            
            # 这里可以添加获取agent状态的方法
            # status_result = multi_agent_tools.get_agent_status(agent_id)
            # assert status_result["status"] == "running"
        
        # 第3步: 正常终止agent
        with patch('psutil.process_iter') as mock_iter:
            mock_process = Mock()
            mock_process.pid = 99999
            mock_process.cmdline.return_value = ["python", "agibot.py", "--agent-id", agent_id]
            mock_process.terminate.return_value = None
            mock_iter.return_value = [mock_process]
            
            terminate_result = multi_agent_tools.terminate_agibot(agent_id=agent_id)
            assert terminate_result["status"] == "success"
            mock_process.terminate.assert_called_once()
        
        # 第4步: 验证agent已终止
        with patch('psutil.process_iter', return_value=[]):
            # 再次尝试终止应该返回"未找到"
            terminate_again = multi_agent_tools.terminate_agibot(agent_id=agent_id)
            assert terminate_again["status"] == "error"
            assert "not found" in terminate_again["message"].lower() 