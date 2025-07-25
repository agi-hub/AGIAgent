#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
代理通信集成测试
测试agent之间的消息路由和通信机制
"""

import pytest
import os
import tempfile
import shutil
import json
import time
import threading
from unittest.mock import patch, Mock
import sys

# 添加src路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))

from tools.message_system import MessageSystem, MessageRouter, MessageType
from tools.multiagents import MultiAgentTools

@pytest.mark.integration
class TestAgentCommunication:
    """代理通信集成测试类"""
    
    @pytest.fixture
    def temp_workspace(self):
        """创建临时工作空间"""
        temp_dir = tempfile.mkdtemp(prefix="agent_comm_test_")
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def message_system(self, temp_workspace):
        """创建消息系统"""
        mailbox_dir = os.path.join(temp_workspace, "mailboxes")
        os.makedirs(mailbox_dir, exist_ok=True)
        return MessageSystem(mailbox_dir=mailbox_dir)
    
    @pytest.fixture
    def multi_agent_tools(self, temp_workspace):
        """创建多代理工具"""
        return MultiAgentTools(workspace_root=temp_workspace)
    
    def test_agent_registration_and_discovery(self, message_system):
        """测试代理注册和发现"""
        # 注册多个代理
        agents = ["data_analyst", "web_developer", "report_generator", "project_manager"]
        
        for agent_id in agents:
            result = message_system.register_agent(agent_id)
            assert result["status"] == "success"
            assert agent_id in result["message"]
        
        # 测试代理发现
        active_agents = message_system.get_active_agents()
        assert len(active_agents) == len(agents)
        
        for agent_id in agents:
            assert agent_id in active_agents
    
    def test_point_to_point_messaging(self, message_system):
        """测试点对点消息传递"""
        # 注册发送方和接收方
        sender_id = "sender_agent"
        receiver_id = "receiver_agent"
        
        message_system.register_agent(sender_id)
        message_system.register_agent(receiver_id)
        
        # 发送消息
        message_content = "Hello from sender agent"
        send_result = message_system.send_message(
            sender_id=sender_id,
            recipient_id=receiver_id,
            content=message_content,
            message_type=MessageType.COLLABORATION.value
        )
        
        assert send_result["status"] == "success"
        assert "message_id" in send_result
        
        # 接收消息
        messages = message_system.get_messages(receiver_id)
        assert len(messages) == 1
        
        received_message = messages[0]
        assert received_message["sender_id"] == sender_id
        assert received_message["content"] == message_content
        assert received_message["message_type"] == MessageType.COLLABORATION.value
        assert "timestamp" in received_message
        assert "message_id" in received_message
    
    def test_broadcast_messaging(self, message_system):
        """测试广播消息"""
        # 注册多个代理
        agents = ["agent_1", "agent_2", "agent_3", "agent_4"]
        manager_id = "manager"
        
        message_system.register_agent(manager_id)
        for agent_id in agents:
            message_system.register_agent(agent_id)
        
        # 发送广播消息
        broadcast_content = "All agents: Project deadline is tomorrow!"
        broadcast_result = message_system.broadcast_message(
            sender_id=manager_id,
            content=broadcast_content,
            message_type=MessageType.BROADCAST.value
        )
        
        assert broadcast_result["status"] == "success"
        assert broadcast_result["recipients_count"] == len(agents)
        
        # 验证所有代理都收到了广播消息
        for agent_id in agents:
            messages = message_system.get_messages(agent_id)
            assert len(messages) == 1
            
            broadcast_msg = messages[0]
            assert broadcast_msg["sender_id"] == manager_id
            assert broadcast_msg["content"] == broadcast_content
            assert broadcast_msg["message_type"] == MessageType.BROADCAST.value
    
    def test_message_routing_with_priorities(self, message_system):
        """测试带优先级的消息路由"""
        sender_id = "urgent_sender"
        receiver_id = "busy_receiver"
        
        message_system.register_agent(sender_id)
        message_system.register_agent(receiver_id)
        
        # 发送不同优先级的消息
        messages_to_send = [
            {"content": "Low priority message", "priority": "low"},
            {"content": "High priority message", "priority": "high"},
            {"content": "Normal priority message", "priority": "normal"},
            {"content": "Critical priority message", "priority": "critical"}
        ]
        
        for msg in messages_to_send:
            message_system.send_message(
                sender_id=sender_id,
                recipient_id=receiver_id,
                content=msg["content"],
                message_type=MessageType.TASK_REQUEST.value,
                priority=msg["priority"]
            )
        
        # 接收消息并验证优先级排序
        received_messages = message_system.get_messages(receiver_id, sort_by_priority=True)
        assert len(received_messages) == 4
        
        # 验证优先级排序（critical > high > normal > low）
        priorities = [msg.get("priority", "normal") for msg in received_messages]
        expected_order = ["critical", "high", "normal", "low"]
        assert priorities == expected_order
    
    def test_message_acknowledgment(self, message_system):
        """测试消息确认机制"""
        sender_id = "sender"
        receiver_id = "receiver"
        
        message_system.register_agent(sender_id)
        message_system.register_agent(receiver_id)
        
        # 发送需要确认的消息
        send_result = message_system.send_message(
            sender_id=sender_id,
            recipient_id=receiver_id,
            content="Please acknowledge this message",
            message_type=MessageType.TASK_REQUEST.value,
            require_ack=True
        )
        
        message_id = send_result["message_id"]
        
        # 模拟接收方确认消息
        ack_result = message_system.acknowledge_message(
            agent_id=receiver_id,
            message_id=message_id
        )
        
        assert ack_result["status"] == "success"
        
        # 验证发送方收到确认
        sender_notifications = message_system.get_notifications(sender_id)
        ack_notifications = [n for n in sender_notifications if n["type"] == "acknowledgment"]
        assert len(ack_notifications) == 1
        assert ack_notifications[0]["message_id"] == message_id
    
    def test_concurrent_messaging(self, message_system):
        """测试并发消息处理"""
        # 注册多个代理
        num_agents = 10
        agents = [f"agent_{i}" for i in range(num_agents)]
        
        for agent_id in agents:
            message_system.register_agent(agent_id)
        
        results = []
        errors = []
        
        def message_worker(sender_idx):
            """并发消息发送工作函数"""
            try:
                sender_id = f"agent_{sender_idx}"
                
                # 每个代理向其他代理发送消息
                for receiver_idx in range(num_agents):
                    if receiver_idx != sender_idx:
                        receiver_id = f"agent_{receiver_idx}"
                        
                        result = message_system.send_message(
                            sender_id=sender_id,
                            recipient_id=receiver_id,
                            content=f"Message from {sender_id} to {receiver_id}",
                            message_type=MessageType.COLLABORATION.value
                        )
                        results.append((sender_idx, receiver_idx, result))
                        
            except Exception as e:
                errors.append((sender_idx, str(e)))
        
        # 启动并发线程
        threads = []
        for i in range(num_agents):
            thread = threading.Thread(target=message_worker, args=(i,))
            threads.append(thread)
            thread.start()
        
        # 等待所有线程完成
        for thread in threads:
            thread.join()
        
        # 验证并发操作结果
        assert len(errors) == 0, f"并发消息发送出现错误: {errors}"
        
        expected_messages = num_agents * (num_agents - 1)  # 每个代理向其他代理发送消息
        assert len(results) == expected_messages
        
        # 验证所有消息都成功发送
        successful_sends = sum(1 for _, _, result in results if result["status"] == "success")
        assert successful_sends == expected_messages
    
    def test_message_filtering_and_routing(self, message_system):
        """测试消息过滤和路由"""
        # 设置消息过滤规则
        receiver_id = "filtered_receiver"
        message_system.register_agent(receiver_id)
        
        # 设置过滤规则：只接收特定类型的消息
        filter_rules = {
            "allowed_types": [MessageType.TASK_REQUEST.value, MessageType.SYSTEM.value],
            "blocked_senders": ["spam_agent"],
            "keywords_filter": ["urgent", "important"]
        }
        
        message_system.set_message_filter(receiver_id, filter_rules)
        
        # 发送各种类型的消息
        test_messages = [
            {"sender": "normal_agent", "type": MessageType.TASK_REQUEST.value, "content": "Urgent task request", "should_receive": True},
            {"sender": "normal_agent", "type": MessageType.COLLABORATION.value, "content": "Collaboration message", "should_receive": False},
            {"sender": "spam_agent", "type": MessageType.TASK_REQUEST.value, "content": "Urgent request", "should_receive": False},
            {"sender": "system", "type": MessageType.SYSTEM.value, "content": "System notification", "should_receive": True},
            {"sender": "normal_agent", "type": MessageType.TASK_REQUEST.value, "content": "Regular request", "should_receive": False},  # 没有关键词
        ]
        
        for msg in test_messages:
            message_system.register_agent(msg["sender"])
            message_system.send_message(
                sender_id=msg["sender"],
                recipient_id=receiver_id,
                content=msg["content"],
                message_type=msg["type"]
            )
        
        # 验证过滤结果
        received_messages = message_system.get_messages(receiver_id)
        expected_count = sum(1 for msg in test_messages if msg["should_receive"])
        assert len(received_messages) == expected_count
    
    def test_agent_status_updates(self, message_system, multi_agent_tools):
        """测试代理状态更新"""
        agent_id = "status_test_agent"
        manager_id = "manager"
        
        message_system.register_agent(agent_id)
        message_system.register_agent(manager_id)
        
        # 模拟代理状态更新
        status_updates = [
            {"status": "started", "progress": 0, "message": "Task started"},
            {"status": "in_progress", "progress": 25, "message": "25% completed"},
            {"status": "in_progress", "progress": 50, "message": "50% completed"},
            {"status": "in_progress", "progress": 75, "message": "75% completed"},
            {"status": "completed", "progress": 100, "message": "Task completed successfully"}
        ]
        
        for update in status_updates:
            result = multi_agent_tools.send_status_update_to_manager(
                agent_id=agent_id,
                status=update["status"],
                progress=update["progress"],
                message=update["message"]
            )
            assert result["status"] == "success"
            time.sleep(0.1)  # 短暂延迟确保时间戳不同
        
        # 验证管理器收到状态更新
        manager_messages = message_system.get_messages(manager_id)
        status_messages = [msg for msg in manager_messages if msg["message_type"] == MessageType.STATUS_UPDATE.value]
        
        assert len(status_messages) == len(status_updates)
        
        # 验证状态更新内容
        for i, status_msg in enumerate(status_messages):
            expected_update = status_updates[i]
            message_data = json.loads(status_msg["content"])
            
            assert message_data["status"] == expected_update["status"]
            assert message_data["progress"] == expected_update["progress"]
            assert message_data["message"] == expected_update["message"]
    
    def test_message_persistence_and_recovery(self, temp_workspace):
        """测试消息持久化和恢复"""
        mailbox_dir = os.path.join(temp_workspace, "mailboxes")
        
        # 创建第一个消息系统实例
        message_system1 = MessageSystem(mailbox_dir=mailbox_dir)
        
        # 注册代理并发送消息
        agent_id = "persistent_agent"
        message_system1.register_agent(agent_id)
        
        message_system1.send_message(
            sender_id="sender",
            recipient_id=agent_id,
            content="Persistent message",
            message_type=MessageType.TASK_REQUEST.value
        )
        
        # 获取消息ID
        messages = message_system1.get_messages(agent_id)
        assert len(messages) == 1
        original_message = messages[0]
        
        # 销毁第一个实例
        del message_system1
        
        # 创建新的消息系统实例（模拟重启）
        message_system2 = MessageSystem(mailbox_dir=mailbox_dir)
        
        # 验证消息持久化
        recovered_messages = message_system2.get_messages(agent_id)
        assert len(recovered_messages) == 1
        
        recovered_message = recovered_messages[0]
        assert recovered_message["content"] == original_message["content"]
        assert recovered_message["sender_id"] == original_message["sender_id"]
        assert recovered_message["message_type"] == original_message["message_type"]
    
    def test_message_expiration_and_cleanup(self, message_system):
        """测试消息过期和清理"""
        agent_id = "expiry_test_agent"
        message_system.register_agent(agent_id)
        
        # 发送带有过期时间的消息
        current_time = time.time()
        
        # 立即过期的消息
        message_system.send_message(
            sender_id="sender",
            recipient_id=agent_id,
            content="Expired message",
            message_type=MessageType.TASK_REQUEST.value,
            expires_at=current_time - 1  # 已过期
        )
        
        # 未过期的消息
        message_system.send_message(
            sender_id="sender",
            recipient_id=agent_id,
            content="Valid message",
            message_type=MessageType.TASK_REQUEST.value,
            expires_at=current_time + 3600  # 1小时后过期
        )
        
        # 触发过期消息清理
        message_system.cleanup_expired_messages()
        
        # 验证只有未过期的消息被保留
        messages = message_system.get_messages(agent_id)
        assert len(messages) == 1
        assert messages[0]["content"] == "Valid message"
    
    def test_message_routing_performance(self, message_system):
        """测试消息路由性能"""
        # 注册大量代理
        num_agents = 100
        agents = [f"perf_agent_{i}" for i in range(num_agents)]
        
        start_time = time.time()
        
        # 批量注册代理
        for agent_id in agents:
            message_system.register_agent(agent_id)
        
        registration_time = time.time() - start_time
        
        # 发送大量消息
        num_messages = 1000
        start_time = time.time()
        
        for i in range(num_messages):
            sender_idx = i % num_agents
            receiver_idx = (i + 1) % num_agents
            
            message_system.send_message(
                sender_id=agents[sender_idx],
                recipient_id=agents[receiver_idx],
                content=f"Performance test message {i}",
                message_type=MessageType.COLLABORATION.value
            )
        
        messaging_time = time.time() - start_time
        
        # 性能验证
        print(f"注册{num_agents}个代理耗时: {registration_time:.3f}秒")
        print(f"发送{num_messages}条消息耗时: {messaging_time:.3f}秒")
        print(f"消息吞吐量: {num_messages/messaging_time:.1f} 消息/秒")
        
        # 性能断言
        assert registration_time < 5.0, f"代理注册时间过长: {registration_time}秒"
        assert messaging_time < 10.0, f"消息发送时间过长: {messaging_time}秒"
        assert (num_messages/messaging_time) > 50, f"消息吞吐量过低: {num_messages/messaging_time:.1f} 消息/秒"