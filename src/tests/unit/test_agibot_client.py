#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AGIBotClient核心功能测试
测试AGIBotClient的主要接口和基本功能
"""

import pytest
import os
import tempfile
import shutil
from unittest.mock import patch, Mock
import sys

# 添加src路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from main import AGIBotClient

class TestAGIBotClient:
    """AGIBotClient核心测试类"""
    
    @pytest.fixture
    def temp_workspace(self):
        """创建临时工作空间"""
        temp_dir = tempfile.mkdtemp(prefix="agibot_client_test_")
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def agibot_client(self):
        """创建AGIBotClient实例"""
        return AGIBotClient(
            api_key="test_key",
            model="test_model",
            api_base="https://test-api.example.com",
            debug_mode=True
        )
    
    def test_client_initialization(self):
        """测试客户端初始化"""
        client = AGIBotClient(
            api_key="test_key",
            model="test_model"
        )
        
        assert client.api_key == "test_key"
        assert client.model == "test_model"
        assert client.single_task_mode == True  # 默认值
        assert client.debug_mode == False  # 默认值
    
    def test_client_initialization_missing_params(self):
        """测试缺少必要参数的初始化"""
        with pytest.raises(ValueError, match="api_key is required"):
            AGIBotClient(model="test_model")
        
        with pytest.raises(ValueError, match="model is required"):
            AGIBotClient(api_key="test_key")
    
    def test_chat_basic_success(self, agibot_client, temp_workspace):
        """测试基础聊天功能成功案例"""
        messages = [{"role": "user", "content": "创建一个简单的Hello World程序"}]
        
        # 模拟成功的执行
        with patch.object(agibot_client, '_create_main_instance') as mock_main:
            mock_main_instance = Mock()
            mock_main_instance.run.return_value = True
            mock_main.return_value = mock_main_instance
            
            result = agibot_client.chat(messages=messages, dir=temp_workspace)
            
            assert result["success"] == True
            assert result["message"] == "Task completed successfully"
            assert "output_dir" in result
            assert "execution_time" in result
            assert result["execution_time"] > 0
    
    def test_chat_basic_failure(self, agibot_client, temp_workspace):
        """测试基础聊天功能失败案例"""
        messages = [{"role": "user", "content": "执行一个会失败的任务"}]
        
        # 模拟失败的执行
        with patch.object(agibot_client, '_create_main_instance') as mock_main:
            mock_main_instance = Mock()
            mock_main_instance.run.return_value = False
            mock_main.return_value = mock_main_instance
            
            result = agibot_client.chat(messages=messages, dir=temp_workspace)
            
            assert result["success"] == False
            assert "execution rounds" in result["message"]
    
    def test_chat_invalid_messages(self, agibot_client):
        """测试无效消息格式"""
        invalid_messages_list = [
            [],  # 空列表
            [{"role": "system", "content": "系统消息"}],  # 无用户消息
            [{"role": "user"}],  # 缺少content
            [{"content": "内容"}],  # 缺少role
            "invalid",  # 非列表格式
        ]
        
        for invalid_messages in invalid_messages_list:
            result = agibot_client.chat(messages=invalid_messages)
            assert result["success"] == False
            assert "error" in result["details"]
    
    def test_chat_auto_directory_generation(self, agibot_client):
        """测试自动目录生成"""
        messages = [{"role": "user", "content": "测试自动目录"}]
        
        with patch.object(agibot_client, '_create_main_instance') as mock_main:
            mock_main_instance = Mock()
            mock_main_instance.run.return_value = True
            mock_main.return_value = mock_main_instance
            
            # 不提供dir参数
            result = agibot_client.chat(messages=messages)
            
            assert result["success"] == True
            assert "agibot_output_" in result["output_dir"]
    
    def test_chat_with_custom_loops(self, agibot_client, temp_workspace):
        """测试自定义循环次数"""
        messages = [{"role": "user", "content": "测试循环次数"}]
        
        with patch.object(agibot_client, '_create_main_instance') as mock_main:
            mock_main_instance = Mock()
            mock_main_instance.run.return_value = True
            mock_main.return_value = mock_main_instance
            
            result = agibot_client.chat(
                messages=messages, 
                dir=temp_workspace,
                loops=10
            )
            
            # 验证main.run被调用时传入了正确的loops参数
            mock_main_instance.run.assert_called_once_with(
                user_requirement="测试循环次数",
                loops=10
            )
            assert result["details"]["loops"] == 10
    
    def test_client_configuration_options(self):
        """测试客户端配置选项"""
        # 测试所有配置选项
        client = AGIBotClient(
            api_key="test_key",
            model="test_model",
            api_base="https://custom-api.com",
            debug_mode=True,
            detailed_summary=False,
            single_task_mode=False,
            interactive_mode=True,
            streaming=True,
            MCP_config_file="custom_mcp.json",
            prompts_folder="custom_prompts",
            link_dir="/custom/link"
        )
        
        assert client.api_base == "https://custom-api.com"
        assert client.debug_mode == True
        assert client.detailed_summary == False
        assert client.single_task_mode == False
        assert client.interactive_mode == True
        assert client.streaming == True
        assert client.MCP_config_file == "custom_mcp.json"
        assert client.prompts_folder == "custom_prompts"
        assert client.link_dir == "/custom/link"
    
    def test_chat_exception_handling(self, agibot_client, temp_workspace):
        """测试聊天过程中的异常处理"""
        messages = [{"role": "user", "content": "测试异常处理"}]
        
        # 模拟初始化过程中的异常
        with patch.object(agibot_client, '_create_main_instance', side_effect=Exception("初始化失败")):
            result = agibot_client.chat(messages=messages, dir=temp_workspace)
            
            assert result["success"] == False
            assert "初始化失败" in result["message"]
    
    def test_chat_result_structure(self, agibot_client, temp_workspace):
        """测试聊天结果结构"""
        messages = [{"role": "user", "content": "测试结果结构"}]
        
        with patch.object(agibot_client, '_create_main_instance') as mock_main:
            mock_main_instance = Mock()
            mock_main_instance.run.return_value = True
            mock_main.return_value = mock_main_instance
            
            result = agibot_client.chat(messages=messages, dir=temp_workspace)
            
            # 验证返回结果包含所有必要字段
            required_fields = [
                "success", "message", "output_dir", 
                "workspace_dir", "execution_time", "details"
            ]
            for field in required_fields:
                assert field in result
            
            # 验证details结构
            assert "requirement" in result["details"]
            assert "loops" in result["details"]
            assert "mode" in result["details"]
            assert "model" in result["details"]
    
    def test_multiple_chat_sessions(self, agibot_client):
        """测试多次聊天会话"""
        messages1 = [{"role": "user", "content": "第一个任务"}]
        messages2 = [{"role": "user", "content": "第二个任务"}]
        
        with patch.object(agibot_client, '_create_main_instance') as mock_main:
            mock_main_instance = Mock()
            mock_main_instance.run.return_value = True
            mock_main.return_value = mock_main_instance
            
            # 执行两次聊天
            result1 = agibot_client.chat(messages=messages1)
            result2 = agibot_client.chat(messages=messages2)
            
            assert result1["success"] == True
            assert result2["success"] == True
            assert result1["output_dir"] != result2["output_dir"]  # 不同的输出目录
    
    def test_workspace_isolation(self, agibot_client):
        """测试工作空间隔离"""
        messages = [{"role": "user", "content": "测试工作空间隔离"}]
        
        with patch.object(agibot_client, '_create_main_instance') as mock_main:
            mock_main_instance = Mock()
            mock_main_instance.run.return_value = True
            mock_main.return_value = mock_main_instance
            
            # 使用不同的目录
            result1 = agibot_client.chat(messages=messages, dir="workspace1")
            result2 = agibot_client.chat(messages=messages, dir="workspace2")
            
            assert "workspace1" in result1["output_dir"]
            assert "workspace2" in result2["output_dir"]
            assert result1["output_dir"] != result2["output_dir"] 