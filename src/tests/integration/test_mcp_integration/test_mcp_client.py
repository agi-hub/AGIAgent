#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MCP(Model Context Protocol)集成测试
测试MCP客户端的集成和通信功能
"""

import pytest
import os
import json
import tempfile
import shutil
from unittest.mock import patch, Mock, MagicMock
import sys

# 添加src路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))

try:
    from tools.mcp_client import MCPClient
    from tools.cli_mcp_wrapper import get_cli_mcp_wrapper
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False

@pytest.mark.integration
@pytest.mark.skipif(not MCP_AVAILABLE, reason="MCP client not available")
class TestMCPIntegration:
    """MCP集成测试类"""
    
    @pytest.fixture
    def temp_workspace(self):
        """创建临时工作空间"""
        temp_dir = tempfile.mkdtemp(prefix="mcp_test_")
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def mock_mcp_config(self, temp_workspace):
        """创建模拟MCP配置"""
        config = {
            "mcpServers": {
                "filesystem": {
                    "command": "npx",
                    "args": ["-y", "@modelcontextprotocol/server-filesystem", temp_workspace],
                    "env": {}
                },
                "git": {
                    "command": "npx", 
                    "args": ["-y", "@modelcontextprotocol/server-git", "--repository", temp_workspace],
                    "env": {}
                },
                "memory": {
                    "command": "npx",
                    "args": ["-y", "@modelcontextprotocol/server-memory"],
                    "env": {}
                }
            }
        }
        
        config_file = os.path.join(temp_workspace, "mcp_servers.json")
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        return config_file
    
    @pytest.fixture
    def mcp_client(self, mock_mcp_config):
        """创建MCP客户端"""
        return MCPClient(config_file=mock_mcp_config)
    
    def test_mcp_client_initialization(self, mcp_client):
        """测试MCP客户端初始化"""
        assert mcp_client is not None
        assert hasattr(mcp_client, 'list_tools')
        assert hasattr(mcp_client, 'call_tool')
        assert hasattr(mcp_client, 'list_resources')
    
    def test_mcp_server_connection(self, mcp_client):
        """测试MCP服务器连接"""
        # 模拟服务器连接
        with patch.object(mcp_client, '_connect_to_server') as mock_connect:
            mock_connect.return_value = True
            
            result = mcp_client.connect_servers()
            
            assert result["status"] == "success"
            assert "connected_servers" in result
    
    def test_mcp_tool_listing(self, mcp_client):
        """测试MCP工具列表"""
        # 模拟工具列表响应
        mock_tools = [
            {
                "name": "read_file",
                "description": "Read contents of a file",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Path to file"}
                    },
                    "required": ["path"]
                }
            },
            {
                "name": "write_file", 
                "description": "Write contents to a file",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Path to file"},
                        "content": {"type": "string", "description": "File content"}
                    },
                    "required": ["path", "content"]
                }
            }
        ]
        
        with patch.object(mcp_client, '_send_request') as mock_request:
            mock_request.return_value = {"tools": mock_tools}
            
            tools = mcp_client.list_tools()
            
            assert len(tools) == 2
            assert tools[0]["name"] == "read_file"
            assert tools[1]["name"] == "write_file"
    
    def test_mcp_tool_execution(self, mcp_client):
        """测试MCP工具执行"""
        # 模拟文件读取工具调用
        mock_response = {
            "content": [
                {
                    "type": "text",
                    "text": "This is the content of the file"
                }
            ]
        }
        
        with patch.object(mcp_client, '_send_request') as mock_request:
            mock_request.return_value = mock_response
            
            result = mcp_client.call_tool(
                tool_name="read_file",
                arguments={"path": "/test/file.txt"}
            )
            
            assert result["status"] == "success"
            assert "content" in result
            assert result["content"][0]["text"] == "This is the content of the file"
    
    def test_mcp_resource_listing(self, mcp_client):
        """测试MCP资源列表"""
        mock_resources = [
            {
                "uri": "file:///test/project/src/main.py",
                "name": "main.py",
                "description": "Main application file",
                "mimeType": "text/python"
            },
            {
                "uri": "file:///test/project/README.md",
                "name": "README.md", 
                "description": "Project documentation",
                "mimeType": "text/markdown"
            }
        ]
        
        with patch.object(mcp_client, '_send_request') as mock_request:
            mock_request.return_value = {"resources": mock_resources}
            
            resources = mcp_client.list_resources()
            
            assert len(resources) == 2
            assert resources[0]["name"] == "main.py"
            assert resources[1]["name"] == "README.md"
    
    def test_mcp_resource_reading(self, mcp_client):
        """测试MCP资源读取"""
        mock_content = {
            "contents": [
                {
                    "uri": "file:///test/file.py",
                    "mimeType": "text/python",
                    "text": "print('Hello, World!')"
                }
            ]
        }
        
        with patch.object(mcp_client, '_send_request') as mock_request:
            mock_request.return_value = mock_content
            
            result = mcp_client.read_resource("file:///test/file.py")
            
            assert result["status"] == "success"
            assert result["content"]["text"] == "print('Hello, World!')"
            assert result["content"]["mimeType"] == "text/python"
    
    def test_mcp_error_handling(self, mcp_client):
        """测试MCP错误处理"""
        # 模拟服务器错误
        with patch.object(mcp_client, '_send_request') as mock_request:
            mock_request.side_effect = Exception("MCP server connection failed")
            
            result = mcp_client.call_tool(
                tool_name="read_file",
                arguments={"path": "/nonexistent/file.txt"}
            )
            
            assert result["status"] == "error"
            assert "connection failed" in result["message"].lower()
    
    def test_mcp_multiple_servers(self, temp_workspace):
        """测试多个MCP服务器"""
        # 创建包含多个服务器的配置
        multi_server_config = {
            "mcpServers": {
                "filesystem": {
                    "command": "npx",
                    "args": ["-y", "@modelcontextprotocol/server-filesystem", temp_workspace]
                },
                "git": {
                    "command": "npx",
                    "args": ["-y", "@modelcontextprotocol/server-git", "--repository", temp_workspace]
                },
                "brave_search": {
                    "command": "npx",
                    "args": ["-y", "@modelcontextprotocol/server-brave-search"],
                    "env": {"BRAVE_API_KEY": "test_key"}
                }
            }
        }
        
        config_file = os.path.join(temp_workspace, "multi_mcp_config.json")
        with open(config_file, 'w') as f:
            json.dump(multi_server_config, f)
        
        client = MCPClient(config_file=config_file)
        
        # 模拟多服务器连接
        with patch.object(client, '_connect_to_server') as mock_connect:
            mock_connect.return_value = True
            
            result = client.connect_servers()
            
            assert result["status"] == "success"
            assert len(result["connected_servers"]) == 3
    
    def test_cli_mcp_wrapper(self, mock_mcp_config):
        """测试CLI MCP包装器"""
        wrapper = get_cli_mcp_wrapper(mock_mcp_config)
        
        assert wrapper is not None
        
        # 模拟CLI工具调用
        with patch.object(wrapper, 'call_tool') as mock_call:
            mock_call.return_value = {
                "status": "success",
                "result": "Tool executed successfully"
            }
            
            result = wrapper.call_tool("filesystem", "read_file", {"path": "test.txt"})
            
            assert result["status"] == "success"
            mock_call.assert_called_once()
    
    def test_mcp_tool_schema_validation(self, mcp_client):
        """测试MCP工具模式验证"""
        # 定义工具模式
        tool_schema = {
            "name": "calculate",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "operation": {"type": "string", "enum": ["add", "subtract", "multiply", "divide"]},
                    "a": {"type": "number"},
                    "b": {"type": "number"}
                },
                "required": ["operation", "a", "b"]
            }
        }
        
        # 测试有效参数
        valid_args = {"operation": "add", "a": 5, "b": 3}
        validation_result = mcp_client.validate_tool_arguments("calculate", valid_args, tool_schema)
        assert validation_result["valid"] == True
        
        # 测试无效参数
        invalid_args = {"operation": "invalid", "a": "not_a_number"}
        validation_result = mcp_client.validate_tool_arguments("calculate", invalid_args, tool_schema)
        assert validation_result["valid"] == False
        assert len(validation_result["errors"]) > 0
    
    def test_mcp_session_management(self, mcp_client):
        """测试MCP会话管理"""
        # 开始新会话
        session_result = mcp_client.start_session()
        assert session_result["status"] == "success"
        session_id = session_result["session_id"]
        
        # 在会话中执行操作
        with patch.object(mcp_client, '_send_request') as mock_request:
            mock_request.return_value = {"result": "success"}
            
            result = mcp_client.call_tool_in_session(
                session_id=session_id,
                tool_name="test_tool",
                arguments={"param": "value"}
            )
            
            assert result["status"] == "success"
        
        # 结束会话
        end_result = mcp_client.end_session(session_id)
        assert end_result["status"] == "success"
    
    def test_mcp_concurrent_requests(self, mcp_client):
        """测试MCP并发请求"""
        import threading
        import time
        
        results = []
        errors = []
        
        def worker(worker_id):
            try:
                with patch.object(mcp_client, '_send_request') as mock_request:
                    mock_request.return_value = {"result": f"Worker {worker_id} result"}
                    
                    result = mcp_client.call_tool(
                        tool_name="test_tool",
                        arguments={"worker_id": worker_id}
                    )
                    results.append((worker_id, result))
                    
            except Exception as e:
                errors.append((worker_id, str(e)))
        
        # 启动并发请求
        threads = []
        for i in range(5):
            thread = threading.Thread(target=worker, args=(i,))
            threads.append(thread)
            thread.start()
        
        # 等待完成
        for thread in threads:
            thread.join()
        
        # 验证并发请求结果
        assert len(errors) == 0, f"并发请求出现错误: {errors}"
        assert len(results) == 5
    
    def test_mcp_timeout_handling(self, mcp_client):
        """测试MCP超时处理"""
        # 模拟超时
        with patch.object(mcp_client, '_send_request') as mock_request:
            mock_request.side_effect = TimeoutError("Request timeout")
            
            result = mcp_client.call_tool(
                tool_name="slow_tool",
                arguments={},
                timeout=1.0
            )
            
            assert result["status"] == "error"
            assert "timeout" in result["message"].lower()
    
    def test_mcp_reconnection(self, mcp_client):
        """测试MCP重连机制"""
        # 模拟连接丢失和重连
        connection_attempts = []
        
        def mock_connect(server_name):
            connection_attempts.append(server_name)
            if len(connection_attempts) < 3:  # 前两次失败
                raise Exception("Connection failed")
            return True  # 第三次成功
        
        with patch.object(mcp_client, '_connect_to_server', side_effect=mock_connect):
            result = mcp_client.connect_with_retry("filesystem", max_retries=3)
            
            assert result["status"] == "success"
            assert len(connection_attempts) == 3  # 重试了3次
    
    def test_mcp_capability_negotiation(self, mcp_client):
        """测试MCP能力协商"""
        # 模拟服务器能力
        server_capabilities = {
            "experimental": {},
            "logging": {"level": "info"},
            "prompts": {"listChanged": True},
            "resources": {"subscribe": True, "listChanged": True},
            "tools": {"listChanged": True}
        }
        
        with patch.object(mcp_client, '_negotiate_capabilities') as mock_negotiate:
            mock_negotiate.return_value = server_capabilities
            
            capabilities = mcp_client.get_server_capabilities("filesystem")
            
            assert "tools" in capabilities
            assert "resources" in capabilities
            assert capabilities["logging"]["level"] == "info"
    
    def test_mcp_streaming_responses(self, mcp_client):
        """测试MCP流式响应"""
        # 模拟流式响应
        def mock_stream_response():
            chunks = [
                {"type": "text", "text": "Processing..."},
                {"type": "text", "text": "Still processing..."},
                {"type": "text", "text": "Complete!"}
            ]
            for chunk in chunks:
                yield chunk
        
        with patch.object(mcp_client, '_stream_request', return_value=mock_stream_response()):
            responses = list(mcp_client.call_tool_streaming(
                tool_name="streaming_tool",
                arguments={}
            ))
            
            assert len(responses) == 3
            assert responses[0]["text"] == "Processing..."
            assert responses[-1]["text"] == "Complete!"
    
    def test_mcp_configuration_validation(self, temp_workspace):
        """测试MCP配置验证"""
        # 创建无效配置
        invalid_config = {
            "mcpServers": {
                "invalid_server": {
                    # 缺少必需的command字段
                    "args": ["--test"],
                    "env": {}
                }
            }
        }
        
        config_file = os.path.join(temp_workspace, "invalid_config.json")
        with open(config_file, 'w') as f:
            json.dump(invalid_config, f)
        
        # 验证配置应该失败
        try:
            client = MCPClient(config_file=config_file)
            validation_result = client.validate_configuration()
            assert validation_result["valid"] == False
            assert len(validation_result["errors"]) > 0
        except Exception as e:
            # 配置验证失败抛出异常也是合理的
            assert "command" in str(e).lower() or "invalid" in str(e).lower()