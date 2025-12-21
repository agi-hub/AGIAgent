#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MCP (Model Context Protocol) Client Implementation
Supports communication with external MCP servers and tool calls

=== Protocol Adapter System ===

This system supports multiple non-standard SSE protocols through protocol adapters that automatically convert request and response formats.

Custom adapter example:
```python
# Create custom adapter
def custom_request_transformer(tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "action": tool_name,
        "params": parameters
    }

def custom_response_transformer(data: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "content": [
            {
                "type": "text",
                "text": json.dumps(data, ensure_ascii=False)
            }
        ]
    }

custom_adapter = ProtocolAdapter(
    name="custom_api",
    url_pattern="myapi.com",
    request_transformer=custom_request_transformer,
    response_transformer=custom_response_transformer
)

# Register adapter
mcp_client.register_protocol_adapter(custom_adapter)
```

Configuration file example:
```json
{
    "mcpServers": {
        "CustomAPI": {
            "url": "https://myapi.com/v1/search",
            "auth_token": "your_api_key",
            "protocol_adapter": "custom_api"
        }
    }
}
```

=== Protocol Differences Comparison ===

Standard MCP SSE protocol:
- Uses JSON-RPC 2.0 format
- Request format: {"jsonrpc": "2.0", "method": "tools/call", "params": {...}}
- Response format: {"jsonrpc": "2.0", "result": {...}}

Proprietary API protocols (like Baidu AI Search):
- Uses proprietary request format
- Request format: {"query": "search content", "search_type": "news"}
- Response format: {"references": [...]}

Protocol adapters automatically handle these differences to ensure a unified interface experience.
"""

import json
import asyncio
import subprocess
import os
import time
import requests
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
from .print_system import print_current, print_system, print_error, print_debug

# ========================================
# 🚀 延迟导入优化：FastMCP 集成延迟加载
# ========================================
# FastMCP 是重量级框架（~3秒），只在实际使用 MCP 功能时才加载

FASTMCP_INTEGRATION_AVAILABLE = None  # 未初始化状态
_fastmcp_integration_checked = False

def _check_fastmcp_integration():
    """检查 FastMCP 集成是否可用（延迟检查）"""
    global FASTMCP_INTEGRATION_AVAILABLE, _fastmcp_integration_checked
    
    if _fastmcp_integration_checked:
        return FASTMCP_INTEGRATION_AVAILABLE
    
    try:
        # 延迟导入 fastmcp_wrapper（只检查是否可导入，不真正导入）
        import importlib.util
        spec = importlib.util.find_spec('.fastmcp_wrapper', package='src.tools')
        FASTMCP_INTEGRATION_AVAILABLE = spec is not None
    except:
        FASTMCP_INTEGRATION_AVAILABLE = False
    
    _fastmcp_integration_checked = True
    return FASTMCP_INTEGRATION_AVAILABLE

def get_fastmcp_wrapper():
    """获取 FastMCP wrapper（延迟加载）"""
    if _check_fastmcp_integration():
        from .fastmcp_wrapper import get_fastmcp_wrapper as _get_wrapper
        return _get_wrapper()
    return None

def initialize_fastmcp_wrapper(*args, **kwargs):
    """初始化 FastMCP wrapper（延迟加载）"""
    if _check_fastmcp_integration():
        from .fastmcp_wrapper import initialize_fastmcp_wrapper as _init
        return _init(*args, **kwargs)
    return False

def is_fastmcp_initialized():
    """检查 FastMCP 是否已初始化（延迟加载）"""
    if _check_fastmcp_integration():
        from .fastmcp_wrapper import is_fastmcp_initialized as _is_init
        return _is_init()
    return False

# 兼容性：FASTMCP_AVAILABLE 属性
def _get_fastmcp_available():
    """获取 FASTMCP_AVAILABLE 状态（延迟）"""
    if _check_fastmcp_integration():
        from .fastmcp_wrapper import FASTMCP_AVAILABLE as _available
        return _available
    return False

FASTMCP_AVAILABLE = property(lambda self: _get_fastmcp_available())

# Setup logging
logger = logging.getLogger(__name__)

class MCPTransportType(Enum):
    """MCP transport types"""
    STDIO = "stdio"
    HTTP = "http"
    SSE = "sse"

@dataclass
class ProtocolAdapter:
    """Protocol adapter configuration"""
    name: str
    url_pattern: str  # URL matching pattern
    request_transformer: Callable[[str, Dict[str, Any]], Dict[str, Any]]  # Request transformer
    response_transformer: Callable[[Dict[str, Any]], Dict[str, Any]]  # Response transformer
    headers_transformer: Optional[Callable[[str], Dict[str, str]]] = None  # Request headers transformer

@dataclass
class MCPServer:
    """MCP server configuration"""
    name: str
    transport_type: MCPTransportType
    url: Optional[str] = None
    command: Optional[str] = None
    args: List[str] = field(default_factory=list)
    env: Dict[str, str] = field(default_factory=dict)
    auth_token: Optional[str] = None
    capabilities: List[str] = field(default_factory=list)
    timeout: int = 30
    retry_count: int = 3
    enabled: bool = True
    protocol_adapter: Optional[str] = None  # Specify which protocol adapter to use

class MCPClient:
    """MCP Client"""
    def __init__(self, config_path: str = "config/mcp_servers.json", workspace_dir: Optional[str] = None):
        self.config_path = config_path
        self.workspace_dir = workspace_dir
        self.servers: Dict[str, MCPServer] = {}
        self.connections: Dict[str, Any] = {}
        self.tools: Dict[str, Dict[str, Any]] = {}
        self.initialized = False
        self.protocol_adapters: Dict[str, ProtocolAdapter] = {}

        # FastMCP integration
        self.fastmcp_available = FASTMCP_INTEGRATION_AVAILABLE and FASTMCP_AVAILABLE
        self.fastmcp_wrapper = None
        self.fastmcp_initialized = False

        self._init_builtin_adapters()
    
    def _init_builtin_adapters(self):
        """Initialize built-in protocol adapters"""
        
        # Baidu AI Search adapter
        def baidu_request_transformer(tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
            """Baidu AI Search request transformer"""
            query = parameters.get("query", "")
            search_data = {"query": query}
            
            # Add optional parameters
            for key in ["search_type", "language", "num_results"]:
                if key in parameters:
                    search_data[key] = parameters[key]
            
            return search_data
        
        def baidu_response_transformer(data: Dict[str, Any]) -> Dict[str, Any]:
            """Baidu AI Search response transformer"""
            if "references" in data:
                formatted_results = []
                for ref in data["references"]:
                    formatted_results.append({
                        "title": ref.get("title", ""),
                        "content": ref.get("content", ""),
                        "url": ref.get("url", "")
                    })
                
                result_text = f"Found {len(formatted_results)} search results:\n\n"
                for i, r in enumerate(formatted_results, 1):
                    result_text += f"{i}. **{r['title']}**\n{r['content']}\nLink: {r['url']}\n\n"
                
                return {
                    "content": [
                        {
                            "type": "text",
                            "text": result_text
                        }
                    ]
                }
            else:
                return {
                    "content": [
                        {
                            "type": "text", 
                            "text": f"Search completed, but no results found."
                        }
                    ]
                }
        
        def baidu_headers_transformer(api_key: str) -> Dict[str, str]:
            """Baidu AI Search headers transformer"""
            return {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
        
        self.protocol_adapters["baidu_aisearch"] = ProtocolAdapter(
            name="baidu_aisearch",
            url_pattern="appbuilder.baidu.com.*ai_search",
            request_transformer=baidu_request_transformer,
            response_transformer=baidu_response_transformer,
            headers_transformer=baidu_headers_transformer
        )
        
        # Tencent Cloud API adapter example
        def tencent_request_transformer(tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
            """Tencent Cloud API request transformer"""
            return {
                "Action": tool_name,
                "Region": parameters.get("region", "ap-beijing"),
                "Version": "2021-03-01",
                **parameters
            }
        
        def tencent_response_transformer(data: Dict[str, Any]) -> Dict[str, Any]:
            """Tencent Cloud API response transformer"""
            return {
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps(data, ensure_ascii=False, indent=2)
                    }
                ]
            }
        
        self.protocol_adapters["tencent_cloud"] = ProtocolAdapter(
            name="tencent_cloud",
            url_pattern="tencentcloudapi.com",
            request_transformer=tencent_request_transformer,
            response_transformer=tencent_response_transformer
        )
        
        # Elasticsearch adapter example
        def elasticsearch_request_transformer(tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
            """Elasticsearch request transformer"""
            if tool_name == "search":
                return {
                    "query": {
                        "multi_match": {
                            "query": parameters.get("query", ""),
                            "fields": parameters.get("fields", ["*"])
                        }
                    },
                    "size": parameters.get("size", 10)
                }
            return parameters
        
        def elasticsearch_response_transformer(data: Dict[str, Any]) -> Dict[str, Any]:
            """Elasticsearch response transformer"""
            hits = data.get("hits", {}).get("hits", [])
            result_text = f"Found {len(hits)} results:\n\n"
            
            for i, hit in enumerate(hits, 1):
                source = hit.get("_source", {})
                result_text += f"{i}. {json.dumps(source, ensure_ascii=False)}\n\n"
            
            return {
                "content": [
                    {
                        "type": "text",
                        "text": result_text
                    }
                ]
            }
        
        self.protocol_adapters["elasticsearch"] = ProtocolAdapter(
            name="elasticsearch",
            url_pattern="elasticsearch|elastic",
            request_transformer=elasticsearch_request_transformer,
            response_transformer=elasticsearch_response_transformer
        )
    
    def _detect_protocol_adapter(self, server: MCPServer) -> Optional[ProtocolAdapter]:
        """Detect which protocol adapter should be used for the server"""
        # Prioritize adapter specified in configuration
        if server.protocol_adapter and server.protocol_adapter in self.protocol_adapters:
            return self.protocol_adapters[server.protocol_adapter]
        
        # Auto-detect based on URL pattern
        if server.url:
            import re
            for adapter in self.protocol_adapters.values():
                if re.search(adapter.url_pattern, server.url, re.IGNORECASE):
                    return adapter
        
        return None
    
    async def initialize(self) -> bool:
        """Initialize MCP client"""
        try:
            # Initialize FastMCP wrapper if available
            if self.fastmcp_available:
                try:
                    self.fastmcp_wrapper = get_fastmcp_wrapper(self.config_path, workspace_dir=self.workspace_dir)
                    self.fastmcp_initialized = await initialize_fastmcp_wrapper(self.config_path, workspace_dir=self.workspace_dir)
                    if self.fastmcp_initialized:
                        logger.info("FastMCP wrapper initialized successfully")
                    else:
                        logger.warning("FastMCP wrapper initialization failed, falling back to traditional MCP")
                except Exception as e:
                    logger.warning(f"FastMCP wrapper initialization error: {e}, falling back to traditional MCP")
                    self.fastmcp_initialized = False
            
            # Load configuration
            if not self._load_config():
                # print_current("⚠️ MCP configuration file not found or empty")
                return True  # No configuration is not an error
            
            #print_current(f"🔌 MCP client initialization completed")
            self.initialized = True
            return True
            
        except Exception as e:
            logger.error(f"MCP client initialization failed: {e}")
            print_error(f"❌ MCP client initialization failed: {e}")
            return False
    
    def _load_config(self) -> bool:
        """Load configuration file"""
        try:
            if not os.path.exists(self.config_path):
                return False
            
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            # Process mcpServers configuration
            servers_config = {}
            
            if "mcpServers" in config:
                servers_config = config["mcpServers"]
            
            for server_name, server_config in servers_config.items():
                try:
                    # Handle NPX/NPM format servers with FastMCP if available
                    if server_config.get("command") and not server_config.get("url"):
                        # This is a command-based server (NPX/NPM format)
                        if self.fastmcp_initialized and self.fastmcp_wrapper and self.fastmcp_wrapper.supports_server(server_name):
                            # FastMCP can handle this server
                            # Continue processing as STDIO type
                            transport_type = MCPTransportType.STDIO
                        else:
                            # Skip it as it should be handled by cli-mcp wrapper
                            continue
                    
                    # Determine transport type based on configuration
                    elif server_config.get("url"):
                        # Check if URL contains SSE identifier
                        url = server_config.get("url", "").lower()
                        if "sse" in url:
                            transport_type = MCPTransportType.SSE
                            print_current(f"🔌 {server_name} identified as SSE transport type")
                        else:
                            transport_type = MCPTransportType.HTTP
                    elif server_config.get("command"):
                        # This branch should rarely be reached now due to the filter above
                        # But keeping for any edge cases
                        transport_type = MCPTransportType.STDIO
                        print_current(f"⚠️  {server_name} using STDIO transport (unexpected)")
                    else:
                        transport_type = MCPTransportType(server_config.get("transport", "stdio"))
                    
                    # Extract API key
                    api_key = None
                    url = server_config.get("url", "")
                    if url and "api_key=" in url:
                        # Extract API key from URL
                        url_parts = url.split("?", 1)
                        if len(url_parts) > 1:
                            query_params = url_parts[1]
                            for param in query_params.split("&"):
                                if param.startswith("api_key="):
                                    api_key_part = param.split("=", 1)[1]
                                    # Handle Bearer+ prefix
                                    if api_key_part.startswith("Bearer+"):
                                        api_key = api_key_part.replace("Bearer+", "")
                                    else:
                                        api_key = api_key_part
                                    break
                    
                    server = MCPServer(
                        name=server_name,
                        transport_type=transport_type,
                        url=server_config.get("url"),
                        command=server_config.get("command"),
                        args=server_config.get("args", []),
                        env=server_config.get("env", {}),
                        auth_token=api_key or server_config.get("auth_token"),
                        capabilities=server_config.get("capabilities", []),
                        timeout=server_config.get("timeout", 30),
                        retry_count=server_config.get("retry_count", 3),
                        enabled=server_config.get("enabled", True),
                        protocol_adapter=server_config.get("protocol_adapter")
                    )
                    
                    self.servers[server_name] = server
                    
                except Exception as e:
                    logger.error(f"Failed to parse server configuration {server_name}: {e}")
                    continue
            
            if len(self.servers) > 0:
                print_system(f"📋 MCP client loaded {len(self.servers)} servers")
            else:
                print_system(f"📋 MCP client found no applicable servers (NPX/NPM servers handled by cli-mcp wrapper)")
                
            return len(self.servers) >= 0  # Allow zero servers as NPX/NPM servers are handled elsewhere
            
        except Exception as e:
            logger.error(f"Failed to load MCP configuration: {e}")
            return False
    
    def is_mcp_tool(self, tool_name: str) -> bool:
        """Check if it's an MCP tool"""
        # Check if tool name is a configured server name
        return tool_name in self.servers

    def get_available_tools(self) -> List[str]:
        """Get list of available MCP tools"""
        tools = []
        for server_name, server in self.servers.items():
            if server.enabled:
                # Use server's original name directly as tool name
                tools.append(server_name)
        
        return list(set(tools))  # Remove duplicates
    
    def get_tool_definition(self, tool_name: str) -> Dict[str, Any]:
        """Get tool definition"""
        if not self.is_mcp_tool(tool_name):
            return {}
        
        # Return definition based on server name
        server = self.servers.get(tool_name)
        if not server:
            return {}
        
        base_definition = {
            "description": f"Use {tool_name} server for operations",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "tool_name": {
                        "type": "string",
                        "description": "Name of the tool to call"
                    },
                    "parameters": {
                        "type": "object",
                        "description": "Tool parameters"
                    }
                },
                "required": ["tool_name"]
            }
        }
        
        return base_definition
    
    async def call_tool(self, tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Call MCP tool"""
        if not self.initialized:
            return {"status": "failed", "error": "MCP client not initialized"}
        
        if not self.is_mcp_tool(tool_name):
            return {"status": "failed", "error": f"Not a valid MCP tool: {tool_name}"}
        
        try:
            # Call directly using server name
            server = self.servers.get(tool_name)
            if server and server.enabled:
                # Extract actual tool name from parameters, use default if not provided
                actual_tool_name = parameters.get("tool_name", "search")
                actual_parameters = parameters.get("parameters", parameters)
                
                return await self._call_mcp_tool(server, actual_tool_name, actual_parameters)
            else:
                return {"status": "failed", "error": f"Server {tool_name} is not available"}
            
        except Exception as e:
            logger.error(f"Failed to call MCP tool: {e}")
            return {"status": "failed", "error": f"Failed to call MCP tool: {e}"}
    
    async def _call_mcp_tool(self, server: MCPServer, tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Actually call MCP tool"""
        if server.transport_type == MCPTransportType.STDIO:
            return await self._call_stdio_tool(server, tool_name, parameters)
        elif server.transport_type == MCPTransportType.HTTP:
            return await self._call_http_tool(server, tool_name, parameters)
        elif server.transport_type == MCPTransportType.SSE:
            return await self._call_sse_tool(server, tool_name, parameters)
        else:
            return {"status": "failed", "error": f"Unsupported transport type: {server.transport_type.value}"}
    
    async def _call_stdio_tool(self, server: MCPServer, tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Call tool via STDIO"""
        try:
            # Try FastMCP first if available and supports this server
            if (self.fastmcp_initialized and 
                self.fastmcp_wrapper and 
                self.fastmcp_wrapper.supports_server(server.name)):
                
                print_debug(f"🚀 Using FastMCP for server {server.name}, tool {tool_name}")
                
                try:
                    result = await self.fastmcp_wrapper.call_server_tool(server.name, tool_name, parameters)
                    
                    if result.get("status") == "success":
                        # Convert FastMCP result format to standard MCP result format
                        fastmcp_result = result.get("result", "")
                        
                        # Check if result is already in MCP content format
                        if isinstance(fastmcp_result, dict) and "content" in fastmcp_result:
                            return fastmcp_result
                        elif isinstance(fastmcp_result, str):
                            return {
                                "content": [
                                    {
                                        "type": "text",
                                        "text": fastmcp_result
                                    }
                                ]
                            }
                        else:
                            return {
                                "content": [
                                    {
                                        "type": "text", 
                                        "text": str(fastmcp_result)
                                    }
                                ]
                            }
                    else:
                        # FastMCP failed, fall back to traditional method
                        print_debug(f"⚠️ FastMCP failed for {server.name}: {result.get('error', 'Unknown error')}, falling back to subprocess")
                        
                except Exception as e:
                    print_debug(f"⚠️ FastMCP call exception for {server.name}: {str(e)}, falling back to subprocess")
            
            # Traditional subprocess method (fallback or when FastMCP not available)
            print_debug(f"💻 Using subprocess for server {server.name}, tool {tool_name}")
            
            # Check if command exists
            if not server.command:
                return {"status": "failed", "error": "MCP server command not configured"}
            
            # Prepare environment variables
            env = os.environ.copy()
            env.update(server.env)
            
            # Build command
            cmd = [server.command] + server.args
            
            # Prepare request
            request = {
                "jsonrpc": "2.0",
                "id": int(time.time() * 1000),
                "method": "tools/call",
                "params": {
                    "name": tool_name,
                    "arguments": parameters
                }
            }
            
            # Debug logging
            print_debug(f"🔧 STDIO MCP call: {tool_name}")
            print_debug(f"💻 Command: {' '.join(cmd)}")
            print_debug(f"📤 Request: {json.dumps(request, indent=2)}")
            
            # Start process
            process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
                text=True,
                encoding='utf-8',
                errors='ignore'
            )
            
            # Send request and get response
            request_json = json.dumps(request)
            stdout, stderr = process.communicate(input=request_json, timeout=server.timeout)
            
            # Debug logging for response
            print_debug(f"📥 STDOUT length: {len(stdout)} chars")
            print_debug(f"STDERR length: {len(stderr)} chars")
            
            if stderr.strip():
                print_debug(f"STDERR content: {stderr.strip()}")
            
            if process.returncode != 0:
                error_msg = f"MCP tool execution failed (exit code: {process.returncode})"
                if stderr.strip():
                    error_msg += f", stderr: {stderr.strip()}"
                if stdout.strip():
                    error_msg += f", stdout: {stdout.strip()}"
                return {"status": "failed", "error": error_msg}
            
            # Debug logging for stdout content
            if len(stdout) > 500:
                print_debug(f"📥 STDOUT (first 200 chars): {stdout[:200]}...")
                print_debug(f"📥 STDOUT (last 200 chars): ...{stdout[-200:]}")
            else:
                print_debug(f"📥 STDOUT content: {stdout}")
            
            # Parse response
            try:
                if not stdout.strip():
                    return {"status": "failed", "error": "Empty response from MCP server"}
                
                response = json.loads(stdout)
                if "result" in response:
                    print_debug(f"✅ Successfully parsed MCP response with result")
                    return response["result"]
                elif "error" in response:
                    print_debug(f"❌ MCP server returned error: {response['error']}")
                    return {"status": "failed", "error": response["error"]}
                else:
                    print_debug(f"⚠️ Unknown response format: {list(response.keys())}")
                    return {"status": "failed", "error": f"Unknown response format, keys: {list(response.keys())}"}
            except json.JSONDecodeError as e:
                error_msg = f"Cannot parse JSON response. Error: {str(e)}"
                if len(stdout) <= 1000:
                    error_msg += f"\nFull response: {repr(stdout)}"
                else:
                    error_msg += f"\nResponse preview (first 500 chars): {repr(stdout[:500])}"
                    error_msg += f"\nResponse preview (last 500 chars): {repr(stdout[-500:])}"
                
                # Also include stderr if available
                if stderr.strip():
                    error_msg += f"\nSTDERR: {stderr.strip()}"
                
                print_debug(f"JSON parsing failed: {error_msg}")
                return {"status": "failed", "error": error_msg}
            
        except subprocess.TimeoutExpired:
            error_msg = f"MCP tool call timeout (>{server.timeout}s)"
            print_debug(f"⏰ {error_msg}")
            return {"status": "failed", "error": error_msg}
        except Exception as e:
            error_msg = f"MCP tool call exception: {str(e)}"
            print_debug(f"💥 {error_msg}")
            return {"status": "failed", "error": error_msg}
    
    async def _call_http_tool(self, server: MCPServer, tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Call tool via HTTP"""
        try:
            # Check if URL exists
            if not server.url:
                return {"status": "failed", "error": "MCP server URL not configured"}
            
            # Standard MCP request
            request_data = {
                "jsonrpc": "2.0",
                "id": int(time.time() * 1000),
                "method": "tools/call",
                "params": {
                    "name": tool_name,
                    "arguments": parameters
                }
            }
            
            # Prepare headers
            headers = {
                "Content-Type": "application/json",
                "Accept": "application/json"
            }
            
            # Add authentication header
            if server.auth_token:
                headers["Authorization"] = f"Bearer {server.auth_token}"
            
            # Send HTTP request
            loop = asyncio.get_running_loop()
            response = await loop.run_in_executor(
                None, 
                self._make_http_request,
                server.url,
                request_data,
                headers,
                server.timeout
            )
            
            if response.get("status") == "error":
                return {"status": "failed", "error": response.get("message", "HTTP request failed")}
            
            # Parse response
            response_data = response.get("data", {})
            
            if "result" in response_data:
                return response_data["result"]
            elif "error" in response_data:
                return {"status": "failed", "error": response_data["error"]}
            else:
                return {"status": "failed", "error": "Unknown response format"}
            
        except Exception as e:
            return {"status": "failed", "error": str(e)}
    
    def _make_http_request(self, url: str, data: Dict[str, Any], headers: Dict[str, str], timeout: int) -> Dict[str, Any]:
        """Synchronous HTTP request (run in executor)"""
        try:
            response = requests.post(
                url,
                json=data,
                headers=headers,
                timeout=timeout
            )
            
            if response.status_code == 200:
                return {"status": "success", "data": response.json()}
            else:
                return {"status": "failed", "message": f"HTTP status code: {response.status_code}"}
                
        except requests.exceptions.Timeout:
            return {"status": "failed", "message": "HTTP request timeout"}
        except requests.exceptions.RequestException as e:
            return {"status": "failed", "message": f"HTTP request exception: {str(e)}"}
        except Exception as e:
            return {"status": "failed", "message": f"Unknown error: {str(e)}"}
    
    async def _call_sse_tool(self, server: MCPServer, tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Call MCP tool via SSE (supports protocol adapters)"""
        try:
            print_debug(f"🔌 Calling SSE MCP tool: {tool_name}")
            
            # Check if URL exists
            if not server.url:
                return {"status": "failed", "error": "MCP server URL not configured"}
            
            # Detect if protocol adapter is needed
            adapter = self._detect_protocol_adapter(server)
            
            if adapter:
                print_current(f"🔧 Using protocol adapter: {adapter.name}")
                return await self._call_sse_with_adapter(server, tool_name, parameters, adapter)
            else:
                print_debug(f"📡 Using standard MCP SSE protocol")
                return await self._call_generic_sse(server, tool_name, parameters)
            
        except Exception as e:
            logger.error(f"SSE tool call exception: {e}")
            return {"status": "failed", "error": str(e)}
    
    async def _call_sse_with_adapter(self, server: MCPServer, tool_name: str, parameters: Dict[str, Any], adapter: ProtocolAdapter) -> Dict[str, Any]:
        """Call SSE tool using protocol adapter"""
        try:
            # Check if URL exists
            if not server.url:
                return {"status": "failed", "error": "MCP server URL not configured for adapter"}
            
            # Parse URL, remove query parameters to get base URL
            url_parts = server.url.split("?")
            base_url = url_parts[0]
            
            # Build API endpoint URL
            if "/mcp/sse" in base_url:
                api_base_url = base_url.replace("/mcp/sse", "")
            else:
                api_base_url = base_url
            
            print_current(f"🌐 Adapter API endpoint: {api_base_url}")
            
            # Use adapter to transform request data
            request_data = adapter.request_transformer(tool_name, parameters)
            print_current(f"📊 Transformed request data: {request_data}")
            
            # Use adapter to transform headers
            if adapter.headers_transformer and server.auth_token:
                headers = adapter.headers_transformer(server.auth_token)
            else:
                headers = {
                    "Content-Type": "application/json"
                }
                if server.auth_token:
                    headers["Authorization"] = f"Bearer {server.auth_token}"
            
            # Send request
            loop = asyncio.get_running_loop()
            response = await loop.run_in_executor(
                None,
                self._make_adapter_request,
                api_base_url,
                request_data,
                headers,
                server.timeout
            )
            
            if response.get("status") == "error":
                return {"status": "failed", "error": response.get("message", "Adapter request failed")}
            
            # Use adapter to transform response data
            raw_data = response.get("data", {})
            return adapter.response_transformer(raw_data)
            
        except Exception as e:
            logger.error(f"Adapter SSE call exception: {e}")
            return {"status": "failed", "error": str(e)}
    
    def _make_adapter_request(self, url: str, data: Dict[str, Any], headers: Dict[str, str], timeout: int) -> Dict[str, Any]:
        """Send adapter request"""
        try:
            response = requests.post(
                url,
                json=data,
                headers=headers,
                timeout=timeout
            )
            
            if response.status_code == 200:
                return {"status": "success", "data": response.json()}
            else:
                return {"status": "failed", "message": f"HTTP status code: {response.status_code}, response: {response.text[:200]}"}
                
        except requests.exceptions.Timeout:
            return {"status": "failed", "message": "Adapter request timeout"}
        except requests.exceptions.RequestException as e:
            return {"status": "failed", "message": f"Adapter request exception: {str(e)}"}
        except Exception as e:
            return {"status": "failed", "message": f"Unknown error: {str(e)}"}
    
    async def _call_generic_sse(self, server: MCPServer, tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Generic SSE protocol handling"""
        try:
            # Check if URL exists
            if not server.url:
                return {"status": "failed", "error": "MCP server URL not configured for generic SSE"}
            
            # Parse URL, remove query parameters to get base URL
            url_parts = server.url.split("?")
            base_url = url_parts[0]
            
            # Prepare standard MCP request
            request_data = {
                "jsonrpc": "2.0",
                "id": int(time.time() * 1000),
                "method": "tools/call",
                "params": {
                    "name": tool_name,
                    "arguments": parameters
                }
            }
            
            # Prepare headers
            headers = {
                "Content-Type": "application/json",
                "Accept": "text/event-stream",
                "Cache-Control": "no-cache"
            }
            
            # Add authentication header
            if server.auth_token:
                headers["Authorization"] = f"Bearer {server.auth_token}"
            
            print_current(f"🌐 Generic SSE endpoint: {base_url}")
            print_current(f"📝 Request data: {request_data}")
            
            # Send SSE request
            loop = asyncio.get_running_loop()
            response = await loop.run_in_executor(
                None,
                self._make_sse_request,
                base_url,
                request_data,
                headers,
                server.timeout
            )
            
            if response.get("status") == "error":
                return {"status": "failed", "error": response.get("message", "SSE request failed")}
            
            # Parse SSE response
            data = response.get("data", {})
            
            if "result" in data:
                return data["result"]
            elif "content" in data:
                return data
            else:
                return {"content": [{"type": "text", "text": str(data)}]}
                
        except Exception as e:
            logger.error(f"Generic SSE call exception: {e}")
            return {"status": "failed", "error": str(e)}
    
    def _make_sse_request(self, url: str, data: Dict[str, Any], headers: Dict[str, str], timeout: int) -> Dict[str, Any]:
        """Send SSE request (generic implementation)"""
        try:
            # For SSE, we send POST request then handle streaming response
            response = requests.post(
                url,
                json=data,
                headers=headers,
                timeout=timeout,
                stream=True
            )
            
            if response.status_code == 200:
                # Handle SSE streaming response
                content = ""
                for line in response.iter_lines(decode_unicode=True):
                    if line:
                        # Parse SSE event
                        if line.startswith("data: "):
                            event_data = line[6:]  # Remove "data: " prefix
                            try:
                                # Try to parse JSON
                                parsed_data = json.loads(event_data)
                                if "result" in parsed_data:
                                    return {"status": "success", "data": parsed_data}
                                elif "error" in parsed_data:
                                    return {"status": "failed", "message": parsed_data["error"]}
                                else:
                                    content += str(parsed_data)
                            except json.JSONDecodeError:
                                # If not JSON, treat as plain text
                                content += event_data + "\n"
                
                # Return collected content
                return {
                    "status": "success", 
                    "data": {
                        "content": [
                            {
                                "type": "text",
                                "text": content.strip()
                            }
                        ]
                    }
                }
            else:
                return {"status": "failed", "message": f"HTTP status code: {response.status_code}, response: {response.text[:200]}"}
                
        except requests.exceptions.Timeout:
            return {"status": "failed", "message": "SSE request timeout"}
        except requests.exceptions.RequestException as e:
            return {"status": "failed", "message": f"SSE request exception: {str(e)}"}
        except Exception as e:
            return {"status": "failed", "message": f"Unknown error: {str(e)}"}
    
    async def cleanup(self):
        """Cleanup MCP client"""
        try:
            # Cleanup FastMCP wrapper if initialized
            if self.fastmcp_wrapper and self.fastmcp_initialized:
                try:
                    await self.fastmcp_wrapper.cleanup()
                    print_current("🧹 FastMCP wrapper cleaned up")
                except Exception as e:
                    print_current(f"⚠️ FastMCP wrapper cleanup error: {e}")
                finally:
                    self.fastmcp_wrapper = None
                    self.fastmcp_initialized = False
            
            # Close all connections
            for connection in self.connections.values():
                try:
                    if hasattr(connection, 'close'):
                        await connection.close()
                except:
                    pass
            
            self.connections.clear()
            self.tools.clear()
            self.initialized = False
            # print_current("🔌 MCP client cleaned up")
            
        except Exception as e:
            logger.error(f"MCP client cleanup error: {e}")
    
    def cleanup_sync(self):
        """Synchronous cleanup method"""
        try:
            # Cleanup FastMCP wrapper if initialized
            if self.fastmcp_wrapper and self.fastmcp_initialized:
                try:
                    self.fastmcp_wrapper.cleanup_sync()
                    print_current("🧹 FastMCP wrapper cleaned up synchronously")
                except Exception as e:
                    print_current(f"⚠️ FastMCP wrapper sync cleanup error: {e}")
                finally:
                    self.fastmcp_wrapper = None
                    self.fastmcp_initialized = False
            
            # Synchronous cleanup operations
            self.connections.clear()
            self.tools.clear()
            self.initialized = False
            # print_current("🔌 MCP client cleaned up synchronously")
        except Exception as e:
            logger.error(f"MCP client sync cleanup error: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get client status"""
        status = {
            "initialized": self.initialized,
            "servers": {
                name: {
                    "enabled": server.enabled,
                    "transport": server.transport_type.value
                }
                for name, server in self.servers.items()
            },
            "total_servers": len(self.servers),
            "fastmcp": {
                "available": self.fastmcp_available,
                "initialized": self.fastmcp_initialized,
                "wrapper": self.fastmcp_wrapper is not None
            }
        }
        
        # Add FastMCP specific status if available
        if self.fastmcp_wrapper and self.fastmcp_initialized:
            try:
                fastmcp_status = self.fastmcp_wrapper.get_status()
                status["fastmcp"]["status"] = fastmcp_status
            except Exception as e:
                status["fastmcp"]["status_error"] = str(e)
        
        return status

    def register_protocol_adapter(self, adapter: ProtocolAdapter):
        """Register custom protocol adapter"""
        self.protocol_adapters[adapter.name] = adapter
        print_current(f"✅ Registered protocol adapter: {adapter.name}")
    
    def list_protocol_adapters(self) -> List[str]:
        """List all available protocol adapters"""
        return list(self.protocol_adapters.keys())
    
    def get_protocol_adapter(self, name: str) -> Optional[ProtocolAdapter]:
        """Get protocol adapter by name"""
        return self.protocol_adapters.get(name)

# Create global MCP client instance
_global_mcp_client: Optional[MCPClient] = None

def get_mcp_client() -> MCPClient:
    """Get global MCP client instance"""
    global _global_mcp_client
    if _global_mcp_client is None:
        _global_mcp_client = MCPClient()
    return _global_mcp_client

async def initialize_mcp_client() -> bool:
    """Initialize global MCP client"""
    client = get_mcp_client()
    return await client.initialize()

async def cleanup_mcp_client():
    """Clean up global MCP client"""
    global _global_mcp_client
    if _global_mcp_client:
        await _global_mcp_client.cleanup()
        _global_mcp_client = None

def safe_cleanup_mcp_client():
    """Safely clean up MCP client in any context"""
    global _global_mcp_client
    if _global_mcp_client:
        try:
            # Try to get current running event loop
            loop = asyncio.get_running_loop()
            # If there's a running event loop, schedule cleanup task
            loop.create_task(_global_mcp_client.cleanup())
        except RuntimeError:
            # No running event loop, try to create a new one to run cleanup
            try:
                asyncio.run(_global_mcp_client.cleanup())
            except RuntimeError:
                # If creating new event loop also fails, use synchronous cleanup
                try:
                    _global_mcp_client.connections.clear()
                    _global_mcp_client.tools.clear()
                    _global_mcp_client.initialized = False
                    # print_current("🔌 MCP client cleaned up synchronously")
                except Exception as sync_e:
                    print_debug(f"⚠️ MCP client synchronous cleanup failed: {sync_e}")
        except Exception as e:
            # If all methods fail, at least clear references
            print_debug(f"⚠️ MCP client cleanup failed: {e}")
            try:
                _global_mcp_client.connections.clear()
                _global_mcp_client.tools.clear()
                _global_mcp_client.initialized = False
            except Exception as cleanup_e:
                print_debug(f"⚠️ MCP client reference cleanup failed: {cleanup_e}")
        
        _global_mcp_client = None

# Test and example code
if __name__ == "__main__":
    """Test protocol adapter system"""
    
    # Create custom adapter example
    def demo_request_transformer(tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Demo request transformer"""
        return {
            "action": tool_name,
            "data": parameters,
            "timestamp": int(time.time())
        }
    
    def demo_response_transformer(data: Dict[str, Any]) -> Dict[str, Any]:
        """Demo response transformer"""
        return {
            "content": [
                {
                    "type": "text",
                    "text": f"Demo response: {json.dumps(data, ensure_ascii=False, indent=2)}"
                }
            ]
        }
    
    # Create demo adapter
    demo_adapter = ProtocolAdapter(
        name="demo_api",
        url_pattern="demo.api.com",
        request_transformer=demo_request_transformer,
        response_transformer=demo_response_transformer
    )
    
    # Initialize client
    client = MCPClient("config/mcp_servers.json")
    
    # Register custom adapter
    client.register_protocol_adapter(demo_adapter)
    
    # List all adapters
    print("Available protocol adapters:")
    for adapter_name in client.list_protocol_adapters():
        print(f"  - {adapter_name}")
    
    # Async test
    async def test_adapters():
        await client.initialize()
        
        # Test Baidu AI Search
        if "AISearch" in client.servers:
            print("\n🔍 Testing Baidu AI Search:")
            result = await client.call_tool("AISearch", {"query": "artificial intelligence"})
            print(f"Result: {result}")
        
        # Test list tools
        print("\n🔧 Available tools:")
        tools = client.get_available_tools()
        for tool_name in tools:
            tool_info = client.get_tool_definition(tool_name)
            print(f"  - {tool_name}: {tool_info.get('description', 'No description')}")
    
    # Run test
    print("\nStarting protocol adapter system test...")
    asyncio.run(test_adapters()) 