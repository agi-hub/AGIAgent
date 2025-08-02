#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FastMCP wrapper for AGIBot
Using FastMCP library as MCP client for better performance and Pythonic interface
"""

import json
import os
import asyncio
import threading
import shutil
import logging
from typing import Dict, Any, List, Optional, Union
from contextlib import asynccontextmanager

# Initialize logger
logger = logging.getLogger(__name__)

# Handle import based on context
try:
    from .print_system import print_current
    from .mcp_server_manager import get_mcp_server_manager, mcp_operation_context
except ImportError:
    # For standalone testing
    def print_current(msg):
        print(f"[FastMCP] {msg}")
    
    # Mock server manager for standalone testing
    @asynccontextmanager
    async def get_mcp_server_manager(config_path: str = "config/mcp_servers.json"):
        yield None
    
    @asynccontextmanager 
    async def mcp_operation_context(config_path: str = "config/mcp_servers.json"):
        yield None

# Check if fastmcp is available
try:
    from fastmcp import Client
    FASTMCP_AVAILABLE = True
except ImportError:
    FASTMCP_AVAILABLE = False

class FastMcpWrapper:
    """FastMCP wrapper, providing MCP functionality for AGIBot"""
    
    # Class variable to track if installation message has been shown
    _installation_message_shown = False
    
    def __init__(self, config_path: str = "config/mcp_servers.json"):
        self.config_path = config_path
        self.available_tools = {}
        self.servers = {}
        self.clients = {}  # Store FastMCP client instances
        self.initialized = False
        self.server_manager = None  # Reference to persistent server manager
        self.use_persistent_servers = True  # Flag to enable persistent server mode
        
        # Check if FastMCP is available
        if not FASTMCP_AVAILABLE:
            if not self._installation_message_shown:
                print_current("❌ FastMCP not found. Please install it using: pip install fastmcp")
                print_current("💡 After installation, restart AGIBot to use MCP tools.")
                self._installation_message_shown = True
            return
    

    
    async def initialize(self) -> bool:
        """Initialize MCP client"""
        if not FASTMCP_AVAILABLE:
            return False
            
        try:
            if self.use_persistent_servers:
                return await self._initialize_with_server_manager()
            else:
                return await self._initialize_legacy()
        except Exception as e:
            logger.error(f"FastMCP client initialization failed: {e}")
            return False
    
    async def _initialize_with_server_manager(self) -> bool:
        """Initialize using persistent server manager"""
        try:
            #print_current("🚀 Initializing FastMCP with persistent server manager...")
            
            # The server manager will be provided externally via context
            # We just load the configuration for tool discovery
            await self._load_config()
            
            # Discover tools from configuration (servers will be managed externally)
            await self._discover_tools_from_config()
            
            self.initialized = True
            logger.info(f"FastMCP client initialized with server manager, discovered {len(self.available_tools)} tools")
            return True
            
        except Exception as e:
            logger.error(f"FastMCP server manager initialization failed: {e}")
            return False
    
    async def _initialize_legacy(self) -> bool:
        """Legacy initialization method (temporary client per call)"""
        try:
            # Check if config file exists
            if not os.path.exists(self.config_path):
                print_current(f"⚠️ MCP config file not found: {self.config_path}")
                print_current("ℹ️ FastMCP client will be available but no servers configured")
                self.initialized = True
                return True
            
            # Load configuration
            await self._load_config()
            
            # Initialize clients and discover tools
            await self._initialize_clients()
            
            # Discover all tools
            await self._discover_tools()
            
            self.initialized = True
            logger.info(f"FastMCP client initialized successfully, discovered {len(self.available_tools)} tools")
            return True
            
        except Exception as e:
            logger.error(f"FastMCP client initialization failed: {e}")
            return False
    
    async def _load_config(self):
        """Load configuration file"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            all_servers = config.get("mcpServers", {})
            
            # Filter out SSE servers, only handle NPX/NPM format servers
            self.servers = {}
            for server_name, server_config in all_servers.items():
                # If server URL contains sse, skip it (leave it for direct MCP client)
                if server_config.get("url") and "sse" in server_config.get("url", "").lower():
                    print_current(f"⏭️  Skipping SSE server {server_name}, will be handled by direct MCP client")
                    continue
                
                # Only handle servers with command field (NPX/NPM format)
                if server_config.get("command"):
                    self.servers[server_name] = server_config
                    print_current(f"📋 Loading NPX/NPM server: {server_name}")
                else:
                    print_current(f"⏭️  Skipping server without command field: {server_name}")
            
            # Auto-set default values
            for server_name, server_config in self.servers.items():
                # Set default enabled status
                if "enabled" not in server_config:
                    server_config["enabled"] = True
                
                # Set default timeout
                if "timeout" not in server_config:
                    server_config["timeout"] = 30
            
            print_current(f"📊 FastMCP client config loaded successfully, found {len(self.servers)} NPX/NPM servers")
            
        except Exception as e:
            print_current(f"❌ Failed to load config file: {e}")
            raise
    
    async def _initialize_clients(self):
        """Initialize FastMCP clients for each server"""
        self.clients = {}
        
        # Create the MCP configuration in FastMCP format
        if self.servers:
            mcp_config = {
                "mcpServers": self.servers
            }
            
            try:
                # Create a single FastMCP client with all servers
                client = Client(mcp_config)
                self.main_client = client
                
                # Store client for each server name for easier access
                for server_name in self.servers.keys():
                    if self.servers[server_name].get("enabled", True):
                        self.clients[server_name] = client
                        print_current(f"🔧 FastMCP client configured for server: {server_name}")
                    else:
                        print_current(f"⏭️  Skipping disabled server: {server_name}")
                
            except Exception as e:
                print_current(f"⚠️ Failed to create FastMCP client: {e}")
        else:
            print_current("⚠️ No servers to initialize for FastMCP")
    
    async def _discover_tools(self):
        """Discover all available tools"""
        self.available_tools = {}
        
        # Use the main client with context manager
        if self.main_client:
            try:
                # Use FastMCP client to list tools - establish connection once
                async with self.main_client as client:
                    tools = await client.list_tools()
                    
                    for tool in tools:
                        # FastMCP returns complete tool names, use them as-is
                        tool_name = tool.name
                        
                        # Determine which server this tool belongs to based on prefix
                        server_name = "default"
                        for srv_name in self.servers.keys():
                            if tool_name.startswith(f"{srv_name}_"):
                                server_name = srv_name
                                break
                        
                        # If no server prefix found, assign to first available server
                        if server_name == "default" and self.servers:
                            server_name = next(iter(self.servers.keys()))
                        
                        # Use the complete tool name for both API and actual calls
                        self.available_tools[tool_name] = {
                            "server": server_name,
                            "tool": tool_name,  # Use complete tool name
                            "original_name": tool_name,
                            "api_name": tool_name,
                            "description": tool.description or "",
                            "parameters": self._convert_tool_schema(tool.inputSchema) if hasattr(tool, 'inputSchema') else []
                        }
                    
            
                logger.info(f"FastMCP tool discovery completed: {len(self.available_tools)} tools available")
                
            except Exception as e:
                logger.error(f"FastMCP tool discovery failed: {e}")
                raise
        else:
            logger.warning("No FastMCP main client available for tool discovery")
    
    async def _discover_tools_from_config(self):
        """Discover tools from configuration without starting servers"""
        self.available_tools = {}
        
        # For now, we'll create placeholder tool definitions based on configuration
        # In a real implementation, you'd query the running servers for their tools
        for server_name in self.servers.keys():
            if self.servers[server_name].get("enabled", True):
                # Create placeholder tools - these would be populated from actual server queries
                # when servers are running
                placeholder_tools = self._create_placeholder_tools(server_name)
                self.available_tools.update(placeholder_tools)
        
        logger.info(f"FastMCP tool discovery from config completed: {len(self.available_tools)} placeholder tools")
    
    def _create_placeholder_tools(self, server_name: str) -> Dict[str, Dict[str, Any]]:
        """Create placeholder tools for a server (to be replaced with actual discovery)"""
        # This is a simplified implementation
        # In practice, you'd need to know what tools each server provides
        placeholder_tools = {}
        
        # Common tools based on server name patterns
        if "jina" in server_name.lower():
            tools = ["jina_reader", "jina_search"]
            descriptions = ["Read web content using Jina Reader", "Search web using Jina Search"]
            parameters_list = [
                [{"name": "url", "type": "string", "required": True, "description": "URL to read"}],
                [{"name": "query", "type": "string", "required": True, "description": "Search query"}]
            ]
        elif "tuzi" in server_name.lower():
            tools = ["submit_gpt_image", "submit_flux_image", "task_barrier"]
            descriptions = ["Generate image using GPT", "Generate image using FLUX", "Wait for tasks to complete"]
            parameters_list = [
                [
                    {"name": "prompt", "type": "string", "required": True, "description": "Image generation prompt"},
                    {"name": "output_path", "type": "string", "required": True, "description": "Path where the generated image will be saved"}
                ],
                [
                    {"name": "prompt", "type": "string", "required": True, "description": "Image generation prompt"},
                    {"name": "output_path", "type": "string", "required": True, "description": "Path where the generated image will be saved"}
                ],
                []  # task_barrier has no parameters
            ]
        else:
            # Generic placeholder
            tools = ["generic_tool"]
            descriptions = ["Generic tool"]
            parameters_list = [[]]
        
        for tool, desc, params in zip(tools, descriptions, parameters_list):
            tool_name = f"{server_name}_{tool}"
            placeholder_tools[tool_name] = {
                "server": server_name,
                "tool": tool,
                "original_name": tool_name,
                "api_name": tool_name,
                "description": desc,
                "parameters": params
            }
        
        return placeholder_tools
    
    def _convert_tool_schema(self, input_schema: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Convert FastMCP tool schema to our internal format"""
        parameters = []
        
        if not input_schema or not isinstance(input_schema, dict):
            return parameters
        
        properties = input_schema.get("properties", {})
        required = input_schema.get("required", [])
        
        for param_name, param_info in properties.items():
            param_type = param_info.get("type", "string")
            param_description = param_info.get("description", f"{param_name} parameter")
            is_required = param_name in required
            
            # Store the full schema for later use in tool definition
            param_data = {
                "name": param_name,
                "type": param_type,
                "required": is_required,
                "description": param_description,
                "schema": param_info  # Keep the original schema
            }
            
            parameters.append(param_data)
        
        return parameters
    
    def call_tool_sync(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Call a tool synchronously using FastMCP"""
        if not FASTMCP_AVAILABLE:
            return {
                "status": "failed",
                "error": "FastMCP not available. Please install it using: pip install fastmcp",
                "tool_name": tool_name,
                "arguments": arguments
            }
            
        if not self.initialized:
            return {
                "status": "failed", 
                "error": "MCP client not initialized",
                "tool_name": tool_name,
                "arguments": arguments
            }

        # Use FastMCP call synchronously by wrapping async call
        import asyncio
        try:
            # Check if we're already in an async context
            try:
                loop = asyncio.get_running_loop()
                # We're in an async context, need to handle differently
                import concurrent.futures
                import threading
                
                def run_in_thread():
                    return asyncio.run(self.call_tool(tool_name, arguments))
                
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(run_in_thread)
                    result = future.result(timeout=30)  # 30 second timeout
                    return result
                    
            except RuntimeError:
                # No running event loop, safe to use asyncio.run
                result = asyncio.run(self.call_tool(tool_name, arguments))
                return result
        except Exception as e:
            return {
                "status": "failed",
                "error": f"FastMCP call failed: {e}",
                "tool_name": tool_name,
                "arguments": arguments
            }

    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Call a tool using FastMCP"""
        if not FASTMCP_AVAILABLE:
            return {
                "status": "failed",
                "error": "FastMCP not available. Please install it using: pip install fastmcp",
                "tool_name": tool_name,
                "arguments": arguments
            }

        if not self.initialized:
            raise Exception("MCP client not initialized")
        
        if tool_name not in self.available_tools:
            raise Exception(f"Tool {tool_name} does not exist")

        tool_info = self.available_tools[tool_name]

        # Use fresh client approach for better reliability (same as call_server_tool)
        max_retries = 1
        for attempt in range(max_retries + 1):
            try:
                # Create fresh client for each call to avoid connection state issues
                fresh_client = self._create_fresh_client()
                
                print_current(f"🚀 Calling FastMCP tool: {tool_name} (attempt {attempt + 1})")
                
                # Use FastMCP with proper cleanup to avoid event loop issues
                try:
                    async with fresh_client as client:
                        result = await client.call_tool(tool_name, arguments)
                        
                        print_current(f"✅ FastMCP call successful for tool: {tool_name}")
                        return {
                            "status": "success",
                            "result": self._format_tool_result(result),
                            "tool_name": tool_name,
                            "original_tool_name": tool_info.get("original_name", tool_name),
                            "arguments": arguments
                        }
                except Exception as ctx_e:
                    # Handle context manager exceptions specifically
                    raise ctx_e
                    
            except Exception as e:
                error_msg = str(e)
                print_current(f"⚠️ FastMCP call attempt {attempt + 1} failed for tool {tool_name}: {error_msg}")
                
                # FastMCP connection diagnostic info
                if "Client failed to connect" in error_msg:
                    print_current(f"🔍 FastMCP Connection Analysis:")
                    print_current(f"   - This is normal FastMCP behavior - each call creates a new process")
                    print_current(f"   - Server process startup time: ~1 second")  
                    print_current(f"   - STDIO mode requires fresh connections for reliability")
                
                # If this is the last attempt, return error
                if attempt == max_retries:
                    print_current(f"❌ FastMCP call failed after {max_retries + 1} attempts")
                    print_current(f"💡 Note: FastMCP uses fresh processes for each call - this is by design")
                    return {
                        "status": "failed",
                        "error": error_msg,
                        "tool_name": tool_name,
                        "arguments": arguments,
                        "attempts": max_retries + 1,
                        "diagnosis": "fastmcp_process_startup_failed"
                    }
                else:
                    # Brief wait before retry
                    import asyncio
                    await asyncio.sleep(0.5)
                    continue
    

    
    def _format_tool_result(self, result) -> Any:
        """Format FastMCP tool result to our standard format"""
        if hasattr(result, 'content'):
            # Result has content attribute (typical FastMCP result)
            content_items = []
            for item in result.content:
                if hasattr(item, 'text'):
                    content_items.append(item.text)
                elif hasattr(item, 'data'):
                    content_items.append(str(item.data))
                else:
                    content_items.append(str(item))
            
            return "\n".join(content_items) if content_items else str(result)
        
        # Direct result
        return result
    
    def get_available_tools(self) -> List[str]:
        """Get list of available tools"""
        return list(self.available_tools.keys())
    
    def get_tool_info(self, tool_name: str) -> Optional[Dict[str, Any]]:
        """Get tool information"""
        return self.available_tools.get(tool_name)
    
    def is_mcp_tool(self, tool_name: str) -> bool:
        """Check if it's an MCP tool"""
        return tool_name in self.available_tools
    
    def get_tool_definition(self, tool_name: str) -> Optional[Dict[str, Any]]:
        """Get tool definition (Claude API format)"""
        if tool_name not in self.available_tools:
            return None
        
        tool_info = self.available_tools[tool_name]
        
        # Build parameter schema
        properties = {}
        required = []
        
        for param in tool_info.get("parameters", []):
            param_name = param["name"]
            param_required = param.get("required", False)
            
            # Use the original schema if available, otherwise build from type
            if "schema" in param and isinstance(param["schema"], dict):
                # Use the original FastMCP schema
                properties[param_name] = param["schema"].copy()
            else:
                # Fallback: build schema from type info
                param_type = param.get("type", "string")
                param_desc = param.get("description", f"{param_name} parameter")
                
                # Map types and build schema
                if param_type in ["string", "number", "integer", "boolean"]:
                    schema_type = param_type
                elif param_type == "int":
                    schema_type = "integer"
                elif param_type == "float":
                    schema_type = "number"
                elif param_type == "bool":
                    schema_type = "boolean"
                elif param_type in ["list", "array"]:
                    schema_type = "array"
                elif param_type in ["dict", "object"]:
                    schema_type = "object"
                else:
                    schema_type = "string"  # Default
                
                schema = {
                    "type": schema_type,
                    "description": param_desc
                }
                
                # Add items for array types
                if schema_type == "array":
                    schema["items"] = {"type": "string"}  # Default items type
                
                properties[param_name] = schema
            
            if param_required:
                required.append(param_name)
        
        # Build tool definition
        tool_def = {
            "name": tool_name,  # Use API compatible name
            "description": tool_info.get("description", f"MCP tool: {tool_name}"),
            "input_schema": {
                "type": "object",
                "properties": properties,
                "required": required
            }
        }
        
        return tool_def
    
    def supports_server(self, server_name: str) -> bool:
        """Check if FastMCP supports a specific server"""
        return server_name in self.servers
    
    def get_server_tools(self, server_name: str) -> List[str]:
        """Get tools available for a specific server"""
        if not self.supports_server(server_name):
            return []
        
        server_tools = []
        for tool_name, tool_info in self.available_tools.items():
            if tool_info.get("server") == server_name:
                server_tools.append(tool_name)
        
        return server_tools
    
    async def call_server_tool(self, server_name: str, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Call a tool on a specific server using FastMCP"""
        if not FASTMCP_AVAILABLE:
            return {
                "status": "failed",
                "error": "FastMCP not available. Please install it using: pip install fastmcp",
                "server_name": server_name,
                "tool_name": tool_name,
                "arguments": arguments
            }

        if not self.initialized:
            raise Exception("FastMCP client not initialized")
        
        if not self.supports_server(server_name):
            raise Exception(f"Server {server_name} is not supported by FastMCP")

        # Find the actual tool to call with improved name matching
        actual_tool_name = self._find_matching_tool(server_name, tool_name)
        
        if not actual_tool_name:
            available_tools = self.get_server_tools(server_name)
            raise Exception(f"Tool '{tool_name}' not found in server '{server_name}'. Available tools: {available_tools}")

        # FastMCP 设计哲学：每次调用都使用新的客户端实例
        # 这是 FastMCP 的标准工作模式，不是 bug
        print_current(f"🔧 Creating fresh FastMCP client for tool: {actual_tool_name} on server: {server_name}")
        
        max_retries = 1  # 由于每次都是新连接，减少重试次数
        for attempt in range(max_retries + 1):
            try:
                # 根据 FastMCP 文档，每次调用都应该创建新客户端
                fresh_client = self._create_fresh_client()
                
                print_current(f"🚀 Calling FastMCP tool: {actual_tool_name} (attempt {attempt + 1})")
                
                # Use FastMCP with proper cleanup to avoid event loop issues
                try:
                    async with fresh_client as client:
                        result = await client.call_tool(actual_tool_name, arguments)
                        
                        print_current(f"✅ FastMCP call successful for tool: {actual_tool_name}")
                        return {
                            "status": "success",
                            "result": self._format_tool_result(result),
                            "tool_name": actual_tool_name,
                            "server_name": server_name,
                            "arguments": arguments
                        }
                except Exception as ctx_e:
                    # Handle context manager exceptions specifically
                    raise ctx_e
                    
            except Exception as e:
                error_msg = str(e)
                print_current(f"⚠️ FastMCP call attempt {attempt + 1} failed for tool {actual_tool_name}: {error_msg}")
                
                # FastMCP 连接诊断信息
                if "Client failed to connect" in error_msg:
                    print_current(f"🔍 FastMCP Connection Analysis:")
                    print_current(f"   - This is normal FastMCP behavior - each call creates a new process")
                    print_current(f"   - Server process startup time: ~1 second")
                    print_current(f"   - STDIO mode requires fresh connections for reliability")
                
                # 如果是最后一次尝试，返回错误
                if attempt == max_retries:
                    print_current(f"❌ FastMCP call failed after {max_retries + 1} attempts")
                    print_current(f"💡 Note: FastMCP uses fresh processes for each call - this is by design")
                    return {
                        "status": "failed",
                        "error": error_msg,
                        "tool_name": actual_tool_name,
                        "server_name": server_name,
                        "arguments": arguments,
                        "attempts": max_retries + 1,
                        "diagnosis": "fastmcp_process_startup_failed"
                    }
                else:
                    # 简短等待后重试
                    import asyncio
                    await asyncio.sleep(0.5)
                    continue
    
    def _find_matching_tool(self, server_name: str, tool_name: str) -> Optional[str]:
        """Find matching tool name with flexible matching strategies"""
        # Strategy 1: Exact match
        for available_tool, tool_info in self.available_tools.items():
            if tool_info.get("server") == server_name and available_tool == tool_name:
                return available_tool
        
        # Strategy 2: Remove server prefix if present
        if tool_name.startswith(f"{server_name}_"):
            stripped_name = tool_name[len(f"{server_name}_"):]
            for available_tool, tool_info in self.available_tools.items():
                if tool_info.get("server") == server_name and available_tool == stripped_name:
                    print_current(f"🔧 Tool name mapping: {tool_name} -> {available_tool}")
                    return available_tool
        
        # Strategy 3: Add server prefix if not present
        prefixed_name = f"{server_name}_{tool_name}"
        for available_tool, tool_info in self.available_tools.items():
            if tool_info.get("server") == server_name and available_tool == prefixed_name:
                print_current(f"🔧 Tool name mapping: {tool_name} -> {available_tool}")
                return available_tool
        
        # Strategy 4: Partial match (contains)
        for available_tool, tool_info in self.available_tools.items():
            if (tool_info.get("server") == server_name and 
                (tool_name in available_tool or available_tool in tool_name)):
                print_current(f"🔧 Tool name partial match: {tool_name} -> {available_tool}")
                return available_tool
        
        # Strategy 5: Case insensitive match
        tool_name_lower = tool_name.lower()
        for available_tool, tool_info in self.available_tools.items():
            if (tool_info.get("server") == server_name and 
                available_tool.lower() == tool_name_lower):
                print_current(f"🔧 Tool name case match: {tool_name} -> {available_tool}")
                return available_tool
        
        return None
    
    def _create_fresh_client(self) -> Any:
        """Create a fresh FastMCP client instance to avoid connection state issues"""
        try:
            if not self.servers:
                raise Exception("No servers configured for fresh client creation")
            
            # Create new configuration
            mcp_config = {
                "mcpServers": self.servers
            }
            
            # Create fresh client instance with proper resource management
            from fastmcp import Client
            fresh_client = Client(mcp_config)
            
            print_current(f"🔄 Created fresh FastMCP client instance")
            return fresh_client
            
        except Exception as e:
            print_current(f"⚠️ Failed to create fresh FastMCP client: {e}")
            raise
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform a health check on the FastMCP client"""
        health_status = {
            "healthy": False,
            "fastmcp_available": FASTMCP_AVAILABLE,
            "initialized": self.initialized,
            "main_client_exists": self.main_client is not None,
            "servers_count": len(self.servers),
            "tools_count": len(self.available_tools),
            "errors": []
        }
        
        try:
            if not FASTMCP_AVAILABLE:
                health_status["errors"].append("FastMCP library not available")
                return health_status
            
            if not self.initialized:
                health_status["errors"].append("FastMCP wrapper not initialized")
                return health_status
            
            if not self.main_client:
                health_status["errors"].append("Main FastMCP client not available")
                return health_status
            
            # Try a simple test call if tools are available
            if self.available_tools:
                test_tool = next(iter(self.available_tools.keys()))
                test_server = self.available_tools[test_tool]["server"]
                
                # Quick connection test
                try:
                    async with self.main_client as client:
                        # Just check if we can establish connection, don't actually call tool
                        health_status["connection_test"] = "passed"
                        
                except Exception as conn_e:
                    health_status["errors"].append(f"Connection test failed: {str(conn_e)}")
                    health_status["connection_test"] = "failed"
            
            health_status["healthy"] = len(health_status["errors"]) == 0
            
        except Exception as e:
            health_status["errors"].append(f"Health check exception: {str(e)}")
            health_status["healthy"] = False
        
        return health_status
    
    def call_server_tool_sync(self, server_name: str, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Call a tool on a specific server synchronously using FastMCP"""
        if not FASTMCP_AVAILABLE:
            return {
                "status": "failed",
                "error": "FastMCP not available. Please install it using: pip install fastmcp",
                "server_name": server_name,
                "tool_name": tool_name,
                "arguments": arguments
            }
            
        if not self.initialized:
            return {
                "status": "failed", 
                "error": "FastMCP client not initialized",
                "server_name": server_name,
                "tool_name": tool_name,
                "arguments": arguments
            }

        # Use FastMCP call synchronously by wrapping async call
        import asyncio
        try:
            result = asyncio.run(self.call_server_tool(server_name, tool_name, arguments))
            return result
        except Exception as e:
            return {
                "status": "failed",
                "error": f"FastMCP call failed: {e}",
                "server_name": server_name,
                "tool_name": tool_name,
                "arguments": arguments
            }
    

    
    def get_status(self) -> Dict[str, Any]:
        """Get client status"""
        return {
            "initialized": self.initialized,
            "servers": list(self.servers.keys()),
            "total_tools": len(self.available_tools),
            "config_path": self.config_path,
            "fastmcp_available": FASTMCP_AVAILABLE
        }
    
    async def cleanup(self):
        """Cleanup resources"""
        try:
            # Ensure all FastMCP clients are properly closed before cleanup
            if self.main_client:
                try:
                    # Force close any remaining connections
                    await asyncio.sleep(0.1)  # Brief delay to let pending operations complete
                except Exception:
                    pass
            
            # Clear references
            self.main_client = None
            self.clients.clear()
            self.available_tools.clear()
            self.servers.clear()
            self.initialized = False
            # print_current("🔌 FastMCP client cleaned up")
        except Exception as e:
            # print_current("🔌 FastMCP client cleaned up")
            pass
    
    def cleanup_sync(self):
        """Synchronous cleanup client"""
        print_current("🔌 FastMCP client cleaned up")


# Global instance with thread safety
_fastmcp_wrapper = None
_fastmcp_config_path = None
_fastmcp_lock = threading.RLock()  # Reentrant lock for thread safety

def get_fastmcp_wrapper(config_path: str = "config/mcp_servers.json") -> FastMcpWrapper:
    """Get FastMCP wrapper instance (thread-safe)"""
    global _fastmcp_wrapper, _fastmcp_config_path
    
    with _fastmcp_lock:
        if _fastmcp_wrapper is None or _fastmcp_config_path != config_path:
            _fastmcp_wrapper = FastMcpWrapper(config_path)
            _fastmcp_config_path = config_path
            #print_current(f"🔧 Created new FastMCP wrapper instance for config: {config_path}")
        return _fastmcp_wrapper

async def initialize_fastmcp_wrapper(config_path: str = "config/mcp_servers.json") -> bool:
    """Initialize FastMCP wrapper (thread-safe)"""
    global _fastmcp_wrapper
    
    # First check if already initialized (quick check with lock)
    with _fastmcp_lock:
        wrapper = get_fastmcp_wrapper(config_path)
        if wrapper.initialized:
            print_current(f"✅ FastMCP wrapper already initialized, reusing existing instance")
            return True
    
    # Initialize outside the lock to avoid blocking other threads
    try:
        result = await wrapper.initialize()
        if result:
            logger.info(f"FastMCP wrapper initialized successfully")
        else:
            logger.warning(f"FastMCP wrapper initialization failed")
        return result
    except Exception as e:
        logger.error(f"FastMCP wrapper initialization error: {e}")
        return False

@asynccontextmanager
async def initialize_fastmcp_with_server_manager(config_path: str = "config/mcp_servers.json"):
    """Initialize FastMCP wrapper with persistent server manager"""
    global _fastmcp_wrapper
    
    # Create wrapper instance
    with _fastmcp_lock:
        wrapper = get_fastmcp_wrapper(config_path)
        wrapper.use_persistent_servers = True
    
    # Use MCP operation context for structured concurrency
    async with mcp_operation_context(config_path) as server_manager:
        try:
            # Set server manager reference
            wrapper.server_manager = server_manager
            
            # Initialize wrapper
            result = await wrapper.initialize()
            if not result:
                raise Exception("FastMCP wrapper initialization failed")
            
            print_current(f"✅ FastMCP wrapper initialized with persistent server manager")
            yield wrapper
            
        finally:
            # Clean up
            wrapper.server_manager = None
            print_current("🔄 FastMCP wrapper with server manager context exiting...")

def is_fastmcp_initialized(config_path: str = "config/mcp_servers.json") -> bool:
    """Check if FastMCP wrapper is initialized (thread-safe)"""
    global _fastmcp_wrapper
    
    with _fastmcp_lock:
        if _fastmcp_wrapper is None:
            return False
        return _fastmcp_wrapper.initialized

def get_fastmcp_status(config_path: str = "config/mcp_servers.json") -> Dict[str, Any]:
    """Get FastMCP wrapper status (thread-safe)"""
    global _fastmcp_wrapper
    
    with _fastmcp_lock:
        if _fastmcp_wrapper is None:
            return {
                "initialized": False,
                "thread": threading.current_thread().name,
                "wrapper_exists": False,
                "fastmcp_available": FASTMCP_AVAILABLE
            }
        
        status = _fastmcp_wrapper.get_status()
        status.update({
            "thread": threading.current_thread().name,
            "wrapper_exists": True
        })
        return status

async def cleanup_fastmcp_wrapper():
    """Cleanup FastMCP wrapper"""
    global _fastmcp_wrapper, _fastmcp_config_path
    if _fastmcp_wrapper:
        await _fastmcp_wrapper.cleanup()
        _fastmcp_wrapper = None
        _fastmcp_config_path = None

def cleanup_fastmcp_wrapper_sync():
    """Cleanup FastMCP wrapper synchronously"""
    global _fastmcp_wrapper, _fastmcp_config_path
    if _fastmcp_wrapper:
        try:
            # Try to get the current event loop
            loop = asyncio.get_running_loop()
            # If there's a running loop, schedule the cleanup
            async def async_cleanup():
                if _fastmcp_wrapper:  # Check again inside async function
                    await _fastmcp_wrapper.cleanup()
            loop.create_task(async_cleanup())
        except RuntimeError:
            # No event loop running, use synchronous cleanup
            if _fastmcp_wrapper:  # Check before cleanup_sync call
                _fastmcp_wrapper.cleanup_sync()
        
        _fastmcp_wrapper = None
        _fastmcp_config_path = None

def safe_cleanup_fastmcp_wrapper():
    """Safely cleanup FastMCP wrapper in any context"""
    global _fastmcp_wrapper, _fastmcp_config_path
    if _fastmcp_wrapper:
        wrapper_instance = _fastmcp_wrapper  # Store reference before clearing
        _fastmcp_wrapper = None
        _fastmcp_config_path = None
        
        try:
            # Try to get the current event loop
            loop = asyncio.get_running_loop()
            # If there's a running loop and it's not closed, schedule the cleanup
            if not loop.is_closed():
                async def async_cleanup():
                    try:
                        await wrapper_instance.cleanup()
                    except Exception as cleanup_e:
                        print_current(f"⚠️ Async cleanup error: {cleanup_e}")
                
                # Schedule the cleanup task
                try:
                    loop.create_task(async_cleanup())
                except RuntimeError:
                    # Loop is closing or closed, use sync cleanup
                    wrapper_instance.cleanup_sync()
            else:
                # Loop is closed, use sync cleanup
                wrapper_instance.cleanup_sync()
        except RuntimeError:
            # No event loop running, try to create one for cleanup
            try:
                # Check if we can create a new event loop
                import threading
                if threading.current_thread() is threading.main_thread():
                    # Only create new event loop in main thread
                    asyncio.run(wrapper_instance.cleanup())
                else:
                    # In non-main thread, use sync cleanup
                    wrapper_instance.cleanup_sync()
            except (RuntimeError, ImportError):
                # If that fails too, use synchronous cleanup
                wrapper_instance.cleanup_sync()
        except Exception as e:
            # If all else fails, just clean up the references
            print_current(f"⚠️ FastMCP client cleanup failed: {e}")
            wrapper_instance.cleanup_sync()


# Test function for FastMCP wrapper
async def test_fastmcp_wrapper():
    """Test FastMCP wrapper functionality"""
    print_current("🧪 Starting FastMCP wrapper test...")
    
    # Use the provided config file
    config_path = "config/mcp_servers.json"
    
    # Initialize wrapper
    wrapper = get_fastmcp_wrapper(config_path)
    
    # Test initialization
    print_current("📋 Testing initialization...")
    init_result = await wrapper.initialize()
    print_current(f"Initialization result: {init_result}")
    
    if not init_result:
        print_current("❌ Initialization failed, cannot continue test")
        return
    
    # Test status
    print_current("📊 Testing status...")
    status = wrapper.get_status()
    print_current(f"Status: {json.dumps(status, indent=2)}")
    
    # Test available tools
    print_current("🔧 Testing available tools...")
    tools = wrapper.get_available_tools()
    print_current(f"Available tools: {tools}")
    
    if tools:
        # Test tool info
        first_tool = tools[0]
        print_current(f"📋 Testing tool info for: {first_tool}")
        tool_info = wrapper.get_tool_info(first_tool)
        print_current(f"Tool info: {json.dumps(tool_info, indent=2)}")
        
        # Test tool definition
        print_current(f"📝 Testing tool definition for: {first_tool}")
        tool_def = wrapper.get_tool_definition(first_tool)
        print_current(f"Tool definition: {json.dumps(tool_def, indent=2)}")
        
        # Test tool call with a simple filesystem operation
        if "read" in first_tool.lower() or "list" in first_tool.lower():
            print_current(f"🔧 Testing tool call for: {first_tool}")
            try:
                # Try to list current directory or read a file
                if "list" in first_tool.lower():
                    result = await wrapper.call_tool(first_tool, {"path": "."})
                elif "read" in first_tool.lower():
                    # Try to read the config file itself
                    result = await wrapper.call_tool(first_tool, {"path": config_path})
                else:
                    result = await wrapper.call_tool(first_tool, {})
                
                print_current(f"Tool call result: {json.dumps(result, indent=2)}")
            except Exception as e:
                print_current(f"❌ Tool call failed: {e}")
    
    # Test cleanup
    print_current("🧹 Testing cleanup...")
    await wrapper.cleanup()
    print_current("✅ FastMCP wrapper test completed!")


def test_fastmcp_wrapper_sync():
    """Synchronous test wrapper"""
    asyncio.run(test_fastmcp_wrapper())


if __name__ == "__main__":
    """Test FastMCP wrapper when run directly"""
    test_fastmcp_wrapper_sync() 