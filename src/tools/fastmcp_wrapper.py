#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FastMCP wrapper for AGIBot
Using FastMCP library as MCP client with persistent server management
"""

import json
import os
import asyncio
import threading
import logging
import warnings
import atexit
from typing import Dict, Any, List, Optional
from contextlib import asynccontextmanager

# Initialize logger
logger = logging.getLogger(__name__)

# Suppress asyncio BaseSubprocessTransport warnings
warnings.filterwarnings("ignore", message=".*Event loop is closed.*")
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*coroutine.*was never awaited.*")
warnings.filterwarnings("ignore", category=ResourceWarning, message=".*subprocess.*")
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*BaseSubprocessTransport.*")

# Suppress specific asyncio transport cleanup warnings
import sys
if sys.version_info >= (3, 8):
    # For Python 3.8+, suppress specific transport cleanup warnings
    import asyncio.base_subprocess
    original_del = asyncio.base_subprocess.BaseSubprocessTransport.__del__
    
    def safe_del(self):
        """Safe cleanup for subprocess transport"""
        try:
            if hasattr(self, '_loop') and self._loop and not self._loop.is_closed():
                original_del(self)
            # If loop is closed, just ignore the cleanup silently
        except (RuntimeError, AttributeError):
            # Silently ignore cleanup errors when event loop is closed
            pass
    
    asyncio.base_subprocess.BaseSubprocessTransport.__del__ = safe_del

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
    """FastMCP wrapper with persistent server management"""
    
    # Class variable to track if installation message has been shown
    _installation_message_shown = False
    
    def __init__(self, config_path: str = "config/mcp_servers.json"):
        self.config_path = config_path
        self.available_tools = {}
        self.servers = {}
        self.initialized = False
        self.server_manager = None  # Reference to persistent server manager
        
        # Check if FastMCP is available
        if not FASTMCP_AVAILABLE:
            if not self._installation_message_shown:
                print_current("❌ FastMCP not found. Please install it using: pip install fastmcp")
                print_current("💡 After installation, restart AGIBot to use MCP tools.")
                self._installation_message_shown = True
            return
    
    async def initialize(self) -> bool:
        """Initialize MCP client with persistent server manager"""
        if not FASTMCP_AVAILABLE:
            return False
            
        try:
            # Load configuration for tool discovery
            await self._load_config()
            
            # Discover tools from running servers
            await self._discover_tools_from_servers()
            
            self.initialized = True
            logger.info(f"FastMCP client initialized, discovered {len(self.available_tools)} tools")
            return True
            
        except Exception as e:
            logger.error(f"FastMCP client initialization failed: {e}")
            return False
    
    async def _load_config(self):
        """Load configuration file"""
        try:
            if not os.path.exists(self.config_path):
                print_current(f"⚠️ MCP config file not found: {self.config_path}")
                return
            
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            all_servers = config.get("mcpServers", {})
            
            # Filter servers - only handle NPX/NPM format servers with commands
            self.servers = {}
            for server_name, server_config in all_servers.items():
                # Skip SSE servers
                if server_config.get("url") and "sse" in server_config.get("url", "").lower():
                    print_current(f"⏭️  Skipping SSE server {server_name}")
                    continue
                
                # Only handle servers with command field
                if server_config.get("command"):
                    self.servers[server_name] = server_config
                    print_current(f"📋 Loading server: {server_name}")
                else:
                    print_current(f"⏭️  Skipping server without command: {server_name}")
            
            # Set default values
            for server_name, server_config in self.servers.items():
                if "enabled" not in server_config:
                    server_config["enabled"] = True
                if "timeout" not in server_config:
                    server_config["timeout"] = 30
            
            print_current(f"📊 Loaded configuration for {len(self.servers)} servers")
            
        except Exception as e:
            print_current(f"❌ Failed to load config file: {e}")
            raise
    
    async def _discover_tools_from_servers(self):
        """Discover tools from all configured servers"""
        self.available_tools = {}
        
        if not self.server_manager:
            print_current("⚠️ No server manager available for tool discovery")
            return
        
        # Try to discover tools from each enabled server
        for server_name in self.servers.keys():
            if self.servers[server_name].get("enabled", True):
                try:
                    discovered_tools = await self._discover_tools_from_server(server_name)
                    if discovered_tools:
                        self.available_tools.update(discovered_tools)
                        print_current(f"✅ Discovered {len(discovered_tools)} tools from {server_name}")
                    else:
                        print_current(f"⚠️ No tools discovered from {server_name}")
                except Exception as e:
                    print_current(f"❌ Failed to discover tools from {server_name}: {e}")
        
        logger.info(f"Tool discovery completed: {len(self.available_tools)} tools total")
    
    async def _discover_tools_standalone(self):
        """Discover tools from all configured servers without server manager (standalone mode)"""
        self.available_tools = {}
        
        print_current("🔍 Starting standalone tool discovery for FastMCP servers")
        
        # Try to discover tools from each enabled server
        for server_name in self.servers.keys():
            if self.servers[server_name].get("enabled", True):
                try:
                    discovered_tools = await self._discover_tools_from_server_standalone(server_name)
                    if discovered_tools:
                        self.available_tools.update(discovered_tools)
                        print_current(f"✅ Discovered {len(discovered_tools)} tools from {server_name}")
                    else:
                        print_current(f"⚠️ No tools discovered from {server_name}")
                except Exception as e:
                    print_current(f"❌ Failed to discover tools from {server_name}: {e}")
        
        logger.info(f"Standalone tool discovery completed: {len(self.available_tools)} tools total")
    
    async def _discover_tools_from_server(self, server_name: str) -> Dict[str, Dict[str, Any]]:
        """Discover tools from a specific server using server manager"""
        try:
            # Check if server is ready
            if not await self.server_manager.is_server_ready(server_name):
                print_current(f"⚠️ Server {server_name} is not ready")
                return {}
            
            return await self._discover_tools_from_server_standalone(server_name)
                
        except Exception as e:
            print_current(f"⚠️ Failed to discover tools from {server_name}: {e}")
            return {}
    
    async def _discover_tools_from_server_standalone(self, server_name: str) -> Dict[str, Dict[str, Any]]:
        """Discover tools from a specific server without server manager (standalone mode)"""
        try:
            server_config = self.servers[server_name]
            command = server_config.get("command")
            args = server_config.get("args", [])
            
            if not command:
                return {}
            
            print_current(f"🔍 Discovering tools from {server_name}")
            
            # Create temporary FastMCP client to query the server
            from fastmcp import Client
            from fastmcp.mcp_config import MCPConfig
            
            # Create MCP configuration
            mcp_config = MCPConfig(
                mcpServers={
                    server_name: {
                        "command": command,
                        "args": args,
                        "transport": "stdio"
                    }
                }
            )
            
            # Query tools using temporary client with timeout
            import sys
            import io
            from contextlib import redirect_stderr
            
            stderr_buffer = io.StringIO()
            
            try:
                # 创建一个更安静的环境来隐藏FastMCP logo
                import os
                import tempfile
                
                # 保存原始的stderr
                original_stderr = os.dup(2)
                
                # 创建临时文件来重定向stderr
                with tempfile.NamedTemporaryFile(mode='w', delete=True) as temp_file:
                    # 重定向stderr到临时文件
                    os.dup2(temp_file.fileno(), 2)
                    
                    try:
                        # Add timeout to prevent hanging (Python 3.10 compatible)
                        async with Client(mcp_config) as client:
                            # Use asyncio.wait_for for Python 3.10 compatibility
                            tools = await asyncio.wait_for(client.list_tools(), timeout=10)
                    finally:
                        # 恢复原始的stderr
                        os.dup2(original_stderr, 2)
                        os.close(original_stderr)
                
                discovered_tools = {}
                for tool in tools:
                    tool_name = f"{server_name}_{tool.name}"
                    
                    # Convert tool schema to our format
                    parameters = self._convert_tool_schema(tool.inputSchema) if hasattr(tool, 'inputSchema') else []
                    
                    discovered_tools[tool_name] = {
                        "server": server_name,
                        "tool": tool.name,
                        "original_name": tool.name,
                        "api_name": tool_name,
                        "description": tool.description or f"Tool from {server_name}",
                        "parameters": parameters
                    }
                
                return discovered_tools
            except asyncio.TimeoutError:
                print_current(f"⚠️ Tool discovery timeout for {server_name}")
                return {}
            except Exception as e:
                print_current(f"⚠️ Tool discovery error for {server_name}: {e}")
                return {}
                
        except Exception as e:
            print_current(f"⚠️ Failed to discover tools from {server_name}: {e}")
            return {}
    
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
            
            param_data = {
                "name": param_name,
                "type": param_type,
                "required": is_required,
                "description": param_description,
                "schema": param_info  # Keep the original schema
            }
            
            parameters.append(param_data)
        
        return parameters
    
    async def _call_tool_standalone(self, server_name: str, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Call a tool in standalone mode without server manager"""
        try:
            server_config = self.servers[server_name]
            command = server_config.get("command")
            args = server_config.get("args", [])
            
            if not command:
                return {"status": "failed", "error": f"No command configured for server {server_name}"}
            
            # Create temporary FastMCP client to call the tool
            from fastmcp import Client
            from fastmcp.mcp_config import MCPConfig
            
            # Create MCP configuration
            mcp_config = MCPConfig(
                mcpServers={
                    server_name: {
                        "command": command,
                        "args": args,
                        "transport": "stdio"
                    }
                }
            )
            
            # Call tool using temporary client with stderr redirection
            import os
            import tempfile
            
            # 保存原始的stderr
            original_stderr = os.dup(2)
            
            try:
                # 创建临时文件来重定向stderr
                with tempfile.NamedTemporaryFile(mode='w', delete=True) as temp_file:
                    # 重定向stderr到临时文件
                    os.dup2(temp_file.fileno(), 2)
                    
                    try:
                        async with Client(mcp_config) as client:
                            # Call the specific tool
                            tool_result = await asyncio.wait_for(
                                client.call_tool(tool_name, arguments), 
                                timeout=30
                            )
                            
                            return {
                                "status": "success",
                                "result": tool_result
                            }
                    finally:
                        # 恢复原始的stderr
                        os.dup2(original_stderr, 2)
                        os.close(original_stderr)
                        
            except asyncio.TimeoutError:
                return {"status": "failed", "error": f"Tool call timeout for {tool_name}"}
            except Exception as e:
                return {"status": "failed", "error": f"Tool call error: {e}"}
                
        except Exception as e:
            return {"status": "failed", "error": f"Failed to call tool {tool_name}: {e}"}
    
    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Call a tool using the persistent server manager"""
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
        server_name = tool_info["server"]
        original_tool_name = tool_info["original_name"]

        try:
            if self.server_manager:
                # Use persistent server manager if available
                if not await self.server_manager.is_server_ready(server_name):
                    raise Exception(f"Server {server_name} is not ready")

                print_current(f"🚀 Calling tool: {tool_name} on persistent server: {server_name}")
                result = await self.server_manager.call_server_tool(server_name, original_tool_name, arguments)
            else:
                # Fallback to standalone mode without server manager
                print_current(f"🚀 Calling tool: {tool_name} in standalone mode")
                result = await self._call_tool_standalone(server_name, original_tool_name, arguments)
            
            if result.get("status") == "success":
                print_current(f"✅ Persistent server call successful: {tool_name}")
                return {
                    "status": "success",
                    "result": self._format_tool_result(result.get("result")),
                    "tool_name": tool_name,
                    "original_tool_name": original_tool_name,
                    "arguments": arguments
                }
            else:
                raise Exception(result.get("error", "Unknown error"))
                    
        except Exception as e:
            error_msg = str(e)
            print_current(f"❌ Tool call failed for {tool_name}: {error_msg}")
            return {
                "status": "failed",
                "error": error_msg,
                "tool_name": tool_name,
                "arguments": arguments
            }
    
    async def call_server_tool(self, server_name: str, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Call a tool on a specific server (backward compatibility method)"""
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
                "error": "MCP client not initialized",
                "server_name": server_name,
                "tool_name": tool_name,
                "arguments": arguments
            }

        # Find the full tool name (usually server_name_tool_name)
        full_tool_name = None
        
        # Strategy 1: Try exact tool name first
        if tool_name in self.available_tools:
            tool_info = self.available_tools[tool_name]
            if tool_info.get("server") == server_name:
                full_tool_name = tool_name
        
        # Strategy 2: Try with server prefix
        if not full_tool_name:
            prefixed_name = f"{server_name}_{tool_name}"
            if prefixed_name in self.available_tools:
                full_tool_name = prefixed_name
        
        # Strategy 3: Search for tool in server's available tools
        if not full_tool_name:
            server_tools = self.get_server_tools(server_name)
            for available_tool in server_tools:
                tool_info = self.available_tools[available_tool]
                if tool_info.get("original_name") == tool_name:
                    full_tool_name = available_tool
                    break
        
        if not full_tool_name:
            return {
                "status": "failed",
                "error": f"Tool '{tool_name}' not found in server '{server_name}'. Available tools: {self.get_server_tools(server_name)}",
                "server_name": server_name,
                "tool_name": tool_name,
                "arguments": arguments
            }
        
        # Call the tool using the full tool name
        result = await self.call_tool(full_tool_name, arguments)
        
        # Add server name to result for compatibility
        if isinstance(result, dict):
            result["server_name"] = server_name
        
        return result

    def call_tool_sync(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Call a tool synchronously"""
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

        # Simple async to sync conversion
        try:
            return asyncio.run(self.call_tool(tool_name, arguments))
        except Exception as e:
            return {
                "status": "failed",
                "error": f"Tool call failed: {e}",
                "tool_name": tool_name,
                "arguments": arguments
            }
    
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
            "name": tool_name,
            "description": tool_info.get("description", f"MCP tool: {tool_name}"),
            "input_schema": {
                "type": "object",
                "properties": properties,
                "required": required
            }
        }
        
        return tool_def
    
    def supports_server(self, server_name: str) -> bool:
        """Check if a specific server is supported"""
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
    
    def get_status(self) -> Dict[str, Any]:
        """Get client status"""
        return {
            "initialized": self.initialized,
            "servers": list(self.servers.keys()),
            "total_tools": len(self.available_tools),
            "config_path": self.config_path,
            "fastmcp_available": FASTMCP_AVAILABLE,
            "server_manager_available": self.server_manager is not None
        }
    
    async def cleanup(self):
        """Cleanup resources gracefully"""
        try:
            print_current("🔄 Starting FastMCP client cleanup...")
            
            # First, clear tool references to prevent new calls
            self.available_tools.clear()
            self.servers.clear()
            
            # If we have a server manager reference, let it know we're cleaning up
            if self.server_manager:
                try:
                    # The server manager will handle its own cleanup
                    # We just need to clear our reference
                    self.server_manager = None
                except Exception as e:
                    print_current(f"⚠️ Error clearing server manager reference: {e}")
            
            # Mark as not initialized
            self.initialized = False
            
            # Give a small delay to allow any pending operations to complete
            await asyncio.sleep(0.1)
            
            print_current("🔌 FastMCP client cleaned up")
            
        except Exception as e:
            print_current(f"⚠️ FastMCP cleanup error: {e}")
            # Continue with cleanup even if there are errors


# Global instance with thread safety
_fastmcp_wrapper = None
_fastmcp_config_path = None
_fastmcp_lock = threading.RLock()


def get_fastmcp_wrapper(config_path: str = "config/mcp_servers.json") -> FastMcpWrapper:
    """Get FastMCP wrapper instance (thread-safe)"""
    global _fastmcp_wrapper, _fastmcp_config_path
    
    with _fastmcp_lock:
        if _fastmcp_wrapper is None or _fastmcp_config_path != config_path:
            _fastmcp_wrapper = FastMcpWrapper(config_path)
            _fastmcp_config_path = config_path
        return _fastmcp_wrapper


@asynccontextmanager
async def initialize_fastmcp_with_server_manager(config_path: str = "config/mcp_servers.json"):
    """Initialize FastMCP wrapper with persistent server manager"""
    global _fastmcp_wrapper
    
    # Create wrapper instance
    with _fastmcp_lock:
        wrapper = get_fastmcp_wrapper(config_path)
    
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
            print_current("🔄 FastMCP wrapper context exiting...")


async def initialize_fastmcp_wrapper(config_path: str = "config/mcp_servers.json") -> bool:
    """Initialize FastMCP wrapper (backward compatibility)"""
    global _fastmcp_wrapper
    
    try:
        # Create wrapper instance
        with _fastmcp_lock:
            wrapper = get_fastmcp_wrapper(config_path)
        
        # For backward compatibility, we'll initialize without server manager first
        # This allows basic functionality without requiring the full server manager context
        if not wrapper.initialized:
            # Load configuration
            await wrapper._load_config()
            
            # Try to discover tools using standalone tool discovery (without server manager)
            try:
                await wrapper._discover_tools_standalone()
                print_current(f"✅ FastMCP wrapper initialized with {len(wrapper.available_tools)} tools discovered")
            except Exception as tool_discovery_error:
                print_current(f"⚠️ FastMCP wrapper basic initialization completed, tool discovery will retry later: {tool_discovery_error}")
            
            # Mark as initialized
            wrapper.initialized = True
        
        return True
    except Exception as e:
        logger.error(f"FastMCP wrapper initialization failed: {e}")
        return False


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
    """Cleanup FastMCP wrapper synchronously with improved error handling"""
    global _fastmcp_wrapper, _fastmcp_config_path
    if _fastmcp_wrapper:
        try:
            # Check if there's an existing event loop
            try:
                loop = asyncio.get_running_loop()
                if loop.is_closed():
                    raise RuntimeError("Loop is closed")
                # If we're in an async context, create a task instead of using run()
                task = loop.create_task(_fastmcp_wrapper.cleanup())
                # Don't wait for completion to avoid blocking
            except RuntimeError:
                # No running loop, safe to use asyncio.run()
                try:
                    asyncio.run(_fastmcp_wrapper.cleanup())
                except Exception:
                    # If asyncio.run fails, do manual cleanup
                    _manual_cleanup()
        except Exception:
            # If all async methods fail, do manual cleanup
            _manual_cleanup()
        finally:
            _fastmcp_wrapper = None
            _fastmcp_config_path = None

def _manual_cleanup():
    """Manual cleanup when async methods fail"""
    global _fastmcp_wrapper
    try:
        if _fastmcp_wrapper:
            # Manual cleanup without async
            _fastmcp_wrapper.available_tools.clear()
            _fastmcp_wrapper.servers.clear()
            _fastmcp_wrapper.server_manager = None
            _fastmcp_wrapper.initialized = False
    except Exception:
        pass  # Silently ignore cleanup errors

def safe_cleanup_fastmcp_wrapper():
    """Safe cleanup FastMCP wrapper with comprehensive error handling"""
    try:
        cleanup_fastmcp_wrapper_sync()
    except Exception as e:
        try:
            print_current(f"⚠️ FastMCP cleanup error: {e}")
        except:
            pass  # Even print may fail if everything is shutting down
        # Try manual cleanup as last resort
        try:
            _manual_cleanup()
            global _fastmcp_wrapper, _fastmcp_config_path
            _fastmcp_wrapper = None
            _fastmcp_config_path = None
        except:
            pass


# Register cleanup at exit to ensure clean shutdown
def _atexit_cleanup():
    """Emergency cleanup at program exit"""
    try:
        safe_cleanup_fastmcp_wrapper()
    except:
        pass  # Silently handle any errors during exit

# Register the exit cleanup handler
atexit.register(_atexit_cleanup)


# Test function for FastMCP wrapper
async def test_fastmcp_wrapper():
    """Test FastMCP wrapper functionality"""
    print_current("🧪 Starting FastMCP wrapper test...")
    
    config_path = "config/mcp_servers.json"
    
    # Test with server manager
    async with initialize_fastmcp_with_server_manager(config_path) as wrapper:
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
    
    print_current("✅ FastMCP wrapper test completed!")


def test_fastmcp_wrapper_sync():
    """Synchronous test wrapper"""
    asyncio.run(test_fastmcp_wrapper())


if __name__ == "__main__":
    """Test FastMCP wrapper when run directly"""
    test_fastmcp_wrapper_sync()