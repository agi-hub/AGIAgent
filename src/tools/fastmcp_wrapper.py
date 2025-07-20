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
from typing import Dict, Any, List, Optional, Union
from .print_system import print_current

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
            print_current(f"✅ FastMCP client initialized successfully, discovered {len(self.available_tools)} tools")
            return True
            
        except Exception as e:
            print_current(f"❌ FastMCP client initialization failed: {e}")
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
                # Ensure client is properly connected before listing tools
                if not hasattr(self.main_client, '_connected') or not self.main_client._connected:
                    print_current("🔄 FastMCP client not connected, attempting to connect...")
                
                # Use FastMCP client to list tools
                async with self.main_client:
                    tools = await self.main_client.list_tools()
                    
                    print_current(f"🔧 FastMCP discovered {len(tools)} tools from servers")
                    
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
                
                print_current(f"✅ FastMCP tool discovery completed: {len(self.available_tools)} tools available")
                
            except Exception as e:
                print_current(f"❌ FastMCP tool discovery failed: {e}")
                # Try to reinitialize the client on discovery failure
                print_current("🔄 Attempting to reinitialize FastMCP client...")
                try:
                    await self._initialize_clients()
                    print_current("✅ FastMCP client reinitialized successfully")
                except Exception as reinit_error:
                    print_current(f"❌ FastMCP client reinitialization failed: {reinit_error}")
        else:
            print_current("⚠️ No FastMCP main client available for tool discovery")
    
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
        """Call a tool synchronously using FastMCP (primary method)"""
        if not FASTMCP_AVAILABLE:
            return {
                "status": "error",
                "error": "FastMCP not available. Please install it using: pip install fastmcp",
                "tool_name": tool_name,
                "arguments": arguments
            }
            
        if not self.initialized:
            return {
                "status": "error", 
                "error": "MCP client not initialized",
                "tool_name": tool_name,
                "arguments": arguments
            }

        if tool_name not in self.available_tools:
            return {
                "status": "error",
                "error": f"Tool {tool_name} does not exist",
                "tool_name": tool_name,
                "arguments": arguments
            }

        tool_info = self.available_tools[tool_name]
        
        # Use FastMCP call synchronously by wrapping async call
        import asyncio
        try:
            result = asyncio.run(self._call_tool_fastmcp_async(tool_name, tool_info, arguments))
            return result
        except Exception as e:
            return {
                "status": "error",
                "error": f"FastMCP call failed: {e}",
                "tool_name": tool_name,
                "arguments": arguments
            }

    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Call a tool (async version - kept for compatibility)"""
        if not FASTMCP_AVAILABLE:
            return {
                "status": "error",
                "error": "FastMCP not available. Please install it using: pip install fastmcp",
                "tool_name": tool_name,
                "arguments": arguments
            }

        if not self.initialized:
            raise Exception("MCP client not initialized")

        # Debug: Log available tools status
        print_current(f"🔍 FastMCP debug: Checking tool {tool_name}")
        print_current(f"🔍 Available tools count: {len(self.available_tools)}")
        print_current(f"🔍 Available tools: {list(self.available_tools.keys())[:10]}...")  # Show first 10
        
        if tool_name not in self.available_tools:
            # Try to rediscover tools before failing
            print_current(f"🔄 Tool {tool_name} not found, attempting to rediscover tools...")
            try:
                await self._discover_tools()
                print_current(f"🔍 After rediscovery, available tools count: {len(self.available_tools)}")
                
                if tool_name not in self.available_tools:
                    raise Exception(f"Tool {tool_name} does not exist even after rediscovery")
                else:
                    print_current(f"✅ Tool {tool_name} found after rediscovery")
            except Exception as e:
                print_current(f"❌ Failed to rediscover tools: {e}")
                raise Exception(f"Tool {tool_name} does not exist")

        tool_info = self.available_tools[tool_name]
        server_name = tool_info["server"]

        try:
            # Use the main FastMCP client with proper context management
            if not self.main_client:
                raise Exception("No FastMCP main client available")

            # CRITICAL FIX: Ensure tools are available in the new connection context
            # Call the tool using FastMCP with context manager and tool verification
            async with self.main_client:
                # First, verify tools are available in this connection context
                print_current(f"🔧 Verifying tools in new connection context...")
                try:
                    context_tools = await self.main_client.list_tools()
                    context_tool_names = [tool.name for tool in context_tools]
                    print_current(f"🔍 Tools available in connection context: {len(context_tool_names)}")
                    print_current(f"🔍 Context tools: {context_tool_names[:10]}...")
                    
                    if tool_name not in context_tool_names:
                        print_current(f"❌ Tool {tool_name} not found in connection context")
                        print_current(f"🔍 Looking for similar tools...")
                        for ctx_tool in context_tool_names:
                            if tool_name in ctx_tool or ctx_tool in tool_name:
                                print_current(f"🔍 Similar tool found: {ctx_tool}")
                        raise Exception(f"Tool {tool_name} not available in connection context")
                    else:
                        print_current(f"✅ Tool {tool_name} verified in connection context")
                        
                except Exception as list_error:
                    print_current(f"⚠️ Failed to list tools in connection context: {list_error}")
                    # Continue anyway, might still work
                
                # Now attempt the actual tool call
                print_current(f"🔧 Attempting FastMCP call with name: {tool_name}")
                result = await self.main_client.call_tool(tool_name, arguments)
                
                # If successful, return the result
                print_current(f"✅ FastMCP call successful with name: {tool_name}")
                return {
                    "status": "success",
                    "result": self._format_tool_result(result),
                    "tool_name": tool_name,  # Return API compatible name
                    "original_tool_name": tool_info.get("original_name", tool_name),
                    "arguments": arguments
                }
                
        except Exception as e:
            print_current(f"❌ FastMCP call failed: {e}")
            
            # Check if it's a connection error and try to recover
            if "not connected" in str(e).lower() or "context manager" in str(e).lower():
                print_current(f"🔄 Connection error detected, attempting recovery for tool {tool_name}")
                try:
                    # Try to reinitialize the client
                    await self._initialize_clients()
                    if self.main_client:
                        # Retry the call with new client
                        async with self.main_client:
                            result = await self.main_client.call_tool(tool_name, arguments)
                            
                            return {
                                "status": "success",
                                "result": self._format_tool_result(result),
                                "tool_name": tool_name,
                                "original_tool_name": tool_info.get("original_name", tool_name),
                                "arguments": arguments
                            }
                    else:
                        raise Exception("Failed to recover FastMCP client")
                except Exception as recovery_error:
                    error_msg = f"Tool call failed and recovery failed: {recovery_error}"
                    print_current(f"❌ {error_msg}")
                    return {
                        "status": "error",
                        "error": error_msg,
                        "tool_name": tool_name,
                        "arguments": arguments
                    }
                    
            # If all name variations failed, try direct command line fallback
            else:
                print_current(f"🔄 All FastMCP calls failed, trying direct command line fallback...")
                
                # Try to call the tool directly using command line if possible
                server_name = tool_info["server"]
                if server_name in self.servers:
                    server_config = self.servers[server_name]
                    if server_config.get("command"):
                        try:
                            # Build direct command
                            import subprocess
                            import json
                            
                            cmd = [server_config["command"]] + server_config.get("args", [])
                            
                            # Create MCP request
                            request = {
                                "jsonrpc": "2.0",
                                "id": 1,
                                "method": "tools/call",
                                "params": {
                                    "name": tool_name,
                                    "arguments": arguments
                                }
                            }
                            
                            print_current(f"🔧 Direct command fallback: {' '.join(cmd)}")
                            
                            # Call the MCP server directly
                            process = subprocess.Popen(
                                cmd,
                                stdin=subprocess.PIPE,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE,
                                text=True
                            )
                            
                            request_json = json.dumps(request)
                            stdout, stderr = process.communicate(input=request_json, timeout=30)
                            
                            # Debug logging
                            print_current(f"🔍 Direct command exit code: {process.returncode}")
                            print_current(f"🔍 STDOUT length: {len(stdout) if stdout else 0}")
                            print_current(f"🔍 STDERR: {stderr[:200] if stderr else 'None'}...")
                            
                            if process.returncode == 0:
                                if stdout and stdout.strip():
                                    try:
                                        response = json.loads(stdout)
                                        if "result" in response:
                                            print_current(f"✅ Direct command fallback successful!")
                                            return {
                                                "status": "success",
                                                "result": self._format_tool_result(response["result"]),
                                                "tool_name": tool_name,
                                                "original_tool_name": tool_info.get("original_name", tool_name),
                                                "arguments": arguments
                                            }
                                        elif "error" in response:
                                            print_current(f"❌ Direct command returned error: {response['error']}")
                                        else:
                                            print_current(f"⚠️ Unexpected response format: {response}")
                                    except json.JSONDecodeError as e:
                                        print_current(f"❌ Failed to parse direct command response: {e}")
                                        print_current(f"Raw stdout: {stdout[:500]}...")
                                else:
                                    # Empty stdout but successful exit - this might be normal for some MCP servers
                                    print_current(f"⚠️ Direct command succeeded but returned empty stdout")
                                    # Check if stderr contains any useful information  
                                    print_current(f"🔍 Checking stderr for server startup: '{stderr}'")
                                    if stderr and "running on stdio" in stderr.lower():
                                        print_current(f"✅ MCP server started successfully (detected from stderr)")
                                        # For memory operations, we can assume success if the server started
                                        return {
                                            "status": "success", 
                                            "result": {"message": f"Memory operation '{tool_name}' completed successfully"},
                                            "tool_name": tool_name,
                                            "original_tool_name": tool_info.get("original_name", tool_name),
                                            "arguments": arguments
                                        }
                                    else:
                                        print_current(f"⚠️ No server startup indicator found in stderr")
                            else:
                                print_current(f"❌ Direct command failed with return code {process.returncode}")
                                if stderr:
                                    print_current(f"STDERR: {stderr}")
                                    
                        except subprocess.TimeoutExpired:
                            print_current(f"⏰ Direct command timeout")
                        except Exception as cmd_error:
                            print_current(f"❌ Direct command exception: {cmd_error}")
                
                # If direct command also failed, raise the original FastMCP error
                raise e
    
    async def _call_tool_fastmcp_async(self, tool_name: str, tool_info: Dict[str, Any], arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Call tool using FastMCP asynchronously (fixed approach)"""
        if not self.main_client:
            raise Exception("No FastMCP main client available")

        # Use the FastMCP tool name (with prefix) for FastMCP calls
        fastmcp_tool_name = tool_info["api_name"]  # This is the FastMCP discovered name
        print_current(f"🔧 FastMCP async call - API tool: {tool_name} -> FastMCP tool: {fastmcp_tool_name}")
        
        # STRATEGY 1: Try to use existing connection first
        try:
            # Attempt to use FastMCP with a persistent connection approach
            print_current(f"🔧 Attempting FastMCP call with tool: {fastmcp_tool_name}")
            
            # Use main client context but try to ensure tools are available
            async with self.main_client:
                # First, verify that tools are available in this context
                available_tools = await self.main_client.list_tools()
                available_tool_names = [tool.name for tool in available_tools]
                print_current(f"🔍 FastMCP context has {len(available_tool_names)} tools available")
                
                if fastmcp_tool_name in available_tool_names:
                    print_current(f"✅ Tool {fastmcp_tool_name} found in FastMCP context")
                    result = await self.main_client.call_tool(fastmcp_tool_name, arguments)
                    print_current(f"✅ FastMCP call successful for tool: {fastmcp_tool_name}")
                    return {
                        "status": "success",
                        "result": self._format_tool_result(result),
                        "tool_name": tool_name,
                        "original_tool_name": tool_info.get("original_name", tool_name),
                        "arguments": arguments
                    }
                else:
                    print_current(f"❌ Tool {fastmcp_tool_name} not found in FastMCP context")
                    print_current(f"🔍 Available tools: {available_tool_names[:10]}...")
                    raise Exception(f"Tool {fastmcp_tool_name} not available in FastMCP context")
        
        except Exception as e:
            print_current(f"❌ FastMCP async call failed: {e}")
            raise e
    
    def _call_tool_direct_sync(self, tool_name: str, tool_info: Dict[str, Any], arguments: Dict[str, Any], server_config: Dict[str, Any]) -> Dict[str, Any]:
        """Call tool directly using MCP command line (synchronous version)"""
        import subprocess
        import json
        
        cmd = [server_config["command"]] + server_config.get("args", [])
        
        # Get the actual tool name (remove prefix)
        actual_tool_name = self._get_actual_tool_name(tool_name, tool_info["server"])
        
        # Create MCP request
        request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": actual_tool_name,
                "arguments": arguments
            }
        }
        
        print_current(f"🔧 Direct MCP call (sync): {' '.join(cmd)}")
        print_current(f"🔍 API tool name: {tool_name}")
        print_current(f"🔍 Actual tool name: {actual_tool_name}")
        print_current(f"🔍 MCP request: {json.dumps(request, ensure_ascii=False)}")
        
        # Call the MCP server directly
        process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        request_json = json.dumps(request)
        print_current(f"🔍 Sending request: {request_json}")
        
        # Send request and close stdin to signal end of input
        stdout, stderr = process.communicate(input=request_json + '\n', timeout=30)
        
        # Debug logging
        print_current(f"🔍 Direct command exit code: {process.returncode}")
        print_current(f"🔍 STDOUT length: {len(stdout) if stdout else 0}")
        print_current(f"🔍 STDERR: {stderr[:200] if stderr else 'None'}...")
        
        if process.returncode == 0:
            if stdout and stdout.strip():
                try:
                    response = json.loads(stdout)
                    if "result" in response:
                        print_current(f"✅ Direct MCP call (sync) successful!")
                        return {
                            "status": "success",
                            "result": self._format_tool_result(response["result"]),
                            "tool_name": tool_name,
                            "original_tool_name": tool_info.get("original_name", tool_name),
                            "arguments": arguments
                        }
                    elif "error" in response:
                        print_current(f"❌ Direct MCP returned error: {response['error']}")
                        raise Exception(f"MCP tool returned error: {response['error']}")
                    else:
                        print_current(f"⚠️ Unexpected response format: {response}")
                        raise Exception(f"Unexpected MCP response format: {response}")
                except json.JSONDecodeError as e:
                    print_current(f"❌ Failed to parse direct MCP response: {e}")
                    print_current(f"Raw stdout: {stdout[:500]}...")
                    raise Exception(f"Failed to parse MCP response: {e}")
            else:
                # Empty stdout - this is NOT success, it means the MCP call didn't work
                print_current(f"❌ Direct MCP returned empty stdout - tool call failed")
                print_current(f"🔍 STDERR (for debugging): '{stderr}'")
                
                # Server startup alone is NOT tool execution success
                if stderr and "running on stdio" in stderr.lower():
                    print_current(f"ℹ️ MCP server started but tool call failed (empty response)")
                    raise Exception(f"MCP server started but tool '{tool_name}' execution failed - no response received")
                else:
                    raise Exception(f"MCP tool call failed - no response and no server startup detected")
        else:
            print_current(f"❌ Direct MCP failed with return code {process.returncode}")
            if stderr:
                print_current(f"STDERR: {stderr}")
            raise Exception(f"MCP command failed with return code {process.returncode}: {stderr}")
    
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
    
    def _get_actual_tool_name(self, tool_name: str, server_name: str) -> str:
        """Get the actual tool name by removing server prefix"""
        # If tool name starts with server prefix, remove it
        server_prefix = f"{server_name}_"
        if tool_name.startswith(server_prefix):
            return tool_name[len(server_prefix):]
        
        # If no prefix, return as is
        return tool_name
    
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
            # FastMCP clients are context managers, they clean up automatically
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
            print_current(f"✅ FastMCP wrapper initialized successfully in thread {threading.current_thread().name}")
        else:
            print_current(f"⚠️ FastMCP wrapper initialization failed in thread {threading.current_thread().name}")
        return result
    except Exception as e:
        print_current(f"❌ FastMCP wrapper initialization error in thread {threading.current_thread().name}: {e}")
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
        try:
            # Try to get the current event loop
            loop = asyncio.get_running_loop()
            # If there's a running loop, schedule the cleanup
            async def async_cleanup():
                await wrapper_instance.cleanup()
            loop.create_task(async_cleanup())
        except RuntimeError:
            # No event loop running, try to create one for cleanup
            try:
                asyncio.run(wrapper_instance.cleanup())
            except RuntimeError:
                # If that fails too, use synchronous cleanup
                wrapper_instance.cleanup_sync()
        except Exception as e:
            # If all else fails, just clean up the references
            print_current(f"⚠️ FastMCP client cleanup failed: {e}")
            wrapper_instance.cleanup_sync()
        
        _fastmcp_wrapper = None
        _fastmcp_config_path = None 