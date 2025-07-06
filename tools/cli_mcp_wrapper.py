#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cli-mcp wrapper for AGIBot
Using mature cli-mcp library as MCP client
"""

import json
import subprocess
import os
import asyncio
from typing import Dict, Any, List, Optional, Union
from .print_system import print_current

class CliMcpWrapper:
    """cli-mcp wrapper, providing MCP functionality for AGIBot"""
    
    # Class variable to track if installation message has been shown
    _installation_message_shown = False
    
    def __init__(self, config_path: str = "mcp.json"):
        self.config_path = config_path
        self.available_tools = {}
        self.servers = {}
        self.initialized = False
        
        # Ensure config file exists
        if not os.path.exists(self.config_path):
            self._create_default_config()
    
    def _create_default_config(self):
        """Create default configuration file"""
        default_config = {
            "mcpServers": {
                "filesystem": {
                    "command": "C:\\Program Files\\nodejs\\npx.cmd",
                    "args": ["@modelcontextprotocol/server-filesystem", os.getcwd()],
                    "env": {}
                }
            }
        }
        
        with open(self.config_path, 'w', encoding='utf-8') as f:
            json.dump(default_config, f, ensure_ascii=False, indent=2)
        
        print_current(f"✅ Created default MCP config file: {self.config_path}")
    
    async def initialize(self) -> bool:
        """Initialize MCP client"""
        try:
            # Load configuration
            await self._load_config()
            
            # Discover all tools
            await self._discover_tools()
            
            self.initialized = True
            print_current(f"✅ cli-mcp client initialized successfully, discovered {len(self.available_tools)} tools")
            return True
            
        except Exception as e:
            print_current(f"❌ cli-mcp client initialization failed: {e}")
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
            
            print_current(f"📊 cli-mcp client config loaded successfully, found {len(self.servers)} NPX/NPM servers")
            
        except Exception as e:
            print_current(f"❌ Failed to load config file: {e}")
            raise
    
    async def _discover_tools(self):
        """Discover all available tools"""
        self.available_tools = {}
        
        for server_name, server_config in self.servers.items():
            # Only handle enabled servers
            if not server_config.get("enabled", True):
                print_current(f"⏭️  Skipping disabled server: {server_name}")
                continue
                
            try:
                tools = await self._list_server_tools(server_name)
                for tool_name, tool_info in tools.items():
                    # Use server.tool format to avoid conflicts, but replace dots with underscores for Claude API compatibility
                    original_full_name = f"{server_name}.{tool_name}"
                    api_compatible_name = f"{server_name}_{tool_name}"
                    
                    self.available_tools[api_compatible_name] = {
                        "server": server_name,
                        "tool": tool_name,
                        "original_name": original_full_name,
                        "api_name": api_compatible_name,
                        "description": tool_info.get("description", ""),
                        "parameters": tool_info.get("parameters", [])
                    }
                
                print_current(f"🔧 Server {server_name} discovered {len(tools)} tools")
                
            except Exception as e:
                print_current(f"⚠️ Server {server_name} tool discovery failed: {e}")
    
    async def _list_server_tools(self, server_name: str) -> Dict[str, Any]:
        """List tools for a specific server"""
        try:
            # Get server configuration
            server_config = self.servers.get(server_name, {})
            
            # Prepare environment variables
            env = os.environ.copy()
            env.update(server_config.get('env', {}))
            
            # Use subprocess to call cli-mcp
            result = await asyncio.create_subprocess_exec(
                "cli-mcp", "list", server_name,
                "--configpath", self.config_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env  # Pass environment variables
            )
            
            stdout, stderr = await result.communicate()
            
            if result.returncode != 0:
                raise Exception(f"cli-mcp list failed: {stderr.decode()}")
            
            # Parse output
            tools = {}
            output = stdout.decode()
            
            # Simple parsing of output format
            lines = output.strip().split('\n')
            for line in lines:
                if line.startswith('- '):
                    parts = line[2:].split(':', 1)
                    if len(parts) >= 2:
                        tool_name = parts[0].strip()
                        description = parts[1].strip()
                        
                        # Parse parameter information (if any)
                        parameters = []
                        if "Parameters:" in line:
                            param_part = line.split("Parameters:")[1].split("(*")[0].strip()
                            if param_part:
                                # Simple parsing of parameter format: *param1(type), param2(type)
                                param_items = param_part.split(',')
                                for param_item in param_items:
                                    param_item = param_item.strip()
                                    if param_item:
                                        required = param_item.startswith('*')
                                        if required:
                                            param_item = param_item[1:]
                                        
                                        if '(' in param_item and ')' in param_item:
                                            param_name = param_item.split('(')[0].strip()
                                            param_type = param_item.split('(')[1].split(')')[0].strip()
                                        else:
                                            param_name = param_item
                                            param_type = "string"
                                        
                                        if param_name:
                                            parameters.append({
                                                "name": param_name,
                                                "type": param_type,
                                                "required": required,
                                                "description": f"{param_name} parameter"
                                            })
                        
                        tools[tool_name] = {
                            "description": description,
                            "parameters": parameters
                        }
            
            return tools
            
        except FileNotFoundError as e:
            # cli-mcp command not found
            if "cli-mcp" in str(e):
                if not self._installation_message_shown:
                    print_current(f"❌ cli-mcp command not found. Please install it using: pip install cli-mcp")
                    print_current(f"💡 After installation, restart AGIBot to use MCP tools.")
                    self._installation_message_shown = True
            else:
                print_current(f"❌ Failed to list tools for server {server_name}: {e}")
            return {}
        except Exception as e:
            error_msg = str(e)
            if "No such file or directory" in error_msg and "cli-mcp" in error_msg:
                if not self._installation_message_shown:
                    print_current(f"❌ cli-mcp command not found. Please install it using: pip install cli-mcp")
                    print_current(f"💡 After installation, restart AGIBot to use MCP tools.")
                    self._installation_message_shown = True
            else:
                print_current(f"❌ Failed to list tools for server {server_name}: {e}")
            return {}
    
    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Call a tool"""
        if not self.initialized:
            raise Exception("MCP client not initialized")
        
        if tool_name not in self.available_tools:
            raise Exception(f"Tool {tool_name} does not exist")
        
        tool_info = self.available_tools[tool_name]
        server_name = tool_info["server"]
        actual_tool_name = tool_info["tool"]
        
        try:
            # Get server configuration
            server_config = self.servers.get(server_name, {})
            
            # Prepare environment variables
            env = os.environ.copy()
            env.update(server_config.get('env', {}))
            
            # Prepare arguments
            args_json = json.dumps(arguments)
            
            # Call cli-mcp using the original tool name
            result = await asyncio.create_subprocess_exec(
                "cli-mcp", "call", server_name, actual_tool_name, args_json,
                "--configpath", self.config_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env  # Pass environment variables
            )
            
            stdout, stderr = await result.communicate()
            
            if result.returncode != 0:
                error_msg = stderr.decode()
                raise Exception(f"Tool call failed: {error_msg}")
            
            # Parse result
            try:
                result_data = json.loads(stdout.decode())
                return {
                    "status": "success",
                    "result": result_data.get("result", result_data),
                    "tool_name": tool_name,  # Return API compatible name
                    "original_tool_name": tool_info.get("original_name", tool_name),
                    "arguments": arguments
                }
            except json.JSONDecodeError:
                # If not JSON, return raw text
                return {
                    "status": "success",
                    "result": stdout.decode(),
                    "tool_name": tool_name,  # Return API compatible name
                    "original_tool_name": tool_info.get("original_name", tool_name),
                    "arguments": arguments
                }
                
        except FileNotFoundError as e:
            # cli-mcp command not found
            if "cli-mcp" in str(e):
                error_msg = "cli-mcp command not found. Please install it using: pip install cli-mcp"
                if not self._installation_message_shown:
                    print_current(f"❌ {error_msg}")
                    print_current(f"💡 After installation, restart AGIBot to use MCP tools.")
                    self._installation_message_shown = True
                return {
                    "status": "error",
                    "error": error_msg,
                    "tool_name": tool_name,
                    "arguments": arguments
                }
            else:
                print_current(f"❌ Failed to call tool {tool_name}: {e}")
                return {
                    "status": "error",
                    "error": str(e),
                    "tool_name": tool_name,
                    "arguments": arguments
                }
        except Exception as e:
            error_msg = str(e)
            if "No such file or directory" in error_msg and "cli-mcp" in error_msg:
                friendly_error = "cli-mcp command not found. Please install it using: pip install cli-mcp"
                if not self._installation_message_shown:
                    print_current(f"❌ {friendly_error}")
                    print_current(f"💡 After installation, restart AGIBot to use MCP tools.")
                    self._installation_message_shown = True
                return {
                    "status": "error",
                    "error": friendly_error,
                    "tool_name": tool_name,
                    "arguments": arguments
                }
            else:
                print_current(f"❌ Failed to call tool {tool_name}: {e}")
                return {
                    "status": "error",
                    "error": str(e),
                    "tool_name": tool_name,
                    "arguments": arguments
                }
    
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
            param_type = param["type"]
            param_required = param.get("required", False)
            param_desc = param.get("description", f"{param_name} parameter")
            
            # Map types
            schema_type = "string"  # Default type
            if param_type in ["string", "number", "integer", "boolean", "array", "object"]:
                schema_type = param_type
            elif param_type == "int":
                schema_type = "integer"
            elif param_type == "float":
                schema_type = "number"
            elif param_type == "bool":
                schema_type = "boolean"
            elif param_type == "list":
                schema_type = "array"
            elif param_type == "dict":
                schema_type = "object"
            
            properties[param_name] = {
                "type": schema_type,
                "description": param_desc
            }
            
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
    
    def get_status(self) -> Dict[str, Any]:
        """Get client status"""
        return {
            "initialized": self.initialized,
            "servers": list(self.servers.keys()),
            "total_tools": len(self.available_tools),
            "config_path": self.config_path
        }
    
    def cleanup(self):
        """Cleanup resources"""
        try:
            # Close any open connections
            self.available_tools.clear()
            self.servers.clear()
            self.initialized = False
            # print_current("🔌 cli-mcp client cleaned up")
        except Exception as e:
            # print_current("🔌 cli-mcp client cleaned up")
            pass
    
    def cleanup_sync(self):
        """Synchronous cleanup client"""
        print_current("🔌 cli-mcp client cleaned up")


# Global instance
_cli_mcp_wrapper = None
_cli_mcp_config_path = None

def get_cli_mcp_wrapper(config_path: str = "mcp.json") -> CliMcpWrapper:
    """Get cli-mcp wrapper instance"""
    global _cli_mcp_wrapper, _cli_mcp_config_path
    if _cli_mcp_wrapper is None or _cli_mcp_config_path != config_path:
        _cli_mcp_wrapper = CliMcpWrapper(config_path)
        _cli_mcp_config_path = config_path
    return _cli_mcp_wrapper

async def initialize_cli_mcp_wrapper(config_path: str = "mcp.json") -> bool:
    """Initialize cli-mcp wrapper"""
    wrapper = get_cli_mcp_wrapper(config_path)
    return await wrapper.initialize()

async def cleanup_cli_mcp_wrapper():
    """Cleanup cli-mcp wrapper"""
    global _cli_mcp_wrapper, _cli_mcp_config_path
    if _cli_mcp_wrapper:
        await _cli_mcp_wrapper.cleanup()
        _cli_mcp_wrapper = None
        _cli_mcp_config_path = None

def cleanup_cli_mcp_wrapper_sync():
    """Cleanup cli-mcp wrapper synchronously"""
    global _cli_mcp_wrapper, _cli_mcp_config_path
    if _cli_mcp_wrapper:
        try:
            # Try to get the current event loop
            loop = asyncio.get_running_loop()
            # If there's a running loop, schedule the cleanup
            loop.create_task(_cli_mcp_wrapper.cleanup())
        except RuntimeError:
            # No event loop running, use synchronous cleanup
            _cli_mcp_wrapper.cleanup_sync()
        
        _cli_mcp_wrapper = None
        _cli_mcp_config_path = None

def safe_cleanup_cli_mcp_wrapper():
    """Safely cleanup cli-mcp wrapper in any context"""
    global _cli_mcp_wrapper, _cli_mcp_config_path
    if _cli_mcp_wrapper:
        try:
            # Try to get the current event loop
            loop = asyncio.get_running_loop()
            # If there's a running loop, schedule the cleanup
            loop.create_task(_cli_mcp_wrapper.cleanup())
        except RuntimeError:
            # No event loop running, try to create one for cleanup
            try:
                asyncio.run(_cli_mcp_wrapper.cleanup())
            except RuntimeError:
                # If that fails too, use synchronous cleanup
                _cli_mcp_wrapper.cleanup_sync()
        except Exception as e:
            # If all else fails, just clean up the references
            print_current(f"⚠️ cli-mcp client cleanup failed: {e}")
            _cli_mcp_wrapper.cleanup_sync()
        
        _cli_mcp_wrapper = None
        _cli_mcp_config_path = None