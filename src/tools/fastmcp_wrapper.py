#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FastMCP wrapper for AGIAgent
Using FastMCP library as MCP client with persistent server management
"""

import json
import os
import asyncio
import threading
import logging
import warnings
import atexit
import signal
import psutil
import tempfile
import io
from typing import Dict, Any, List, Optional, Set
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
    from .print_system import print_current, print_debug
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
    
    def __init__(self, config_path: str = "config/mcp_servers.json", workspace_dir: Optional[str] = None):
        self.config_path = config_path
        self.available_tools = {}
        self.servers = {}
        self.initialized = False
        self.server_manager = None  # Reference to persistent server manager
        self._persistent_clients = {}  # Cache for persistent MCP clients
        # All servers are now treated as stateful for maximum reliability
        # This ensures persistent connections and state preservation for all MCP servers
        self._stateful_servers = None  # All servers are stateful by default

        # Shared event loop and thread for stateful servers
        self._shared_loop = None
        self._shared_thread = None
        self._loop_lock = threading.Lock()

        # Enhanced subprocess tracking for forced cleanup
        self._tracked_processes: Set[int] = set()  # Track subprocess PIDs
        self._server_processes: Dict[str, Dict[str, Any]] = {}  # Track server -> process info
        self._cleanup_lock = threading.RLock()  # Thread-safe cleanup operations

        # Process health monitoring
        self._health_check_interval = 60  # Check every 60 seconds
        self._health_check_thread = None
        self._health_check_active = False
        self._last_health_check = 0

        # Workspace directory for MCP servers
        if workspace_dir:
            # If workspace_dir is provided, use it directly
            self._workspace_dir = workspace_dir
            print_debug(f"📁 Using specified workspace directory: {workspace_dir}")
        else:
            # Fall back to automatic detection
            self._workspace_dir = self._get_default_workspace_dir()

    def _get_default_workspace_dir(self) -> str:
        """Get default workspace directory for MCP servers"""
        try:
            current_dir = os.getcwd()
            print_debug(f"🔍 Current working directory: {current_dir}")

            # Strategy 1: Check if we're currently in an output_xxx/workspace directory
            dir_name = os.path.basename(current_dir)
            if dir_name == 'workspace':
                parent_dir = os.path.dirname(current_dir)
                parent_name = os.path.basename(parent_dir)
                if parent_name.startswith('output_'):
                    print_debug(f"📁 Already in workspace directory: {current_dir}")
                    return current_dir

            # Strategy 2: Check if we're in an output_xxx directory (directly)
            if dir_name.startswith('output_') and os.path.exists(os.path.join(current_dir, 'workspace')):
                workspace_dir = os.path.join(current_dir, latest_output_dir, 'workspace')
                print_debug(f"📁 Found workspace directory in current output dir: {workspace_dir}")
                return workspace_dir

            # Strategy 3: Look for the most recent output_xxx directory in current directory
            try:
                items = os.listdir(current_dir)
                output_dirs = [item for item in items
                              if item.startswith('output_') and os.path.isdir(os.path.join(current_dir, item))]

                if output_dirs:
                    # Sort by modification time (most recent first)
                    output_dirs.sort(key=lambda x: os.path.getmtime(os.path.join(current_dir, x)), reverse=True)
                    latest_output_dir = output_dirs[0]
                    workspace_dir = os.path.join(current_dir, latest_output_dir, 'workspace')

                    if os.path.exists(workspace_dir):
                        print_debug(f"📁 Found latest workspace directory: {workspace_dir}")
                        return workspace_dir
                    else:
                        # Create workspace in the latest output directory
                        try:
                            os.makedirs(workspace_dir, exist_ok=True)
                            print_debug(f"📁 Created workspace directory in latest output: {workspace_dir}")
                            return workspace_dir
                        except Exception as e:
                            print_debug(f"⚠️ Could not create workspace in latest output dir: {e}")
            except Exception as e:
                print_debug(f"⚠️ Error searching output directories: {e}")

            # Strategy 4: Look for output_xxx directories in parent directory
            parent_dir = os.path.dirname(current_dir)
            if os.path.exists(parent_dir):
                try:
                    items = os.listdir(parent_dir)
                    output_dirs = [item for item in items
                                  if item.startswith('output_') and os.path.isdir(os.path.join(parent_dir, item))]

                    if output_dirs:
                        # Sort by modification time (most recent first)
                        output_dirs.sort(key=lambda x: os.path.getmtime(os.path.join(parent_dir, x)), reverse=True)
                        latest_output_dir = output_dirs[0]
                        workspace_dir = os.path.join(parent_dir, latest_output_dir, 'workspace')

                        if os.path.exists(workspace_dir):
                            # print_current(f"📁 Found workspace directory in parent: {workspace_dir}")
                            return workspace_dir
                except Exception as e:
                    print_current(f"⚠️ Error searching parent directory: {e}")

            # Strategy 5: Check if workspace directory exists in current directory (only as last resort)
            workspace_dir = os.path.join(current_dir, 'workspace')
            if os.path.exists(workspace_dir):
                # print_current(f"📁 Found workspace directory in current dir (fallback): {workspace_dir}")
                return workspace_dir

            # Strategy 6: Create workspace directory in current directory if nothing else works
            try:
                os.makedirs(workspace_dir, exist_ok=True)
                # print_current(f"📁 Created workspace directory (fallback): {workspace_dir}")
            except Exception as e:
                # print_current(f"⚠️ Could not create workspace directory: {e}")
                # Fallback to current directory
                workspace_dir = current_dir
                # print_current(f"📁 Using current directory as workspace (final fallback): {workspace_dir}")

            return workspace_dir

        except Exception as e:
            print_current(f"⚠️ Error determining workspace directory: {e}")
            return os.getcwd()

        # Check if FastMCP is available
        if not FASTMCP_AVAILABLE:
            if not self._installation_message_shown:
                print_current("FastMCP not found. Please install it using: pip install fastmcp")
                print_current("💡 After installation, restart AGIAgent to use MCP tools.")
                self._installation_message_shown = True
            return
    
    async def initialize(self) -> bool:
        """Initialize MCP client with persistent server manager"""
        if not FASTMCP_AVAILABLE:
            print_current("FastMCP not available")
            return False
            
        try:
            # Initialize shared event loop for all servers (all are treated as stateful)
            self._initialize_shared_loop()
            
            # Load configuration for tool discovery
            await self._load_config()
            
            # Discover tools from running servers
            if self.server_manager:
                # Use server manager if available
                await self._discover_tools_from_servers()
            else:
                await self._discover_tools_standalone()
            
            # Start health monitoring for subprocess management
            self._start_health_monitoring()

            self.initialized = True
            # print_current(f"✅ FastMCP initialization completed successfully with {len(self.available_tools)} tools")
            logger.info(f"FastMCP client initialized, discovered {len(self.available_tools)} tools")
            
            # Register wrapper to agent context if available
            try:
                from .agent_context import get_current_agent_id, set_agent_fastmcp_wrapper
                current_agent_id = get_current_agent_id()
                if current_agent_id:
                    set_agent_fastmcp_wrapper(current_agent_id, self)
            except Exception as e:
                # Silently ignore agent context errors
                pass
            
            return True
            
        except Exception as e:
            # print_current(f"❌ FastMCP initialization failed: {e}")
            logger.error(f"FastMCP client initialization failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _run_shared_loop(self):
        """Run the shared event loop in a separate thread"""
        asyncio.set_event_loop(self._shared_loop)
        try:
            self._shared_loop.run_forever()
        except Exception as e:
            print_current(f"⚠️ Shared event loop error: {e}")
        finally:
            self._shared_loop.close()

    def _initialize_shared_loop(self):
        """Initialize shared event loop for all servers (all are treated as stateful)"""
        with self._loop_lock:
            if not self._shared_loop or self._shared_loop.is_closed():
                self._shared_loop = asyncio.new_event_loop()
                self._shared_thread = threading.Thread(target=self._run_shared_loop, daemon=True)
                self._shared_thread.start()
                # print_current("🔄 Shared event loop initialized for all servers")

    def _get_server_tool_info(self, tool_name: str) -> tuple:
        """Get server name and original tool name for a tool"""
        if tool_name not in self.available_tools:
            return None, None
        
        tool_info = self.available_tools[tool_name]
        server_name = tool_info["server"]
        original_tool_name = tool_info["original_name"]
        return server_name, original_tool_name

    def _track_subprocess(self, server_name: str, process_info: Dict[str, Any]):
        """Track a subprocess for later cleanup"""
        with self._cleanup_lock:
            if 'pid' in process_info:
                self._tracked_processes.add(process_info['pid'])
            self._server_processes[server_name] = process_info.copy()

            # Use time.time() instead of asyncio.get_event_loop().time() to avoid event loop issues in threads
            import time
            self._server_processes[server_name]['tracked_at'] = time.time()

            # print_current(f"📋 Tracking subprocess for server: {server_name} (PID: {process_info.get('pid', 'unknown')})")

    def _force_kill_processes(self):
        """Force kill all tracked subprocesses"""
        with self._cleanup_lock:
            killed_count = 0
            for pid in list(self._tracked_processes):
                try:
                    process = psutil.Process(pid)
                    # Kill the entire process tree
                    for child in process.children(recursive=True):
                        try:
                            child.kill()
                            killed_count += 1
                        except (psutil.NoSuchProcess, psutil.AccessDenied):
                            pass
                    try:
                        process.kill()
                        killed_count += 1
                        # print_current(f"💀 Force killed process: {pid}")
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
                finally:
                    self._tracked_processes.discard(pid)

            if killed_count > 0:
                # print_current(f"💀 Force killed {killed_count} subprocess(es)")
                pass

    def _start_health_monitoring(self):
        """Start background health monitoring"""
        with self._cleanup_lock:
            if self._health_check_thread and self._health_check_thread.is_alive():
                return  # Already running

            self._health_check_active = True
            self._health_check_thread = threading.Thread(
                target=self._health_monitor_loop,
                daemon=True,
                name="FastMCP-HealthMonitor"
            )
            self._health_check_thread.start()
            # print_current("🏥 Started subprocess health monitoring")

    def _stop_health_monitoring(self):
        """Stop background health monitoring"""
        with self._cleanup_lock:
            self._health_check_active = False
            if self._health_check_thread and self._health_check_thread.is_alive():
                self._health_check_thread.join(timeout=2.0)

    def _health_monitor_loop(self):
        """Background health monitoring loop"""
        import time

        while self._health_check_active:
            try:
                current_time = time.time()
                if current_time - self._last_health_check >= self._health_check_interval:
                    self._perform_health_check()
                    self._last_health_check = current_time
            except Exception as e:
                # print_current(f"⚠️ Health check error: {e}")
                pass

            time.sleep(10)  # Check every 10 seconds

    def _perform_health_check(self):
        """Perform health check on tracked processes"""
        with self._cleanup_lock:
            dead_processes = []

            # Use time.time() instead of asyncio.get_event_loop().time() to avoid event loop issues in threads
            import time
            current_time = time.time()

            for pid in list(self._tracked_processes):
                try:
                    process = psutil.Process(pid)
                    if not process.is_running():
                        dead_processes.append(pid)
                        # print_current(f"⚠️ Detected dead process: {pid}")
                    elif hasattr(process, 'status') and process.status() in [psutil.STATUS_ZOMBIE]:
                        dead_processes.append(pid)
                        # print_current(f"👻 Detected zombie process: {pid}")
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    dead_processes.append(pid)
                except Exception as e:
                    # print_current(f"⚠️ Error checking process {pid}: {e}")
                    dead_processes.append(pid)

            # Clean up dead processes
            for pid in dead_processes:
                self._tracked_processes.discard(pid)

            # Clean up old temporary discovery processes
            old_temp_processes = []
            for server_name, process_info in list(self._server_processes.items()):
                if (process_info.get('temporary') and
                    current_time - process_info.get('tracked_at', 0) > 300):  # 5 minutes
                    old_temp_processes.append(server_name)

            for server_name in old_temp_processes:
                if server_name in self._server_processes:
                    process_info = self._server_processes[server_name]
                    if 'pid' in process_info:
                        try:
                            process = psutil.Process(process_info['pid'])
                            process.terminate()
                            process.wait(timeout=5)
                            # print_current(f"🧹 Cleaned up old temporary process: {server_name}")
                        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.TimeoutExpired):
                            pass
                    del self._server_processes[server_name]

    def _cleanup_server_processes(self):
        """Clean up all server-related processes gracefully, then forcefully"""
        with self._cleanup_lock:
            # First try graceful termination
            for server_name, process_info in list(self._server_processes.items()):
                try:
                    if 'pid' in process_info:
                        pid = process_info['pid']
                        process = psutil.Process(pid)
                        # Send SIGTERM first
                        process.terminate()
                        # print_current(f"🛑 Sent SIGTERM to server process: {server_name} (PID: {pid})")
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
                except Exception as e:
                    # print_current(f"⚠️ Error terminating server process {server_name}: {e}")
                    pass

            # Wait a bit for graceful shutdown
            import time
            time.sleep(1.0)

            # Force kill any remaining processes
            self._force_kill_processes()

            # Clear tracking data
            self._server_processes.clear()
            self._tracked_processes.clear()

    async def cleanup_persistent_clients(self):
        """Clean up all persistent MCP clients"""
        # print_current("🔄 Starting comprehensive cleanup of MCP clients and processes...")

        # First clean up server processes (most critical)
        try:
            self._cleanup_server_processes()
        except Exception as e:
            # print_current(f"⚠️ Error during process cleanup: {e}")
            pass

        # Then clean up persistent clients
        for server_name, client_info in list(self._persistent_clients.items()):
            try:
                if client_info['entered']:
                    await client_info['client'].__aexit__(None, None, None)
                    # print_current(f"🧹 Cleaned up persistent client for server: {server_name}")
            except Exception as e:
                # print_current(f"⚠️ Error cleaning up client for {server_name}: {e}")
                pass
        self._persistent_clients.clear()

        # Clean up shared loop
        with self._loop_lock:
            if self._shared_loop and not self._shared_loop.is_closed():
                self._shared_loop.call_soon_threadsafe(self._shared_loop.stop)
                if self._shared_thread and self._shared_thread.is_alive():
                    self._shared_thread.join(timeout=2.0)

        # print_current("✅ Comprehensive cleanup completed")
    
    def __del__(self):
        """Destructor to ensure cleanup"""
        try:
            # First, force kill any tracked processes (most critical)
            if hasattr(self, '_tracked_processes') and self._tracked_processes:
                try:
                    self._force_kill_processes()
                except:
                    pass

            # Then try to clean up persistent clients
            if hasattr(self, '_persistent_clients') and self._persistent_clients:
                import asyncio
                try:
                    # Try to clean up if there's an event loop
                    try:
                        loop = asyncio.get_event_loop()
                        if loop.is_running():
                            # Schedule cleanup as a task
                            asyncio.create_task(self.cleanup_persistent_clients())
                        else:
                            # Run cleanup synchronously
                            asyncio.run(self.cleanup_persistent_clients())
                    except RuntimeError:
                        # No event loop available (e.g., in different thread)
                        # Just clear references without async cleanup
                        pass
                except:
                    # If cleanup fails, at least clear the references
                    self._persistent_clients.clear()
        except:
            # Silent cleanup in destructor
            pass
    
    async def _load_config(self):
        """Load configuration file"""
        try:
            if not os.path.exists(self.config_path):
                # print_current(f"⚠️ MCP config file not found: {self.config_path}")
                return
            
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            all_servers = config.get("mcpServers", {})
            
            # Filter servers - handle both command-based and HTTP-based servers
            self.servers = {}
            for server_name, server_config in all_servers.items():
                # Skip SSE servers (handled by mcp_client)
                if server_config.get("url") and "sse" in server_config.get("url", "").lower():
                    # print_current(f"⏭️  Skipping SSE server {server_name}")
                    continue

                # Handle servers with command field (NPX/NPM format)
                if server_config.get("command"):
                    self.servers[server_name] = server_config
                    print_debug(f"📋 Loading command-based server: {server_name}")

                # Handle HTTP servers (streamable HTTP protocol)
                elif server_config.get("url"):
                    self.servers[server_name] = server_config
                    print_debug(f"🌐 Loading HTTP server: {server_name} -> {server_config.get('url')}")

                # Skip servers without command or url
                else:
                    print_debug(f"⏭️  Skipping server {server_name} (no command or url)")
            
            # Set default values
            for server_name, server_config in self.servers.items():
                if "enabled" not in server_config:
                    server_config["enabled"] = True
                if "timeout" not in server_config:
                    server_config["timeout"] = 30
            

            
        except Exception as e:
            # print_current(f"❌ Failed to load config file: {e}")
            raise
    
    async def _discover_tools_from_servers(self):
        """Discover tools from all configured servers"""
        self.available_tools = {}
        
        if not self.server_manager:
            # print_current("⚠️ No server manager available for tool discovery")
            return
        
        # Try to discover tools from each enabled server
        for server_name in self.servers.keys():
            if self.servers[server_name].get("enabled", True):
                try:
                    discovered_tools = await self._discover_tools_from_server(server_name)
                    if discovered_tools:
                        self.available_tools.update(discovered_tools)
                        # print_current(f"✅ Discovered {len(discovered_tools)} tools from {server_name}")
                    else:
                        # print_current(f"⚠️ No tools discovered from {server_name}")
                        pass
                except Exception as e:
                    # print_current(f"❌ Failed to discover tools from {server_name}: {e}")
                    pass
        
        logger.info(f"Tool discovery completed: {len(self.available_tools)} tools total")
    
    async def _discover_tools_standalone(self):
        """Discover tools from all configured servers without server manager (standalone mode)"""
        self.available_tools = {}
        
        # print_current("🔍 Starting standalone tool discovery for FastMCP servers")
        
        # Try to discover tools from each enabled server
        for server_name in self.servers.keys():
            if self.servers[server_name].get("enabled", True):
                try:
                    discovered_tools = await self._discover_tools_from_server_standalone(server_name)
                    if discovered_tools:
                        self.available_tools.update(discovered_tools)
                        # print_current(f"✅ Discovered {len(discovered_tools)} tools from {server_name}")
                    else:
                        # print_current(f"⚠️ No tools discovered from {server_name}")
                        pass
                except Exception as e:
                    # print_current(f"❌ Failed to discover tools from {server_name}: {e}")
                    import traceback
                    traceback.print_exc()
        
        logger.info(f"Standalone tool discovery completed: {len(self.available_tools)} tools total")
    
    async def _discover_tools_from_server(self, server_name: str) -> Dict[str, Dict[str, Any]]:
        """Discover tools from a specific server using server manager"""
        try:
            # Check if server is ready
            if not await self.server_manager.is_server_ready(server_name):
                # print_current(f"⚠️ Server {server_name} is not ready")
                return {}
            
            return await self._discover_tools_from_server_standalone(server_name)
                
        except Exception as e:
            # print_current(f"⚠️ Failed to discover tools from {server_name}: {e}")
            return {}
    
    async def _discover_tools_from_server_standalone(self, server_name: str) -> Dict[str, Dict[str, Any]]:
        """Discover tools from a specific server without server manager (standalone mode)"""
        try:
            server_config = self.servers[server_name]
            command = server_config.get("command")
            url = server_config.get("url")
            args = server_config.get("args", [])

            # Check if this is an HTTP server
            is_http_server = bool(url and not command)

            if not command and not url:
                return {}

            # print_current(f"🔍 Discovering tools from {server_name} ({'HTTP' if is_http_server else 'Command'})")

            # Create temporary FastMCP client to query the server
            from fastmcp import Client
            from fastmcp.mcp_config import MCPConfig

            if is_http_server:
                # For HTTP servers, use URL directly as transport
                transport = url
                server_config_for_fastmcp = {
                    "url": url,
                    "headers": server_config.get("headers", {}),
                    "transport": "http"
                }
            else:
                # For command-based servers, use stdio transport
                server_config_for_fastmcp = {
                    "command": command,
                    "args": args,
                    "transport": "stdio"
                }

            # Add workspace directory (cwd - current working directory for subprocess)
            if self._workspace_dir and os.path.exists(self._workspace_dir):
                server_config_for_fastmcp["cwd"] = self._workspace_dir

            # Add environment variables if they exist
            env_vars = server_config.get("env", {})

            # Auto-detect relevant environment variables
            # Look for all environment variables that contain API_KEY, TOKEN, SECRET, or KEY
            auto_env_vars = {}
            server_name_lower = server_name.lower()

            # Look for common API-related environment variables
            api_related_patterns = ['API_KEY', 'TOKEN', 'SECRET', 'KEY']

            for env_var in os.environ:
                env_var_upper = env_var.upper()
                # If environment variable contains any of the API-related patterns
                for pattern in api_related_patterns:
                    if pattern in env_var_upper:
                        auto_env_vars[env_var] = os.environ[env_var]
                        break

                # Also include environment variables that contain the server name
                if server_name_lower in env_var.lower():
                    auto_env_vars[env_var] = os.environ[env_var]

            # For HTTP servers, we don't need MCPConfig, just use URL directly
            if is_http_server:
                transport = url
                # For HTTP servers, we can pass headers directly to the Client
                client_kwargs = {}
                if server_config.get("headers"):
                    print_debug(f"⚠️ Headers configuration found for {server_name} but FastMCP Client doesn't support direct headers parameter")
                    # TODO: Find alternative way to pass headers to HTTP requests
            else:
                # For command-based servers, create MCPConfig
                # Merge config env vars with auto-detected vars
                if env_vars or auto_env_vars:
                    final_env_vars = {**auto_env_vars, **env_vars}
                    server_config_for_fastmcp["env"] = final_env_vars

                mcp_config = MCPConfig(
                    mcpServers={
                        server_name: server_config_for_fastmcp
                    }
                )
                transport = mcp_config
                client_kwargs = {}

            # Query tools using temporary client with timeout
            from contextlib import redirect_stderr

            stderr_buffer = io.StringIO()

            try:

                original_stderr = os.dup(2)

                # write logo to file
                with tempfile.NamedTemporaryFile(mode='w', delete=True) as temp_file:

                    os.dup2(temp_file.fileno(), 2)

                    try:
                        # Add timeout to prevent hanging (Python 3.10 compatible)
                        async with Client(transport, **client_kwargs) as client:
                            
                            # Track subprocess for tool discovery (temporary client)
                            # Only track subprocesses for command-based servers
                            if not is_http_server:
                                try:
                                    if hasattr(client, '_process') and client._process:
                                        temp_process_info = {
                                            'pid': client._process.pid,
                                            'command': command,
                                            'args': args,
                                            'server_name': f"{server_name}_discovery",
                                            'temporary': True
                                        }
                                        self._track_subprocess(f"{server_name}_discovery", temp_process_info)
                                    elif hasattr(client, '_transport') and hasattr(client._transport, '_proc'):
                                        proc = client._transport._proc
                                        if proc:
                                            temp_process_info = {
                                                'pid': proc.pid,
                                                'command': command,
                                                'args': args,
                                                'server_name': f"{server_name}_discovery",
                                                'temporary': True
                                            }
                                            self._track_subprocess(f"{server_name}_discovery", temp_process_info)
                                except Exception as track_error:
                                    # Silently ignore tracking errors for discovery
                                    pass

                            # Use asyncio.wait_for for Python 3.10 compatibility
                            tools = await asyncio.wait_for(client.list_tools(), timeout=10)
                    finally:

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
                import traceback
                traceback.print_exc()
                return {}
                
        except Exception as e:
            print_current(f"⚠️ Failed to discover tools from {server_name}: {e}")
            import traceback
            traceback.print_exc()
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
        """Call a tool in standalone mode with persistent client caching"""
        try:
            server_config = self.servers[server_name]
            command = server_config.get("command")
            url = server_config.get("url")
            args = server_config.get("args", [])

            # Check if this is an HTTP server
            is_http_server = bool(url and not command)

            if not command and not url:
                return {"status": "failed", "error": f"No command or URL configured for server {server_name}"}

            # Check if we have a persistent client for this server
            need_new_client = (server_name not in self._persistent_clients or
                             not self._persistent_clients[server_name].get('entered', False))

            if need_new_client:
                # Create new persistent FastMCP client
                from fastmcp import Client
                from fastmcp.mcp_config import MCPConfig

                if is_http_server:
                    # For HTTP servers, use URL directly as transport
                    transport = url
                    client_kwargs = {}
                    # Note: FastMCP Client doesn't support headers parameter directly
                    # Headers will be handled by the underlying HTTP client if needed
                    if server_config.get("headers"):
                        print_debug(f"⚠️ Headers configuration found for {server_name} but FastMCP Client doesn't support direct headers parameter")
                        # TODO: Find alternative way to pass headers to HTTP requests
                else:
                    # For command-based servers, create MCPConfig
                    server_config_for_fastmcp = {
                        "command": command,
                        "args": args,
                        "transport": "stdio"
                    }

                    # Add workspace directory (cwd - current working directory for subprocess)
                    if self._workspace_dir and os.path.exists(self._workspace_dir):
                        server_config_for_fastmcp["cwd"] = self._workspace_dir

                    # Add environment variables if they exist
                    env_vars = server_config.get("env", {})

                    # Auto-detect relevant environment variables
                    # Look for all environment variables that contain API_KEY, TOKEN, SECRET, or KEY
                    auto_env_vars = {}
                    server_name_lower = server_name.lower()

                    # Look for common API-related environment variables
                    api_related_patterns = ['API_KEY', 'TOKEN', 'SECRET', 'KEY']

                    for env_var in os.environ:
                        env_var_upper = env_var.upper()
                        # If environment variable contains any of the API-related patterns
                        for pattern in api_related_patterns:
                            if pattern in env_var_upper:
                                auto_env_vars[env_var] = os.environ[env_var]
                                break

                        # Also include environment variables that contain the server name
                        if server_name_lower in env_var.lower():
                            auto_env_vars[env_var] = os.environ[env_var]

                    # Merge config env vars with auto-detected vars
                    if env_vars or auto_env_vars:
                        final_env_vars = {**auto_env_vars, **env_vars}
                        server_config_for_fastmcp["env"] = final_env_vars

                    mcp_config = MCPConfig(
                        mcpServers={
                            server_name: server_config_for_fastmcp
                        }
                    )
                    transport = mcp_config
                    client_kwargs = {}

                # Create and initialize persistent client
                client = Client(transport, **client_kwargs)
                try:
                    await client.__aenter__()
                    self._persistent_clients[server_name] = {
                        'client': client,
                        'entered': True
                    }

                    # Track subprocess information for cleanup (only for command-based servers)
                    if not is_http_server:
                        try:
                            # Try to get process information from the client
                            # Note: This is implementation-dependent and may need adjustment based on FastMCP internals
                            if hasattr(client, '_process') and client._process:
                                process_info = {
                                    'pid': client._process.pid,
                                    'command': command,
                                    'args': args,
                                    'server_name': server_name
                                }
                                self._track_subprocess(server_name, process_info)
                            elif hasattr(client, '_transport') and hasattr(client._transport, '_proc'):
                                # Alternative way to get process info
                                proc = client._transport._proc
                                if proc:
                                    process_info = {
                                        'pid': proc.pid,
                                        'command': command,
                                        'args': args,
                                        'server_name': server_name
                                    }
                                    self._track_subprocess(server_name, process_info)
                        except Exception as track_error:
                            # print_current(f"⚠️ Could not track subprocess for {server_name}: {track_error}")
                            pass

                    # print_current(f"🔗 Created and connected persistent MCP client for server: {server_name}")
                except Exception as e:
                    # print_current(f"❌ Failed to connect persistent client for {server_name}: {e}")
                    pass
            
            client_info = self._persistent_clients[server_name]
            client = client_info['client']
            
            # Call tool using persistent client with stderr redirection
            # 保存原始的stderr
            original_stderr = os.dup(2)
            
            try:
                # 创建临时文件来重定向stderr
                with tempfile.NamedTemporaryFile(mode='w', delete=True) as temp_file:
                    # 重定向stderr到临时文件
                    os.dup2(temp_file.fileno(), 2)
                    
                    try:
                        # Check if client is still connected, if not, reconnect
                        try:
                            # Call the specific tool using the persistent connection
                            tool_result = await asyncio.wait_for(
                                client.call_tool(tool_name, arguments), 
                                timeout=300
                            )
                            
                            # print_current(f"✅ Tool call successful on persistent connection: {tool_name}")
                            return {
                                "status": "success",
                                "result": tool_result
                            }
                        except Exception as call_error:
                            # If the call fails due to connection issues, try to reconnect
                            error_str = str(call_error).lower()
                            if "not connected" in error_str or "connection" in error_str:
                                # print_current(f"🔄 Reconnecting client for {server_name} due to connection issue")
                                # Clean up the old client
                                try:
                                    await client.__aexit__(None, None, None)
                                except:
                                    pass
                                
                                # Create new client and reconnect
                                from fastmcp import Client
                                from fastmcp.mcp_config import MCPConfig
                                
                                server_config_for_new_client = {
                                    "command": command,
                                    "args": args,
                                    "transport": "stdio"
                                }

                                # Add workspace directory (cwd - current working directory for subprocess)
                                if self._workspace_dir and os.path.exists(self._workspace_dir):
                                    server_config_for_new_client["cwd"] = self._workspace_dir

                                env_vars = server_config.get("env", {})

                                # Auto-detect relevant environment variables
                                # Look for all environment variables that contain API_KEY, TOKEN, SECRET, or KEY
                                auto_env_vars = {}
                                server_name_lower = server_name.lower()

                                # Look for common API-related environment variables
                                api_related_patterns = ['API_KEY', 'TOKEN', 'SECRET', 'KEY']

                                for env_var in os.environ:
                                    env_var_upper = env_var.upper()
                                    # If environment variable contains any of the API-related patterns
                                    for pattern in api_related_patterns:
                                        if pattern in env_var_upper:
                                            auto_env_vars[env_var] = os.environ[env_var]
                                            break

                                    # Also include environment variables that contain the server name
                                    if server_name_lower in env_var.lower():
                                        auto_env_vars[env_var] = os.environ[env_var]

                                # Merge config env vars with auto-detected vars
                                if env_vars or auto_env_vars:
                                    final_env_vars = {**auto_env_vars, **env_vars}
                                    server_config_for_new_client["env"] = final_env_vars
                                
                                mcp_config = MCPConfig(
                                    mcpServers={
                                        server_name: server_config_for_new_client
                                    }
                                )
                                
                                new_client = Client(mcp_config)
                                await new_client.__aenter__()
                                self._persistent_clients[server_name] = {
                                    'client': new_client,
                                    'entered': True
                                }

                                # Track subprocess for the new client
                                try:
                                    if hasattr(new_client, '_process') and new_client._process:
                                        process_info = {
                                            'pid': new_client._process.pid,
                                            'command': command,
                                            'args': args,
                                            'server_name': server_name
                                        }
                                        self._track_subprocess(server_name, process_info)
                                    elif hasattr(new_client, '_transport') and hasattr(new_client._transport, '_proc'):
                                        proc = new_client._transport._proc
                                        if proc:
                                            process_info = {
                                                'pid': proc.pid,
                                                'command': command,
                                                'args': args,
                                                'server_name': server_name
                                            }
                                            self._track_subprocess(server_name, process_info)
                                except Exception as track_error:
                                    # print_current(f"⚠️ Could not track subprocess for reconnected {server_name}: {track_error}")
                                    pass
                                
                                # Retry the tool call with new client
                                tool_result = await asyncio.wait_for(
                                    new_client.call_tool(tool_name, arguments), 
                                    timeout=300
                                )
                                
                                # print_current(f"✅ Tool call successful after reconnection: {tool_name}")
                                return {
                                    "status": "success",
                                    "result": tool_result
                                }
                            else:
                                # Re-raise non-connection errors
                                raise call_error
                    finally:
                        # 恢复原始的stderr
                        os.dup2(original_stderr, 2)
                        os.close(original_stderr)
                        
            except asyncio.TimeoutError:
                # print_current(f"⏰ Tool call timeout for {tool_name} (300s)")
                return {"status": "failed", "error": f"Tool call timeout for {tool_name}"}
            except Exception as e:
                # print_current(f"❌ Tool call error for {tool_name}: {e}")
                # All servers are treated as stateful - be careful about connection cleanup
                error_str = str(e).lower()
                if "not connected" in error_str or "connection" in error_str or "broken pipe" in error_str:
                    # print_current(f"⚠️ Connection error on server {server_name}, will attempt automatic reconnection")
                    # For all servers, attempt reconnection to preserve state
                    if server_name in self._persistent_clients:
                        try:
                            # Clean up the old client
                            client_info = self._persistent_clients[server_name]
                            if client_info['entered']:
                                await client_info['client'].__aexit__(None, None, None)
                            del self._persistent_clients[server_name]
                            
                            # Attempt immediate reconnection for all servers
                            # print_current(f"🔄 Attempting immediate reconnection for server {server_name}")
                            # This will be handled by the reconnection logic in the next call
                            
                        except Exception as cleanup_e:
                            # print_current(f"⚠️ Error during server cleanup: {cleanup_e}")
                            pass
                
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

                # print_current(f"🚀 Calling tool: {tool_name} on persistent server: {server_name}")
                result = await self.server_manager.call_server_tool(server_name, original_tool_name, arguments)
            else:
                # Fallback to standalone mode without server manager
                # print_current(f"🚀 Calling tool: {tool_name} in standalone mode")
                result = await self._call_tool_standalone(server_name, original_tool_name, arguments)
            
            if result.get("status") == "success":
                # print_current(f"✅ Persistent server call successful: {tool_name}")
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
            # print_current(f"❌ Tool call failed for {tool_name}: {error_msg}")
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
        """Call a tool synchronously with persistent connection support"""
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

        # Get server information
        server_name, original_tool_name = self._get_server_tool_info(tool_name)
        if not server_name:
            return {
                "status": "failed",
                "error": f"Tool {tool_name} not found",
                "tool_name": tool_name,
                "arguments": arguments
            }

        # All servers are now treated as stateful for maximum reliability
        # Use shared event loop to maintain persistent connections for all servers
        return self._call_tool_stateful_sync(tool_name, arguments)

    def _call_tool_stateful_sync(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Call a tool using shared event loop (all servers are treated as stateful)"""
        try:
            # Ensure shared loop is available
            if not self._ensure_shared_loop():
                return {
                    "status": "failed",
                    "error": "Shared event loop not available",
                    "tool_name": tool_name,
                    "arguments": arguments
                }
            
            # Submit the coroutine to the shared event loop
            future = asyncio.run_coroutine_threadsafe(
                self.call_tool(tool_name, arguments), 
                self._shared_loop
            )
            
            # Wait for result with timeout
            result = future.result(timeout=300)  # 5 minute timeout
            return result
                
        except Exception as e:
            return {
                "status": "failed",
                "error": f"Stateful tool call failed: {e}",
                "tool_name": tool_name,
                "arguments": arguments
            }

    def _ensure_shared_loop(self) -> bool:
        """Ensure shared event loop is available and running"""
        with self._loop_lock:
            if self._shared_loop and not self._shared_loop.is_closed():
                return True
            
            # Reinitialize if needed
            try:
                import time
                import threading
                
                loop_ready = threading.Event()
                
                def run_shared_loop():
                    """Run the shared event loop in a separate thread"""
                    self._shared_loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(self._shared_loop)
                    loop_ready.set()  # Signal that loop is ready
                    try:
                        self._shared_loop.run_forever()
                    except Exception as e:
                        print_current(f"⚠️ Shared event loop error: {e}")
                    finally:
                        self._shared_loop.close()
                
                self._shared_thread = threading.Thread(target=run_shared_loop, daemon=True)
                self._shared_thread.start()
                
                # Wait for loop to be ready
                if loop_ready.wait(timeout=5.0):
                    # print_current("🔄 Shared event loop re-initialized for all servers")
                    return True
                else:
                    # print_current("⚠️ Failed to initialize shared event loop (timeout)")
                    return False
                    
            except Exception as e:
                # print_current(f"⚠️ Error initializing shared event loop: {e}")
                return False
    
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
    
    def set_workspace_dir(self, workspace_dir: str):
        """Set custom workspace directory for MCP servers"""
        if os.path.exists(workspace_dir) and os.path.isdir(workspace_dir):
            self._workspace_dir = workspace_dir
            print_current(f"📂 Custom workspace directory set: {workspace_dir}")
        else:
            print_current(f"⚠️ Invalid workspace directory: {workspace_dir}")

    def get_workspace_dir(self) -> str:
        """Get current workspace directory"""
        return self._workspace_dir

    def get_status(self) -> Dict[str, Any]:
        """Get client status"""
        return {
            "initialized": self.initialized,
            "servers": list(self.servers.keys()),
            "total_tools": len(self.available_tools),
            "config_path": self.config_path,
            "workspace_dir": self._workspace_dir,
            "fastmcp_available": FASTMCP_AVAILABLE,
            "server_manager_available": self.server_manager is not None,
            "tracked_processes": len(self._tracked_processes),
            "server_processes": len(self._server_processes)
        }
    
    async def cleanup(self):
        """Cleanup resources gracefully"""
        try:
            # print_current("🔄 Starting FastMCP client cleanup...")

            # First, clear tool references to prevent new calls
            self.available_tools.clear()
            self.servers.clear()

            # Critical: Force kill all tracked processes first
            try:
                self._cleanup_server_processes()
            except Exception as e:
                # print_current(f"⚠️ Error during process cleanup: {e}")
                pass

            # If we have a server manager reference, let it know we're cleaning up
            if self.server_manager:
                try:
                    # The server manager will handle its own cleanup
                    # We just need to clear our reference
                    self.server_manager = None
                except Exception as e:
                    # print_current(f"⚠️ Error clearing server manager reference: {e}")
                    pass

                        # Stop health monitoring
            self._stop_health_monitoring()

            # Mark as not initialized
            self.initialized = False

            # Give a small delay to allow any pending operations to complete
            await asyncio.sleep(0.1)

            # print_current("🔌 FastMCP client cleaned up")

        except Exception as e:
            # print_current(f"⚠️ FastMCP cleanup error: {e}")
            # Continue with cleanup even if there are errors
            pass


# Global instance with thread safety
_fastmcp_wrapper = None
_fastmcp_config_path = None
_fastmcp_lock = threading.RLock()


def get_fastmcp_wrapper(config_path: str = "config/mcp_servers.json", workspace_dir: Optional[str] = None) -> FastMcpWrapper:
    """Get FastMCP wrapper instance (thread-safe)"""
    global _fastmcp_wrapper, _fastmcp_config_path

    # Try to get agent-specific wrapper first
    try:
        from .agent_context import get_current_agent_id, get_agent_fastmcp_wrapper
        current_agent_id = get_current_agent_id()
        
        if current_agent_id:
            agent_wrapper = get_agent_fastmcp_wrapper(current_agent_id)
            if agent_wrapper and agent_wrapper.initialized:
                return agent_wrapper
    except Exception as e:
        # Silently ignore agent context errors, fall back to global wrapper
        pass

    with _fastmcp_lock:
        # For custom config paths (containing timestamps), we should reuse existing instances
        # to avoid creating new instances for each run
        is_custom_config = 'mcp_servers_custom_' in config_path
        is_standard_config = config_path == "config/mcp_servers.json" or config_path.endswith("/config/mcp_servers.json")
        
        # Priority 1: If we have an initialized instance with tools, always try to reuse it first
        # This ensures that even if config paths are different, we can reuse a working instance
        if (_fastmcp_wrapper and _fastmcp_wrapper.initialized and 
            hasattr(_fastmcp_wrapper, 'available_tools') and len(_fastmcp_wrapper.available_tools) > 0):
            return _fastmcp_wrapper

        # Check if we need to create a new instance
        need_new_instance = False
        
        if _fastmcp_wrapper is None:
            # No existing instance, create new one
            need_new_instance = True
        elif _fastmcp_config_path != config_path:
            # Config path changed - this means a different agent with different config
            # But if we have an initialized instance with tools, we can still reuse it
            if (_fastmcp_wrapper and _fastmcp_wrapper.initialized and 
                hasattr(_fastmcp_wrapper, 'available_tools') and len(_fastmcp_wrapper.available_tools) > 0):
                return _fastmcp_wrapper
            else:
                # Create new instance for different config files
                need_new_instance = True
        
        if need_new_instance:
            _fastmcp_wrapper = FastMcpWrapper(config_path, workspace_dir)
            _fastmcp_config_path = config_path
        
        return _fastmcp_wrapper


@asynccontextmanager
async def initialize_fastmcp_with_server_manager(config_path: str = "config/mcp_servers.json", workspace_dir: Optional[str] = None):
    """Initialize FastMCP wrapper with persistent server manager"""
    global _fastmcp_wrapper

    # Create wrapper instance
    with _fastmcp_lock:
        wrapper = get_fastmcp_wrapper(config_path, workspace_dir)
    
    # Use MCP operation context for structured concurrency
    async with mcp_operation_context(config_path) as server_manager:
        try:
            # Set server manager reference
            wrapper.server_manager = server_manager
            
            # Initialize wrapper
            result = await wrapper.initialize()
            if not result:
                raise Exception("FastMCP wrapper initialization failed")
            
            # print_current(f"✅ FastMCP wrapper initialized with persistent server manager")
            yield wrapper
            
        finally:
            # Clean up
            wrapper.server_manager = None
            # print_current("🔄 FastMCP wrapper context exiting...")


async def initialize_fastmcp_wrapper(config_path: str = "config/mcp_servers.json", workspace_dir: Optional[str] = None) -> bool:
    """Initialize FastMCP wrapper (backward compatibility)"""
    global _fastmcp_wrapper, _fastmcp_config_path

    try:
        # Create wrapper instance
        with _fastmcp_lock:
            wrapper = get_fastmcp_wrapper(config_path, workspace_dir)
            # Ensure global variables are updated
            _fastmcp_wrapper = wrapper
            _fastmcp_config_path = config_path
        
        # For backward compatibility, we'll initialize without server manager first
        # This allows basic functionality without requiring the full server manager context
        if not wrapper.initialized:
            # Load configuration
            await wrapper._load_config()

            # Try to discover tools using standalone tool discovery (without server manager)
            try:
                await wrapper._discover_tools_standalone()
                # print_current(f"✅ FastMCP wrapper initialized with {len(wrapper.available_tools)} tools discovered")
            except Exception as tool_discovery_error:
                # print_current(f"⚠️ FastMCP wrapper basic initialization completed, tool discovery will retry later: {tool_discovery_error}")
                pass

            # Mark as initialized
            wrapper.initialized = True
        
        # Register wrapper to agent context if available
        try:
            from .agent_context import get_current_agent_id, set_agent_fastmcp_wrapper
            current_agent_id = get_current_agent_id()
            if current_agent_id and wrapper.initialized:
                set_agent_fastmcp_wrapper(current_agent_id, wrapper)
        except Exception as e:
            # Silently ignore agent context errors
            pass
        
        return True
    except Exception as e:
        logger.error(f"FastMCP wrapper initialization failed: {e}")
        # print_current(f"❌ FastMCP wrapper initialization failed: {e}")
        import traceback
        traceback.print_exc()
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
            # Critical: Force kill processes first
            try:
                if hasattr(_fastmcp_wrapper, '_cleanup_server_processes'):
                    _fastmcp_wrapper._cleanup_server_processes()
            except Exception:
                # If that fails, try force kill
                try:
                    if hasattr(_fastmcp_wrapper, '_force_kill_processes'):
                        _fastmcp_wrapper._force_kill_processes()
                except Exception:
                    pass

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


# Enhanced cleanup with signal handling
def _signal_cleanup(signum=None, frame=None):
    """Signal handler for clean shutdown on various signals"""
    signal_name = "unknown"
    if signum:
        try:
            signal_name = signal.Signals(signum).name
        except:
            signal_name = str(signum)

    # print_current(f"🛑 Received signal {signal_name}, performing emergency cleanup...")

    try:
        # Force kill all tracked processes first (most critical)
        global _fastmcp_wrapper
        if _fastmcp_wrapper and hasattr(_fastmcp_wrapper, '_force_kill_processes'):
            try:
                _fastmcp_wrapper._force_kill_processes()
            except Exception as e:
                # print_current(f"⚠️ Signal cleanup process kill failed: {e}")
                pass

        # Then perform normal cleanup
        safe_cleanup_fastmcp_wrapper()
    except Exception as e:
        # print_current(f"⚠️ Signal cleanup failed: {e}")
        pass

    # Re-raise the signal to allow normal exit behavior
    if signum:
        os._exit(128 + signum)  # Exit with signal code

# Register signal handlers for common termination signals
def _register_signal_handlers():
    """Register signal handlers for clean shutdown"""
    signals_to_handle = [
        signal.SIGTERM,  # Termination signal
        signal.SIGINT,   # Interrupt (Ctrl+C)
    ]

    # SIGHUP is Unix-only, check if it exists before adding it
    if hasattr(signal, 'SIGHUP'):
        signals_to_handle.append(signal.SIGHUP)  # Hangup

    # Only register signals that are available on this platform
    for sig in signals_to_handle:
        try:
            signal.signal(sig, _signal_cleanup)
        except (ValueError, OSError):
            # Signal not available on this platform
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

# Register signal handlers
_register_signal_handlers()


# Test function for FastMCP wrapper
async def test_fastmcp_wrapper():
    """Test FastMCP wrapper functionality"""
    # print_current("🧪 Starting FastMCP wrapper test...")
    
    config_path = "config/mcp_servers.json"
    
    # Test with server manager
    async with initialize_fastmcp_with_server_manager(config_path) as wrapper:
        # Test status
        # print_current("📊 Testing status...")
        status = wrapper.get_status()
        # print_current(f"Status: {json.dumps(status, indent=2)}")
        
        # Test available tools
        # print_current("🔧 Testing available tools...")
        tools = wrapper.get_available_tools()
        # print_current(f"Available tools: {tools}")
        
        if tools:
            # Test tool info
            first_tool = tools[0]
            # print_current(f"📋 Testing tool info for: {first_tool}")
            tool_info = wrapper.get_tool_info(first_tool)
            # print_current(f"Tool info: {json.dumps(tool_info, indent=2)}")
            
            # Test tool definition
            # print_current(f"📝 Testing tool definition for: {first_tool}")
            tool_def = wrapper.get_tool_info(first_tool)
            # print_current(f"Tool definition: {json.dumps(tool_def, indent=2)}")
    
    # print_current("✅ FastMCP wrapper test completed!")


def test_fastmcp_wrapper_sync():
    """Synchronous test wrapper"""
    asyncio.run(test_fastmcp_wrapper())


if __name__ == "__main__":
    """Test FastMCP wrapper when run directly"""
    test_fastmcp_wrapper_sync()