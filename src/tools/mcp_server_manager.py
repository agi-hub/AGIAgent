#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MCP Server Manager
Manages persistent MCP server processes with proper lifecycle and signal handling
"""

import asyncio
import json
import os
import signal
import threading
import weakref
from contextlib import asynccontextmanager
from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass
from enum import Enum
import logging

from .print_system import print_current, print_debug, print_error

logger = logging.getLogger(__name__)


class ServerState(Enum):
    """MCP Server states"""
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    FAILED = "failed"


@dataclass
class MCPServerProcess:
    """MCP Server process information"""
    name: str
    command: List[str]
    env: Dict[str, str]
    process: Optional[asyncio.subprocess.Process] = None
    state: ServerState = ServerState.STOPPED
    start_time: Optional[float] = None
    failure_count: int = 0
    max_failures: int = 3


class MCPServerManager:
    """
    Persistent MCP Server Manager
    
    This manager keeps MCP servers running persistently and provides
    proper lifecycle management with signal-based shutdown.
    """
    
    def __init__(self, config_path: str = "config/mcp_servers.json"):
        self.config_path = config_path
        self.servers: Dict[str, MCPServerProcess] = {}
        self.shutdown_event = asyncio.Event()
        self._task_group: Optional[asyncio.TaskGroup] = None
        self._management_tasks: Set[asyncio.Task] = set()
        self._lock = asyncio.Lock()
        
        # Signal handling
        self._original_sigint_handler = None
        self._original_sigterm_handler = None
        self._shutdown_callbacks: List[callable] = []
        
    async def __aenter__(self):
        """Async context manager entry"""
        await self.start()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.shutdown()
        
    async def start(self):
        """Start the MCP server manager"""
        try:
            print_current("🚀 Starting MCP Server Manager...")
            
            # Setup signal handlers
            self._setup_signal_handlers()
            
            # Load configuration
            await self._load_configuration()
            
            # Start all configured servers
            async with asyncio.TaskGroup() as tg:
                self._task_group = tg
                
                for server_name, server in self.servers.items():
                    if server.state == ServerState.STOPPED:
                        task = tg.create_task(self._start_server(server_name))
                        self._management_tasks.add(task)
                        task.add_done_callback(self._management_tasks.discard)
            
            print_current(f"✅ MCP Server Manager started with {len(self.servers)} servers")
            
        except Exception as e:
            logger.error(f"Failed to start MCP Server Manager: {e}")
            raise
            
    async def shutdown(self):
        """Shutdown all MCP servers gracefully"""
        print_current("🔄 Shutting down MCP Server Manager...")
        
        # Signal shutdown to all components
        self.shutdown_event.set()
        
        # Cancel all management tasks
        for task in list(self._management_tasks):
            if not task.done():
                task.cancel()
                
        # Wait for tasks to complete
        if self._management_tasks:
            await asyncio.gather(*self._management_tasks, return_exceptions=True)
            
        # Stop all servers
        async with self._lock:
            stop_tasks = []
            for server_name in list(self.servers.keys()):
                stop_tasks.append(self._stop_server(server_name))
                
            if stop_tasks:
                await asyncio.gather(*stop_tasks, return_exceptions=True)
        
        # Restore signal handlers
        self._restore_signal_handlers()
        
        # Run shutdown callbacks
        for callback in self._shutdown_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback()
                else:
                    callback()
            except Exception as e:
                print_debug(f"⚠️ Error in shutdown callback: {e}")
        
        print_current("✅ MCP Server Manager shut down")
        
    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown"""
        try:
            # Store original handlers
            self._original_sigint_handler = signal.signal(signal.SIGINT, self._signal_handler)
            self._original_sigterm_handler = signal.signal(signal.SIGTERM, self._signal_handler)
            print_debug("📡 Signal handlers installed")
        except ValueError:
            # Not running in main thread, signal handling not available
            print_debug("⚠️ Signal handling not available (not in main thread)")
    
    def _restore_signal_handlers(self):
        """Restore original signal handlers"""
        try:
            if self._original_sigint_handler is not None:
                signal.signal(signal.SIGINT, self._original_sigint_handler)
            if self._original_sigterm_handler is not None:
                signal.signal(signal.SIGTERM, self._original_sigterm_handler)
            print_debug("📡 Signal handlers restored")
        except ValueError:
            pass
            
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        print_current(f"📡 Received signal {signum}, initiating shutdown...")
        
        # Schedule shutdown in the event loop
        loop = asyncio.get_running_loop()
        if loop.is_running():
            loop.create_task(self.shutdown())
        else:
            asyncio.run(self.shutdown())
    
    async def _load_configuration(self):
        """Load MCP server configuration"""
        try:
            if not os.path.exists(self.config_path):
                print_debug(f"⚠️ Configuration file not found: {self.config_path}")
                return
                
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            servers_config = config.get("mcpServers", {})
            
            for server_name, server_config in servers_config.items():
                # Only handle command-based servers (NPX/NPM format)
                if not server_config.get("command"):
                    continue
                    
                # Skip disabled servers
                if not server_config.get("enabled", True):
                    continue
                    
                # Build command
                command = [server_config["command"]] + server_config.get("args", [])
                env = os.environ.copy()
                env.update(server_config.get("env", {}))
                
                server_process = MCPServerProcess(
                    name=server_name,
                    command=command,
                    env=env,
                    max_failures=server_config.get("max_failures", 3)
                )
                
                self.servers[server_name] = server_process
                
            print_current(f"📋 Loaded configuration for {len(self.servers)} MCP servers")
            
        except Exception as e:
            logger.error(f"Failed to load MCP configuration: {e}")
            raise
    
    async def _start_server(self, server_name: str) -> bool:
        """Start a specific MCP server"""
        async with self._lock:
            server = self.servers.get(server_name)
            if not server:
                print_error(f"❌ Server {server_name} not found")
                return False
                
            if server.state in [ServerState.STARTING, ServerState.RUNNING]:
                print_debug(f"📋 Server {server_name} already starting/running")
                return True
                
            server.state = ServerState.STARTING
            
        try:
            print_current(f"🚀 Starting MCP server: {server_name}")
            
            # Start the process
            process = await asyncio.create_subprocess_exec(
                *server.command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                stdin=asyncio.subprocess.PIPE,
                env=server.env
            )
            
            async with self._lock:
                server.process = process
                server.state = ServerState.RUNNING
                server.start_time = asyncio.get_running_loop().time()
                server.failure_count = 0
            
            print_current(f"✅ MCP server {server_name} started (PID: {process.pid})")
            
            # Monitor the process
            monitor_task = asyncio.create_task(self._monitor_server(server_name))
            self._management_tasks.add(monitor_task)
            monitor_task.add_done_callback(self._management_tasks.discard)
            
            return True
            
        except Exception as e:
            async with self._lock:
                server.state = ServerState.FAILED
                server.failure_count += 1
                
            print_error(f"❌ Failed to start MCP server {server_name}: {e}")
            
            # Try to restart if not exceeded max failures
            if server.failure_count < server.max_failures:
                print_current(f"🔄 Retrying server {server_name} in 5 seconds...")
                await asyncio.sleep(5)
                return await self._start_server(server_name)
                
            return False
    
    async def _stop_server(self, server_name: str) -> bool:
        """Stop a specific MCP server"""
        async with self._lock:
            server = self.servers.get(server_name)
            if not server or not server.process:
                return True
                
            if server.state in [ServerState.STOPPED, ServerState.STOPPING]:
                return True
                
            server.state = ServerState.STOPPING
            process = server.process
            
        try:
            print_current(f"🛑 Stopping MCP server: {server_name}")
            
            # Try graceful termination first
            process.terminate()
            
            try:
                await asyncio.wait_for(process.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                # Force kill if graceful termination failed
                print_debug(f"⚠️ Force killing MCP server: {server_name}")
                process.kill()
                await process.wait()
            
            async with self._lock:
                server.process = None
                server.state = ServerState.STOPPED
                
            print_current(f"✅ MCP server {server_name} stopped")
            return True
            
        except Exception as e:
            print_error(f"❌ Error stopping MCP server {server_name}: {e}")
            return False
    
    async def _monitor_server(self, server_name: str):
        """Monitor a server process for unexpected termination"""
        try:
            server = self.servers.get(server_name)
            if not server or not server.process:
                return
                
            # Wait for process to exit
            returncode = await server.process.wait()
            
            async with self._lock:
                if server.state == ServerState.STOPPING:
                    # Expected termination
                    return
                    
                print_error(f"❌ MCP server {server_name} exited unexpectedly (code: {returncode})")
                server.state = ServerState.FAILED
                server.failure_count += 1
                server.process = None
            
            # Auto-restart if not shutting down and under failure limit
            if not self.shutdown_event.is_set() and server.failure_count < server.max_failures:
                print_current(f"🔄 Auto-restarting MCP server: {server_name}")
                await asyncio.sleep(2)  # Brief delay before restart
                await self._start_server(server_name)
                
        except Exception as e:
            print_error(f"❌ Error monitoring MCP server {server_name}: {e}")
    
    def get_server_status(self, server_name: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific server"""
        server = self.servers.get(server_name)
        if not server:
            return None
            
        return {
            "name": server.name,
            "state": server.state.value,
            "pid": server.process.pid if server.process else None,
            "start_time": server.start_time,
            "failure_count": server.failure_count,
            "max_failures": server.max_failures
        }
    
    def get_all_server_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all servers"""
        return {
            name: self.get_server_status(name)
            for name in self.servers.keys()
        }
    
    def add_shutdown_callback(self, callback: callable):
        """Add a callback to be called during shutdown"""
        self._shutdown_callbacks.append(callback)
    
    async def is_server_ready(self, server_name: str) -> bool:
        """Check if a server is ready to handle requests"""
        server = self.servers.get(server_name)
        return server is not None and server.state == ServerState.RUNNING
    
    async def wait_for_server(self, server_name: str, timeout: float = 30.0) -> bool:
        """Wait for a server to be ready"""
        start_time = asyncio.get_running_loop().time()
        
        while asyncio.get_running_loop().time() - start_time < timeout:
            if await self.is_server_ready(server_name):
                return True
            await asyncio.sleep(0.1)
            
        return False


# Global server manager instance
_global_server_manager: Optional[MCPServerManager] = None
_manager_lock = threading.Lock()


@asynccontextmanager
async def get_mcp_server_manager(config_path: str = "config/mcp_servers.json"):
    """Get or create global MCP server manager with proper lifecycle"""
    global _global_server_manager
    
    with _manager_lock:
        if _global_server_manager is None:
            _global_server_manager = MCPServerManager(config_path)
    
    async with _global_server_manager:
        yield _global_server_manager


async def cleanup_global_server_manager():
    """Cleanup global server manager"""
    global _global_server_manager
    
    if _global_server_manager:
        await _global_server_manager.shutdown()
        _global_server_manager = None


# Structured concurrency context manager for MCP operations
@asynccontextmanager
async def mcp_operation_context(config_path: str = "config/mcp_servers.json"):
    """
    Structured concurrency context for MCP operations
    
    This ensures all MCP servers are properly managed within the context
    and cleaned up when the context exits.
    """
    async with get_mcp_server_manager(config_path) as manager:
        try:
            # Wait for all servers to be ready
            ready_servers = []
            for server_name in manager.servers.keys():
                if await manager.wait_for_server(server_name, timeout=10.0):
                    ready_servers.append(server_name)
                else:
                    print_error(f"⚠️ Server {server_name} failed to start within timeout")
            
            print_current(f"✅ {len(ready_servers)} MCP servers ready for operations")
            yield manager
            
        finally:
            print_current("🔄 MCP operation context exiting...")


if __name__ == "__main__":
    """Test the MCP server manager"""
    
    async def test_server_manager():
        print_current("🧪 Testing MCP Server Manager...")
        
        async with mcp_operation_context() as manager:
            print_current("📊 Server status:")
            status = manager.get_all_server_status()
            for name, info in status.items():
                print_current(f"  {name}: {info['state']} (PID: {info['pid']})")
            
            # Simulate some work
            await asyncio.sleep(2)
            
            print_current("✅ Test completed")
    
    asyncio.run(test_server_manager())