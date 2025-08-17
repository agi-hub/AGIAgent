#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright (c) 2025 AGI Bot Research Group.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import os
import sys
import time
import json
import threading
import queue
import importlib
from typing import Dict, Any, Optional, List
from datetime import datetime
from src.tools.agent_context import get_current_agent_id
from .print_system import print_current, print_error, print_debug, set_output_directory

# Import print system module
ps_mod = importlib.import_module('src.tools.print_system')

# Global round synchronization manager - singleton pattern
class GlobalRoundSyncManager:
    """Global round synchronization manager, ensuring only one synchronization thread runs"""
    _instance = None
    _lock = threading.Lock()
    _thread = None
    _active = False
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not hasattr(self, '_initialized'):
            self._initialized = True
            self._workspace_root = None
            self._debug_mode = False
    
    def start(self, workspace_root: str = None, debug_mode: bool = False):
        """Start global synchronization manager"""
        with self._lock:
            if not self._active:
                try:
                    from src.config_loader import get_enable_round_sync
                    if get_enable_round_sync():
                        self._workspace_root = workspace_root
                        self._debug_mode = debug_mode
                        self._active = True
                        self._thread = threading.Thread(target=self._round_sync_manager_loop, name="RoundSyncManager", daemon=True)
                        self._thread.start()
                        if debug_mode:
                            print_debug("🌐 Global RoundSyncManager started")
                except Exception as e:
                    if debug_mode:
                        print_debug(f"⚠️ Failed to start Global RoundSyncManager: {e}")
    
    def stop(self):
        """Stop global synchronization manager"""
        with self._lock:
            self._active = False
    
    def is_active(self):
        """Check if synchronization manager is active"""
        return self._active
    
    def get_status(self):
        """Get synchronization manager status"""
        return {
            "active": self._active,
            "workspace_root": self._workspace_root,
            "debug_mode": self._debug_mode,
            "thread_alive": self._thread.is_alive() if self._thread else False
        }
    
    def _round_sync_manager_loop(self):
        """Global synchronization manager loop - same as original logic"""
        try:
            from src.config_loader import get_sync_round
            import json
            base_dir = self._workspace_root
            if base_dir and os.path.basename(base_dir) == 'workspace':
                base_dir = os.path.dirname(base_dir)
            if not base_dir:
                base_dir = os.getcwd()
            signal_file = os.path.join(base_dir, '.agibot_round_sync.signal')
            sync_epoch = 0
            # manager loop
            while self._active:
                try:
                    time.sleep(0.1)
                    # collect registered agents from message router
                    try:
                        from .message_system import get_message_router
                        router = get_message_router(self._workspace_root, cleanup_on_init=False)
                        agents = [a for a in router.get_all_agents() if a != 'manager']
                    except Exception:
                        agents = []
                    if not agents:
                        continue
                    # check each agent status file; ignore finished agents (based on status only)
                    considered_agents = []
                    waiting_flags = []
                    for agent_id in agents:
                        status_file = None
                        if self._workspace_root:
                            if os.path.basename(self._workspace_root) == 'workspace':
                                status_file = os.path.join(os.path.dirname(self._workspace_root), f'.agibot_spawn_{agent_id}_status.json')
                            else:
                                status_file = os.path.join(self._workspace_root, f'.agibot_spawn_{agent_id}_status.json')
                        if not status_file or not os.path.exists(status_file):
                            # no file yet → not started; skip from consideration to avoid deadlock
                            continue
                        try:
                            with open(status_file, 'r', encoding='utf-8') as f:
                                data = json.load(f)
                        except Exception:
                            continue

                        # Determine finished: use status only (simplified)
                        status_val = (data.get('status') or '').lower()
                        finished = status_val in (
                            'completed', 'terminated', 'failed', 'success', 'max_rounds_reached'
                        )

                        if finished:
                            # ignore finished agents for barrier counting
                            continue

                        # Ignore not-started agents to avoid deadlock (they'll join next window)
                        try:
                            if int(data.get('current_loop', 0)) < 1:
                                continue
                        except Exception:
                            continue
                        considered_agents.append(agent_id)
                        waiting_flags.append(bool(data.get('wait_for_sync', False)))

                    # If no running-and-started agents remain, no barrier is needed; stop sync loop gracefully
                    if not considered_agents:
                        # No participants; avoid keeping barrier active forever
                        time.sleep(0.5)
                        continue
                    if considered_agents and all(waiting_flags):
                        # release signal by increasing epoch
                        try:
                            sync_epoch += 1
                            with open(signal_file, 'w', encoding='utf-8') as f:
                                f.write(str(sync_epoch))
                            # allow agents to proceed and clear their wait flags
                            time.sleep(0.1)
                        except Exception:
                            pass
                except Exception:
                    time.sleep(0.5)
        except Exception as e:
            if self._debug_mode:
                print_debug(f"⚠️ Global RoundSyncManager loop error: {e}")

# Global synchronization manager instance
_global_sync_manager = GlobalRoundSyncManager()

# Register cleanup function when program exits
import atexit
atexit.register(_global_sync_manager.stop)

class MultiAgentTools:
    def __init__(self, workspace_root: str = None, debug_mode: bool = False, 
                 max_concurrent_agents: int = 5):
        """
        Initialize multi-agent tools with a workspace root directory.
        
        Args:
            workspace_root: Root directory for workspace files
            debug_mode: Enable debug logging
            max_concurrent_agents: Maximum number of concurrent agents (default: 5)
        """
        self.workspace_root = workspace_root
        self.debug_mode = debug_mode  # Save debug mode setting
        self.max_concurrent_agents = max_concurrent_agents
        
        # Add session-level AGIBot task tracking
        self.session_spawned_tasks = set()
        # Add thread tracking dictionary
        self.active_threads = {}  # task_id -> thread
        
        # Add terminated agents tracking
        self.terminated_agents = set()  # Track terminated agents
        self.completed_agents = set()   # Track completed agents
        
        # Save generated agent IDs for reference normalization
        self.generated_agent_ids = []
        


        # Use global synchronization manager
        try:
            _global_sync_manager.start(self.workspace_root, self.debug_mode)
        except Exception as e:
            print_debug(f"⚠️ Failed to start Global RoundSyncManager: {e}")



    def spawn_agent(self, task_description: str, agent_id: str = None, api_key: str = None, model: str = None, max_loops: int = 25, MCP_config_file: str = None, prompts_folder: str = None, **kwargs) -> Dict[str, Any]:
        """
        Spawn a new AGIBot instance to handle a specific task asynchronously.
        This allows for complex task decomposition and parallel execution.
        All spawned agents run asynchronously in the background.
        
        Args:
            task_description: Description of the task for the new AGIBot instance
            agent_id: Custom agent ID (optional, will auto-generate if not provided). Must match format 'agent_XXX'
            api_key: API key for the new instance (optional, will use current if not provided)
            model: Model name for the new instance (optional, will use current if not provided)  
            max_loops: Maximum execution loops for the new instance
            MCP_config_file: Custom MCP configuration file path (optional, defaults to 'config/mcp_servers.json')
            prompts_folder: Custom prompts folder path (optional, defaults to 'prompts'). Allows using different prompt templates and tool interfaces
            **kwargs: Additional parameters for AGIBotClient
            
        Returns:
            Dict containing spawn information and agent ID
        """
        import threading
        import time
        import uuid
        import json
        from datetime import datetime
        from .id_manager import generate_agent_id
        
        try:
            # Handle parameter type conversion to ensure numeric parameters are correct types
            try:
                # Convert numeric parameters
                if isinstance(max_loops, str):
                    max_loops = int(max_loops)
                
                # Ensure parameter types are correct
                max_loops = int(max_loops)
                
            except (ValueError, TypeError) as e:
                return {
                    "status": "failed",
                    "message": f"Invalid parameter types: max_loops={max_loops}",
                    "error": str(e)
                }
            
            # Get streaming configuration from config/config.txt
            try:
                from src.config_loader import get_streaming
                streaming = get_streaming()
            except:
                streaming = False  # Default fallback
            
            # Import AGIBot Client from main module
            from src.main import AGIBotClient
            
            # Handle agent ID generation or validation
            if agent_id is not None:
                # Validate user-provided agent ID format
                if not self._is_valid_agent_id_format(agent_id):
                    return {
                        "status": "failed",
                        "message": f"Invalid agent ID format: '{agent_id}'. Must match pattern 'agent_XXX' where XXX is a 3-digit number (e.g., 'agent_001')",
                        "provided_agent_id": agent_id
                    }
                
                # Check if agent ID is already in use
                if self._is_agent_id_in_use(agent_id):
                    return {
                        "status": "failed", 
                        "message": f"Agent ID '{agent_id}' is already in use. Please choose a different ID or let the system auto-generate one.",
                        "provided_agent_id": agent_id,
                        "active_agents": list(self.generated_agent_ids)
                    }
            else:
                # Auto-generate unique agent ID using sequential numbering
                agent_id = generate_agent_id("agent", self.workspace_root)
            
            # For code compatibility, task_id is the same as agent_id
            task_id = agent_id
            
            # Normalize Agent references in task description
            task_description = task_description
            
            # Determine the parent AGIBot's working directory (always use shared workspace mode)
            if hasattr(self, 'workspace_root') and self.workspace_root:
                # If workspace_root ends with 'workspace', use its parent directory
                if os.path.basename(self.workspace_root) == 'workspace':
                    output_directory = os.path.dirname(self.workspace_root)
                # If workspace_root contains a 'workspace' subdirectory, use workspace_root
                elif os.path.exists(os.path.join(self.workspace_root, 'workspace')):
                    output_directory = self.workspace_root
                # Otherwise, try to find current output directory based on naming pattern
                else:
                    # Look for output_ pattern in current working directory
                    current_dir = os.getcwd()
                    if 'output_' in os.path.basename(current_dir):
                        output_directory = current_dir
                    else:
                        # Fallback to creating new output directory
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        output_directory = f"output_{timestamp}"
            else:
                # Fallback to creating new output directory
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_directory = f"output_{timestamp}"
            
            # Get current API configuration if not provided
            if api_key is None:
                from src.config_loader import get_api_key
                api_key = get_api_key()
            
            if model is None:
                from src.config_loader import get_model
                model = get_model()
            
            # Validate required parameters
            if not api_key:
                return {
                    "status": "failed",
                    "message": "API key not found. Please provide api_key parameter or set it in config/config.txt",
                    "agent_id": agent_id
                }
            
            if not model:
                return {
                    "status": "failed", 
                    "message": "Model not found. Please provide model parameter or set it in config/config.txt",
                    "agent_id": agent_id
                }
            
            # Create output directory
            abs_output_dir = os.path.abspath(output_directory)
            os.makedirs(abs_output_dir, exist_ok=True)
            
            # Always use shared workspace mode - child AGIBot works in parent's workspace
            if hasattr(self, 'workspace_root') and self.workspace_root:
                if os.path.basename(self.workspace_root) == 'workspace':
                    parent_output_dir = os.path.dirname(self.workspace_root)
                else:
                    parent_output_dir = self.workspace_root
            else:
                parent_output_dir = abs_output_dir
            
            workspace_dir = parent_output_dir
            parent_workspace = os.path.join(parent_output_dir, "workspace")
            os.makedirs(parent_workspace, exist_ok=True)

            # Validate MCP configuration file if specified
            if MCP_config_file is not None:
                mcp_config_path = None
                search_locations = []
                
                # If absolute path provided, use as-is
                if os.path.isabs(MCP_config_file):
                    if os.path.exists(MCP_config_file):
                        mcp_config_path = MCP_config_file
                    search_locations.append(MCP_config_file)
                else:
                    # Search in multiple locations for relative path
                    potential_paths = [
                        # 1. Current working directory
                        os.path.join(os.getcwd(), MCP_config_file),
                        # 2. Workspace directory
                        os.path.join(workspace_dir, MCP_config_file),
                        # 3. Config directory in project root
                        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config", MCP_config_file),
                        # 4. Project root directory  
                        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), MCP_config_file),
                        # 5. Relative to workspace root if it exists
                        os.path.join(self.workspace_root, MCP_config_file) if hasattr(self, 'workspace_root') and self.workspace_root else None,
                    ]
                    
                    # Filter out None values and check each path
                    for path in potential_paths:
                        if path is not None:
                            search_locations.append(path)
                            if os.path.exists(path):
                                mcp_config_path = path
                                break
                
                # If MCP config file not found, return error
                if mcp_config_path is None:
                    return {
                        "status": "failed",
                        "message": f"MCP configuration file '{MCP_config_file}' not found. Searched locations: {search_locations}",
                        "agent_id": agent_id,
                        "searched_locations": search_locations
                    }
                
                # Update MCP_config_file to use the found absolute path
                MCP_config_file = mcp_config_path

            # Create status file for tracking
            status_file_path = os.path.join(abs_output_dir, f".agibot_spawn_{agent_id}_status.json")
            initial_status = {
                "agent_id": agent_id,
                "status": "running",
                "task_description": task_description,
                "start_time": datetime.now().isoformat(),
                "completion_time": None,
                "output_directory": abs_output_dir,
                "working_directory": workspace_dir,
                "model": model,
                "max_loops": max_loops,
                "current_loop": 0,  # Add current loop field, initialized to 0
                "error": None
            }
            
            # Write initial status
            try:
                with open(status_file_path, 'w', encoding='utf-8') as f:
                    json.dump(initial_status, f, indent=2, ensure_ascii=False)
            except Exception as e:
                print_current(f"⚠️ Warning: Could not create status file: {e}")
            
            # Define the async task execution function
            def execute_agibot_task():
                try:
                    # Set up agent context for print operations
                    set_output_directory(workspace_dir)

                    # Print spawn start info (will be routed to agent log under current outdir)
                    print_current(task_id, f"🚀 AGIBot {task_id} started")
                    
                    # Agent id will be injected into AGIBotClient, not print system
                    
                    # Register AGIBot mailbox
                    try:
                        from .message_system import get_message_router
                        router = get_message_router(workspace_dir, cleanup_on_init=False)
                        router.register_agent(task_id)
                        print_current(task_id, f"📬 Mailbox registered")
                    except Exception as e:
                        # Remove unnecessary log output
                        pass
                    
                    # Ensure all direct prints during agent execution are routed to agent logs
                    try:
                        from .print_system import with_agent_print
                    except Exception:
                        with_agent_print = None

                    if with_agent_print is not None:
                        with with_agent_print(task_id):
                            # Create AGIBot client with agent_id
                            debug_mode_to_use = kwargs.get('debug_mode', self.debug_mode)
                            client = AGIBotClient(
                                api_key=api_key,
                                model=model,
                                debug_mode=debug_mode_to_use,
                                detailed_summary=kwargs.get('detailed_summary', True),
                                single_task_mode=kwargs.get('single_task_mode', True),
                                interactive_mode=kwargs.get('interactive_mode', False),
                                streaming=streaming,
                                MCP_config_file=MCP_config_file,
                                prompts_folder=prompts_folder,
                                agent_id=task_id
                            )

                            # Execute the task
                            response = client.chat(
                                messages=[{"role": "user", "content": task_description}],
                                dir=workspace_dir,
                                loops=max_loops,
                                continue_mode=kwargs.get('continue_mode', False)
                            )
                    else:
                        # Fallback without context manager
                        debug_mode_to_use = kwargs.get('debug_mode', self.debug_mode)
                        client = AGIBotClient(
                            api_key=api_key,
                            model=model,
                            debug_mode=debug_mode_to_use,
                            detailed_summary=kwargs.get('detailed_summary', True),
                            single_task_mode=kwargs.get('single_task_mode', True),
                            interactive_mode=kwargs.get('interactive_mode', False),
                            streaming=streaming,
                            MCP_config_file=MCP_config_file,
                            prompts_folder=prompts_folder,
                            agent_id=task_id
                        )

                        # Execute the task
                        response = client.chat(
                            messages=[{"role": "user", "content": task_description}],
                            dir=workspace_dir,
                            loops=max_loops,
                            continue_mode=kwargs.get('continue_mode', False)
                        )
                    
                    # 🔧 Check if terminate signal was received
                    is_terminated = False
                    if isinstance(response.get('message'), str) and 'AGENT_TERMINATED' in response.get('message', ''):
                        is_terminated = True
                        print_current(task_id, f"🛑 Agent {task_id} received terminate signal and will exit")
                    
                    # Update status file with completion
                    if is_terminated:
                        # 🔧 New: handle terminate signal status update
                        terminate_status = {
                            "agent_id": task_id,
                            "status": "terminated",
                            "task_description": task_description,
                            "start_time": initial_status["start_time"],
                            "completion_time": datetime.now().isoformat(),
                            "output_directory": abs_output_dir,
                            "working_directory": workspace_dir,
                            "model": model,
                            "max_loops": max_loops,
                            "current_loop": response.get("current_loop", 0),  # Add loop information when terminated
                            "error": None,
                            "status": "success",
                            "terminated": True,
                            "response": response
                        }
                        
                        try:
                            with open(status_file_path, 'w', encoding='utf-8') as f:
                                json.dump(terminate_status, f, indent=2, ensure_ascii=False)
                                f.flush()
                                import os
                                os.fsync(f.fileno()) if hasattr(f, 'fileno') else None
                            # Remove unnecessary log output
                        except Exception as e:
                            # Remove unnecessary log output
                            pass
                            
                    elif response["success"]:
                        completion_status = {
                            "agent_id": task_id,
                            "status": "completed",
                            "task_description": task_description,
                            "start_time": initial_status["start_time"],
                            "completion_time": datetime.now().isoformat(),
                            "output_directory": abs_output_dir,
                            "working_directory": workspace_dir,
                            "model": model,
                            "max_loops": max_loops,
                            "current_loop": response.get("current_loop", max_loops),  # Add final loop information
                            "error": None,
                            "status": "success",
                            "response": response
                        }
                        
                        # 🔧 New: add agent to completed_agents set
                        if hasattr(self, 'completed_agents'):
                            self.completed_agents.add(task_id)
                        
                        try:
                            with open(status_file_path, 'w', encoding='utf-8') as f:
                                json.dump(completion_status, f, indent=2, ensure_ascii=False)
                                f.flush()
                                import os
                                os.fsync(f.fileno()) if hasattr(f, 'fileno') else None
                        except Exception as e:
                            # Remove unnecessary log output
                            pass
                    else:
                        response_message = response.get('message', 'Task failed')
                        if "reached maximum execution rounds" in response_message or "max_rounds_reached" in response_message:
                            status = "max_rounds_reached"
                            error_message = "Reached maximum rounds"
                        else:
                            status = "failed"
                            error_message = response_message
                        
                        completion_status = {
                            "agent_id": task_id,
                            "status": status,
                            "task_description": task_description,
                            "start_time": initial_status["start_time"],
                            "completion_time": datetime.now().isoformat(),
                            "output_directory": abs_output_dir,
                            "working_directory": workspace_dir,
                            "model": model,
                            "max_loops": max_loops,
                            "current_loop": response.get("current_loop", max_loops),  # Add loop information when failed
                            "error": error_message,
                            "response": response
                        }
                        
                        try:
                            with open(status_file_path, 'w', encoding='utf-8') as f:
                                json.dump(completion_status, f, indent=2, ensure_ascii=False)
                                f.flush()
                                import os
                                os.fsync(f.fileno()) if hasattr(f, 'fileno') else None
                        except Exception as e:
                            # Remove unnecessary log output
                            pass
                    
                    time.sleep(0.5)
                    
                    # Print completion status
                    if is_terminated:
                        print_current(task_id, f"🛑 AGIBot spawn {task_id} terminated successfully")
                    elif response["success"]:
                        print_current(task_id, f"✅ AGIBot spawn {task_id} completed successfully")
                    else:
                        response_message = response.get('message', 'Unknown error')
                        if "reached maximum execution rounds" in response_message or "max_rounds_reached" in response_message:
                            print_current(task_id, f"⚠️ AGIBot spawn {task_id} reached maximum execution rounds")
                        else:
                            print_current(task_id, f"❌ AGIBot spawn {task_id} failed: {response_message}")
                    
                    # remove active_threads after finished
                    if hasattr(self, 'active_threads') and task_id in self.active_threads:
                        del self.active_threads[task_id]
                    

                        
                except Exception as e:
                    error_msg = str(e)
                    print_current(task_id, f"❌ AGIBot spawn {task_id} error: {error_msg}")
                    
                    # Update status file with error
                    error_status = {
                        "agent_id": task_id,
                        "status": "failed",
                        "task_description": task_description,
                        "start_time": initial_status["start_time"],
                        "completion_time": datetime.now().isoformat(),
                        "output_directory": abs_output_dir,
                        "working_directory": workspace_dir,
                        "model": model,
                        "max_loops": max_loops,
                        "current_loop": 0,  # Add loop information on error (usually 0)
                        "error": error_msg,
                        "status": "failed",
                        "response": {
                            "status": "failed",
                            "message": error_msg
                        }
                    }
                    
                    try:
                        with open(status_file_path, 'w', encoding='utf-8') as f:
                            json.dump(error_status, f, indent=2, ensure_ascii=False)
                            f.flush()
                            import os
                            os.fsync(f.fileno()) if hasattr(f, 'fileno') else None
                    except Exception as e:
                        # Remove unnecessary log output
                        pass
                    
                    # remove from list if error
                    if hasattr(self, 'active_threads') and task_id in self.active_threads:
                        del self.active_threads[task_id]
                    
                    time.sleep(0.5)
                    
            

            thread = threading.Thread(target=execute_agibot_task, daemon=True)
            thread.start()
            
            # Wait a moment to let the thread start
            time.sleep(0.1)
            
            # Add the task ID started in this session to the tracking set
            self.session_spawned_tasks.add(task_id)
            
            # Save thread reference for later checking
            self.active_threads[task_id] = thread
            
            # Save generated agent ID for reference normalization
            self.generated_agent_ids.append(task_id)
            
            # Base result information
            result = {
                "status": "success", 
                "message": f"AGIBot instance spawned successfully with agent ID: {agent_id}",
                "agent_id": agent_id,
                "output_directory": abs_output_dir,
                "working_directory": workspace_dir,
                "workspace_files_directory": os.path.join(abs_output_dir, "workspace"),
                "task_description": task_description,
                "model": model,
                "max_loops": max_loops,
                "status_file": status_file_path,
                "agent_communication_note": f"✅ Use agent ID '{task_id}' for all message sending and receiving operations",
                "spawn_mode": "asynchronous",
                "thread_started": thread.is_alive(),
                "thread_id": thread.ident if thread else None,
                "execution_note": "Task running asynchronously in dedicated thread with shared workspace"
            }
            
            # All agents run asynchronously
            result["note"] = f"✅ AGIBot {task_id} is running asynchronously in background. Task will execute independently and send messages when completed."
            result["success"] = True
            result["thread_id"] = thread.ident if thread else None
            
            return result
            
        except Exception as e:
            return {
                "status": "failed",
                "message": f"Failed to spawn AGIBot instance: {str(e)}",
                "task_id": task_id if 'task_id' in locals() else "unknown"
            }

    def send_P2P_message(self, receiver_id: str, content) -> Dict[str, Any]:
        """
        Send message to specified agent or manager. Use 'manager' as receiver_id to send messages to the manager.
        
        Args:
            receiver_id: Receiver agent ID (use 'manager' for manager)
            content: Message content
            
        Returns:
            Send result dictionary
        """
        try:
            from .message_system import Message, MessageType, MessagePriority, get_message_router
            
            # Get message router
            router = get_message_router(self.workspace_root, cleanup_on_init=False)
            
            # Use default message type and priority
            msg_type = MessageType.COLLABORATION
            msg_priority = MessagePriority.NORMAL
            
            # Forcefully obtain the real agent_id of the current thread as sender_id
            current_agent_id = get_current_agent_id()
            sender_id = current_agent_id if current_agent_id else "manager"
            
            # If the caller provided a sender info that does not match the current thread, issue a warning
            if hasattr(self, '_override_sender_check') and current_agent_id:
                print_debug(f"📤 Message sender auto-detected as: {sender_id}")
            
            # Handle content parameter - convert string to dict if needed
            if isinstance(content, str):
                message_content = {"text": content}
            elif isinstance(content, dict):
                message_content = content
            else:
                message_content = {"text": str(content)}
            
            # Create message
            message = Message(
                sender_id=sender_id,
                receiver_id=receiver_id,
                message_type=msg_type,
                content=message_content,
                priority=msg_priority
            )
            
            # Use the correct sender's mailbox to send the message
            sender_mailbox = router.get_mailbox(sender_id)
            if not sender_mailbox:
                sender_mailbox = router.register_agent(sender_id)
                if not sender_mailbox:
                    return {"status": "failed", "message": f"Failed to register sender agent: {sender_id}"}
            
            # Send message
            success = sender_mailbox.send_message(message)
            
            # Trigger routing processing immediately after sending message
            if success:
                try:
                    processed_count = router.process_all_messages_once()
                except Exception as e:
                    print_current(f"⚠️ Error processing messages after send: {e}")
                
                return {
                    "status": "success",
                    "message": "Message sent successfully",
                    "message_id": message.message_id,
                    "sender_id": sender_id,
                    "receiver_id": receiver_id
                }
            else:
                return {
                    "status": "failed",
                    "message": "Failed to send message",
                    "sender_id": sender_id,
                    "receiver_id": receiver_id
                }
            
        except Exception as e:
            return {
                "status": "failed",
                "message": f"Error sending message: {str(e)}",
                "receiver_id": receiver_id if 'receiver_id' in locals() else "unknown"
            }

    def read_received_messages(self, include_read: bool = False) -> Dict[str, Any]:
        """
        Get messages from current agent's mailbox. Messages are automatically marked as read after retrieval.
        
        Args:
            include_read: Whether to include read messages
            
        Returns:
            Message list and statistics
        """
        try:
            from .message_system import get_message_router
            import glob
            
            # Get current agent ID
            current_agent_id = get_current_agent_id()
            agent_id = current_agent_id if current_agent_id else "manager"
            
            # Use workspace_root directly
            try:
                router = get_message_router(self.workspace_root, cleanup_on_init=False)
                mailbox = router.get_mailbox(agent_id)
                
                if not mailbox:
                    # Collect available agents for error reporting
                    available_agents = router.get_all_agents()
                    return {
                        "status": "failed", 
                        "message": f"Agent '{agent_id}' mailbox not found",
                        "agent_id": agent_id,
                        "workspace_root": self.workspace_root,
                        "available_agents": available_agents
                    }
                
                found_workspace = self.workspace_root
                
            except Exception as e:
                return {
                    "status": "failed",
                    "message": f"Error accessing workspace: {str(e)}",
                    "agent_id": agent_id,
                    "workspace_root": self.workspace_root
                }
            
            # Get messages
            if include_read:
                messages = mailbox.get_all_messages()
            else:
                messages = mailbox.get_unread_messages()
            
            # Automatically mark messages as read after retrieval
            for message in messages:
                try:
                    mailbox.mark_as_read(message.message_id)
                except Exception as e:
                    print_current(f"⚠️ Warning: Could not mark message {message.message_id} as read: {e}")
            
            # Convert message format
            messages_data = [msg.to_dict() for msg in messages]
            
            # Get mailbox statistics
            stats = mailbox.get_message_stats()
            
            result = {
                "status": "success",
                "agent_id": agent_id,
                "message_count": len(messages_data),
                "messages": messages_data,
                "mailbox_stats": stats,
                "found_in_workspace": found_workspace,
                "auto_marked_as_read": len(messages_data)
            }
            
            return result
            
        except Exception as e:
            return {
                "status": "failed",
                "message": f"Error getting messages: {str(e)}",
                "agent_id": agent_id if 'agent_id' in locals() else "unknown"
            }

    def get_agent_messages_summary(self, include_read: bool = False) -> Dict[str, Any]:
        """
        Get a summary of agent messages without marking them as read.
        Useful for debugging result display issues.
        
        Args:
            include_read: Whether to include read messages
            
        Returns:
            Message summary and diagnostic information
        """
        try:
            from .message_system import get_message_router
            import json
            
            # Get current agent ID
            current_agent_id = get_current_agent_id()
            agent_id = current_agent_id if current_agent_id else "manager"
            
            router = get_message_router(self.workspace_root, cleanup_on_init=False)
            mailbox = router.get_mailbox(agent_id)
            
            if not mailbox:
                return {
                    "status": "failed", 
                    "message": f"Agent '{agent_id}' mailbox not found",
                    "agent_id": agent_id,
                    "workspace_root": self.workspace_root
                }
            
            # Get messages WITHOUT marking as read
            if include_read:
                messages = mailbox.get_all_messages()
            else:
                messages = mailbox.get_unread_messages()
            
            # Create summary without full message content
            message_summaries = []
            total_content_size = 0
            
            for i, msg in enumerate(messages):
                try:
                    # Calculate content size
                    content_str = json.dumps(msg.content) if hasattr(msg, 'content') else "{}"
                    content_size = len(content_str)
                    total_content_size += content_size
                    
                    summary = {
                        "message_id": msg.message_id,
                        "sender_id": msg.sender_id,
                        "receiver_id": msg.receiver_id,
                        "message_type": msg.message_type.value if hasattr(msg.message_type, 'value') else str(msg.message_type),
                        "timestamp": msg.timestamp,
                        "delivered": getattr(msg, 'delivered', False),
                        "read": getattr(msg, 'read', False),
                        "content_size_bytes": content_size,
                        "content_preview": content_str[:100] + "..." if len(content_str) > 100 else content_str
                    }
                    message_summaries.append(summary)
                    
                except Exception as e:
                    print_current(f"⚠️ Error processing message {i+1}: {e}")
                    message_summaries.append({
                        "message_id": getattr(msg, 'message_id', f'unknown_{i}'),
                        "error": f"Failed to process: {str(e)}"
                    })
            
            # Get mailbox statistics
            stats = mailbox.get_message_stats()
            
            result = {
                "status": "success",
                "agent_id": agent_id,
                "workspace_root": self.workspace_root,
                "mailbox_path": mailbox.inbox_dir,
                "message_count": len(messages),
                "total_content_size_bytes": total_content_size,
                "message_summaries": message_summaries,
                "mailbox_stats": stats,
                "diagnostic_info": {
                    "mailbox_exists": os.path.exists(mailbox.inbox_dir),
                    "include_read": include_read,
                    "messages_not_marked_as_read": True
                }
            }
            
            return result
            
        except Exception as e:
            return {
                "status": "failed",
                "message": f"Error getting message summary: {str(e)}",
                "agent_id": agent_id if 'agent_id' in locals() else "unknown"
            }

    def send_status_update_to_manager(self, agent_id: str, round_number: int, task_completed: bool, 
                                     llm_response_preview: str, tool_calls_summary: list, 
                                     current_task_description: str = "", error_message: str = None) -> Dict[str, Any]:
        """
        Send status update to manager
        
        Args:
            agent_id: Agent ID (can be 'current_agent' to auto-detect current agent)
            round_number: Round number
            task_completed: Whether task is completed
            llm_response_preview: LLM response preview
            tool_calls_summary: Tool calls summary
            current_task_description: Current task description
            error_message: Error message if any
            
        Returns:
            Send result
        """
        try:
            from .message_system import Message, MessageType, MessagePriority, StatusUpdateMessage, get_message_router
            
            # 🔧 Always use current thread's real agent_id
            current_agent_id = get_current_agent_id()
            if current_agent_id:
                actual_agent_id = current_agent_id
                # If LLM's passed agent_id doesn't match current thread's agent_id
                if agent_id != "current_agent" and agent_id != current_agent_id:
                    print_debug(f"⚠️ Agent ID mismatch! LLM provided '{agent_id}' but current thread is '{current_agent_id}'. Using correct ID.")
            else:
                # If agent_id is not set
                if agent_id == "current_agent" or not agent_id:
                    actual_agent_id = "manager"
                else:
                    actual_agent_id = agent_id
            
            # Get message router
            router = get_message_router(self.workspace_root, cleanup_on_init=False)
            
            # Create status update content
            content = StatusUpdateMessage.create_content(
                round_number=round_number,
                task_completed=task_completed,
                llm_response_preview=llm_response_preview,
                tool_calls_summary=tool_calls_summary,
                current_task_description=current_task_description,
                error_message=error_message
            )
            
            # Create message
            message = Message(
                sender_id=actual_agent_id,
                receiver_id="manager",
                message_type=MessageType.STATUS_UPDATE,
                content=content,
                priority=MessagePriority.NORMAL
            )
            
            # Get sender mailbox
            sender_mailbox = router.get_mailbox(actual_agent_id)
            if not sender_mailbox:
                sender_mailbox = router.register_agent(actual_agent_id)
                if not sender_mailbox:
                    return {
                        "status": "failed",
                        "message": f"Failed to register agent: {actual_agent_id}",
                        "agent_id": actual_agent_id
                    }
            
            # Send message
            success = sender_mailbox.send_message(message)
            
            # Trigger routing processing immediately after sending status update
            if success:
                try:
                    processed_count = router.process_all_messages_once()
                except Exception as e:
                    print_current(f"⚠️ Error processing messages after status update: {e}")
            
            return {
                "status": "success" if success else "failed",
                "message": "Status update sent to manager successfully" if success else "Failed to send status update to manager",
                "message_id": message.message_id,
                "agent_id": actual_agent_id,
                "round_number": round_number
            }
            
        except Exception as e:
            return {
                "status": "failed",
                "message": f"Error sending status update to manager: {str(e)}",
                "agent_id": actual_agent_id if 'actual_agent_id' in locals() else agent_id
            }

    def send_broadcast_message(self, content) -> Dict[str, Any]:
        """
        Broadcast message to all agents
        
        Args:
            content: Message content
            
        Returns:
            Broadcast result
        """
        try:
            from .message_system import MessageType, get_message_router
            
            # Get message router
            router = get_message_router(self.workspace_root, cleanup_on_init=False)
            
            # Use default message type
            msg_type = MessageType.BROADCAST
            
            # Handle content parameter - convert string to dict if needed
            if isinstance(content, str):
                message_content = {"text": content}
            elif isinstance(content, dict):
                message_content = content
            else:
                message_content = {"text": str(content)}
            
            # Broadcast message
            sent_count = router.broadcast_message("manager", message_content)
            
            # 🔧 Trigger routing processing immediately after broadcasting message
            if sent_count > 0:
                try:
                    processed_count = router.process_all_messages_once()
                except Exception as e:
                    print_current(f"⚠️ Error processing messages after broadcast: {e}")
            
            return {
                "status": "success",
                "message": f"Broadcast sent to {sent_count} agents",
                "sent_count": sent_count
            }
            
        except Exception as e:
            return {
                "status": "failed",
                "message": f"Error broadcasting message: {str(e)}"
            }


    def get_agent_session_info(self) -> Dict[str, Any]:
        """
        Get comprehensive information about current agent session including session statistics, 
        agent details, and list of all active agents.
        
        Returns:
            Session information dictionary including active agents list
        """
        try:
            from .message_system import get_message_router
            import glob
            import threading
            
            # Get message router
            router = get_message_router(self.workspace_root, cleanup_on_init=False)
            
            # Get all registered agents
            all_agents = router.get_all_agents()
            
            # Get active agents based on thread status
            active_agents_info = []
            
            # Check active threads 
            if hasattr(self, 'active_threads'):
                for agent_id, thread in self.active_threads.items():
                    if isinstance(thread, threading.Thread) and thread.is_alive():
                        active_agents_info.append({
                            "agent_id": agent_id,
                            "status": "active",
                            "task_description": f"Agent {agent_id}",
                            "start_time": "2025-01-01T00:00:00",
                            "thread_id": thread.ident,
                            "thread_name": thread.name
                        })
            
            # Also check registered agents in message system
            try:
                # Collect all possible workspace paths
                workspace_paths = []
                
                # Add current workspace_root
                if self.workspace_root:
                    workspace_paths.append(self.workspace_root)
                
                # Add current directory
                workspace_paths.append(os.getcwd())
                
                # Search all output_* directories
                output_dirs = glob.glob("output_*")
                for output_dir in output_dirs:
                    if os.path.isdir(output_dir):
                        workspace_paths.append(os.path.abspath(output_dir))
                
                # Deduplicate
                workspace_paths = list(set(workspace_paths))
                
                # Try to get registered agents from all possible workspace paths
                all_registered_agents = set()
                for workspace_path in workspace_paths:
                    try:
                        router = get_message_router(workspace_path, cleanup_on_init=False)
                        registered_agents = router.get_all_agents()
                        all_registered_agents.update(registered_agents)
                    except Exception:
                        pass
                
                # Add agents registered in message system but not in thread tracking
                existing_agent_ids = {agent["agent_id"] for agent in active_agents_info}
                for agent_id in all_registered_agents:
                    if agent_id not in existing_agent_ids and agent_id != "manager":
                        # 🔧 Check agent status - terminated, completed, max_rounds_reached, failed, or registered
                        if agent_id in self.terminated_agents:
                            status = "terminated"
                            status_icon = "🔴"
                        elif agent_id in self.completed_agents:
                            status = "completed"
                            status_icon = "🟢"
                        else:
                            # Check status file for more accurate status
                            status = self._get_agent_status_from_file(agent_id)
                            if status == "terminated":
                                status_icon = "🔴"
                                self.terminated_agents.add(agent_id)  # Update cache
                            elif status == "completed":
                                status_icon = "🟢"
                                self.completed_agents.add(agent_id)  # Update cache
                            elif status == "max_rounds_reached":
                                status_icon = "🟠"
                                # Treat max_rounds_reached as a completion status
                                self.completed_agents.add(agent_id)  # Update cache
                            elif status == "failed":
                                status_icon = "🔴"
                                self.terminated_agents.add(agent_id)  # Update cache
                            elif status == "running":
                                status = "running"
                                status_icon = "🟢"
                            elif status == "unknown":
                                # Agent has registered mailbox but status unknown
                                status = "idle"
                                status_icon = "🟡"
                            else:
                                # Other unknown status
                                status = "unknown"
                                status_icon = "⚫"
                        
                        active_agents_info.append({
                            "agent_id": agent_id,
                            "status": status,
                            "status_icon": status_icon,
                            "task_description": f"Agent {agent_id}",
                            "start_time": "2025-01-01T00:00:00"
                        })
                        
            except Exception as e:
                print_current(f"⚠️ Error checking message system agents: {e}")
            
            # Statistics
            total_agents = len(all_agents)
            active_agents = len(active_agents_info)
            
            # Terminal output
            print_current("📊 ===========================================")
            print_current("📊 AGIBot Session Information")
            print_current("📊 ===========================================")
            print_current(f"📊 Total Agents: {total_agents}")
            print_current(f"📊 Active Agents: {active_agents}")
            print_current(f"📊 Completed Agents: {len(self.completed_agents)}")
            print_current(f"📊 Failed Agents: {len(self.terminated_agents)}")
            print_current(f"📊 Message System Status: Active")
            print_current(f"📊 Registered Agents: {', '.join(all_agents) if all_agents else 'None'}")
            print_current("📊 ===========================================")
            
            if active_agents_info:
                print_current("🤖 Active AGIBot List:")
                for i, agent in enumerate(active_agents_info, 1):
                    # 🔧 Use more detailed status icons and status descriptions
                    status_icon = agent.get("status_icon", "🔵")
                    if not status_icon:
                        if agent.get("status") == "active":
                            status_icon = "🟢"
                        elif agent.get("status") == "terminated":
                            status_icon = "🔴"
                        elif agent.get("status") == "completed":
                            status_icon = "✅"
                        else:
                            status_icon = "🔵"
                    
                    # Add more detailed status descriptions
                    status_desc = agent.get('status', 'unknown')
                    if status_desc == "completed":
                        # Try to get completion time information
                        try:
                            import datetime
                            completion_status = self._get_agent_completion_info(agent['agent_id'])
                            if completion_status:
                                completion_time = completion_status.get('completion_time')
                                if completion_time:
                                    # Calculate how long ago completion time was from now
                                    from datetime import datetime as dt
                                    completed_dt = dt.fromisoformat(completion_time.replace('Z', '+00:00') if completion_time.endswith('Z') else completion_time)
                                    now_dt = dt.now()
                                    time_diff = now_dt - completed_dt
                                    if time_diff.total_seconds() < 60:
                                        status_desc = f"completed (just completed)"
                                    elif time_diff.total_seconds() < 3600:
                                        minutes = int(time_diff.total_seconds() / 60)
                                        status_desc = f"completed ({minutes} minutes ago)"
                                    else:
                                        hours = int(time_diff.total_seconds() / 3600)
                                        status_desc = f"completed ({hours} hours ago)"
                        except Exception:
                            pass
                    elif status_desc == "max_rounds_reached":
                        status_desc = "exit (max rounds reached)"
                    elif status_desc == "failed":
                        status_desc = "exit (failed)"
                    elif status_desc == "terminated":
                        status_desc = "exit (terminated)"
                    
                    print_current(f"🤖 {i}. {status_icon} {agent['agent_id']} - {status_desc}")
                    if agent.get("thread_id"):
                        print_current(f"   └─ Thread ID: {agent['thread_id']}, Thread Name: {agent.get('thread_name', 'Unknown')}")
            else:
                print_current("🤖 No active AGIBot detected")
            
            result = {
                "status": "success",
                "session_id": "default_session",
                "session_start_time": "2025-01-01T00:00:00",
                "total_agents": total_agents,
                "active_agents": active_agents,
                "completed_agents": len(self.completed_agents),
                "failed_agents": len(self.terminated_agents),
                "message_system_active": True,
                "registered_agents": all_agents,
                "active_agents_info": active_agents_info,
                "mailbox_stats": {
                    "total_mailboxes": total_agents,
                    "active_mailboxes": active_agents
                }
            }
            
            return result
            
        except Exception as e:
            error_msg = f"Error getting session info: {str(e)}"
            print_current(f"❌ {error_msg}")
            return {
                "status": "failed",
                "message": error_msg
            }

    def terminate_agent(self, agent_id: str, reason: str = None) -> Dict[str, Any]:
        """
        Terminate a specific AGIBot agent by sending a terminate signal.
        
        Args:
            agent_id: ID of the agent to terminate. Use "self" or leave empty to terminate current agent.
            reason: Reason for termination (optional)
            
        Returns:
            Termination result dictionary
        """
        try:
            from .message_system import Message, MessageType, MessagePriority, get_message_router
            from datetime import datetime
            
            # Get current agent ID (usually manager)
            current_agent_id = get_current_agent_id()
            sender_id = current_agent_id if current_agent_id else "manager"
            
            if not agent_id or agent_id == "self" or agent_id == "current_agent":
                if current_agent_id:
                    agent_id = current_agent_id
                else:
                    return {
                        "status": "failed",
                        "message": "Cannot determine current agent ID for self-termination",
                        "agent_id": agent_id
                    }
            
            # Validate agent ID format
            if not self._is_valid_agent_id_format(agent_id):
                return {
                    "status": "failed",
                    "message": f"Invalid agent ID format: {agent_id}. Expected format: agent_XXX",
                    "agent_id": agent_id
                }
            
            # Check if agent exists and is active
            agent_exists = False
            
            # Check in active threads
            if hasattr(self, 'active_threads') and agent_id in self.active_threads:
                thread = self.active_threads[agent_id]
                if thread.is_alive():
                    agent_exists = True
            
            # Check in message system
            try:
                router = get_message_router(self.workspace_root, cleanup_on_init=False)
                mailbox = router.get_mailbox(agent_id)
                if mailbox:
                    agent_exists = True
            except Exception:
                pass
            
            if not agent_exists:
                return {
                    "status": "failed",
                    "message": f"Agent '{agent_id}' not found or already terminated",
                    "agent_id": agent_id
                }
            
            # Create terminate message
            terminate_content = {
                "signal": "terminate",
                "reason": reason or "Terminated by request",
                "timestamp": datetime.now().isoformat(),
                "sender": sender_id
            }
            
            # Send terminate message to the agent
            try:
                router = get_message_router(self.workspace_root, cleanup_on_init=False)
                
                # Create terminate message
                message = Message(
                    sender_id=sender_id,
                    receiver_id=agent_id,
                    message_type=MessageType.SYSTEM,
                    content=terminate_content,
                    priority=MessagePriority.HIGH
                )
                
                # Get sender mailbox
                sender_mailbox = router.get_mailbox(sender_id)
                if not sender_mailbox:
                    sender_mailbox = router.register_agent(sender_id)
                
                # Send the terminate message
                success = sender_mailbox.send_message(message)
                
                if success:
                    try:
                        processed_count = router.process_all_messages_once()
                    except Exception as e:
                        print_current(f"⚠️ Error processing messages after terminate signal: {e}")

                    
                    # Remove from active threads tracking if exists
                    if hasattr(self, 'active_threads') and agent_id in self.active_threads:
                        del self.active_threads[agent_id]
                    
                    # Remove from generated agent IDs if exists
                    if hasattr(self, 'generated_agent_ids') and agent_id in self.generated_agent_ids:
                        self.generated_agent_ids.remove(agent_id)
                    
                    # Add to terminated agents
                    self.terminated_agents.add(agent_id)
                    
                    return {
                        "status": "success",
                        "message": f"Terminate signal sent to agent {agent_id} successfully",
                        "agent_id": agent_id,
                        "reason": reason or "Terminated by request",
                        "message_id": message.message_id
                    }
                else:
                    return {
                        "status": "failed",
                        "message": f"Failed to send terminate signal to agent {agent_id}",
                        "agent_id": agent_id
                    }
                    
            except Exception as e:
                return {
                    "status": "failed",
                    "message": f"Error sending terminate signal to agent {agent_id}: {str(e)}",
                    "agent_id": agent_id
                }
        
        except Exception as e:
            return {
                "status": "failed", 
                "message": f"Error terminating agent {agent_id}: {str(e)}",
                "agent_id": agent_id if 'agent_id' in locals() else "unknown"
            }

    
    def _is_valid_agent_id_format(self, agent_id: str) -> bool:
        """
        Validate agent ID format
        
        Args:
            agent_id: Agent ID to validate
            
        Returns:
            True if agent ID format is valid
        """
        import re
        
        # Allowed formats:
        # 1. manager (special admin ID)
        # 2. agent_XXX (supports letter, number, underscore combinations)
        if agent_id == "manager":
            return True
            
        # agent_prefix, followed by letter, number, or underscore combinations
        pattern = r'^agent_[a-zA-Z0-9_]+$'
        return bool(re.match(pattern, agent_id))
    
    def _is_agent_id_in_use(self, agent_id: str) -> bool:
        """
        Check if agent ID is already in use
        
        Args:
            agent_id: Agent ID to check
            
        Returns:
            True if agent ID is already in use
        """
        # Check generated agent IDs in the current session
        if agent_id in self.generated_agent_ids:
            return True
        
        # Check active threads
        if agent_id in self.active_threads:
            return True
        
        # Check if a mailbox corresponding to the agent ID is already registered
        try:
            from .message_system import get_message_router
            router = get_message_router(self.workspace_root, cleanup_on_init=False)
            if hasattr(router, 'mailboxes') and agent_id in router.mailboxes:
                return True
        except Exception:
            pass
        
        return False
    
    def _get_agent_completion_info(self, agent_id: str) -> dict:
        """
        Get agent completion information
        
        Args:
            agent_id: Agent ID
            
        Returns:
            Dictionary containing completion information, returns empty dict if not found
        """
        try:
            import json
            
            status_file_paths = []
            
            if self.workspace_root:
                if os.path.basename(self.workspace_root) == "workspace":
                    outdir = os.path.dirname(self.workspace_root)
                    status_file_paths.append(f"{outdir}/.agibot_spawn_{agent_id}_status.json")
                else:
                    status_file_paths.append(f"{self.workspace_root}/.agibot_spawn_{agent_id}_status.json")
            
            #status_file_paths.append(f".agibot_spawn_{agent_id}_status.json")

            for status_file in status_file_paths:
                if os.path.exists(status_file):
                    try:
                        with open(status_file, 'r', encoding='utf-8') as f:
                            status_data = json.load(f)
                        
                        if status_data.get("status") == "completed" or (status_data.get("success", False) and status_data.get("completion_time")):
                            return {
                                'completion_time': status_data.get('completion_time'),
                                'start_time': status_data.get('start_time'),
                                'task_description': status_data.get('task_description'),
                                'success': status_data.get('success', False)
                            }

                        break
                            
                    except (json.JSONDecodeError, IOError):
                        continue
            
            return {}
            
        except Exception as e:
            if self.debug_mode:
                print_current(f"⚠️ Error reading completion info for {agent_id}: {e}")
            return {}

    def _get_agent_status_from_file(self, agent_id: str) -> str:
        """
        Get agent status from status file
        
        Args:
            agent_id: Agent ID
            
        Returns:
            Agent status: 'terminated', 'completed', 'running', or 'unknown'
        """
        try:
            import json
            
            status_file_paths = []
            
            if self.workspace_root:
                if os.path.basename(self.workspace_root) == "workspace":
                    outdir = os.path.dirname(self.workspace_root)
                    status_file_paths.append(f"{outdir}/.agibot_spawn_{agent_id}_status.json")
                else:
                    status_file_paths.append(f"{self.workspace_root}/.agibot_spawn_{agent_id}_status.json")
            
            
            for status_file in status_file_paths:
                if os.path.exists(status_file):
                    try:
                        with open(status_file, 'r', encoding='utf-8') as f:
                            status_data = json.load(f)
                        
                        if status_data.get("terminated", False):
                            return "terminated"
                        elif status_data.get("success", False) and status_data.get("completion_time"):
                            return "completed"
                        elif status_data.get("status") == "terminated":
                            return "terminated"
                        elif status_data.get("status") == "completed":
                            return "completed"
                        elif status_data.get("status") == "max_rounds_reached":
                            return "max_rounds_reached"
                        elif status_data.get("status") == "failed":
                            return "failed"
                        elif status_data.get("status") == "running":
                            return "running"
                        
                        if self.debug_mode:
                            print_current(f"🔍 Found status file for {agent_id}: {status_file}, status: {status_data.get('status', 'unknown')}")
                        
                        break
                            
                    except (json.JSONDecodeError, IOError):
                        continue
            
            return "unknown"
            
        except Exception as e:
            if self.debug_mode:
                print_current(f"⚠️ Error reading status file for {agent_id}: {e}")
            return "unknown"

    def cleanup(self):
        """Clean up multi-agent system resources"""
        try:
            # Clean up active threads
            if hasattr(self, 'active_threads'):
                active_threads = 0
                
                for task_id, thread in self.active_threads.items():
                    if isinstance(thread, threading.Thread) and thread.is_alive():
                        print_current(f"⏳ Waiting for thread {task_id} to complete...")
                        active_threads += 1
                
                if active_threads > 0:
                    print_current(f"⏳ Waiting for {active_threads} threads to complete...")
                
                # Clear thread dictionary
                self.active_threads.clear()
            
            # Clean up session tracking
            if hasattr(self, 'session_spawned_tasks'):
                self.session_spawned_tasks.clear()
            
            # Clean up generated agent IDs
            if hasattr(self, 'generated_agent_ids'):
                self.generated_agent_ids.clear()
            
            # Clean up message router
            try:
                from .message_system import get_message_router
                router = get_message_router()
                if router and hasattr(router, 'stop'):
                    router.stop()
            except Exception as e:
                print_current(f"⚠️ Error cleaning up message router: {e}")
            
        except Exception as e:
            print_error(f"❌ Error cleaning up multi-agent system resources: {e}")

    def __del__(self):
        """Destructor, ensure resources are cleaned up"""
        try:
            self.cleanup()
        except:
            pass


    def update_agent_current_loop(self, agent_id: str, current_loop: int) -> Dict[str, Any]:
        """
        Update the current loop information in agent status file
        
        Args:
            agent_id: Agent ID
            current_loop: Current loop number
            
        Returns:
            Update result
        """
        try:
            from datetime import datetime
            import json
            import os
            
            # Find status file
            status_file_path = None
            
            # First search in current working directory
            if hasattr(self, 'workspace_root') and self.workspace_root:
                # Try to find in parent directory of workspace
                parent_dir = os.path.dirname(self.workspace_root)
                potential_status_file = os.path.join(parent_dir, f".agibot_spawn_{agent_id}_status.json")
                if os.path.exists(potential_status_file):
                    status_file_path = potential_status_file
            
            # If not found in workspace directory, try current directory
            if not status_file_path:
                potential_status_file = os.path.join(os.getcwd(), f".agibot_spawn_{agent_id}_status.json")
                if os.path.exists(potential_status_file):
                    status_file_path = potential_status_file
            
            # Recursively search all possible directories
            if not status_file_path:
                search_dirs = [os.getcwd()]
                if hasattr(self, 'workspace_root') and self.workspace_root:
                    search_dirs.extend([
                        self.workspace_root,
                        os.path.dirname(self.workspace_root),
                        os.path.dirname(os.path.dirname(self.workspace_root))
                    ])
                
                for search_dir in search_dirs:
                    if os.path.exists(search_dir):
                        potential_file = os.path.join(search_dir, f".agibot_spawn_{agent_id}_status.json")
                        if os.path.exists(potential_file):
                            status_file_path = potential_file
                            break
            
            if not status_file_path:
                return {
                    "status": "failed",
                    "message": f"Status file for agent {agent_id} not found",
                    "agent_id": agent_id
                }
            
            # Read existing status
            try:
                with open(status_file_path, 'r', encoding='utf-8') as f:
                    status_data = json.load(f)
            except Exception as e:
                return {
                    "status": "failed",
                    "message": f"Failed to read status file: {e}",
                    "agent_id": agent_id,
                    "status_file_path": status_file_path
                }
            
            # Update loop information
            status_data["current_loop"] = current_loop
            status_data["last_loop_update"] = datetime.now().isoformat()
            
            # Write back to status file
            try:
                with open(status_file_path, 'w', encoding='utf-8') as f:
                    json.dump(status_data, f, indent=2, ensure_ascii=False)
                    f.flush()
                    import os
                    os.fsync(f.fileno()) if hasattr(f, 'fileno') else None
                
                return {
                    "status": "success",
                    "message": f"Updated current_loop for agent {agent_id} to {current_loop}",
                    "agent_id": agent_id,
                    "current_loop": current_loop,
                    "status_file_path": status_file_path
                }
                
            except Exception as e:
                return {
                    "status": "failed",
                    "message": f"Failed to write status file: {e}",
                    "agent_id": agent_id,
                    "status_file_path": status_file_path
                }
                
        except Exception as e:
            return {
                "status": "failed",
                "message": f"Error updating agent current loop: {str(e)}",
                "agent_id": agent_id
            }