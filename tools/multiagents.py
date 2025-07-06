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
import threading
import time
import queue
from typing import Dict, Any
from .print_system import print_agent, print_system, print_current


class MultiAgentTools:
    def __init__(self, workspace_root: str = None, debug_mode: bool = False):
        """Initialize multi-agent tools with a workspace root directory."""
        self.workspace_root = workspace_root or os.getcwd()
        self.debug_mode = debug_mode  # Save debug mode setting
        
        # Add session-level AGIBot task tracking
        self.session_spawned_tasks = set()
        # Add thread tracking dictionary
        self.active_threads = {}  # task_id -> thread
        
        # Save generated agent IDs for reference normalization
        self.generated_agent_ids = []

    def spawn_agibot(self, task_description: str, agent_id: str = None, output_directory: str = None, api_key: str = None, model: str = None, max_loops: int = 25, wait_for_completion: bool = False, shared_workspace: bool = True, streaming: bool = False, **kwargs) -> Dict[str, Any]:
        """
        Spawn a new AGIBot instance to handle a specific task asynchronously.
        This allows for complex task decomposition and parallel execution.
        
        Args:
            task_description: Description of the task for the new AGIBot instance
            agent_id: Custom agent ID (optional, will auto-generate if not provided). Must match format 'agent_XXX'
            output_directory: Directory where the new AGIBot should save its output (optional, will use parent's if not provided)
            api_key: API key for the new instance (optional, will use current if not provided)
            model: Model name for the new instance (optional, will use current if not provided)  
            max_loops: Maximum execution loops for the new instance
            wait_for_completion: Whether to wait for the spawned AGIBot to complete (default: False)
            shared_workspace: Whether to share parent's workspace directory (default: True)
            streaming: Whether to use streaming output (default: False for python interface, overrides config.txt)
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
            # 🔧 Fix: Handle parameter type conversion to ensure boolean and numeric parameters are correct types
            # This solves the parallel execution issue caused by string parameters
            try:
                # Convert boolean parameters
                if isinstance(wait_for_completion, str):
                    wait_for_completion = wait_for_completion.lower() in ('true', '1', 'yes', 'on')
                
                if isinstance(shared_workspace, str):
                    shared_workspace = shared_workspace.lower() in ('true', '1', 'yes', 'on')
                
                if isinstance(streaming, str):
                    streaming = streaming.lower() in ('true', '1', 'yes', 'on')
                
                # Convert numeric parameters
                if isinstance(max_loops, str):
                    max_loops = int(max_loops)
                
                # Ensure parameter types are correct
                wait_for_completion = bool(wait_for_completion)
                shared_workspace = bool(shared_workspace)
                streaming = bool(streaming)
                max_loops = int(max_loops)
                
            except (ValueError, TypeError) as e:
                return {
                    "status": "error",
                    "message": f"Invalid parameter types: max_loops={max_loops}, wait_for_completion={wait_for_completion}, shared_workspace={shared_workspace}, streaming={streaming}",
                    "error": str(e)
                }
            
            # Import AGIBot Client from main module
            import sys
            import os
            parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            sys.path.insert(0, parent_dir)
            from main import AGIBotClient
            
            # Handle agent ID generation or validation
            if agent_id is not None:
                # Validate user-provided agent ID format
                if not self._is_valid_agent_id_format(agent_id):
                    return {
                        "status": "error",
                        "message": f"Invalid agent ID format: '{agent_id}'. Must match pattern 'agent_XXX' where XXX is a 3-digit number (e.g., 'agent_001')",
                        "provided_agent_id": agent_id
                    }
                
                # Check if agent ID is already in use
                if self._is_agent_id_in_use(agent_id):
                    return {
                        "status": "error", 
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
            task_description = self._normalize_agent_references(task_description, agent_id)
            
            # Save original output_directory for comparison
            original_output_directory = output_directory
            
            # Determine the parent AGIBot's working directory
            if shared_workspace:
                # In shared workspace mode, always use parent AGIBot's output directory
                # Ignore any provided output_directory to ensure true workspace sharing
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
                
            else:
                # In independent workspace mode, use provided output_directory or auto-generate
                if output_directory is None:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    output_directory = f"output_{timestamp}"
                else:
                    pass
            
            # Get current API configuration if not provided
            if api_key is None:
                from config_loader import get_api_key
                api_key = get_api_key()
            
            if model is None:
                from config_loader import get_model
                model = get_model()
            
            # Validate required parameters
            if not api_key:
                return {
                    "status": "error",
                    "message": "API key not found. Please provide api_key parameter or set it in config.txt",
                    "agent_id": agent_id
                }
            
            if not model:
                return {
                    "status": "error", 
                    "message": "Model not found. Please provide model parameter or set it in config.txt",
                    "agent_id": agent_id
                }
            
            # Create output directory
            abs_output_dir = os.path.abspath(output_directory)
            os.makedirs(abs_output_dir, exist_ok=True)
            
            if shared_workspace:
                # For shared workspace, we need child AGIBot to work in parent's workspace
                # Since AGIBotMain always creates "workspace" subdir under its output dir,
                # we need to give child the parent's base output directory so that:
                # parent workspace = parent_output/workspace = child_output/workspace = same directory
                
                # Get parent's base output directory (one level up from workspace)
                if hasattr(self, 'workspace_root') and self.workspace_root:
                    if os.path.basename(self.workspace_root) == 'workspace':
                        parent_output_dir = os.path.dirname(self.workspace_root)
                    else:
                        parent_output_dir = self.workspace_root
                else:
                    # Fallback: use current working directory pattern
                    parent_output_dir = abs_output_dir
                
                # Child AGIBot uses parent's base output directory
                # This way, child will create workspace at parent_output/workspace = parent's workspace
                workspace_dir = parent_output_dir
                
                # Ensure parent's workspace exists
                parent_workspace = os.path.join(parent_output_dir, "workspace")
                os.makedirs(parent_workspace, exist_ok=True)
                
            else:
                # For independent workspace, create separate directory structure
                workspace_dir = os.path.join(abs_output_dir, "workspace")
                os.makedirs(workspace_dir, exist_ok=True)

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
                "shared_workspace": shared_workspace,
                "model": model,
                "max_loops": max_loops,
                "error": None
            }
            
            # Write initial status
            try:
                with open(status_file_path, 'w', encoding='utf-8') as f:
                    json.dump(initial_status, f, indent=2, ensure_ascii=False)
            except Exception as e:
                print_agent("manager", f"⚠️ Warning: Could not create status file: {e}")
            
            # Define the async task execution function
            def execute_agibot_task():
                import sys
                
                try:
                    # Print spawn start info directly to terminal
                    print_agent(task_id, f"🚀 AGIBot {task_id} started")
                    
                    # 🔧 Fix display tag issue: set current agent ID
                    from .print_system import set_agent_id
                    set_agent_id(task_id)
                    
                    # Register AGIBot mailbox
                    try:
                        from .message_system import get_message_router
                        # 🔧 Fix: Avoid cleaning existing mailboxes when starting new agent
                        router = get_message_router(workspace_dir, cleanup_on_init=False)
                        router.register_agent(task_id)
                        print_agent(task_id, f"📬 Mailbox registered for agent {task_id}")
                    except Exception as e:
                        print_agent(task_id, f"⚠️ Warning: Failed to register mailbox: {e}")
                        # Continue execution, but log the error
                    
                    # Create AGIBot client - no output redirection
                    # 🔧 Fix: Use instance's debug_mode, use default value if not specified in kwargs
                    debug_mode_to_use = kwargs.get('debug_mode', self.debug_mode)
                    client = AGIBotClient(
                        api_key=api_key,
                        model=model,
                        debug_mode=debug_mode_to_use,
                        detailed_summary=kwargs.get('detailed_summary', True),
                        single_task_mode=kwargs.get('single_task_mode', True),
                        interactive_mode=kwargs.get('interactive_mode', False),
                        streaming=streaming
                    )
                    
                    # Execute the task - use workspace directory as working directory
                    response = client.chat(
                        messages=[{"role": "user", "content": task_description}],
                        dir=workspace_dir,
                        loops=max_loops,
                        continue_mode=kwargs.get('continue_mode', False)
                    )
                    
                    # Update status file with completion
                    # 🔧 Fix: Distinguish between real failure and reaching max rounds
                    if response["success"]:
                        completion_status = {
                            "task_id": task_id,
                            "status": "completed",
                            "task_description": task_description,
                            "start_time": initial_status["start_time"],
                            "completion_time": datetime.now().isoformat(),
                            "output_directory": abs_output_dir,
                            "working_directory": workspace_dir,
                            "shared_workspace": shared_workspace,
                            "model": model,
                            "max_loops": max_loops,
                            "error": None,
                            "success": True,
                            "response": response
                        }
                    else:
                        response_message = response.get('message', 'Task failed')
                        if "reached maximum execution rounds" in response_message or "max_rounds_reached" in response_message:
                            status = "max_rounds_reached"
                            error_message = "Reached maximum rounds"
                        else:
                            status = "failed"
                            error_message = response_message
                        
                        completion_status = {
                            "task_id": task_id,
                            "status": status,
                            "task_description": task_description,
                            "start_time": initial_status["start_time"],
                            "completion_time": datetime.now().isoformat(),
                            "output_directory": abs_output_dir,
                            "working_directory": workspace_dir,
                            "shared_workspace": shared_workspace,
                            "model": model,
                            "max_loops": max_loops,
                            "error": error_message,
                            "success": False,
                            "response": response
                        }
                    
                    try:
                        with open(status_file_path, 'w', encoding='utf-8') as f:
                            json.dump(completion_status, f, indent=2, ensure_ascii=False)
                            # 🔧 Fix timing race condition: ensure status file is fully written before ending
                            f.flush()
                            import os
                            os.fsync(f.fileno()) if hasattr(f, 'fileno') else None
                    except Exception as e:
                        print_agent(task_id, f"⚠️ Warning: Could not update status file: {e}")
                    
                    # 🔧 Add longer delay to ensure status file is fully written to disk
                    time.sleep(0.5)
                    
                    # Print completion status directly to terminal
                    if response["success"]:
                        print_agent(task_id, f"✅ AGIBot spawn {task_id} completed successfully")
                    else:
                        response_message = response.get('message', 'Unknown error')
                        if "reached maximum execution rounds" in response_message or "max_rounds_reached" in response_message:
                            print_agent(task_id, f"⚠️ AGIBot spawn {task_id} reached maximum execution rounds")
                        else:
                            print_agent(task_id, f"❌ AGIBot spawn {task_id} failed: {response_message}")
                    
                    # 🔧 Fix display tag issue: reset agent ID after task completion
                    set_agent_id(None)
                        
                except Exception as e:
                    error_msg = str(e)
                    print_agent(task_id, f"❌ AGIBot spawn {task_id} error: {error_msg}")
                    
                    # Update status file with error
                    error_status = {
                        "task_id": task_id,
                        "status": "failed",
                        "task_description": task_description,
                        "start_time": initial_status["start_time"],
                        "completion_time": datetime.now().isoformat(),
                        "output_directory": abs_output_dir,
                        "working_directory": workspace_dir,
                        "shared_workspace": shared_workspace,
                        "model": model,
                        "max_loops": max_loops,
                        "error": error_msg,
                        "success": False,
                        "response": {
                            "success": False,
                            "message": error_msg
                        }
                    }
                    
                    try:
                        with open(status_file_path, 'w', encoding='utf-8') as f:
                            json.dump(error_status, f, indent=2, ensure_ascii=False)
                            # 🔧 Fix timing race condition: ensure error status file is also fully written
                            f.flush()
                            import os
                            os.fsync(f.fileno()) if hasattr(f, 'fileno') else None
                    except Exception as e:
                        print_agent(task_id, f"⚠️ Warning: Could not update status file with error: {e}")
                    
                    # 🔧 Add longer delay to ensure status file is fully written to disk
                    time.sleep(0.5)
                    
                    # 🔧 Fix display tag issue: reset agent ID after exception handling
                    set_agent_id(None)
            
            # Start the task in a separate thread
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
            
            # Simple parent status (no details to avoid confusion)
            pass
            
            # Base result information
            result = {
                "status": "success", 
                "message": f"AGIBot instance spawned successfully with agent ID: {agent_id}",
                "agent_id": agent_id,
                "output_directory": abs_output_dir,
                "working_directory": workspace_dir,  # The directory where child AGIBot runs
                "workspace_files_directory": os.path.join(abs_output_dir, "workspace") if shared_workspace else workspace_dir,
                "task_description": task_description,
                "model": model,
                "max_loops": max_loops,
                "thread_started": thread.is_alive(),
                "shared_workspace": shared_workspace,
                "status_file": status_file_path,
                "agent_communication_note": f"✅ Use agent ID '{task_id}' for all message sending and receiving operations",
                "spawn_mode": "asynchronous" if not wait_for_completion else "synchronous"
            }
            
            if wait_for_completion:
                print_agent("manager", f"⏳ Waiting for AGIBot spawn {task_id} to complete...")
                
                result["note"] = "Waiting for task completion..."
                
                # Wait for the thread to complete
                thread.join()
                
                # Read final status from file
                try:
                    with open(status_file_path, 'r', encoding='utf-8') as f:
                        final_status = json.load(f)
                    result.update({
                        "status": final_status["status"],
                        "completion_time": final_status["completion_time"], 
                        "success": final_status.get("success", False),
                        "error": final_status.get("error", None),
                        "note": "Task completed synchronously."
                    })
                except Exception as e:
                    result.update({
                        "status": "error",
                        "note": f"Task thread completed but status file could not be read: {e}"
                    })
                
                print_agent("manager", f"✅ Spawn {task_id} completed")
                
            else:
                result["note"] = f"✅ AGIBot {task_id} is running asynchronously in background. Task will execute independently and send messages when completed."
                result["success"] = True  # Explicitly mark as success
                result["thread_id"] = thread.ident if thread else None
            
            return result
            
        except Exception as e:
            return {
                "status": "error",
                "message": f"Failed to spawn AGIBot instance: {str(e)}",
                "task_id": task_id if 'task_id' in locals() else "unknown"
            }
    
    def wait_for_agibot_spawns(self, task_ids: list = None, output_directories: list = None, check_interval: int = 5, max_wait_time: int = 3600) -> Dict[str, Any]:
        """
        Wait for multiple spawned AGIBot instances to complete.
        
        Args:
            task_ids: List of task IDs to wait for (optional, will use session-tracked tasks if not provided)
            output_directories: List of output directories to check (optional, only used if task_ids is also provided)
            check_interval: Interval in seconds between status checks (default: 5)
            max_wait_time: Maximum time to wait in seconds (default: 3600 = 1 hour)
            
        Returns:
            Dict containing completion status of all tasks
            
        Note:
            - If no task_ids provided, will use tasks spawned in current session
            - No longer auto-discovers all historical tasks to avoid confusion
        """
        import time
        import json
        import glob
        from datetime import datetime
        
        try:
            # 🔧 Fix: Handle all parameter type conversions to ensure numeric parameters are correct types
            # Fix type comparison error: ensure check_interval and max_wait_time are numeric types
            try:
                check_interval = int(check_interval) if check_interval else 5
                max_wait_time = int(max_wait_time) if max_wait_time else 3600
            except (ValueError, TypeError) as e:
                return {
                    "status": "error",
                    "message": f"Invalid parameter types: check_interval={check_interval}, max_wait_time={max_wait_time}",
                    "error": str(e)
                }
            
            print_current(f"🔧 Parameter validation: check_interval={check_interval}s, max_wait_time={max_wait_time}s")
            
            # 🔧 Fix: Handle task_ids parameter type conversion
            if task_ids is not None:
                # If task_ids is a string (JSON format), try to parse it as a list
                if isinstance(task_ids, str):
                    import json
                    try:
                        task_ids = json.loads(task_ids)
                        print_current(f"🔧 Parsed task_ids from string: {task_ids}")
                    except json.JSONDecodeError as e:
                        return {
                            "status": "error",
                            "message": f"Invalid task_ids format. Expected list or JSON array string, got: {task_ids}",
                            "error": str(e)
                        }
                
                # Ensure task_ids is a list
                if not isinstance(task_ids, list):
                    return {
                        "status": "error",
                        "message": f"task_ids must be a list, got {type(task_ids).__name__}: {task_ids}"
                    }
            
            # Also handle output_directories parameter
            if output_directories is not None:
                if isinstance(output_directories, str):
                    import json
                    try:
                        output_directories = json.loads(output_directories)
                        print_current(f"🔧 Parsed output_directories from string: {output_directories}")
                    except json.JSONDecodeError:
                        # If not JSON, might be a single directory path
                        output_directories = [output_directories]
                
                if not isinstance(output_directories, list):
                    output_directories = [output_directories]
            
            # Prioritize session-tracked tasks if task_ids not provided
            if not task_ids and not output_directories:
                # Use tasks started in current session
                if hasattr(self, 'session_spawned_tasks') and self.session_spawned_tasks:
                    task_ids = list(self.session_spawned_tasks)
                else:
                    return {
                        "status": "error",
                        "message": "No AGIBot spawn tasks found in current session. Please provide task_ids or spawn AGIBots first.",
                        "discovered_tasks": 0
                    }
            
            # If task_ids provided, use the provided task IDs
            if task_ids:
                task_info = []
                print_current(f"🔍 Searching for status files for task IDs: {task_ids}")
                
                for task_id in task_ids:
                    # 🔧 Modify: Skip root directory scan, only search for status files in output_* directories
                    possible_status_files = []
                    
                    # Check output_* directories under current directory
                    for output_pattern in glob.glob("output_*"):
                        if os.path.isdir(output_pattern):
                            status_path = os.path.join(output_pattern, f".agibot_spawn_{task_id}_status.json")
                            if os.path.exists(status_path):
                                possible_status_files.append((status_path, output_pattern))
                    
                    if possible_status_files:
                        # Select the latest status file (sorted by directory name, usually containing timestamp)
                        possible_status_files.sort(key=lambda x: x[1], reverse=True)
                        status_file, output_dir = possible_status_files[0]
                        task_info.append({"task_id": task_id, "output_dir": output_dir, "status_file": status_file})
                    else:
                        print_current(f"⚠️ Warning: Status file not found for task {task_id}")
                        print_current(f"🔍 Searched only in output_* subdirectories (root directory scanning disabled)")
                
                if not task_info:
                    return {
                        "status": "error",
                        "message": f"No valid status files found for provided task IDs: {task_ids}",
                        "provided_task_ids": task_ids,
                        "parameters_used": {
                            "task_ids": str(task_ids),
                            "check_interval": str(check_interval),
                            "max_wait_time": str(max_wait_time)
                        }
                    }
            # Handle output_directories parameter (if provided)
            if output_directories and task_ids:
                # If both task_ids and output_directories are provided, use specified directories
                task_info = []
                for task_id, output_dir in zip(task_ids, output_directories):
                    status_file = os.path.join(output_dir, f".agibot_spawn_{task_id}_status.json")
                    task_info.append({"task_id": task_id, "output_dir": output_dir, "status_file": status_file})
            
            if not task_info:
                return {
                    "status": "error",
                    "message": "No valid tasks found to wait for",
                    "provided_task_ids": task_ids,
                    "provided_directories": output_directories
                }
            
            print_system(f"⏳ Waiting for {len(task_info)} AGIBot spawn tasks to complete...")
            print_system(f"🔍 Check interval: {check_interval} seconds")
            print_system(f"⏰ Max wait time: {max_wait_time} seconds")
            
            start_time = time.time()
            completed_tasks = {}
            
            while True:
                current_time = time.time()
                elapsed_time = current_time - start_time
                
                # Check if max wait time exceeded
                if elapsed_time > max_wait_time:
                    return {
                        "status": "timeout",
                        "message": f"Maximum wait time ({max_wait_time} seconds) exceeded",
                        "elapsed_time": elapsed_time,
                        "completed_tasks": completed_tasks,
                        "remaining_tasks": [t["task_id"] for t in task_info if t["task_id"] not in completed_tasks]
                    }
                
                # Check status of each task
                pending_tasks = []
                for task in task_info:
                    task_id = task["task_id"]
                    
                    # Skip already completed tasks
                    if task_id in completed_tasks:
                        continue
                    
                    status_file = task["status_file"]
                    
                    # 🔧 Improve: Thread alive check - Add time delay check to avoid false positives
                    thread_alive = True  # Assume thread is alive by default
                    thread_dead_confirmed = False
                    if hasattr(self, 'active_threads') and task_id in self.active_threads:
                        thread = self.active_threads[task_id]
                        thread_alive = thread.is_alive()
                        
                        # If thread is not alive, wait a short time again to confirm, avoid timing issues
                        if not thread_alive:
                            time.sleep(0.5)  # Wait 500ms
                            thread_alive = thread.is_alive()  # Check again
                            if not thread_alive:
                                thread_dead_confirmed = True
                                print_agent(task_id, f"⚠️ Thread for task {task_id} confirmed dead after double check")
                    
                    if os.path.exists(status_file):
                        try:
                            with open(status_file, 'r', encoding='utf-8') as f:
                                status_data = json.load(f)
                            
                            # 🔧 Fix timing race condition: Avoid falsely marking normal completion tasks
                            # Only mark as failed if thread actually died and status file still shows running
                            if (status_data["status"] == "running" and thread_dead_confirmed and
                                elapsed_time > 10):  # Only check for death after at least 10 seconds of running
                                
                                # 🔧 Use multiple retry mechanism, give status file more time to write
                                retry_count = 0
                                max_retries = 8  # Increase retry count
                                retry_delay = 0.3  # 300ms interval
                                status_updated = False
                                
                                print_agent(task_id, f"⏳ Starting {max_retries} retries to check status file update...")
                                
                                while retry_count < max_retries and not status_updated:
                                    retry_count += 1
                                    time.sleep(retry_delay)
                                    
                                    try:
                                        with open(status_file, 'r', encoding='utf-8') as f:
                                            updated_status = json.load(f)
                                        
                                        # If status file has been updated to a non-running state, it means the task completed normally
                                        if updated_status["status"] != "running":
                                            print_agent(task_id, f"✅ Task {task_id} completed normally (status updated after {retry_count} retries)")
                                            status_data = updated_status  # Use updated status
                                            status_updated = True
                                        else:
                                            print_agent(task_id, f"🔄 Retry {retry_count}/{max_retries}: Status still running, waiting...")
                                    
                                    except Exception as e:
                                        print_agent(task_id, f"⚠️ Error re-reading status file on retry {retry_count}: {e}")
                                        # Continue retrying
                                
                                # If all retries failed, mark as true abnormal exit
                                if not status_updated:
                                    print_agent(task_id, f"💀 Task {task_id} thread died without updating status file (confirmed after {max_retries} retries)")
                                    # Update status file
                                    status_data["status"] = "failed"
                                    status_data["completion_time"] = datetime.now().isoformat()
                                    status_data["error"] = "Thread terminated unexpectedly without updating status"
                                    status_data["success"] = False
                                    
                                    try:
                                        with open(status_file, 'w', encoding='utf-8') as f:
                                            json.dump(status_data, f, indent=2, ensure_ascii=False)
                                        print_agent(task_id, f"📝 Updated status file for dead thread {task_id}")
                                    except Exception as e:
                                        print_agent(task_id, f"⚠️ Failed to update status file for {task_id}: {e}")
                                else:
                                    print_agent(task_id, f"🎯 Successfully detected status update after retries")
                            
                            # 🔧 Fix: Only add to completed_tasks when task is truly completed
                            if status_data["status"] != "running":
                                # Task is completed (success or failure)
                                completed_tasks[task_id] = {
                                    "status": status_data["status"],
                                    "completion_time": status_data["completion_time"],
                                    "output_directory": task["output_dir"],
                                    "success": status_data.get("success", status_data["status"] == "completed"),
                                    "error": status_data.get("error", None),
                                    "log_file": status_data.get("log_file", None)
                                }
                                
                                print_agent(task_id, f"✅ Task {task_id}: {status_data['status'].upper()}")
                                
                                # Clean up thread reference
                                if hasattr(self, 'active_threads') and task_id in self.active_threads:
                                    del self.active_threads[task_id]
                            else:
                                # Task is still running
                                pending_tasks.append(task)
                            
                        except Exception as e:
                            print_agent(task_id, f"⚠️ Error reading status for task {task_id}: {e}")
                            pending_tasks.append(task)
                    else:
                        # Status file doesn't exist yet
                        if thread_dead_confirmed and hasattr(self, 'active_threads') and task_id in self.active_threads:
                            # Thread is dead but no status file - only mark as failed after running for a long time
                            if elapsed_time > 30:  # Only consider it a true failure after at least 30 seconds of running
                                print_agent(task_id, f"💀 Task {task_id} thread died without creating status file")
                                completed_tasks[task_id] = {
                                    "status": "failed",
                                    "completion_time": datetime.now().isoformat(),
                                    "output_directory": task["output_dir"],
                                    "success": False,
                                    "error": "Thread terminated before creating status file",
                                    "log_file": None
                                }
                                del self.active_threads[task_id]
                            else:
                                # Still in startup phase, continue waiting
                                pending_tasks.append(task)
                        else:
                            # Status file doesn't exist yet, task might still be starting
                            pending_tasks.append(task)
                
                # All tasks completed
                if not pending_tasks:
                    successful_tasks = sum(1 for t in completed_tasks.values() if t["success"])
                    failed_tasks = len(completed_tasks) - successful_tasks
                    
                    print_system(f"🎉 All tasks completed! Success: {successful_tasks}, Failed: {failed_tasks}")
                    
                    return {
                        "status": "completed",
                        "message": f"All {len(completed_tasks)} tasks completed",
                        "elapsed_time": elapsed_time,
                        "total_tasks": len(completed_tasks),
                        "successful_tasks": successful_tasks,
                        "failed_tasks": failed_tasks,
                        "completed_tasks": completed_tasks
                    }
                
                # Update task_info to only include pending tasks
                task_info = pending_tasks
                
                # Show progress
                print_system(f"⏳ {len(completed_tasks)} completed, {len(pending_tasks)} pending... (elapsed: {elapsed_time:.1f}s)")
                
                # Wait before next check
                time.sleep(check_interval)
                
        except Exception as e:
            return {
                "status": "error",
                "message": f"Error while waiting for AGIBot spawns: {str(e)}",
                "error": str(e)
            }

    def send_message_to_agent(self, receiver_id: str, message_type: str, content: dict, priority: str = "normal") -> Dict[str, Any]:
        """
        Send message to specified agent
        
        Args:
            receiver_id: Receiver agent ID
            message_type: Message type (status_update, task_request, collaboration, broadcast, system, error)
            content: Message content
            priority: Message priority (low, normal, high, urgent)
            
        Returns:
            Send result dictionary
        """
        try:
            from .message_system import Message, MessageType, MessagePriority, get_message_router
            from .print_system import get_agent_id
            
            # Get message router
            router = get_message_router(self.workspace_root, cleanup_on_init=False)
            
            # Convert enum types
            try:
                msg_type = MessageType(message_type)
            except ValueError:
                return {"status": "error", "message": f"Invalid message type: {message_type}"}
            
            try:
                msg_priority = MessagePriority[priority.upper()]
            except KeyError:
                msg_priority = MessagePriority.NORMAL
            
            # 🔧 Get current agent ID as sender_id, if not, use manager
            current_agent_id = get_agent_id()
            sender_id = current_agent_id if current_agent_id else "manager"
            
            # Create message
            message = Message(
                sender_id=sender_id,
                receiver_id=receiver_id,
                message_type=msg_type,
                content=content,
                priority=msg_priority
            )
            
            # 🔧 Use the correct sender's mailbox to send the message
            sender_mailbox = router.get_mailbox(sender_id)
            if not sender_mailbox:
                sender_mailbox = router.register_agent(sender_id)
            
            # Send message
            success = sender_mailbox.send_message(message)
            
            # 🔧 Fix: Manually trigger message processing, instead of background processing
            if success:
                try:
                    processed_count = router.process_all_messages_once()
                    if processed_count > 0:
                        print_current(f"📬 Processed {processed_count} messages after sending")
                except Exception as e:
                    print_current(f"⚠️ Error processing messages after send: {e}")
            
            return {
                "status": "success" if success else "failed",
                "message": "Message sent successfully" if success else "Failed to send message",
                "receiver_id": receiver_id,
                "message_type": message_type,
                "message_id": message.message_id
            }
            
        except Exception as e:
            return {
                "status": "error",
                "message": f"Error sending message: {str(e)}",
                "receiver_id": receiver_id,
                "message_type": message_type
            }

    def get_agent_messages(self, agent_id: str = "manager", include_read: bool = False) -> Dict[str, Any]:
        """
        Get messages for specified agent
        
        Args:
            agent_id: Agent ID
            include_read: Whether to include read messages
            
        Returns:
            Message list and statistics
        """
        try:
            from .message_system import get_message_router
            import glob
            
            # 🔧 Fix: Try to find agent mailbox from multiple possible workspace paths
            workspace_paths = []
            
            # Add current workspace_root
            if self.workspace_root:
                workspace_paths.append(self.workspace_root)
            
            # Add current directory
            workspace_paths.append(os.getcwd())
            
            # Search all output_* directories (these are usually agent working directories)
            output_dirs = glob.glob("output_*")
            for output_dir in output_dirs:
                if os.path.isdir(output_dir):
                    workspace_paths.append(os.path.abspath(output_dir))
            
            # Deduplicate
            workspace_paths = list(set(workspace_paths))
            
            # Try to get agent mailbox from all possible workspace paths
            mailbox = None
            found_workspace = None
            all_available_agents = set()
            
            for workspace_path in workspace_paths:
                try:
                    router = get_message_router(workspace_path, cleanup_on_init=False)
                    temp_mailbox = router.get_mailbox(agent_id)
                    available_agents = router.get_all_agents()
                    all_available_agents.update(available_agents)
                    
                    if temp_mailbox:
                        mailbox = temp_mailbox
                        found_workspace = workspace_path
                        break  # Stop searching once found
                        
                except Exception as e:
                    # Ignore single path errors, continue trying other paths
                    continue
            
            if not mailbox:
                # Provide more detailed error information, including all found agents
                available_agents_list = list(all_available_agents)
                if available_agents_list:
                    return {
                        "status": "error", 
                        "message": f"Agent '{agent_id}' not found in any workspace. Available agents: {', '.join(available_agents_list)}",
                        "available_agents": available_agents_list,
                        "searched_workspaces": workspace_paths
                    }
                else:
                    return {
                        "status": "error", 
                        "message": f"Agent '{agent_id}' not found. No agents are currently registered in any workspace.",
                        "available_agents": [],
                        "searched_workspaces": workspace_paths
                    }
            
            # Get messages
            if include_read:
                # Get all messages (including read)
                messages = mailbox.get_all_messages()
            else:
                # Only get unread messages
                messages = mailbox.get_unread_messages()
            
            # Convert message format
            messages_data = [msg.to_dict() for msg in messages]
            
            # Get mailbox statistics
            stats = mailbox.get_message_stats()
            
            return {
                "status": "success",
                "agent_id": agent_id,
                "message_count": len(messages_data),
                "messages": messages_data,
                "mailbox_stats": stats,
                "found_in_workspace": found_workspace
            }
            
        except Exception as e:
            return {
                "status": "error",
                "message": f"Error getting messages: {str(e)}",
                "agent_id": agent_id
            }

    def send_message_to_manager(self, message_content: str, message_type: str = "system") -> Dict[str, Any]:
        """
        Send message to manager (convenience function)
        
        Args:
            message_content: Message content
            message_type: Message type (system, status_update, task_request, etc.) - default: system
            
        Returns:
            Send result
        """
        # Convenience function: internally calls the generic send_message_to_agent
        return self.send_message_to_agent(
            receiver_id="manager",
            message_type=message_type,
            content={"message": message_content}
        )

    def send_status_update_to_manager(self, agent_id: str, round_number: int, task_completed: bool, 
                                     llm_response_preview: str, tool_calls_summary: list, 
                                     current_task_description: str = "", error_message: str = None) -> Dict[str, Any]:
        """
        Send status update to manager
        
        Args:
            agent_id: Agent ID
            round_number: Current round number
            task_completed: Whether task is completed
            llm_response_preview: LLM response preview
            tool_calls_summary: Tool calls summary
            current_task_description: Current task description
            error_message: Error message
            
        Returns:
            Send result
        """
        try:
            from .message_system import Message, MessageType, MessagePriority, StatusUpdateMessage, get_message_router
            
            # Get message router
            router = get_message_router(self.workspace_root)
            
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
                sender_id=agent_id,
                receiver_id="manager",
                message_type=MessageType.STATUS_UPDATE,
                content=content,
                priority=MessagePriority.NORMAL
            )
            
            # Get sender mailbox
            sender_mailbox = router.get_mailbox(agent_id)
            if not sender_mailbox:
                sender_mailbox = router.register_agent(agent_id)
            
            # Send message
            success = sender_mailbox.send_message(message)
            
            return {
                "status": "success" if success else "failed",
                "message": "Status update sent to manager successfully" if success else "Failed to send status update to manager",
                "message_id": message.message_id,
                "agent_id": agent_id,
                "round_number": round_number
            }
            
        except Exception as e:
            return {
                "status": "error",
                "message": f"Error sending status update to manager: {str(e)}",
                "agent_id": agent_id
            }

    def broadcast_message_to_agents(self, message_type: str, content: dict) -> Dict[str, Any]:
        """
        Broadcast message to all agents
        
        Args:
            message_type: Message type
            content: Message content
            
        Returns:
            Broadcast result
        """
        try:
            from .message_system import MessageType, get_message_router
            
            # Get message router
            router = get_message_router(self.workspace_root)
            
            # Convert message type
            try:
                msg_type = MessageType(message_type)
            except ValueError:
                return {"status": "error", "message": f"Invalid message type: {message_type}"}
            
            # Broadcast message
            sent_count = router.broadcast_message("manager", content)
            
            return {
                "status": "success",
                "message": f"Broadcast sent to {sent_count} agents",
                "sent_count": sent_count,
                "message_type": message_type
            }
            
        except Exception as e:
            return {
                "status": "error",
                "message": f"Error broadcasting message: {str(e)}",
                "message_type": message_type
            }

    def mark_message_as_read(self, message_id: str, agent_id: str = "manager") -> Dict[str, Any]:
        """
        Mark message as read
        
        Args:
            message_id: Message ID
            agent_id: Agent ID
            
        Returns:
            Operation result
        """
        try:
            from .message_system import get_message_router
            
            # Get message router
            router = get_message_router(self.workspace_root)
            
            # Get agent mailbox
            mailbox = router.get_mailbox(agent_id)
            if not mailbox:
                return {"status": "error", "message": "Agent mailbox not found"}
            
            # Mark message as read
            success = mailbox.mark_as_read(message_id)
            
            return {
                "status": "success" if success else "failed",
                "message": "Message marked as read" if success else "Failed to mark message as read",
                "message_id": message_id,
                "agent_id": agent_id
            }
            
        except Exception as e:
            return {
                "status": "error",
                "message": f"Error marking message as read: {str(e)}",
                "message_id": message_id,
                "agent_id": agent_id
            }

    def get_agent_session_info(self) -> Dict[str, Any]:
        """
        Get agent session information
        
        Returns:
            Session information dictionary
        """
        try:
            from .message_system import get_message_router
            
            # Get message router
            router = get_message_router(self.workspace_root, cleanup_on_init=False)
            
            # Get all registered agents
            all_agents = router.get_all_agents()
            
            # Statistics
            total_agents = len(all_agents)
            active_agents = total_agents  # Simplified processing, all registered agents are considered active
            
            # 🔧 Add terminal print output
            print_current("📊 ===========================================")
            print_current("📊 AGIBot Session Information")
            print_current("📊 ===========================================")
            print_current(f"📊 Total Agents: {total_agents}")
            print_current(f"📊 Active Agents: {active_agents}")
            print_current(f"📊 Completed Agents: 0")
            print_current(f"📊 Failed Agents: 0")
            print_current(f"📊 Message System Status: Active")
            print_current(f"📊 Registered Agents: {', '.join(all_agents) if all_agents else 'None'}")
            print_current("📊 ===========================================")
            
            result = {
                "status": "success",
                "session_id": "default_session",
                "session_start_time": "2025-01-01T00:00:00",
                "total_agents": total_agents,
                "active_agents": active_agents,
                "completed_agents": 0,
                "failed_agents": 0,
                "message_system_active": True,
                "registered_agents": all_agents,
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
                "status": "error",
                "message": error_msg
            }

    def list_active_agents(self) -> Dict[str, Any]:
        """
        List currently active agents based on thread status
        
        Returns:
            Active agents list
        """
        try:
            # Based on actual thread status to detect active agents, not relying on potentially cleaned-up mailboxes
            agents_info = []
            
            # Check active threads
            if hasattr(self, 'active_threads'):
                for agent_id, thread in self.active_threads.items():
                    if thread.is_alive():
                        agents_info.append({
                            "task_id": agent_id,
                            "status": "active",
                            "task_description": f"Agent {agent_id}",
                            "start_time": "2025-01-01T00:00:00",
                            "rounds_completed": 0,
                            "current_task": "Running",
                            "last_update": "2025-01-01T00:00:00",
                            "thread_id": thread.ident,
                            "thread_name": thread.name
                        })
            
            # 🔧 Fix: Also check registered agents in message system as an alternative - use multiple workspace_root paths
            try:
                from .message_system import get_message_router
                import glob
                
                # Collect all possible workspace paths
                workspace_paths = []
                
                # Add current workspace_root
                if self.workspace_root:
                    workspace_paths.append(self.workspace_root)
                
                # Add current directory
                workspace_paths.append(os.getcwd())
                
                # Search all output_* directories (these are usually agent working directories)
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
                    except Exception as e:
                        # Ignore single path errors, continue trying other paths
                        pass
                
                # Add agents registered in message system but not in thread tracking
                existing_agent_ids = {agent["task_id"] for agent in agents_info}
                for agent_id in all_registered_agents:
                    if agent_id not in existing_agent_ids and agent_id != "manager":
                        agents_info.append({
                            "task_id": agent_id,
                            "status": "registered",
                            "task_description": f"Agent {agent_id}",
                            "start_time": "2025-01-01T00:00:00",
                            "rounds_completed": 0,
                            "current_task": "Unknown",
                            "last_update": "2025-01-01T00:00:00"
                        })
                        
            except Exception as e:
                print_current(f"⚠️ Error checking message system agents: {e}")
            
            # 🔧 Add terminal print output
            print_current("🤖 ===========================================")
            print_current("🤖 Active AGIBot List")
            print_current("🤖 ===========================================")
            print_current(f"🤖 Detected Active Agents: {len(agents_info)}")
            
            if agents_info:
                for i, agent in enumerate(agents_info, 1):
                    status_icon = "🟢" if agent.get("status") == "active" else "🔵"
                    print_current(f"🤖 {i}. {status_icon} {agent['task_id']} - {agent['status']}")
                    if agent.get("thread_id"):
                        print_current(f"   └─ Thread ID: {agent['thread_id']}, Thread Name: {agent.get('thread_name', 'Unknown')}")
                    else:
                        print_current(f"   └─ Mailbox Registration Status: {agent['status']}")
            else:
                print_current("🤖 No active AGIBot detected")
                print_current("🤖 Tip: Check thread tracking and message system registration status")
                
                # Add debug information
                if hasattr(self, 'active_threads'):
                    print_current(f"🤖 Thread tracking dictionary size: {len(self.active_threads)}")
                    if self.active_threads:
                        print_current(f"🤖 Agents in thread tracking: {list(self.active_threads.keys())}")
                else:
                    print_current("🤖 Thread tracking not initialized")
            
            print_current("🤖 ===========================================")
            
            return {
                "status": "success",
                "active_count": len(agents_info),
                "agents": agents_info,
                "detection_method": "thread_status_primary_multi_workspace"
            }
            
        except Exception as e:
            return {
                "status": "error",
                "message": f"Error listing active agents: {str(e)}",
                "active_count": 0,
                "agents": []
            }
    
    def debug_thread_status(self) -> Dict[str, Any]:
        """
        Debug function: Check current active thread status
        
        Returns:
            Thread status information
        """
        try:
            if not hasattr(self, 'active_threads'):
                return {
                    "status": "info",
                    "message": "No thread tracking initialized",
                    "active_threads": {}
                }
            
            thread_status = {}
            dead_threads = []
            
            for task_id, thread in self.active_threads.items():
                is_alive = thread.is_alive()
                thread_status[task_id] = {
                    "thread_id": thread.ident,
                    "thread_name": thread.name,
                    "is_alive": is_alive,
                    "is_daemon": thread.daemon
                }
                if not is_alive:
                    dead_threads.append(task_id)
            
            return {
                "status": "success",
                "message": f"Found {len(self.active_threads)} tracked threads, {len(dead_threads)} dead",
                "total_threads": len(self.active_threads),
                "alive_threads": len(self.active_threads) - len(dead_threads),
                "dead_threads": dead_threads,
                "thread_details": thread_status
            }
            
        except Exception as e:
            return {
                "status": "error",
                "message": f"Error checking thread status: {str(e)}"
            }
    
    def debug_workspace_paths(self) -> Dict[str, Any]:
        """
        Debug function: Check all possible workspace paths and message system status
        
        Returns:
            Workspace paths and message system diagnostic information
        """
        try:
            from .message_system import get_message_router
            import glob
            
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
            
            # Check message system status for each path
            workspace_status = {}
            all_agents = set()
            
            for workspace_path in workspace_paths:
                try:
                    router = get_message_router(workspace_path, cleanup_on_init=False)
                    agents = router.get_all_agents()
                    mailbox_dir = getattr(router, 'mailbox_root', None)
                    
                    workspace_status[workspace_path] = {
                        "status": "accessible",
                        "agents": agents,
                        "agent_count": len(agents),
                        "mailbox_directory": mailbox_dir,
                        "mailbox_exists": os.path.exists(mailbox_dir) if mailbox_dir else False
                    }
                    
                    all_agents.update(agents)
                    
                except Exception as e:
                    workspace_status[workspace_path] = {
                        "status": "error",
                        "error": str(e),
                        "agents": [],
                        "agent_count": 0
                    }
            
            return {
                "status": "success",
                "message": f"Found {len(workspace_paths)} workspace paths, {len(all_agents)} total agents",
                "current_workspace_root": self.workspace_root,
                "current_working_directory": os.getcwd(),
                "searched_paths": workspace_paths,
                "workspace_details": workspace_status,
                "all_discovered_agents": list(all_agents),
                "total_unique_agents": len(all_agents)
            }
            
        except Exception as e:
            return {
                "status": "error",
                "message": f"Error checking workspace paths: {str(e)}"
            }
    
    def _normalize_agent_references(self, task_description: str, current_agent_id: str) -> str:
        """
        Normalize agent references in task description, converting non-standard formats to correct agent ID format
        """
        import re
        
        # Create a map of generated agent IDs (excluding the current agent)
        available_agents = [aid for aid in self.generated_agent_ids if aid != current_agent_id]
        
        # Define common error formats and their replacement rules
        replacements = []
        
        # If there are other agents, create a map
        if available_agents:
            # AGIBot-1, AGIBot-2, AGIBot-3 format
            for i, agent_id in enumerate(available_agents, 1):
                replacements.append((f"AGIBot-{i}", agent_id))
                replacements.append((f"AGIBot {i}", agent_id))
                replacements.append((f"Agent-{i}", agent_id))
                replacements.append((f"Agent {i}", agent_id))
                replacements.append((f"Agent_{i}", agent_id))
                replacements.append((f"agent-{i}", agent_id))
                replacements.append((f"agent {i}", agent_id))
                replacements.append((f"agent_{i}", agent_id))
        
        # Perform replacements
        normalized_desc = task_description
        for old_ref, new_ref in replacements:
            normalized_desc = normalized_desc.replace(old_ref, new_ref)
        
        # If replacements were found, print a hint message
        if normalized_desc != task_description:
            print_current(f"📝 Normalized agent references in task description:")
            for old_ref, new_ref in replacements:
                if old_ref in task_description:
                    print_current(f"   {old_ref} → {new_ref}")
        
        return normalized_desc
    
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
        # For example: agent_001, agent_main, agent_primary, agent_test_1 etc.
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
            router = get_message_router(self.workspace_root)
            # If mailbox already exists, it means the agent ID is already in use
            if hasattr(router, 'mailboxes') and agent_id in router.mailboxes:
                return True
        except Exception:
            # If message system cannot be checked, ignore this check
            pass
        
        return False

    def cleanup(self):
        """Clean up multi-agent system resources"""
        try:
            # Clean up active threads
            if hasattr(self, 'active_threads'):
                for task_id, thread in self.active_threads.items():
                    if thread.is_alive():
                        print_current(f"⏳ Waiting for thread {task_id} to complete...")
                        # Do not force termination, let the thread end naturally
                
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
            print_current(f"❌ Error cleaning up multi-agent system resources: {e}")

    def __del__(self):
        """Destructor, ensure resources are cleaned up"""
        try:
            self.cleanup()
        except:
            pass