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
        
        # 🔧 新增：添加terminated agents跟踪
        self.terminated_agents = set()  # 跟踪已终止的agents
        self.completed_agents = set()   # 跟踪已完成的agents
        
        # Save generated agent IDs for reference normalization
        self.generated_agent_ids = []

    def spawn_agibot(self, task_description: str, agent_id: str = None, output_directory: str = None, api_key: str = None, model: str = None, max_loops: int = 25, wait_for_completion: bool = False, shared_workspace: bool = True, MCP_config_file: str = None, prompts_folder: str = None, **kwargs) -> Dict[str, Any]:
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
            # Handle parameter type conversion to ensure boolean and numeric parameters are correct types
            try:
                # Convert boolean parameters
                if isinstance(wait_for_completion, str):
                    wait_for_completion = wait_for_completion.lower() in ('true', '1', 'yes', 'on')
                
                if isinstance(shared_workspace, str):
                    shared_workspace = shared_workspace.lower() in ('true', '1', 'yes', 'on')
                
                # Convert numeric parameters
                if isinstance(max_loops, str):
                    max_loops = int(max_loops)
                
                # Ensure parameter types are correct
                wait_for_completion = bool(wait_for_completion)
                shared_workspace = bool(shared_workspace)
                max_loops = int(max_loops)
                
            except (ValueError, TypeError) as e:
                return {
                    "status": "error",
                    "message": f"Invalid parameter types: max_loops={max_loops}, wait_for_completion={wait_for_completion}, shared_workspace={shared_workspace}",
                    "error": str(e)
                }
            
            # Get streaming configuration from config.txt
            try:
                from config_loader import get_streaming
                streaming = get_streaming()
            except:
                streaming = False  # Default fallback
            
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
            
            # Determine the parent AGIBot's working directory
            if shared_workspace:
                # In shared workspace mode, always use parent AGIBot's output directory
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
                
            else:
                # For independent workspace, create separate directory structure
                workspace_dir = os.path.join(abs_output_dir, "workspace")
                os.makedirs(workspace_dir, exist_ok=True)

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
                        "status": "error",
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
                print_current(f"⚠️ Warning: Could not create status file: {e}")
            
            # Define the async task execution function
            def execute_agibot_task():
                import sys
                
                try:
                    # Print spawn start info directly to terminal
                    print_agent(task_id, f"🚀 AGIBot {task_id} started")
                    
                    # Set current agent ID
                    from .print_system import set_agent_id
                    set_agent_id(task_id)
                    
                    # Register AGIBot mailbox
                    try:
                        from .message_system import get_message_router
                        router = get_message_router(workspace_dir, cleanup_on_init=False)
                        router.register_agent(task_id)
                        print_agent(task_id, f"📬 Mailbox registered for agent {task_id}")
                    except Exception as e:
                        print_agent(task_id, f"⚠️ Warning: Failed to register mailbox: {e}")
                    
                    # Create AGIBot client
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
                        prompts_folder=prompts_folder
                    )
                    
                    # Execute the task
                    response = client.chat(
                        messages=[{"role": "user", "content": task_description}],
                        dir=workspace_dir,
                        loops=max_loops,
                        continue_mode=kwargs.get('continue_mode', False)
                    )
                    
                    # 🔧 修复：检查是否收到了terminate信号
                    is_terminated = False
                    if isinstance(response.get('message'), str) and 'AGENT_TERMINATED' in response.get('message', ''):
                        is_terminated = True
                        print_agent(task_id, f"🛑 Agent {task_id} received terminate signal and will exit")
                    
                    # Update status file with completion
                    if is_terminated:
                        # 🔧 新增：处理terminate信号的状态更新
                        terminate_status = {
                            "agent_id": task_id,
                            "status": "terminated",
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
                            "terminated": True,
                            "response": response
                        }
                        
                        try:
                            with open(status_file_path, 'w', encoding='utf-8') as f:
                                json.dump(terminate_status, f, indent=2, ensure_ascii=False)
                                f.flush()
                                import os
                                os.fsync(f.fileno()) if hasattr(f, 'fileno') else None
                            print_agent(task_id, f"📁 Status file updated: terminated")
                        except Exception as e:
                            print_agent(task_id, f"⚠️ Warning: Could not update status file for termination: {e}")
                            
                    elif response["success"]:
                        completion_status = {
                            "agent_id": task_id,
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
                        
                        # 🔧 新增：将agent添加到completed_agents集合
                        if hasattr(self, 'completed_agents'):
                            self.completed_agents.add(task_id)
                        
                        try:
                            with open(status_file_path, 'w', encoding='utf-8') as f:
                                json.dump(completion_status, f, indent=2, ensure_ascii=False)
                                f.flush()
                                import os
                                os.fsync(f.fileno()) if hasattr(f, 'fileno') else None
                        except Exception as e:
                            print_agent(task_id, f"⚠️ Warning: Could not update status file: {e}")
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
                                f.flush()
                                import os
                                os.fsync(f.fileno()) if hasattr(f, 'fileno') else None
                        except Exception as e:
                            print_agent(task_id, f"⚠️ Warning: Could not update status file: {e}")
                    
                    time.sleep(0.5)
                    
                    # Print completion status
                    if is_terminated:
                        print_agent(task_id, f"🛑 AGIBot spawn {task_id} terminated successfully")
                    elif response["success"]:
                        print_agent(task_id, f"✅ AGIBot spawn {task_id} completed successfully")
                    else:
                        response_message = response.get('message', 'Unknown error')
                        if "reached maximum execution rounds" in response_message or "max_rounds_reached" in response_message:
                            print_agent(task_id, f"⚠️ AGIBot spawn {task_id} reached maximum execution rounds")
                        else:
                            print_agent(task_id, f"❌ AGIBot spawn {task_id} failed: {response_message}")
                    
                    # 🔧 修复：在完成后从active_threads中移除
                    if hasattr(self, 'active_threads') and task_id in self.active_threads:
                        del self.active_threads[task_id]
                    
                    # Reset agent ID after task completion
                    set_agent_id(None)
                        
                except Exception as e:
                    error_msg = str(e)
                    print_agent(task_id, f"❌ AGIBot spawn {task_id} error: {error_msg}")
                    
                    # Update status file with error
                    error_status = {
                        "agent_id": task_id,
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
                            f.flush()
                            import os
                            os.fsync(f.fileno()) if hasattr(f, 'fileno') else None
                    except Exception as e:
                        print_agent(task_id, f"⚠️ Warning: Could not update status file with error: {e}")
                    
                    # 🔧 修复：在出错后也从active_threads中移除
                    if hasattr(self, 'active_threads') and task_id in self.active_threads:
                        del self.active_threads[task_id]
                    
                    time.sleep(0.5)
                    
                    # Reset agent ID after exception handling
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
            
            # Base result information
            result = {
                "status": "success", 
                "message": f"AGIBot instance spawned successfully with agent ID: {agent_id}",
                "agent_id": agent_id,
                "output_directory": abs_output_dir,
                "working_directory": workspace_dir,
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
                print_agent(f"⏳ Waiting for AGIBot spawn {task_id} to complete...")
                
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
                
                print_agent(f"✅ Spawn {task_id} completed")
                
            else:
                result["note"] = f"✅ AGIBot {task_id} is running asynchronously in background. Task will execute independently and send messages when completed."
                result["success"] = True
                result["thread_id"] = thread.ident if thread else None
            
            return result
            
        except Exception as e:
            return {
                "status": "error",
                "message": f"Failed to spawn AGIBot instance: {str(e)}",
                "task_id": task_id if 'task_id' in locals() else "unknown"
            }

    def send_message_to_agent_or_manager(self, receiver_id: str, message_type: str, content: dict, priority: str = "normal") -> Dict[str, Any]:
        """
        Send message to specified agent or manager. Use 'manager' as receiver_id to send messages to the manager.
        
        Args:
            receiver_id: Receiver agent ID (use 'manager' for manager)
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
            
            # Get current agent ID as sender_id, if not, use manager
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
            
            # Use the correct sender's mailbox to send the message
            sender_mailbox = router.get_mailbox(sender_id)
            if not sender_mailbox:
                sender_mailbox = router.register_agent(sender_id)
                if not sender_mailbox:
                    return {"status": "error", "message": f"Failed to register sender agent: {sender_id}"}
            
            # Send message
            success = sender_mailbox.send_message(message)
            
            # 🔧 修复：发送消息后立即触发路由处理，确保消息被传递到目标邮箱
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
                    "receiver_id": receiver_id,
                    "message_type": message_type,
                    "priority": priority
                }
            else:
                return {
                    "status": "failed",
                    "message": "Failed to send message",
                    "sender_id": sender_id,
                    "receiver_id": receiver_id,
                    "message_type": message_type
                }
            
        except Exception as e:
            return {
                "status": "error",
                "message": f"Error sending message: {str(e)}",
                "receiver_id": receiver_id if 'receiver_id' in locals() else "unknown"
            }

    def get_agent_messages(self, include_read: bool = False) -> Dict[str, Any]:
        """
        Get messages from current agent's mailbox. Messages are automatically marked as read after retrieval.
        
        Args:
            include_read: Whether to include read messages
            
        Returns:
            Message list and statistics
        """
        try:
            from .message_system import get_message_router
            from .print_system import get_agent_id
            import glob
            
            # Get current agent ID
            current_agent_id = get_agent_id()
            agent_id = current_agent_id if current_agent_id else "manager"
            
            # 直接使用workspace_root
            try:
                router = get_message_router(self.workspace_root, cleanup_on_init=False)
                mailbox = router.get_mailbox(agent_id)
                
                if not mailbox:
                    # Collect available agents for error reporting
                    available_agents = router.get_all_agents()
                    return {
                        "status": "error", 
                        "message": f"Agent '{agent_id}' mailbox not found",
                        "agent_id": agent_id,
                        "workspace_root": self.workspace_root,
                        "available_agents": available_agents
                    }
                
                found_workspace = self.workspace_root
                
            except Exception as e:
                return {
                    "status": "error",
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
                "status": "error",
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
            from .print_system import get_agent_id
            import json
            
            # Get current agent ID
            current_agent_id = get_agent_id()
            agent_id = current_agent_id if current_agent_id else "manager"
            
            router = get_message_router(self.workspace_root, cleanup_on_init=False)
            mailbox = router.get_mailbox(agent_id)
            
            if not mailbox:
                return {
                    "status": "error", 
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
                "status": "error",
                "message": f"Error getting message summary: {str(e)}",
                "agent_id": agent_id if 'agent_id' in locals() else "unknown"
            }

    def diagnose_mailbox_paths(self) -> Dict[str, Any]:
        """
        Diagnose mailbox path configuration and find all available mailboxes.
        
        Returns:
            Diagnostic information about mailbox paths and available messages
        """
        try:
            from .message_system import get_message_router
            from .print_system import get_agent_id
            import glob
            import os
            
            # Get current agent ID
            current_agent_id = get_agent_id()
            agent_id = current_agent_id if current_agent_id else "manager"
            
            # Current workspace_root information
            current_workspace = self.workspace_root
            print_current(f"🔍 Current workspace_root: {current_workspace}")
            
            # Find all output_ directories
            output_dirs = []
            for pattern in ["output_*", "./output_*", "../output_*"]:
                found_dirs = glob.glob(pattern)
                for dir_path in found_dirs:
                    if os.path.isdir(dir_path):
                        abs_path = os.path.abspath(dir_path)
                        output_dirs.append(abs_path)
            
            # Remove duplicates
            output_dirs = list(set(output_dirs))
            print_current(f"🔍 Found output directories: {output_dirs}")
            
            # Check each directory for mailboxes
            mailbox_info = []
            for output_dir in output_dirs:
                mailbox_dir = os.path.join(output_dir, "mailboxes")
                if os.path.exists(mailbox_dir):
                    # Check for manager mailbox specifically
                    manager_mailbox = os.path.join(mailbox_dir, "manager")
                    if os.path.exists(manager_mailbox):
                        inbox_dir = os.path.join(manager_mailbox, "inbox")
                        if os.path.exists(inbox_dir):
                            # Count messages in inbox
                            message_files = glob.glob(os.path.join(inbox_dir, "*.json"))
                            mailbox_info.append({
                                "output_dir": output_dir,
                                "mailbox_dir": mailbox_dir,
                                "agent_id": "manager",
                                "inbox_dir": inbox_dir,
                                "message_count": len(message_files),
                                "message_files": [os.path.basename(f) for f in message_files[:5]]  # Show first 5
                            })
            
            # Try to access messages with current workspace_root
            current_router_result = None
            try:
                router = get_message_router(current_workspace, cleanup_on_init=False)
                mailbox = router.get_mailbox(agent_id)
                if mailbox:
                    current_router_result = {
                        "mailbox_found": True,
                        "inbox_dir": mailbox.inbox_dir,
                        "exists": os.path.exists(mailbox.inbox_dir),
                        "message_count": len(mailbox.get_all_messages()) if os.path.exists(mailbox.inbox_dir) else 0
                    }
                else:
                    current_router_result = {"mailbox_found": False}
            except Exception as e:
                current_router_result = {"error": str(e)}
            
            # Find the best workspace_root candidate
            best_candidate = None
            max_messages = 0
            for info in mailbox_info:
                if info["message_count"] > max_messages:
                    max_messages = info["message_count"]
                    best_candidate = info["output_dir"]
            
            return {
                "status": "success",
                "current_workspace_root": current_workspace,
                "current_router_result": current_router_result,
                "output_directories": output_dirs,
                "mailbox_info": mailbox_info,
                "best_candidate": best_candidate,
                "recommendation": f"Use set_workspace_root('{best_candidate}') to access messages" if best_candidate else "No mailboxes with messages found",
                "agent_id": agent_id
            }
            
        except Exception as e:
            return {
                "status": "error",
                "message": f"Error diagnosing mailbox paths: {str(e)}"
            }

    def set_workspace_root(self, new_workspace_root: str) -> Dict[str, Any]:
        """
        Set new workspace root directory for message access.
        
        Args:
            new_workspace_root: New workspace root directory path
            
        Returns:
            Result of workspace root change
        """
        try:
            # Validate path exists
            if not os.path.exists(new_workspace_root):
                return {
                    "status": "error",
                    "message": f"Path does not exist: {new_workspace_root}"
                }
            
            old_workspace = self.workspace_root
            self.workspace_root = os.path.abspath(new_workspace_root)
            
            # Test access to mailboxes with new path
            try:
                from .message_system import get_message_router
                from .print_system import get_agent_id
                
                current_agent_id = get_agent_id()
                agent_id = current_agent_id if current_agent_id else "manager"
                
                router = get_message_router(self.workspace_root, cleanup_on_init=False)
                mailbox = router.get_mailbox(agent_id)
                
                if mailbox and os.path.exists(mailbox.inbox_dir):
                    message_count = len(mailbox.get_all_messages())
                    return {
                        "status": "success",
                        "message": f"Workspace root changed successfully",
                        "old_workspace_root": old_workspace,
                        "new_workspace_root": self.workspace_root,
                        "mailbox_found": True,
                        "inbox_dir": mailbox.inbox_dir,
                        "message_count": message_count
                    }
                else:
                    return {
                        "status": "warning",
                        "message": f"Workspace root changed but no mailbox found for {agent_id}",
                        "old_workspace_root": old_workspace,
                        "new_workspace_root": self.workspace_root,
                        "mailbox_found": False
                    }
                    
            except Exception as e:
                # Rollback on error
                self.workspace_root = old_workspace
                return {
                    "status": "error",
                    "message": f"Error testing new workspace root: {str(e)}",
                    "workspace_root": self.workspace_root
                }
                
        except Exception as e:
            return {
                "status": "error",
                "message": f"Error setting workspace root: {str(e)}"
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
            from .print_system import get_agent_id
            
            # 🔧 修复：自动获取当前智能体ID，支持current_agent参数
            if agent_id == "current_agent" or not agent_id:
                current_agent_id = get_agent_id()
                actual_agent_id = current_agent_id if current_agent_id else "manager"
            else:
                actual_agent_id = agent_id
            
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
                        "status": "error",
                        "message": f"Failed to register agent: {actual_agent_id}",
                        "agent_id": actual_agent_id
                    }
            
            # Send message
            success = sender_mailbox.send_message(message)
            
            # 🔧 修复：发送状态更新后立即触发路由处理
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
                "status": "error",
                "message": f"Error sending status update to manager: {str(e)}",
                "agent_id": actual_agent_id if 'actual_agent_id' in locals() else agent_id
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
            
            # 🔧 修复：广播消息后立即触发路由处理
            if sent_count > 0:
                try:
                    processed_count = router.process_all_messages_once()
                except Exception as e:
                    print_current(f"⚠️ Error processing messages after broadcast: {e}")
            
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
            
            # Get message router
            router = get_message_router(self.workspace_root, cleanup_on_init=False)
            
            # Get all registered agents
            all_agents = router.get_all_agents()
            
            # Get active agents based on thread status
            active_agents_info = []
            
            # Check active threads
            if hasattr(self, 'active_threads'):
                for agent_id, thread in self.active_threads.items():
                    if thread.is_alive():
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
                        # 🔧 修复：检查agent状态 - terminated, completed, 或 registered
                        if agent_id in self.terminated_agents:
                            status = "terminated"
                            status_icon = "🔴"
                        elif agent_id in self.completed_agents:
                            status = "completed"
                            status_icon = "🟢"
                        else:
                            # 检查状态文件获取更准确的状态
                            status = self._get_agent_status_from_file(agent_id)
                            if status == "terminated":
                                status_icon = "🔴"
                                self.terminated_agents.add(agent_id)  # 更新缓存
                            elif status == "completed":
                                status_icon = "🟢"
                                self.completed_agents.add(agent_id)  # 更新缓存
                            elif status == "running":
                                status = "running"
                                status_icon = "🟢"
                            elif status == "unknown":
                                # 智能体已注册邮箱但状态不明，可能是刚启动或等待中
                                status = "idle"
                                status_icon = "🟡"
                            else:
                                # 其他未知状态
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
                    # 🔧 修复：使用更详细的状态图标和状态描述
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
                    
                    # 添加更详细的状态描述
                    status_desc = agent.get('status', 'unknown')
                    if status_desc == "completed":
                        # 尝试获取完成时间信息
                        try:
                            import datetime
                            completion_status = self._get_agent_completion_info(agent['agent_id'])
                            if completion_status:
                                completion_time = completion_status.get('completion_time')
                                if completion_time:
                                    # 计算完成时间距离现在多久
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
                "status": "error",
                "message": error_msg
            }

    def terminate_agibot(self, agent_id: str, reason: str = None) -> Dict[str, Any]:
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
            from .print_system import get_agent_id
            from datetime import datetime
            
            # Get current agent ID (usually manager)
            current_agent_id = get_agent_id()
            sender_id = current_agent_id if current_agent_id else "manager"
            
            if not agent_id or agent_id == "self" or agent_id == "current_agent":
                if current_agent_id:
                    agent_id = current_agent_id
                else:
                    return {
                        "status": "error",
                        "message": "Cannot determine current agent ID for self-termination",
                        "agent_id": agent_id
                    }
            
            # Validate agent ID format
            if not self._is_valid_agent_id_format(agent_id):
                return {
                    "status": "error",
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
                    "status": "error",
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
                        "status": "error",
                        "message": f"Failed to send terminate signal to agent {agent_id}",
                        "agent_id": agent_id
                    }
                    
            except Exception as e:
                return {
                    "status": "error",
                    "message": f"Error sending terminate signal to agent {agent_id}: {str(e)}",
                    "agent_id": agent_id
                }
        
        except Exception as e:
            return {
                "status": "error", 
                "message": f"Error terminating agent {agent_id}: {str(e)}",
                "agent_id": agent_id if 'agent_id' in locals() else "unknown"
            }

    def _normalize_agent_references(self, task_description: str, current_agent_id: str) -> str:
        """
        Normalize agent references in task description
        
        Args:
            task_description: Task description
            current_agent_id: Current agent ID
            
        Returns:
            Normalized task description
        """
        # This is a simplified version - in the full implementation, 
        # this would replace generic agent references with specific agent IDs
        return task_description
    
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
            router = get_message_router(self.workspace_root)
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
        从状态文件中获取agent的状态
        
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
                for task_id, thread in self.active_threads.items():
                    if thread.is_alive():
                        print_current(f"⏳ Waiting for thread {task_id} to complete...")
                
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