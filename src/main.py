#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright (c) 2025 AGI Agent Research Group.

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

"""
AGI Agent Main Program

A complete automated task processing workflow:
1. Receive user requirement input
2. Call multi-round task executor to execute tasks
3. Package working directory to tar.gz file
"""

# Application name macro definition
APP_NAME = "AGI Agent"

# Imports
from src.tools.print_system import print_current, print_system
from src.tools import Tools
import importlib
from src.tools.debug_system import install_debug_system, track_operation, finish_operation
from typing import Dict, Any, Optional, List
import os
import sys
import argparse
import json
import atexit
from datetime import datetime
from src.multi_round_executor import MultiRoundTaskExecutor
from src.config_loader import get_api_key, get_api_base, get_model, get_truncation_length
from src.routine_utils import append_routine_to_requirement
from src.tools.agent_context import get_current_agent_id, set_current_agent_id
# Configuration file to store last output directory
LAST_OUTPUT_CONFIG_FILE = ".agia_last_output.json"

# Global cleanup flag
_cleanup_executed = False

def global_cleanup():
    """Global cleanup function to ensure all resources are properly released"""
    global _cleanup_executed
    if _cleanup_executed:
        return
    _cleanup_executed = True
    
    try:
        #print_current("🔄 Starting global cleanup...")
        
        # Import here to avoid circular imports
        # Note: AgentManager class is not implemented, skipping cleanup
        
        # Cleanup MCP clients first (most important for subprocess cleanup)
        try:
            from tools.cli_mcp_wrapper import safe_cleanup_cli_mcp_wrapper
            safe_cleanup_cli_mcp_wrapper()
        except Exception as e:
            print_current(f"⚠️ CLI-MCP cleanup warning: {e}")
        
        try:
            from tools.fastmcp_wrapper import safe_cleanup_fastmcp_wrapper
            safe_cleanup_fastmcp_wrapper()
        except Exception as e:
            print_current(f"⚠️ FastMCP cleanup warning: {e}")
        
        try:
            from tools.mcp_client import safe_cleanup_mcp_client
            safe_cleanup_mcp_client()
        except Exception as e:
            print_current(f"⚠️ MCP client cleanup warning: {e}")
        
        # Stop message router if it exists
        try:
            from tools.message_system import get_message_router
            router = get_message_router()
            if router:
                router.stop()
        except Exception as e:
            print_current(f"⚠️ Message router cleanup warning: {e}")
        
        # Cleanup debug system
        try:
            from tools.debug_system import get_debug_system
            debug_sys = get_debug_system()
            debug_sys.cleanup()
        except Exception as e:
            print_current(f"⚠️ Debug system cleanup warning: {e}")
        
        # Small delay to allow cleanup to complete
        import time
        time.sleep(0.2)
        
        # Force garbage collection
        import gc
        gc.collect()
        
        #print_current("✅ Global cleanup completed")
        
    except Exception as e:
        print_current(f"⚠️ Error during final cleanup: {e}")

def signal_handler(signum, frame):
    """Handle interrupt signals"""
    print_current(f"\n⚠️ Signal received {signum}，正在清理...")
    global_cleanup()
    sys.exit(1)

def save_last_output_dir(out_dir: str, requirement: str = None):
    """
    Save the last output directory and requirement to configuration file
    
    Args:
        out_dir: Output directory path
        requirement: User requirement (optional)
    """
    try:
        # Check if current agent is manager - only manager should update the file
        current_agent_id = get_current_agent_id()
        
        # Only allow manager (None or "manager") to update the configuration file
        if current_agent_id is not None and current_agent_id != "manager":
            print_current(f"🔒 Agent {current_agent_id} skipping .agia_last_output.json update (only manager can update)")
            return
        
        config = {
            "last_output_dir": os.path.abspath(out_dir),
            "last_requirement": requirement,
            "timestamp": datetime.now().isoformat()
        }
        with open(LAST_OUTPUT_CONFIG_FILE, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print_current(f"⚠️ Failed to save last output directory: {e}")

def load_last_output_dir() -> Optional[str]:
    """
    Load the last output directory from configuration file
    
    Returns:
        Last output directory path, or None if not found
    """
    try:
        if not os.path.exists(LAST_OUTPUT_CONFIG_FILE):
            return None
        
        with open(LAST_OUTPUT_CONFIG_FILE, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        last_dir = config.get("last_output_dir")
        if last_dir and os.path.exists(last_dir):
            return last_dir
        else:
            print_current(f"⚠️ Last output directory does not exist: {last_dir}")
            return None
            
    except Exception as e:
        print_current(f"⚠️ Failed to load last output directory: {e}")
        return None

def load_last_requirement() -> Optional[str]:
    """
    Load the last user requirement from configuration file
    
    Returns:
        Last user requirement, or None if not found
    """
    try:
        if not os.path.exists(LAST_OUTPUT_CONFIG_FILE):
            return None
        
        with open(LAST_OUTPUT_CONFIG_FILE, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        return config.get("last_requirement")
            
    except Exception as e:
        print_current(f"⚠️ Failed to load last requirement: {e}")
        return None

class AGIAgentMain:
    def __init__(self, 
                 out_dir: str = "output", 
                 api_key: Optional[str] = None, 
                 model: Optional[str] = None, 
                 api_base: Optional[str] = None, 
                 debug_mode: bool = False, 
                 detailed_summary: bool = True, 
                 single_task_mode: bool = True,
                 interactive_mode: bool = False,
                 continue_mode: bool = False,
                 streaming: Optional[bool] = None,
                 MCP_config_file: Optional[str] = None,
                 prompts_folder: Optional[str] = None,
                 link_dir: Optional[str] = None,
                 user_id: Optional[str] = None,
                 routine_file: Optional[str] = None,
                 plan_mode: bool = False):

        """
        Initialize AGI Agent main program
        
        Args:
            out_dir: Output directory
            api_key: API key
            model: Model name
            api_base: API base URL
            debug_mode: Whether to enable DEBUG mode
            detailed_summary: Whether to enable detailed summary mode, retaining more technical information
            single_task_mode: Whether to enable single task mode, skip task decomposition and execute directly
            interactive_mode: Whether to enable interactive mode, ask user confirmation at each step
            continue_mode: Whether to continue from last output directory
            streaming: Whether to use streaming output
            MCP_config_file: Custom MCP configuration file path (optional, defaults to 'config/mcp_servers.json')
            prompts_folder: Custom prompts folder path (optional, defaults to 'prompts')
            link_dir: Link to external code directory (optional)
            routine_file: Routine file path to include in task planning (optional)
        """
        # Handle last requirement loading for continue mode
        self.last_requirement = None
        if continue_mode:
            last_req = load_last_requirement()
            if last_req:
                self.last_requirement = last_req
                print_system(f"🔄 Continue mode: Last requirement loaded: {last_req[:100]}{'...' if len(last_req) > 100 else ''}")
            else:
                print_system("ℹ️ Continue mode: No previous requirement found")
        
        # Load API key from config/config.txt if not provided
        if api_key is None:
            api_key = get_api_key()
            if api_key is None:
                raise ValueError("API key not found. Please provide api_key parameter or set it in config/config.txt")

        # Load model from config/config.txt if not provided
        if model is None:
            model = get_model()
            if model is None:
                raise ValueError("Model not found. Please provide model parameter or set it in config/config.txt")

        # Load API base from config/config.txt if not provided
        if api_base is None:
            api_base = get_api_base()
            if api_base is None:
                raise ValueError("API base URL not found. Please provide api_base parameter or set it in config/config.txt")

        # Store the validated parameters
        self.out_dir = out_dir
        self.api_key = api_key
        self.model = model
        self.api_base = api_base
        self.debug_mode = debug_mode
        self.detailed_summary = detailed_summary
        self.single_task_mode = single_task_mode
        self.interactive_mode = interactive_mode
        self.streaming = streaming
        self.MCP_config_file = MCP_config_file
        self.prompts_folder = prompts_folder
        self.link_dir = link_dir
        self.user_id = user_id
        self.routine_file = routine_file
        self.plan_mode = plan_mode
        
        # Ensure output directory exists
        os.makedirs(out_dir, exist_ok=True)
        
        # Set paths
        self.logs_dir = os.path.join(out_dir, "logs")  # Simplified: direct logs directory
        
        # Ensure logs directory exists  
        os.makedirs(self.logs_dir, exist_ok=True)
        
        # Set up workspace directory
        self.workspace_dir = os.path.join(self.out_dir, "workspace")
        os.makedirs(self.workspace_dir, exist_ok=True)
        
        # Create symbolic link if link_dir is provided
        if self.link_dir:
            self._create_workspace_link()
        
        ps_mod = importlib.import_module('src.tools.print_system')
        ps_mod.set_output_directory(self.out_dir)
        
        # Tools will be initialized in ToolExecutor to avoid duplicate initialization
        self.tools = None
            
    def _create_workspace_link(self):
        """
        Create symbolic link in workspace directory pointing to external code directory
        """
        if not self.link_dir:
            return
            
        # Convert to absolute path
        link_target = os.path.abspath(self.link_dir)
        
        # Check if target directory exists
        if not os.path.exists(link_target):
            print_current(f"⚠️ Warning: Link target directory does not exist: {link_target}")
            return
            
        if not os.path.isdir(link_target):
            print_current(f"⚠️ Warning: Link target is not a directory: {link_target}")
            return
            
        # Create link name (use the basename of the target directory)
        link_name = os.path.basename(link_target.rstrip(os.sep))
        if not link_name:
            link_name = "linked_code"
            
        link_path = os.path.join(self.workspace_dir, link_name)
        
        # Remove existing link or directory if exists
        if os.path.exists(link_path) or os.path.islink(link_path):
            try:
                if os.path.islink(link_path):
                    os.unlink(link_path)
                    print_current(f"🔗 Removed existing symbolic link: {link_path}")
                else:
                    print_current(f"⚠️ Warning: A file/directory already exists at link location: {link_path}")
                    return
            except Exception as e:
                print_current(f"⚠️ Warning: Failed to remove existing link: {e}")
                return
        
        # Create symbolic link
        try:
            os.symlink(link_target, link_path)
            print_current(f"🔗 Created symbolic link: {link_path} -> {link_target}")
            print_current(f"🎯 AGI Agent will now operate on external code directory: {link_target}")
        except Exception as e:
            print_current(f"⚠️ Warning: Failed to create symbolic link: {e}")
            print_current(f"    This may be due to insufficient permissions or unsupported file system")
        
    def get_user_requirement(self, requirement: Optional[str] = None) -> str:
        """
        Get user requirement
        
        Args:
            requirement: If provided, use directly, otherwise prompt user for input or use last requirement in continue mode
            
        Returns:
            User requirement string (merged with previous requirement if in continue mode)
        """
        # Check if we need to merge with previous requirement in continue mode
        if requirement and hasattr(self, 'last_requirement') and self.last_requirement:
            # Merge new requirement with previous requirement
            merged_requirement = f"{requirement}. This is a task that continues to be executed. The previous task requirements is: {self.last_requirement}"
            print_system(f"🔄 Continue mode: Merging new requirement with previous context")
            print_system(f"   New requirement: {requirement}")
            print_system(f"   Previous requirement: {self.last_requirement}")
            print_system(f"   Final merged requirement will be passed to AI")
            return merged_requirement
        
        if requirement:
            current_agent_id = get_current_agent_id()
            if current_agent_id:
                print_current(f"Received user requirement: {requirement}")
            else:
                print_current(f"Received user requirement: {requirement}")
            return requirement
        
        # Check if we should use last requirement from continue mode
        if hasattr(self, 'last_requirement') and self.last_requirement:
            print_system(f"🔄 Using last requirement from continue mode:")
            print_system(f"   {self.last_requirement}")
            print_system("   (Press Enter to use this requirement, or type a new one)")
            print_system("-" * 50)
            
            try:
                user_input = input("New requirement (or press Enter to continue): ").strip()
                if user_input:
                    # Merge new requirement with previous requirement
                    merged_requirement = f"{user_input}. This is a task that continues to be executed. The previous task requirements is: {self.last_requirement}"
                    print_system(f"Using merged requirement: {user_input}")
                    print_system(f"With context: Previous task - {self.last_requirement}")
                    return merged_requirement
                else:
                    print_system("Using previous requirement")
                    return self.last_requirement
            except (KeyboardInterrupt, EOFError):
                print_current("\nUsing previous requirement")
                return self.last_requirement
        
        print_system(f"=== {APP_NAME} Automated Task Processing System ===")
        print_system("Please describe your requirements, the system will automatically decompose tasks and execute:")
        print_system("(Press Enter to finish)")
        print_system("-" * 50)
        
        try:
            requirement = input("Enter your requirement: ").strip()
            if not requirement:
                print_current("No valid requirement entered")
                sys.exit(1)
        except EOFError:
            print_current("No valid requirement entered")
            sys.exit(1)
            
        except KeyboardInterrupt:
            print_current("\nUser cancelled input")
            sys.exit(0)
        
        return requirement
    
    def execute_plan_mode(self, user_requirement: str) -> bool:
        """
        Execute plan mode: interact with user to create plan.md document
        
        Args:
            user_requirement: User requirement
            
        Returns:
            Whether plan.md was successfully created
        """
        try:
            # Set working directory
            workspace_dir = os.path.join(self.out_dir, "workspace")
            
            # Create multi-round task executor with plan_mode=True
            executor = MultiRoundTaskExecutor(
                subtask_loops=50,  # Use reasonable number of loops for planning
                logs_dir=self.logs_dir,
                workspace_dir=workspace_dir,
                debug_mode=self.debug_mode,
                api_key=self.api_key,
                model=self.model,
                api_base=self.api_base,
                detailed_summary=self.detailed_summary,
                interactive_mode=False,  # Don't use interactive mode, use talk_to_user instead
                streaming=self.streaming,
                MCP_config_file=self.MCP_config_file,
                prompts_folder=self.prompts_folder,
                user_id=self.user_id,
                plan_mode=True  # Enable plan mode
            )
            
            # Construct single task for plan creation
            plan_task = {
                'Task ID': '1',
                'Task Name': 'Plan Creation',
                'Task Description': user_requirement,
                'Execution Status': '0',
                'Dependent Tasks': ''
            }
            
            print_system(f"🚀 Starting plan creation for: {user_requirement}")
            
            # Execute plan creation task
            task_result = executor.execute_single_task(plan_task, 0, 1, "")
            
            # Check if user interrupted execution
            if task_result.get("status") == "user_interrupted":
                print_current("🛑 Plan creation stopped by user")
                try:
                    executor.cleanup()
                except:
                    pass
                return False
            
            # Check if plan.md was created
            plan_md_path = os.path.join(workspace_dir, "plan.md")
            if os.path.exists(plan_md_path):
                print_current(f"✅ Plan document created successfully: {plan_md_path}")
                try:
                    executor.cleanup()
                except:
                    pass
                return True
            else:
                print_current("⚠️ Plan mode completed but plan.md was not found in workspace")
                try:
                    executor.cleanup()
                except:
                    pass
                return False
                
        except Exception as e:
            print_current(f"❌ Plan mode execution error: {e}")
            # Clean up resources if executor was created
            try:
                if 'executor' in locals():
                    executor.cleanup()
            except:
                pass
            return False
    
    def execute_single_task(self, user_requirement: str, loops: int = 3) -> bool:
        """
        Execute single task (skip task decomposition)
        
        Args:
            user_requirement: User requirement
            loops: Task execution rounds
            
        Returns:
            Whether execution was successful
        """
        
        try:
            # Set working directory
            workspace_dir = os.path.join(self.out_dir, "workspace")
            
            # 🔧 Check current agent_id and pass to executor
            current_agent_id = get_current_agent_id()
            if current_agent_id:
                print_current(f"🏷️ Using agent ID for task execution: {current_agent_id}")
            
            # Create multi-round task executor
            executor = MultiRoundTaskExecutor(
                subtask_loops=loops,
                logs_dir=self.logs_dir,
                workspace_dir=workspace_dir,
                debug_mode=self.debug_mode,
                api_key=self.api_key,
                model=self.model,
                api_base=self.api_base,
                detailed_summary=self.detailed_summary,
                interactive_mode=self.interactive_mode,
                streaming=self.streaming,
                MCP_config_file=self.MCP_config_file,
                prompts_folder=self.prompts_folder,
                user_id=self.user_id
            )
            
            # 🔧 Ensure executor uses correct agent_id
            if current_agent_id and hasattr(executor, 'executor') and hasattr(executor.executor, 'tools'):
                try:
                    # Set agent_id in executor's tools
                    if hasattr(executor.executor.tools, 'set_agent_context'):
                        executor.executor.tools.set_agent_context(current_agent_id)
                except Exception as e:
                    print_current(f"⚠️ Warning: Could not set agent context: {e}")
            
            # Append routine content to user requirement if provided
            enhanced_requirement = user_requirement
            if self.routine_file:
                enhanced_requirement = append_routine_to_requirement(user_requirement, self.routine_file)
            
            # Construct single task
            single_task = {
                'Task ID': '1',
                'Task Name': 'User Requirement Execution',
                'Task Description': enhanced_requirement,
                'Execution Status': '0',
                'Dependent Tasks': ''
            }
            
            print_system(f"🚀 Starting task execution ({loops} rounds max)")
            
            # Execute single task
            task_result = executor.execute_single_task(single_task, 0, 1, "")
            
            # Check if user interrupted execution
            if task_result.get("status") == "user_interrupted":
                print_current("🛑 Single task execution stopped by user")
                # Clean up resources before returning
                try:
                    executor.cleanup()
                except:
                    pass
                return False
            
            if task_result.get("status") == "completed":
                # Clean up resources before returning
                try:
                    executor.cleanup()
                except:
                    pass
                
                return True
            elif task_result.get("status") == "max_rounds_reached":
                print_current("⚠️ Task reached maximum execution rounds")
                
                # Clean up resources before returning
                try:
                    executor.cleanup()
                except:
                    pass
                
                return False
            else:
                # 🔧 Fix: distinguish between real failure and reaching max rounds  
                print_current("⚠️ Single task execution reached maximum rounds")
                # Clean up resources before returning
                try:
                    executor.cleanup()
                except:
                    pass
                return False
                
        except Exception as e:
            print_current(f"❌ Single task execution error: {e}")
            # Clean up resources if executor was created
            try:
                if 'executor' in locals():
                    executor.cleanup()
            except:
                pass
            return False
    
    def ask_user_confirmation(self, message: str, default_yes: bool = True) -> bool:
        """
        Ask user for confirmation in interactive mode
        
        Args:
            message: Confirmation message to display
            default_yes: Whether to default to 'yes' if user just presses Enter
            
        Returns:
            True if user confirms, False otherwise
        """
        if not self.interactive_mode:
            return True  # In non-interactive mode, always continue
        
        try:
            default_hint = "(Y/n)" if default_yes else "(y/N)"
            response = input(f"\n{message} {default_hint}: ").strip().lower()
            
            if not response:  # Empty response, use default
                return default_yes
            
            return response in ['y', 'yes', 'yes', 'confirm']
            
        except (KeyboardInterrupt, EOFError):
            print_current("\n❌ User cancelled operation")
            return False
    
    def run(self, user_requirement: Optional[str] = None, loops: int = 3) -> bool:
        """
        Run complete workflow
        
        Args:
            user_requirement: User requirement (optional)
            loops: Execution rounds for each task
            
        Returns:
            Whether successfully completed
        """
        track_operation("Main Program Execution")
        
        workspace_dir = os.path.join(self.out_dir, "workspace")
        
        # Step 1: Get user requirement
        track_operation("Get User Requirement")
        requirement = self.get_user_requirement(user_requirement)
        finish_operation("Get User Requirement")
        
        if not requirement:
            print_current("Invalid user requirement")
            return False
        
        # Save current output directory and requirement immediately after confirmation
        # This enables --continue functionality even if workflow is interrupted
        save_last_output_dir(self.out_dir, requirement)
        print_system(f"💾 Configuration saved for future --continue operations")
        
        # Plan mode: interact with user to create plan.md, then exit
        if self.plan_mode:
            track_operation("Plan Mode Execution")
            if not self.execute_plan_mode(requirement):
                print_current("Plan mode execution failed")
                finish_operation("Plan Mode Execution")
                finish_operation("Main Program Execution")
                return False
            finish_operation("Plan Mode Execution")
            finish_operation("Main Program Execution")
            print_current("🎉 Plan mode completed!")
            return True
        
        # Interactive mode confirmation before task execution
        if self.interactive_mode:
            task_description = f"Execute task: {requirement}"
            if not self.ask_user_confirmation(f"🤖 Ready to execute task:\n   {requirement}\n\nProceed with execution?"):
                print_current("Task execution cancelled by user")
                return False
        
        # Execute single task
        track_operation("Single Task Execution")
        if not self.execute_single_task(requirement, loops):
            print_current("⚠️ Single task execution reached maximum rounds")  # Fix: distinguish between failure and reaching max rounds
            finish_operation("Single Task Execution")
            finish_operation("Main Program Execution")
            return False
        finish_operation("Single Task Execution")
        
        # Task execution completed
        #print_current(f"📁 All output files saved at: {os.path.abspath(self.out_dir)}")
        # print_current(f"💻 User files saved at: {os.path.abspath(workspace_dir)}")
        
        print_current("🎉 Workflow completed!")
        finish_operation("Main Program Execution")
        return True


class AGIAgentClient:
    """
    AGI Agent Python Library Interface
    
    Provides OpenAI-like chat interface for programmatic access to AGI Agent functionality.
                Does not rely on config/config.txt file - all configuration is passed during initialization.
    
    Example usage:
        client = AGIAgentClient(
            api_key="your_api_key",
            model="claude-3-sonnet-20240229"
        )
        
        response = client.chat(
            messages=[{"role": "user", "content": "Build a calculator app"}],
            dir="my_project"
        )
        
        if response["success"]:
            print_current(f"Task completed! Output: {response['output_dir']}")
        else:
            print_current(f"Task failed: {response['message']}")
    """
    
    def __init__(self, 
                 api_key: Optional[str] = None,
                 model: Optional[str] = None,
                 api_base: Optional[str] = None,
                 debug_mode: bool = False,
                 detailed_summary: bool = True,
                 single_task_mode: bool = True,
                 interactive_mode: bool = False,
                 streaming: Optional[bool] = None,
                 MCP_config_file: Optional[str] = None,
                 prompts_folder: Optional[str] = None,
                 link_dir: Optional[str] = None,
                 user_id: Optional[str] = None,
                 agent_id: Optional[str] = None,
                 routine_file: Optional[str] = None):
        """
        Initialize AGI Agent Client
        
        Args:
            api_key: API key for LLM service (optional, will read from config/config.txt if not provided)
            model: Model name (optional, will read from config/config.txt if not provided)
            api_base: API base URL (optional)
            debug_mode: Whether to enable DEBUG mode
            detailed_summary: Whether to enable detailed summary mode
            single_task_mode: Whether to use single task mode (default: True)
            interactive_mode: Whether to enable interactive mode
            streaming: Whether to use streaming output (None to use config/config.txt)
            MCP_config_file: Custom MCP configuration file path (optional, defaults to 'config/mcp_servers.json')
            prompts_folder: Custom prompts folder path (optional, defaults to 'prompts')
            link_dir: Link to external code directory (optional)
            routine_file: Routine file path to include in task planning (optional)
        """
        # Import config loader functions
        from .config_loader import get_api_key, get_model, get_api_base
        
        # Use provided values or read from config
        self.api_key = api_key or get_api_key()
        self.model = model or get_model()
        
        if not self.api_key:
            raise ValueError("api_key is required. Either provide it as parameter or set it in config/config.txt")
        if not self.model:
            raise ValueError("model is required. Either provide it as parameter or set it in config/config.txt")
            
        # Use provided api_base or read from config
        self.api_base = api_base or get_api_base()
        self.debug_mode = debug_mode
        self.detailed_summary = detailed_summary
        self.single_task_mode = single_task_mode
        self.interactive_mode = interactive_mode
        self.streaming = streaming
        self.MCP_config_file = MCP_config_file
        self.prompts_folder = prompts_folder
        self.link_dir = link_dir
        self.routine_file = routine_file
        # Store agent id and publish to agent context
        self.agent_id = agent_id
        
        set_current_agent_id(agent_id)
        
    def chat(self, 
             messages: list,
             dir: Optional[str] = None,
             loops: int = 50,
             continue_mode: bool = False,
             **kwargs) -> dict:
        """
        Chat interface similar to OpenAI's chat completions
        
        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
                     Currently only supports single user message
                     Example: [{"role": "user", "content": "Build a calculator app"}]
            dir: Output directory for results (optional, will auto-generate if not provided)
             loops: Maximum execution rounds per task (default: 50, -1 for infinite loop)
            continue_mode: Whether to continue from previous execution (default: False)
            **kwargs: Additional parameters (reserved for future use)
            
        Returns:
            Dictionary containing:
            - success: bool - Whether execution was successful
            - message: str - Result message or error description
            - output_dir: str - Path to output directory
            - workspace_dir: str - Path to workspace directory
            - execution_time: float - Execution time in seconds
            - details: dict - Additional execution details
        """
        import time
        from datetime import datetime
        
        start_time = time.time()
        
        # Validate messages format
        if not isinstance(messages, list) or len(messages) == 0:
            return {
                "success": False,
                "message": "messages must be a non-empty list",
                "output_dir": None,
                "workspace_dir": None,
                "execution_time": 0,
                "details": {"error": "Invalid messages format"}
            }
        
        # Extract user requirement from messages
        user_message = None
        for msg in messages:
            if isinstance(msg, dict) and msg.get("role") == "user":
                user_message = msg.get("content", "").strip()
                break
        
        if not user_message:
            return {
                "success": False,
                "message": "No user message found in messages",
                "output_dir": None,
                "workspace_dir": None,
                "execution_time": 0,
                "details": {"error": "No valid user message"}
            }
        
        # Generate output directory if not provided
        if dir is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            dir = f"agia_output_{timestamp}"
        
        try:
            # 🔧 Check current thread's agent_id (set in agent_context when AGIAgentClient initializes)
            current_agent_id = get_current_agent_id()
            

            from src.tools.print_system import set_output_directory
            set_output_directory(dir)
            
            # Create AGI Agent main instance
            main_app = AGIAgentMain(
                out_dir=dir,
                api_key=self.api_key,
                model=self.model,
                api_base=self.api_base,
                debug_mode=self.debug_mode,
                detailed_summary=self.detailed_summary,
                single_task_mode=self.single_task_mode,
                interactive_mode=self.interactive_mode,
                continue_mode=continue_mode,
                streaming=self.streaming,
                MCP_config_file=self.MCP_config_file,
                prompts_folder=self.prompts_folder,
                link_dir=self.link_dir,
                routine_file=self.routine_file
            )
            
            # 🔧 If agent_id exists
            if current_agent_id:
                print_current(f"🏷️ AGIAgentClient using agent ID: {current_agent_id}")
            
            # Execute the task
            if current_agent_id:
                print_current(f"🚀 Executing task: {user_message}")
            else:
                print_current(f"🚀 Executing task: {user_message}")
            success = main_app.run(
                user_requirement=user_message,
                loops=loops
            )
            
            execution_time = time.time() - start_time
            workspace_dir = os.path.join(dir, "workspace")
            
            if success:
                return {
                    "success": True,
                    "message": "Task completed successfully",
                    "output_dir": os.path.abspath(dir),
                    "workspace_dir": os.path.abspath(workspace_dir) if os.path.exists(workspace_dir) else None,
                    "execution_time": execution_time,
                    "details": {
                        "requirement": user_message,
                        "loops": loops,
                        "mode": "single_task" if self.single_task_mode else "multi_task",
                        "model": self.model
                    }
                }
            else:
                return {
                    "success": False,
                    "message": "Task execution reached maximum execution rounds",
                    "output_dir": os.path.abspath(dir),
                    "workspace_dir": os.path.abspath(workspace_dir) if os.path.exists(workspace_dir) else None,
                    "execution_time": execution_time,
                    "details": {
                        "requirement": user_message,
                        "loops": loops,
                        "mode": "single_task" if self.single_task_mode else "multi_task",
                        "model": self.model,
                        "error": "Reached maximum execution rounds"
                    }
                }
                
        except Exception as e:
            execution_time = time.time() - start_time
            return {
                "success": False,
                "message": f"Execution error: {str(e)}",
                "output_dir": os.path.abspath(dir) if os.path.exists(dir) else None,
                "workspace_dir": None,
                "execution_time": execution_time,
                "details": {
                    "requirement": user_message,
                    "error": str(e),
                    "error_type": type(e).__name__
                }
            }
    
    def get_models(self) -> list:
        """
        Get list of supported models (placeholder for future implementation)
        
        Returns:
            List of supported model names
        """
        return [
            "gpt-4",
            "gpt-4-turbo",
            "gpt-3.5-turbo",
            "claude-3-sonnet-20240229",
            "claude-3-haiku-20240307",
            "claude-3-opus-20240229",
            "claude-3-5-sonnet-20241022"
        ]
    
    def get_config(self) -> dict:
        """
        Get current client configuration
        
        Returns:
            Dictionary containing current configuration
        """
        return {
            "api_key": f"{self.api_key[:10]}...{self.api_key[-5:]}" if len(self.api_key) > 15 else self.api_key,
            "model": self.model,
            "api_base": self.api_base,
            "debug_mode": self.debug_mode,
            "detailed_summary": self.detailed_summary,
            "single_task_mode": self.single_task_mode,
            "interactive_mode": self.interactive_mode
        }


# Convenience function for quick usage
def create_client(api_key: Optional[str] = None, model: Optional[str] = None, user_id: Optional[str] = None, **kwargs) -> AGIAgentClient:
    """
    Convenience function to create AGI Agent client
    
    Args:
        api_key: API key for LLM service (optional, will read from config/config.txt if not provided)
        model: Model name (optional, will read from config/config.txt if not provided)
        user_id: User ID for MCP knowledge base tools (optional)
        **kwargs: Additional configuration parameters
        
    Returns:
        AGIAgentClient instance
    """
    return AGIAgentClient(api_key=api_key, model=model, user_id=user_id, **kwargs)


def print_ascii_banner():
    """Print ASCII art banner for AGI Agent"""
    # This function is imported from agia.py
    pass


def main():
    """
    Main function - handle command line parameters
    """
    from config_loader import load_config
    config = load_config()
    enable_debug_system = config.get('enable_debug_system', 'False').lower() == 'true'
    if enable_debug_system:
        install_debug_system(
            enable_stack_trace=True,
            enable_memory_monitor=True, 
            enable_execution_tracker=True,
            show_activation_message=False  # Always silent by default
        )
    
    # Register cleanup handlers
    atexit.register(global_cleanup)
    # Note: signal handlers are now managed by debug system
    
    # Print ASCII banner at startup
    print_ascii_banner()
    
    parser = argparse.ArgumentParser(
        description=f"{APP_NAME} Automated Task Processing System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Usage Examples:
  # Single task mode (default) - directly execute user requirement without task decomposition
  python main.py "Fix game sound effect playback issue"
  python main.py --requirement "Fix game sound effect playback issue"
  python main.py -r "Fix game sound effect playback issue"
  python main.py --singletask "Optimize code performance"
  
  # Continue from last output directory
  python main.py --continue "Continue working on the previous task"
  python main.py -c --requirement "Add new features to existing project"
  
  # Interactive mode - prompt user for requirement input
  python main.py
  
  # Specify output directory and execution rounds
  python main.py --dir my_project --loops 5 "Requirement description"
  
  # Infinite loop execution (until task completion or manual interruption)
  python main.py --loops -1 "Requirement description"
  
  # Use custom model configuration
  python main.py --api-key YOUR_KEY --model gpt-4 --base-url https://api.openai.com/v1 "My requirement"
        """
    )
    
    parser.add_argument(
        "requirement_positional",
        nargs="?",
        help="User requirement description (positional argument)"
    )
    
    parser.add_argument(
        "--requirement", "-r",
        help="User requirement description. If not provided, will enter interactive mode to prompt user input"
    )
    
    parser.add_argument(
        "--dir", "-d",
        default=f"output_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        help="Output directory for storing logs (default: output_timestamp)"
    )
    
    parser.add_argument(
        "--loops", "-l",
        type=int,
        default=50,
        help="Execution rounds for each subtask (default: 50, -1 for infinite loop)"
    )
    
    parser.add_argument(
        "--api-key",
        default=None,
        help="API key"
    )
    
    parser.add_argument(
        "--model", "-m",
        default=None,
        help="LLM model name (will load from config/config.txt if not specified)"
    )
    
    parser.add_argument(
        "--api-base",
        default=None,
        help="API base URL"
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="Enable DEBUG mode, record detailed LLM call information to llmcall.csv file"
    )
    
    parser.add_argument(
        "--detailed-summary",
        action="store_true",
        default=True,
        help="Enable detailed summary mode, retain more technical information and execution context (enabled by default)"
    )
    
    parser.add_argument(
        "--simple-summary",
        action="store_true",
        default=False,
        help="Use simplified summary mode, only retain basic information"
    )
    
    parser.add_argument(
        "--singletask",
        action="store_true",
        default=True,
        help="Enable single task mode, skip task decomposition and directly execute user requirement (default mode)"
    )
    
    parser.add_argument(
        "--version", "-v",
        action="version",
        version=f"{APP_NAME} v0.1.0"
    )
    
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        default=False,
        help="Enable interactive mode, ask user confirmation at each step"
    )
    
    parser.add_argument(
        "--continue", "-c",
        action="store_true",
        default=False,
        dest="continue_mode",
        help="Continue from last output directory (ignores --dir if last directory exists)"
    )
    
    args = parser.parse_args()
    
    # Handle requirement argument priority: positional argument takes precedence over --requirement/-r
    if args.requirement_positional:
        args.requirement = args.requirement_positional
    
    # Check for conflicting parameters: --continue and --dir
    user_specified_out_dir = '--dir' in sys.argv or '-d' in sys.argv
    if args.continue_mode and user_specified_out_dir:
        # User specified both --continue/-c and --dir
        print_current("⚠️  Warning: Both --continue/-c and --dir parameters were specified.")
        print_current("    The --continue/-c parameter takes priority and --dir will be ignored.")
        print_current("    If you want to use a specific output directory, don't use --continue/-c.")
    
    # Check if no parameters provided, if so use default parameters
    if len(sys.argv) == 1:  # Only script name, no other parameters
        print_current("🔧 No parameters provided, using default configuration...")
        # Set default parameters
        # args.requirement = "build a tetris game"
        # args.requirement = "make up some electronic sound in the sounds directory and remove the chinese characters in the GUI"
        #args.out_dir = "output_test"
        args.loops = 50
        #args.model = "gpt-4.1"
        #args.base_url = "https://api.openai-proxy.org/v1"
        args.api_key = None
        args.model = None  # Let it load from config/config.txt
        args.api_base = None
        print_current(f"📁 Output directory: {args.dir}")
        print_current(f"🔄 Execution rounds: {args.loops}")
        print_current(f"🤖 Model: Will load from config/config.txt")

    
    # Get API key
    api_key = args.api_key
    
    # Determine summary mode
    detailed_summary = not args.simple_summary if hasattr(args, 'simple_summary') else args.detailed_summary
    
    # Determine task mode (always single task mode now)
    single_task_mode = True
    
    # Create and run main program
    try:
        main_app = AGIAgentMain(
            out_dir=args.dir,
            api_key=api_key,
            model=args.model,
            api_base=args.api_base,
            debug_mode=args.debug,
            detailed_summary=detailed_summary,
            single_task_mode=single_task_mode,
            interactive_mode=args.interactive,
            continue_mode=args.continue_mode
        )
        
        success = main_app.run(
            user_requirement=args.requirement,
            loops=args.loops
        )
        
        sys.exit(0 if success else 1)
        
    except KeyboardInterrupt:
        print_current("\nUser interrupted program execution")
        sys.exit(1)
    except Exception as e:
        print_current(f"Program execution error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 