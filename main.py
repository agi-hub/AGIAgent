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

"""
AGI Bot Main Program

A complete automated task processing workflow:
1. Receive user requirement input
2. Call task decomposer to create todo.csv
3. Call multi-round task executor to execute tasks
4. Package working directory to tar.gz file
"""

# Application name macro definition
APP_NAME = "AGI Bot"

from tools.print_system import set_agent_id, print_manager, print_system, print_current
from tools import Tools
from tools.debug_system import install_debug_system, track_operation, finish_operation

def is_jupyter_environment():
    """
    Check if the code is running in a Jupyter environment
    
    Returns:
        bool: True if running in Jupyter, False otherwise
    """
    try:
        # Check if get_ipython function exists (available in IPython/Jupyter)
        get_ipython
        return True
    except NameError:
        pass
    
    # Check for Jupyter-specific environment variables
    try:
        import os
        jupyter_env_vars = [
            'JUPYTER_RUNTIME_DIR', 
            'JUPYTER_CONFIG_DIR',
            'JPY_SESSION_NAME',
            'KERNEL_ID'
        ]
        if any(var in os.environ for var in jupyter_env_vars):
            return True
    except:
        pass
    # Check for Google Colab environment
    try:
        import os
        # Check environment variables
        if any(var in os.environ for var in ['COLAB_GPU', 'COLAB_TPU_ADDR']):
            return True
    except:
        pass
    return False

def is_library_mode():
    """
    Check if the code is being used as a library (not run directly)
    
    Returns:
        bool: True if used as library, False if run directly
    """
    import inspect
    
    # Check if we're being called from main() function
    # If main() is not in the call stack, it's likely library usage
    frame = inspect.currentframe()
    try:
        while frame:
            if frame.f_code.co_name == 'main' and frame.f_globals.get('__name__') == '__main__':
                return False  # main() is being called directly
            frame = frame.f_back
        return True  # main() not found in stack, likely library usage
    finally:
        del frame

def should_show_banner():
    """
    Determine whether to show the ASCII banner
    
    Returns:
        bool: True if banner should be shown, False otherwise
    """
    # Don't show banner in Jupyter environments
    if is_jupyter_environment():
        return False
    
    # Check if we're running from command line (main script execution)
    import inspect
    frame = inspect.currentframe()
    try:
        # Look for main() function in call stack with __name__ == '__main__'
        while frame:
            if (frame.f_code.co_name == 'main' and 
                frame.f_globals.get('__name__') == '__main__'):
                return True  # Direct command line execution
            frame = frame.f_back
    finally:
        del frame
    
    # If main() not found in call stack, it's likely library usage
    return False

def print_ascii_banner():
    """Print ASCII art banner for AGI Bot (only if appropriate environment)"""
    if not should_show_banner():
        return
    
    # ANSI color codes - Bright blue
    BRIGHT_BLUE = '\033[94m'
    RESET = '\033[0m'
    
    banner = f"""{BRIGHT_BLUE}
                                                             
       █████╗  ██████╗ ██╗    ██████╗  ██████╗ ████████╗     
      ██╔══██╗██╔════╝ ██║    ██╔══██╗██╔═══██╗╚══██╔══╝     
      ███████║██║  ███╗██║    ██████╔╝██║   ██║   ██║        
      ██╔══██║██║   ██║██║    ██╔══██╗██║   ██║   ██║        
      ██║  ██║╚██████╔╝██║    ██████╔╝╚██████╔╝   ██║        
      ╚═╝  ╚═╝ ╚═════╝ ╚═╝    ╚═════╝  ╚═════╝    ╚═╝        
                                                             
         🚀 Autonomous Task Execution System                 
         🧠 LLM-Powered Cognitive Architecture              
                                                             {RESET}
    """
    print(banner)

import os
import sys
import argparse
import json
import atexit
import signal
from datetime import datetime
from task_decomposer import TaskDecomposer
from multi_round_executor import MultiRoundTaskExecutor
from typing import Dict, Any
from config_loader import get_api_key, get_api_base, get_model, get_truncation_length, get_summary_report
from tools.print_system import print_system, print_manager, print_current

# Configuration file to store last output directory
LAST_OUTPUT_CONFIG_FILE = ".agibot_last_output.json"

# Global cleanup flag
_cleanup_executed = False

def global_cleanup():
    """Global cleanup function to ensure all resources are properly released"""
    global _cleanup_executed
    if _cleanup_executed:
        return
    _cleanup_executed = True
    
    try:
        
        # Import here to avoid circular imports
        # Note: AgentManager class is not implemented, skipping cleanup
        
        # Stop message router if it exists
        try:
            from tools.message_system import get_message_router
            router = get_message_router()
            if router:
                router.stop()
        except:
            pass
        
        # Cleanup debug system
        try:
            from tools.debug_system import get_debug_system
            debug_sys = get_debug_system()
            debug_sys.cleanup()
        except:
            pass
        
        # Force garbage collection
        import gc
        gc.collect()
        
        
    except Exception as e:
        print_current(f"⚠️ Error during final cleanup: {e}")

def signal_handler(signum, frame):
    """Handle interrupt signals"""
    print_current(f"\n⚠️ 收到信号 {signum}，正在清理...")
    global_cleanup()
    sys.exit(1)

def save_last_output_dir(out_dir: str):
    """
    Save the last output directory to configuration file
    
    Args:
        out_dir: Output directory path
    """
    try:
        config = {
            "last_output_dir": os.path.abspath(out_dir),
            "timestamp": datetime.now().isoformat()
        }
        with open(LAST_OUTPUT_CONFIG_FILE, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print_current(f"⚠️ Failed to save last output directory: {e}")

def load_last_output_dir() -> str:
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

class AGIBotMain:
    def __init__(self, 
                 out_dir: str = "output", 
                 api_key: str = None, 
                 model: str = None, 
                 api_base: str = None, 
                 debug_mode: bool = False, 
                 detailed_summary: bool = True, 
                 single_task_mode: bool = True,
                 interactive_mode: bool = False,
                 continue_mode: bool = False,
                 streaming: bool = None):

        """
        Initialize AGI Bot main program
        
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
        """
        # Handle continue mode - load last output directory if requested
        if continue_mode:
            last_dir = load_last_output_dir()
            # If an explicit out_dir is provided and the directory exists, prioritize using the provided directory
            # This is mainly to support directory selection in GUI mode
            # Check if absolute or relative path exists
            out_dir_abs = os.path.abspath(out_dir)
            if out_dir != "output" and (os.path.exists(out_dir) or os.path.exists(out_dir_abs)):
                print_system(f"🔄 Continue mode: Using specified directory: {out_dir}")
            elif last_dir:
                out_dir = last_dir
                print_system(f"🔄 Continue mode: Using last output directory: {out_dir}")
            else:
                print_system("⚠️ Continue mode requested but no valid last output directory found")
                print_system("    Creating new output directory instead")
        
        # Load API key from config.txt if not provided
        if api_key is None:
            api_key = get_api_key()
            if api_key is None:
                raise ValueError("API key not found. Please provide api_key parameter or set it in config.txt")
        
        # Load model from config.txt if not provided
        if model is None:
            model = get_model()
            if model is None:
                raise ValueError("Model not found. Please provide model parameter or set it in config.txt")
        
        # Load API base from config.txt if not provided
        if api_base is None:
            api_base = get_api_base()
            if api_base is None:
                raise ValueError("API base URL not found. Please provide api_base parameter or set it in config.txt")
        

        # print_current(f"🚀 {APP_NAME} Automated Task Processing System")  # Commented out to reduce terminal noise

        # print_current(f"🤖 LLM Configuration:")  # Commented out to reduce terminal noise
        # print_current(f"   Model: {model}")  # Commented out to reduce terminal noise
        # print_current(f"   API Base: {api_base}")  # Commented out to reduce terminal noise
        # if api_key:
        #     print_current(f"   API Key: {api_key[:20]}...{api_key[-10:] if len(api_key) > 30 else api_key}")  # Commented out to reduce terminal noise
        # else:
        #     print_current(f"   API Key: Not set")  # Commented out to reduce terminal noise
        # print_current(f"📁 Output Directory: {out_dir}")  # Commented out to reduce terminal noise
        # if debug_mode:
        #     print_current(f"🐛 DEBUG Mode: Enabled")  # Commented out to reduce terminal noise
        # if detailed_summary:
        #     print_current(f"📋 Detailed Summary: Enabled (retaining more technical information)")  # Commented out to reduce terminal noise
        # else:
        #     print_current(f"📋 Detailed Summary: Disabled (using simplified summary)")  # Commented out to reduce terminal noise
        # if interactive_mode:
        #     print_current(f"🤝 Interactive Mode: Enabled (ask user confirmation at each step)")  # Commented out to reduce terminal noise
        # else:
        #     print_current(f"🤖 Automatic Mode: Enabled (no user confirmation required)")  # Commented out to reduce terminal noise
        # if continue_mode:
        #     print_current(f"🔄 Continue Mode: Enabled (using last output directory)")  # Commented out to reduce terminal noise

        
        self.out_dir = out_dir
        self.api_key = api_key
        self.model = model
        self.api_base = api_base
        self.debug_mode = debug_mode
        self.detailed_summary = detailed_summary
        self.single_task_mode = single_task_mode
        self.interactive_mode = interactive_mode
        self.streaming = streaming
        
        # Ensure output directory exists
        os.makedirs(out_dir, exist_ok=True)
        
        # Set paths
        self.todo_csv_path = os.path.join(out_dir, "todo.csv")
        self.logs_dir = os.path.join(out_dir, "logs")  # Simplified: direct logs directory
        
        # Ensure logs directory exists  
        os.makedirs(self.logs_dir, exist_ok=True)
        
        # Set up workspace directory
        self.workspace_dir = os.path.join(self.out_dir, "workspace")
        os.makedirs(self.workspace_dir, exist_ok=True)
        
        # Set agent ID for main AGIBot
        set_agent_id("manager")
        
        # Initialize tools with workspace root
        self.tools = Tools(self.workspace_dir)
        
        # Only create task decomposer in multi-task mode
        if not single_task_mode:
            self.task_decomposer = TaskDecomposer(api_key=api_key, model=model, api_base=api_base)
        else:
            self.task_decomposer = None
        
    def get_user_requirement(self, requirement: str = None) -> str:
        """
        Get user requirement
        
        Args:
            requirement: If provided, use directly, otherwise prompt user for input
            
        Returns:
            User requirement string
        """
        if requirement:
            # 🔧 修复：根据当前agent ID打印信息，而不是总是使用manager
            from tools.print_system import get_agent_id
            current_agent_id = get_agent_id()
            if current_agent_id:
                print_current(f"Received user requirement: {requirement}")
            else:
                print_manager(f"Received user requirement: {requirement}")
            return requirement
        
        print_system(f"=== {APP_NAME} Automated Task Processing System ===")
        print_system("Please describe your requirements, the system will automatically decompose tasks and execute:")
        print_system("(Supports multi-line input, enter two empty lines to finish)")
        print_system("-" * 50)
        
        lines = []
        empty_line_count = 0
        
        while True:
            try:
                line = input()
                if line.strip() == "":
                    empty_line_count += 1
                    if empty_line_count >= 2:
                        break
                    lines.append("")
                else:
                    empty_line_count = 0
                    lines.append(line)
            except KeyboardInterrupt:
                print_current("\nUser cancelled input")
                sys.exit(0)
            except EOFError:
                break
        
        requirement = "\n".join(lines).strip()
        
        if not requirement:
            print_current("❌ No valid requirement entered")
            sys.exit(1)
            
        return requirement
    
    def decompose_task(self, user_requirement: str) -> bool:
        """
        Execute task decomposition
        
        Args:
            user_requirement: User requirement
            
        Returns:
            Whether todo.csv was successfully created
        """
        print_manager("🔧 Starting task decomposition...")
        
        try:
            # Set working directory
            workspace_dir = os.path.join(self.out_dir, "workspace")
            
            # Execute task decomposition, pass working directory information
            result = self.task_decomposer.decompose_task(
                user_requirement, 
                self.todo_csv_path,
                workspace_dir=workspace_dir
            )
            print_current(f"Task decomposition result: {result}")
            
            # Check if CSV file was successfully created
            # First check expected location
            if not os.path.exists(self.todo_csv_path):
                print_current("❌ Task decomposition failed: Failed to create todo.csv file")
                return False
            
            return True
            
        except Exception as e:
            print_current(f"❌ Task decomposition error: {e}")
            return False
    
    def execute_tasks(self, loops: int = 3) -> bool:
        """
        Execute tasks
        
        Args:
            loops: Execution rounds for each task
            
        Returns:
            Whether execution was successful
        """
        
        try:
            # Set working directory
            workspace_dir = os.path.join(self.out_dir, "workspace")
            
            # Create multi-round task executor, pass all configuration parameters
            executor = MultiRoundTaskExecutor(
                subtask_loops=loops,
                logs_dir=self.logs_dir,
                workspace_dir=workspace_dir,
                debug_mode=self.debug_mode,
                api_key=self.api_key,
                model=self.model,
                api_base=self.api_base,
                detailed_summary=self.detailed_summary,
                interactive_mode=self.interactive_mode
            )
            
            # Execute all tasks
            report = executor.execute_all_tasks(self.todo_csv_path)
            
            if "error" in report:
                print_current(f"❌ Task execution failed: {report['error']}")
                # Clean up resources before returning
                try:
                    executor.cleanup()
                except:
                    pass
                return False
            
            # Clean up resources before returning
            try:
                executor.cleanup()
            except:
                pass
            return True
            
        except Exception as e:
            print_current(f"❌ Task execution error: {e}")
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
            
            # 🔧 检查当前agent_id并传递给执行器
            from tools.print_system import get_agent_id, set_agent_id
            current_agent_id = get_agent_id()
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
                streaming=self.streaming
            )
            
            # 🔧 确保executor使用正确的agent_id
            if current_agent_id and hasattr(executor, 'executor') and hasattr(executor.executor, 'tools'):
                try:
                    # 在executor的工具中设置agent_id
                    if hasattr(executor.executor.tools, 'set_agent_context'):
                        executor.executor.tools.set_agent_context(current_agent_id)
                except Exception as e:
                    print_current(f"⚠️ Warning: Could not set agent context: {e}")
            
            # Construct single task
            single_task = {
                'Task ID': '1',
                'Task Name': 'User Requirement Execution',
                'Task Description': user_requirement,
                'Execution Status': '0',
                'Dependent Tasks': ''
            }
            
            print_manager(f"🚀 Starting task execution ({loops} rounds max)")
            
            # Execute single task
            task_result = executor.execute_single_task(single_task, 0, 1, "")
            
            # Check if user interrupted execution
            if task_result.get("status") == "user_interrupted":
                print_current("🛑 Single task execution stopped by user")
                print_current("📋 Skipping report generation due to user interruption")
                # Clean up resources before returning
                try:
                    executor.cleanup()
                except:
                    pass
                return False
            
            if task_result.get("status") == "completed":
                
                # Save simple execution report
                execution_report = {
                    "start_time": datetime.now().isoformat(),
                    "end_time": datetime.now().isoformat(),
                    "total_tasks": 1,
                    "completed_tasks": [task_result],
                    "failed_tasks": [],
                    "execution_summary": f"Single task mode execution completed\nTask: {user_requirement}",
                    "workspace_dir": workspace_dir,
                    "mode": "single_task"
                }
                
                # Save execution report
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                # 🔧 添加agent ID到报告文件名
                from tools.print_system import get_agent_id
                current_agent_id = get_agent_id()
                if current_agent_id:
                    report_file = os.path.join(self.logs_dir, f"single_task_report_{current_agent_id}_{timestamp}.json")
                else:
                    report_file = os.path.join(self.logs_dir, f"single_task_report_{timestamp}.json")
                
                try:
                    import json
                    with open(report_file, 'w', encoding='utf-8') as f:
                        json.dump(execution_report, f, ensure_ascii=False, indent=2)
                except Exception as e:
                    print_current(f"⚠️ Report save failed: {e}")
                
                # Generate human-readable Markdown report
                try:
                    self.generate_single_task_markdown_report(execution_report, timestamp)
                except Exception as e:
                    print_current(f"⚠️ Markdown report generation failed: {e}")
                
                # Generate detailed summary report (if enabled in config)
                if get_summary_report():
                    try:
                        self.generate_single_task_summary_report(execution_report, timestamp)
                    except Exception as e:
                        print_current(f"⚠️ Detailed summary report generation failed: {e}")
                
                # Clean up resources before returning
                try:
                    executor.cleanup()
                except:
                    pass
                
                return True
            elif task_result.get("status") == "max_rounds_reached":
                # For max rounds reached, still generate reports but return False to indicate partial success
                print_current("⚠️ Task reached maximum execution rounds")
                
                # Save execution report for max rounds case
                execution_report = {
                    "start_time": datetime.now().isoformat(),
                    "end_time": datetime.now().isoformat(),
                    "total_tasks": 1,
                    "completed_tasks": [],
                    "max_rounds_reached_tasks": [task_result],  # 🔧 修复：不再标记为failed_tasks
                    "execution_summary": f"Single task mode execution reached max rounds\nTask: {user_requirement}",
                    "workspace_dir": workspace_dir,
                    "mode": "single_task"
                }
                
                # Save execution report
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                # 🔧 添加agent ID到报告文件名
                from tools.print_system import get_agent_id
                current_agent_id = get_agent_id()
                if current_agent_id:
                    report_file = os.path.join(self.logs_dir, f"single_task_report_{current_agent_id}_{timestamp}.json")
                else:
                    report_file = os.path.join(self.logs_dir, f"single_task_report_{timestamp}.json")
                
                try:
                    import json
                    with open(report_file, 'w', encoding='utf-8') as f:
                        json.dump(execution_report, f, ensure_ascii=False, indent=2)
                    print_current(f"📋 Execution report saved to: {report_file}")
                except Exception as e:
                    print_current(f"⚠️ Report save failed: {e}")
                
                # Clean up resources before returning
                try:
                    executor.cleanup()
                except:
                    pass
                
                return False
            else:
                # 🔧 修复：区分真正的失败和达到最大轮数  
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
    
    def generate_single_task_markdown_report(self, report: Dict[str, Any], timestamp: str):
        """
        Generate human-readable Markdown report for single task mode
        
        Args:
            report: Execution report
            timestamp: Timestamp
        """
        try:
            # 🔧 添加agent ID到markdown报告文件名
            from tools.print_system import get_agent_id
            current_agent_id = get_agent_id()
            if current_agent_id:
                markdown_file = os.path.join(self.logs_dir, f"single_task_report_{current_agent_id}_{timestamp}.md")
            else:
                markdown_file = os.path.join(self.logs_dir, f"single_task_report_{timestamp}.md")
            
            start_time = datetime.fromisoformat(report["start_time"])
            end_time = datetime.fromisoformat(report["end_time"])
            duration = end_time - start_time
            
            task = report["completed_tasks"][0] if report["completed_tasks"] else {}
            task_name = task.get("task_name", "User Requirement Execution")
            history = task.get("history", [])
            summary = task.get("summary", "No summary")
            
            # Build Markdown content
            markdown_content = f"""# {APP_NAME} Single Task Execution Report

## 📊 Execution Overview

- **Execution Mode**: Single Task Mode (direct execution of user requirement)
- **Execution Time**: {start_time.strftime('%Y-%m-%d %H:%M:%S')}
- **Total Duration**: {duration}
- **Task Status**: ✅ Execution Completed
- **Working Directory**: {report.get("workspace_dir", "Not specified")}

---

## 📋 Task Details

### {task_name}

**Task Summary**:
{summary}

---

"""
            
            # Add execution history details
            if history:
                markdown_content += "## 🔄 Execution History\n\n"
                
                for round_info in history:
                    if isinstance(round_info, dict) and "round" in round_info:
                        round_num = round_info["round"]
                        prompt = round_info.get("prompt", "")
                        result = round_info.get("result", "")
                        task_completed = round_info.get("task_completed", False)
                        timestamp_str = round_info.get("timestamp", "")
                        
                        markdown_content += f"### Round {round_num} Execution\n\n"
                        
                        if timestamp_str:
                            try:
                                exec_time = datetime.fromisoformat(timestamp_str)
                                markdown_content += f"**Execution Time**: {exec_time.strftime('%H:%M:%S')}\n\n"
                            except:
                                pass
                        
                        # Add user prompt (simplified display)
                        if prompt:
                            # Extract task description part
                            if "Task Description:" in prompt:
                                desc_start = prompt.find("Task Description:") + len("Task Description:")
                                desc_end = prompt.find("\n\n", desc_start)
                                if desc_end != -1:
                                    task_desc = prompt[desc_start:desc_end].strip()
                                    markdown_content += f"**Task Requirement**: {task_desc}\n\n"
                        
                        # Parse result content
                        if result:
                            # Separate LLM response and tool execution results
                            if "--- Tool Execution Results ---" in result:
                                parts = result.split("--- Tool Execution Results ---")
                                llm_response = parts[0].strip()
                                tool_results = parts[1].strip() if len(parts) > 1 else ""
                                
                                # LLM response
                                if llm_response:
                                    markdown_content += f"**LLM Analysis and Planning**:\n```\n{llm_response}\n```\n\n"
                                
                                # Tool execution results
                                if tool_results:
                                    markdown_content += f"**Tool Execution Results**:\n```\n{tool_results}\n```\n\n"
                            else:
                                # Plain text response
                                markdown_content += f"**Execution Result**:\n```\n{result}\n```\n\n"
                        
                        # Task completion status
                        if task_completed:
                            markdown_content += "**Status**: 🎉 Task completed, ending iteration early\n\n"
                        else:
                            markdown_content += "**Status**: 🔄 Continue to next round\n\n"
                        
                        markdown_content += "---\n\n"
                    
                    elif "error" in round_info:
                        # Handle error records
                        round_num = round_info.get("round", "Unknown")
                        error_msg = round_info.get("error", "Unknown error")
                        markdown_content += f"### ❌ Round {round_num} Execution Error\n\n"
                        markdown_content += f"**Error Message**: {error_msg}\n\n---\n\n"
            
            # Add system information
            markdown_content += f"""---

## 🔧 System Information

This report was generated by {APP_NAME} Automated Task Processing System.

- **System Version**: {APP_NAME} v1.0
- **Execution Mode**: Single Task Mode
- **Report Format**: Human-readable Markdown format  
- **Generation Time**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

### 📁 Related Files

- JSON format detailed log: `single_task_report_{timestamp}.json`
- Task history record: `task_1_log.json`
""" + (f"- LLM call record: `llmcall.csv` (DEBUG mode)" if self.debug_mode else "") + f"""

### 💡 Usage Instructions

- **Single Task Mode**: Directly execute user requirements without task decomposition, suitable for simple, clear requirements
- **Multi-round Execution**: System will perform multiple rounds of dialogue and tool calls as needed to ensure task completion
- **Intelligent Stopping**: When LLM determines task is completed, it will automatically stop subsequent rounds

---

*This report contains complete task execution process, tool call details and final results for review and analysis.*
"""
            
            # Save Markdown file
            with open(markdown_file, 'w', encoding='utf-8') as f:
                f.write(markdown_content)
            
        except Exception as e:
            print_current(f"⚠️ Single task Markdown report generation failed: {e}")
    
    def generate_single_task_summary_report(self, report: Dict[str, Any], timestamp: str):
        """
        Use LLM to generate detailed summary report for single task mode (retain detail information, remove multi-round markers)
        
        Args:
            report: Execution report
            timestamp: Timestamp
        """
        try:
            # Save summary report to workspace directory for easy user access
            workspace_dir = report.get("workspace_dir")
            # 🔧 添加agent ID到summary文件名
            from tools.print_system import get_agent_id
            current_agent_id = get_agent_id()
            if current_agent_id:
                summary_filename = f"task_summary_{current_agent_id}_{timestamp}.md"
            else:
                summary_filename = f"task_summary_{timestamp}.md"
            
            if workspace_dir and os.path.exists(workspace_dir):
                summary_file = os.path.join(workspace_dir, summary_filename)
            else:
                summary_file = os.path.join(self.logs_dir, summary_filename)
            
            task = report["completed_tasks"][0] if report["completed_tasks"] else {}
            task_name = task.get("task_name", "User Requirement Execution")
            history = task.get("history", [])
            
            start_time = datetime.fromisoformat(report["start_time"])
            end_time = datetime.fromisoformat(report["end_time"])
            duration = end_time - start_time
            
            # Extract execution information (without round division)
            user_requirement = ""
            llm_responses = []
            tool_outputs = []
            final_result = ""
            
            for round_info in history:
                if isinstance(round_info, dict):
                    # Extract user requirement (only first time)
                    if "prompt" in round_info and not user_requirement:
                        prompt = round_info["prompt"]
                        if "Task Description:" in prompt:
                            desc_start = prompt.find("Task Description:") + len("Task Description:")
                            desc_end = prompt.find("\n\n", desc_start)
                            if desc_end != -1:
                                user_requirement = prompt[desc_start:desc_end].strip()
                    
                    # Extract execution results
                    if "result" in round_info:
                        result = round_info["result"]
                        
                        # Separate LLM response and tool execution results
                        if "--- Tool Execution Results ---" in result:
                            parts = result.split("--- Tool Execution Results ---")
                            llm_response = parts[0].strip()
                            tool_result = parts[1].strip() if len(parts) > 1 else ""
                            
                            if llm_response:
                                llm_responses.append(llm_response)
                            if tool_result:
                                tool_outputs.append(tool_result)
                        else:
                            # No tool execution results case
                            if result.strip():
                                llm_responses.append(result.strip())
                        
                        # Check if it's the final completion round
                        if round_info.get("task_completed", False):
                            final_result = result
            
            # Build summary prompt, requiring retention of detailed information
            summary_prompt = f"""Please generate a detailed summary report based on the following single task execution information.

Requirements:
1. Retain all important information and detailed content from LLM output
2. Remove "Round X execution" and other multi-round markers, integrate content into coherent description
3. Retain technical details, analysis process and specific implementation solutions
4. Retain all key conclusions, configuration examples, code snippets, etc.
5. Organize content in a flowing manner, avoid repetition
6. Maintain original technical depth and information completeness
7. Format in markdown, ensure readability

Single task execution information:
User requirement: {user_requirement}

LLM analysis and output:
{chr(10).join([f"Response {i+1}: {resp}" for i, resp in enumerate(llm_responses)])}

Tool execution results:
{chr(10).join([f"Tool output {i+1}: {output}" for i, output in enumerate(tool_outputs)])}

Final result: {final_result}

Please generate a markdown format detailed summary report, retaining all important information but removing round markers."""
            
            try:
                print_current(f"🧠 Using LLM to generate detailed summary report...")
                
                # Create temporary multi-round task executor to call LLM
                from multi_round_executor import MultiRoundTaskExecutor
                temp_executor = MultiRoundTaskExecutor(
                    subtask_loops=1,
                    logs_dir=self.logs_dir,
                    workspace_dir=None,
                    debug_mode=False,
                    api_key=self.api_key,
                    model=self.model,
                    api_base=self.api_base,
                    detailed_summary=False
                )
                
                # Use LLM to generate summary
                if temp_executor.executor.is_claude:
                    # Use Anthropic Claude API - batch call
                    import requests
                    
                    def get_ip_location_info():
                        try:
                            response = requests.get('http://ipapi.co/json/', timeout=5)
                            if response.status_code == 200:
                                data = response.json()
                                return {
                                    'ip': data.get('ip', 'Unknown'),
                                    'city': data.get('city', 'Unknown'),
                                    'region': data.get('region', 'Unknown'),
                                    'country': data.get('country_name', 'Unknown'),
                                    'country_code': data.get('country_code', 'Unknown'),
                                    'timezone': data.get('timezone', 'Unknown')
                                }
                        except:
                            pass
                        return {'ip': 'Unknown', 'city': 'Unknown', 'region': 'Unknown', 'country': 'Unknown', 'country_code': 'Unknown', 'timezone': 'Unknown'}
                    
                    current_date = datetime.now()
                    location_info = get_ip_location_info()
                    system_prompt = f"""You are a professional technical document organization assistant, skilled at retaining technical details while organizing clear and coherent reports. You need to retain all important technical information, analysis processes and specific implementations, just remove multi-round execution markers to make content more flowing.

**Current Date Information**:
- Current Date: {current_date.strftime('%Y-%m-%d')}
- Current Time: {current_date.strftime('%Y-%m-%d %H:%M:%S')}

**Location Information**:
- City: {location_info['city']}
- Country: {location_info['country']}"""
                    
                    print_current("🔄 Starting generation of single task summary...")
                    response = temp_executor.executor.client.messages.create(
                        model=self.model,
                        max_tokens=temp_executor.executor._get_max_tokens_for_model(self.model),
                        system=system_prompt,
                        messages=[
                            {"role": "user", "content": summary_prompt}
                        ],
                        temperature=0.7
                    )
                    
                    # Get complete response content (Claude API structure)
                    if response.content and len(response.content) > 0:
                        llm_summary = response.content[0].text
                    else:
                        llm_summary = "Summary generation failed: API returned empty response"
                        print_current("⚠️ Warning: Claude API returned empty content list")
                    print_current("📋 Single task summary generated")
                    
                else:
                    # Use OpenAI API - batch call
                    import requests
                    
                    def get_ip_location_info():
                        try:
                            response = requests.get('http://ipapi.co/json/', timeout=5)
                            if response.status_code == 200:
                                data = response.json()
                                return {
                                    'ip': data.get('ip', 'Unknown'),
                                    'city': data.get('city', 'Unknown'),
                                    'region': data.get('region', 'Unknown'),
                                    'country': data.get('country_name', 'Unknown'),
                                    'country_code': data.get('country_code', 'Unknown'),
                                    'timezone': data.get('timezone', 'Unknown')
                                }
                        except:
                            pass
                        return {'ip': 'Unknown', 'city': 'Unknown', 'region': 'Unknown', 'country': 'Unknown', 'country_code': 'Unknown', 'timezone': 'Unknown'}
                    
                    current_date = datetime.now()
                    location_info = get_ip_location_info()
                    system_prompt = f"""You are a professional technical document organization assistant, skilled at retaining technical details while organizing clear and coherent reports. You need to retain all important technical information, analysis processes and specific implementations, just remove multi-round execution markers to make content more flowing.

**Current Date Information**:
- Current Date: {current_date.strftime('%Y-%m-%d')}
- Current Time: {current_date.strftime('%Y-%m-%d %H:%M:%S')}

**Location Information**:
- City: {location_info['city']}
- Country: {location_info['country']}"""
                    
                    print_current("🔄 Starting generation of single task summary...")
                    response = temp_executor.executor.client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": summary_prompt}
                        ],
                        max_tokens=temp_executor.executor._get_max_tokens_for_model(self.model),
                        temperature=0.7,
                        top_p=0.9
                    )
                    
                    # Get complete response content
                    if response.choices and len(response.choices) > 0:
                        llm_summary = response.choices[0].message.content
                    else:
                        llm_summary = "Summary generation failed: API returned empty response"
                        print_current("⚠️ Warning: OpenAI API returned empty choices list")
                    print_current("📋 Single task summary generated")
                
                # Build final summary report
                final_summary = f"""# {APP_NAME} Task Summary Report

> 📊 **Quick Overview**: Single task execution completed | Duration: {duration} | Status: ✅ Success

---

## 📋 Task Details

**User Requirement**: {user_requirement}

---

## 🎯 Execution Results and Analysis

{llm_summary}

---

## 📊 Execution Overview

- **Execution Time**: {start_time.strftime('%Y-%m-%d %H:%M:%S')}
- **Total Duration**: {duration}
- **Task Status**: ✅ Execution Completed
- **Working Directory**: {report.get("workspace_dir", "Not specified")}

---

📅 **Report Generation Time**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
📁 **Detailed Log**: `single_task_report_{timestamp}.md`  

---

*This report was automatically generated by {APP_NAME}, retaining complete task execution information and technical details.*
"""
                
                # Save summary report
                with open(summary_file, 'w', encoding='utf-8') as f:
                    f.write(final_summary)
                
                print_current(f"📋 Detailed summary report saved to: {summary_file}")
                
            except Exception as e:
                print_current(f"⚠️ LLM summary generation failed: {e}")
                # Use backup plan - directly organize existing information
                self._generate_detailed_single_task_summary(report, timestamp, user_requirement, llm_responses, tool_outputs, final_result)
            
        except Exception as e:
            print_current(f"⚠️ Single task summary report generation failed: {e}")
    
    def _generate_detailed_single_task_summary(self, report: Dict[str, Any], timestamp: str, user_requirement: str, llm_responses: list, tool_outputs: list, final_result: str):
        """
        Generate detailed single task summary report (backup plan) - retain detailed information
        """
        try:
            # Save summary report to workspace directory for easy user access
            workspace_dir = report.get("workspace_dir")
            if workspace_dir and os.path.exists(workspace_dir):
                summary_file = os.path.join(workspace_dir, f"task_summary_{timestamp}.md")
            else:
                summary_file = os.path.join(self.logs_dir, f"task_summary_{timestamp}.md")
            
            start_time = datetime.fromisoformat(report["start_time"])
            duration = datetime.fromisoformat(report["end_time"]) - start_time
            
            detailed_summary = f"""# {APP_NAME} Task Summary Report

> 📊 **Quick Overview**: Single task execution completed | Duration: {duration} | Status: ✅ Success

---

## 📋 Task Details

**User Requirement**: {user_requirement}

---

## 🎯 Execution Process and Results

"""
            
            # Add LLM analysis and planning
            if llm_responses:
                detailed_summary += "### Analysis and Execution\n\n"
                for i, response in enumerate(llm_responses):
                    # Clean content, remove multi-round markers
                    clean_response = response.replace("Round 1 Execution", "").replace("Round 2 Execution", "").replace("Round 3 Execution", "")
                    clean_response = clean_response.replace("## Round", "## ").replace("### Round", "### ")
                    if clean_response.strip():
                        detailed_summary += f"{clean_response}\n\n"
            
            # Add tool execution results (if there's important information)
            if tool_outputs:
                detailed_summary += "### Execution Results\n\n"
                for output in tool_outputs:
                    # Only add meaningful tool outputs
                    if len(output) > 50 and any(keyword in output for keyword in ['success', 'completed', 'created', 'modified', 'result', 'content']):
                        # Use configured history truncation length
                        history_truncation_length = get_truncation_length()
                        detailed_summary += f"```\n{output[:history_truncation_length]}...\n```\n\n"
            
            # Add final result
            if final_result:
                clean_final = final_result.replace("TASK_COMPLETED:", "").strip()
                if clean_final:
                    detailed_summary += f"### Final Conclusion\n\n{clean_final}\n\n"
            
            detailed_summary += f"""---

## 📊 Execution Overview

- **Execution Time**: {start_time.strftime('%Y-%m-%d %H:%M:%S')}
- **Total Duration**: {duration}
- **Task Status**: ✅ Execution Completed
- **Working Directory**: {report.get("workspace_dir", "Not specified")}

---

📅 **Report Generation Time**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
📁 **Detailed Log**: `single_task_report_{timestamp}.md`  

---

*This report was automatically generated by {APP_NAME}, retaining complete task execution information and technical details.*
"""
            
            # Save detailed summary
            with open(summary_file, 'w', encoding='utf-8') as f:
                f.write(detailed_summary)
            
            print_current(f"📋 Detailed summary report saved to: {summary_file}")
            
        except Exception as e:
            print_current(f"⚠️ Detailed summary report generation failed: {e}")
    
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
    
    def run(self, user_requirement: str = None, loops: int = 3) -> bool:
        """
        Run complete workflow
        
        Args:
            user_requirement: User requirement (optional)
            loops: Execution rounds for each task
            
        Returns:
            Whether successfully completed
        """
        track_operation("主程序执行")
        
        workspace_dir = os.path.join(self.out_dir, "workspace")
        
        if not self.single_task_mode:
            print_current(f"📋 Task File: {os.path.abspath(self.todo_csv_path)}")
        
        # Step 1: Get user requirement
        track_operation("获取用户需求")
        requirement = self.get_user_requirement(user_requirement)
        finish_operation("获取用户需求")
        
        if not requirement:
            print_current("❌ Invalid user requirement")
            return False
        
        # Choose execution path based on mode
        if self.single_task_mode:
            # Single task mode: directly execute user requirement
            track_operation("单任务执行")
            if not self.execute_single_task(requirement, loops):
                print_current("⚠️ Single task execution reached maximum rounds")  # 🔧 修复：区分失败和达到最大轮数
                finish_operation("单任务执行")
                finish_operation("主程序执行")
                return False
            finish_operation("单任务执行")
                
        else:
            # Step 2: Task decomposition
            track_operation("任务分解")
            if not self.decompose_task(requirement):
                print_current("❌ Task decomposition failed, program terminated")
                finish_operation("任务分解")
                finish_operation("主程序执行")
                return False
            finish_operation("任务分解")
            
            # Interactive mode confirmation is handled by individual task execution
            # No need for pre-execution confirmation here
            
            # Step 3: Execute tasks
            track_operation("多任务执行")
            if not self.execute_tasks(loops):
                print_current("❌ Task execution failed")
                finish_operation("多任务执行")
                finish_operation("主程序执行")
                return False
            finish_operation("多任务执行")
        
        # Task execution completed
        print_current(f"📁 All output files saved at: {os.path.abspath(self.out_dir)}")
        print_current(f"💻 Code files saved at: {os.path.abspath(workspace_dir)}")
        
        # Save current output directory for future continue operations
        save_last_output_dir(self.out_dir)
        
        print_current("\n🎉 Workflow completed!")
        finish_operation("主程序执行")
        return True


class AGIBotClient:
    """
    AGI Bot Python Library Interface
    
    Provides OpenAI-like chat interface for programmatic access to AGI Bot functionality.
    Does not rely on config.txt file - all configuration is passed during initialization.
    
    Example usage:
        client = AGIBotClient(
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
                 api_key: str,
                 model: str,
                 api_base: str = None,
                 debug_mode: bool = False,
                 detailed_summary: bool = True,
                 single_task_mode: bool = True,
                 interactive_mode: bool = False,
                 streaming: bool = None):
        """
        Initialize AGI Bot Client
        
        Args:
            api_key: API key for LLM service
            model: Model name (e.g., 'gpt-4', 'claude-3-sonnet-20240229')
            api_base: API base URL (optional)
            debug_mode: Whether to enable DEBUG mode
            detailed_summary: Whether to enable detailed summary mode
            single_task_mode: Whether to use single task mode (default: True)
            interactive_mode: Whether to enable interactive mode
            streaming: Whether to use streaming output (None to use config.txt)
        """
        if not api_key:
            raise ValueError("api_key is required")
        if not model:
            raise ValueError("model is required")
            
        self.api_key = api_key
        self.model = model
        self.api_base = api_base
        self.debug_mode = debug_mode
        self.detailed_summary = detailed_summary
        self.single_task_mode = single_task_mode
        self.interactive_mode = interactive_mode
        self.streaming = streaming
        
    def chat(self, 
             messages: list,
             dir: str = None,
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
            loops: Maximum execution rounds per task (default: 50)
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
            dir = f"agibot_output_{timestamp}"
        
        try:
            # 🔧 检查当前线程的agent_id并设置到主线程
            from tools.print_system import get_agent_id, set_agent_id
            current_agent_id = get_agent_id()
            
            # Create AGI Bot main instance
            main_app = AGIBotMain(
                out_dir=dir,
                api_key=self.api_key,
                model=self.model,
                api_base=self.api_base,
                debug_mode=self.debug_mode,
                detailed_summary=self.detailed_summary,
                single_task_mode=self.single_task_mode,
                interactive_mode=self.interactive_mode,
                continue_mode=continue_mode,
                streaming=self.streaming
            )
            
            # 🔧 如果有agent_id，设置到主线程中
            if current_agent_id:
                set_agent_id(current_agent_id)
                print_current(f"🏷️ AGIBotClient using agent ID: {current_agent_id}")
            
            # Execute the task
            if current_agent_id:
                print_current(f"🚀 Executing task: {user_message}")
            else:
                print_manager(f"🚀 Executing task: {user_message}")
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
                # 🔧 修复：区分失败和达到最大轮数
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
def create_client(api_key: str, model: str, **kwargs) -> AGIBotClient:
    """
    Convenience function to create AGI Bot client
    
    Args:
        api_key: API key for LLM service
        model: Model name
        **kwargs: Additional configuration parameters
        
    Returns:
        AGIBotClient instance
    """
    return AGIBotClient(api_key=api_key, model=model, **kwargs)


def main():
    """
    Main function - handle command line parameters
    """
    # Install debug system first (re-enabled for debugging freeze issues)
    debug_system = install_debug_system(
        enable_stack_trace=True,
        enable_memory_monitor=True, 
        enable_execution_tracker=True
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
  python main.py --requirement "Fix game sound effect playback issue"
  python main.py --singletask --requirement "Optimize code performance"
  
  # Multi-task mode - automatically decompose task into multiple subtasks for execution
  python main.py --todo --requirement "Develop a complete Python Web application"
  
  # Continue from last output directory
  python main.py --continue --requirement "Continue working on the previous task"
  python main.py -c --requirement "Add new features to existing project"
  
  # Interactive mode - prompt user for requirement input
  python main.py  # Single task mode
  python main.py --todo  # Multi-task mode
  
  # Specify output directory and execution rounds
  python main.py --dir my_project --loops 5 --requirement "Requirement description"
  
  # Use custom model configuration
  python main.py --api-key YOUR_KEY --model gpt-4 --base-url https://api.openai.com/v1
        """
    )
    
    parser.add_argument(
        "--requirement", "-r",
        help="User requirement description. If not provided, will enter interactive mode to prompt user input"
    )
    
    parser.add_argument(
        "--dir", "-d",
        default=f"output_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        help="Output directory for storing todo.csv and logs (default: output_timestamp)"
    )
    
    parser.add_argument(
        "--loops", "-l",
        type=int,
        default=50,
        help="Execution rounds for each subtask (default: 50)"
    )
    
    parser.add_argument(
        "--api-key",
        default=None,
        help="API key"
    )
    
    parser.add_argument(
        "--model", "-m",
        default=None,
        help="LLM model name (will load from config.txt if not specified)"
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
        "--todo",
        action="store_true",
        default=False,
        help="Enable multi-task mode, use task decomposer to decompose requirement into multiple subtasks"
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
    
    # Check for conflicting parameters: --continue and --dir
    user_specified_out_dir = '--dir' in sys.argv or '-d' in sys.argv
    if args.continue_mode and user_specified_out_dir:
        # User specified both --continue/-c and --dir
        print_current("⚠️  Warning: Both --continue/-c and --dir parameters were specified.")
        print_current("    The --continue/-c parameter takes priority and --dir will be ignored.")
        print_current("    If you want to use a specific output directory, don't use --continue/-c.")
        print()
    
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
        args.model = None  # Let it load from config.txt
        args.api_base = None
        print_current(f"📁 Output directory: {args.dir}")
        print_current(f"🔄 Execution rounds: {args.loops}")
        print_current(f"🤖 Model: Will load from config.txt")
        print()
    
    # Get API key
    api_key = args.api_key
    
    # Determine summary mode
    detailed_summary = not args.simple_summary if hasattr(args, 'simple_summary') else args.detailed_summary
    
    # Determine task mode
    single_task_mode = not args.todo if hasattr(args, 'todo') else args.singletask
    
    # Create and run main program
    try:
        main_app = AGIBotMain(
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