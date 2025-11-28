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
AGI Agent Command Line Interface

This file provides the command line interface for AGI Agent.
All source code has been moved to the src/ directory.
"""

# Add src directory to Python path
import os
import sys
import warnings

# Suppress asyncio warnings that occur during FastMCP cleanup - must be early
warnings.filterwarnings("ignore", message=".*Event loop is closed.*")
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*coroutine.*was never awaited.*")
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*was never awaited.*")

# Also suppress at the asyncio level
import asyncio
import logging
import atexit
logging.getLogger('asyncio').setLevel(logging.CRITICAL)

# Set environment variable to suppress asyncio debug output
os.environ['PYTHONWARNINGS'] = 'ignore::RuntimeWarning:asyncio'
os.environ['PYTHONASYNCIODEBUG'] = '0'

# Install custom exception hook to suppress asyncio cleanup warnings
def custom_excepthook(exc_type, exc_value, exc_traceback):
    """Custom exception hook that filters out asyncio cleanup warnings"""
    if exc_type == RuntimeError and "Event loop is closed" in str(exc_value):
        # Suppress this specific warning
        return
    # For all other exceptions, use the default handler
    sys.__excepthook__(exc_type, exc_value, exc_traceback)

sys.excepthook = custom_excepthook

# Application name macro definition
APP_NAME = "AGIAgent"

from src.tools.print_system import print_current
from src.tools.debug_system import install_debug_system
from src.main import AGIAgentMain

import argparse
import json
import atexit
from datetime import datetime
from typing import Dict, Any, Optional

# Configuration file to store last output directory
LAST_OUTPUT_CONFIG_FILE = ".agia_last_output.json"

# Global cleanup flag
_cleanup_executed = False

def is_jupyter_environment():
    """
    Check if the code is running in a Jupyter environment
    
    Returns:
        bool: True if running in Jupyter, False otherwise
    """
    try:
        # Check if we're in a Jupyter environment by looking for common indicators
        import sys
        # Check for Jupyter-specific modules in sys.modules
        jupyter_modules = ['ipykernel', 'jupyter_client', 'IPython']
        for module in jupyter_modules:
            if module in sys.modules:
                return True
        
        # Check environment variables that indicate Jupyter
        import os
        jupyter_env_vars = ['JPY_PARENT_PID', 'JUPYTER_DATA_DIR', 'IPYTHON']
        for var in jupyter_env_vars:
            if var in os.environ:
                return True
                
    except Exception:
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
    """Print ASCII art banner for AGI Agent (only if appropriate environment)"""
    if not should_show_banner():
        return
    
    # ANSI color codes - Bright blue
    BRIGHT_BLUE = '\033[94m'
    RESET = '\033[0m'
    
    banner = f"""{BRIGHT_BLUE}
                                                                  
       █████╗  ██████╗ ██╗     █████╗  ██████╗ ███████╗███╗   ██╗████████╗
      ██╔══██╗██╔════╝ ██║    ██╔══██╗██╔════╝ ██╔════╝████╗  ██║╚══██╔══╝
      ███████║██║  ███╗██║    ███████║██║  ███╗█████╗  ██╔██╗ ██║   ██║   
      ██╔══██║██║   ██║██║    ██╔══██║██║   ██║██╔══╝  ██║╚██╗██║   ██║   
      ██║  ██║╚██████╔╝██║    ██║  ██║╚██████╔╝███████╗██║ ╚████║   ██║   
      ╚═╝  ╚═╝ ╚═════╝ ╚═╝    ╚═╝  ╚═╝ ╚═════╝ ╚══════╝╚═╝  ╚═══╝   ╚═╝   
                                                                  
                     🚀 Autonomous Task Execution System                      
                    🧠 LLM-Powered Cognitive Architecture                   
                                                                  {RESET}
    """
    print(banner)

def global_cleanup():
    """Global cleanup function to ensure all resources are properly released"""
    global _cleanup_executed
    if _cleanup_executed:
        return
    _cleanup_executed = True
    
    try:
        import time
        
        # Import here to avoid circular imports
        # Note: AgentManager class is not implemented, skipping cleanup
        
        # Stop message router if it exists
        try:
            from src.tools.message_system import get_message_router
            router = get_message_router()
            if router:
                router.stop()
        except:
            pass
        
        # Cleanup debug system
        try:
            from src.tools.debug_system import get_debug_system
            debug_sys = get_debug_system()
            debug_sys.cleanup()
        except Exception as e:
            print_current(f"⚠️ Debug system cleanup warning: {e}")
        
        # Cleanup global code index manager
        try:
            from src.tools.global_code_index_manager import GlobalCodeIndexManager
            manager = GlobalCodeIndexManager()
            manager.cleanup_all()
        except Exception as e:
            print_current(f"⚠️ Code index cleanup warning: {e}")
        
        # Small delay to allow daemon threads to finish current operations
        time.sleep(0.1)
        
        # Force garbage collection
        import gc
        gc.collect()
        
        
    except Exception as e:
        print_current(f"⚠️ Error during final cleanup: {e}")

def signal_handler(signum, frame):
    """Handle interrupt signals"""
    print_current(f"\n⚠️ Signal received {signum}, cleaning up...")
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
        from src.tools.print_system import get_agent_id
        current_agent_id = get_agent_id()
        
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

def main():
    """
    Main function - handle command line parameters
    """
    # Parse arguments first to get output directory
    parser = argparse.ArgumentParser(
        description=f"{APP_NAME} Automated Task Processing System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Usage Examples:
  # Single task mode (default) - directly execute user requirement without task decomposition
  python agia.py "Fix game sound effect playback issue"
  python agia.py --requirement "Fix game sound effect playback issue"
  python agia.py -r "Fix game sound effect playback issue"
  python agia.py --singletask "Optimize code performance"
  
  # Multi-task mode - automatically decompose task into multiple subtasks for execution
  python agia.py --todo "Develop a complete Python Web application"
  python agia.py --todo --requirement "Develop a complete Python Web application"
  
  # Multi-task mode with routine file - use routine guidelines for task planning
  python agia.py --todo --routine "my_routine.txt" "Develop a web application"
  python agia.py --todo -u "routines/development.md" --requirement "Build an API server"
  
  # Single task mode with routine file - append routine content to user requirement
  python agia.py --routine "my_routine.txt" "Create a Python script"
  python agia.py -u "routines/coding_style.md" --requirement "Refactor existing code"
  
  # Continue from last output directory
  python agia.py --continue "Continue working on the previous task"
  python agia.py -c --requirement "Add new features to existing project"
  
  # Link to external code directory - AGI Agent will operate on external code
  python agia.py --link-dir /path/to/your/project "Add new features to the project"
  python agia.py --link-dir ./my_project --dir output_folder "Refactor the codebase"
  
  # Interactive mode - prompt user for requirement input
  python agia.py  # Single task mode
  python agia.py --todo  # Multi-task mode
  
  # Specify output directory and execution rounds
  python agia.py --dir my_project --loops 5 "Requirement description"
  
  # Infinite loop execution (until task completion or manual interruption)
  python agia.py --loops -1 "Requirement description"
  
  # Use custom model configuration
  python agia.py --api-key YOUR_KEY --model gpt-4 --base-url https://api.openai.com/v1 "My requirement"
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
        help="Output directory for storing todo.md and logs (default: output_timestamp)"
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
        "--todo",
        action="store_true",
        default=False,
        help="Enable multi-task mode, use task decomposer to decompose requirement into multiple subtasks"
    )
    
    parser.add_argument(
        "--version", "-v",
        action="version",
        version=f"{APP_NAME} v1.0.2"
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
    
    parser.add_argument(
        "--link-dir",
        default=None,
        help="Link to external code directory. Creates a symbolic link in workspace directory pointing to the specified path, allowing AGI Agent to manipulate external code projects."
    )
    
    parser.add_argument(
        "--routine", "-u",
        type=str,
        default=None,
        help="Routine file path to include routine guidelines. In --todo mode: integrates into task planning. In single-task mode: appends to user requirement."
    )
    
    parser.add_argument(
        "--plan",
        action="store_true",
        default=False,
        help="Enable plan mode: interact with user to create plan.md document, then exit without executing tasks"
    )
    
    args = parser.parse_args()
    
    # Handle requirement argument priority: positional argument takes precedence over --requirement/-r
    if args.requirement_positional:
        args.requirement = args.requirement_positional
    
    # Handle continue mode BEFORE setting up output directory
    if args.continue_mode:
        user_specified_out_dir = '--dir' in sys.argv or '-d' in sys.argv
        if user_specified_out_dir:
            # User specified both --continue/-c and --dir
            print("⚠️  Warning: Both --continue/-c and --dir parameters were specified.")
            print("    The --continue/-c parameter takes priority and --dir will be ignored.")
            print("    If you want to use a specific output directory, don't use --continue/-c.")
        
        # Load last output directory
        last_dir = load_last_output_dir()
        
        # If an explicit out_dir is provided and the directory exists, prioritize using the provided directory
        # This is mainly to support directory selection in GUI mode
        # Check if absolute or relative path exists
        out_dir_abs = os.path.abspath(args.dir)
        if args.dir != "output" and (os.path.exists(args.dir) or os.path.exists(out_dir_abs)):
            print(f"🔄 Continue mode: Using specified directory: {args.dir}")
        elif last_dir:
            args.dir = last_dir
            print(f"🔄 Continue mode: Using last output directory: {args.dir}")
        else:
            print("⚠️ Continue mode requested but no valid last output directory found")
            print("    Creating new output directory instead")
    
    # Check if no parameters provided, if so use default parameters
    if len(sys.argv) == 1:  # Only script name, no other parameters
        print("🔧 No parameters provided, using default configuration...")
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
        print(f"📁 Output directory: {args.dir}")
        print(f"🔄 Execution rounds: {args.loops}")
        print(f"🤖 Model: Will load from config/config.txt")

    # Set up output directory for logging BEFORE importing any modules that might produce logs
    # This ensures all subsequent logs go to the correct directory
    from src.tools.print_system import set_output_directory
    set_output_directory(args.dir)
    
    # Install debug system after setting output directory
    from src.config_loader import load_config
    config = load_config()
    enable_debug_system = config.get('enable_debug_system', 'False').lower() == 'true'
    if enable_debug_system:
        install_debug_system(
            enable_stack_trace=True,
            enable_memory_monitor=True, 
            enable_execution_tracker=True
        )
    
    # Register cleanup handlers
    atexit.register(global_cleanup)
    # Note: signal handlers are now managed by debug system
    
    # Print ASCII banner at startup
    print_ascii_banner()

    
    # Get API key
    api_key = args.api_key
    
    # Determine summary mode
    detailed_summary = not args.simple_summary if hasattr(args, 'simple_summary') else args.detailed_summary
    
    # Determine task mode
    single_task_mode = not args.todo if hasattr(args, 'todo') else args.singletask
    
    # Get plan mode
    plan_mode = args.plan if hasattr(args, 'plan') else False
    
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
            continue_mode=args.continue_mode,
            link_dir=args.link_dir,
            routine_file=args.routine,
            plan_mode=plan_mode
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