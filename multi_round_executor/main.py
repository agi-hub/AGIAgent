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

 #!/usr/bin/env python3
"""
Main entry point for the multi-round task executor
"""

import os
import argparse
from .executor import MultiRoundTaskExecutor


def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(description="Multi-round task executor")
    parser.add_argument("csv_file", help="todo.csv file path")
    parser.add_argument("--loops", type=int, default=3, 
                       help="Number of execution rounds for each subtask (default: 3)")
    parser.add_argument("--logs-dir", default="logs", 
                       help="Log directory path (default: logs)")
    parser.add_argument("--workspace-dir", 
                       help="Workspace directory path (default: auto-create based on CSV filename)")
    parser.add_argument("--debug", action="store_true", 
                       help="Enable DEBUG mode")
    parser.add_argument("--simple-summary", action="store_true",
                       help="Use simplified summary mode (default: detailed summary)")
    parser.add_argument("--api-key", 
                       help="API key for the language model")
    parser.add_argument("--model", default=None,
                       help="Model name (will load from config.txt if not specified)")
    parser.add_argument("--api-base", default=None,
                       help="API base URL")
    
    args = parser.parse_args()
    
    # Check if CSV file exists
    if not os.path.exists(args.csv_file):
        print(f"Error: File not found {args.csv_file}")
        return 1
    
    # Determine workspace directory
    if args.workspace_dir:
        workspace_dir = args.workspace_dir
    else:
        # Auto-create workspace directory based on CSV filename
        csv_basename = os.path.splitext(os.path.basename(args.csv_file))[0]
        workspace_dir = f"{csv_basename}_workspace"
    
    print(f"📁 Workspace directory will be set to: {workspace_dir}")
    
    # Create and run executor
    executor = MultiRoundTaskExecutor(
        subtask_loops=args.loops,
        logs_dir=args.logs_dir,
        workspace_dir=workspace_dir,
        debug_mode=args.debug,
        detailed_summary=not args.simple_summary,
        api_key=args.api_key,
        model=args.model,
        api_base=args.api_base
    )
    
    try:
        # Execute all tasks
        result = executor.execute_all_tasks(args.csv_file)
        
        # Return appropriate exit code
        if result.get("success", False):
            return 0
        else:
            return 1
            
    except KeyboardInterrupt:
        print("\n⚠️ Execution interrupted by user")
        return 130
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())