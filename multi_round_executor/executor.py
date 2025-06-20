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
Main executor class for multi-round task execution
"""

import os
import re
from datetime import datetime
from typing import List, Dict, Any, Optional

from tool_executor import ToolExecutor
from config_loader import get_model, get_truncation_length, get_summary_history
from .config import (
    DEFAULT_SUBTASK_LOOPS, DEFAULT_LOGS_DIR, DEFAULT_MODEL,
    DEFAULT_DETAILED_SUMMARY, extract_session_timestamp
)
from .task_loader import TaskLoader
from .summary_generator import SummaryGenerator
from .report_generator import ReportGenerator
from .debug_recorder import DebugRecorder
from .task_checker import TaskChecker


class MultiRoundTaskExecutor:
    """Main executor for multi-round task execution"""
    
    def __init__(self, subtask_loops: int = DEFAULT_SUBTASK_LOOPS, 
                 logs_dir: str = DEFAULT_LOGS_DIR, workspace_dir: str = None, 
                 debug_mode: bool = False, api_key: str = None, 
                 model: str = DEFAULT_MODEL, api_base: str = None, 
                 detailed_summary: bool = DEFAULT_DETAILED_SUMMARY,
                 interactive_mode: bool = False):
        """
        Initialize multi-round task executor
        
        Args:
            subtask_loops: Number of rounds to execute for each subtask
            logs_dir: Log directory path
            workspace_dir: Working directory for storing code files
            debug_mode: Whether to enable DEBUG mode
            api_key: API key
            model: Model name
            api_base: API base URL
            detailed_summary: Whether to generate detailed summary
            interactive_mode: Whether to enable interactive mode
        """
        self.subtask_loops = subtask_loops
        self.logs_dir = logs_dir
        self.workspace_dir = workspace_dir
        self.debug_mode = debug_mode
        self.api_key = api_key
        
        # Load model from config.txt if not provided
        if model is None:
            model = get_model()
            if model is None:
                raise ValueError("Model not found. Please provide model parameter or set it in config.txt")
        self.model = model
        
        self.api_base = api_base
        self.detailed_summary = detailed_summary
        self.interactive_mode = interactive_mode
        self.task_summaries = []  # Store summaries of all tasks
        
        # Extract session timestamp
        session_timestamp = extract_session_timestamp(logs_dir)
        if session_timestamp:
            print(f"📅 Detected session timestamp: {session_timestamp}")
        
        # Ensure directories exist
        os.makedirs(logs_dir, exist_ok=True)
        if workspace_dir:
            os.makedirs(workspace_dir, exist_ok=True)
            print(f"📁 Workspace directory created: {workspace_dir}")
        
        # Initialize tool executor
        self.executor = ToolExecutor(
            api_key=api_key,
            model=model, 
            api_base=api_base,
            workspace_dir=workspace_dir,
            debug_mode=debug_mode,
            logs_dir=logs_dir,
            session_timestamp=session_timestamp,
            interactive_mode=interactive_mode
        )
        
        # Initialize module components
        self.task_loader = TaskLoader()
        self.summary_generator = SummaryGenerator(self.executor, detailed_summary)
        self.report_generator = ReportGenerator(self.executor, logs_dir, workspace_dir)
        self.debug_recorder = DebugRecorder(debug_mode)
        self.task_checker = TaskChecker()
        
        # Print configuration
        self._print_configuration()
    
    def _print_configuration(self):
        """Print executor configuration"""
        # print(f"\n📋 Task executor configuration:")  # Commented out to reduce terminal noise
        # print(f"   Subtask execution rounds: {self.subtask_loops}")  # Commented out to reduce terminal noise
        # print(f"   Log directory: {self.logs_dir}")  # Commented out to reduce terminal noise
        # if self.workspace_dir:
        #     print(f"   Workspace directory: {self.workspace_dir}")  # Commented out to reduce terminal noise
        # if self.debug_mode:
        #     print(f"   DEBUG mode: Enabled")  # Commented out to reduce terminal noise
        # if self.detailed_summary:
        #     print(f"   Detailed summary: Enabled (retain more technical information)")  # Commented out to reduce terminal noise
        # else:
        #     print(f"   Detailed summary: Disabled (use simplified summary)")  # Commented out to reduce terminal noise
        
        # Print history summarization status
        # summary_history_enabled = get_summary_history()
        # print(f"   History Summarization: {'✅ Enabled' if summary_history_enabled else '❌ Disabled'}")  # Commented out to reduce terminal noise
        # print()  # Commented out to reduce terminal noise
        pass
    
    def load_todo_csv(self, csv_file: str) -> List[Dict[str, Any]]:
        """
        Load todo.csv file
        
        Args:
            csv_file: CSV file path
            
        Returns:
            Task list
        """
        return self.task_loader.load_todo_csv(csv_file)
    
    def execute_single_task(self, task: Dict[str, Any], task_index: int, 
                           total_tasks: int, previous_tasks_summary: str = "") -> Dict[str, Any]:
        """
        Execute multi-round calls for a single task
        
        Args:
            task: Task information
            task_index: Current task index (starting from 0)
            total_tasks: Total number of tasks
            previous_tasks_summary: Intelligent summary of prerequisite tasks
            
        Returns:
            Task execution results and history
        """
        task_id = task.get('Task ID', task.get('任务编号', ''))
        task_name = task.get('Task Name', task.get('任务名称', ''))
        task_desc = task.get('Task Description', task.get('任务详细描述', ''))
        
        print(f"📋 Starting task execution [{task_index + 1}/{total_tasks}] - Task{task_id}: {task_name}")

        
        # Initialize task history
        task_history = []
        
        # Add prerequisite task summary
        if previous_tasks_summary:
            context_summary = f"Summary of prerequisite task completion:\n\n{previous_tasks_summary}\n\nPlease continue executing the current task based on the above completed task information.\n\n"
            task_history.append({
                "role": "system",
                "content": context_summary,
                "timestamp": datetime.now().isoformat()
            })
            print(f"📚 Loaded prerequisite task intelligent summary")
        
        # Add legacy task summaries as backup
        if self.task_summaries and not previous_tasks_summary:
            context_summary = "The following is summary information of previously completed tasks:\n\n"
            for i, summary in enumerate(self.task_summaries, 1):
                context_summary += f"Task{i} summary: {summary}\n\n"
            context_summary += "Please continue executing the current task based on the above completed task information.\n\n"
            task_history.append({
                "role": "system",
                "content": context_summary,
                "timestamp": datetime.now().isoformat()
            })
            print(f"📚 Loaded {len(self.task_summaries)} prerequisite task summaries (basic mode)")
        
        # Execute multi-round task
        task_history = self._execute_task_rounds(task_id, task_name, task_desc, task_history)
        
        # Check if user interrupted execution
        user_interrupted = any(record.get("user_interrupted", False) for record in task_history)
        if user_interrupted:
            print(f"🛑 Task {task_id} execution stopped by user")
            return {
                "task_id": task_id,
                "task_name": task_name,
                "task_description": task_desc,
                "history": task_history,
                "status": "user_interrupted"
            }
        
        # Check if task was actually completed or just reached max rounds
        completed_rounds = [h for h in task_history if isinstance(h, dict) and h.get("task_completed", False)]
        if completed_rounds:
            print(f"✅ Task {task_id} execution completed successfully")
            status = "completed"
        else:
            print(f"⚠️ Task {task_id} reached maximum rounds without explicit completion")
            status = "max_rounds_reached"
        
        return {
            "task_id": task_id,
            "task_name": task_name,
            "task_description": task_desc,
            "history": task_history,
            "status": status
        }
    
    def _execute_task_rounds(self, task_id: str, task_name: str, task_desc: str, 
                            task_history: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Execute multiple rounds for a single task
        
        Args:
            task_id: Task ID
            task_name: Task name
            task_desc: Task description
            task_history: Existing task history
            
        Returns:
            Updated task history
        """
        base_prompt = f"Task description: {task_desc}"
        max_rounds = self.subtask_loops
        task_round = 1  # Renamed from round_num to task_round for clarity
        task_completed = False
        
        while task_round <= max_rounds and not task_completed:
            #print(f"\n🔄 --- Task Round {task_round}/{max_rounds} execution ---")
            
            # Use base prompt directly - round info will be handled by _build_new_user_message
            current_prompt = base_prompt
            
            try:
                print(f"⏳ Starting task round {task_round} execution...")
                
                # Prepare history for LLM
                history_for_llm = [record for record in task_history 
                                 if "prompt" in record and "result" in record and not record.get("error")]
                
                # Execute task - this will handle internal tool calling rounds
                # The tool executor's internal rounds are separate from task rounds
                result = self.executor.execute_subtask(
                    current_prompt, 
                    "prompts.txt",
                    task_history=history_for_llm,
                    execution_round=task_round
                )
                
                # Check if user interrupted execution
                if result.startswith("USER_INTERRUPTED:"):
                    print(f"🛑 User interrupted execution: {result}")
                    # Record the interruption
                    round_record = {
                        "task_round": task_round, 
                        "prompt": current_prompt,
                        "result": result,
                        "task_completed": False,
                        "user_interrupted": True,
                        "timestamp": datetime.now().isoformat()
                    }
                    task_history.append(round_record)
                    print(f"🛑 Task execution stopped by user at task round {task_round}")
                    break
                
                # Check task completion
                task_completed = self.task_checker.check_task_completion(result)
                
                # Record debug information
                self._record_debug_info(task_id, task_name, task_round, current_prompt, result, task_completed, len(task_history))
                
                # Record round results
                round_record = {
                    "task_round": task_round, 
                    "prompt": current_prompt,
                    "result": result,
                    "task_completed": task_completed,
                    "timestamp": datetime.now().isoformat()
                }
                task_history.append(round_record)
                
                print(f"✅ Task round {task_round} execution completed")
                
                # Update codebase
                try:
                    self.executor.tools.perform_incremental_update()
                except Exception as e:
                    print(f"⚠️ Codebase incremental update failed: {e}")
                
                # Display execution summary
                self._display_round_summary(task_round, result)
                
                # Check if task is completed
                if task_completed:
                    print(f"🎉 Large model determined task is completed, ending task iteration early!")
                    break
                else:
                    task_round += 1
                
            except Exception as e:
                error_msg = str(e)
                
                # Record error in debug
                self.debug_recorder.record_llm_call(
                    task_id=task_id,
                    task_name=task_name,
                    round_num=task_round,  # Keep as round_num for debug recorder compatibility
                    prompt=current_prompt,
                    llm_output="",
                    tool_name="",
                    tool_params="",
                    tool_result="",
                    task_completed=False,
                    history_length=len(task_history),
                    error_msg=error_msg
                )
                
                error_record = {
                    "task_round": task_round,  # Changed from "round" to "task_round"
                    "prompt": current_prompt,
                    "error": error_msg,
                    "task_completed": False,
                    "timestamp": datetime.now().isoformat()
                }
                task_history.append(error_record)
                print(f"❌ Task round {task_round} execution error: {e}")
                task_round += 1
        
        return task_history
    

    
    def _record_debug_info(self, task_id: str, task_name: str, round_num: int, 
                          current_prompt: str, result: str, task_completed: bool, 
                          history_length: int):
        """Record debug information for the round"""
        if not self.debug_mode:
            return
        
        # Parse tool call information
        tool_name = ""
        tool_params = ""
        tool_result = ""
        llm_output = ""
        
        if "--- Tool Execution Result ---" in result or "--- 工具执行结果 ---" in result:
            separator = "--- Tool Execution Result ---" if "--- Tool Execution Result ---" in result else "--- 工具执行结果 ---"
            parts = result.split(separator)
            llm_output = parts[0].strip()
            tool_result = parts[1].strip() if len(parts) > 1 else ""
            
            # Try to parse tool name and parameters
            tool_call = self.executor.parse_tool_call(llm_output)
            if tool_call:
                tool_name = tool_call.get("name", "")
                import json
                tool_params = json.dumps(tool_call.get("arguments", {}), ensure_ascii=False)
        else:
            llm_output = result
            tool_result = "No tool call"
        
        # Record LLM call information
        self.debug_recorder.record_llm_call(
            task_id=task_id,
            task_name=task_name,
            round_num=round_num,
            prompt=current_prompt,
            llm_output=llm_output,
            tool_name=tool_name,
            tool_params=tool_params,
            tool_result=tool_result,
            task_completed=task_completed,
            history_length=history_length
        )
    
    def _display_round_summary(self, task_round: int, result: str):
        """Display task round execution summary"""
        if "--- Tool Execution Result ---" in result or "--- 工具执行结果 ---" in result:
            # Separate LLM response and tool execution results
            separator = "--- Tool Execution Result ---" if "--- Tool Execution Result ---" in result else "--- 工具执行结果 ---"
            parts = result.split(separator)
            llm_response = parts[0].strip()
            tool_results = parts[1].strip() if len(parts) > 1 else ""
            
            print(f"\n📝 Task Round {task_round} response summary:")
            # 使用配置的历史截断长度
            truncation_length = get_truncation_length()
            print(f"   LLM analysis: {llm_response[:truncation_length]}{'...' if len(llm_response) > truncation_length else ''}")
            
            if tool_results:
                print(f"   🛠️ Tool call executed, detailed results displayed above")
        else:
            # Pure text response - avoid repeating the content that was just streamed
            # Extract key completion information instead of repeating the full text
            if "TASK_COMPLETED:" in result:
                task_completed_match = re.search(r'TASK_COMPLETED:\s*(.+)', result)
                completion_msg = task_completed_match.group(1).strip() if task_completed_match else "Task marked as completed"
                print(f"\n📝 Task Round {task_round} response summary: ✅ {completion_msg}")
            else:
                # For non-completion responses, show a brief summary
                lines = result.strip().split('\n')
                first_meaningful_line = next((line.strip() for line in lines if line.strip() and not line.strip().startswith('🤖')), '')
                
                if first_meaningful_line:
                    # 使用配置的历史截断长度
                    truncation_length = get_truncation_length()
                    summary_text = first_meaningful_line[:truncation_length]
                    if len(first_meaningful_line) > truncation_length:
                        summary_text += "..."
                    print(f"\n📝 Task Round {task_round} response summary: {summary_text}")
                else:
                    print(f"\n📝 Task Round {task_round} response summary: [Response content displayed above]")
    
    def execute_all_tasks(self, csv_file: str) -> Dict[str, Any]:
        """
        Process todo.csv as a single comprehensive task
        
        Args:
            csv_file: CSV file path
            
        Returns:
            Execution report
        """
        # Validate CSV file
        if not self.task_loader.validate_csv_file(csv_file):
            return {"error": f"CSV file validation failed: {csv_file}"}
        
        # Read CSV content
        try:
            csv_content = self.task_loader.read_csv_content(csv_file)
        except Exception as e:
            return {"error": str(e)}
        
        print(f"\n🚀 Starting task execution")
        print(f"📄 Task file: {csv_file}")

        
        # Initialize execution report
        execution_report = {
            "start_time": datetime.now().isoformat(),
            "todo_file": csv_file,
            "subtask_loops": self.subtask_loops,
            "task_result": None,
            "task_history": [],
            "execution_summary": "",
            "workspace_dir": self.workspace_dir,
            "success": False
        }
        
        # Construct overall task description
        task_info = {
            "Task ID": "TODO_FILE",
            "Task Name": f"Process todo file: {os.path.basename(csv_file)}",
            "Task Description": f"""
Please process all tasks in the following todo.csv file. File content:

{csv_content}

Please execute all tasks listed in the file in a reasonable order. You can use various tools to complete these tasks, including but not limited to:
- Creating files and directories
- Writing code
- Running commands
- Searching content
- Editing files

Please ensure all tasks are properly handled. If some tasks have dependencies, please execute them in the correct order.
"""
        }
        
        try:
            print(f"\n🎯 Starting todo file processing")
            
            # Execute overall task
            task_result = self.execute_single_task(task_info, 0, 1, "")
            execution_report["task_result"] = task_result
            execution_report["task_history"] = task_result.get("history", [])
            execution_report["success"] = True
            
            print(f"✅ Todo file processing completed")
            
        except Exception as e:
            execution_report["error"] = str(e)
            execution_report["success"] = False
            print(f"❌ Todo file processing failed: {e}")
        
        # Complete execution
        execution_report["end_time"] = datetime.now().isoformat()
        execution_report["execution_summary"] = self.report_generator.generate_execution_summary(execution_report)
        
        # Save execution report
        print(f"\n💾 Saving execution report...")
        self.report_generator.save_execution_report(execution_report)
        
        # Output completion information
        self.print_completion_report(execution_report)
        
        return execution_report
    
    def print_completion_report(self, report: Dict[str, Any]):
        """
        Print completion report
        
        Args:
            report: Execution report
        """
        print("🎉 Task execution completed!")
        print(report["execution_summary"])
        
        # Check if it's the new todo file processing mode
        if "todo_file" in report:
            # New mode: process entire todo file
            if not report.get("success", False) and "error" in report:
                print(f"\n❌ Error occurred during execution:")
                print(f"  Error message: {report['error']}")
            
            task_history = report.get("task_history", [])
            if task_history:
                rounds_completed = len([h for h in task_history if isinstance(h, dict) and "task_round" in h])
                print(f"\n📊 Execution details:")
                print(f"  Completed rounds: {rounds_completed}")
                
                # Check if there are task completion flags
                completed_rounds = [h for h in task_history if isinstance(h, dict) and h.get("task_completed", False)]
                if completed_rounds:
                    print(f"  Early completion: Task completion detected in round {completed_rounds[0].get('task_round', '?')}")
                    
        else:
            # Legacy mode: compatibility support
            failed_tasks = report.get("failed_tasks", [])
            if failed_tasks:
                print("\n❌ Failed tasks:")
                for failed_task in failed_tasks:
                    print(f"  - Task {failed_task['task_id']}: {failed_task['task_name']}")
                    print(f"    Error: {failed_task['error']}")
        
        print(f"\n📁 Detailed log files saved in: {self.logs_dir}/")
    
    def generate_smart_summary(self, completed_tasks: List[Dict[str, Any]]) -> str:
        """
        Generate intelligent summary of completed tasks
        
        Args:
            completed_tasks: List of completed tasks
            
        Returns:
            Intelligent summary text
        """
        return self.summary_generator.generate_smart_summary(completed_tasks)