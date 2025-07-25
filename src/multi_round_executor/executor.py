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
from tools.print_system import print_manager, print_current, print_agent
from tools.debug_system import track_operation, finish_operation
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
                 interactive_mode: bool = False, streaming: bool = None,
                 MCP_config_file: str = None, prompts_folder: str = None):
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
            streaming: Whether to use streaming output (None to use config.txt)
            MCP_config_file: Custom MCP configuration file path (optional, defaults to 'config/mcp_servers.json')
            prompts_folder: Custom prompts folder path (optional, defaults to 'prompts')
        """
        # Ensure subtask_loops is an integer to prevent type comparison errors
        self.subtask_loops = int(subtask_loops) if subtask_loops is not None else DEFAULT_SUBTASK_LOOPS
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
        self.streaming = streaming
        self.MCP_config_file = MCP_config_file
        self.prompts_folder = prompts_folder
        self.task_summaries = []  # Store summaries of all tasks
        
        # Extract session timestamp
        session_timestamp = extract_session_timestamp(logs_dir)
        
        # Ensure directories exist
        os.makedirs(logs_dir, exist_ok=True)
        if workspace_dir:
            os.makedirs(workspace_dir, exist_ok=True)
        
        # Initialize tool executor
        self.executor = ToolExecutor(
            api_key=api_key,
            model=model, 
            api_base=api_base,
            workspace_dir=workspace_dir,
            debug_mode=debug_mode,
            logs_dir=logs_dir,
            session_timestamp=session_timestamp,
            interactive_mode=interactive_mode,
            streaming=streaming,
            MCP_config_file=MCP_config_file,
            prompts_folder=prompts_folder
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
        # print_current(f"\n📋 Task executor configuration:")  # Commented out to reduce terminal noise
        # print_current(f"   Subtask execution rounds: {self.subtask_loops}")  # Commented out to reduce terminal noise
        # print_current(f"   Log directory: {self.logs_dir}")  # Commented out to reduce terminal noise
        # if self.workspace_dir:
        #     print_current(f"   Workspace directory: {self.workspace_dir}")  # Commented out to reduce terminal noise
        # if self.debug_mode:
        #     print_current(f"   DEBUG mode: Enabled")  # Commented out to reduce terminal noise
        # if self.detailed_summary:
        #     print_current(f"   Detailed summary: Enabled (retain more technical information)")  # Commented out to reduce terminal noise
        # else:
        #     print_current(f"   Detailed summary: Disabled (use simplified summary)")  # Commented out to reduce terminal noise
        
        # Print history summarization status
        # summary_history_enabled = get_summary_history()
        # print_current(f"   History Summarization: {'✅ Enabled' if summary_history_enabled else '❌ Disabled'}")  # Commented out to reduce terminal noise
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
        task_id = task.get('Task ID', task.get('任务编号', '')) or ''
        task_name = task.get('Task Name', task.get('任务名称', '')) or ''
        task_desc = task.get('Task Description', task.get('任务详细描述', '')) or ''
        
        # 🔧 Set agent_id at the beginning of task execution
        from tools.print_system import get_agent_id, set_agent_id
        current_agent_id = get_agent_id()
        if not current_agent_id and task_id and 'agent' in task_id.lower():
            # If agent_id not set but task_id contains agent, set it
            if task_id.startswith('agent_'):
                set_agent_id(task_id)
                print_current(f"🏷️ Set agent ID from task_id: {task_id}")
            else:
                import re
                agent_match = re.search(r'agent[_\-]?\d+', task_id, re.IGNORECASE)
                if agent_match:
                    set_agent_id(agent_match.group())
                    print_current(f"🏷️ Set agent ID from task_id pattern: {agent_match.group()}")
        
        track_operation(f"Execute task: {task_name} ({task_id})")


        
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
            print_current(f"📚 Loaded prerequisite task intelligent summary")
        
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
            print_current(f"📚 Loaded {len(self.task_summaries)} prerequisite task summaries (basic mode)")
        
        # Execute multi-round task
        task_history = self._execute_task_rounds(task_id, task_name, task_desc, task_history)
        
        # Check if user interrupted execution
        user_interrupted = any(record.get("user_interrupted", False) for record in task_history)
        if user_interrupted:
            print_current(f"🛑 Task {task_id} execution stopped by user")
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
            print_current(f"✅ Task {task_id} execution completed successfully")
            status = "completed"
        else:
            print_current(f"⚠️ Task {task_id} reached maximum rounds without explicit completion")
            status = "max_rounds_reached"
        
        finish_operation(f"Execute task: {task_name} ({task_id})")
        
        # 🔧 New: Get the last executed round information from task history
        last_round = 0
        for record in reversed(task_history):
            if isinstance(record, dict) and "task_round" in record:
                last_round = record["task_round"]
                break
        
        return {
            "task_id": task_id,
            "task_name": task_name,
            "task_description": task_desc,
            "history": task_history,
            "status": status,
            "current_loop": last_round  # Add current loop information
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
        
        # 🔧 新增：获取当前智能体ID用于轮次调度
        from tools.print_system import get_agent_id
        current_agent_id = get_agent_id()
        
        # 🔧 新增：检查是否启用了轮次调度器
        round_scheduler = None
        try:
            from tools.priority_scheduler import get_priority_scheduler
            round_scheduler = get_priority_scheduler()
            if hasattr(round_scheduler, 'request_next_round'):
                print_current(f"🎮 Using round-level fairness scheduler for {current_agent_id or 'manager'}")
            else:
                round_scheduler = None
        except:
            round_scheduler = None
        
        while task_round <= max_rounds and not task_completed:
            #print_current(f"\n🔄 --- Task Round {task_round}/{max_rounds} execution ---")
            
            # Use base prompt directly - round info will be handled by _build_new_user_message
            current_prompt = base_prompt
            
            try:
                
                # Prepare history for LLM - include error records so model can learn from mistakes
                history_for_llm = [record for record in task_history 
                                 if "prompt" in record and ("result" in record or "error" in record)]
                
                # Check if we need to summarize history to keep it manageable
                total_history_length = sum(len(str(record.get("prompt", ""))) + len(str(record.get("result", ""))) 
                                         for record in history_for_llm)
                
                # If history is too long, try to summarize it before passing to execute_subtask
                if hasattr(self.executor, 'summary_history') and self.executor.summary_history and \
                   hasattr(self.executor, 'summary_trigger_length') and \
                   total_history_length > self.executor.summary_trigger_length and \
                   len(history_for_llm) > 1:  # Only summarize if we have multiple records
                    
                    print_current(f"📊 History length ({total_history_length} chars) exceeds trigger, attempting to summarize...")
                    
                    # Check if we can use executor's summarization capability
                    if hasattr(self.executor, 'conversation_summarizer') and self.executor.conversation_summarizer:
                        try:
                            # Convert to conversation format
                            conversation_records = []
                            for record in history_for_llm:
                                conversation_records.append({
                                    "role": "user",
                                    "content": record["prompt"]
                                })
                                conversation_records.append({
                                    "role": "assistant", 
                                    "content": record["result"]
                                })
                            
                            # Generate summary
                            latest_result = history_for_llm[-1]["result"] if history_for_llm else ""
                            history_summary = self.executor.conversation_summarizer.generate_conversation_history_summary(
                                conversation_records, 
                                latest_result
                            )
                            
                            # Replace history with summary record
                            summary_record = {
                                "prompt": "Previous task execution summary",
                                "result": f"## Task History Summary\n\n{history_summary}",
                                "task_completed": False,
                                "timestamp": datetime.now().isoformat(),
                                "is_summary": True  # Mark as summary record
                            }
                            
                            # Keep only the summary and recent records (last 2 rounds)
                            recent_records = history_for_llm[-2:] if len(history_for_llm) > 2 else []
                            history_for_llm = [summary_record] + recent_records
                            
                            # Update the main task_history to prevent future growth
                            # Keep non-LLM records (system messages, etc.) and replace LLM history
                            non_llm_records = [record for record in task_history 
                                             if not ("prompt" in record and "result" in record) or record.get("error")]
                            task_history = non_llm_records + history_for_llm
                            
                            print_current(f"✅ History summarized and replaced: {total_history_length} → {len(history_summary)} chars")
                            
                        except Exception as e:
                            print_current(f"⚠️ History summarization failed: {e}")
                            # Keep recent history only as fallback
                            history_for_llm = history_for_llm[-3:] if len(history_for_llm) > 3 else history_for_llm
                            print_current(f"📋 Using recent history subset: {len(history_for_llm)} records")
                
                # 🔧 Fix display label issue: ensure correct agent_id is set before executing subtask
                from tools.print_system import get_agent_id, set_agent_id
                current_agent_id = get_agent_id()
                if current_agent_id is None:
                    # Try to infer agent_id from task ID
                    if task_id and task_id.startswith('agent_'):
                        set_agent_id(task_id)
                        print_current(f"🏷️ Set agent ID: {task_id}")
                    elif task_id and 'agent' in task_id.lower():
                        # Handle other possible agent ID formats
                        import re
                        agent_match = re.search(r'agent[_\-]?\d+', task_id, re.IGNORECASE)
                        if agent_match:
                            set_agent_id(agent_match.group())
                            print_current(f"🏷️ Set agent ID: {agent_match.group()}")
                
                # Execute task - this will handle internal tool calling rounds
                # The tool executor's internal rounds are separate from task rounds
                result = self.executor.execute_subtask(
                    current_prompt, 
                    "prompts.txt",
                    task_history=history_for_llm,
                    execution_round=task_round
                )
                
                # Handle possible optimized history return
                optimized_history = None
                if isinstance(result, tuple):
                    result, optimized_history = result
                    print_current(f"🔄 Received optimized history from single-round executor: {len(optimized_history)} records")
                    
                    # Update main task_history with optimized version
                    # Keep non-LLM records (system messages) and replace LLM history
                    non_llm_records = [record for record in task_history 
                                     if not ("prompt" in record and ("result" in record or "error" in record))]
                    task_history.clear()
                    task_history.extend(non_llm_records + optimized_history)
                    print_current(f"✅ Main task history updated with optimized records")
                
                # Check if user interrupted execution
                if result.startswith("USER_INTERRUPTED:"):
                    print_current(f"🛑 User interrupted execution: {result}")
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
                    print_current(f"🛑 Task execution stopped by user at task round {task_round}")
                    break
                
                # 🔧 New: Check if terminate signal is received
                if "AGENT_TERMINATED:" in result:
                    # Record the termination
                    round_record = {
                        "task_round": task_round, 
                        "prompt": current_prompt,
                        "result": result,
                        "task_completed": True,  # 🔧 Mark as completed to avoid showing as failed
                        "agent_terminated": True,
                        "timestamp": datetime.now().isoformat()
                    }
                    task_history.append(round_record)
                    print_current(f"🛑 Task execution terminated at task round {task_round}")
                    task_completed = True  # 🔧 Set task completion flag to ensure loop exit
                    break
                # Check task completion
                task_completed = self.task_checker.check_task_completion(result)
                
                # Record debug information
                self._record_debug_info(task_id, task_name, task_round, current_prompt, result, task_completed, len(task_history))
                
                # Record round results - but be careful not to make history too long again
                round_record = {
                    "task_round": task_round, 
                    "prompt": current_prompt,
                    "result": result,
                    "task_completed": task_completed,
                    "timestamp": datetime.now().isoformat()
                }
                task_history.append(round_record)
                

                
                print_current(f"✅ Task round {task_round} execution completed")
                
                # 🔧 New: Update current loop information in agent status file
                if current_agent_id and hasattr(self.executor, 'multi_agent_tools') and self.executor.multi_agent_tools:
                    try:
                        update_result = self.executor.multi_agent_tools.update_agent_current_loop(
                            agent_id=current_agent_id,
                            current_loop=task_round
                        )
                        if update_result.get("status") == "success":
                            print_agent(current_agent_id, f"📝 Status file updated: current_loop = {task_round}")
                        else:
                            print_agent(current_agent_id, f"⚠️ Status update failed: {update_result.get('message', 'Unknown error')}")
                    except Exception as e:
                        print_agent(current_agent_id, f"⚠️ Status update error: {e}")
                
                # 🔧 Remove synchronous incremental update call, now handled automatically by background thread
                # Commented out original synchronous update code:
                # try:
                #     self.executor.tools.perform_incremental_update()
                # except Exception as e:
                #     print_current(f"⚠️ Codebase incremental update failed: {e}")
                
                
                # Check if task is completed
                if task_completed:
                    break
                else:
                    # 🔧 新增：轮次级别的公平性调度
                    if round_scheduler and current_agent_id:
                        # 请求执行下一轮的权限
                        print_agent(current_agent_id, f"🎫 Requesting permission for round {task_round + 1}/{max_rounds}")
                        
                        # 调用轮次调度器
                        permission_granted = round_scheduler.request_next_round(
                            agent_id=current_agent_id,
                            current_round=task_round,
                            max_rounds=max_rounds,
                            wait_timeout=60.0  # 等待60秒
                        )
                        
                        if permission_granted:
                            task_round += 1
                            print_agent(current_agent_id, f"✅ Permission granted, proceeding to round {task_round}")
                        else:
                            print_agent(current_agent_id, f"⏰ Round permission timeout, stopping execution")
                            # 添加超时记录
                            timeout_record = {
                                "task_round": task_round,
                                "message": "Round scheduling timeout - execution stopped for fairness",
                                "timestamp": datetime.now().isoformat(),
                                "scheduler_timeout": True
                            }
                            task_history.append(timeout_record)
                            break
                    else:
                        # 传统模式：直接递增轮次
                        task_round += 1
                
            except Exception as e:
                error_msg = str(e)
                error_str = error_msg.lower()
                
                # Check if this is a retryable error (like overloaded)
                retryable_errors = [
                    'overloaded', 'rate limit', 'too many requests', 
                    'service unavailable', 'timeout', 'temporary failure',
                    'server error', '429', '503', '502', '500'
                ]
                
                # Find which error keyword matched
                matched_error_keyword = None
                for error_keyword in retryable_errors:
                    if error_keyword in error_str:
                        matched_error_keyword = error_keyword
                        break
                
                is_retryable = matched_error_keyword is not None
                
                if is_retryable:
                    # Get or initialize retry count for this round
                    retry_key = f"round_{task_round}_retries"
                    if not hasattr(self, '_round_retries'):
                        self._round_retries = {}
                    
                    current_retries = self._round_retries.get(retry_key, 0)
                    max_retries = 3  # Maximum retry attempts for retryable errors
                    
                    if current_retries < max_retries:
                        # Retry current round
                        self._round_retries[retry_key] = current_retries + 1
                        retry_delay = min(2 ** current_retries, 10)  # Exponential backoff, max 10 seconds
                        
                        print_manager(f"⚠️ {matched_error_keyword.title() if matched_error_keyword else 'Unknown'} error detected in task round {task_round}: {error_msg}")
                        print_manager(f"💡 Consider switching to a different model or trying again later")
                        print_manager(f"🔄 You can change the model in config.txt and restart AGIBot")
                        print_manager(f"🔄 Retrying in {retry_delay} seconds... (attempt {current_retries + 1}/{max_retries})")
                        
                        # Wait before retry
                        import time
                        time.sleep(retry_delay)
                        
                        # Record retry attempt but don't advance to next round
                        retry_record = {
                            "task_round": task_round,
                            "prompt": current_prompt,
                            "result": f"🔄 Retry attempt {current_retries + 1}/{max_retries} after {matched_error_keyword} error: {error_msg}",
                            "error": error_msg,
                            "task_completed": False,
                            "is_retry": True,
                            "retry_attempt": current_retries + 1,
                            "timestamp": datetime.now().isoformat()
                        }
                        task_history.append(retry_record)
                        
                        # Record debug information for retry
                        self.debug_recorder.record_llm_call(
                            task_id=task_id,
                            task_name=task_name,
                            round_num=task_round,
                            prompt=current_prompt,
                            llm_output="",
                            tool_name="",
                            tool_params="",
                            tool_result="",
                            task_completed=False,
                            history_length=len(task_history),
                            error_msg=f"Retry {current_retries + 1}/{max_retries}: {error_msg}"
                        )
                        
                        # Continue to retry (don't increment task_round)
                        continue
                    else:
                        # Max retries exceeded
                        print_manager(f"❌ Maximum retries ({max_retries}) exceeded for task round {task_round}")
                        print_manager(f"💡 Moving to next round. Consider switching to a different model.")
                        
                        # Clean up retry counter for this round
                        if retry_key in self._round_retries:
                            del self._round_retries[retry_key]
                
                # Record error in debug (for non-retryable errors or max retries exceeded)
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
                    "result": f"❌ Execution Error: {error_msg}",  # Put error in result field so LLM can see it
                    "error": error_msg,  # Keep error field for debugging
                    "task_completed": False,
                    "timestamp": datetime.now().isoformat()
                }
                task_history.append(error_record)
                print_manager(f"❌ Task round {task_round} execution error: {e}")
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
        
        if "--- Tool Execution Result ---" in result or "--- Tool Execution Result ---" in result:
            separator = "--- Tool Execution Result ---"
            parts = result.split(separator)
            llm_output = parts[0].strip()
            tool_result = parts[1].strip() if len(parts) > 1 else ""
            
            # Try to parse tool name and parameters
            tool_calls = self.executor.parse_tool_calls(llm_output)
            tool_call = tool_calls[0] if tool_calls else None
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
    
    def execute_all_tasks(self, todo_file: str) -> Dict[str, Any]:
        """
        Process todo file (CSV or Markdown) as a single comprehensive task
        
        Args:
            todo_file: Todo file path (CSV or Markdown)
            
        Returns:
            Execution report
        """
        # Determine file type and validate
        file_ext = os.path.splitext(todo_file)[1].lower()
        
        if file_ext == '.csv':
            # Handle CSV file
            if not self.task_loader.validate_csv_file(todo_file):
                return {"error": f"CSV file validation failed: {todo_file}"}
            
            try:
                file_content = self.task_loader.read_csv_content(todo_file)
                file_type = "CSV"
            except Exception as e:
                return {"error": str(e)}
                
        elif file_ext == '.md':
            # Handle Markdown file
            if not os.path.exists(todo_file):
                return {"error": f"Markdown file not found: {todo_file}"}
            
            try:
                with open(todo_file, 'r', encoding='utf-8') as f:
                    file_content = f.read()
                file_type = "Markdown"
            except Exception as e:
                return {"error": f"Failed to read Markdown file: {e}"}
        else:
            return {"error": f"Unsupported file type: {file_ext}. Only .csv and .md files are supported."}
        
        print_manager(f"\n🚀 Starting task execution")
        print_manager(f"📄 Task file: {todo_file} ({file_type})")

        
        # Initialize execution report
        execution_report = {
            "start_time": datetime.now().isoformat(),
            "todo_file": todo_file,
            "file_type": file_type,
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
            "Task Name": f"Process todo file: {os.path.basename(todo_file)}",
            "Task Description": f"""
Please process all tasks in the following todo {file_type.lower()} file. File content:

{file_content}

Please execute all tasks listed in the file in a reasonable order. You can use various tools to complete these tasks, including but not limited to:
- Creating files and directories
- Writing code
- Running commands
- Searching content
- Editing files
- Updating task status using the todo_update tool

For Markdown todo files, you can use the todo_update tool to mark tasks as completed:
- todo_update(task_id=1, status="in_progress", description="Starting task")  
- todo_update(task_id=1, status="completed", description="Task finished")

Please ensure all tasks are properly handled. If some tasks have dependencies, please execute them in the correct order.
"""
        }
        
        try:
            print_manager(f"\n🎯 Starting todo file processing")
            
            # Execute overall task
            task_result = self.execute_single_task(task_info, 0, 1, "")
            execution_report["task_result"] = task_result
            execution_report["task_history"] = task_result.get("history", [])
            execution_report["success"] = True
            
            print_manager(f"✅ Todo file processing completed")
            
        except Exception as e:
            execution_report["error"] = str(e)
            execution_report["success"] = False
            print_manager(f"❌ Todo file processing failed: {e}")
        
        # Complete execution
        execution_report["end_time"] = datetime.now().isoformat()
        execution_report["execution_summary"] = self.report_generator.generate_execution_summary(execution_report)
        
        # Save execution report
        print_current(f"\n💾 Saving execution report...")
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
        print(report["execution_summary"])
        
        # Check if it's the new todo file processing mode
        if "todo_file" in report:
            # New mode: process entire todo file
            if not report.get("success", False) and "error" in report:
                print_current(f"\n❌ Error occurred during execution:")
                print_current(f"  Error message: {report['error']}")
            
            task_history = report.get("task_history", [])
            if task_history:
                rounds_completed = len([h for h in task_history if isinstance(h, dict) and "task_round" in h])
                print_current(f"\n📊 Execution details:")
                print_current(f"  Completed rounds: {rounds_completed}")
                
                # Check if there are task completion flags
                completed_rounds = [h for h in task_history if isinstance(h, dict) and h.get("task_completed", False)]
                if completed_rounds:
                    print_current(f"  Early completion: Task completion detected in round {completed_rounds[0].get('task_round', '?')}")
                    
        else:
            # Legacy mode: compatibility support
            failed_tasks = report.get("failed_tasks", [])
            if failed_tasks:
                print_current("\n❌ Failed tasks:")
                for failed_task in failed_tasks:
                    print_current(f"  - Task {failed_task['task_id']}: {failed_task['task_name']}")
                    print_current(f"    Error: {failed_task['error']}")
        
        print_current(f"\n📁 Detailed log files saved in: {self.logs_dir}/")
    
    def generate_smart_summary(self, completed_tasks: List[Dict[str, Any]]) -> str:
        """
        Generate intelligent summary of completed tasks
        
        Args:
            completed_tasks: List of completed tasks
            
        Returns:
            Intelligent summary text
        """
        return self.summary_generator.generate_smart_summary(completed_tasks)
    
    def cleanup(self):
        """Clean up all resources and threads"""
        try:

            # Clean up ToolExecutor
            if hasattr(self, 'executor') and self.executor:
                self.executor.cleanup()
            
        except Exception as e:
            print_current(f"⚠️ Error during MultiRoundTaskExecutor cleanup: {e}")