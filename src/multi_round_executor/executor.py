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
Main executor class for multi-round task execution
"""

import os
from datetime import datetime
from typing import List, Dict, Any, Optional

from tool_executor import ToolExecutor
from config_loader import get_model, get_truncation_length, get_summary_history, get_enable_round_sync, get_sync_round
from src.tools.print_system import print_current, print_current, print_current, print_debug
from src.tools.agent_context import get_current_agent_id
from src.tools.debug_system import track_operation, finish_operation
from .debug_recorder import DebugRecorder
from .task_checker import TaskChecker
import re

def extract_session_timestamp(logs_dir: str) -> str:
    """
    Extract session timestamp from logs directory path
    
    Args:
        logs_dir: Log directory path
        
    Returns:
        Session timestamp or None
    """
    session_timestamp = None
    if logs_dir:
        # Extract parent directory name
        parent_dir = os.path.dirname(logs_dir) if logs_dir != "logs" else logs_dir
        parent_name = os.path.basename(parent_dir)
        
        # Check if it matches output_YYYYMMDD_HHMMSS format
        match = re.search(r'(\d{8}_\d{6})', parent_name)
        if match:
            session_timestamp = match.group(1)
            
    return session_timestamp

class MultiRoundTaskExecutor:
    """Main executor for multi-round task execution"""
    
    def __init__(self, subtask_loops: int = 50, 
                 logs_dir: str = "logs", workspace_dir: str = None, 
                 debug_mode: bool = False, api_key: str = None, 
                 model: str = 'GPT-4.1', api_base: str = None, 
                 detailed_summary: bool = False,
                 interactive_mode: bool = False, streaming: bool = None,
                 MCP_config_file: str = None, prompts_folder: str = None,
                 user_id: str = None):
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
            user_id: User ID for MCP knowledge base tools
        """
        # Ensure subtask_loops is an integer to prevent type comparison errors
        self.subtask_loops = int(subtask_loops) 
        self.logs_dir = logs_dir
        self.workspace_dir = workspace_dir
        self.debug_mode = debug_mode
        self.api_key = api_key
        
        # Load model from config.txt if not provided
        self.model = model
        
        self.api_base = api_base
        self.detailed_summary = detailed_summary
        self.interactive_mode = interactive_mode
        self.streaming = streaming
        self.MCP_config_file = MCP_config_file
        self.prompts_folder = prompts_folder
        self.user_id = user_id
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
            prompts_folder=prompts_folder,
            user_id=user_id
        )
        
        # Initialize module components
        self.debug_recorder = DebugRecorder(debug_mode)
        self.task_checker = TaskChecker()
        # Track last seen sync epoch per agent to avoid file race conditions
        self._agent_last_sync_epoch: Dict[str, int] = {}
        
    def _get_status_file_path(self, agent_id: str) -> Optional[str]:
        """Compute status file path for agent if available via multi_agent_tools workspace"""
        try:
            if not agent_id:
                return None
            if hasattr(self.executor, 'multi_agent_tools') and self.executor.multi_agent_tools:
                workspace_root = self.executor.multi_agent_tools.workspace_root
                import os
                if workspace_root:
                    if os.path.basename(workspace_root) == 'workspace':
                        outdir = os.path.dirname(workspace_root)
                        return f"{outdir}/.agia_spawn_{agent_id}_status.json"
                    return f"{workspace_root}/.agia_spawn_{agent_id}_status.json"
        except Exception:
            return None
        return None

    def _is_agent_finished(self, data: Dict[str, Any]) -> bool:
        """Determine if an agent has finished based on simplified status field"""
        try:
            status_val = (data.get('status') or '').lower()
            return status_val in ('completed', 'terminated', 'failed', 'success', 'max_rounds_reached')
        except Exception:
            return False

    def _list_running_participants(self, exclude_agent_id: Optional[str] = None) -> list:
        """Return list of agent_ids that are started (current_loop>=1) and not finished"""
        participants = []
        finished_participants = []
        try:
            from tools.message_system import get_message_router
            import json, os
            router = get_message_router(self.executor.multi_agent_tools.workspace_root, cleanup_on_init=False) if hasattr(self.executor, 'multi_agent_tools') and self.executor.multi_agent_tools else None
            agents = router.get_all_agents() if router else []
            for aid in agents:
                if aid == 'manager':
                    continue
                if exclude_agent_id and aid == exclude_agent_id:
                    continue
                path = self._get_status_file_path(aid)
                if not path or not os.path.exists(path):
                    continue
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                except Exception:
                    continue

                # Check if agent has started
                try:
                    if int(data.get('current_loop', 0)) < 1:
                        continue
                except Exception:
                    continue

                if self._is_agent_finished(data):
                    finished_participants.append(aid)
                else:
                    participants.append(aid)

            # For synchronization purposes, include finished agents that might still need to sync
            # This prevents deadlock when one agent finishes before others in a sync window
            current_agent_id = get_current_agent_id()
            if current_agent_id and current_agent_id in finished_participants:
                # Current agent is finished, include other finished agents for sync counting
                all_participants = participants + finished_participants
                return all_participants
            else:
                # Current agent is still running, only count running participants
                return participants

        except Exception:
            return participants

    def _set_agent_wait_for_sync(self, agent_id: str, waiting: bool) -> None:
        """Mark agent status file with wait_for_sync flag"""
        path = self._get_status_file_path(agent_id)
        if not path:
            return
        try:
            import json, os
            if os.path.exists(path):
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            else:
                data = {}
            data['wait_for_sync'] = bool(waiting)
            data['wait_for_sync_updated_at'] = datetime.now().isoformat()
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
                f.flush()
                os.fsync(f.fileno()) if hasattr(f, 'fileno') else None
        except Exception as e:
            if self.debug_mode:
                print_debug(f"⚠️ Failed to write wait_for_sync to {path}: {e}")

    def _wait_for_global_sync_signal(self) -> None:
        """Block current thread until a global sync signal epoch increases; each agent consumes once per window.
        If no participants remain, return immediately to avoid deadlock."""
        try:
            import time, os, json
            # Determine base directory to place sync files: use workspace root or current dir
            base_dir = None
            if hasattr(self.executor, 'multi_agent_tools') and self.executor.multi_agent_tools:
                base_dir = self.executor.multi_agent_tools.workspace_root
                if base_dir and os.path.basename(base_dir) == 'workspace':
                    base_dir = os.path.dirname(base_dir)
            if not base_dir:
                base_dir = os.getcwd()
            signal_file = os.path.join(base_dir, '.agia_round_sync.signal')
            # read current epoch
            def read_epoch() -> int:
                try:
                    with open(signal_file, 'r', encoding='utf-8') as f:
                        txt = f.read().strip()
                        return int(txt) if txt.isdigit() else 0
                except Exception:
                    return 0
            current_agent_id = get_current_agent_id()
            last_seen = self._agent_last_sync_epoch.get(current_agent_id or 'manager', 0)

            # Check if current agent is finished
            current_agent_finished = False
            if current_agent_id:
                path = self._get_status_file_path(current_agent_id)
                if path and os.path.exists(path):
                    try:
                        with open(path, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                        current_agent_finished = self._is_agent_finished(data)
                    except Exception:
                        pass

            # wait until epoch increases
            wait_start_time = time.time()
            max_wait_time = 300  # 5 minutes max wait to prevent infinite waiting

            while True:
                # Check for timeout
                if time.time() - wait_start_time > max_wait_time:
                    if self.debug_mode:
                        print_debug(f"⚠️ Sync signal wait timeout for agent {current_agent_id}")
                    break

                # For finished agents, wait for signal but don't require other participants
                if current_agent_finished:
                    epoch = read_epoch()
                    if epoch > last_seen:
                        self._agent_last_sync_epoch[current_agent_id or 'manager'] = epoch
                        break
                else:
                    # For running agents, check if other participants exist
                    others = self._list_running_participants(exclude_agent_id=current_agent_id)
                    if not others:
                        # No other running participants, can proceed
                        break
                    epoch = read_epoch()
                    if epoch > last_seen:
                        self._agent_last_sync_epoch[current_agent_id or 'manager'] = epoch
                        break

                time.sleep(0.3)
        except Exception as e:
            if self.debug_mode:
                print_debug(f"⚠️ Error waiting for sync signal: {e}")
    
    
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
        task_id = task.get('Task ID', task.get('Task Number', '')) or ''
        task_name = task.get('Task Name', task.get('Task Name', '')) or ''
        task_desc = task.get('Task Description', task.get('Task Detailed Description', '')) or ''
        

        
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
            print_current(f"✅ Task completed successfully")
            status = "completed"
        else:
            print_current(f"⚠️ Task reached maximum rounds without explicit completion")
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
        # Sync barrier config
        enable_round_sync = get_enable_round_sync()
        sync_step = get_sync_round() if enable_round_sync else None
        
        current_agent_id = get_current_agent_id()
        
        # 🔧 New: Check if Round Scheduler is Enabled
        round_scheduler = None
        try:
            from tools.priority_scheduler import get_priority_scheduler
            round_scheduler = get_priority_scheduler()
            if not hasattr(round_scheduler, 'request_next_round'):
                round_scheduler = None
        except:
            round_scheduler = None
        # If Round Synchronization Barrier is Enabled
        try:
            if get_enable_round_sync():
                round_scheduler = None

        except Exception:
            pass
        
        while task_round <= max_rounds and not task_completed:
            print_debug(f"\n🔄 Current round {task_round} / total rounds {max_rounds}")

            # Barrier check before executing this round
            # Changed to: When There is at Least One Started and Unfinished Non-manager Agent
            is_manager = (current_agent_id == 'manager' or not current_agent_id)
            multi_agent_active = False
            current_agent_finished = False

            # Check if current agent is finished
            if current_agent_id:
                path = self._get_status_file_path(current_agent_id)
                if path and os.path.exists(path):
                    try:
                        import json
                        with open(path, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                        current_agent_finished = self._is_agent_finished(data)
                    except Exception:
                        pass

            try:
                participants = self._list_running_participants()
                multi_agent_active = bool(participants)
            except Exception:
                multi_agent_active = False

            barrier_applicable = (
                bool(enable_round_sync)
                and bool(sync_step)
                and not current_agent_finished  # Don't apply barrier to finished agents
                and (
                    (not is_manager)  # Regular Spawned Agent Always Follows
                    or (is_manager and multi_agent_active)  # Manager Also Follows When Other Agents Exist
                )
            )
            if barrier_applicable:
                try:
                    # Determine if this agent should wait (when next round would exceed its sync window)
                    # An agent can run rounds: 1..sync_step, then wait; (sync_step+1)..(2*sync_step), then wait; etc.
                    # We block when we are at the first round of a new window: rounds (sync_step+1), (2*sync_step+1), ...
                    need_sync = ((task_round - 1) % sync_step == 0) and (task_round > 1)
                    if need_sync:
                        # Update Status File to Indicate Waiting for Sync (Only for Non-manager)
                        if (
                            current_agent_id
                            and current_agent_id != 'manager'
                            and hasattr(self.executor, 'multi_agent_tools')
                            and self.executor.multi_agent_tools
                        ):
                            try:
                                # Reuse status file update util to set wait flag
                                self._set_agent_wait_for_sync(current_agent_id, True)
                                print_current(current_agent_id, f"⏸️ Waiting for sync barrier before round {task_round}")
                            except Exception as e:
                                print_current(current_agent_id, f"⚠️ Failed to mark wait_for_sync: {e}")
                        elif is_manager:
                            # Manager Only Outputs Prompt When Waiting
                            print_current(f"⏸️ Waiting for sync barrier before round {task_round}")
                        # Block until sync signal epoch increases (barrier release)
                        self._wait_for_global_sync_signal()
                        # Clear wait flag after sync
                        if (
                            current_agent_id
                            and current_agent_id != 'manager'
                            and hasattr(self.executor, 'multi_agent_tools')
                            and self.executor.multi_agent_tools
                        ):
                            try:
                                self._set_agent_wait_for_sync(current_agent_id, False)
                            except Exception:
                                pass
                except Exception as e:
                    print_debug(f"⚠️ Sync barrier check error: {e}")
            
            # Interactive mode confirmation before each round
            if self.interactive_mode and task_round > 1:  # Skip confirmation for first round
                try:
                    response = input(f"\n🤖 Continue to round {task_round} for task: {task_name}? (Y/n): ").strip().lower()
                    if response and response not in ['y', 'yes', 'yes', '']:
                        print_current("🛑 Task execution stopped by user")
                        # Mark as user interrupted
                        task_history.append({
                            "round": task_round,
                            "prompt": "User cancelled execution",
                            "result": "Task cancelled by user",
                            "user_interrupted": True,
                            "timestamp": datetime.now().isoformat()
                        })
                        return task_history
                except (KeyboardInterrupt, EOFError):
                    print_current("\n🛑 Task execution stopped by user")
                    task_history.append({
                        "round": task_round,
                        "prompt": "User cancelled execution",
                        "result": "Task cancelled by user (KeyboardInterrupt)",
                        "user_interrupted": True,
                        "timestamp": datetime.now().isoformat()
                    })
                    return task_history
            
            # Use base prompt directly - round info will be handled by _build_new_user_message
            current_prompt = base_prompt
            
            try:
                
                # Prepare history for LLM - include error records so model can learn from mistakes
                # 🔧 优化：只包含有结果的记录，避免重复的提示词
                history_for_llm = [record for record in task_history 
                                 if "result" in record or "error" in record]
                
                # Check if we need to summarize history to keep it manageable
                # First, determine which records would be summarized (excluding last 2 rounds)
                records_to_summarize = history_for_llm[:-2] if len(history_for_llm) > 2 else []
                recent_records = history_for_llm[-2:] if len(history_for_llm) > 2 else []
                
                # Calculate the length of content that would actually be summarized
                records_to_summarize_length = sum(len(str(record.get("result", ""))) 
                                           for record in records_to_summarize)
                recent_records_length = sum(len(str(record.get("result", ""))) 
                                         for record in recent_records)
                total_history_length = records_to_summarize_length + recent_records_length
                
                # Only summarize if the content to be summarized is actually substantial
                if hasattr(self.executor, 'summary_history') and self.executor.summary_history and \
                   hasattr(self.executor, 'summary_trigger_length') and \
                   records_to_summarize_length > self.executor.summary_trigger_length and \
                   len(records_to_summarize) > 0:  # Only summarize if we have records to summarize
                    
                    print_current(f"📊 Content to summarize ({records_to_summarize_length} chars) exceeds trigger ({self.executor.summary_trigger_length} chars), attempting to summarize...")
                    
                    # Check if we can use executor's summarization capability
                    if hasattr(self.executor, 'conversation_summarizer') and self.executor.conversation_summarizer:
                        try:
                            # Convert to conversation format, excluding recent records to avoid overlap
                            conversation_records = []
                            
                            # Check if we have enough records to summarize
                            if not records_to_summarize:
                                print_current(f"📋 Not enough history to summarize (only {len(history_for_llm)} records), using recent records only")
                                # Keep recent history only as fallback
                                history_for_llm = history_for_llm[-3:] if len(history_for_llm) > 3 else history_for_llm
                                # Skip summarization and proceed with current history_for_llm
                            else:
                                # Convert records to conversation format
                                for record in records_to_summarize:
                                    # 🔧 优化：使用简化的用户消息，避免重复提示词
                                    user_content = f"Round {record.get('task_round', 'N/A')} execution"
                                    conversation_records.append({
                                        "role": "user",
                                        "content": user_content
                                    })
                                    conversation_records.append({
                                        "role": "assistant", 
                                        "content": record["result"]
                                    })
                                
                                # Generate summary (excluding recent records)
                                latest_result = recent_records[-1]["result"] if recent_records else ""
                                history_summary = self.executor.conversation_summarizer.generate_conversation_history_summary(
                                    conversation_records, 
                                    latest_result
                                )
                                
                                # Validate summary content
                                if not history_summary or len(history_summary.strip()) < 10:
                                    print_current("⚠️ Generated summary is too short or empty, using basic fallback")
                                    # Use basic conversation summary as fallback
                                    history_summary = self.executor.conversation_summarizer._generate_basic_conversation_summary(
                                        conversation_records, latest_result
                                    )
                                
                                # Replace history with summary record
                                summary_record = {
                                    "task_round": "summary",
                                    "result": f"## Earlier Conversation Summary\n\n{history_summary}",
                                    "task_completed": False,
                                    "timestamp": datetime.now().isoformat(),
                                    "is_summary": True  # Mark as summary record
                                }
                                
                                # Combine summary with recent records (no overlap now)
                                history_for_llm = [summary_record] + recent_records
                                
                                # Update the main task_history to prevent future growth
                                # Keep non-LLM records (system messages, etc.) and replace LLM history
                                non_llm_records = [record for record in task_history 
                                                 if not ("result" in record) or record.get("error")]
                                task_history = non_llm_records + history_for_llm
                                
                                # Calculate actual new length after replacement
                                summary_length = len(history_summary) if history_summary else 0
                                summary_record_length = len(summary_record["result"])
                                recent_records_length = sum(len(str(r.get("result", ""))) for r in recent_records)
                                new_total_length = summary_record_length + recent_records_length
                                
                                if history_summary:
                                    print_current(f"✅ Summary completed:")
                                    print_current(f"   - Summary part: {records_to_summarize_length} → {summary_length} chars ({(1 - summary_length/records_to_summarize_length)*100:.1f}% reduction)")
                                    print_current(f"   - Recent records (last 2 rounds): {len(recent_records)} records, {recent_records_length} chars")
                                    print_current(f"   - Total: {total_history_length} → {new_total_length} chars ({(1 - new_total_length/total_history_length)*100:.1f}% reduction)")
                                    
                                    # Print the actual summary content to terminal
                                    print_current("📋 Generated Summary Content:")
                                    print_current("=" * 80)
                                    print_current(history_summary)
                                    print_current("=" * 80)
                                else:
                                    print_current(f"⚠️ Summary failed, using recent records only:")
                                    print_current(f"   - Summary part: {records_to_summarize_length} chars (failed to compress)")
                                    print_current(f"   - Recent records (last 2 rounds): {len(recent_records)} records, {recent_records_length} chars")
                                    print_current(f"   - Total: {total_history_length} → {recent_records_length} chars ({(1 - recent_records_length/total_history_length)*100:.1f}% reduction)")
                            
                        except Exception as e:
                            print_current(f"⚠️ History summarization failed: {e}")
                            # Keep recent history only as fallback
                            history_for_llm = history_for_llm[-3:] if len(history_for_llm) > 3 else history_for_llm
                            print_current(f"📋 Using recent history subset: {len(history_for_llm)} records")
                else:
                    # When summary_history=False, use simple compression instead
                    # Apply the same logic as AI summarization: only compress older records, keep recent 2 rounds intact
                    if hasattr(self.executor, 'simple_compressor') and self.executor.simple_compressor and \
                       len(history_for_llm) > 2:
                        try:
                            # Split history same way as AI summarization: compress older records, keep recent 2 rounds
                            records_to_compress = history_for_llm[:-2] if len(history_for_llm) > 2 else []
                            recent_records_to_keep = history_for_llm[-2:] if len(history_for_llm) > 2 else history_for_llm
                            
                            if records_to_compress:
                                #print_current(f"🗜️ Using simple compressor: compressing {len(records_to_compress)} older records, keeping {len(recent_records_to_keep)} recent records intact")
                                compressed_older_records = self.executor.simple_compressor.compress_history(records_to_compress)
                                # Combine compressed older records with uncompressed recent records
                                history_for_llm = compressed_older_records + recent_records_to_keep
                            # If no older records to compress, keep all records as-is
                        except Exception as e:
                            print_debug(f"⚠️ Simple history compression failed: {e}")
                
                # 🔧 Ensure correct agent_id is set in agent context before executing subtask
                current_agent_id = get_current_agent_id()
                
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
                    print_debug(f"🔄 Received optimized history from single-round executor: {len(optimized_history)} records")
                    
                    # Update main task_history with optimized version
                    # Keep non-LLM records (system messages) and replace LLM history
                    non_llm_records = [record for record in task_history 
                                     if not ("result" in record) or record.get("error")]
                    task_history.clear()
                    task_history.extend(non_llm_records + optimized_history)
                    print_debug(f"✅ Main task history updated with optimized records")
                
                # Check if user interrupted execution
                if result.startswith("USER_INTERRUPTED:"):
                    print_current(f"🛑 User interrupted execution: {result}")
                    # Record the interruption
                    round_record = {
                        "task_round": task_round, 
                        "result": result,
                        "task_completed": False,
                        "user_interrupted": True,
                        "timestamp": datetime.now().isoformat()
                    }
                    task_history.append(round_record)
                    print_current(f"🛑 Task execution stopped by user at task round {task_round}")
                    break
                
                # 🔧 Check if terminate signal is received (through proper message system)
                # Note: Terminate signals should come from _check_terminate_messages() in tool_executor
                # not from searching LLM response content directly
                if result and isinstance(result, str) and result.startswith("AGENT_TERMINATED:"):
                    # Only process if it's a proper terminate signal format from message system
                    # Record the termination
                    round_record = {
                        "task_round": task_round,
                        "result": result,
                        "task_completed": True,  # Mark as completed to avoid showing as failed
                        "agent_terminated": True,
                        "timestamp": datetime.now().isoformat()
                    }
                    task_history.append(round_record)
                    print_current(f"🛑 Task execution terminated at task round {task_round}")
                    task_completed = True  # Set task completion flag to ensure loop exit
                    break
                # Check task completion
                task_completed = self.task_checker.check_task_completion(result)
                
                # Record debug information
                self._record_debug_info(task_id, task_name, task_round, current_prompt, result, task_completed, len(task_history))
                
                # 🔧 优化：只在第一轮记录完整提示词，后续轮次只记录轮次信息和结果
                if task_round == 1:
                    # 第一轮：记录完整的提示词和结果
                    round_record = {
                        "task_round": task_round, 
                        "prompt": current_prompt,  # 只在第一轮记录完整提示词
                        "result": result,
                        "task_completed": task_completed,
                        "timestamp": datetime.now().isoformat()
                    }
                else:
                    # 后续轮次：只记录轮次信息和结果，不重复提示词
                    round_record = {
                        "task_round": task_round, 
                        "result": result,
                        "task_completed": task_completed,
                        "timestamp": datetime.now().isoformat()
                    }
                task_history.append(round_record)
                

                
                print_debug(f"✅ Task round {task_round} execution completed")
                
                # 🔧 New: Update current loop information in agent status file
                if current_agent_id and hasattr(self.executor, 'multi_agent_tools') and self.executor.multi_agent_tools:
                    try:
                        update_result = self.executor.multi_agent_tools.update_agent_current_loop(
                            agent_id=current_agent_id,
                            current_loop=task_round
                        )
                        if update_result.get("status") == "success":
                            print_current(current_agent_id, f"📝 Status file updated: current_loop = {task_round}")
                        else:
                            print_current(current_agent_id, f"⚠️ Status update failed: {update_result.get('message', 'Unknown error')}")
                    except Exception as e:
                        print_current(current_agent_id, f"⚠️ Status update error: {e}")
                
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
                    # 🔧 New: Round-level Fairness Scheduling
                    if round_scheduler and current_agent_id:
                        # Request Permission to Execute Next Round
                        print_current(current_agent_id, f"🎫 Requesting permission for round {task_round + 1}/{max_rounds}")
                        
                        # Call Round Scheduler
                        permission_granted = round_scheduler.request_next_round(
                            agent_id=current_agent_id,
                            current_round=task_round,
                            max_rounds=max_rounds,
                            wait_timeout=60.0  # Wait 60 Seconds
                        )
                        
                        if permission_granted:
                            task_round += 1
                            print_current(current_agent_id, f"✅ Permission granted, proceeding to round {task_round}")
                        else:
                            print_current(current_agent_id, f"⏰ Round permission timeout, stopping execution")
                            # Add Timeout Record
                            timeout_record = {
                                "task_round": task_round,
                                "message": "Round scheduling timeout - execution stopped for fairness",
                                "timestamp": datetime.now().isoformat(),
                                "scheduler_timeout": True
                            }
                            task_history.append(timeout_record)
                            break
                    else:
                        # Traditional Mode: Directly Increment Round
                        task_round += 1
                
            except Exception as e:
                error_msg = str(e)
                error_str = error_msg.lower()

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
                    "result": f"❌ Execution Error: {error_msg}",  # Put error in result field so LLM can see it
                    "error": error_msg,  # Keep error field for debugging
                    "task_completed": False,
                    "timestamp": datetime.now().isoformat()
                }
                task_history.append(error_record)
                print_current(f"❌ Task round {task_round} execution error: {e}")
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
 

    
    def cleanup(self):
        """Clean up all resources and threads"""
        try:

            # Clean up ToolExecutor
            if hasattr(self, 'executor') and self.executor:
                self.executor.cleanup()
            
        except Exception as e:
            print_debug(f"⚠️ Error during MultiRoundTaskExecutor cleanup: {e}")