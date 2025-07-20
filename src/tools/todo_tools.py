#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Todo Tools - LLM tools for managing todo.md task status
"""

import os
import sys
import json
from datetime import datetime
from typing import List, Dict, Any, Optional

# Add parent directory to path to import todo_status_updater
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from todo_status_updater import TodoStatusUpdater, TaskStatus
except ImportError:
    # Fallback for when running from different contexts
    import importlib.util
    todo_updater_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'todo_status_updater.py')
    if os.path.exists(todo_updater_path):
        spec = importlib.util.spec_from_file_location("todo_status_updater", todo_updater_path)
        todo_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(todo_module)
        TodoStatusUpdater = todo_module.TodoStatusUpdater
        TaskStatus = todo_module.TaskStatus

class TodoTools:
    """Tools for managing todo.md file from LLM"""
    
    def __init__(self, workspace_dir: str = None, create_backup: bool = False):
        """
        Initialize todo tools
        
        Args:
            workspace_dir: Workspace directory containing todo.md
            create_backup: Whether to create backup files (default: False)
        """
        self.workspace_dir = workspace_dir or os.getcwd()
        self.todo_file = os.path.join(self.workspace_dir, "todo.md")
        self.updater = TodoStatusUpdater(self.todo_file, create_backup=create_backup)
    
    def update_task_status(self, task_id: int, status: str, description: str = None) -> Dict[str, Any]:
        """
        Update task status in todo.md file
        
        Args:
            task_id: Task ID to update (1, 2, 3, etc.)
            status: New status ('pending', 'in_progress', 'completed', 'blocked', 'skipped')
            description: Optional description of the status change
            
        Returns:
            Result dictionary with success status and message
        """
        try:
            # Validate status
            status_map = {
                'pending': TaskStatus.PENDING,
                'in_progress': TaskStatus.IN_PROGRESS,
                'inprogress': TaskStatus.IN_PROGRESS,
                'progress': TaskStatus.IN_PROGRESS,
                'completed': TaskStatus.COMPLETED,
                'complete': TaskStatus.COMPLETED,
                'done': TaskStatus.COMPLETED,
                'finished': TaskStatus.COMPLETED,
                'blocked': TaskStatus.BLOCKED,
                'block': TaskStatus.BLOCKED,
                'skipped': TaskStatus.SKIPPED,
                'skip': TaskStatus.SKIPPED
            }
            
            status_lower = status.lower().strip()
            if status_lower not in status_map:
                return {
                    "success": False,
                    "error": f"Invalid status '{status}'. Valid options: {list(status_map.keys())}"
                }
            
            task_status = status_map[status_lower]
            
            # Update task status
            success = self.updater.update_task_status(task_id, task_status, description)
            
            if success:
                return {
                    "success": True,
                    "message": f"Task {task_id} status updated to '{task_status.value}'",
                    "task_id": task_id,
                    "new_status": task_status.value,
                    "description": description
                }
            else:
                return {
                    "success": False,
                    "error": f"Failed to update task {task_id}"
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": f"Error updating task status: {str(e)}"
            }
    
    def mark_task_completed(self, task_id: int, completion_note: str = None) -> Dict[str, Any]:
        """
        Mark a task as completed
        
        Args:
            task_id: Task ID to mark as completed
            completion_note: Optional note about task completion
            
        Returns:
            Result dictionary
        """
        return self.update_task_status(task_id, 'completed', completion_note)
    
    def mark_task_in_progress(self, task_id: int, progress_note: str = None) -> Dict[str, Any]:
        """
        Mark a task as in progress
        
        Args:
            task_id: Task ID to mark as in progress
            progress_note: Optional note about current progress
            
        Returns:
            Result dictionary
        """
        return self.update_task_status(task_id, 'in_progress', progress_note)
    
    def mark_task_blocked(self, task_id: int, block_reason: str = None) -> Dict[str, Any]:
        """
        Mark a task as blocked
        
        Args:
            task_id: Task ID to mark as blocked
            block_reason: Reason why task is blocked
            
        Returns:
            Result dictionary
        """
        return self.update_task_status(task_id, 'blocked', block_reason)
    
    def get_task_progress(self) -> Dict[str, Any]:
        """
        Get current task progress summary
        
        Returns:
            Progress summary dictionary
        """
        try:
            summary = self.updater.get_task_summary()
            
            if "error" in summary:
                return {
                    "success": False,
                    "error": summary["error"]
                }
            
            # Format for LLM consumption
            result = {
                "success": True,
                "total_tasks": summary["total_tasks"],
                "completed": summary["completed"],
                "in_progress": summary["in_progress"],
                "pending": summary["pending"],
                "blocked": summary["blocked"],
                "skipped": summary["skipped"],
                "completion_percentage": summary["completion_percentage"],
                "tasks": summary["tasks"]
            }
            
            # Add progress visualization
            if summary["total_tasks"] > 0:
                completed_ratio = summary["completed"] / summary["total_tasks"]
                bar_length = 20
                filled_length = int(bar_length * completed_ratio)
                progress_bar = '▰' * filled_length + '▱' * (bar_length - filled_length)
                result["progress_bar"] = f"{progress_bar} {summary['completion_percentage']}%"
            
            return result
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Error getting task progress: {str(e)}"
            }
    
    def list_tasks(self) -> Dict[str, Any]:
        """
        List all tasks from todo.md
        
        Returns:
            List of tasks with their current status
        """
        try:
            content = self.updater.read_todo_file()
            if content is None:
                return {
                    "success": False,
                    "error": "Could not read todo.md file"
                }
            
            tasks = self.updater.parse_tasks(content)
            
            task_list = []
            for task in tasks:
                task_list.append({
                    "id": task['id'],
                    "name": task['name'],
                    "status": task['status'].value,
                    "status_icon": self.updater._status_to_icon(task['status'])
                })
            
            return {
                "success": True,
                "tasks": task_list,
                "total_count": len(task_list)
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Error listing tasks: {str(e)}"
            }
    
    def get_next_pending_task(self) -> Dict[str, Any]:
        """
        Get the next pending task that needs to be worked on
        
        Returns:
            Next pending task information
        """
        try:
            task_list = self.list_tasks()
            if not task_list["success"]:
                return task_list
            
            # Find first pending task
            for task in task_list["tasks"]:
                if task["status"] == "pending":
                    return {
                        "success": True,
                        "next_task": task,
                        "suggestion": f"Consider working on Task {task['id']}: {task['name']}"
                    }
            
            # No pending tasks found
            return {
                "success": True,
                "next_task": None,
                "message": "No pending tasks found. All tasks may be completed or in progress."
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Error finding next task: {str(e)}"
            }
    
    def bulk_update_status(self, updates: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Update multiple task statuses at once
        
        Args:
            updates: List of update dictionaries with 'task_id', 'status', and optional 'description'
            
        Returns:
            Bulk update results
        """
        results = []
        success_count = 0
        
        for update in updates:
            task_id = update.get('task_id')
            status = update.get('status')
            description = update.get('description')
            
            if not task_id or not status:
                results.append({
                    "task_id": task_id,
                    "success": False,
                    "error": "Missing task_id or status"
                })
                continue
            
            result = self.update_task_status(task_id, status, description)
            results.append(result)
            
            if result["success"]:
                success_count += 1
        
        return {
            "success": success_count > 0,
            "total_updates": len(updates),
            "successful_updates": success_count,
            "failed_updates": len(updates) - success_count,
            "results": results
        }

    def auto_suggest_next_action(self) -> Dict[str, Any]:
        """
        Analyze current progress and suggest next actions
        
        Returns:
            Suggested next actions based on current task status
        """
        try:
            progress = self.get_task_progress()
            if not progress["success"]:
                return progress
            
            suggestions = []
            
            # Check for blocked tasks that need attention
            if progress["blocked"] > 0:
                suggestions.append({
                    "priority": "high",
                    "action": "resolve_blocked_tasks",
                    "message": f"You have {progress['blocked']} blocked task(s) that need attention"
                })
            
            # Check for in-progress tasks
            if progress["in_progress"] > 0:
                suggestions.append({
                    "priority": "medium",
                    "action": "continue_progress",
                    "message": f"Continue working on {progress['in_progress']} task(s) currently in progress"
                })
            
            # Suggest starting next pending task
            if progress["pending"] > 0:
                next_task = self.get_next_pending_task()
                if next_task["success"] and next_task.get("next_task"):
                    suggestions.append({
                        "priority": "normal",
                        "action": "start_next_task",
                        "message": next_task["suggestion"]
                    })
            
            # All tasks completed
            if progress["completed"] == progress["total_tasks"] and progress["total_tasks"] > 0:
                suggestions.append({
                    "priority": "celebration",
                    "action": "all_completed",
                    "message": "🎉 Congratulations! All tasks have been completed!"
                })
            
            return {
                "success": True,
                "current_progress": progress,
                "suggestions": suggestions
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Error generating suggestions: {str(e)}"
            } 