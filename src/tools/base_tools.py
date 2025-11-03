#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from .print_system import print_system, print_current, print_error
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

import os
import subprocess
import threading
import time
import queue
from datetime import datetime
from typing import Dict, Any, Optional

from .code_repository_parser import CodeRepositoryParser

# Supported file extensions for code parsing
SUPPORTED_EXTENSIONS = [
    '.py', '.js', '.ts', '.tsx', '.jsx', '.java', '.cpp', '.c', '.h', '.hpp', 
    '.cs', '.php', '.rb', '.go', '.rs', '.swift', '.kt', '.scala', '.sh', '.bat', 
    '.ps1', '.sql', '.html', '.css', '.scss', '.less', '.xml', '.json', '.yaml', 
    '.yml', '.toml', '.cfg', '.ini', '.md', '.txt', '.dockerfile', '.makefile'
]


class BaseTools:
    def __init__(self, workspace_root: str = None, model: str = None):
        """Initialize tools with a workspace root directory."""
        self.workspace_root = workspace_root
        self.model = model  # Store model name for vision API detection
        self.last_edit = None  # Store last edit info (used for debugging and context)

        # Initialize code repository parser
        self.code_parser = None
        self._init_code_parser()
        
        # Initialize terminal tools
        self.terminal_tools = None
        self._init_terminal_tools()

        # Initialize sensor data collector
        self.sensor_collector = None
        self._init_sensor_collector()

    def _init_code_parser(self):
        """Initialize code repository parser with background update enabled"""
        try:
            # Only initialize code parser if we have a valid workspace_root
            # Don't create code index for project root directory
            if not self.workspace_root:
                self.code_parser = None
                return
                
            # Check if workspace_root looks like a valid output directory
            # (should contain or be a workspace directory)
            workspace_path = os.path.abspath(self.workspace_root)
            workspace_name = os.path.basename(workspace_path)
            
            # Only create code parser for workspace directories or directories containing workspace
            is_workspace_dir = workspace_name == "workspace"
            has_workspace_subdir = os.path.exists(os.path.join(workspace_path, "workspace"))
            
            # Check if it's a parent directory containing multiple output_* subdirectories
            # (This case doesn't need code parser, so we skip silently)
            is_parent_of_outputs = False
            if not (is_workspace_dir or has_workspace_subdir):
                # Check if this is a parent directory containing output_* subdirectories
                try:
                    if os.path.isdir(workspace_path):
                        subdirs = [d for d in os.listdir(workspace_path) 
                                 if os.path.isdir(os.path.join(workspace_path, d)) 
                                 and d.startswith('output_')]
                        if len(subdirs) > 0:
                            is_parent_of_outputs = True
                except (OSError, PermissionError):
                    pass
            
            if not (is_workspace_dir or has_workspace_subdir):
                # Only print warning if it's not a parent of output directories
                # (parent directories are expected and don't need code parser)
                if not is_parent_of_outputs:
                    print_current(f"⚠️ Workspace path '{workspace_path}' doesn't appear to be a valid workspace directory, skipping code parser initialization")
                self.code_parser = None
                return
            
            from .global_code_index_manager import get_global_code_index_manager
            
            # Use global code index manager
            manager = get_global_code_index_manager()
            
            self.code_parser = manager.get_parser(
                workspace_root=self.workspace_root,
                supported_extensions=SUPPORTED_EXTENSIONS
            )
            
            if self.code_parser is None:
                print_current("Failed to get code parser from global manager")
                
        except Exception as e:
            print_current(f"Failed to initialize code repository parser: {e}")
            self.code_parser = None

    def _get_code_index_path(self) -> str:
        """Get the path to the code index database (proxy method)"""
        if self.code_parser:
            workspace_root = self.workspace_root or os.getcwd()
            return self.code_parser._get_code_index_path(workspace_root)
        return ""

    def _rebuild_code_index(self):
        """Rebuild code index (proxy method)"""
        if self.code_parser:
            return self.code_parser._rebuild_code_index()
        return False

    def perform_incremental_update(self):
        """Perform incremental update (proxy method)"""
        if self.code_parser:
            return self.code_parser.perform_incremental_update()
        return False

    def _resolve_path(self, path: str) -> str:
        """Resolve a path to an absolute path, cleaning up any redundant workspace prefixes."""
        if os.path.isabs(path):
            return path
        
        if path.startswith('workspace/'):
            workspace_dir_name = os.path.basename(self.workspace_root)
            if workspace_dir_name in ['workspace', 'output']:
                path = path[10:]
                # print_current(f"⚠️  Path cleanup: removed redundant 'workspace/' prefix, using: {path}")
        
        resolved_path = os.path.join(self.workspace_root, path)
        # print_current(f"🔍 Path resolution: '{path}' -> '{resolved_path}'")
        return resolved_path

    def _init_terminal_tools(self):
        """Initialize terminal tools for user interaction"""
        try:
            from .terminal_tools import TerminalTools
            self.terminal_tools = TerminalTools(workspace_root=self.workspace_root)
        except Exception as e:
            print_error(f"❌ Failed to initialize terminal tools: {e}")
            self.terminal_tools = None

    def _init_sensor_collector(self):
        """Initialize sensor data collector"""
        try:
            from .sensor_tools import SensorDataCollector
            self.sensor_collector = SensorDataCollector(workspace_root=self.workspace_root, model=self.model)
        except Exception as e:
            print_current(f"❌ Failed to initialize sensor data collector: {e}")
            self.sensor_collector = None

    def get_sensor_data(self, type: int, source: str, para: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Acquire physical world information including images, videos, audio, and sensor data.
        
        Args:
            type: Data type (1=image, 2=video, 3=audio, 4=sensor)
            source: Source identifier (file path or device path)
            para: Parameters dictionary
            
        Returns:
            Dictionary containing sensor data acquisition results
        """
        if self.sensor_collector:
            return self.sensor_collector.get_sensor_data(type, source, para)
        else:
            return {
                'status': 'failed',
                'data': None,
                'dataformat': None,
                'error': 'Sensor data collector not initialized',
                'timestamp': None
            }

    def talk_to_user(self, query: str, timeout: int = 10) -> Dict[str, Any]:
        """
        Display a question to the user and wait for keyboard input with timeout.
        
        Args:
            query: The question to display to the user
            timeout: Maximum time to wait for user response (default: 10 seconds)
            
        Returns:
            Dict containing the user's response or timeout indication
        """
        if self.terminal_tools:
            return self.terminal_tools.talk_to_user(query, timeout)
        else:
            return {
                'status': 'failed',
                'query': query,
                'user_response': 'no user response',
                'timeout': timeout,
                'response_time': 'error',
                'error': 'Terminal tools not initialized'
            }

    def idle(self, message: str = None, reason: str = None) -> Dict[str, Any]:
        """
        Idle tool - represents doing nothing in this round, mainly used for multi-agent synchronization.
        
        Args:
            message: Optional message explaining why idling (default: None)
            reason: Optional reason for idling (default: None)
            
        Returns:
            Dict containing idle status and optional message
        """
        result = {
            'status': 'idle',
            'action': 'no_action_taken',
            'description': 'This round is idle - no operations performed'
        }
        
        if message:
            result['message'] = message
        
        if reason:
            result['reason'] = reason
        
        # Add timestamp for synchronization purposes
        import datetime
        result['timestamp'] = datetime.datetime.now().isoformat()
        
        print_current("💤 Idle - No action taken this round")
        if message:
            print_current(f"   Message: {message}")
        if reason:
            print_current(f"   Reason: {reason}")
        
        return result

    def todo_update(self, task_id: int, status: str = None, description: str = None, action: str = "update_status") -> Dict[str, Any]:
        """
        Update todo task status and return complete file content.
        
        Args:
            task_id: Task ID to update (1, 2, 3, etc.)
            status: New status ('pending', 'in_progress', 'completed', 'blocked', 'skipped')
            description: Optional description explaining the status change
            action: Action to perform ('update_status', 'get_progress', 'list_tasks', 'get_next_task')
            
        Returns:
            Dict containing operation result and complete todo.md file content
        """
        try:
            # Initialize todo tools with correct directory
            # todo.md is typically in the parent of workspace directory (output_xxx/todo.md, not output_xxx/workspace/todo.md)
            from .todo_tools import TodoTools
            
            # If workspace_root ends with 'workspace', use parent directory for todo.md
            todo_dir = self.workspace_root
            if self.workspace_root and os.path.basename(self.workspace_root) == 'workspace':
                todo_dir = os.path.dirname(self.workspace_root)
            
            todo_tools = TodoTools(todo_dir, create_backup=False)
            
            result = {
                'success': False,
                'action': action,
                'task_id': task_id,
                'todo_file_content': '',
                'message': '',
                'error': None
            }
            
            # Handle different actions
            if action == "update_status":
                if not status:
                    result['error'] = "Status is required for update_status action"
                    return result
                
                # Update task status
                update_result = todo_tools.update_task_status(task_id, status, description)
                
                if update_result.get('status') == 'success':
                    result['success'] = True
                    result['message'] = update_result['message']
                    result['new_status'] = update_result.get('new_status')
                    result['description'] = update_result.get('description')
                else:
                    result['error'] = update_result.get('error', 'Unknown error')
                    
            elif action == "get_progress":
                # Get current progress
                progress_result = todo_tools.get_task_progress()
                
                if progress_result.get('status') == 'success':
                    result['success'] = True
                    result['progress'] = progress_result
                    result['message'] = f"Progress: {progress_result['completion_percentage']}% complete"
                else:
                    result['error'] = progress_result.get('error', 'Failed to get progress')
                    
            elif action == "list_tasks":
                # List all tasks
                list_result = todo_tools.list_tasks()
                
                if list_result.get('status') == 'success':
                    result['success'] = True
                    result['tasks'] = list_result['tasks']
                    result['total_count'] = list_result['total_count']
                    result['message'] = f"Found {list_result['total_count']} tasks"
                else:
                    result['error'] = list_result.get('error', 'Failed to list tasks')
                    
            elif action == "get_next_task":
                # Get next pending task
                next_result = todo_tools.get_next_pending_task()
                
                if next_result.get('status') == 'success':
                    result['success'] = True
                    result['next_task'] = next_result.get('next_task')
                    result['message'] = next_result.get('suggestion', next_result.get('message', ''))
                else:
                    result['error'] = next_result.get('error', 'Failed to get next task')
            else:
                result['error'] = f"Invalid action: {action}"
                
            # Always try to read and return the complete todo.md file content
            try:
                # Use the same todo_dir logic for reading the file
                todo_file_path = os.path.join(todo_dir, "todo.md")
                if os.path.exists(todo_file_path):
                    with open(todo_file_path, 'r', encoding='utf-8') as f:
                        result['todo_file_content'] = f.read()
                    
                    # Add file info
                    result['todo_file_path'] = todo_file_path
                    result['file_exists'] = True
                else:
                    result['todo_file_content'] = "# Todo file not found\n\nNo todo.md file exists in the current workspace."
                    result['file_exists'] = False
                    
            except Exception as file_error:
                result['todo_file_content'] = f"# Error reading todo file\n\nError: {str(file_error)}"
                result['file_read_error'] = str(file_error)
            
            # Add timestamp
            from datetime import datetime
            result['timestamp'] = datetime.now().isoformat()
            
            # Print status update for user visibility
            if result['success']:
                if action == "update_status":
                    print_current(f"✅ Task {task_id} status updated to: {status}")
                    if description:
                        print_current(f"   📝 Note: {description}")
                elif action == "get_progress":
                    progress = result.get('progress', {})
                    print_current(f"📊 Progress: {progress.get('completion_percentage', 0)}% ({progress.get('completed', 0)}/{progress.get('total_tasks', 0)} tasks)")
                elif action == "list_tasks":
                    print_current(f"📋 Listed {result.get('total_count', 0)} tasks")
                elif action == "get_next_task":
                    next_task = result.get('next_task')
                    if next_task:
                        print_current(f"👆 Next task: #{next_task['id']} {next_task['name']}")
                    else:
                        print_current("🎉 No pending tasks found")
            else:
                print_current(f"❌ Todo update failed: {result.get('error', 'Unknown error')}")
            
            return result
            
        except Exception as e:
            from datetime import datetime
            error_result = {
                'success': False,
                'action': action,
                'task_id': task_id,
                'error': f"Todo update tool error: {str(e)}",
                'todo_file_content': '',
                'timestamp': datetime.now().isoformat()
            }
            
            # Try to read file content even if there's an error
            try:
                # Use the same todo_dir logic for error case
                todo_dir = self.workspace_root
                if self.workspace_root and os.path.basename(self.workspace_root) == 'workspace':
                    todo_dir = os.path.dirname(self.workspace_root)
                
                todo_file_path = os.path.join(todo_dir, "todo.md")
                if os.path.exists(todo_file_path):
                    with open(todo_file_path, 'r', encoding='utf-8') as f:
                        error_result['todo_file_content'] = f.read()
            except:
                pass
                
            print_current(f"❌ Todo update error: {str(e)}")
            return error_result

    def cleanup_todo_backups(self, workspace_dir: str = None) -> Dict[str, Any]:
        """
        Clean up todo.md.backup files in the specified directory and subdirectories.
        
        Args:
            workspace_dir: Directory to clean (default: current workspace_root)
            
        Returns:
            Dict containing cleanup results
        """
        try:
            import glob
            
            # Determine cleanup directory
            cleanup_dir = workspace_dir or self.workspace_root or os.getcwd()
            
            # If workspace_root ends with 'workspace', also check parent directory
            if self.workspace_root and os.path.basename(self.workspace_root) == 'workspace':
                parent_dir = os.path.dirname(self.workspace_root)
                search_dirs = [cleanup_dir, parent_dir]
            else:
                search_dirs = [cleanup_dir]
            
            removed_files = []
            total_size = 0
            
            for search_dir in search_dirs:
                if not os.path.exists(search_dir):
                    continue
                    
                # Find all .backup files
                backup_pattern = os.path.join(search_dir, "**", "*.backup")
                backup_files = glob.glob(backup_pattern, recursive=True)
                
                for backup_file in backup_files:
                    try:
                        file_size = os.path.getsize(backup_file)
                        os.remove(backup_file)
                        removed_files.append(backup_file)
                        total_size += file_size
                        print_current(f"🗑️ Removed: {backup_file}")
                    except Exception as e:
                        print_current(f"⚠️ Failed to remove {backup_file}: {e}")
            
            result = {
                'success': True,
                'removed_count': len(removed_files),
                'removed_files': removed_files,
                'total_size_bytes': total_size,
                'search_directories': search_dirs
            }
            
            if removed_files:
                size_mb = total_size / (1024 * 1024)
                print_current(f"✅ Cleaned up {len(removed_files)} backup files ({size_mb:.2f} MB)")
            else:
                print_current("ℹ️ No backup files found to clean up")
                
            return result
            
        except Exception as e:
            error_result = {
                'success': False,
                'error': f"Cleanup error: {str(e)}",
                'removed_count': 0,
                'removed_files': [],
                'total_size_bytes': 0
            }
            print_current(f"❌ Backup cleanup error: {str(e)}")
            return error_result

