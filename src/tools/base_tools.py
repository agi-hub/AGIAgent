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
            timeout: Maximum time to wait for user response (default: 10 seconds, -1 to disable timeout)
            
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

