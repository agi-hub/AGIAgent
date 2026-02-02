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
        self._agent_id = None  # Store agent_id for this tools instance

        # Initialize code repository parser
        self.code_parser = None
        self._init_code_parser()
        
        # Initialize terminal tools
        self.terminal_tools = None
        self._init_terminal_tools()

        # Initialize sensor data collector
        self.sensor_collector = None
        self._init_sensor_collector()
    
    def set_agent_context(self, agent_id: str):
        """Set agent ID for this tools instance and update context."""
        self._agent_id = agent_id
        from .agent_context import set_current_agent_id
        set_current_agent_id(agent_id)

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

    def talk_to_user(self, query: str, timeout: int = 120) -> Dict[str, Any]:
        """
        Display a question to the user and wait for keyboard input with timeout.
        
        Args:
            query: The question to display to the user
            timeout: Maximum time to wait for user response (default: 120 seconds, -1 to disable timeout)
            
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

    def idle(self, reason: str = None, sleep: float = 10) -> Dict[str, Any]:
        """
        Idle tool - represents doing nothing in this round, mainly used for multi-agent synchronization.
        Monitors current agent's inbox for new extmsg messages and exits sleep early if detected.
        Default sleep time is 10 seconds to wait for other agents.
        
        Args:
            reason: Optional reason for idling (default: None)
            sleep: Sleep time in seconds (default: 10). Will sleep and monitor for new extmsg messages.
                   If extmsg is detected, will exit early. Otherwise waits for full sleep time.
                   Set to -1 for infinite sleep (only wakes up when user sends a message).
            
        Returns:
            Dict containing idle status and optional reason
        """
        result = {
            'status': 'idle',
            'action': 'no_action_taken',
            'description': 'This round is idle - no operations performed'
        }
        
        if reason:
            result['reason'] = reason
        
        # Add timestamp for synchronization purposes
        import datetime
        result['timestamp'] = datetime.datetime.now().isoformat()
        
        print_current("💤 Idle - No action taken this round")
        if reason:
            print_current(f"   Reason: {reason}")
        
        # Get current agent's inbox directory and mailbox
        # Use current agent_id if available, otherwise use "manager"
        agent_inbox_dir = None
        mailbox = None
        agent_id = None
        try:
            # Try to get workspace root to determine mailbox path
            workspace_root = self.workspace_root
            if workspace_root:
                from .message_system import get_message_router
                from .agent_context import get_current_agent_id
                
                router = get_message_router(workspace_root, cleanup_on_init=False)
                
                # Get current agent ID from multiple sources
                agent_id = get_current_agent_id()
                
                # If not set in context, try to get from tools instance
                if not agent_id and hasattr(self, '_agent_id') and self._agent_id:
                    agent_id = self._agent_id
                
                # If still not set, try to extract from workspace_root path
                # e.g., /path/to/agents_output/agent_001/workspace -> agent_001
                if not agent_id and workspace_root:
                    import re
                    # Look for agent_XXX pattern in the path
                    match = re.search(r'agent_\d+', workspace_root)
                    if match:
                        agent_id = match.group(0)
                
                # Default to "manager" if still not set
                if not agent_id:
                    agent_id = "manager"
                
                mailbox = router.get_mailbox(agent_id)
                
                # If mailbox not found, try to register the agent
                if not mailbox:
                    try:
                        mailbox = router.register_agent(agent_id)
                    except Exception as e:
                        print_current(f"   ⚠️ Failed to register mailbox for {agent_id}: {e}")
                
                if mailbox:
                    agent_inbox_dir = mailbox.inbox_dir
        except Exception as e:
            print_current(f"   ⚠️ Could not access message router: {e}")
        
        # Handle infinite sleep (sleep == -1)
        if sleep == -1:
            print_current(f"   ⏸️  Infinite sleep mode - waiting for user message to wake up...")
            
            if agent_inbox_dir and os.path.exists(agent_inbox_dir) and mailbox:
                # Infinite loop checking for new messages
                check_interval = 0.5  # Check every 0.5 seconds
                while True:
                    time.sleep(check_interval)
                    
                    # Check for unread extmsg messages
                    try:
                        unread_messages = mailbox.get_unread_messages()
                        
                        # Check if any unread message is an extmsg
                        for msg in unread_messages:
                            if msg.message_id.startswith("extmsg_"):
                                print_current(f"   ⚡ Detected new extmsg message ({msg.message_id}), waking up from infinite sleep")
                                
                                # Extract message content
                                message_content = ""
                                if isinstance(msg.content, dict):
                                    message_content = msg.content.get('text', str(msg.content))
                                else:
                                    message_content = str(msg.content)
                                
                                # Format message for agent
                                formatted_message = f"""
**NEW MESSAGE RECEIVED:**
- Message ID: {msg.message_id}
- From: {msg.sender_id}
- To: {msg.receiver_id}
- Type: {msg.message_type.value}
- Priority: {msg.priority.value}
- Content: {message_content}
- Timestamp: {msg.timestamp}

**ACTION REQUIRED:** Please process this message immediately.
"""
                                
                                result['early_exit'] = True
                                result['extmsg_detected'] = msg.message_id
                                result['infinite_sleep'] = True
                                result['message'] = formatted_message
                                result['description'] = f"Idle interrupted: New extmsg message ({msg.message_id}) received from {msg.sender_id}. Message content: {message_content}"
                                result['content'] = formatted_message  # Add content field for better LLM visibility
                                result['new_message_content'] = message_content  # Direct content access
                                result['new_message_sender'] = msg.sender_id
                                
                                # Mark message as read since we've processed it
                                try:
                                    mailbox.mark_as_read(msg.message_id)
                                    print_current(f"   ✅ Marked message {msg.message_id} as read")
                                except Exception as e:
                                    print_current(f"   ⚠️ Warning: Could not mark message as read: {e}")
                                
                                return result
                    except Exception as e:
                        print_current(f"   ⚠️ Error checking for extmsg: {e}")
            else:
                print_current(f"   ⚠️ Cannot enter infinite sleep mode: {agent_id}'s inbox not available")
                result['error'] = 'inbox_not_available'
        
        # If sleep time is specified (and not -1), monitor inbox and sleep
        elif sleep > 0:
            print_current(f"   Sleeping for {sleep} seconds, monitoring for new extmsg messages...")
            
            # Monitor inbox and sleep
            if agent_inbox_dir and os.path.exists(agent_inbox_dir) and mailbox:
                # Sleep with periodic checks for new messages
                check_interval = 0.5  # Check every 0.5 seconds
                elapsed_time = 0.0
                early_exit = False
                
                while elapsed_time < sleep:
                    time.sleep(min(check_interval, sleep - elapsed_time))
                    elapsed_time += check_interval
                    
                    # Always check for unread extmsg messages (not just when file count changes)
                    # This ensures we detect messages even if they arrive between checks
                    try:
                        unread_messages = mailbox.get_unread_messages()
                        # Check if any unread message is an extmsg
                        for msg in unread_messages:
                            if msg.message_id.startswith("extmsg_"):
                                print_current(f"   ⚡ Detected new extmsg message ({msg.message_id}), exiting sleep early")
                                
                                # Extract message content
                                message_content = ""
                                if isinstance(msg.content, dict):
                                    message_content = msg.content.get('text', str(msg.content))
                                else:
                                    message_content = str(msg.content)
                                
                                # Format message for agent
                                formatted_message = f"""
**NEW MESSAGE RECEIVED:**
- Message ID: {msg.message_id}
- From: {msg.sender_id}
- To: {msg.receiver_id}
- Type: {msg.message_type.value}
- Priority: {msg.priority.value}
- Content: {message_content}
- Timestamp: {msg.timestamp}

**ACTION REQUIRED:** Please process this message immediately.
"""
                                
                                result['early_exit'] = True
                                result['extmsg_detected'] = msg.message_id
                                result['message'] = formatted_message
                                result['description'] = f"Idle interrupted: New extmsg message ({msg.message_id}) received from {msg.sender_id}. Message content: {message_content}"
                                result['content'] = formatted_message  # Add content field for better LLM visibility
                                result['new_message_content'] = message_content  # Direct content access
                                result['new_message_sender'] = msg.sender_id
                                
                                # Mark message as read since we've processed it
                                try:
                                    mailbox.mark_as_read(msg.message_id)
                                    print_current(f"   ✅ Marked message {msg.message_id} as read")
                                except Exception as e:
                                    print_current(f"   ⚠️ Warning: Could not mark message as read: {e}")
                                
                                early_exit = True
                                break
                        
                        if early_exit:
                            break
                    except Exception as e:
                        print_current(f"   ⚠️ Error checking for extmsg: {e}")
                
                if not early_exit:
                    print_current(f"   ✅ Sleep completed ({elapsed_time:.1f} seconds)")
            else:
                # If inbox directory not found, just sleep normally
                if agent_inbox_dir:
                    print_current(f"   ⚠️ {agent_id}'s inbox directory not found: {agent_inbox_dir}, sleeping normally")
                time.sleep(sleep)
                print_current(f"   ✅ Sleep completed ({sleep} seconds)")
        
        return result

    def read_img(self, query: str, image_path: str) -> Dict[str, Any]:
        """
        Read and analyze an image using the vision model, returning text description.
        
        This tool uses the configured vision_model to analyze images and return
        text descriptions instead of base64-encoded image data.
        
        Args:
            query: Query or instruction for what to analyze in the image
            image_path: Path to the image file (supports relative and absolute paths)
            
        Returns:
            Dictionary containing:
            - status: 'success' or 'failed'
            - text: Text description of the image (if successful)
            - error: Error message (if failed)
            - image_path: Path to the analyzed image
        """
        from .read_img import ImageReader
        
        # Initialize image reader with workspace root
        image_reader = ImageReader(workspace_root=self.workspace_root)
        
        # Delegate to ImageReader
        return image_reader.read_img(query, image_path)

