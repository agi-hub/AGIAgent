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

from flask import Flask, render_template, request, jsonify, send_file, send_from_directory
from flask_socketio import SocketIO, emit, join_room, leave_room
import os
import sys
import threading
from datetime import datetime
import shutil
import zipfile
from werkzeug.utils import secure_filename
import multiprocessing
import queue
import re

# Add parent directory to path to import config_loader
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config_loader import get_language, get_gui_default_data_directory

# Check current directory, switch to parent directory if in GUI directory
current_dir = os.getcwd()
current_dir_name = os.path.basename(current_dir)

if current_dir_name == 'GUI':
    parent_dir = os.path.dirname(current_dir)
    os.chdir(parent_dir)
    print(f"🔄 Detected startup in GUI directory, switched to parent directory: {parent_dir}")
else:
    print(f"📁 Current working directory: {current_dir}")

# Add parent directory to path to import main.py
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Application name macro definition
APP_NAME = "AGI Bot"

from src.main import AGIBotMain

# Determine template directory - always relative to this app.py file
app_dir = os.path.dirname(os.path.abspath(__file__))
template_dir = os.path.join(app_dir, 'templates')

print(f"📁 Template directory: {template_dir}")
print(f"📁 Template exists: {os.path.exists(template_dir)}")

app = Flask(__name__, template_folder=template_dir)
app.config['SECRET_KEY'] = f'{APP_NAME.lower().replace(" ", "_")}_gui_secret_key'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading', 
                   ping_timeout=60, ping_interval=25)

# Internationalization text configuration
I18N_TEXTS = {
    'zh': {
        # Page title and basic information
        'page_title': f'{APP_NAME}',
        'app_title': f'{APP_NAME}',
        'app_subtitle': '',
        'chat_title': '执行日志',
        'connected': f'已连接到 {APP_NAME}',
        'reconnect_failed': '重连失败，请刷新页面',
        
        # Button text
        'execute_direct': '直接执行',
        'execute_plan': '计划模式', 
        'new_directory': '新建目录',
        'stop_task': '停止任务',
        'refresh': '刷新',
        'upload': '上传',
        'download': '下载',
        'rename': '重命名',
        'delete': '删除',
        'confirm': '确认',
        'cancel': '取消',
        
        # Button tooltips
        'direct_tooltip': '直接执行 - 不进行任务分解',
        'plan_tooltip': '计划模式 - 先分解任务再执行',
        'new_tooltip': '新建目录 - 创建新的工作目录',
        'refresh_tooltip': '刷新目录列表',
        'upload_tooltip': '上传文件到Workspace',
        'download_tooltip': '下载目录为ZIP（排除workspace_code_index）',
        'rename_tooltip': '重命名目录',
        'delete_tooltip': '删除目录',
        
        # Input boxes and placeholders
        'input_placeholder': '请输入您的需求...',
        'rename_placeholder': '请输入新的目录名称',
        
        # Modal titles
        'upload_title': '上传文件到Workspace',
        'rename_title': '重命名目录',
        'confirm_rename': '确认重命名',
        
        # Status messages
        'task_running': '任务正在运行中...',
        'no_task_running': '当前没有任务在运行',
        'task_stopped': '任务已被用户停止',
        'task_completed': '任务执行完成！',
        'task_completed_with_errors': '任务达到最大轮数，可能未完全完成',
        'task_failed': '任务执行失败',
        'directory_created': '已创建新工作目录',
        'directory_selected': '已选择目录',
        'directory_renamed': '目录重命名成功',
        'directory_deleted': '目录删除成功',
        'files_uploaded': '文件上传成功',
        'refresh_success': '目录列表已刷新',
        
        # Mode information
        'plan_mode_info': '🔄 启用计划模式：将先分解任务再执行',
        'direct_mode_info': '⚡ 直接执行模式：不进行任务分解',
        'new_mode_info': '新建目录模式 - 点击绿色按钮创建新工作目录，或选择现有目录',
        'selected_dir_info': '已选择目录',
        
        # Error messages
        'error_no_requirement': '请提供有效的需求',
        'error_task_running': '已有任务正在运行',
        'error_no_directory': '请先选择目录',
        'error_no_files': '请先选择文件',
        'error_delete_confirm': '确定要删除目录',
        'error_delete_warning': '此操作不可撤销，将永久删除该目录及其所有内容。',
        'error_rename_empty': '新名称不能为空',
        'error_rename_same': '新名称与原名称相同或包含无效字符',
        'error_directory_exists': '目标目录已存在',
        'error_directory_not_found': '目录不存在',
        'error_permission_denied': '权限不足',
        'error_file_too_large': '文件过大无法显示',
        'error_file_not_supported': '不支持预览此文件类型',
        
        # File operations
        'file_size': '文件大小',
        'download_file': '下载文件',
        'office_preview_note': 'Office文档预览',
        'office_download_note': '下载文件: 下载到本地使用Office软件打开',
        
        # Tool execution status
        'tool_running': '执行中',
        'tool_success': '成功',
        'tool_error': '错误',
        'function_calling': '调用中',
        
        # Configuration options
        'config_options': '配置选项',
        'show_config_options': '显示配置选项',
        'hide_config_options': '隐藏配置选项',
        'enable_web_search': '搜索网络',
        'enable_knowledge_base': '搜索知识库',
        'enable_multi_agent': '启动多智能体',
        'enable_long_term_memory': '启动长期记忆',
        'enable_mcp': '启动MCP',
        'enable_jieba': '启用中文分词',
        
        # Others
        'deleting': '删除中...',
        'renaming': '重命名中...',
        'uploading': '上传中...',
        'loading': '加载中...',
        'system_message': '系统消息',
        'welcome_message': f'欢迎使用 {APP_NAME}！请在下方输入您的需求，系统将自动为您处理任务。',
        'workspace_title': '工作目录',
        'file_preview': '文件预览',
        'data_directory_info': '数据目录',
        'disconnected': '与服务器断开连接',
        'reconnected': '已重新连接到服务器',
        'drag_files': '拖拽文件到此处或点击选择文件',
        'upload_hint': '支持多文件上传，文件将保存到选定目录的workspace文件夹中',
        'select_files': '选择文件',
        
        # Additional bilingual text
        'new_messages': '条新消息',
        'auto_scroll': '自动滚动',
        'scroll_to_bottom': '滚动到底部',
        'plan_mode_suffix': ' (计划模式)',
        'continue_mode_info': '继续模式 - 将使用上次的工作目录',
        'create_or_select_directory': '请先点击绿色按钮创建新工作目录，或选择右侧的现有目录',
        'select_directory_first': '请先创建或选择一个工作目录',
        'current_name': '当前名称：',
        'new_name': '新名称：',
        'rename_info': '将使用您输入的名称作为目录名',
        'paused': '已暂停',
        'load_directory_failed': '加载目录失败',
        'network_error': '网络错误',
        'upload_network_error': '网络错误，上传失败',
        'rename_failed': '重命名失败',
        'rename_error': '重命名出错',
        'refresh_failed': '刷新失败',
        'attempt': '尝试',
        'create_directory_failed': '创建目录失败',
        'preview': '预览',
        'page_info': '第 {0} 页，共 {1} 页',
        'upload_to': '上传文件到',
        'workspace': '/workspace',
        'select_directory_error': '请先选择目录',
        'uploading_files': '正在上传 {0} 个文件...',
        'upload_progress': '上传进度: {0}%',
        'upload_failed_http': '上传失败: HTTP {0}',
        
        # Directory operations
        'directory_created_with_workspace': '已创建新工作目录: {0} (包含workspace子目录)',
        'directory_list_refreshed': '目录列表已刷新',
        'no_files_selected': '没有选择文件',
        'no_valid_files': '没有选择有效文件',
        'target_directory_not_exist': '目标目录不存在',
        'upload_success': '成功上传 {0} 个文件',
        'new_name_empty': '新名称不能为空',
        
        # Multi-user support
        'api_key_label': 'API Key:',
        'api_key_placeholder': '输入API Key (可选)',
        'api_key_tooltip': '输入您的API Key，留空则使用默认用户模式',
        'connect_btn': '连接',
        'disconnect_btn': '断开',
        'connecting': '连接中...',
        'user_connected': '已连接',
        'user_disconnected': '未连接',
        'user_reconnecting': '重连中...',
        'user_connection_failed': '连接失败',
        'default_user': '默认用户',
        'user_prefix': '用户',
    },
    'en': {
        # Page title and basic info
        'page_title': f'{APP_NAME}',
        'app_title': f'{APP_NAME}', 
        'app_subtitle': '',
        'chat_title': 'Execution Log',
        'connected': f'Connected to {APP_NAME}',
        'reconnect_failed': 'Reconnection failed, please refresh the page',
        
        # Button text
        'execute_direct': 'Execute',
        'execute_plan': 'Plan Mode',
        'new_directory': 'New Directory', 
        'stop_task': 'Stop Task',
        'refresh': 'Refresh',
        'upload': 'Upload',
        'download': 'Download',
        'rename': 'Rename',
        'delete': 'Delete',
        'confirm': 'Confirm',
        'cancel': 'Cancel',
        
        # Button tooltips
        'direct_tooltip': 'Direct execution - no task decomposition',
        'plan_tooltip': 'Plan mode - decompose tasks before execution',
        'new_tooltip': 'New directory - create new workspace',
        'refresh_tooltip': 'Refresh directory list',
        'upload_tooltip': 'Upload files to Workspace',
        'download_tooltip': 'Download directory as ZIP (excluding workspace_code_index)',
        'rename_tooltip': 'Rename directory',
        'delete_tooltip': 'Delete directory',
        
        # Input and placeholders
        'input_placeholder': 'Enter your requirements...',
        'rename_placeholder': 'Enter new directory name',
        
        # Modal titles
        'upload_title': 'Upload Files to Workspace',
        'rename_title': 'Rename Directory',
        'confirm_rename': 'Confirm Rename',
        
        # Status messages
        'task_running': 'Task is running...',
        'no_task_running': 'No task is currently running',
        'task_stopped': 'Task stopped by user',
        'task_completed': 'Task completed successfully!',
        'task_completed_with_errors': 'Task reached maximum rounds, may not be fully completed',
        'task_failed': 'Task execution failed',
        'directory_created': 'New workspace directory created',
        'directory_selected': 'Directory selected',
        'directory_renamed': 'Directory renamed successfully',
        'directory_deleted': 'Directory deleted successfully',
        'files_uploaded': 'Files uploaded successfully',
        'refresh_success': 'Directory list refreshed',
        
        # Mode info
        'plan_mode_info': '🔄 Plan mode enabled: Tasks will be decomposed before execution',
        'direct_mode_info': '⚡ Direct execution mode: No task decomposition',
        'new_mode_info': 'New directory mode - Click green button to create new workspace, or select existing directory',
        'selected_dir_info': 'Selected directory',
        
        # Error messages
        'error_no_requirement': 'Please provide a valid requirement',
        'error_task_running': 'A task is already running',
        'error_no_directory': 'Please select a directory first',
        'error_no_files': 'Please select files first',
        'error_delete_confirm': 'Are you sure you want to delete directory',
        'error_delete_warning': 'This operation cannot be undone and will permanently delete the directory and all its contents.',
        'error_rename_empty': 'New name cannot be empty',
        'error_rename_same': 'New name is the same as original or contains invalid characters',
        'error_directory_exists': 'Target directory already exists',
        'error_directory_not_found': 'Directory not found',
        'error_permission_denied': 'Permission denied',
        'error_file_too_large': 'File too large to display',
        'error_file_not_supported': 'File type not supported for preview',
        
        # File operations
        'file_size': 'File Size',
        'download_file': 'Download File',
        'office_preview_note': 'Office Document Preview',
        'office_download_note': 'Download File: Download to local and open with Office software',
        
        # Tool execution status
        'tool_running': 'Running',
        'tool_success': 'Success',
        'tool_error': 'Error',
        'function_calling': 'Calling',
        
        # Configuration options
        'config_options': 'Configuration Options',
        'show_config_options': 'Show Configuration',
        'hide_config_options': 'Hide Configuration',
        'enable_web_search': 'Web Search',
        'enable_knowledge_base': 'Knowledge Base',
        'enable_multi_agent': 'Multi-Agent',
        'enable_long_term_memory': 'Long-term Memory',
        'enable_mcp': 'Enable MCP',
        'enable_jieba': 'Chinese Segmentation',
        
        # Others
        'deleting': 'Deleting...',
        'renaming': 'Renaming...',
        'uploading': 'Uploading...',
        'loading': 'Loading...',
        'system_message': 'System Message',
        'welcome_message': f'Welcome to {APP_NAME}! Please enter your requirements below, and the system will automatically process tasks for you.',
        'workspace_title': 'Workspace',
        'file_preview': 'File Preview',
        'data_directory_info': 'Data Directory',
        'disconnected': 'Disconnected from server',
        'reconnected': 'Reconnected to server',
        'drag_files': 'Drag files here or click to select files',
        'upload_hint': 'Supports multiple file upload, files will be saved to the workspace folder of the selected directory',
        'select_files': 'Select Files',
        
        # Additional bilingual text
        'new_messages': 'new messages',
        'auto_scroll': 'Auto Scroll',
        'scroll_to_bottom': 'Scroll to Bottom',
        'plan_mode_suffix': ' (Plan Mode)',
        'continue_mode_info': 'Continue mode - Will use the previous workspace directory',
        'create_or_select_directory': 'Please click the green button to create a new workspace directory, or select an existing directory on the right',
        'select_directory_first': 'Please create or select a workspace directory first',
        'current_name': 'Current Name:',
        'new_name': 'New Name:',
        'rename_info': 'The name you enter will be used as the directory name',
        'paused': 'Paused',
        'load_directory_failed': 'Failed to load directories',
        'network_error': 'Network error',
        'upload_network_error': 'Network error, upload failed',
        'rename_failed': 'Rename failed',
        'rename_error': 'Rename error',
        'refresh_failed': 'Refresh failed',
        'attempt': 'attempt',
        'create_directory_failed': 'Failed to create directory',
        'preview': 'Preview',
        'page_info': 'Page {0} of {1}',
        'upload_to': 'Upload files to',
        'workspace': '/workspace',
        'select_directory_error': 'Please select a directory first',
        'uploading_files': 'Uploading {0} files...',
        'upload_progress': 'Upload progress: {0}%',
        'upload_failed_http': 'Upload failed: HTTP {0}',
        
        # Directory operations
        'directory_created_with_workspace': 'New workspace directory created: {0} (with workspace subdirectory)',
        'directory_list_refreshed': 'Directory list refreshed',
        'no_files_selected': 'No files selected',
        'no_valid_files': 'No valid files selected',
        'target_directory_not_exist': 'Target directory does not exist',
        'upload_success': 'Successfully uploaded {0} files',
        'new_name_empty': 'New name cannot be empty',
    }
}

def get_i18n_texts():
    """Get internationalization text for current language"""
    current_lang = get_language()
    return I18N_TEXTS.get(current_lang, I18N_TEXTS['en'])

def execute_agibot_task_process_target(user_requirement, output_queue, out_dir=None, continue_mode=False, plan_mode=False, gui_config=None, session_id=None):
    # Get i18n texts for this process
    i18n = get_i18n_texts()
    """
    This function runs in a separate process.
    It cannot use the `socketio` object directly.
    It communicates back to the main process via the queue.
    """
    try:
        output_queue.put({'event': 'task_started', 'data': {'message': 'Task execution started...'}})
        
        if not out_dir:
            # Get GUI default data directory from config for new directories
            from src.config_loader import get_gui_default_data_directory
            config_data_dir = get_gui_default_data_directory()
            if config_data_dir:
                base_dir = config_data_dir
            else:
                base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_dir = os.path.join(base_dir, f"output_{timestamp}")
        
        if continue_mode:
            output_queue.put({'event': 'output', 'data': {'message': f"Continuing with existing directory: {out_dir}", 'type': 'info'}})
        else:
            output_queue.put({'event': 'output', 'data': {'message': f"Creating output directory: {out_dir}", 'type': 'info'}})
        
        # Process GUI configuration options
        if gui_config is None:
            gui_config = {}
        
        # Set default values based on user requirements
        enable_web_search = gui_config.get('enable_web_search', False)
        enable_knowledge_base = gui_config.get('enable_knowledge_base', False)
        enable_multi_agent = gui_config.get('enable_multi_agent', False)
        enable_long_term_memory = gui_config.get('enable_long_term_memory', False)
        enable_mcp = gui_config.get('enable_mcp', False)
        enable_jieba = gui_config.get('enable_jieba', True)
        
        # Log GUI configuration
        output_queue.put({'event': 'output', 'data': {'message': f"GUI Configuration:", 'type': 'info'}})
        output_queue.put({'event': 'output', 'data': {'message': f"  - Web Search: {enable_web_search}", 'type': 'info'}})
        output_queue.put({'event': 'output', 'data': {'message': f"  - Knowledge Base: {enable_knowledge_base}", 'type': 'info'}})
        output_queue.put({'event': 'output', 'data': {'message': f"  - Multi-Agent: {enable_multi_agent}", 'type': 'info'}})
        output_queue.put({'event': 'output', 'data': {'message': f"  - Long-term Memory: {enable_long_term_memory}", 'type': 'info'}})
        output_queue.put({'event': 'output', 'data': {'message': f"  - MCP: {enable_mcp}", 'type': 'info'}})
        output_queue.put({'event': 'output', 'data': {'message': f"  - Chinese Segmentation: {enable_jieba}", 'type': 'info'}})
        
        # Create a temporary configuration that overrides config.txt for GUI mode
        # We'll use environment variables to pass these settings to the AGIBot system
        original_env = {}
        if enable_web_search:
            original_env['AGIBOT_WEB_SEARCH'] = os.environ.get('AGIBOT_WEB_SEARCH', '')
            os.environ['AGIBOT_WEB_SEARCH'] = 'true'
        if enable_multi_agent:
            original_env['AGIBOT_MULTI_AGENT'] = os.environ.get('AGIBOT_MULTI_AGENT', '')
            os.environ['AGIBOT_MULTI_AGENT'] = 'true'
        if enable_jieba:
            original_env['AGIBOT_ENABLE_JIEBA'] = os.environ.get('AGIBOT_ENABLE_JIEBA', '')
            os.environ['AGIBOT_ENABLE_JIEBA'] = 'true'
        if enable_long_term_memory:
            original_env['AGIBOT_LONG_TERM_MEMORY'] = os.environ.get('AGIBOT_LONG_TERM_MEMORY', '')
            os.environ['AGIBOT_LONG_TERM_MEMORY'] = 'true'
        
        # Set parameters based on mode
        if plan_mode:
            output_queue.put({'event': 'output', 'data': {'message': f"Plan mode enabled: Using task decomposition (--todo)", 'type': 'info'}})
            single_task_mode = False  # Plan mode uses task decomposition
        else:
            output_queue.put({'event': 'output', 'data': {'message': f"Normal mode: Direct execution (single task)", 'type': 'info'}})
            single_task_mode = True   # Default mode executes directly
        
        # Determine MCP config file based on GUI setting
        mcp_config_file = None
        if enable_mcp:
            mcp_config_file = "config/mcp_servers.json"  # Use default MCP config when enabled
            output_queue.put({'event': 'output', 'data': {'message': f"MCP enabled with config: {mcp_config_file}", 'type': 'info'}})
        
        agibot = AGIBotMain(
            out_dir=out_dir,
            debug_mode=False,
            detailed_summary=True,
            single_task_mode=single_task_mode,  # Set based on plan_mode
            interactive_mode=False,  # Disable interactive mode
            continue_mode=continue_mode,
            MCP_config_file=mcp_config_file  # Set based on GUI MCP option
        )
        
        output_queue.put({'event': 'output', 'data': {'message': f"Initialized {APP_NAME} with output directory: {out_dir}", 'type': 'info'}})
        output_queue.put({'event': 'output', 'data': {'message': f"Starting task execution...", 'type': 'info'}})
        output_queue.put({'event': 'output', 'data': {'message': f"User requirement: {user_requirement}", 'type': 'user'}})
        
        class QueueSocketHandler:
            def __init__(self, q, socket_type='info'):
                self.q = q
                self.socket_type = socket_type
                self.buffer = ""
            
            def filter_code_edit_content(self, line):
                """Filter code_edit content in tool execution parameters for GUI display"""
                # Check if line contains Parameters with code_edit field
                if "Parameters:" in line and "'code_edit':" in line:
                    # Find the start of code_edit content
                    code_edit_start = line.find("'code_edit': '")
                    if code_edit_start != -1:
                        # Find the position after 'code_edit': '
                        content_start = code_edit_start + len("'code_edit': '")
                        
                        # Find the next ', which should end the code_edit field
                        # We need to be careful about escaped quotes
                        content_end = content_start
                        quote_count = 0
                        while content_end < len(line):
                            if line[content_end] == "'":
                                # Check if it's escaped
                                if content_end > 0 and line[content_end-1] != "\\":
                                    quote_count += 1
                                    if quote_count == 1:  # Found the closing quote
                                        break
                            content_end += 1
                        
                        if content_end < len(line):
                            # Extract the content between quotes
                            content = line[content_start:content_end]
                            
                            # If content is longer than 10 characters, truncate it
                            if len(content) > 10:
                                truncated_content = content[:10] + "..."
                                filtered_line = line[:content_start] + truncated_content + line[content_end:]
                                return filtered_line
                
                return line
            
            def write(self, message):
                self.buffer += message
                if '\n' in self.buffer:
                    *lines, self.buffer = self.buffer.split('\n')
                    for line in lines:
                        if line.strip():
                            # Filter code_edit content for GUI display
                            filtered_line = self.filter_code_edit_content(line.strip())
                            
                            # Check if it's warning or progress info, if so display as normal info instead of error
                            line_lower = filtered_line.lower()
                            if ('warning' in line_lower or 
                                'progress' in line_lower or 
                                'processing files' in line_lower or
                                filtered_line.startswith('Processing files:') or
                                'userwarning' in line_lower or
                                'warnings.warn' in line_lower):
                                message_type = 'info'
                            else:
                                message_type = self.socket_type
                            # Display warning and progress info as normal info
                            self.q.put({'event': 'output', 'data': {'message': filtered_line, 'type': message_type}})

            def flush(self):
                pass
            
            def final_flush(self):
                if self.buffer.strip():
                    # Check if it's warning or progress info, if so display as normal info instead of error
                    buffer_lower = self.buffer.lower()
                    if ('warning' in buffer_lower or 
                        'progress' in buffer_lower or 
                        'processing files' in buffer_lower or
                        self.buffer.strip().startswith('Processing files:') or
                        'userwarning' in buffer_lower or
                        'warnings.warn' in buffer_lower):
                        message_type = 'info'
                    else:
                        message_type = self.socket_type
                    # Display warning and progress info as normal info
                    self.q.put({'event': 'output', 'data': {'message': self.buffer.strip(), 'type': message_type}})
                    self.buffer = ""

        original_stdout = sys.stdout
        original_stderr = sys.stderr
        
        stdout_handler = QueueSocketHandler(output_queue, 'info')
        stderr_handler = QueueSocketHandler(output_queue, 'error')

        try:
            sys.stdout = stdout_handler
            sys.stderr = stderr_handler
            
            success = agibot.run(user_requirement=user_requirement, loops=50)
            
            # Ensure important completion information is displayed
            workspace_dir = os.path.join(out_dir, "workspace")
            output_queue.put({'event': 'output', 'data': {'message': f"📁 All files saved at: {os.path.abspath(out_dir)}", 'type': 'success'}})
            
            # Extract directory name for GUI display (relative to GUI data directory)
            dir_name = os.path.basename(out_dir)
            
            if success:
                output_queue.put({'event': 'task_completed', 'data': {'message': i18n['task_completed'], 'output_dir': dir_name, 'success': True}})
            else:
                output_queue.put({'event': 'task_completed', 'data': {'message': i18n['task_completed_with_errors'], 'output_dir': dir_name, 'success': False}})
        finally:
            stdout_handler.final_flush()
            stderr_handler.final_flush()
            sys.stdout = original_stdout
            sys.stderr = original_stderr
            
    except Exception as e:
        import traceback
        tb_str = traceback.format_exc()
        output_queue.put({'event': 'error', 'data': {'message': f'Task execution failed in process: {str(e)}\\n{tb_str}'}})
    finally:
        output_queue.put({'event': 'STOP'})

class AGIBotGUI:
    def __init__(self):
        # User session management
        self.user_sessions = {}  # session_id -> UserSession
        
        # Get GUI default data directory from config, fallback to current directory
        config_data_dir = get_gui_default_data_directory()
        if config_data_dir:
            self.base_data_dir = config_data_dir
            print(f"📁 Using configured GUI base data directory: {self.base_data_dir}")
        else:
            self.base_data_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            print(f"📁 Using default GUI base data directory: {self.base_data_dir}")
        
        # Ensure base directory exists
        os.makedirs(self.base_data_dir, exist_ok=True)
        
        # Create default userdata directory
        self.default_user_dir = os.path.join(self.base_data_dir, 'userdata')
        os.makedirs(self.default_user_dir, exist_ok=True)
        print(f"📁 Default user directory: {self.default_user_dir}")

class UserSession:
    def __init__(self, session_id, api_key=None):
        self.session_id = session_id
        self.api_key = api_key
        self.current_process = None
        self.output_queue = None
        self.current_output_dir = None  # Track current execution output directory
        self.last_output_dir = None     # Track last used output directory
        self.selected_output_dir = None # Track user selected output directory
        
        # Determine user directory based on API key
        if api_key:
            # Use API key hash as directory name for security
            import hashlib
            api_key_hash = hashlib.sha256(api_key.encode()).hexdigest()[:16]
            self.user_dir_name = f"user_{api_key_hash}"
        else:
            self.user_dir_name = "userdata"
    
    def get_user_directory(self, base_dir):
        """Get the user's base directory path"""
        return os.path.join(base_dir, self.user_dir_name)
        
    def get_user_session(self, session_id, api_key=None):
        """Get or create user session"""
        if session_id not in self.user_sessions:
            self.user_sessions[session_id] = UserSession(session_id, api_key)
        return self.user_sessions[session_id]
    
    def get_output_directories(self, user_session):
        """Get all directories containing workspace subdirectory for specific user"""
        result = []
        
        # Get user's directory
        user_output_dir = user_session.get_user_directory(self.base_data_dir)
        os.makedirs(user_output_dir, exist_ok=True)
        
        try:
            # Traverse all subdirectories in user's directory
            for item in os.listdir(user_output_dir):
                item_path = os.path.join(user_output_dir, item)
                
                # Check if it's a directory
                if os.path.isdir(item_path):
                    # Check if it contains workspace subdirectory
                    workspace_path = os.path.join(item_path, 'workspace')
                    if os.path.exists(workspace_path) and os.path.isdir(workspace_path):
                        # Get directory information
                        stat = os.stat(item_path)
                        size = self.get_directory_size(item_path)
                        
                        result.append({
                            'name': item,
                            'path': item_path,
                            'size': self.format_size(size),
                            'modified_time': datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S'),
                            'files': self.get_directory_structure(item_path),
                            'is_current': item == user_session.current_output_dir,  # Mark if it's current directory
                            'is_selected': item == user_session.selected_output_dir,  # Mark if it's selected directory
                            'is_last': item == user_session.last_output_dir  # Mark if it's last used directory
                        })
        except (OSError, PermissionError) as e:
            print(f"Error reading directories: {e}")
        
        # Sort by modification time
        result.sort(key=lambda x: os.path.getmtime(x['path']), reverse=True)
        return result
    
    def get_directory_size(self, directory):
        """Calculate directory size"""
        total_size = 0
        try:
            for dirpath, dirnames, filenames in os.walk(directory):
                for filename in filenames:
                    filepath = os.path.join(dirpath, filename)
                    if os.path.exists(filepath):
                        total_size += os.path.getsize(filepath)
        except (OSError, IOError):
            pass
        return total_size
    
    def format_size(self, size_bytes):
        """Format file size"""
        if size_bytes == 0:
            return "0 B"
        size_names = ["B", "KB", "MB", "GB", "TB"]
        i = 0
        while size_bytes >= 1024.0 and i < len(size_names) - 1:
            size_bytes /= 1024.0
            i += 1
        return f"{size_bytes:.1f} {size_names[i]}"
    
    def get_directory_structure(self, directory, max_depth=3, current_depth=0, base_dir=None):
        """Get directory structure"""
        if current_depth > max_depth:
            return []
        
        # If first call, set base_dir to parent directory of current directory
        if base_dir is None:
            base_dir = os.path.dirname(directory)
        
        items = []
        try:
            for item in os.listdir(directory):
                item_path = os.path.join(directory, item)
                # Calculate relative path to base_dir
                relative_path = os.path.relpath(item_path, base_dir)
                # Convert Windows path separators to Unix style
                relative_path = relative_path.replace('\\', '/')
                
                if os.path.isdir(item_path):
                    children = self.get_directory_structure(item_path, max_depth, current_depth + 1, base_dir)
                    items.append({
                        'name': item,
                        'type': 'directory',
                        'path': relative_path,
                        'children': children
                    })
                else:
                    items.append({
                        'name': item,
                        'type': 'file',
                        'path': relative_path,
                        'size': self.format_size(os.path.getsize(item_path))
                    })
        except (OSError, PermissionError):
            pass
        
        return sorted(items, key=lambda x: (x['type'] == 'file', x['name']))

gui_instance = AGIBotGUI()

def queue_reader_thread(session_id):
    """Reads from the queue and emits messages to the client via SocketIO."""
    print(f"Queue reader thread started for user {session_id}.")
    
    if session_id not in gui_instance.user_sessions:
        print(f"❌ User session {session_id} not found")
        return
    
    user_session = gui_instance.user_sessions[session_id]
    
    while True:
        try:
            if user_session.current_process and not user_session.current_process.is_alive() and user_session.output_queue.empty():
                print(f"Process finished and queue is empty for user {session_id}, stopping reader.")
                break

            message = user_session.output_queue.get(timeout=1)
            
            if message.get('event') == 'STOP':
                print(f"Received STOP sentinel for user {session_id}.")
                break
            
            # If task completion message, save last used directory and clear current directory mark
            if message.get('event') in ['task_completed', 'error']:
                if user_session.current_output_dir:
                    user_session.last_output_dir = user_session.current_output_dir
                    # If current directory is the selected directory, keep the selection
                    # This ensures user can continue in the same directory
                    if user_session.selected_output_dir == user_session.current_output_dir:
                        print(f"🔄 Keeping selected directory for user {session_id}: {user_session.selected_output_dir}")
                    else:
                        # If different directories, clear selection to avoid confusion
                        print(f"🔄 Clearing selected directory for user {session_id} (was {user_session.selected_output_dir}, current {user_session.current_output_dir})")
                        user_session.selected_output_dir = None
                user_session.current_output_dir = None
            
            # Emit to user's specific room
            socketio.emit(message['event'], message.get('data', {}), room=session_id)
        except queue.Empty:
            continue
        except Exception as e:
            print(f"Error in queue_reader_thread for user {session_id}: {e}")
            break
    
    print(f"Queue reader thread finished for user {session_id}.")
    if user_session.current_process:
        user_session.current_process.join(timeout=1)
    user_session.current_process = None
    user_session.output_queue = None
    if user_session.current_output_dir:
        user_session.last_output_dir = user_session.current_output_dir
    user_session.current_output_dir = None  # Clear current directory mark

@app.route('/')
def index():
    """Main page"""
    i18n = get_i18n_texts()
    current_lang = get_language()
    return render_template('index.html', i18n=i18n, lang=current_lang)

@app.route('/test_toggle_simple.html')
def test_toggle_simple():
    """Expand/collapse functionality test page"""
    return send_from_directory('.', 'test_toggle_simple.html')

@app.route('/simple_test.html')
def simple_test():
    """Simple test page"""
    return send_from_directory('.', 'simple_test.html')

@app.route('/api/output-dirs')
def get_output_dirs():
    """Get output directory list"""
    try:
        # Get API key from query parameters
        api_key = request.args.get('api_key')
        
        # Create a temporary session for API calls (since no socket connection)
        temp_session_id = f"api_{request.remote_addr}_{id(request)}"
        user_session = gui_instance.get_user_session(temp_session_id, api_key)
        
        dirs = gui_instance.get_output_directories(user_session)
        return jsonify({'success': True, 'directories': dirs})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/download/<path:dir_name>')
def download_directory(dir_name):
    """Download directory as zip file (excluding workspace_code_index directory)"""
    try:
        # Get API key from query parameters or headers
        api_key = request.args.get('api_key') or request.headers.get('X-API-Key')
        
        # Create a temporary session for API calls
        temp_session_id = f"api_{request.remote_addr}_{id(request)}"
        user_session = gui_instance.get_user_session(temp_session_id, api_key)
        user_base_dir = user_session.get_user_directory(gui_instance.base_data_dir)
        
        dir_path = os.path.join(user_base_dir, secure_filename(dir_name))
        if not os.path.exists(dir_path) or not os.path.isdir(dir_path):
            return jsonify({'success': False, 'error': 'Directory not found'})
        
        # Create temporary zip file
        temp_file = f"/tmp/{dir_name}.zip"
        
        with zipfile.ZipFile(temp_file, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(dir_path):
                # Exclude workspace_code_index directory
                if 'workspace_code_index' in root:
                    print(f"Excluding directory: {root}")  # Debug info
                    continue
                
                for file in files:
                    file_path = os.path.join(root, file)
                    # 计算相对路径
                    arcname = os.path.join(dir_name, os.path.relpath(file_path, dir_path))
                    zipf.write(file_path, arcname)
        
        return send_file(temp_file, as_attachment=True, download_name=f"{dir_name}.zip")
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/file/<path:file_path>')
def get_file_content(file_path):
    """Get file content"""
    try:
        # Get API key from query parameters or headers
        api_key = request.args.get('api_key') or request.headers.get('X-API-Key')
        
        # Create a temporary session for API calls
        temp_session_id = f"api_{request.remote_addr}_{id(request)}"
        user_session = gui_instance.get_user_session(temp_session_id, api_key)
        user_base_dir = user_session.get_user_directory(gui_instance.base_data_dir)
        
        # Use the passed path directly, don't use secure_filename as we need to maintain path structure
        full_path = os.path.join(user_base_dir, file_path)
        
        # Security check: ensure path is within user's output directory
        real_output_dir = os.path.realpath(user_base_dir)
        real_file_path = os.path.realpath(full_path)
        if not real_file_path.startswith(real_output_dir):
            return jsonify({'success': False, 'error': 'Access denied'})
        
        if not os.path.exists(full_path) or not os.path.isfile(full_path):
            return jsonify({'success': False, 'error': f'File not found: {file_path}'})
        
        # Check file size to avoid reading oversized files
        file_size = os.path.getsize(full_path)
        if file_size > 5 * 1024 * 1024:  # 5MB
            return jsonify({'success': False, 'error': 'File too large to display'})
        
        # Get file extension
        _, ext = os.path.splitext(full_path.lower())
        
        # Decide how to handle based on file type
        if ext in ['.html', '.htm']:
            with open(full_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            return jsonify({
                'success': True, 
                'content': content, 
                'type': 'html',
                'size': gui_instance.format_size(file_size)
            })
        elif ext in ['.md', '.markdown']:
            with open(full_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            return jsonify({
                'success': True, 
                'content': content, 
                'type': 'markdown',
                'size': gui_instance.format_size(file_size)
            })
        elif ext == '.pdf':
            # PDF文件直接返回文件路径，让前端处理
            return jsonify({
                'success': True, 
                'type': 'pdf',
                'file_path': file_path,
                'size': gui_instance.format_size(file_size)
            })
        elif ext in ['.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx']:
            # Office文档预览
            return jsonify({
                'success': True, 
                'type': 'office',
                'file_path': file_path,
                'file_ext': ext,
                'size': gui_instance.format_size(file_size)
            })
        elif ext in ['.py', '.js', '.jsx', '.ts', '.tsx', '.css', '.json', '.txt', '.log', '.yaml', '.yml', 
                     '.c', '.cpp', '.cc', '.cxx', '.h', '.hpp', '.java', '.go', '.rs', '.php', '.rb', 
                     '.sh', '.bash', '.zsh', '.fish', '.ps1', '.bat', '.cmd', '.xml', '.sql', '.r', 
                     '.scala', '.kt', '.swift', '.dart', '.lua', '.perl', '.pl', '.vim', '.dockerfile', 
                     '.makefile', '.cmake', '.gradle', '.properties', '.ini', '.cfg', '.conf', '.toml']:
            with open(full_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # Language mapping for syntax highlighting
            language_map = {
                '.py': 'python',
                '.js': 'javascript', 
                '.jsx': 'javascript',
                '.ts': 'typescript',
                '.tsx': 'typescript',
                '.css': 'css',
                '.json': 'json',
                '.c': 'c',
                '.cpp': 'cpp',
                '.cc': 'cpp',
                '.cxx': 'cpp',
                '.h': 'c',
                '.hpp': 'cpp',
                '.java': 'java',
                '.go': 'go',
                '.rs': 'rust',
                '.php': 'php',
                '.rb': 'ruby',
                '.sh': 'bash',
                '.bash': 'bash',
                '.zsh': 'bash',
                '.fish': 'bash',
                '.ps1': 'powershell',
                '.bat': 'batch',
                '.cmd': 'batch',
                '.xml': 'xml',
                '.sql': 'sql',
                '.r': 'r',
                '.scala': 'scala',
                '.kt': 'kotlin',
                '.swift': 'swift',
                '.dart': 'dart',
                '.lua': 'lua',
                '.perl': 'perl',
                '.pl': 'perl',
                '.vim': 'vim',
                '.dockerfile': 'dockerfile',
                '.makefile': 'makefile',
                '.cmake': 'cmake',
                '.gradle': 'gradle',
                '.yaml': 'yaml',
                '.yml': 'yaml',
                '.toml': 'toml',
                '.txt': 'text',
                '.log': 'text'
            }
            
            language = language_map.get(ext, ext[1:])  # Default to remove dot
            
            return jsonify({
                'success': True, 
                'content': content, 
                'type': 'code',
                'language': language,
                'size': gui_instance.format_size(file_size)
            })
        elif ext == '.csv':
            # CSV file table preview
            import csv
            import io
            
            try:
                # Read CSV file
                with open(full_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                
                # Parse CSV content
                csv_reader = csv.reader(io.StringIO(content))
                rows = list(csv_reader)
                
                if not rows:
                    return jsonify({'success': False, 'error': 'CSV file is empty'})
                
                # Get header (first row)
                headers = rows[0] if rows else []
                data_rows = rows[1:] if len(rows) > 1 else []
                
                # Limit displayed rows to avoid frontend lag
                max_rows = 1000
                if len(data_rows) > max_rows:
                    data_rows = data_rows[:max_rows]
                    truncated = True
                    total_rows = len(rows) - 1  # Subtract header
                else:
                    truncated = False
                    total_rows = len(data_rows)
                
                return jsonify({
                    'success': True,
                    'type': 'csv',
                    'headers': headers,
                    'data': data_rows,
                    'total_rows': total_rows,
                    'displayed_rows': len(data_rows),
                    'truncated': truncated,
                    'size': gui_instance.format_size(file_size)
                })
                
            except UnicodeDecodeError:
                # Try other encodings
                try:
                    with open(full_path, 'r', encoding='gbk', errors='ignore') as f:
                        content = f.read()
                    
                    csv_reader = csv.reader(io.StringIO(content))
                    rows = list(csv_reader)
                    
                    if not rows:
                        return jsonify({'success': False, 'error': 'CSV file is empty'})
                    
                    headers = rows[0] if rows else []
                    data_rows = rows[1:] if len(rows) > 1 else []
                    
                    max_rows = 1000
                    if len(data_rows) > max_rows:
                        data_rows = data_rows[:max_rows]
                        truncated = True
                        total_rows = len(rows) - 1
                    else:
                        truncated = False
                        total_rows = len(data_rows)
                    
                    return jsonify({
                        'success': True,
                        'type': 'csv',
                        'headers': headers,
                        'data': data_rows,
                        'total_rows': total_rows,
                        'displayed_rows': len(data_rows),
                        'truncated': truncated,
                        'encoding': 'gbk',
                        'size': gui_instance.format_size(file_size)
                    })
                except Exception:
                    return jsonify({'success': False, 'error': 'CSV file encoding not supported, please try UTF-8 or GBK encoding'})
            
            except Exception as e:
                return jsonify({'success': False, 'error': f'CSV file parsing failed: {str(e)}'})
        else:
            return jsonify({'success': False, 'error': 'File type not supported for preview'})
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/pdf/<path:file_path>')
def serve_pdf(file_path):
    """Serve PDF file directly"""
    try:
        # Get API key from query parameters or headers
        api_key = request.args.get('api_key') or request.headers.get('X-API-Key')
        
        # Create a temporary session for API calls
        temp_session_id = f"api_{request.remote_addr}_{id(request)}"
        user_session = gui_instance.get_user_session(temp_session_id, api_key)
        user_base_dir = user_session.get_user_directory(gui_instance.base_data_dir)
        
        # Use the passed path directly, don't use secure_filename as we need to maintain path structure
        full_path = os.path.join(user_base_dir, file_path)
        
        # Security check: ensure path is within user's output directory
        real_output_dir = os.path.realpath(user_base_dir)
        real_file_path = os.path.realpath(full_path)
        if not real_file_path.startswith(real_output_dir):
            return jsonify({'success': False, 'error': 'Access denied'})
        
        if not os.path.exists(full_path) or not os.path.isfile(full_path):
            return jsonify({'success': False, 'error': f'File not found: {file_path}'})
        
        # Check if it's a PDF file
        if not full_path.lower().endswith('.pdf'):
            return jsonify({'success': False, 'error': 'Not a PDF file'})
        
        return send_file(full_path, mimetype='application/pdf')
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/download-file/<path:file_path>')
def download_file(file_path):
    """Download file directly"""
    try:
        # Get API key from query parameters or headers
        api_key = request.args.get('api_key') or request.headers.get('X-API-Key')
        
        # Create a temporary session for API calls
        temp_session_id = f"api_{request.remote_addr}_{id(request)}"
        user_session = gui_instance.get_user_session(temp_session_id, api_key)
        user_base_dir = user_session.get_user_directory(gui_instance.base_data_dir)
        
        # Use the passed path directly, don't use secure_filename as we need to maintain path structure
        full_path = os.path.join(user_base_dir, file_path)
        
        # Security check: ensure path is within user's output directory
        real_output_dir = os.path.realpath(user_base_dir)
        real_file_path = os.path.realpath(full_path)
        if not real_file_path.startswith(real_output_dir):
            return jsonify({'success': False, 'error': 'Access denied'})
        
        if not os.path.exists(full_path) or not os.path.isfile(full_path):
            return jsonify({'success': False, 'error': f'File not found: {file_path}'})
        
        # Get file extension and set appropriate mimetype
        _, ext = os.path.splitext(full_path.lower())
        
        # Define mimetypes for different file types
        mimetype_map = {
            '.pdf': 'application/pdf',
            '.doc': 'application/msword',
            '.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
            '.xls': 'application/vnd.ms-excel',
            '.xlsx': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            '.ppt': 'application/vnd.ms-powerpoint',
            '.pptx': 'application/vnd.openxmlformats-officedocument.presentationml.presentation',
            '.txt': 'text/plain',
            '.html': 'text/html',
            '.css': 'text/css',
            '.js': 'application/javascript',
            '.json': 'application/json',
            '.xml': 'application/xml',
            '.zip': 'application/zip',
            '.png': 'image/png',
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.gif': 'image/gif',
            '.svg': 'image/svg+xml'
        }
        
        # Get mimetype or use default
        mimetype = mimetype_map.get(ext, 'application/octet-stream')
        
        # Get filename for download
        filename = os.path.basename(full_path)
        
        return send_file(full_path, 
                        mimetype=mimetype, 
                        as_attachment=True, 
                        download_name=filename)
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/upload-to-cloud/<path:file_path>')
def upload_to_cloud(file_path):
    """Upload file to cloud storage for preview"""
    try:
        import requests
        
        # Get API key from query parameters or headers
        api_key = request.args.get('api_key') or request.headers.get('X-API-Key')
        
        # Create a temporary session for API calls
        temp_session_id = f"api_{request.remote_addr}_{id(request)}"
        user_session = gui_instance.get_user_session(temp_session_id, api_key)
        user_base_dir = user_session.get_user_directory(gui_instance.base_data_dir)
        
        # Use the passed path directly, don't use secure_filename as we need to maintain path structure
        full_path = os.path.join(user_base_dir, file_path)
        
        # Security check: ensure path is within user's output directory
        real_output_dir = os.path.realpath(user_base_dir)
        real_file_path = os.path.realpath(full_path)
        if not real_file_path.startswith(real_output_dir):
            return jsonify({'success': False, 'error': 'Access denied'})
        
        if not os.path.exists(full_path) or not os.path.isfile(full_path):
            return jsonify({'success': False, 'error': f'File not found: {file_path}'})
        
        # Check file size (most free services have limits)
        file_size = os.path.getsize(full_path)
        if file_size > 100 * 1024 * 1024:  # 100MB limit
            return jsonify({'success': False, 'error': 'File too large (max 100MB)'})
        
        filename = os.path.basename(full_path)
        
        # For testing purposes, use local file URL when cloud services fail
        # This allows us to test the preview functionality
        # Can be controlled by environment variable CLOUD_PREVIEW_TEST_MODE
        test_mode = os.environ.get('CLOUD_PREVIEW_TEST_MODE', 'false').lower() == 'true'
        
        # Disable test mode for now to enable real cloud upload
        # if test_mode:
        #     # Return local file URL for testing
        #     local_url = f"{request.host_url}api/download-file/{file_path}"
        #     return jsonify({
        #         'success': True,
        #         'cloud_url': local_url,
        #         'service': 'Local Test',
        #         'expires': 'Session'
        #     })
        
        # Try multiple cloud storage services
        cloud_services = [
            {
                'name': 'transfer.sh',
                'url': 'https://transfer.sh',
                'method': 'transfer'
            },
            {
                'name': '0x0.st',
                'url': 'https://0x0.st',
                'method': '0x0'
            },
            {
                'name': 'File.io',
                'url': 'https://file.io',
                'method': 'fileio'
            }
        ]
        
        for service in cloud_services:
            try:
                if service['method'] == 'transfer':
                    # transfer.sh upload
                    with open(full_path, 'rb') as f:
                        response = requests.put(f"{service['url']}/{filename}", data=f, timeout=30)
                    
                    if response.status_code == 200:
                        cloud_url = response.text.strip()
                        if cloud_url.startswith('http'):
                            return jsonify({
                                'success': True,
                                'cloud_url': cloud_url,
                                'service': service['name'],
                                'expires': '14 days'
                            })
                
                elif service['method'] == 'fileio':
                    # File.io upload
                    with open(full_path, 'rb') as f:
                        files = {'file': (filename, f)}
                        response = requests.post(service['url'], files=files, timeout=30)
                    
                    if response.status_code == 200:
                        try:
                            data = response.json()
                            if data.get('success'):
                                return jsonify({
                                    'success': True,
                                    'cloud_url': data['link'],
                                    'service': service['name'],
                                    'expires': '14 days'
                                })
                        except ValueError as e:
                            print(f"File.io JSON parse error: {e}, response: {response.text[:200]}")
                            # File.io might return plain text URL
                            if response.text.startswith('http'):
                                return jsonify({
                                    'success': True,
                                    'cloud_url': response.text.strip(),
                                    'service': service['name'],
                                    'expires': '14 days'
                                })
                
                elif service['method'] == '0x0':
                    # 0x0.st upload
                    with open(full_path, 'rb') as f:
                        files = {'file': (filename, f)}
                        response = requests.post(service['url'], files=files, timeout=30)
                    
                    if response.status_code == 200:
                        cloud_url = response.text.strip()
                        if cloud_url.startswith('http'):
                            return jsonify({
                                'success': True,
                                'cloud_url': cloud_url,
                                'service': service['name'],
                                'expires': '365 days'
                            })
                

                            
            except Exception as e:
                print(f"Failed to upload to {service['name']}: {e}")
                continue
        
        return jsonify({'success': False, 'error': 'All cloud storage services failed'})
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@socketio.on('connect')
def handle_connect(auth):
    """WebSocket connection processing with authentication"""
    i18n = get_i18n_texts()
    
    # Get user authentication info
    api_key = None
    if auth and 'api_key' in auth:
        api_key = auth['api_key']
    
    # Create or get user session
    session_id = request.sid
    user_session = gui_instance.get_user_session(session_id, api_key)
    
    # Create user directory if not exists
    user_dir = user_session.get_user_directory(gui_instance.base_data_dir)
    os.makedirs(user_dir, exist_ok=True)
    
    # Join user to their own room for isolated communication
    join_room(session_id)
    
    print(f"🔗 User connected: {session_id}, API Key: {'***' if api_key else 'None'}, Directory: {os.path.basename(user_dir)}")
    
    emit('status', {'message': i18n['connected']}, room=session_id)

@socketio.on('disconnect')
def handle_disconnect():
    """Handle user disconnection"""
    session_id = request.sid
    
    if session_id in gui_instance.user_sessions:
        user_session = gui_instance.user_sessions[session_id]
        
        # Clean up any running processes
        if user_session.current_process and user_session.current_process.is_alive():
            print(f"🛑 Terminating process for disconnected user {session_id}")
            user_session.current_process.terminate()
            user_session.current_process.join(timeout=5)  # Wait up to 5 seconds
        
        # Leave room
        leave_room(session_id)
        
        # Remove user session
        del gui_instance.user_sessions[session_id]
        
        print(f"🔌 User disconnected and cleaned up: {session_id}")
    else:
        print(f"🔌 User disconnected (no session found): {session_id}")

@socketio.on('execute_task')
def handle_execute_task(data):
    """Handle task execution request"""
    i18n = get_i18n_texts()
    session_id = request.sid
    
    # Get user session
    if session_id not in gui_instance.user_sessions:
        emit('error', {'message': 'User session not found'}, room=session_id)
        return
    
    user_session = gui_instance.user_sessions[session_id]
    
    if user_session.current_process and user_session.current_process.is_alive():
        emit('error', {'message': i18n['error_task_running']}, room=session_id)
        return

    user_requirement = data.get('requirement', '')
    if not user_requirement.strip():
        emit('error', {'message': i18n['error_no_requirement']}, room=session_id)
        return
    
    task_type = data.get('type', 'continue')  # 'new', 'continue', 'selected'
    plan_mode = data.get('plan_mode', False)  # Whether to use plan mode (task decomposition)
    selected_directory = data.get('selected_directory')  # Directory name from frontend
    gui_config = data.get('gui_config', {})  # GUI configuration options
    
    # Get user's base directory
    user_base_dir = user_session.get_user_directory(gui_instance.base_data_dir)
    
    # Debug logging
    print(f"🔍 Execute task debug info for user {session_id}:")
    print(f"   Task type: {task_type}")
    print(f"   Plan mode: {plan_mode}")
    print(f"   Frontend selected_directory: {selected_directory}")
    print(f"   Backend user_session.selected_output_dir: {user_session.selected_output_dir}")
    print(f"   Backend user_session.last_output_dir: {user_session.last_output_dir}")
    print(f"   User base directory: {user_base_dir}")
    
    if task_type == 'new':
        # New task: create new output directory
        out_dir = None
        continue_mode = False
    elif task_type == 'selected':
        # Use selected directory - prioritize frontend passed directory name
        target_dir_name = selected_directory or user_session.selected_output_dir
        if target_dir_name:
            out_dir = os.path.join(user_base_dir, target_dir_name)
            # Update backend state to match frontend
            user_session.selected_output_dir = target_dir_name
            print(f"🎯 Using selected directory: {target_dir_name} (from {'frontend' if selected_directory else 'backend state'})")
        else:
            out_dir = None
            print("⚠️ Selected task type but no directory specified")
        # Check if selected directory is newly created (not in last_output_dir)
        # If it's a new directory, should use continue_mode=False
        if target_dir_name != user_session.last_output_dir:
            continue_mode = False  # New directory, don't continue previous work
        else:
            continue_mode = True   # Existing directory, continue previous work
    else:
        # Continue mode: use last output directory - convert to absolute path
        if user_session.last_output_dir:
            out_dir = os.path.join(user_base_dir, user_session.last_output_dir)
        else:
            out_dir = None
        continue_mode = True
        
        # If no last directory, create new one
        if not out_dir:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_dir = os.path.join(user_base_dir, f"output_{timestamp}")
    
    user_session.output_queue = multiprocessing.Queue()
    
    user_session.current_process = multiprocessing.Process(
        target=execute_agibot_task_process_target,
        args=(user_requirement, user_session.output_queue, out_dir, continue_mode, plan_mode, gui_config, session_id)
    )
    user_session.current_process.daemon = True
    user_session.current_process.start()
    
    # Set current output directory name (extract from absolute path if needed)
    if out_dir:
        user_session.current_output_dir = os.path.basename(out_dir)
    else:
        user_session.current_output_dir = None

    threading.Thread(target=queue_reader_thread, args=(session_id,), daemon=True).start()

@socketio.on('select_directory')
def handle_select_directory(data):
    """Handle directory selection request"""
    session_id = request.sid
    if session_id not in gui_instance.user_sessions:
        return
    
    user_session = gui_instance.user_sessions[session_id]
    dir_name = data.get('dir_name', '')
    if dir_name:
        user_session.selected_output_dir = dir_name
        emit('directory_selected', {'dir_name': dir_name}, room=session_id)
    else:
        user_session.selected_output_dir = None
        emit('directory_selected', {'dir_name': None}, room=session_id)

@socketio.on('stop_task')
def handle_stop_task():
    """Handle stop task request"""
    i18n = get_i18n_texts()
    session_id = request.sid
    
    if session_id not in gui_instance.user_sessions:
        return
    
    user_session = gui_instance.user_sessions[session_id]
    
    if user_session.current_process and user_session.current_process.is_alive():
        print(f"Received stop request for user {session_id}. Terminating process.")
        user_session.current_process.terminate()
        user_session.current_output_dir = None  # Clear current directory mark
        emit('task_stopped', {'message': i18n['task_stopped'], 'type': 'error'}, room=session_id)
    else:
        emit('output', {'message': i18n['no_task_running'], 'type': 'info'}, room=session_id)

@socketio.on('create_new_directory')
def handle_create_new_directory():
    """Handle create new directory request"""
    session_id = request.sid
    if session_id not in gui_instance.user_sessions:
        return
    
    user_session = gui_instance.user_sessions[session_id]
    user_base_dir = user_session.get_user_directory(gui_instance.base_data_dir)
    
    try:
        i18n = get_i18n_texts()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        new_dir_name = f"output_{timestamp}"
        new_dir_path = os.path.join(user_base_dir, new_dir_name)
        
        # Create main directory
        os.makedirs(new_dir_path, exist_ok=True)
        
        # Create workspace subdirectory
        workspace_dir = os.path.join(new_dir_path, 'workspace')
        os.makedirs(workspace_dir, exist_ok=True)
        
        # Set as currently selected directory
        user_session.selected_output_dir = new_dir_name
        
        emit('directory_created', {
            'dir_name': new_dir_name,
            'success': True,
            'message': i18n['directory_created_with_workspace'].format(new_dir_name)
        }, room=session_id)
        
    except Exception as e:
        emit('directory_created', {
            'success': False,
            'error': str(e)
        }, room=session_id)

@app.route('/api/refresh-dirs', methods=['POST'])
def refresh_directories():
    """Manually refresh directory list"""
    try:
        i18n = get_i18n_texts()
        
        # Get API key from JSON data, query parameters or headers
        api_key = None
        if request.json:
            api_key = request.json.get('api_key')
        if not api_key:
            api_key = request.args.get('api_key') or request.headers.get('X-API-Key')
        
        # Create a temporary session for API calls
        temp_session_id = f"api_{request.remote_addr}_{id(request)}"
        user_session = gui_instance.get_user_session(temp_session_id, api_key)
        
        # Use existing method to get directory list for this user
        directories = gui_instance.get_output_directories(user_session)
        return jsonify({
            'success': True,
            'directories': directories,
            'message': i18n['directory_list_refreshed']
        })
    except Exception as e:
        print(f"Failed to refresh directories: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# File upload functionality
@app.route('/api/upload/<path:dir_name>', methods=['POST'])
def upload_files(dir_name):
    """Upload files to workspace of specified directory"""
    try:
        i18n = get_i18n_texts()
        
        # Get API key from form data, query parameters or headers
        api_key = request.form.get('api_key') or request.args.get('api_key') or request.headers.get('X-API-Key')
        
        # Create a temporary session for API calls
        temp_session_id = f"api_{request.remote_addr}_{id(request)}"
        user_session = gui_instance.get_user_session(temp_session_id, api_key)
        user_base_dir = user_session.get_user_directory(gui_instance.base_data_dir)
        
        if 'files' not in request.files:
            return jsonify({'success': False, 'error': i18n['no_files_selected']})
        
        files = request.files.getlist('files')
        if not files or all(f.filename == '' for f in files):
            return jsonify({'success': False, 'error': i18n['no_valid_files']})
        
        # Target directory path
        target_dir = os.path.join(user_base_dir, secure_filename(dir_name))
        if not os.path.exists(target_dir):
            return jsonify({'success': False, 'error': i18n['target_directory_not_exist']})
        
        # workspace directory path
        workspace_dir = os.path.join(target_dir, 'workspace')
        os.makedirs(workspace_dir, exist_ok=True)
        
        uploaded_files = []
        for file in files:
            if file.filename:
                # Custom secure filename handling, preserve Chinese characters
                safe_filename = sanitize_filename(file.filename)
                if not safe_filename:
                    continue
                
                # If file already exists, add timestamp
                if os.path.exists(os.path.join(workspace_dir, safe_filename)):
                    name, ext = os.path.splitext(safe_filename)
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    safe_filename = f"{name}_{timestamp}{ext}"
                
                file_path = os.path.join(workspace_dir, safe_filename)
                
                file.save(file_path)
                uploaded_files.append(safe_filename)
        
        return jsonify({
            'success': True,
            'message': i18n['upload_success'].format(len(uploaded_files)),
            'files': uploaded_files
        })
        
    except Exception as e:
        print(f"File upload failed: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

def sanitize_filename(filename, is_directory=False):
    """
    Custom filename sanitization function, preserve Chinese characters but remove dangerous characters
    """
    if not filename:
        return None
    
    # Remove path separators and other dangerous characters, but preserve Chinese characters
    # Allow: letters, numbers, Chinese characters, dots, underscores, hyphens, spaces, parentheses
    filename = re.sub(r'[<>:"/\\|?*]', '', filename)
    
    # Remove leading and trailing spaces and dots
    filename = filename.strip(' .')
    
    # If filename is empty, return None
    if not filename:
        return None
    
    # For directory names, allow starting with dots (like .git, etc.)
    # Limit filename length
    if len(filename) > 255:
        filename = filename[:255]
    
    return filename

@app.route('/api/rename-directory/<path:old_name>', methods=['PUT'])
def rename_directory(old_name):
    """Rename output directory"""
    try:
        i18n = get_i18n_texts()
        
        # Get API key from form data, query parameters or headers
        api_key = request.json.get('api_key') if request.json else None
        if not api_key:
            api_key = request.args.get('api_key') or request.headers.get('X-API-Key')
        
        # Create a temporary session for API calls
        temp_session_id = f"api_{request.remote_addr}_{id(request)}"
        user_session = gui_instance.get_user_session(temp_session_id, api_key)
        user_base_dir = user_session.get_user_directory(gui_instance.base_data_dir)
        
        data = request.get_json()
        new_name = data.get('new_name', '').strip()
        
        if not new_name:
            return jsonify({'success': False, 'error': i18n['new_name_empty']})
        
        # Check if it's currently executing directory for any user with same API key
        # (This is a simplification - in practice we might want to check all sessions with same API key)
        if hasattr(user_session, 'current_output_dir') and old_name == user_session.current_output_dir:
            return jsonify({'success': False, 'error': 'Cannot rename directory currently in use'})
        
        # Use custom secure filename handling, preserve more characters
        new_name_safe = sanitize_filename(new_name, is_directory=True)
        if not new_name_safe:
            return jsonify({'success': False, 'error': 'Invalid directory name'})
        
        # Build complete path
        old_path = os.path.join(user_base_dir, secure_filename(old_name))
        new_path = os.path.join(user_base_dir, new_name_safe)
        
        # Debug info
        print(f"Rename debug info:")
        print(f"  Original old_name: {old_name}")
        print(f"  Original new_name: {new_name}")
        print(f"  Safe old_name: {new_name_safe}")
        print(f"  Old path: {old_path}")
        print(f"  New path: {new_path}")
        print(f"  Paths are same: {old_path == new_path}")
        
        # If processed paths are the same, it means the new name is invalid
        if old_path == new_path:
            return jsonify({'success': False, 'error': 'New name is the same as original or contains invalid characters'})
        
        # Security check: ensure paths are within expected directory
        real_old_path = os.path.realpath(old_path)
        real_new_path = os.path.realpath(new_path)
        expected_parent = os.path.realpath(user_base_dir)
        
        if not real_old_path.startswith(expected_parent) or not real_new_path.startswith(expected_parent):
            return jsonify({'success': False, 'error': 'Paths are not safe'})
        
        # Check if original directory exists
        if not os.path.exists(old_path):
            return jsonify({'success': False, 'error': 'Original directory does not exist'})
        
        # Check if new directory exists
        if os.path.exists(new_path):
            return jsonify({'success': False, 'error': 'Target directory already exists'})
        
        print(f"Renaming directory: {old_path} -> {new_path}")
        
        # Rename directory
        os.rename(old_path, new_path)
        
        # Update user session related states
        if hasattr(user_session, 'selected_output_dir') and user_session.selected_output_dir == old_name:
            user_session.selected_output_dir = new_name_safe
        if hasattr(user_session, 'last_output_dir') and user_session.last_output_dir == old_name:
            user_session.last_output_dir = new_name_safe
        
        print(f"Successfully renamed directory: {old_name} -> {new_name_safe}")
        
        return jsonify({
            'success': True, 
            'message': f'Directory renamed successfully: {old_name} -> {new_name_safe}',
            'old_name': old_name,
            'new_name': new_name_safe
        })
        
    except Exception as e:
        print(f"Failed to rename directory: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/delete-directory/<path:dir_name>', methods=['DELETE'])
def delete_directory(dir_name):
    """Delete specified output directory"""
    try:
        # Get API key from query parameters or headers
        api_key = request.args.get('api_key') or request.headers.get('X-API-Key')
        
        # Create a temporary session for API calls
        temp_session_id = f"api_{request.remote_addr}_{id(request)}"
        user_session = gui_instance.get_user_session(temp_session_id, api_key)
        user_base_dir = user_session.get_user_directory(gui_instance.base_data_dir)
        
        # Security check directory name
        safe_dir_name = secure_filename(dir_name)
        target_dir = os.path.join(user_base_dir, safe_dir_name)
        
        # Security check: ensure directory is within user's output directory
        real_output_dir = os.path.realpath(user_base_dir)
        real_target_dir = os.path.realpath(target_dir)
        if not real_target_dir.startswith(real_output_dir):
            return jsonify({'success': False, 'error': 'Access denied: Invalid directory path'})
        
        # Check if directory exists
        if not os.path.exists(target_dir):
            return jsonify({'success': False, 'error': f'Directory not found: {dir_name}'})
        
        # Check if directory contains workspace subdirectory (ensure it's a workspace directory)
        workspace_path = os.path.join(target_dir, 'workspace')
        if not os.path.exists(workspace_path) or not os.path.isdir(workspace_path):
            return jsonify({'success': False, 'error': 'Only directories with workspace subdirectory can be deleted'})
        
        # Check if it's currently executing directory for any user with same API key
        if hasattr(user_session, 'current_output_dir') and user_session.current_output_dir == dir_name:
            return jsonify({'success': False, 'error': 'Cannot delete currently executing directory'})
        
        print(f"Deleting directory: {target_dir}")
        
        # Delete directory and all its contents
        shutil.rmtree(target_dir)
        
        # Clean user session related states
        if hasattr(user_session, 'last_output_dir') and user_session.last_output_dir == dir_name:
            user_session.last_output_dir = None
        if hasattr(user_session, 'selected_output_dir') and user_session.selected_output_dir == dir_name:
            user_session.selected_output_dir = None
        
        print(f"Successfully deleted directory: {dir_name}")
        
        return jsonify({
            'success': True, 
            'message': f'Directory "{dir_name}" has been successfully deleted'
        })
        
    except PermissionError as e:
        print(f"Permission error deleting directory {dir_name}: {str(e)}")
        return jsonify({'success': False, 'error': f'Permission denied: {str(e)}'})
    except Exception as e:
        print(f"Error deleting directory {dir_name}: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})



if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    socketio.run(app, host='0.0.0.0', port=port, debug=False) 