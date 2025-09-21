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

from flask import Flask, render_template, request, jsonify, send_file, send_from_directory, after_this_request
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
import time
import json
import psutil
from collections import defaultdict
from threading import Lock, Semaphore
import argparse


# Determine template and static directories FIRST - always relative to this app.py file
# Get the directory where app.py is located (before any directory changes)
app_dir = os.path.dirname(os.path.abspath(__file__))
template_dir = os.path.join(app_dir, 'templates')
static_dir = os.path.join(app_dir, 'static')

# Add parent directory to path to import config_loader
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config_loader import get_language, get_gui_default_data_directory
from auth_manager import AuthenticationManager

# Import Mermaid processor

try:
    from src.tools.mermaid_processor import mermaid_processor
    MERMAID_PROCESSOR_AVAILABLE = True
except ImportError:
    print("⚠️ Mermaid processor not available")
    MERMAID_PROCESSOR_AVAILABLE = False

# Import SVG optimizers
try:
    from src.utils.advanced_svg_optimizer import AdvancedSVGOptimizer, OptimizationLevel
    SVG_OPTIMIZER_AVAILABLE = True
except ImportError:
    print("⚠️ Advanced SVG optimizer not available")
    SVG_OPTIMIZER_AVAILABLE = False

try:
    from src.utils.llm_svg_optimizer import create_llm_optimizer_from_env
    LLM_SVG_OPTIMIZER_AVAILABLE = True
except ImportError:
    print("⚠️ LLM SVG optimizer not available")
    LLM_SVG_OPTIMIZER_AVAILABLE = False

# Check current directory, switch to parent directory if in GUI directory
current_dir = os.getcwd()
current_dir_name = os.path.basename(current_dir)

if current_dir_name == 'GUI':
    parent_dir = os.path.dirname(current_dir)
    os.chdir(parent_dir)
else:
    pass

# Add parent directory to path to import main.py
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Application name macro definition
APP_NAME = "AGI Agent"

from src.main import AGIAgentMain




# Concurrency control and performance monitoring class
class ConcurrencyManager:
    """Concurrency Control and Performance Monitoring Manager"""
    
    def __init__(self, max_concurrent_tasks=16, max_connections=40, task_timeout=3600):  # 60 minute timeout (Expand by 1x)
        self.max_concurrent_tasks = max_concurrent_tasks
        self.max_connections = max_connections
        self.task_timeout = task_timeout  # 任务超时时间（Seconds）
        
        # Concurrency control
        self.task_semaphore = Semaphore(max_concurrent_tasks)
        self.active_tasks = {}  # session_id -> task_info
        self.task_queue = queue.Queue()  # Task queuing
        self.connection_count = 0
        self.lock = Lock()
        

        
        # Performance monitoring
        self.metrics = {
            'total_tasks': 0,
            'completed_tasks': 0,
            'failed_tasks': 0,
            'avg_task_duration': 0.0,
            'active_connections': 0,
            'peak_memory_usage': 0.0,
            'last_updated': time.time()
        }
        
        # Resource monitoring
        self.resource_monitor_active = True
        self.resource_thread = threading.Thread(target=self._monitor_resources, daemon=True)
        self.resource_thread.start()
        
        # Timeout monitoring
        self.timeout_monitor_active = True
        self.timeout_thread = threading.Thread(target=self._monitor_timeouts, daemon=True)
        self.timeout_thread.start()
        

    
    def can_accept_connection(self):
        """Check if new connections can be accepted"""
        with self.lock:
            return self.connection_count < self.max_connections
    
    def add_connection(self):
        """Add connection"""
        with self.lock:
            if self.connection_count < self.max_connections:
                self.connection_count += 1
                self.metrics['active_connections'] = self.connection_count
                return True
            return False
    
    def remove_connection(self):
        """Remove connection"""
        with self.lock:
            if self.connection_count > 0:
                self.connection_count -= 1
                self.metrics['active_connections'] = self.connection_count
    
    def can_start_task(self, session_id):
        """Check if new tasks can be started"""
        # Non-blocking check semaphore
        acquired = self.task_semaphore.acquire(blocking=False)
        if acquired:
            with self.lock:
                self.active_tasks[session_id] = {
                    'start_time': time.time(),
                    'status': 'running'
                }
                self.metrics['total_tasks'] += 1
            return True
        return False
    
    def finish_task(self, session_id, success=True):
        """Complete task"""
        self.task_semaphore.release()
        
        with self.lock:
            if session_id in self.active_tasks:
                task_info = self.active_tasks.pop(session_id)
                duration = time.time() - task_info['start_time']
                
                if success:
                    self.metrics['completed_tasks'] += 1
                else:
                    self.metrics['failed_tasks'] += 1
                
                # Update average execution time
                total_completed = self.metrics['completed_tasks'] + self.metrics['failed_tasks']
                if total_completed > 0:
                    current_avg = self.metrics['avg_task_duration']
                    self.metrics['avg_task_duration'] = (current_avg * (total_completed - 1) + duration) / total_completed
    
    def get_metrics(self):
        """Get performance metrics"""
        with self.lock:
            metrics_copy = self.metrics.copy()
            metrics_copy['active_tasks'] = len(self.active_tasks)
            metrics_copy['queue_size'] = self.task_queue.qsize()
            return metrics_copy
    
    def _monitor_resources(self):
        """Resource monitoring thread"""
        while self.resource_monitor_active:
            try:
                # Monitor memory usage
                process = psutil.Process()
                memory_info = process.memory_info()
                memory_mb = memory_info.rss / 1024 / 1024
                
                with self.lock:
                    if memory_mb > self.metrics['peak_memory_usage']:
                        self.metrics['peak_memory_usage'] = memory_mb
                    self.metrics['last_updated'] = time.time()
                
                time.sleep(30)  # Monitor every 30 seconds
            except Exception as e:
                print(f"⚠️ Resource monitoring error: {e}")
                time.sleep(60)
    
    def _monitor_timeouts(self):
        """Timeout monitoring thread"""
        while self.timeout_monitor_active:
            try:
                current_time = time.time()
                timeout_sessions = []
                
                with self.lock:
                    for session_id, task_info in self.active_tasks.items():
                        if current_time - task_info['start_time'] > self.task_timeout:
                            timeout_sessions.append(session_id)
                
                # Handle timeout tasks
                for session_id in timeout_sessions:
                    print(f"⏰ Task timeout detected for user {session_id}")
                    self._handle_task_timeout(session_id)
                

                
                time.sleep(60)  # Check timeout every minute
            except Exception as e:
                print(f"⚠️ Timeout monitoring error: {e}")
                time.sleep(120)
    
    def _handle_task_timeout(self, session_id):
        """Handle task timeout"""
        # This method needs to set callback after GUI instance initialization
        if hasattr(self, '_timeout_callback') and self._timeout_callback:
            self._timeout_callback(session_id)
        else:
            print(f"⚠️ Timeout handling callback not set: {session_id}")
    
    def set_timeout_callback(self, callback):
        """Set timeout handling callback"""
        self._timeout_callback = callback
    

    
    def get_task_runtime(self, session_id):
        """Get task running time"""
        with self.lock:
            if session_id in self.active_tasks:
                return time.time() - self.active_tasks[session_id]['start_time']
            return 0
    

    
    def stop(self):
        """Stop monitoring"""
        self.resource_monitor_active = False
        self.timeout_monitor_active = False



app = Flask(__name__, template_folder=template_dir, static_folder=static_dir)
app.config['SECRET_KEY'] = f'{APP_NAME.lower().replace(" ", "_")}_gui_secret_key'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading', 
                   ping_timeout=3600, ping_interval=60)  # 1小时超时，1分钟心跳

# 修复 werkzeug write() before start_response 错误
#import logging
#logging.getLogger('werkzeug').setLevel(logging.CRITICAL)

I18N_TEXTS = {
    'zh': {
        # Page title and basic information
        'page_title': f'{APP_NAME}',
        'app_title': f'{APP_NAME}',
        'app_subtitle': '',
        'connected': f'已连接到 {APP_NAME}',
        
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
        'clear_chat': '清扫',
        
        # Button tooltips
        'direct_tooltip': '直接执行 - 不进行任务分解',
        'plan_tooltip': '计划模式 - 先分解任务再执行',
        'new_tooltip': '新建目录 - 创建新的工作目录',
        'refresh_tooltip': '刷新目录列表',
        'upload_tooltip': '上传文件到Workspace',
        'download_tooltip': '下载目录为ZIP（排除code_index）',
        'rename_tooltip': '重命名目录',
        'delete_tooltip': '删除目录',
        'clear_chat_tooltip': '清空日志显示区域和历史对话',
        
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
        'chat_cleared': '日志和历史对话已清空',
        'confirm_clear_chat': '确定要清空所有日志和历史对话吗？此操作不可撤销。',
        
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
        
        # PDF preview
        'pdf_pages': '共 {pages} 页',
        'pdf_pages_simple': '共 {pages} 页 (简化模式)',
        'download_pdf': '下载PDF',
        'pdf_loading': '正在加载所有页面...',
        'pdf_render_error': 'PDF页面渲染失败',
        
        # Delete warnings
        'delete_current_executing_warning': '⚠️ 警告：这是当前正在执行的目录！',
        'delete_selected_warning': '⚠️ 警告：这是当前选择的目录！',
        
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
        'image': '图片',
        'dimensions': '尺寸',
        'total_rows': '总行数',
        'columns': '列数',
        
        # Configuration options
        'config_options': '配置选项',
        'show_config_options': '显示配置选项',
        'hide_config_options': '隐藏配置选项',
        'routine_file': '指导文件',
        'no_routine': '无',
        'enable_web_search': '搜索网络',
        'enable_multi_agent': '启动多智能体',
        'enable_long_term_memory': '启动长期记忆',
        'enable_mcp': 'MCP工具配置',
        'enable_jieba': '启用中文分词',
        
        # Others
        'deleting': '删除中...',
        'renaming': '重命名中...',
        'uploading': '上传中...',
        'edit_mermaid_placeholder': '编辑Mermaid内容...',
        'convert_to_images': '将mermaid转换为PNG和SVG图像',
        'convert_to_images_short': '转换为图像',
        'loading': '加载中...',
        'system_message': '系统消息',
        'welcome_message': f'欢迎使用 {APP_NAME}！请在下方输入您的需求，系统将自动为您处理任务。',
        'workspace_title': '工作目录',
        'file_preview': '文件预览',
        'data_directory_info': '数据目录',
        'disconnected': '与服务器断开连接',
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
        'upload_completed': '上传文档已完成',
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
        'user_connection_failed': '连接失败',
        'default_user': '默认用户',
        'user_prefix': '用户',
        'guest_user': '访客用户',
        'temporary_connection': '临时连接',
        'auto_login_from_url': '已通过URL参数自动登录',
        'session_restored': '已恢复上次登录会话',
        
        # Model selection
        'model_label': '模型:',
        'model_tooltip': '选择要使用的AI模型',
        'model_claude_sonnet': 'claude-sonnet-4-0 (高精度)',
        'model_gpt_4': 'gpt-4.1 (高效率)',
        'config_error_title': '配置错误',
        'config_error_invalid_key': 'API Key配置无效，请检查config/config.txt文件中的GUI API configuration部分',
        
        # Custom model config dialog
        'custom_config_title': '自定义模型配置',
        'custom_api_key_label': 'API Key:',
        'custom_api_base_label': 'API Base URL:',
        'custom_model_label': '模型名称:',
        'custom_max_tokens_label': 'Max Output Tokens:',
        'custom_api_key_placeholder': '请输入API Key',
        'custom_api_base_placeholder': '请输入API Base URL（如：https://api.example.com/v1）',
        'custom_model_placeholder': '请输入模型名称（如：gpt-4）',
        'custom_max_tokens_placeholder': '请输入最大输出token数量（默认：8192）',
        'custom_config_save': '保存配置',
        'custom_config_cancel': '取消',
        'custom_config_required': '所有字段都是必填的',
        
        # Additional UI elements
        'new_messages': '条新消息',
        'auto_scrolling': '自动滚动',
        'uploading': '上传中...',
        'running_input_placeholder': '任务执行中，您可以输入新需求（等待当前任务完成后执行）...',
        'reload': '重新加载',
        'save': '保存',
        'type_label': '类型',
        'language': '语言',
        'image': '图片',
        'dimensions': '尺寸',
        'total_rows': '总行数',
        'columns': '列数',
        'preview': '预览',
        'office_preview_title': 'Office文档预览',
        'office_download_instruction': 'Office文档需要下载到本地查看：',
        'download_file': '下载文件',
        'usage_instructions': '使用说明',
        'office_instruction_1': '点击"下载文件"按钮将文件保存到本地',
        'office_instruction_2': '使用Microsoft Office、WPS或其他兼容软件打开',
        'office_instruction_3': '支持.doc、.docx、.xls、.xlsx、.ppt、.pptx等格式',
        'office_offline_note': '为了支持离线部署，云存储预览功能已被移除。请下载文件到本地查看。',
        'source_mode': '源码模式',
        'preview_mode': '预览模式',
        'save_markdown_title': '保存当前Markdown文本',
        'save_mermaid_title': '保存当前Mermaid文件',
        'toggle_to_preview_title': '切换到预览模式',
        'toggle_to_source_title': '切换到源码模式',
        
        # Mermaid conversion
        'mermaid_conversion_completed': 'Mermaid图表转换完成',
        'mermaid_svg_png_format': '（SVG和PNG格式）',
        'mermaid_svg_only': '（仅SVG格式）',
        'mermaid_png_only': '（仅PNG格式）',
        
        # Configuration validation
        'config_missing': '模型配置信息缺失',
        'config_incomplete': '配置信息不完整：缺少 API Key、API Base 或模型名称',
        'custom_label': '自定义',
        'task_starting': '🚀 任务开始执行...',
        
        # Directory status messages
        'no_workspace_directories': '暂无工作目录（包含workspace子目录的目录）',
        'current_executing': '当前执行',
        'selected': '已选择',
        'last_used': '上次使用',
        'expand_collapse': '展开/收起',
        'upload_to_workspace': '上传文件到Workspace',
        'download_as_zip': '下载目录为ZIP（排除code_index）',
        'rename_directory': '重命名目录',
        'delete_directory': '删除目录',
        'confirm_delete_directory': '确定要删除目录',
        'delete_warning': '此操作不可撤销，将永久删除该目录及其所有内容。',
        'guest_cannot_execute': 'guest用户为演示账户，无法执行新任务。',
        'guest_cannot_create': 'guest用户为演示账户，无法创建新目录。',
        'guest_cannot_delete': 'guest用户为演示账户，无法删除目录。',
        'guest_cannot_save': 'guest用户为演示账户，无法保存。',
        'guest_cannot_convert': 'guest用户为演示账户，无法转换图表。',
        'guest_cannot_rename': 'guest用户为演示账户，无法重命名目录。',
        'guest_cannot_upload': 'guest用户为演示账户，无法上传文件。',
        'select_valid_config': '请选择有效的模型配置',
        'config_validation_failed': '配置验证失败，请检查网络连接',
    },
    'en': {
        # Page title and basic info
        'page_title': f'{APP_NAME}',
        'app_title': f'{APP_NAME}', 
        'app_subtitle': '',
        'connected': f'Connected to {APP_NAME}',
        
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
        'clear_chat': 'Clean',
        
        # Button tooltips
        'direct_tooltip': 'Direct execution - no task decomposition',
        'plan_tooltip': 'Plan mode - decompose tasks before execution',
        'new_tooltip': 'New directory - create new workspace',
        'refresh_tooltip': 'Refresh directory list',
        'upload_tooltip': 'Upload files to Workspace',
        'download_tooltip': 'Download directory as ZIP (excluding code_index)',
        'rename_tooltip': 'Rename directory',
        'delete_tooltip': 'Delete directory',
        'clear_chat_tooltip': 'Clear chat log and conversation history',
        
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
        'chat_cleared': 'Chat log and conversation history cleared',
        'confirm_clear_chat': 'Are you sure you want to clear all chat logs and conversation history? This operation cannot be undone.',
        
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
        
        # PDF preview
        'pdf_pages': 'Total {pages} pages',
        'pdf_pages_simple': 'Total {pages} pages (Simple mode)',
        'download_pdf': 'Download PDF',
        'pdf_loading': 'Loading all pages...',
        'pdf_render_error': 'PDF page rendering failed',
        
        # Delete warnings
        'delete_current_executing_warning': '⚠️ Warning: This is the currently executing directory!',
        'delete_selected_warning': '⚠️ Warning: This is the currently selected directory!',
        
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
        'image': 'Image',
        'dimensions': 'Dimensions',
        'total_rows': 'Total Rows',
        'columns': 'Columns',
        
        # Configuration options
        'config_options': 'Configuration Options',
        'show_config_options': 'Show Configuration',
        'hide_config_options': 'Hide Configuration',
        'routine_file': 'Routine File',
        'no_routine': 'None',
        'enable_web_search': 'Web Search',
        'enable_multi_agent': 'Multi-Agent',
        'enable_long_term_memory': 'Long-term Memory',
        'enable_mcp': 'Enable MCP',
        'enable_jieba': 'Chinese Segmentation',
        
        # Others
        'deleting': 'Deleting...',
        'renaming': 'Renaming...',
        'uploading': 'Uploading...',
        'edit_mermaid_placeholder': 'Edit Mermaid content...',
        'convert_to_images': 'Convert Mermaid to PNG and SVG images',
        'convert_to_images_short': 'Convert to Images',
        'loading': 'Loading...',
        'system_message': 'System Message',
        'welcome_message': f'Welcome to {APP_NAME}! Please enter your requirements below, and the system will automatically process tasks for you.',
        'workspace_title': 'Workspace',
        'file_preview': 'File Preview',
        'data_directory_info': 'Data Directory',
        'disconnected': 'Disconnected from server',
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
        'upload_completed': 'Upload completed',
        'upload_failed_http': 'Upload failed: HTTP {0}',
        
        # Directory operations
        'directory_created_with_workspace': 'New workspace directory created: {0} (with workspace subdirectory)',
        'directory_list_refreshed': 'Directory list refreshed',
        'no_files_selected': 'No files selected',
        'no_valid_files': 'No valid files selected',
        'target_directory_not_exist': 'Target directory does not exist',
        'upload_success': 'Successfully uploaded {0} files',
        'new_name_empty': 'New name cannot be empty',
        
        # Multi-user support
        'api_key_label': 'API Key:',
        'api_key_placeholder': 'Enter API Key (optional)',
        'api_key_tooltip': 'Enter your API Key, leave empty for default user mode',
        'connect_btn': 'Connect',
        'disconnect_btn': 'Disconnect',
        'connecting': 'Connecting...',
        'user_connected': 'Connected',
        'user_disconnected': 'Disconnected',
        'user_connection_failed': 'Connection Failed',
        'default_user': 'Default User',
        'user_prefix': 'User',
        'guest_user': 'Guest User',
        'temporary_connection': 'Temporary Connection',
        'auto_login_from_url': 'Auto-logged in via URL parameter',
        'session_restored': 'Previous login session restored',
        
        # Model selection
        'model_label': 'Model:',
        'model_tooltip': 'Select AI model to use',
        'model_claude_sonnet': 'claude-sonnet-4-0 (High Accuracy)',
        'model_gpt_4': 'gpt-4.1 (High Efficiency)',
        'config_error_title': 'Configuration Error',
        'config_error_invalid_key': 'Invalid API Key configuration, please check GUI API configuration in config/config.txt',
        
        # Custom model config dialog
        'custom_config_title': 'Custom Model Configuration',
        'custom_api_key_label': 'API Key:',
        'custom_api_base_label': 'API Base URL:',
        'custom_model_label': 'Model Name:',
        'custom_max_tokens_label': 'Max Output Tokens:',
        'custom_api_key_placeholder': 'Enter API Key',
        'custom_api_base_placeholder': 'Enter API Base URL (e.g., https://api.example.com/v1)',
        'custom_model_placeholder': 'Enter model name (e.g., gpt-4)',
        'custom_max_tokens_placeholder': 'Enter max output tokens (default: 8192)',
        'custom_config_save': 'Save Configuration',
        'custom_config_cancel': 'Cancel',
        'custom_config_required': 'All fields are required',
        
        # Additional UI elements
        'new_messages': 'new messages',
        'auto_scrolling': 'Auto Scroll',
        'uploading': 'Uploading...',
        'running_input_placeholder': 'Task is running. You can type a new request (will execute after current task)...',
        'reload': 'Reload',
        'save': 'Save',
        'type_label': 'Type',
        'language': 'Language',
        'image': 'Image',
        'dimensions': 'Dimensions',
        'total_rows': 'Total Rows',
        'columns': 'Columns',
        'preview': 'Preview',
        'office_preview_title': 'Office Document Preview',
        'office_download_instruction': 'Office documents need to be downloaded for local viewing:',
        'download_file': 'Download File',
        'usage_instructions': 'Usage Instructions',
        'office_instruction_1': 'Click the "Download File" button to save the file locally',
        'office_instruction_2': 'Open with Microsoft Office, WPS, or other compatible software',
        'office_instruction_3': 'Supports .doc, .docx, .xls, .xlsx, .ppt, .pptx and other formats',
        'office_offline_note': 'To support offline deployment, cloud storage preview functionality has been removed. Please download files for local viewing.',
        'source_mode': 'Source Mode',
        'preview_mode': 'Preview Mode',
        'save_markdown_title': 'Save current Markdown text',
        'save_mermaid_title': 'Save current Mermaid file',
        'toggle_to_preview_title': 'Switch to preview mode',
        'toggle_to_source_title': 'Switch to source mode',
        
        # Mermaid conversion
        'mermaid_conversion_completed': 'Mermaid chart conversion completed',
        'mermaid_svg_png_format': ' (SVG and PNG formats)',
        'mermaid_svg_only': ' (SVG format only)',
        'mermaid_png_only': ' (PNG format only)',
        
        # Configuration validation
        'config_missing': 'Model configuration information missing',
        'config_incomplete': 'Incomplete configuration: missing API Key, API Base, or model name',
        'custom_label': 'Custom',
        'task_starting': '🚀 Task starting...',
        
        # Directory status messages
        'no_workspace_directories': 'No workspace directories (directories containing workspace subdirectories)',
        'current_executing': 'Currently Executing',
        'selected': 'Selected',
        'last_used': 'Last Used',
        'expand_collapse': 'Expand/Collapse',
        'upload_to_workspace': 'Upload Files to Workspace',
        'download_as_zip': 'Download Directory as ZIP (excluding code_index)',
        'rename_directory': 'Rename Directory',
        'delete_directory': 'Delete Directory',
        'confirm_delete_directory': 'Are you sure you want to delete directory',
        'delete_warning': 'This operation cannot be undone and will permanently delete the directory and all its contents.',
        'guest_cannot_execute': 'Guest user is a demo account and cannot execute new tasks.',
        'guest_cannot_create': 'Guest user is a demo account and cannot create new directories.',
        'guest_cannot_delete': 'Guest user is a demo account and cannot delete directories.',
        'guest_cannot_save': 'Guest user is a demo account and cannot save.',
        'guest_cannot_convert': 'Guest user is a demo account and cannot convert charts.',
        'guest_cannot_rename': 'Guest user is a demo account and cannot rename directories.',
        'guest_cannot_upload': 'Guest user is a demo account and cannot upload files.',
        'select_valid_config': 'Please select a valid model configuration',
        'config_validation_failed': 'Configuration validation failed, please check network connection',
    }
}

def get_i18n_texts():
    """Get internationalization text for current language"""
    current_lang = get_language()
    return I18N_TEXTS.get(current_lang, I18N_TEXTS['en'])

def execute_agia_task_process_target(user_requirement, output_queue, out_dir=None, continue_mode=False, plan_mode=False, gui_config=None, session_id=None, detailed_requirement=None, user_id=None):
    # Get i18n texts for this process
    i18n = get_i18n_texts()
    """
    This function runs in a separate process.
    It cannot use the `socketio` object directly.
    It communicates back to the main process via the queue.
    """
    try:
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
        
        # Process GUI configuration options
        if gui_config is None:
            gui_config = {}
        
        # Set default values based on user requirements
        enable_web_search = gui_config.get('enable_web_search', True)
        enable_multi_agent = gui_config.get('enable_multi_agent', False)
        enable_long_term_memory = gui_config.get('enable_long_term_memory', True)  # Default selection
        enable_mcp = gui_config.get('enable_mcp', False)
        enable_jieba = gui_config.get('enable_jieba', True)  # Default selection
        
        # Routine file configuration from GUI
        routine_file = gui_config.get('routine_file')
        if routine_file:
            # 检查是否是workspace文件（以routine_开头）
            if routine_file.startswith('routine_'):
                # 直接使用workspace根目录下的文件
                routine_file = os.path.join(os.getcwd(), routine_file)
            else:
                # 根据语言配置选择routine文件夹
                current_lang = get_language()
                if current_lang == 'zh':
                    routine_file = os.path.join(os.getcwd(), 'routine_zh', routine_file)
                else:
                    routine_file = os.path.join(os.getcwd(), 'routine', routine_file)
            
            if not os.path.exists(routine_file):
                output_queue.put({'event': 'output', 'data': {'message': f"Warning: Routine file not found: {routine_file}", 'type': 'warning'}})
                routine_file = None

        # Model configuration from GUI
        selected_model = gui_config.get('selected_model', 'claude-sonnet-4')
        model_api_key = gui_config.get('model_api_key')
        model_api_base = gui_config.get('model_api_base')
        

        
        # Create a temporary configuration that overrides config.txt for GUI mode
        # We'll use environment variables to pass these settings to the AGIAgent system
        original_env = {}
        
        # Model configuration: GUI setting overrides config.txt
        if model_api_key:
            original_env['AGIBOT_API_KEY'] = os.environ.get('AGIBOT_API_KEY', '')
            os.environ['AGIBOT_API_KEY'] = model_api_key
        if model_api_base:
            original_env['AGIBOT_API_BASE'] = os.environ.get('AGIBOT_API_BASE', '')
            os.environ['AGIBOT_API_BASE'] = model_api_base
        if selected_model:
            original_env['AGIBOT_MODEL'] = os.environ.get('AGIBOT_MODEL', '')
            os.environ['AGIBOT_MODEL'] = selected_model
        
        # Web search: only set if GUI enables it
        if enable_web_search:
            original_env['AGIBOT_WEB_SEARCH'] = os.environ.get('AGIBOT_WEB_SEARCH', '')
            os.environ['AGIBOT_WEB_SEARCH'] = 'true'
        
        # Multi-agent: GUI setting overrides config.txt (set environment variable explicitly)
        original_env['AGIBOT_MULTI_AGENT'] = os.environ.get('AGIBOT_MULTI_AGENT', '')
        if enable_multi_agent:
            os.environ['AGIBOT_MULTI_AGENT'] = 'true'
        else:
            os.environ['AGIBOT_MULTI_AGENT'] = 'false'
        
        # Jieba: GUI setting overrides config.txt (set environment variable explicitly)
        original_env['AGIBOT_ENABLE_JIEBA'] = os.environ.get('AGIBOT_ENABLE_JIEBA', '')
        if enable_jieba:
            os.environ['AGIBOT_ENABLE_JIEBA'] = 'true'
        else:
            os.environ['AGIBOT_ENABLE_JIEBA'] = 'false'
        
        # Long-term memory: GUI setting overrides config.txt (set environment variable explicitly)
        original_env['AGIBOT_LONG_TERM_MEMORY'] = os.environ.get('AGIBOT_LONG_TERM_MEMORY', '')
        if enable_long_term_memory:
            os.environ['AGIBOT_LONG_TERM_MEMORY'] = 'true'
        else:
            os.environ['AGIBOT_LONG_TERM_MEMORY'] = 'false'
        
        # Set parameters based on mode
        if plan_mode:
            single_task_mode = False  # Plan mode uses task decomposition
        else:
            single_task_mode = True   # Default mode executes directly
        
        # Determine MCP config file based on GUI setting
        mcp_config_file = None
        if enable_mcp:
            # Get selected MCP servers from GUI config
            selected_mcp_servers = gui_config.get('selected_mcp_servers', [])

            if selected_mcp_servers:
                # Generate custom MCP config file based on selected servers
                mcp_config_file = generate_custom_mcp_config(selected_mcp_servers, out_dir)
            else:
                # Use default MCP config if no servers selected
                mcp_config_file = "config/mcp_servers.json"
        
        agia = AGIAgentMain(
            out_dir=out_dir,
            debug_mode=False,
            detailed_summary=True,
            single_task_mode=single_task_mode,  # Set based on plan_mode
            interactive_mode=False,  # Disable interactive mode
            continue_mode=False,  # Always use False for GUI mode to avoid shared .agia_last_output.json
            MCP_config_file=mcp_config_file,  # Set based on GUI MCP option
            user_id=user_id,  # Pass user ID for MCP knowledge base tools
            routine_file=routine_file  # Pass routine file to main application
        )
        
        # Use detailed_requirement if provided (contains conversation history)
        base_requirement = detailed_requirement if detailed_requirement else user_requirement
        
        # Helper function to format file size
        def format_size(size_bytes):
            """Format file size"""
            if size_bytes == 0:
                return "0 B"
            size_names = ["B", "KB", "MB", "GB", "TB"]
            i = 0
            while size_bytes >= 1024.0 and i < len(size_names) - 1:
                size_bytes /= 1024.0
                i += 1
            return f"{size_bytes:.1f} {size_names[i]}"
        
        # Add workspace path information to the prompt
        workspace_info = ""
        if out_dir:
            # Display user-selected directory path
            workspace_info = f"\n\nCurrently selected directory: {out_dir}"
            
            # Check workspace subdirectory
            workspace_dir = os.path.join(out_dir, "workspace")
            if os.path.exists(workspace_dir):
                workspace_info += f"\nworkspace subdirectory path: {workspace_dir}\nworkspace subdirectory content:"
                try:
                    # List workspace contents for context
                    workspace_files = []
                    md_files = []
                    for root, dirs, files in os.walk(workspace_dir):
                        for file in files:
                            file_path = os.path.join(root, file)
                            rel_path = os.path.relpath(file_path, workspace_dir)
                            file_size = os.path.getsize(file_path)
                            
                            if file.endswith('.md'):
                                md_files.append(f"  - {rel_path} ({format_size(file_size)})")
                            else:
                                workspace_files.append(f"  - {rel_path} ({format_size(file_size)})")
                    
                    # Prioritize displaying MD files
                    if md_files:
                        workspace_info += "\nMD files:"
                        workspace_info += "\n" + "\n".join(md_files)
                    
                    if workspace_files:
                        workspace_info += "\nOther files:"
                        workspace_info += "\n" + "\n".join(workspace_files)
                    
                    if not md_files and not workspace_files:
                        workspace_info += "\n  (Empty directory)"
                        
                except Exception as e:
                    workspace_info += f"\n  (Cannot read directory content: {str(e)})"
            else:
                workspace_info += f"\nNote: workspace subdirectory does not exist"
        
        # Add search configuration hints to the prompt based on GUI settings
        search_hints = []
        if not enable_web_search:
            search_hints.append("[Don't search network]")
        
        # Combine base requirement with workspace info and search hints
        requirement_parts = []
        if search_hints:
            requirement_parts.append(' '.join(search_hints))
        requirement_parts.append(base_requirement)
        if workspace_info:
            requirement_parts.append(workspace_info)
        
        final_requirement = ' '.join(requirement_parts)
        
        # Send user requirement as separate message
        output_queue.put({'event': 'output', 'data': {'message': f"User requirement: {user_requirement}", 'type': 'user'}})
        
        # Send task_started event to update UI buttons
        output_queue.put({'event': 'task_started', 'data': {'message': i18n.get('task_starting', '🚀 Task starting...')}})
        
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
            
            def should_filter_message(self, line):
                """Filter out redundant system messages that are already displayed in GUI"""
                # Don't filter error messages, warnings, or important notifications
                line_lower = line.lower()
                if any(keyword in line_lower for keyword in ['error', 'warning', 'failed', 'exception', 'traceback']):
                    return False
                
                # List of message patterns to filter out (only redundant status messages)
                filter_patterns = [
                    "Received user requirement:",
                    "Currently selected directory:",
                    "workspace subdirectory path:",
                    "workspace subdirectory content:",
                    "Note: workspace subdirectory does not exist",
                    "With conversation context included",
                    "(Empty directory)",
                    "(Cannot read directory content:",
                    "MD files:",
                    "Other files:"
                ]
                
                # Check if line matches any filter pattern
                for pattern in filter_patterns:
                    if pattern in line:
                        return True
                
                # Filter file list items that start with "  - " but only if they look like file paths
                if line.strip().startswith("- ") and ("(" in line and ")" in line):
                    return True
                
                # Also filter empty lines and lines with only whitespace/special chars
                if not line.strip() or line.strip() in ['', '---', '===', '***']:
                    return True
                    
                return False
            
            def write(self, message):
                self.buffer += message
                if '\n' in self.buffer:
                    *lines, self.buffer = self.buffer.split('\n')
                    for line in lines:
                        if line.strip():
                            # Filter code_edit content for GUI display
                            filtered_line = self.filter_code_edit_content(line.strip())
                            
                            # Filter out redundant system messages that are already displayed in GUI
                            if self.should_filter_message(filtered_line):
                                continue
                            
                            # Check if it's warning or progress info, if so display as normal info instead of error
                            line_lower = filtered_line.lower()
                            if ('warning' in line_lower or
                                'progress' in line_lower or
                                'processing files' in line_lower or
                                filtered_line.startswith('Processing files:') or
                                'userwarning' in line_lower or
                                'warnings.warn' in line_lower or
                                '⚠️' in filtered_line or  # 中文警告符号
                                filtered_line.startswith('W: ') or  # apt warning format
                                'W: ' in filtered_line):  # apt warning format
                                message_type = 'info'
                            else:
                                message_type = self.socket_type
                            # Display warning and progress info as normal info
                            self.q.put({'event': 'output', 'data': {'message': filtered_line, 'type': message_type}})

            def flush(self):
                pass
            
            def final_flush(self):
                if self.buffer.strip():
                    # Filter out redundant system messages
                    if self.should_filter_message(self.buffer.strip()):
                        self.buffer = ""
                        return
                    
                    # Check if it's warning or progress info, if so display as normal info instead of error
                    buffer_lower = self.buffer.lower()
                    if ('warning' in buffer_lower or
                        'progress' in buffer_lower or
                        'processing files' in buffer_lower or
                        self.buffer.strip().startswith('Processing files:') or
                        'userwarning' in buffer_lower or
                        'warnings.warn' in buffer_lower or
                        '⚠️' in self.buffer or  # 中文警告符号
                        self.buffer.strip().startswith('W: ') or  # apt warning format
                        'W: ' in self.buffer):  # apt warning format
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
            
            success = agia.run(user_requirement=final_requirement, loops=50)
            
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

class AGIAgentGUI:
    def __init__(self):
        # User session management
        self.user_sessions = {}  # session_id -> UserSession
        
        # Initialize authentication manager
        self.auth_manager = AuthenticationManager()
        
        # Initialize concurrency manager
        self.concurrency_manager = ConcurrencyManager(
            max_concurrent_tasks=16,  # Maximum concurrent tasks (Expand by 1x)
            max_connections=40        # 最大Connect数 (Expand by 1x)
        )
        
        # Get GUI default data directory from config, fallback to current directory
        config_data_dir = get_gui_default_data_directory()
        if config_data_dir:
            self.base_data_dir = config_data_dir
        else:
            self.base_data_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        # Ensure base directory exists
        os.makedirs(self.base_data_dir, exist_ok=True)
        
        # Don't create default userdata directory until needed
        self.default_user_dir = os.path.join(self.base_data_dir, 'userdata')
        
        # Start session cleanup thread
        self.session_cleanup_active = True
        self.session_cleanup_thread = threading.Thread(target=self._cleanup_idle_sessions, daemon=True)
        self.session_cleanup_thread.start()
        
        # Set timeout handling callback
        self.concurrency_manager.set_timeout_callback(self._handle_user_task_timeout)
        

    
    def get_user_session(self, session_id, api_key=None):
        """Get or create user session with authentication"""
        # Convert empty string to None for guest access
        if api_key == "":
            api_key = None
            
        # Always authenticate (including guest access)
        auth_result = self.auth_manager.authenticate_api_key(api_key)
        if not auth_result["authenticated"]:
            print(f"🚫 Authentication failed for session {session_id}: {auth_result['error']}")
            return None
        
        # Store guest status and user info
        is_guest = auth_result.get("is_guest", False)
        user_info = auth_result["user_info"]
        
        if session_id not in self.user_sessions:
            # Create authenticated session
            if self.auth_manager.create_session(api_key, session_id):
                self.user_sessions[session_id] = UserSession(session_id, api_key, user_info)
                session_type = "guest" if is_guest else "authenticated"
                print(f"✅ Created {session_type} session for {session_id}")
            else:
                print(f"🚫 Failed to create session for {session_id}")
                return None
        else:
            # Update API key if it has changed
            existing_session = self.user_sessions[session_id]
            if existing_session.api_key != api_key:
                print(f"🔄 API key changed for session {session_id}")
                # Re-authenticate and update session
                if self.auth_manager.create_session(api_key, session_id):
                    self.user_sessions[session_id] = UserSession(session_id, api_key, user_info)
                else:
                    print(f"🚫 Failed to update session for {session_id}")
                    return None
        
        return self.user_sessions[session_id]
    
    def _cleanup_idle_sessions(self):
        """Clean up idle session thread"""
        while self.session_cleanup_active:
            try:
                current_time = time.time()
                idle_sessions = []
                
                # Check idle sessions (no activity for over 2 hours)
                for session_id, user_session in self.user_sessions.items():
                    # Check if authentication session is still valid
                    session_info = self.auth_manager.validate_session(session_id)
                    if not session_info:
                        idle_sessions.append(session_id)
                        continue
                    
                    # Check if there are running processes
                    if user_session.current_process and user_session.current_process.is_alive():
                        continue  # 有活动进程，不清理
                
                # Clean up idle sessions
                for session_id in idle_sessions:
                    print(f"🧹 Cleaning up idle session: {session_id}")
                    self._cleanup_session(session_id)
                
                # Check every 30 minutes
                time.sleep(1800)
            except Exception as e:
                print(f"⚠️ Session cleanup error: {e}")
                time.sleep(3600)  # Wait 1 hour before retrying on error
    
    def _cleanup_session(self, session_id):
        """Clean up specified session"""
        try:
            if session_id in self.user_sessions:
                user_session = self.user_sessions[session_id]
                
                # Clean up running processes
                if user_session.current_process and user_session.current_process.is_alive():
                    print(f"🛑 Terminating process for cleanup session {session_id}")
                    user_session.current_process.terminate()
                    user_session.current_process.join(timeout=5)
                
                # Clean up queue
                if user_session.output_queue:
                    try:
                        while not user_session.output_queue.empty():
                            user_session.output_queue.get_nowait()
                    except:
                        pass
                
                # Clean up session history (keep last 5)
                if len(user_session.conversation_history) > 5:
                    user_session.conversation_history = user_session.conversation_history[-5:]
                
                # Destroy authentication session
                self.auth_manager.destroy_session(session_id)
                
                # Remove user session
                del self.user_sessions[session_id]
                
                print(f"🧹 Session cleaned up: {session_id}")
        except Exception as e:
            print(f"⚠️ Session cleanup error {session_id}: {e}")
    
    def _handle_user_task_timeout(self, session_id):
        """Handle user task timeout"""
        try:
            if session_id in self.user_sessions:
                user_session = self.user_sessions[session_id]

                # Terminate process
                if user_session.current_process and user_session.current_process.is_alive():
                    print(f"🛑 Terminating timeout process for user {session_id}")
                    user_session.current_process.terminate()
                    user_session.current_process.join(timeout=10)

                    # Send timeout message to user
                    from flask_socketio import emit
                    emit('task_timeout', {
                        'message': f'Task execution timeout ({self.concurrency_manager.task_timeout}seconds)'
                    }, room=session_id)

                # Release task resources - call finish_task to clean up active_tasks
                self.concurrency_manager.finish_task(session_id, success=False)
                print(f"✅ Cleaned up timeout task for user {session_id}")
        except Exception as e:
            print(f"⚠️ Error handling user task timeout: {e}")
    

    
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

class UserSession:
    def __init__(self, session_id, api_key=None, user_info=None):
        self.session_id = session_id
        self.api_key = api_key
        self.user_info = user_info or {}
        self.current_process = None
        self.output_queue = None
        self.current_output_dir = None  # Track current execution output directory
        self.last_output_dir = None     # Track last used output directory
        self.selected_output_dir = None # Track user selected output directory
        self.conversation_history = []  # Store conversation history for this user
        
        # Determine user directory based on user info
        if user_info and user_info.get("is_guest", False):
            # Guest user gets a special directory
            self.user_dir_name = "guest"
        elif user_info and user_info.get("name"):
            # Use username as directory name, sanitize for filesystem safety
            import re
            username = user_info.get("name")
            # Remove or replace characters that are not safe for directory names
            safe_username = re.sub(r'[<>:"/\\|?*]', '_', username)
            # Remove leading/trailing spaces and dots
            safe_username = safe_username.strip(' .')
            # Ensure it's not empty after sanitization
            if not safe_username:
                safe_username = "user"
            self.user_dir_name = safe_username
        elif api_key:
            # Fallback: Use API key hash as directory name for security
            import hashlib
            api_key_hash = hashlib.sha256(api_key.encode()).hexdigest()[:16]
            self.user_dir_name = f"user_{api_key_hash}"
        else:
            self.user_dir_name = "userdata"
    
    def get_user_directory(self, base_dir):
        """Get the user's base directory path"""
        return os.path.join(base_dir, self.user_dir_name)
    
    def add_to_conversation_history(self, user_input, result_summary=None):
        """Add a conversation turn to history"""
        conversation_entry = {
            'timestamp': datetime.now().isoformat(),
            'user_input': user_input,
            'result_summary': result_summary or "Task executed",
            'output_dir': self.current_output_dir
        }
        self.conversation_history.append(conversation_entry)
        
        # Keep only last 10 conversations to avoid memory issues
        if len(self.conversation_history) > 10:
            self.conversation_history = self.conversation_history[-10:]
    
    def get_summarized_requirements(self):
        """Summarize conversation history into a comprehensive requirement"""
        if not self.conversation_history:
            return None
        
        # Create a summary of all previous requests
        history_summary = []
        for entry in self.conversation_history:
            history_summary.append(f"User requested: {entry['user_input']}")
        
        # Combine into a comprehensive requirement
        summarized_req = "\n".join(history_summary[-5:])  # Last 5 entries
        return summarized_req

gui_instance = AGIAgentGUI()

def create_temp_session_id(request, api_key=None):
    """Create a temporary session ID for API calls with user isolation"""
    import hashlib
    api_key_hash = hashlib.sha256((api_key or "default").encode()).hexdigest()[:8]
    return f"api_{request.remote_addr}_{api_key_hash}_{id(request)}"

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
                # Release task resources
                task_success = message.get('event') == 'task_completed'
                gui_instance.concurrency_manager.finish_task(session_id, success=task_success)
                
                # Get updated metrics
                metrics = gui_instance.concurrency_manager.get_metrics()
                status_msg = "Complete" if task_success else "Failed"
                print(f"✅ Task {status_msg} for user {session_id}")
                print(f"📊 Updated metrics - Active tasks: {metrics['active_tasks']}, Completed: {metrics['completed_tasks']}, Failed: {metrics['failed_tasks']}")
                
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
                
                # Add to conversation history if we have context from last executed task
                if hasattr(user_session, '_current_task_requirement'):
                    result_summary = "Task completed successfully" if task_success else "Task failed or had errors"
                    user_session.add_to_conversation_history(user_session._current_task_requirement, result_summary)
                    delattr(user_session, '_current_task_requirement')
                
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
    mcp_servers = get_mcp_servers_config()
    return render_template('index.html', i18n=i18n, lang=current_lang, mcp_servers=mcp_servers)

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
        temp_session_id = create_temp_session_id(request, api_key)
        user_session = gui_instance.get_user_session(temp_session_id, api_key)
        
        if not user_session:
            return jsonify({'success': False, 'error': 'Authentication failed'}), 401
        
        dirs = gui_instance.get_output_directories(user_session)
        return jsonify({'success': True, 'directories': dirs})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/download/<path:dir_name>')
def download_directory(dir_name):
    """Download directory as zip file (excluding code_index directory)"""
    try:
        # Get API key from query parameters or headers
        api_key = request.args.get('api_key') or request.headers.get('X-API-Key')
        
        # Create a temporary session for API calls
        temp_session_id = create_temp_session_id(request, api_key)
        user_session = gui_instance.get_user_session(temp_session_id, api_key)
        user_base_dir = user_session.get_user_directory(gui_instance.base_data_dir)
        
        dir_path = os.path.join(user_base_dir, secure_filename(dir_name))
        if not os.path.exists(dir_path) or not os.path.isdir(dir_path):
            return jsonify({'success': False, 'error': 'Directory not found'})
        
        # Create temporary zip file in a more reliable location
        import tempfile
        temp_dir = tempfile.gettempdir()
        temp_file = os.path.join(temp_dir, f"{dir_name}_{os.getpid()}_{int(datetime.now().timestamp())}.zip")
        
        try:
            with zipfile.ZipFile(temp_file, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as zipf:
                for root, dirs, files in os.walk(dir_path):
                    # Exclude code_index directory and other unwanted directories
                    dirs_to_exclude = {'code_index', '__pycache__', '.git', '.vscode', 'node_modules'}
                    if any(excluded in root for excluded in dirs_to_exclude):
                        print(f"Excluding directory: {root}")  # Debug info
                        continue
                    
                    for file in files:
                        # Skip unwanted files
                        if file.startswith('.') and file not in {'.gitignore', '.env.example'}:
                            continue
                        if file.endswith(('.pyc', '.pyo', '.DS_Store', 'Thumbs.db')):
                            continue
                            
                        file_path = os.path.join(root, file)
                        try:
                            # Calculate relative path for archive
                            rel_path = os.path.relpath(file_path, dir_path)
                            arcname = os.path.join(dir_name, rel_path).replace('\\', '/')
                            zipf.write(file_path, arcname)
                        except (OSError, IOError) as file_error:
                            print(f"Warning: Could not add file {file_path} to zip: {file_error}")
                            continue
            
            # Verify that the zip file was created and is not empty
            if not os.path.exists(temp_file) or os.path.getsize(temp_file) == 0:
                return jsonify({'success': False, 'error': 'Failed to create zip file or zip file is empty'})
            
            print(f"ZIP file created successfully: {temp_file}, size: {os.path.getsize(temp_file)} bytes")
            
            # Schedule cleanup after the request is complete
            @after_this_request
            def remove_temp_file(response):
                try:
                    if os.path.exists(temp_file):
                        os.remove(temp_file)
                        print(f"Cleaned up temporary file: {temp_file}")
                except Exception as cleanup_error:
                    print(f"Warning: Could not clean up temporary file {temp_file}: {cleanup_error}")
                return response
            
            # Return the file with proper headers
            return send_file(
                temp_file, 
                as_attachment=True, 
                download_name=f"{dir_name}.zip",
                mimetype='application/zip'
            )
            
        except Exception as zip_error:
            # Clean up temporary file on error
            if os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except:
                    pass
            raise zip_error
    
    except Exception as e:
        print(f"Download error: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/list-directory', methods=['POST'])
def list_directory():
    """List directory contents (single level). Used by Markdown image switcher."""
    try:
        data = request.get_json() or {}
        rel_path = data.get('path', '')

        # Auth
        api_key = request.args.get('api_key') or request.headers.get('X-API-Key') or data.get('api_key')
        temp_session_id = create_temp_session_id(request, api_key)
        user_session = gui_instance.get_user_session(temp_session_id, api_key)
        user_base_dir = user_session.get_user_directory(gui_instance.base_data_dir)

        full_path = os.path.join(user_base_dir, rel_path)
        real_output_dir = os.path.realpath(user_base_dir)
        real_file_path = os.path.realpath(full_path)
        if not real_file_path.startswith(real_output_dir):
            return jsonify({'success': False, 'error': 'Access denied'})
        if not os.path.exists(full_path) or not os.path.isdir(full_path):
            return jsonify({'success': False, 'error': f'Directory not found: {rel_path}'})

        items = []
        for name in os.listdir(full_path):
            item_path = os.path.join(full_path, name)
            if os.path.isfile(item_path):
                try:
                    size = os.path.getsize(item_path)
                except Exception:
                    size = 0
                items.append({'name': name, 'type': 'file', 'size': size})
            else:
                items.append({'name': name, 'type': 'directory'})

        items.sort(key=lambda x: (x.get('type') == 'file', x['name']))
        return jsonify({'success': True, 'files': items})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/file/<path:file_path>')
def get_file_content(file_path):
    """Get file content"""
    try:
        # Get API key from query parameters or headers
        api_key = request.args.get('api_key') or request.headers.get('X-API-Key')
        
        # Create a temporary session for API calls
        temp_session_id = create_temp_session_id(request, api_key)
        user_session = gui_instance.get_user_session(temp_session_id, api_key)
        user_base_dir = user_session.get_user_directory(gui_instance.base_data_dir)
        
        # URL decode the file path to handle Chinese characters
        import urllib.parse
        file_path = urllib.parse.unquote(file_path)
        
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
        if file_size > 50 * 1024 * 1024:  # 50MB
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
            # PDF files directly return file path
            return jsonify({
                'success': True, 
                'type': 'pdf',
                'file_path': file_path,
                'size': gui_instance.format_size(file_size)
            })
        elif ext in ['.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx']:
            # Office document preview
            return jsonify({
                'success': True, 
                'type': 'office',
                'file_path': file_path,
                'file_ext': ext,
                'size': gui_instance.format_size(file_size)
            })
        elif ext == '.tex':
            # LaTeX file - treat as code file
            with open(full_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            return jsonify({
                'success': True, 
                'content': content, 
                'type': 'code',
                'language': 'latex',
                'size': gui_instance.format_size(file_size)
            })
        elif ext in ['.py', '.js', '.jsx', '.ts', '.tsx', '.css', '.json', '.txt', '.log', '.yaml', '.yml', 
                     '.c', '.cpp', '.cc', '.cxx', '.h', '.hpp', '.java', '.go', '.rs', '.php', '.rb', 
                     '.sh', '.bash', '.zsh', '.fish', '.ps1', '.bat', '.cmd', '.xml', '.sql', '.r', 
                     '.scala', '.kt', '.swift', '.dart', '.lua', '.perl', '.pl', '.vim', '.dockerfile', 
                     '.makefile', '.cmake', '.gradle', '.properties', '.ini', '.cfg', '.conf', '.toml', '.mmd', '.out']:
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
                '.log': 'text',
                '.mmd': 'mermaid',
                '.out': 'text'
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
        elif ext in ['.png', '.jpg', '.jpeg', '.gif', '.svg', '.bmp', '.webp', '.ico']:
            # Image file handling
            import base64
            
            try:
                # Check if request wants raw image data (from img tag) or JSON (from preview)
                accept_header = request.headers.get('Accept', '')
                wants_raw_image = (
                    'image/' in accept_header or 
                    request.args.get('raw') == 'true' or
                    'text/html' in accept_header  # img tags typically send this
                )
                
                # Determine MIME type
                mime_types = {
                    '.png': 'image/png',
                    '.jpg': 'image/jpeg', 
                    '.jpeg': 'image/jpeg',
                    '.gif': 'image/gif',
                    '.svg': 'image/svg+xml',
                    '.bmp': 'image/bmp',
                    '.webp': 'image/webp',
                    '.ico': 'image/x-icon'
                }
                mime_type = mime_types.get(ext, 'image/jpeg')
                
                if wants_raw_image:
                    # Return raw image data for img tags
                    with open(full_path, 'rb') as f:
                        image_data = f.read()
                    
                    from flask import Response
                    return Response(
                        image_data,
                        mimetype=mime_type,
                        headers={
                            'Content-Length': len(image_data),
                            'Cache-Control': 'no-cache, no-store, must-revalidate'  # Disable caching for immediate updates
                        }
                    )
                else:
                    # Return JSON for preview functionality
                    with open(full_path, 'rb') as f:
                        image_data = f.read()
                    
                    # Convert to base64 for embedding in response
                    image_base64 = base64.b64encode(image_data).decode('utf-8')
                    
                    # Get image dimensions if possible
                    image_info = {}
                    try:
                        from PIL import Image
                        with Image.open(full_path) as img:
                            image_info = {
                                'width': img.width,
                                'height': img.height,
                                'format': img.format
                            }
                    except (ImportError, Exception):
                        # PIL not available or image cannot be processed
                        image_info = {'width': 'Unknown', 'height': 'Unknown', 'format': ext[1:].upper()}
                    
                    return jsonify({
                        'success': True,
                        'type': 'image',
                        'data': f"data:{mime_type};base64,{image_base64}",
                        'file_path': file_path,
                        'image_info': image_info,
                        'size': gui_instance.format_size(file_size)
                    })
                
            except Exception as e:
                return jsonify({'success': False, 'error': f'Failed to load image: {str(e)}'})
        else:
            return jsonify({'success': False, 'error': 'File type not supported for preview'})
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/pdf/<path:file_path>')
def serve_pdf(file_path):
    """Serve PDF file directly"""
    try:
        print(f"PDF request for: {file_path}")  # Add debug logs
        
        # Get API key from query parameters or headers
        api_key = request.args.get('api_key') or request.headers.get('X-API-Key')
        print(f"API key: {api_key[:10] + '...' if api_key else 'None'}")  # Add debug logs
        
        # Create a temporary session for API calls
        temp_session_id = create_temp_session_id(request, api_key)
        user_session = gui_instance.get_user_session(temp_session_id, api_key)
        user_base_dir = user_session.get_user_directory(gui_instance.base_data_dir)
        
        # URL decode the file path to handle Chinese characters
        import urllib.parse
        file_path = urllib.parse.unquote(file_path)
        
        # Use the passed path directly, don't use secure_filename as we need to maintain path structure
        full_path = os.path.join(user_base_dir, file_path)
        print(f"Full path: {full_path}")  # Add debug logs
        
        # Security check: ensure path is within user's output directory
        real_output_dir = os.path.realpath(user_base_dir)
        real_file_path = os.path.realpath(full_path)
        if not real_file_path.startswith(real_output_dir):
            print(f"Access denied: {real_file_path} not in {real_output_dir}")
            return jsonify({'success': False, 'error': 'Access denied'})
        
        if not os.path.exists(full_path) or not os.path.isfile(full_path):
            print(f"File not found: {full_path}")
            return jsonify({'success': False, 'error': f'File not found: {file_path}'})
        
        # Check if it's a PDF file
        if not full_path.lower().endswith('.pdf'):
            print(f"Not a PDF file: {full_path}")
            return jsonify({'success': False, 'error': 'Not a PDF file'})
        
        # Verify PDF file structure
        try:
            with open(full_path, 'rb') as f:
                header = f.read(8)
                if not header.startswith(b'%PDF-'):
                    print(f"Invalid PDF header: {header}")
                    return jsonify({'success': False, 'error': 'Invalid PDF file structure'})
        except Exception as pdf_check_error:
            print(f"PDF validation error: {pdf_check_error}")
            return jsonify({'success': False, 'error': f'PDF validation failed: {str(pdf_check_error)}'})
        
        print(f"Serving PDF file: {full_path}")
        response = send_file(full_path, mimetype='application/pdf')
        
        # Add CORS headers
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Methods'] = 'GET, OPTIONS'
        response.headers['Access-Control-Allow-Headers'] = 'X-API-Key, Content-Type'
        
        return response
    
    except Exception as e:
        print(f"PDF serve error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/download-file/<path:file_path>')
def download_file(file_path):
    """Download file directly"""
    try:
        # Get API key from query parameters or headers
        api_key = request.args.get('api_key') or request.headers.get('X-API-Key')
        
        # Create a temporary session for API calls
        temp_session_id = create_temp_session_id(request, api_key)
        user_session = gui_instance.get_user_session(temp_session_id, api_key)
        user_base_dir = user_session.get_user_directory(gui_instance.base_data_dir)
        
        # URL decode the file path to handle Chinese characters
        import urllib.parse
        file_path = urllib.parse.unquote(file_path)
        
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

# Cloud upload functionality has been removed for offline deployment

def convert_markdown_to_latex_only(full_path, file_path, user_base_dir):
    """Convert Markdown to LaTeX only"""
    import subprocess
    from pathlib import Path
    
    try:
        md_path = Path(full_path)
        base_name = md_path.stem
        output_dir = md_path.parent
        latex_file = output_dir / f"{base_name}.tex"
        
        # Use trans_md_to_pdf.py script to convert to LaTeX
        trans_script = Path(__file__).parent.parent / "src" / "utils" / "trans_md_to_pdf.py"
        
        if trans_script.exists():
            cmd = [
                'python3',
                str(trans_script),
                md_path.name,  # Use filename instead of full path
                latex_file.name,  # Use filename instead of full path
                '--latex'  # Add LaTeX flag
            ]
            
            # Execute command in markdown file directory
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(output_dir))
            
            if latex_file.exists():
                file_size = latex_file.stat().st_size
                return {
                    'status': 'success',
                    'markdown_file': file_path,
                    'conversions': {
                        'latex': {
                            'status': 'success',
                            'file': str(latex_file.relative_to(user_base_dir)),
                            'size': file_size,
                            'size_kb': f"{file_size / 1024:.1f} KB"
                        }
                    }
                }
            else:
                # Try direct pandoc conversion as fallback
                cmd = [
                    'pandoc',
                    md_path.name,
                    '-o', latex_file.name,
                    '--to', 'latex'
                ]
                
                # Add common options for LaTeX
                cmd.extend([
                    '-V', 'fontsize=12pt',
                    '-V', 'geometry:margin=2.5cm',
                    '-V', 'geometry:a4paper',
                    '-V', 'linestretch=2.0',
                    '--highlight-style=tango',
                    '-V', 'colorlinks=true',
                    '-V', 'linkcolor=blue',
                    '-V', 'urlcolor=blue',
                    '--toc',
                    '--wrap=preserve'
                ])
                
                result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(output_dir))
                
                if latex_file.exists():
                    file_size = latex_file.stat().st_size
                    return {
                        'status': 'success',
                        'markdown_file': file_path,
                        'conversions': {
                            'latex': {
                                'status': 'success',
                                'file': str(latex_file.relative_to(user_base_dir)),
                                'size': file_size,
                                'size_kb': f"{file_size / 1024:.1f} KB",
                                'method': 'direct_pandoc'
                            }
                        }
                    }
                else:
                    return {
                        'status': 'failed',
                        'markdown_file': file_path,
                        'error': f'LaTeX conversion failed: {result.stderr if result.stderr else "Unknown error"}'
                    }
        else:
            return {
                'status': 'failed',
                'markdown_file': file_path,
                'error': 'trans_md_to_pdf.py script not found'
            }
            
    except Exception as e:
        return {
            'status': 'failed',
            'markdown_file': file_path,
            'error': f'LaTeX conversion exception: {str(e)}'
        }


@app.route('/api/convert-markdown', methods=['POST'])
def convert_markdown():
    """Convert Markdown files to Word and PDF formats"""
    try:
        data = request.get_json()
        file_path = data.get('file_path')
        format_type = data.get('format', 'both')  # 'word', 'pdf', 'latex', or 'both'
        
        # Get API key from query parameters or headers
        api_key = request.args.get('api_key') or request.headers.get('X-API-Key') or data.get('api_key')
        
        # Create a temporary session for API calls
        temp_session_id = create_temp_session_id(request, api_key)
        user_session = gui_instance.get_user_session(temp_session_id, api_key)
        user_base_dir = user_session.get_user_directory(gui_instance.base_data_dir)
        
        if not file_path:
            return jsonify({'success': False, 'error': 'File path cannot be empty'})
        
        # URL decode the file path to handle Chinese characters
        import urllib.parse
        file_path = urllib.parse.unquote(file_path)
        
        # Use the passed path directly
        full_path = os.path.join(user_base_dir, file_path)
        
        # Security check: ensure path is within user's output directory
        real_output_dir = os.path.realpath(user_base_dir)
        real_file_path = os.path.realpath(full_path)
        if not real_file_path.startswith(real_output_dir):
            return jsonify({'success': False, 'error': 'Access denied'})
        
        if not os.path.exists(full_path) or not os.path.isfile(full_path):
            return jsonify({'success': False, 'error': f'File does not exist: {file_path}'})
        
        # Check if it's a markdown file
        _, ext = os.path.splitext(full_path.lower())
        if ext not in ['.md', '.markdown']:
            return jsonify({'success': False, 'error': 'Only supports Markdown file conversion'})
        
        # Create Tools instance directly to access FileSystemTools
        from src.tools import Tools
        tools = Tools(
            workspace_root=user_base_dir,
            out_dir=user_base_dir
        )
        
        # Call the conversion method from FileSystemTools
        print(f"🔍 Conversion debug information:")
        print(f"  file_path: {file_path}")
        print(f"  full_path: {full_path}")
        print(f"  user_base_dir: {user_base_dir}")
        print(f"  workspace_root: {tools.workspace_root}")
        
        # Handle LaTeX conversion separately if requested
        if format_type == 'latex':
            conversion_result = convert_markdown_to_latex_only(full_path, file_path, user_base_dir)
        else:
            conversion_result = tools._convert_markdown_to_formats(full_path, file_path, format_type)
        
        print(f"  Conversion result: {conversion_result}")
        
        if conversion_result.get('status') == 'success':
            # Check for partial success (some conversions failed)
            conversions = conversion_result.get('conversions', {})
            failed_conversions = [k for k, v in conversions.items() if v.get('status') == 'failed']
            
            response_data = {
                'success': True,
                'message': 'Conversion completed',
                'conversions': conversions,
                'converted_files': []
            }
            
            # Add warnings for failed conversions
            if failed_conversions:
                warnings = []
                for conv_type in failed_conversions:
                    conv_error = conversions[conv_type].get('error', 'Unknown error')
                    if 'Cannot load file' in conv_error or 'Invalid' in conv_error:
                        warnings.append(f'{conv_type.upper()} conversion failed due to image format issues. Consider converting WebP/TIFF images to PNG/JPEG.')
                    elif 'Cannot determine size' in conv_error or 'BoundingBox' in conv_error:
                        warnings.append(f'{conv_type.upper()} conversion failed due to image size/boundary issues.')
                    elif 'PDF engines' in conv_error:
                        warnings.append(f'{conv_type.upper()} conversion failed: No PDF engines available. Install xelatex, lualatex, pdflatex, wkhtmltopdf, or weasyprint.')
                    else:
                        warnings.append(f'{conv_type.upper()} conversion failed: {conv_error}')
                
                response_data['warnings'] = warnings
                response_data['partial_success'] = True
            
            return jsonify(response_data)
        else:
            error_msg = conversion_result.get('error', 'Conversion failed')
            user_friendly_error = error_msg
            suggestions = []
            
            # Provide user-friendly error messages and suggestions
            if 'Cannot load file' in error_msg or 'Invalid' in error_msg:
                user_friendly_error = 'Image format compatibility issues detected'
                suggestions.append('Convert WebP, TIFF, or other incompatible images to PNG or JPEG format')
                suggestions.append('Remove or replace problematic images')
            elif 'Cannot determine size' in error_msg or 'BoundingBox' in error_msg:
                user_friendly_error = 'Image size or boundary issues detected'
                suggestions.append('Ensure images have valid dimensions and formats')
                suggestions.append('Try resaving images in a standard format like PNG')
            elif 'PDF engines' in error_msg:
                user_friendly_error = 'PDF conversion engines not available'
                suggestions.append('Install LaTeX (xelatex, lualatex, pdflatex) for high-quality PDF output')
                suggestions.append('Install wkhtmltopdf or weasyprint as alternatives')
                suggestions.append('Word document conversion may still work as a fallback')
            
            return jsonify({
                'success': False,
                'error': user_friendly_error,
                'original_error': error_msg,
                'suggestions': suggestions,
                'message': conversion_result.get('message', 'Conversion failed')
            })
    
    except Exception as e:
        print(f"Markdown conversion error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': f'Error occurred during conversion: {str(e)}'})

@app.route('/api/convert-mermaid-to-images', methods=['POST'])
def convert_mermaid_to_images():
    """Convert Mermaid chart to SVG and PNG images"""
    try:
        data = request.get_json()
        file_path = data.get('file_path')
        mermaid_content = data.get('mermaid_content')
        
        # Get API key from query parameters or headers
        api_key = request.args.get('api_key') or request.headers.get('X-API-Key') or data.get('api_key')
        
        # Create a temporary session for API calls
        temp_session_id = create_temp_session_id(request, api_key)
        user_session = gui_instance.get_user_session(temp_session_id, api_key)
        user_base_dir = user_session.get_user_directory(gui_instance.base_data_dir)
        
        if not file_path:
            return jsonify({'success': False, 'error': 'File path cannot be empty'})
        
        if not mermaid_content:
            return jsonify({'success': False, 'error': 'Mermaid content cannot be empty'})
        
        if not MERMAID_PROCESSOR_AVAILABLE:
            return jsonify({'success': False, 'error': 'Mermaid processor not available'})
        
        # URL decode the file path to handle Chinese characters
        import urllib.parse
        file_path = urllib.parse.unquote(file_path)
        
        # Use the passed path directly
        full_path = os.path.join(user_base_dir, file_path)
        
        # Security check: ensure path is within user's output directory
        real_output_dir = os.path.realpath(user_base_dir)
        real_file_path = os.path.realpath(full_path)
        if not real_file_path.startswith(real_output_dir):
            return jsonify({'success': False, 'error': 'Access denied'})
        
        if not os.path.exists(full_path) or not os.path.isfile(full_path):
            return jsonify({'success': False, 'error': f'File does not exist: {file_path}'})
        
        # Check if it's a mermaid file
        _, ext = os.path.splitext(full_path.lower())
        if ext not in ['.mmd']:
            return jsonify({'success': False, 'error': 'Only supports .mmd file conversion'})
        
        # Generate base filename from original file (without extension)
        base_name = os.path.splitext(os.path.basename(full_path))[0]
        file_dir = os.path.dirname(full_path)

        # Check if we're already in an images directory
        # If so, use the current directory to avoid nested images folders
        if os.path.basename(file_dir).lower() == 'images':
            images_dir = file_dir
        else:
            # Create images directory if it doesn't exist
            images_dir = os.path.join(file_dir, 'images')
            os.makedirs(images_dir, exist_ok=True)
        
        # Generate output paths
        svg_path = os.path.join(images_dir, f"{base_name}.svg")
        png_path = os.path.join(images_dir, f"{base_name}.png")
        
        print(f"🎨 Converting Mermaid chart to images:")
        print(f"  File: {full_path}")
        print(f"  SVG output: {svg_path}")
        print(f"  PNG output: {png_path}")
        
        # Use mermaid processor to generate images
        from pathlib import Path
        svg_success, png_success = mermaid_processor._generate_mermaid_image(
            mermaid_content, 
            Path(svg_path), 
            Path(png_path)
        )
        
        if svg_success or png_success:
            i18n = get_i18n_texts()
            result = {
                'success': True,
                'message': i18n['mermaid_conversion_completed']
            }
            
            if svg_success:
                rel_svg_path = os.path.relpath(svg_path, user_base_dir)
                result['svg_path'] = rel_svg_path
                result['svg_full_path'] = svg_path
            
            if png_success:
                rel_png_path = os.path.relpath(png_path, user_base_dir)
                result['png_path'] = rel_png_path
                result['png_full_path'] = png_path
                
            if svg_success and png_success:
                result['message'] += i18n['mermaid_svg_png_format']
            elif svg_success:
                result['message'] += i18n['mermaid_svg_only']
            elif png_success:
                result['message'] += i18n['mermaid_png_only']
            
            print(f"✅ Mermaid conversion successful: SVG={svg_success}, PNG={png_success}")
            return jsonify(result)
        else:
            print(f"❌ Mermaid conversion failed")
            return jsonify({
                'success': False,
                'error': 'Failed to generate images from Mermaid chart'
            })
    
    except Exception as e:
        print(f"❌ Mermaid conversion error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': f'Error occurred during conversion: {str(e)}'})

@app.route('/api/metrics')
def get_performance_metrics():
    """Get current performance metrics"""
    try:
        metrics = gui_instance.concurrency_manager.get_metrics()
        
        # Add system resource information
        import psutil
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        
        system_metrics = {
            'cpu_percent': cpu_percent,
            'memory_percent': memory.percent,
            'memory_used_mb': memory.used / 1024 / 1024,
            'memory_total_mb': memory.total / 1024 / 1024
        }
        
        return jsonify({
            'success': True,
            'metrics': metrics,
            'system': system_metrics,
            'timestamp': time.time()
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@socketio.on('connect')
def handle_connect(auth):
    """WebSocket connection processing with authentication"""
    i18n = get_i18n_texts()
    session_id = request.sid
    
    # Check if new connections can be accepted
    if not gui_instance.concurrency_manager.can_accept_connection():
        emit('connection_rejected', {
            'message': 'Server connection limit reached'
        }, room=session_id)
        print(f"🚫 Connection rejected for {session_id}: Server at capacity")
        return False
    
    # Get user authentication info
    api_key = None
    if auth and 'api_key' in auth:
        api_key = auth['api_key']
    
    # Create or get user session with authentication
    user_session = gui_instance.get_user_session(session_id, api_key)
    
    if not user_session:
        # Authentication failed
        emit('auth_failed', {'message': 'Authentication failed. Please check your API key.'}, room=session_id)
        print(f"🚫 Connection rejected for {session_id}: Authentication failed")
        return False
    
    # Add connection to concurrency manager
    if not gui_instance.concurrency_manager.add_connection():
        emit('connection_rejected', {
            'message': 'Server connection limit reached'
        }, room=session_id)
        print(f"🚫 Connection rejected for {session_id}: Failed to add connection")
        return False
    
    # Create user directory if not exists
    user_dir = user_session.get_user_directory(gui_instance.base_data_dir)
    os.makedirs(user_dir, exist_ok=True)
    
    # Join user to their own room for isolated communication
    join_room(session_id)
    
    # Send connection status with user info
    is_guest = user_session.user_info.get("is_guest", False)
    user_name = user_session.user_info.get("name", "unknown")
    
    # Get current performance metrics
    metrics = gui_instance.concurrency_manager.get_metrics()
    
    print(f"🔗 User connected: {session_id}, User: {user_name}, API Key: {'***' if api_key else 'Guest'}, Directory: {os.path.basename(user_dir)}")
    print(f"📊 Current metrics - Active connections: {metrics['active_connections']}, Active tasks: {metrics['active_tasks']}")
    
    # Send status with guest indicator and performance info
    connection_data = {
        'message': i18n['connected'],
        'is_guest': is_guest,
        'user_name': user_name,
        'user_info': user_session.user_info,
        'server_metrics': {
            'active_connections': metrics['active_connections'],
            'active_tasks': metrics['active_tasks'],
            'queue_size': metrics['queue_size']
        }
    }
    
    emit('status', connection_data, room=session_id)

@socketio.on('disconnect')
def handle_disconnect():
    """Handle user disconnection"""
    session_id = request.sid

    # Remove connection from concurrency manager
    gui_instance.concurrency_manager.remove_connection()

    if session_id in gui_instance.user_sessions:
        user_session = gui_instance.user_sessions[session_id]

        # Leave room and clean up session immediately
        leave_room(session_id)

        # Terminate any running processes
        if user_session.current_process and user_session.current_process.is_alive():
            print(f"🛑 Terminating process for disconnected user {session_id}")
            user_session.current_process.terminate()
            user_session.current_process.join(timeout=5)

        # Clean up active task if exists
        gui_instance.concurrency_manager.finish_task(session_id, success=False)
        print(f"✅ Cleaned up task for disconnected user {session_id}")

        # Clean up session
        gui_instance.auth_manager.destroy_session(session_id)
        del gui_instance.user_sessions[session_id]

        print(f"🔌 User {session_id} disconnected and cleaned up")

        # Get updated metrics
        metrics = gui_instance.concurrency_manager.get_metrics()
        print(f"📊 Updated metrics - Active connections: {metrics['active_connections']}, Active tasks: {metrics['active_tasks']}")
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
    
    # Generate detailed requirement with conversation history for continuing tasks
    detailed_requirement = None
    if task_type in ['continue', 'selected'] and user_session.conversation_history:
        # For continue/selected tasks, include conversation context
        history_context = user_session.get_summarized_requirements()
        if history_context:
            # 🔧 Fix: adjust prompt order - current first
            detailed_requirement = f"Current request: {user_requirement}\n\nPrevious conversation context:\n{history_context}"
    
    # Get user's base directory
    user_base_dir = user_session.get_user_directory(gui_instance.base_data_dir)
    

    
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
            # 🔧 Fix: if user selected selected mode but didn't specify directory
            emit('error', {'message': i18n['select_directory_first']}, room=session_id)
            return
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
        
        # 🔧 Fix: if user didn't select directory and there's no last used directory
        if not out_dir and not user_session.selected_output_dir:
            emit('error', {'message': i18n['select_directory_first']}, room=session_id)
            return
    
    # Check if new tasks can be started
    if not gui_instance.concurrency_manager.can_start_task(session_id):
        emit('task_queued', {
            'message': 'Current server tasks are busy...',
            'queue_position': gui_instance.concurrency_manager.task_queue.qsize() + 1
        }, room=session_id)
        print(f"⏳ Task queued for user {session_id}: server at capacity")
        return
    
    user_session.output_queue = multiprocessing.Queue()
    
    # Get user ID (sha256_hash) for MCP knowledge base tools
    user_id = None
    if user_session.api_key:
        import hashlib
        user_id = hashlib.sha256(user_session.api_key.encode()).hexdigest()
    
    try:
        user_session.current_process = multiprocessing.Process(
            target=execute_agia_task_process_target,
            args=(user_requirement, user_session.output_queue, out_dir, continue_mode, plan_mode, gui_config, session_id, detailed_requirement, user_id)
        )
        user_session.current_process.daemon = True
        user_session.current_process.start()
        
        # Get current performance metrics
        metrics = gui_instance.concurrency_manager.get_metrics()
        print(f"🚀 Task started for user {session_id}")
        print(f"📊 Current metrics - Active tasks: {metrics['active_tasks']}, Completed: {metrics['completed_tasks']}")
        
    except Exception as e:
        # If process startup fails
        gui_instance.concurrency_manager.finish_task(session_id, success=False)
        emit('error', {'message': f'Task startup failed: {str(e)}'}, room=session_id)
        print(f"❌ Failed to start task for user {session_id}: {e}")
        return
    
    # Set current output directory name (extract from absolute path if needed)
    if out_dir:
        user_session.current_output_dir = os.path.basename(out_dir)
    else:
        user_session.current_output_dir = None
    
    # Store current task for conversation history
    user_session._current_task_requirement = user_requirement

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

@socketio.on('get_metrics')
def handle_get_metrics():
    """Handle real-time metrics request"""
    session_id = request.sid
    try:
        metrics = gui_instance.concurrency_manager.get_metrics()
        
        # Add current user's task running time
        runtime = gui_instance.concurrency_manager.get_task_runtime(session_id)
        
        # Add system resource information (lightweight)
        import psutil
        cpu_percent = psutil.cpu_percent(interval=0)  # Don't wait
        memory = psutil.virtual_memory()
        
        response_data = {
            'metrics': metrics,
            'system': {
                'cpu_percent': cpu_percent,
                'memory_percent': memory.percent
            },
            'user_task_runtime': runtime,
            'timestamp': time.time()
        }
        
        emit('metrics_update', response_data, room=session_id)
    except Exception as e:
        emit('error', {'message': f'Failed to get performance metrics: {str(e)}'}, room=session_id)

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

        # 🔧 Fix: save current conversation to history when stopping task
        if hasattr(user_session, '_current_task_requirement'):
            print(f"💾 Saving interrupted conversation to history for user {session_id}")
            user_session.add_to_conversation_history(
                user_session._current_task_requirement,
                "Task stopped by user"
            )
            delattr(user_session, '_current_task_requirement')

        user_session.current_process.terminate()
        user_session.current_output_dir = None  # Clear current directory mark

        # 🔧 Fix: Clean up active task to prevent timeout detection
        if hasattr(gui_instance, 'finish_task'):
            gui_instance.finish_task(session_id, success=False)

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

@socketio.on('clear_chat')
def handle_clear_chat():
    """Handle clear chat request"""
    session_id = request.sid
    if session_id not in gui_instance.user_sessions:
        return
    
    try:
        i18n = get_i18n_texts()
        
        # Clear server-side conversation history
        user_session = gui_instance.user_sessions[session_id]
        user_session.conversation_history.clear()
        print(f"🧹 Cleared conversation history for user {session_id}")
        
        emit('chat_cleared', {
            'success': True,
            'message': i18n['chat_cleared']
        }, room=session_id)
        
    except Exception as e:
        emit('chat_cleared', {
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
        temp_session_id = create_temp_session_id(request, api_key)
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

@app.route('/api/file-count/<path:dir_name>', methods=['GET'])
def get_file_count(dir_name):
    """Get file count in specified directory's workspace folder"""
    try:
        # Get API key from query parameters or headers
        api_key = request.args.get('api_key') or request.headers.get('X-API-Key')
        
        # Create a temporary session for API calls
        temp_session_id = create_temp_session_id(request, api_key)
        user_session = gui_instance.get_user_session(temp_session_id, api_key)
        user_base_dir = user_session.get_user_directory(gui_instance.base_data_dir)
        
        # Target directory path
        target_dir = os.path.join(user_base_dir, secure_filename(dir_name))
        if not os.path.exists(target_dir):
            return jsonify({
                'success': False,
                'error': 'Directory not found'
            }), 404
        
        # workspace directory path
        workspace_dir = os.path.join(target_dir, 'workspace')
        if not os.path.exists(workspace_dir):
            return jsonify({
                'success': True,
                'file_count': 0
            })
        
        # Count files recursively in workspace directory
        file_count = 0
        for root, dirs, files in os.walk(workspace_dir):
            file_count += len(files)
        
        return jsonify({
            'success': True,
            'file_count': file_count
        })
    except Exception as e:
        print(f"Failed to get file count: {str(e)}")
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
        temp_session_id = create_temp_session_id(request, api_key)
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
        temp_session_id = create_temp_session_id(request, api_key)
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
        temp_session_id = create_temp_session_id(request, api_key)
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

@app.route('/api/delete-file', methods=['DELETE'])
def delete_file():
    """Delete specified file from workspace"""
    try:
        # Get file path from request
        data = request.get_json()
        file_path = data.get('file_path') if data else request.args.get('file_path')
        
        if not file_path:
            return jsonify({'success': False, 'error': 'File path is required'})
        
        # Get API key from query parameters or headers
        api_key = request.args.get('api_key') or request.headers.get('X-API-Key')
        if data:
            api_key = api_key or data.get('api_key')
        
        # Create a temporary session for API calls
        temp_session_id = create_temp_session_id(request, api_key)
        user_session = gui_instance.get_user_session(temp_session_id, api_key)
        user_base_dir = user_session.get_user_directory(gui_instance.base_data_dir)
        
        # Construct full file path
        full_file_path = os.path.join(user_base_dir, file_path)
        
        # Security check: ensure file is within user's directory
        real_user_dir = os.path.realpath(user_base_dir)
        real_file_path = os.path.realpath(full_file_path)
        if not real_file_path.startswith(real_user_dir):
            return jsonify({'success': False, 'error': 'Access denied: Invalid file path'})
        
        # Check if path exists
        if not os.path.exists(full_file_path):
            return jsonify({'success': False, 'error': f'Path not found: {file_path}'})
        
        if os.path.isfile(full_file_path):
            print(f"Deleting file: {full_file_path}")
            # Delete the file
            os.remove(full_file_path)
        elif os.path.isdir(full_file_path):
            print(f"Deleting folder: {full_file_path}")
            # Delete the folder and all its contents
            shutil.rmtree(full_file_path)
        else:
            return jsonify({'success': False, 'error': f'Path is neither a file nor a directory: {file_path}'})
        
        print(f"Successfully deleted file: {file_path}")
        
        return jsonify({
            'success': True, 
            'message': f'File "{os.path.basename(file_path)}" has been successfully deleted'
        })
        
    except PermissionError as e:
        print(f"Permission error deleting file {file_path}: {str(e)}")
        return jsonify({'success': False, 'error': f'Permission denied: {str(e)}'})
    except Exception as e:
        print(f"Error deleting file {file_path}: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/routine-files', methods=['GET'])
def get_routine_files():
    """Get list of routine files from routine directory and workspace files starting with 'routine_'"""
    try:
        routine_files = []
        workspace_dir = os.getcwd()
        
        # 根据语言配置选择routine文件夹
        current_lang = get_language()
        if current_lang == 'zh':
            routine_dir = os.path.join(workspace_dir, 'routine_zh')
        else:
            routine_dir = os.path.join(workspace_dir, 'routine')
        
        # 1. 添加routine文件夹下的文件
        if os.path.exists(routine_dir) and os.path.isdir(routine_dir):
            for filename in os.listdir(routine_dir):
                if os.path.isfile(os.path.join(routine_dir, filename)):
                    # Remove file extension
                    name_without_ext = os.path.splitext(filename)[0]
                    routine_files.append({
                        'name': name_without_ext,
                        'filename': filename,
                        'type': 'routine_folder'
                    })
        
        # 2. 添加当前workspace下routine_开头的文件
        for filename in os.listdir(workspace_dir):
            if filename.startswith('routine_') and os.path.isfile(os.path.join(workspace_dir, filename)):
                # Remove file extension and 'routine_' prefix
                name_without_ext = os.path.splitext(filename)[0]
                display_name = name_without_ext[8:] if name_without_ext.startswith('routine_') else name_without_ext
                routine_files.append({
                    'name': display_name,
                    'filename': filename,
                    'type': 'workspace_file'
                })
        
        # 按名称排序
        routine_files.sort(key=lambda x: x['name'])
        
        return jsonify({
            'success': True,
            'files': routine_files
        })
        
    except Exception as e:
        print(f"Error getting routine files: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e),
            'files': []
        }), 500

@app.route('/api/validate-config', methods=['POST'])
def validate_config():
    """Validate GUI configuration and return model-specific config"""
    try:
        from src.config_loader import get_gui_config, validate_gui_config
        
        data = request.get_json()
        model_config = data.get('config')  # 新的结构：完整的配置对象
        
        if not model_config:
            i18n = get_i18n_texts()
            return jsonify({
                'success': False,
                'error': i18n['config_missing']
            })
        
        # 从配置对象中提取信息
        api_key = model_config.get('api_key')
        api_base = model_config.get('api_base')
        model_name = model_config.get('model')
        max_tokens = model_config.get('max_tokens', 8192)
        
        # 验证必需字段
        if not api_key or not api_base or not model_name:
            if 'i18n' not in locals():
                i18n = get_i18n_texts()
            return jsonify({
                'success': False,
                'error': i18n['config_incomplete']
            })
        
        # 验证max_tokens是有效的数字
        try:
            max_tokens = int(max_tokens) if max_tokens else 8192
            if max_tokens <= 0:
                max_tokens = 8192
        except (ValueError, TypeError):
            max_tokens = 8192
        
        # 对于内置的GLM-4.5配置，进行额外的配置文件验证
        if model_config.get('value') == 'glm-4.5':
            # 读取GUI配置并验证
            gui_config = get_gui_config()
            is_valid, error_message = validate_gui_config(gui_config)
            
            if not is_valid:
                return jsonify({
                    'success': False,
                    'error': error_message
                })
        
        # 返回验证后的配置
        return jsonify({
            'success': True,
            'config': {
                'api_key': api_key,
                'api_base': api_base,
                'model': model_name,
                'max_tokens': max_tokens
            }
        })
        
    except Exception as e:
        print(f"Error validating configuration: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Configuration validation failed: {str(e)}'
        })

@app.route('/api/save-file', methods=['POST'])
def save_file():
    """Save file content back to disk (universal file save endpoint)."""
    try:
        data = request.get_json() or {}
        rel_path = data.get('file_path')
        content = data.get('content', '')
        if not rel_path:
            return jsonify({'success': False, 'error': 'File path is required'})

        api_key = request.args.get('api_key') or request.headers.get('X-API-Key') or data.get('api_key')
        temp_session_id = create_temp_session_id(request, api_key)
        user_session = gui_instance.get_user_session(temp_session_id, api_key)
        user_base_dir = user_session.get_user_directory(gui_instance.base_data_dir)

        full_path = os.path.join(user_base_dir, rel_path)
        real_output_dir = os.path.realpath(user_base_dir)
        real_file_path = os.path.realpath(full_path)
        if not real_file_path.startswith(real_output_dir):
            return jsonify({'success': False, 'error': 'Access denied'})

        # Ensure parent dir exists
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        # Save content
        with open(full_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return jsonify({'success': True, 'path': rel_path})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/save-markdown', methods=['POST'])
def save_markdown():
    """Save modified Markdown content back to disk."""
    try:
        data = request.get_json() or {}
        rel_path = data.get('path')
        content = data.get('content', '')
        if not rel_path:
            return jsonify({'success': False, 'error': 'File path is required'})

        api_key = request.args.get('api_key') or request.headers.get('X-API-Key') or data.get('api_key')
        temp_session_id = create_temp_session_id(request, api_key)
        user_session = gui_instance.get_user_session(temp_session_id, api_key)
        user_base_dir = user_session.get_user_directory(gui_instance.base_data_dir)

        full_path = os.path.join(user_base_dir, rel_path)
        real_output_dir = os.path.realpath(user_base_dir)
        real_file_path = os.path.realpath(full_path)
        if not real_file_path.startswith(real_output_dir):
            return jsonify({'success': False, 'error': 'Access denied'})

        # Ensure parent dir exists
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        # Save content
        with open(full_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return jsonify({'success': True, 'path': rel_path})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/render-markdown', methods=['POST'])
def render_markdown():
    """Render Markdown content to HTML for preview."""
    try:
        data = request.get_json() or {}
        content = data.get('content', '')
        
        if not content:
            return jsonify({'success': False, 'error': 'Content is required'})
        
        # 使用现有的markdown处理逻辑
        import markdown
        from markdown.extensions import codehilite, tables, toc, fenced_code
        
        # 配置markdown扩展
        extensions = [
            'markdown.extensions.tables',
            'markdown.extensions.fenced_code',
            'markdown.extensions.codehilite',
            'markdown.extensions.toc',
            'markdown.extensions.attr_list',
            'markdown.extensions.def_list',
            'markdown.extensions.footnotes',
            'markdown.extensions.md_in_html'
        ]
        
        # 创建markdown实例
        md = markdown.Markdown(
            extensions=extensions,
            extension_configs={
                'codehilite': {
                    'css_class': 'highlight',
                    'use_pygments': True
                },
                'toc': {
                    'permalink': True
                }
            }
        )
        
        # 转换为HTML
        html = md.convert(content)
        
        return jsonify({'success': True, 'html': html})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/gui-configs', methods=['GET'])
def get_gui_configs():
    """Get available GUI model configurations"""
    try:
        from src.config_loader import get_gui_config
        
        # 读取GUI配置
        gui_config = get_gui_config()
        
        # 返回固定的两个选项：GLM-4.5 和自定义
        i18n = get_i18n_texts()
        configs = [
            {
                'value': 'glm-4.5',
                'label': 'GLM-4.5',
                'api_key': gui_config.get('api_key', ''),
                'api_base': gui_config.get('api_base', ''),
                'model': gui_config.get('model', ''),
                'max_tokens': gui_config.get('max_tokens', 8192),
                'display_name': 'GLM-4.5'
            },
            {
                'value': 'custom',
                'label': i18n['custom_label'],
                'api_key': '',
                'api_base': '',
                'model': '',
                'max_tokens': 8192,
                'display_name': i18n['custom_label']
            }
        ]
        
        return jsonify({
            'success': True,
            'configs': configs
        })
        
    except Exception as e:
        print(f"Error getting GUI configurations: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        })


@app.route('/api/optimize-svg', methods=['POST'])
def optimize_svg():
    """Optimize SVG file using either traditional or LLM-based optimization"""
    try:
        data = request.get_json() or {}
        file_path = data.get('file_path')
        use_llm = data.get('use_llm', False)
        api_key = request.args.get('api_key') or request.headers.get('X-API-Key') or data.get('api_key')

        if not file_path:
            return jsonify({'success': False, 'error': 'File path is required'})

        # Validate file path and permissions
        temp_session_id = create_temp_session_id(request, api_key)
        user_session = gui_instance.get_user_session(temp_session_id, api_key)
        user_base_dir = user_session.get_user_directory(gui_instance.base_data_dir)

        full_path = os.path.join(user_base_dir, file_path)
        real_output_dir = os.path.realpath(user_base_dir)
        real_file_path = os.path.realpath(full_path)

        if not real_file_path.startswith(real_output_dir):
            return jsonify({'success': False, 'error': 'Access denied'})

        if not os.path.exists(full_path):
            return jsonify({'success': False, 'error': 'File not found'})

        # Check if it's an SVG file
        if not full_path.lower().endswith('.svg'):
            return jsonify({'success': False, 'error': 'File must be an SVG file'})

        # Read original SVG content
        with open(full_path, 'r', encoding='utf-8') as f:
            original_content = f.read()

        optimization_report = None
        optimized_content = original_content

        if use_llm and LLM_SVG_OPTIMIZER_AVAILABLE:
            # Use LLM-based optimization
            try:
                optimizer = create_llm_optimizer_from_env()
                optimized_content, report = optimizer.optimize_svg_with_llm(original_content)

                optimization_report = {
                    'method': 'LLM',
                    'llm_provider': getattr(optimizer, 'provider', 'unknown'),
                    'llm_model': getattr(optimizer, 'model', 'unknown'),
                    'original_issues_count': len(report.get('original_issues', [])),
                    'changes_made': report.get('changes_made', []),
                    'issues_fixed': report.get('issues_fixed', [])
                }
            except Exception as llm_error:
                print(f"LLM optimization failed, falling back to traditional: {str(llm_error)}")
                use_llm = False

        if not use_llm and SVG_OPTIMIZER_AVAILABLE:
            # Use traditional optimization
            try:
                optimizer = AdvancedSVGOptimizer(OptimizationLevel.STANDARD)
                optimized_content, report = optimizer.optimize_svg_with_report(original_content)

                optimization_report = {
                    'method': 'Traditional',
                    'original_issues_count': len(report.original_issues),
                    'fixed_issues_count': len(report.fixed_issues),
                    'remaining_issues_count': len(report.remaining_issues)
                }
            except Exception as trad_error:
                print(f"Traditional optimization failed: {str(trad_error)}")
                return jsonify({'success': False, 'error': f'Optimization failed: {str(trad_error)}'})

        # Create backup if content changed
        if optimized_content != original_content:
            backup_path = full_path + '.optimized_backup'
            try:
                with open(backup_path, 'w', encoding='utf-8') as f:
                    f.write(original_content)
                print(f"Created backup: {backup_path}")
            except Exception as backup_error:
                print(f"Warning: Failed to create backup: {str(backup_error)}")

            # Save optimized content
            with open(full_path, 'w', encoding='utf-8') as f:
                f.write(optimized_content)

        # Generate success message
        if optimized_content != original_content:
            message = f"SVG文件已成功优化！"
            if optimization_report:
                if use_llm and optimization_report.get('method') == 'LLM':
                    message += f"\\n\\n🤖 AI优化完成"
                    message += f"\\n• 使用模型: {optimization_report.get('llm_provider', 'unknown')} - {optimization_report.get('llm_model', 'unknown')}"
                    message += f"\\n• 检测到问题: {optimization_report.get('original_issues_count', 0)}"
                    if optimization_report.get('changes_made'):
                        message += f"\\n• 主要改进: {len(optimization_report['changes_made'])} 项"
                    if optimization_report.get('issues_fixed'):
                        message += f"\\n• 修复问题: {len(optimization_report['issues_fixed'])} 个"
                else:
                    message += f"\\n\\n传统优化完成"
                    message += f"\\n• 检测到问题: {optimization_report.get('original_issues_count', 0)}"
                    message += f"\\n• 已修复问题: {optimization_report.get('fixed_issues_count', 0)}"
                    message += f"\\n• 剩余问题: {optimization_report.get('remaining_issues_count', 0)}"
        else:
            message = "SVG文件已经是最佳状态，无需优化"

        return jsonify({
            'success': True,
            'message': message,
            'optimization_report': optimization_report,
            'used_llm': use_llm
        })

    except Exception as e:
        print(f"SVG optimization error: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'SVG optimization failed: {str(e)}'
        })


def get_mcp_servers_config():
    """Get MCP servers configuration from mcp_servers_GUI.json for GUI

    Returns:
        dict: MCP servers configuration, or empty dict if failed
    """
    try:
        # Path to the example MCP config file
        example_config_path = os.path.join(os.getcwd(), 'config', 'mcp_servers_GUI.json')

        # Check if example config exists
        if not os.path.exists(example_config_path):
            print(f"Warning: MCP example config not found at {example_config_path}")
            return {}

        # Load the example configuration
        with open(example_config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)

        # Return the mcpServers section
        return config.get('mcpServers', {})

    except Exception as e:
        print(f"Error loading MCP servers config: {str(e)}")
        return {}


def generate_custom_mcp_config(selected_servers, out_dir):
    """Generate a custom MCP configuration file based on selected servers.

    Args:
        selected_servers: List of selected MCP server names
        out_dir: Output directory for the task

    Returns:
        str: Path to the generated MCP configuration file, or None if failed
    """
    try:
        # Path to the example MCP config file
        example_config_path = os.path.join(os.getcwd(), 'config', 'mcp_servers_GUI.json')

        # Check if example config exists
        if not os.path.exists(example_config_path):
            print(f"Warning: MCP example config not found at {example_config_path}")
            return None

        # Load the example configuration
        with open(example_config_path, 'r', encoding='utf-8') as f:
            example_config = json.load(f)

        # Create custom config with only selected servers
        custom_config = {"mcpServers": {}}

        # Add selected servers to custom config
        for server_name in selected_servers:
            if server_name in example_config.get('mcpServers', {}):
                custom_config['mcpServers'][server_name] = example_config['mcpServers'][server_name]
            else:
                print(f"Warning: MCP server '{server_name}' not found in example config")

        # Generate filename with timestamp to avoid conflicts
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        config_filename = f"mcp_servers_custom_{timestamp}.json"
        custom_config_path = os.path.join(out_dir, config_filename)

        # Write custom configuration to file
        with open(custom_config_path, 'w', encoding='utf-8') as f:
            json.dump(custom_config, f, indent=2, ensure_ascii=False)

        print(f"Generated custom MCP config: {custom_config_path}")
        return custom_config_path

    except Exception as e:
        print(f"Error generating custom MCP config: {str(e)}")
        return None


if __name__ == '__main__':
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='AGIAgent GUI Server')
    parser.add_argument('--port', '-p', type=int, default=5004, 
                       help='指定服务器启动端口 (默认: 5004)')
    args = parser.parse_args()
    
    # 优先使用命令行参数，其次使用环境变量，最后使用默认值
    port = args.port if args.port else int(os.environ.get('PORT', 5004))
    
    print(f"启动AGIAgent GUI服务器，端口: {port}")
    socketio.run(app, host='0.0.0.0', port=port, debug=False, allow_unsafe_werkzeug=True) 

