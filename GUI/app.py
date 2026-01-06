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

# 尝试使用 gevent 进行 monkey patching 以支持异步
try:
    from gevent import monkey
    monkey.patch_all()
    ASYNC_MODE = 'gevent'
except ImportError:
    ASYNC_MODE = 'threading'

from flask import Flask, render_template, request, jsonify, send_file, send_from_directory, after_this_request, abort, Response
from flask_socketio import SocketIO, emit, join_room, leave_room
import os
import sys
import threading
import datetime
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

# Note: We use the default multiprocessing start method
# 'fork' is faster but unsafe in multi-threaded environment (Flask/SocketIO)
# 'spawn' is slower but safer


# Determine template and static directories FIRST - always relative to this app.py file
# Get the directory where app.py is located (before any directory changes)
app_dir = os.path.dirname(os.path.abspath(__file__))
template_dir = os.path.join(app_dir, 'templates')
static_dir = os.path.join(app_dir, 'static')

# Add parent directory to path to import config_loader
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config_loader import get_language, get_gui_default_data_directory, load_config
from auth_manager import AuthenticationManager

# Import Mermaid processor

try:
    from src.tools.mermaid_processor import mermaid_processor
    MERMAID_PROCESSOR_AVAILABLE = True
except ImportError:
    MERMAID_PROCESSOR_AVAILABLE = False

# Import SVG optimizers
try:
    from src.utils.advanced_svg_optimizer import AdvancedSVGOptimizer, OptimizationLevel
    SVG_OPTIMIZER_AVAILABLE = True
except ImportError:
    #print("⚠️ Advanced SVG optimizer not available")
    SVG_OPTIMIZER_AVAILABLE = False

try:
    from src.utils.llm_svg_optimizer import create_llm_optimizer_from_env
    LLM_SVG_OPTIMIZER_AVAILABLE = True
except ImportError:
    #print("⚠️ LLM SVG optimizer not available")
    LLM_SVG_OPTIMIZER_AVAILABLE = False

# Import SVG to PNG converter
try:
    from src.tools.svg_to_png import EnhancedSVGToPNGConverter
    SVG_TO_PNG_CONVERTER_AVAILABLE = True
except ImportError:
    #print("⚠️ SVG to PNG converter not available")
    SVG_TO_PNG_CONVERTER_AVAILABLE = False

# Import agent status visualizer functions
try:
    # Import from same directory as app.py (GUI directory)
    from agent_status_visualizer import (
        find_status_files, load_status_file, find_message_files,
        find_tool_calls_from_logs, find_mermaid_figures_from_plan,
        find_status_updates, find_latest_output_dir
    )
    AGENT_VISUALIZER_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Agent status visualizer not available: {e}")
    AGENT_VISUALIZER_AVAILABLE = False

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
    
    def __init__(self, max_concurrent_tasks=16, max_connections=40, task_timeout=3600, gui_instance=None):  # 60 minute timeout (Expand by 1x)
        self.max_concurrent_tasks = max_concurrent_tasks
        self.max_connections = max_connections
        self.task_timeout = task_timeout  # 任务超时时间（Seconds）
        self.gui_instance = gui_instance  # Reference to GUI instance for session cleanup
        
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
        
        # Unified resource and monitoring thread
        self.monitor_active = True
        self.monitor_thread = threading.Thread(target=self._unified_monitor, daemon=True)
        self.monitor_thread.start()
        

    
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
    
    def _unified_monitor(self):
        """Unified resource and monitoring thread - handles resources, timeouts, and session cleanup"""
        resource_check_counter = 0
        timeout_check_counter = 0
        session_cleanup_counter = 0
        
        while self.monitor_active:
            try:
                # Check resources every 30 seconds (every 6 cycles of 5 seconds)
                resource_check_counter += 1
                if resource_check_counter >= 6:
                    resource_check_counter = 0
                    try:
                        process = psutil.Process()
                        memory_info = process.memory_info()
                        memory_mb = memory_info.rss / 1024 / 1024
                        
                        with self.lock:
                            if memory_mb > self.metrics['peak_memory_usage']:
                                self.metrics['peak_memory_usage'] = memory_mb
                            self.metrics['last_updated'] = time.time()
                    except Exception as e:
                        pass  # Ignore metrics error
                
                # Check timeouts every 60 seconds (every 12 cycles of 5 seconds)
                timeout_check_counter += 1
                if timeout_check_counter >= 12:
                    timeout_check_counter = 0
                    try:
                        current_time = time.time()
                        timeout_sessions = []
                        
                        with self.lock:
                            for session_id, task_info in self.active_tasks.items():
                                if current_time - task_info['start_time'] > self.task_timeout:
                                    timeout_sessions.append(session_id)
                        
                        # Handle timeout tasks
                        for session_id in timeout_sessions:
                            self._handle_task_timeout(session_id)
                    except Exception as e:
                        pass
                
                # Check idle sessions every 30 minutes (every 360 cycles of 5 seconds)
                session_cleanup_counter += 1
                if session_cleanup_counter >= 360:
                    session_cleanup_counter = 0
                    if self.gui_instance:
                        try:
                            self._cleanup_idle_sessions_for_gui()
                        except Exception as e:
                            pass
                
                # Sleep 5 seconds per cycle
                time.sleep(5)
                
            except Exception as e:
                time.sleep(10)
    
    def _cleanup_idle_sessions_for_gui(self):
        """Clean up idle sessions - integrated from GUI class"""
        if not self.gui_instance:
            return
            
        try:
            current_time = time.time()
            idle_sessions = []
            
            # Check idle sessions (no activity for over 2 hours)
            for session_id, user_session in self.gui_instance.user_sessions.items():
                # Check if authentication session is still valid
                session_info = self.gui_instance.auth_manager.validate_session(session_id)
                if not session_info:
                    idle_sessions.append(session_id)
                    continue
                
                # Check if there are running processes
                if user_session.current_process and user_session.current_process.is_alive():
                    continue  # 有活动进程，不清理
            
            # Clean up idle sessions
            for session_id in idle_sessions:
                try:
                    if hasattr(self.gui_instance, '_cleanup_session'):
                        self.gui_instance._cleanup_session(session_id)
                except Exception as e:
                    pass  # Silent cleanup
        except Exception as e:
            pass  # Cleanup error
    
    def _handle_task_timeout(self, session_id):
        """Handle task timeout"""
        # This method needs to set callback after GUI instance initialization
        if hasattr(self, '_timeout_callback') and self._timeout_callback:
            self._timeout_callback(session_id)
    
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
        self.monitor_active = False
        if hasattr(self, 'monitor_thread') and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=2)



app = Flask(__name__, template_folder=template_dir, static_folder=static_dir)
app.config['SECRET_KEY'] = f'{APP_NAME.lower().replace(" ", "_")}_gui_secret_key'
# 🔧 优化ping配置：增加ping超时时间到600秒（10分钟），防止任务执行期间连接断开
# ping_interval=60秒发送一次ping，ping_timeout=600秒超时（10分钟）
# 客户端每55秒发送心跳，服务器每60秒发送ping，双重保活机制
# 使用 gevent 异步模式（如果可用），否则回退到 threading
socketio = SocketIO(app, cors_allowed_origins="*", async_mode=ASYNC_MODE, 
                   ping_timeout=600, ping_interval=60, 
                   # 🔧 添加更多配置以支持更好的重连
                   logger=False, engineio_logger=False,
                   # 允许HTTP长轮询作为fallback
                   allow_upgrades=True)  


import logging
logging.getLogger('werkzeug').setLevel(logging.CRITICAL)

I18N_TEXTS = {
    'zh': {
        # Page title and basic information
        'page_title': f'{APP_NAME}',
        'app_title': f'{APP_NAME}',
        'app_subtitle': '',
        'connected': '',  # 已删除连接成功消息
        
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
        'direct_tooltip': '发起任务',
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
        'no_task_assigned': '未布置任务',
        'creating_directory': '正在自动创建新工作目录...',
        'directory_created': '已创建新工作目录',
        'directory_selected': '已选择目录',
        'directory_renamed': '目录重命名成功',
        'directory_deleted': '目录删除成功',
        'files_uploaded': '文件上传成功',
        'refresh_success': '目录列表已刷新',
        'chat_cleared': '日志和历史对话已清空',
        'confirm_clear_chat': '确定要清空所有日志和历史对话吗？此操作不可撤销。',
        
        # Mode information
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
        'pdfjs_not_loaded': 'PDF.js 未加载，无法预览PDF文件',
        'docx_load_failed': '文档加载失败: {error}',
        'preview_failed': '预览失败',
        
        # Delete warnings
        'delete_current_executing_warning': '⚠️ 警告：这是当前正在执行的目录！',
        'delete_selected_warning': '⚠️ 警告：这是当前选择的目录！',
        
        # File operations
        'file_size': '文件大小',
        'download_file': '下载文件',
        'office_preview_note': 'Office文档预览',
        'office_download_note': '下载文件: 下载到本地使用Office软件打开',
        'drag_unselected_dir_warning': '请先选择此工作目录后再拖动',
        
        # Tool execution status
        'tool_running': '执行中',
        'tool_success': '成功',
        'tool_error': '错误',
        'function_calling': '调用中',
        'tool_call': '工具调用',
        'json_output': 'JSON输出',
        'image': '图片',
        'dimensions': '尺寸',
        'total_rows': '总行数',
        'columns': '列数',
        
        # Configuration options
        'config_options': '配置选项',
        'show_config_options': '显示配置选项',
        'hide_config_options': '隐藏配置选项',
        'routine_file': '技能',
        'task_type': '模式选择',
        'no_routine': '请选择...',
        'enable_web_search': '搜索网络',
        'enable_multi_agent': '启动多智能体',
        'enable_long_term_memory': '启动长期记忆',
        'enable_mcp': 'MCP工具配置',
        'enable_jieba': '启用中文分词',
        'enable_thinking': '启用思考模式',
        'execution_mode': '执行模式',
        'agent_mode': 'Agent模式',
        'plan_mode': 'Plan模式',
        'user_input_request': '用户输入请求',
        'enter_your_response': '请输入您的回复...',
        'submit': '提交',
        'append_task': '追加任务',
        'append_task_empty': '请输入要追加的任务内容',
        'append_task_success': '任务已成功发送给智能体',
        'append_task_sent': '任务已追加到inbox',
        
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
        'continue_mode_info': '继续模式 - 将使用上次的工作目录',
        'create_or_select_directory': '请先点击绿色按钮创建新工作目录，或选择右侧的现有目录',
        'select_directory_first': '请先创建或者选择一个工作目录，鼠标单击工作目录中的某个文件夹，直到变为蓝色代表选中',
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
        'please_connect': '请先连接服务器',
        'uploading_files': '正在上传 {0} 个文件',
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
        'connection_error': '连接错误',
        'reconnecting': '正在尝试重新连接...',
        'connection_interrupted_reconnecting': '连接中断，正在尝试重新连接...',
        'reconnect_attempt': '正在尝试重新连接',
        'reconnect_success': '已重新连接到服务器',
        'reconnect_failed_cleanup': '自动重连失败，已清空工作目录，请重新连接',
        'reconnect_error': '自动重连出错',
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
        'save_to_config_confirm': '已设置为临时配置，是否将此配置保存到 config/config.txt 作为长期配置？\n\n这将更新配置文件中的默认模型设置。',
        'save_to_config_success': '配置已成功保存到 config.txt',
        'save_to_config_failed': '保存到 config.txt 失败',
        'save_to_config_error': '保存到 config.txt 时发生错误',
        
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
        'office_instruction_3': '',
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
        'task_emitted': '✅ 任务已发起',
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
        
        # SVG Editor buttons
        'edit_svg': '编辑',
        'ai_optimize_svg': 'AI润色',
        'restore_svg': '恢复',
        'delete_svg': '删除',
        'edit_svg_tooltip': '编辑SVG图',
        'ai_optimize_svg_tooltip': 'AI智能重新设计SVG图',
        'restore_svg_tooltip': '恢复原图',
        'delete_svg_tooltip': '删除SVG图',
        
        # Markdown diagram reparse
        'reparse_diagrams': '解析图表',
        'reparse_diagrams_title': '重新解析Markdown中的Mermaid图表和SVG代码块',
        
        # Document conversion messages
        'converting': '转换中...',
        'mermaid_conversion_success': 'Mermaid图表转换成功！',
        'conversion_failed': '转换失败',
        'unknown_error': '未知错误',
        'word_conversion_success': 'Word文档转换成功并开始下载！',
        'word_conversion_failed': 'Word文档转换失败',
        'pdf_conversion_success': 'PDF文档转换成功并开始下载！',
        'pdf_conversion_failed': 'PDF文档转换失败',
        'latex_generation_success': 'LaTeX源文件生成成功并开始下载！',
        'latex_generation_failed': 'LaTeX源文件生成失败',
        'generation_failed': '生成失败',
        'file_label': '文件',
        'size_label': '大小',
        'svg_file': 'SVG文件',
        'png_file': 'PNG文件',
        
        # Dialog messages
        'confirm_delete_svg': '确定要删除这个SVG图吗？',
        'confirm_delete_image': '确定要删除这张图片吗？',
        'delete_image_failed': '删除图片失败',
        'no_markdown_to_save': '未检测到可保存的Markdown内容',
        'cannot_determine_file_path': '无法确定当前Markdown文件路径',
        'confirm_delete_elements': '确定要删除选中的 {count} 个元素吗？此操作无法撤销。',
        'confirm_delete_elements_en': 'Are you sure you want to delete the selected {count} elements? This action cannot be undone.',
        
        # Console log messages (for debugging, but should be consistent)
        'edit_svg_file': '编辑SVG文件',
        'delete_image': '删除图片',
        'image_deleted_auto_save': '图片删除后已自动保存markdown文件',
        'image_switched_auto_save': '图片切换后已自动保存markdown文件',
        'svg_deleted_auto_save': 'SVG删除后已自动保存markdown文件',
        'auto_save_error': '自动保存时出错',
        'guest_skip_auto_save': 'Guest用户跳过自动保存',
        'no_markdown_auto_save': '无Markdown内容可自动保存',
        'cannot_determine_path_auto_save': '无法确定Markdown文件路径，跳过自动保存',
        'markdown_auto_saved': 'Markdown已自动保存',
        'auto_save_failed': '自动保存失败',
        'auto_save_markdown_failed': '自动保存Markdown失败',
        
        # Additional error messages
        'cannot_get_svg_path': '无法获取SVG文件路径',
        'cannot_get_image_path': '无法获取图片文件路径',
        'cannot_get_file_path': '无法获取文件路径',
        'cannot_get_current_file_path': '无法获取当前文件路径',
        'cannot_determine_mermaid_path': '无法确定当前Mermaid文件路径',
        'cannot_determine_markdown_path': '无法确定当前Markdown文件路径',
        'delete_svg_failed': '删除SVG失败',
        'conversion_request_failed': '转换请求失败',
        'conversion_error': '转换错误',
        'error_during_conversion': '转换过程中发生错误',
        'generation_error': '生成错误',
        'error_during_generation': '生成过程中发生错误',
        
        # Virtual terminal
        'virtual_terminal_disabled': '该版本的虚拟终端已禁用，请下载自部署版本，并在config.txt中配置GUI_virtual_terminal=True',
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
        'no_task_assigned': 'No task assigned',
        'creating_directory': 'Creating new workspace directory...',
        'directory_created': 'New workspace directory created',
        'directory_selected': 'Directory selected',
        'directory_renamed': 'Directory renamed successfully',
        'directory_deleted': 'Directory deleted successfully',
        'files_uploaded': 'Files uploaded successfully',
        'refresh_success': 'Directory list refreshed',
        'chat_cleared': 'Chat log and conversation history cleared',
        'confirm_clear_chat': 'Are you sure you want to clear all chat logs and conversation history? This operation cannot be undone.',
        
        # Mode info
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
        'pdfjs_not_loaded': 'PDF.js not loaded, unable to preview PDF files',
        'docx_load_failed': 'Document load failed: {error}',
        'preview_failed': 'Preview Failed',
        
        # Delete warnings
        'delete_current_executing_warning': '⚠️ Warning: This is the currently executing directory!',
        'delete_selected_warning': '⚠️ Warning: This is the currently selected directory!',
        
        # File operations
        'file_size': 'File Size',
        'download_file': 'Download File',
        'office_preview_note': 'Office Document Preview',
        'office_download_note': 'Download File: Download to local and open with Office software',
        'drag_unselected_dir_warning': 'Please select this workspace directory first before dragging',
        
        # Tool execution status
        'tool_running': 'Running',
        'tool_success': 'Success',
        'tool_error': 'Error',
        'function_calling': 'Calling',
        'tool_call': 'Tool Call',
        'json_output': 'JSON Output',
        'image': 'Image',
        'dimensions': 'Dimensions',
        'total_rows': 'Total Rows',
        'columns': 'Columns',
        
        # Configuration options
        'config_options': 'Configuration Options',
        'show_config_options': 'Show Configuration',
        'hide_config_options': 'Hide Configuration',
        'routine_file': 'Skills',
        'task_type': 'Mode Selection',
        'no_routine': 'Please select...',
        'enable_web_search': 'Web Search',
        'enable_multi_agent': 'Multi-Agent',
        'enable_long_term_memory': 'Long-term Memory',
        'enable_mcp': 'Enable MCP',
        'enable_jieba': 'Chinese Segmentation',
        'enable_thinking': 'Enable Thinking',
        'execution_mode': 'Execution Mode',
        'agent_mode': 'Agent Mode',
        'plan_mode': 'Plan Mode',
        'user_input_request': 'User Input Request',
        'enter_your_response': 'Enter your response...',
        'submit': 'Submit',
        'append_task': 'Append Task',
        'append_task_empty': 'Please enter task content to append',
        'append_task_success': 'Task successfully sent to agent',
        'append_task_sent': 'Task appended to inbox',
        
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
        'continue_mode_info': 'Continue mode - Will use the previous workspace directory',
        'create_or_select_directory': 'Please click the green button to create a new workspace directory, or select an existing directory on the right',
        'select_directory_first': 'Please create or select a workspace directory, then click a folder in the workspace list until it turns blue to confirm the selection',
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
        'please_connect': 'Please connect to server first',
        'attempt': 'attempt',
        'create_directory_failed': 'Failed to create directory',
        'preview': 'Preview',
        'page_info': 'Page {0} of {1}',
        'upload_to': 'Upload files to',
        'workspace': '/workspace',
        'select_directory_error': 'Please select a directory first',
        'uploading_files': 'Uploading {0} files',
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
        'connection_error': 'Connection error',
        'reconnecting': 'Attempting to reconnect...',
        'connection_interrupted_reconnecting': 'Connection interrupted, attempting to reconnect...',
        'reconnect_attempt': 'Attempting to reconnect',
        'reconnect_success': 'Reconnected to server',
        'reconnect_failed_cleanup': 'Auto reconnection failed. Workspace has been cleared, please reconnect.',
        'reconnect_error': 'Auto reconnection error',
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
        'save_to_config_confirm': 'Already configured for temporary setting. Would you like to save this configuration to config/config.txt as a long-term configuration?\n\nThis will update the default model settings in the config file.',
        'save_to_config_success': 'Configuration successfully saved to config.txt',
        'save_to_config_failed': 'Failed to save to config.txt',
        'save_to_config_error': 'An error occurred while saving to config.txt',
        
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
        'office_instruction_3': '',
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
        'task_emitted': '✅ Task Emitted',
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
        
        # SVG Editor buttons
        'edit_svg': 'Edit',
        'ai_optimize_svg': 'AI Polish',
        'restore_svg': 'Restore',
        'delete_svg': 'Delete',
        'edit_svg_tooltip': 'Edit SVG image',
        'ai_optimize_svg_tooltip': 'AI intelligent redesign SVG image',
        'restore_svg_tooltip': 'Restore original image',
        'delete_svg_tooltip': 'Delete SVG image',
        
        # Markdown diagram reparse
        'reparse_diagrams': 'Parse Diagrams',
        'reparse_diagrams_title': 'Reparse Mermaid charts and SVG code blocks in Markdown',
        
        # Document conversion messages
        'converting': 'Converting...',
        'mermaid_conversion_success': 'Mermaid chart conversion successful!',
        'conversion_failed': 'Conversion failed',
        'unknown_error': 'Unknown error',
        'word_conversion_success': 'Word document conversion successful and download started!',
        'word_conversion_failed': 'Word document conversion failed',
        'pdf_conversion_success': 'PDF document conversion successful and download started!',
        'pdf_conversion_failed': 'PDF document conversion failed',
        'latex_generation_success': 'LaTeX source file generation successful and download started!',
        'latex_generation_failed': 'LaTeX source file generation failed',
        'generation_failed': 'Generation failed',
        'file_label': 'File',
        'size_label': 'Size',
        'svg_file': 'SVG file',
        'png_file': 'PNG file',
        
        # Dialog messages
        'confirm_delete_svg': 'Are you sure you want to delete this SVG image?',
        'confirm_delete_image': 'Are you sure you want to delete this image?',
        'delete_image_failed': 'Failed to delete image',
        'no_markdown_to_save': 'No Markdown content detected to save',
        'cannot_determine_file_path': 'Cannot determine current Markdown file path',
        'confirm_delete_elements': 'Are you sure you want to delete the selected {count} elements? This action cannot be undone.',
        'confirm_delete_elements_en': 'Are you sure you want to delete the selected {count} elements? This action cannot be undone.',
        
        # Console log messages (for debugging, but should be consistent)
        'edit_svg_file': 'Edit SVG file',
        'delete_image': 'Delete image',
        'image_deleted_auto_save': 'Markdown file auto-saved after image deletion',
        'image_switched_auto_save': 'Markdown file auto-saved after image switch',
        'svg_deleted_auto_save': 'Markdown file auto-saved after SVG deletion',
        'auto_save_error': 'Auto-save error',
        'guest_skip_auto_save': 'Guest user skips auto-save',
        'no_markdown_auto_save': 'No Markdown content to auto-save',
        'cannot_determine_path_auto_save': 'Cannot determine Markdown file path, skip auto-save',
        'markdown_auto_saved': 'Markdown auto-saved',
        'auto_save_failed': 'Auto-save failed',
        'auto_save_markdown_failed': 'Auto-save Markdown failed',
        
        # Additional error messages
        'cannot_get_svg_path': 'Cannot get SVG file path',
        'cannot_get_image_path': 'Cannot get image file path',
        'cannot_get_file_path': 'Cannot get file path',
        'cannot_get_current_file_path': 'Cannot get current file path',
        'cannot_determine_mermaid_path': 'Cannot determine current Mermaid file path',
        'cannot_determine_markdown_path': 'Cannot determine current Markdown file path',
        'delete_svg_failed': 'Failed to delete SVG',
        'conversion_request_failed': 'Conversion request failed',
        'conversion_error': 'Conversion error',
        'error_during_conversion': 'Error occurred during conversion',
        'generation_error': 'Generation error',
        'error_during_generation': 'Error occurred during generation',
        
        # Virtual terminal
        'virtual_terminal_disabled': 'Configuration disabled. Please download the standalone version and set GUI_virtual_terminal=True in config.txt',
    }
}

def get_i18n_texts():
    """Get internationalization text for current language"""
    current_lang = get_language()
    return I18N_TEXTS.get(current_lang, I18N_TEXTS['en'])

def execute_agia_task_process_target(user_requirement, output_queue, input_queue, out_dir=None, continue_mode=False, plan_mode=False, gui_config=None, session_id=None, detailed_requirement=None, user_id=None, attached_files=None):
    """
    This function runs in a separate process.
    It cannot use the `socketio` object directly.
    It communicates back to the main process via the queue.
    User input is received via input_queue in GUI mode.
    """
    # Store input_queue in a way that talk_to_user can access it
    import sys
    import __main__
    __main__._agia_gui_input_queue = input_queue
    
    try:

        # Get i18n texts for this process (after sending initial message)
        i18n = get_i18n_texts()
        
        if not out_dir:
            # Get GUI default data directory from config for new directories
            from src.config_loader import get_gui_default_data_directory
            config_data_dir = get_gui_default_data_directory()
            if config_data_dir:
                base_dir = config_data_dir
            else:
                base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            out_dir = os.path.join(base_dir, f"output_{timestamp}")
        
        # Process GUI configuration options
        if gui_config is None:
            gui_config = {}
        
        # Get language from gui_config if available, otherwise use default
        user_lang = gui_config.get('language')
        if user_lang and user_lang in ('zh', 'en'):
            i18n = I18N_TEXTS.get(user_lang, I18N_TEXTS['en'])
        
        # Set default values based on user requirements
        enable_web_search = gui_config.get('enable_web_search', True)
        enable_multi_agent = gui_config.get('enable_multi_agent', False)
        enable_long_term_memory = gui_config.get('enable_long_term_memory', True)  # Default selection
        enable_mcp = gui_config.get('enable_mcp', False)
        enable_jieba = gui_config.get('enable_jieba', True)  # Default selection
        enable_thinking = gui_config.get('enable_thinking', False)  # Default disabled
        
        # Execution rounds configuration from GUI
        execution_rounds = gui_config.get('execution_rounds', 50)  # Default to 50 if not provided
        
        # Routine file configuration from GUI
        routine_file = gui_config.get('routine_file')
        if routine_file:
            # 检查是否是workspace文件（以routine_开头）
            if routine_file.startswith('routine_'):
                # 直接使用workspace根目录下的文件
                routine_file = os.path.join(os.getcwd(), routine_file)
            else:
                # 根据语言配置选择routine文件夹
                # 优先使用前端传递的语言参数，如果没有则使用服务器端配置
                current_lang = gui_config.get('language')
                # 确保语言参数有效（'zh' 或 'en'），否则使用服务器端配置
                if not current_lang or current_lang not in ('zh', 'en'):
                    current_lang = get_language()
                if current_lang == 'zh':
                    routine_file = os.path.join(os.getcwd(), 'routine_zh', routine_file)
                else:
                    routine_file = os.path.join(os.getcwd(), 'routine', routine_file)
            
            if not os.path.exists(routine_file):
                output_queue.put({'event': 'output', 'data': {'message': f"Warning: Routine file not found: {routine_file}", 'type': 'warning'}})
                routine_file = None

        # Model configuration from GUI
        selected_model = gui_config.get('selected_model')
        model_api_key = gui_config.get('model_api_key')
        model_api_base = gui_config.get('model_api_base')
        
        # 如果前端没有提供 api_key 和 api_base（内置配置），从服务器端读取
        # 对于内置配置，前端可能会发送 api_key 和 api_base（从服务器获取的），也可能不发送
        if not model_api_key or not model_api_base:
            from src.config_loader import get_gui_config, get_all_model_configs
            
            # 首先尝试从所有配置中找到匹配selected_model的配置
            if selected_model:
                all_configs = get_all_model_configs()
                matching_config = None
                for config in all_configs:
                    if config.get('model', '').strip() == selected_model.strip():
                        # 优先选择enabled的配置
                        if config.get('enabled', True):
                            matching_config = config
                            break
                        elif not matching_config:
                            # 如果没有enabled的，保存第一个匹配的作为备选
                            matching_config = config
                
                if matching_config:
                    if not model_api_key:
                        model_api_key = matching_config.get('api_key', '')
                    if not model_api_base:
                        model_api_base = matching_config.get('api_base', '')
            
            # 如果还是没有找到，使用GUI API配置作为fallback
            if not model_api_key or not model_api_base:
                gui_config_from_server = get_gui_config()
                
                # 如果服务器端有配置，就使用它
                if gui_config_from_server.get('api_key') and gui_config_from_server.get('api_base'):
                    if not model_api_key:
                        model_api_key = gui_config_from_server.get('api_key')
                    if not model_api_base:
                        model_api_base = gui_config_from_server.get('api_base')
                    # 如果 selected_model 为空、None、空字符串或为默认值，使用服务器端的模型名称
                    if not selected_model or selected_model == '' or selected_model == 'claude-sonnet-4':
                        selected_model = gui_config_from_server.get('model', selected_model or 'claude-sonnet-4')
        
        # 验证配置是否完整
        if not model_api_key or not model_api_base or not selected_model:
            missing_items = []
            if not model_api_key:
                missing_items.append('API Key')
            if not model_api_base:
                missing_items.append('API Base')
            if not selected_model:
                missing_items.append('模型名称')
            error_msg = f"配置信息不完整：缺少 {', '.join(missing_items)}。请检查 config/config.txt 中的 GUI API 配置部分。"
            output_queue.put({'event': 'error', 'data': {'message': error_msg}})
            return
        
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
        # In plan mode, we still use single_task_mode=True, but plan_mode will be handled separately in run()
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
        
        # Set environment variable for GUI mode detection
        os.environ['AGIA_GUI_MODE'] = 'true'
        
        # Print task initiation message before starting AGIAgent
        output_queue.put({'event': 'output', 'data': {'message': i18n['task_emitted'], 'type': 'info'}})
        
        agia = AGIAgentMain(
            out_dir=out_dir,
            debug_mode=False,
            detailed_summary=True,
            single_task_mode=single_task_mode,  # Set based on plan_mode
            interactive_mode=False,  # Disable interactive mode
            continue_mode=False,  # Always use False for GUI mode to avoid shared .agia_last_output.json
            MCP_config_file=mcp_config_file,  # Set based on GUI MCP option
            user_id=user_id,  # Pass user ID for MCP knowledge base tools
            routine_file=routine_file,  # Pass routine file to main application
            plan_mode=plan_mode,  # Pass plan_mode to AGIAgentMain
            enable_thinking=enable_thinking  # Pass thinking mode to AGIAgentMain
        )
        
        # Use detailed_requirement if provided (contains conversation history)
        base_requirement = detailed_requirement if detailed_requirement else user_requirement
        
        # Process attached files - add file path references instead of content
        if attached_files:
            file_references = []
            for file_info in attached_files:
                file_path = file_info.get('path', '')
                file_name = file_info.get('name', '')
                reference = file_info.get('reference', '')
                if file_path and file_name:
                    file_references.append(f"\n\n--- 文件引用: {file_name} ---\n文件路径: {file_path}\n--- 文件引用结束: {file_name} ---\n")
            
            if file_references:
                base_requirement = base_requirement + ''.join(file_references)
        
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
                    # List workspace contents for context (limit to first 50 files for performance)
                    workspace_files = []
                    md_files = []
                    max_files = 50  # Limit to avoid long delays with large directories
                    file_count = 0
                    
                    for root, dirs, files in os.walk(workspace_dir):
                        # Skip hidden directories and common large directories
                        dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['node_modules', '__pycache__', 'venv', '.git']]
                        
                        for file in files:
                            if file_count >= max_files:
                                break
                            
                            file_path = os.path.join(root, file)
                            rel_path = os.path.relpath(file_path, workspace_dir)
                            file_size = os.path.getsize(file_path)
                            
                            if file.endswith('.md'):
                                md_files.append(f"  - {rel_path} ({format_size(file_size)})")
                            else:
                                workspace_files.append(f"  - {rel_path} ({format_size(file_size)})")
                            
                            file_count += 1
                        
                        if file_count >= max_files:
                            break
                    
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
        
        class QueueSocketHandler:
            def __init__(self, q, socket_type='info'):
                self.q = q
                self.socket_type = socket_type
                self.buffer = ""
                # 保存原始的stderr引用，用于调试输出（避免递归）
                self._original_stderr = sys.__stderr__
            
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
                # IMPORTANT: Don't filter GUI_USER_INPUT_REQUEST, QUERY, and TIMEOUT messages here!
                # These messages need to enter the queue so queue_reader_thread can detect them.
                # They will be filtered later in queue_reader_thread before emitting to frontend.
                # if '🔔 GUI_USER_INPUT_REQUEST' in line or line.startswith('QUERY: ') or line.startswith('TIMEOUT: '):
                #     return True
                
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
                
                # Check if buffer contains \r (carriage return) indicating progress bar update
                has_carriage_return = '\r' in self.buffer
                
                if '\n' in self.buffer:
                    *lines, self.buffer = self.buffer.split('\n')
                    for line in lines:
                        if line.strip():
                            # Filter code_edit content for GUI display (preserve leading spaces)
                            line_rstrip = line.rstrip()  # Only remove trailing spaces, preserve leading spaces
                            filtered_line = self.filter_code_edit_content(line_rstrip)
                            
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
                            
                            # Detect if this is a progress bar update (contains \r)
                            is_update = '\r' in line
                            # Remove \r from the message for display
                            filtered_line = filtered_line.replace('\r', '')
                            
                            # Display warning and progress info as normal info
                            self.q.put({'event': 'output', 'data': {'message': filtered_line, 'type': message_type, 'is_update': is_update}})
                elif has_carriage_return and self.buffer:
                    # Handle progress bar update without newline (buffer ends with \r)
                    # Clean the buffer: remove \r and trailing whitespace
                    buffer_clean = self.buffer.replace('\r', '').rstrip()
                    if buffer_clean:
                        # Filter code_edit content
                        filtered_line = self.filter_code_edit_content(buffer_clean)
                        
                        # Filter out redundant system messages
                        if not self.should_filter_message(filtered_line):
                            # Check if it's warning or progress info
                            line_lower = filtered_line.lower()
                            if ('warning' in line_lower or
                                'progress' in line_lower or
                                'processing files' in line_lower or
                                filtered_line.startswith('Processing files:') or
                                'userwarning' in line_lower or
                                'warnings.warn' in line_lower or
                                '⚠️' in filtered_line or
                                filtered_line.startswith('W: ') or
                                'W: ' in filtered_line):
                                message_type = 'info'
                            else:
                                message_type = self.socket_type
                            
                            # This is definitely an update (has \r)
                            self.q.put({'event': 'output', 'data': {'message': filtered_line, 'type': message_type, 'is_update': True}})
                        # Clear buffer after processing update
                        self.buffer = ""
                # 修复丢字问题：如果buffer中没有\n也没有\r，但buffer长度超过阈值（比如1024字符），也应该flush
                # 这样可以避免长消息被分成多个chunk时，最后一部分没有换行符导致丢失
                elif len(self.buffer) > 1024:
                    # Buffer太长但没有换行符，强制flush以避免丢失
                    buffer_rstrip = self.buffer.rstrip()
                    if buffer_rstrip:
                        filtered_line = self.filter_code_edit_content(buffer_rstrip)
                        if not self.should_filter_message(filtered_line):
                            line_lower = filtered_line.lower()
                            if ('warning' in line_lower or
                                'progress' in line_lower or
                                'processing files' in line_lower or
                                filtered_line.startswith('Processing files:') or
                                'userwarning' in line_lower or
                                'warnings.warn' in line_lower or
                                '⚠️' in filtered_line or
                                filtered_line.startswith('W: ') or
                                'W: ' in filtered_line):
                                message_type = 'info'
                            else:
                                message_type = self.socket_type
                            
                            self.q.put({'event': 'output', 'data': {'message': filtered_line, 'type': message_type, 'is_update': False}})
                    self.buffer = ""

            def flush(self):
                # Flush buffer to queue if it contains content
                # This ensures that messages are sent immediately when flush() is called
                # 修复丢字问题：即使buffer中没有换行符，也应该发送buffer中的内容
                if self.buffer:
                    # 处理buffer中的所有内容，即使没有换行符
                    # 先检查是否有完整的行（以\n结尾）
                    if '\n' in self.buffer:
                        # 有完整的行，按行处理
                        *lines, remaining = self.buffer.split('\n')
                        for line in lines:
                            if line.strip():
                                line_rstrip = line.rstrip()
                                filtered_line = self.filter_code_edit_content(line_rstrip)
                                if not self.should_filter_message(filtered_line):
                                    line_lower = filtered_line.lower()
                                    if ('warning' in line_lower or
                                        'progress' in line_lower or
                                        'processing files' in line_lower or
                                        filtered_line.startswith('Processing files:') or
                                        'userwarning' in line_lower or
                                        'warnings.warn' in line_lower or
                                        '⚠️' in filtered_line or
                                        filtered_line.startswith('W: ') or
                                        'W: ' in filtered_line):
                                        message_type = 'info'
                                    else:
                                        message_type = self.socket_type
                                    
                                    is_update = '\r' in line
                                    buffer_clean = filtered_line.replace('\r', '')
                                    self.q.put({'event': 'output', 'data': {'message': buffer_clean, 'type': message_type, 'is_update': is_update}})
                        # 保留剩余部分（可能不完整）
                        self.buffer = remaining
                    else:
                        # 没有换行符，直接处理整个buffer
                        buffer_rstrip = self.buffer.rstrip()
                        if buffer_rstrip:
                            filtered_line = self.filter_code_edit_content(buffer_rstrip)
                            if not self.should_filter_message(filtered_line):
                                line_lower = filtered_line.lower()
                                if ('warning' in line_lower or
                                    'progress' in line_lower or
                                    'processing files' in line_lower or
                                    filtered_line.startswith('Processing files:') or
                                    'userwarning' in line_lower or
                                    'warnings.warn' in line_lower or
                                    '⚠️' in filtered_line or
                                    filtered_line.startswith('W: ') or
                                    'W: ' in filtered_line):
                                    message_type = 'info'
                                else:
                                    message_type = self.socket_type
                                
                                is_update = '\r' in self.buffer
                                buffer_clean = filtered_line.replace('\r', '')
                                self.q.put({'event': 'output', 'data': {'message': buffer_clean, 'type': message_type, 'is_update': is_update}})
                        # 清空buffer，因为已经处理了所有内容
                        self.buffer = ""
            
            def final_flush(self):
                if self.buffer.strip():
                    # Filter out redundant system messages (preserve leading spaces)
                    buffer_rstrip = self.buffer.rstrip()  # Only remove trailing spaces, preserve leading spaces
                    if self.should_filter_message(buffer_rstrip):
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
                        '⚠️' in self.buffer or  
                        self.buffer.strip().startswith('W: ') or  # apt warning format
                        'W: ' in self.buffer):  # apt warning format
                        message_type = 'info'
                    else:
                        message_type = self.socket_type
                    
                    # Detect if this is a progress bar update (contains \r)
                    is_update = '\r' in self.buffer
                    # Remove \r from the message for display
                    buffer_rstrip = buffer_rstrip.replace('\r', '')
                    
                    # Display warning and progress info as normal info
                    self.q.put({'event': 'output', 'data': {'message': buffer_rstrip, 'type': message_type, 'is_update': is_update}})
                    self.buffer = ""

        original_stdout = sys.stdout
        original_stderr = sys.stderr
        
        stdout_handler = QueueSocketHandler(output_queue, 'info')
        stderr_handler = QueueSocketHandler(output_queue, 'error')

        try:
            sys.stdout = stdout_handler
            sys.stderr = stderr_handler
            
            success = agia.run(user_requirement=final_requirement, loops=execution_rounds)
            
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
        
        # Initialize concurrency manager with reference to this GUI instance
        self.concurrency_manager = ConcurrencyManager(
            max_concurrent_tasks=16,  # Maximum concurrent tasks (Expand by 1x)
            max_connections=40,       # 最大Connect数 (Expand by 1x)
            gui_instance=self         # Pass GUI instance for unified monitoring
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
        
        # Session cleanup is now handled by ConcurrencyManager unified monitor
        # No separate thread needed
        
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
            pass  # Authentication failed
            return None
        
        # Store guest status and user info
        is_guest = auth_result.get("is_guest", False)
        user_info = auth_result["user_info"]
        
        if session_id not in self.user_sessions:
            # Create authenticated session
            if self.auth_manager.create_session(api_key, session_id):
                self.user_sessions[session_id] = UserSession(session_id, api_key, user_info)
                session_type = "guest" if is_guest else "authenticated"
            else:
                return None
        else:
            # Update API key if it has changed
            existing_session = self.user_sessions[session_id]
            if existing_session.api_key != api_key:
                # Re-authenticate and update session
                if self.auth_manager.create_session(api_key, session_id):
                    self.user_sessions[session_id] = UserSession(session_id, api_key, user_info)
                else:
                    return None
        
        return self.user_sessions[session_id]
    
    def _cleanup_session(self, session_id):
        """Clean up specified session"""
        try:
            if session_id in self.user_sessions:
                user_session = self.user_sessions[session_id]
                
                # Clean up running processes
                if user_session.current_process and user_session.current_process.is_alive():
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
                
        except Exception as e:
                pass  # Session cleanup error
    
    def _handle_user_task_timeout(self, session_id):
        """Handle user task timeout"""
        try:
            if session_id in self.user_sessions:
                user_session = self.user_sessions[session_id]

                # Terminate process
                if user_session.current_process and user_session.current_process.is_alive():
                    user_session.current_process.terminate()
                    user_session.current_process.join(timeout=10)

                    # Send timeout message to user
                    from flask_socketio import emit
                    emit('task_timeout', {
                        'message': f'Task execution timeout ({self.concurrency_manager.task_timeout}seconds)'
                    }, room=session_id)

                # Release task resources - call finish_task to clean up active_tasks
                self.concurrency_manager.finish_task(session_id, success=False)
        except Exception as e:
            pass
    

    
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
                        
                        # 获取任务描述
                        task_description = self.get_task_description_from_manager_out(item_path)
                        
                        result.append({
                        'name': item,
                        'path': item_path,
                        'size': self.format_size(size),
                        'files': self.get_directory_structure(item_path),
                        'is_current': item == user_session.current_output_dir,  # Mark if it's current directory
                        'is_selected': item == user_session.selected_output_dir,  # Mark if it's selected directory
                            'is_last': item == user_session.last_output_dir,  # Mark if it's last used directory
                        'task_description': task_description  # 任务描述
                        })
        except (OSError, PermissionError) as e:
            pass
        
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
    
    def get_directory_structure(self, directory, max_depth=10, current_depth=0, base_dir=None):
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
    
    def get_task_description_from_manager_out(self, directory_path):
        """从manager.out文件中读取任务描述
        
        Args:
            directory_path: 目录路径
            
        Returns:
            str: 任务描述（最后一个user_requirement），如果没有找到则返回i18n翻译后的"未布置任务"
        """
        # 获取i18n文本
        i18n = get_i18n_texts()
        no_task_text = i18n.get('no_task_assigned', '未布置任务')
        
        manager_out_path = os.path.join(directory_path, 'logs', 'manager.out')
        
        # 检查文件是否存在
        if not os.path.exists(manager_out_path):
            return no_task_text
        
        try:
            # 读取文件内容
            with open(manager_out_path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
            
            # 从后往前查找"Received user requirement:"行（获取最后一个，即最新的用户需求）
            task_description = None
            for line in reversed(lines):
                if "Received user requirement:" in line:
                    # 提取冒号后面的内容
                    parts = line.split("Received user requirement:", 1)
                    if len(parts) > 1:
                        task_description = parts[1].strip()
                        break
            
            # 如果找到了任务描述，返回它；否则返回默认值
            return task_description if task_description else no_task_text
            
        except (IOError, OSError, UnicodeDecodeError) as e:
            # 如果读取失败，返回默认值
            return no_task_text

class UserSession:
    def __init__(self, session_id, api_key=None, user_info=None):
        self.session_id = session_id
        self.api_key = api_key
        self.user_info = user_info or {}
        self.client_session_id = None  # 客户端持久化会话ID
        self.current_process = None
        self.output_queue = None
        self.input_queue = None  # Queue for user input in GUI mode
        self.current_output_dir = None  # Track current execution output directory
        self.last_output_dir = None     # Track last used output directory
        self.selected_output_dir = None # Track user selected output directory
        self.conversation_history = []  # Store conversation history for this user
        self.queue_reader_stop_flag = None  # 用于停止queue_reader_thread的标志
        self.queue_reader_thread = None  # 当前运行的queue_reader_thread引用
        self.terminal_cwd = None  # 终端当前工作目录，用于维护cd命令的状态
        
        # Determine user directory based on user info
        # Priority: name (if exists and not "guest") > is_guest > api_key hash > default
        if user_info and user_info.get("name"):
            username = user_info.get("name")
            # Only use "guest" directory if name is explicitly "guest" AND is_guest is True
            if username.lower() == "guest" and user_info.get("is_guest", False):
                self.user_dir_name = "guest"
            else:
                # Use username as directory name, sanitize for filesystem safety
                import re
                # Remove or replace characters that are not safe for directory names
                safe_username = re.sub(r'[<>:"/\\|?*]', '_', username)
                # Remove leading/trailing spaces and dots
                safe_username = safe_username.strip(' .')
                # Ensure it's not empty after sanitization
                if not safe_username:
                    safe_username = "user"
                self.user_dir_name = safe_username
        elif user_info and user_info.get("is_guest", False):
            # Guest user without name gets a special directory
            self.user_dir_name = "guest"
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
    
    def get_terminal_cwd(self, base_dir, force_recalculate=False):
        """Get terminal current working directory, initialize if not set"""
        if self.terminal_cwd is None or force_recalculate:
            # 确定要使用的工作目录（按优先级）
            target_dir = None
            
            # 优先级1: selected_output_dir (用户选择的工作目录)
            if self.selected_output_dir:
                target_dir = self.selected_output_dir
            # 优先级2: current_output_dir (当前执行的任务目录)
            elif self.current_output_dir:
                target_dir = self.current_output_dir
            # 优先级3: last_output_dir (最后使用的目录)
            elif self.last_output_dir:
                target_dir = self.last_output_dir
            
            if target_dir:
                # 使用工作目录的workspace子目录
                user_dir = self.get_user_directory(base_dir)
                workspace_dir = os.path.join(user_dir, target_dir, 'workspace')
                if os.path.exists(workspace_dir) and os.path.isdir(workspace_dir):
                    self.terminal_cwd = workspace_dir
                else:
                    # workspace不存在，使用工作目录本身
                    output_dir = os.path.join(user_dir, target_dir)
                    if os.path.exists(output_dir) and os.path.isdir(output_dir):
                        self.terminal_cwd = output_dir
                    else:
                        # 工作目录不存在，使用用户目录
                        self.terminal_cwd = self.get_user_directory(base_dir)
                        os.makedirs(self.terminal_cwd, exist_ok=True)
            else:
                # 没有可用的工作目录，使用用户目录
                self.terminal_cwd = self.get_user_directory(base_dir)
                os.makedirs(self.terminal_cwd, exist_ok=True)
            
            # 确保返回绝对路径
            if self.terminal_cwd:
                self.terminal_cwd = os.path.abspath(self.terminal_cwd)
        return self.terminal_cwd
    
    def set_terminal_cwd(self, new_cwd):
        """Set terminal current working directory"""
        if new_cwd:
            # 确保是绝对路径
            new_cwd = os.path.abspath(new_cwd)
            if os.path.exists(new_cwd) and os.path.isdir(new_cwd):
                self.terminal_cwd = new_cwd
                return True
        return False
    
    def add_to_conversation_history(self, user_input, result_summary=None):
        """Add a conversation turn to history"""
        conversation_entry = {
            'timestamp': datetime.datetime.now().isoformat(),
            'user_input': user_input,
            'result_summary': result_summary or "Task executed",
            'output_dir': self.current_output_dir
        }
        self.conversation_history.append(conversation_entry)
        
        # Keep only last 10 conversations to avoid memory issues
        if len(self.conversation_history) > 10:
            self.conversation_history = self.conversation_history[-10:]
    
    def get_summarized_requirements(self):
        """从manager.out文件中提取历史user requirements并汇总

        从用户的各个output目录中读取manager.out文件，提取所有历史user requirements，
        按时间排序并返回最近的几个需求。

        Returns:
            str: 汇总的历史需求，如果没有找到则返回None
        """
        # 获取用户的所有output目录
        user_output_dirs = []
        try:
            user_base_dir = self.get_user_directory(gui_instance.base_data_dir)
            if os.path.exists(user_base_dir):
                # 遍历用户目录下的所有子目录
                for item in os.listdir(user_base_dir):
                    item_path = os.path.join(user_base_dir, item)
                    if os.path.isdir(item_path):
                        # 检查是否包含workspace子目录（表示是有效的output目录）
                        workspace_path = os.path.join(item_path, 'workspace')
                        if os.path.exists(workspace_path) and os.path.isdir(workspace_path):
                            user_output_dirs.append(item_path)
        except (OSError, PermissionError):
            pass

        if not user_output_dirs:
            return None

        # 从每个目录的manager.out文件中提取历史需求
        all_requirements = []
        for output_dir in user_output_dirs:
            manager_out_path = os.path.join(output_dir, 'logs', 'manager.out')
            if os.path.exists(manager_out_path):
                try:
                    with open(manager_out_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()

                    # 查找所有"Received user requirement:"行
                    lines = content.split('\n')
                    for line in lines:
                        if "Received user requirement:" in line:
                            # 提取冒号后面的内容
                            parts = line.split("Received user requirement:", 1)
                            if len(parts) > 1:
                                requirement = parts[1].strip()
                                if requirement:  # 确保不为空
                                    # 获取目录的修改时间作为时间戳
                                    try:
                                        dir_mtime = os.path.getmtime(output_dir)
                                        timestamp = datetime.datetime.fromtimestamp(dir_mtime).isoformat()
                                    except:
                                        timestamp = datetime.datetime.now().isoformat()

                                    all_requirements.append({
                                        'requirement': requirement,
                                        'timestamp': timestamp,
                                        'output_dir': os.path.basename(output_dir)
                                    })
                except (IOError, OSError, UnicodeDecodeError):
                    continue

        if not all_requirements:
            return None

        # 按时间戳排序（最新的在前）
        all_requirements.sort(key=lambda x: x['timestamp'], reverse=True)

        # 取最近的5个需求
        recent_requirements = all_requirements[:5]

        # 生成汇总文本
        history_summary = []
        for req in recent_requirements:
            history_summary.append(f"User requested: {req['requirement']}")

        return "\n".join(history_summary)

gui_instance = AGIAgentGUI()

def create_temp_session_id(request, api_key=None):
    """Create a temporary session ID for API calls with user isolation"""
    import hashlib
    api_key_hash = hashlib.sha256((api_key or "default").encode()).hexdigest()[:8]
    # Use consistent session ID based on IP and API key, not request ID
    return f"api_{request.remote_addr}_{api_key_hash}"

def stop_queue_reader_thread(user_session):
    """安全地停止queue_reader_thread"""
    import datetime
    if user_session.queue_reader_stop_flag:
        print(f"[{datetime.datetime.now().isoformat()}] 🛑 Stopping old queue reader thread: session_id={user_session.session_id}")
        user_session.queue_reader_stop_flag.set()
        # 给线程一点时间退出
        import time
        time.sleep(0.5)
        user_session.queue_reader_stop_flag = None
        user_session.queue_reader_thread = None

def queue_reader_thread(session_id):
    """Reads from the queue and emits messages to the client via SocketIO."""
    
    def safe_emit(event, data=None, room=None):
        """安全地发送消息，捕获所有异常以避免线程崩溃"""
        try:
            if data is None:
                socketio.emit(event, room=room or session_id)
            else:
                socketio.emit(event, data, room=room or session_id)
        except Exception as emit_error:
            # 如果发送失败（通常是客户端已断开），静默处理
            # 如果是因为客户端断开，应该退出线程
            if 'disconnected' in str(emit_error).lower() or 'not connected' in str(emit_error).lower():
                return False
        return True
    
    if session_id not in gui_instance.user_sessions:
        return
    
    user_session = gui_instance.user_sessions[session_id]
    
    # 创建新的停止标志
    import threading
    stop_flag = threading.Event()
    user_session.queue_reader_stop_flag = stop_flag
    
    while True:
        try:
            # 检查停止标志
            if stop_flag.is_set():
                print(f"[{datetime.datetime.now().isoformat()}] 🛑 Queue reader thread stopped by flag: session_id={session_id}")
                break
                
            if user_session.current_process and not user_session.current_process.is_alive() and user_session.output_queue.empty():
                break

            message = user_session.output_queue.get(timeout=1)
            
            if message.get('event') == 'STOP':
                break
            
            # Check for GUI_USER_INPUT_REQUEST marker in output messages
            # Also check for QUERY: and TIMEOUT: messages that might arrive out of order
            if message.get('event') == 'output':
                data = message.get('data', {})
                msg_text = data.get('message', '')
                
                # Check if this is a QUERY: message (might arrive before GUI_USER_INPUT_REQUEST)
                if msg_text.startswith('QUERY: '):
                    # Store query for later use
                    if not hasattr(user_session, '_pending_user_query'):
                        user_session._pending_user_query = {}
                    user_session._pending_user_query['query'] = msg_text[7:]  # Remove 'QUERY: ' prefix
                    # Don't emit this system message to frontend - it's only for internal processing
                    continue
                
                # Check if this is a TIMEOUT: message
                elif msg_text.startswith('TIMEOUT: '):
                    # Store timeout for later use
                    if not hasattr(user_session, '_pending_user_query'):
                        user_session._pending_user_query = {}
                    timeout_str = msg_text[9:]  # Remove 'TIMEOUT: ' prefix
                    try:
                        user_session._pending_user_query['timeout'] = int(timeout_str)
                    except:
                        user_session._pending_user_query['timeout'] = 10
                    # Don't emit this system message to frontend - it's only for internal processing
                    continue
                
                # Check for GUI_USER_INPUT_REQUEST marker
                elif '🔔 GUI_USER_INPUT_REQUEST' in msg_text:
                    # Extract query and timeout from subsequent messages or use stored values
                    query = None
                    timeout = 10
                    timeout_found = False
                    
                    # Check if we already have stored query/timeout from previous messages
                    if hasattr(user_session, '_pending_user_query'):
                        query = user_session._pending_user_query.get('query')
                        stored_timeout = user_session._pending_user_query.get('timeout')
                        if stored_timeout is not None:
                            timeout = stored_timeout
                            timeout_found = True
                        # Clear stored values
                        delattr(user_session, '_pending_user_query')
                    
                    # Store messages that are not QUERY/TIMEOUT for later emission
                    pending_messages = []
                    # Read more messages to get query and timeout (increased from 15 to 30)
                    # Also increase timeout per message to handle slow message delivery
                    for _ in range(30):  # Read up to 30 more messages to ensure we get QUERY and TIMEOUT
                        # 检查停止标志
                        if stop_flag.is_set():
                            break
                        try:
                            next_msg = user_session.output_queue.get(timeout=2.0)  # Increased timeout from 1.0 to 2.0
                            if next_msg.get('event') == 'output':
                                next_data = next_msg.get('data', {})
                                next_text = next_data.get('message', '')
                                if next_text.startswith('QUERY: '):
                                    query = next_text[7:]  # Remove 'QUERY: ' prefix
                                elif next_text.startswith('TIMEOUT: '):
                                    timeout_str = next_text[9:]  # Remove 'TIMEOUT: ' prefix
                                    try:
                                        timeout = int(timeout_str)
                                        timeout_found = True
                                    except:
                                        timeout = 10
                                else:
                                    # Store other messages to emit later
                                    pending_messages.append(next_msg)
                            else:
                                # Store non-output messages to emit later
                                pending_messages.append(next_msg)
                            
                            # If we found both query and timeout, we can break
                            if query and timeout_found:
                                break
                        except queue.Empty:
                            # If queue is empty, wait a bit more and try to read remaining messages
                            # This handles the case where messages are still being written
                            import time
                            time.sleep(0.1)  # Small delay to allow messages to arrive
                            # Try one more time with shorter timeout
                            try:
                                next_msg = user_session.output_queue.get(timeout=0.5)
                                if next_msg.get('event') == 'output':
                                    next_data = next_msg.get('data', {})
                                    next_text = next_data.get('message', '')
                                    if next_text.startswith('QUERY: '):
                                        query = next_text[7:]
                                    elif next_text.startswith('TIMEOUT: '):
                                        timeout_str = next_text[9:]
                                        try:
                                            timeout = int(timeout_str)
                                            timeout_found = True
                                        except:
                                            timeout = 10
                                    else:
                                        pending_messages.append(next_msg)
                                else:
                                    pending_messages.append(next_msg)
                                if query and timeout_found:
                                    break
                            except queue.Empty:
                                break
                    
                    # If we found query (either from stored value or from queue), send the request
                    if query:
                        # Send user_input_request event to GUI
                        if not safe_emit('user_input_request', {
                            'query': query,
                            'timeout': timeout
                        }):
                            break
                        # Emit pending messages that were read while looking for QUERY/TIMEOUT
                        for pending_msg in pending_messages:
                            if not safe_emit(pending_msg['event'], pending_msg.get('data', {})):
                                break
                        continue  # Don't emit the marker message itself
                    else:
                        # If query not found after all attempts, emit all pending messages
                        # Emit all pending messages including the marker
                        for pending_msg in pending_messages:
                            if not safe_emit(pending_msg['event'], pending_msg.get('data', {})):
                                break
                        # Still emit the original marker message so user can see something happened
                        if not safe_emit(message['event'], message.get('data', {})):
                            break
            
            # If task completion message, save last used directory and clear current directory mark
            if message.get('event') in ['task_completed', 'error']:
                # Release task resources
                task_success = message.get('event') == 'task_completed'
                gui_instance.concurrency_manager.finish_task(session_id, success=task_success)
                
                # Get updated metrics
                metrics = gui_instance.concurrency_manager.get_metrics()
                status_msg = "Complete" if task_success else "Failed"
                
                if user_session.current_output_dir:
                    user_session.last_output_dir = user_session.current_output_dir
                    # If current directory is the selected directory, keep the selection
                    # This ensures user can continue in the same directory
                    if user_session.selected_output_dir == user_session.current_output_dir:
                        pass
                    else:
                        # If different directories, clear selection to avoid confusion
                        user_session.selected_output_dir = None
                
                # Add to conversation history if we have context from last executed task
                if hasattr(user_session, '_current_task_requirement'):
                    result_summary = "Task completed successfully" if task_success else "Task failed or had errors"
                    user_session.add_to_conversation_history(user_session._current_task_requirement, result_summary)
                    delattr(user_session, '_current_task_requirement')
                
                user_session.current_output_dir = None
            
            # Emit to user's specific room (but filter out system markers)
            if message.get('event') == 'output':
                data = message.get('data', {})
                msg_text = data.get('message', '')
                # Don't emit system markers to frontend (they're handled internally)
                if '🔔 GUI_USER_INPUT_REQUEST' in msg_text or msg_text.startswith('QUERY: ') or msg_text.startswith('TIMEOUT: '):
                    continue  # Skip emitting these system messages
            
            if not safe_emit(message['event'], message.get('data', {})):
                break  # 客户端已断开，退出线程
        except queue.Empty:
            continue
        except Exception as e:
            # 静默处理异常，避免线程崩溃
            break
    
    # 清理停止标志
    if user_session.queue_reader_stop_flag == stop_flag:
        user_session.queue_reader_stop_flag = None
        user_session.queue_reader_thread = None
    
    if user_session.current_process and hasattr(user_session.current_process, '_popen') and user_session.current_process._popen is not None:
        try:
            user_session.current_process.join(timeout=1)
        except Exception as e:
            pass
    user_session.current_process = None
    user_session.output_queue = None
    if user_session.current_output_dir:
        user_session.last_output_dir = user_session.current_output_dir
    user_session.current_output_dir = None  # Clear current directory mark

@app.route('/')
def index():
    """Main page"""
    # Support language switching via URL parameter
    lang_param = request.args.get('lang')
    if lang_param and lang_param in ('zh', 'en'):
        current_lang = lang_param
    else:
        current_lang = get_language()
    
    i18n = get_i18n_texts()
    # Override i18n if language is specified via URL
    if lang_param and lang_param in ('zh', 'en'):
        i18n = I18N_TEXTS.get(lang_param, I18N_TEXTS['en'])
    
    mcp_servers = get_mcp_servers_config()
    
    # Load GUI virtual terminal configuration
    config = load_config()
    gui_virtual_terminal = config.get('GUI_virtual_terminal', 'False').lower() == 'true'
    
    return render_template('index.html', i18n=i18n, lang=current_lang, mcp_servers=mcp_servers, gui_virtual_terminal=gui_virtual_terminal)

@app.route('/terminal')
def terminal():
    """Terminal page"""
    i18n = get_i18n_texts()
    current_lang = get_language()
    
    # Load GUI virtual terminal configuration
    config = load_config()
    gui_virtual_terminal = config.get('GUI_virtual_terminal', 'False').lower() == 'true'
    
    return render_template('terminal.html', i18n=i18n, lang=current_lang, gui_virtual_terminal=gui_virtual_terminal)

@app.route('/register')
def register():
    """User registration page"""
    i18n = get_i18n_texts()
    current_lang = get_language()
    return render_template('register.html', i18n=i18n, lang=current_lang)

@app.route('/api/register', methods=['POST'])
def api_register():
    """API endpoint for user registration"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'error': '无效的请求数据'}), 400

        username = data.get('username', '').strip()
        phone_number = data.get('phone_number', '').strip()

        if not username or not phone_number:
            return jsonify({'success': False, 'error': '用户名和手机号为必填项'}), 400

        # Register user
        result = gui_instance.auth_manager.register_user(username, phone_number)

        if result['success']:
            return jsonify({
                'success': True,
                'api_key': result['api_key'],
                'user_info': result['user_info'],
                'message': '注册成功！请妥善保存您的API密钥。'
            })
        else:
            return jsonify({'success': False, 'error': result['error']}), 400

    except Exception as e:
        return jsonify({'success': False, 'error': '注册过程中发生错误'}), 500

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
        
        # Security check: normalize path and prevent path traversal
        # Don't use secure_filename as it destroys Chinese characters
        normalized_dir_name = os.path.normpath(dir_name)
        if '..' in normalized_dir_name or normalized_dir_name.startswith('/'):
            return jsonify({'success': False, 'error': 'Access denied: Invalid directory path'})
        
        dir_path = os.path.join(user_base_dir, normalized_dir_name)
        
        # Security check: ensure directory is within user's output directory
        real_output_dir = os.path.realpath(user_base_dir)
        real_dir_path = os.path.realpath(dir_path)
        if not real_dir_path.startswith(real_output_dir):
            return jsonify({'success': False, 'error': 'Access denied: Invalid directory path'})
        
        if not os.path.exists(dir_path) or not os.path.isdir(dir_path):
            return jsonify({'success': False, 'error': 'Directory not found'})
        
        # Create temporary zip file in a more reliable location
        import tempfile
        import io
        
        # Create zip file in memory to avoid file system timing issues
        memory_file = io.BytesIO()
        
        try:
            with zipfile.ZipFile(memory_file, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as zipf:
                for root, dirs, files in os.walk(dir_path):
                    # Exclude code_index directory and other unwanted directories
                    dirs_to_exclude = {'code_index', '__pycache__', '.git', '.vscode', 'node_modules'}
                    if any(excluded in root for excluded in dirs_to_exclude):
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
                            continue
            
            # Get the zip file size and seek to beginning
            memory_file.seek(0, 2)  # Seek to end
            file_size = memory_file.tell()
            memory_file.seek(0)  # Seek to beginning
            
            # Verify that the zip file is not empty
            if file_size == 0:
                return jsonify({'success': False, 'error': 'Failed to create zip file or zip file is empty'})
            
            # Return the file with proper headers
            # Using memory file means no cleanup needed
            return send_file(
                memory_file, 
                as_attachment=True, 
                download_name=f"{dir_name}.zip",
                mimetype='application/zip'
            )
            
        except Exception as zip_error:
            # No cleanup needed for memory file
            raise zip_error
    
    except Exception as e:
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

@app.route('/api/office-file/<path:file_path>', methods=['GET', 'OPTIONS'])
def get_office_file(file_path):
    """Get office file for browser-based preview (mammoth.js)"""
    # Handle CORS preflight request
    if request.method == 'OPTIONS':
        response = Response()
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Methods'] = 'GET, OPTIONS'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type, X-API-Key'
        response.headers['Access-Control-Max-Age'] = '3600'
        return response
    
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
            abort(403)
        
        if not os.path.exists(full_path) or not os.path.isfile(full_path):
            abort(404)
        
        # Get file extension and set appropriate mimetype
        _, ext = os.path.splitext(full_path.lower())
        
        # Define mimetypes for office files
        mimetype_map = {
            '.doc': 'application/msword',
            '.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
            '.xls': 'application/vnd.ms-excel',
            '.xlsx': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            '.ppt': 'application/vnd.ms-powerpoint',
            '.pptx': 'application/vnd.openxmlformats-officedocument.presentationml.presentation',
        }
        
        mimetype = mimetype_map.get(ext, 'application/octet-stream')
        
        # Create response with CORS headers for browser-based preview
        response = send_file(full_path, mimetype=mimetype)
        
        # Add CORS headers to allow browser to load the file
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Methods'] = 'GET, OPTIONS'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type, X-API-Key'
        response.headers['Access-Control-Max-Age'] = '3600'
        
        return response
    
    except Exception as e:
        print(f"Error serving office file {file_path}: {str(e)}")
        import traceback
        traceback.print_exc()
        abort(500)

@app.route('/api/file/<path:file_path>')
def get_file_content(file_path):
    """Get file content"""
    try:
        # Get API key from query parameters or headers
        api_key = request.args.get('api_key') or request.headers.get('X-API-Key')
        
        # Create a temporary session for API calls
        temp_session_id = create_temp_session_id(request, api_key)
        user_session = gui_instance.get_user_session(temp_session_id, api_key)
        if not user_session:
            # 提供更详细的错误信息用于调试
            import traceback
            debug_info = {
                'api_key_provided': bool(api_key),
                'api_key_length': len(api_key) if api_key else 0,
                'temp_session_id': temp_session_id,
                'remote_addr': request.remote_addr
            }
            print(f"❌ SVG预览认证失败: {debug_info}")
            return jsonify({
                'success': False, 
                'error': 'Authentication failed or session creation failed. Please ensure you are connected with a valid API key.',
                'debug': debug_info if os.environ.get('FLASK_DEBUG') == '1' else None
            })
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
                'file_path': file_path,  # Add file path for HTML preview
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
        pass
        
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
        
        # Check if it's a PDF file
        if not full_path.lower().endswith('.pdf'):
            return jsonify({'success': False, 'error': 'Not a PDF file'})
        
        # Verify PDF file structure
        try:
            with open(full_path, 'rb') as f:
                header = f.read(8)
                if not header.startswith(b'%PDF-'):
                    return jsonify({'success': False, 'error': 'Invalid PDF file structure'})
        except Exception as pdf_check_error:
            return jsonify({'success': False, 'error': f'PDF validation failed: {str(pdf_check_error)}'})
        
        response = send_file(full_path, mimetype='application/pdf')
        
        # Add CORS headers
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Methods'] = 'GET, OPTIONS'
        response.headers['Access-Control-Allow-Headers'] = 'X-API-Key, Content-Type'
        
        return response
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/static-file/<path:file_path>')
def serve_static_file(file_path):
    """Serve static files for HTML preview (JS, CSS, images, etc.)"""
    try:
        # Get API key from query parameters or headers
        api_key = request.args.get('api_key') or request.headers.get('X-API-Key')
        
        # Create a temporary session for API calls
        temp_session_id = create_temp_session_id(request, api_key)
        user_session = gui_instance.get_user_session(temp_session_id, api_key)
        if not user_session:
            return jsonify({'success': False, 'error': 'Authentication failed or session creation failed'}), 403
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
            abort(403)
        
        if not os.path.exists(full_path) or not os.path.isfile(full_path):
            abort(404)
        
        # Get file extension and determine mimetype
        _, ext = os.path.splitext(full_path.lower())
        
        # Define mimetypes for different file types
        mimetype_map = {
            '.js': 'application/javascript',
            '.css': 'text/css',
            '.html': 'text/html',
            '.htm': 'text/html',
            '.json': 'application/json',
            '.xml': 'application/xml',
            '.txt': 'text/plain',
            '.png': 'image/png',
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.gif': 'image/gif',
            '.svg': 'image/svg+xml',
            '.webp': 'image/webp',
            '.ico': 'image/x-icon',
            '.bmp': 'image/bmp',
            '.woff': 'font/woff',
            '.woff2': 'font/woff2',
            '.ttf': 'font/ttf',
            '.eot': 'application/vnd.ms-fontobject',
            '.otf': 'font/otf',
            '.mp3': 'audio/mpeg',
            '.wav': 'audio/wav',
            '.ogg': 'audio/ogg',
            '.mp4': 'video/mp4',
            '.webm': 'video/webm',
            '.avi': 'video/x-msvideo',
            '.mov': 'video/quicktime'
        }
        
        mimetype = mimetype_map.get(ext, 'application/octet-stream')
        
        # For text-based files, try to read with UTF-8 encoding
        if ext in ['.js', '.css', '.html', '.htm', '.json', '.svg', '.xml', '.txt']:
            try:
                with open(full_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                return Response(content, mimetype=mimetype, headers={
                    'Cache-Control': 'no-cache',
                    'Access-Control-Allow-Origin': '*'
                })
            except UnicodeDecodeError:
                # Fallback to binary mode if UTF-8 fails
                pass
        
        # For binary files or if UTF-8 failed, serve as binary
        return send_file(full_path, mimetype=mimetype, as_attachment=False)
        
    except Exception as e:
        print(f"Error serving static file {file_path}: {str(e)}")
        import traceback
        traceback.print_exc()
        abort(500)

@app.route('/api/html-preview/<path:file_path>')
def serve_html_preview(file_path):
    """Serve HTML file with proper base URL for relative resource loading"""
    try:
        # Get API key from query parameters or headers
        api_key = request.args.get('api_key') or request.headers.get('X-API-Key')
        
        # Create a temporary session for API calls
        temp_session_id = create_temp_session_id(request, api_key)
        user_session = gui_instance.get_user_session(temp_session_id, api_key)
        if not user_session:
            return jsonify({'success': False, 'error': 'Authentication failed or session creation failed'}), 403
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
            abort(403)
        
        if not os.path.exists(full_path) or not os.path.isfile(full_path):
            abort(404)
        
        # Read HTML content
        with open(full_path, 'r', encoding='utf-8', errors='ignore') as f:
            html_content = f.read()
        
        # Get the directory of the HTML file for base URL
        file_dir = os.path.dirname(file_path)
        
        # Inject base tag to handle relative paths
        if file_dir:
            # Ensure the base URL ends with a slash for proper relative path resolution
            base_url = f"/api/static-file/{file_dir}/"
        else:
            base_url = "/api/static-file/"
        
        # Don't add API key to base URL as it doesn't work properly with relative paths
        # Instead, we'll modify the HTML content to include API key in script/link tags
        
        # Process HTML content to add API key to relative resource URLs
        import re
        
        # Function to add API key to relative URLs
        def add_api_key_to_url(url):
            if url.startswith(('http://', 'https://', '//', 'data:', 'javascript:', 'mailto:')):
                return url  # Don't modify absolute URLs or special schemes
            if url.startswith('/'):
                return url  # Don't modify root-relative URLs
            
            # Add API key to relative URLs
            separator = '&' if '?' in url else '?'
            if api_key and api_key != 'default':
                return f"{base_url}{url}{separator}api_key={api_key}"
            else:
                return f"{base_url}{url}"
        
        # Replace src attributes in script tags
        html_content = re.sub(
            r'(<script[^>]+src=")([^"]+)(")',
            lambda m: m.group(1) + add_api_key_to_url(m.group(2)) + m.group(3),
            html_content,
            flags=re.IGNORECASE
        )
        
        # Replace href attributes in link tags (CSS, etc.)
        html_content = re.sub(
            r'(<link[^>]+href=")([^"]+)(")',
            lambda m: m.group(1) + add_api_key_to_url(m.group(2)) + m.group(3),
            html_content,
            flags=re.IGNORECASE
        )
        
        # Replace src attributes in img tags
        html_content = re.sub(
            r'(<img[^>]+src=")([^"]+)(")',
            lambda m: m.group(1) + add_api_key_to_url(m.group(2)) + m.group(3),
            html_content,
            flags=re.IGNORECASE
        )
        
        # Also handle single quotes
        html_content = re.sub(
            r"(<script[^>]+src=')([^']+)(')",
            lambda m: m.group(1) + add_api_key_to_url(m.group(2)) + m.group(3),
            html_content,
            flags=re.IGNORECASE
        )
        
        html_content = re.sub(
            r"(<link[^>]+href=')([^']+)(')",
            lambda m: m.group(1) + add_api_key_to_url(m.group(2)) + m.group(3),
            html_content,
            flags=re.IGNORECASE
        )
        
        html_content = re.sub(
            r"(<img[^>]+src=')([^']+)(')",
            lambda m: m.group(1) + add_api_key_to_url(m.group(2)) + m.group(3),
            html_content,
            flags=re.IGNORECASE
        )
        
        return Response(html_content, mimetype='text/html')
        
    except Exception as e:
        print(f"Error serving HTML preview {file_path}: {str(e)}")
        import traceback
        traceback.print_exc()
        abort(500)

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
                sys.executable,  # Use current Python executable instead of hardcoded 'python3'
                str(trans_script),
                md_path.name,  # Use filename instead of full path
                latex_file.name,  # Use filename instead of full path
                '--latex'  # Add LaTeX flag
            ]
            
            # Execute command in markdown file directory
            result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='ignore', cwd=str(output_dir))
            
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
                    '--syntax-highlighting=tango',
                    '-V', 'colorlinks=true',
                    '-V', 'linkcolor=blue',
                    '-V', 'urlcolor=blue',
                    '--toc',
                    '--wrap=preserve'
                ])
                
                result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='ignore', cwd=str(output_dir))
                
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
        from src.tools.print_system import set_output_directory
        from src.tools.agent_context import set_current_agent_id
        
        # Set up logging directory for conversion operations
        # This ensures print_debug() can write to manager.log in the user's output directory
        set_output_directory(user_base_dir)
        set_current_agent_id('manager')  # Set agent ID to 'manager' so logs go to manager.log
        
        tools = Tools(
            workspace_root=user_base_dir,
            out_dir=user_base_dir
        )
        
        # Call the conversion method from FileSystemTools
        
        # Handle LaTeX conversion separately if requested
        if format_type == 'latex':
            conversion_result = convert_markdown_to_latex_only(full_path, file_path, user_base_dir)
        else:
            conversion_result = tools._convert_markdown_to_formats(full_path, file_path, format_type)
        
        
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
            
            # Add warnings for failed conversions and log detailed errors
            if failed_conversions:
                warnings = []
                for conv_type in failed_conversions:
                    conv_result = conversions[conv_type]
                    conv_error = conv_result.get('error', 'Unknown error')
                    
                    # Log detailed error information
                    try:
                        from src.tools.print_system import print_debug
                        print_debug(f"❌ {conv_type.upper()} conversion failed for file: {file_path}")
                        print_debug(f"Error: {conv_error}")
                        if conv_result.get('stderr'):
                            print_debug(f"stderr: {conv_result.get('stderr')}")
                        if conv_result.get('stdout'):
                            print_debug(f"stdout: {conv_result.get('stdout')}")
                        if conv_result.get('return_code') is not None:
                            print_debug(f"Return code: {conv_result.get('return_code')}")
                    except Exception:
                        pass  # If logging fails, continue
                    
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
            
            # Log conversion failure to manager.log
            try:
                from src.tools.print_system import print_debug
                print_debug(f"❌ Markdown conversion failed for file: {file_path}")
                print_debug(f"Error: {error_msg}")
                if conversion_result.get('conversions'):
                    for conv_type, conv_result in conversion_result.get('conversions', {}).items():
                        if conv_result.get('status') == 'failed':
                            print_debug(f"  {conv_type.upper()} conversion failed: {conv_result.get('error', 'Unknown error')}")
            except Exception:
                pass  # If logging fails, continue with error response
            
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
        import traceback
        error_traceback = traceback.format_exc()
        traceback.print_exc()
        
        # Log the error to manager.log if logging is set up
        try:
            from src.tools.print_system import print_debug
            print_debug(f"❌ PDF conversion error: {str(e)}")
            print_debug(f"Traceback:\n{error_traceback}")
        except Exception:
            # If logging fails, at least we have the traceback printed above
            pass
        
        return jsonify({
            'success': False, 
            'error': f'Error occurred during conversion: {str(e)}',
            'traceback': error_traceback if app.debug else None  # Only include traceback in debug mode
        })

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
        if not user_session:
            return jsonify({'success': False, 'error': 'Authentication failed or session creation failed. Please ensure you are connected with a valid API key.'})
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
            
            return jsonify(result)
        else:
            return jsonify({
                'success': False,
                'error': 'Failed to generate images from Mermaid chart'
            })
    
    except Exception as e:
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
    import datetime
    i18n = get_i18n_texts()
    session_id = request.sid
    
    # Get user authentication info and client session ID
    api_key = None
    client_session_id = None
    if auth:
        api_key = auth.get('api_key')
        # Convert empty string to None for guest access
        if api_key == "":
            api_key = None
        client_session_id = auth.get('client_session_id')
    
    
    # 检查是否有待恢复的会话（使用client_session_id匹配）
    recovered_session = None
    old_socket_sid = None
    if client_session_id:
        # 优先使用client_session_id匹配
        for old_sid, pending_info in list(_pending_cleanup_sessions.items()):
            if pending_info.get('client_session_id') == client_session_id:
                # 找到同一客户端的待清理会话，恢复它
                recovered_session = pending_info['user_session']
                old_socket_sid = old_sid
                del _pending_cleanup_sessions[old_sid]
                # 也从旧的 user_sessions 中移除
                if old_sid in gui_instance.user_sessions:
                    del gui_instance.user_sessions[old_sid]
                print(f"[{datetime.datetime.now().isoformat()}] 🔄 Restoring session by client_sid: old_socket_sid={old_sid}, new_socket_sid={session_id}, client_sid={client_session_id}")
                break
    
    # 如果没有通过client_session_id恢复，尝试通过api_key恢复（兼容旧版本）
    if not recovered_session and api_key:
        for old_sid, pending_info in list(_pending_cleanup_sessions.items()):
            if pending_info['api_key'] == api_key:
                # 找到同一用户的待清理会话，恢复它
                recovered_session = pending_info['user_session']
                old_socket_sid = old_sid
                del _pending_cleanup_sessions[old_sid]
                # 也从旧的 user_sessions 中移除
                if old_sid in gui_instance.user_sessions:
                    del gui_instance.user_sessions[old_sid]
                print(f"[{datetime.datetime.now().isoformat()}] 🔄 Restoring session by api_key: old_socket_sid={old_sid}, new_socket_sid={session_id}")
                break
    
    # Check if new connections can be accepted
    if not gui_instance.concurrency_manager.can_accept_connection():
        emit('connection_rejected', {
            'message': 'Server connection limit reached'
        }, room=session_id)
        return False
    
    # Create or get user session with authentication
    if recovered_session:
        # 使用恢复的会话
        user_session = recovered_session
        # 更新session_id到新的socket session_id
        user_session.session_id = session_id
        # 保存client_session_id
        if client_session_id:
            user_session.client_session_id = client_session_id
        gui_instance.user_sessions[session_id] = user_session
        # 重新创建认证会话 - 使用保存的api_key
        gui_instance.auth_manager.create_session(user_session.api_key, session_id)
    else:
        user_session = gui_instance.get_user_session(session_id, api_key)
        # 保存client_session_id
        if user_session and client_session_id:
            user_session.client_session_id = client_session_id
    
    if not user_session:
        # Authentication failed
        emit('auth_failed', {'message': 'Authentication failed. Please check your API key.'}, room=session_id)
        return False
    
    # Add connection to concurrency manager
    if not gui_instance.concurrency_manager.add_connection():
        emit('connection_rejected', {
            'message': 'Server connection limit reached'
        }, room=session_id)
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
    
    
    # 检查是否有正在运行的任务（重连恢复的情况）
    # 🔧 Fix: 更准确地检查任务状态
    # 1. 检查进程对象是否存在
    # 2. 检查进程是否真的在运行
    # 3. 检查是否有当前输出目录（任务完成时会清理 current_output_dir）
    # 4. 检查会话是否在活跃任务列表中
    has_process = user_session.current_process is not None
    process_alive = has_process and user_session.current_process.is_alive()
    has_output_dir = user_session.current_output_dir is not None
    is_in_active_tasks = session_id in gui_instance.concurrency_manager.active_tasks
    
    # 只有当进程真的在运行，且有输出目录，且在活跃任务列表中时，才认为任务在运行
    task_running = process_alive and has_output_dir and is_in_active_tasks
    
    # 🔧 Fix: 如果进程对象存在但任务实际上不在运行，清理进程对象
    if has_process and not task_running:
        print(f"[{datetime.datetime.now().isoformat()}] 🔧 Cleaning up stale process object: session_id={session_id}, process_alive={process_alive}, has_output_dir={has_output_dir}, is_in_active_tasks={is_in_active_tasks}")
        user_session.current_process = None
    
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
        },
        'task_running': task_running,  # 告知客户端是否有任务在运行
        'recovered': recovered_session is not None,  # 告知客户端这是恢复的会话
        # 🔧 恢复文件夹选择状态
        'selected_output_dir': user_session.selected_output_dir,
        'last_output_dir': user_session.last_output_dir,
        'current_output_dir': user_session.current_output_dir
    }
    
    emit('status', connection_data, room=session_id)
    
    # 如果是恢复的会话且有任务在运行，重新启动 queue_reader_thread
    if recovered_session and task_running:
        print(f"[{datetime.datetime.now().isoformat()}] 🔄 Restarting message reading thread: session_id={session_id}")
        # 先停止旧的线程（如果存在）
        stop_queue_reader_thread(user_session)
        # 启动新线程
        new_thread = threading.Thread(target=queue_reader_thread, args=(session_id,), daemon=True)
        user_session.queue_reader_thread = new_thread
        new_thread.start()

# 存储待清理的会话（等待重连）
_pending_cleanup_sessions = {}  # {session_id: {'user_session': ..., 'disconnect_time': ..., 'api_key': ...}}
RECONNECT_GRACE_PERIOD = 120  # 等待重连的时间（秒）

@socketio.on('disconnect')
def handle_disconnect():
    """Handle user disconnection - 延迟清理，等待可能的重连"""
    session_id = request.sid
    import datetime
    disconnect_reason = getattr(request, 'disconnect_reason', 'unknown')
    print(f"[{datetime.datetime.now().isoformat()}] ❌ Server detected connection disconnect: session_id={session_id}, reason={disconnect_reason}")

    # Remove connection from concurrency manager
    gui_instance.concurrency_manager.remove_connection()

    if session_id in gui_instance.user_sessions:
        user_session = gui_instance.user_sessions[session_id]
        api_key = user_session.user_info.get('api_key', '')
        client_session_id = getattr(user_session, 'client_session_id', None)
        
        # 判断等待时间：有任务运行时等待更长时间，空闲时等待较短时间
        has_running_task = user_session.current_process and user_session.current_process.is_alive()
        grace_period = RECONNECT_GRACE_PERIOD if has_running_task else 30  # 空闲时等待30秒
        
        # 日志中显示client_session_id
        if client_session_id:
            print(f"[{datetime.datetime.now().isoformat()}] ⏳ Connection disconnected, waiting {grace_period} seconds for reconnection: socket_sid={session_id}, client_sid={client_session_id}, has_task={has_running_task}")
        else:
            print(f"[{datetime.datetime.now().isoformat()}] ⏳ Connection disconnected, waiting {grace_period} seconds for reconnection: socket_sid={session_id}, has_task={has_running_task}")
        
        # 保存到待清理列表
        _pending_cleanup_sessions[session_id] = {
            'user_session': user_session,
            'disconnect_time': time.time(),
            'api_key': api_key,
            'client_session_id': client_session_id,  # 保存client_session_id用于重连匹配
            'has_running_task': has_running_task
        }
        
        # 从当前会话中移除，但不终止进程
        try:
            leave_room(session_id)
        except Exception:
            pass
        
        # 不删除 user_sessions 中的记录，让重连时可以恢复
        # 启动延迟清理线程
        def delayed_cleanup(sid, wait_time):
            time.sleep(wait_time)
            if sid in _pending_cleanup_sessions:
                print(f"[{datetime.datetime.now().isoformat()}] ⏰ Reconnection timeout, cleaning up session: session_id={sid}")
                _cleanup_disconnected_session(sid)
        
        cleanup_thread = threading.Thread(target=delayed_cleanup, args=(session_id, grace_period), daemon=True)
        cleanup_thread.start()
    else:
        pass

def _cleanup_disconnected_session(session_id):
    """清理断开的会话"""
    import datetime
    
    # 从待清理列表中移除
    pending_info = _pending_cleanup_sessions.pop(session_id, None)
    
    if session_id in gui_instance.user_sessions:
        user_session = gui_instance.user_sessions[session_id]
    elif pending_info:
        user_session = pending_info['user_session']
    else:
        return

    # 获取client_session_id用于日志
    client_session_id = getattr(user_session, 'client_session_id', None)

    # Leave room
    try:
        leave_room(session_id)
    except Exception:
        pass

    # Terminate any running processes
    if user_session.current_process and user_session.current_process.is_alive():
        try:
            if client_session_id:
                print(f"[{datetime.datetime.now().isoformat()}] 🛑 终止运行中的任务: socket_sid={session_id}, client_sid={client_session_id}")
            else:
                print(f"[{datetime.datetime.now().isoformat()}] 🛑 终止运行中的任务: socket_sid={session_id}")
            user_session.current_process.terminate()
            user_session.current_process.join(timeout=5)
        except Exception:
            pass

    # Clean up active task if exists
    try:
        gui_instance.concurrency_manager.finish_task(session_id, success=False)
    except Exception:
        pass

    # Clean up session
    try:
        gui_instance.auth_manager.destroy_session(session_id)
        if session_id in gui_instance.user_sessions:
            del gui_instance.user_sessions[session_id]
    except Exception:
        pass

    # 日志中显示client_session_id
    if client_session_id:
        print(f"[{datetime.datetime.now().isoformat()}] 🧹 Session cleaned up: socket_sid={session_id}, client_sid={client_session_id}")
    else:
        print(f"[{datetime.datetime.now().isoformat()}] 🧹 Session cleaned up: socket_sid={session_id}")

@socketio.on('heartbeat')
def handle_heartbeat(data):
    """Handle heartbeat from client to keep connection alive"""
    session_id = request.sid
    client_timestamp = data.get('timestamp', 0)
    
    # 🔧 增强：记录心跳接收情况，用于调试连接问题
    import datetime
    if session_id in gui_instance.user_sessions:
        # 验证并更新会话，这会更新last_accessed时间
        gui_instance.auth_manager.validate_session(session_id)
        # 可选：记录心跳日志（仅在调试模式下）
        # print(f"[{datetime.datetime.now().isoformat()}] 💓 Heartbeat received: session_id={session_id}")
    else:
        # 如果会话不存在，记录警告
        print(f"[{datetime.datetime.now().isoformat()}] ⚠️ Heartbeat from unknown session: session_id={session_id}")
    
    # 发送心跳响应，确认连接正常
    emit('heartbeat_ack', {'timestamp': client_timestamp, 'server_time': time.time()}, room=session_id)

@socketio.on('execute_task')
def handle_execute_task(data):
    """Handle task execution request"""
    # Get language from gui_config if available, otherwise use default
    gui_config = data.get('gui_config', {})
    user_lang = gui_config.get('language', get_language())
    i18n = I18N_TEXTS.get(user_lang, I18N_TEXTS['en'])
    session_id = request.sid
    
    # 🔧 添加调试日志以跟踪函数调用
    # print(f"[{datetime.datetime.now().isoformat()}] 📥 handle_execute_task called: session_id={session_id}")
    
    # Get user session
    if session_id not in gui_instance.user_sessions:
        emit('error', {'message': 'User session not found'}, room=session_id)
        return
    
    user_session = gui_instance.user_sessions[session_id]
    
    if user_session.current_process and user_session.current_process.is_alive():
        emit('error', {'message': i18n['error_task_running']}, room=session_id)
        return

    user_requirement = data.get('requirement', '')
    # Allow empty requirement to start the program
    
    task_type = data.get('type', 'continue')  # 'new', 'continue', 'selected'
    # Ensure plan_mode is boolean (handle string 'true'/'false' from frontend)
    plan_mode_raw = data.get('plan_mode', False)
    if isinstance(plan_mode_raw, str):
        plan_mode = plan_mode_raw.lower() in ('true', '1', 'yes')
    else:
        plan_mode = bool(plan_mode_raw)
    selected_directory = data.get('selected_directory')  # Directory name from frontend
    gui_config = data.get('gui_config', {})  # GUI configuration options
    attached_files = data.get('attached_files', [])  # Attached file information
    
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
        return
    
    # 🔧 停止旧的queue_reader_thread（如果存在）
    # 这很重要，因为我们即将创建新的队列
    stop_queue_reader_thread(user_session)
    
    user_session.output_queue = multiprocessing.Queue()
    user_session.input_queue = multiprocessing.Queue()  # Queue for user input in GUI mode
    
    # Get user ID (sha256_hash) for MCP knowledge base tools
    user_id = None
    if user_session.api_key:
        import hashlib
        user_id = hashlib.sha256(user_session.api_key.encode()).hexdigest()
    
    try:
        # 🚀 Create and start process with highest priority (minimize delay)
        user_session.current_process = multiprocessing.Process(
            target=execute_agia_task_process_target,
            args=(user_requirement, user_session.output_queue, user_session.input_queue, out_dir, continue_mode, plan_mode, gui_config, session_id, detailed_requirement, user_id, attached_files)
        )
        user_session.current_process.daemon = True
        user_session.current_process.start()
        
        # Get current performance metrics
        metrics = gui_instance.concurrency_manager.get_metrics()
        
        # Start queue reader thread after process is confirmed started
        # Messages will be buffered in queue, so slight delay is fine
        new_thread = threading.Thread(target=queue_reader_thread, args=(session_id,), daemon=True)
        user_session.queue_reader_thread = new_thread
        new_thread.start()
        
    except Exception as e:
        # If process startup fails
        gui_instance.concurrency_manager.finish_task(session_id, success=False)
        emit('error', {'message': f'Task startup failed: {str(e)}'}, room=session_id)
        return
    
    # Set current output directory name (extract from absolute path if needed)
    if out_dir:
        user_session.current_output_dir = os.path.basename(out_dir)
    else:
        user_session.current_output_dir = None
    
    # Store current task for conversation history
    user_session._current_task_requirement = user_requirement

@socketio.on('terminal_connect')
def handle_terminal_connect():
    """Handle terminal connection - send initial working directory"""
    session_id = request.sid
    
    if session_id not in gui_instance.user_sessions:
        emit('terminal_error', {'error': 'User session not found'}, room=session_id)
        return
    
    user_session = gui_instance.user_sessions[session_id]
    
    # 重置terminal_cwd，强制重新计算工作目录，确保使用最新的选择状态
    user_session.terminal_cwd = None
    cwd = user_session.get_terminal_cwd(gui_instance.base_data_dir, force_recalculate=True)
    
    # 发送工作目录信息
    emit('terminal_init', {'working_directory': cwd}, room=session_id)

@socketio.on('terminal_input')
def handle_terminal_input(data):
    """Handle terminal command input from browser terminal"""
    import subprocess
    import platform
    import re
    import os
    session_id = request.sid
    
    if session_id not in gui_instance.user_sessions:
        emit('terminal_error', {'error': 'User session not found'}, room=session_id)
        return
    
    user_session = gui_instance.user_sessions[session_id]
    command = data.get('command', '').strip()
    
    if not command:
        emit('command_complete', {}, room=session_id)
        return
    
    # 检查退出命令
    if command.lower() in ('exit', 'quit'):
        emit('terminal_output', {'output': '\r\n'}, room=session_id)
        emit('command_complete', {}, room=session_id)
        return
    
    try:
        # 获取当前工作目录（维护cd命令的状态）
        cwd = user_session.get_terminal_cwd(gui_instance.base_data_dir)
        
        # 确保cwd是绝对路径
        if cwd:
            cwd = os.path.abspath(cwd)
        
        # 根据操作系统选择shell
        cmd_lower = command.strip().lower()
        
        if platform.system() == 'Windows':
            shell = True
            executable = None  # 使用cmd.exe（Windows默认）
            # Windows上先设置UTF-8编码，然后执行命令
            # 使用chcp 65001设置UTF-8编码
            # 如果命令是cd命令，需要特殊处理以正确切换目录和更新提示符
            if cmd_lower.startswith('cd'):
                # cd命令处理：提取目录路径并更新terminal_cwd
                cd_match = re.match(r'cd\s+(?:/d\s+)?["\']?([^"\']+)["\']?(?:\s+&&\s+prompt\s+\$P\$G)?', command, re.IGNORECASE)
                if cd_match:
                    target_dir = cd_match.group(1)
                    # 解析相对路径或绝对路径
                    if os.path.isabs(target_dir):
                        new_cwd = target_dir
                    else:
                        new_cwd = os.path.join(cwd, target_dir)
                    new_cwd = os.path.normpath(os.path.abspath(new_cwd))
                    
                    # 更新terminal_cwd状态
                    if user_session.set_terminal_cwd(new_cwd):
                        # 切换成功，使用cd /d切换目录和盘符
                        # 移除echo %CD%以避免重复输出路径
                        full_command = f'cd /d "{new_cwd}"'
                        cwd = new_cwd  # 更新当前cwd用于subprocess
                    else:
                        # 目录不存在，显示错误
                        full_command = f'echo Error: Directory not found: {target_dir}'
                else:
                    # 如果无法解析，尝试执行原命令
                    full_command = command
            else:
                # 对于其他命令，使用系统默认编码（Windows通常是GBK/CP936）
                # 如果是python命令，添加-u参数以禁用缓冲，确保输出实时显示
                cmd_lower_check = command.strip().lower()
                if cmd_lower_check.startswith('python') and '-u' not in cmd_lower_check:
                    # 在python命令中添加-u参数
                    python_match = re.match(r'(python\s+)(.*)', command, re.IGNORECASE)
                    if python_match:
                        # Python命令不使用chcp，直接执行，使用系统默认编码
                        full_command = f'{python_match.group(1)}-u {python_match.group(2)}'
                    else:
                        full_command = command
                else:
                    full_command = command
            # Windows使用系统默认编码（通常是GBK/CP936），而不是UTF-8
            import locale
            encoding = locale.getpreferredencoding() or 'gbk'
        else:
            # Linux/Mac处理
            shell = True
            executable = '/bin/bash'
            
            # Linux下也需要处理cd命令
            # 支持: cd dir, cd "dir", cd 'dir', cd ~, cd -, cd .., cd dir/
            # 也支持: cd "dir" && command (组合命令)
            if cmd_lower.startswith('cd'):
                # 检查是否是组合命令 (cd ... && command)
                and_pos = command.find(' && ')
                if and_pos != -1:
                    # 是组合命令，提取cd部分
                    cd_part = command[:and_pos].strip()
                    rest_command = command[and_pos + 4:].strip()
                    
                    # 解析cd命令
                    cd_match = re.match(r'cd\s+(?:"([^"]+)"|\'([^\']+)\'|([^\s]+))', cd_part)
                    if cd_match:
                        target_dir = cd_match.group(1) or cd_match.group(2) or cd_match.group(3)
                        target_dir = target_dir.rstrip('/')
                        
                        # 处理特殊目录
                        if target_dir == '-':
                            new_cwd = os.path.dirname(cwd) if cwd != os.path.sep else cwd
                        elif target_dir.startswith('~'):
                            new_cwd = os.path.expanduser(target_dir)
                        else:
                            if os.path.isabs(target_dir):
                                new_cwd = target_dir
                            else:
                                # 处理相对路径
                                # 检查当前工作目录是否已经是workspace目录
                                # 如果target_dir包含output_xxx/workspace这样的路径，且当前cwd已经是workspace，需要去掉output_xxx/workspace前缀
                                cwd_basename = os.path.basename(cwd)
                                if cwd_basename == 'workspace':
                                    # 当前目录已经是workspace，检查target_dir是否包含output_xxx/workspace模式
                                    # 例如：target_dir = "output_20260104_102756/workspace" 或 "output_20260104_102756/workspace/subdir"
                                    parts = target_dir.split('/')
                                    workspace_idx = -1
                                    for i, part in enumerate(parts):
                                        if part == 'workspace':
                                            workspace_idx = i
                                            break
                                    
                                    if workspace_idx != -1:
                                        # 找到workspace，使用workspace之后的部分
                                        if workspace_idx + 1 < len(parts):
                                            # workspace后面还有路径
                                            target_dir = '/'.join(parts[workspace_idx + 1:])
                                        else:
                                            # workspace后面没有路径，说明就是workspace本身
                                            target_dir = '.'
                                
                                # 如果target_dir以用户目录名开头，去掉它（因为cwd已经是用户目录了）
                                user_dir_name = user_session.user_dir_name
                                if target_dir.startswith(user_dir_name + '/'):
                                    # 去掉用户目录名前缀
                                    target_dir = target_dir[len(user_dir_name) + 1:]
                                elif target_dir.startswith(user_dir_name + '\\'):
                                    # Windows路径分隔符
                                    target_dir = target_dir[len(user_dir_name) + 1:]
                                
                                new_cwd = os.path.join(cwd, target_dir)
                        new_cwd = os.path.abspath(os.path.normpath(new_cwd))
                        
                        # 更新terminal_cwd状态
                        if user_session.set_terminal_cwd(new_cwd):
                            # 切换成功，执行组合命令，使用新的cwd作为subprocess的工作目录
                            full_command = command  # 保持原命令不变
                            cwd = new_cwd  # 更新当前cwd用于subprocess
                        else:
                            # 目录不存在，显示错误
                            full_command = f'echo "Error: Directory not found: {target_dir}"'
                    else:
                        # 无法解析cd部分，执行原命令
                        full_command = command
                else:
                    # 单独的cd命令
                    cd_match = re.match(r'cd\s+(?:"([^"]+)"|\'([^\']+)\'|([^\s]+))', command)
                    if cd_match:
                        # 获取匹配的目录路径（三个组中只有一个会有值）
                        target_dir = cd_match.group(1) or cd_match.group(2) or cd_match.group(3)
                        target_dir = target_dir.rstrip('/')  # 移除末尾的斜杠
                        
                        # 处理特殊目录
                        if target_dir == '-':
                            # cd - 回到上一个目录（这里简化处理，使用父目录）
                            new_cwd = os.path.dirname(cwd) if cwd != os.path.sep else cwd
                        elif target_dir.startswith('~'):
                            # 处理 ~ 和 ~user
                            new_cwd = os.path.expanduser(target_dir)
                        else:
                            # 解析相对路径或绝对路径
                            if os.path.isabs(target_dir):
                                new_cwd = target_dir
                            else:
                                # 处理相对路径
                                # 如果target_dir以用户目录名开头，去掉它（因为cwd已经是用户目录了）
                                user_dir_name = user_session.user_dir_name
                                if target_dir.startswith(user_dir_name + '/'):
                                    # 去掉用户目录名前缀
                                    target_dir = target_dir[len(user_dir_name) + 1:]
                                elif target_dir.startswith(user_dir_name + '\\'):
                                    # Windows路径分隔符
                                    target_dir = target_dir[len(user_dir_name) + 1:]
                                
                                new_cwd = os.path.join(cwd, target_dir)
                        new_cwd = os.path.abspath(os.path.normpath(new_cwd))
                        
                        # 更新terminal_cwd状态
                        if user_session.set_terminal_cwd(new_cwd):
                            # 切换成功，执行cd命令（不输出pwd，避免重复）
                            # 注意：在Linux下，cd命令在子shell中执行，不会影响父进程的工作目录
                            # 但是我们已经更新了terminal_cwd状态，后续命令会使用新的cwd
                            full_command = f'cd "{new_cwd}"'
                            cwd = new_cwd  # 更新当前cwd用于subprocess
                        else:
                            # 目录不存在，显示错误
                            full_command = f'echo "Error: Directory not found: {target_dir}"'
                    else:
                        # 如果无法解析，尝试执行原命令
                        full_command = command
            else:
                # 非cd命令，直接执行
                # 确保cwd是workspace目录（如果terminal_cwd已设置）
                full_command = command
            encoding = 'utf-8'
        
        # 准备环境变量（确保pip等命令使用无缓冲输出）
        import os
        env = os.environ.copy()
        # 为pip命令设置环境变量以确保实时输出
        cmd_lower_for_env = command.strip().lower()
        if 'pip' in cmd_lower_for_env:
            env['PYTHONUNBUFFERED'] = '1'
            env['PIP_PROGRESS_BAR'] = 'on'
            # 确保pip输出不被缓冲
            if 'install' in cmd_lower_for_env:
                env['PIP_DISABLE_PIP_VERSION_CHECK'] = '1'
        
        # 执行命令
        # 对于Windows，使用二进制模式读取以更好地处理格式
        if platform.system() == 'Windows':
            process = subprocess.Popen(
                full_command,
                shell=shell,
                executable=executable,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                stdin=subprocess.PIPE,
                text=False,  # 使用二进制模式
                bufsize=0,  # 无缓冲
                cwd=cwd,
                env=env  # 传递环境变量
            )
        else:
            # Linux/Mac: 使用二进制模式读取，以正确处理\r字符（用于ls等多列格式化）
            process = subprocess.Popen(
                full_command,
                shell=shell,
                executable=executable,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                stdin=subprocess.PIPE,
                text=False,  # 使用二进制模式，以便正确处理\r
                bufsize=0,  # 无缓冲
                cwd=cwd,
                env=env  # 传递环境变量
            )
        
        # 读取输出并实时发送
        def read_output():
            # 使用应用上下文，因为这是在后台线程中运行
            with app.app_context():
                try:
                    import io
                    import time
                    import select
                    # 对于Windows，使用二进制模式读取，然后手动解码，以更好地处理格式
                    if platform.system() == 'Windows':
                        # 使用二进制模式读取，更频繁地读取以支持进度条
                        buffer = b''
                        last_flush_time = time.time()
                        flush_interval = 0.1  # 每100ms刷新一次缓冲区
                        
                        while True:
                            # 尝试读取可用数据（非阻塞方式）
                            chunk = None
                            try:
                                # 使用read1()如果可用，它会读取至少1字节但不等待完整缓冲区
                                if hasattr(process.stdout, 'read1'):
                                    chunk = process.stdout.read1(8192)
                                else:
                                    # 回退到read(1)以获取更及时的响应
                                    chunk = process.stdout.read(1)
                            except:
                                pass
                            
                            if chunk:
                                buffer += chunk
                                last_flush_time = time.time()
                            
                            # 处理缓冲区中的完整行和进度条
                            processed = False
                            while buffer:
                                # 找到第一个换行符或回车符
                                nl_pos = buffer.find(b'\n')
                                cr_pos = buffer.find(b'\r')
                                
                                if nl_pos != -1 and (cr_pos == -1 or nl_pos <= cr_pos):
                                    # 发送到换行符（包含换行符）
                                    to_send = buffer[:nl_pos + 1]
                                    buffer = buffer[nl_pos + 1:]
                                    try:
                                        decoded = to_send.decode(encoding, errors='replace')
                                        socketio.emit('terminal_output', {'output': decoded}, room=session_id)
                                    except:
                                        pass
                                    processed = True
                                elif cr_pos != -1:
                                    # 检查是否是\r\n组合
                                    if cr_pos + 1 < len(buffer) and buffer[cr_pos + 1] == ord(b'\n'):
                                        # \r\n组合，发送到\n
                                        to_send = buffer[:cr_pos + 2]
                                        buffer = buffer[cr_pos + 2:]
                                        try:
                                            decoded = to_send.decode(encoding, errors='replace')
                                            socketio.emit('terminal_output', {'output': decoded}, room=session_id)
                                        except:
                                            pass
                                        processed = True
                                    else:
                                        # 单独的\r，需要特殊处理
                                        # 对于ls等命令，\r用于格式化多列输出，需要正确处理
                                        # 找到\r后面的内容直到下一个\r或\n
                                        next_cr = buffer.find(b'\r', cr_pos + 1)
                                        next_nl = buffer.find(b'\n', cr_pos + 1)
                                        
                                        # 确定发送的结束位置
                                        if next_nl != -1 and (next_cr == -1 or next_nl <= next_cr):
                                            # 有换行符，发送到换行符（包含\r和\n）
                                            to_send = buffer[:next_nl + 1]
                                            buffer = buffer[next_nl + 1:]
                                        elif next_cr != -1:
                                            # 有下一个\r，发送从当前\r到下一个\r之前的内容（包含当前\r）
                                            to_send = buffer[:next_cr]
                                            buffer = buffer[next_cr:]
                                        else:
                                            # 没有找到下一个\r或\n
                                            # 检查是否应该等待更多数据
                                            # 如果缓冲区中\r后面的内容足够长（超过200字节），可能是完整的格式化行
                                            # 否则等待更多数据或刷新间隔
                                            content_after_cr = len(buffer) - cr_pos - 1
                                            current_time = time.time()
                                            if content_after_cr > 200 or (current_time - last_flush_time >= flush_interval):
                                                # 发送当前\r和后面的所有内容
                                                to_send = buffer
                                                buffer = b''
                                                last_flush_time = current_time
                                            else:
                                                # 缓冲区不够长且未到刷新时间，等待更多数据
                                                break
                                        
                                        if to_send:
                                            try:
                                                decoded = to_send.decode(encoding, errors='replace')
                                                # 保持原始格式，让xterm.js正确处理\r（用于ls等多列格式化）
                                                socketio.emit('terminal_output', {'output': decoded}, room=session_id)
                                            except:
                                                pass
                                            processed = True
                                else:
                                    # 没有找到换行符或回车符
                                    # 如果缓冲区有内容且超过刷新间隔，发送部分内容（用于实时输出）
                                    current_time = time.time()
                                    if buffer and (current_time - last_flush_time >= flush_interval):
                                        # 发送缓冲区内容（不等待换行）
                                        to_send = buffer
                                        buffer = b''
                                        try:
                                            decoded = to_send.decode(encoding, errors='replace')
                                            socketio.emit('terminal_output', {'output': decoded}, room=session_id)
                                        except:
                                            pass
                                        last_flush_time = current_time
                                    break
                            
                            # 检查进程是否已结束
                            if process.poll() is not None:
                                # 进程已结束，发送剩余缓冲区内容
                                if buffer:
                                    try:
                                        decoded = buffer.decode(encoding, errors='replace')
                                        socketio.emit('terminal_output', {'output': decoded}, room=session_id)
                                    except:
                                        pass
                                break
                            
                            # 如果没有数据且未处理任何内容，短暂休眠避免CPU占用过高
                            if not chunk and not processed:
                                time.sleep(0.01)
                    else:
                        # Linux/Mac使用二进制模式读取，以正确处理\r字符（用于ls等多列格式化）
                        import select
                        buffer = b''
                        last_flush_time = time.time()
                        flush_interval = 0.1
                        
                        while True:
                            # 检查进程是否已结束
                            process_ended = (process.poll() is not None)
                            
                            # 尝试读取可用数据（非阻塞方式）
                            chunk = None
                            try:
                                # 在二进制模式下，直接读取bytes
                                if hasattr(process.stdout, 'read1'):
                                    chunk = process.stdout.read1(8192)
                                else:
                                    # 回退到read()以获取更及时的响应
                                    if hasattr(select, 'select'):
                                        try:
                                            ready, _, _ = select.select([process.stdout], [], [], 0.1)
                                            if ready:
                                                chunk = process.stdout.read(8192)
                                        except:
                                            pass
                                    if not chunk:
                                        # 尝试直接读取
                                        try:
                                            chunk = process.stdout.read(8192)
                                        except:
                                            pass
                            except:
                                pass
                            
                            if chunk:
                                # 确保chunk是bytes类型
                                if isinstance(chunk, str):
                                    chunk = chunk.encode(encoding)
                                buffer += chunk
                                last_flush_time = time.time()
                            
                            # 处理缓冲区中的完整行和进度条
                            processed = False
                            while buffer:
                                # 找到第一个换行符或回车符
                                nl_pos = buffer.find(b'\n')
                                cr_pos = buffer.find(b'\r')
                                
                                if nl_pos != -1 and (cr_pos == -1 or nl_pos <= cr_pos):
                                    # 发送到换行符（包含换行符）
                                    to_send = buffer[:nl_pos + 1]
                                    buffer = buffer[nl_pos + 1:]
                                    try:
                                        decoded = to_send.decode(encoding, errors='replace')
                                        socketio.emit('terminal_output', {'output': decoded}, room=session_id)
                                    except:
                                        pass
                                    processed = True
                                elif cr_pos != -1:
                                    # 检查是否是\r\n组合
                                    if cr_pos + 1 < len(buffer) and buffer[cr_pos + 1] == ord(b'\n'):
                                        # \r\n组合，发送到\n
                                        to_send = buffer[:cr_pos + 2]
                                        buffer = buffer[cr_pos + 2:]
                                        try:
                                            decoded = to_send.decode(encoding, errors='replace')
                                            socketio.emit('terminal_output', {'output': decoded}, room=session_id)
                                        except:
                                            pass
                                        processed = True
                                    else:
                                        # 单独的\r，需要特殊处理
                                        # 对于ls等命令，\r用于格式化多列输出，需要正确处理
                                        # 找到\r后面的内容直到下一个\r或\n
                                        next_cr = buffer.find(b'\r', cr_pos + 1)
                                        next_nl = buffer.find(b'\n', cr_pos + 1)
                                        
                                        # 确定发送的结束位置
                                        if next_nl != -1 and (next_cr == -1 or next_nl <= next_cr):
                                            # 有换行符，发送从开头到换行符（包含\r和\n）
                                            to_send = buffer[:next_nl + 1]
                                            buffer = buffer[next_nl + 1:]
                                        elif next_cr != -1:
                                            # 有下一个\r，发送从当前\r到下一个\r之前的内容（包含当前\r）
                                            to_send = buffer[:next_cr]
                                            buffer = buffer[next_cr:]
                                        else:
                                            # 没有找到下一个\r或\n
                                            # 对于ls等多列输出，\r用于回到行首，需要立即发送
                                            # 检查\r后面是否有内容
                                            content_after_cr = len(buffer) - cr_pos - 1
                                            current_time = time.time()
                                            
                                            # 如果\r后面有内容，发送从开头到\r及后面的内容（最多到缓冲区末尾或刷新间隔）
                                            # 这样可以确保\r字符能够立即被xterm.js处理
                                            if content_after_cr > 0:
                                                # 有内容，发送从开头到当前缓冲区末尾（包含\r和后面的内容）
                                                # 降低阈值，确保\r能够及时发送
                                                if content_after_cr > 100 or (current_time - last_flush_time >= flush_interval):
                                                    to_send = buffer
                                                    buffer = b''
                                                    last_flush_time = current_time
                                                else:
                                                    # 内容较少，等待更多数据或刷新间隔
                                                    break
                                            else:
                                                # \r后面没有内容，立即发送\r字符
                                                to_send = buffer[:cr_pos + 1]
                                                buffer = buffer[cr_pos + 1:]
                                        
                                        if to_send:
                                            try:
                                                decoded = to_send.decode(encoding, errors='replace')
                                                # 保持原始格式，让xterm.js正确处理\r（用于ls等多列格式化）
                                                socketio.emit('terminal_output', {'output': decoded}, room=session_id)
                                            except:
                                                pass
                                            processed = True
                                else:
                                    # 没有找到换行符或回车符
                                    # 如果缓冲区有内容且超过刷新间隔，发送部分内容（用于实时输出）
                                    current_time = time.time()
                                    if buffer and (current_time - last_flush_time >= flush_interval):
                                        # 发送缓冲区内容（不等待换行）
                                        to_send = buffer
                                        buffer = b''
                                        try:
                                            decoded = to_send.decode(encoding, errors='replace')
                                            socketio.emit('terminal_output', {'output': decoded}, room=session_id)
                                        except:
                                            pass
                                        last_flush_time = current_time
                                    break
                            
                            # 检查进程是否已结束
                            if process_ended:
                                # 进程已结束，发送剩余缓冲区内容
                                if buffer:
                                    try:
                                        decoded = buffer.decode(encoding, errors='replace')
                                        socketio.emit('terminal_output', {'output': decoded}, room=session_id)
                                    except:
                                        pass
                                break
                            
                            # 如果没有数据且未处理任何内容，短暂休眠避免CPU占用过高
                            if not chunk and not processed:
                                time.sleep(0.01)
                        
                        # 确保发送所有剩余的缓冲区内容
                        if buffer:
                            try:
                                decoded = buffer.decode(encoding, errors='replace')
                                socketio.emit('terminal_output', {'output': decoded}, room=session_id)
                            except:
                                pass
                    
                    process.stdout.close()
                    return_code = process.wait()
                    
                    # 如果是cd命令（包括组合命令中的cd）且成功执行，发送更新后的提示符
                    if ('cd' in cmd_lower and (' && ' in command or cmd_lower.startswith('cd'))) and return_code == 0:
                        # 获取当前目录并发送更新后的提示符
                        current_dir = user_session.get_terminal_cwd(gui_instance.base_data_dir)
                        # 发送提示符更新事件
                        socketio.emit('terminal_prompt_update', {'directory': current_dir}, room=session_id)
                    
                    socketio.emit('command_complete', {}, room=session_id)
                except Exception as e:
                    socketio.emit('terminal_error', {'error': str(e)}, room=session_id)
        
        # 在后台线程中读取输出
        output_thread = threading.Thread(target=read_output, daemon=True)
        output_thread.start()
        
    except Exception as e:
        emit('terminal_error', {'error': f'Command execution failed: {str(e)}'}, room=session_id)
        emit('command_complete', {}, room=session_id)

@socketio.on('terminal_autocomplete')
def handle_terminal_autocomplete(data):
    """Handle terminal autocomplete request"""
    import os
    import glob
    session_id = request.sid
    
    if session_id not in gui_instance.user_sessions:
        return
    
    user_session = gui_instance.user_sessions[session_id]
    line = data.get('line', '')
    cursor = data.get('cursor', len(line))
    working_dir = data.get('working_dir', '')
    
    # 获取当前工作目录
    cwd = user_session.get_terminal_cwd(gui_instance.base_data_dir)
    if working_dir:
        cwd = working_dir
    
    # 提取要补全的部分（从行开始到光标位置）
    text_before_cursor = line[:cursor] if cursor <= len(line) else line
    parts = text_before_cursor.split()
    
    if not parts:
        # 没有输入，返回空
        emit('terminal_autocomplete_result', {'completions': []}, room=session_id)
        return
    
    last_part = parts[-1]
    
    # 如果是路径补全（包含路径分隔符）
    if '/' in last_part or '\\' in last_part:
        # 路径补全
        dir_part = os.path.dirname(last_part) or '.'
        file_part = os.path.basename(last_part)
        
        if not os.path.isabs(dir_part):
            dir_part = os.path.join(cwd, dir_part)
        
        dir_part = os.path.normpath(dir_part)
        
        if os.path.isdir(dir_part):
            try:
                pattern = os.path.join(dir_part, file_part + '*')
                matches = glob.glob(pattern)
                completions = []
                for m in matches:
                    name = os.path.basename(m)
                    if os.path.isdir(m):
                        completions.append(name + os.sep)
                    else:
                        completions.append(name)
                completions.sort()
            except Exception:
                completions = []
        else:
            completions = []
    else:
        # 命令/文件名补全 - 查找当前目录下的文件和目录
        try:
            pattern = os.path.join(cwd, last_part + '*')
            matches = glob.glob(pattern)
            completions = []
            for m in matches:
                name = os.path.basename(m)
                if os.path.isdir(m):
                    completions.append(name + os.sep)
                else:
                    completions.append(name)
            completions.sort()
        except Exception:
            completions = []
    
    # 限制补全结果数量
    completions = completions[:20]
    
    emit('terminal_autocomplete_result', {'completions': completions}, room=session_id)

@socketio.on('user_input_response')
def handle_user_input_response(data):
    """Handle user input response from GUI"""
    session_id = request.sid
    
    if session_id not in gui_instance.user_sessions:
        return
    
    user_session = gui_instance.user_sessions[session_id]
    user_input = data.get('input', '')
    
    # Put user input into the input queue
    if user_session.input_queue:
        try:
            user_session.input_queue.put(user_input)
        except Exception as e:
            emit('error', {'message': f'Failed to send user input: {str(e)}'}, room=session_id)

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
        
        # 获取logs目录下的所有.out文件列表
        out_files = []
        try:
            user_base_dir = user_session.get_user_directory(gui_instance.base_data_dir)
            logs_dir = os.path.join(user_base_dir, dir_name, 'logs')
            if os.path.exists(logs_dir):
                # 查找所有.out文件
                for filename in os.listdir(logs_dir):
                    if filename.endswith('.out'):
                        # 移除.out后缀，只保留文件名
                        agent_name = filename[:-4]  # 移除'.out'
                        out_files.append(agent_name)
                # 排序，确保manager在最后（如果存在）
                out_files.sort(key=lambda x: (x != 'manager', x))
        except Exception as e:
            logger.warning(f"Failed to list .out files for directory {dir_name}: {str(e)}")
        
        # 不再自动读取manager.out文件内容，改为由用户手动点击加载按钮触发
        emit('directory_selected', {
            'dir_name': dir_name,
            'out_files': out_files
        }, room=session_id)
    else:
        user_session.selected_output_dir = None
        emit('directory_selected', {'dir_name': None, 'out_files': []}, room=session_id)

@socketio.on('load_history')
def handle_load_history(data):
    """Handle load history request"""
    session_id = request.sid
    if session_id not in gui_instance.user_sessions:
        return
    
    user_session = gui_instance.user_sessions[session_id]
    dir_name = data.get('dir_name', '') or user_session.selected_output_dir
    agent_name = data.get('agent_name', 'manager')  # 默认为manager
    
    if not dir_name:
        emit('history_loaded', {
            'success': False,
            'error': 'No directory selected'
        }, room=session_id)
        return
    
    # 尝试读取指定的.out文件内容
    out_content = None
    try:
        user_base_dir = user_session.get_user_directory(gui_instance.base_data_dir)
        out_file_path = os.path.join(user_base_dir, dir_name, 'logs', f'{agent_name}.out')
        
        if os.path.exists(out_file_path):
            with open(out_file_path, 'r', encoding='utf-8') as f:
                out_content = f.read()
            emit('history_loaded', {
                'success': True,
                'manager_out_content': out_content  # 保持字段名不变以兼容前端
            }, room=session_id)
        else:
            emit('history_loaded', {
                'success': False,
                'error': f'{agent_name}.out file not found'
            }, room=session_id)
    except Exception as e:
        logger.warning(f"Failed to read {agent_name}.out for directory {dir_name}: {str(e)}")
        emit('history_loaded', {
            'success': False,
            'error': str(e)
        }, room=session_id)

@socketio.on('append_task')
def handle_append_task(data):
    """Handle append task request - add user request to manager inbox (multi-agent mode only)"""
    session_id = request.sid
    if session_id not in gui_instance.user_sessions:
        emit('error', {'message': 'Session not found'}, room=session_id)
        return
    
    user_session = gui_instance.user_sessions[session_id]
    content = data.get('content', '').strip()
    
    if not content:
        emit('error', {'message': 'Task content cannot be empty'}, room=session_id)
        return
    
    try:
        # Get current output directory
        user_base_dir = user_session.get_user_directory(gui_instance.base_data_dir)
        output_dir = None
        
        if user_session.current_output_dir:
            output_dir = os.path.join(user_base_dir, user_session.current_output_dir)
        elif user_session.selected_output_dir:
            output_dir = os.path.join(user_base_dir, user_session.selected_output_dir)
        elif user_session.last_output_dir:
            output_dir = os.path.join(user_base_dir, user_session.last_output_dir)
        
        if not output_dir or not os.path.exists(output_dir):
            emit('error', {'message': 'No valid output directory found. Please start a task first.'}, room=session_id)
            return
        
        # Import functions from add_user_request.py
        import re
        from datetime import datetime
        
        # Find next extmsg ID
        inbox_dir = os.path.join(output_dir, "mailboxes", "manager", "inbox")
        os.makedirs(inbox_dir, exist_ok=True)
        
        max_id = 0
        pattern = re.compile(r'extmsg_(\d+)\.json')
        
        if os.path.exists(inbox_dir):
            for filename in os.listdir(inbox_dir):
                match = pattern.match(filename)
                if match:
                    msg_id = int(match.group(1))
                    max_id = max(max_id, msg_id)
        
        next_id = max_id + 1
        message_id = f"extmsg_{next_id:06d}"
        
        # Create message object
        message = {
            "message_id": message_id,
            "sender_id": "user",
            "receiver_id": "manager",
            "message_type": "collaboration",
            "content": {
                "text": content
            },
            "priority": 2,
            "requires_response": False,
            "timestamp": datetime.datetime.now().isoformat(),
            "delivered": False,
            "read": False
        }
        
        # Write message file
        file_path = os.path.join(inbox_dir, f"{message_id}.json")
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(message, f, indent=2, ensure_ascii=False)
        
        emit('append_task_success', {
            'message': f'Task appended successfully',
            'message_id': message_id,
            'file_path': file_path
        }, room=session_id)
        
    except Exception as e:
        emit('error', {'message': f'Failed to append task: {str(e)}'}, room=session_id)

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
def handle_stop_task(data=None):
    """Handle stop task request with force option"""
    i18n = get_i18n_texts()
    session_id = request.sid
    
    if session_id not in gui_instance.user_sessions:
        return
    
    user_session = gui_instance.user_sessions[session_id]
    
    # Check if force stop is requested
    force_stop = False
    if data and isinstance(data, dict):
        force_stop = data.get('force', False)
    
    if user_session.current_process and user_session.current_process.is_alive():
        # 🔧 Fix: save current conversation to history when stopping task
        if hasattr(user_session, '_current_task_requirement'):
            user_session.add_to_conversation_history(
                user_session._current_task_requirement,
                "Task stopped by user"
            )
            delattr(user_session, '_current_task_requirement')

        try:
            if force_stop:
                # Force kill the process immediately
                user_session.current_process.kill()
                emit('output', {'message': '🛑 强制停止任务中...', 'type': 'warning'}, room=session_id)
            else:
                # Try graceful termination first
                user_session.current_process.terminate()
                emit('output', {'message': '⏹️ 正在停止任务...', 'type': 'info'}, room=session_id)
                
                # Wait a short time for graceful termination
                import time
                time.sleep(0.5)
                
                # If still alive after 0.5 seconds, force kill
                if user_session.current_process and user_session.current_process.is_alive():
                    user_session.current_process.kill()
                    emit('output', {'message': '🛑 任务未响应，已强制停止', 'type': 'warning'}, room=session_id)
        except Exception as e:
            # If terminate/kill fails, try to find and kill child processes
            try:
                import psutil
                import os
                if user_session.current_process and hasattr(user_session.current_process, 'pid'):
                    parent = psutil.Process(user_session.current_process.pid)
                    for child in parent.children(recursive=True):
                        try:
                            child.kill()
                        except:
                            pass
                    try:
                        parent.kill()
                    except:
                        pass
            except:
                pass
            
            emit('output', {'message': f'⚠️ 停止任务时出错: {str(e)}', 'type': 'error'}, room=session_id)
        
        user_session.current_output_dir = None  # Clear current directory mark

        # 🔧 Fix: Clean up active task to prevent timeout detection
        if hasattr(gui_instance, 'finish_task'):
            gui_instance.finish_task(session_id, success=False)

        emit('task_stopped', {'message': i18n['task_stopped'], 'type': 'error'}, room=session_id)
    else:
        # 当没有运行中的任务时，直接返回，不显示消息
        pass

@socketio.on('create_new_directory')
def handle_create_new_directory(data=None):
    """Handle create new directory request"""
    session_id = request.sid
    
    try:
        # Check if session exists
        if session_id not in gui_instance.user_sessions:
            # Get language from data if available, otherwise use default
            user_lang = data.get('language', get_language()) if data else get_language()
            i18n = I18N_TEXTS.get(user_lang, I18N_TEXTS['en'])
            emit('directory_created', {
                'success': False,
                'error': i18n.get('session_not_found', 'Session not found. Please reconnect.')
            }, room=session_id)
            return
        
        user_session = gui_instance.user_sessions[session_id]
        user_base_dir = user_session.get_user_directory(gui_instance.base_data_dir)
        
        # Get language from data if available, otherwise use default
        user_lang = data.get('language', get_language()) if data else get_language()
        i18n = I18N_TEXTS.get(user_lang, I18N_TEXTS['en'])
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        new_dir_name = f"output_{timestamp}"
        new_dir_path = os.path.join(user_base_dir, new_dir_name)
        
        # Create main directory
        os.makedirs(new_dir_path, exist_ok=True)
        
        # Create workspace subdirectory
        workspace_dir = os.path.join(new_dir_path, 'workspace')
        os.makedirs(workspace_dir, exist_ok=True)
        
        # Set as currently selected directory
        user_session.selected_output_dir = new_dir_name
        
        # Clear conversation history when creating new workspace
        user_session.conversation_history.clear()
        
        emit('directory_created', {
            'dir_name': new_dir_name,
            'success': True,
            'message': i18n['directory_created_with_workspace'].format(new_dir_name)
        }, room=session_id)
        
    except Exception as e:
        # Get language from data if available, otherwise use default
        user_lang = data.get('language', get_language()) if data else get_language()
        i18n = I18N_TEXTS.get(user_lang, I18N_TEXTS['en'])
        emit('directory_created', {
            'success': False,
            'error': str(e)
        }, room=session_id)

@socketio.on('clear_chat')
def handle_clear_chat(data=None):
    """Handle clear chat request"""
    session_id = request.sid
    if session_id not in gui_instance.user_sessions:
        return
    
    try:
        # Get language from data if available, otherwise use default
        user_lang = get_language()
        if data and isinstance(data, dict):
            user_lang = data.get('language', user_lang)
        i18n = I18N_TEXTS.get(user_lang, I18N_TEXTS['en'])
        
        # Clear server-side conversation history
        user_session = gui_instance.user_sessions[session_id]
        user_session.conversation_history.clear()
        
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
        
        # Security check: normalize path and prevent path traversal
        # Don't use secure_filename as it destroys Chinese characters
        normalized_dir_name = os.path.normpath(dir_name)
        if '..' in normalized_dir_name or normalized_dir_name.startswith('/'):
            return jsonify({'success': False, 'error': 'Access denied: Invalid directory path'}), 403
        
        # Target directory path
        target_dir = os.path.join(user_base_dir, normalized_dir_name)
        
        # Security check: ensure directory is within user's output directory
        real_output_dir = os.path.realpath(user_base_dir)
        real_target_dir = os.path.realpath(target_dir)
        if not real_target_dir.startswith(real_output_dir):
            return jsonify({'success': False, 'error': 'Access denied: Invalid directory path'}), 403
        
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
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/out-files/<path:dir_name>', methods=['GET'])
def get_out_files(dir_name):
    """Get list of .out files in specified directory's logs folder"""
    try:
        # Get API key from query parameters or headers
        api_key = request.args.get('api_key') or request.headers.get('X-API-Key')
        
        # Create a temporary session for API calls
        temp_session_id = create_temp_session_id(request, api_key)
        user_session = gui_instance.get_user_session(temp_session_id, api_key)
        user_base_dir = user_session.get_user_directory(gui_instance.base_data_dir)
        
        # Security check: normalize path and prevent path traversal
        normalized_dir_name = os.path.normpath(dir_name)
        if '..' in normalized_dir_name or normalized_dir_name.startswith('/'):
            return jsonify({'success': False, 'error': 'Access denied: Invalid directory path'}), 403
        
        # Target directory path
        target_dir = os.path.join(user_base_dir, normalized_dir_name)
        
        # Security check: ensure directory is within user's output directory
        real_output_dir = os.path.realpath(user_base_dir)
        real_target_dir = os.path.realpath(target_dir)
        if not real_target_dir.startswith(real_output_dir):
            return jsonify({'success': False, 'error': 'Access denied: Invalid directory path'}), 403
        
        if not os.path.exists(target_dir):
            return jsonify({
                'success': False,
                'error': 'Directory not found'
            }), 404
        
        # Get logs directory
        logs_dir = os.path.join(target_dir, 'logs')
        out_files = []
        
        if os.path.exists(logs_dir):
            # Find all .out files
            for filename in os.listdir(logs_dir):
                if filename.endswith('.out'):
                    # Remove .out suffix, keep only filename
                    agent_name = filename[:-4]  # Remove '.out'
                    out_files.append(agent_name)
            # Sort, ensuring manager is last (if exists)
            out_files.sort(key=lambda x: (x != 'manager', x))
        
        return jsonify({
            'success': True,
            'out_files': out_files
        })
    except Exception as e:
        logger.warning(f"Failed to list .out files for directory {dir_name}: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# File upload functionality
@app.route('/agent-status-visualizer')
def agent_status_visualizer():
    """Serve agent status visualizer page"""
    if not AGENT_VISUALIZER_AVAILABLE:
        return "Agent status visualizer is not available", 404
    
    # Get API key from query parameters or headers
    api_key = request.args.get('api_key') or request.headers.get('X-API-Key')
    temp_session_id = create_temp_session_id(request, api_key)
    user_session = gui_instance.get_user_session(temp_session_id, api_key)
    if not user_session:
        return "Authentication failed. Please provide a valid API key.", 401
    user_base_dir = user_session.get_user_directory(gui_instance.base_data_dir)
    
    # Get directory from query parameter (selected directory)
    dir_name = request.args.get('dir')
    
    # Try to find the output directory
    output_dir = None
    if dir_name:
        # Use the selected directory from query parameter
        # Ensure dir_name doesn't already contain user directory path
        # If it does, extract just the directory name
        if os.path.sep in dir_name or '/' in dir_name:
            # dir_name might contain user directory, extract just the basename
            dir_name = os.path.basename(dir_name)
        output_dir = os.path.join(user_base_dir, dir_name)
        if not os.path.exists(output_dir):
            return f"Directory not found: {dir_name} (searched in: {user_base_dir})", 404
    elif user_session.current_output_dir:
        output_dir = os.path.join(user_base_dir, user_session.current_output_dir)
    elif user_session.last_output_dir:
        output_dir = os.path.join(user_base_dir, user_session.last_output_dir)
    else:
        # Try to find latest output directory
        latest_dir = find_latest_output_dir(user_base_dir)
        if latest_dir:
            output_dir = latest_dir
    
    # Read agent_status_visualizer.html from templates directory
    html_path = os.path.join(template_dir, 'agent_status_visualizer.html')
    
    if not os.path.exists(html_path):
        return f"Agent status visualizer HTML not found at {html_path}", 404
    
    try:
        with open(html_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        # Replace API endpoints to use new routes
        # Use regex to replace more accurately
        html_content = re.sub(r"'/api/status'", "'/api/agent-status'", html_content)
        html_content = re.sub(r'"/api/status"', '"/api/agent-status"', html_content)
        html_content = re.sub(r"'/api/reload'", "'/api/agent-status-reload'", html_content)
        html_content = re.sub(r'"/api/reload"', '"/api/agent-status-reload"', html_content)
        html_content = re.sub(r"'/api/files/", "'/api/agent-status-files/", html_content)
        html_content = re.sub(r'"/api/files/', '"/api/agent-status-files/', html_content)
        # Also replace in apiUrl() function calls
        html_content = re.sub(r"apiUrl\(['\"]api/status['\"]\)", "apiUrl('api/agent-status')", html_content)
        html_content = re.sub(r'apiUrl\(["\']api/reload["\']\)', "apiUrl('api/agent-status-reload')", html_content)
        html_content = re.sub(r"apiUrl\(['\"]api/files/", "apiUrl('api/agent-status-files/", html_content)
        
        # Inject JavaScript to get dir and api_key parameters from URL and pass them to API calls
        dir_param = dir_name if dir_name else ''
        api_key_param = api_key if api_key else ''
        inject_script = f"""
        <script>
            // Get directory and API key parameters from URL
            const urlParams = new URLSearchParams(window.location.search);
            const dirParam = urlParams.get('dir') || '{dir_param}';
            const apiKeyParam = urlParams.get('api_key') || '{api_key_param}';
            
            // Override fetch to automatically add dir and api_key parameters to API calls
            const originalFetch = window.fetch;
            window.fetch = function(url, options) {{
                if (typeof url === 'string') {{
                    // Handle agent-status related API calls
                    if (url.includes('/api/agent-status') || url.includes('/api/reload') || url.includes('/api/files/')) {{
                        const urlObj = new URL(url, window.location.origin);
                        if (dirParam && !urlObj.searchParams.has('dir')) {{
                            urlObj.searchParams.set('dir', dirParam);
                        }}
                        if (apiKeyParam && !urlObj.searchParams.has('api_key')) {{
                            urlObj.searchParams.set('api_key', apiKeyParam);
                        }}
                        url = urlObj.toString();
                    }}
                }}
                return originalFetch.call(this, url, options);
            }};
        </script>
        """
        
        # Insert the script before closing </head> tag
        html_content = html_content.replace('</head>', inject_script + '</head>')
        
        return html_content, 200, {'Content-Type': 'text/html; charset=utf-8'}
    except Exception as e:
        return f"Error loading agent status visualizer: {str(e)}", 500

@app.route('/api/agent-status')
def agent_status_api():
    """API endpoint to get current agent statuses and messages"""
    if not AGENT_VISUALIZER_AVAILABLE:
        return jsonify({'error': 'Agent status visualizer not available'}), 404
    
    try:
        # Get API key from query parameters or headers
        api_key = request.args.get('api_key') or request.headers.get('X-API-Key')
        temp_session_id = create_temp_session_id(request, api_key)
        user_session = gui_instance.get_user_session(temp_session_id, api_key)
        if not user_session:
            return jsonify({'error': 'Authentication failed. Please provide a valid API key.'}), 401
        user_base_dir = user_session.get_user_directory(gui_instance.base_data_dir)
        
        # Get directory from query parameter (selected directory)
        dir_name = request.args.get('dir')
        
        # Try to find the output directory
        output_dir = None
        if dir_name:
            # Use the selected directory from query parameter
            # Ensure dir_name doesn't already contain user directory path
            # If it does, extract just the directory name
            if os.path.sep in dir_name or '/' in dir_name:
                # dir_name might contain user directory, extract just the basename
                dir_name = os.path.basename(dir_name)
            output_dir = os.path.join(user_base_dir, dir_name)
            if not os.path.exists(output_dir):
                return jsonify({'error': f'Directory not found: {dir_name} (searched in: {user_base_dir})'}), 404
        elif user_session.current_output_dir:
            output_dir = os.path.join(user_base_dir, user_session.current_output_dir)
        elif user_session.last_output_dir:
            output_dir = os.path.join(user_base_dir, user_session.last_output_dir)
        else:
            # Try to find latest output directory
            latest_dir = find_latest_output_dir(user_base_dir)
            if latest_dir:
                output_dir = latest_dir
        
        if not output_dir or not os.path.exists(output_dir):
            return jsonify({
                'error': 'Output directory not found',
                'agents': {},
                'messages': [],
                'agent_ids': [],
                'output_directory': output_dir or '未设置',
                'timestamp': datetime.datetime.now().isoformat()
            }), 404
        
        # Load all agent statuses
        status_files = find_status_files(output_dir)
        agent_statuses = {}
        
        for status_file in status_files:
            status_data = load_status_file(status_file)
            if status_data:
                agent_id = status_data.get('agent_id', 'unknown')
                agent_statuses[agent_id] = status_data
        
        # Also add manager if not present
        if 'manager' not in agent_statuses:
            agent_statuses['manager'] = {
                'agent_id': 'manager',
                'status': 'running',
                'current_loop': 0
            }
        
        # Load all messages
        messages = find_message_files(output_dir)
        sorted_messages = sorted(messages, key=lambda x: x.get('timestamp', '') or '')
        
        # Load tool calls from log files
        tool_calls = find_tool_calls_from_logs(output_dir)
        
        # Load mermaid figures from plan.md
        mermaid_figures = find_mermaid_figures_from_plan(output_dir)
        
        # Load status updates from status files
        status_updates = find_status_updates(output_dir)
        
        # Get all unique agent IDs
        agent_ids = set(agent_statuses.keys())
        for msg in messages:
            agent_ids.add(msg.get('sender_id', ''))
            agent_ids.add(msg.get('receiver_id', ''))
        agent_ids = sorted([aid for aid in agent_ids if aid])
        
        return jsonify({
            'agents': agent_statuses,
            'messages': sorted_messages,
            'tool_calls': tool_calls,
            'status_updates': status_updates,
            'mermaid_figures': mermaid_figures,
            'agent_ids': agent_ids,
            'output_directory': output_dir,
            'timestamp': datetime.datetime.now().isoformat(),
            'message_count': len(sorted_messages)
        })
    except Exception as e:
        import traceback
        error_msg = str(e)
        traceback.print_exc()
        return jsonify({
            'error': f'Error loading status: {error_msg}',
            'agents': {},
            'messages': [],
            'agent_ids': [],
            'output_directory': 'Error',
            'timestamp': datetime.datetime.now().isoformat()
        }), 500

@app.route('/api/agent-status-reload', methods=['POST'])
def agent_status_reload():
    """API endpoint to reload and find the latest output directory"""
    if not AGENT_VISUALIZER_AVAILABLE:
        return jsonify({'success': False, 'message': 'Agent status visualizer not available'}), 404
    
    try:
        # Get API key from query parameters or headers
        api_key = request.args.get('api_key') or request.headers.get('X-API-Key')
        temp_session_id = create_temp_session_id(request, api_key)
        user_session = gui_instance.get_user_session(temp_session_id, api_key)
        if not user_session:
            return jsonify({'error': 'Authentication failed. Please provide a valid API key.'}), 401
        user_base_dir = user_session.get_user_directory(gui_instance.base_data_dir)
        
        # Get directory from query parameter (selected directory)
        dir_name = request.args.get('dir')
        
        # If dir parameter is provided, use it; otherwise find latest
        if dir_name:
            # Ensure dir_name doesn't already contain user directory path
            # If it does, extract just the directory name
            if os.path.sep in dir_name or '/' in dir_name:
                # dir_name might contain user directory, extract just the basename
                dir_name = os.path.basename(dir_name)
            new_output_dir = os.path.join(user_base_dir, dir_name)
            if not os.path.exists(new_output_dir):
                return jsonify({
                    'success': False,
                    'message': f'Directory not found: {dir_name} (searched in: {user_base_dir})',
                    'output_directory': 'Not set'
                }), 404
        else:
            # Find latest output directory
            new_output_dir = find_latest_output_dir(user_base_dir)
        
        if new_output_dir and os.path.exists(new_output_dir):
            # Update user session's last output dir
            rel_path = os.path.relpath(new_output_dir, user_base_dir)
            user_session.last_output_dir = rel_path
            
            return jsonify({
                'success': True,
                'output_directory': new_output_dir,
                'message': f'Reloaded: {os.path.basename(new_output_dir)}'
            })
        else:
            return jsonify({
                'success': False,
                'message': 'No output directory found',
                'output_directory': 'Not set'
            }), 404
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error: {str(e)}'
        }), 500

@app.route('/api/agent-status-files/<path:path>')
def agent_status_files(path):
    """Serve files from output directory (for mermaid images)"""
    if not AGENT_VISUALIZER_AVAILABLE:
        return jsonify({'error': 'Agent status visualizer not available'}), 404
    
    try:
        # Get API key from query parameters or headers
        api_key = request.args.get('api_key') or request.headers.get('X-API-Key')
        temp_session_id = create_temp_session_id(request, api_key)
        user_session = gui_instance.get_user_session(temp_session_id, api_key)
        if not user_session:
            return jsonify({'error': 'Authentication failed. Please provide a valid API key.'}), 401
        user_base_dir = user_session.get_user_directory(gui_instance.base_data_dir)
        
        # Get directory from query parameter (selected directory)
        dir_name = request.args.get('dir')
        
        # Try to find the output directory
        output_dir = None
        if dir_name:
            # Use the selected directory from query parameter
            # Ensure dir_name doesn't already contain user directory path
            # If it does, extract just the directory name
            if os.path.sep in dir_name or '/' in dir_name:
                # dir_name might contain user directory, extract just the basename
                dir_name = os.path.basename(dir_name)
            output_dir = os.path.join(user_base_dir, dir_name)
            if not os.path.exists(output_dir):
                return jsonify({'error': f'Directory not found: {dir_name} (searched in: {user_base_dir})'}), 404
        elif user_session.current_output_dir:
            # Ensure current_output_dir doesn't already contain user directory path
            current_dir = user_session.current_output_dir
            if os.path.sep in current_dir or '/' in current_dir:
                current_dir = os.path.basename(current_dir)
            output_dir = os.path.join(user_base_dir, current_dir)
        elif user_session.last_output_dir:
            # Ensure last_output_dir doesn't already contain user directory path
            last_dir = user_session.last_output_dir
            if os.path.sep in last_dir or '/' in last_dir:
                last_dir = os.path.basename(last_dir)
            output_dir = os.path.join(user_base_dir, last_dir)
        else:
            latest_dir = find_latest_output_dir(user_base_dir)
            if latest_dir:
                output_dir = latest_dir
        
        if not output_dir:
            return jsonify({'error': 'Output directory not set'}), 404
        
        # Construct full path
        file_path = os.path.join(output_dir, path)
        
        # Security check: ensure path is within OUTPUT_DIR
        real_output_dir = os.path.realpath(output_dir)
        real_file_path = os.path.realpath(file_path)
        if not real_file_path.startswith(real_output_dir):
            return jsonify({'error': 'Invalid path'}), 403
        
        if not os.path.exists(file_path):
            return jsonify({'error': 'File not found'}), 404
        
        return send_from_directory(output_dir, path)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

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
        
        # Security check: normalize path and prevent path traversal
        # Don't use secure_filename as it destroys Chinese characters
        normalized_dir_name = os.path.normpath(dir_name)
        if '..' in normalized_dir_name or normalized_dir_name.startswith('/'):
            return jsonify({'success': False, 'error': 'Access denied: Invalid directory path'})
        
        # Target directory path
        target_dir = os.path.join(user_base_dir, normalized_dir_name)
        
        # Security check: ensure directory is within user's output directory
        real_output_dir = os.path.realpath(user_base_dir)
        real_target_dir = os.path.realpath(target_dir)
        if not real_target_dir.startswith(real_output_dir):
            return jsonify({'success': False, 'error': 'Access denied: Invalid directory path'})
        
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
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    safe_filename = f"{name}_{timestamp}{ext}"
                
                file_path = os.path.join(workspace_dir, safe_filename)
                
                file.save(file_path)
                uploaded_files.append(safe_filename)
        
        # 构造上传成功消息，只显示文件名，不显示文件数量
        files_str = ', '.join(uploaded_files)
        # 通过检查i18n字典中的upload_success键来判断语言
        upload_success_text = i18n.get('upload_success', '')
        if '成功上传' in upload_success_text or upload_success_text.startswith('成功上传'):
            message = f'成功上传文件: {files_str}'
        else:
            message = f'Successfully uploaded files: {files_str}'
        
        return jsonify({
            'success': True,
            'message': message,
            'files': uploaded_files
        })
        
    except Exception as e:
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
        
        # Security check: normalize old path and prevent path traversal
        # Don't use secure_filename as it destroys Chinese characters
        normalized_old_name = os.path.normpath(old_name)
        if '..' in normalized_old_name or normalized_old_name.startswith('/'):
            return jsonify({'success': False, 'error': 'Access denied: Invalid directory path'})
        
        # Build complete path
        old_path = os.path.join(user_base_dir, normalized_old_name)
        new_path = os.path.join(user_base_dir, new_name_safe)
        
        # Debug info
        
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
        
        
        # Rename directory
        os.rename(old_path, new_path)
        
        # Update user session related states
        if hasattr(user_session, 'selected_output_dir') and user_session.selected_output_dir == old_name:
            user_session.selected_output_dir = new_name_safe
        if hasattr(user_session, 'last_output_dir') and user_session.last_output_dir == old_name:
            user_session.last_output_dir = new_name_safe
        
        
        return jsonify({
            'success': True, 
            'message': f'Directory renamed successfully: {old_name} -> {new_name_safe}',
            'old_name': old_name,
            'new_name': new_name_safe
        })
        
    except Exception as e:
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
        
        # Security check: normalize path and prevent path traversal
        # Don't use secure_filename as it destroys Chinese characters
        normalized_dir_name = os.path.normpath(dir_name)
        if '..' in normalized_dir_name or normalized_dir_name.startswith('/'):
            return jsonify({'success': False, 'error': 'Access denied: Invalid directory path'})
        
        # Construct target directory path (preserve Chinese characters)
        target_dir = os.path.join(user_base_dir, normalized_dir_name)
        
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
        
        
        # Delete directory and all its contents
        shutil.rmtree(target_dir)

        # Check if deletion was successful with retry logic
        max_retries = 3
        for attempt in range(max_retries):
            if not os.path.exists(target_dir):
                # Directory successfully deleted
                # Clean user session related states
                if hasattr(user_session, 'last_output_dir') and user_session.last_output_dir == dir_name:
                    user_session.last_output_dir = None
                if hasattr(user_session, 'selected_output_dir') and user_session.selected_output_dir == dir_name:
                    user_session.selected_output_dir = None

                return jsonify({'success': True})
            else:
                # Directory still exists, wait 1 second before retry (except on last attempt)
                if attempt < max_retries - 1:
                    import time
                    time.sleep(1)

        # If we reach here, deletion failed after all retries
        return jsonify({'success': False, 'error': f'Directory deletion failed after {max_retries} attempts'})
        
    except PermissionError as e:
        return jsonify({'success': False, 'error': f'Permission denied: {str(e)}'})
    except Exception as e:
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
            # Delete the file
            os.remove(full_file_path)
        elif os.path.isdir(full_file_path):
            # Delete the folder and all its contents
            shutil.rmtree(full_file_path)
        else:
            return jsonify({'success': False, 'error': f'Path is neither a file nor a directory: {file_path}'})
        
        
        return jsonify({
            'success': True, 
            'message': f'File "{os.path.basename(file_path)}" has been successfully deleted'
        })
        
    except PermissionError as e:
        return jsonify({'success': False, 'error': f'Permission denied: {str(e)}'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/routine-files', methods=['GET'])
def get_routine_files():
    """Get list of routine files from routine directory and workspace files starting with 'routine_'"""
    try:
        routine_files = []
        workspace_dir = os.getcwd()
        
        # 根据URL参数或语言配置选择routine文件夹
        lang_param = request.args.get('lang')
        if lang_param and lang_param in ('zh', 'en'):
            current_lang = lang_param
        else:
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
        
        # 按名称排序（反向排序，推荐类文件在上边）
        routine_files.sort(key=lambda x: x['name'], reverse=True)
        
        return jsonify({
            'success': True,
            'files': routine_files
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'files': []
        }), 500

@app.route('/api/validate-config', methods=['POST'])
def validate_config():
    """Validate GUI configuration (without returning sensitive information)"""
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
        
        config_value = model_config.get('value')
        model_name = model_config.get('model')
        max_tokens = model_config.get('max_tokens', 8192)
        
        # 验证max_tokens是有效的数字
        try:
            max_tokens = int(max_tokens) if max_tokens else 8192
            if max_tokens <= 0:
                max_tokens = 8192
        except (ValueError, TypeError):
            max_tokens = 8192
        
        # 如果是内置配置（不是 'custom'），从服务器端读取并验证
        if config_value and config_value != 'custom':
            gui_config = get_gui_config()
            config_model = gui_config.get('model', 'glm-4.5')
            
            # 验证模型名称是否存在
            if not model_name:
                # 如果前端没有提供模型名称，使用服务器端的模型名称
                model_name = config_model
            
            if config_value == config_model:
                # 读取GUI配置并验证
                is_valid, error_message = validate_gui_config(gui_config)
                
                if not is_valid:
                    return jsonify({
                        'success': False,
                        'error': error_message
                    })
            
            # 验证模型名称是否存在
            if not model_name:
                i18n = get_i18n_texts()
                return jsonify({
                    'success': False,
                    'error': i18n['config_incomplete']
                })
            
            # 对于内置配置，只返回非敏感信息
            return jsonify({
                'success': True,
                'config': {
                    # 不返回 api_key 和 api_base，这些敏感信息只在发起任务时从服务器端读取
                    'model': model_name,
                    'max_tokens': max_tokens
                }
            })
        else:
            # 自定义配置：验证用户输入的配置
            api_key = model_config.get('api_key')
            api_base = model_config.get('api_base')
            
            # 验证必需字段
            if not api_key or not api_base or not model_name:
                i18n = get_i18n_texts()
                return jsonify({
                    'success': False,
                    'error': i18n['config_incomplete']
                })
            
            # 对于自定义配置，只返回非敏感信息（前端已经有完整配置）
            return jsonify({
                'success': True,
                'config': {
                    'model': model_name,
                    'max_tokens': max_tokens
                }
            })
        
    except Exception as e:
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
        if not user_session:
            return jsonify({'success': False, 'error': 'Authentication failed or session creation failed. Please ensure you are connected with a valid API key.'})
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
        
        # Auto-convert SVG to PNG if the saved file is an SVG
        if rel_path.lower().endswith('.svg') and SVG_TO_PNG_CONVERTER_AVAILABLE:
            try:
                from pathlib import Path
                svg_path = Path(full_path)
                png_path = svg_path.with_suffix('.png')
                
                converter = EnhancedSVGToPNGConverter()
                success, message = converter.convert(svg_path, png_path, enhance_chinese=True, dpi=300)
                

            except Exception as e:
                # 转换失败不影响SVG保存成功
                print(f"⚠️ SVG转PNG出错: {e}")
        
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
        if not user_session:
            return jsonify({'success': False, 'error': 'Authentication failed or session creation failed. Please ensure you are connected with a valid API key.'})
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


@app.route('/api/reparse-markdown-diagrams', methods=['POST'])
def reparse_markdown_diagrams():
    """重新解析Markdown文件中的Mermaid图表和SVG代码块"""
    try:
        data = request.get_json() or {}
        rel_path = data.get('path')
        
        if not rel_path:
            return jsonify({'success': False, 'error': 'File path is required'})
        
        # 获取用户会话
        api_key = request.args.get('api_key') or request.headers.get('X-API-Key') or data.get('api_key')
        temp_session_id = create_temp_session_id(request, api_key)
        user_session = gui_instance.get_user_session(temp_session_id, api_key)
        user_base_dir = user_session.get_user_directory(gui_instance.base_data_dir)
        
        # 获取完整路径
        full_path = os.path.join(user_base_dir, rel_path)
        real_output_dir = os.path.realpath(user_base_dir)
        real_file_path = os.path.realpath(full_path)
        
        # 安全检查
        if not real_file_path.startswith(real_output_dir):
            return jsonify({'success': False, 'error': 'Access denied'})
        
        if not os.path.exists(real_file_path):
            return jsonify({'success': False, 'error': 'File not found'})
        
        if not rel_path.lower().endswith('.md'):
            return jsonify({'success': False, 'error': 'Only markdown files are supported'})
        
        # 使用FileSystemTools的process_markdown_diagrams方法
        from src.tools.file_system_tools import FileSystemTools
        
        fs_tools = FileSystemTools(workspace_root=user_base_dir)
        result = fs_tools.process_markdown_diagrams(rel_path)
        
        if result.get('status') in ['success', 'skipped']:
            return jsonify({
                'success': True,
                'message': result.get('message', 'Processing completed'),
                'details': {
                    'mermaid': result.get('mermaid_processing', {}),
                    'svg': result.get('svg_processing', {})
                }
            })
        else:
            return jsonify({
                'success': False,
                'error': result.get('message', 'Processing failed'),
                'details': result
            })
            
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/gui-configs', methods=['GET'])
def get_gui_configs():
    """Get available GUI model configurations (without sensitive information)"""
    try:
        from src.config_loader import get_all_model_configs, get_gui_config
        
        # 读取所有模型配置（包括注释掉的）
        all_configs = get_all_model_configs()
        
        # 读取当前激活的GUI配置（用于确定默认选择）
        gui_config = get_gui_config()
        current_model = gui_config.get('model', '')
        current_api_base = gui_config.get('api_base', '')
        
        i18n = get_i18n_texts()
        configs = []
        
        # 添加所有找到的配置
        for config in all_configs:
            model = config.get('model', '')
            api_base = config.get('api_base', '')
            display_name = config.get('display_name', model)
            
            # 创建唯一标识符（使用model和api_base的组合）
            config_id = f"{model}__{api_base}"
            
            configs.append({
                'value': config_id,
                'label': display_name,
                # 不返回 api_key 和 api_base，保护敏感信息
                'model': model,
                'max_tokens': config.get('max_tokens', 8192),
                'display_name': display_name,
                'enabled': config.get('enabled', True)
            })
        
        # 添加自定义选项
        configs.append({
            'value': 'custom',
            'label': i18n['custom_label'],
            'model': '',
            'max_tokens': 8192,
            'display_name': i18n['custom_label'],
            'enabled': True
        })
        
        return jsonify({
            'success': True,
            'configs': configs,
            'current_model': current_model,
            'current_api_base': current_api_base
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        })


@app.route('/api/get-model-config', methods=['POST'])
def get_model_config():
    """Get model configuration details by config ID (including sensitive information)"""
    try:
        from src.config_loader import get_all_model_configs
        
        data = request.json
        config_id = data.get('config_id', '')
        
        if not config_id:
            return jsonify({
                'success': False,
                'error': 'Config ID is required'
            })
        
        # Handle custom config
        if config_id == 'custom':
            return jsonify({
                'success': True,
                'config': {
                    'value': 'custom',
                    'model': '',
                    'api_key': '',
                    'api_base': '',
                    'max_tokens': 8192
                }
            })
        
        # Parse config_id (format: "model__api_base")
        if '__' not in config_id:
            return jsonify({
                'success': False,
                'error': 'Invalid config ID format'
            })
        
        model, api_base = config_id.split('__', 1)
        
        # Get all configs and find matching one
        all_configs = get_all_model_configs()
        matching_config = None
        
        for config in all_configs:
            if config.get('model', '').strip() == model.strip() and \
               config.get('api_base', '').strip() == api_base.strip():
                matching_config = config
                break
        
        if not matching_config:
            return jsonify({
                'success': False,
                'error': 'Configuration not found'
            })
        
        # Return config with sensitive information (only for server-side use)
        return jsonify({
            'success': True,
            'config': {
                'value': config_id,
                'model': matching_config.get('model', ''),
                'api_key': matching_config.get('api_key', ''),
                'api_base': matching_config.get('api_base', ''),
                'max_tokens': matching_config.get('max_tokens', 8192),
                'display_name': matching_config.get('display_name', matching_config.get('model', ''))
            }
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        })


@app.route('/api/save-to-config', methods=['POST'])
def save_to_config():
    """Save custom model configuration to config.txt"""
    try:
        data = request.json
        api_key = data.get('api_key', '').strip()
        api_base = data.get('api_base', '').strip()
        model = data.get('model', '').strip()
        max_tokens = data.get('max_tokens', 8192)
        
        # Validate required fields
        if not api_key or not api_base or not model:
            return jsonify({
                'success': False,
                'error': 'All fields are required'
            })
        
        # Path to config.txt
        config_path = os.path.join(os.getcwd(), 'config', 'config.txt')
        
        if not os.path.exists(config_path):
            return jsonify({
                'success': False,
                'error': 'config.txt file not found'
            })
        
        # Read the current config file
        with open(config_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # Update the first uncommented configuration section
        updated_lines = []
        found_first_config = False
        lines_updated = 0
        
        for i, line in enumerate(lines):
            stripped = line.strip()
            
            # Skip empty lines and comments
            if not stripped or stripped.startswith('#'):
                updated_lines.append(line)
                continue
            
            # Check if this line contains a config key-value pair
            if '=' in line and not found_first_config:
                key = line.split('=')[0].strip()
                
                # Update the first configuration block (top-most uncommented configs)
                if key == 'api_key' and lines_updated == 0:
                    updated_lines.append(f'api_key={api_key}\n')
                    lines_updated += 1
                elif key == 'api_base' and lines_updated == 1:
                    updated_lines.append(f'api_base={api_base}\n')
                    lines_updated += 1
                elif key == 'model' and lines_updated == 2:
                    updated_lines.append(f'model={model}\n')
                    lines_updated += 1
                elif key == 'max_tokens' and lines_updated == 3:
                    updated_lines.append(f'max_tokens={max_tokens}\n')
                    lines_updated += 1
                    found_first_config = True  # We've updated all needed fields
                else:
                    updated_lines.append(line)
            else:
                updated_lines.append(line)
        
        # Write back to config.txt
        with open(config_path, 'w', encoding='utf-8') as f:
            f.writelines(updated_lines)
        
        # Clear config cache so changes take effect immediately
        from src.config_loader import clear_config_cache
        clear_config_cache()
        
        return jsonify({
            'success': True,
            'message': 'Configuration saved to config.txt successfully'
        })
        
    except Exception as e:
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
        if not user_session:
            return jsonify({'success': False, 'error': 'Authentication failed or session creation failed. Please ensure you are connected with a valid API key.'})
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
                return jsonify({'success': False, 'error': f'Optimization failed: {str(trad_error)}'})

        # Create backup if content changed
        if optimized_content != original_content:
            backup_path = full_path + '.optimized_backup'
            try:
                with open(backup_path, 'w', encoding='utf-8') as f:
                    f.write(original_content)
            except Exception as backup_error:
                pass

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
            return {}

        # Load the example configuration
        with open(example_config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)

        # Return the mcpServers section
        return config.get('mcpServers', {})

    except Exception as e:
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
                pass

        # Generate filename with timestamp to avoid conflicts
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        config_filename = f"mcp_servers_custom_{timestamp}.json"
        custom_config_path = os.path.join(out_dir, config_filename)

        # Write custom configuration to file
        with open(custom_config_path, 'w', encoding='utf-8') as f:
            json.dump(custom_config, f, indent=2, ensure_ascii=False)

        return custom_config_path

    except Exception as e:
        return None


if __name__ == '__main__':
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='AGIAgent GUI Server')
    parser.add_argument('--port', '-p', type=int, default=5002, 
                       help='Port specified to use')
    args = parser.parse_args()
    
    # 优先使用命令行参数，其次使用环境变量，最后使用默认值
    port = args.port if args.port else int(os.environ.get('PORT', 5002))
    
    print(f"🚀 Starting AGIAgent GUI Server on port {port}")
    socketio.run(app, host='0.0.0.0', port=port, debug=False, allow_unsafe_werkzeug=True) 
    print(f"🚀 Wait for 5 seconds and open the browser with url 127.0.0.1:{port}")
