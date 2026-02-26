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

from flask import Flask, render_template, request, jsonify, send_file, send_from_directory, after_this_request, abort, Response, redirect
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
from typing import Optional
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
    from src.utils.llm_svg_optimizer import create_image_generation_optimizer_from_config
    LLM_SVG_OPTIMIZER_AVAILABLE = True
except ImportError:
    #print("⚠️ Image generation SVG optimizer not available")
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

# Application name macro definition (will be updated by AppManager)
APP_NAME = "AGI Agent"

from src.main import AGIAgentMain
from app_manager import AppManager




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
        'welcome_message': f'你好呀，我是一个聪明能干的智能体。很高兴见到你～请选择一个技能，执行效果更好哦',
        'workspace_title': '工作目录',
        'file_preview': '文件预览',
        'data_directory_info': '数据目录',
        'disconnected': '与服务器断开连接',
        'drag_files': '拖拽文件到此处或点击选择文件',
        'drop_files_to_upload': '拖放文件到此处上传',
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
        'please_connect': '当前没有登陆，请先注册并使用API Key连接。您也可以使用空API Key连接后参观已有案例',
        'uploading_files': '正在上传 {0} 个文件',
        'upload_progress': '上传进度: {0}%',
        'upload_completed': '上传文档已完成',
        'upload_failed_http': '上传失败: HTTP {0}',
        
        # Directory operations
        'directory_created_with_workspace': '已创建新工作目录: {0} (包含workspace子目录)',
        'directory_list_refreshed': '目录列表已刷新',
        'refreshing_directories': '正在刷新目录',
        'no_files_selected': '没有选择文件',
        'no_valid_files': '没有选择有效文件',
        'target_directory_not_exist': '目标目录不存在',
        'upload_success': '成功上传 {0} 个文件',
        'new_name_empty': '新名称不能为空',
        
        # Multi-user support
        'api_key_label': '登录码',
        'api_key_placeholder': 'API Key (可选)',
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
        'model_label': '模型',
        'model_tooltip': '选择要使用的AI模型',
        'model_claude_sonnet': 'claude-sonnet-4-0 (高精度)',
        'model_gpt_4': 'gpt-4.1 (高效率)',
        'config_error_title': '配置错误',
        'config_error_invalid_key': '登陆码（login key）配置无效，请检查config/config.txt文件中的GUI API configuration部分',
        
        # Custom model config dialog
        'custom_config_title': '自定义模型配置',
        'custom_api_key_label': '登陆码',
        'custom_api_base_label': 'API Base URL:',
        'custom_model_label': '模型名称',
        'custom_max_tokens_label': 'Max Output Tokens:',
        'custom_api_key_placeholder': '请输入登陆码',
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
        
        # Platform selection
        'default_platform': '主平台',
        
        # Contact us
        'contact_us': '联系我们',
        'contact_message_label': '留言内容',
        'contact_message_placeholder': '请输入您的留言...',
        'contact_current_dir_label': '当前工作目录',
        'contact_contact_info_label': '您的联系方式（邮箱或电话，选填）',
        'contact_contact_info_placeholder': '请输入您的邮箱或电话（选填）',
        'contact_submit_success': '留言已提交，感谢您的反馈！',
        'contact_submit_error': '提交失败',
        'contact_message_empty': '请输入留言内容',
        
        # History labels
        'oldest': '最老',
        'newest': '最新',
        
        # Help
        'help': '帮助',
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
        'welcome_message': f'I am ready. Please enter your requirements below, and I will automatically process tasks for you.',
        'workspace_title': 'Workspace',
        'file_preview': 'File Preview',
        'data_directory_info': 'Data Directory',
        'disconnected': 'Disconnected from server',
        'drag_files': 'Drag files here or click to select files',
        'drop_files_to_upload': 'Drop files here to upload',
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
        'please_connect': 'Currently not logged in. Please register and connect with API Key, or connect without API Key to view existing cases',
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
        'refreshing_directories': 'Refreshing directories...',
        'no_files_selected': 'No files selected',
        'no_valid_files': 'No valid files selected',
        'target_directory_not_exist': 'Target directory does not exist',
        'upload_success': 'Successfully uploaded {0} files',
        'new_name_empty': 'New name cannot be empty',
        
        # Multi-user support
        'api_key_label': 'login key',
        'api_key_placeholder': 'Enter login key (optional)',
        'api_key_tooltip': 'Enter your login key, leave empty for default user mode',
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
        'model_label': 'Model',
        'model_tooltip': 'Select AI model to use',
        'model_claude_sonnet': 'claude-sonnet-4-0 (High Accuracy)',
        'model_gpt_4': 'gpt-4.1 (High Efficiency)',
        'config_error_title': 'Configuration Error',
        'config_error_invalid_key': 'Invalid login key configuration, please check GUI API configuration in config/config.txt',
        
        # Custom model config dialog
        'custom_config_title': 'Custom Model Configuration',
        'custom_api_key_label': 'login key',
        'custom_api_base_label': 'API Base URL:',
        'custom_model_label': 'Model Name:',
        'custom_max_tokens_label': 'Max Output Tokens:',
        'custom_api_key_placeholder': 'Enter login key',
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
        
        # Platform selection
        'default_platform': 'Default Platform',
        
        # Contact us
        'contact_us': 'Contact Us',
        'contact_message_label': 'Message',
        'contact_message_placeholder': 'Please enter your message...',
        'contact_current_dir_label': 'Current Workspace Directory',
        'contact_contact_info_label': 'Your Contact Information (Email or Phone, Optional)',
        'contact_contact_info_placeholder': 'Please enter your email or phone (optional)',
        'contact_submit_success': 'Message submitted, thank you for your feedback!',
        'contact_submit_error': 'Submission failed',
        'contact_message_empty': 'Please enter your message',
        
        # History labels
        'oldest': 'Oldest',
        'newest': 'Newest',
        
        # Help
        'help': 'Help',
    }
}

def get_i18n_texts():
    """Get internationalization text for current language"""
    current_lang = get_language()
    return I18N_TEXTS.get(current_lang, I18N_TEXTS['en'])

def execute_agia_task_process_target(user_requirement, output_queue, input_queue, out_dir=None, continue_mode=False, plan_mode=False, gui_config=None, session_id=None, detailed_requirement=None, user_id=None, attached_files=None, app_name=None, user_dir=None):
    """
    This function runs in a separate process.
    It cannot use the `socketio` object directly.
    It communicates back to the main process via the queue.
    User input is received via input_queue in GUI mode.
    
    Args:
        app_name: Application name (e.g., 'patent') for app-specific configuration
        user_dir: User directory path for checking shared directory
    """
    # Store input_queue in a way that talk_to_user can access it
    import sys
    import __main__
    __main__._agia_gui_input_queue = input_queue
    
    try:
        # Initialize AppManager in this process
        # Determine base_dir (project root)
        current_file = os.path.abspath(__file__)
        gui_dir = os.path.dirname(current_file)
        base_dir = os.path.dirname(gui_dir)
        app_manager = AppManager(app_name=app_name, base_dir=base_dir)
        
        # Get i18n texts for this process (after sending initial message)
        i18n = get_i18n_texts()
        
        if not out_dir:
            # Get GUI default data directory from config for new directories
            # Use app-specific config file if available
            from src.config_loader import get_gui_default_data_directory
            config_file = "config/config.txt"  # default
            if app_manager.is_app_mode():
                app_config_path = app_manager.get_config_path()
                if app_config_path:
                    config_file = app_config_path
            config_data_dir = get_gui_default_data_directory(config_file)
            if config_data_dir:
                base_data_dir = config_data_dir
            else:
                base_data_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            
            # Create output directory in user directory, not directly in base_data_dir
            # user_dir should be the full path to user's directory (e.g., /mnt/data_colordoc/user1)
            if user_dir and os.path.exists(user_dir):
                # Use provided user_dir
                user_output_base = user_dir
            else:
                # Fallback: create in base_data_dir/userdata if user_dir not provided
                user_output_base = os.path.join(base_data_dir, 'userdata')
                os.makedirs(user_output_base, exist_ok=True)
            
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            out_dir = os.path.join(user_output_base, f"output_{timestamp}")
        
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
        
        # Get prompts folder and routine path from AppManager
        prompts_folder = None
        routine_file = None
        
        # Set app-specific config file if available
        if app_manager.is_app_mode():
            config_path = app_manager.get_config_path(user_dir=user_dir)
            if config_path:
                os.environ['AGIA_CONFIG_FILE'] = config_path

        # 简化设计：根据app_name直接查找routine文件
        # 前端必须传递app_name和routine_file，后端直接根据app查找
        routine_file_from_gui = gui_config.get('routine_file')
        
        if routine_file_from_gui:
            # 检查是否是workspace文件（以routine_开头）
            if routine_file_from_gui.startswith('routine_'):
                # 直接使用workspace根目录下的文件
                routine_file = os.path.join(os.getcwd(), routine_file_from_gui)
            elif app_name and app_manager.is_app_mode():
                # 如果有app_name且app模式已启用，直接根据app查找routine文件
                if app_manager.app_config and app_manager.app_dir:
                    routine_path_config = app_manager.app_config.get('routine_path', 'routine')
                    app_routine_dir = os.path.join(app_manager.app_dir, routine_path_config)
                    app_routine_dir = os.path.abspath(app_routine_dir)
                    
                    # 优先检查用户shared目录
                    if user_dir:
                        shared_routine_dir = os.path.join(user_dir, 'shared', routine_path_config)
                        if os.path.exists(shared_routine_dir) and os.path.isdir(shared_routine_dir):
                            shared_routine_file = os.path.join(shared_routine_dir, routine_file_from_gui)
                            if os.path.exists(shared_routine_file):
                                routine_file = shared_routine_file
                    
                    # 如果shared目录没有，使用app目录
                    if not routine_file and os.path.exists(app_routine_dir) and os.path.isdir(app_routine_dir):
                        app_routine_file = os.path.join(app_routine_dir, routine_file_from_gui)
                        if os.path.exists(app_routine_file):
                            routine_file = app_routine_file
                    
                    # 如果找不到，显示警告（使用app目录路径）
                    if not routine_file:
                        warning_path = os.path.join(app_routine_dir, routine_file_from_gui)
                        output_queue.put({'event': 'output', 'data': {'message': f"Warning: Routine file not found: {warning_path}", 'type': 'warning'}})
                else:
                    # app配置加载失败
                    output_queue.put({'event': 'output', 'data': {'message': f"Warning: App config not found for app: {app_name}", 'type': 'warning'}})
            else:
                # 没有app_name或不在app模式，使用默认routine目录（向后兼容）
                prompts_folder = app_manager.get_prompts_folder(user_dir=user_dir)
                current_lang = gui_config.get('language')
                if not current_lang or current_lang not in ('zh', 'en'):
                    current_lang = get_language()
                if current_lang == 'zh':
                    routine_file = os.path.join(os.getcwd(), 'routine_zh', routine_file_from_gui)
                else:
                    routine_file = os.path.join(os.getcwd(), 'routine', routine_file_from_gui)
                
                if not os.path.exists(routine_file):
                    output_queue.put({'event': 'output', 'data': {'message': f"Warning: Routine file not found: {routine_file}", 'type': 'warning'}})
                    routine_file = None
        
        # 获取prompts文件夹（如果还没有获取）
        if prompts_folder is None:
            prompts_folder = app_manager.get_prompts_folder(user_dir=user_dir)

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
                # Use app-specific config file if available
                config_file = "config/config.txt"
                if app_manager.is_app_mode():
                    app_config_path = app_manager.get_config_path(user_dir=user_dir)
                    if app_config_path:
                        config_file = app_config_path
                
                gui_config_from_server = get_gui_config(config_file)
                
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
        
        agia = AGIAgentMain(
            out_dir=out_dir,
            debug_mode=False,
            detailed_summary=True,
            single_task_mode=single_task_mode,  # Set based on plan_mode
            interactive_mode=False,  # Disable interactive mode
            continue_mode=False,  # Always use False for GUI mode to avoid shared .agia_last_output.json
            MCP_config_file=mcp_config_file,  # Set based on GUI MCP option
            prompts_folder=prompts_folder,  # Use app-specific prompts folder if available
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
    def __init__(self, app_name: Optional[str] = None):
        # User session management
        self.user_sessions = {}  # session_id -> UserSession
        
        # Initialize authentication manager
        self.auth_manager = AuthenticationManager()
        
        # Save initial app_name for resetting to default platform
        self.initial_app_name = app_name
        
        # Initialize app manager
        self.app_manager = AppManager(app_name=app_name)
        
        # Update global APP_NAME if app is configured
        global APP_NAME
        if self.app_manager.is_app_mode():
            APP_NAME = self.app_manager.get_app_name()
        
        # Initialize concurrency manager with reference to this GUI instance
        self.concurrency_manager = ConcurrencyManager(
            max_concurrent_tasks=16,  # Maximum concurrent tasks (Expand by 1x)
            max_connections=40,       # 最大Connect数 (Expand by 1x)
            gui_instance=self         # Pass GUI instance for unified monitoring
        )
        
        # Get GUI default data directory from config, fallback to current directory
        # If in app mode, use app-specific config file
        config_file = "config/config.txt"  # default
        if self.app_manager.is_app_mode():
            app_config_path = self.app_manager.get_config_path()
            if app_config_path:
                config_file = app_config_path
        config_data_dir = get_gui_default_data_directory(config_file)
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
        
        # 性能优化：添加缓存机制
        self._directory_cache = {}  # 缓存目录结构和大小
        self._task_description_cache = {}  # 缓存任务描述
        self._cache_lock = Lock()  # 缓存锁
        self._cache_timeout = 30  # 缓存超时时间（秒）
        
    def switch_app(self, app_name: Optional[str], session_id: Optional[str] = None):
        """
        动态切换应用平台
        
        Args:
            app_name: 应用名称（如 'patent'），如果为None则重置为默认模式
            session_id: 会话ID，如果提供则切换指定用户的app，否则切换全局默认app（向后兼容）
        """
        # 创建临时 AppManager 来获取配置路径（用于更新 base_data_dir）
        temp_app_manager = AppManager(app_name=app_name)
        
        # 更新 base_data_dir 以使用新的 app 配置（无论是否有 session_id，都需要更新全局 base_data_dir）
        config_file = "config/config.txt"  # default
        if temp_app_manager.is_app_mode():
            app_config_path = temp_app_manager.get_config_path()
            if app_config_path:
                config_file = app_config_path
        config_data_dir = get_gui_default_data_directory(config_file)
        if config_data_dir:
            self.base_data_dir = config_data_dir
        else:
            self.base_data_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        # Ensure base directory exists
        os.makedirs(self.base_data_dir, exist_ok=True)
        
        if session_id:
            # 会话级切换：直接更新用户的 AppManager 实例
            if session_id in self.user_sessions:
                user_session = self.user_sessions[session_id]
                # 直接创建并更新 AppManager 实例，简单高效
                user_session.app_manager = temp_app_manager
                user_session.current_app_name = app_name  # 保留用于日志和调试
        else:
            # 全局切换（向后兼容，用于初始化或默认模式）
            # 使用已创建的 AppManager 实例
            self.app_manager = temp_app_manager
            
            # 更新全局 APP_NAME
            global APP_NAME
            if self.app_manager.is_app_mode():
                APP_NAME = self.app_manager.get_app_name()
            else:
                APP_NAME = "AGI Agent"
            
            # 更新环境变量 AGIA_APP_NAME（保持向后兼容）
            if app_name:
                os.environ['AGIA_APP_NAME'] = app_name
            else:
                # 如果设置为None，清除环境变量
                if 'AGIA_APP_NAME' in os.environ:
                    del os.environ['AGIA_APP_NAME']
    
    def ensure_app_switched_for_request(self, request, session_id: Optional[str] = None):
        """
        确保当前请求的 base_data_dir 是正确的（根据 URL 自动切换 app）
        这个方法应该在所有使用 base_data_dir 的 API 路由中调用
        
        Args:
            request: Flask request 对象
            session_id: 会话ID（可选）
        """
        app_name = get_app_name_from_url(request)
        
        # 获取当前应该使用的 base_data_dir
        temp_app_manager = AppManager(app_name=app_name)
        config_file = "config/config.txt"  # default
        if temp_app_manager.is_app_mode():
            app_config_path = temp_app_manager.get_config_path()
            if app_config_path:
                config_file = app_config_path
        
        expected_data_dir = get_gui_default_data_directory(config_file)
        if not expected_data_dir:
            expected_data_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        # 如果当前 base_data_dir 不正确，则切换
        if self.base_data_dir != expected_data_dir:
            if session_id:
                self.switch_app(app_name, session_id=session_id)
            else:
                # 如果没有 session_id，先创建临时 session
                api_key = request.args.get('api_key') or request.headers.get('X-API-Key')
                if api_key:
                    temp_session_id = create_temp_session_id(request, api_key)
                    user_session = self.get_user_session(temp_session_id, api_key)
                    if user_session:
                        self.switch_app(app_name, session_id=temp_session_id)
                else:
                    self.switch_app(app_name)
    
    def get_base_data_dir_for_request(self, request):
        """
        根据请求的 URL 获取正确的 base_data_dir（不修改全局变量）
        这个方法用于在需要时获取正确的数据目录，而不影响全局状态
        
        Args:
            request: Flask request 对象
        
        Returns:
            正确的 base_data_dir 路径
        """
        app_name = get_app_name_from_url(request)
        temp_app_manager = AppManager(app_name=app_name)
        config_file = "config/config.txt"  # default
        if temp_app_manager.is_app_mode():
            app_config_path = temp_app_manager.get_config_path()
            if app_config_path:
                config_file = app_config_path
        
        data_dir = get_gui_default_data_directory(config_file)
        if not data_dir:
            data_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        return data_dir
    
    def get_user_app_manager(self, session_id: Optional[str] = None) -> AppManager:
        """
        根据session_id获取用户专属的AppManager实例
        
        Args:
            session_id: 会话ID，如果为None则返回全局默认AppManager
        
        Returns:
            AppManager实例
        """
        if session_id and session_id in self.user_sessions:
            # 直接返回用户 session 中存储的 AppManager 实例
            # 简单高效，避免重复创建对象
            return self.user_sessions[session_id].app_manager
        
        # 返回全局默认AppManager（向后兼容）
        return self.app_manager
    
    def get_base_data_dir_for_session(self, session_id: Optional[str] = None) -> str:
        """
        根据session_id获取正确的 base_data_dir（用于 socket 事件）
        
        Args:
            session_id: 会话ID，如果为None则返回全局默认 base_data_dir
        
        Returns:
            正确的 base_data_dir 路径
        """
        if session_id and session_id in self.user_sessions:
            user_session = self.user_sessions[session_id]
            app_manager = user_session.app_manager
            
            # 使用用户 session 的 AppManager 来获取配置路径
            config_file = "config/config.txt"  # default
            if app_manager.is_app_mode():
                app_config_path = app_manager.get_config_path()
                if app_config_path:
                    config_file = app_config_path
            
            data_dir = get_gui_default_data_directory(config_file)
            if data_dir:
                return data_dir
        
        # Fallback to global base_data_dir
        return self.base_data_dir

    
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
    

    
    def get_output_directories(self, user_session, base_data_dir=None):
        """
        Get all directories containing workspace subdirectory for specific user (optimized)
        
        Args:
            user_session: User session object
            base_data_dir: Optional base data directory path. If None, uses self.base_data_dir (for backward compatibility)
        """
        result = []
        
        # Use provided base_data_dir or fall back to instance variable
        if base_data_dir is None:
            base_data_dir = self.base_data_dir
        
        # Get user's directory
        user_output_dir = user_session.get_user_directory(base_data_dir)
        os.makedirs(user_output_dir, exist_ok=True)
        
        try:
            # 优化：使用os.scandir代替os.listdir，减少系统调用
            with os.scandir(user_output_dir) as entries:
                for entry in entries:
                    if not entry.is_dir():
                        continue
                    
                    item = entry.name
                    item_path = entry.path
                    
                    # Check if it contains workspace subdirectory
                    workspace_path = os.path.join(item_path, 'workspace')
                    if not os.path.exists(workspace_path) or not os.path.isdir(workspace_path):
                        continue
                    
                    try:
                        # 优化：使用entry.stat()获取修改时间，避免额外的stat调用
                        stat = entry.stat()
                        
                        # 性能优化：移除目录大小计算（前端不需要显示目录大小，只显示文件大小）
                        # 这样可以节省大量时间，特别是对于包含大量文件的目录
                        
                        # 获取任务描述（已优化，带缓存）
                        task_description = self.get_task_description_from_manager_out(item_path)
                        
                        # 性能优化：加载完整目录结构（保留其他优化，但取消深度限制）
                        # 使用完整的递归加载，确保所有子文件夹的文件都能显示
                        files_structure = self.get_directory_structure(item_path, lazy_load=False)
                        
                        result.append({
                            'name': item,
                            'path': item_path,
                            'size': '0 B',  # 不再计算目录大小，节省时间
                            'files': files_structure,  # 延迟加载的目录结构
                            'is_current': item == user_session.current_output_dir,
                            'is_selected': item == user_session.selected_output_dir,
                            'is_last': item == user_session.last_output_dir,
                            'task_description': task_description,
                            'mtime': stat.st_mtime  # 保存修改时间用于排序
                        })
                    except (OSError, PermissionError):
                        continue
        except (OSError, PermissionError) as e:
            pass
        
        # Sort by modification time (使用已保存的mtime，避免重复stat调用)
        result.sort(key=lambda x: x.get('mtime', 0), reverse=True)
        return result
    
    def get_directory_size(self, directory):
        """Calculate directory size (optimized with caching)"""
        # 检查缓存
        cache_key = f"size_{directory}"
        with self._cache_lock:
            if cache_key in self._directory_cache:
                cached_data = self._directory_cache[cache_key]
                if time.time() - cached_data['timestamp'] < self._cache_timeout:
                    return cached_data['size']
        
        # 优化：使用更快的统计方法
        total_size = 0
        try:
            # 使用os.walk但优化：减少系统调用
            for dirpath, dirnames, filenames in os.walk(directory):
                # 跳过不需要遍历的目录（如code_index, __pycache__等）
                dirnames[:] = [d for d in dirnames if d not in {'code_index', '__pycache__', '.git', '.vscode', 'node_modules'}]
                
                for filename in filenames:
                    filepath = os.path.join(dirpath, filename)
                    try:
                        # 直接使用stat获取大小，避免额外的exists检查
                        stat_info = os.stat(filepath)
                        total_size += stat_info.st_size
                    except (OSError, IOError):
                        continue
        except (OSError, IOError):
            pass
        
        # 更新缓存
        with self._cache_lock:
            self._directory_cache[cache_key] = {
                'size': total_size,
                'timestamp': time.time()
            }
        
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
    
    def get_directory_structure(self, directory, max_depth=10, current_depth=0, base_dir=None, lazy_load=False):
        """Get directory structure (optimized but without depth limitation)
        
        Args:
            directory: 目录路径
            max_depth: 最大递归深度（保留参数以兼容，但实际不限制）
            current_depth: 当前深度
            base_dir: 基础目录路径
            lazy_load: 是否延迟加载（已取消，始终完整加载）
        """
        if current_depth > max_depth:
            return []
        
        # If first call, set base_dir to parent directory of current directory
        if base_dir is None:
            base_dir = os.path.dirname(directory)
        
        # 检查缓存（仅对顶层目录使用缓存）
        if current_depth == 0:
            cache_key = f"struct_{directory}"
            with self._cache_lock:
                if cache_key in self._directory_cache:
                    cached_data = self._directory_cache[cache_key]
                    # 检查目录修改时间
                    try:
                        dir_mtime = os.path.getmtime(directory)
                        # 🔧 修复：同时检查workspace子目录的修改时间，确保上传文件后缓存能正确失效
                        workspace_path = os.path.join(directory, 'workspace')
                        workspace_mtime = None
                        if os.path.exists(workspace_path) and os.path.isdir(workspace_path):
                            workspace_mtime = os.path.getmtime(workspace_path)
                        
                        # 如果父目录和workspace目录的修改时间都没有变化，且缓存未过期，则使用缓存
                        if dir_mtime == cached_data.get('dir_mtime'):
                            cached_workspace_mtime = cached_data.get('workspace_mtime')
                            if workspace_mtime is None or workspace_mtime == cached_workspace_mtime:
                                if time.time() - cached_data['timestamp'] < self._cache_timeout:
                                    return cached_data['structure']
                    except OSError:
                        pass
        
        items = []
        try:
            # 优化：使用os.scandir代替os.listdir + os.path.isdir，减少系统调用
            with os.scandir(directory) as entries:
                for entry in entries:
                    try:
                        item_path = entry.path
                        item_name = entry.name
                        
                        # Calculate relative path to base_dir
                        relative_path = os.path.relpath(item_path, base_dir)
                        # Convert Windows path separators to Unix style
                        relative_path = relative_path.replace('\\', '/')
                        
                        if entry.is_dir():
                            # 跳过不需要遍历的目录
                            if item_name.lower() in {'code_index', '__pycache__', '.git', '.vscode', 'node_modules'}:
                                continue
                            
                            # 取消深度限制：完整递归加载所有子目录
                            # 确保web_search_results/images等深层目录的文件都能显示
                            children = self.get_directory_structure(item_path, max_depth, current_depth + 1, base_dir, lazy_load)
                            items.append({
                                'name': item_name,
                                'type': 'directory',
                                'path': relative_path,
                                'children': children
                            })
                        elif entry.is_file():
                            # 过滤掉以tmp开头的PDF文件（临时文件）
                            if item_name.lower().startswith('tmp') and item_name.lower().endswith('.pdf'):
                                continue
                            
                            # 优化：使用entry.stat()而不是os.path.getsize，减少系统调用
                            try:
                                stat_info = entry.stat()
                                file_size = stat_info.st_size
                            except OSError:
                                file_size = 0
                            
                            items.append({
                                'name': item_name,
                                'type': 'file',
                                'path': relative_path,
                                'size': self.format_size(file_size)
                            })
                    except (OSError, PermissionError):
                        continue
        except (OSError, PermissionError):
            pass
        
        result = sorted(items, key=lambda x: (x['type'] == 'file', x['name']))
        
        # 更新缓存（仅对顶层目录）
        if current_depth == 0:
            try:
                dir_mtime = os.path.getmtime(directory)
                # 🔧 修复：同时保存workspace子目录的修改时间
                workspace_path = os.path.join(directory, 'workspace')
                workspace_mtime = None
                if os.path.exists(workspace_path) and os.path.isdir(workspace_path):
                    workspace_mtime = os.path.getmtime(workspace_path)
                
                with self._cache_lock:
                    self._directory_cache[cache_key] = {
                        'structure': result,
                        'dir_mtime': dir_mtime,
                        'workspace_mtime': workspace_mtime,
                        'timestamp': time.time()
                    }
            except OSError:
                pass
        
        return result
    
    def clear_directory_cache(self, directory):
        """清除指定目录的缓存
        
        Args:
            directory: 要清除缓存的目录路径
        """
        try:
            cache_key = f"struct_{directory}"
            with self._cache_lock:
                if cache_key in self._directory_cache:
                    del self._directory_cache[cache_key]
        except Exception:
            # 如果清除缓存失败，不影响主流程，静默处理
            pass
    
    def get_task_description_from_manager_out(self, directory_path):
        """从manager.out文件中读取任务描述（带缓存优化）
        
        Args:
            directory_path: 目录路径
            
        Returns:
            str: 任务描述（第一个user_requirement），如果没有找到则返回i18n翻译后的"未布置任务"
        """
        # 检查缓存
        cache_key = f"task_{directory_path}"
        with self._cache_lock:
            if cache_key in self._task_description_cache:
                cached_data = self._task_description_cache[cache_key]
                # 检查文件修改时间，如果文件未修改则使用缓存
                manager_out_path = os.path.join(directory_path, 'logs', 'manager.out')
                if os.path.exists(manager_out_path):
                    try:
                        file_mtime = os.path.getmtime(manager_out_path)
                        if file_mtime == cached_data['file_mtime']:
                            return cached_data['description']
                    except OSError:
                        pass
        
        # 获取i18n文本
        i18n = get_i18n_texts()
        no_task_text = i18n.get('no_task_assigned', '未布置任务')
        
        manager_out_path = os.path.join(directory_path, 'logs', 'manager.out')
        
        # 检查文件是否存在
        if not os.path.exists(manager_out_path):
            return no_task_text
        
        try:
            # 获取文件修改时间用于缓存验证
            file_mtime = os.path.getmtime(manager_out_path)
            
            # 优化：只读取文件的前几KB，通常任务描述在文件开头
            # 如果文件很大，只读取前50KB
            max_read_size = 50 * 1024  # 50KB
            with open(manager_out_path, 'r', encoding='utf-8', errors='ignore') as f:
                # 先读取部分内容
                content = f.read(max_read_size)
                lines = content.split('\n')
            
            # 从前往后查找"Received user requirement:"行（获取第一个，即最老的用户需求）
            task_description = None
            for line in lines:
                if "Received user requirement:" in line:
                    # 提取冒号后面的内容
                    parts = line.split("Received user requirement:", 1)
                    if len(parts) > 1:
                        task_description = parts[1].strip()
                        break
            
            # 如果找到了任务描述，返回它；否则返回默认值
            result = task_description if task_description else no_task_text
            
            # 更新缓存
            with self._cache_lock:
                self._task_description_cache[cache_key] = {
                    'description': result,
                    'file_mtime': file_mtime,
                    'timestamp': time.time()
                }
            
            return result
            
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
        self.current_app_name = None  # 用户当前选择的app名称（如'patent'），None表示使用默认模式（保留用于日志和调试）
        
        # 直接存储 AppManager 实例，简化逻辑并提升性能
        # 默认使用 None（默认模式），在 switch_app 时会更新
        self.app_manager = AppManager(app_name=None)
        
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
    
    def get_summarized_requirements(self, output_dir=None, language=None):
        """从manager.out文件中提取历史user requirements并汇总

        从指定的output目录（或当前工作目录）中读取manager.out文件，提取历史user requirements，
        按时间排序并返回最近的几个需求。

        Args:
            output_dir: 可选，指定的output目录路径。如果为None，则从当前工作目录读取。
            language: 可选，语言代码（'zh' 或 'en'）。如果为None，则使用配置文件中的语言。

        Returns:
            str: 汇总的历史需求，如果没有找到则返回None
        """
        # 确定要读取的目录
        target_dirs = []
        
        if output_dir:
            # 如果指定了目录，只从该目录读取
            if os.path.exists(output_dir):
                target_dirs = [output_dir]
        else:
            # 如果没有指定目录，尝试使用当前工作目录
            # 使用 session 特定的 base_data_dir
            session_base_data_dir = gui_instance.get_base_data_dir_for_session(self.session_id)
            current_dir = None
            if self.current_output_dir:
                user_base_dir = self.get_user_directory(session_base_data_dir)
                current_dir = os.path.join(user_base_dir, self.current_output_dir)
            elif self.selected_output_dir:
                user_base_dir = self.get_user_directory(session_base_data_dir)
                current_dir = os.path.join(user_base_dir, self.selected_output_dir)
            elif self.last_output_dir:
                user_base_dir = self.get_user_directory(session_base_data_dir)
                current_dir = os.path.join(user_base_dir, self.last_output_dir)
            
            if current_dir and os.path.exists(current_dir):
                target_dirs = [current_dir]

        if not target_dirs:
            return None

        # 从指定目录的manager.out文件中提取历史需求
        all_requirements = []
        for output_dir in target_dirs:
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
                                    # 获取文件的修改时间作为时间戳（更准确）
                                    try:
                                        file_mtime = os.path.getmtime(manager_out_path)
                                        timestamp = datetime.datetime.fromtimestamp(file_mtime).isoformat()
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

        # 按时间戳排序（最老的在前）
        all_requirements.sort(key=lambda x: x['timestamp'], reverse=False)

        # 取最近的5个需求
        recent_requirements = all_requirements[:5]

        # 生成汇总文本
        # 获取i18n文本，使用传入的语言参数或默认语言
        if language and language in ('zh', 'en'):
            i18n = I18N_TEXTS.get(language, I18N_TEXTS['en'])
        else:
            i18n = get_i18n_texts()
        history_summary = []
        total_count = len(recent_requirements)
        for idx, req in enumerate(recent_requirements):
            # 索引0是最老的，最后一个是最新的
            if idx == 0:
                label = f"1. ({i18n.get('oldest', '最老')})"
            elif idx == total_count - 1:
                label = f"{idx + 1}. ({i18n.get('newest', '最新')})"
            else:
                label = f"{idx + 1}."
            history_summary.append(f"{label} {req['requirement']}")

        return "\n".join(history_summary)

# Initialize GUI instance - app_name will be set from environment variable or command line
# This allows --app parameter to work even though gui_instance is created at module level
_app_name_from_env = os.environ.get('AGIA_APP_NAME', None)
gui_instance = AGIAgentGUI(app_name=_app_name_from_env)

def create_temp_session_id(request, api_key=None):
    """Create a temporary session ID for API calls with user isolation"""
    import hashlib
    api_key_hash = hashlib.sha256((api_key or "default").encode()).hexdigest()[:8]
    # Use consistent session ID based on IP and API key, not request ID
    return f"api_{request.remote_addr}_{api_key_hash}"

def get_session_id_from_request(request, api_key=None):
    """
    从请求中获取session_id
    
    优先级：
    1. WebSocket连接：使用request.sid（如果可用）
    2. Cookie中的session_id（如果存在）
    3. Header中的X-Session-ID（如果存在）
    4. 基于API key创建临时session_id（向后兼容）
    
    Returns:
        session_id字符串，如果无法获取则返回None
    """
    # 尝试从WebSocket获取（如果是在SocketIO上下文中）
    try:
        if hasattr(request, 'sid') and request.sid:
            return request.sid
    except:
        pass
    
    # 尝试从Cookie获取
    session_id = request.cookies.get('session_id')
    if session_id:
        return session_id
    
    # 尝试从Header获取
    session_id = request.headers.get('X-Session-ID')
    if session_id:
        return session_id
    
    # 向后兼容：如果没有session_id，基于API key创建临时session_id
    # 但返回None，让调用者决定是否创建临时session
    return None

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

# Reserved paths that should not be treated as app names
RESERVED_PATHS = ['terminal', 'register', 'agent-status-visualizer', 'api', 'static']

def get_app_name_from_url(request):
    """
    从请求的 URL 路径中提取 app_name
    
    优先级：
    1. Referer header 中的路径（如果用户从 /colordoc 访问 API）
    2. 当前请求路径（支持从 /colordoc/api/xxx 或 /api/xxx 中提取）
    3. 从 session 中获取（如果存在）
    
    Args:
        request: Flask request 对象
    
    Returns:
        app_name 字符串，如果不是 app 路径则返回 None
    """
    app_name = None
    
    def validate_app_name(potential_app_name):
        """验证 app_name 是否有效"""
        if not potential_app_name or potential_app_name in RESERVED_PATHS:
            return None
        # 验证 app 是否存在（包括隐藏应用，通过检查文件系统）
        try:
            apps_dir = os.path.join(gui_instance.app_manager.base_dir, 'apps')
            app_path = os.path.join(apps_dir, potential_app_name)
            app_json = os.path.join(app_path, 'app.json')
            if os.path.isdir(app_path) and os.path.exists(app_json):
                return potential_app_name
        except Exception:
            pass
        return None
    
    # 首先尝试从 Referer header 获取（用户从 /colordoc 页面访问 API）
    referer = request.headers.get('Referer') or request.headers.get('Referrer')
    if referer:
        try:
            from urllib.parse import urlparse
            parsed = urlparse(referer)
            # 🔧 修复：如果 Referer 是主平台（/），明确返回 None，不使用 session 中的 app_name
            if parsed.path == '/' or not parsed.path or parsed.path == '':
                # 主平台访问，明确返回 None
                return None
            path_parts = [p for p in parsed.path.split('/') if p]
            # 遍历路径的所有部分，找到第一个有效的 app_name
            for part in path_parts:
                validated = validate_app_name(part)
                if validated:
                    app_name = validated
                    break
        except Exception:
            pass
    
    # 如果从 Referer 没找到，尝试从当前路径获取
    if not app_name:
        try:
            current_path = request.path if hasattr(request, 'path') else '/'
            path_parts = [p for p in current_path.split('/') if p]
            # 遍历路径的所有部分，找到第一个有效的 app_name
            # 这样可以支持 /colordoc/api/xxx 这样的路径格式
            for part in path_parts:
                validated = validate_app_name(part)
                if validated:
                    app_name = validated
                    break
        except Exception:
            pass
    
    # 🔧 修复：如果当前路径是主平台（/）或API路径（/api/xxx），不应该从session中获取app_name
    # 这样可以确保访问主平台时使用默认配置，而不是之前访问的app配置
    is_main_platform = False
    try:
        current_path = request.path if hasattr(request, 'path') else '/'
        # 如果路径是 / 或 /api/xxx，说明是访问主平台或API，不应该从session获取
        if current_path == '/' or current_path.startswith('/api/'):
            is_main_platform = True
    except Exception:
        pass
    
    # 如果仍然没找到，且不是主平台访问，尝试从 session 中获取（如果存在）
    if not app_name and not is_main_platform:
        try:
            # 尝试从请求中获取 session_id
            api_key = request.args.get('api_key') or request.headers.get('X-API-Key')
            if api_key:
                temp_session_id = create_temp_session_id(request, api_key)
                if temp_session_id in gui_instance.user_sessions:
                    user_session = gui_instance.user_sessions[temp_session_id]
                    if hasattr(user_session, 'app_manager') and user_session.app_manager.is_app_mode():
                        # 使用 app_name 属性（目录名），而不是 get_app_name()（显示名称）
                        app_name = user_session.app_manager.app_name
        except Exception:
            pass
    
    return app_name

def render_index_page(app_name_param=None, session_id=None):
    """Helper function to render index page with specified app"""
    # Support language switching via URL parameter
    lang_param = request.args.get('lang')
    if lang_param and lang_param in ('zh', 'en'):
        current_lang = lang_param
    else:
        # 尝试从浏览器Accept-Language头检测语言
        accept_language = request.headers.get('Accept-Language', '')
        if accept_language:
            # 检查是否包含中文
            if 'zh' in accept_language.lower():
                current_lang = 'zh'
            else:
                current_lang = get_language()
        else:
            current_lang = get_language()
    
    # 确保i18n与current_lang保持一致
    # 如果current_lang是通过URL参数或浏览器Accept-Language设置的，应该使用对应的i18n文本
    i18n = I18N_TEXTS.get(current_lang, I18N_TEXTS['en'])
    
    mcp_servers = get_mcp_servers_config()
    
    # If app_name_param is provided and session_id exists, ensure user_session.app_manager is set
    if app_name_param and session_id and session_id in gui_instance.user_sessions:
        user_session = gui_instance.user_sessions[session_id]
        # 检查 app_manager.app_name 而不是 current_app_name，保持一致性
        if user_session.app_manager.app_name != app_name_param:
            # Switch app for this user session
            gui_instance.switch_app(app_name_param, session_id=session_id)
    
    # Get user-specific AppManager if session_id is provided
    # Otherwise use global AppManager (backward compatibility)
    user_app_manager = gui_instance.get_user_app_manager(session_id) if session_id else gui_instance.app_manager
    
    # Load GUI virtual terminal configuration
    # Use app-specific config file if available
    config_file = "config/config.txt"
    if user_app_manager.is_app_mode():
        # Get user_dir if session_id exists for user-specific config path
        user_dir = None
        if session_id and session_id in gui_instance.user_sessions:
            user_session = gui_instance.user_sessions[session_id]
            # 使用 session 特定的 base_data_dir
            session_base_data_dir = gui_instance.get_base_data_dir_for_session(session_id)
            user_dir = user_session.get_user_directory(session_base_data_dir)
        app_config_path = user_app_manager.get_config_path(user_dir=user_dir)
        if app_config_path:
            config_file = app_config_path
    
    config = load_config(config_file)
    gui_virtual_terminal = config.get('GUI_virtual_terminal', 'False').lower() == 'true'
    
    # Load GUI button display configurations
    gui_show_infinite_execute_button = config.get('GUI_show_infinite_execute_button', 'True').lower() == 'true'
    gui_show_multi_agent_button = config.get('GUI_show_multi_agent_button', 'True').lower() == 'true'
    gui_show_agent_view_button = config.get('GUI_show_agent_view_button', 'True').lower() == 'true'
    
    # Get app information for initial render (to avoid double display)
    app_name = user_app_manager.get_app_name()
    app_logo_path = user_app_manager.get_logo_path()
    app_logo_url = None
    if app_logo_path:
        project_root = user_app_manager.base_dir
        apps_dir = os.path.join(project_root, 'apps')
        if app_logo_path.startswith(apps_dir):
            rel_path = os.path.relpath(app_logo_path, apps_dir)
            rel_path = rel_path.replace('\\', '/')
            app_logo_url = f'/api/app-logo/{rel_path}'
        elif app_logo_path.startswith(project_root):
            rel_path = os.path.relpath(app_logo_path, project_root)
            rel_path = rel_path.replace('\\', '/')
            app_logo_url = f'/static/{rel_path}'
    
    is_app_mode = user_app_manager.is_app_mode()
    is_hidden = user_app_manager.is_hidden() if is_app_mode else False
    
    return render_template('index.html', 
                         i18n=i18n, 
                         lang=current_lang, 
                         mcp_servers=mcp_servers, 
                         gui_virtual_terminal=gui_virtual_terminal,
                         gui_show_infinite_execute_button=gui_show_infinite_execute_button,
                         gui_show_multi_agent_button=gui_show_multi_agent_button,
                         gui_show_agent_view_button=gui_show_agent_view_button,
                         app_name=app_name,
                         app_logo_url=app_logo_url,
                         is_app_mode=is_app_mode,
                         is_hidden=is_hidden)

@app.route('/')
def index():
    """Main page - resets to initial platform specified at startup"""
    # Try to get session_id from request
    api_key = request.args.get('api_key') or request.headers.get('X-API-Key')
    session_id = get_session_id_from_request(request, api_key)
    
    # If no session_id but we have api_key, create/get user session
    if not session_id and api_key:
        temp_session_id = create_temp_session_id(request, api_key)
        user_session = gui_instance.get_user_session(temp_session_id, api_key)
        if user_session:
            session_id = temp_session_id
    
    # Always reset to initial platform (or None for default) when accessing root path
    # This ensures base_data_dir is correctly updated
    if session_id:
        gui_instance.switch_app(gui_instance.initial_app_name, session_id=session_id)
    else:
        # No session, reset global app (backward compatibility)
        gui_instance.switch_app(gui_instance.initial_app_name)
    
    return render_index_page(session_id=session_id)

@app.route('/<app_name>')
def index_with_app(app_name):
    """Main page with app specified via path, e.g., /patent, /colordoc"""
    # Exclude reserved paths
    if app_name in RESERVED_PATHS:
        abort(404)
    
    # Try to get session_id from request
    api_key = request.args.get('api_key') or request.headers.get('X-API-Key')
    session_id = get_session_id_from_request(request, api_key)
    
    # If no session_id but we have api_key, create/get user session
    if not session_id and api_key:
        temp_session_id = create_temp_session_id(request, api_key)
        user_session = gui_instance.get_user_session(temp_session_id, api_key)
        if user_session:
            session_id = temp_session_id
    
    # Check if app exists (including hidden apps that can be accessed via URL)
    apps_dir = os.path.join(gui_instance.app_manager.base_dir, 'apps')
    app_path = os.path.join(apps_dir, app_name)
    app_json = os.path.join(app_path, 'app.json')
    
    # Validate app exists by checking directory and app.json file
    app_exists = os.path.isdir(app_path) and os.path.exists(app_json)
    
    if app_exists:
        # Switch to the specified platform for this user
        # IMPORTANT: Even if session_id doesn't exist yet (no WebSocket connection),
        # we should create/get a user session and set current_app_name so it's ready
        # when the WebSocket connection is established
        if not session_id:
            # Create/get user session even without WebSocket connection
            temp_session_id = create_temp_session_id(request, api_key)
            user_session = gui_instance.get_user_session(temp_session_id, api_key)
            if user_session:
                session_id = temp_session_id
        
        # Now switch app for this user session (if session_id exists)
        if session_id:
            gui_instance.switch_app(app_name, session_id=session_id)
        else:
            # No session could be created, switch global app (backward compatibility)
            gui_instance.switch_app(app_name)
        return render_index_page(app_name_param=app_name, session_id=session_id)
    else:
        # Invalid app name, redirect to root
        return redirect('/')

@app.route('/terminal')
def terminal():
    """Terminal page"""
    i18n = get_i18n_texts()
    current_lang = get_language()
    
    # Try to get session_id from request
    api_key = request.args.get('api_key') or request.headers.get('X-API-Key')
    session_id = get_session_id_from_request(request, api_key)
    
    # If no session_id but we have api_key, create/get user session
    if not session_id and api_key:
        temp_session_id = create_temp_session_id(request, api_key)
        user_session = gui_instance.get_user_session(temp_session_id, api_key)
        if user_session:
            session_id = temp_session_id
    
    # Get user-specific AppManager if session_id exists
    user_app_manager = gui_instance.get_user_app_manager(session_id) if session_id else gui_instance.app_manager
    
    # Load GUI virtual terminal configuration
    # Use app-specific config file if available
    config_file = "config/config.txt"
    if user_app_manager.is_app_mode():
        app_config_path = user_app_manager.get_config_path()
        if app_config_path:
            config_file = app_config_path
    
    config = load_config(config_file)
    gui_virtual_terminal = config.get('GUI_virtual_terminal', 'False').lower() == 'true'
    
    return render_template('terminal.html', i18n=i18n, lang=current_lang, gui_virtual_terminal=gui_virtual_terminal)

@app.route('/register')
def register():
    """User registration page"""
    # Determine current language:
    # 1) URL parameter ?lang=zh/en
    # 2) Fallback to config file language
    lang_param = request.args.get('lang')
    if lang_param in ('zh', 'en'):
        current_lang = lang_param
    else:
        current_lang = get_language()

    # Load i18n texts for the resolved language
    i18n = I18N_TEXTS.get(current_lang, I18N_TEXTS['en'])
    # 获取来源页面参数，用于返回时跳转到正确的页面
    from_page = request.args.get('from', '/')
    return render_template('register.html', i18n=i18n, lang=current_lang, from_page=from_page)

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
            # 创建用户目录和shared目录，并拷贝应用配置
            user_info = result['user_info']
            if user_info and not user_info.get('existing_user', False):
                # 只有新用户才创建shared目录
                try:
                    # 确定用户目录名称（与UserSession逻辑一致）
                    username = user_info.get("name", "")
                    if username.lower() == "guest" and user_info.get("is_guest", False):
                        user_dir_name = "guest"
                    else:
                        import re
                        safe_username = re.sub(r'[<>:"/\\|?*]', '_', username)
                        safe_username = safe_username.strip(' .')
                        user_dir_name = safe_username if safe_username else "user"
                    
                    # 使用请求特定的 base_data_dir，避免并发问题
                    request_base_data_dir = gui_instance.get_base_data_dir_for_request(request)
                    user_dir = os.path.join(request_base_data_dir, user_dir_name)
                    os.makedirs(user_dir, exist_ok=True)
                    
                    # 如果当前有激活的应用，拷贝应用配置到shared目录
                    if gui_instance.app_manager.is_app_mode():
                        gui_instance.app_manager.copy_app_to_shared(user_dir)
                except Exception as e:
                    # 如果创建shared目录失败，不影响注册流程
                    print(f"⚠️ Warning: Failed to create shared directory for user {username}: {e}")
            
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
    """Get output directory list (optimized with lazy loading)"""
    try:
        # Get API key from query parameters
        api_key = request.args.get('api_key')
        
        # Create a temporary session for API calls (since no socket connection)
        temp_session_id = create_temp_session_id(request, api_key)
        user_session = gui_instance.get_user_session(temp_session_id, api_key)
        
        if not user_session:
            return jsonify({'success': False, 'error': 'Authentication failed'}), 401
        
        # 确保根据 URL 切换正确的 app，以使用正确的 base_data_dir
        gui_instance.ensure_app_switched_for_request(request, temp_session_id)
        
        # 使用请求特定的 base_data_dir，避免并发问题
        request_base_data_dir = gui_instance.get_base_data_dir_for_request(request)
        
        dirs = gui_instance.get_output_directories(user_session, base_data_dir=request_base_data_dir)
        return jsonify({'success': True, 'directories': dirs})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/load-directory-structure', methods=['POST'])
def load_directory_structure():
    """Load directory structure on demand (for lazy loading optimization)"""
    try:
        data = request.get_json() or {}
        dir_name = data.get('dir_name', '')
        sub_path = data.get('sub_path', '')  # 子目录的相对路径
        
        # Get API key
        api_key = request.args.get('api_key') or request.headers.get('X-API-Key') or data.get('api_key')
        temp_session_id = create_temp_session_id(request, api_key)
        user_session = gui_instance.get_user_session(temp_session_id, api_key)
        
        if not user_session:
            return jsonify({'success': False, 'error': 'Authentication failed'}), 401
        
        # 确保根据 URL 切换正确的 app
        gui_instance.ensure_app_switched_for_request(request, temp_session_id)
        request_base_data_dir = gui_instance.get_base_data_dir_for_request(request)
        user_base_dir = user_session.get_user_directory(request_base_data_dir)
        
        # 构建完整路径
        if sub_path:
            full_path = os.path.join(user_base_dir, dir_name, sub_path)
        else:
            full_path = os.path.join(user_base_dir, dir_name)
        
        # 安全检查
        real_output_dir = os.path.realpath(user_base_dir)
        real_file_path = os.path.realpath(full_path)
        if not real_file_path.startswith(real_output_dir):
            return jsonify({'success': False, 'error': 'Access denied'}), 403
        
        if not os.path.exists(full_path) or not os.path.isdir(full_path):
            return jsonify({'success': False, 'error': 'Directory not found'}), 404
        
        # 加载目录结构（不延迟加载，完整加载）
        base_dir = os.path.dirname(full_path) if sub_path else os.path.dirname(user_base_dir)
        structure = gui_instance.get_directory_structure(full_path, lazy_load=False, base_dir=base_dir)
        
        return jsonify({'success': True, 'structure': structure})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/download/<path:dir_name>')
def download_directory(dir_name):
    """Download directory as zip file (excluding code_index directory)"""
    try:
        # Get API key from query parameters or headers
        api_key = request.args.get('api_key') or request.headers.get('X-API-Key')
        
        # Create a temporary session for API calls
        temp_session_id = create_temp_session_id(request, api_key)
        user_session = gui_instance.get_user_session(temp_session_id, api_key)
        
        # 确保根据 URL 切换正确的 app，以使用正确的 base_data_dir
        gui_instance.ensure_app_switched_for_request(request, temp_session_id)
        
        # 使用请求特定的 base_data_dir，避免并发问题
        request_base_data_dir = gui_instance.get_base_data_dir_for_request(request)
        user_base_dir = user_session.get_user_directory(request_base_data_dir)
        
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
        
        # 根据 URL 路径自动切换 app（如果从 /colordoc 或 /patent 访问，或从 / 访问需要重置）
        app_name = get_app_name_from_url(request)
        # 如果从 / 访问（app_name 为 None），也需要切换以重置到默认配置
        # 如果从 /colordoc 等访问（app_name 不为 None），切换到对应 app
        gui_instance.switch_app(app_name, session_id=temp_session_id)
        
        # 使用请求特定的 base_data_dir，避免并发问题
        request_base_data_dir = gui_instance.get_base_data_dir_for_request(request)
        user_base_dir = user_session.get_user_directory(request_base_data_dir)

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
                # 过滤掉以tmp开头的PDF文件（临时文件）
                if name.lower().startswith('tmp') and name.lower().endswith('.pdf'):
                    continue
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
        
        # 确保根据 URL 切换正确的 app，以使用正确的 base_data_dir
        gui_instance.ensure_app_switched_for_request(request, temp_session_id)
        
        # 使用请求特定的 base_data_dir，避免并发问题
        request_base_data_dir = gui_instance.get_base_data_dir_for_request(request)
        user_base_dir = user_session.get_user_directory(request_base_data_dir)
        
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
        
        # 确保根据 URL 切换正确的 app，以使用正确的 base_data_dir
        gui_instance.ensure_app_switched_for_request(request, temp_session_id)
        
        # 使用请求特定的 base_data_dir，避免并发问题
        request_base_data_dir = gui_instance.get_base_data_dir_for_request(request)
        user_base_dir = user_session.get_user_directory(request_base_data_dir)
        
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
                     '.makefile', '.cmake', '.gradle', '.properties', '.ini', '.cfg', '.conf', '.toml', '.mmd', '.out', '.v']:
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
                '.out': 'text',
                '.v': 'verilog'
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
        
        # 确保根据 URL 切换正确的 app，以使用正确的 base_data_dir
        gui_instance.ensure_app_switched_for_request(request, temp_session_id)
        
        # 使用请求特定的 base_data_dir，避免并发问题
        request_base_data_dir = gui_instance.get_base_data_dir_for_request(request)
        user_base_dir = user_session.get_user_directory(request_base_data_dir)
        
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
        
        # 确保根据 URL 切换正确的 app，以使用正确的 base_data_dir
        gui_instance.ensure_app_switched_for_request(request, temp_session_id)
        
        # 使用请求特定的 base_data_dir，避免并发问题
        request_base_data_dir = gui_instance.get_base_data_dir_for_request(request)
        user_base_dir = user_session.get_user_directory(request_base_data_dir)
        
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

@app.route('/api/user-guide')
def serve_user_guide():
    """Serve user guide PDF file"""
    try:
        # Get the project root directory (parent of GUI directory)
        project_root = os.path.dirname(app_dir)
        user_guide_path = os.path.join(project_root, 'md', 'user_guide.pdf')
        
        if not os.path.exists(user_guide_path) or not os.path.isfile(user_guide_path):
            abort(404)
        
        # Check if download parameter is set
        download = request.args.get('download', 'false').lower() == 'true'
        
        return send_file(user_guide_path, mimetype='application/pdf', as_attachment=download, download_name='user_guide.pdf' if download else None)
    except Exception as e:
        print(f"Error serving user guide: {str(e)}")
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
        
        # 确保根据 URL 切换正确的 app，以使用正确的 base_data_dir
        gui_instance.ensure_app_switched_for_request(request, temp_session_id)
        
        # 使用请求特定的 base_data_dir，避免并发问题
        request_base_data_dir = gui_instance.get_base_data_dir_for_request(request)
        user_base_dir = user_session.get_user_directory(request_base_data_dir)
        
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
        
        # 确保根据 URL 切换正确的 app，以使用正确的 base_data_dir
        gui_instance.ensure_app_switched_for_request(request, temp_session_id)
        
        # 使用请求特定的 base_data_dir，避免并发问题
        request_base_data_dir = gui_instance.get_base_data_dir_for_request(request)
        user_base_dir = user_session.get_user_directory(request_base_data_dir)
        
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
        
        # 确保根据 URL 切换正确的 app，以使用正确的 base_data_dir
        gui_instance.ensure_app_switched_for_request(request, temp_session_id)
        
        # 使用请求特定的 base_data_dir，避免并发问题
        request_base_data_dir = gui_instance.get_base_data_dir_for_request(request)
        user_base_dir = user_session.get_user_directory(request_base_data_dir)
        
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
        
        # 确保根据 URL 切换正确的 app，以使用正确的 base_data_dir
        gui_instance.ensure_app_switched_for_request(request, temp_session_id)
        
        # 使用请求特定的 base_data_dir，避免并发问题
        request_base_data_dir = gui_instance.get_base_data_dir_for_request(request)
        user_base_dir = user_session.get_user_directory(request_base_data_dir)
        
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
    app_name_from_client = None  # 从客户端获取的 app_name（从 URL 路径）
    if auth:
        api_key = auth.get('api_key')
        # Convert empty string to None for guest access
        if api_key == "":
            api_key = None
        client_session_id = auth.get('client_session_id')
        app_name_from_client = auth.get('app_name')  # 前端传递的 app_name（从 URL 路径获取）
    
    
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
        # 🔧 修复：检查新的 API key 是否与旧的 API key 不同
        # 如果不同，说明用户切换了账户（例如从访客切换到注册账户），应该销毁旧会话并创建新会话
        old_api_key = recovered_session.api_key
        # 统一处理：将空字符串转换为 None 以便比较
        old_api_key_normalized = None if (old_api_key == "" or old_api_key is None) else old_api_key
        new_api_key_normalized = None if (api_key == "" or api_key is None) else api_key
        
        if old_api_key_normalized != new_api_key_normalized:
            # API key 不同，销毁旧会话并创建新会话
            # 销毁旧的认证会话
            gui_instance.auth_manager.destroy_session(old_socket_sid)
            # 不恢复旧会话，而是创建新会话
            user_session = gui_instance.get_user_session(session_id, api_key)
        else:
            # API key 相同，使用恢复的会话
            user_session = recovered_session
            # 更新session_id到新的socket session_id
            user_session.session_id = session_id
            # 保存client_session_id
            if client_session_id:
                user_session.client_session_id = client_session_id
            # 🔧 修复：无论是否有 app_name_from_client，都要根据当前tab的URL路径设置正确的app_manager
            # 这样可以确保每个tab使用正确的app配置，即使它们共享同一个user_session对象
            # 如果 app_name_from_client 为 None，表示主平台，应该设置为默认模式
            user_session.app_manager = AppManager(app_name=app_name_from_client)
            user_session.current_app_name = app_name_from_client
            gui_instance.user_sessions[session_id] = user_session
            # 重新创建认证会话 - 使用保存的api_key
            gui_instance.auth_manager.create_session(user_session.api_key, session_id)
        
        # 保存client_session_id（无论是否恢复会话）
        if user_session and client_session_id:
            user_session.client_session_id = client_session_id
        # 🔧 修复：无论是否有 app_name_from_client，都要根据当前tab的URL路径设置正确的app_manager
        # 这样可以确保每个tab使用正确的app配置，即使它们共享同一个user_session对象
        if user_session:
            user_session.app_manager = AppManager(app_name=app_name_from_client)
            user_session.current_app_name = app_name_from_client
    else:
        user_session = gui_instance.get_user_session(session_id, api_key)
        # 保存client_session_id
        if user_session and client_session_id:
            user_session.client_session_id = client_session_id
        # 🔧 修复：无论是否有 app_name_from_client，都要根据当前tab的URL路径设置正确的app_manager
        # 这样可以确保每个tab使用正确的app配置，即使它们共享同一个user_session对象
        if user_session:
            user_session.app_manager = AppManager(app_name=app_name_from_client)
            user_session.current_app_name = app_name_from_client
    
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
    # 使用 session 特定的 base_data_dir
    session_base_data_dir = gui_instance.get_base_data_dir_for_session(session_id)
    user_dir = user_session.get_user_directory(session_base_data_dir)
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

    # Remove connection from concurrency manager
    gui_instance.concurrency_manager.remove_connection()

    if session_id in gui_instance.user_sessions:
        user_session = gui_instance.user_sessions[session_id]
        api_key = user_session.user_info.get('api_key', '')
        client_session_id = getattr(user_session, 'client_session_id', None)
        
        # 判断等待时间：有任务运行时等待更长时间，空闲时等待较短时间
        has_running_task = user_session.current_process and user_session.current_process.is_alive()
        grace_period = RECONNECT_GRACE_PERIOD if has_running_task else 30  # 空闲时等待30秒
        

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
    
    # 🔧 修复：对于 WebSocket 请求，优先使用 session 中的 app_manager.app_name
    # 因为 WebSocket 请求的路径可能是 /socket.io/...，无法从路径识别 app_name
    # 而 session 中的 app_manager 已经在连接时根据 URL 路径正确设置了
    session_app_name = user_session.app_manager.app_name if user_session.app_manager else None
    if session_app_name:
        # 使用 session 中的 app_name 来获取 base_data_dir
        temp_app_manager = AppManager(app_name=session_app_name)
        config_file = "config/config.txt"  # default
        if temp_app_manager.is_app_mode():
            app_config_path = temp_app_manager.get_config_path()
            if app_config_path:
                config_file = app_config_path
        request_base_data_dir = get_gui_default_data_directory(config_file)
        if not request_base_data_dir:
            request_base_data_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    else:
        # 如果没有 session app_name，尝试从请求 URL 获取（用于 HTTP 请求）
        gui_instance.ensure_app_switched_for_request(request, session_id)
        request_base_data_dir = gui_instance.get_base_data_dir_for_request(request)
    
    # Get user's base directory using request-specific base_data_dir
    user_base_dir = user_session.get_user_directory(request_base_data_dir)
    
    # Determine output directory first (needed for loading history from correct directory)
    if task_type == 'new':
        # New task: create new output directory
        # For new tasks, we'll create the directory in execute_agia_task_process_target
        # but we need to pass the correct base_data_dir via app_name
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
    
    # Generate detailed requirement with conversation history for continuing tasks
    # 🔧 Fix: Only load history from current working directory (out_dir)
    # 🔧 Fix: Remove conversation_history check - first round should also load history from manager.out
    detailed_requirement = None
    if task_type in ['continue', 'selected'] and out_dir:
        # For continue/selected tasks, include conversation context from current directory only
        # This will read from manager.out file if it exists, even for first round after reconnection
        # 🔧 Fix: Pass user language to get_summarized_requirements for correct i18n
        history_context = user_session.get_summarized_requirements(output_dir=out_dir, language=user_lang)
        if history_context:
            # 🔧 Fix: adjust prompt order - current first
            detailed_requirement = f"{user_requirement}\n\nPrevious conversation context:\n{history_context}"
    
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
        # Get app_name and user_dir for app-specific configuration
        # 🔧 修复：优先使用 session 中的 app_manager.app_name，确保与 base_data_dir 一致
        # 因为 WebSocket 请求无法从路径识别 app_name，而 session 中的 app_manager 已经在连接时正确设置了
        app_name = user_session.app_manager.app_name if user_session.app_manager else None
        
        # 如果 session 中没有，尝试从请求数据中获取（前端传递）
        if not app_name:
            app_name = data.get('app_name') or gui_config.get('app_name')
        
        # 如果还是没有，尝试从连接的 URL 获取（WebSocket 连接时可能传递了）
        if not app_name:
            # 尝试从 request 的 headers 或环境变量中获取
            # 注意：WebSocket 连接可能没有 Referer header，所以优先使用前端传递的值
            try:
                # 检查是否有 origin header，可能包含路径信息
                origin = request.headers.get('Origin') or request.headers.get('Referer')
                if origin:
                    from urllib.parse import urlparse
                    parsed = urlparse(origin)
                    path_parts = [p for p in parsed.path.split('/') if p]
                    if path_parts:
                        potential_app_name = path_parts[0]
                        if potential_app_name not in RESERVED_PATHS:
                            available_apps = gui_instance.app_manager.list_available_apps()
                            app_names = [app['name'] for app in available_apps]
                            if potential_app_name in app_names:
                                app_name = potential_app_name
            except Exception:
                pass
        
        # Use request-specific base_data_dir for user_dir
        user_dir = user_session.get_user_directory(request_base_data_dir)
        
        user_session.current_process = multiprocessing.Process(
            target=execute_agia_task_process_target,
            args=(user_requirement, user_session.output_queue, user_session.input_queue, out_dir, continue_mode, plan_mode, gui_config, session_id, detailed_requirement, user_id, attached_files, app_name, user_dir)
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
    # 使用 session 特定的 base_data_dir
    session_base_data_dir = gui_instance.get_base_data_dir_for_session(session_id)
    cwd = user_session.get_terminal_cwd(session_base_data_dir, force_recalculate=True)
    
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
        # 使用 session 特定的 base_data_dir
        session_base_data_dir = gui_instance.get_base_data_dir_for_session(session_id)
        cwd = user_session.get_terminal_cwd(session_base_data_dir)
        
        # 确保cwd是绝对路径
        if cwd:
            cwd = os.path.abspath(cwd)
        
        # 如果命令中包含cd到workspace的路径，提前更新cwd为workspace目录
        # 这样可以确保subprocess在正确的目录执行，避免命令中的cd失败
        # 同时可以简化命令，移除不必要的cd到workspace的部分
        if '&&' in command and 'workspace' in command:
            # 检查是否是cd命令链，且目标目录包含workspace
            cd_match = re.search(r'cd\s+["\']?([^"\']+)["\']?\s+&&', command)
            if cd_match:
                target_dir = cd_match.group(1)
                # 检查是否是output_xxx/workspace格式
                if target_dir.startswith('output_') and '/workspace' in target_dir:
                    # 提取output_xxx部分
                    output_dir = target_dir.split('/')[0]
                    user_dir = user_session.get_user_directory(session_base_data_dir)
                    workspace_dir = os.path.join(user_dir, output_dir, 'workspace')
                    if os.path.exists(workspace_dir) and os.path.isdir(workspace_dir):
                        # 更新cwd为workspace目录，这样subprocess会在正确的目录执行
                        workspace_abs = os.path.abspath(workspace_dir)
                        # 检查当前cwd是否是用户基础目录
                        if os.path.basename(cwd) != 'workspace' and cwd == user_dir:
                            # 当前cwd是用户基础目录，更新为workspace目录
                            cwd = workspace_abs
                            user_session.set_terminal_cwd(cwd)
                            # 从命令中移除cd到workspace的部分，只保留后续命令
                            # 例如：cd "output_xxx/workspace" && python script.py -> python script.py
                            # 或者：cd "output_xxx/workspace" && cd "subdir" && python script.py -> cd "subdir" && python script.py
                            command_parts = command.split('&&')
                            if len(command_parts) > 1:
                                # 移除第一个cd命令（cd到workspace的部分）
                                remaining_parts = command_parts[1:]
                                command = ' && '.join(part.strip() for part in remaining_parts)
                                # 更新cmd_lower用于后续处理
                                cmd_lower = command.strip().lower()
                elif target_dir == 'workspace' or target_dir.endswith('/workspace'):
                    # cd "workspace" 或 cd "xxx/workspace"格式
                    user_dir = user_session.get_user_directory(session_base_data_dir)
                    # 尝试找到workspace目录
                    workspace_dir = os.path.join(user_dir, 'workspace')
                    if not os.path.exists(workspace_dir):
                        # 尝试在output_xxx目录下找workspace
                        import glob
                        output_dirs = glob.glob(os.path.join(user_dir, 'output_*'))
                        for output_dir in output_dirs:
                            potential_workspace = os.path.join(output_dir, 'workspace')
                            if os.path.exists(potential_workspace) and os.path.isdir(potential_workspace):
                                workspace_dir = potential_workspace
                                break
                    if os.path.exists(workspace_dir) and os.path.isdir(workspace_dir):
                        workspace_abs = os.path.abspath(workspace_dir)
                        if os.path.basename(cwd) != 'workspace' and cwd == user_dir:
                            cwd = workspace_abs
                            user_session.set_terminal_cwd(cwd)
                            # 从命令中移除cd到workspace的部分
                            command_parts = command.split('&&')
                            if len(command_parts) > 1:
                                remaining_parts = command_parts[1:]
                                command = ' && '.join(part.strip() for part in remaining_parts)
                                cmd_lower = command.strip().lower()
        
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
            # macOS使用zsh，Linux使用bash
            if platform.system() == 'Darwin':  # macOS
                executable = '/bin/zsh'
            else:  # Linux
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
                        
                        # 检查目标目录是否与当前目录相同
                        if os.path.abspath(new_cwd) == os.path.abspath(cwd):
                            # 目标目录与当前目录相同，不需要cd，直接执行后续命令
                            # 提取cd命令之后的部分
                            after_cd = command.split('&&', 1)
                            if len(after_cd) > 1:
                                # 有后续命令，只执行后续命令
                                full_command = after_cd[1].strip()
                            else:
                                # 只有cd命令，执行cd .（无操作但保持一致性）
                                full_command = 'cd .'
                            # cwd保持不变
                        elif user_session.set_terminal_cwd(new_cwd):
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
                        
                        # 检查目标目录是否与当前目录相同
                        if os.path.abspath(new_cwd) == os.path.abspath(cwd):
                            # 目标目录与当前目录相同，执行cd .（无操作但保持一致性）
                            full_command = 'cd .'
                            # cwd保持不变
                        elif user_session.set_terminal_cwd(new_cwd):
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
        
        # 在执行命令前，输出命令本身（仅用于自动执行的命令，用户手动输入的命令已经在终端中显示）
        # 检查是否是自动执行的命令（通过检查命令是否包含引号中的脚本名，这通常是一键运行按钮触发的）
        # 对于用户手动输入的命令，命令已经在终端中显示了，不需要重复输出
        # 这里我们只在命令看起来像是自动执行的情况下输出（包含引号的脚本执行命令）
        # 但实际上，用户手动输入的命令已经在终端中显示了，所以这里不输出命令
        # 如果需要显示命令，可以在前端处理，但通常不需要，因为用户输入时已经显示了
        
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
                    
                    # 命令执行完成后，更新提示符（无论是否成功）
                    # 使用 session 特定的 base_data_dir
                    session_base_data_dir = gui_instance.get_base_data_dir_for_session(session_id)
                    current_dir = user_session.get_terminal_cwd(session_base_data_dir)
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
    # 使用 session 特定的 base_data_dir
    session_base_data_dir = gui_instance.get_base_data_dir_for_session(session_id)
    cwd = user_session.get_terminal_cwd(session_base_data_dir)
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
    
    # 标识是否是目录子项补全（需要追加而不是替换）
    is_dir_completion = False
    
    # 如果是路径补全（包含路径分隔符）
    if '/' in last_part or '\\' in last_part:
        # 路径补全
        # 首先检查输入的路径本身是否是一个完整的目录（类似Linux的tab补全行为）
        test_path = last_part.rstrip('/\\')  # 移除末尾的分隔符
        if not os.path.isabs(test_path):
            test_path = os.path.join(cwd, test_path)
        test_path = os.path.normpath(test_path)
        
        # 如果输入路径本身是一个目录，补全其子项
        if os.path.isdir(test_path):
            # 输入路径是完整目录，补全其子项
            is_dir_completion = True
            try:
                matches = glob.glob(os.path.join(test_path, '*'))
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
            # 正常路径补全：提取目录部分和文件名部分
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
    
    emit('terminal_autocomplete_result', {'completions': completions, 'is_dir_completion': is_dir_completion}, room=session_id)

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
            # 使用 session 特定的 base_data_dir
            session_base_data_dir = gui_instance.get_base_data_dir_for_session(session_id)
            user_base_dir = user_session.get_user_directory(session_base_data_dir)
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
        # 使用 session 特定的 base_data_dir
        session_base_data_dir = gui_instance.get_base_data_dir_for_session(session_id)
        user_base_dir = user_session.get_user_directory(session_base_data_dir)
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
        # 使用 session 特定的 base_data_dir
        session_base_data_dir = gui_instance.get_base_data_dir_for_session(session_id)
        user_base_dir = user_session.get_user_directory(session_base_data_dir)
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
        # 使用 session 特定的 base_data_dir
        session_base_data_dir = gui_instance.get_base_data_dir_for_session(session_id)
        user_base_dir = user_session.get_user_directory(session_base_data_dir)
        
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
        
        # 确保根据 URL 切换正确的 app，以使用正确的 base_data_dir
        gui_instance.ensure_app_switched_for_request(request, temp_session_id)
        
        # 使用请求特定的 base_data_dir，避免并发问题
        request_base_data_dir = gui_instance.get_base_data_dir_for_request(request)
        
        # Use existing method to get directory list for this user with request-specific base_data_dir
        directories = gui_instance.get_output_directories(user_session, base_data_dir=request_base_data_dir)
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
        
        # 确保根据 URL 切换正确的 app，以使用正确的 base_data_dir
        gui_instance.ensure_app_switched_for_request(request, temp_session_id)
        
        # 使用请求特定的 base_data_dir，避免并发问题
        request_base_data_dir = gui_instance.get_base_data_dir_for_request(request)
        user_base_dir = user_session.get_user_directory(request_base_data_dir)
        
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

@app.route('/api/file-snapshots/<path:file_path>', methods=['GET'])
def get_file_snapshots(file_path):
    """Get list of snapshot versions for a file"""
    try:
        # Get API key from query parameters or headers
        api_key = request.args.get('api_key') or request.headers.get('X-API-Key')
        
        # Create a temporary session for API calls
        temp_session_id = create_temp_session_id(request, api_key)
        user_session = gui_instance.get_user_session(temp_session_id, api_key)
        
        # 确保根据 URL 切换正确的 app，以使用正确的 base_data_dir
        gui_instance.ensure_app_switched_for_request(request, temp_session_id)
        
        # 使用请求特定的 base_data_dir，避免并发问题
        request_base_data_dir = gui_instance.get_base_data_dir_for_request(request)
        user_base_dir = user_session.get_user_directory(request_base_data_dir)
        
        # URL decode the file path
        import urllib.parse
        file_path = urllib.parse.unquote(file_path)
        
        # Security check: normalize path and prevent path traversal
        normalized_file_path = os.path.normpath(file_path)
        if '..' in normalized_file_path or normalized_file_path.startswith('/'):
            return jsonify({'success': False, 'error': 'Access denied: Invalid file path'}), 403
        
        # Full path to the file
        full_file_path = os.path.join(user_base_dir, normalized_file_path)
        
        # Security check: ensure path is within user's output directory
        real_output_dir = os.path.realpath(user_base_dir)
        real_file_path = os.path.realpath(full_file_path)
        if not real_file_path.startswith(real_output_dir):
            return jsonify({'success': False, 'error': 'Access denied: Invalid file path'}), 403
        
        # Check if file exists
        if not os.path.exists(full_file_path) or not os.path.isfile(full_file_path):
            return jsonify({'success': False, 'error': 'File not found'}), 404
        
        # Find workspace directory (could be in output_xxx/workspace or directly in user_base_dir/workspace)
        workspace_dir = None
        # Try to find workspace directory by checking if file_path contains workspace
        path_parts = normalized_file_path.split(os.sep)
        if 'workspace' in path_parts:
            workspace_idx = path_parts.index('workspace')
            workspace_path_parts = path_parts[:workspace_idx + 1]
            potential_workspace = os.path.join(user_base_dir, *workspace_path_parts)
            if os.path.exists(potential_workspace) and os.path.isdir(potential_workspace):
                workspace_dir = potential_workspace
        
        # If not found, try common locations
        if not workspace_dir:
            # Try output_xxx/workspace pattern
            for item in os.listdir(user_base_dir):
                item_path = os.path.join(user_base_dir, item)
                if os.path.isdir(item_path) and item.startswith('output_'):
                    potential_workspace = os.path.join(item_path, 'workspace')
                    if os.path.exists(potential_workspace) and os.path.isdir(potential_workspace):
                        # Check if file is under this workspace
                        rel_path_from_workspace = os.path.relpath(full_file_path, potential_workspace)
                        if not rel_path_from_workspace.startswith('..'):
                            workspace_dir = potential_workspace
                            break
        
        if not workspace_dir:
            return jsonify({'success': True, 'snapshots': []})
        
        # Get snapshot directory (parent of workspace)
        parent_dir = os.path.dirname(workspace_dir)
        snapshot_base_dir = os.path.join(parent_dir, 'file_snapshot')
        
        if not os.path.exists(snapshot_base_dir):
            return jsonify({'success': True, 'snapshots': []})
        
        # Get relative path from workspace
        rel_path_from_workspace = os.path.relpath(full_file_path, workspace_dir)
        file_dir = os.path.dirname(rel_path_from_workspace)
        file_name = os.path.basename(rel_path_from_workspace)
        
        # Split filename and extension
        name_parts = file_name.rsplit('.', 1)
        if len(name_parts) == 2:
            base_name, extension = name_parts
            extension = '.' + extension
        else:
            base_name = file_name
            extension = ''
        
        # Get snapshot directory for this file
        snapshot_file_dir = os.path.join(snapshot_base_dir, file_dir) if file_dir else snapshot_base_dir
        
        if not os.path.exists(snapshot_file_dir):
            return jsonify({'success': True, 'snapshots': []})
        
        # Find all matching snapshot files
        snapshots = []
        import re
        # Pattern: base_name_XXX.extension where XXX is 3 digits
        pattern = re.compile(rf'^{re.escape(base_name)}_(\d{{3}}){re.escape(extension)}$')
        
        for filename in os.listdir(snapshot_file_dir):
            match = pattern.match(filename)
            if match:
                snapshot_path = os.path.join(snapshot_file_dir, filename)
                if os.path.isfile(snapshot_path):
                    snapshot_id = int(match.group(1))
                    mtime = os.path.getmtime(snapshot_path)
                    snapshots.append({
                        'filename': filename,
                        'snapshot_id': snapshot_id,
                        'modified_time': mtime,
                        'modified_time_str': datetime.datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M:%S')
                    })
        
        # Sort by modified time (newest first)
        snapshots.sort(key=lambda x: x['modified_time'], reverse=True)
        
        return jsonify({'success': True, 'snapshots': snapshots})
        
    except Exception as e:
        import traceback
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500

@app.route('/api/restore-file', methods=['POST'])
def restore_file():
    """Restore a file from snapshot"""
    try:
        data = request.get_json() or {}
        file_path = data.get('file_path')
        snapshot_filename = data.get('snapshot_filename')
        
        if not file_path or not snapshot_filename:
            return jsonify({'success': False, 'error': 'Missing file_path or snapshot_filename'}), 400
        
        # Get API key from query parameters or headers
        api_key = request.args.get('api_key') or request.headers.get('X-API-Key') or data.get('api_key')
        
        # Create a temporary session for API calls
        temp_session_id = create_temp_session_id(request, api_key)
        user_session = gui_instance.get_user_session(temp_session_id, api_key)
        
        # 确保根据 URL 切换正确的 app，以使用正确的 base_data_dir
        gui_instance.ensure_app_switched_for_request(request, temp_session_id)
        
        # 使用请求特定的 base_data_dir，避免并发问题
        request_base_data_dir = gui_instance.get_base_data_dir_for_request(request)
        user_base_dir = user_session.get_user_directory(request_base_data_dir)
        
        # URL decode the file path
        import urllib.parse
        file_path = urllib.parse.unquote(file_path)
        
        # Security check: normalize path and prevent path traversal
        normalized_file_path = os.path.normpath(file_path)
        if '..' in normalized_file_path or normalized_file_path.startswith('/'):
            return jsonify({'success': False, 'error': 'Access denied: Invalid file path'}), 403
        
        # Full path to the file
        full_file_path = os.path.join(user_base_dir, normalized_file_path)
        
        # Security check: ensure path is within user's output directory
        real_output_dir = os.path.realpath(user_base_dir)
        real_file_path = os.path.realpath(full_file_path)
        if not real_file_path.startswith(real_output_dir):
            return jsonify({'success': False, 'error': 'Access denied: Invalid file path'}), 403
        
        # Check if file exists
        if not os.path.exists(full_file_path) or not os.path.isfile(full_file_path):
            return jsonify({'success': False, 'error': 'File not found'}), 404
        
        # Find workspace directory (same logic as get_file_snapshots)
        workspace_dir = None
        path_parts = normalized_file_path.split(os.sep)
        if 'workspace' in path_parts:
            workspace_idx = path_parts.index('workspace')
            workspace_path_parts = path_parts[:workspace_idx + 1]
            potential_workspace = os.path.join(user_base_dir, *workspace_path_parts)
            if os.path.exists(potential_workspace) and os.path.isdir(potential_workspace):
                workspace_dir = potential_workspace
        
        if not workspace_dir:
            for item in os.listdir(user_base_dir):
                item_path = os.path.join(user_base_dir, item)
                if os.path.isdir(item_path) and item.startswith('output_'):
                    potential_workspace = os.path.join(item_path, 'workspace')
                    if os.path.exists(potential_workspace) and os.path.isdir(potential_workspace):
                        rel_path_from_workspace = os.path.relpath(full_file_path, potential_workspace)
                        if not rel_path_from_workspace.startswith('..'):
                            workspace_dir = potential_workspace
                            break
        
        if not workspace_dir:
            return jsonify({'success': False, 'error': 'Workspace directory not found'}), 404
        
        # Get snapshot directory
        parent_dir = os.path.dirname(workspace_dir)
        snapshot_base_dir = os.path.join(parent_dir, 'file_snapshot')
        
        # Get relative path from workspace
        rel_path_from_workspace = os.path.relpath(full_file_path, workspace_dir)
        file_dir = os.path.dirname(rel_path_from_workspace)
        file_name = os.path.basename(rel_path_from_workspace)
        
        # Get snapshot file path
        snapshot_file_dir = os.path.join(snapshot_base_dir, file_dir) if file_dir else snapshot_base_dir
        snapshot_path = os.path.join(snapshot_file_dir, snapshot_filename)
        
        # Security check: ensure snapshot is within snapshot directory
        real_snapshot_base = os.path.realpath(snapshot_base_dir)
        real_snapshot_path = os.path.realpath(snapshot_path)
        if not real_snapshot_path.startswith(real_snapshot_base):
            return jsonify({'success': False, 'error': 'Access denied: Invalid snapshot path'}), 403
        
        if not os.path.exists(snapshot_path) or not os.path.isfile(snapshot_path):
            return jsonify({'success': False, 'error': 'Snapshot file not found'}), 404
        
        # Read current file content
        with open(full_file_path, 'r', encoding='utf-8') as f:
            current_content = f.read()
        
        # Create snapshot of current file before restoring
        # Split filename and extension
        name_parts = file_name.rsplit('.', 1)
        if len(name_parts) == 2:
            base_name, extension = name_parts
            extension = '.' + extension
        else:
            base_name = file_name
            extension = ''
        
        # Ensure snapshot directory exists
        os.makedirs(snapshot_file_dir, exist_ok=True)
        
        # Find the next available snapshot ID
        snapshot_id = 0
        while True:
            new_snapshot_filename = f"{base_name}_{snapshot_id:03d}{extension}"
            new_snapshot_path = os.path.join(snapshot_file_dir, new_snapshot_filename)
            
            if not os.path.exists(new_snapshot_path):
                break
                
            snapshot_id += 1
            
            # Safety check
            if snapshot_id > 999:
                return jsonify({'success': False, 'error': 'Too many snapshots'}), 500
        
        # Save current file as snapshot
        with open(new_snapshot_path, 'w', encoding='utf-8') as f:
            f.write(current_content)
        
        # Read snapshot content
        with open(snapshot_path, 'r', encoding='utf-8') as f:
            snapshot_content = f.read()
        
        # Restore file
        with open(full_file_path, 'w', encoding='utf-8') as f:
            f.write(snapshot_content)
        
        return jsonify({
            'success': True,
            'message': 'File restored successfully',
            'new_snapshot': new_snapshot_filename
        })
        
    except Exception as e:
        import traceback
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
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
        
        # 确保根据 URL 切换正确的 app，以使用正确的 base_data_dir
        gui_instance.ensure_app_switched_for_request(request, temp_session_id)
        
        # 使用请求特定的 base_data_dir，避免并发问题
        request_base_data_dir = gui_instance.get_base_data_dir_for_request(request)
        user_base_dir = user_session.get_user_directory(request_base_data_dir)
        
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
    # 使用请求特定的 base_data_dir，避免并发问题
    request_base_data_dir = gui_instance.get_base_data_dir_for_request(request)
    user_base_dir = user_session.get_user_directory(request_base_data_dir)
    
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
        # 使用请求特定的 base_data_dir，避免并发问题
        request_base_data_dir = gui_instance.get_base_data_dir_for_request(request)
        user_base_dir = user_session.get_user_directory(request_base_data_dir)
        
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
        # 使用请求特定的 base_data_dir，避免并发问题
        request_base_data_dir = gui_instance.get_base_data_dir_for_request(request)
        user_base_dir = user_session.get_user_directory(request_base_data_dir)
        
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
        # 使用请求特定的 base_data_dir，避免并发问题
        request_base_data_dir = gui_instance.get_base_data_dir_for_request(request)
        user_base_dir = user_session.get_user_directory(request_base_data_dir)
        
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
        
        # URL decode the path to handle encoded characters
        import urllib.parse
        decoded_path = urllib.parse.unquote(path)
        
        # Convert URL path (forward slashes) to OS-specific path separators
        # This handles Windows paths correctly
        normalized_path = decoded_path.replace('/', os.sep)
        
        # Construct full path
        file_path = os.path.join(output_dir, normalized_path)
        
        # Security check: ensure path is within OUTPUT_DIR
        real_output_dir = os.path.realpath(output_dir)
        real_file_path = os.path.realpath(file_path)
        if not real_file_path.startswith(real_output_dir):
            return jsonify({'error': 'Invalid path'}), 403
        
        if not os.path.exists(file_path):
            return jsonify({'error': 'File not found'}), 404
        
        # Determine MIME type based on file extension
        _, ext = os.path.splitext(file_path.lower())
        mime_types = {
            '.svg': 'image/svg+xml',
            '.png': 'image/png',
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.gif': 'image/gif',
            '.webp': 'image/webp'
        }
        mimetype = mime_types.get(ext, 'application/octet-stream')
        
        # Use decoded_path (with forward slashes) for send_from_directory
        # send_from_directory uses safe_join internally, which expects forward slashes
        # even on Windows, because it's designed for URL paths
        # The path should be relative to output_dir
        try:
            # Use decoded_path (forward slashes) - safe_join will handle it correctly
            # Explicitly set mimetype for SVG files
            return send_from_directory(output_dir, decoded_path, mimetype=mimetype)
        except Exception as send_error:
            # If send_from_directory fails, use send_file directly as fallback
            from flask import send_file
            return send_file(file_path, mimetype=mimetype)
    except Exception as e:
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500

@app.route('/api/upload/<path:dir_name>', methods=['POST'])
def upload_files(dir_name):
    """Upload files to workspace of specified directory"""
    uploaded_files = []  # 在try块外初始化，确保异常处理中可以访问
    try:
        i18n = get_i18n_texts()
        
        # Get API key from form data, query parameters or headers
        api_key = request.form.get('api_key') or request.args.get('api_key') or request.headers.get('X-API-Key')
        
        # Create a temporary session for API calls
        temp_session_id = create_temp_session_id(request, api_key)
        user_session = gui_instance.get_user_session(temp_session_id, api_key)
        
        # 确保根据 URL 切换正确的 app，以使用正确的 base_data_dir
        gui_instance.ensure_app_switched_for_request(request, temp_session_id)
        
        # 使用请求特定的 base_data_dir，避免并发问题
        request_base_data_dir = gui_instance.get_base_data_dir_for_request(request)
        user_base_dir = user_session.get_user_directory(request_base_data_dir)
        
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
        
        # 🔧 修复：清除目录缓存，确保上传后的文件能立即显示
        # 清除目标目录的缓存，这样刷新时能获取到最新的文件列表
        # 即使清除缓存失败，也不影响文件上传的成功响应
        try:
            gui_instance.clear_directory_cache(target_dir)
        except Exception:
            # 清除缓存失败不影响文件上传成功，静默处理
            pass
        
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
        # 记录错误日志以便调试
        import traceback
        error_trace = traceback.format_exc()
        print(f"Upload error: {str(e)}\n{error_trace}")
        
        # 如果文件已经上传成功，即使后续处理出错，也返回成功
        # 检查是否有文件已经保存
        if uploaded_files:
            files_str = ', '.join(uploaded_files)
            return jsonify({
                'success': True,
                'message': f'文件已上传: {files_str}（部分操作可能未完成）',
                'files': uploaded_files
            })
        
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
        
        # 确保根据 URL 切换正确的 app，以使用正确的 base_data_dir
        gui_instance.ensure_app_switched_for_request(request, temp_session_id)
        
        # 使用请求特定的 base_data_dir，避免并发问题
        request_base_data_dir = gui_instance.get_base_data_dir_for_request(request)
        user_base_dir = user_session.get_user_directory(request_base_data_dir)
        
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
        
        # 确保根据 URL 切换正确的 app，以使用正确的 base_data_dir
        gui_instance.ensure_app_switched_for_request(request, temp_session_id)
        
        # 使用请求特定的 base_data_dir，避免并发问题
        request_base_data_dir = gui_instance.get_base_data_dir_for_request(request)
        user_base_dir = user_session.get_user_directory(request_base_data_dir)
        
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
        
        # 确保根据 URL 切换正确的 app，以使用正确的 base_data_dir
        gui_instance.ensure_app_switched_for_request(request, temp_session_id)
        
        # 使用请求特定的 base_data_dir，避免并发问题
        request_base_data_dir = gui_instance.get_base_data_dir_for_request(request)
        user_base_dir = user_session.get_user_directory(request_base_data_dir)
        
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
def get_routine_files_route():
    """API endpoint for getting routine files list"""
    # 获取语言参数（优先从URL参数获取）
    lang_param = request.args.get('lang')
    
    # 优先从 URL 路径获取 app_name（如从 /colordoc 页面访问）
    app_name = get_app_name_from_url(request)
    
    # 如果从 URL 获取到了 app_name，直接使用它创建 AppManager
    # 这样不依赖 session，更简单可靠
    if app_name:
        app_manager = AppManager(app_name=app_name)
        return get_routine_files(app_manager=app_manager, lang_param=lang_param)
    
    # 如果没有从 URL 获取到，fallback 到 session（向后兼容）
    api_key = request.args.get('api_key') or request.headers.get('X-API-Key')
    session_id = get_session_id_from_request(request, api_key)
    
    # If no session_id but we have api_key, create/get user session
    if not session_id and api_key:
        temp_session_id = create_temp_session_id(request, api_key)
        user_session = gui_instance.get_user_session(temp_session_id, api_key)
        if user_session:
            session_id = temp_session_id
    
    return get_routine_files(session_id=session_id, lang_param=lang_param)

@app.route('/api/app-list', methods=['GET'])
def get_app_list():
    """Get list of available applications"""
    try:
        # Try to get session_id from request
        api_key = request.args.get('api_key') or request.headers.get('X-API-Key')
        session_id = get_session_id_from_request(request, api_key)
        
        # If no session_id but we have api_key, create/get user session
        if not session_id and api_key:
            temp_session_id = create_temp_session_id(request, api_key)
            user_session = gui_instance.get_user_session(temp_session_id, api_key)
            if user_session:
                session_id = temp_session_id
        
        # Get user-specific AppManager if session_id exists
        user_app_manager = gui_instance.get_user_app_manager(session_id) if session_id else gui_instance.app_manager
        
        apps = gui_instance.app_manager.list_available_apps()  # Use global for listing all apps
        current_app = user_app_manager.app_name
        current_path = request.path if hasattr(request, 'path') else '/'
        return jsonify({
            'success': True,
            'apps': apps,
            'current_app': current_app,
            'current_path': current_path
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'apps': [],
            'current_app': None,
            'error': str(e)
        })

@app.route('/api/switch-app', methods=['POST'])
def api_switch_app():
    """Switch application platform for the current user"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'error': 'Invalid request data'}), 400
        
        app_name = data.get('app_name')
        # If app_name is empty string or None, reset to default
        if app_name == '':
            app_name = None
        
        # Try to get session_id from request
        api_key = request.args.get('api_key') or request.headers.get('X-API-Key') or (data.get('api_key') if isinstance(data, dict) else None)
        session_id = get_session_id_from_request(request, api_key)
        
        # If no session_id but we have api_key, create/get user session
        if not session_id and api_key:
            temp_session_id = create_temp_session_id(request, api_key)
            user_session = gui_instance.get_user_session(temp_session_id, api_key)
            if user_session:
                session_id = temp_session_id
        
        # Validate app_name if provided (check file system, not just visible apps)
        if app_name:
            # Check if app exists by checking directory and app.json file (including hidden apps)
            apps_dir = os.path.join(gui_instance.app_manager.base_dir, 'apps')
            app_path = os.path.join(apps_dir, app_name)
            app_json = os.path.join(app_path, 'app.json')
            app_exists = os.path.isdir(app_path) and os.path.exists(app_json)
            
            if not app_exists:
                return jsonify({
                    'success': False,
                    'error': f'Invalid app name: {app_name}'
                }), 400
        
        # Switch platform for this user (if session_id exists)
        if session_id:
            gui_instance.switch_app(app_name, session_id=session_id)
            # Get user-specific AppManager to return correct app name
            user_app_manager = gui_instance.get_user_app_manager(session_id)
            current_app_name = user_app_manager.get_app_name()
        else:
            # No session, switch global app (backward compatibility)
            gui_instance.switch_app(app_name)
            current_app_name = gui_instance.app_manager.get_app_name()
        
        # Determine redirect URL
        if app_name:
            redirect_url = f'/{app_name}'
        else:
            redirect_url = '/'
        
        return jsonify({
            'success': True,
            'redirect': redirect_url,
            'app_name': current_app_name
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/app-info')
def get_app_info():
    """Get current application information (name and logo) for the current user"""
    try:
        # 优先从 URL 路径获取 app_name（如从 /colordoc 页面访问）
        app_name = get_app_name_from_url(request)
        
        # 如果从 URL 获取到了 app_name，直接使用它创建 AppManager
        if app_name:
            user_app_manager = AppManager(app_name=app_name)
        else:
            # Fallback 到 session（向后兼容）
            api_key = request.args.get('api_key') or request.headers.get('X-API-Key')
            session_id = get_session_id_from_request(request, api_key)
            
            # If no session_id but we have api_key, create/get user session
            if not session_id and api_key:
                temp_session_id = create_temp_session_id(request, api_key)
                user_session = gui_instance.get_user_session(temp_session_id, api_key)
                if user_session:
                    session_id = temp_session_id
            
            # Get user-specific AppManager if session_id exists
            user_app_manager = gui_instance.get_user_app_manager(session_id) if session_id else gui_instance.app_manager
        
        app_name = user_app_manager.get_app_name()
        # Get logo path (no user_dir needed for logo display on main page)
        logo_path = user_app_manager.get_logo_path()
        
        # Convert logo path to URL if it exists
        logo_url = None
        if logo_path:
            # Get relative path from project root
            project_root = user_app_manager.base_dir
            # If logo is in apps directory, serve it via a special route
            apps_dir = os.path.join(project_root, 'apps')
            if logo_path.startswith(apps_dir):
                rel_path = os.path.relpath(logo_path, apps_dir)
                # Normalize path separators for URL
                rel_path = rel_path.replace('\\', '/')
                logo_url = f'/api/app-logo/{rel_path}'
            elif logo_path.startswith(project_root):
                # If logo is elsewhere in project, try static route
                rel_path = os.path.relpath(logo_path, project_root)
                rel_path = rel_path.replace('\\', '/')
                logo_url = f'/static/{rel_path}'
        
        return jsonify({
            'success': True,
            'app_name': app_name,
            'logo_url': logo_url,
            'is_app_mode': user_app_manager.is_app_mode()
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'app_name': 'AGI Agent',
            'logo_url': None,
            'is_app_mode': False,
            'error': str(e)
        })

@app.route('/api/app-logo/<path:logo_path>')
def get_app_logo(logo_path):
    """Serve app logo file"""
    try:
        project_root = gui_instance.app_manager.base_dir
        apps_dir = os.path.join(project_root, 'apps')
        # Normalize the path to handle any path traversal attempts
        logo_path = os.path.normpath(logo_path)
        # Remove any leading slashes or dots
        logo_path = logo_path.lstrip('/').lstrip('.')
        if '..' in logo_path:
            abort(403)
        
        full_path = os.path.join(apps_dir, logo_path)
        
        # Security check: ensure path is within apps directory
        real_apps_dir = os.path.realpath(apps_dir)
        real_file_path = os.path.realpath(full_path)
        if not real_file_path.startswith(real_apps_dir):
            abort(403)
        
        if not os.path.exists(full_path):
            abort(404)
        
        # Determine mimetype based on file extension
        mimetype = None
        if logo_path.lower().endswith('.png'):
            mimetype = 'image/png'
        elif logo_path.lower().endswith('.jpg') or logo_path.lower().endswith('.jpeg'):
            mimetype = 'image/jpeg'
        elif logo_path.lower().endswith('.svg'):
            mimetype = 'image/svg+xml'
        elif logo_path.lower().endswith('.gif'):
            mimetype = 'image/gif'
        
        return send_file(full_path, mimetype=mimetype)
    except Exception as e:
        print(f"Error serving app logo {logo_path}: {e}")
        abort(404)


@app.route('/api/user-guide-image')
def user_guide_image():
    """Serve user guide image from project md/images directory"""
    try:
        # Try to get project root from gui_instance if available
        project_root = None
        try:
            if 'gui_instance' in globals() and hasattr(gui_instance, 'app_manager'):
                project_root = gui_instance.app_manager.base_dir
        except Exception:
            project_root = None

        if not project_root:
            # Fallback: GUI directory的上一级就是项目根目录
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        img_path = os.path.join(project_root, 'md', 'images', 'user_guide.png')
        if not os.path.exists(img_path):
            abort(404)

        return send_file(img_path, mimetype='image/png')
    except Exception as e:
        print(f"Error serving user guide image: {e}")
        abort(404)

def get_routine_files(session_id=None, app_manager=None, lang_param=None):
    """Get list of routine files from routine directory and workspace files starting with 'routine_'
    
    Args:
        session_id: Optional session ID to get user-specific app configuration (向后兼容)
        app_manager: Optional AppManager instance (优先使用，从 URL 路径获取)
        lang_param: Optional language parameter from request (优先使用)
    """
    try:
        routine_files = []
        workspace_dir = os.getcwd()
        
        # 优先使用传入的 app_manager（从 URL 路径获取）
        # 如果没有，则从 session 获取（向后兼容）
        if app_manager:
            user_app_manager = app_manager
        elif session_id:
            user_app_manager = gui_instance.get_user_app_manager(session_id)
        else:
            user_app_manager = gui_instance.app_manager
        
        # 检查是否处于应用模式
        app_routine_dir = None
        is_app_mode = False
        try:
            is_app_mode = user_app_manager.is_app_mode()
            if is_app_mode:
                # Get user_dir if session_id exists for user-specific routine path
                user_dir = None
                if session_id and session_id in gui_instance.user_sessions:
                    user_session = gui_instance.user_sessions[session_id]
                    # 使用 session 特定的 base_data_dir
                    session_base_data_dir = gui_instance.get_base_data_dir_for_session(session_id)
                    user_dir = user_session.get_user_directory(session_base_data_dir)
                app_routine_dir = user_app_manager.get_routine_path(user_dir=user_dir)
        except Exception as e:
            print(f"Warning: Error checking app mode: {e}")
        
        # 如果处于应用模式且找到了应用的routine目录，优先使用应用的routine目录
        app_files_loaded = False
        if is_app_mode and app_routine_dir and os.path.exists(app_routine_dir) and os.path.isdir(app_routine_dir):
            # 从应用的routine目录加载文件
            try:
                for filename in os.listdir(app_routine_dir):
                    file_path = os.path.join(app_routine_dir, filename)
                    if os.path.isfile(file_path):
                        # Remove file extension
                        name_without_ext = os.path.splitext(filename)[0]
                        routine_files.append({
                            'name': name_without_ext,
                            'filename': filename,
                            'type': 'routine_folder'
                        })
                        app_files_loaded = True
                #print(f"DEBUG: Loaded {len(routine_files)} files from app routine directory")
            except Exception as e:
                print(f"Warning: Error reading app routine directory {app_routine_dir}: {e}")
        
        # 如果应用模式下没有加载到文件，或者非应用模式，使用默认routine目录
        if not app_files_loaded:
            # 非应用模式：根据URL参数或语言配置选择routine文件夹
            # 优先使用传入的lang_param（前端传递的），如果没有则尝试从request获取，然后从session获取，最后才使用配置文件
            if lang_param and lang_param in ('zh', 'en'):
                current_lang = lang_param
            else:
                # 尝试从request获取语言参数（向后兼容）
                request_lang = request.args.get('lang') if hasattr(request, 'args') else None
                if request_lang and request_lang in ('zh', 'en'):
                    current_lang = request_lang
                else:
                    # 尝试从session获取语言设置
                    if session_id and session_id in gui_instance.user_sessions:
                        user_session = gui_instance.user_sessions[session_id]
                        gui_config = user_session.gui_config if hasattr(user_session, 'gui_config') else {}
                        session_lang = gui_config.get('language')
                        if session_lang and session_lang in ('zh', 'en'):
                            current_lang = session_lang
                        else:
                            current_lang = get_language()
                    else:
                        current_lang = get_language()
            
            if current_lang == 'zh':
                routine_dir = os.path.join(workspace_dir, 'routine_zh')
            else:
                routine_dir = os.path.join(workspace_dir, 'routine')
            

            # 1. 添加routine文件夹下的文件
            if os.path.exists(routine_dir) and os.path.isdir(routine_dir):
                try:
                    for filename in os.listdir(routine_dir):
                        file_path = os.path.join(routine_dir, filename)
                        if os.path.isfile(file_path):
                            # Remove file extension
                            name_without_ext = os.path.splitext(filename)[0]
                            routine_files.append({
                                'name': name_without_ext,
                                'filename': filename,
                                'type': 'routine_folder'
                            })
                except Exception as e:
                    print(f"Warning: Error reading routine directory {routine_dir}: {e}")
        
        # 2. 添加当前workspace下routine_开头的文件（应用模式和非应用模式都支持）
        try:
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
        except Exception as e:
            print(f"Warning: Error reading workspace directory {workspace_dir}: {e}")
        
        # 按名称排序（反向排序，推荐类文件在上边）
        routine_files.sort(key=lambda x: x['name'], reverse=True)
        
        return jsonify({
            'success': True,
            'files': routine_files
        })
        
    except Exception as e:
        import traceback
        error_msg = f"Error in get_routine_files: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
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
            # Try to get session_id from request for user-specific config
            api_key = request.args.get('api_key') or request.headers.get('X-API-Key')
            session_id = get_session_id_from_request(request, api_key)
            if not session_id and api_key:
                temp_session_id = create_temp_session_id(request, api_key)
                user_session = gui_instance.get_user_session(temp_session_id, api_key)
                if user_session:
                    session_id = temp_session_id
            
            user_app_manager = gui_instance.get_user_app_manager(session_id) if session_id else gui_instance.app_manager
            
            # Use app-specific config file if available
            config_file = "config/config.txt"
            if user_app_manager.is_app_mode():
                app_config_path = user_app_manager.get_config_path()
                if app_config_path:
                    config_file = app_config_path
            
            gui_config = get_gui_config(config_file)
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
        # 使用请求特定的 base_data_dir，避免并发问题
        request_base_data_dir = gui_instance.get_base_data_dir_for_request(request)
        user_base_dir = user_session.get_user_directory(request_base_data_dir)

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
        # 使用请求特定的 base_data_dir，避免并发问题
        request_base_data_dir = gui_instance.get_base_data_dir_for_request(request)
        user_base_dir = user_session.get_user_directory(request_base_data_dir)

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
        # 使用请求特定的 base_data_dir，避免并发问题
        request_base_data_dir = gui_instance.get_base_data_dir_for_request(request)
        user_base_dir = user_session.get_user_directory(request_base_data_dir)
        
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
        
        # 读取当前激活的GUI配置（用于确定默认选择）
        # Try to get session_id from request for user-specific config
        api_key = request.args.get('api_key') or request.headers.get('X-API-Key')
        session_id = get_session_id_from_request(request, api_key)
        if not session_id and api_key:
            temp_session_id = create_temp_session_id(request, api_key)
            user_session = gui_instance.get_user_session(temp_session_id, api_key)
            if user_session:
                session_id = temp_session_id
        
        user_app_manager = gui_instance.get_user_app_manager(session_id) if session_id else gui_instance.app_manager
        
        # Use app-specific config file if available
        config_file = "config/config.txt"
        if user_app_manager.is_app_mode():
            app_config_path = user_app_manager.get_config_path()
            if app_config_path:
                config_file = app_config_path
        
        # 检查配置文件是否存在
        if not os.path.exists(config_file):
            return jsonify({
                'success': False,
                'error': f'Configuration file not found: {config_file}',
                'configs': []
            })
        
        # 读取所有模型配置（包括注释掉的）
        try:
            all_configs = get_all_model_configs(config_file)
        except Exception as e:
            import traceback
            traceback.print_exc()
            return jsonify({
                'success': False,
                'error': f'Failed to parse model configurations: {str(e)}',
                'configs': []
            })
        
        # 读取GUI配置
        try:
            gui_config = get_gui_config(config_file)
        except Exception as e:
            import traceback
            traceback.print_exc()
            # GUI配置加载失败不影响模型配置列表的返回
            gui_config = {}
        
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
            'error': str(e),
            'configs': []
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
    """Generate image from SVG file using image generation API"""
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
        # 使用请求特定的 base_data_dir，避免并发问题
        request_base_data_dir = gui_instance.get_base_data_dir_for_request(request)
        user_base_dir = user_session.get_user_directory(request_base_data_dir)

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
            original_svg_content = f.read()

        optimization_report = None
        generated_image_path = None

        if use_llm and LLM_SVG_OPTIMIZER_AVAILABLE:
            # 先检查是否已经生成过_aicreate图像
            # 生成_aicreate文件路径：原文件名_aicreate.png
            from pathlib import Path
            svg_path_obj = Path(full_path)
            aicreate_path = svg_path_obj.parent / f"{svg_path_obj.stem}_aicreate.png"
            
            # 检查文件是否存在
            if aicreate_path.exists():
                # 已存在，直接使用该文件，不重新生成
                generated_image_path = str(aicreate_path)
                
                # 转换为相对路径
                if os.path.isabs(generated_image_path):
                    try:
                        generated_image_path = os.path.relpath(generated_image_path, user_base_dir)
                    except ValueError:
                        pass
                
                # 清理路径（移除output_*/workspace/前缀）
                original_rel_path = file_path.replace('\\', '/')
                original_dir = os.path.dirname(original_rel_path).replace('\\', '/')
                original_dir_parts = original_dir.split('/') if original_dir else []
                cleaned_dir_parts = []
                skip_next = False
                for i, part in enumerate(original_dir_parts):
                    if skip_next:
                        skip_next = False
                        continue
                    if part.startswith('output_'):
                        if i + 1 < len(original_dir_parts) and original_dir_parts[i + 1] == 'workspace':
                            skip_next = True
                        continue
                    if part == 'workspace' and i > 0 and original_dir_parts[i - 1].startswith('output_'):
                        continue
                    cleaned_dir_parts.append(part)
                
                generated_filename = os.path.basename(generated_image_path)
                if cleaned_dir_parts:
                    generated_image_path = '/'.join(cleaned_dir_parts) + '/' + generated_filename
                else:
                    generated_image_path = generated_filename
                
                # 创建优化报告
                optimization_report = {
                    'method': 'ImageGeneration',
                    'model': 'cached',
                    'api_base': 'cached',
                    'output_path': generated_image_path,
                    'image_format': 'png',
                    'cached': True  # 标记为缓存文件
                }
                
                # 更新markdown文件中的图像链接
                updated_markdown_files = _update_markdown_image_links(
                    user_base_dir, file_path, generated_image_path
                )
                
                if updated_markdown_files:
                    optimization_report['updated_markdown_files'] = updated_markdown_files
            else:
                # 不存在，调用大模型生成
                try:
                    optimizer = create_image_generation_optimizer_from_config()
                    generated_image_path, report = optimizer.generate_image_from_svg(original_svg_content, full_path)

                    # Convert absolute path to relative path, and clean up path
                    if generated_image_path:
                        if os.path.isabs(generated_image_path):
                            try:
                                generated_image_path = os.path.relpath(generated_image_path, user_base_dir)
                            except ValueError:
                                # If paths are on different drives, use absolute path
                                pass
                        
                        # Extract directory structure from original file path, removing output_*/workspace/ prefix
                        # Example: "images/example.svg" -> "images/example_aicreate.png"
                        # Example: "output_20260125_154810/workspace/example.svg" -> "example_aicreate.png"
                        # Example: "output_20260125_154810/workspace/images/example.svg" -> "images/example_aicreate.png"
                        
                        # Get original file's directory structure (relative to user_base_dir)
                        original_rel_path = file_path.replace('\\', '/')
                        original_dir = os.path.dirname(original_rel_path).replace('\\', '/')
                        
                        # Remove output_*/workspace/ prefix from original directory
                        original_dir_parts = original_dir.split('/') if original_dir else []
                        cleaned_dir_parts = []
                        skip_next = False
                        for i, part in enumerate(original_dir_parts):
                            if skip_next:
                                skip_next = False
                                continue
                            # Skip "output_*" pattern directories
                            if part.startswith('output_'):
                                # Check if next part is "workspace"
                                if i + 1 < len(original_dir_parts) and original_dir_parts[i + 1] == 'workspace':
                                    skip_next = True
                                continue
                            # Skip "workspace" directory if it's after an output_* directory
                            if part == 'workspace' and i > 0 and original_dir_parts[i - 1].startswith('output_'):
                                continue
                            cleaned_dir_parts.append(part)
                        
                        # Get generated image filename
                        generated_filename = os.path.basename(generated_image_path)
                        
                        # Combine cleaned directory with filename
                        if cleaned_dir_parts:
                            generated_image_path = '/'.join(cleaned_dir_parts) + '/' + generated_filename
                        else:
                            generated_image_path = generated_filename

                    optimization_report = {
                        'method': 'ImageGeneration',
                        'model': report.get('model', 'unknown'),
                        'api_base': report.get('api_base', 'unknown'),
                        'output_path': generated_image_path,
                        'image_format': report.get('image_format', 'png')
                    }

                    # Update markdown files that reference this SVG
                    updated_markdown_files = _update_markdown_image_links(
                        user_base_dir, file_path, generated_image_path
                    )

                    if updated_markdown_files:
                        optimization_report['updated_markdown_files'] = updated_markdown_files

                except Exception as img_gen_error:
                    import traceback
                    error_details = traceback.format_exc()
                    return jsonify({
                        'success': False,
                        'error': f'Image generation failed: {str(img_gen_error)}\n\n{error_details}'
                    })

        if not use_llm or not LLM_SVG_OPTIMIZER_AVAILABLE:
            return jsonify({
                'success': False,
                'error': 'Image generation is not available. Please ensure image_generation_api_key is configured in config/config.txt'
            })

        # Generate success message
        if optimization_report:
            output_path = optimization_report.get('output_path', 'unknown')
            # 只显示文件名，不显示完整路径
            if output_path != 'unknown':
                output_path = os.path.basename(output_path)
                message = f"图像生成成功，输出文件: {output_path}"
            else:
                message = f"图像生成成功"
        else:
            message = f"图像生成成功"

        return jsonify({
            'success': True,
            'message': message,
            'optimization_report': optimization_report,
            'used_llm': use_llm,
            'generated_image_path': generated_image_path
        })

    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        return jsonify({
            'success': False,
            'error': f'SVG image generation failed: {str(e)}\n\n{error_details}'
        })


def _update_markdown_image_links(base_dir: str, original_svg_path: str, new_image_path: str) -> list:
    """
    更新markdown文件中引用SVG文件的图像链接
    
    Args:
        base_dir: 基础目录
        original_svg_path: 原始SVG文件路径（相对路径）
        new_image_path: 新生成的图像文件路径（相对路径）
        
    Returns:
        已更新的markdown文件列表
    """
    import re
    from pathlib import Path
    
    updated_files = []
    
    # 查找所有markdown文件
    base_path = Path(base_dir)
    md_files = list(base_path.rglob('*.md'))
    
    # 准备匹配模式：匹配引用原始SVG文件的图像链接
    # 支持多种格式：![alt](path), ![alt](path "title"), <img src="path">
    svg_filename = os.path.basename(original_svg_path)
    svg_name_without_ext = os.path.splitext(svg_filename)[0]
    
    for md_file in md_files:
        try:
            with open(md_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            original_content = content
            content_changed = False
            
            # 模式1: ![alt](svg_path)
            pattern1 = r'!\[([^\]]*)\]\(([^)]*' + re.escape(svg_filename) + r'[^)]*)\)'
            def replace_func1(match):
                nonlocal content_changed
                alt_text = match.group(1)
                old_path = match.group(2)
                # 检查路径是否匹配（支持相对路径和绝对路径）
                if svg_filename in old_path or svg_name_without_ext in old_path:
                    content_changed = True
                    return f'![{alt_text}]({new_image_path})'
                return match.group(0)
            
            content = re.sub(pattern1, replace_func1, content)
            
            # 模式2: <img src="svg_path" alt="alt">
            pattern2 = r'<img\s+([^>]*src=["\']([^"\']*' + re.escape(svg_filename) + r'[^"\']*)["\'][^>]*)>'
            def replace_func2(match):
                nonlocal content_changed
                img_attrs = match.group(1)
                old_src = match.group(2)
                if svg_filename in old_src or svg_name_without_ext in old_src:
                    content_changed = True
                    # 替换src属性
                    new_attrs = re.sub(r'src=["\'][^"\']*["\']', f'src="{new_image_path}"', img_attrs)
                    return f'<img {new_attrs}>'
                return match.group(0)
            
            content = re.sub(pattern2, replace_func2, content)
            
            # 如果内容有变化，保存文件
            if content_changed:
                with open(md_file, 'w', encoding='utf-8') as f:
                    f.write(content)
                rel_path = os.path.relpath(md_file, base_dir)
                updated_files.append(rel_path)
                
        except Exception as e:
            # 忽略无法处理的文件
            print(f"Warning: Failed to update markdown file {md_file}: {e}")
            continue
    
    return updated_files


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


@app.route('/api/contact-us', methods=['POST'])
def api_contact_us():
    """处理联系我们留言提交"""
    try:
        data = request.get_json() or {}
        session_id = data.get('session_id', 'Unknown')
        message = data.get('message', '').strip()
        current_dir = data.get('current_dir', '').strip()
        contact_info = data.get('contact_info', '').strip()
        
        if not message:
            return jsonify({
                'success': False,
                'error': 'Message cannot be empty'
            })
        
        # 获取gui_default_data_directory配置的目录
        gui_data_dir = get_gui_default_data_directory()
        if not gui_data_dir or not os.path.exists(gui_data_dir):
            # 如果配置的目录不存在，使用当前工作目录
            gui_data_dir = os.getcwd()
        
        # 在gui_default_data_directory下创建contact_messages目录（如果不存在）
        contact_dir = os.path.join(gui_data_dir, 'contact_messages')
        os.makedirs(contact_dir, exist_ok=True)
        
        # 保存留言到文件
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'contact_{timestamp}_{session_id[:8]}.txt'
        filepath = os.path.join(contact_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(f'Session ID: {session_id}\n')
            f.write(f'Timestamp: {datetime.datetime.now().isoformat()}\n')
            if current_dir:
                f.write(f'Current Directory: {current_dir}\n')
            if contact_info:
                f.write(f'Contact Information: {contact_info}\n')
            f.write(f'Message:\n{message}\n')
        
        return jsonify({
            'success': True,
            'message': 'Message received successfully'
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        })


if __name__ == '__main__':
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='AGIAgent GUI Server')
    parser.add_argument('--port', '-p', type=int, default=5002, 
                       help='Port specified to use')
    parser.add_argument('--app', '-a', type=str, default=None,
                       help='Application name (e.g., patent, national_project)')
    args = parser.parse_args()
    
    # 优先使用命令行参数，其次使用环境变量，最后使用默认值
    port = args.port if args.port else int(os.environ.get('PORT', 5002))
    app_name = args.app if args.app else os.environ.get('AGIA_APP_NAME', None)
    
    # 如果通过命令行指定了app_name，更新环境变量并重新创建gui_instance
    if app_name:
        os.environ['AGIA_APP_NAME'] = app_name
        # 重新创建gui_instance以应用app_name
        import __main__
        if hasattr(__main__, 'gui_instance'):
            __main__.gui_instance = AGIAgentGUI(app_name=app_name)
        # Also update the module-level gui_instance
        import sys
        current_module = sys.modules[__name__]
        current_module.gui_instance = AGIAgentGUI(app_name=app_name)
    else:
        # 如果没有指定app_name，确保initial_app_name为None（默认平台）
        gui_instance.initial_app_name = None
    
    print(f"🚀 Starting AGIAgent GUI Server on port {port}")
    if app_name:
        print(f"📱 Application mode: {app_name} ({gui_instance.app_manager.get_app_name()})")
    socketio.run(app, host='0.0.0.0', port=port, debug=False, allow_unsafe_werkzeug=True) 
    print(f"🚀 Wait for 5 seconds and open the browser with url 127.0.0.1:{port}")
