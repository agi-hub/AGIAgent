#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AGIBot Output Manager
Manages terminal vs log file output based on message importance
"""

import os
import sys
import threading
import time
from datetime import datetime
from typing import Optional, Any, Dict, Set
from enum import Enum
from contextlib import contextmanager


class MessageImportance(Enum):
    """Message importance levels"""
    CRITICAL = "CRITICAL"    # Always show in terminal
    IMPORTANT = "IMPORTANT"  # Show in terminal
    INFO = "INFO"           # Log to file only
    DEBUG = "DEBUG"         # Log to file only


class OutputManager:
    """Manages output routing between terminal and log files"""
    
    _instance = None
    _creation_lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._creation_lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if hasattr(self, '_initialized'):
            return
        
        self._initialized = True
        self._log_dir = None
        self._log_files = {}  # category -> file handle
        self._terminal_categories: Set[str] = {
            'llm_response',      # LLM response content
            'tool_calls',        # Tool call information
            'tool_results',      # Tool execution results
            'task_completion',   # Task completion status
            'user_interaction',  # User interaction
            'error'              # Error information
        }
        self._lock = threading.RLock()
        
        # Simplified log structure - all logs go to one file
        self._log_categories = {
            'system': 'agibot.log',           # System initialization, configuration, etc.
            'debug': 'agibot.log',            # Debug information
            'tools': 'agibot.log',            # Detailed tool call logs
            'llm': 'agibot.log',              # LLM call logs
            'execution': 'agibot.log',        # Task execution logs
            'general': 'agibot.log'           # General logs
        }
        self._unified_log_file = None
    def set_output_directory(self, out_dir: str):
        """Set output directory and initialize log files"""
        with self._lock:
            # Close existing log files
            self._close_log_files()
            
            # Set new log directory
            self._log_dir = os.path.join(out_dir, "logs")
            os.makedirs(self._log_dir, exist_ok=True)
            
            # Initialize log files
            self._init_log_files()
    
    def _init_log_files(self):
        """Initialize unified log file"""
        if not self._log_dir:
            return
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Create unified log file
        log_path = os.path.join(self._log_dir, 'agibot.log')
        try:
            file_handle = open(log_path, 'a', encoding='utf-8', buffering=1)
            file_handle.write(f"\n=== Log session started at {timestamp} ===\n")
            file_handle.flush()
            self._unified_log_file = file_handle
            
            # All categories use the same file handle
            for category in self._log_categories.keys():
                self._log_files[category] = file_handle
        except Exception as e:
            print(f"Warning: Failed to create log file {log_path}: {e}")
    
    def _close_log_files(self):
        """Close unified log file"""
        if self._unified_log_file and not self._unified_log_file.closed:
            try:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                self._unified_log_file.write(f"=== Log session ended at {timestamp} ===\n\n")
                self._unified_log_file.close()
            except:
                pass
        self._log_files.clear()
        self._unified_log_file = None
    
    def log_message(self, message: str, category: str = 'general', 
                   importance: MessageImportance = MessageImportance.INFO,
                   show_in_terminal: bool = None):
        """
        Log a message with specified category and importance
        
        Args:
            message: The message to log
            category: Log category (system, debug, tools, llm, execution, general)
            importance: Message importance level
            show_in_terminal: Override terminal display (None = auto-decide)
        """
        with self._lock:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            formatted_message = f"[{timestamp}] {message}"
            
            # Always log to file
            self._write_to_log_file(formatted_message, category)
            
            # Decide terminal output
            if show_in_terminal is None:
                # Auto-decide based on importance and category
                show_in_terminal = (
                    importance in [MessageImportance.CRITICAL, MessageImportance.IMPORTANT] or
                    category in self._terminal_categories
                )
            
            return show_in_terminal
    
    def _write_to_log_file(self, message: str, category: str):
        """Write message to unified log file with category prefix"""
        # 🔧 核心修复：当文件句柄为None时，尝试恢复并写入
        if self._unified_log_file is None:
            # 尝试恢复日志文件句柄
            if self._recover_log_file():
                # 恢复成功，继续正常写入流程
                pass
            else:
                # 恢复失败，无法写入日志文件
                return
        
        # 检查文件句柄是否有效
        if self._unified_log_file and hasattr(self._unified_log_file, 'closed') and not self._unified_log_file.closed:
            try:
                # Add category prefix for better organization
                categorized_message = f"[{category.upper()}] {message}"
                self._unified_log_file.write(categorized_message + '\n')
                self._unified_log_file.flush()
            except Exception as e:
                # 文件写入失败，尝试恢复
                if self._recover_log_file():
                    # 恢复成功，重试写入
                    try:
                        categorized_message = f"[{category.upper()}] {message}"
                        self._unified_log_file.write(categorized_message + '\n')
                        self._unified_log_file.flush()
                    except:
                        pass  # 重试失败，放弃写入
    
    def log_llm_response(self, response: str, show_in_terminal: bool = True):
        """Log LLM response"""
        return self.log_message(
            f"LLM Response: {response}",
            category='llm',
            importance=MessageImportance.IMPORTANT,
            show_in_terminal=show_in_terminal
        )
    
    def log_tool_call(self, tool_name: str, params: Dict[str, Any], show_in_terminal: bool = True):
        """Log tool call"""
        params_str = str(params)[:200] + "..." if len(str(params)) > 200 else str(params)
        return self.log_message(
            f"Tool Call: {tool_name} with params: {params_str}",
            category='tools',
            importance=MessageImportance.IMPORTANT,
            show_in_terminal=show_in_terminal
        )
    
    def log_tool_result(self, tool_name: str, result: Any, show_in_terminal: bool = True):
        """Log tool result"""
        result_str = str(result)[:500] + "..." if len(str(result)) > 500 else str(result)
        return self.log_message(
            f"Tool Result ({tool_name}): {result_str}",
            category='tools',
            importance=MessageImportance.IMPORTANT,
            show_in_terminal=show_in_terminal
        )
    
    def log_system_info(self, message: str, show_in_terminal: bool = False):
        """Log system information"""
        return self.log_message(
            message,
            category='system',
            importance=MessageImportance.INFO,
            show_in_terminal=show_in_terminal
        )
    
    def log_debug(self, message: str, show_in_terminal: bool = False):
        """Log debug information"""
        return self.log_message(
            message,
            category='debug',
            importance=MessageImportance.DEBUG,
            show_in_terminal=show_in_terminal
        )
    
    def log_error(self, message: str, show_in_terminal: bool = True):
        """Log error message"""
        return self.log_message(
            f"ERROR: {message}",
            category='general',
            importance=MessageImportance.CRITICAL,
            show_in_terminal=show_in_terminal
        )
    
    def _recover_log_file(self) -> bool:
        """尝试恢复日志文件句柄"""
        try:
            # 如果已经有日志目录，直接重新初始化文件
            if self._log_dir and os.path.exists(self._log_dir):
                self._init_log_files()
                return self._unified_log_file is not None
            
            # 如果没有日志目录，尝试查找可能的输出目录
            possible_dirs = []
            
            # 查找最新的output目录
            import glob
            output_patterns = ['output_*', './output_*', '../output_*']
            for pattern in output_patterns:
                dirs = glob.glob(pattern)
                if dirs:
                    # 按修改时间排序，取最新的
                    dirs.sort(key=lambda x: os.path.getmtime(x) if os.path.exists(x) else 0, reverse=True)
                    possible_dirs.extend(dirs)
            
            # 如果找到输出目录，使用第一个进行初始化
            if possible_dirs:
                self.set_output_directory(possible_dirs[0])
                return self._unified_log_file is not None
            
            return False
        except Exception:
            return False
    
    def cleanup(self):
        """Clean up resources"""
        with self._lock:
            self._close_log_files()


# Global output manager instance
_output_manager = None

def get_output_manager() -> OutputManager:
    """Get output manager instance"""
    global _output_manager
    if _output_manager is None:
        _output_manager = OutputManager()
    return _output_manager


def init_output_manager(out_dir: str):
    """Initialize output manager with output directory"""
    manager = get_output_manager()
    manager.set_output_directory(out_dir)


# Convenience functions
def log_to_file(message: str, category: str = 'general', 
               importance: MessageImportance = MessageImportance.INFO) -> bool:
    """
    Log message to file and return whether it should be shown in terminal
    
    Returns:
        bool: True if message should be shown in terminal
    """
    return get_output_manager().log_message(message, category, importance)


def log_llm_response(response: str, show_in_terminal: bool = True) -> bool:
    """Log LLM response"""
    return get_output_manager().log_llm_response(response, show_in_terminal)


def log_tool_call(tool_name: str, params: Dict[str, Any], show_in_terminal: bool = True) -> bool:
    """Log tool call"""
    return get_output_manager().log_tool_call(tool_name, params, show_in_terminal)


def log_tool_result(tool_name: str, result: Any, show_in_terminal: bool = True) -> bool:
    """Log tool result"""
    return get_output_manager().log_tool_result(tool_name, result, show_in_terminal)


def log_system_info(message: str, show_in_terminal: bool = False) -> bool:
    """Log system information"""
    return get_output_manager().log_system_info(message, show_in_terminal)


def log_debug(message: str, show_in_terminal: bool = False) -> bool:
    """Log debug information"""  
    return get_output_manager().log_debug(message, show_in_terminal)


def log_error(message: str, show_in_terminal: bool = True) -> bool:
    """Log error message"""
    return get_output_manager().log_error(message, show_in_terminal)