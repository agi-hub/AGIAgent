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

"""
Configuration constants and settings for the multi-round task executor
"""

import os
import re

# Default configuration values
DEFAULT_SUBTASK_LOOPS = 3
DEFAULT_LOGS_DIR = "logs"
DEFAULT_MODEL = None


# Summary configuration
DEFAULT_DETAILED_SUMMARY = True
SUMMARY_WORD_LIMIT = 400

# Report configuration  
REPORT_TIMESTAMP_FORMAT = "%Y%m%d_%H%M%S"
DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S"

# Debug configuration
def get_debug_truncation_config():
    """动态从配置文件获取调试截断配置"""
    try:
        # 导入config_loader模块
        import sys
        import os
        sys.path.append(os.path.dirname(os.path.dirname(__file__)))
        from config_loader import get_truncation_length
        
        truncation = get_truncation_length()
        
        return {
            'DEBUG_PROMPT_LIMIT': truncation,
            'DEBUG_OUTPUT_LIMIT': min(truncation, 2000),  # 限制输出不超过2000或主截断长度
            'DEBUG_RESULT_LIMIT': truncation
        }
    except Exception:
        # 如果读取配置失败，使用默认值
        return {
            'DEBUG_PROMPT_LIMIT': 1000,
            'DEBUG_OUTPUT_LIMIT': 2000,
            'DEBUG_RESULT_LIMIT': 1000
        }

# 获取动态配置
_debug_config = get_debug_truncation_config()
DEBUG_PROMPT_LIMIT = _debug_config['DEBUG_PROMPT_LIMIT']
DEBUG_OUTPUT_LIMIT = _debug_config['DEBUG_OUTPUT_LIMIT']
DEBUG_RESULT_LIMIT = _debug_config['DEBUG_RESULT_LIMIT']

# Task completion keywords
TASK_COMPLETION_KEYWORDS = [
    'complete', 'success', 'create', 'fix', 'solve' 
]

# Error keywords
ERROR_KEYWORDS = ['error', 'fail', 'exception']


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


def read_language_config(config_file: str = None) -> str:
    """
    Read language configuration from config file
    
    Args:
        config_file: Config file path, defaults to config.txt in current directory
        
    Returns:
        Language code ('en' or 'zh')
    """
    if config_file is None:
        config_file = os.path.join(os.path.dirname(__file__), "..", "config.txt")
    
    try:
        if os.path.exists(config_file):
            with open(config_file, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                if "LANG=en" in content:
                    return "en"
                else:
                    return "zh"
        else:
            # Default to Chinese if config file doesn't exist
            return "zh"
    except Exception:
        return "zh"