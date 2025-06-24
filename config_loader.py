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

import os
from typing import Dict, Optional

def load_config(config_file: str = "config.txt", verbose: bool = False) -> Dict[str, str]:
    """
    Load configuration from config.txt file
    
    Args:
        config_file: Path to the configuration file
        verbose: Whether to print debug information
        
    Returns:
        Dictionary containing configuration key-value pairs
    """
    config = {}
    
    if not os.path.exists(config_file):
        if verbose:
            print(f"Warning: Configuration file {config_file} not found")
        return config
    
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            line_number = 0
            for line in f:
                line_number += 1
                original_line = line.rstrip('\n\r')  # 保留原始行用于调试
                line = line.strip()
                
                # 跳过空行
                if not line:
                    continue
                
                # 跳过纯注释行（以#开头的行）
                if line.startswith('#'):
                    if verbose:
                        print(f"Skipping commented line {line_number}: {original_line}")
                    continue
                
                # 处理包含等号的行
                if '=' in line:
                    # 处理行内注释：在#之前分割
                    if '#' in line:
                        line = line.split('#')[0].strip()
                    
                    # 分割键值对
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip()
                    
                    if key:  # 确保键不为空
                        config[key] = value
                        if verbose:
                            print(f"Loaded config: {key} = {value}")
                    else:
                        if verbose:
                            print(f"Warning: Empty key found on line {line_number}: {original_line}")
                else:
                    if verbose:
                        print(f"Warning: Invalid config line {line_number} (no '=' found): {original_line}")
                    
    except Exception as e:
        print(f"Error reading configuration file {config_file}: {e}")
    
    return config

def get_api_key(config_file: str = "config.txt") -> Optional[str]:
    """
    Get API key from configuration file
    
    Args:
        config_file: Path to the configuration file
        
    Returns:
        API key string or None if not found
    """
    config = load_config(config_file)
    return config.get('api_key')

def get_api_base(config_file: str = "config.txt") -> Optional[str]:
    """
    Get API base URL from configuration file
    
    Args:
        config_file: Path to the configuration file
        
    Returns:
        API base URL string or None if not found
    """
    config = load_config(config_file)
    return config.get('api_base')

def get_config_value(key: str, default: Optional[str] = None, config_file: str = "config.txt") -> Optional[str]:
    """
    Get a specific configuration value
    
    Args:
        key: Configuration key
        default: Default value if key not found
        config_file: Path to the configuration file
        
    Returns:
        Configuration value or default
    """
    config = load_config(config_file)
    return config.get(key, default)

def get_model(config_file: str = "config.txt") -> Optional[str]:
    """
    Get model name from configuration file
    
    Args:
        config_file: Path to the configuration file
        
    Returns:
        Model name string or None if not found
    """
    config = load_config(config_file)
    return config.get('model')

def get_max_tokens(config_file: str = "config.txt") -> Optional[int]:
    """
    Get max_tokens from configuration file with model-specific defaults
    
    Args:
        config_file: Path to the configuration file
        
    Returns:
        Max tokens integer with model-specific default if not manually set
    """
    config = load_config(config_file)
    max_tokens_str = config.get('max_tokens')
    
    # If user manually set max_tokens, use their setting
    if max_tokens_str:
        try:
            return int(max_tokens_str)
        except ValueError:
            print(f"Warning: Invalid max_tokens value '{max_tokens_str}' in config file, must be an integer")
            # Fall through to model-specific defaults
    
    # If no manual setting, use model-specific defaults
    model = get_model(config_file)
    if model:
        model_lower = model.lower()
        
        # DeepSeek and OpenAI models: 8192
        if ('deepseek' in model_lower or 
            'gpt-' in model_lower or 
            'o1-' in model_lower or
            'chatgpt' in model_lower):
            return 8192
            
        # Anthropic Claude models: 16384
        elif ('claude' in model_lower or 
              'anthropic' in model_lower):
            return 16384
            
        # Other models (Ollama, Qwen, Doubao, etc.): 4096
        else:
            return 4096
    
    # Fallback default if no model specified
    return 4096

def get_streaming(config_file: str = "config.txt") -> bool:
    """
    Get streaming configuration from configuration file
    
    Args:
        config_file: Path to the configuration file
        
    Returns:
        Boolean indicating whether to use streaming output (default: False)
    """
    config = load_config(config_file)
    streaming_str = config.get('streaming', 'False').lower()
    
    # Convert string to boolean
    if streaming_str in ('true', '1', 'yes', 'on'):
        return True
    elif streaming_str in ('false', '0', 'no', 'off'):
        return False
    else:
        print(f"Warning: Invalid streaming value '{streaming_str}' in config file, using default False")
        return False

def get_language(config_file: str = "config.txt") -> str:
    """
    Get language configuration from configuration file
    
    Args:
        config_file: Path to the configuration file
        
    Returns:
        Language code string (default: 'en' for English)
    """
    config = load_config(config_file)
    lang = config.get('LANG', 'en').lower()
    
    # Support common language codes
    if lang in ('zh', 'zh-cn', 'chinese', '中文'):
        return 'zh'
    elif lang in ('en', 'english', 'eng'):
        return 'en'
    else:
        print(f"Warning: Unsupported language '{lang}' in config file, defaulting to English")
        return 'en'

def get_truncation_length(config_file: str = "config.txt") -> int:
    """
    Get truncation length from configuration file
    
    Args:
        config_file: Path to the configuration file
        
    Returns:
        Truncation length integer (default: 10000)
    """
    config = load_config(config_file)
    truncation_str = config.get('truncation_length')
    
    if truncation_str:
        try:
            truncation_length = int(truncation_str)
            if truncation_length <= 0:
                print(f"Warning: Invalid truncation_length value '{truncation_str}' in config file, must be positive integer, using default 10000")
                return 10000
            return truncation_length
        except ValueError:
            print(f"Warning: Invalid truncation_length value '{truncation_str}' in config file, must be an integer, using default 10000")
            return 10000
    
    return 10000  # Default truncation length

# def get_history_truncation_length(config_file: str = "config.txt") -> int:
#     """
#     Get history truncation length from configuration file
#     DEPRECATED: This function is no longer used as we now use summarization instead of truncation
#     
#     Args:
#         config_file: Path to the configuration file
#         
#     Returns:
#         History truncation length integer (default: 1000)
#     """
#     config = load_config(config_file)
#     history_truncation_str = config.get('history_truncation_length')
#     
#     if history_truncation_str:
#         try:
#             truncation_length = int(history_truncation_str)
#             if truncation_length <= 0:
#                 print(f"Warning: Invalid history_truncation_length value '{history_truncation_str}' in config file, must be positive integer, using default 1000")
#                 return 1000
#             return truncation_length
#         except ValueError:
#             print(f"Warning: Invalid history_truncation_length value '{history_truncation_str}' in config file, must be an integer, using default 1000")
#             return 1000
#     
#     # 如果没有设置，则使用主截断长度的 1/10，但不少于1000
#     main_truncation = get_truncation_length(config_file)
#     return max(1000, main_truncation // 10)

def get_web_content_truncation_length(config_file: str = "config.txt") -> int:
    """
    Get web content truncation length from configuration file
    
    Args:
        config_file: Path to the configuration file
        
    Returns:
        Web content truncation length integer (default: 50000)
    """
    config = load_config(config_file)
    web_truncation_str = config.get('web_content_truncation_length')
    
    if web_truncation_str:
        try:
            truncation_length = int(web_truncation_str)
            if truncation_length <= 0:
                print(f"Warning: Invalid web_content_truncation_length value '{web_truncation_str}' in config file, must be positive integer, using default 50000")
                return 50000
            return truncation_length
        except ValueError:
            print(f"Warning: Invalid web_content_truncation_length value '{web_truncation_str}' in config file, must be an integer, using default 50000")
            return 50000
    
    # 如果没有设置，则使用主截断长度的 5 倍，但不少于50000
    main_truncation = get_truncation_length(config_file)
    return max(50000, main_truncation * 5)

def get_summary_history(config_file: str = "config.txt") -> bool:
    """
    Get summary_history configuration from configuration file
    
    Args:
        config_file: Path to the configuration file
        
    Returns:
        Boolean indicating whether to use history summarization (default: False)
    """
    config = load_config(config_file)
    summary_history_str = config.get('summary_history', 'False').lower()
    
    # Convert string to boolean
    if summary_history_str in ('true', '1', 'yes', 'on'):
        return True
    elif summary_history_str in ('false', '0', 'no', 'off'):
        return False
    else:
        print(f"Warning: Invalid summary_history value '{summary_history_str}' in config file, using default False")
        return False

def get_summary_max_length(config_file: str = "config.txt") -> int:
    """
    Get summary max length from configuration file
    
    Args:
        config_file: Path to the configuration file
        
    Returns:
        Summary max length integer (default: 5000)
    """
    config = load_config(config_file)
    summary_max_length_str = config.get('summary_max_length')
    
    if summary_max_length_str:
        try:
            summary_max_length = int(summary_max_length_str)
            if summary_max_length <= 0:
                print(f"Warning: Invalid summary_max_length value '{summary_max_length_str}' in config file, must be positive integer, using default 5000")
                return 5000
            return summary_max_length
        except ValueError:
            print(f"Warning: Invalid summary_max_length value '{summary_max_length_str}' in config file, must be an integer, using default 5000")
            return 5000
    
    return 5000  # Default summary max length

def get_summary_trigger_length(config_file: str = "config.txt") -> int:
    """
    Get summary trigger length from configuration file
    
    Args:
        config_file: Path to the configuration file
        
    Returns:
        Summary trigger length integer (default: 20000)
    """
    config = load_config(config_file)
    summary_trigger_length_str = config.get('summary_trigger_length')
    
    if summary_trigger_length_str:
        try:
            summary_trigger_length = int(summary_trigger_length_str)
            if summary_trigger_length <= 0:
                print(f"Warning: Invalid summary_trigger_length value '{summary_trigger_length_str}' in config file, must be positive integer, using default 20000")
                return 20000
            return summary_trigger_length
        except ValueError:
            print(f"Warning: Invalid summary_trigger_length value '{summary_trigger_length_str}' in config file, must be an integer, using default 20000")
            return 20000
    
    return 20000  # Default summary trigger length

def get_simplified_search_output(config_file: str = "config.txt") -> bool:
    """
    Get simplified search output configuration from configuration file
    
    Args:
        config_file: Path to the configuration file
        
    Returns:
        Boolean indicating whether to use simplified output for search tools (default: True)
    """
    config = load_config(config_file)
    simplified_output_str = config.get('simplified_search_output', 'True').lower()
    
    # Convert string to boolean
    if simplified_output_str in ('true', '1', 'yes', 'on'):
        return True
    elif simplified_output_str in ('false', '0', 'no', 'off'):
        return False
    else:
        print(f"Warning: Invalid simplified_search_output value '{simplified_output_str}' in config file, using default True")
        return True

def get_summary_report(config_file: str = "config.txt") -> bool:
    """
    Get summary report generation configuration from configuration file
    
    Args:
        config_file: Path to the configuration file
        
    Returns:
        Boolean indicating whether to generate summary reports (default: False)
    """
    config = load_config(config_file)
    summary_report_str = config.get('summary_report', 'False').lower()
    
    # Convert string to boolean
    if summary_report_str in ('true', '1', 'yes', 'on'):
        return True
    elif summary_report_str in ('false', '0', 'no', 'off'):
        return False
    else:
        print(f"Warning: Invalid summary_report value '{summary_report_str}' in config file, using default False")
        return False

def get_gui_default_data_directory(config_file: str = "config.txt") -> Optional[str]:
    """
    Get GUI default user data directory from configuration file
    
    Args:
        config_file: Path to the configuration file
        
    Returns:
        GUI default data directory path or None if not set or invalid
    """
    config = load_config(config_file)
    gui_data_dir = config.get('gui_default_data_directory')
    
    if not gui_data_dir:
        return None
    
    # Expand user home directory if path starts with ~
    gui_data_dir = os.path.expanduser(gui_data_dir)
    
    # Convert relative path to absolute path
    if not os.path.isabs(gui_data_dir):
        gui_data_dir = os.path.abspath(gui_data_dir)
    
    # Check if directory exists
    if os.path.exists(gui_data_dir) and os.path.isdir(gui_data_dir):
        return gui_data_dir
    else:
        print(f"Warning: GUI default data directory '{gui_data_dir}' does not exist or is not a directory")
        return None

def get_auto_fix_interactive_commands(config_file: str = "config.txt") -> bool:
    """
    Get auto fix interactive commands configuration from configuration file
    
    Args:
        config_file: Path to the configuration file
        
    Returns:
        Boolean indicating whether to automatically fix interactive commands (default: False)
    """
    config = load_config(config_file)
    auto_fix_str = config.get('auto_fix_interactive_commands', 'False').lower()
    
    # Convert string to boolean
    if auto_fix_str in ('true', '1', 'yes', 'on'):
        return True
    elif auto_fix_str in ('false', '0', 'no', 'off'):
        return False
    else:
        print(f"Warning: Invalid auto_fix_interactive_commands value '{auto_fix_str}' in config file, using default False")
        return False

def get_web_search_summary(config_file: str = "config.txt") -> bool:
    """
    Get web search summary configuration from configuration file
    
    Args:
        config_file: Path to the configuration file
        
    Returns:
        Boolean indicating whether to enable AI summarization of web search results (default: True)
    """
    config = load_config(config_file)
    web_summary_str = config.get('web_search_summary', 'True').lower()
    
    # Convert string to boolean
    if web_summary_str in ('true', '1', 'yes', 'on'):
        return True
    elif web_summary_str in ('false', '0', 'no', 'off'):
        return False
    else:
        print(f"Warning: Invalid web_search_summary value '{web_summary_str}' in config file, using default True")
        return True