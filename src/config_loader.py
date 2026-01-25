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

import os
from typing import Dict, Optional, Tuple

# Global cache configuration
_config_cache: Dict[str, Dict[str, str]] = {}
_config_file_mtime: Dict[str, float] = {}

def clear_config_cache() -> None:
    """
    Clear configuration file cache
    """
    global _config_cache, _config_file_mtime
    _config_cache.clear()
    _config_file_mtime.clear()

def load_config(config_file: str = "config/config.txt", verbose: bool = False) -> Dict[str, str]:
    """
    Load configuration from config/config.txt file (with caching support)
    
    Args:
        config_file: Path to the configuration file
        verbose: Whether to print debug information
        
    Returns:
        Dictionary containing configuration key-value pairs
    """
    global _config_cache, _config_file_mtime
    
    # Check environment variable for app-specific config file
    if config_file == "config/config.txt" and os.environ.get('AGIA_CONFIG_FILE'):
        config_file = os.environ.get('AGIA_CONFIG_FILE')
    
    # Check if file exists
    if not os.path.exists(config_file):
        if verbose:
            print(f"Warning: Configuration file {config_file} not found")
        return {}
    
    try:
        # Get file modification time
        current_mtime = os.path.getmtime(config_file)
        
        # Check if cache is valid
        if (config_file in _config_cache and 
            config_file in _config_file_mtime and 
            _config_file_mtime[config_file] == current_mtime):
            if verbose:
                print(f"Using cached configuration for {config_file}")
            return _config_cache[config_file].copy()
        
        # Need to re-parse file
        if verbose:
            print(f"Loading configuration from {config_file}")
        
        config = {}
        
        with open(config_file, 'r', encoding='utf-8') as f:
            line_number = 0
            for line in f:
                line_number += 1
                original_line = line.rstrip('\n\r')  # Keep original line for debugging
                line = line.strip()
                
                # Skip empty lines
                if not line:
                    continue
                
                # 跳过纯注释行（以#开头的行）
                if line.startswith('#'):
                    if verbose:
                        print(f"Skipping commented line {line_number}: {original_line}")
                    continue
                
                # Process lines containing equals sign
                if '=' in line:
                    # 处理行内注释：在#之前分割
                    if '#' in line:
                        line = line.split('#')[0].strip()
                    
                    # Split key-value pairs
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip()
                    
                    if key:  # Ensure key is not empty
                        config[key] = value
                        if verbose:
                            print(f"Loaded config: {key} = {value}")
                    else:
                        if verbose:
                            print(f"Warning: Empty key found on line {line_number}: {original_line}")
                else:
                    if verbose:
                        print(f"Warning: Invalid config line {line_number} (no '=' found): {original_line}")
        
        # Update cache
        _config_cache[config_file] = config.copy()
        _config_file_mtime[config_file] = current_mtime
        
        return config
                    
    except Exception as e:
        print(f"Error reading configuration file {config_file}: {e}")
        return {}

def get_api_key(config_file: str = "config/config.txt") -> Optional[str]:
    """
    Get API key from environment variable or configuration file
    
    Args:
        config_file: Path to the configuration file
        
    Returns:
        API key string or None if not found
    """
    # Check environment variable first (for GUI override)
    env_value = os.environ.get('AGIBOT_API_KEY')
    if env_value:
        return env_value
    
    config = load_config(config_file)
    api_key = config.get('api_key')
    
    # If not found in config file, try AGIAGENT_API_KEY environment variable
    if not api_key:
        api_key = os.environ.get('AGIAGENT_API_KEY')
    
    return api_key

def get_api_base(config_file: str = "config/config.txt") -> Optional[str]:
    """
    Get API base URL from environment variable or configuration file
    
    Args:
        config_file: Path to the configuration file
        
    Returns:
        API base URL string or None if not found
    """
    # Check environment variable first (for GUI override)
    env_value = os.environ.get('AGIBOT_API_BASE')
    if env_value:
        return env_value
    
    config = load_config(config_file)
    api_base = config.get('api_base')
    
    # If not found in config file, try AGIAGENT_API_BASE environment variable
    if not api_base:
        api_base = os.environ.get('AGIAGENT_API_BASE')
    
    return api_base

def get_config_value(key: str, default: Optional[str] = None, config_file: str = "config/config.txt") -> Optional[str]:
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

def get_enable_round_sync(config_file: str = "config/config.txt") -> bool:
    """
    Get whether round synchronization barrier is enabled
    
    Args:
        config_file: Path to the configuration file
        
    Returns:
        True if enabled, False otherwise (default: False)
    """
    config = load_config(config_file)
    value = config.get('enable_round_sync', 'false').strip().lower()
    return value in ('1', 'true', 'yes', 'on')

def get_sync_round(config_file: str = "config/config.txt") -> int:
    """
    Get sync round step (N), number of rounds allowed per sync window
    
    Args:
        config_file: Path to the configuration file
        
    Returns:
        Integer N (default: 2)
    """
    config = load_config(config_file)
    value = config.get('sync_round', '2').strip()
    try:
        n = int(value)
        return max(1, n)
    except Exception:
        return 2

def get_model(config_file: str = "config/config.txt") -> Optional[str]:
    """
    Get model name from environment variable or configuration file
    
    Args:
        config_file: Path to the configuration file
        
    Returns:
        Model name string or None if not found
    """
    # Check environment variable first (for GUI override)
    env_value = os.environ.get('AGIBOT_MODEL')
    if env_value:
        return env_value
    
    config = load_config(config_file)
    model = config.get('model')
    
    # If not found in config file, try AGIAGENT_MODEL environment variable
    if not model:
        model = os.environ.get('AGIAGENT_MODEL')
    
    return model

def get_max_tokens(config_file: str = "config/config.txt") -> Optional[int]:
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
    
    # If not found in config file, try AGIAGENT_MAX_OUT_TOKENS environment variable
    if not max_tokens_str:
        max_tokens_env = os.environ.get('AGIAGENT_MAX_OUT_TOKENS')
        if max_tokens_env:
            try:
                return int(max_tokens_env)
            except ValueError:
                print(f"Warning: Invalid AGIAGENT_MAX_OUT_TOKENS value '{max_tokens_env}', must be an integer")
    
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

def get_streaming(config_file: str = "config/config.txt") -> bool:
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

def get_language(config_file: str = "config/config.txt") -> str:
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
    if lang in ('zh', 'zh-cn', 'chinese', 'Chinese'):
        return 'zh'
    elif lang in ('en', 'english', 'eng'):
        return 'en'
    else:
        print(f"Warning: Unsupported language '{lang}' in config file, defaulting to English")
        return 'en'

def get_truncation_length(config_file: str = "config/config.txt") -> int:
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
#     # If not set
#     main_truncation = get_truncation_length(config_file)
#     return max(1000, main_truncation // 10)

def get_web_content_truncation_length(config_file: str = "config/config.txt") -> int:
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
    
    # If not set, use 5 times the main truncation length, but not less than 50000
    main_truncation = get_truncation_length(config_file)
    return max(50000, main_truncation * 5)

def get_compression_min_length(config_file: str = "config/config.txt") -> int:
    """
    Get compression min length from configuration file
    
    Args:
        config_file: Path to the configuration file
        
    Returns:
        Compression min length integer (default: 500)
    """
    config = load_config(config_file)
    compression_min_length_str = config.get('compression_min_length')
    
    if compression_min_length_str:
        try:
            compression_min_length = int(compression_min_length_str)
            if compression_min_length <= 0:
                print(f"Warning: Invalid compression_min_length value '{compression_min_length_str}' in config file, must be positive integer, using default 500")
                return 500
            return compression_min_length
        except ValueError:
            print(f"Warning: Invalid compression_min_length value '{compression_min_length_str}' in config file, must be an integer, using default 500")
            return 500
    
    return 500  # Default compression min length

def get_compression_head_length(config_file: str = "config/config.txt") -> int:
    """
    Get compression head length from configuration file
    
    Args:
        config_file: Path to the configuration file
        
    Returns:
        Compression head length integer (default: 100)
    """
    config = load_config(config_file)
    compression_head_length_str = config.get('compression_head_length')
    
    if compression_head_length_str:
        try:
            compression_head_length = int(compression_head_length_str)
            if compression_head_length <= 0:
                print(f"Warning: Invalid compression_head_length value '{compression_head_length_str}' in config file, must be positive integer, using default 100")
                return 100
            return compression_head_length
        except ValueError:
            print(f"Warning: Invalid compression_head_length value '{compression_head_length_str}' in config file, must be an integer, using default 100")
            return 100
    
    return 100  # Default compression head length

def get_compression_tail_length(config_file: str = "config/config.txt") -> int:
    """
    Get compression tail length from configuration file
    
    Args:
        config_file: Path to the configuration file
        
    Returns:
        Compression tail length integer (default: 100)
    """
    config = load_config(config_file)
    compression_tail_length_str = config.get('compression_tail_length')
    
    if compression_tail_length_str:
        try:
            compression_tail_length = int(compression_tail_length_str)
            if compression_tail_length <= 0:
                print(f"Warning: Invalid compression_tail_length value '{compression_tail_length_str}' in config file, must be positive integer, using default 100")
                return 100
            return compression_tail_length
        except ValueError:
            print(f"Warning: Invalid compression_tail_length value '{compression_tail_length_str}' in config file, must be an integer, using default 100")
            return 100
    
    return 100  # Default compression tail length

def get_summary_history(config_file: str = "config/config.txt") -> bool:
    """
    Get summary history configuration from configuration file
    
    Args:
        config_file: Path to the configuration file
        
    Returns:
        Boolean indicating whether to enable history summarization (default: True)
    """
    config = load_config(config_file)
    summary_history_str = config.get('summary_history', 'True').lower()
    
    # Convert string to boolean
    if summary_history_str in ('true', '1', 'yes', 'on'):
        return True
    elif summary_history_str in ('false', '0', 'no', 'off'):
        return False
    else:
        print(f"Warning: Invalid summary_history value '{summary_history_str}' in config file, using default True")
        return True

def get_summary_max_length(config_file: str = "config/config.txt") -> int:
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

def get_summary_trigger_length(config_file: str = "config/config.txt") -> int:
    """
    Get summary trigger length from configuration file
    
    Args:
        config_file: Path to the configuration file
        
    Returns:
        Summary trigger length integer (default: 100000)
    """
    config = load_config(config_file)
    summary_trigger_length_str = config.get('summary_trigger_length')
    
    if summary_trigger_length_str:
        try:
            summary_trigger_length = int(summary_trigger_length_str)
            if summary_trigger_length <= 0:
                print(f"Warning: Invalid summary_trigger_length value '{summary_trigger_length_str}' in config file, must be positive integer, using default 100000")
                return 100000
            return summary_trigger_length
        except ValueError:
            print(f"Warning: Invalid summary_trigger_length value '{summary_trigger_length_str}' in config file, must be an integer, using default 100000")
            return 100000
    
    return 100000  # Default summary trigger length

def get_compression_target_length(config_file: str = "config/config.txt") -> int:
    """
    Get compression target length from configuration file
    When compression is triggered, history will be compressed to this target length
    (usually smaller than summary_trigger_length to prevent repeated compression cycles)
    
    Args:
        config_file: Path to the configuration file
    
    Returns:
        Compression target length integer (default: 70% of summary_trigger_length)
    """
    config = load_config(config_file)
    compression_target_length_str = config.get('compression_target_length')
    
    if compression_target_length_str:
        try:
            compression_target_length = int(compression_target_length_str)
            if compression_target_length <= 0:
                # If invalid, calculate default as 70% of trigger length
                trigger_length = get_summary_trigger_length(config_file)
                default_value = int(trigger_length * 0.7)
                print(f"Warning: Invalid compression_target_length value '{compression_target_length_str}' in config file, must be positive integer, using default {default_value} (70% of summary_trigger_length)")
                return default_value
            return compression_target_length
        except ValueError:
            # If invalid, calculate default as 70% of trigger length
            trigger_length = get_summary_trigger_length(config_file)
            default_value = int(trigger_length * 0.7)
            print(f"Warning: Invalid compression_target_length value '{compression_target_length_str}' in config file, must be an integer, using default {default_value} (70% of summary_trigger_length)")
            return default_value
    
    # Default: 70% of summary_trigger_length
    trigger_length = get_summary_trigger_length(config_file)
    return int(trigger_length * 0.7)

def get_simplified_search_output(config_file: str = "config/config.txt") -> bool:
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

def get_summary_report(config_file: str = "config/config.txt") -> bool:
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

def get_gui_default_data_directory(config_file: str = "config/config.txt") -> Optional[str]:
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

def get_auto_fix_interactive_commands(config_file: str = "config/config.txt") -> bool:
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

def get_web_search_summary(config_file: str = "config/config.txt") -> bool:
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

def get_multi_agent(config_file: str = "config/config.txt") -> bool:
    """
    Get multi-agent configuration from configuration file or environment variable
    
    Args:
        config_file: Path to the configuration file
        
    Returns:
        Boolean indicating whether multi-agent mode is enabled (default: True)
    """
    # Check environment variable first (for GUI override)
    env_value = os.environ.get('AGIBOT_MULTI_AGENT')
    if env_value is not None:
        env_value_lower = env_value.lower()
        if env_value_lower in ('true', '1', 'yes', 'on'):
            return True
        elif env_value_lower in ('false', '0', 'no', 'off'):
            return False
    
    # Fall back to config file
    config = load_config(config_file)
    multi_agent_str = config.get('multi_agent', 'True').lower()
    
    # Convert string to boolean
    if multi_agent_str in ('true', '1', 'yes', 'on'):
        return True
    elif multi_agent_str in ('false', '0', 'no', 'off'):
        return False
    else:
        # Default to True if invalid value
        return True

def get_enable_jieba(config_file: str = "config/config.txt") -> bool:
    """
    Get jieba Chinese segmentation configuration from configuration file or environment variable
    
    Args:
        config_file: Path to the configuration file
        
    Returns:
        Boolean indicating whether jieba Chinese segmentation is enabled (default: False)
    """
    # Check environment variable first (for GUI override)
    env_value = os.environ.get('AGIBOT_ENABLE_JIEBA')
    if env_value is not None:
        env_value_lower = env_value.lower()
        if env_value_lower in ('true', '1', 'yes', 'on'):
            return True
        elif env_value_lower in ('false', '0', 'no', 'off'):
            return False
    
    # Fall back to config file
    config = load_config(config_file)
    enable_jieba_str = config.get('enable_jieba', 'False').lower()
    
    # Convert string to boolean
    if enable_jieba_str in ('true', '1', 'yes', 'on'):
        return True
    elif enable_jieba_str in ('false', '0', 'no', 'off'):
        return False
    else:
        # Default to False if invalid value
        return False

def get_emoji_disabled(config_file: str = "config/config.txt") -> bool:
    """
    Get emoji display configuration from configuration file or environment variable
    
    Args:
        config_file: Path to the configuration file
        
    Returns:
        Boolean indicating whether emoji display is disabled (default: False, meaning emoji enabled)
    """
    # Check environment variable first (for GUI override)
    env_value = os.environ.get('AGIBOT_EMOJI_DISABLED')
    if env_value is not None:
        env_value_lower = env_value.lower()
        if env_value_lower in ('true', '1', 'yes', 'on'):
            return True
        elif env_value_lower in ('false', '0', 'no', 'off'):
            return False
    
    # Fall back to config file
    config = load_config(config_file)
    emoji_disabled_str = config.get('emoji_disabled', 'False').lower()
    
    # Convert string to boolean
    if emoji_disabled_str in ('true', '1', 'yes', 'on'):
        return True
    elif emoji_disabled_str in ('false', '0', 'no', 'off'):
        return False
    else:
        # Default to False if invalid value (emoji enabled)
        return False



def get_tool_calling_format(config_file: str = "config/config.txt") -> bool:
    """
    Get tool calling format configuration from configuration file
    
    Args:
        config_file: Path to the configuration file
        
    Returns:
        Boolean indicating whether to use standard tool calling (True) or chat-based tool calling (False)
        Default: True (use standard tool calling when available)
    """
    config = load_config(config_file)
    tool_calling_format_str = config.get('Tool_calling_format', 'True').lower()
    
    # Convert string to boolean
    if tool_calling_format_str in ('true', '1', 'yes', 'on'):
        return True
    elif tool_calling_format_str in ('false', '0', 'no', 'off'):
        return False
    else:
        # Default to True if invalid value
        return True

def get_enable_thinking(config_file: str = "config/config.txt") -> bool:
    """
    Get thinking support configuration from configuration file
    
    Args:
        config_file: Path to the configuration file
        
    Returns:
        Boolean indicating whether to enable and display thinking process from reasoning models
        Default: True (enabled for models that support thinking)
    """
    config = load_config(config_file)
    enable_thinking_str = config.get('enable_thinking', 'True').lower()
    
    # Convert string to boolean
    if enable_thinking_str in ('true', '1', 'yes', 'on'):
        return True
    elif enable_thinking_str in ('false', '0', 'no', 'off'):
        return False
    else:
        # Default to True if invalid value
        return True

def get_tool_call_parse_format(config_file: str = "config/config.txt") -> str:
    """
    Get tool call parsing format configuration from configuration file
    
    Args:
        config_file: Path to the configuration file
        
    Returns:
        String indicating which format to try first: "json" or "xml"
        Default: "json"
    """
    config = load_config(config_file)
    parse_format_str = config.get('tool_call_parse_format', 'json').lower().strip()
    
    # Validate and return format
    if parse_format_str in ('json', 'xml'):
        return parse_format_str
    else:
        print(f"Warning: Invalid tool_call_parse_format value '{parse_format_str}' in config file, must be 'json' or 'xml', using default 'json'")
        return 'json'

def get_gui_config(config_file: str = "config/config.txt") -> Dict[str, Optional[str]]:
    """
    Get GUI API configuration from configuration file
    
    Reads the GUI API configuration section which should contain:
    - api_key: API key for the model
    - api_base: Base URL for the API
    - model: Model name (can be overridden by GUI selection)
    
    Args:
        config_file: Path to the configuration file
        
    Returns:
        Dictionary containing GUI configuration values
    """
    config = load_config(config_file)
    
    # Parse the config file to find GUI API configuration section
    gui_config = {}
    
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        in_gui_section = False
        for line in lines:
            line = line.strip()
            
            # Skip empty lines
            if not line:
                continue
                
            # Check for GUI API configuration section
            if line.startswith('# GUI API configuration'):
                in_gui_section = True
                continue
            
            # Check if we've reached another section
            if line.startswith('#') and 'configuration' in line and in_gui_section:
                # We've moved to another configuration section
                break
                
            # If we're in the GUI section and find a config line
            if in_gui_section and '=' in line and not line.startswith('#'):
                # Handle inline comments
                if '#' in line:
                    line = line.split('#')[0].strip()
                
                key, value = line.split('=', 1)
                key = key.strip()
                value = value.strip()
                
                if key in ['api_key', 'api_base', 'model', 'max_tokens']:
                    gui_config[key] = value
                    
    except Exception as e:
        print(f"Error reading GUI configuration from {config_file}: {e}")
        
    return gui_config

def validate_gui_config(gui_config: Dict[str, Optional[str]]) -> Tuple[bool, str]:
    """
    Validate GUI configuration
    
    Args:
        gui_config: Dictionary containing GUI configuration
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    api_key = gui_config.get('api_key')
    api_base = gui_config.get('api_base')
    
    # Check if API key is set and not the default placeholder
    if not api_key or api_key.strip() == 'your key':
        return False, "Invalid API Key configuration. Please check the GUI API configuration section in config/config.txt."
    
    # Check if API base is set
    if not api_base or api_base.strip() == '':
        return False, "Invalid API Base configuration. Please check the GUI API configuration section in config/config.txt."
    return True, ""

def get_vision_model(config_file: str = "config/config.txt") -> Optional[str]:
    """
    Get vision model name from configuration file
    Falls back to main model if not configured
    
    Args:
        config_file: Path to the configuration file
        
    Returns:
        Vision model name string or None
    """
    config = load_config(config_file)
    vision_model = config.get('vision_model')
    
    # If vision_model is not set or is placeholder, return main model
    if not vision_model or vision_model.strip() == '' or vision_model.strip() == 'your key':
        return get_model(config_file)
    
    return vision_model

def get_vision_api_key(config_file: str = "config/config.txt") -> Optional[str]:
    """
    Get vision API key from configuration file
    Falls back to main API key if not configured
    
    Args:
        config_file: Path to the configuration file
        
    Returns:
        Vision API key string or None
    """
    config = load_config(config_file)
    vision_api_key = config.get('vision_api_key')
    
    # If vision_api_key is not set or is placeholder, return main api_key
    if not vision_api_key or vision_api_key.strip() == '' or vision_api_key.strip() == 'your key':
        return get_api_key(config_file)
    
    return vision_api_key

def get_vision_api_base(config_file: str = "config/config.txt") -> Optional[str]:
    """
    Get vision API base URL from configuration file
    Falls back to main API base if not configured
    
    Args:
        config_file: Path to the configuration file
        
    Returns:
        Vision API base URL string or None
    """
    config = load_config(config_file)
    vision_api_base = config.get('vision_api_base')
    
    # If vision_api_base is not set or is placeholder, return main api_base
    if not vision_api_base or vision_api_base.strip() == '':
        return get_api_base(config_file)
    
    return vision_api_base

def get_vision_max_tokens(config_file: str = "config/config.txt") -> Optional[int]:
    """
    Get vision max_tokens from configuration file
    Falls back to main max_tokens if not configured
    
    Args:
        config_file: Path to the configuration file
        
    Returns:
        Vision max tokens integer or None
    """
    config = load_config(config_file)
    vision_max_tokens_str = config.get('vision_max_tokens')
    
    # If vision_max_tokens is set, use it
    if vision_max_tokens_str:
        try:
            return int(vision_max_tokens_str)
        except ValueError:
            print(f"Warning: Invalid vision_max_tokens value '{vision_max_tokens_str}' in config file, using main max_tokens")
    
    # Fall back to main max_tokens
    return get_max_tokens(config_file)

def has_vision_config(config_file: str = "config/config.txt") -> bool:
    """
    Check if vision-specific configuration exists
    
    Args:
        config_file: Path to the configuration file
        
    Returns:
        True if vision config is set, False if using main model config
    """
    config = load_config(config_file)
    vision_model = config.get('vision_model')
    
    # Check if vision_model is properly configured
    if vision_model and vision_model.strip() != '' and vision_model.strip() != 'your key':
        return True
    
    return False

def get_image_generation_api_key(config_file: str = "config/config.txt") -> Optional[str]:
    """
    Get image generation API key from configuration file
    Falls back to main API key if not configured
    
    Args:
        config_file: Path to the configuration file
        
    Returns:
        Image generation API key string or None
    """
    config = load_config(config_file)
    image_generation_api_key = config.get('image_generation_api_key')
    
    # If image_generation_api_key is not set or is placeholder, return main api_key
    if not image_generation_api_key or image_generation_api_key.strip() == '' or image_generation_api_key.strip() == 'your key':
        return get_api_key(config_file)
    
    return image_generation_api_key

def get_image_generation_api_base(config_file: str = "config/config.txt") -> Optional[str]:
    """
    Get image generation API base URL from configuration file
    Falls back to main API base if not configured
    
    Args:
        config_file: Path to the configuration file
        
    Returns:
        Image generation API base URL string or None
    """
    config = load_config(config_file)
    image_generation_api_base = config.get('image_generation_api_base')
    
    # If image_generation_api_base is not set or is placeholder, return main api_base
    if not image_generation_api_base or image_generation_api_base.strip() == '':
        return get_api_base(config_file)
    
    return image_generation_api_base

def get_image_generation_model(config_file: str = "config/config.txt") -> Optional[str]:
    """
    Get image generation model name from configuration file
    Defaults to google/gemini-3-pro-image-preview if not configured
    
    Args:
        config_file: Path to the configuration file
        
    Returns:
        Image generation model name string or default
    """
    config = load_config(config_file)
    image_generation_model = config.get('image_generation_model')
    
    # If image_generation_model is not set or is placeholder, return default
    if not image_generation_model or image_generation_model.strip() == '' or image_generation_model.strip() == 'your key':
        return 'google/gemini-3-pro-image-preview'  # Default model
    
    return image_generation_model

def get_svg_optimizer_api_key(config_file: str = "config/config.txt") -> Optional[str]:
    """
    Get SVG optimizer API key from configuration file
    Falls back to main API key if not configured
    
    Args:
        config_file: Path to the configuration file
        
    Returns:
        SVG optimizer API key string or None
    """
    config = load_config(config_file)
    svg_optimizer_api_key = config.get('svg_optimizer_api_key')
    
    # If svg_optimizer_api_key is not set or is placeholder, return main api_key
    if not svg_optimizer_api_key or svg_optimizer_api_key.strip() == '' or svg_optimizer_api_key.strip() == 'your key':
        return get_api_key(config_file)
    
    return svg_optimizer_api_key

def get_svg_optimizer_api_base(config_file: str = "config/config.txt") -> Optional[str]:
    """
    Get SVG optimizer API base URL from configuration file
    Falls back to main API base if not configured
    
    Args:
        config_file: Path to the configuration file
        
    Returns:
        SVG optimizer API base URL string or None
    """
    config = load_config(config_file)
    svg_optimizer_api_base = config.get('svg_optimizer_api_base')
    
    # If svg_optimizer_api_base is not set or is placeholder, return main api_base
    if not svg_optimizer_api_base or svg_optimizer_api_base.strip() == '':
        return get_api_base(config_file)
    
    return svg_optimizer_api_base

def get_svg_optimizer_model(config_file: str = "config/config.txt") -> Optional[str]:
    """
    Get SVG optimizer model name from configuration file
    Defaults to google/gemini-2.0-flash-exp:free if not configured
    
    Args:
        config_file: Path to the configuration file
        
    Returns:
        SVG optimizer model name string or default
    """
    config = load_config(config_file)
    svg_optimizer_model = config.get('svg_optimizer_model')
    
    # If svg_optimizer_model is not set or is placeholder, return default
    if not svg_optimizer_model or svg_optimizer_model.strip() == '':
        return 'google/gemini-2.0-flash-exp:free'  # Default model
    
    return svg_optimizer_model

def get_admit_task_completed_with_tools(config_file: str = "config/config.txt") -> bool:
    """
    Get TASK_COMPLETED signal handling configuration from configuration file
    
    Controls whether to admit TASK_COMPLETED signal when it appears along with tool calls.
    When enabled (True), if LLM outputs TASK_COMPLETED along with tool calls, the task will be completed immediately.
    When disabled (False), TASK_COMPLETED signal will be dropped and tools will be executed.
    
    Args:
        config_file: Path to the configuration file
        
    Returns:
        Boolean indicating whether to admit TASK_COMPLETED with tools (default: False)
    """
    config = load_config(config_file)
    admit_str = config.get('admit_task_completed_with_tools', 'False').lower()
    
    # Convert string to boolean
    if admit_str in ('true', '1', 'yes', 'on'):
        return True
    elif admit_str in ('false', '0', 'no', 'off'):
        return False
    else:
        # Default to False if invalid value
        return False

def get_temperature(config_file: str = "config/config.txt") -> float:
    """
    Get temperature parameter from configuration file
    
    Controls randomness in output (0.0 to 2.0)
    Lower values: More deterministic, focused output
    Higher values: More creative and diverse output
    
    Args:
        config_file: Path to the configuration file
        
    Returns:
        Temperature float value (default: 0.7)
    """
    config = load_config(config_file)
    temperature_str = config.get('temperature', '0.7').strip()
    
    try:
        temperature = float(temperature_str)
        # Validate range
        if temperature < 0.0 or temperature > 2.0:
            print(f"Warning: Invalid temperature value '{temperature_str}' in config file, must be between 0.0 and 2.0, using default 0.7")
            return 0.7
        return temperature
    except ValueError:
        print(f"Warning: Invalid temperature value '{temperature_str}' in config file, must be a float, using default 0.7")
        return 0.7

def get_top_p(config_file: str = "config/config.txt") -> float:
    """
    Get top_p (nucleus sampling) parameter from configuration file
    
    Controls diversity via nucleus sampling (0.0 to 1.0)
    Lower values: More focused, conservative output
    Higher values: More diverse output
    
    Args:
        config_file: Path to the configuration file
        
    Returns:
        Top-p float value (default: 0.8)
    """
    config = load_config(config_file)
    top_p_str = config.get('top_p', '0.8').strip()
    
    try:
        top_p = float(top_p_str)
        # Validate range
        if top_p < 0.0 or top_p > 1.0:
            print(f"Warning: Invalid top_p value '{top_p_str}' in config file, must be between 0.0 and 1.0, using default 0.8")
            return 0.8
        return top_p
    except ValueError:
        print(f"Warning: Invalid top_p value '{top_p_str}' in config file, must be a float, using default 0.8")
        return 0.8

def get_all_model_configs(config_file: str = "config/config.txt") -> list:
    """
    Get all model configurations from config file (including commented ones)
    
    Parses the config file to find all model configuration blocks, including those
    that are commented out. Each configuration block should contain:
    - api_key: API key for the model
    - api_base: Base URL for the API
    - model: Model name
    - max_tokens: Maximum tokens (optional)
    
    Args:
        config_file: Path to the configuration file
        
    Returns:
        List of dictionaries, each containing a model configuration with keys:
        - name: Display name (from comment or model name)
        - api_key: API key
        - api_base: API base URL
        - model: Model name
        - max_tokens: Max tokens (default: 8192)
        - enabled: Whether this configuration is currently active (not commented)
    """
    all_configs = []
    
    if not os.path.exists(config_file):
        return all_configs
    
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # Find the start marker: "# Model configuration section"
        start_index = -1
        for i, line in enumerate(lines):
            if line.strip() == '# Model configuration section':
                start_index = i
                break
        
        # If start marker not found, return empty list
        if start_index == -1:
            return []
        
        # Start parsing from the line after the start marker
        i = start_index + 1
        while i < len(lines):
            line = lines[i].rstrip('\n\r')
            stripped_line = line.strip()
            
            # Stop parsing if we reach the end marker
            if stripped_line == '# This is the end of model config.':
                break
            
            # Check for configuration section headers
            # Look for lines starting with # that contain configuration-related keywords
            # and are followed by api_key/api_base/model lines
            # But exclude lines that are actual config lines (contain = immediately after #)
            is_config_header = False
            if stripped_line.startswith('#'):
                # Skip if this is a config line (contains = after removing #)
                header_without_hash = stripped_line[1:].strip()
                if '=' in header_without_hash and not header_without_hash.startswith('#'):
                    # This is a config line, not a header
                    i += 1
                    continue
                
                header_without_hash = stripped_line[1:].strip()
                # Check if next few lines contain config keys (api_key, api_base, model)
                # If a comment line is followed by config items, it's a configuration section header
                if i + 1 < len(lines):
                    next_lines = ''.join(lines[i+1:min(i+5, len(lines))])  # Check next 4 lines
                    if any(key in next_lines for key in ['api_key', 'api_base', 'model=']):
                        is_config_header = True
            
            if is_config_header:
                # Extract section name
                section_name = stripped_line.replace('#', '').strip()
                
                # Extract provider name for display
                display_name = section_name
                if 'API configuration' in section_name:
                    parts = section_name.split()
                    if len(parts) > 0:
                        if parts[0] == 'GUI':
                            display_name = 'GUI' + ((' ' + parts[1]) if len(parts) > 1 else '')
                        else:
                            display_name = parts[0]
                elif section_name.endswith('models'):
                    display_name = section_name.replace(' models', '').replace(' model', '')
                elif section_name.startswith('# '):
                    display_name = section_name[2:]
                
                # Start collecting config from next lines
                i += 1
                current_config = {'name': display_name}
                section_enabled = True  # Track if section is enabled
                found_model = False
                found_api_base = False
                last_line_was_commented = False
                
                # Read config lines until we hit another section or non-config content
                while i < len(lines):
                    line = lines[i].rstrip('\n\r')
                    stripped_line = line.strip()
                    
                    # Stop if we hit the end marker
                    if stripped_line == '# This is the end of model config.':
                        break
                    
                    # Stop if we hit another configuration section header
                    # Check if this is a header (not a config line)
                    if stripped_line.startswith('#'):
                        header_without_hash = stripped_line[1:].strip()
                        # If it's a header (no = sign) and followed by config keys, it's a new section
                        if '=' not in header_without_hash:
                            # Check if next lines contain config keys - if so, this is a new config section
                            if i + 1 < len(lines):
                                next_lines = ''.join(lines[i+1:min(i+5, len(lines))])
                                if any(key in next_lines for key in ['api_key', 'api_base', 'model=']):
                                    # If we've already found model and api_base, this is definitely a new section
                                    if found_model and found_api_base:
                                        break
                                    # If it's followed by config items, it's a new section header
                                    break
                    
                    # Stop if we hit a non-empty, non-comment line that's not a config line
                    # (but allow config lines even if commented)
                    if stripped_line and not stripped_line.startswith('#') and '=' not in stripped_line:
                        # Check if we've collected enough config (at least model and api_base)
                        if found_model and found_api_base:
                            break
                    
                    # Process configuration lines (including commented ones)
                    if '=' in stripped_line:
                        # Check if line is commented
                        line_is_commented = stripped_line.startswith('#')
                        last_line_was_commented = line_is_commented
                        config_line = stripped_line
                        if line_is_commented:
                            config_line = config_line[1:].strip()
                            # If first config line is commented, whole section is likely commented
                            if not found_model and not found_api_base:
                                section_enabled = False
                        
                        # Handle inline comments
                        if '#' in config_line:
                            config_line = config_line.split('#')[0].strip()
                        
                        if '=' in config_line:
                            key, value = config_line.split('=', 1)
                            key = key.strip()
                            value = value.strip()
                            
                            if key == 'api_key':
                                current_config['api_key'] = value
                            elif key == 'api_base':
                                current_config['api_base'] = value
                                found_api_base = True
                            elif key == 'model':
                                current_config['model'] = value
                                found_model = True
                                # Use model name as fallback display name
                                if not current_config.get('name') or current_config['name'] == 'Unknown':
                                    current_config['name'] = value
                            elif key == 'max_tokens':
                                try:
                                    current_config['max_tokens'] = int(value)
                                except ValueError:
                                    current_config['max_tokens'] = 8192
                    
                    i += 1
                
                # Save config if it has required fields
                # Only save if the config is enabled (not fully commented out)
                if current_config.get('model') and current_config.get('api_base'):
                    current_config['enabled'] = section_enabled and not last_line_was_commented
                    # Set default max_tokens if not set
                    if 'max_tokens' not in current_config:
                        current_config['max_tokens'] = 8192
                    # Only include enabled (non-commented) configurations
                    if current_config['enabled']:
                        all_configs.append(current_config.copy())
                
                # Continue to next iteration (don't increment i here as it's already incremented)
                continue
            
            i += 1
        
        # Remove duplicates and filter invalid configs
        seen = set()
        unique_configs = []
        for config in all_configs:
            model = config.get('model', '').strip()
            api_base = config.get('api_base', '').strip()
            api_key = config.get('api_key', '').strip()
            
            # Skip if missing required fields
            if not model or not api_base:
                continue
            
            # Include all configs, even with placeholder api_key values
            # Users can still select them and provide their own keys
            
            key = (model, api_base)
            if key not in seen:
                seen.add(key)
                # Create display name: provider name + model name
                name = config.get('name', model)
                if name != model and name != 'Unknown':
                    config['display_name'] = f"{name} - {model}"
                else:
                    config['display_name'] = model
                unique_configs.append(config)
        
        return unique_configs
        
    except Exception as e:
        print(f"Error reading all model configurations from {config_file}: {e}")
        import traceback
        traceback.print_exc()
        return []


def get_zhipu_search_api_key(config_file: str = "config/config.txt") -> Optional[str]:
    """
    Get Zhipu AI web search API key from configuration

    Args:
        config_file: Path to the configuration file

    Returns:
        Zhipu search API key if configured, None otherwise
    """
    return get_config_value("zhipu_search_api_key", config_file=config_file)


def get_zhipu_search_engine(config_file: str = "config/config.txt") -> str:
    """
    Get Zhipu AI web search engine configuration

    Args:
        config_file: Path to the configuration file

    Returns:
        Search engine type, defaults to "search_std"
    """
    return get_config_value("zhipu_search_engine", default="search_std", config_file=config_file)
