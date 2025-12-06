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
    return config.get('api_key')

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
    return config.get('api_base')

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
    return config.get('model')

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