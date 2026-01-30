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

Configuration Management Module
"""

import os
import json
import time
import shutil
from typing import Dict, Any, Tuple, Optional, List, Union
from pathlib import Path

from .exceptions import ConfigError
from .logger import get_logger
from .config_validator import ConfigValidator

logger = get_logger(__name__)


class ConfigLoader:
    """Configuration loader class for unified management of all configuration parameters"""

    def __init__(self, config_file: Optional[Union[str, List[str]]] = None, auto_reload: bool = False):
        """
        Initialize configuration loader

        Args:
            config_file: Configuration file path(s), can be a string or list of strings.
                        If None, will use default: ["config/config.txt", "config/config_mem.txt"]
            auto_reload: Whether to enable auto-reload
        """
        # Handle different input types and set default config files
        if config_file is None:
            # Default configuration: try to find config files in standard locations
            self.config_files = self._get_default_config_files()
        elif isinstance(config_file, str):
            # Single config file (backward compatibility)
            self.config_files = [config_file]
        elif isinstance(config_file, list):
            # Multiple config files
            self.config_files = config_file
        else:
            raise ValueError("config_file must be a string, list of strings, or None")
        
        # For backward compatibility, keep config_file attribute
        self.config_file = self.config_files[0] if self.config_files else "config.txt"
        
        self.auto_reload = auto_reload
        self.last_modified = {}  # Track modification times for each file
        self.config = {}
        self.env_config = {}

        # Load environment variable configuration
        self._load_env_config()

        # Load configuration files
        self._load_config()

        # Validate configuration
        self._validate_config()

    def _get_default_config_files(self) -> List[str]:
        """
        Get default configuration file paths
        
        Returns:
            List of default configuration file paths
        """
        # Try to determine project root directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Navigate up to find project root (look for config directory)
        project_root = current_dir
        for _ in range(10):  # Limit search depth
            config_dir = os.path.join(project_root, "config")
            if os.path.exists(config_dir):
                break
            parent = os.path.dirname(project_root)
            if parent == project_root:  # Reached filesystem root
                break
            project_root = parent
        else:
            # Fallback: assume config is in current directory or parent directories
            project_root = current_dir
        
        # Default config files in order of preference
        default_files = [
            os.path.join(project_root, "config", "config.txt"),
            os.path.join(project_root, "config", "config_memory.txt"),
            # Fallback options
            os.path.join(project_root, "config.txt"),
            "config.txt"
        ]
        
        # Only return files that exist, or at least the first one as fallback
        existing_files = [f for f in default_files if os.path.exists(f)]
        if existing_files:
            return existing_files
        else:
            # Return the first default file even if it doesn't exist
            return [default_files[0]]

    def _load_env_config(self) -> Dict[str, str]:
        """Load configuration from environment variables"""
        env_config = {}

        # Define environment variable mapping
        env_mapping = {
            "MEM_API_KEY": "api_key",
            "MEM_API_BASE": "api_base",
            "MEM_MODEL": "mem_model",
            "MEM_EMBEDDING_MODEL": "embedding_model",
            "MEM_AGENT_MODEL": "agent_model",
            "MEM_MAX_TOKENS": "max_tokens",
            "MEM_SIMILARITY_THRESHOLD": "similarity_threshold",
            "MEM_DEFAULT_TOP_K": "default_top_k",
            "MEM_EMBEDDING_WEIGHT": "default_embedding_weight",
            "MEM_TFIDF_WEIGHT": "default_tfidf_weight",
            "MEM_MAX_DAYS_BACK": "default_max_days_back",
            "MEM_SORT_BY": "default_sort_by",
            "MEM_LLM_TOOL_TEMPERATURE": "llm_tool_temperature",
            "MEM_LLM_TOOL_MAX_TOKENS": "llm_tool_max_tokens",
            "MEM_LLM_TOOL_TOP_P": "llm_tool_top_p",
            "MEM_LLM_TIMEOUT": "llm_timeout",
            "MEM_STREAMING": "streaming",
            "MEM_LANG": "LANG",
            "MEM_LOG_ENABLED": "log_enabled"
        }

        for env_var, config_key in env_mapping.items():
            value = os.getenv(env_var)
            if value:
                env_config[config_key] = value
                logger.debug(f"Loaded configuration from environment variable: {config_key} = {value}")

        self.env_config = env_config
        return env_config

    def _load_config(self) -> Dict[str, str]:
        """
        Read configuration files and merge their contents

        Returns:
            Merged configuration dictionary
        """
        config = {}
        
        # Load each configuration file in order
        for config_file in self.config_files:
            file_config = self._load_single_config_file(config_file)
            if file_config:
                # Merge configuration (later files override earlier ones)
                config.update(file_config)
        
        # If no config was loaded, use default configuration
        if not config:
            logger.warning("No configuration files could be loaded, using default configuration")
            config = self._get_default_config()

        # Merge environment variable configuration (environment variables have highest priority)
        merged_config = config.copy()
        merged_config.update(self.env_config)

        self.config = merged_config
        return merged_config

    def _load_single_config_file(self, config_file: str) -> Optional[Dict[str, str]]:
        """
        Load a single configuration file

        Args:
            config_file: Path to configuration file

        Returns:
            Configuration dictionary or None if failed
        """
        config = {}
        try:
            # Check if file exists
            if not os.path.exists(config_file):
                logger.debug(f"Configuration file does not exist: {config_file}")
                return None
            
            # Check if file has been modified (for auto-reload)
            current_modified = os.path.getmtime(config_file)
            if self.auto_reload and config_file in self.last_modified:
                if current_modified > self.last_modified[config_file]:
                    logger.info(f"Configuration file update detected, reloading: {config_file}")
            
            self.last_modified[config_file] = current_modified

            # Read configuration file
            with open(config_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    # Skip comments and empty lines
                    if line and not line.startswith('#') and '=' in line:
                        try:
                            key, value = line.split('=', 1)
                            config[key.strip()] = value.strip()
                        except ValueError as e:
                            logger.warning(f"Configuration file {config_file} line {line_num} format error: {line}")

            logger.info(f"Successfully loaded configuration file: {config_file}")
            return config

        except Exception as e:
            logger.error(f"Failed to read configuration file {config_file}: {e}")
            return None

    def reload_config(self) -> bool:
        """
        Reload configuration

        Returns:
            Whether reload was successful
        """
        try:
            logger.info("Starting configuration reload...")

            # Reload environment variables
            self._load_env_config()

            # Reload configuration file
            self._load_config()

            # Re-validate configuration
            self._validate_config()

            logger.info("Configuration reload completed")
            return True
        except Exception as e:
            logger.error(f"Configuration reload failed: {e}")
            return False

    def _validate_config(self):
        """Validate configuration"""
        try:
            validation_result = self.validate_config()
            if not validation_result['valid']:
                logger.error("Configuration validation failed:")
                for error in validation_result['errors']:
                    logger.error(f"  - {error}")

                # If required configuration error, throw exception
                if any('Missing required configuration' in error for error in validation_result['errors']):
                    raise ConfigError("Configuration validation failed, missing required configuration items")

            if validation_result['warnings']:
                logger.warning("Configuration warnings:")
                for warning in validation_result['warnings']:
                    logger.warning(f"  - {warning}")

        except Exception as e:
            logger.error(f"Error occurred during configuration validation: {e}")
            # Don't throw exception, allow using default configuration

    def _get_default_config(self) -> Dict[str, str]:
        """Get default configuration"""
        return {
            # API configuration
            'api_key': '',
            'api_base': 'https://api.siliconflow.cn/v1',
            'mem_model': 'Pro/Qwen/Qwen2.5-7B-Instruct',
            'max_tokens': '4096',
            'embedding_model': 'Pro/BAAI/bge-m3',
            'agent_model': 'deepseek-ai/DeepSeek-V3',
            'streaming': 'True',

            # Truncation configuration
            'truncation_length': '10000',
            'history_truncation_length': '10000',
            'web_content_truncation_length': '50000',

            # History summary configuration
            'summary_history': 'True',
            'summary_max_length': '5000',
            'summary_trigger_length': '30000',

            # Memory management system configuration
            'similarity_threshold': '0.70',
            'default_top_k': '5',
            'default_embedding_weight': '0.5',
            'default_tfidf_weight': '0.5',
            'default_max_days_back': '30',
            'default_sort_by': 'recall_count',

            # LLM tool call configuration
            'llm_tool_temperature': '0.0',
            'llm_tool_max_tokens': '500',
            'llm_tool_top_p': '0.1',
            'llm_timeout': '10',

            # Language configuration
            'LANG': 'zh',
            'log_enabled': 'True'
        }

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value

        Args:
            key: Configuration key
            default: Default value

        Returns:
            Configuration value
        """
        # If auto-reload is enabled, check if any config file has been modified
        if self.auto_reload:
            should_reload = False
            for config_file in self.config_files:
                if os.path.exists(config_file):
                    current_modified = os.path.getmtime(config_file)
                    if config_file not in self.last_modified or current_modified > self.last_modified[config_file]:
                        should_reload = True
                        break
            if should_reload:
                self.reload_config()

        return self.config.get(key, default)

    def get_int(self, key: str, default: int = 0) -> int:
        """
        Get integer configuration value

        Args:
            key: Configuration key
            default: Default value

        Returns:
            Integer configuration value
        """
        try:
            return int(self.get(key, default))
        except (ValueError, TypeError):
            logger.warning(f"Configuration item {key} cannot be converted to integer, using default value {default}")
            return default

    def get_float(self, key: str, default: float = 0.0) -> float:
        """
        Get float configuration value

        Args:
            key: Configuration key
            default: Default value

        Returns:
            Float configuration value
        """
        try:
            return float(self.get(key, default))
        except (ValueError, TypeError):
            logger.warning(f"Configuration item {key} cannot be converted to float, using default value {default}")
            return default

    def get_bool(self, key: str, default: bool = False) -> bool:
        """
        Get boolean configuration value

        Args:
            key: Configuration key
            default: Default value

        Returns:
            Boolean configuration value
        """
        value = self.get(key, str(default)).lower()
        return value in ('true', '1', 'yes', 'on')

    def get_tuple(self, key1: str, key2: str, default: Tuple[float, float] = (0.5, 0.5)) -> Tuple[float, float]:
        """
        Get tuple configuration value (for weights, etc.)

        Args:
            key1: First configuration key
            key2: Second configuration key
            default: Default value

        Returns:
            Tuple configuration value
        """
        try:
            val1 = self.get_float(key1, default[0])
            val2 = self.get_float(key2, default[1])
            return (val1, val2)
        except Exception as e:
            logger.warning(f"Failed to get tuple configuration: {e}, using default value {default}")
            return default

    # Memory management system specific configuration methods
    def get_similarity_threshold(self) -> float:
        """Get similarity threshold"""
        return self.get_float('similarity_threshold', 0.70)

    def get_max_tokens(self) -> int:
        """Get maximum token count"""
        return self.get_int('max_tokens', 4096)

    def get_default_top_k(self) -> int:
        """Get default search result count"""
        return self.get_int('default_top_k', 5)

    def get_default_weights(self) -> Tuple[float, float]:
        """Get default search weights"""
        return self.get_tuple('default_embedding_weight', 'default_tfidf_weight', (0.5, 0.5))

    def get_default_max_days_back(self) -> int:
        """Get default memoir update days"""
        return self.get_int('default_max_days_back', 30)

    def get_default_sort_by(self) -> str:
        """Get default sorting method"""
        return self.get('default_sort_by', 'recall_count')

    def get_llm_tool_params(self) -> Dict[str, Any]:
        """Get LLM tool call parameters"""
        return {
            'temperature': self.get_float('llm_tool_temperature', 0.0),
            'max_tokens': self.get_int('llm_tool_max_tokens', 500),
            'top_p': self.get_float('llm_tool_top_p', 0.1),
            'timeout': self.get_int('llm_timeout', 10)
        }

    def get_api_config(self) -> Dict[str, str]:
        """Get API configuration"""
        return {
            'api_key': self.get('api_key'),
            'api_base': self.get('api_base'),
            'model': self.get('mem_model'),
            'embedding_model': self.get('embedding_model'),
            'agent_model': self.get('agent_model')
        }

    def get_truncation_config(self) -> Dict[str, int]:
        """Get truncation configuration"""
        return {
            'truncation_length': self.get_int('truncation_length', 10000),
            'history_truncation_length': self.get_int('history_truncation_length', 10000),
            'web_content_truncation_length': self.get_int('web_content_truncation_length', 50000)
        }

    def get_summary_config(self) -> Dict[str, Any]:
        """Get summary configuration"""
        return {
            'summary_history': self.get_bool('summary_history', True),
            'summary_max_length': self.get_int('summary_max_length', 5000),
            'summary_trigger_length': self.get_int('summary_trigger_length', 30000)
        }

    def get_log_config(self) -> Dict[str, Any]:
        """Get logging configuration"""
        return {
            'log_enabled': self.get_bool('log_enabled', True)
        }

    def validate_config(self) -> Dict[str, Any]:
        """
        Validate configuration validity

        Returns:
            Validation result dictionary
        """
        validation_result = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'fixes': []
        }

        # Check required configurations
        required_configs = ['api_key', 'api_base',
                            'mem_model', 'embedding_model']
        for config in required_configs:
            if not self.get(config):
                validation_result['valid'] = False
                validation_result['errors'].append(f"Missing required configuration: {config}")
                # Provide fix suggestions
                if config == 'api_key':
                    validation_result['fixes'].append(
                        f"Please set {config} or environment variable MEM_API_KEY")
                elif config == 'api_base':
                    validation_result['fixes'].append(
                        f"Please set {config} or environment variable MEM_API_BASE")
                else:
                    validation_result['fixes'].append(f"Please set {config} or the corresponding environment variable")

        # Check value ranges
        similarity_threshold = self.get_similarity_threshold()
        if not 0.0 <= similarity_threshold <= 1.0:
            validation_result['warnings'].append(
                f"Similarity threshold should be between 0.0-1.0, current value: {similarity_threshold}")
            validation_result['fixes'].append(
                "It is recommended to set similarity_threshold to a value between 0.0-1.0")

        max_tokens = self.get_max_tokens()
        if max_tokens <= 0:
            validation_result['warnings'].append(
                f"Maximum token count should be greater than 0, current value: {max_tokens}")
            validation_result['fixes'].append("It is recommended to set max_tokens to a positive integer")

        # Check weight configuration
        weights = self.get_default_weights()
        if abs(sum(weights) - 1.0) > 0.01:
            validation_result['warnings'].append(
                f"Search weights should sum up to approximately 1.0, current value: {sum(weights)}")
            validation_result['fixes'].append(
                "It is recommended to adjust default_embedding_weight and default_tfidf_weight to sum to 1.0")

        # Check API key format
        api_key = self.get('api_key')
        if api_key and not api_key.startswith('sk-'):
            validation_result['warnings'].append("API key format may be incorrect, should start with 'sk-'")

        return validation_result

    def print_config_summary(self):
        """Print configuration summary"""
        logger.info("=== Configuration Summary ===")
        logger.info(f"Configuration file: {self.config_file}")
        logger.info(f"Auto-reload: {'Enabled' if self.auto_reload else 'Disabled'}")
        logger.info(f"Environment variable configurations: {len(self.env_config)}")
        logger.info(f"Similarity threshold: {self.get_similarity_threshold()}")
        logger.info(f"Maximum token count: {self.get_max_tokens()}")
        logger.info(f"Default search result count: {self.get_default_top_k()}")
        logger.info(f"Default search weights: {self.get_default_weights()}")
        logger.info(f"Default sorting method: {self.get_default_sort_by()}")
        logger.info(f"Default memoir update days: {self.get_default_max_days_back()}")

        # Validate configuration
        validation = self.validate_config()
        if not validation['valid']:
            logger.error("Configuration validation failed:")
            for error in validation['errors']:
                logger.error(f"  - {error}")

        if validation['warnings']:
            logger.warning("Configuration warnings:")
            for warning in validation['warnings']:
                logger.warning(f"  - {warning}")

        if validation['fixes']:
            logger.info("Fix suggestions:")
            for fix in validation['fixes']:
                logger.info(f"  - {fix}")

        if validation['valid'] and not validation['warnings']:
            logger.info("Configuration validation passed")

    def save_config(self, config_dict: Dict[str, Any], backup: bool = True) -> bool:
        """
        Save configuration to file

        Args:
            config_dict: Configuration dictionary to save
            backup: Whether to backup original file

        Returns:
            Whether saving was successful
        """
        try:
            # Backup original file
            if backup and os.path.exists(self.config_file):
                backup_file = f"{self.config_file}.backup.{int(time.time())}"
                shutil.copy2(self.config_file, backup_file)
                logger.info(f"Configuration file backed up: {backup_file}")

            # Ensure directory exists
            os.makedirs(os.path.dirname(self.config_file), exist_ok=True)

            # Save configuration
            with open(self.config_file, 'w', encoding='utf-8') as f:
                f.write("# Intelligent Memory Management System Configuration File\n")
                f.write(f"# Generated time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

                for key, value in config_dict.items():
                    f.write(f"{key}={value}\n")

            logger.info(f"Configuration saved to: {self.config_file}")
            return True
        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")
            return False

    def update_config(self, updates: Dict[str, Any]) -> bool:
        """
        Update configuration

        Args:
            updates: Configuration dictionary to update

        Returns:
            Whether updating was successful
        """
        try:
            # Read existing configuration
            current_config = {}
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#') and '=' in line:
                            key, value = line.split('=', 1)
                            current_config[key.strip()] = value.strip()

            # Update configuration
            current_config.update(updates)

            # Save updated configuration
            return self.save_config(current_config)
        except Exception as e:
            logger.error(f"Failed to update configuration: {e}")
            return False

    def get_config_source(self, key: str) -> str:
        """
        Get configuration value source

        Args:
            key: Configuration key

        Returns:
            Configuration source ('file', 'env', 'default')
        """
        if key in self.env_config:
            return 'env'
        elif key in self.config:
            return 'file'
        else:
            return 'default'

    def export_config(self, format: str = 'json') -> str:
        """
        Export configuration

        Args:
            format: Export format ('json', 'env')

        Returns:
            Exported configuration string
        """
        try:
            if format == 'json':
                return json.dumps(self.config, ensure_ascii=False, indent=2)
            elif format == 'env':
                env_lines = []
                for key, value in self.config.items():
                    env_lines.append(f"{key.upper()}={value}")
                return '\n'.join(env_lines)
            else:
                raise ValueError(f"Unsupported export format: {format}")
        except Exception as e:
            logger.error(f"Failed to export configuration: {e}")
            return ""

    def import_config(self, config_data: str, format: str = 'json') -> bool:
        """
        Import configuration

        Args:
            config_data: Configuration data string
            format: Data format ('json', 'env')

        Returns:
            Whether importing was successful
        """
        try:
            if format == 'json':
                imported_config = json.loads(config_data)
            elif format == 'env':
                imported_config = {}
                for line in config_data.split('\n'):
                    line = line.strip()
                    if line and '=' in line and not line.startswith('#'):
                        key, value = line.split('=', 1)
                        imported_config[key.lower()] = value
            else:
                raise ValueError(f"Unsupported import format: {format}")

            # Update configuration
            self.config.update(imported_config)

            # Save to file
            return self.save_config(self.config)
        except Exception as e:
            logger.error(f"Failed to import configuration: {e}")
            return False

    def reset_to_defaults(self) -> bool:
        """
        Reset to default configuration

        Returns:
            Whether resetting was successful
        """
        try:
            default_config = self._get_default_config()
            return self.save_config(default_config)
        except Exception as e:
            logger.error(f"Failed to reset configuration: {e}")
            return False

    def get_config_stats(self) -> Dict[str, Any]:
        """
        Get configuration statistics

        Returns:
            Configuration statistics
        """
        stats = {
            'total_configs': len(self.config),
            'file_configs': len([k for k in self.config.keys() if k not in self.env_config]),
            'env_configs': len(self.env_config),
            'default_configs': len(self._get_default_config()),
            'validation_result': self.validate_config()
        }
        return stats
