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

Unified Log Manager
"""

import logging
import os
import sys
from datetime import datetime
from typing import Optional, Dict, Any, List
from pathlib import Path


def _load_simple_config() -> Dict[str, Any]:
    """
    Simple configuration loader for logger (avoid circular import with ConfigLoader)
    Reads from multiple config files in priority order
    
    Returns:
        Configuration dictionary
    """
    config = {}
    
    # Default config files in priority order (later files override earlier ones)
    default_config_files = [
        "config/config.txt",
        "config/config_memory.txt"
    ]
    
    # Try to find project root and config files
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_roots = [
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))),  # 4 levels up
        os.path.dirname(os.path.dirname(os.path.dirname(current_dir))),  # 3 levels up
        os.path.dirname(os.path.dirname(current_dir)),  # 2 levels up
        os.path.dirname(current_dir),  # 1 level up
        current_dir,  # current dir
        os.getcwd()  # working directory
    ]
    
    config_files_found = []
    for root in project_roots:
        for config_file in default_config_files:
            full_path = os.path.join(root, config_file)
            if os.path.exists(full_path) and full_path not in config_files_found:
                config_files_found.append(full_path)
    
    # Also try the simple config.txt in various locations
    simple_config_files = ["config.txt", "config/config.txt"]
    for root in project_roots:
        for config_file in simple_config_files:
            full_path = os.path.join(root, config_file)
            if os.path.exists(full_path) and full_path not in config_files_found:
                config_files_found.append(full_path)
    
    # Load configuration from files (later files override earlier ones)
    for config_file in config_files_found:
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    # Skip comments and empty lines
                    if not line or line.startswith('#'):
                        continue
                    # Parse key=value lines
                    if '=' in line:
                        key, value = line.split('=', 1)
                        key = key.strip()
                        value = value.strip()
                        config[key] = value
        except Exception as e:
            # Continue reading other files even if one fails
            pass
    
    return config


class MemoryLogger:
    """Memory system log manager"""

    def __init__(
        self,
        name: str = "memory_system",
        log_level: str = "INFO",
        log_file: Optional[str] = None,
        max_file_size: int = 10 * 1024 * 1024,  # 10MB
        backup_count: int = 5,
        log_enabled: bool = True
    ):
        """
        Initialize log manager

        Args:
            name: Logger name
            log_level: Log level
            log_file: Log file path
            max_file_size: Maximum file size
            backup_count: Number of backup files
            log_enabled: Whether to enable console logging output
        """
        self.name = name
        self.log_level = getattr(logging, log_level.upper())
        self.log_file = log_file
        self.max_file_size = max_file_size
        self.backup_count = backup_count
        self.log_enabled = log_enabled

        # Create logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(self.log_level)

        # Clear existing handlers
        self.logger.handlers.clear()

        # Add handlers
        if self.log_enabled:
            self._add_console_handler()
        if log_file:
            self._add_file_handler()

        # Set format
        self._set_formatter()

    def _add_console_handler(self):
        """Add console handler"""
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(self.log_level)
        self.logger.addHandler(console_handler)

    def _add_file_handler(self):
        """Add file handler"""
        if not self.log_file:
            return

        # Ensure log directory exists
        log_dir = os.path.dirname(self.log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)

        # Use RotatingFileHandler for log rotation
        from logging.handlers import RotatingFileHandler
        file_handler = RotatingFileHandler(
            self.log_file,
            maxBytes=self.max_file_size,
            backupCount=self.backup_count,
            encoding='utf-8'
        )
        file_handler.setLevel(self.log_level)
        self.logger.addHandler(file_handler)

    def _set_formatter(self):
        """Set log format"""
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        for handler in self.logger.handlers:
            handler.setFormatter(formatter)

    def get_logger(self) -> logging.Logger:
        """Get logger"""
        return self.logger

    def info(self, message: str):
        """Log info message"""
        self.logger.info(message)

    def warning(self, message: str):
        """Log warning message"""
        self.logger.warning(message)

    def error(self, message: str):
        """Log error message"""
        self.logger.error(message)

    def debug(self, message: str):
        """Log debug message"""
        self.logger.debug(message)

    def critical(self, message: str):
        """Log critical error message"""
        self.logger.critical(message)


# Global log manager instance
_memory_logger = None


def get_logger(name: str = "memory_system") -> logging.Logger:
    """
    Get logger instance

    Args:
        name: Logger name

    Returns:
        Logger instance
    """
    global _memory_logger

    if _memory_logger is None:
        # Read configuration from multiple config files
        config = _load_simple_config()
        
        # Get log level from config (fallback to environment variable)
        log_level = config.get("log_level", os.getenv("MEMORY_LOG_LEVEL", "INFO"))
        
        # Get log file from config or environment variable
        log_file = config.get("log_file", os.getenv("MEMORY_LOG_FILE"))
        
        # Get log_enabled from config (fallback to environment variable)
        log_enabled_str = config.get("log_enabled", os.getenv("MEMORY_LOG_ENABLED", "True"))
        log_enabled = log_enabled_str.lower() in ('true', '1', 'yes', 'on')

        _memory_logger = MemoryLogger(
            name=name,
            log_level=log_level,
            log_file=log_file,
            log_enabled=log_enabled
        )
        
        # If log_enabled is False, completely disable all logging
        if not log_enabled:
            logging.disable(logging.CRITICAL)

    return _memory_logger.get_logger()


def setup_logging(
    name: str = "memory_system",
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    log_enabled: Optional[bool] = None
) -> MemoryLogger:
    """
    Setup log configuration

    Args:
        name: Logger name
        log_level: Log level
        log_file: Log file path
        log_enabled: Whether to enable console logging output (None to auto-detect from config)

    Returns:
        Log manager instance
    """
    global _memory_logger

    # If log_enabled is not specified, try to read from config files
    if log_enabled is None:
        config = _load_simple_config()
        log_enabled_str = config.get("log_enabled", os.getenv("MEMORY_LOG_ENABLED", "True"))
        log_enabled = log_enabled_str.lower() in ('true', '1', 'yes', 'on')

    _memory_logger = MemoryLogger(
        name=name,
        log_level=log_level,
        log_file=log_file,
        log_enabled=log_enabled
    )

    # If log_enabled is False, completely disable all logging
    if not log_enabled:
        logging.disable(logging.CRITICAL)

    return _memory_logger
