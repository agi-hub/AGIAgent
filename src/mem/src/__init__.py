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

Intelligent Memory Management System

Main interfaces:
- MemManagerAgent: Unified management interface
- PreliminaryMemoryManager: Primary memory management
- MemoirManager: Advanced memory organization

Utility modules:
- ConfigLoader: Configuration management
- MemoryLogger: Log management
- SecurityManager: Security management
- PerformanceMonitor: Performance monitoring
"""

# Core modules
from .core.memory_manager import MemManagerAgent
from .core.preliminary import PreliminaryMemoryManager
from .core.memoir import MemoirManager

# Client modules
from .clients.llm_client import LLMClient
from .clients.embedding_client import EmbeddingClient

# Data models
from .models.memory_cell import MemCell, MemoirEntry

# Utility modules
from .utils.config import ConfigLoader
from .utils.logger import MemoryLogger, get_logger, setup_logging
from .utils.security import SecurityManager, get_security_manager
from .utils.monitor import PerformanceMonitor, get_performance_monitor
from .utils.cache_strategy import FileCacheStrategy, cache_result
from .utils.embedding_cache import EmbeddingCacheManager, get_global_cache_manager
from .utils.exceptions import (
    MemorySystemError, ConfigError, LLMClientError,
    EmbeddingError, StorageError, ValidationError
)

__version__ = "2.0.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

__all__ = [
    # Core interfaces
    "MemManagerAgent",
    "PreliminaryMemoryManager",
    "MemoirManager",

    # Clients
    "LLMClient",
    "EmbeddingClient",

    # Data models
    "MemCell",
    "MemoirEntry",

    # Utility modules
    "ConfigLoader",
    "MemoryLogger",
    "get_logger",
    "setup_logging",
    "SecurityManager",
    "get_security_manager",
    "PerformanceMonitor",
    "get_performance_monitor",
    "FileCacheStrategy",
    "cache_result",
    "EmbeddingCacheManager",
    "get_global_cache_manager",

    # Exception classes
    "MemorySystemError",
    "ConfigError",
    "LLMClientError",
    "EmbeddingError",
    "StorageError",
    "ValidationError"
]
