"""
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
