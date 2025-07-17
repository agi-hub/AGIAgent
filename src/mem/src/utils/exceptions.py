"""
Custom exception class definitions
"""


class MemorySystemError(Exception):
    """Base exception class for memory system"""
    pass


class ConfigError(MemorySystemError):
    """Configuration related exceptions"""
    pass


class LLMClientError(MemorySystemError):
    """LLM client exceptions"""
    pass


class EmbeddingError(MemorySystemError):
    """Embedding related exceptions"""
    pass


class StorageError(MemorySystemError):
    """Storage related exceptions"""
    pass


class ValidationError(MemorySystemError):
    """Data validation exceptions"""
    pass
