#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified Embedding Cache Manager
Specialized for caching summary embeddings of memories to improve retrieval performance
"""

import os
import json
import time
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import hashlib
from .logger import get_logger

logger = get_logger(__name__)


class EmbeddingCacheManager:
    """Unified Embedding Cache Manager"""

    def __init__(self, cache_path: str, max_cache_size: int = 10000):
        """
        Initialize Embedding Cache Manager

        Args:
            cache_path: Cache storage path (must be the embedding_cache subdirectory under the memory library)
            max_cache_size: Maximum number of cache items
        """
        self.cache_path = cache_path
        self.max_cache_size = max_cache_size

        # Ensure cache directory exists
        os.makedirs(cache_path, exist_ok=True)

        # In-memory cache
        self.memory_cache = {}

        # Cache metadata
        self.cache_metadata = {}

        # Initialize cache
        self._init_cache()

        logger.info(f"Embedding cache manager initialized, cache path: {cache_path}")

    def _init_cache(self):
        """Initialize cache"""
        try:
            metadata_path = os.path.join(
                self.cache_path, "cache_metadata.json")
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    self.cache_metadata = json.load(f)
                logger.info(f"Loaded cache metadata, containing {len(self.cache_metadata)} items")
            else:
                self.cache_metadata = {}
                logger.info("Created new cache metadata")
        except Exception as e:
            logger.error(f"Failed to initialize cache: {e}")
            self.cache_metadata = {}

    def _get_cache_key(self, text: str, memory_type: str = "general") -> str:
        """
        Generate cache key

        Args:
            text: Text content
            memory_type: Memory type (preliminary/memoir)

        Returns:
            Cache key
        """
        # Generate hash using the combination of text content and memory type
        content = f"{memory_type}:{text}"
        return hashlib.md5(content.encode('utf-8')).hexdigest()

    def get_cached_embedding(self, text: str, memory_type: str = "general") -> Optional[np.ndarray]:
        """
        Get cached embedding

        Args:
            text: Text content
            memory_type: Memory type

        Returns:
            Embedding vector, returns None if not found
        """
        try:
            cache_key = self._get_cache_key(text, memory_type)

            # Check in-memory cache
            if cache_key in self.memory_cache:
                logger.debug(f"Get embedding from memory cache: {cache_key}")
                return self.memory_cache[cache_key]

            # Check disk cache
            if cache_key in self.cache_metadata:
                cache_file = os.path.join(self.cache_path, f"{cache_key}.npy")
                if os.path.exists(cache_file):
                    embedding = np.load(cache_file)
                    # Load into memory cache
                    self.memory_cache[cache_key] = embedding
                    logger.debug(f"Loaded embedding from disk cache: {cache_key}")
                    return embedding

            return None
        except Exception as e:
            logger.error(f"Failed to get cached embedding: {e}")
            return None

    def cache_embedding(self, text: str, embedding: np.ndarray, memory_type: str = "general") -> bool:
        """
        Cache embedding

        Args:
            text: Text content
            embedding: Embedding vector
            memory_type: Memory type

        Returns:
            Whether caching succeeded
        """
        try:
            cache_key = self._get_cache_key(text, memory_type)

            # Save to memory cache
            self.memory_cache[cache_key] = embedding

            # Save to disk cache
            cache_file = os.path.join(self.cache_path, f"{cache_key}.npy")
            np.save(cache_file, embedding)

            # Update metadata
            self.cache_metadata[cache_key] = {
                "text_length": len(text),
                "embedding_shape": embedding.shape,
                "memory_type": memory_type,
                "created_time": time.time(),
                "text_preview": text[:100] + "..." if len(text) > 100 else text
            }

            # Save metadata
            self._save_metadata()

            # Periodically clean up cache
            self._cleanup_cache()

            logger.debug(f"Cached embedding: {cache_key}")
            return True
        except Exception as e:
            logger.error(f"Failed to cache embedding: {e}")
            return False

    def delete_cached_embedding(self, text: str, memory_type: str = "general") -> bool:
        """
        Delete cached embedding

        Args:
            text: Text content
            memory_type: Memory type

        Returns:
            Whether deletion succeeded
        """
        try:
            cache_key = self._get_cache_key(text, memory_type)

            # Delete from memory cache
            if cache_key in self.memory_cache:
                del self.memory_cache[cache_key]

            # Delete from disk cache
            cache_file = os.path.join(self.cache_path, f"{cache_key}.npy")
            if os.path.exists(cache_file):
                os.remove(cache_file)

            # Delete metadata
            if cache_key in self.cache_metadata:
                del self.cache_metadata[cache_key]

            # Save updated metadata
            self._save_metadata()

            logger.debug(f"Deleted embedding cache: {cache_key}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete embedding cache: {e}")
            return False

    def _save_metadata(self):
        """Save metadata"""
        try:
            metadata_path = os.path.join(
                self.cache_path, "cache_metadata.json")
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(self.cache_metadata, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Failed to save cache metadata: {e}")

    def _cleanup_cache(self):
        """Clean up cache"""
        try:
            # If the number of cache items exceeds the limit, delete the oldest items
            if len(self.cache_metadata) > self.max_cache_size:
                # Sort by creation time
                sorted_items = sorted(
                    self.cache_metadata.items(),
                    key=lambda x: x[1].get('created_time', 0)
                )

                # Delete the oldest items
                items_to_remove = len(sorted_items) - self.max_cache_size
                for i in range(items_to_remove):
                    cache_key, metadata = sorted_items[i]
                    self.delete_cached_embedding(
                        metadata.get('text_preview', ''),
                        metadata.get('memory_type', 'general')
                    )

                logger.info(f"Cleaned up {items_to_remove} cache items")

        except Exception as e:
            logger.error(f"Failed to clean up cache: {e}")

    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics

        Returns:
            Cache statistics information
        """
        try:
            total_items = len(self.cache_metadata)
            memory_items = len(self.memory_cache)

            # Calculate cache size
            total_size_mb = 0
            for cache_key in self.cache_metadata:
                cache_file = os.path.join(self.cache_path, f"{cache_key}.npy")
                if os.path.exists(cache_file):
                    total_size_mb += os.path.getsize(cache_file) / \
                        (1024 * 1024)

            # Statistics by type
            type_stats = {}
            for metadata in self.cache_metadata.values():
                memory_type = metadata.get('memory_type', 'unknown')
                type_stats[memory_type] = type_stats.get(memory_type, 0) + 1

            return {
                "total_items": total_items,
                "memory_items": memory_items,
                "total_size_mb": round(total_size_mb, 2),
                "max_cache_size": self.max_cache_size,
                "type_stats": type_stats,
                "cache_path": self.cache_path
            }
        except Exception as e:
            logger.error(f"Failed to get cache statistics: {e}")
            return {
                "total_items": 0,
                "memory_items": 0,
                "total_size_mb": 0,
                "max_cache_size": self.max_cache_size,
                "type_stats": {},
                "cache_path": self.cache_path
            }

    def clear_cache(self, memory_type: str = None) -> Dict[str, Any]:
        """
        Clear cache

        Args:
            memory_type: Memory type, if None clears all cache

        Returns:
            Clearing result
        """
        try:
            cleared_count = 0
            total_size_mb = 0

            if memory_type is None:
                # Clear all cache
                for cache_key in list(self.cache_metadata.keys()):
                    metadata = self.cache_metadata[cache_key]
                    cache_file = os.path.join(
                        self.cache_path, f"{cache_key}.npy")
                    if os.path.exists(cache_file):
                        total_size_mb += os.path.getsize(
                            cache_file) / (1024 * 1024)
                        os.remove(cache_file)
                    cleared_count += 1

                # Clear in-memory cache and metadata
                self.memory_cache.clear()
                self.cache_metadata.clear()
            else:
                # Clear cache of specified type
                for cache_key in list(self.cache_metadata.keys()):
                    metadata = self.cache_metadata[cache_key]
                    if metadata.get('memory_type') == memory_type:
                        cache_file = os.path.join(
                            self.cache_path, f"{cache_key}.npy")
                        if os.path.exists(cache_file):
                            total_size_mb += os.path.getsize(
                                cache_file) / (1024 * 1024)
                            os.remove(cache_file)

                        # Delete from in-memory cache
                        if cache_key in self.memory_cache:
                            del self.memory_cache[cache_key]

                        # Delete metadata
                        del self.cache_metadata[cache_key]
                        cleared_count += 1

            # Save updated metadata
            self._save_metadata()

            logger.info(
                f"Cleared {cleared_count} cache items, released {round(total_size_mb, 2)} MB space")

            return {
                "success": True,
                "cleared_count": cleared_count,
                "cleared_size_mb": round(total_size_mb, 2),
                "memory_type": memory_type
            }

        except Exception as e:
            logger.error(f"Failed to clear cache: {e}")
            return {
                "success": False,
                "error": str(e),
                "cleared_count": 0,
                "cleared_size_mb": 0
            }

    def preload_cache(self, memory_manager, memory_type: str, max_items: int = 100) -> Dict[str, Any]:
        """
        Preload cache

        Args:
            memory_manager: Memory manager
            memory_type: Memory type
            max_items: Maximum number of items to preload

        Returns:
            Preloading result
        """
        try:
            loaded_count = 0
            skipped_count = 0

            # Get memory list
            memories = memory_manager.list_all() if hasattr(
                memory_manager, 'list_all') else []

            for memory in memories[:max_items]:
                try:
                    # Get summary text
                    summary = getattr(memory, 'summary', '')
                    if not summary:
                        continue

                    # Check if already cached
                    if self.get_cached_embedding(summary, memory_type) is not None:
                        skipped_count += 1
                        continue

                    # Create embedding and cache
                    # Here, you need to call the embedding client, skip for now
                    # embedding = memory_manager.embedding_client.create_embedding(summary)
                    # self.cache_embedding(summary, embedding, memory_type)
                    loaded_count += 1

                except Exception as e:
                    logger.warning(f"Failed to preload memory: {e}")
                    continue

            logger.info(f"Preloading complete: Loaded {loaded_count} items, skipped {skipped_count} items")

            return {
                "success": True,
                "loaded_count": loaded_count,
                "skipped_count": skipped_count,
                "memory_type": memory_type
            }

        except Exception as e:
            logger.error(f"Failed to preload cache: {e}")
            return {
                "success": False,
                "error": str(e),
                "loaded_count": 0,
                "skipped_count": 0
            }


# Global cache manager instance
_global_cache_manager = None


def get_global_cache_manager(storage_path: str = None) -> EmbeddingCacheManager:
    """
    Get global cache manager instance

    Args:
        storage_path: Storage path, if None uses default path

    Returns:
        Global cache manager instance
    """
    global _global_cache_manager
    if _global_cache_manager is None:
        # If storage path is not provided, use default path
        if storage_path is None:
            default_cache_path = "memory/embedding_cache"
        else:
            # Create embedding_cache subdirectory under the storage path
            default_cache_path = os.path.join(storage_path, "embedding_cache")

        _global_cache_manager = EmbeddingCacheManager(default_cache_path)
    return _global_cache_manager
