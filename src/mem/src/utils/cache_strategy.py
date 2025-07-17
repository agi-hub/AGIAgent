"""
Cache Strategy Manager
"""

import time
import json
import os
from typing import Dict, Any, Optional, List
from functools import wraps
from .exceptions import StorageError


class CacheStrategy:
    """Cache strategy base class"""

    def __init__(self, cache_dir: str, max_size: int = 1000):
        self.cache_dir = cache_dir
        self.max_size = max_size
        self.cache_info_file = os.path.join(cache_dir, "cache_info.json")
        self._ensure_cache_dir()

    def _ensure_cache_dir(self):
        """Ensure cache directory exists"""
        os.makedirs(self.cache_dir, exist_ok=True)

    def get(self, key: str) -> Optional[Any]:
        """Get cache value"""
        raise NotImplementedError

    def set(self, key: str, value: Any, ttl: int = 3600):
        """Set cache value"""
        raise NotImplementedError

    def delete(self, key: str):
        """Delete cache"""
        raise NotImplementedError

    def clear(self):
        """Clear cache"""
        raise NotImplementedError

    def cleanup_expired(self):
        """Clean up expired cache"""
        raise NotImplementedError


class FileCacheStrategy(CacheStrategy):
    """File cache strategy"""

    def __init__(self, cache_dir: str, max_size: int = 1000):
        super().__init__(cache_dir, max_size)
        self.cache_info = self._load_cache_info()

    def _load_cache_info(self) -> Dict[str, Any]:
        """Load cache information"""
        if os.path.exists(self.cache_info_file):
            try:
                with open(self.cache_info_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                return {"entries": {}, "access_count": {}}
        return {"entries": {}, "access_count": {}}

    def _save_cache_info(self):
        """Save cache information"""
        try:
            with open(self.cache_info_file, 'w', encoding='utf-8') as f:
                json.dump(self.cache_info, f, ensure_ascii=False, indent=2)
        except IOError as e:
            raise StorageError(f"Failed to save cache information: {e}")

    def get(self, key: str) -> Optional[Any]:
        """Get cache value"""
        if key not in self.cache_info["entries"]:
            return None

        entry = self.cache_info["entries"][key]
        file_path = os.path.join(self.cache_dir, f"{key}.cache")

        # Check if file exists
        if not os.path.exists(file_path):
            self.delete(key)
            return None

        # Check if expired
        if time.time() > entry["expire_time"]:
            self.delete(key)
            return None

        # Read cache file
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Update access count
            self.cache_info["access_count"][key] = self.cache_info["access_count"].get(
                key, 0) + 1
            self._save_cache_info()

            return data
        except (json.JSONDecodeError, IOError):
            self.delete(key)
            return None

    def set(self, key: str, value: Any, ttl: int = 3600):
        """Set cache value"""
        file_path = os.path.join(self.cache_dir, f"{key}.cache")

        # Check cache size limit
        if len(self.cache_info["entries"]) >= self.max_size:
            self._evict_least_used()

        # Save cache file
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(value, f, ensure_ascii=False, indent=2)
        except IOError as e:
            raise StorageError(f"Failed to save cache file: {e}")

        # Update cache information
        self.cache_info["entries"][key] = {
            "file_path": file_path,
            "create_time": time.time(),
            "expire_time": time.time() + ttl,
            "size": os.path.getsize(file_path)
        }
        self.cache_info["access_count"][key] = 0
        self._save_cache_info()

    def delete(self, key: str):
        """Delete cache"""
        if key in self.cache_info["entries"]:
            file_path = self.cache_info["entries"][key]["file_path"]
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
            except OSError:
                pass

            del self.cache_info["entries"][key]
            if key in self.cache_info["access_count"]:
                del self.cache_info["access_count"][key]
            self._save_cache_info()

    def clear(self):
        """Clear cache"""
        # Delete all cache files
        for entry in self.cache_info["entries"].values():
            try:
                if os.path.exists(entry["file_path"]):
                    os.remove(entry["file_path"])
            except OSError:
                pass

        # Clear cache information
        self.cache_info = {"entries": {}, "access_count": {}}
        self._save_cache_info()

    def cleanup_expired(self):
        """Clean up expired cache"""
        current_time = time.time()
        expired_keys = []

        for key, entry in self.cache_info["entries"].items():
            if current_time > entry["expire_time"]:
                expired_keys.append(key)

        for key in expired_keys:
            self.delete(key)

    def _evict_least_used(self):
        """Evict least used cache"""
        if not self.cache_info["entries"]:
            return

        # Find cache with least access count
        least_used_key = min(
            self.cache_info["access_count"].keys(),
            key=lambda k: self.cache_info["access_count"][k]
        )

        self.delete(least_used_key)


def cache_result(ttl: int = 3600, strategy: str = "file", cache_dir: str = None):
    """Cache decorator"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            cache_key = f"{func.__name__}_{hash(str(args) + str(sorted(kwargs.items())))}"

            # Determine cache directory
            if cache_dir is None:
                # Try to get storage_path from parameters
                storage_path = None
                for arg in args:
                    if hasattr(arg, 'storage_path'):
                        storage_path = arg.storage_path
                        break

                if storage_path:
                    # Create cache directory under storage path
                    cache_directory = os.path.join(storage_path, "cache")
                else:
                    # Default to global cache directory
                    cache_directory = "cache"
            else:
                cache_directory = cache_dir

            # Get cache strategy instance
            cache_strategy = FileCacheStrategy(cache_directory)

            # Try to get from cache
            cached_result = cache_strategy.get(cache_key)
            if cached_result is not None:
                return cached_result

            # Execute function and cache result
            result = func(*args, **kwargs)
            cache_strategy.set(cache_key, result, ttl)

            return result
        return wrapper
    return decorator
