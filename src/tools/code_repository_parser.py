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
import json
import pickle
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional
from dataclasses import dataclass
from collections import defaultdict
import re
from tqdm import tqdm
import logging
import time
import threading
import warnings
from datetime import datetime
from .print_system import print_system, print_current, print_debug, print_error

# Configure logging BEFORE importing jieba to suppress debug output
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# Disable jieba debug logging
jieba_logger = logging.getLogger('jieba')
jieba_logger.setLevel(logging.ERROR)

# ========================================
# 🚀 延迟导入优化：重量级库延迟加载
# ========================================
# 这些库只在实际使用代码索引功能时才导入，避免启动时加载
# numpy、sklearn 等库会在后台线程中首次使用时加载

# 延迟导入标志
_LAZY_IMPORTS_LOADED = False
_LAZY_IMPORTS_LOCK = threading.Lock()

# 全局变量用于存储延迟导入的模块
np = None
TfidfVectorizer = None
cosine_similarity = None

def _ensure_lazy_imports():
    """确保延迟导入的库已加载（线程安全）"""
    global _LAZY_IMPORTS_LOADED, np, TfidfVectorizer, cosine_similarity
    
    if _LAZY_IMPORTS_LOADED:
        return
    
    with _LAZY_IMPORTS_LOCK:
        # 双重检查锁定模式
        if _LAZY_IMPORTS_LOADED:
            return
        
        try:
            print_debug("⏳ 首次使用代码索引功能，正在加载机器学习库...")
            
            # 导入 numpy
            import numpy as _np
            np = _np
            
            # 导入 sklearn
            from sklearn.feature_extraction.text import TfidfVectorizer as _TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity as _cosine_similarity
            TfidfVectorizer = _TfidfVectorizer
            cosine_similarity = _cosine_similarity
            
            _LAZY_IMPORTS_LOADED = True
            print_debug("✅ 机器学习库加载完成")
            
        except ImportError as e:
            print_error(f"❌ 无法导入必需的机器学习库: {e}")
            raise

# ========================================
# 🚀 Jieba 延迟导入优化
# ========================================
# jieba 用于中文分词，只在实际需要中文分词时才加载
# 这样可以避免启动时加载 jieba 及其词典文件（较慢）

jieba = None
JIEBA_ENABLED = None  # 未初始化状态
_JIEBA_CHECKED = False
_JIEBA_LOCK = threading.Lock()

def _ensure_jieba_loaded():
    """确保 jieba 已加载（延迟加载，线程安全）"""
    global jieba, JIEBA_ENABLED, _JIEBA_CHECKED
    
    if _JIEBA_CHECKED:
        return JIEBA_ENABLED
    
    with _JIEBA_LOCK:
        # 双重检查锁定
        if _JIEBA_CHECKED:
            return JIEBA_ENABLED
        
        try:
            # Check jieba setting
            import sys
            import os
            sys.path.append(os.path.dirname(os.path.dirname(__file__)))
            from config_loader import get_enable_jieba
            
            JIEBA_ENABLED = get_enable_jieba()
            
            if JIEBA_ENABLED:
                print_debug("⏳ 首次使用中文分词，正在加载 jieba...")
                
                # Suppress jieba initialization output and pkg_resources deprecation warning
                import warnings
                warnings.filterwarnings('ignore', category=UserWarning, module='jieba')
                warnings.filterwarnings('ignore', category=UserWarning, message='.*pkg_resources.*')
                
                # Redirect jieba stderr to suppress prints
                import contextlib
                import io
                
                with contextlib.redirect_stderr(io.StringIO()):
                    import jieba as _jieba
                    import jieba.analyse
                    # Configure jieba to be quiet
                    _jieba.setLogLevel(logging.ERROR)
                    jieba = _jieba
                
                print_debug("✅ jieba 加载完成")
            else:
                jieba = None
                print_debug("ℹ️ 中文分词功能未启用")
                
        except ImportError:
            # If config_loader is not available, default to disabled
            JIEBA_ENABLED = False
            jieba = None
        
        _JIEBA_CHECKED = True
        return JIEBA_ENABLED

# Vectorization related libraries - removed sentence_transformers
# Vector database related libraries (延迟加载)
faiss = None
FAISS_AVAILABLE = None  # Will be checked on first use

def _check_faiss_available():
    """检查 faiss 是否可用（延迟检查）"""
    global faiss, FAISS_AVAILABLE
    
    if FAISS_AVAILABLE is not None:
        return FAISS_AVAILABLE
    
    try:
        # 修复 faiss 库中的 distutils 弃用警告问题
        # 使用 monkey patch 将 LooseVersion 替换为 packaging.version.Version
        import sys
        from packaging import version as packaging_version
        
        # 创建一个兼容 LooseVersion 的包装类
        class LooseVersionCompat:
            """兼容 LooseVersion 的包装类，使用 packaging.version.Version"""
            def __init__(self, vstring):
                self.version = packaging_version.parse(vstring)
            
            def __ge__(self, other):
                if isinstance(other, str):
                    other = packaging_version.parse(other)
                return self.version >= other
            
            def __le__(self, other):
                if isinstance(other, str):
                    other = packaging_version.parse(other)
                return self.version <= other
            
            def __gt__(self, other):
                if isinstance(other, str):
                    other = packaging_version.parse(other)
                return self.version > other
            
            def __lt__(self, other):
                if isinstance(other, str):
                    other = packaging_version.parse(other)
                return self.version < other
            
            def __eq__(self, other):
                if isinstance(other, str):
                    other = packaging_version.parse(other)
                return self.version == other
        
        # 在导入 faiss 之前，替换 distutils.version.LooseVersion
        import distutils.version
        original_loose_version = distutils.version.LooseVersion
        distutils.version.LooseVersion = LooseVersionCompat
        
        # 直接导入 faiss（问题已通过 monkey patch 修复，无需抑制警告）
        import faiss as _faiss
        
        # 恢复原始的 LooseVersion（可选，但保持干净）
        distutils.version.LooseVersion = original_loose_version
        
        faiss = _faiss
        FAISS_AVAILABLE = True
        print_debug("✅ FAISS 库已加载（已修复 distutils 弃用警告）")
    except ImportError:
        FAISS_AVAILABLE = False
        print_debug("ℹ️ FAISS 不可用，将使用 numpy 进行向量存储")
    except Exception as e:
        # 如果修复失败，尝试直接导入（可能会有警告，但功能正常）
        try:
            import faiss as _faiss
            faiss = _faiss
            FAISS_AVAILABLE = True
            print_debug(f"✅ FAISS 库已加载（修复失败但导入成功: {e}）")
        except ImportError:
            FAISS_AVAILABLE = False
            print_debug("ℹ️ FAISS 不可用，将使用 numpy 进行向量存储")
    
    return FAISS_AVAILABLE

# Add global code index manager
_global_parsers = {}  # workspace_root -> CodeRepositoryParser instance
_global_parsers_lock = threading.Lock()

def get_global_code_parser(workspace_root: str, **kwargs) -> 'CodeRepositoryParser':
    """
    Get global code parser instance, ensuring only one instance per workspace
    
    Args:
        workspace_root: Workspace root directory
        **kwargs: Additional parameters passed to CodeRepositoryParser constructor
        
    Returns:
        CodeRepositoryParser instance
    """
    workspace_root = os.path.abspath(workspace_root)
    
    with _global_parsers_lock:
        if workspace_root not in _global_parsers:
            # Create new parser instance
            parser = CodeRepositoryParser(
                root_path=workspace_root,
                **kwargs
            )
            _global_parsers[workspace_root] = parser
            print_current(f"🔧 Created new global code parser for workspace: {workspace_root}")
        else:
            print_current(f"🔄 Reusing existing global code parser for workspace: {workspace_root}")
        
        return _global_parsers[workspace_root]

def cleanup_global_parsers():
    """Clean up all global code parsers"""
    with _global_parsers_lock:
        for workspace_root, parser in _global_parsers.items():
            try:
                parser.cleanup()
                print_current(f"🧹 Cleaned up global code parser for: {workspace_root}")
            except Exception as e:
                print_current(f"⚠️ Error cleaning up parser for {workspace_root}: {e}")
        _global_parsers.clear()

@dataclass
class CodeSegment:
    """Code segment data structure"""
    content: str
    file_path: str
    start_line: int
    end_line: int
    segment_id: str
    
    def __getstate__(self):
        """Support for pickling"""
        return self.__dict__
    
    def __setstate__(self, state):
        """Support for unpickling"""
        self.__dict__.update(state)

@dataclass
class SearchResult:
    """Search result data structure"""
    segment: CodeSegment
    score: float
    search_type: str  # 'vector' or 'keyword'

@dataclass
class FileTimestamp:
    """File timestamp data structure"""
    file_path: str
    last_modified: float
    file_size: int
    last_checked: float

class IncrementalUpdateThread:
    """Independent incremental update thread that periodically checks and updates code repository indexes"""
    
    def __init__(self, code_parser, update_interval: float = 1.0):
        """
        Initialize incremental update thread
        
        Args:
            code_parser: CodeRepositoryParser instance
            update_interval: Update interval in seconds, default 1 second
        """
        self.code_parser = code_parser
        self.update_interval = update_interval
        self.running = False
        self.thread = None
        self.lock = threading.Lock()  # For thread safety
        self.last_update_time = 0
        self.total_updates = 0
        self.successful_updates = 0
        
        # Add anti-duplicate mechanism
        self.last_changes_hash = None
        self.last_successful_update_time = 0
        self.min_update_interval = 2.0
        
        # Statistics
        self.stats = {
            'total_checks': 0,
            'total_updates': 0,
            'successful_updates': 0,
            'failed_updates': 0,
            'skipped_updates': 0,
            'last_update_time': None,
            'last_error': None
        }
    
    def start(self):
        """Start incremental update thread"""
        if self.running:
            print_current("⚠️ Incremental update thread is already running")
            return
        
        self.running = True
        self.thread = threading.Thread(target=self._update_loop, name="CodeRepoUpdater", daemon=True)
        self.thread.start()
        #print_current(f"🚀 Incremental update thread started, update interval: {self.update_interval} seconds")
    
    def stop(self):
        """Stop incremental update thread"""
        if not self.running:
            return
        
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=5)  # Wait up to 5 seconds
    
    def _update_loop(self):
        """Main update loop"""
        #print_current("🔄 Incremental update thread started running")
        
        while self.running:
            try:
                start_time = time.time()
                
                # Use lock to ensure thread safety
                with self.lock:
                    if self.code_parser and hasattr(self.code_parser, 'incremental_update'):
                        self.stats['total_checks'] += 1
                        
                        try:
                            # Execute incremental update
                            update_result = self.code_parser.incremental_update()
                            
                            # Check if there were actual changes
                            has_changes = any(count > 0 for count in update_result.values())
                            
                            if has_changes:
                                # Calculate changes hash for anti-duplicate
                                import hashlib
                                changes_str = f"{update_result['files_added']}_{update_result['files_modified']}_{update_result['files_deleted']}"
                                current_changes_hash = hashlib.md5(changes_str.encode()).hexdigest()
                                current_time = time.time()
                                
                                # Check if this is a duplicate update
                                time_since_last_update = current_time - self.last_successful_update_time
                                is_duplicate = (self.last_changes_hash == current_changes_hash and 
                                              time_since_last_update < self.min_update_interval)
                                
                                if is_duplicate:
                                    self.stats['skipped_updates'] += 1
                                    # Don't print message for skipped duplicate updates
                                else:
                                    # This is a valid update
                                    self.stats['total_updates'] += 1
                                    
                                    # Save database if needed
                                    db_path = self.code_parser._get_code_index_path()
                                    self.code_parser.save_database(db_path)
                                    
                                    self.stats['successful_updates'] += 1
                                    self.stats['last_update_time'] = datetime.now().isoformat()
                                    
                                    # Update anti-duplicate tracking
                                    self.last_changes_hash = current_changes_hash
                                    self.last_successful_update_time = current_time
                                    
                                    # Print update message
                                    total_changes = sum(update_result.values())

                                    workspace_name = os.path.basename(self.code_parser.root_path)
                                    #print_current(f"🔄 Background thread ({workspace_name}): Code repository update completed: {total_changes} file changes")
                            
                        except Exception as e:
                            self.stats['failed_updates'] += 1
                            self.stats['last_error'] = str(e)
                            # Only log the error type and basic message to avoid verbose output
                            error_msg = str(e)
                            if "empty vocabulary" in error_msg.lower():
                                print_debug(f"❌ Incremental update failed: empty vocabulary; perhaps the documents only contain stop words")
                            else:
                                print_debug(f"❌ Incremental update failed: {type(e).__name__}: {error_msg}")
                
                # Calculate update time for this iteration
                elapsed = time.time() - start_time
                
                # Dynamically adjust sleep time to ensure stable update interval
                sleep_time = max(0, self.update_interval - elapsed)
                time.sleep(sleep_time)
                
            except Exception as e:
                self.stats['failed_updates'] += 1
                self.stats['last_error'] = str(e)
                print_current(f"❌ Incremental update thread error: {e}")
                time.sleep(self.update_interval)  # Wait one cycle after error
    
    def get_stats(self) -> Dict[str, Any]:
        """Get update statistics"""
        with self.lock:
            return self.stats.copy()
    
    def is_running(self) -> bool:
        """Check if thread is running"""
        return self.running and self.thread and self.thread.is_alive()
    
    def __getstate__(self):
        """Custom pickle state method to handle thread locks"""
        state = self.__dict__.copy()
        # Remove unpicklable thread locks and thread objects
        state.pop('lock', None)
        state.pop('thread', None)
        # Set running to False since thread won't be active after unpickling
        state['running'] = False
        return state
    
    def __setstate__(self, state):
        """Custom unpickle state method to recreate thread locks"""
        self.__dict__.update(state)
        # Recreate thread lock
        self.lock = threading.Lock()
        # Thread will be recreated when needed
        self.thread = None
        self.running = False

class CodeRepositoryParser:
    """Code repository parser"""
    
    def __init__(self, 
                 root_path: str,
                 segment_size: int = 200,
                 supported_extensions: Optional[List[str]] = None,
                 enable_background_update: bool = True,
                 update_interval: float = 1.0):
        """
        Initialize code repository parser
        
        Args:
            root_path: Code repository root path
            segment_size: Number of lines per segment
            supported_extensions: Supported file extensions
            enable_background_update: Whether to enable background incremental updates
            update_interval: Background update interval in seconds
        """
        self.root_path = Path(root_path)
        self.segment_size = segment_size
        
        self._agia_initialized = False
        
        # Default supported code file extensions
        if supported_extensions is None:
            self.supported_extensions = {
                '.py', '.js', '.jsx', '.ts', '.tsx', '.java', '.cpp', '.c', '.h',
                '.cs', '.go', '.rs', '.rb', '.php', '.swift', '.kt', '.scala',
                '.r', '.m', '.sql', '.sh', '.ps1', '.md', '.txt', '.json', '.yaml', '.yml'
            }
        else:
            self.supported_extensions = set(supported_extensions)
        
        # Data storage
        self.code_segments: List[CodeSegment] = []
        self.segment_vectors: Optional[Any] = None  # Will be np.ndarray after lazy import
        self.vector_index = None
        
        # File timestamp records
        self.file_timestamps: Dict[str, FileTimestamp] = {}
        
        # TF-IDF database (延迟初始化)
        self.tfidf_vectorizer = None  # Will be initialized when needed
        self.tfidf_matrix = None
        self._tfidf_config = {
            'max_features': 8000,
            'stop_words': None,
            'ngram_range': (1, 2),
            'min_df': 1,
            'max_df': 1.0,
            'sublinear_tf': True,
            'norm': 'l2'
        }
        
        # Background update thread
        self.background_update_thread = None
        self.enable_background_update = enable_background_update
        self.update_interval = update_interval
        self._update_lock = threading.Lock()  # For protecting data access
        
        # If background update is enabled, thread will be started later (after initialization)
        self._background_update_enabled = enable_background_update
    
    def _ensure_tfidf_vectorizer(self):
        """确保 TF-IDF vectorizer 已初始化（延迟加载）"""
        if self.tfidf_vectorizer is not None:
            return
        
        # 先确保延迟导入的库已加载
        _ensure_lazy_imports()
        
        # 使用全局的 TfidfVectorizer 类
        global TfidfVectorizer
        
        # 创建 TF-IDF vectorizer 实例
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=self._tfidf_config['max_features'],
            stop_words=self._tfidf_config['stop_words'],
            ngram_range=self._tfidf_config['ngram_range'],
            tokenizer=self._tokenize_code,
            token_pattern=None,
            min_df=self._tfidf_config['min_df'],
            max_df=self._tfidf_config['max_df'],
            sublinear_tf=self._tfidf_config['sublinear_tf'],
            norm=self._tfidf_config['norm']
        )
    
    def start_background_update(self):
        """Start background incremental update thread"""
        if not self._background_update_enabled:
            print_current("📍 Background update is disabled, skipping thread start")
            return
        
        if self.background_update_thread and self.background_update_thread.is_running():
            print_current("🔄 Background update thread is already running")
            return
        
        self.background_update_thread = IncrementalUpdateThread(
            code_parser=self,
            update_interval=self.update_interval
        )
        self.background_update_thread.start()
        print_system(f"🚀 Started background code index update thread (interval: {self.update_interval}s)")
    
    def stop_background_update(self):
        """Stop background incremental update thread"""
        if self.background_update_thread:
            self.background_update_thread.stop()
            self.background_update_thread = None
    
    def get_background_update_stats(self) -> Dict[str, Any]:
        """Get background update statistics"""
        if self.background_update_thread:
            return self.background_update_thread.get_stats()
        return {}
    
    def _tokenize_code(self, text: str) -> List[str]:
        """
        Code text tokenizer
        
        Args:
            text: Code text
            
        Returns:
            List of tokenized results
        """
        # Extract identifiers, keywords, etc. from code
        # Use regex to match variable names, function names, class names, etc.
        identifier_pattern = r'\b[a-zA-Z_][a-zA-Z0-9_]*\b'
        identifiers = re.findall(identifier_pattern, text)
        
        # Add Chinese word segmentation support (for Chinese comments) if jieba is enabled
        # 🚀 延迟加载：首次使用中文分词时才加载 jieba
        if _ensure_jieba_loaded():
            global jieba
            chinese_text = re.sub(r'[^\u4e00-\u9fff]+', ' ', text)
            if chinese_text.strip():
                try:
                    chinese_tokens = jieba.lcut(chinese_text)
                    identifiers.extend([token for token in chinese_tokens if len(token) > 1])
                except Exception as e:
                    # If jieba fails, continue without Chinese segmentation
                    logger.warning(f"Jieba segmentation failed: {e}")
        
        # Remove only very common stop words, keep programming keywords for better search
        stop_words = {
            'the', 'and', 'or', 'not', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should',
            'may', 'might', 'must', 'can', 'cannot', 'cant', 'wont', 'dont', 'doesnt',
            'didnt', 'isnt', 'arent', 'wasnt', 'werent', 'hasnt', 'havent', 'hadnt',
            'wouldnt', 'couldnt', 'shouldnt', 'mustnt', 'neednt', 'daren', 'darent'
        }
        
        # Keep more tokens including programming keywords for better code search
        return [token.lower() for token in identifiers 
                if token.lower() not in stop_words and len(token) > 1]
    
    def _is_code_file(self, file_path: Path) -> bool:
        """
        Check if it's a code file
        
        Args:
            file_path: File path
            
        Returns:
            Whether it's a code file
        """
        # Check file extension
        if file_path.suffix.lower() not in self.supported_extensions:
            return False
        
        # Exclude dictionary files and other non-code files
        file_name = file_path.name.lower()
        excluded_patterns = [
            'ppocr_keys',  # OCR dictionary files
            'dict',        # Dictionary files
            'vocab',       # Vocabulary files
            'stopwords',   # Stop word files
            'wordlist',    # Word list files
            'corpus',      # Corpus files
            'resources',   # Resource files (if they contain data, not code)
        ]
        
        # Check if file contains dictionary-like patterns
        if any(pattern in file_name for pattern in excluded_patterns):
            return False
        
        # Additional check: if it's a .txt file, exclude if it contains mostly non-ASCII characters
        if file_path.suffix.lower() == '.txt':
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    sample = f.read(1000)  # Read first 1000 characters
                    if sample:
                        # Count non-ASCII characters
                        non_ascii_count = sum(1 for char in sample if ord(char) > 127)
                        # If more than 80% are non-ASCII, likely a dictionary file
                        if non_ascii_count / len(sample) > 0.8:
                            return False
            except Exception:
                pass  # If we can't read the file, include it anyway
        
        return True
    
    def _read_file_safely(self, file_path: Path) -> Optional[str]:
        """
        Safely read file content
        
        Args:
            file_path: File path
            
        Returns:
            File content or None
        """
        encodings = ['utf-8', 'gbk', 'utf-16', 'iso-8859-1']
        
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    return f.read()
            except (UnicodeDecodeError, UnicodeError):
                continue
            except Exception as e:
                logger.warning(f"Error reading file {file_path}: {e}")
                break
        
        return None
    
    def _get_file_timestamp(self, file_path: Path) -> Optional[FileTimestamp]:
        """
        Get file timestamp information
        
        Args:
            file_path: File path
            
        Returns:
            File timestamp information
        """
        try:
            stat = file_path.stat()
            relative_path = str(file_path.relative_to(self.root_path))
            
            return FileTimestamp(
                file_path=relative_path,
                last_modified=stat.st_mtime,
                file_size=stat.st_size,
                last_checked=time.time()
            )
        except Exception as e:
            logger.warning(f"Error getting timestamp for {file_path}: {e}")
            return None
    
    def _has_file_changed(self, file_path: Path) -> bool:
        """
        Check if file has changed
        
        Args:
            file_path: File path
            
        Returns:
            Whether file has changed
        """
        try:
            relative_path = str(file_path.relative_to(self.root_path))
            current_timestamp = self._get_file_timestamp(file_path)
            
            if not current_timestamp:
                return True  # Cannot get timestamp, consider it changed
            
            if relative_path not in self.file_timestamps:
                return True  # New file
            
            stored_timestamp = self.file_timestamps[relative_path]
            
            # Check modification time and file size
            return (current_timestamp.last_modified != stored_timestamp.last_modified or
                    current_timestamp.file_size != stored_timestamp.file_size)
        
        except Exception as e:
            logger.warning(f"Error checking file change for {file_path}: {e}")
            return True  # Consider it changed when error occurs
    
    def _update_file_timestamp(self, file_path: Path):
        """
        Update file timestamp record
        
        Args:
            file_path: File path
        """
        timestamp = self._get_file_timestamp(file_path)
        if timestamp:
            self.file_timestamps[timestamp.file_path] = timestamp
    
    def _segment_code(self, content: str, file_path: str) -> List[CodeSegment]:
        """
        Segment code content
        
        Args:
            content: File content
            file_path: File path
            
        Returns:
            List of code segments
        """
        lines = content.split('\n')
        segments = []
        
        # For markdown documents, use intelligent segmentation
        if file_path.endswith('.md'):
            segments = self._segment_markdown_intelligently(lines, file_path)
            if segments:
                return segments
        
        # Use smaller segment size for markdown documents
        segment_size = self.segment_size
        if file_path.endswith('.md'):
            segment_size = min(50, self.segment_size)  # Use 50 lines or smaller for markdown
        
        for i in range(0, len(lines), segment_size):
            end_idx = min(i + segment_size, len(lines))
            segment_content = '\n'.join(lines[i:end_idx])
            
            # Filter empty segments
            if segment_content.strip():
                segment_id = f"{file_path}:{i+1}:{end_idx}"
                segment = CodeSegment(
                    content=segment_content,
                    file_path=file_path,
                    start_line=i + 1,
                    end_line=end_idx,
                    segment_id=segment_id
                )
                segments.append(segment)
        
        return segments
    
    def _segment_markdown_intelligently(self, lines: List[str], file_path: str) -> List[CodeSegment]:
        """
        Intelligently segment markdown documents by header structure
        
        Args:
            lines: Document line list
            file_path: File path
            
        Returns:
            List of code segments
        """
        segments = []
        current_segment_start = 0
        
        for i, line in enumerate(lines):
            # Detect markdown headers (# ## ### etc.)
            if line.strip().startswith('#') and len(line.strip()) > 1:
                # If new header found, save previous segment first
                if i > current_segment_start:
                    segment_content = '\n'.join(lines[current_segment_start:i])
                    if segment_content.strip():
                        segment_id = f"{file_path}:{current_segment_start+1}:{i}"
                        segment = CodeSegment(
                            content=segment_content,
                            file_path=file_path,
                            start_line=current_segment_start + 1,
                            end_line=i,
                            segment_id=segment_id
                        )
                        segments.append(segment)
                
                current_segment_start = i
        
        # Handle last segment
        if current_segment_start < len(lines):
            segment_content = '\n'.join(lines[current_segment_start:])
            if segment_content.strip():
                segment_id = f"{file_path}:{current_segment_start+1}:{len(lines)}"
                segment = CodeSegment(
                    content=segment_content,
                    file_path=file_path,
                    start_line=current_segment_start + 1,
                    end_line=len(lines),
                    segment_id=segment_id
                )
                segments.append(segment)
        
        # If too few segments (less than 3), fallback to fixed-size segmentation
        if len(segments) < 3:
            return []
        
        return segments
    
    def _remove_file_segments(self, file_path: str):
        """
        Remove all code segments from specified file
        
        Args:
            file_path: File path to remove
        """
        # Remove code segments
        original_count = len(self.code_segments)
        self.code_segments = [seg for seg in self.code_segments if seg.file_path != file_path]
        removed_count = original_count - len(self.code_segments)
        
        if removed_count > 0:
            logger.info(f"Removed {removed_count} segments from file: {file_path}")
        
        # Reset vector-related data, need to rebuild
        self.segment_vectors = None
        self.vector_index = None
        self.tfidf_matrix = None
    
    def _process_single_file(self, file_path: Path) -> int:
        """
        Process single file and add to code segments
        
        Args:
            file_path: File path
            
        Returns:
            Number of code segments added
        """
        try:
            content = self._read_file_safely(file_path)
            if content is None:
                return 0
            
            relative_path = str(file_path.relative_to(self.root_path))
            segments = self._segment_code(content, relative_path)
            
            # Add new code segments
            self.code_segments.extend(segments)
            
            # Update file timestamp
            self._update_file_timestamp(file_path)
            
            return len(segments)
            
        except Exception as e:
            logger.warning(f"Error processing file {file_path}: {e}")
            return 0
    
    def _get_all_code_files(self) -> List[Path]:
        """
        Get all code files including those in symlinked directories
        
        Returns:
            List of Path objects for all code files
        """
        code_files = []
        visited_dirs = set()  # To avoid infinite loops with circular symlinks
        
        def _traverse_directory(path: Path, depth: int = 0):
            """Recursively traverse directory, handling symlinks"""
            if depth > 10:  # Prevent infinite recursion
                return
                
            try:
                # Convert to absolute path and resolve to detect circular references
                abs_path = path.resolve()
                if abs_path in visited_dirs:
                    return
                visited_dirs.add(abs_path)
                
                # Use iterdir() instead of rglob() to handle symlinks manually
                for item in path.iterdir():
                    try:
                        if item.is_file():
                            # Regular file or symlink to file
                            if self._is_code_file(item):
                                code_files.append(item)
                        elif item.is_dir():
                            # Regular directory or symlink to directory
                            _traverse_directory(item, depth + 1)
                        elif item.is_symlink():
                            # Handle symlink explicitly
                            target = item.resolve()
                            if target.exists():
                                if target.is_file() and self._is_code_file(item):
                                    code_files.append(item)
                                elif target.is_dir():
                                    _traverse_directory(item, depth + 1)
                    except (OSError, RuntimeError) as e:
                        # Skip files/directories that can't be accessed
                        logger.warning(f"Skipping {item}: {e}")
                        continue
                        
            except (OSError, RuntimeError) as e:
                logger.warning(f"Error accessing directory {path}: {e}")
                return
        
        # Start traversal from root path
        _traverse_directory(self.root_path)
        return code_files

    def check_repository_changes(self) -> Dict[str, List[str]]:
        """
        Check file changes in code repository (without internal locking)
        
        Note: This method does not use internal locks. If thread safety is required,
        the caller should handle locking.
        
        Returns:
            Change information dictionary containing lists of added, modified, deleted files
        """
        logger.info("Checking repository changes...")
        
        changes = {
            'added': [],
            'modified': [],
            'deleted': []
        }
        
        # Get all current code files (including symlinked files)
        current_files = set()
        for file_path in self._get_all_code_files():
            current_files.add(str(file_path.relative_to(self.root_path)))
        
        # Check changes in known files
        for file_path in current_files:
            full_path = self.root_path / file_path
            if self._has_file_changed(full_path):
                if file_path in self.file_timestamps:
                    changes['modified'].append(file_path)
                else:
                    changes['added'].append(file_path)
        
        # Check deleted files
        stored_files = set(self.file_timestamps.keys())
        for file_path in stored_files:
            if file_path not in current_files:
                changes['deleted'].append(file_path)
        
        logger.info(f"Repository changes: {len(changes['added'])} added, "
                   f"{len(changes['modified'])} modified, {len(changes['deleted'])} deleted")
        
        return changes
    
    def incremental_update(self) -> Dict[str, int]:
        """
        Incremental update of code repository (thread-safe version)
        
        Returns:
            Update statistics
        """
        logger.info("Starting incremental update...")
        
        stats = {
            'files_added': 0,
            'files_modified': 0,
            'files_deleted': 0,
            'segments_added': 0,
            'segments_removed': 0
        }
        
        # Perform all operations under a single lock to ensure atomicity
        with self._update_lock:
            # Check changes within the same lock scope
            changes = {
                'added': [],
                'modified': [],
                'deleted': []
            }
            
            # Get all current code files (including symlinked files)
            current_files = set()
            for file_path in self._get_all_code_files():
                current_files.add(str(file_path.relative_to(self.root_path)))
            
            # Check changes in known files
            for file_path in current_files:
                full_path = self.root_path / file_path
                if self._has_file_changed(full_path):
                    if file_path in self.file_timestamps:
                        changes['modified'].append(file_path)
                    else:
                        changes['added'].append(file_path)
            
            # Check deleted files
            stored_files = set(self.file_timestamps.keys())
            for file_path in stored_files:
                if file_path not in current_files:
                    changes['deleted'].append(file_path)
            
            logger.info(f"Repository changes: {len(changes['added'])} added, "
                       f"{len(changes['modified'])} modified, {len(changes['deleted'])} deleted")
            
            # If no changes, return directly
            if not any(changes.values()):
                logger.info("No changes detected, indexes unchanged")
                return stats
            
            # Handle deleted files
            for file_path in changes['deleted']:
                self._remove_file_segments(file_path)
                if file_path in self.file_timestamps:
                    del self.file_timestamps[file_path]
                stats['files_deleted'] += 1
                logger.info(f"Removed deleted file: {file_path}")
            
            # Handle modified files
            for file_path in changes['modified']:
                # Remove old code segments
                original_count = len(self.code_segments)
                self._remove_file_segments(file_path)
                removed_count = original_count - len(self.code_segments)
                stats['segments_removed'] += removed_count
                
                # Reprocess file
                full_path = self.root_path / file_path
                added_count = self._process_single_file(full_path)
                stats['segments_added'] += added_count
                stats['files_modified'] += 1
                
                logger.info(f"Updated modified file: {file_path} "
                           f"(removed {removed_count}, added {added_count} segments)")
            
            # Handle added files
            for file_path in changes['added']:
                full_path = self.root_path / file_path
                added_count = self._process_single_file(full_path)
                stats['segments_added'] += added_count
                stats['files_added'] += 1
                
                logger.info(f"Added new file: {file_path} ({added_count} segments)")
            
            # If there are changes, rebuild indexes
            logger.info("Rebuilding indexes due to changes...")
            self._build_vector_database()
            self._build_tfidf_database()
        
        logger.info(f"Incremental update completed: "
                   f"{stats['files_added']} files added, "
                   f"{stats['files_modified']} files modified, "
                   f"{stats['files_deleted']} files deleted, "
                   f"{stats['segments_added']} segments added, "
                   f"{stats['segments_removed']} segments removed")
        
        return stats
    
    def parse_repository(self, force_rebuild: bool = False):
        """
        Parse code repository
        
        Args:
            force_rebuild: Whether to force rebuild, ignoring incremental updates
        """
        if not force_rebuild and self.code_segments and self.file_timestamps:
            # Try incremental update
            logger.info("Attempting incremental update...")
            self.incremental_update()
            return
        
        logger.info(f"Starting to parse repository: {self.root_path}")
        
        # Clear existing data
        self.code_segments = []
        self.file_timestamps = {}
        
        # Traverse all code files (including symlinked files)
        code_files = self._get_all_code_files()
        
        logger.info(f"Found {len(code_files)} code files")
        if len(code_files) == 0:
            logger.debug(f"No code files found in directory: {self.root_path}")
            #logger.warning(f"Supported extensions: {', '.join(sorted(self.supported_extensions))}")
            return
        
        # Process each file
        for file_path in code_files:
            self._process_single_file(file_path)
        
        logger.info(f"Total segments created: {len(self.code_segments)}")
        if len(self.code_segments) == 0:
            logger.warning("No code segments were created from the processed files")
            return
        
        # Build vector database
        self._build_vector_database()
        
        # Build TF-IDF database
        self._build_tfidf_database()
    
    def _build_vector_database(self):
        """Build vector database"""
        logger.info("Building vector database...")
        
        # 🚀 延迟加载：确保机器学习库已导入
        _ensure_lazy_imports()
        global np, TfidfVectorizer
        
        # 确保 TF-IDF vectorizer 已初始化
        self._ensure_tfidf_vectorizer()
        
        if not self.code_segments:
            logger.debug("No code segments to vectorize")
            return
        
        # Prepare text data
        texts = [segment.content for segment in self.code_segments]
        
        # Check if we have enough valid content
        non_empty_texts = [text for text in texts if text and text.strip()]
        if len(non_empty_texts) == 0:
            logger.warning("No non-empty code segments found for vectorization")
            return
        
        logger.info(f"Found {len(non_empty_texts)} non-empty code segments for vectorization")
        
        try:
            # Vectorize with TF-IDF
            logger.info("Vectorizing with TF-IDF...")
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
            self.segment_vectors = tfidf_matrix.toarray()
        except ValueError as e:
            error_msg = str(e).lower()
            if "empty vocabulary" in error_msg:
                logger.warning("TF-IDF vectorizer encountered empty vocabulary during vector database build")
                # Create a minimal vocabulary to avoid the error
                dummy_text = "function class variable method import return if else for while"
                texts_with_dummy = texts + [dummy_text]
                try:
                    tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts_with_dummy)
                    # Remove the dummy document from the matrix
                    self.segment_vectors = tfidf_matrix[:-1].toarray()
                    logger.info("Vector database built with fallback method")
                except Exception as fallback_e:
                    logger.error(f"Failed to build vector database even with fallback: {fallback_e}")
                    self.segment_vectors = None
                    return
            elif "max_df corresponds to < documents than min_df" in error_msg:
                logger.warning(f"Not enough documents ({len(texts)}) for current min_df/max_df settings. Adjusting parameters...")
                # Create a new vectorizer with adjusted parameters for small document sets
                fallback_vectorizer = TfidfVectorizer(
                    max_features=min(5000, len(texts) * 100),
                    stop_words=None,
                    ngram_range=(1, 1),  # Use only unigrams for small datasets
                    tokenizer=self._tokenize_code,
                    token_pattern=None,
                    min_df=1,
                    max_df=1.0,  # Include all terms
                    sublinear_tf=True,
                    norm='l2'
                )
                try:
                    tfidf_matrix = fallback_vectorizer.fit_transform(texts)
                    self.segment_vectors = tfidf_matrix.toarray()
                    logger.info(f"Vector database built with adjusted parameters for {len(texts)} documents")
                except Exception as fallback_e:
                    logger.error(f"Failed to build vector database even with adjusted parameters: {fallback_e}")
                    self.segment_vectors = None
                    return
            else:
                logger.error(f"Failed to build vector database: {e}")
                self.segment_vectors = None
                return
        
        # Build FAISS index (if available)
        if _check_faiss_available() and self.segment_vectors is not None:
            global faiss
            logger.info("Building FAISS index...")
            dimension = self.segment_vectors.shape[1]
            self.vector_index = faiss.IndexFlatIP(dimension)  # Inner product index
            
            # Normalize vectors
            faiss.normalize_L2(self.segment_vectors.astype(np.float32))
            self.vector_index.add(self.segment_vectors.astype(np.float32))
        
        logger.info("Vector database built successfully")
    
    def _build_tfidf_database(self):
        """Build TF-IDF database"""
        logger.info("Building TF-IDF database...")
        
        # 🚀 延迟加载：确保机器学习库已导入
        _ensure_lazy_imports()
        
        # 确保 TF-IDF vectorizer 已初始化
        self._ensure_tfidf_vectorizer()
        
        if not self.code_segments:
            logger.debug("No code segments for TF-IDF indexing")
            return
        
        # Prepare text data
        texts = [segment.content for segment in self.code_segments]
        
        # Check if we have enough valid content
        non_empty_texts = [text for text in texts if text and text.strip()]
        if len(non_empty_texts) == 0:
            logger.warning("No non-empty code segments found for TF-IDF indexing")
            return
        
        try:
            # Build TF-IDF matrix
            self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
            logger.info(f"TF-IDF database built with {self.tfidf_matrix.shape[1]} features")
        except ValueError as e:
            error_msg = str(e).lower()
            if "empty vocabulary" in error_msg:
                logger.warning("TF-IDF vectorizer encountered empty vocabulary, likely due to documents containing only stop words or very short content")
                # Create a minimal vocabulary to avoid the error
                # Add a dummy document with common programming terms
                dummy_text = "function class variable method import return if else for while"
                texts_with_dummy = texts + [dummy_text]
                try:
                    self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts_with_dummy)
                    # Remove the dummy document from the matrix
                    self.tfidf_matrix = self.tfidf_matrix[:-1]
                    logger.info(f"TF-IDF database built with fallback method, {self.tfidf_matrix.shape[1]} features")
                except Exception as fallback_e:
                    logger.error(f"Failed to build TF-IDF database even with fallback: {fallback_e}")
                    self.tfidf_matrix = None
            elif "max_df corresponds to < documents than min_df" in error_msg:
                logger.warning(f"Not enough documents ({len(texts)}) for current min_df/max_df settings in TF-IDF. Using fallback vectorizer...")
                # Create a new vectorizer with adjusted parameters for small document sets
                fallback_vectorizer = TfidfVectorizer(
                    max_features=min(5000, len(texts) * 100),
                    stop_words=None,
                    ngram_range=(1, 1),  # Use only unigrams for small datasets
                    tokenizer=self._tokenize_code,
                    token_pattern=None,
                    min_df=1,
                    max_df=1.0,  # Include all terms
                    sublinear_tf=True,
                    norm='l2'
                )
                try:
                    self.tfidf_matrix = fallback_vectorizer.fit_transform(texts)
                    logger.info(f"TF-IDF database built with adjusted parameters, {self.tfidf_matrix.shape[1]} features")
                except Exception as fallback_e:
                    logger.error(f"Failed to build TF-IDF database even with adjusted parameters: {fallback_e}")
                    self.tfidf_matrix = None
            else:
                logger.error(f"Failed to build TF-IDF database: {e}")
                self.tfidf_matrix = None
    
    def vector_search(self, query: str, top_k: int = 5) -> List[SearchResult]:
        """
        Vector similarity search (thread-safe version)
        
        Args:
            query: Query text
            top_k: Number of results to return
            
        Returns:
            List of search results
        """
        # 🚀 延迟加载：确保机器学习库已导入
        _ensure_lazy_imports()
        global np, cosine_similarity
        
        # Use lock to ensure thread safety when reading data
        with self._update_lock:
            if self.segment_vectors is None:
                logger.warning("Vector database not built")
                return []
            
            # Convert query to vector
            query_vector = self.tfidf_vectorizer.transform([query]).toarray()
            
            # Search similar vectors
            if self.vector_index is not None and _check_faiss_available():
                # Use FAISS search
                global faiss
                faiss.normalize_L2(query_vector.astype(np.float32))
                scores, indices = self.vector_index.search(
                    query_vector.astype(np.float32), 
                    min(top_k, len(self.code_segments))
                )
                
                results = []
                for score, idx in zip(scores[0], indices[0]):
                    if idx < len(self.code_segments):
                        result = SearchResult(
                            segment=self.code_segments[idx],
                            score=float(score),
                            search_type='vector'
                        )
                        results.append(result)
            else:
                # Use numpy to calculate cosine similarity
                similarities = cosine_similarity(query_vector, self.segment_vectors)[0]
                top_indices = np.argsort(similarities)[-top_k:][::-1]
                
                results = []
                for idx in top_indices:
                    result = SearchResult(
                        segment=self.code_segments[idx],
                        score=float(similarities[idx]),
                        search_type='vector'
                    )
                    results.append(result)
            
            return results
    
    def keyword_search(self, query: str, top_k: int = 5) -> List[SearchResult]:
        """
        TF-IDF similarity search (thread-safe version)
        
        Args:
            query: Query text
            top_k: Number of results to return
            
        Returns:
            List of search results
        """
        # 🚀 延迟加载：确保机器学习库已导入
        _ensure_lazy_imports()
        global cosine_similarity
        
        # Use lock to ensure thread safety when reading data
        with self._update_lock:
            if self.tfidf_matrix is None:
                logger.warning("TF-IDF database not built")
                return []
            
            # Convert query to TF-IDF vector
            query_vector = self.tfidf_vectorizer.transform([query])
            
            # Calculate similarity with all segments
            similarities = cosine_similarity(query_vector, self.tfidf_matrix)[0]
            top_indices = np.argsort(similarities)[-top_k:][::-1]
            
            results = []
            for idx in top_indices:
                if similarities[idx] > 0:  # Only return results with similarity
                    result = SearchResult(
                        segment=self.code_segments[idx],
                        score=float(similarities[idx]),
                        search_type='keyword'
                    )
                    results.append(result)
            
            return results
    
    def hybrid_search(self, query: str, vector_top_k: int = 5, keyword_top_k: int = 5) -> List[SearchResult]:
        """
        Hybrid search: combine vector search and keyword search
        
        Args:
            query: Query text
            vector_top_k: Number of vector search results
            keyword_top_k: Number of keyword search results
            
        Returns:
            Merged search results list
        """
        vector_results = self.vector_search(query, vector_top_k)
        keyword_results = self.keyword_search(query, keyword_top_k)
        
        # Merge results, remove duplicates
        all_results = {}
        
        # Vector search results weight
        for result in vector_results:
            segment_id = result.segment.segment_id
            if segment_id not in all_results:
                all_results[segment_id] = result
            else:
                # If same segment appears in both searches, take higher score
                if result.score > all_results[segment_id].score:
                    all_results[segment_id] = result
        
        # Keyword search results
        for result in keyword_results:
            segment_id = result.segment.segment_id
            if segment_id not in all_results:
                all_results[segment_id] = result
            else:
                # Hybrid scoring: vector search + keyword search
                combined_score = (all_results[segment_id].score + result.score) / 2
                all_results[segment_id].score = combined_score
                all_results[segment_id].search_type = 'hybrid'
        
        # Sort by score
        sorted_results = sorted(all_results.values(), key=lambda x: x.score, reverse=True)
        
        return sorted_results
    
    def get_repository_stats(self) -> Dict[str, Any]:
        """
        Get code repository statistics
        
        Returns:
            Statistics dictionary
        """
        tfidf_features = self.tfidf_matrix.shape[1] if self.tfidf_matrix is not None else 0
        
        stats = {
            'total_files': len(self.file_timestamps),
            'total_segments': len(self.code_segments),
            'tfidf_features': tfidf_features,
            'last_update': datetime.now().isoformat(),
            'file_types': defaultdict(int),
            'files_by_size': {'small': 0, 'medium': 0, 'large': 0}
        }
        
        # Statistics by file type
        for file_path in self.file_timestamps:
            ext = Path(file_path).suffix.lower()
            stats['file_types'][ext] += 1
        
        # Statistics by file size distribution
        for timestamp in self.file_timestamps.values():
            if timestamp.file_size < 10000:  # < 10KB
                stats['files_by_size']['small'] += 1
            elif timestamp.file_size < 100000:  # < 100KB
                stats['files_by_size']['medium'] += 1
            else:
                stats['files_by_size']['large'] += 1
        
        return stats
    
    def save_database(self, save_path: str):
        """Save database to file"""
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save code segments
        with open(save_path / 'code_segments.pkl', 'wb') as f:
            pickle.dump(self.code_segments, f)
        
        # Save file timestamps
        timestamps_data = {
            file_path: {
                'file_path': ts.file_path,
                'last_modified': ts.last_modified,
                'file_size': ts.file_size,
                'last_checked': ts.last_checked
            }
            for file_path, ts in self.file_timestamps.items()
        }
        with open(save_path / 'file_timestamps.json', 'w', encoding='utf-8') as f:
            json.dump(timestamps_data, f, ensure_ascii=False, indent=2)
        
        # Save vector data
        if self.segment_vectors is not None:
            # 🚀 延迟加载：确保 numpy 已导入
            _ensure_lazy_imports()
            global np
            np.save(save_path / 'segment_vectors.npy', self.segment_vectors)
        
        # Save TF-IDF model and matrix
        if self.tfidf_vectorizer is not None:
            with open(save_path / 'tfidf_vectorizer.pkl', 'wb') as f:
                pickle.dump(self.tfidf_vectorizer, f)
        
        if self.tfidf_matrix is not None:
            with open(save_path / 'tfidf_matrix.pkl', 'wb') as f:
                pickle.dump(self.tfidf_matrix, f)
        

        
        # Save FAISS index
        if self.vector_index is not None and _check_faiss_available():
            global faiss
            faiss.write_index(self.vector_index, str(save_path / 'faiss_index.idx'))
        
        # Save statistics
        stats = self.get_repository_stats()
        with open(save_path / 'repository_stats.json', 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Database saved to {save_path}")
    
    def load_database(self, load_path: str):
        """Load database from file"""
        load_path = Path(load_path)
        
        # 🚀 延迟加载：确保机器学习库已导入（因为可能需要加载向量数据）
        _ensure_lazy_imports()
        global np
        
        # Load code segments
        with open(load_path / 'code_segments.pkl', 'rb') as f:
            self.code_segments = pickle.load(f)
        
        # Load file timestamps
        timestamps_file = load_path / 'file_timestamps.json'
        if timestamps_file.exists():
            with open(timestamps_file, 'r', encoding='utf-8') as f:
                timestamps_data = json.load(f)
            
            self.file_timestamps = {}
            for file_path, data in timestamps_data.items():
                self.file_timestamps[file_path] = FileTimestamp(
                    file_path=data['file_path'],
                    last_modified=data['last_modified'],
                    file_size=data['file_size'],
                    last_checked=data['last_checked']
                )
        
        # Load vector data
        vector_file = load_path / 'segment_vectors.npy'
        if vector_file.exists():
            self.segment_vectors = np.load(vector_file)
        
        # Load TF-IDF model and matrix
        tfidf_vectorizer_file = load_path / 'tfidf_vectorizer.pkl'
        if tfidf_vectorizer_file.exists():
            with open(tfidf_vectorizer_file, 'rb') as f:
                self.tfidf_vectorizer = pickle.load(f)
        
        tfidf_matrix_file = load_path / 'tfidf_matrix.pkl'
        if tfidf_matrix_file.exists():
            with open(tfidf_matrix_file, 'rb') as f:
                self.tfidf_matrix = pickle.load(f)
        

        
        # Load FAISS index
        faiss_index_file = load_path / 'faiss_index.idx'
        if faiss_index_file.exists() and _check_faiss_available():
            global faiss
            self.vector_index = faiss.read_index(str(faiss_index_file))
        
        logger.info(f"Database loaded from {load_path}")
    
    def cleanup(self):
        """Clean up resources used by the parser"""
        try:
            # First stop background update thread
            self.stop_background_update()
            
            # Use lock to ensure cleanup process safety
            with self._update_lock:
                # Clear large data structures to free memory
                self.code_segments.clear()
                self.segment_vectors = None
                self.tfidf_matrix = None
                self.file_timestamps.clear()
                
                # Clear vector index if it exists
                if hasattr(self, 'vector_index') and self.vector_index is not None:
                    self.vector_index = None
                
                # Clear TF-IDF vectorizer
                if hasattr(self, 'tfidf_vectorizer') and self.tfidf_vectorizer is not None:
                    self.tfidf_vectorizer = None
                    
        except Exception as e:
            logger.warning(f"Error during CodeRepositoryParser cleanup: {e}")
    
    def __getstate__(self):
        """Custom pickle state method to handle thread locks"""
        state = self.__dict__.copy()
        # Remove unpicklable thread locks
        state.pop('_update_lock', None)
        # Remove background update thread (it will be recreated if needed)
        state.pop('background_update_thread', None)
        return state
    
    def __setstate__(self, state):
        """Custom unpickle state method to recreate thread locks"""
        self.__dict__.update(state)
        # Recreate thread lock
        self._update_lock = threading.Lock()
        # Background update thread will be recreated when needed
        self.background_update_thread = None

    def _get_code_index_path(self, workspace_root: Optional[str] = None) -> str:
        """
        Get the path to the code index database
        
        Args:
            workspace_root: Workspace root directory (optional, uses self.root_path if not provided)
            
        Returns:
            Path to the code index database
        """
        if workspace_root is None:
            workspace_root = str(self.root_path)
            
        workspace_name = os.path.basename(workspace_root.rstrip('/'))
        
        # 🔧 Fix: Use simpler "code_index" name for workspace directories
        if workspace_name == "workspace":
            db_path = "code_index"
        else:
            db_path = f"{workspace_name.replace('/', '_')}_code_index"
        
        if not os.path.isabs(db_path):
            # 🔧 Fix: Place code index in the workspace's parent directory (output directory)
            # instead of the project root when workspace_root ends with 'workspace'
            if workspace_name == "workspace":
                # If workspace_root is a 'workspace' subdirectory, place index in its parent
                parent_dir = os.path.dirname(workspace_root)
                db_path = os.path.join(parent_dir, db_path)
            else:
                # Otherwise, place in the same directory as workspace_root
                db_path = os.path.join(workspace_root, db_path)
        
        return db_path

    def init_code_parser(self, workspace_root: Optional[str] = None, supported_extensions: Optional[List[str]] = None, 
                         skip_initial_update: bool = False) -> bool:
        """
        Initialize code repository parser with database loading
        
        Args:
            workspace_root: Workspace root directory (optional, uses self.root_path if not provided)
            supported_extensions: Supported file extensions (optional)
            skip_initial_update: Whether to skip initial synchronous update (default: False, ensure initial indexing)
            
        Returns:
            True if initialization successful, False otherwise
        """
        try:
            # Update root path if provided
            if workspace_root is not None:
                self.root_path = Path(workspace_root)
            
            # Update supported extensions if provided
            if supported_extensions is not None:
                self.supported_extensions = set(supported_extensions)
            
            # Get database path
            db_path = self._get_code_index_path(workspace_root)
            
            initialization_success = False
            
            # Try to load existing database
            if os.path.exists(f"{db_path}/code_segments.pkl"):
                try:
                    print_system(f"📚 Loading existing code index database from: {db_path}")
                    self.load_database(db_path)
                    
                    # Always perform initial check to ensure index is up-to-date
                    if not skip_initial_update:
                        print_debug("🔍 Performing initial repository check to ensure index is current...")
                        # Check for repository changes (perform initial check synchronously)
                        changes = self.check_repository_changes()
                        if any(changes.values()):
                            # Perform incremental update
                            update_result = self.incremental_update()
                            if any(count > 0 for count in update_result.values()):
                                self.save_database(db_path)
                            total_changes = sum(update_result.values())
                            print_current(f"📊 Updated code index: {total_changes} changes processed")
                        else:
                            print_debug("✅ Code index is already up-to-date")
                    
                    # Verify that we have segments available for search
                    if len(self.code_segments) == 0:
                        # Check if this is an empty directory (no code files)
                        code_files = self._get_all_code_files()
                        if len(code_files) == 0:
                            # Empty directory - no need to rebuild, just mark as successful
                            initialization_success = True
                            print_system("📁 Code index initialized for empty workspace")
                        else:
                            # Has code files but no segments - need to rebuild
                            print_debug("⚠️ No code segments found in database, rebuilding index...")
                            initialization_success = self._rebuild_code_index(db_path)
                    else:
                        initialization_success = True
                        print_debug(f"📚 Code index loaded successfully: {len(self.code_segments)} code segments available")
                    
                except Exception as e:
                    print_system(f"⚠️ Failed to load code index database: {e}, will recreate")
                    initialization_success = self._rebuild_code_index(db_path)
            else:
                # Create new database - always perform initial indexing
                print_system(f"🏗️ Creating new code index database at: {db_path}")
                initialization_success = self._rebuild_code_index(db_path)
                if initialization_success:
                    print_system(f"✅ New code index created successfully: {len(self.code_segments)} code segments indexed")
            
            # If initialization successful, start background update thread
            if initialization_success:
                self.start_background_update()
                print_system(f"✅ Code repository parser initialization completed, background update enabled")
                print_system(f"🔍 Code search functionality is now available")
            
            return initialization_success
                
        except Exception as e:
            print_error(f"❌ Failed to initialize code repository parser: {e}")
            return False

    def _rebuild_code_index(self, db_path: Optional[str] = None) -> bool:
        """
        Rebuild code index
        
        Args:
            db_path: Database path (optional, will generate if not provided)
            
        Returns:
            True if rebuild successful, False otherwise
        """
        try:
            if db_path is None:
                db_path = self._get_code_index_path()
            
            print_system(f"🔧 Building code index for workspace: {os.path.basename(str(self.root_path))}")
            print_system(f"📁 Scanning directory: {self.root_path}")
            
            # Parse repository
            self.parse_repository(force_rebuild=True)
            
            # Check if we have segments
            #if len(self.code_segments) == 0:
                #print_system("ℹ️ No code files found in directory - this is normal for empty or new projects")
                #print_current(f"📋 Supported extensions: {', '.join(sorted(self.supported_extensions))}")
                #print_current("✅ Code index initialized for empty workspace")
            #else:
                #print_current(f"📊 Indexed {len(self.code_segments)} code segments from {len(self.file_timestamps)} files")
            
            # Save database (even if empty)
            self.save_database(db_path)
            #print_current(f"💾 Code index saved to: {db_path}")
            return True
            
        except Exception as e:
            print_current(f"❌ Failed to rebuild code index: {e}")
            return False

    def perform_incremental_update(self, db_path: Optional[str] = None) -> bool:
        """
        Perform incremental update
        
        Args:
            db_path: Database path (optional, will generate if not provided)
            
        Returns:
            True if update successful, False otherwise
        """
        try:
            if db_path is None:
                db_path = self._get_code_index_path()
                
            update_result = self.incremental_update()
            
            if any(count > 0 for count in update_result.values()):
                self.save_database(db_path)
                return True
            return True
            
        except Exception as e:
            print_current(f"⚠️ Code repository update failed: {e}")
            return False


def test_code_repository_parser():
    """Test code repository parser"""
    print_current("=== Code Repository Parser Test ===")
    
    # Initialize parser (use current directory as test)
    parser = CodeRepositoryParser(
        root_path=".",  # Current directory
        segment_size=100,  # Reduce segment size for testing
    )
    
    # Parse repository
    print_current("1. Parsing code repository...")
    parser.parse_repository()
    
    print_current(f"Total parsed {len(parser.code_segments)} code segments")
    
    # Test queries
    test_queries = [
        "file reading and processing",
        "class definition",
        "vector search",
        "database storage",
        "exception handling"
    ]
    
    print_current("\n2. Testing search functionality...")
    
    for query in test_queries:
        print_current(f"\nQuery: '{query}'")
        print_current("-" * 50)
        
        # Vector search
        print_current("Vector search results:")
        vector_results = parser.vector_search(query, top_k=3)
        for i, result in enumerate(vector_results[:3], 1):
            print_current(f"  {i}. File: {result.segment.file_path}")
            print_current(f"     Lines: {result.segment.start_line}-{result.segment.end_line}")
            print_current(f"     Score: {result.score:.4f}")
            print_current(f"     Type: {result.search_type}")
            print_current(f"     Preview: {result.segment.content[:100]}...")
        
        # Keyword search
        print_current("Keyword search results:")
        keyword_results = parser.keyword_search(query, top_k=3)
        for i, result in enumerate(keyword_results[:3], 1):
            print_current(f"  {i}. File: {result.segment.file_path}")
            print_current(f"     Lines: {result.segment.start_line}-{result.segment.end_line}")
            print_current(f"     Score: {result.score:.4f}")
            print_current(f"     Type: {result.search_type}")
            print_current(f"     Preview: {result.segment.content[:100]}...")
        
        # Hybrid search
        print_current("Hybrid search results:")
        hybrid_results = parser.hybrid_search(query, vector_top_k=3, keyword_top_k=3)
        for i, result in enumerate(hybrid_results[:3], 1):
            print_current(f"  {i}. File: {result.segment.file_path}")
            print_current(f"     Lines: {result.segment.start_line}-{result.segment.end_line}")
            print_current(f"     Score: {result.score:.4f}")
            print_current(f"     Type: {result.search_type}")
            print_current(f"     Preview: {result.segment.content[:100]}...")
    
    # Test save and load
    print_current("\n3. Testing database save and load...")
    save_path = "test_database"
    parser.save_database(save_path)
    
    # Create new parser instance and load database
    new_parser = CodeRepositoryParser(root_path=".")
    new_parser.load_database(save_path)
    
    print_current(f"Segment count after loading: {len(new_parser.code_segments)}")
    
    # Test search functionality after loading
    test_query = "test query"
    results = new_parser.hybrid_search(test_query, vector_top_k=2, keyword_top_k=2)
    print_current(f"Search for '{test_query}' after loading got {len(results)} results")
    
    print_current("\n=== Test Complete ===")


if __name__ == "__main__":
    test_code_repository_parser() 