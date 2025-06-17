#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright (c) 2025 AGI Bot Research Group.

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
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional
from dataclasses import dataclass
from collections import defaultdict
import re
from tqdm import tqdm
import logging
import time
from datetime import datetime

# Machine learning related libraries
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import jieba
import jieba.analyse

# Vectorization related libraries - removed sentence_transformers
# Vector database related libraries
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    # print("Warning: faiss not available. Will use numpy for vector storage.")  # Moved warning display to main.py

# Configure logging - only show WARNING and above level logs
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

@dataclass
class CodeSegment:
    """Code segment data structure"""
    content: str
    file_path: str
    start_line: int
    end_line: int
    segment_id: str

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

class CodeRepositoryParser:
    """Code repository parser"""
    
    def __init__(self, 
                 root_path: str,
                 segment_size: int = 200,
                 supported_extensions: List[str] = None):
        """
        Initialize code repository parser
        
        Args:
            root_path: Code repository root path
            segment_size: Number of lines per segment
            supported_extensions: Supported file extensions
        """
        self.root_path = Path(root_path)
        self.segment_size = segment_size
        
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
        self.segment_vectors: Optional[np.ndarray] = None
        self.vector_index = None
        
        # File timestamp records
        self.file_timestamps: Dict[str, FileTimestamp] = {}
        
        # TF-IDF database
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=10000,
            stop_words=None,  # Don't use stop words for code
            ngram_range=(1, 3),
            tokenizer=self._tokenize_code,
            token_pattern=None  # Explicitly set to None to avoid warning when using custom tokenizer
        )
        self.tfidf_matrix = None
    
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
        
        # Add Chinese word segmentation support (for Chinese comments)
        chinese_text = re.sub(r'[^\u4e00-\u9fff]+', ' ', text)
        if chinese_text.strip():
            chinese_tokens = jieba.lcut(chinese_text)
            identifiers.extend([token for token in chinese_tokens if len(token) > 1])
        
        # Remove common programming keywords (can be adjusted as needed)
        programming_keywords = {
            'if', 'else', 'for', 'while', 'def', 'class', 'import', 'from',
            'return', 'try', 'except', 'with', 'as', 'in', 'is', 'and', 'or', 'not'
        }
        
        return [token.lower() for token in identifiers 
                if token.lower() not in programming_keywords and len(token) > 2]
    
    def _is_code_file(self, file_path: Path) -> bool:
        """
        Check if it's a code file
        
        Args:
            file_path: File path
            
        Returns:
            Whether it's a code file
        """
        return file_path.suffix.lower() in self.supported_extensions
    
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
    
    def _get_file_timestamp(self, file_path: Path) -> FileTimestamp:
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
    
    def check_repository_changes(self) -> Dict[str, List[str]]:
        """
        Check file changes in code repository
        
        Returns:
            Change information dictionary containing lists of added, modified, deleted files
        """
        logger.info("Checking repository changes...")
        
        changes = {
            'added': [],
            'modified': [],
            'deleted': []
        }
        
        # Get all current code files
        current_files = set()
        for file_path in self.root_path.rglob('*'):
            if file_path.is_file() and self._is_code_file(file_path):
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
        Incremental update of code repository
        
        Returns:
            Update statistics
        """
        logger.info("Starting incremental update...")
        
        # Check changes
        changes = self.check_repository_changes()
        
        stats = {
            'files_added': 0,
            'files_modified': 0,
            'files_deleted': 0,
            'segments_added': 0,
            'segments_removed': 0
        }
        
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
        if any(changes.values()):
            logger.info("Rebuilding indexes due to changes...")
            self._build_vector_database()
            self._build_tfidf_database()
        else:
            logger.info("No changes detected, indexes unchanged")
        
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
        
        # Traverse all code files
        code_files = []
        for file_path in self.root_path.rglob('*'):
            if file_path.is_file() and self._is_code_file(file_path):
                code_files.append(file_path)
        
        logger.info(f"Found {len(code_files)} code files")
        
        # Process each file
        for file_path in tqdm(code_files, desc="Processing files"):
            self._process_single_file(file_path)
        
        logger.info(f"Total segments created: {len(self.code_segments)}")
        
        # Build vector database
        self._build_vector_database()
        
        # Build TF-IDF database
        self._build_tfidf_database()
    
    def _build_vector_database(self):
        """Build vector database"""
        logger.info("Building vector database...")
        
        if not self.code_segments:
            logger.debug("No code segments to vectorize")
            return
        
        # Prepare text data
        texts = [segment.content for segment in self.code_segments]
        
        # Vectorize with TF-IDF
        logger.info("Vectorizing with TF-IDF...")
        tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
        self.segment_vectors = tfidf_matrix.toarray()
        
        # Build FAISS index (if available)
        if FAISS_AVAILABLE and self.segment_vectors is not None:
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
        
        if not self.code_segments:
            logger.debug("No code segments for TF-IDF indexing")
            return
        
        # Prepare text data
        texts = [segment.content for segment in self.code_segments]
        
        # Build TF-IDF matrix
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
        
        logger.info(f"TF-IDF database built with {self.tfidf_matrix.shape[1]} features")
    
    def vector_search(self, query: str, top_k: int = 5) -> List[SearchResult]:
        """
        Vector similarity search
        
        Args:
            query: Query text
            top_k: Number of results to return
            
        Returns:
            List of search results
        """
        if self.segment_vectors is None:
            logger.warning("Vector database not built")
            return []
        
        # Convert query to vector
        query_vector = self.tfidf_vectorizer.transform([query]).toarray()
        
        # Search similar vectors
        if self.vector_index is not None and FAISS_AVAILABLE:
            # Use FAISS search
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
        TF-IDF similarity search
        
        Args:
            query: Query text
            top_k: Number of results to return
            
        Returns:
            List of search results
        """
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
            np.save(save_path / 'segment_vectors.npy', self.segment_vectors)
        
        # Save TF-IDF model and matrix
        if self.tfidf_vectorizer is not None:
            with open(save_path / 'tfidf_vectorizer.pkl', 'wb') as f:
                pickle.dump(self.tfidf_vectorizer, f)
        
        if self.tfidf_matrix is not None:
            with open(save_path / 'tfidf_matrix.pkl', 'wb') as f:
                pickle.dump(self.tfidf_matrix, f)
        

        
        # Save FAISS index
        if self.vector_index is not None and FAISS_AVAILABLE:
            faiss.write_index(self.vector_index, str(save_path / 'faiss_index.idx'))
        
        # Save statistics
        stats = self.get_repository_stats()
        with open(save_path / 'repository_stats.json', 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Database saved to {save_path}")
    
    def load_database(self, load_path: str):
        """Load database from file"""
        load_path = Path(load_path)
        
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
        if faiss_index_file.exists() and FAISS_AVAILABLE:
            self.vector_index = faiss.read_index(str(faiss_index_file))
        
        logger.info(f"Database loaded from {load_path}")


def test_code_repository_parser():
    """Test code repository parser"""
    print("=== Code Repository Parser Test ===")
    
    # Initialize parser (use current directory as test)
    parser = CodeRepositoryParser(
        root_path=".",  # Current directory
        segment_size=100,  # Reduce segment size for testing
    )
    
    # Parse repository
    print("1. Parsing code repository...")
    parser.parse_repository()
    
    print(f"Total parsed {len(parser.code_segments)} code segments")
    
    # Test queries
    test_queries = [
        "file reading and processing",
        "class definition",
        "vector search",
        "database storage",
        "exception handling"
    ]
    
    print("\n2. Testing search functionality...")
    
    for query in test_queries:
        print(f"\nQuery: '{query}'")
        print("-" * 50)
        
        # Vector search
        print("Vector search results:")
        vector_results = parser.vector_search(query, top_k=3)
        for i, result in enumerate(vector_results[:3], 1):
            print(f"  {i}. File: {result.segment.file_path}")
            print(f"     Lines: {result.segment.start_line}-{result.segment.end_line}")
            print(f"     Score: {result.score:.4f}")
            print(f"     Type: {result.search_type}")
            print(f"     Preview: {result.segment.content[:100]}...")
            print()
        
        # Keyword search
        print("Keyword search results:")
        keyword_results = parser.keyword_search(query, top_k=3)
        for i, result in enumerate(keyword_results[:3], 1):
            print(f"  {i}. File: {result.segment.file_path}")
            print(f"     Lines: {result.segment.start_line}-{result.segment.end_line}")
            print(f"     Score: {result.score:.4f}")
            print(f"     Type: {result.search_type}")
            print(f"     Preview: {result.segment.content[:100]}...")
            print()
        
        # Hybrid search
        print("Hybrid search results:")
        hybrid_results = parser.hybrid_search(query, vector_top_k=3, keyword_top_k=3)
        for i, result in enumerate(hybrid_results[:3], 1):
            print(f"  {i}. File: {result.segment.file_path}")
            print(f"     Lines: {result.segment.start_line}-{result.segment.end_line}")
            print(f"     Score: {result.score:.4f}")
            print(f"     Type: {result.search_type}")
            print(f"     Preview: {result.segment.content[:100]}...")
            print()
    
    # Test save and load
    print("\n3. Testing database save and load...")
    save_path = "test_database"
    parser.save_database(save_path)
    
    # Create new parser instance and load database
    new_parser = CodeRepositoryParser(root_path=".")
    new_parser.load_database(save_path)
    
    print(f"Segment count after loading: {len(new_parser.code_segments)}")
    
    # Test search functionality after loading
    test_query = "test query"
    results = new_parser.hybrid_search(test_query, vector_top_k=2, keyword_top_k=2)
    print(f"Search for '{test_query}' after loading got {len(results)} results")
    
    print("\n=== Test Complete ===")


if __name__ == "__main__":
    test_code_repository_parser() 