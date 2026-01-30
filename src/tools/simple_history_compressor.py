#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright (c) 2025 AGI Agent Research Group.

Simple History Compressor - Handle history compression by truncating long JSON fields
"""

import json
import re
from typing import Dict, Any, List, Union
from datetime import datetime
from .print_system import print_current


class SimpleHistoryCompressor:
    """A simple history compressor that compresses history records by truncating long field contents."""
    
    def __init__(self, 
                 min_length: int = 500,
                 head_length: int = 100,
                 tail_length: int = 100,
                 ellipsis: str = "\n...[omitted {} chars]...\n",
                 aggressive_mode: bool = False):
        """
        Initialize the simple history compressor.
        
        Args:
            min_length: Minimum number of characters to trigger compression. Content shorter than this will not be compressed.
            head_length: Number of characters to keep at the beginning.
            tail_length: Number of characters to keep at the end.
            ellipsis: Ellipsis format, {} will be replaced by the number of omitted characters.
            aggressive_mode: If True, use more aggressive compression (lower thresholds, smaller head/tail).
        """
        if aggressive_mode:
            # More aggressive compression settings
            self.min_length = min(min_length, 200)
            self.head_length = min(head_length, 50)
            self.tail_length = min(tail_length, 50)
        else:
            self.min_length = min_length
            self.head_length = head_length
            self.tail_length = tail_length
        self.ellipsis = ellipsis
        self.aggressive_mode = aggressive_mode
        
    def compress_history(self, task_history: List[Dict[str, Any]], target_compression_ratio: float = None, trigger_length: int = None) -> List[Dict[str, Any]]:
        """
        Compress the history records.
        
        Args:
            task_history: The original list of history records.
            target_compression_ratio: Optional target compression ratio (0.0-1.0). If provided, 
                                     will iteratively compress until target is reached.
            trigger_length: Optional trigger length threshold. If provided, compression will only 
                          occur if total content length exceeds this threshold.
            
        Returns:
            The compressed list of history records.
        """
        if not task_history:
            return task_history
        
        # Check trigger length if provided
        if trigger_length is not None:
            total_length = sum(self._calculate_record_size(record) for record in task_history)
            if total_length <= trigger_length:
                # Content does not exceed trigger length, return original history
                return task_history
            
        #print_current(f"🗜️ Starting simple history compression, original record count: {len(task_history)}")
        
        compressed_history = []
        total_original_chars = 0
        total_compressed_chars = 0
        compressed_fields_count = 0
        
        for i, record in enumerate(task_history):
            compressed_record = self._compress_single_record(record.copy())
            compressed_history.append(compressed_record)
            
            # Statistics for compression effect
            original_size = self._calculate_record_size(record)
            compressed_size = self._calculate_record_size(compressed_record)
            
            total_original_chars += original_size
            total_compressed_chars += compressed_size
            
            # Count the number of compressed fields
            compressed_fields_count += self._count_compressed_fields(record, compressed_record)
        
        # If target compression ratio is specified and not achieved, apply more aggressive compression
        if target_compression_ratio is not None and total_original_chars > 0:
            current_ratio = 1.0 - (total_compressed_chars / total_original_chars)
            if current_ratio < target_compression_ratio:
                # Apply more aggressive compression iteratively
                iteration = 0
                max_iterations = 5
                while current_ratio < target_compression_ratio and iteration < max_iterations:
                    iteration += 1
                    # Temporarily reduce head/tail lengths for more aggressive compression
                    original_head = self.head_length
                    original_tail = self.tail_length
                    original_min = self.min_length
                    
                    # Progressively reduce lengths
                    self.head_length = max(20, int(self.head_length * 0.7))
                    self.tail_length = max(20, int(self.tail_length * 0.7))
                    self.min_length = max(100, int(self.min_length * 0.7))
                    
                    # Re-compress
                    compressed_history = []
                    total_compressed_chars = 0
                    for record in task_history:
                        compressed_record = self._compress_single_record(record.copy())
                        compressed_history.append(compressed_record)
                        total_compressed_chars += self._calculate_record_size(compressed_record)
                    
                    current_ratio = 1.0 - (total_compressed_chars / total_original_chars) if total_original_chars > 0 else 0
                    
                    # Restore original values
                    self.head_length = original_head
                    self.tail_length = original_tail
                    self.min_length = original_min
        
        return compressed_history
    
    def _compress_single_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compress a single history record.
        
        Args:
            record: A single history record.
            
        Returns:
            The compressed record.
        """
        # List of fields to check
        fields_to_check = ['prompt', 'result', 'content', 'response', 'output', 'data']
        
        for field in fields_to_check:
            if field in record:
                record[field] = self._compress_field_content(record[field])
        
        return record
    
    def _compress_field_content(self, content: Any) -> Any:
        """
        Compress the content of a field.
        
        Args:
            content: The field content.
            
        Returns:
            The compressed content.
        """
        if isinstance(content, str):
            return self._compress_string(content)
        elif isinstance(content, dict):
            return self._compress_dict(content)
        elif isinstance(content, list):
            return self._compress_list(content)
        else:
            return content
    
    def _compress_string(self, text: str) -> str:
        """
        Compress string content.
        
        Args:
            text: The original string.
            
        Returns:
            The compressed string.
        """
        if not text or len(text) <= self.min_length:
            return text
        
        # Check if the content looks like JSON
        if self._looks_like_json(text):
            return self._compress_json_string(text)
        
        # For normal strings, directly truncate and keep head and tail
        return self._truncate_string(text)
    
    def _compress_dict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compress dictionary content.
        
        Args:
            data: The original dictionary.
            
        Returns:
            The compressed dictionary.
        """
        compressed_dict = {}
        
        for key, value in data.items():
            compressed_dict[key] = self._compress_field_content(value)
        
        return compressed_dict
    
    def _compress_list(self, data: List[Any]) -> List[Any]:
        """
        Compress list content.
        
        Args:
            data: The original list.
            
        Returns:
            The compressed list.
        """
        compressed_list = []
        
        for item in data:
            compressed_list.append(self._compress_field_content(item))
        
        return compressed_list
    
    def _looks_like_json(self, text: str) -> bool:
        """
        Check if a string looks like JSON format.
        
        Args:
            text: The string to check.
            
        Returns:
            Whether it looks like JSON format.
        """
        text = text.strip()
        return (text.startswith('{') and text.endswith('}')) or \
               (text.startswith('[') and text.endswith(']'))
    
    def _compress_json_string(self, text: str) -> str:
        """
        Compress a JSON-formatted string.
        
        Args:
            text: The JSON-formatted string.
            
        Returns:
            The compressed string.
        """
        try:
            # Try to parse JSON
            json_data = json.loads(text)
            # Recursively compress JSON content
            compressed_json = self._compress_field_content(json_data)
            # Convert back to string
            return json.dumps(compressed_json, ensure_ascii=False, indent=2)
        except (json.JSONDecodeError, Exception):
            # If not valid JSON, treat as normal string
            return self._truncate_string(text)
    
    def _truncate_string(self, text: str) -> str:
        """
        Truncate a string, keeping the head and tail parts.
        
        Args:
            text: The original string.
            
        Returns:
            The truncated string.
        """
        if len(text) <= self.min_length:
            return text
        
        # Special handling for strings containing "Tool execution results:"
        marker = "Tool execution results:"
        if marker in text:
            return self._truncate_string_with_marker(text, marker)
        
        # Calculate the number of omitted characters
        omitted_chars = len(text) - self.head_length - self.tail_length
        
        # Make sure not negative
        if omitted_chars <= 0:
            return text
        
        # Get head and tail parts
        head_part = text[:self.head_length]
        tail_part = text[-self.tail_length:]
        
        # Create ellipsis
        ellipsis_text = self.ellipsis.format(omitted_chars)
        
        return head_part + ellipsis_text + tail_part
    
    def _truncate_string_with_marker(self, text: str, marker: str) -> str:
        """
        Truncate a string that contains a marker, compressing parts before and after the marker separately.
        
        Args:
            text: The original string containing the marker.
            marker: The marker string (e.g., "Tool execution results:").
            
        Returns:
            The truncated string with marker preserved.
        """
        # Find the marker position
        marker_pos = text.find(marker)
        if marker_pos == -1:
            # Should not happen, but fallback to normal truncation (avoid recursion)
            omitted_chars = len(text) - self.head_length - self.tail_length
            if omitted_chars <= 0:
                return text
            head_part = text[:self.head_length]
            tail_part = text[-self.tail_length:]
            ellipsis_text = self.ellipsis.format(omitted_chars)
            return head_part + ellipsis_text + tail_part
        
        # Split into three parts: before marker, marker, after marker
        before_marker = text[:marker_pos]
        marker_text = marker
        after_marker = text[marker_pos + len(marker):]
        
        # Compress the part before marker (if long enough)
        if len(before_marker) > self.min_length:
            omitted_before = len(before_marker) - self.head_length - self.tail_length
            if omitted_before > 0:
                before_head = before_marker[:self.head_length]
                before_tail = before_marker[-self.tail_length:]
                before_ellipsis = self.ellipsis.format(omitted_before)
                compressed_before = before_head + before_ellipsis + before_tail
            else:
                compressed_before = before_marker
        else:
            compressed_before = before_marker
        
        # Compress the part after marker (if long enough)
        if len(after_marker) > self.min_length:
            omitted_after = len(after_marker) - self.head_length - self.tail_length
            if omitted_after > 0:
                after_head = after_marker[:self.head_length]
                after_tail = after_marker[-self.tail_length:]
                after_ellipsis = self.ellipsis.format(omitted_after)
                compressed_after = after_head + after_ellipsis + after_tail
            else:
                compressed_after = after_marker
        else:
            compressed_after = after_marker
        
        # Combine: compressed before + marker + compressed after
        return compressed_before + marker_text + compressed_after
    
    def _calculate_record_size(self, record: Dict[str, Any]) -> int:
        """
        Calculate the number of characters in a record.
        
        Args:
            record: The history record.
            
        Returns:
            Number of characters.
        """
        try:
            return len(json.dumps(record, ensure_ascii=False))
        except Exception:
            return len(str(record))
    
    def _count_compressed_fields(self, original_record: Dict[str, Any], 
                                compressed_record: Dict[str, Any]) -> int:
        """
        Count the number of compressed fields.
        
        Args:
            original_record: The original record.
            compressed_record: The compressed record.
            
        Returns:
            The number of compressed fields.
        """
        compressed_count = 0
        fields_to_check = ['prompt', 'result', 'content', 'response', 'output', 'data']
        
        for field in fields_to_check:
            if field in original_record and field in compressed_record:
                original_size = len(str(original_record[field]))
                compressed_size = len(str(compressed_record[field]))
                if compressed_size < original_size:
                    compressed_count += 1
        
        return compressed_count
    
    def get_compression_stats(self, original_history: List[Dict[str, Any]], 
                            compressed_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Get compression statistics.
        
        Args:
            original_history: The original history records.
            compressed_history: The compressed history records.
            
        Returns:
            Compression statistics.
        """
        total_original_chars = sum(self._calculate_record_size(record) for record in original_history)
        total_compressed_chars = sum(self._calculate_record_size(record) for record in compressed_history)
        
        compression_ratio = (1 - total_compressed_chars / total_original_chars) * 100 if total_original_chars > 0 else 0
        saved_chars = total_original_chars - total_compressed_chars
        
        # Estimate token savings (rough estimate: 1 token ≈ 4 chars)
        estimated_token_savings = saved_chars // 4
        
        return {
            'original_chars': total_original_chars,
            'compressed_chars': total_compressed_chars,
            'saved_chars': saved_chars,
            'compression_ratio': compression_ratio,
            'estimated_token_savings': estimated_token_savings,
            'original_records': len(original_history),
            'compressed_records': len(compressed_history)
        }
