#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright (c) 2025 AGI Bot Research Group.

History Optimizer - Handle image data optimization in multi-turn conversations
"""

import re
import hashlib
import base64
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime
from .print_system import print_current
import os # Added for file path checks


class HistoryOptimizer:
    """History optimizer, specialized in handling image data deduplication and optimization"""
    
    def __init__(self, workspace_root: Optional[str] = None):
        """
        Initialize history optimizer
        
        Args:
            workspace_root: Workspace root directory path
        """
        self.workspace_root = workspace_root
        self.image_cache = {}  # Image cache {hash: metadata}
        self.image_references = {}  # Image references {hash: reference_id}
        self.processed_rounds = set()  # Processed rounds
        
    def optimize_history_for_context(self, task_history: List[Dict[str, Any]], 
                                     keep_recent_images: int = 1) -> List[Dict[str, Any]]:
        """
        Optimize history records to reduce token consumption
        
        Args:
            task_history: Original history records
            keep_recent_images: Keep original image data from the most recent N rounds (default 1 round)
            
        Returns:
            Optimized history records
        """
        if not task_history:
            return task_history
        
        # Remove verbose print statements, keep only essential info
        # print_current(f"🔧 Starting history optimization, original records: {len(task_history)}")
        # print_current(f"📸 Will keep original image data from the most recent {keep_recent_images} rounds")
        
        optimized_history = []
        image_summary = {}  # Record summary information for all images
        
        # Calculate the range of latest rounds that need protection
        total_records = len(task_history)
        protected_start_index = max(0, total_records - keep_recent_images)
        
        for i, record in enumerate(task_history):
            # Determine if this is a latest round that needs protection
            is_recent_record = i >= protected_start_index
            
            if is_recent_record:
                # Latest rounds: preserve original image data, but still need to record in summary (for indexing)
                optimized_record = self._analyze_record_without_optimization(record, i, image_summary)
                # print_current(f"🔒 Protecting round {i}, preserving original image data")
            else:
                # Historical rounds: apply optimization
                optimized_record = self._optimize_single_record(record, i, image_summary)
                # print_current(f"🔧 Optimizing round {i}, converting image data to references")
            
            optimized_history.append(optimized_record)
        
        # If images were processed, add image index description
        if image_summary:
            index_record = self._create_image_index_record(image_summary, keep_recent_images)
            optimized_history.insert(0, index_record)
        
        # Simplified output with just the key results
        if image_summary:
            total_optimized = sum(info['occurrences'] - (1 if info.get('has_recent_original', False) else 0) 
                                for info in image_summary.values())
            # print_current(f"✅ History optimization completed, optimized records: {len(optimized_history)}")
            # print_current(f"📊 Image data optimization: {len(image_summary)} images processed")
        
        return optimized_history
    
    def _optimize_single_record(self, record: Dict[str, Any], record_index: int, 
                               image_summary: Dict[str, Any]) -> Dict[str, Any]:
        """
        Optimize a single history record
        
        Args:
            record: Single history record
            record_index: Record index
            image_summary: Image summary dictionary
            
        Returns:
            Optimized record
        """
        # Copy record to avoid modifying original data
        optimized_record = record.copy()
        
        # Check and optimize prompt field
        if 'prompt' in record:
            optimized_record['prompt'] = self._optimize_text_content(
                record['prompt'], record_index, 'prompt', image_summary
            )
        
        # Check and optimize result field
        if 'result' in record:
            optimized_record['result'] = self._optimize_text_content(
                record['result'], record_index, 'result', image_summary
            )
        
        # Check and optimize content field
        if 'content' in record:
            optimized_record['content'] = self._optimize_text_content(
                record['content'], record_index, 'content', image_summary
            )
        
        return optimized_record
    
    def _optimize_text_content(self, text: str, record_index: int, field_name: str, 
                              image_summary: Dict[str, Any]) -> str:
        """
        Optimize image data in text content
        
        Args:
            text: Original text
            record_index: Record index
            field_name: Field name
            image_summary: Image summary dictionary
            
        Returns:
            Optimized text
        """
        if not text or not isinstance(text, str):
            return text
        
        # Detect base64 image data
        base64_pattern = r'[A-Za-z0-9+/]{500,}={0,2}'
        matches = list(re.finditer(base64_pattern, text))
        
        if not matches:
            return text
        
        # print_current(f"🖼️ Found {len(matches)} image data in record {record_index}'s {field_name} field")
        
        optimized_text = text
        offset = 0  # Track text offset after replacement
        
        for match_index, match in enumerate(matches):
            base64_data = match.group()
            
            # Calculate image hash
            image_hash = hashlib.md5(base64_data.encode()).hexdigest()[:16]
            
            # Generate image reference ID
            image_ref_id = f"IMG_{image_hash}"
            
            # Check and extract file path markers
            file_info = self._extract_file_path_info(base64_data)
            
            # Clean base64 data (remove file path markers)
            clean_base64_data = self._clean_base64_data(base64_data)
            
            # Analyze image information
            image_info = self._analyze_image_data(clean_base64_data, record_index, match_index)
            
            # Add to image summary
            if image_ref_id not in image_summary:
                image_summary[image_ref_id] = {
                    'hash': image_hash,
                    'size_chars': len(clean_base64_data),
                    'estimated_size_kb': len(clean_base64_data) * 3 // 4 // 1024,  # Base64 decoded size estimation
                    'first_seen_record': record_index,
                    'occurrences': 1,
                    'format_info': image_info,
                    'file_info': file_info,  # Add file information
                    'original_data': clean_base64_data[:100] + "..." if len(clean_base64_data) > 100 else clean_base64_data
                }
            else:
                image_summary[image_ref_id]['occurrences'] += 1
                # Update file information (if new one is more complete)
                if file_info and not image_summary[image_ref_id].get('file_info'):
                    image_summary[image_ref_id]['file_info'] = file_info
            
            # Create image reference replacement text
            replacement_text = self._create_image_reference(image_ref_id, image_info)
            
            # Calculate position in adjusted text
            start_pos = match.start() + offset
            end_pos = match.end() + offset
            
            # Replace original data (including markers) with reference
            optimized_text = (optimized_text[:start_pos] + 
                            replacement_text + 
                            optimized_text[end_pos:])
            
            # Update offset
            offset += len(replacement_text) - len(base64_data)
            
            # Show optimization effect (use cleaned data length for statistics)
            # print_current(f"   ✅ Image {match_index+1}: {len(clean_base64_data)} chars → {len(replacement_text)} chars (reduced {len(clean_base64_data) - len(replacement_text)} chars)")
        
        return optimized_text
    
    def _extract_file_path_info(self, text_with_marker: str) -> Dict[str, str]:
        """
        Extract file path information from marked text
        
        Args:
            text_with_marker: Text that may contain file path markers
            
        Returns:
            Dictionary containing file path information
        """
        file_info = {}
        
        # Check saved file marker [FILE_SAVED:path]
        saved_match = re.search(r'\[FILE_SAVED:([^\]]+)\]', text_with_marker)
        if saved_match:
            file_info['file_path'] = saved_match.group(1)
            file_info['file_type'] = 'saved'
        
        # Check source file marker [FILE_SOURCE:path]
        source_match = re.search(r'\[FILE_SOURCE:([^\]]+)\]', text_with_marker)
        if source_match:
            file_info['source_path'] = source_match.group(1)
            file_info['file_type'] = 'source' if not file_info.get('file_type') else file_info['file_type']
        
        return file_info
    
    def _clean_base64_data(self, text_with_marker: str) -> str:
        """
        Remove file path markers from text, return cleaned base64 data
        
        Args:
            text_with_marker: Text that may contain file path markers
            
        Returns:
            Cleaned base64 data
        """
        # Remove all file path markers
        cleaned_text = re.sub(r'\[FILE_SAVED:[^\]]+\]', '', text_with_marker)
        cleaned_text = re.sub(r'\[FILE_SOURCE:[^\]]+\]', '', cleaned_text)
        
        return cleaned_text
    
    def _analyze_record_without_optimization(self, record: Dict[str, Any], record_index: int, 
                                           image_summary: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze images in record without optimization, for latest rounds
        
        Args:
            record: History record
            record_index: Record index  
            image_summary: Image summary dictionary
            
        Returns:
            Original record (not optimized)
        """
        # Analyze image data but don't replace
        for field in ['prompt', 'result', 'content']:
            if field in record and isinstance(record[field], str):
                self._analyze_text_content_without_replacement(
                    record[field], record_index, field, image_summary
                )
        
        return record.copy()  # Return original record copy
    
    def _analyze_text_content_without_replacement(self, text: str, record_index: int, 
                                                field_name: str, image_summary: Dict[str, Any]) -> None:
        """
        Analyze image data in text without replacement
        
        Args:
            text: Original text
            record_index: Record index
            field_name: Field name
            image_summary: Image summary dictionary
        """
        if not text or not isinstance(text, str):
            return
        
        # Detect base64 image data
        base64_pattern = r'[A-Za-z0-9+/]{500,}={0,2}'
        matches = list(re.finditer(base64_pattern, text))
        
        if not matches:
            return
        
        # print_current(f"📸 Found {len(matches)} image data in round {record_index}'s {field_name} field (preserving original data)")
        
        for match_index, match in enumerate(matches):
            base64_data = match.group()
            
            # Calculate image hash
            image_hash = hashlib.md5(base64_data.encode()).hexdigest()[:16]
            
            # Generate image reference ID
            image_ref_id = f"IMG_{image_hash}"
            
            # Check and extract file path markers
            file_info = self._extract_file_path_info(base64_data)
            
            # Clean base64 data (remove file path markers)
            clean_base64_data = self._clean_base64_data(base64_data)
            
            # Analyze image information
            image_info = self._analyze_image_data(clean_base64_data, record_index, match_index)
            
            # Add to image summary (but mark as having preserved original data)
            if image_ref_id not in image_summary:
                image_summary[image_ref_id] = {
                    'hash': image_hash,
                    'size_chars': len(clean_base64_data),
                    'estimated_size_kb': len(clean_base64_data) * 3 // 4 // 1024,
                    'first_seen_record': record_index,
                    'occurrences': 1,
                    'format_info': image_info,
                    'file_info': file_info,  # Add file information
                    'original_data': clean_base64_data[:100] + "..." if len(clean_base64_data) > 100 else clean_base64_data,
                    'has_recent_original': True  # Mark: original data preserved in latest round
                }
            else:
                image_summary[image_ref_id]['occurrences'] += 1
                # Update file information (if new one is more complete)
                if file_info and not image_summary[image_ref_id].get('file_info'):
                    image_summary[image_ref_id]['file_info'] = file_info
                # If this is the latest occurrence, mark as having original data
                if record_index >= image_summary[image_ref_id].get('latest_record', 0):
                    image_summary[image_ref_id]['has_recent_original'] = True
                    image_summary[image_ref_id]['latest_record'] = record_index
    
    def _analyze_image_data(self, base64_data: str, record_index: int, match_index: int) -> Dict[str, Any]:
        """
        Analyze image data, extract useful information
        
        Args:
            base64_data: Base64 encoded image data
            record_index: Record index
            match_index: Match index
            
        Returns:
            Image information dictionary
        """
        try:
            # Try to decode partial data to get format information
            decoded_start = base64.b64decode(base64_data[:100] + "==")
            
            # Detect image format
            if decoded_start.startswith(b'\xff\xd8\xff'):
                format_type = 'JPEG'
            elif decoded_start.startswith(b'\x89PNG'):
                format_type = 'PNG'
            elif decoded_start.startswith(b'GIF8'):
                format_type = 'GIF'
            elif decoded_start.startswith(b'BM'):
                format_type = 'BMP'
            else:
                format_type = 'Unknown'
            
            return {
                'format': format_type,
                'size_chars': len(base64_data),
                'estimated_size_kb': len(base64_data) * 3 // 4 // 1024,
                'record_index': record_index,
                'match_index': match_index
            }
        except Exception as e:
            return {
                'format': 'Unknown',
                'size_chars': len(base64_data),
                'estimated_size_kb': len(base64_data) * 3 // 4 // 1024,
                'record_index': record_index,
                'match_index': match_index,
                'error': str(e)
            }
    
    def _create_image_reference(self, image_ref_id: str, image_info: Dict[str, Any]) -> str:
        """
        Create image reference text
        
        Args:
            image_ref_id: Image reference ID
            image_info: Image information
            
        Returns:
            Image reference text
        """
        format_type = image_info.get('format', 'Unknown')
        size_kb = image_info.get('estimated_size_kb', 0)
        
        reference_text = f"[IMAGE_REF:{image_ref_id}|{format_type}|{size_kb}KB]"
        
        return reference_text
    
    def _create_image_index_record(self, image_summary: Dict[str, Any], keep_recent_images: int) -> Dict[str, Any]:
        """
        Create image index record
        
        Args:
            image_summary: Image summary dictionary
            keep_recent_images: Number of most recent rounds to preserve
            
        Returns:
            Image index record
        """
        index_content = "## 📸 Image Data Index\n\n"
        index_content += f"The following is an index of image data involved in this conversation. Image data from the most recent {keep_recent_images} rounds has been preserved in original format for large model analysis:\n\n"
        
        total_original_size = 0
        total_compressed_size = 0
        total_saved_size = 0
        accessible_files = []  # List of accessible files
        
        for ref_id, info in image_summary.items():
            original_size = info['size_chars']
            compressed_size = len(self._create_image_reference(ref_id, info))
            has_recent = info.get('has_recent_original', False)
            
            # Calculate actual saved size (excluding rounds with preserved original data)
            optimized_occurrences = info['occurrences'] - (1 if has_recent else 0)
            
            total_original_size += original_size * info['occurrences']
            total_compressed_size += (compressed_size * optimized_occurrences + 
                                    (original_size if has_recent else 0))
            total_saved_size += original_size * optimized_occurrences - compressed_size * optimized_occurrences
            
            index_content += f"- **{ref_id}**: {info['format_info']['format']} format, "
            index_content += f"{info['estimated_size_kb']}KB, appears {info['occurrences']} times"
            
            # Check if there are accessible file paths
            file_info = info.get('file_info', {})
            if file_info.get('file_path'):
                file_path = file_info['file_path']
                # Check if file exists
                if os.path.exists(file_path):
                    accessible_files.append({
                        'ref_id': ref_id,
                        'file_path': file_path,
                        'format': info['format_info']['format'],
                        'size_kb': info['estimated_size_kb']
                    })
                    index_content += f" 📁 File accessible: `{file_path}`"
                else:
                    index_content += f" ⚠️ File path: `{file_path}` (does not exist)"
            elif file_info.get('source_path'):
                source_path = file_info['source_path']
                if os.path.exists(source_path):
                    accessible_files.append({
                        'ref_id': ref_id,
                        'file_path': source_path,
                        'format': info['format_info']['format'],
                        'size_kb': info['estimated_size_kb'],
                        'is_source': True
                    })
                    index_content += f" 📁 Source file accessible: `{source_path}`"
                else:
                    index_content += f" ⚠️ Source file path: `{source_path}` (does not exist)"
            
            if has_recent:
                index_content += f" 🔒 Latest round preserves original data\n"
                index_content += f"  Optimization status: {optimized_occurrences} rounds referenced, 1 round preserved\n"
            else:
                index_content += f" 🔧 All referenced\n"
                index_content += f"  Optimization status: {info['occurrences']} rounds all referenced\n"
            
            index_content += f"  Original size: {original_size:,} chars → Reference size: {compressed_size} chars\n"
        
        compression_ratio = (1 - total_compressed_size / total_original_size) * 100 if total_original_size > 0 else 0
        
        index_content += f"\n📊 **Smart Optimization Statistics:**\n"
        index_content += f"- Original total size: {total_original_size:,} characters\n"
        index_content += f"- Optimized size: {total_compressed_size:,} characters\n"
        index_content += f"- Actual savings: {total_saved_size:,} characters\n"
        index_content += f"- Compression ratio: {compression_ratio:.1f}%\n"
        index_content += f"- Estimated token savings: ~{total_saved_size // 4:,} tokens\n\n"
        
        # Add file access guide
        if accessible_files:
            index_content += f"📁 **Re-accessible image files ({len(accessible_files)} files):**\n"
            for file_info in accessible_files:
                file_type = "Source file" if file_info.get('is_source') else "Saved file"
                index_content += f"- `{file_info['ref_id']}`: Can be re-accessed via `read_file(\"{file_info['file_path']}\")` ({file_type})\n"
            index_content += "\n"
        
        index_content += f"**Optimization strategy:** Preserve original image data from the most recent {keep_recent_images} rounds to support large model analysis, use references for historical rounds to save tokens.\n"
        index_content += "**Image reference format:** `[IMAGE_REF:ID|FORMAT|SIZE]` - To re-analyze historical images, use the above file paths or get_sensor_data tool to re-acquire.\n"
        
        return {
            'role': 'system',
            'content': index_content,
            'timestamp': datetime.now().isoformat(),
            'is_image_index': True,
            'optimization_strategy': f"keep_recent_{keep_recent_images}_rounds",
            'accessible_files': accessible_files  # File list for tool use
        }
    
    def calculate_optimization_stats(self, original_history: List[Dict[str, Any]], 
                                   optimized_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate optimization statistics
        
        Args:
            original_history: Original history records
            optimized_history: Optimized history records
            
        Returns:
            Optimization statistics
        """
        from utils.cacheeff import estimate_token_count
        
        # Calculate original and optimized content length
        original_content = ""
        optimized_content = ""
        
        for record in original_history:
            for field in ['prompt', 'result', 'content']:
                if field in record:
                    original_content += str(record[field])
        
        for record in optimized_history:
            for field in ['prompt', 'result', 'content']:
                if field in record:
                    optimized_content += str(record[field])
        
        # Calculate token estimation
        original_tokens = estimate_token_count(original_content, has_images=True)
        optimized_tokens = estimate_token_count(optimized_content, has_images=False)
        
        stats = {
            'original_length': len(original_content),
            'optimized_length': len(optimized_content),
            'compression_ratio': (1 - len(optimized_content) / len(original_content)) * 100 if len(original_content) > 0 else 0,
            'original_tokens_estimated': original_tokens,
            'optimized_tokens_estimated': optimized_tokens,
            'token_savings_estimated': original_tokens - optimized_tokens,
            'token_savings_ratio': (1 - optimized_tokens / original_tokens) * 100 if original_tokens > 0 else 0
        }
        
        return stats
    
