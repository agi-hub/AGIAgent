#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from .print_system import print_system, print_current, print_system_info, print_debug
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
import glob
import re
import fnmatch
import threading
import time
import shutil
import subprocess
from typing import List, Dict, Any, Optional, Tuple, Union


class FileSystemTools:
    def __init__(self, workspace_root: Optional[str] = None):
        """Initialize the FileSystemTools with a workspace root directory."""
        self.workspace_root = workspace_root or os.getcwd()
        self.last_edit = None
        self.snapshot_dir = "file_snapshot"
        self._check_system_grep_available()
    
    def _check_system_grep_available(self):
        """Check if system grep command is available"""
        self.system_grep_available = shutil.which('grep') is not None
        if self.system_grep_available:
            print_system_info("🚀 System grep detected, will use for faster searching")
        else:
            print_system_info("⚠️ System grep not available, using Python fallback")
    
    def _resolve_path(self, path: str) -> str:
        """Resolve a path relative to the workspace root."""
        if os.path.isabs(path):
            return path
        return os.path.join(self.workspace_root, path)
    
    def _create_file_snapshot(self, file_path: str, target_file: str) -> bool:
        """
        Create a snapshot of the existing file before editing.
        
        Args:
            file_path: Absolute path to the file to snapshot
            target_file: Relative path to the file (used for snapshot naming)
            
        Returns:
            True if snapshot was created successfully, False otherwise
        """
        if not os.path.exists(file_path) or not os.path.isfile(file_path):
            return False
        
        try:
            # Create snapshot directory if it doesn't exist
            parent_dir = os.path.dirname(self.workspace_root)
            snapshot_base_dir = os.path.join(parent_dir, self.snapshot_dir)
            os.makedirs(snapshot_base_dir, exist_ok=True)
            
            # Get the directory structure of the original file
            file_dir = os.path.dirname(target_file)
            file_name = os.path.basename(target_file)
            
            # Split filename and extension
            name_parts = file_name.rsplit('.', 1)
            if len(name_parts) == 2:
                base_name, extension = name_parts
                extension = '.' + extension
            else:
                base_name = file_name
                extension = ''
            
            # Create snapshot directory structure
            snapshot_file_dir = os.path.join(snapshot_base_dir, file_dir) if file_dir else snapshot_base_dir
            os.makedirs(snapshot_file_dir, exist_ok=True)
            
            # Find the next available ID
            snapshot_id = 0
            while True:
                snapshot_filename = f"{base_name}_{snapshot_id:03d}{extension}"
                snapshot_path = os.path.join(snapshot_file_dir, snapshot_filename)
                
                if not os.path.exists(snapshot_path):
                    break
                    
                snapshot_id += 1
                
                # Safety check to avoid infinite loop
                if snapshot_id > 999:
                    print_current(f"⚠️ Too many snapshots for file {target_file}, skipping snapshot creation")
                    return False
            
            # Create the snapshot by copying the file
            shutil.copy2(file_path, snapshot_path)
            
            # Make relative path for display
            # 🔧 Modified: Adjust relative path calculation for external file_snapshot
            if os.path.basename(self.workspace_root) == "workspace":
                # If file_snapshot is outside workspace, calculate relative to parent directory
                parent_dir = os.path.dirname(self.workspace_root)
                relative_snapshot_path = os.path.relpath(snapshot_path, parent_dir)
            else:
                # Otherwise, use original logic
                relative_snapshot_path = os.path.relpath(snapshot_path, self.workspace_root)
            print_current(f"📸 Created file snapshot: {relative_snapshot_path}")
            
            return True
            
        except Exception as e:
            print_current(f"⚠️ Failed to create snapshot for {target_file}: {e}")
            return False

    def read_file(self, target_file: str, should_read_entire_file: bool = False, 
                 start_line_one_indexed: Optional[int] = None, end_line_one_indexed_inclusive: Optional[int] = None,
                 **kwargs) -> Dict[str, Any]:
        """
        Read the contents of a file.
        """
        # Ignore additional parameters
        if kwargs:
            print_current(f"⚠️  Ignoring additional parameters: {list(kwargs.keys())}")
        
        print_current(f"🎯 Requested to read file: {target_file}")
        print_current(f"📂 Current working directory: {self.workspace_root}")
        
        file_path = self._resolve_path(target_file)
        
        if not os.path.exists(file_path):
            print_current(f"❌ File does not exist: {file_path}")
            try:
                current_dir_files = os.listdir(self.workspace_root)
                print_current(f"📁 Files in current directory: {current_dir_files}")
            except Exception as e:
                print_current(f"⚠️ Cannot list current directory: {e}")
            
            return {
                'file': target_file,
                'error': f'File not found: {file_path}',
                'workspace_root': self.workspace_root,
                'resolved_path': file_path,
                'exists': False
            }
        
        if not os.path.isfile(file_path):
            print_current(f"❌ Path is not a file: {file_path}")
            return {
                'file': target_file,
                'error': f'Path is not a file: {file_path}',
                'workspace_root': self.workspace_root,
                'resolved_path': file_path,
                'exists': True,
                'is_file': False
            }
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                all_lines = f.readlines()
            
            total_lines = len(all_lines)
            
            if should_read_entire_file:
                max_entire_lines = 500
                if total_lines <= max_entire_lines:
                    content = ''.join(all_lines)
                    return {
                        'file': target_file,
                        'content': content,
                        'total_lines': total_lines,
                        'resolved_path': file_path,
                        'status': 'success'
                    }
                else:
                    # File too large, truncate to max_entire_lines
                    content_lines = all_lines[:max_entire_lines]
                    content = ''.join(content_lines)
                    after_summary = f"... {total_lines - max_entire_lines} lines truncated ..."
                    print_current(f"📄 Read entire file (truncated), showing first {max_entire_lines} lines of {total_lines}")
                    return {
                        'file': target_file,
                        'content': content,
                        'after_summary': after_summary,
                        'total_lines': total_lines,
                        'lines_shown': max_entire_lines,
                        'truncated': True,
                        'resolved_path': file_path,
                        'status': 'success'
                    }
            else:
                if start_line_one_indexed is None:
                    start_line_one_indexed = 1
                if end_line_one_indexed_inclusive is None:
                    end_line_one_indexed_inclusive = min(start_line_one_indexed + 249, total_lines)
                
                start_idx = max(0, start_line_one_indexed - 1)
                end_idx = min(total_lines, end_line_one_indexed_inclusive)
                
                if end_idx - start_idx > 250:
                    end_idx = start_idx + 250
                
                print_current(f"📄 Read partial file: lines {start_line_one_indexed}-{end_line_one_indexed_inclusive} (actual: {start_idx+1}-{end_idx})")
                
                content_lines = all_lines[start_idx:end_idx]
                content = ''.join(content_lines)
                
                before_summary = f"... {start_idx} lines before ..." if start_idx > 0 else ""
                after_summary = f"... {total_lines - end_idx} lines after ..." if end_idx < total_lines else ""
                
                return {
                    'file': target_file,
                    'content': content,
                    'before_summary': before_summary,
                    'after_summary': after_summary,
                    'start_line': start_line_one_indexed,
                    'end_line': end_line_one_indexed_inclusive,
                    'total_lines': total_lines,
                    'resolved_path': file_path,
                    'status': 'success'
                }
        except UnicodeDecodeError as e:
            print_current(f"❌ File encoding error: {e}")
            return {
                'file': target_file,
                'error': f'Unicode decode error: {str(e)}',
                'resolved_path': file_path,
                'status': 'encoding_error'
            }
        except Exception as e:
            print_current(f"❌ Error occurred while reading file: {e}")
            return {
                'file': target_file,
                'error': str(e),
                'resolved_path': file_path,
                'status': 'error'
            }

    def list_dir(self, relative_workspace_path: str = "", **kwargs) -> Dict[str, Any]:
        """
        List the contents of a directory.
        """
        # Ignore additional parameters
        if kwargs:
            print_current(f"⚠️  Ignoring additional parameters: {list(kwargs.keys())}")
        
        dir_path = self._resolve_path(relative_workspace_path)
        
        try:
            entries = os.listdir(dir_path)
            
            files = []
            directories = []
            
            for entry in entries:
                entry_path = os.path.join(dir_path, entry)
                
                if os.path.isdir(entry_path):
                    directories.append(entry)
                else:
                    files.append(entry)
            
            return {
                'path': relative_workspace_path,
                'directories': sorted(directories),
                'files': sorted(files)
            }
        except Exception as e:
            return {
                'path': relative_workspace_path,
                'error': str(e)
            }

    def grep_search(self, query: str, include_pattern: Optional[str] = None, 
                   exclude_pattern: Optional[str] = None, case_sensitive: bool = False,
                   max_results: Optional[int] = None, **kwargs) -> Dict[str, Any]:
        """
        Run a text-based regex search over files with intelligent query optimization.
        
        Args:
            query: Search pattern/regex
            include_pattern: File pattern to include (e.g., '*.py')
            exclude_pattern: File pattern to exclude (e.g., 'output_*/*')
            case_sensitive: Whether search should be case sensitive
            max_results: Maximum number of results to return (None = unlimited)
        """
        # Ignore additional parameters
        if kwargs:
            print_current(f"⚠️  Ignoring additional parameters: {list(kwargs.keys())}")
        
        # Intelligent query optimization for LLM-generated complex queries
        optimized_result, should_split = self._optimize_query_for_performance(query)
        
        if should_split:
            print_current(f"🔧 Complex query detected, optimizing for better performance...")
            # Type assertion: we know optimized_result is a list when should_split is True
            query_groups = optimized_result if isinstance(optimized_result, list) else [str(optimized_result)]
            return self._execute_split_search(query_groups, include_pattern or "", exclude_pattern or "", case_sensitive, max_results)
        else:
            print_current(f"Searching for: {optimized_result}")
            # Type assertion: we know optimized_result is a string when should_split is False
            query_str = str(optimized_result) if not isinstance(optimized_result, list) else optimized_result[0]
            return self._execute_single_search(query_str, include_pattern or "", exclude_pattern or "", case_sensitive, max_results)

    def edit_file(self, target_file: str, edit_mode: str, code_edit: str, instructions: Optional[str] = None, 
                  **kwargs) -> Dict[str, Any]:
        """
        使用三种模式编辑文件或创建新文件。
        
        Args:
            target_file: 文件路径
            edit_mode: 编辑模式 - "lines_replace", "append", "full_replace"
            code_edit: 要编辑的代码/文本内容
            instructions: 可选的编辑说明
            
        Returns:
            包含编辑结果的字典
        
        Edit Modes:
            - lines_replace: 智能替换模式，使用existing code标记进行精确合并
            - append: 追加到文件末尾
            - full_replace: 完全替换文件内容
        """
        # Check for dummy placeholder file created by hallucination detection
        if target_file == "dummy_file_placeholder.txt" or target_file.endswith("/dummy_file_placeholder.txt"):
            print_current(f"🚨 HALLUCINATION PREVENTION: Detected dummy placeholder file '{target_file}' - skipping actual file operation")
            return {
                'file': target_file,
                'status': 'skipped',
                'error': 'Dummy placeholder file detected - hallucination prevention active',
                'hallucination_prevention': True
            }
        
        # Default to append mode if auto mode is set for safety
        if edit_mode == "auto":
            edit_mode = "append"
        
        # 兼容旧的edit_mode名称
        if edit_mode == "auto":
            edit_mode = "lines_replace"
        elif edit_mode in ["replace_lines", "insert_lines"]:
            edit_mode = "lines_replace"
        
        # Ignore additional parameters
        if kwargs:
            print_current(f"⚠️  Ignoring additional parameters: {list(kwargs.keys())}")
        
        file_path = self._resolve_path(target_file)
        file_exists = os.path.exists(file_path)
        
        # Clean markdown code block markers from code_edit
        cleaned_code_edit = self._clean_markdown_markers(code_edit)
        
        # Auto-correct HTML entities in code_edit
        cleaned_code_edit = self._fix_html_entities(cleaned_code_edit)
        
        # Process markdown content (convert \n markers and ensure newline at end)
        cleaned_code_edit = self._process_markdown_content(cleaned_code_edit, target_file)
        
        self.last_edit = {
            'target_file': target_file,
            'instructions': instructions,
            'code_edit': cleaned_code_edit,
            'edit_mode': edit_mode
        }
        
        try:
            # Create directory if needed
            dir_path = os.path.dirname(file_path)
            if dir_path:
                os.makedirs(dir_path, exist_ok=True)
            
            # Read original content if file exists
            original_content = ""
            if file_exists:
                with open(file_path, 'r', encoding='utf-8') as f:
                    original_content = f.read()
                
                # Create snapshot before editing (only for existing files)
                snapshot_created = self._create_file_snapshot(file_path, target_file)
                
                # Safety check for unintentional content loss
                if self._is_risky_edit(original_content, cleaned_code_edit, edit_mode, file_exists):
                    return {
                        'file': target_file,
                        'status': 'safety_blocked',
                        'error': 'Edit blocked: Risk of content loss detected. Use specific edit_mode or read file first.',
                        'safety_suggestion': 'Consider using edit_mode="lines_replace", "append", or "full_replace"',
                        'original_length': len(original_content),
                        'edit_mode': edit_mode,
                        'snapshot_created': snapshot_created
                    }
            
            # Process edit based on mode
            new_content = self._process_edit_by_mode(
                original_content, cleaned_code_edit, edit_mode, target_file
            )
            
            # Write the new content
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(new_content)
            
            # Determine action and status
            if not file_exists:
                action = 'created'
                status = 'created'
            elif edit_mode == "append":
                action = 'appended'
                status = 'appended'
            else:
                action = 'modified'
                status = 'edited'
            
            return {
                'file': target_file,
                'status': status,
                'action': action,
                'edit_mode': edit_mode,
                'snapshot_created': file_exists  # Only create snapshot for existing files
            }
            
        except Exception as e: 
            return {
                'file': target_file,
                'status': 'error',
                'error': str(e),
                'edit_mode': edit_mode
            }

    def _clean_markdown_markers(self, code_content: str) -> str:
        """
        Clean markdown code block markers from code content.
        
        Args:
            code_content: Raw code content that might contain markdown markers
            
        Returns:
            Cleaned code content without markdown markers
        """
        import re
        
        # Store original content for comparison
        original_content = code_content
        
        # Check if content has markdown code block markers
        has_start_marker = False
        has_end_marker = False
        
        # Check for start marker (```[language]) at the beginning
        start_marker_pattern = r'^```[^\n]*\n'
        if re.match(start_marker_pattern, code_content):
            has_start_marker = True
            print_current(f"🧹 Found markdown code block start marker")
        
        # Check for end marker (```) at the end
        end_marker_pattern = r'\n```\s*$'
        if re.search(end_marker_pattern, code_content):
            has_end_marker = True
            print_current(f"🧹 Found markdown code block end marker")
        
        # If no markers found, return original content to preserve exact formatting
        if not has_start_marker and not has_end_marker:
            return original_content
        
        # Remove markers using regex to preserve internal formatting
        cleaned_content = original_content
        
        if has_start_marker:
            # Remove start marker (```[language]\n)
            cleaned_content = re.sub(start_marker_pattern, '', cleaned_content)
            print_current(f"🧹 Removed markdown code block start marker")
        
        if has_end_marker:
            # Remove end marker (\n```)
            cleaned_content = re.sub(end_marker_pattern, '', cleaned_content)
            print_current(f"🧹 Removed markdown code block end marker")
        
        print_current(f"✅ Cleaned markdown markers while preserving formatting")
        return cleaned_content

    def _fix_html_entities(self, code_content: str) -> str:
        """
        Auto-correct HTML entities in code content using Python's html.unescape().
        This handles all standard HTML entities, not just < and >.
        
        Args:
            code_content: Code content that might contain HTML entities
            
        Returns:
            Code content with HTML entities corrected
        """
        import html
        
        original_content = code_content
        
        # Use Python's built-in html.unescape() for comprehensive entity decoding
        code_content = html.unescape(code_content)
        
        # Log if any changes were made
        if original_content != code_content:
            # Count common entities for logging
            common_entities = {
                '&lt;': '<',
                '&gt;': '>',
                '&amp;': '&',
                '&quot;': '"',
                '&#x27;': "'",
                '&#39;': "'"
            }
            
            corrections = []
            for entity, char in common_entities.items():
                count = original_content.count(entity)
                if count > 0:
                    corrections.append(f"{entity} → {char} ({count} times)")
            
            # If there are other entities not in our common list, mention them generically
            if corrections:
                print_current(f"🔧 Auto-corrected HTML entities: {', '.join(corrections)}")
            else:
                print_current(f"🔧 Auto-corrected HTML entities (various types found)")
        
        return code_content

    def _process_edit_by_mode(self, original_content: str, code_edit: str, edit_mode: str, 
                             target_file: str) -> str:
        """
        Process edit based on the specified mode.

        Args:
            original_content: The original file content
            code_edit: The new content to add/replace
            edit_mode: Edit mode - "lines_replace", "append", "full_replace"
            target_file: Target file path

        Returns:
            The new file content after applying the edit
        """
        if edit_mode == "lines_replace":
            # Smart replacement mode using existing code markers
            return self._process_code_edit(original_content, code_edit, target_file)
        elif edit_mode == "append":
            # Append to the end of the file
            return self._append_content(original_content, code_edit)
        elif edit_mode == "full_replace":
            # Completely replace the file content
            return code_edit
        else:
            # Compatibility mode: default to lines_replace
            print_current(f"⚠️ Unknown edit_mode '{edit_mode}', defaulting to lines_replace")
            return self._process_code_edit(original_content, code_edit, target_file)

    def _replace_lines(self, content: str, new_content: str, start_line_one_indexed: int, end_line_one_indexed_inclusive: int) -> str:
        """
        Replace specified line range with new content.
        
        Args:
            content: Original file content
            new_content: Content to replace with
            start_line_one_indexed: Starting line number (1-indexed, inclusive)
            end_line_one_indexed_inclusive: Ending line number (1-indexed, inclusive)
        
        Returns:
            Updated content with lines replaced
        """
        if start_line_one_indexed is None or end_line_one_indexed_inclusive is None:
            raise ValueError("start_line_one_indexed and end_line_one_indexed_inclusive must be specified for replace_lines mode")
        
        lines = content.split('\n')
        new_lines = new_content.split('\n')
        
        # Convert to 0-indexed
        start_idx = start_line_one_indexed - 1
        end_idx = end_line_one_indexed_inclusive - 1
        
        # Validate line range
        if start_idx < 0:
            raise ValueError(f"start_line_one_indexed must be >= 1, got {start_line_one_indexed}")
        if end_idx >= len(lines):
            raise ValueError(f"end_line_one_indexed_inclusive {end_line_one_indexed_inclusive} exceeds file length ({len(lines)} lines)")
        if start_idx > end_idx:
            raise ValueError(f"start_line_one_indexed ({start_line_one_indexed}) must be <= end_line_one_indexed_inclusive ({end_line_one_indexed_inclusive})")
        
        print_current(f"🔧 Replacing lines {start_line_one_indexed}-{end_line_one_indexed_inclusive} ({end_idx - start_idx + 1} lines) with {len(new_lines)} new lines")
        
        # Replace the specified lines
        result = lines[:start_idx] + new_lines + lines[end_idx + 1:]
        return '\n'.join(result)

    def _insert_lines(self, content: str, new_content: str, insert_position: int) -> str:
        """
        Insert new content at the specified line position.
        
        Args:
            content: Original file content
            new_content: Content to insert
            insert_position: Line number to insert before (1-indexed)
        
        Returns:
            Updated content with lines inserted
        """
        if insert_position is None:
            raise ValueError("insert_position must be specified for insert_lines mode")
        
        lines = content.split('\n')
        new_lines = new_content.split('\n')
        
        # Convert to 0-indexed
        insert_idx = insert_position - 1
        
        # Validate insert position
        if insert_idx < 0:
            raise ValueError(f"insert_position must be >= 1, got {insert_position}")
        if insert_idx > len(lines):
            raise ValueError(f"insert_position {insert_position} exceeds file length + 1 ({len(lines) + 1})")
        
        print_current(f"📍 Inserting {len(new_lines)} lines at position {insert_position}")
        
        # Insert the new lines
        result = lines[:insert_idx] + new_lines + lines[insert_idx:]
        return '\n'.join(result)

    def _append_content(self, content: str, new_content: str) -> str:
        """
        Append content to the end of the file.
        
        Args:
            content: Original file content
            new_content: Content to append
        
        Returns:
            Updated content with content appended
        """
        # Remove any existing code markers from the append content
        clean_content = new_content.replace('// ... existing code ...', '').strip()
        clean_content = clean_content.replace('# ... existing code ...', '').strip()
        clean_content = clean_content.replace('/* ... existing code ...', '').strip()
        clean_content = clean_content.replace('<!-- ... existing code ...', '').strip()
        
        if not content:
            print_current("📝 Creating new file with append content")
            return clean_content
        
        # Ensure there's a newline before appending if the file doesn't end with one
        if content and not content.endswith('\n'):
            result = content + '\n' + clean_content
        else:
            result = content + clean_content
        
        print_current(f"➕ Appending {len(clean_content.split(chr(10)))} lines to end of file")
        return result

    def _process_code_edit(self, original_content: str, code_edit: str, target_file: str) -> str:
        """
        Process code edits using context anchor matching algorithm for precise positioning.
        """
        comment_markers = self._get_comment_markers(target_file)
        
        # Standard existing code markers for auto mode
        existing_code_patterns = [
            '// ... existing code ...',
            '# ... existing code ...',
            '/* ... existing code ...',
            '<!-- ... existing code -->',
        ]
        
        used_pattern = None
        for pattern in existing_code_patterns:
            if pattern in code_edit:
                used_pattern = pattern
                break
        
        if not used_pattern:
            # Reject edits without existing code markers to prevent accidental full file replacement
            raise ValueError(
                "EDIT REJECTED: No existing code markers found in lines_replace mode. "
                "To prevent accidental file replacement, you must use one of these markers: "
                f"{', '.join(existing_code_patterns)}. "
                "If you want to replace the entire file, use edit_mode='full_replace' instead. "
                "If you want to add content to the end, use edit_mode='append'."
            )
        
        edit_parts = code_edit.split(used_pattern)
        
        if len(edit_parts) == 1:
            return code_edit
        
        print_current(f"🔧 Found existing code marker: {used_pattern}")
        print_current(f"📝 Edit split into {len(edit_parts)} parts")
        
        # Use improved context anchor matching algorithm
        return self._apply_context_based_edit(original_content, edit_parts, used_pattern)
    
    def _apply_context_based_edit(self, original_content: str, edit_parts: List[str], marker: str) -> str:
        """
        Smart code merging algorithm based on context anchors.
        
        Args:
            original_content: Original file content
            edit_parts: Edit parts split by existing code marker
            marker: The existing code marker used
        
        Returns:
            Merged file content
        """
        if len(edit_parts) < 2:
            return original_content
            
        original_lines = original_content.split('\n')
        
        # 调试打印：显示编辑部分的内容
        print_current(f"🔍 DEBUG: Edit parts count: {len(edit_parts)}")
        for i, part in enumerate(edit_parts):
            part_preview = part.strip()[:100].replace('\n', '\\n') if part.strip() else "(empty)"
            print_current(f"🔍 DEBUG: Edit part {i}: {part_preview}...")
        
        if len(edit_parts) == 2:
            # Simple case: one existing code marker
            return self._apply_single_marker_edit(original_lines, edit_parts[0], edit_parts[1])
        else:
            # Complex case: multiple existing code markers
            return self._apply_multiple_marker_edit(original_lines, edit_parts, marker)
    
    def _apply_single_marker_edit(self, original_lines: List[str], before_part: str, after_part: str) -> str:
        """
        Handle the case of a single existing code marker, using context anchors for precise positioning.
        """
        before_lines = [line.rstrip() for line in before_part.split('\n') if line.strip()]
        after_lines = [line.rstrip() for line in after_part.split('\n') if line.strip()]
        
        # 调试打印：显示before和after的内容
        print_current(f"🔍 DEBUG: Before lines count: {len(before_lines)}")
        if before_lines:
            print_current(f"🔍 DEBUG: Before lines preview: {before_lines[:3]}...")
        print_current(f"🔍 DEBUG: After lines count: {len(after_lines)}")
        if after_lines:
            print_current(f"🔍 DEBUG: After lines preview: {after_lines[:3]}...")
        
        # Case 1: No content at all
        if not before_lines and not after_lines:
            print_current("🔍 DEBUG: Case 1 - No content, returning original")
            return '\n'.join(original_lines)
        
        # Case 2: Only before content - try context matching first, fallback to beginning
        if before_lines and not after_lines:
            print_current("🔍 DEBUG: Case 2 - Only before content, trying context matching")
            # Try to find context for before_lines first
            best_match = self._find_best_context_match(original_lines, before_lines, [])
            if best_match is not None:
                insert_pos, match_type, anchor_size = best_match
                print_current(f"🎯 Found context match for before_lines at line {insert_pos}")
                if match_type == "before":
                    new_content_lines = before_lines[:-anchor_size] if len(before_lines) > anchor_size else []
                    result_lines = (original_lines[:insert_pos + anchor_size] + 
                                  new_content_lines + original_lines[insert_pos + anchor_size:])
                    return '\n'.join(result_lines)
            
            print_current("🔍 DEBUG: No context match found, inserting at beginning")
            return self._insert_at_beginning(original_lines, before_lines)
        
        # Case 3: Only after content - ALWAYS try context matching first
        if after_lines and not before_lines:
            print_current("🔍 DEBUG: Case 3 - Only after content, trying context matching")
            # Try to find context for after_lines first
            best_match = self._find_best_context_match(original_lines, [], after_lines)
            if best_match is not None:
                insert_pos, match_type, anchor_size = best_match
                print_current(f"🎯 Found context match for after_lines at line {insert_pos}")
                if match_type == "after":
                    # Replace the anchor content with new content
                    replace_size = max(anchor_size, len(after_lines) if after_lines else 0)
                    print_current(f"🔄 Case 3: Replacing {replace_size} lines starting from line {insert_pos + 1}")
                    print_current(f"🔍 DEBUG: anchor_size={anchor_size}, after_lines={len(after_lines)}, replace_size={replace_size}")
                    result_lines = (original_lines[:insert_pos] + after_lines + original_lines[insert_pos + replace_size:])
                    return '\n'.join(result_lines)
            
            print_current("🔍 DEBUG: No context match found, inserting at end")
            return self._insert_at_end(original_lines, after_lines)
        
        # Case 4: Both before and after content, need to find insertion position
        if before_lines and after_lines:
            print_current("🔍 DEBUG: Case 4 - Both before and after content, using context anchors")
            return self._insert_with_context_anchors(original_lines, before_lines, after_lines)
        
        # Fallback
        print_current("🔍 DEBUG: Fallback - returning original")
        return '\n'.join(original_lines)
    
    def _insert_with_context_anchors(self, original_lines: List[str], before_lines: List[str], after_lines: List[str]) -> str:
        """
        Use context anchors to find the precise insertion position with improved matching.
        """
        print_current(f"🔍 DEBUG: Original file has {len(original_lines)} lines")
        
        # Show what we're looking for
        if before_lines:
            print_current(f"🔍 DEBUG: Looking for BEFORE content ending with: {before_lines[-3:] if len(before_lines) >= 3 else before_lines}")
        if after_lines:
            print_current(f"🔍 DEBUG: Looking for AFTER content starting with: {after_lines[:3] if len(after_lines) >= 3 else after_lines}")
        
        # Strategy 1: Try to find the best match for complete context sequence
        best_match = self._find_best_context_match(original_lines, before_lines, after_lines)
        
        if best_match is not None:
            insert_pos, match_type, anchor_size = best_match
            print_current(f"🎯 Found best context match at line {insert_pos} (type: {match_type}, anchor_size: {anchor_size})")
            
            if match_type == "before":
                # Replace the anchor content with new content
                print_current(f"🔄 Replacing {anchor_size} lines starting from line {insert_pos + 1}")
                # For "before" type, we want to replace the matched content plus find the appropriate range
                # Calculate how many lines we should replace based on before_lines and after_lines
                replace_size = max(anchor_size, len(before_lines) if before_lines else 0)
                result_lines = (original_lines[:insert_pos] + 
                              before_lines + after_lines + 
                              original_lines[insert_pos + replace_size:])
                return '\n'.join(result_lines)
            elif match_type == "after":
                # Replace the anchor content with new content
                print_current(f"🔄 Replacing {anchor_size} lines starting from line {insert_pos + 1}")
                # For "after" type, we want to replace a range that covers the content we're updating
                # The range should cover at least the anchor_size, but potentially more based on after_lines
                replace_size = max(anchor_size, len(after_lines) if after_lines else 0)
                print_current(f"🔍 DEBUG: Will replace {replace_size} lines (anchor_size={anchor_size}, after_lines={len(after_lines)})")
                print_current(f"🔍 DEBUG: Original file range to replace: lines {insert_pos + 1} to {insert_pos + replace_size}")
                print_current(f"🔍 DEBUG: Keeping lines 0 to {insert_pos}, then inserting {len(before_lines) + len(after_lines)} new lines, then keeping from line {insert_pos + replace_size + 1}")
                result_lines = (original_lines[:insert_pos] + before_lines + 
                              after_lines + original_lines[insert_pos + replace_size:])
                print_current(f"🔍 DEBUG: Result will have {len(result_lines)} lines (original had {len(original_lines)})")
                return '\n'.join(result_lines)
        
        # Strategy 2: Try traditional anchor matching with larger anchor sizes for unique matches
        print_current("🔍 DEBUG: Trying traditional anchor matching...")
        for anchor_size in [5, 4, 3, 2, 1]:  # Try larger anchors first
            # Try before context
            if anchor_size <= len(before_lines):
                before_anchor = before_lines[-anchor_size:]
                matches = self._find_all_anchor_positions(original_lines, before_anchor)
                
                print_current(f"🔍 DEBUG: Traditional matching - anchor_size {anchor_size}, before anchor: {before_anchor}, matches: {matches}")
                
                if len(matches) == 1:  # Unique match found
                    insert_pos = matches[0]
                    print_current(f"🎯 Found unique anchor position using before context at line {insert_pos + anchor_size} (anchor_size: {anchor_size})")
                    new_content_lines = before_lines[:-anchor_size] if len(before_lines) > anchor_size else []
                    result_lines = (original_lines[:insert_pos + anchor_size] + 
                                  new_content_lines + after_lines + 
                                  original_lines[insert_pos + anchor_size:])
                    return '\n'.join(result_lines)
            
            # Try after context
            if anchor_size <= len(after_lines):
                after_anchor = after_lines[:anchor_size]
                matches = self._find_all_anchor_positions(original_lines, after_anchor)
                
                print_current(f"🔍 DEBUG: Traditional matching - anchor_size {anchor_size}, after anchor: {after_anchor}, matches: {matches}")
                
                if len(matches) == 1:  # Unique match found
                    insert_pos = matches[0]
                    print_current(f"🎯 Found unique anchor position using after context at line {insert_pos} (anchor_size: {anchor_size})")
                    new_content_lines = after_lines[anchor_size:] if len(after_lines) > anchor_size else []
                    result_lines = (original_lines[:insert_pos] + before_lines + 
                                  new_content_lines + original_lines[insert_pos:])
                    return '\n'.join(result_lines)
        
        # Strategy 3: Fallback to safe append mode
        print_current("⚠️ No suitable anchor found, appending to end")
        return self._safe_append_edit(original_lines, before_lines, after_lines)
    
    def _find_anchor_position(self, original_lines: List[str], anchor_lines: List[str], search_from_start: bool = True) -> Optional[int]:
        """
        Find the anchor position in the original file.
        
        Args:
            original_lines: Original file lines
            anchor_lines: Anchor lines
            search_from_start: Whether to search from the start
        
        Returns:
            Line number of the anchor position, or None if not found
        """
        if not anchor_lines:
            return None
        
        anchor_text = [line.strip() for line in anchor_lines if line.strip()]
        if not anchor_text:
            return None
        
        matches = []
        search_range = range(len(original_lines) - len(anchor_text) + 1)
        
        for i in search_range:
            match = True
            for j, anchor_line in enumerate(anchor_text):
                original_line = original_lines[i + j].strip()
                # Use fuzzy matching, ignore whitespace differences
                if not self._lines_match(original_line, anchor_line):
                    match = False
                    break
            
            if match:
                matches.append(i)
        
        # Check for uniqueness
        if len(matches) == 1:
            return matches[0]
        elif len(matches) > 1:
            print_current(f"⚠️ Multiple anchor matches found: {matches}, using {'first' if search_from_start else 'last'}")
            return matches[0] if search_from_start else matches[-1]
        
        return None

    def _find_all_anchor_positions(self, original_lines: List[str], anchor_lines: List[str]) -> List[int]:
        """
        Find all possible anchor positions in the original file.
        
        Args:
            original_lines: Original file lines
            anchor_lines: Anchor lines
        
        Returns:
            List of line numbers where the anchor matches
        """
        if not anchor_lines:
            return []
        
        # Clean anchor lines for comparison
        anchor_text = [line.strip() for line in anchor_lines if line.strip()]
        if not anchor_text:
            return []
        
        matches = []
        
        for i in range(len(original_lines) - len(anchor_text) + 1):
            match = True
            for j, anchor_line in enumerate(anchor_text):
                original_line = original_lines[i + j].strip()
                # Use fuzzy matching, ignore whitespace differences
                if not self._lines_match(original_line, anchor_line):
                    match = False
                    break
            
            if match:
                matches.append(i)
        
        return matches

    def _find_best_context_match(self, original_lines: List[str], before_lines: List[str], after_lines: List[str]) -> Optional[Tuple[int, str, int]]:
        """
        Find the best context match by analyzing both before and after context together.
        
        Args:
            original_lines: Original file lines
            before_lines: Lines that should appear before the insertion point
            after_lines: Lines that should appear after the insertion point
        
        Returns:
            Tuple of (insert_position, match_type, anchor_size) or None if no good match found
        """
        best_score = 0
        best_match = None
        
        print_current("🔍 DEBUG: Starting best context match search...")
        
        # Try different anchor sizes for comprehensive matching
        for anchor_size in [5, 4, 3, 2, 1]:
            # Try before context anchors
            if anchor_size <= len(before_lines):
                before_anchor = before_lines[-anchor_size:]
                before_matches = self._find_all_anchor_positions(original_lines, before_anchor)
                
                print_current(f"🔍 DEBUG: Anchor size {anchor_size}, before anchor: {before_anchor}, found {len(before_matches)} matches at positions: {before_matches}")
                
                for match_pos in before_matches:
                    # Calculate context score by checking surrounding lines
                    score = self._calculate_context_score(original_lines, match_pos, before_lines, after_lines, anchor_size, "before")
                    
                    # Show context around the match position
                    context_start = max(0, match_pos - 2)
                    context_end = min(len(original_lines), match_pos + anchor_size + 2)
                    context_lines = original_lines[context_start:context_end]
                    context_preview = [f"L{context_start + i + 1}: {line}" for i, line in enumerate(context_lines)]
                    
                    print_current(f"🔍 DEBUG: Before match at line {match_pos + 1}, score: {score}")
                    print_current(f"🔍 DEBUG: Context around match:\n" + "\n".join(context_preview))
                    
                    if score > best_score:
                        best_score = score
                        best_match = (match_pos, "before", anchor_size)
            
            # Try after context anchors
            if anchor_size <= len(after_lines):
                after_anchor = after_lines[:anchor_size]
                after_matches = self._find_all_anchor_positions(original_lines, after_anchor)
                
                print_current(f"🔍 DEBUG: Anchor size {anchor_size}, after anchor: {after_anchor}, found {len(after_matches)} matches at positions: {after_matches}")
                
                for match_pos in after_matches:
                    # Calculate context score by checking surrounding lines
                    score = self._calculate_context_score(original_lines, match_pos, before_lines, after_lines, anchor_size, "after")
                    
                    # Show context around the match position
                    context_start = max(0, match_pos - 2)
                    context_end = min(len(original_lines), match_pos + anchor_size + 2)
                    context_lines = original_lines[context_start:context_end]
                    context_preview = [f"L{context_start + i + 1}: {line}" for i, line in enumerate(context_lines)]
                    
                    print_current(f"🔍 DEBUG: After match at line {match_pos + 1}, score: {score}")
                    print_current(f"🔍 DEBUG: Context around match:\n" + "\n".join(context_preview))
                    
                    if score > best_score:
                        best_score = score
                        best_match = (match_pos, "after", anchor_size)
        
        # Only return match if score is high enough (at least 2 matching context lines)
        if best_score >= 2 and best_match is not None:
            print_current(f"🔍 DEBUG: Best match found: position {best_match[0]}, type {best_match[1]}, anchor_size {best_match[2]}, score {best_score}")
            return best_match
        else:
            print_current(f"🔍 DEBUG: No good match found, best score was {best_score} (minimum required: 2)")
            return None

    def _calculate_context_score(self, original_lines: List[str], match_pos: int, before_lines: List[str], after_lines: List[str], anchor_size: int, match_type: str) -> int:
        """
        Calculate a context score for how well the surrounding lines match.
        
        Args:
            original_lines: Original file lines
            match_pos: Position of the potential match
            before_lines: Expected before context
            after_lines: Expected after context
            anchor_size: Size of the anchor being tested
            match_type: Type of match ("before" or "after")
        
        Returns:
            Score indicating quality of context match (higher is better)
        """
        score = anchor_size  # Base score from anchor match
        
        if match_type == "before":
            # Check additional context before the anchor
            additional_before = before_lines[:-anchor_size] if len(before_lines) > anchor_size else []
            check_pos = match_pos - len(additional_before)
            
            if check_pos >= 0:
                for i, line in enumerate(additional_before):
                    if check_pos + i < len(original_lines):
                        if self._lines_match(original_lines[check_pos + i].strip(), line.strip()):
                            score += 1
            
            # Check after context at the position where new content would be inserted
            check_pos = match_pos + anchor_size
            for i, line in enumerate(after_lines[:3]):  # Check first 3 lines of after context
                if check_pos + i < len(original_lines):
                    if self._lines_match(original_lines[check_pos + i].strip(), line.strip()):
                        score += 1
        
        elif match_type == "after":
            # Check before context at the position where new content would be inserted
            check_pos = match_pos - len(before_lines[:3])  # Check last 3 lines of before context
            
            if check_pos >= 0:
                for i, line in enumerate(before_lines[-3:]):  # Check last 3 lines
                    if check_pos + i < len(original_lines):
                        if self._lines_match(original_lines[check_pos + i].strip(), line.strip()):
                            score += 1
            
            # Check additional context after the anchor
            additional_after = after_lines[anchor_size:] if len(after_lines) > anchor_size else []
            check_pos = match_pos + anchor_size
            
            for i, line in enumerate(additional_after):
                if check_pos + i < len(original_lines):
                    if self._lines_match(original_lines[check_pos + i].strip(), line.strip()):
                        score += 1
        
        return score
    
    def _lines_match(self, line1: str, line2: str) -> bool:
        """
        Check if two lines match (ignoring whitespace differences).
        """
        # Remove extra whitespace and compare
        clean1 = ' '.join(line1.split())
        clean2 = ' '.join(line2.split())
        return clean1 == clean2
    
    def _insert_at_beginning(self, original_lines: List[str], new_lines: List[str]) -> str:
        """Insert content at the beginning of the file."""
        print_current(f"📍 Inserting {len(new_lines)} lines at beginning")
        return '\n'.join(new_lines + original_lines)
    
    def _insert_at_end(self, original_lines: List[str], new_lines: List[str]) -> str:
        """Insert content at the end of the file."""
        print_current(f"📍 Inserting {len(new_lines)} lines at end")
        return '\n'.join(original_lines + new_lines)
    
    def _safe_append_edit(self, original_lines: List[str], before_lines: List[str], after_lines: List[str]) -> str:
        """Safe append edit mode."""
        result_lines = original_lines[:]
        
        if before_lines:
            # Insert before content at a suitable position
            insert_pos = len(result_lines) // 2  # Middle position
            result_lines = result_lines[:insert_pos] + before_lines + result_lines[insert_pos:]
        
        if after_lines:
            # Append after content
            result_lines.extend(after_lines)
        
        return '\n'.join(result_lines)
    
    def _apply_multiple_marker_edit(self, original_lines: List[str], edit_parts: List[str], marker: str) -> str:
        """
        Handle complex cases with multiple existing code markers using anchor detection.
        """
        print_current(f"🔧 Handling complex edit with {len(edit_parts)} parts")
        
        # 调试打印：显示编辑部分
        for i, part in enumerate(edit_parts):
            part_preview = part.strip()[:100].replace('\n', '\\n') if part.strip() else "(empty)"
            print_current(f"🔍 DEBUG: Multiple marker edit part {i}: {part_preview}...")
        
        # For multiple markers, we need to process them sequentially
        # Start with the original content
        current_content = '\n'.join(original_lines)
        
        # Process pairs of edit parts (before_marker, after_marker)
        for i in range(0, len(edit_parts) - 1, 2):
            if i + 1 < len(edit_parts):
                before_part = edit_parts[i]
                after_part = edit_parts[i + 1]
                
                print_current(f"🔍 DEBUG: Processing marker pair {i//2 + 1}")
                print_current(f"🔍 DEBUG: Before part: {before_part.strip()[:100] if before_part.strip() else '(empty)'}...")
                print_current(f"🔍 DEBUG: After part: {after_part.strip()[:100] if after_part.strip() else '(empty)'}...")
                
                # Apply single marker edit logic for each pair
                current_lines = current_content.split('\n')
                current_content = self._apply_single_marker_edit(current_lines, before_part, after_part)
        
        return current_content

    def _get_comment_markers(self, target_file: str) -> Dict[str, str]:
        """Get appropriate comment markers for the file type."""
        ext = os.path.splitext(target_file)[1].lower()
        
        comment_map = {
            '.py': {'line': '#', 'block_start': '"""', 'block_end': '"""'},
            '.js': {'line': '//', 'block_start': '/*', 'block_end': '*/'},
            '.ts': {'line': '//', 'block_start': '/*', 'block_end': '*/'},
            '.jsx': {'line': '//', 'block_start': '/*', 'block_end': '*/'},
            '.tsx': {'line': '//', 'block_start': '/*', 'block_end': '*/'},
            '.java': {'line': '//', 'block_start': '/*', 'block_end': '*/'},
            '.c': {'line': '//', 'block_start': '/*', 'block_end': '*/'},
            '.cpp': {'line': '//', 'block_start': '/*', 'block_end': '*/'},
            '.html': {'line': '', 'block_start': '<!--', 'block_end': '-->'},
            '.xml': {'line': '', 'block_start': '<!--', 'block_end': '-->'},
            '.css': {'line': '', 'block_start': '/*', 'block_end': '*/'},
            '.md': {'line': '', 'block_start': '<!--', 'block_end': '-->'},
            '.sh': {'line': '#', 'block_start': '', 'block_end': ''},
            '.yaml': {'line': '#', 'block_start': '', 'block_end': ''},
            '.yml': {'line': '#', 'block_start': '', 'block_end': ''},
        }
        
        return comment_map.get(ext, {'line': '//', 'block_start': '/*', 'block_end': '*/'})


    def file_search(self, query: str, **kwargs) -> Dict[str, Any]:
        """
        Fast file search based on fuzzy matching against file path.
        """
        # Ignore additional parameters
        if kwargs:
            print_current(f"⚠️  Ignoring additional parameters: {list(kwargs.keys())}")
        
        print_current(f"Searching for file: {query}")
        
        results = []
        max_results = 10
        
        # Enable followlinks=True to traverse symbolic links
        for root, _, files in os.walk(self.workspace_root, followlinks=True):
            for file in files:
                file_path = os.path.join(root, file)
                rel_path = os.path.relpath(file_path, self.workspace_root)
                
                if query.lower() in rel_path.lower():
                    results.append(rel_path)
                    
                    if len(results) >= max_results:
                        break
            
            if len(results) >= max_results:
                break
        
        return {
            'query': query,
            'results': results,
            'total_matches': len(results),
            'max_results': max_results,
            'truncated': len(results) >= max_results
        }

    def delete_file(self, target_file: str, **kwargs) -> Dict[str, Any]:
        """
        Delete a file at the specified path.
        """
        # Ignore additional parameters
        if kwargs:
            print_current(f"⚠️  Ignoring additional parameters: {list(kwargs.keys())}")
        
        file_path = self._resolve_path(target_file)
        
        if not os.path.exists(file_path):
            return {
                'file': target_file,
                'status': 'error',
                'error': 'File does not exist'
            }
        
        try:
            if os.path.isdir(file_path):
                return {
                    'file': target_file,
                    'status': 'error',
                    'error': 'Target is a directory, not a file'
                }
            
            os.remove(file_path)
            
            return {
                'file': target_file,
                'status': 'deleted'
            }
        except Exception as e:
            return {
                'file': target_file,
                'status': 'error',
                'error': str(e)
            }

    def _is_risky_edit(self, original_content: str, new_content: str, edit_mode: str, file_exists: bool) -> bool:
        """
        Check if the edit operation risks accidentally deleting file content.
        
        Args:
            original_content: The original file content
            new_content: The new content to write
            edit_mode: The editing mode
            file_exists: Whether the file already exists
        
        Returns:
            True if risky, False if safe
        """
        # No need to check for new files or explicit modes
        if not file_exists or edit_mode in ["append", "full_replace"]:
            return False
        
        # For lines_replace mode, be more flexible with safety checks
        if edit_mode == "lines_replace":
            # Check if existing code marker is present
            existing_code_patterns = [
                '// ... existing code ...',
                '# ... existing code ...',
                '/* ... existing code ...',
                '<!-- ... existing code -->',
            ]
            
            has_existing_marker = any(pattern in new_content for pattern in existing_code_patterns)
            
            # If no existing code marker found, only warn for very short content
            if not has_existing_marker:
                # Only block if the new content is suspiciously short compared to original
                if len(original_content.strip()) > 200 and len(new_content.strip()) < 30:
                    print_current("🚨 Safety check: Original file has substantial content but new content is very short")
                    return True
                else:
                    # Allow the edit but log a warning
                    print_current("⚠️ lines_replace mode without existing code markers - treating as full replacement")
                    return False
            
            return False
        
        # For other modes, apply stricter checks
        return False

    def _optimize_query_for_performance(self, query: str) -> Tuple[Union[str, List[str]], bool]:
        """
        Optimize LLM-generated complex queries for better performance.
        Returns (optimized_query_or_groups, should_split)
        """
        # Count OR operations in the query
        or_count = query.count('|')
        
        # If query has too many OR operations, suggest splitting
        if or_count >= 8:  # Threshold for complex queries
            print_current(f"⚠️ Complex query detected with {or_count + 1} terms")
            # Split query into logical groups
            terms = query.split('|')
            
            # Group related terms
            groups = self._group_related_terms(terms)
            
            return groups, True
        
        # Apply exclude pattern optimizations for common patterns
        optimized_query = query
        return optimized_query, False
    
    def _group_related_terms(self, terms: List[str]) -> List[str]:
        """Group related search terms to reduce query complexity"""
        groups = []
        
        # Group 1: API and LLM core terms
        api_terms = [t.strip() for t in terms if any(keyword in t.lower() for keyword in ['openai', 'gpt', 'llm', 'api'])]
        if api_terms:
            groups.append('|'.join(api_terms))
        
        # Group 2: Chinese LLM models
        chinese_llm = [t.strip() for t in terms if any(keyword in t.lower() for keyword in ['chatglm', 'baichuan', 'qwen', 'glm'])]
        if chinese_llm:
            groups.append('|'.join(chinese_llm))
        
        # Group 3: Completion and response terms
        completion_terms = [t.strip() for t in terms if any(keyword in t.lower() for keyword in ['completion', 'chat_completion', 'response'])]
        if completion_terms:
            groups.append('|'.join(completion_terms))
        
        # Add remaining terms
        used_terms = set()
        for group in groups:
            used_terms.update(group.split('|'))
        
        remaining = [t.strip() for t in terms if t.strip() not in used_terms]
        if remaining:
            groups.append('|'.join(remaining))
        
        return groups
    
    def _execute_split_search(self, query_groups: List[str], include_pattern: str, exclude_pattern: str, case_sensitive: bool, max_results: Optional[int]) -> Dict[str, Any]:
        """Execute search as multiple smaller queries"""
        all_results = []
        all_queries = []
        
        # Improved exclude pattern for better performance
        if not exclude_pattern:
            exclude_pattern = "output_*/*|final_test_output/*|__pycache__/*|*.egg-info/*|cache/*"
        else:
            exclude_pattern = f"{exclude_pattern}|output_*/*|__pycache__/*|*.egg-info/*"
        
        for i, group_query in enumerate(query_groups):
            print_current(f"🔍 Executing search group {i+1}/{len(query_groups)}: {group_query}")
            
            result = self._execute_single_search(group_query, include_pattern, exclude_pattern, case_sensitive, max_results)
            
            if result['results']:
                all_results.extend(result['results'])
                all_queries.append(group_query)
                
                # Limit total results to prevent overwhelming output
                if max_results is not None and len(all_results) >= max_results:
                    print_current(f"⚠️ Reached result limit ({max_results}), stopping further searches")
                    break
        
        # Deduplicate results based on file and line number
        seen = set()
        unique_results = []
        for result in all_results:
            key = (result['file'], result['line_number'])
            if key not in seen:
                seen.add(key)
                unique_results.append(result)
        
        return {
            'query': ' + '.join(all_queries),
            'results': unique_results[:50],  # Limit final results
            'total_matches': len(unique_results),
            'max_results': 50,
            'truncated': len(unique_results) > 50,
            'optimization_applied': True,
            'query_groups': len(query_groups)
        }
    
    def _execute_single_search(self, query: str, include_pattern: str, exclude_pattern: str, case_sensitive: bool, max_results: Optional[int]) -> Dict[str, Any]:
        """Execute a single search query"""
        # Try system grep first if available
        if hasattr(self, 'system_grep_available') and self.system_grep_available:
            return self._execute_system_grep_search(query, include_pattern, exclude_pattern, case_sensitive, max_results)
        else:
            # Fallback to Python implementation
            return self._execute_python_search_original(query, include_pattern, exclude_pattern, case_sensitive, max_results)
    
    def _execute_python_search_original(self, query: str, include_pattern: str, exclude_pattern: str, case_sensitive: bool, max_results: Optional[int]) -> Dict[str, Any]:
        """Execute a single search query using Python (original implementation)"""
        results = []
        # Enhanced exclude pattern for better performance
        if not exclude_pattern:
            exclude_pattern = "output_*/*|final_test_output/*|__pycache__/*|*.egg-info/*|cache/*"
        
        # Enable followlinks=True to traverse symbolic links
        for root, _, files in os.walk(self.workspace_root, followlinks=True):
            for file in files:
                file_path = os.path.join(root, file)
                rel_path = os.path.relpath(file_path, self.workspace_root)
                
                if include_pattern and not fnmatch.fnmatch(rel_path, include_pattern):
                    continue
                if exclude_pattern and fnmatch.fnmatch(rel_path, exclude_pattern):
                    continue
                
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        for i, line in enumerate(f, 1):
                            flags = 0 if case_sensitive else re.IGNORECASE
                            if re.search(query, line, flags=flags):
                                results.append({
                                    'file': rel_path,
                                    'line_number': i,
                                    'line': line.rstrip()
                                })
                                
                                if max_results is not None and len(results) >= max_results:
                                    break
                except Exception:
                    continue
                
                if max_results is not None and len(results) >= max_results:
                    break
        
        return {
            'query': query,
            'results': results,
            'total_matches': len(results),
            'max_results': max_results,
            'truncated': max_results is not None and len(results) >= max_results
        }

    def _execute_system_grep_search(self, query: str, include_pattern: str, exclude_pattern: str, case_sensitive: bool, max_results: Optional[int]) -> Dict[str, Any]:
        """Execute search using system grep command"""
        try:
            results = []
            
            # Build better grep command for performance  
            cmd = ['grep', '-E', '-R', '-n', '--binary-files=without-match']  # extended regex, follow symlinks, line numbers, skip binary files
            
            if not case_sensitive:
                cmd.append('-i')  # case insensitive
            
            # Add basic exclude patterns for performance (no wildcards, only exact directory names)
            cmd.extend(['--exclude-dir', '__pycache__', '--exclude-dir', '.git', '--exclude-dir', 'node_modules'])
            
            # For output_* patterns, find actual directory names and exclude them
            if exclude_pattern and 'output_' in exclude_pattern:
                import glob
                output_dirs = glob.glob(os.path.join(self.workspace_root, 'output_*'))
                for output_dir in output_dirs:
                    dir_name = os.path.basename(output_dir)
                    cmd.extend(['--exclude-dir', dir_name])
            
            # Add include pattern if specified
            if include_pattern:
                cmd.extend(['--include', include_pattern])
            
            # Add the query pattern (escape special characters for shell safety)
            cmd.append(query)
            
            # Add search path
            cmd.append(self.workspace_root)
            
            # Execute grep
            print_current(f"🔍 Executing system grep: {' '.join(cmd)} (target: {self.workspace_root})")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode in [0, 1]:  # 0=found, 1=not found (both OK)
                lines = result.stdout.strip().split('\n') if result.stdout.strip() else []
                
                # Filter results manually if grep couldn't exclude everything
                filtered_lines = []
                if exclude_pattern:
                    import fnmatch
                    for line in lines:
                        if ':' in line:
                            filepath_part = line.split(':', 1)[0]
                            rel_path = os.path.relpath(filepath_part, self.workspace_root)
                            
                            # Check if file matches exclude pattern
                            should_exclude = False
                            for pattern in exclude_pattern.split('|'):
                                if pattern.strip() and fnmatch.fnmatch(rel_path, pattern.strip()):
                                    should_exclude = True
                                    break
                            
                            if not should_exclude:
                                filtered_lines.append(line)
                        else:
                            filtered_lines.append(line)
                else:
                    filtered_lines = lines
                
                # Apply max_results limit if specified
                lines_to_process = filtered_lines if max_results is None else filtered_lines[:max_results]
                for line in lines_to_process:
                    if ':' in line:
                        # Split carefully to handle filenames with colons
                        first_colon = line.find(':')
                        if first_colon > 0:
                            filepath_part = line[:first_colon]
                            remainder = line[first_colon + 1:]
                            
                            second_colon = remainder.find(':')
                            if second_colon > 0:
                                line_number_str = remainder[:second_colon]
                                content = remainder[second_colon + 1:]
                                
                                try:
                                    file_path = os.path.relpath(filepath_part, self.workspace_root)
                                    line_number = int(line_number_str) if line_number_str.isdigit() else 1
                                    
                                    results.append({
                                        'file': file_path,
                                        'line_number': line_number,
                                        'line': content
                                    })
                                except (ValueError, OSError):
                                    continue
                
                return {
                    'query': query,
                    'results': results,
                    'total_matches': len(results),
                    'max_results': max_results,
                    'truncated': max_results is not None and len(filtered_lines) > max_results,
                    'search_method': 'system_grep',
                    'performance_boost': True
                }
            else:
                # If grep fails, fall back to Python implementation
                print_debug(f"⚠️ System grep failed (code {result.returncode}), falling back to Python")
                return self._execute_python_search(query, include_pattern, exclude_pattern, case_sensitive, max_results)
                
        except Exception as e:
            print_debug(f"⚠️ System grep error: {e}, falling back to Python")
            return self._execute_python_search(query, include_pattern, exclude_pattern, case_sensitive, max_results)

    def _execute_python_search(self, query: str, include_pattern: str, exclude_pattern: str, case_sensitive: bool, max_results: Optional[int]) -> Dict[str, Any]:
        """Execute search using Python implementation with special character handling"""
        results = []
        
        # Enhanced exclude pattern for better performance
        if not exclude_pattern:
            exclude_pattern = "output_*/*|final_test_output/*|__pycache__/*|*.egg-info/*|cache/*"
        
        # Handle invalid regex patterns by escaping them
        try:
            re.compile(query)
            use_regex = True
        except re.error:
            # If regex is invalid, escape special characters and search as literal string
            query = re.escape(query)
            use_regex = True
            print_current(f"⚠️ Invalid regex pattern, searching as literal string: {query}")
        
        # Enable followlinks=True to traverse symbolic links
        for root, _, files in os.walk(self.workspace_root, followlinks=True):
            for file in files:
                file_path = os.path.join(root, file)
                rel_path = os.path.relpath(file_path, self.workspace_root)
                
                if include_pattern and not fnmatch.fnmatch(rel_path, include_pattern):
                    continue
                if exclude_pattern and fnmatch.fnmatch(rel_path, exclude_pattern):
                    continue
                
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        for i, line in enumerate(f, 1):
                            flags = 0 if case_sensitive else re.IGNORECASE
                            if re.search(query, line, flags=flags):
                                results.append({
                                    'file': rel_path,
                                    'line_number': i,
                                    'line': line.rstrip()
                                })
                                
                                if max_results is not None and len(results) >= max_results:
                                    break
                except Exception:
                    continue
                
                if max_results is not None and len(results) >= max_results:
                    break
        
        return {
            'query': query,
            'results': results,
            'total_matches': len(results),
            'max_results': max_results,
            'truncated': max_results is not None and len(results) >= max_results,
            'search_method': 'python_fallback'
        }

    def _process_markdown_content(self, content: str, target_file: str) -> str:
        """
        Process markdown content with special formatting rules.
        
        Args:
            content: The content to process
            target_file: Target file path to determine if it's a markdown file
            
        Returns:
            Processed content with markdown-specific formatting applied
        """
        # Check if this is a markdown file
        if not target_file.lower().endswith('.md'):
            return content
        
        processed_content = content
        
        # Convert \n markers to actual newlines
        # Only convert literal \n (not already converted newlines)
        if '\\n' in processed_content:
            processed_content = processed_content.replace('\\n', '\n')
            print_current(f"📝 Converted \\n markers to actual newlines in markdown file")
        
        # Ensure file ends with two newlines (two empty lines) for markdown files
        if processed_content:
            if not processed_content.endswith('\n\n'):
                if processed_content.endswith('\n'):
                    # Has one newline, add another one
                    processed_content += '\n'
                    print_debug(f"Added second newline at end of markdown file")
                else:
                    # Has no newline, add two newlines
                    processed_content += '\n\n'
                    print_debug(f"Added two newlines at end of markdown file")
        
        return processed_content

 