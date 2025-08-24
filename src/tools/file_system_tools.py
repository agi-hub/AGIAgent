#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from .print_system import print_system, print_current, print_system, print_debug
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

# Import Mermaid processor for handling charts in markdown files
try:
    from .mermaid_processor import mermaid_processor
    MERMAID_PROCESSOR_AVAILABLE = True
except ImportError:
    print_debug("⚠️ Mermaid processor not available")
    MERMAID_PROCESSOR_AVAILABLE = False


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
            print_system("🚀 System grep detected, will use for faster searching")
        else:
            print_system("⚠️ System grep not available, using Python fallback")
    
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
        
        #print_current(f"🎯 Requested to read file: {target_file}")
        #print_current(f"📂 Current working directory: {self.workspace_root}")
        
        file_path = self._resolve_path(target_file)
        
        if not os.path.exists(file_path):
            print_current(f"❌ File does not exist: {file_path}")
            try:
                current_dir_files = os.listdir(self.workspace_root)
                print_current(f"📁 Files in current directory: {current_dir_files}")
            except Exception as e:
                print_current(f"⚠️ Cannot list current directory: {e}")
            
            return {
                'status': 'failed',
                'file': target_file,
                'error': f'File not found: {file_path}',
                'workspace_root': self.workspace_root,
                'resolved_path': file_path,
                'exists': False
            }
        
        if not os.path.isfile(file_path):
            print_current(f"❌ Path is not a file: {file_path}")
            return {
                'status': 'failed',
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
                        'status': 'success',
                        'file': target_file,
                        'content': content,
                        'total_lines': total_lines,
                        'resolved_path': file_path
                    }
                else:
                    # File too large, truncate to max_entire_lines
                    content_lines = all_lines[:max_entire_lines]
                    content = ''.join(content_lines)
                    after_summary = f"... {total_lines - max_entire_lines} lines truncated ..."
                    print_current(f"📄 Read entire file (truncated), showing first {max_entire_lines} lines of {total_lines}")
                    return {
                        'status': 'success',
                        'file': target_file,
                        'content': content,
                        'after_summary': after_summary,
                        'total_lines': total_lines,
                        'lines_shown': max_entire_lines,
                        'truncated': True,
                        'resolved_path': file_path
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
                    'status': 'success',
                    'file': target_file,
                    'content': content,
                    'before_summary': before_summary,
                    'after_summary': after_summary,
                    'start_line': start_line_one_indexed,
                    'end_line': end_line_one_indexed_inclusive,
                    'total_lines': total_lines,
                    'resolved_path': file_path
                }
        except UnicodeDecodeError as e:
            print_current(f"❌ File encoding error: {e}")
            return {
                'status': 'failed',
                'file': target_file,
                'error': f'Unicode decode error: {str(e)}',
                'resolved_path': file_path
            }
        except Exception as e:
            print_current(f"❌ Error occurred while reading file: {e}")
            return {
                'status': 'failed',
                'file': target_file,
                'error': str(e),
                'resolved_path': file_path
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
                'status': 'success',
                'sub directories': "No sub-directories" if not directories else str(sorted(directories)),
                'files': str(sorted(files))
            }
        except Exception as e:
            return {
                'status': 'failed',
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
                  old_code: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """
        Use three modes to edit files or create new files.
        
        Args:
            target_file: File path
            edit_mode: Edit mode - "lines_replace", "append", "full_replace"
            code_edit: Code/text content to edit
            instructions: Optional edit description
            old_code: For lines_replace mode
            
        Returns:
            Dictionary containing edit results
        
        Edit Modes:
            - lines_replace: Exact replacement mode
            - append: Append to end of file
            - full_replace: Completely replace file content
        """
        # Check for dummy placeholder file created by hallucination detection
        if target_file == "dummy_file_placeholder.txt" or target_file.endswith("/dummy_file_placeholder.txt"):
            print_current(f"🚨 HALLUCINATION PREVENTION: Detected dummy placeholder file '{target_file}' - skipping actual file operation")
            return {
                'status': 'failed',
                'file': target_file,
                'error': 'Dummy placeholder file detected - hallucination prevention active',
                'hallucination_prevention': True
            }
        
        # Default to append mode if auto mode is set for safety
        if edit_mode == "auto":
            edit_mode = "append"
        
        # Compatible with old edit_mode names
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
                original_content, cleaned_code_edit, edit_mode, target_file, old_code
            )
            
            # Write the new content
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(new_content)
            
            # Preprocess bullet formatting for markdown files
            if target_file.lower().endswith('.md'):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content_to_preprocess = f.read()
                    
                    # Apply bullet formatting preprocessing
                    preprocessed_content = self._preprocess_bullet_formatting(content_to_preprocess)
                    
                    # Only rewrite if content has changed
                    if preprocessed_content != content_to_preprocess:
                        with open(file_path, 'w', encoding='utf-8') as f:
                            f.write(preprocessed_content)
                        print_current(f"📝 Applied bullet formatting preprocessing to markdown file")
                except Exception as e:
                    print_current(f"⚠️ Error during bullet formatting preprocessing: {e}")
            
            # Process Mermaid charts if this is a markdown file
            mermaid_result = None
            if target_file.lower().endswith('.md') and MERMAID_PROCESSOR_AVAILABLE:
                try:
                    if mermaid_processor.has_mermaid_charts(file_path):
                        print_current(f"🎨 Detected Mermaid charts in markdown file, processing...")
                        mermaid_result = mermaid_processor.process_markdown_file(file_path)
                        if mermaid_result['status'] == 'success':
                            print_current(f"✅ Mermaid processing completed: {mermaid_result['message']}")
                        else:
                            print_current(f"⚠️ Mermaid processing failed: {mermaid_result.get('message', 'Unknown error')}")
                except Exception as e:
                    print_current(f"⚠️ Error during Mermaid processing: {e}")
                    mermaid_result = {
                        'status': 'failed',
                        'error': str(e),
                        'message': f'Mermaid processing error: {e}'
                    }
            
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
            
            result = {
                'status': 'success',
                'file': target_file,
                'action': action,
                'edit_mode': edit_mode,
                'snapshot_created': file_exists  # Only create snapshot for existing files
            }
            
            # Add Mermaid processing result if applicable
            if mermaid_result is not None:
                result['mermaid_processing'] = mermaid_result
            
            # Convert markdown to Word and PDF if this is a markdown file
            conversion_result = None
            if target_file.lower().endswith('.md'):
                try:
                    conversion_result = self._convert_markdown_to_formats(file_path, target_file)
                    if conversion_result:
                        result['conversion'] = conversion_result
                except Exception as e:
                    print_current(f"⚠️ Error during markdown conversion: {e}")
                    result['conversion'] = {
                        'status': 'failed',
                        'error': str(e),
                        'message': f'Markdown conversion error: {e}'
                    }
            
            return result
            
        except Exception as e: 
            return {
                'status': 'failed',
                'file': target_file,
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
                             target_file: str, old_code: Optional[str] = None) -> str:
        """
        Process edit based on the specified mode.

        Args:
            original_content: The original file content
            code_edit: The new content to add/replace
            edit_mode: Edit mode - "lines_replace", "append", "full_replace"
            target_file: Target file path
            old_code: For lines_replace mode, the exact code to find and replace

        Returns:
            The new file content after applying the edit
        """
        if edit_mode == "lines_replace":
            # Precise replacement mode using exact code matching
            return self._process_precise_code_edit(original_content, code_edit, target_file, old_code)
        elif edit_mode == "append":
            # Append to the end of the file
            return self._append_content(original_content, code_edit)
        elif edit_mode == "full_replace":
            # Completely replace the file content
            return code_edit
        else:
            # Compatibility mode: default to lines_replace
            print_current(f"⚠️ Unknown edit_mode '{edit_mode}', defaulting to lines_replace")
            return self._process_precise_code_edit(original_content, code_edit, target_file, old_code)

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
        # No special processing needed for append content
        clean_content = new_content.strip()
        
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

    def _process_precise_code_edit(self, original_content: str, code_edit: str, target_file: str, old_code: Optional[str] = None) -> str:
        """
        Process code edits using 100% precise matching algorithm.
        
        Args:
            original_content: The original file content
            code_edit: The new content to replace with  
            target_file: Target file path
            old_code: The exact code snippet to find and replace
            
        Returns:
            The new file content after applying the edit
            
        Raises:
            ValueError: If old_code is not provided for lines_replace mode
            ValueError: If old_code is not found exactly in the original content
        """
        if old_code is None:
            raise ValueError(
                "EDIT REJECTED: old_code parameter is required for lines_replace mode. "
                "You must provide the exact code snippet that you want to replace, "
                "including all whitespace and indentation."
            )
        
        # Clean the old_code in the same way as code_edit
        cleaned_old_code = self._clean_markdown_markers(old_code)
        cleaned_old_code = self._fix_html_entities(cleaned_old_code)
        cleaned_old_code = self._process_markdown_content(cleaned_old_code, target_file)
        
        # Clean the new code_edit 
        cleaned_code_edit = self._clean_markdown_markers(code_edit)
        cleaned_code_edit = self._fix_html_entities(cleaned_code_edit)
        cleaned_code_edit = self._process_markdown_content(cleaned_code_edit, target_file)
        
        print_current(f"🎯 Starting precise code replacement")
        print_current(f"📝 Looking for old code snippet ({len(cleaned_old_code)} chars)")
        print_current(f"🔄 Will replace with new code snippet ({len(cleaned_code_edit)} chars)")
        
        # Apply direct string replacement
        return self._apply_precise_replacement(original_content, cleaned_old_code, cleaned_code_edit)
    
    def _apply_precise_replacement(self, original_content: str, old_code: str, new_code: str) -> str:
        """
        Apply precise replacement using exact string matching.
        
        Args:
            original_content: The original file content
            old_code: The exact code snippet to find
            new_code: The new code snippet to replace with
            
        Returns:
            The updated file content
            
        Raises:
            ValueError: If the old_code is not found exactly in the original content
        """
        # Simplified logic: only perform direct exact string replacement
        return self._apply_direct_precise_replacement(original_content, old_code, new_code)
    
    def _apply_direct_precise_replacement(self, original_content: str, old_code: str, new_code: str) -> str:
        """
        Apply direct string replacement.
        
        Args:
            original_content: The original file content
            old_code: The exact code snippet to find
            new_code: The new code snippet to replace with
            
        Returns:
            The updated file content
            
        Raises:
            ValueError: If the old_code is not found exactly once in the original content
        """
        # Count occurrences of old_code in original content
        occurrence_count = original_content.count(old_code)
        
        if occurrence_count == 0:
            raise ValueError(
                f"EDIT REJECTED: The specified old_code was not found in the file. "
                f"Please check that the code snippet matches exactly (including whitespace and indentation). "
                f"Old code snippet (first 200 chars): {repr(old_code[:200])}"
            )
        elif occurrence_count > 1:
            raise ValueError(
                f"EDIT REJECTED: The specified old_code appears {occurrence_count} times in the file. "
                f"For safety, please make the old_code more specific to match exactly one location. "
                f"You can add more context lines to make it unique."
            )
        
        # Perform the replacement
        new_content = original_content.replace(old_code, new_code, 1)
        
        print_current(f"✅ Successfully replaced code snippet (direct replacement)")
        print_current(f"📊 Content size: {len(original_content)} → {len(new_content)} chars")
        
        return new_content
    

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
            'status': 'success',
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
                'status': 'failed',
                'file': target_file,
                'error': 'File does not exist'
            }
        
        try:
            if os.path.isdir(file_path):
                return {
                    'status': 'failed',
                    'file': target_file,
                    'error': 'Target is a directory, not a file'
                }
            
            os.remove(file_path)
            
            return {
                'status': 'success',
                'file': target_file,
                'action': 'deleted'
            }
        except Exception as e:
            return {
                'status': 'failed',
                'file': target_file,
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
        
        # For lines_replace mode, no special safety checks needed since we have old_code parameter
        if edit_mode == "lines_replace":
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
            'status': 'success',
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
            'status': 'success',
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
                    'status': 'success',
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
            'status': 'success',
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
    
    def _preprocess_bullet_formatting(self, content: str) -> str:
        """Preprocess Markdown files to fix bullet point formatting issues
        
        Ensure there is a blank line before bullet points so that Pandoc recognizes them as separate lists.
        """
        lines = content.split('\n')
        processed_lines = []
        
        for i, line in enumerate(lines):
            processed_lines.append(line)
            
            # Check if the current line is not empty, and the next line is a bullet point
            if (line.strip() and 
                i + 1 < len(lines) and 
                lines[i + 1].strip().startswith('- ') and
                not line.strip().startswith('- ') and  # Current line is not a bullet point
                (i == 0 or lines[i - 1].strip() != '')):  # Previous line is not empty
                
                # Add a blank line after the current line to ensure the following bullet point is recognized correctly
                processed_lines.append('')
        
        return '\n'.join(processed_lines)

    def parse_doc_to_md(self, folder_path: str) -> Dict[str, Any]:
        """
        Recursively traverse a folder and convert document files to markdown using markitdown.
        
        Args:
            folder_path: Path to the folder to process
            
        Returns:
            Dictionary with processing results
        """
        try:
            # Import markitdown
            from markitdown import MarkItDown
            markitdown = MarkItDown()
        except ImportError:
            return {
                'status': 'failed',
                'error': 'markitdown library not found. Please install it with: pip install markitdown',
                'folder': folder_path
            }
        
        # Resolve the folder path
        resolved_folder = self._resolve_path(folder_path)
        
        if not os.path.exists(resolved_folder):
            return {
                'status': 'failed',
                'error': f'Folder not found: {resolved_folder}',
                'folder': folder_path
            }
        
        if not os.path.isdir(resolved_folder):
            return {
                'status': 'failed',
                'error': f'Path is not a directory: {resolved_folder}',
                'folder': folder_path
            }
        
        # Define supported document extensions
        doc_extensions = {'.docx', '.doc', '.pdf', '.xlsx', '.xls', '.pptx', '.ppt', '.txt', '.rtf'}
        
        processed_files = []
        failed_files = []
        skipped_files = []
        
        print_current(f"📁 Starting document parsing in folder: {resolved_folder}")
        
        # Recursively walk through all files
        for root, dirs, files in os.walk(resolved_folder):
            for file in files:
                file_path = os.path.join(root, file)
                file_ext = os.path.splitext(file)[1].lower()
                
                # Check if it's a supported document type
                if file_ext not in doc_extensions:
                    continue
                
                # Get relative path for display
                rel_path = os.path.relpath(file_path, self.workspace_root)
                
                # Generate output markdown filename
                base_name = os.path.splitext(file)[0]
                md_filename = base_name + '.md'
                md_path = os.path.join(root, md_filename)
                
                # Skip if markdown file already exists and is newer than source
                if os.path.exists(md_path):
                    src_mtime = os.path.getmtime(file_path)
                    md_mtime = os.path.getmtime(md_path)
                    if md_mtime >= src_mtime:
                        skipped_files.append({
                            'file': rel_path,
                            'reason': 'markdown file already exists and is newer'
                        })
                        continue
                
                print_current(f"🔄 Converting: {rel_path}")
                
                try:
                    # Convert document to markdown
                    result = markitdown.convert(file_path)
                    
                    if result and hasattr(result, 'text_content'):
                        markdown_content = result.text_content
                        
                        # Write the markdown content to file
                        with open(md_path, 'w', encoding='utf-8') as f:
                            f.write(markdown_content)
                        
                        processed_files.append({
                            'source': rel_path,
                            'output': os.path.relpath(md_path, self.workspace_root),
                            'size': len(markdown_content)
                        })
                        
                        print_current(f"✅ Converted: {rel_path} → {os.path.basename(md_path)}")
                    else:
                        failed_files.append({
                            'file': rel_path,
                            'error': 'No content returned from markitdown conversion'
                        })
                        print_current(f"❌ Failed to convert: {rel_path} - No content returned")
                        
                except Exception as e:
                    failed_files.append({
                        'file': rel_path,
                        'error': str(e)
                    })
                    print_current(f"❌ Failed to convert: {rel_path} - {str(e)}")
        
        # Summary
        total_processed = len(processed_files)
        total_failed = len(failed_files)
        total_skipped = len(skipped_files)
        
        print_current(f"📊 Conversion complete:")
        print_current(f"   ✅ Processed: {total_processed} files")
        print_current(f"   ❌ Failed: {total_failed} files")
        print_current(f"   ⏭️ Skipped: {total_skipped} files")
        
        return {
            'status': 'success',
            'folder': folder_path,
            'resolved_folder': resolved_folder,
            'summary': {
                'processed': total_processed,
                'failed': total_failed,
                'skipped': total_skipped,
                'total_files': total_processed + total_failed + total_skipped
            },
            'processed_files': processed_files,
            'failed_files': failed_files,
            'skipped_files': skipped_files
        }

    def _check_pdf_engine_availability(self):
        """Check which PDF engines are available and return the best one"""
        engines = [
            ('xelatex', '--pdf-engine=xelatex'),
            ('lualatex', '--pdf-engine=lualatex'),
            ('pdflatex', '--pdf-engine=pdflatex'),
            ('wkhtmltopdf', '--pdf-engine=wkhtmltopdf'),
            ('weasyprint', '--pdf-engine=weasyprint')
        ]
        
        available_engines = []
        
        for engine_name, engine_option in engines:
            try:
                if engine_name in ['xelatex', 'lualatex', 'pdflatex']:
                    # Check if LaTeX engine is available
                    result = subprocess.run([engine_name, '--version'], 
                                         capture_output=True, text=True, timeout=5)
                    if result.returncode == 0:
                        available_engines.append((engine_name, engine_option))
                elif engine_name == 'wkhtmltopdf':
                    # Check if wkhtmltopdf is available
                    result = subprocess.run(['wkhtmltopdf', '--version'], 
                                         capture_output=True, text=True, timeout=5)
                    if result.returncode == 0:
                        available_engines.append((engine_name, engine_option))
                elif engine_name == 'weasyprint':
                    # Check if weasyprint is available
                    result = subprocess.run(['weasyprint', '--version'], 
                                         capture_output=True, text=True, timeout=5)
                    if result.returncode == 0:
                        available_engines.append((engine_name, engine_option))
            except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.CalledProcessError):
                continue
        
        if not available_engines:
            print("❌ No PDF engines available. Please install at least one of: xelatex, lualatex, pdflatex, wkhtmltopdf, or weasyprint")
            return None, None
        
        # Return the best available engine (prioritize xelatex > lualatex > pdflatex > others)
        priority_order = ['xelatex', 'lualatex', 'pdflatex', 'wkhtmltopdf', 'weasyprint']
        
        for preferred in priority_order:
            for engine_name, engine_option in available_engines:
                if engine_name == preferred:
                    return engine_name, engine_option
        
        # Fallback to first available
        selected_engine = available_engines[0]
        return selected_engine[0], selected_engine[1]

    def _get_engine_specific_options(self, engine_name):
        """Get engine-specific options based on the selected PDF engine"""
        if engine_name in ['xelatex', 'lualatex']:
            # XeLaTeX and LuaLaTeX support CJK fonts
            return [
                '-V', 'CJKmainfont=Noto Serif CJK SC',
                '-V', 'CJKsansfont=Noto Sans CJK SC',
                '-V', 'CJKmonofont=Noto Sans Mono CJK SC',
                '-V', 'mainfont=DejaVu Serif',
                '-V', 'sansfont=DejaVu Sans',
                '-V', 'monofont=DejaVu Sans Mono',
            ]
        elif engine_name == 'pdflatex':
            # pdfLaTeX doesn't support CJK fonts natively, use basic fonts
            return [
                '-V', 'mainfont=DejaVu Serif',
                '-V', 'sansfont=DejaVu Sans',
                '-V', 'monofont=DejaVu Sans Mono',
            ]
        else:
            # wkhtmltopdf and weasyprint don't use LaTeX, return minimal options
            return []

    def _convert_markdown_to_formats(self, file_path: str, target_file: str) -> Dict[str, Any]:
        """
        Convert Markdown files to Word and PDF formats
        
        Args:
            file_path: Absolute path of Markdown file
            target_file: Relative path of Markdown file
            
        Returns:
            Dictionary containing conversion results
        """
        import subprocess
        from pathlib import Path
        
        try:
            md_path = Path(file_path)
            base_name = md_path.stem
            output_dir = md_path.parent
            
            # Generate output filename
            word_file = output_dir / f"{base_name}.docx"
            pdf_file = output_dir / f"{base_name}.pdf"
            
            conversion_results = {
                'status': 'success',
                'markdown_file': target_file,
                'conversions': {}
            }
            
            # Convert to Word document
            print_current(f"📄 Converting Markdown to Word document: {word_file.name}")
            try:
                # Use pandoc to convert to Word
                cmd = [
                    'pandoc',
                    md_path.name,  # Use filename instead of full path
                    '-o', word_file.name,  # Use filename instead of full path
                    '--from', 'markdown',
                    '--to', 'docx'
                ]
                
                # Execute command in markdown file directory
                result = subprocess.run(cmd, check=True, capture_output=True, text=True, cwd=str(output_dir))
                
                if word_file.exists():
                    file_size = word_file.stat().st_size
                    conversion_results['conversions']['word'] = {
                        'status': 'success',
                        'file': str(word_file.relative_to(self.workspace_root)),
                        'size': file_size,
                        'size_kb': f"{file_size / 1024:.1f} KB"
                    }
                    print_current(f"✅ Word document conversion successful: {word_file.name} ({file_size / 1024:.1f} KB)")
                else:
                    conversion_results['conversions']['word'] = {
                        'status': 'failed',
                        'error': 'Word file not generated'
                    }
                    print_current(f"❌ Word document conversion failed: File not generated")
                    
            except subprocess.CalledProcessError as e:
                conversion_results['conversions']['word'] = {
                    'status': 'failed',
                    'error': f'pandoc conversion failed: {e.stderr}'
                }
                print_current(f"❌ Word document conversion failed: {e.stderr}")
            except Exception as e:
                conversion_results['conversions']['word'] = {
                    'status': 'failed',
                    'error': f'Conversion exception: {str(e)}'
                }
                print_current(f"❌ Word document conversion exception: {str(e)}")
            
            # Convert to PDF document
            print_current(f"📄 Converting Markdown to PDF document: {pdf_file.name}")
            try:
                # Use trans_md_to_pdf.py script to convert to PDF
                trans_script = Path(__file__).parent.parent / "utils" / "trans_md_to_pdf.py"
                
                if trans_script.exists():
                    cmd = [
                        'python3',
                        str(trans_script),
                        md_path.name,  # Use filename instead of full path
                        pdf_file.name  # Use filename instead of full path
                    ]
                    
                    # Execute command in markdown file directory
                    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(output_dir))
                    
                    if pdf_file.exists():  # Check file existence only, ignore warnings in returncode
                        file_size = pdf_file.stat().st_size
                        conversion_results['conversions']['pdf'] = {
                            'status': 'success',
                            'file': str(pdf_file.relative_to(self.workspace_root)),
                            'size': file_size,
                            'size_kb': f"{file_size / 1024:.1f} KB"
                        }
                        
                        # Check if there were warnings during conversion
                        if result.returncode != 0:
                            print_current(f"✅ PDF document conversion successful: {pdf_file.name} ({file_size / 1024:.1f} KB)")
                            print_current(f"⚠️  Note: Conversion completed with warnings (non-critical)")
                            if result.stderr:
                                print_current(f"   Warning details: {result.stderr[:200]}...")  # Show first 200 chars
                        else:
                            print_current(f"✅ PDF document conversion successful: {pdf_file.name} ({file_size / 1024:.1f} KB)")
                    else:
                        # If trans_md_to_pdf.py fails
                        print_current(f"⚠️ trans_md_to_pdf.py conversion failed...")
                        error_msg = result.stderr if result.stderr else result.stdout
                        print_current(f"   Error message: {error_msg}")
                        
                        # Analyze error for common issues
                        if error_msg:
                            if "Cannot load file" in error_msg or "Invalid" in error_msg:
                                print_current("🔍 Detected image format compatibility issues")
                                print_current("💡 Suggestion: Consider converting WebP/TIFF images to PNG/JPEG")
                            elif "Cannot determine size" in error_msg or "BoundingBox" in error_msg:
                                print_current("🔍 Detected image size/boundary issues")  
                                print_current("💡 Suggestion: Image preprocessing may help resolve this")
                        
                        # Use pandoc directly for conversion with fallback engine
                        try:
                            # Check available PDF engines
                            engine_name, engine_option = self._check_pdf_engine_availability()
                            if not engine_name:
                                # No PDF engines available - use Word as fallback
                                print_current(f"⚠️ No PDF engines available, using Word document as fallback")
                                conversion_results['conversions']['pdf'] = {
                                    'status': 'fallback_to_word',
                                    'error': 'No PDF engines available (xelatex, lualatex, pdflatex, wkhtmltopdf, weasyprint)',
                                    'fallback_method': 'word_document',
                                    'message': 'Generated Word document instead of PDF due to missing LaTeX/PDF engines'
                                }
                                # Generate an additional Word document as PDF fallback
                                self._generate_fallback_word_document(md_path, output_dir, conversion_results)
                                return conversion_results
                            
                            # Get engine-specific options
                            engine_options = self._get_engine_specific_options(engine_name)
                            
                            direct_cmd = [
                                'pandoc',
                                md_path.name,
                                '-o', pdf_file.name,
                                engine_option,  # Use the selected engine
                            ]
                            
                            # Add engine-specific options
                            direct_cmd.extend(engine_options)
                            
                            # Add common options
                            direct_cmd.extend([
                                '-V', 'fontsize=12pt',
                                '-V', 'geometry:margin=2.5cm',
                                '-V', 'geometry:a4paper',
                                '-V', 'linestretch=1.5',
                                '--highlight-style=tango',
                                '-V', 'colorlinks=true',
                                '-V', 'linkcolor=blue',
                                '-V', 'urlcolor=blue',
                                '--toc',
                                '--wrap=preserve'
                            ])
                            
                            # Add LaTeX-specific options only for LaTeX engines
                            if engine_name in ['xelatex', 'lualatex', 'pdflatex']:
                                direct_cmd.extend([
                                    '-V', 'graphics=true',
                                ])
                            
                            print_current(f"Using fallback PDF engine: {engine_name}")
                            
                            direct_result = subprocess.run(direct_cmd, capture_output=True, text=True, cwd=str(output_dir))
                            
                            if direct_result.returncode == 0 and pdf_file.exists():
                                file_size = pdf_file.stat().st_size
                                conversion_results['conversions']['pdf'] = {
                                    'status': 'success',
                                    'file': str(pdf_file.relative_to(self.workspace_root)),
                                    'size': file_size,
                                    'size_kb': f"{file_size / 1024:.1f} KB",
                                    'method': 'direct_pandoc'
                                }
                                print_current(f"✅ PDF document conversion successful (Direct pandoc): {pdf_file.name} ({file_size / 1024:.1f} KB)")
                            else:
                                conversion_results['conversions']['pdf'] = {
                                    'status': 'failed',
                                    'error': f'Direct pandoc conversion also failed: {direct_result.stderr if direct_result.stderr else "Unknown error"}'
                                }
                                print_current(f"❌ PDF document conversion failed (Direct pandoc): {direct_result.stderr if direct_result.stderr else 'Unknown error'}")
                        except Exception as e:
                            conversion_results['conversions']['pdf'] = {
                                'status': 'failed',
                                'error': f'Direct pandoc conversion exception: {str(e)}'
                            }
                            print_current(f"❌ PDF document conversion exception (Direct pandoc): {str(e)}")
                else:
                    # If trans_md_to_pdf.py doesn't exist
                    print_current(f"⚠️ trans_md_to_pdf.py script doesn't exist")
                    
                    # Check available PDF engines
                    engine_name, engine_option = self._check_pdf_engine_availability()
                    if not engine_name:
                        # No PDF engines available - use Word as fallback
                        print_current(f"⚠️ No PDF engines available, using Word document as fallback")
                        conversion_results['conversions']['pdf'] = {
                            'status': 'fallback_to_word',
                            'error': 'No PDF engines available (xelatex, lualatex, pdflatex, wkhtmltopdf, weasyprint)',
                            'fallback_method': 'word_document',
                            'message': 'Generated Word document instead of PDF due to missing LaTeX/PDF engines'
                        }
                        # Generate an additional Word document as PDF fallback
                        self._generate_fallback_word_document(md_path, output_dir, conversion_results)
                        return conversion_results
                    
                    # Get engine-specific options
                    engine_options = self._get_engine_specific_options(engine_name)
                    
                    cmd = [
                        'pandoc',
                        md_path.name,  # Use filename instead of full path
                        '-o', pdf_file.name,  # Use filename instead of full path
                        engine_option,  # Use the selected engine
                    ]
                    
                    # Add engine-specific options
                    cmd.extend(engine_options)
                    
                    # Add common options
                    cmd.extend([
                        '-V', 'fontsize=12pt',
                        '-V', 'geometry:margin=2.5cm',
                        '-V', 'geometry:a4paper',
                        '-V', 'linestretch=1.5',
                        '--highlight-style=tango',
                        '-V', 'colorlinks=true',
                        '-V', 'linkcolor=blue',
                        '-V', 'urlcolor=blue',
                        '--toc',
                        '--wrap=preserve'
                    ])
                    
                    # Add LaTeX-specific options only for LaTeX engines
                    if engine_name in ['xelatex', 'lualatex', 'pdflatex']:
                        cmd.extend([
                            '-V', 'graphics=true',
                        ])
                    
                    print_current(f"Using fallback PDF engine: {engine_name}")
                    
                    # Execute command in markdown file directory
                    result = subprocess.run(cmd, check=True, capture_output=True, text=True, cwd=str(output_dir))
                    
                    if pdf_file.exists():
                        file_size = pdf_file.stat().st_size
                        conversion_results['conversions']['pdf'] = {
                            'status': 'success',
                            'file': str(pdf_file.relative_to(self.workspace_root)),
                            'size': file_size,
                            'size_kb': f"{file_size / 1024:.1f} KB"
                        }
                        print_current(f"✅ PDF document conversion successful: {pdf_file.name} ({file_size / 1024:.1f} KB)")
                    else:
                        conversion_results['conversions']['pdf'] = {
                            'status': 'failed',
                            'error': 'PDF file not generated'
                        }
                        print_current(f"❌ PDF document conversion failed: File not generated")
                        
            except subprocess.CalledProcessError as e:
                conversion_results['conversions']['pdf'] = {
                    'status': 'failed',
                    'error': f'PDF conversion failed: {e.stderr}'
                }
                print_current(f"❌ PDF document conversion failed: {e.stderr}")
            except Exception as e:
                conversion_results['conversions']['pdf'] = {
                    'status': 'failed',
                    'error': f'PDF conversion exception: {str(e)}'
                }
                print_current(f"❌ PDF document conversion exception: {str(e)}")
            
            # Check conversion results
            successful_conversions = sum(1 for conv in conversion_results['conversions'].values() 
                                       if conv.get('status') == 'success')
            total_conversions = len(conversion_results['conversions'])
            
            if successful_conversions > 0:
                print_current(f"📊 Conversion completed: {successful_conversions}/{total_conversions} formats converted successfully")
            else:
                print_current(f"⚠️ All format conversions failed")
                conversion_results['status'] = 'partial_failure'
            
            return conversion_results
            
        except Exception as e:
            print_current(f"❌ Error occurred during Markdown conversion: {str(e)}")
            return {
                'status': 'failed',
                'markdown_file': target_file,
                'error': str(e),
                'message': f'Error occurred during conversion: {str(e)}'
            }

    def _generate_fallback_word_document(self, md_path, output_dir, conversion_results):
        """
        Generate a Word document as fallback when PDF engines are not available
        
        Args:
            md_path: Path object of the markdown file
            output_dir: Path object of the output directory
            conversion_results: Dict to store conversion results
        """
        try:
            # Generate fallback Word filename with suffix to indicate it's a PDF fallback
            base_name = md_path.stem
            fallback_word_file = output_dir / f"{base_name}_pdf_fallback.docx"
            
            print_current(f"📄 Generating fallback Word document: {fallback_word_file.name}")
            
            # Use pandoc to convert to Word with enhanced formatting for PDF-like appearance
            cmd = [
                'pandoc',
                md_path.name,
                '-o', fallback_word_file.name,
                '--from', 'markdown',
                '--to', 'docx',
                '--toc',  # Include table of contents
                '--highlight-style=tango',  # Code highlighting
                '--reference-doc=' if self._check_reference_doc_exists() else ''
            ]
            
            # Remove empty reference-doc parameter if no reference document exists
            cmd = [arg for arg in cmd if arg]
            
            # Execute command in markdown file directory
            result = subprocess.run(cmd, check=True, capture_output=True, text=True, cwd=str(output_dir))
            
            if fallback_word_file.exists():
                file_size = fallback_word_file.stat().st_size
                conversion_results['conversions']['pdf_fallback_word'] = {
                    'status': 'success',
                    'file': str(fallback_word_file.relative_to(self.workspace_root)),
                    'size': file_size,
                    'size_kb': f"{file_size / 1024:.1f} KB",
                    'type': 'word_document',
                    'message': 'Generated as PDF fallback due to missing LaTeX/PDF engines'
                }
                print_current(f"✅ Fallback Word document generated successfully: {fallback_word_file.name} ({file_size / 1024:.1f} KB)")
                print_current(f"💡 Tip: Install xelatex, lualatex, pdflatex, wkhtmltopdf, or weasyprint for PDF generation")
            else:
                conversion_results['conversions']['pdf_fallback_word'] = {
                    'status': 'failed',
                    'error': 'Fallback Word document not generated'
                }
                print_current(f"❌ Fallback Word document generation failed: File not created")
                
        except subprocess.CalledProcessError as e:
            conversion_results['conversions']['pdf_fallback_word'] = {
                'status': 'failed',
                'error': f'Fallback Word generation failed: {e.stderr}'
            }
            print_current(f"❌ Fallback Word document generation failed: {e.stderr}")
        except Exception as e:
            conversion_results['conversions']['pdf_fallback_word'] = {
                'status': 'failed',
                'error': f'Fallback Word generation exception: {str(e)}'
            }
            print_current(f"❌ Fallback Word document generation exception: {str(e)}")

    def _check_reference_doc_exists(self):
        """Check if a reference Word document exists for styling"""
        # This could be enhanced to look for a reference document in the utils directory
        # For now, return False to use default Word styling
        return False

 