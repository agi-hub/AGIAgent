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
import glob
import re
from typing import List, Dict, Any, Optional


class FileSystemTools:
    def __init__(self, workspace_root: str = None):
        """Initialize the FileSystemTools with a workspace root directory."""
        self.workspace_root = workspace_root or os.getcwd()
        self.last_edit = None
    
    def _resolve_path(self, path: str) -> str:
        """Resolve a path relative to the workspace root."""
        if os.path.isabs(path):
            return path
        return os.path.join(self.workspace_root, path)
    
    def read_file(self, target_file: str, should_read_entire_file: bool = False, 
                 start_line_one_indexed: int = None, end_line_one_indexed_inclusive: int = None,
                 **kwargs) -> Dict[str, Any]:
        """
        Read the contents of a file.
        """
        # Ignore additional parameters
        if kwargs:
            print(f"⚠️  Ignoring additional parameters: {list(kwargs.keys())}")
        
        print(f"🎯 Requested to read file: {target_file}")
        print(f"📂 Current working directory: {self.workspace_root}")
        
        file_path = self._resolve_path(target_file)
        
        if not os.path.exists(file_path):
            print(f"❌ File does not exist: {file_path}")
            try:
                current_dir_files = os.listdir(self.workspace_root)
                print(f"📁 Files in current directory: {current_dir_files}")
            except Exception as e:
                print(f"⚠️ Cannot list current directory: {e}")
            
            return {
                'file': target_file,
                'error': f'File not found: {file_path}',
                'workspace_root': self.workspace_root,
                'resolved_path': file_path,
                'exists': False
            }
        
        if not os.path.isfile(file_path):
            print(f"❌ Path is not a file: {file_path}")
            return {
                'file': target_file,
                'error': f'Path is not a file: {file_path}',
                'workspace_root': self.workspace_root,
                'resolved_path': file_path,
                'exists': True,
                'is_file': False
            }
        
        try:
            print(f"✅ File exists, starting to read: {file_path}")
            with open(file_path, 'r', encoding='utf-8') as f:
                all_lines = f.readlines()
            
            total_lines = len(all_lines)
            print(f"📊 Total lines in file: {total_lines}")
            
            if should_read_entire_file:
                max_entire_lines = 500
                if total_lines <= max_entire_lines:
                    content = ''.join(all_lines)
                    print(f"📄 Read entire file, content length: {len(content)} characters")
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
                    print(f"📄 Read entire file (truncated), showing first {max_entire_lines} lines of {total_lines}")
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
                
                print(f"📄 Read partial file: lines {start_line_one_indexed}-{end_line_one_indexed_inclusive} (actual: {start_idx+1}-{end_idx})")
                
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
            print(f"❌ File encoding error: {e}")
            return {
                'file': target_file,
                'error': f'Unicode decode error: {str(e)}',
                'resolved_path': file_path,
                'status': 'encoding_error'
            }
        except Exception as e:
            print(f"❌ Error occurred while reading file: {e}")
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
            print(f"⚠️  Ignoring additional parameters: {list(kwargs.keys())}")
        
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

    def grep_search(self, query: str, include_pattern: str = None, 
                   exclude_pattern: str = None, case_sensitive: bool = False,
                   **kwargs) -> Dict[str, Any]:
        """
        Run a text-based regex search over files.
        """
        # Ignore additional parameters
        if kwargs:
            print(f"⚠️  Ignoring additional parameters: {list(kwargs.keys())}")
        
        print(f"Searching for: {query}")
        
        results = []
        max_results = 50
        
        for root, _, files in os.walk(self.workspace_root):
            for file in files:
                file_path = os.path.join(root, file)
                rel_path = os.path.relpath(file_path, self.workspace_root)
                
                if include_pattern and not glob.fnmatch.fnmatch(rel_path, include_pattern):
                    continue
                if exclude_pattern and glob.fnmatch.fnmatch(rel_path, exclude_pattern):
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
                                
                                if len(results) >= max_results:
                                    break
                except Exception:
                    continue
                
                if len(results) >= max_results:
                    break
        
        return {
            'query': query,
            'results': results,
            'total_matches': len(results),
            'max_results': max_results,
            'truncated': len(results) >= max_results
        }

    def edit_file(self, target_file: str, code_edit: str, instructions: str = None, 
                  edit_mode: str = "full_replace", start_line: int = None, end_line: int = None, 
                  insert_position: int = None, **kwargs) -> Dict[str, Any]:
        """
        Edit a file or create a new file with multiple editing modes.
        
        Args:
            target_file: Path to the file to edit
            code_edit: The code/text content to add or edit
            instructions: Optional instructions for the edit
            edit_mode: Editing mode - "auto", "replace_lines", "insert_lines", "append", "full_replace"
            start_line: Starting line number for replace_lines mode (1-indexed, inclusive)
            end_line: Ending line number for replace_lines mode (1-indexed, inclusive)
            insert_position: Line number to insert before for insert_lines mode (1-indexed)
        """
        # Check for dummy placeholder file created by hallucination detection
        if target_file == "dummy_file_placeholder.txt" or target_file.endswith("/dummy_file_placeholder.txt"):
            print(f"🚨 HALLUCINATION PREVENTION: Detected dummy placeholder file '{target_file}' - skipping actual file operation")
            return {
                'file': target_file,
                'status': 'skipped',
                'error': 'Dummy placeholder file detected - hallucination prevention active',
                'hallucination_prevention': True
            }
        
        print(f"Editing file: {target_file}")
        if instructions:
            print(f"Instructions: {instructions}")
        else:
            print("Instructions: (not provided)")
        
        # Default to append mode if auto mode is set for safety
        if edit_mode == "auto":
            edit_mode = "append"
            print("📝 Auto mode detected - defaulting to 'append' mode for safety")
        
        print(f"📝 Edit mode: {edit_mode}")
        if edit_mode == "replace_lines" and start_line and end_line:
            print(f"🎯 Replace lines {start_line}-{end_line}")
        elif edit_mode == "insert_lines" and insert_position:
            print(f"📍 Insert at line {insert_position}")
        
        # Ignore additional parameters
        if kwargs:
            print(f"⚠️  Ignoring additional parameters: {list(kwargs.keys())}")
        
        file_path = self._resolve_path(target_file)
        file_exists = os.path.exists(file_path)
        
        # Clean markdown code block markers from code_edit
        cleaned_code_edit = self._clean_markdown_markers(code_edit)
        
        # Auto-correct HTML entities in code_edit
        cleaned_code_edit = self._fix_html_entities(cleaned_code_edit)
        
        self.last_edit = {
            'target_file': target_file,
            'instructions': instructions,
            'code_edit': cleaned_code_edit,
            'edit_mode': edit_mode,
            'start_line': start_line,
            'end_line': end_line,
            'insert_position': insert_position
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
                
                # Safety check for unintentional content loss
                if self._is_risky_edit(original_content, cleaned_code_edit, edit_mode, file_exists):
                    return {
                        'file': target_file,
                        'status': 'safety_blocked',
                        'error': 'Edit blocked: Risk of content loss detected. Use specific edit_mode or read file first.',
                        'safety_suggestion': 'Consider using edit_mode="append", "insert_lines", or "replace_lines" instead of "auto"',
                        'original_length': len(original_content),
                        'edit_mode': edit_mode
                    }
            
            # Process edit based on mode
            new_content = self._process_edit_by_mode(
                original_content, cleaned_code_edit, edit_mode, target_file,
                start_line, end_line, insert_position
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
                'edit_mode': edit_mode
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
            print(f"🧹 Found markdown code block start marker")
        
        # Check for end marker (```) at the end
        end_marker_pattern = r'\n```\s*$'
        if re.search(end_marker_pattern, code_content):
            has_end_marker = True
            print(f"🧹 Found markdown code block end marker")
        
        # If no markers found, return original content to preserve exact formatting
        if not has_start_marker and not has_end_marker:
            return original_content
        
        # Remove markers using regex to preserve internal formatting
        cleaned_content = original_content
        
        if has_start_marker:
            # Remove start marker (```[language]\n)
            cleaned_content = re.sub(start_marker_pattern, '', cleaned_content)
            print(f"🧹 Removed markdown code block start marker")
        
        if has_end_marker:
            # Remove end marker (\n```)
            cleaned_content = re.sub(end_marker_pattern, '', cleaned_content)
            print(f"🧹 Removed markdown code block end marker")
        
        print(f"✅ Cleaned markdown markers while preserving formatting")
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
                print(f"🔧 Auto-corrected HTML entities: {', '.join(corrections)}")
            else:
                print(f"🔧 Auto-corrected HTML entities (various types found)")
        
        return code_content

    def _process_edit_by_mode(self, original_content: str, code_edit: str, edit_mode: str, 
                             target_file: str, start_line: int = None, end_line: int = None, 
                             insert_position: int = None) -> str:
        """
        Process the edit based on the specified mode.
        
        Args:
            original_content: Original file content
            code_edit: New content to add/replace
            edit_mode: The editing mode to use
            target_file: Target file path (for legacy mode)
            start_line: Starting line number (1-indexed, inclusive)
            end_line: Ending line number (1-indexed, inclusive)
            insert_position: Line number to insert before (1-indexed)
        
        Returns:
            The new file content after applying the edit
        """
        if edit_mode == "replace_lines":
            return self._replace_lines(original_content, code_edit, start_line, end_line)
        elif edit_mode == "insert_lines":
            return self._insert_lines(original_content, code_edit, insert_position)
        elif edit_mode == "append":
            return self._append_content(original_content, code_edit)
        elif edit_mode == "full_replace":
            print("🔄 Full file replacement mode")
            return code_edit
        else:  # "auto" mode - use legacy logic
            return self._process_code_edit(original_content, code_edit, target_file)

    def _replace_lines(self, content: str, new_content: str, start_line: int, end_line: int) -> str:
        """
        Replace specified line range with new content.
        
        Args:
            content: Original file content
            new_content: Content to replace with
            start_line: Starting line number (1-indexed, inclusive)
            end_line: Ending line number (1-indexed, inclusive)
        
        Returns:
            Updated content with lines replaced
        """
        if start_line is None or end_line is None:
            raise ValueError("start_line and end_line must be specified for replace_lines mode")
        
        lines = content.split('\n')
        new_lines = new_content.split('\n')
        
        # Convert to 0-indexed
        start_idx = start_line - 1
        end_idx = end_line - 1
        
        # Validate line range
        if start_idx < 0:
            raise ValueError(f"start_line must be >= 1, got {start_line}")
        if end_idx >= len(lines):
            raise ValueError(f"end_line {end_line} exceeds file length ({len(lines)} lines)")
        if start_idx > end_idx:
            raise ValueError(f"start_line ({start_line}) must be <= end_line ({end_line})")
        
        print(f"🔧 Replacing lines {start_line}-{end_line} ({end_idx - start_idx + 1} lines) with {len(new_lines)} new lines")
        
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
        
        print(f"📍 Inserting {len(new_lines)} lines at position {insert_position}")
        
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
            print("📝 Creating new file with append content")
            return clean_content
        
        # Ensure there's a newline before appending if the file doesn't end with one
        if content and not content.endswith('\n'):
            result = content + '\n' + clean_content
        else:
            result = content + clean_content
        
        print(f"➕ Appending {len(clean_content.split(chr(10)))} lines to end of file")
        return result

    def _process_code_edit(self, original_content: str, code_edit: str, target_file: str) -> str:
        """
        Process the code edit and apply it to the original content.
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
            print("⚠️ No existing code marker found - treating as full content replacement")
            return code_edit
        
        edit_parts = code_edit.split(used_pattern)
        
        if len(edit_parts) == 1:
            return code_edit
        
        print(f"🔧 Found existing code marker: {used_pattern}")
        print(f"📝 Edit split into {len(edit_parts)} parts")
        
        original_lines = original_content.split('\n')
        
        if len(edit_parts) == 2:
            before_edit = edit_parts[0]
            after_edit = edit_parts[1]
            
            return self._apply_simple_edit(original_lines, before_edit, after_edit)
        
        elif len(edit_parts) > 2:
            return self._apply_complex_edit(original_lines, edit_parts, used_pattern)
        
        else:
            return code_edit

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

    def _apply_simple_edit(self, original_lines: List[str], before_edit: str, after_edit: str) -> str:
        """
        Apply a simple edit with one existing code marker (simplified logic).
        """
        before_lines = before_edit.strip().split('\n') if before_edit.strip() else []
        after_lines = after_edit.strip().split('\n') if after_edit.strip() else []
        
        if not before_lines and not after_lines:
            return '\n'.join(original_lines)
        
        # Simplified logic: before content + original content + after content
        if before_lines and after_lines:
            # Content before and after: insert in middle
            mid_point = len(original_lines) // 2
            new_lines = before_lines + original_lines[mid_point:] + after_lines
        elif before_lines:
            # Only before content: prepend
            new_lines = before_lines + original_lines
        else:
            # Only after content: append
            new_lines = original_lines + after_lines
        
        return '\n'.join(new_lines)

    def _apply_complex_edit(self, original_lines: List[str], edit_parts: List[str], marker: str) -> str:
        """
        Apply complex edit with multiple existing code markers (simplified).
        """
        non_empty_parts = [part.strip() for part in edit_parts if part.strip()]
        
        if len(non_empty_parts) == 0:
            return '\n'.join(original_lines)
        elif len(non_empty_parts) == 1:
            return non_empty_parts[0]
        
        # Simplified: interleave edit parts with original content
        result_lines = []
        chunk_size = max(1, len(original_lines) // max(1, len(non_empty_parts) - 1))
        
        for i, part in enumerate(non_empty_parts):
            if part:
                result_lines.extend(part.split('\n'))
            
            # Add original content between parts (except after the last part)
            if i < len(non_empty_parts) - 1:
                start_idx = i * chunk_size
                end_idx = min((i + 1) * chunk_size, len(original_lines))
                if start_idx < len(original_lines):
                    result_lines.extend(original_lines[start_idx:end_idx])
        
        return '\n'.join(result_lines)

    def file_search(self, query: str, **kwargs) -> Dict[str, Any]:
        """
        Fast file search based on fuzzy matching against file path.
        """
        # Ignore additional parameters
        if kwargs:
            print(f"⚠️  Ignoring additional parameters: {list(kwargs.keys())}")
        
        print(f"Searching for file: {query}")
        
        results = []
        max_results = 10
        
        for root, _, files in os.walk(self.workspace_root):
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
            print(f"⚠️  Ignoring additional parameters: {list(kwargs.keys())}")
        
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
        检查编辑操作是否有误删除文件内容的风险
        
        Args:
            original_content: 原始文件内容
            new_content: 新内容
            edit_mode: 编辑模式
            file_exists: 文件是否存在
        
        Returns:
            True if risky, False if safe
        """
        # 新文件或明确的模式不需要检查
        if not file_exists or edit_mode in ["append", "insert_lines", "replace_lines", "full_replace"]:
            return False
        
        # 只检查 auto 模式
        if edit_mode != "auto":
            return False
        
        # 如果原文件有内容但是新内容很短，可能有风险
        if len(original_content.strip()) > 100 and len(new_content.strip()) < 50:
            print("🚨 Safety check: Original file has substantial content but new content is very short")
            return True
        
        # 检查是否包含 existing code 标记
        existing_code_patterns = [
            '// ... existing code ...',
            '# ... existing code ...',
            '/* ... existing code ...',
            '<!-- ... existing code -->',
        ]
        
        has_existing_marker = any(pattern in new_content for pattern in existing_code_patterns)
        
        # 如果没有 existing code 标记，且原文件有内容，可能有风险
        if not has_existing_marker and len(original_content.strip()) > 20:
            print("🚨 Safety check: No existing code marker found in auto mode with existing content")
            return True
        
        return False

 