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

    def edit_file(self, target_file: str, code_edit: str, instructions: str = None, append_mode: bool = False, **kwargs) -> Dict[str, Any]:
        """
        Edit a file or create a new file.
        
        Args:
            target_file: Path to the file to edit
            code_edit: The code/text content to add or edit
            instructions: Optional instructions for the edit
            append_mode: If True, append content to the end of the file instead of replacing/editing
        """
        print(f"Editing file: {target_file}")
        if instructions:
            print(f"Instructions: {instructions}")
        else:
            print("Instructions: (not provided)")
        
        if append_mode:
            print("🔗 Append mode enabled - content will be added to the end of the file")
        
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
            'append_mode': append_mode
        }
        
        try:
            if append_mode:
                # Append mode: add content to the end of the file
                if file_exists:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        existing_content = f.read()
                    
                    # Remove any existing code markers from the append content
                    append_content = cleaned_code_edit.replace('// ... existing code ...', '').strip()
                    append_content = append_content.replace('# ... existing code ...', '').strip()
                    append_content = append_content.replace('/* ... existing code ...', '').strip()
                    append_content = append_content.replace('<!-- ... existing code ...', '').strip()
                    
                    # Ensure there's a newline before appending if the file doesn't end with one
                    if existing_content and not existing_content.endswith('\n'):
                        new_content = existing_content + '\n' + append_content
                    else:
                        new_content = existing_content + append_content
                    
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(new_content)
                    
                    return {
                        'file': target_file,
                        'status': 'appended',
                        'action': 'appended',
                        'append_mode': True
                    }
                else:
                    # File doesn't exist, create it with the append content
                    dir_path = os.path.dirname(file_path)
                    if dir_path:
                        os.makedirs(dir_path, exist_ok=True)
                    
                    clean_content = cleaned_code_edit.replace('// ... existing code ...', '').strip()
                    
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(clean_content)
                    
                    return {
                        'file': target_file,
                        'status': 'created',
                        'action': 'created',
                        'append_mode': True
                    }
            else:
                # Original edit mode
                if file_exists:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        original_content = f.read()
                    
                    new_content = self._process_code_edit(original_content, cleaned_code_edit, target_file)
                    
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(new_content)
                    
                    return {
                        'file': target_file,
                        'status': 'edited',
                        'action': 'modified',
                        'append_mode': False
                    }
                else:
                    dir_path = os.path.dirname(file_path)
                    if dir_path:
                        os.makedirs(dir_path, exist_ok=True)
                    
                    clean_content = cleaned_code_edit.replace('// ... existing code ...', '').strip()
                    
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(clean_content)
                    
                    return {
                        'file': target_file,
                        'status': 'created',
                        'action': 'created',
                        'append_mode': False
                    }
        except Exception as e: 
            return {
                'file': target_file,
                'status': 'error',
                'error': str(e),
                'append_mode': append_mode
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

    def _process_code_edit(self, original_content: str, code_edit: str, target_file: str) -> str:
        """
        Process the code edit and apply it to the original content.
        """
        comment_markers = self._get_comment_markers(target_file)
        
        existing_code_patterns = [
            '// ... existing code ...',
            '# ... existing code ...',
            '/* ... existing code ...',
            '<!-- ... existing code ...',
            '# existing code continues...',
            '// existing code continues...',
            '... existing code ...',
        ]
        
        used_pattern = None
        for pattern in existing_code_patterns:
            if pattern in code_edit:
                used_pattern = pattern
                break
        
        if not used_pattern:
            print("⚠️ No existing code marker found, will perform full file replacement")
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
        Apply a simple edit with one existing code marker.
        """
        before_lines = before_edit.strip().split('\n') if before_edit.strip() else []
        after_lines = after_edit.strip().split('\n') if after_edit.strip() else []
        
        if not before_lines and not after_lines:
            return '\n'.join(original_lines)
        
        if before_lines and not after_lines:
            replace_count = min(len(before_lines), len(original_lines))
            
            match_end = self._find_context_match(original_lines, before_lines)
            if match_end is not None:
                replace_count = match_end
            
            new_lines = before_lines + original_lines[replace_count:]
            return '\n'.join(new_lines)
        
        if after_lines and not before_lines:
            insert_point = len(original_lines)
            
            new_lines = original_lines + after_lines
            return '\n'.join(new_lines)
        
        split_point = self._find_best_split_point(original_lines, before_lines, after_lines)
        
        if split_point is not None:
            new_lines = before_lines + original_lines[split_point:] + after_lines
        else:
            new_lines = before_lines + original_lines + after_lines
        
        return '\n'.join(new_lines)

    def _find_context_match(self, original_lines: List[str], edit_lines: List[str]) -> Optional[int]:
        """
        Find where the edit lines match in the original content.
        """
        if not edit_lines or not original_lines:
            return None
        
        for i in range(len(original_lines) - len(edit_lines) + 1):
            matches = 0
            for j, edit_line in enumerate(edit_lines):
                if i + j < len(original_lines):
                    orig_line = original_lines[i + j].strip()
                    edit_line_clean = edit_line.strip()
                    
                    if orig_line == edit_line_clean or (edit_line_clean and edit_line_clean in orig_line):
                        matches += 1
            
            if matches >= len(edit_lines) * 0.5:
                return i + len(edit_lines)
        
        for i in range(min(len(edit_lines), len(original_lines))):
            if edit_lines[i].strip() and edit_lines[i].strip() in original_lines[i]:
                continue
            else:
                return i if i > 0 else len(edit_lines)
        
        return len(edit_lines)

    def _find_best_split_point(self, original_lines: List[str], before_lines: List[str], after_lines: List[str]) -> Optional[int]:
        """
        Find the best point to split the original content for insertion.
        """
        match_end = self._find_context_match(original_lines, before_lines)
        if match_end is not None:
            return match_end
        
        return len(original_lines) // 2

    def _apply_complex_edit(self, original_lines: List[str], edit_parts: List[str], marker: str) -> str:
        """
        Apply complex edit with multiple existing code markers.
        """
        non_empty_parts = [part.strip() for part in edit_parts if part.strip()]
        
        if len(non_empty_parts) == 1:
            return non_empty_parts[0]
        
        result_lines = []
        original_copy = original_lines.copy()
        
        for i, part in enumerate(non_empty_parts):
            part_lines = part.split('\n')
            result_lines.extend(part_lines)
            
            if i < len(non_empty_parts) - 1 and original_copy:
                chunk_size = max(1, len(original_copy) // (len(non_empty_parts) - i))
                chunk = original_copy[:chunk_size]
                original_copy = original_copy[chunk_size:]
                result_lines.extend(chunk)
        
        if original_copy:
            result_lines.extend(original_copy)
        
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

    def reapply(self, target_file: str, **kwargs) -> Dict[str, Any]:
        """
        Reapply the last edit to the specified file.
        """
        print(f"Reapplying edit to: {target_file}")
        
        # Ignore additional parameters
        if kwargs:
            print(f"⚠️  Ignoring additional parameters: {list(kwargs.keys())}")
        
        if not self.last_edit or self.last_edit['target_file'] != target_file:
            return {
                'file': target_file,
                'status': 'error',
                'error': 'No previous edit found for this file'
            }
        
        return self.edit_file(
            target_file=target_file,
            instructions=f"REAPPLY: {self.last_edit['instructions']}",
            code_edit=self.last_edit['code_edit'],
            append_mode=self.last_edit.get('append_mode', False)
        ) 