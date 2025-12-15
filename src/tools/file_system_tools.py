#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from .print_system import print_system, print_current, print_system, print_debug
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
import glob
import re
import fnmatch
import threading
import time
import shutil
import subprocess
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple, Union
from src.config_loader import load_config
from src.tools.agent_context import get_current_log_dir

# Import Mermaid processor for handling charts in markdown files

def remove_emoji_from_text(text):
    """
    从文本中删除emoji字符
    保留普通的中文、英文、数字和标点符号
    """
    if not text:
        return text

    # 使用正则表达式删除emoji
    # 匹配各种emoji Unicode范围
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # 表情符号
        "\U0001F300-\U0001F5FF"  # 杂项符号和象形文字
        "\U0001F680-\U0001F6FF"  # 交通和地图符号
        "\U0001F700-\U0001F77F"  # 炼金术符号
        "\U0001F780-\U0001F7FF"  # 几何形状扩展
        "\U0001F800-\U0001F8FF"  # 补充箭头-C
        "\U0001F900-\U0001F9FF"  # 补充符号和象形文字
        "\U0001FA00-\U0001FA6F"  # 棋牌符号
        "\U0001FA70-\U0001FAFF"  # 符号和象形文字扩展-A
        "\U00002600-\U000026FF"  # 杂项符号
        "\U00002700-\U000027BF"  # 装饰符号
        "\U0001F1E6-\U0001F1FF"  # 地区指示符号（国旗）
        "\U00002B50-\U00002B55"  # 星星等
        "\U0000FE00-\U0000FE0F"  # 变体选择器
        "]+",
        flags=re.UNICODE
    )

    # 删除emoji
    text_without_emoji = emoji_pattern.sub('', text)

    # 清理空格和换行符
    text_without_emoji = re.sub(r'[ \t]+', ' ', text_without_emoji)  # 只合并空格和tab
    text_without_emoji = re.sub(r' *\n *', '\n', text_without_emoji)  # 清理换行符前后的空格
    text_without_emoji = re.sub(r'\n{3,}', '\n\n', text_without_emoji)  # 限制连续换行符数量

    return text_without_emoji.strip()


def create_emoji_free_markdown(input_file):
    """
    创建一个删除了emoji的临时markdown文件

    Args:
        input_file: 输入的markdown文件路径

    Returns:
        str: 临时文件路径，如果失败返回None
    """
    import tempfile

    try:
        # 读取原始markdown文件
        with open(input_file, 'r', encoding='utf-8') as f:
            content = f.read()

        # 删除emoji
        cleaned_content = remove_emoji_from_text(content)

        # 如果内容没有变化，就不需要创建临时文件
        if cleaned_content == content:
            print_debug("📝 No emoji found in markdown, using original file")
            return None

        # 在输入文件所在目录创建临时文件，这样pandoc可以找到它
        input_dir = os.path.dirname(input_file)
        temp_fd, temp_path = tempfile.mkstemp(suffix='.md', prefix='emoji_free_', dir=input_dir)

        try:
            # 写入清理后的内容
            with os.fdopen(temp_fd, 'w', encoding='utf-8') as temp_file:
                temp_file.write(cleaned_content)

            print_debug(f"📝 Created emoji-free temporary markdown: {temp_path}")
            return temp_path

        except Exception as e:
            # 如果写入失败，关闭并删除临时文件
            os.close(temp_fd)
            if os.path.exists(temp_path):
                os.remove(temp_path)
            raise e

    except Exception as e:
        print_debug(f"❌ Error creating emoji-free markdown: {e}")
        return None
# ========================================
# 🚀 延迟导入优化：mermaid_processor 延迟加载
# ========================================
# mermaid_processor 用于处理 Mermaid 图表，只在实际需要时才加载
# 避免启动时加载，节省约 3.2秒

_mermaid_processor = None
MERMAID_PROCESSOR_AVAILABLE = None  # 未初始化状态

def _get_mermaid_processor():
    """延迟获取 Mermaid 处理器（只在需要时导入）"""
    global _mermaid_processor, MERMAID_PROCESSOR_AVAILABLE
    
    if _mermaid_processor is not None:
        return _mermaid_processor
    
    if MERMAID_PROCESSOR_AVAILABLE is False:
        return None
    
    try:
        from .mermaid_processor import MermaidProcessor
        _mermaid_processor = MermaidProcessor(silent_init=True)
        MERMAID_PROCESSOR_AVAILABLE = True
        return _mermaid_processor
    except ImportError:
        print_debug("⚠️ Mermaid processor not available")
        MERMAID_PROCESSOR_AVAILABLE = False
        return None

# Import SVG processor for handling SVG code blocks in markdown files
try:
    from .svg_processor import svg_processor
    SVG_PROCESSOR_AVAILABLE = True
except ImportError:
    print_debug("⚠️ SVG processor not available")
    SVG_PROCESSOR_AVAILABLE = False


class FileSystemTools:
    def __init__(self, workspace_root: Optional[str] = None):
        """Initialize the FileSystemTools with a workspace root directory."""
        self.workspace_root = workspace_root or os.getcwd()
        self.last_edit = None
        self.snapshot_dir = "file_snapshot"
        self._check_system_grep_available()
        
        # Update SVG processor workspace root if available
        if SVG_PROCESSOR_AVAILABLE:
            svg_processor.set_workspace_root(self.workspace_root)
    
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
                    print_debug(f"⚠️ Too many snapshots for file {target_file}, skipping snapshot creation")
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
            
            return True
            
        except Exception as e:
            print_debug(f"⚠️ Failed to create snapshot for {target_file}: {e}")
            return False

    def _save_original_markdown_to_logs(self, file_path: str, target_file: str) -> Optional[str]:
        """
        Save the original markdown file to logs directory before image processing.
        
        Args:
            file_path: Absolute path to the file to save
            target_file: Relative path to the file (used for naming)
            
        Returns:
            Path to the saved file if successful, None otherwise
        """
        if not os.path.exists(file_path) or not os.path.isfile(file_path):
            return None
        
        try:
            # Get logs directory
            logs_dir = get_current_log_dir()
            if not logs_dir:
                # Fallback: use workspace_root parent directory + logs
                parent_dir = os.path.dirname(self.workspace_root)
                logs_dir = os.path.join(parent_dir, "logs")
            
            # Create logs directory if it doesn't exist
            os.makedirs(logs_dir, exist_ok=True)
            
            # Create a subdirectory for original markdown files
            original_md_dir = os.path.join(logs_dir, "original_markdown")
            os.makedirs(original_md_dir, exist_ok=True)
            
            # Get file name and extension
            file_name = os.path.basename(target_file)
            name_parts = file_name.rsplit('.', 1)
            if len(name_parts) == 2:
                base_name, extension = name_parts
                extension = '.' + extension
            else:
                base_name = file_name
                extension = '.md'  # Default to .md if no extension
            
            # Generate timestamp for unique filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # Include milliseconds
            
            # Create filename with timestamp
            saved_filename = f"{base_name}_original_{timestamp}{extension}"
            saved_path = os.path.join(original_md_dir, saved_filename)
            
            # Read and save the original content
            with open(file_path, 'r', encoding='utf-8') as f:
                original_content = f.read()
            
            with open(saved_path, 'w', encoding='utf-8') as f:
                f.write(original_content)
            
            # Make relative path for display
            relative_saved_path = os.path.relpath(saved_path, logs_dir)
            print_debug(f"💾 Saved original markdown file to logs: {relative_saved_path}")
            
            return saved_path
            
        except Exception as e:
            print_debug(f"⚠️ Failed to save original markdown to logs for {target_file}: {e}")
            return None

    def read_file(self, target_file: str, should_read_entire_file: bool = False, 
                 start_line_one_indexed: Optional[int] = None, end_line_one_indexed_inclusive: Optional[int] = None,
                 **kwargs) -> Dict[str, Any]:
        """
        Read the contents of a file.
        """
        # Ignore additional parameters
        if kwargs:
            print_debug(f"⚠️  Ignoring additional parameters: {list(kwargs.keys())}")
    
        
        file_path = self._resolve_path(target_file)
        
        if not os.path.exists(file_path):
            print_debug(f"❌ File does not exist: {file_path}")
            try:
                current_dir_files = os.listdir(self.workspace_root)
                print_debug(f"📁 Files in current directory: {current_dir_files}")
            except Exception as e:
                print_debug(f"⚠️ Cannot list current directory: {e}")
            
            return {
                'status': 'failed',
                'file': target_file,
                'error': f'File not found: {file_path}',
                'workspace_root': self.workspace_root,
                'resolved_path': file_path,
                'exists': False
            }
        
        if not os.path.isfile(file_path):
            print_debug(f"❌ Path is not a file: {file_path}")
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
                    print_debug(f"📄 Read entire file (truncated), showing first {max_entire_lines} lines of {total_lines}")
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
                
                print_debug(f"📄 Read partial file: lines {start_line_one_indexed}-{end_line_one_indexed_inclusive} (actual: {start_idx+1}-{end_idx})")
                
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
            print_debug(f"❌ File encoding error: {e}")
            return {
                'status': 'failed',
                'file': target_file,
                'error': f'Unicode decode error: {str(e)}',
                'resolved_path': file_path
            }
        except Exception as e:
            print_debug(f"❌ Error occurred while reading file: {e}")
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
            print_debug(f"⚠️  Ignoring additional parameters: {list(kwargs.keys())}")
        
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
            print_debug(f"⚠️  Ignoring additional parameters: {list(kwargs.keys())}")
        
        # Intelligent query optimization for LLM-generated complex queries
        optimized_result, should_split = self._optimize_query_for_performance(query)
        
        if should_split:
            print_debug(f"🔧 Complex query detected, optimizing for better performance...")
            # Type assertion: we know optimized_result is a list when should_split is True
            query_groups = optimized_result if isinstance(optimized_result, list) else [str(optimized_result)]
            return self._execute_split_search(query_groups, include_pattern or "", exclude_pattern or "", case_sensitive, max_results)
        else:
            print_debug(f"Searching for: {optimized_result}")
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
            print_debug(f"🚨 HALLUCINATION PREVENTION: Detected dummy placeholder file '{target_file}' - skipping actual file operation")
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
            print_debug(f"⚠️  Ignoring additional parameters: {list(kwargs.keys())}")
        
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
                original_content, code_edit, edit_mode, target_file, old_code
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
                        print_debug(f"📝 Applied bullet formatting preprocessing to markdown file")
                except Exception as e:
                    print_debug(f"⚠️ Error during bullet formatting preprocessing: {e}")
            
            # Save original markdown file to logs before image processing
            original_md_saved_path = None
            if target_file.lower().endswith('.md'):
                try:
                    original_md_saved_path = self._save_original_markdown_to_logs(file_path, target_file)
                except Exception as e:
                    print_debug(f"⚠️ Error saving original markdown to logs: {e}")
            
            # Process Mermaid charts if this is a markdown file
            mermaid_result = None
            if target_file.lower().endswith('.md'):
                # 🚀 延迟加载：只在需要处理 Mermaid 图表时才加载处理器
                processor = _get_mermaid_processor()
                if processor:
                    try:
                        if processor.has_mermaid_charts(file_path):
                            print_debug(f"🎨 Detected Mermaid charts in markdown file, processing...")
                            mermaid_result = processor.process_markdown_file(file_path)
                            if mermaid_result['status'] == 'success':
                                print_debug(f"✅ Mermaid processing completed: {mermaid_result['message']}")
                            else:
                                print_debug(f"⚠️ Mermaid processing failed: {mermaid_result.get('message', 'Unknown error')}")
                    except Exception as e:
                        print_debug(f"⚠️ Error during Mermaid processing: {e}")
                        mermaid_result = {
                            'status': 'failed',
                            'error': str(e),
                            'message': f'Mermaid processing error: {e}'
                        }
            
            # Process .mmd files - convert Mermaid code to SVG and PNG
            mermaid_mmd_result = None
            if target_file.lower().endswith('.mmd'):
                # 🚀 延迟加载：只在需要处理 Mermaid 图表时才加载处理器
                processor = _get_mermaid_processor()
                if processor:
                    try:
                        from pathlib import Path
                        print_debug(f"🎨 Detected .mmd file, converting to SVG and PNG...")
                        
                        # Read the mermaid code from the .mmd file
                        with open(file_path, 'r', encoding='utf-8') as f:
                            mermaid_code = f.read().strip()
                        
                        if not mermaid_code:
                            print_debug(f"⚠️ .mmd file is empty, skipping conversion")
                            mermaid_mmd_result = {
                                'status': 'skipped',
                                'message': 'File is empty'
                            }
                        else:
                            # Generate output paths (same directory, same filename, different extensions)
                            mmd_path = Path(file_path)
                            svg_path = mmd_path.with_suffix('.svg')
                            png_path = mmd_path.with_suffix('.png')
                            
                            # Generate SVG and PNG images
                            svg_success, png_success = processor._generate_mermaid_image(mermaid_code, svg_path, png_path)
                            
                            if svg_success and png_success:
                                print_debug(f"✅ Successfully converted .mmd to SVG and PNG")
                                mermaid_mmd_result = {
                                    'status': 'success',
                                    'message': f'Successfully converted to {svg_path.name} and {png_path.name}',
                                    'svg_path': str(svg_path),
                                    'png_path': str(png_path)
                                }
                            elif svg_success:
                                print_debug(f"⚠️ SVG conversion succeeded but PNG conversion failed")
                                mermaid_mmd_result = {
                                    'status': 'partial',
                                    'message': f'SVG conversion succeeded but PNG conversion failed',
                                    'svg_path': str(svg_path)
                                }
                            else:
                                print_debug(f"❌ Failed to convert .mmd file")
                                mermaid_mmd_result = {
                                    'status': 'failed',
                                    'message': 'Failed to convert .mmd file to SVG/PNG'
                                }
                    except Exception as e:
                        print_debug(f"⚠️ Error during .mmd file processing: {e}")
                        mermaid_mmd_result = {
                            'status': 'failed',
                            'error': str(e),
                            'message': f'.mmd processing error: {e}'
                        }
            
            # Process SVG code blocks if this is a markdown file
            svg_result = None
            if target_file.lower().endswith('.md') and SVG_PROCESSOR_AVAILABLE:
                try:
                    if svg_processor.has_svg_blocks(file_path):
                        print_debug(f"🎨 Detected SVG code blocks in markdown file, processing...")
                        svg_result = svg_processor.process_markdown_file(file_path)
                        if svg_result['status'] == 'success':
                            print_debug(f"✅ SVG processing completed: {svg_result['message']}")
                        else:
                            print_debug(f"⚠️ SVG processing failed: {svg_result.get('message', 'Unknown error')}")
                except Exception as e:
                    print_debug(f"⚠️ Error during SVG processing: {e}")
                    svg_result = {
                        'status': 'failed',
                        'error': str(e),
                        'message': f'SVG processing error: {e}'
                    }
            
            # Validate and comment out invalid local image links after Mermaid/SVG processing
            # This is done after image conversion to check all final image references
            if target_file.lower().endswith('.md'):
                try:
                    self._validate_and_comment_invalid_images(file_path)
                except Exception as e:
                    print_debug(f"⚠️ Error validating images: {e}")
            
            # Ensure markdown file ends with proper newlines after image processing
            # This is important because Mermaid/SVG processors may rewrite the file
            if target_file.lower().endswith('.md'):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        final_content = f.read()
                    
                    # Check if file needs newlines at the end
                    if final_content and not final_content.endswith('\n\n'):
                        if final_content.endswith('\n'):
                            final_content += '\n'
                            print_debug(f"📝 Added second newline at end of markdown file after image processing")
                        else:
                            final_content += '\n\n'
                            print_debug(f"📝 Added two newlines at end of markdown file after image processing")
                        
                        # Write the corrected content back
                        with open(file_path, 'w', encoding='utf-8') as f:
                            f.write(final_content)
                except Exception as e:
                    print_debug(f"⚠️ Error ensuring newlines at end of file: {e}")
            
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
            
            # Add .mmd file processing result if applicable
            if mermaid_mmd_result is not None:
                result['mermaid_mmd_processing'] = mermaid_mmd_result
            
            # Add SVG processing result if applicable
            if svg_result is not None:
                result['svg_processing'] = svg_result
            
            # Convert markdown to Word and PDF if this is a markdown file
            # Skip conversion for plan.md as it's an execution plan, not an output document
            conversion_result = None
            target_file_lower = target_file.lower()
            file_basename = os.path.basename(target_file_lower)
            if target_file_lower.endswith('.md') and file_basename != 'plan.md':
                try:
                    conversion_result = self._convert_markdown_to_formats(file_path, target_file)
                    if conversion_result:
                        result['conversion'] = conversion_result
                except Exception as e:
                    print_debug(f"⚠️ Error during markdown conversion: {e}")
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
            print_debug(f"🧹 Found markdown code block start marker")
        
        # Check for end marker (```) at the end
        end_marker_pattern = r'\n```\s*$'
        if re.search(end_marker_pattern, code_content):
            has_end_marker = True
            print_debug(f"🧹 Found markdown code block end marker")
        
        # If no markers found, return original content to preserve exact formatting
        if not has_start_marker and not has_end_marker:
            return original_content
        
        # Remove markers using regex to preserve internal formatting
        cleaned_content = original_content
        
        if has_start_marker:
            # Remove start marker (```[language]\n)
            cleaned_content = re.sub(start_marker_pattern, '', cleaned_content)
            print_debug(f"🧹 Removed markdown code block start marker")
        
        if has_end_marker:
            # Remove end marker (\n```)
            cleaned_content = re.sub(end_marker_pattern, '', cleaned_content)
            print_debug(f"🧹 Removed markdown code block end marker")
        
        print_debug(f"✅ Cleaned markdown markers while preserving formatting")
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
                print_debug(f"🔧 Auto-corrected HTML entities: {', '.join(corrections)}")
            else:
                print_debug(f"🔧 Auto-corrected HTML entities (various types found)")
        
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
            print_debug(f"⚠️ Unknown edit_mode '{edit_mode}', defaulting to lines_replace")
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
        
        print_debug(f"🔧 Replacing lines {start_line_one_indexed}-{end_line_one_indexed_inclusive} ({end_idx - start_idx + 1} lines) with {len(new_lines)} new lines")
        
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
        
        print_debug(f"📍 Inserting {len(new_lines)} lines at position {insert_position}")
        
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
            print_debug("📝 Creating new file with append content")
            return clean_content
        
        # Ensure there's a newline before appending if the file doesn't end with one
        if content and not content.endswith('\n'):
            result = content + '\n' + clean_content
        else:
            result = content + clean_content
        
        print_debug(f"➕ Appending {len(clean_content.split(chr(10)))} lines to end of file")
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
        
        print_debug(f"🎯 Starting precise code replacement")
        print_debug(f"📝 Looking for old code snippet ({len(cleaned_old_code)} chars)")
        print_debug(f"🔄 Will replace with new code snippet ({len(cleaned_code_edit)} chars)")
        
        # Try direct string replacement first
        try:
            return self._apply_precise_replacement(original_content, cleaned_old_code, cleaned_code_edit)
        except ValueError as e:
            # If direct replacement fails due to whitespace differences, try normalized comparison
            if "old_code was not found" in str(e):
                print_debug("Direct replacement failed, trying normalized whitespace comparison...")
                return self._apply_normalized_replacement(original_content, cleaned_old_code, cleaned_code_edit)
            else:
                raise
    
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
        
        print_debug(f"✅ Successfully replaced code snippet (direct replacement)")
        print_debug(f"📊 Content size: {len(original_content)} → {len(new_content)} chars")
        
        return new_content

    def _apply_normalized_replacement(self, original_content: str, old_code: str, new_code: str) -> str:
        """
        Apply replacement using normalized whitespace comparison.
        This handles cases where old_code has different whitespace (spaces/tabs/newlines) than the file.

        Args:
            original_content: The original file content
            old_code: The code snippet to find (may have whitespace differences)
            new_code: The new code snippet to replace with

        Returns:
            The updated file content

        Raises:
            ValueError: If the normalized old_code is not found exactly once
        """
        def normalize_whitespace(text: str) -> str:
            """Normalize whitespace while preserving structure"""
            # Split into lines and strip each line, then rejoin
            lines = text.split('\n')
            normalized_lines = []
            for line in lines:
                # Strip leading/trailing whitespace but preserve indentation structure
                stripped = line.rstrip()
                if stripped:  # Only add non-empty lines
                    normalized_lines.append(stripped)
                elif normalized_lines:  # Add single empty lines between content
                    if not normalized_lines[-1] == '':
                        normalized_lines.append('')
            return '\n'.join(normalized_lines).rstrip()

        # Normalize both old_code and the search in original_content
        normalized_old = normalize_whitespace(old_code)
        normalized_content = normalize_whitespace(original_content)

        print_debug(f"Normalized old_code ({len(normalized_old)} chars): {repr(normalized_old[:100])}...")

        # Find the normalized old_code in normalized content
        old_pos = normalized_content.find(normalized_old)
        if old_pos == -1:
            raise ValueError(
                f"EDIT REJECTED: Even after normalizing whitespace, the old_code was not found in the file. "
                f"This usually means the code structure is significantly different. "
                f"Normalized old_code (first 200 chars): {repr(normalized_old[:200])}"
            )

        # Check if it appears multiple times
        if normalized_content.count(normalized_old) > 1:
            raise ValueError(
                f"EDIT REJECTED: The normalized old_code appears multiple times in the file. "
                f"Please make it more specific to match exactly one location."
            )

        # Now we need to find the corresponding position in the original content
        # This is tricky because normalization changes positions
        # We'll use a different approach: split both into lines and find matching line sequence

        original_lines = original_content.split('\n')
        old_lines = old_code.split('\n')

        # Find the starting line in original content that matches our old_code
        start_line = -1
        for i in range(len(original_lines) - len(old_lines) + 1):
            if self._lines_match(original_lines[i:i+len(old_lines)], old_lines):
                start_line = i
                break

        if start_line == -1:
            raise ValueError(
                f"EDIT REJECTED: Could not find matching line sequence for old_code in original content."
            )

        # Find the end line
        end_line = start_line + len(old_lines) - 1

        # Perform the replacement
        result_lines = original_lines[:start_line] + [new_code] + original_lines[end_line + 1:]
        new_content = '\n'.join(result_lines)

        print_debug(f"✅ Successfully replaced code snippet using normalized comparison")
        print_debug(f"📊 Content size: {len(original_content)} → {len(new_content)} chars")

        return new_content

    def _lines_match(self, original_lines: list, old_lines: list) -> bool:
        """Check if two line sequences match, allowing for whitespace differences"""
        if len(original_lines) != len(old_lines):
            return False

        for orig, old in zip(original_lines, old_lines):
            # Strip both lines and compare
            if orig.rstrip() != old.rstrip():
                return False
        return True

    def file_search(self, query: str, **kwargs) -> Dict[str, Any]:
        """
        Fast file search based on fuzzy matching against file path.
        """
        # Ignore additional parameters
        if kwargs:
            print_debug(f"⚠️  Ignoring additional parameters: {list(kwargs.keys())}")
        
        print_debug(f"Searching for file: {query}")
        
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
            print_debug(f"⚠️  Ignoring additional parameters: {list(kwargs.keys())}")
        
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
            print_debug(f"⚠️ Complex query detected with {or_count + 1} terms")
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
            print_debug(f"🔍 Executing search group {i+1}/{len(query_groups)}: {group_query}")
            
            result = self._execute_single_search(group_query, include_pattern, exclude_pattern, case_sensitive, max_results)
            
            if result['results']:
                all_results.extend(result['results'])
                all_queries.append(group_query)
                
                # Limit total results to prevent overwhelming output
                if max_results is not None and len(all_results) >= max_results:
                    print_debug(f"⚠️ Reached result limit ({max_results}), stopping further searches")
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
            print_debug(f"🔍 Executing system grep: {' '.join(cmd)} (target: {self.workspace_root})")
            result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='ignore', timeout=30)
            
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
            print_debug(f"⚠️ Invalid regex pattern, searching as literal string: {query}")
        
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

    def merge_file(self, file_list: List[str], output_file: str, **kwargs) -> Dict[str, Any]:
        """
        合并多个文件的内容到一个新文件中
        
        Args:
            file_list: 要合并的文件列表，如 ['file1.txt', 'file2.txt']
            output_file: 合并后的输出文件名，如 'merged.txt'
            
        Returns:
            Dictionary containing merge results
        """
        import subprocess
        from pathlib import Path
        
        try:
            # 验证输入参数
            if not file_list or not isinstance(file_list, list):
                return {
                    'status': 'failed',
                    'error': 'file_list must be a non-empty list of filenames',
                    'file_list': file_list,
                    'output_file': output_file
                }
            
            if not output_file:
                return {
                    'status': 'failed',
                    'error': 'output_file must be provided',
                    'file_list': file_list,
                    'output_file': output_file
                }
            
            # 解析文件路径
            resolved_files = []
            missing_files = []
            
            for file_path in file_list:
                resolved_path = self._resolve_path(file_path)
                if Path(resolved_path).exists():
                    resolved_files.append(resolved_path)
                else:
                    missing_files.append(file_path)
            
            if missing_files:
                return {
                    'status': 'failed',
                    'error': f'Following files not found: {missing_files}',
                    'file_list': file_list,
                    'output_file': output_file,
                    'missing_files': missing_files
                }
            
            # 解析输出文件路径
            output_path = self._resolve_path(output_file)
            output_path_obj = Path(output_path)
            
            print_debug(f"🔄 Merging {len(resolved_files)} files into: {output_file}")
            
            # 使用 cat 命令合并文件
            try:
                cmd = ['cat'] + resolved_files
                with open(output_path, 'w', encoding='utf-8') as outfile:
                    result = subprocess.run(cmd, stdout=outfile, stderr=subprocess.PIPE, text=True, encoding='utf-8', errors='ignore', check=True)
                
                if not output_path_obj.exists():
                    return {
                        'status': 'failed',
                        'error': 'Output file was not created',
                        'file_list': file_list,
                        'output_file': output_file
                    }
                
                # 获取合并后文件的信息
                file_size = output_path_obj.stat().st_size
                relative_output_path = str(output_path_obj.relative_to(self.workspace_root))
                
                print_debug(f"✅ Successfully merged files into: {output_file} ({file_size / 1024:.1f} KB)")
                
                result_data = {
                    'status': 'success',
                    'file_list': file_list,
                    'output_file': output_file,
                    'resolved_output_path': relative_output_path,
                    'merged_files_count': len(resolved_files),
                    'output_size': file_size,
                    'output_size_kb': f"{file_size / 1024:.1f} KB"
                }
                
                # 如果输出文件是 markdown 文件，自动转换为 Word 和 PDF
                if output_file.lower().endswith('.md'):
                    print_debug(f"📄 Detected Markdown file, converting to Word and PDF formats...")
                    try:
                        conversion_result = self._convert_markdown_to_formats(output_path, relative_output_path)
                        result_data['format_conversions'] = conversion_result
                        print_debug(f"✅ Format conversion completed for: {output_file}")
                    except Exception as e:
                        result_data['format_conversions'] = {
                            'status': 'failed',
                            'error': f'Format conversion failed: {str(e)}'
                        }
                        print_debug(f"❌ Format conversion failed for {output_file}: {str(e)}")
                
                return result_data
                
            except subprocess.CalledProcessError as e:
                return {
                    'status': 'failed',
                    'error': f'File merge command failed: {e.stderr}',
                    'file_list': file_list,
                    'output_file': output_file
                }
            
        except Exception as e:
            return {
                'status': 'failed',
                'error': f'Merge operation failed: {str(e)}',
                'file_list': file_list,
                'output_file': output_file
            }

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
        
        print_debug(f"📁 Starting document parsing in folder: {resolved_folder}")
        
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
                
                print_debug(f"🔄 Converting: {rel_path}")
                
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
                        
                        print_debug(f"✅ Converted: {rel_path} → {os.path.basename(md_path)}")
                    else:
                        failed_files.append({
                            'file': rel_path,
                            'error': 'No content returned from markitdown conversion'
                        })
                        print_debug(f"❌ Failed to convert: {rel_path} - No content returned")
                        
                except Exception as e:
                    failed_files.append({
                        'file': rel_path,
                        'error': str(e)
                    })
                    print_debug(f"❌ Failed to convert: {rel_path} - {str(e)}")
        
        # Summary
        total_processed = len(processed_files)
        total_failed = len(failed_files)
        total_skipped = len(skipped_files)
        
        print_debug(f"📊 Conversion complete:")
        print_debug(f"   ✅ Processed: {total_processed} files")
        print_debug(f"   ❌ Failed: {total_failed} files")
        print_debug(f"   ⏭️ Skipped: {total_skipped} files")
        
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

    def convert_docs_to_markdown(self, file_path: str) -> Dict[str, Any]:
        """
        Convert various document formats to Markdown using pandoc and pymupdf.

        Supports: docx, xlsx, html, latex, reStructuredText, pptx, pdf
        Uses pandoc for most formats and pymupdf (fitz) for PDF conversion.
        Images are extracted and saved to doc_images folder.

        Args:
            file_path: Path to the document file to convert

        Returns:
            Dictionary with conversion results
        """
        try:
            # Resolve the file path
            resolved_file = self._resolve_path(file_path)

            if not os.path.exists(resolved_file):
                return {
                    'status': 'failed',
                    'error': f'File not found: {resolved_file}',
                    'file': file_path
                }

            if not os.path.isfile(resolved_file):
                return {
                    'status': 'failed',
                    'error': f'Path is not a file: {resolved_file}',
                    'file': file_path
                }

            # Get file extension and base name
            file_ext = os.path.splitext(resolved_file)[1].lower()
            base_name = os.path.splitext(os.path.basename(resolved_file))[0]

            # Define supported formats and their conversion methods
            pandoc_formats = {
                '.docx': 'docx',
                '.html': 'html',
                '.htm': 'html',
                '.tex': 'latex',
                '.rst': 'rst',
                '.pptx': 'pptx'
            }

            # Generate output markdown path
            output_dir = os.path.dirname(resolved_file)
            md_filename = base_name + '.md'
            md_path = os.path.join(output_dir, md_filename)

            print_debug(f"🔄 Converting: {os.path.basename(resolved_file)}")

            if file_ext in pandoc_formats:
                # Use pandoc for conversion
                success, error_msg = self._convert_with_pandoc(resolved_file, md_path, pandoc_formats[file_ext])

            elif file_ext == '.xlsx':
                # Use pandas for Excel to Markdown conversion
                success, error_msg = self._convert_xlsx_with_pandas(resolved_file, md_path)

            elif file_ext == '.pdf':
                # Use pymupdf (fitz) for PDF conversion
                success, error_msg = self._convert_pdf_with_fitz(resolved_file, md_path, output_dir)

            else:
                return {
                    'status': 'failed',
                    'error': f'Unsupported file format: {file_ext}',
                    'supported_formats': list(pandoc_formats.keys()) + ['.xlsx', '.pdf'],
                    'file': file_path
                }

            if success:
                # Check if markdown file was created
                if os.path.exists(md_path):
                    file_size = os.path.getsize(md_path)
                    print_debug(f"✅ Converted: {os.path.basename(resolved_file)} → {md_filename} ({file_size} bytes)")

                    # Convert image paths to relative paths
                    self._convert_image_paths_to_relative(md_path)

                    return {
                        'status': 'success',
                        'file': file_path,
                        'output': os.path.relpath(md_path, self.workspace_root),
                        'method': 'pandoc' if file_ext in pandoc_formats else 'pymupdf',
                        'size': file_size
                    }
                else:
                    return {
                        'status': 'failed',
                        'error': 'Output file was not created',
                        'file': file_path
                    }
            else:
                return {
                    'status': 'failed',
                    'error': error_msg,
                    'file': file_path
                }

        except Exception as e:
            return {
                'status': 'failed',
                'error': f'Unexpected error: {str(e)}',
                'file': file_path
            }

    def _convert_with_pandoc(self, input_file: str, output_file: str, format_type: str) -> Tuple[bool, str]:
        """Convert document using pandoc"""
        try:
            # Create doc_images directory for extracted images
            output_dir = os.path.dirname(output_file)
            images_dir = os.path.join(output_dir, 'doc_images')
            os.makedirs(images_dir, exist_ok=True)

            # Build pandoc command
            cmd = [
                'pandoc',
                input_file,
                '-f', format_type,
                '-t', 'markdown',
                '-o', output_file,
                '--extract-media=' + images_dir,
                '--wrap=none'
            ]

            # Run pandoc conversion
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='ignore',
                timeout=300  # 5 minute timeout
            )

            if result.returncode == 0:
                return True, None
            else:
                error_msg = result.stderr.strip() if result.stderr else 'Unknown pandoc error'
                return False, f'pandoc conversion failed: {error_msg}'

        except subprocess.TimeoutExpired:
            return False, 'pandoc conversion timed out'
        except Exception as e:
            return False, f'pandoc conversion exception: {str(e)}'

    def _convert_xlsx_with_pandas(self, input_file: str, output_file: str) -> Tuple[bool, str]:
        """
        Convert Excel file to Markdown using pandas

        Args:
            input_file: Path to the Excel file
            output_file: Path for the output Markdown file

        Returns:
            Tuple of (success: bool, error_message: str)
        """
        try:
            import pandas as pd

            # Read Excel file (header=0 means first row is header)
            df = pd.read_excel(
                input_file,
                sheet_name=0,  # Use first sheet by default
                header=0,  # First row as header
                engine="openpyxl"  # Explicitly specify engine for .xlsx
            )

            # Convert to Markdown table (index=False means don't show row indices)
            md_table = df.to_markdown(index=False)

            # Save as Markdown file
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(md_table)

            return True, None

        except ImportError:
            return False, 'pandas or openpyxl not installed. Please install with: pip install pandas openpyxl'
        except Exception as e:
            return False, f'Excel to Markdown conversion failed: {str(e)}'

    def _convert_pdf_with_fitz(self, input_file: str, output_file: str, output_dir: str) -> Tuple[bool, str]:
        """Convert PDF to markdown using pymupdf (fitz)"""
        try:
            import fitz  # pymupdf

            # Open PDF document
            doc = fitz.open(input_file)

            if doc.page_count == 0:
                return False, 'PDF document is empty'

            # Create doc_images directory for extracted images
            images_dir = os.path.join(output_dir, 'doc_images')
            os.makedirs(images_dir, exist_ok=True)

            markdown_content = []
            image_counter = 0

            # Process each page
            for page_num in range(doc.page_count):
                page = doc.load_page(page_num)

                # Extract text
                text = page.get_text()

                if text.strip():
                    # Add page header
                    if doc.page_count > 1:
                        markdown_content.append(f"## Page {page_num + 1}\n")
                    markdown_content.append(text.strip())
                    markdown_content.append("\n---\n")

                # Extract images
                images = page.get_images(full=True)
                for img_index, img in enumerate(images):
                    try:
                        xref = img[0]
                        base_image = doc.extract_image(xref)
                        image_bytes = base_image["image"]
                        image_ext = base_image["ext"]

                        # Generate unique image filename
                        image_filename = f"pdf_image_{page_num + 1}_{img_index + 1}.{image_ext}"
                        image_path = os.path.join(images_dir, image_filename)

                        # Save image
                        with open(image_path, "wb") as img_file:
                            img_file.write(image_bytes)

                        # Add image reference to markdown
                        rel_image_path = os.path.join('doc_images', image_filename)
                        markdown_content.append(f"![Extracted Image]({rel_image_path})\n\n")

                    except Exception as img_error:
                        print_debug(f"Warning: Failed to extract image {img_index} from page {page_num + 1}: {str(img_error)}")
                        continue

            doc.close()

            # Write markdown content
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write('\n'.join(markdown_content))

            return True, None

        except ImportError:
            return False, 'pymupdf (fitz) library not available. Please install with: pip install pymupdf'
        except Exception as e:
            return False, f'PDF conversion exception: {str(e)}'

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
                                         capture_output=True, text=True, encoding='utf-8', errors='ignore', timeout=5)
                    if result.returncode == 0:
                        available_engines.append((engine_name, engine_option))
                elif engine_name == 'wkhtmltopdf':
                    # Check if wkhtmltopdf is available
                    result = subprocess.run(['wkhtmltopdf', '--version'], 
                                         capture_output=True, text=True, encoding='utf-8', errors='ignore', timeout=5)
                    if result.returncode == 0:
                        available_engines.append((engine_name, engine_option))
                elif engine_name == 'weasyprint':
                    # Check if weasyprint is available
                    result = subprocess.run(['weasyprint', '--version'], 
                                         capture_output=True, text=True, encoding='utf-8', errors='ignore', timeout=5)
                    if result.returncode == 0:
                        available_engines.append((engine_name, engine_option))
            except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.CalledProcessError):
                continue
        
        if not available_engines:
            print_debug("❌ No PDF engines available. Please install at least one of: xelatex, lualatex, pdflatex, wkhtmltopdf, or weasyprint")
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

    def process_markdown_diagrams(self, markdown_file: str) -> Dict[str, Any]:
        """
        Process a markdown file to convert Mermaid charts and SVG code blocks to images
        
        Args:
            markdown_file: Path to the markdown file (relative or absolute)
            
        Returns:
            Dictionary containing processing results for both Mermaid and SVG
        """
        try:
            file_path = self._resolve_path(markdown_file)
            
            if not os.path.exists(file_path):
                return {
                    'status': 'failed',
                    'file': markdown_file,
                    'error': f'Markdown file not found: {file_path}',
                    'resolved_path': file_path
                }
            
            if not file_path.lower().endswith('.md'):
                return {
                    'status': 'failed',
                    'file': markdown_file,
                    'error': 'File is not a markdown file (.md extension required)',
                    'resolved_path': file_path
                }
            
            print_debug(f"🎨 Processing markdown diagrams in: {markdown_file}")
            
            processing_results = {
                'status': 'success',
                'file': markdown_file,
                'resolved_path': file_path,
                'mermaid_processing': None,
                'svg_processing': None,
                'summary': {}
            }
            
            # Process Mermaid charts
            # 🚀 延迟加载：只在需要处理 Mermaid 图表时才加载处理器
            processor = _get_mermaid_processor()
            if processor:
                try:
                    if processor.has_mermaid_charts(file_path):
                        print_debug(f"🎨 Detected Mermaid charts, processing...")
                        mermaid_result = processor.process_markdown_file(file_path)
                        processing_results['mermaid_processing'] = mermaid_result
                        
                        if mermaid_result['status'] == 'success':
                            print_debug(f"✅ Mermaid processing completed: {mermaid_result.get('message', '')}")
                        else:
                            print_debug(f"⚠️ Mermaid processing failed: {mermaid_result.get('message', 'Unknown error')}")
                    else:
                        processing_results['mermaid_processing'] = {
                            'status': 'success',
                            'message': 'No Mermaid charts found',
                            'charts_found': 0
                        }
                        print_debug(f"ℹ️ No Mermaid charts found in file")
                        
                except Exception as e:
                    print_debug(f"⚠️ Error during Mermaid processing: {e}")
                    processing_results['mermaid_processing'] = {
                        'status': 'failed',
                        'error': str(e),
                        'message': f'Mermaid processing error: {e}'
                    }
            else:
                processing_results['mermaid_processing'] = {
                    'status': 'skipped',
                    'message': 'Mermaid processor not available'
                }
                print_debug(f"⚠️ Mermaid processor not available")
            
            # Process SVG code blocks
            if SVG_PROCESSOR_AVAILABLE:
                try:
                    if svg_processor.has_svg_blocks(file_path):
                        print_debug(f"🎨 Detected SVG code blocks, processing...")
                        svg_result = svg_processor.process_markdown_file(file_path)
                        processing_results['svg_processing'] = svg_result
                        
                        if svg_result['status'] == 'success':
                            print_debug(f"✅ SVG processing completed: {svg_result.get('message', '')}")
                        else:
                            print_debug(f"⚠️ SVG processing failed: {svg_result.get('message', 'Unknown error')}")
                    else:
                        processing_results['svg_processing'] = {
                            'status': 'success',
                            'message': 'No SVG code blocks found',
                            'svg_blocks_found': 0
                        }
                        print_debug(f"ℹ️ No SVG code blocks found in file")
                        
                except Exception as e:
                    print_debug(f"⚠️ Error during SVG processing: {e}")
                    processing_results['svg_processing'] = {
                        'status': 'failed',
                        'error': str(e),
                        'message': f'SVG processing error: {e}'
                    }
            else:
                processing_results['svg_processing'] = {
                    'status': 'skipped',
                    'message': 'SVG processor not available'
                }
                print_debug(f"⚠️ SVG processor not available")
            
            # Generate summary
            mermaid_charts = 0
            svg_blocks = 0
            mermaid_success = 0
            svg_success = 0
            
            if processing_results['mermaid_processing'] and processing_results['mermaid_processing'].get('status') == 'success':
                mermaid_charts = processing_results['mermaid_processing'].get('charts_found', 0)
                mermaid_success = processing_results['mermaid_processing'].get('successful_conversions', 0)
            
            if processing_results['svg_processing'] and processing_results['svg_processing'].get('status') == 'success':
                svg_blocks = processing_results['svg_processing'].get('svg_blocks_found', 0)
                svg_success = processing_results['svg_processing'].get('successful_conversions', 0)
            
            processing_results['summary'] = {
                'total_diagrams_found': mermaid_charts + svg_blocks,
                'total_successful_conversions': mermaid_success + svg_success,
                'mermaid_charts_found': mermaid_charts,
                'mermaid_successful': mermaid_success,
                'svg_blocks_found': svg_blocks,
                'svg_successful': svg_success
            }
            
            print_debug(f"📊 Processing summary:")
            print_debug(f"   🎯 Total diagrams found: {mermaid_charts + svg_blocks}")
            print_debug(f"   ✅ Successful conversions: {mermaid_success + svg_success}")
            print_debug(f"   📈 Mermaid: {mermaid_success}/{mermaid_charts}")
            print_debug(f"   🎨 SVG: {svg_success}/{svg_blocks}")
            
            return processing_results
            
        except Exception as e:
            print_debug(f"❌ Error processing markdown diagrams: {e}")
            return {
                'status': 'failed',
                'file': markdown_file,
                'error': str(e),
                'message': f'Diagram processing error: {e}'
            }

    def _convert_markdown_to_formats(self, file_path: str, target_file: str, format_type: str = 'both') -> Dict[str, Any]:
        """
        Convert Markdown files to Word, PDF, and LaTeX formats
        
        Args:
            file_path: Absolute path of Markdown file
            target_file: Relative path of Markdown file
            format_type: Format to convert ('word', 'pdf', 'latex', or 'both')
            
        Returns:
            Dictionary containing conversion results
        """
        import subprocess
        from pathlib import Path
        
        try:
            # Load configuration to check auto-conversion settings
            config = load_config()
            
            # Get auto-conversion settings (default to True for word/pdf for backward compatibility, False for latex)
            auto_convert_word = config.get('auto_convert_to_word', 'True').lower() == 'true'
            auto_convert_pdf = config.get('auto_convert_to_pdf', 'True').lower() == 'true'
            auto_convert_latex = config.get('auto_convert_to_latex', 'False').lower() == 'true'
            
            # If format_type is 'both', check individual config settings (for auto-conversion)
            # When format_type is explicitly specified ('word', 'pdf', 'latex'), ignore config and force conversion
            # This allows manual conversion requests to work even when auto-conversion is disabled
            if format_type == 'both':
                # Determine which formats to generate based on config (auto-conversion mode)
                should_generate_word = auto_convert_word
                should_generate_pdf = auto_convert_pdf
                should_generate_latex = auto_convert_latex
            else:
                # When format_type is explicitly specified, ignore config and force conversion (manual request)
                should_generate_word = (format_type == 'word')
                should_generate_pdf = (format_type == 'pdf')
                should_generate_latex = (format_type == 'latex')
            
            # If no formats should be generated, return early
            if not (should_generate_word or should_generate_pdf or should_generate_latex):
                print_debug(f"ℹ️ Auto-conversion disabled for all formats, skipping conversion")
                return {
                    'status': 'skipped',
                    'markdown_file': target_file,
                    'message': 'Auto-conversion disabled for all formats',
                    'conversions': {}
                }
            
            md_path = Path(file_path)
            base_name = md_path.stem
            output_dir = md_path.parent
            
            # Generate output filenames
            word_file = output_dir / f"{base_name}.docx"
            pdf_file = output_dir / f"{base_name}.pdf"
            latex_file = output_dir / f"{base_name}.tex"
            
            conversion_results = {
                'status': 'success',
                'markdown_file': target_file,
                'conversions': {}
            }

            # Track if PDF has already been converted via pywin32
            pdf_already_converted = False

            # Special handling for Windows PDF conversion: check for existing docx and convert directly
            import platform
            if platform.system() == 'Windows' and should_generate_pdf:
                print_debug(f"🔍 Windows system detected, checking for existing Word document: {word_file.name}")
                if word_file.exists():
                    print_debug(f"✅ Found existing Word document: {word_file.name}, converting directly to PDF")
                    try:
                        pdf_conversion_result = self._word_to_pdf(str(word_file), str(pdf_file))
                        if pdf_conversion_result['status'] == 'success':
                                    conversion_results['conversions']['pdf'] = {
                                        'status': 'success',
                                        'file': str(pdf_file.relative_to(self.workspace_root)),
                                        'size': pdf_conversion_result['size'],
                                        'size_kb': pdf_conversion_result['size_kb'],
                                        'method': 'pywin32_from_existing_docx'
                                    }
                                    print_debug(f"✅ PDF document conversion successful (via pywin32 from existing docx): {pdf_file.name} ({pdf_conversion_result['size_kb']})")
                                    # 额外清理一次临时文件，确保没有遗留
                                    self._cleanup_word_temp_files(str(output_dir))
                                    return conversion_results
                        elif pdf_conversion_result['status'] == 'skipped':
                            print_debug(f"ℹ️ Word to PDF conversion skipped: {pdf_conversion_result.get('message', 'Unknown reason')}")
                        else:
                            print_debug(f"⚠️ Word to PDF conversion failed: {pdf_conversion_result.get('error', 'Unknown error')}")
                            print_debug("Falling back to normal PDF conversion process...")
                    except Exception as e:
                        print_debug(f"⚠️ Error during Word to PDF conversion from existing docx: {e}")
                        print_debug("Falling back to normal PDF conversion process...")
                else:
                    print_debug(f"ℹ️ No existing Word document found: {word_file.name}, will generate docx first then convert to PDF")
                    # Generate Word document first, then convert to PDF using pywin32
                    print_debug(f"📄 Generating Word document first: {word_file.name}")
                    temp_files = []  # Track temporary files for cleanup

                    try:
                        # Step 1: Create emoji-free version if needed
                        actual_input_file = md_path.name  # Default to original file
                        try:
                            temp_md_file = create_emoji_free_markdown(str(md_path))
                            if temp_md_file:
                                actual_input_file = os.path.basename(temp_md_file)  # Use filename for pandoc
                                temp_files.append(temp_md_file)
                        except Exception as e:
                            print_debug(f"⚠️ Warning: Failed to create emoji-free markdown: {e}")

                        # Use pandoc to convert to Word with multiple filters
                        # Find the project root directory (where src/ folder exists)
                        current_file = Path(__file__)
                        project_root = current_file.parent.parent.parent  # Go up from src/tools/file_system_tools.py to project root
                        svg_to_png_filter_path = project_root / 'src' / 'utils' / 'word_svg_to_png_filter.lua'
                        image_filter_path = project_root / 'src' / 'utils' / 'word_image_filter.lua'
                        title_color_filter_path = project_root / 'src' / 'utils' / 'word_title_color_filter.lua'
                        reference_doc_path = project_root / 'src' / 'utils' / 'word_reference.docx'

                        cmd = [
                            'pandoc',
                            actual_input_file,  # Use emoji-free file if available
                            '-o', word_file.name,  # Use filename instead of full path
                            '--from', 'markdown',
                            '--to', 'docx'
                        ]

                        # Add reference document if it exists
                        if reference_doc_path.exists():
                            cmd.extend(['--reference-doc', str(reference_doc_path)])
                            print_debug(f"✅ Using Word reference template: {reference_doc_path}")
                        else:
                            print_debug(f"⚠️ Word reference template not found: {reference_doc_path}")

                        # Add SVG to PNG filter if it exists
                        if svg_to_png_filter_path.exists():
                            cmd.extend(['--lua-filter', str(svg_to_png_filter_path)])
                            print_debug(f"✅ Using SVG to PNG filter: {svg_to_png_filter_path}")
                        else:
                            print_debug(f"⚠️ SVG to PNG filter not found: {svg_to_png_filter_path}")

                        # Add image size limit filter if it exists
                        if image_filter_path.exists():
                            cmd.extend(['--lua-filter', str(image_filter_path)])
                            print_debug(f"✅ Using image size limit filter: {image_filter_path}")
                        else:
                            print_debug(f"⚠️ Image size limit filter not found: {image_filter_path}")

                        # Add title color filter if it exists
                        if title_color_filter_path.exists():
                            cmd.extend(['--lua-filter', str(title_color_filter_path)])
                            print_debug(f"✅ Using title color filter: {title_color_filter_path}")
                        else:
                            print_debug(f"⚠️ Title color filter not found: {title_color_filter_path}")

                        # Execute command in markdown file directory
                        result = subprocess.run(cmd, check=True, capture_output=True, text=True, encoding='utf-8', errors='ignore', cwd=str(output_dir))

                        if word_file.exists():
                            # 后处理：修改Word文档中的标题颜色
                            try:
                                postprocessor_path = project_root / 'src' / 'utils' / 'word_style_postprocessor.py'
                                if postprocessor_path.exists():
                                    import sys
                                    postprocess_cmd = [sys.executable, str(postprocessor_path), str(word_file)]
                                    subprocess.run(postprocess_cmd, check=True, capture_output=True, text=True, encoding='utf-8', errors='ignore')
                                    print_debug(f"✅ Word document post-processed for title colors: {word_file.name}")
                                else:
                                    print_debug(f"⚠️ Word style postprocessor not found: {postprocessor_path}")
                            except Exception as e:
                                print_debug(f"⚠️ Word document post-processing failed: {str(e)}")

                            print_debug(f"✅ Word document generated: {word_file.name}")

                            # Now convert the newly generated docx to PDF using pywin32
                            print_debug(f"📄 Converting newly generated Word document to PDF: {pdf_file.name}")
                            try:
                                pdf_conversion_result = self._word_to_pdf(str(word_file), str(pdf_file))
                                if pdf_conversion_result['status'] == 'success':
                                    conversion_results['conversions']['pdf'] = {
                                        'status': 'success',
                                        'file': str(pdf_file.relative_to(self.workspace_root)),
                                        'size': pdf_conversion_result['size'],
                                        'size_kb': pdf_conversion_result['size_kb'],
                                        'method': 'pywin32_from_generated_docx'
                                    }
                                    print_debug(f"✅ PDF document conversion successful (via pywin32 from generated docx): {pdf_file.name} ({pdf_conversion_result['size_kb']})")
                                    # 额外清理一次临时文件，确保没有遗留
                                    self._cleanup_word_temp_files(str(output_dir))
                                    return conversion_results
                                else:
                                    print_debug(f"⚠️ Word to PDF conversion failed: {pdf_conversion_result.get('error', 'Unknown error')}")
                                    print_debug("Falling back to normal PDF conversion process...")
                            except Exception as e:
                                print_debug(f"⚠️ Error during Word to PDF conversion from generated docx: {e}")
                                print_debug("Falling back to normal PDF conversion process...")
                        else:
                            print_debug(f"❌ Word document generation failed: File not generated")
                            print_debug("Falling back to normal PDF conversion process...")

                    except subprocess.CalledProcessError as e:
                        print_debug(f"❌ Word document generation failed: {e.stderr}")
                        print_debug("Falling back to normal PDF conversion process...")
                    except Exception as e:
                        print_debug(f"❌ Word document generation exception: {str(e)}")
                        print_debug("Falling back to normal PDF conversion process...")
                    finally:
                        # Clean up temporary files
                        for temp_file in temp_files:
                            try:
                                if os.path.exists(temp_file):
                                    os.remove(temp_file)
                                    print_debug(f"🧹 Cleaned up temporary file: {temp_file}")
                            except Exception as e:
                                print_debug(f"⚠️ Failed to clean up temporary file {temp_file}: {e}")
                        
                        # 清理Word生成的临时文件
                        self._cleanup_word_temp_files(str(output_dir))

            # Convert to Word document
            if should_generate_word:
                print_debug(f"📄 Converting Markdown to Word document: {word_file.name}")
                temp_files = []  # Track temporary files for cleanup

                try:
                    # Step 1: Create emoji-free version if needed
                    actual_input_file = md_path.name  # Default to original file
                    try:
                        temp_md_file = create_emoji_free_markdown(str(md_path))
                        if temp_md_file:
                            actual_input_file = os.path.basename(temp_md_file)  # Use filename for pandoc
                            temp_files.append(temp_md_file)
                    except Exception as e:
                        print_debug(f"⚠️ Warning: Failed to create emoji-free markdown: {e}")

                    # Use pandoc to convert to Word with multiple filters
                    # Find the project root directory (where src/ folder exists)
                    current_file = Path(__file__)
                    project_root = current_file.parent.parent.parent  # Go up from src/tools/file_system_tools.py to project root
                    svg_to_png_filter_path = project_root / 'src' / 'utils' / 'word_svg_to_png_filter.lua'
                    image_filter_path = project_root / 'src' / 'utils' / 'word_image_filter.lua'
                    title_color_filter_path = project_root / 'src' / 'utils' / 'word_title_color_filter.lua'
                    reference_doc_path = project_root / 'src' / 'utils' / 'word_reference.docx'
                    
                    cmd = [
                        'pandoc',
                        actual_input_file,  # Use emoji-free file if available
                        '-o', word_file.name,  # Use filename instead of full path
                        '--from', 'markdown',
                        '--to', 'docx'
                    ]
                    
                    # Add reference document if it exists
                    if reference_doc_path.exists():
                        cmd.extend(['--reference-doc', str(reference_doc_path)])
                        print_debug(f"✅ Using Word reference template: {reference_doc_path}")
                    else:
                        print_debug(f"⚠️ Word reference template not found: {reference_doc_path}")
                    
                    # Add SVG to PNG filter if it exists
                    if svg_to_png_filter_path.exists():
                        cmd.extend(['--lua-filter', str(svg_to_png_filter_path)])
                        print_debug(f"✅ Using SVG to PNG filter: {svg_to_png_filter_path}")
                    else:
                        print_debug(f"⚠️ SVG to PNG filter not found: {svg_to_png_filter_path}")
                    
                    # Add image size limit filter if it exists
                    if image_filter_path.exists():
                        cmd.extend(['--lua-filter', str(image_filter_path)])
                        print_debug(f"✅ Using image size limit filter: {image_filter_path}")
                    else:
                        print_debug(f"⚠️ Image size limit filter not found: {image_filter_path}")
                    
                    # Add title color filter if it exists
                    if title_color_filter_path.exists():
                        cmd.extend(['--lua-filter', str(title_color_filter_path)])
                        print_debug(f"✅ Using title color filter: {title_color_filter_path}")
                    else:
                        print_debug(f"⚠️ Title color filter not found: {title_color_filter_path}")
                    
                    # Execute command in markdown file directory
                    result = subprocess.run(cmd, check=True, capture_output=True, text=True, encoding='utf-8', errors='ignore', cwd=str(output_dir))
                    
                    if word_file.exists():
                        # 后处理：修改Word文档中的标题颜色
                        try:
                            postprocessor_path = project_root / 'src' / 'utils' / 'word_style_postprocessor.py'
                            if postprocessor_path.exists():
                                import sys
                                postprocess_cmd = [sys.executable, str(postprocessor_path), str(word_file)]
                                subprocess.run(postprocess_cmd, check=True, capture_output=True, text=True, encoding='utf-8', errors='ignore')
                                print_debug(f"✅ Word document post-processed for title colors: {word_file.name}")
                            else:
                                print_debug(f"⚠️ Word style postprocessor not found: {postprocessor_path}")
                        except Exception as e:
                            print_debug(f"⚠️ Word document post-processing failed: {str(e)}")
                        
                        file_size = word_file.stat().st_size
                        conversion_results['conversions']['word'] = {
                            'status': 'success',
                            'file': str(word_file.relative_to(self.workspace_root)),
                            'size': file_size,
                            'size_kb': f"{file_size / 1024:.1f} KB"
                        }
                        print_debug(f"✅ Word document conversion successful: {word_file.name} ({file_size / 1024:.1f} KB)")

                        # 在 Windows 系统上使用 pywin32 从 docx 生成 PDF
                        pdf_already_converted = False
                        if should_generate_pdf:
                            try:
                                pdf_conversion_result = self._word_to_pdf(str(word_file), str(pdf_file))
                                if pdf_conversion_result['status'] == 'success':
                                    conversion_results['conversions']['pdf'] = {
                                        'status': 'success',
                                        'file': str(pdf_file.relative_to(self.workspace_root)),
                                        'size': pdf_conversion_result['size'],
                                        'size_kb': pdf_conversion_result['size_kb'],
                                        'method': 'pywin32_from_docx'
                                    }
                                    print_debug(f"✅ PDF document conversion successful (via pywin32): {pdf_file.name} ({pdf_conversion_result['size_kb']})")
                                    pdf_already_converted = True  # 标记PDF已转换
                                elif pdf_conversion_result['status'] == 'skipped':
                                    print_debug(f"ℹ️ Word to PDF conversion skipped: {pdf_conversion_result.get('message', 'Unknown reason')}")
                                else:
                                    print_debug(f"⚠️ Word to PDF conversion failed: {pdf_conversion_result.get('error', 'Unknown error')}")
                            except Exception as e:
                                print_debug(f"⚠️ Error during Word to PDF conversion: {e}")
                    else:
                        conversion_results['conversions']['word'] = {
                            'status': 'failed',
                            'error': 'Word file not generated'
                        }
                        print_debug(f"❌ Word document conversion failed: File not generated")
                        
                except subprocess.CalledProcessError as e:
                    conversion_results['conversions']['word'] = {
                        'status': 'failed',
                        'error': f'pandoc conversion failed: {e.stderr}'
                    }
                    print_debug(f"❌ Word document conversion failed: {e.stderr}")
                except Exception as e:
                    conversion_results['conversions']['word'] = {
                        'status': 'failed',
                        'error': f'Conversion exception: {str(e)}'
                    }
                    print_debug(f"❌ Word document conversion exception: {str(e)}")
                finally:
                    # Clean up temporary files
                    for temp_file in temp_files:
                        try:
                            if os.path.exists(temp_file):
                                os.remove(temp_file)
                                print_debug(f"🧹 Cleaned up temporary file: {temp_file}")
                        except Exception as e:
                            print_debug(f"⚠️ Failed to clean up temporary file {temp_file}: {e}")
                    
                    # 清理Word生成的临时文件
                    self._cleanup_word_temp_files(str(output_dir))
            
            # Convert to PDF document (synchronous execution)
            if should_generate_pdf and not pdf_already_converted:
                print_debug(f"📄 Starting Markdown to PDF conversion: {pdf_file.name}")
                try:
                    # Use trans_md_to_pdf.py script to convert to PDF
                    trans_script = Path(__file__).parent.parent / "utils" / "trans_md_to_pdf.py"
                    
                    if trans_script.exists():
                        import sys
                        cmd = [
                            sys.executable,  # Use current Python interpreter
                            str(trans_script),
                            md_path.name,  # Use filename instead of full path
                            pdf_file.name
                            ]
                        
                        print_debug(f"🔧 PDF conversion command: {' '.join(cmd)}")
                        print_debug(f"🔧 Working directory: {output_dir}")
                        
                        # Execute command synchronously in markdown file directory
                        result = subprocess.run(
                            cmd,
                            capture_output=True,
                            text=True,
                            encoding='utf-8',
                            errors='ignore',
                            cwd=str(output_dir),
                            timeout=120  # 2 minutes timeout
                        )
                        
                        # Log command output for debugging
                        if result.stdout:
                            print_debug(f"📋 PDF conversion stdout: {result.stdout}")
                        if result.stderr:
                            print_debug(f"⚠️ PDF conversion stderr: {result.stderr}")
                        print_debug(f"📊 PDF conversion return code: {result.returncode}")
                        
                        # Check if PDF file was generated
                        if pdf_file.exists():
                            file_size = pdf_file.stat().st_size
                            conversion_results['conversions']['pdf'] = {
                                'status': 'success',
                                'file': str(pdf_file.relative_to(self.workspace_root)),
                                'size': file_size,
                                'size_kb': f"{file_size / 1024:.1f} KB",
                                'method': 'trans_md_to_pdf_script'
                            }
                            print_debug(f"✅ PDF document conversion successful: {pdf_file.name} ({file_size / 1024:.1f} KB)")
                        else:
                            # Build comprehensive error message
                            error_parts = []
                            if result.returncode != 0:
                                error_parts.append(f"Return code: {result.returncode}")
                            if result.stderr:
                                error_parts.append(f"stderr: {result.stderr}")
                            if result.stdout:
                                error_parts.append(f"stdout: {result.stdout}")
                            if not error_parts:
                                error_parts.append("PDF file not generated (unknown error)")
                            
                            error_msg = " | ".join(error_parts)
                            conversion_results['conversions']['pdf'] = {
                                'status': 'failed',
                                'error': error_msg,
                                'return_code': result.returncode,
                                'stdout': result.stdout if result.stdout else None,
                                'stderr': result.stderr if result.stderr else None
                            }
                            print_debug(f"❌ PDF conversion failed: {error_msg}")
                            
                    else:
                        print_debug(f"⚠️ trans_md_to_pdf.py script not found, skipping PDF conversion")
                        conversion_results['conversions']['pdf'] = {
                            'status': 'skipped',
                            'error': 'trans_md_to_pdf.py script not found'
                        }
                            
                except subprocess.TimeoutExpired:
                    conversion_results['conversions']['pdf'] = {
                        'status': 'failed',
                        'error': 'PDF conversion timeout (exceeded 2 minutes)'
                    }
                    print_debug(f"❌ PDF conversion timeout")
                except Exception as e:
                    conversion_results['conversions']['pdf'] = {
                        'status': 'failed',
                        'error': f'Failed to convert PDF: {str(e)}'
                    }
                    print_debug(f"❌ Failed to convert PDF: {str(e)}")
            
            # Convert to LaTeX document
            if should_generate_latex:
                print_debug(f"📄 Starting Markdown to LaTeX conversion: {latex_file.name}")
                try:
                    # Use trans_md_to_pdf.py script to convert to LaTeX
                    # Resolve path: file_system_tools.py is in src/tools/, so parent.parent is src/, then utils/
                    current_file = Path(__file__).resolve()
                    trans_script = current_file.parent.parent / "utils" / "trans_md_to_pdf.py"
                    
                    if trans_script.exists():
                        import sys
                        cmd = [
                            sys.executable,  # Use current Python interpreter
                            str(trans_script),
                            md_path.name,  # Use filename instead of full path
                            latex_file.name,  # Use filename instead of full path
                            '--latex'  # Add LaTeX flag
                        ]
                        
                        print_debug(f"🔧 LaTeX conversion command: {' '.join(cmd)}")
                        print_debug(f"🔧 Working directory: {output_dir}")
                        
                        # Execute command synchronously in markdown file directory
                        result = subprocess.run(
                            cmd,
                            capture_output=True,
                            text=True,
                            encoding='utf-8',
                            errors='ignore',
                            cwd=str(output_dir),
                            timeout=120  # 2 minutes timeout
                        )
                        
                        # Log command output for debugging
                        if result.stdout:
                            print_debug(f"📋 LaTeX conversion stdout: {result.stdout}")
                        if result.stderr:
                            print_debug(f"⚠️ LaTeX conversion stderr: {result.stderr}")
                        print_debug(f"📊 LaTeX conversion return code: {result.returncode}")
                        
                        # Check if LaTeX file was generated
                        if latex_file.exists():
                            file_size = latex_file.stat().st_size
                            conversion_results['conversions']['latex'] = {
                                'status': 'success',
                                'file': str(latex_file.relative_to(self.workspace_root)),
                                'size': file_size,
                                'size_kb': f"{file_size / 1024:.1f} KB",
                                'method': 'trans_md_to_pdf_script'
                            }
                            print_debug(f"✅ LaTeX document conversion successful: {latex_file.name} ({file_size / 1024:.1f} KB)")
                        else:
                            # Build comprehensive error message
                            error_parts = []
                            if result.returncode != 0:
                                error_parts.append(f"Return code: {result.returncode}")
                            if result.stderr:
                                error_parts.append(f"stderr: {result.stderr}")
                            if result.stdout:
                                error_parts.append(f"stdout: {result.stdout}")
                            if not error_parts:
                                error_parts.append("LaTeX file not generated (unknown error)")
                            
                            error_msg = " | ".join(error_parts)
                            conversion_results['conversions']['latex'] = {
                                'status': 'failed',
                                'error': error_msg,
                                'return_code': result.returncode,
                                'stdout': result.stdout if result.stdout else None,
                                'stderr': result.stderr if result.stderr else None
                            }
                            print_debug(f"❌ LaTeX conversion failed: {error_msg}")
                            
                    else:
                        print_debug(f"⚠️ trans_md_to_pdf.py script not found at: {trans_script}, skipping LaTeX conversion")
                        conversion_results['conversions']['latex'] = {
                            'status': 'skipped',
                            'error': f'trans_md_to_pdf.py script not found at {trans_script}'
                        }
                            
                except subprocess.TimeoutExpired:
                    conversion_results['conversions']['latex'] = {
                        'status': 'failed',
                        'error': 'LaTeX conversion timeout (exceeded 2 minutes)'
                    }
                    print_debug(f"❌ LaTeX conversion timeout")
                except Exception as e:
                    import traceback
                    error_trace = traceback.format_exc()
                    conversion_results['conversions']['latex'] = {
                        'status': 'failed',
                        'error': f'Failed to convert LaTeX: {str(e)}',
                        'traceback': error_trace
                    }
                    print_debug(f"❌ Failed to convert LaTeX: {str(e)}")
                    print_debug(f"❌ Traceback: {error_trace}")
            
            # Check conversion results
            successful_conversions = sum(1 for conv in conversion_results['conversions'].values() 
                                       if conv.get('status') == 'success')
            total_conversions = len(conversion_results['conversions'])
            
            if successful_conversions == 0:
                #print_current(f"⚠️ All format conversions failed")
                conversion_results['status'] = 'partial_failure'
            
            return conversion_results
            
        except Exception as e:
            print_debug(f"❌ Error occurred during Markdown conversion: {str(e)}")
            return {
                'status': 'failed',
                'markdown_file': target_file,
                'error': str(e),
                'message': f'Error occurred during conversion: {str(e)}'
            }

    def _cleanup_word_temp_files(self, directory: str) -> None:
        """
        清理Word应用程序生成的临时文件（以~开头的文件）
        
        Args:
            directory: 要清理的目录路径
        """
        import os
        import glob
        
        try:
            # 查找以~开头的临时文件
            temp_patterns = [
                os.path.join(directory, '~$*'),  # Word临时文件
                os.path.join(directory, '~*.tmp'),  # 通用临时文件
                os.path.join(directory, '~*.docx'),  # Word临时文档
                os.path.join(directory, '~*.doc'),   # Word临时文档
            ]
            
            cleaned_files = []
            for pattern in temp_patterns:
                temp_files = glob.glob(pattern)
                for temp_file in temp_files:
                    try:
                        if os.path.exists(temp_file) and os.path.isfile(temp_file):
                            os.remove(temp_file)
                            cleaned_files.append(temp_file)
                            print_debug(f"🧹 Cleaned up Word temp file: {os.path.basename(temp_file)}")
                    except Exception as e:
                        print_debug(f"⚠️ Failed to clean up temp file {temp_file}: {e}")
            
            if cleaned_files:
                print_debug(f"✅ Cleaned up {len(cleaned_files)} Word temporary files")
            
        except Exception as e:
            print_debug(f"⚠️ Error during Word temp file cleanup: {e}")

    def _word_to_pdf(self, word_path: str, pdf_path: str) -> Dict[str, Any]:
        """
        将 Word 文档转换为 PDF（仅在 Windows 系统上使用 pywin32）

        Args:
            word_path: Word 文件路径（.doc 或 .docx）
            pdf_path: 输出 PDF 文件路径

        Returns:
            Dictionary containing conversion results
        """
        import platform
        import os

        # 仅在 Windows 系统上执行
        if platform.system() != 'Windows':
            return {
                'status': 'skipped',
                'message': 'Word to PDF conversion only available on Windows with pywin32'
            }

        try:
            # 尝试导入 pywin32
            import win32com.client

            # 获取文档所在目录，用于后续清理临时文件
            word_dir = os.path.dirname(os.path.abspath(word_path))

            # 启动 Word 应用
            word = win32com.client.Dispatch("Word.Application")
            word.Visible = False  # 后台运行，不显示窗口

            try:
                # 打开 Word 文档
                doc = word.Documents.Open(os.path.abspath(word_path))

                # 导出为 PDF（FileFormat=17 对应 PDF 格式）
                doc.SaveAs2(
                    os.path.abspath(pdf_path),
                    FileFormat=17  # 17 是 Word 中 PDF 格式的编码
                )

                file_size = os.path.getsize(pdf_path)
                print_debug(f"✅ Word to PDF conversion successful: {pdf_path} ({file_size / 1024:.1f} KB)")

                return {
                    'status': 'success',
                    'file': pdf_path,
                    'size': file_size,
                    'size_kb': f"{file_size / 1024:.1f} KB",
                    'method': 'pywin32'
                }

            except Exception as e:
                print_debug(f"❌ Word to PDF conversion failed: {e}")
                return {
                    'status': 'failed',
                    'error': str(e),
                    'message': f'Word to PDF conversion error: {e}'
                }
            finally:
                # 关闭文档和 Word 应用
                try:
                    if 'doc' in locals() and doc:
                        doc.Close()
                except:
                    pass
                try:
                    word.Quit()
                except:
                    pass
                
                # 等待一小段时间确保Word完全关闭，然后清理临时文件
                import time
                time.sleep(0.5)  # 等待500毫秒
                self._cleanup_word_temp_files(word_dir)

        except ImportError:
            print_debug("⚠️ pywin32 not available, skipping Word to PDF conversion")
            return {
                'status': 'skipped',
                'message': 'pywin32 not installed'
            }
        except Exception as e:
            print_debug(f"❌ Word to PDF conversion setup failed: {e}")
            return {
                'status': 'failed',
                'error': str(e),
                'message': f'Word to PDF setup error: {e}'
            }

    def _convert_image_paths_to_relative(self, md_file_path: str) -> None:
        """
        Convert image paths in markdown by replacing workspace root with '.' .

        Args:
            md_file_path: Path to the markdown file to process
        """
        try:
            import re

            # Read the markdown file
            with open(md_file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Pattern to match markdown image references: ![alt](path)
            image_pattern = r'!\[([^\]]*)\]\(([^)]+)\)'

            def convert_path(match):
                alt_text = match.group(1)
                image_path = match.group(2).strip()

                # Normalize paths for comparison (handle both forward and backward slashes)
                normalized_image_path = os.path.normpath(image_path)
                normalized_workspace_root = os.path.normpath(self.workspace_root)

                # Replace workspace root with '.'
                if normalized_workspace_root in normalized_image_path:
                    converted_path = normalized_image_path.replace(normalized_workspace_root, '.')
                else:
                    # If workspace root not found in path, replace entire path with '.'
                    converted_path = '.'

                return f'![{alt_text}]({converted_path})'

            # Apply the conversion to all image references
            converted_content = re.sub(image_pattern, convert_path, content)

            # Write back only if content changed
            if converted_content != content:
                with open(md_file_path, 'w', encoding='utf-8') as f:
                    f.write(converted_content)
                print_debug(f"🔄 Converted image paths in: {os.path.basename(md_file_path)}")

        except Exception as e:
            print_debug(f"⚠️ Failed to convert image paths in {md_file_path}: {str(e)}")

    def _validate_and_comment_invalid_images(self, md_file_path: str) -> None:
        """
        Check all image links in markdown file and comment out invalid local images.
        
        Args:
            md_file_path: Path to the markdown file to process
        """
        try:
            from pathlib import Path

            # Read the markdown file
            with open(md_file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Pattern to match markdown image references: ![alt](path)
            image_pattern = r'!\[([^\]]*)\]\(([^)]+)\)'
            
            md_dir = Path(md_file_path).parent
            content_changed = False
            invalid_count = 0

            def validate_and_replace(match):
                nonlocal content_changed, invalid_count
                alt_text = match.group(1)
                image_path = match.group(2).strip()
                
                # Skip non-local images (http, https, ftp, data URIs)
                if image_path.startswith(('http://', 'https://', 'ftp://', 'data:')):
                    return match.group(0)  # Return original unchanged
                
                # Build full image path
                if os.path.isabs(image_path):
                    full_image_path = Path(image_path)
                else:
                    # Relative path: resolve relative to markdown file directory
                    full_image_path = (md_dir / image_path).resolve()
                
                # Check if file exists
                if not full_image_path.exists():
                    invalid_count += 1
                    content_changed = True
                    print_debug(f"⚠️ Local image not found: {image_path} (resolved to: {full_image_path}), commenting out")
                    # Convert to HTML comment
                    return f'<!-- ![{alt_text}]({image_path}) -->'
                
                return match.group(0)  # Return original unchanged
            
            # Apply validation to all image references
            validated_content = re.sub(image_pattern, validate_and_replace, content)
            
            # Write back only if content changed
            if content_changed:
                with open(md_file_path, 'w', encoding='utf-8') as f:
                    f.write(validated_content)
                print_debug(f"📝 Commented out {invalid_count} invalid local image(s) in: {os.path.basename(md_file_path)}")
            else:
                print_debug(f"✅ All local images validated successfully in: {os.path.basename(md_file_path)}")

        except Exception as e:
            print_debug(f"⚠️ Error validating images in {md_file_path}: {str(e)}")



 