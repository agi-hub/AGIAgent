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
import shutil
import subprocess
import threading
import time
import queue
from typing import List, Dict, Any, Optional

from .code_repository_parser import CodeRepositoryParser


class BaseTools:
    def __init__(self, workspace_root: str = None):
        """Initialize tools with a workspace root directory."""
        self.workspace_root = workspace_root or os.getcwd()
        self.last_edit = None  # Store the last edit for reapply

        # Initialize code repository parser
        self.code_parser = None
        self._init_code_parser()

    def _get_code_index_path(self) -> str:
        """
        Get the path of the code index database, ensuring it's outside the workspace
        
        Returns:
            Full path of the code index database
        """
        workspace_parent = os.path.dirname(self.workspace_root)
        workspace_name = os.path.basename(self.workspace_root)
        
        index_dir_name = f"{workspace_name}_code_index"
        code_index_path = os.path.join(workspace_parent, index_dir_name)
        
        return code_index_path

    def _init_code_parser(self):
        """Initialize code repository parser"""
        try:
            repo_root = os.path.join(self.workspace_root, "workspace")
            if not os.path.exists(repo_root):
                repo_root = self.workspace_root
            
            # print(f"🔍 Initializing code repository parser, root path: {repo_root}")
            
            supported_extensions = [
                '.py', '.js', '.ts', '.jsx', '.tsx', '.java', '.c', '.cpp', '.h', '.hpp',
                '.cs', '.php', '.rb', '.go', '.rs', '.swift', '.kt', '.scala', 
                '.html', '.css', '.scss', '.less', '.xml', '.json', '.yaml', '.yml',
                '.sql', '.sh', '.bat', '.ps1', '.dockerfile', '.txt', '.md', '.rst'
            ]
            
            self.code_parser = CodeRepositoryParser(
                root_path=repo_root,
                segment_size=200,
                supported_extensions=supported_extensions
            )
            
            db_path = self._get_code_index_path()
            
            if os.path.exists(db_path):
                try:
                    # print(f"📚 Loading existing code index database: {db_path}")
                    self.code_parser.load_database(db_path)
                    # print(f"✅ Code index database loaded successfully")
                except Exception as e:
                    print(f"⚠️ Failed to load code index database: {e}, will recreate")
                    self._rebuild_code_index()
            else:
                # print(f"🆕 Creating new code index database: {db_path}")
                self._rebuild_code_index()
                
        except Exception as e:
            print(f"❌ Failed to initialize code repository parser: {e}")
            self.code_parser = None

    def _rebuild_code_index(self):
        """Rebuild code index"""
        try:
            if self.code_parser:
                # print(f"🔄 Starting to build code index...")
                self.code_parser.parse_repository(force_rebuild=True)
                
                db_path = self._get_code_index_path()
                self.code_parser.save_database(db_path)
                # print(f"✅ Code index build complete, saved to: {db_path}")
        except Exception as e:
            print(f"❌ Failed to rebuild code index: {e}")

    def perform_incremental_update(self):
        """Perform incremental update"""
        try:
            if not self.code_parser:
                return
                
            update_result = self.code_parser.incremental_update()
            
            if any(count > 0 for count in update_result.values()):
                db_path = self._get_code_index_path()
                self.code_parser.save_database(db_path)
                
        except Exception as e:
            print(f"⚠️ Code repository update failed: {e}")

    def _resolve_path(self, path: str) -> str:
        """Resolve a path to an absolute path, cleaning up any redundant workspace prefixes."""
        if os.path.isabs(path):
            return path
        
        if path.startswith('workspace/'):
            workspace_dir_name = os.path.basename(self.workspace_root)
            if workspace_dir_name in ['workspace', 'output']:
                path = path[10:]
                # print(f"⚠️  Path cleanup: removed redundant 'workspace/' prefix, using: {path}")
        
        resolved_path = os.path.join(self.workspace_root, path)
        # print(f"🔍 Path resolution: '{path}' -> '{resolved_path}'")
        return resolved_path

    def _read_process_output_with_timeout(self, process, timeout_inactive=300, max_total_time=600):
        """
        Asynchronously read process output with smart timeout, while displaying real-time output to user
        """
        stdout_lines = []
        stderr_lines = []
        last_output_time = time.time()
        start_time = time.time()
        
        stdout_queue = queue.Queue()
        stderr_queue = queue.Queue()
        
        def read_stdout():
            try:
                for line in iter(process.stdout.readline, ''):
                    if line:
                        stdout_queue.put(('stdout', line, time.time()))
                process.stdout.close()
            except:
                pass
        
        def read_stderr():
            try:
                for line in iter(process.stderr.readline, ''):
                    if line:
                        stderr_queue.put(('stderr', line, time.time()))
                process.stderr.close()
            except:
                pass
        
        stdout_thread = threading.Thread(target=read_stdout)
        stderr_thread = threading.Thread(target=read_stderr)
        stdout_thread.daemon = True
        stderr_thread.daemon = True
        stdout_thread.start()
        stderr_thread.start()
        
        timed_out = False
        
        pass  # Separator line removed
        print("🚀 Command execution started, real-time output as follows:")
        pass  # Separator line removed
        
        try:
            while process.poll() is None:
                current_time = time.time()
                
                got_output = False
                
                try:
                    while True:
                        output_type, line, timestamp = stdout_queue.get_nowait()
                        line_clean = line.rstrip()
                        stdout_lines.append(line_clean)
                        if line_clean:
                            print(f"📤 {line_clean}")
                        last_output_time = timestamp
                        got_output = True
                except queue.Empty:
                    pass
                
                try:
                    while True:
                        output_type, line, timestamp = stderr_queue.get_nowait()
                        line_clean = line.rstrip()
                        stderr_lines.append(line_clean)
                        if line_clean:
                            print(f"⚠️  {line_clean}")
                        last_output_time = timestamp
                        got_output = True
                except queue.Empty:
                    pass
                
                time_since_last_output = current_time - last_output_time
                total_time = current_time - start_time
                
                if total_time > max_total_time:
                    print(f"\n⏰ Process execution exceeded maximum time limit of {max_total_time} seconds, force terminating")
                    timed_out = True
                    break
                elif time_since_last_output > timeout_inactive:
                    print(f"\n⏰ Process has no output for more than {timeout_inactive} seconds, may be stuck, force terminating")
                    timed_out = True
                    break
                
                time.sleep(0.1)
            
            if timed_out:
                try:
                    process.terminate()
                    print("🔄 Attempting graceful process termination...")
                    try:
                        process.wait(timeout=5)
                        print("✅ Process terminated gracefully")
                    except subprocess.TimeoutExpired:
                        print("💀 Force killing process...")
                        process.kill()
                        process.wait()
                        print("✅ Process force terminated")
                except:
                    pass
            
            try:
                while True:
                    output_type, line, timestamp = stdout_queue.get_nowait()
                    line_clean = line.rstrip()
                    stdout_lines.append(line_clean)
                    if line_clean:
                        print(f"📤 {line_clean}")
            except queue.Empty:
                pass
            
            try:
                while True:
                    output_type, line, timestamp = stderr_queue.get_nowait()
                    line_clean = line.rstrip()
                    stderr_lines.append(line_clean)
                    if line_clean:
                        print(f"⚠️  {line_clean}")
            except queue.Empty:
                pass
                
        except KeyboardInterrupt:
            print("\n⏰ User interrupted, terminating process")
            timed_out = True
            try:
                process.terminate()
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            except:
                pass
        
        return_code = process.returncode if process.returncode is not None else -1
        
        pass  # Separator line removed
        if timed_out:
            print("⏰ Command execution timed out")
        elif return_code == 0:
            print("✅ Command execution completed successfully")
        else:
            print(f"❌ Command execution failed, exit code: {return_code}")
        pass  # Separator line removed
        
        return '\n'.join(stdout_lines), '\n'.join(stderr_lines), return_code, timed_out