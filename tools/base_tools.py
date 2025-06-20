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
import subprocess
import threading
import time
import queue

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
        """Get the path to the code index database"""
        workspace_name = os.path.basename(self.workspace_root.rstrip('/'))
        
        if workspace_name == "workspace":
            workspace_name = "test_workspace"
        
        db_path = f"{workspace_name.replace('/', '_')}_code_index"
        
        if not os.path.isabs(db_path):
            db_path = os.path.join(os.path.dirname(__file__), '..', db_path)
        
        return db_path

    def _init_code_parser(self):
        """Initialize code repository parser"""
        try:
            from tools.code_repository_parser import CodeRepositoryParser
            
            self.code_parser = CodeRepositoryParser(
                root_path=self.workspace_root,
                supported_extensions=['.py', '.js', '.ts', '.tsx', '.jsx', '.java', '.cpp', '.c', '.h', '.hpp', 
                          '.cs', '.php', '.rb', '.go', '.rs', '.swift', '.kt', '.scala', '.sh', '.bat', 
                          '.ps1', '.sql', '.html', '.css', '.scss', '.less', '.xml', '.json', '.yaml', 
                          '.yml', '.toml', '.cfg', '.ini', '.md', '.txt', '.dockerfile', '.makefile']
            )

            db_path = self._get_code_index_path()
            if os.path.exists(f"{db_path}/code_segments.pkl"):
                try:
                    # print(f"📚 Loading code index database from: {db_path}")
                    self.code_parser.load_database(db_path)
                    
                    changes = self.code_parser.check_repository_changes()
                    if any(changes.values()):
                        # print(f"🔄 Code files have changed, starting incremental update...")
                        self.perform_incremental_update()
                    else:
                        pass
                        # print(f"✅ Code index is up to date")
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