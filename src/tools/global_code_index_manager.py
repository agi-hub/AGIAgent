#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Global Code Index Manager
Ensures only one code index instance is running in a multi-agent environment
"""

import os
import threading
import atexit
from typing import Dict, Optional
from .print_system import print_current


class GlobalCodeIndexManager:
    """Global code index manager implementing singleton pattern"""
    
    _instance = None
    _lock = threading.Lock()
    _parsers: Dict[str, 'CodeRepositoryParser'] = {}
    _init_locks: Dict[str, threading.Lock] = {}
    
    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self._initialized = True
            # Register cleanup function on exit
            atexit.register(self.cleanup_all)
    
    def get_parser(self, workspace_root: str, **kwargs) -> Optional['CodeRepositoryParser']:
        """
        Get code parser for specified workspace
        
        Args:
            workspace_root: Workspace root directory
            **kwargs: Parameters passed to CodeRepositoryParser
            
        Returns:
            CodeRepositoryParser instance or None
        """
        workspace_root = os.path.abspath(workspace_root)
        
        # Create independent initialization lock for each workspace
        if workspace_root not in self._init_locks:
            with self._lock:
                if workspace_root not in self._init_locks:
                    self._init_locks[workspace_root] = threading.Lock()
        
        # Use workspace-specific lock for initialization
        with self._init_locks[workspace_root]:
            if workspace_root not in self._parsers:
                try:
                    from .code_repository_parser import CodeRepositoryParser
                    
                    # Set default parameters
                    default_kwargs = {
                        'enable_background_update': True,
                        'update_interval': 3.0,
                        'supported_extensions': kwargs.get('supported_extensions', [
                            '.py', '.js', '.ts', '.tsx', '.jsx', '.java', '.cpp', '.c', '.h', '.hpp', 
                            '.cs', '.php', '.rb', '.go', '.rs', '.swift', '.kt', '.scala', '.sh', '.bat', 
                            '.ps1', '.sql', '.html', '.css', '.scss', '.less', '.xml', '.json', '.yaml', 
                            '.yml', '.toml', '.cfg', '.ini', '.md', '.txt', '.dockerfile', '.makefile'
                        ])
                    }
                    default_kwargs.update(kwargs)
                    
                    # Create new parser instance
                    parser = CodeRepositoryParser(
                        root_path=workspace_root,
                        **default_kwargs
                    )
                    
                    # Initialize parser
                    success = parser.init_code_parser(
                        workspace_root=workspace_root,
                        supported_extensions=default_kwargs['supported_extensions'],
                        skip_initial_update=False  # 🔧 Fixed: Ensure initial indexing is performed
                    )
                    
                    if success:
                        self._parsers[workspace_root] = parser
                        print_current(f"🔧 Global manager: Created code parser for workspace: {os.path.basename(workspace_root)}")
                    else:
                        print_current(f"❌ Global manager: Failed to initialize parser for: {workspace_root}")
                        return None
                        
                except Exception as e:
                    print_current(f"❌ Global manager: Error creating parser for {workspace_root}: {e}")
                    return None
            else:
                print_current(f"🔄 Global manager: Reusing existing parser for: {os.path.basename(workspace_root)}")
        
        return self._parsers.get(workspace_root)
    
    def cleanup_parser(self, workspace_root: str):
        """Clean up parser for a specific workspace"""
        workspace_root = os.path.abspath(workspace_root)
        
        with self._lock:
            if workspace_root in self._parsers:
                try:
                    parser = self._parsers[workspace_root]
                    parser.cleanup()
                    del self._parsers[workspace_root]
                except Exception as e:
                    print_current(f"⚠️ Global manager: Error cleaning up parser for {workspace_root}: {e}")
    
    def cleanup_all(self):
        """Clean up all parsers"""
        with self._lock:
            workspaces = list(self._parsers.keys())
            for workspace_root in workspaces:
                try:
                    parser = self._parsers[workspace_root]
                    parser.cleanup()
                except Exception as e:
                    print_current(f"⚠️ Global manager: Error cleaning up parser for {workspace_root}: {e}")
            self._parsers.clear()
    
    def get_stats(self) -> Dict[str, any]:
        """Get manager statistics"""
        with self._lock:
            stats = {
                'total_parsers': len(self._parsers),
                'workspaces': list(self._parsers.keys()),
                'active_background_threads': 0
            }
            
            for parser in self._parsers.values():
                if hasattr(parser, 'background_update_thread') and parser.background_update_thread:
                    if parser.background_update_thread.is_running():
                        stats['active_background_threads'] += 1
            
            return stats


# Global singleton instance
_global_manager = None
_global_manager_lock = threading.Lock()

def get_global_code_index_manager() -> GlobalCodeIndexManager:
    """Get global code index manager instance"""
    global _global_manager
    
    if _global_manager is None:
        with _global_manager_lock:
            if _global_manager is None:
                _global_manager = GlobalCodeIndexManager()
    
    return _global_manager
