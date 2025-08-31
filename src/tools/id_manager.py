#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from .print_system import print_system, print_current
"""
AGIAgent ID Manager
Provides sequential ID generation as an alternative to random UUIDs
"""

import os
import json
import threading
from typing import Dict, Any, Optional
from datetime import datetime


class IDManager:
    """ID Manager - Singleton Pattern"""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls, workspace_root: Optional[str] = None):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, workspace_root: Optional[str] = None):
        if hasattr(self, '_initialized'):
            return
        
        self._initialized = True
        self.workspace_root = workspace_root if workspace_root is not None else "."
        self.state_file = os.path.join(self.workspace_root, ".agia_id_state.json")
        
        # Counters
        self._agent_counter = 1
        self._message_counter = 1
        
        # Thread locks
        self._agent_lock = threading.Lock()
        self._message_lock = threading.Lock()
        
        # Load saved state
        self._load_state()
    
    def _load_state(self):
        """Load ID state"""
        try:
            if os.path.exists(self.state_file):
                with open(self.state_file, 'r', encoding='utf-8') as f:
                    state = json.load(f)
                
                self._agent_counter = state.get('agent_counter', 1)
                self._message_counter = state.get('message_counter', 1)
                
                print_current(f"📊 ID state loaded - Agent: {self._agent_counter}, Message: {self._message_counter}")
        except Exception as e:
            print_current(f"⚠️ Unable to load ID state file: {e}")
            # Use default values
            self._agent_counter = 1
            self._message_counter = 1
    
    def _save_state(self):
        """Save ID state"""
        try:
            os.makedirs(os.path.dirname(self.state_file), exist_ok=True)
            
            state = {
                'agent_counter': self._agent_counter,
                'message_counter': self._message_counter,
                'last_update': datetime.now().isoformat()
            }
            
            with open(self.state_file, 'w', encoding='utf-8') as f:
                json.dump(state, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            print_current(f"⚠️ Unable to save ID state file: {e}")
    
    def generate_agent_id(self, prefix: str = "agent") -> str:
        """Generate agent ID"""
        with self._agent_lock:
            agent_id = f"{prefix}_{self._agent_counter:03d}"
            self._agent_counter += 1
            self._save_state()
            return agent_id
    
    def generate_message_id(self, prefix: str = "msg") -> str:
        """Generate message ID"""
        with self._message_lock:
            message_id = f"{prefix}_{self._message_counter:06d}"
            self._message_counter += 1
            self._save_state()
            return message_id
    
    def get_current_counters(self) -> Dict[str, int]:
        """Get current counter state"""
        return {
            'agent_counter': self._agent_counter,
            'message_counter': self._message_counter
        }
    
    def reset_counters(self, agent_counter: int = 1, message_counter: int = 1):
        """Reset counters (use with caution)"""
        with self._agent_lock, self._message_lock:
            self._agent_counter = agent_counter
            self._message_counter = message_counter
            self._save_state()
            print_current(f"🔄 ID counters reset - Agent: {agent_counter}, Message: {message_counter}")


# Global instance getter function
_global_id_manager = None

def get_id_manager(workspace_root: Optional[str] = None) -> IDManager:
    """Get global ID manager instance"""
    global _global_id_manager
    if _global_id_manager is None:
        _global_id_manager = IDManager(workspace_root)
    return _global_id_manager


def generate_agent_id(prefix: str = "agent", workspace_root: Optional[str] = None) -> str:
    """Convenience function: Generate agent ID"""
    manager = get_id_manager(workspace_root)
    return manager.generate_agent_id(prefix)


def generate_message_id(prefix: str = "msg", workspace_root: Optional[str] = None) -> str:
    """Convenience function: Generate message ID"""
    manager = get_id_manager(workspace_root)
    return manager.generate_message_id(prefix)


def get_id_counters(workspace_root: Optional[str] = None) -> Dict[str, int]:
    """Convenience function: Get current counter state"""
    manager = get_id_manager(workspace_root)
    return manager.get_current_counters() 