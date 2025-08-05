#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AGIBot Print System
Supports prefix-based printing with terminal exclusive locking mechanism
"""

import sys
import threading
import builtins
import time
from typing import Optional, Any
from enum import Enum
from contextlib import contextmanager
try:
    from .output_manager import get_output_manager, MessageImportance
except ImportError:
    from tools.output_manager import get_output_manager, MessageImportance


class PrintType(Enum):
    """Print type enumeration"""
    SYSTEM = "SYS"           # System messages
    MANAGER = "manager"      # Messages from the main AGIBot
    AGENT = "agent"          # Messages from other AGIBots


class PrintSystem:
    """AGIBot print system with terminal exclusive locking mechanism"""
    
    _instance = None
    _creation_lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._creation_lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if hasattr(self, '_initialized'):
            return
        
        self._initialized = True
        # 🔧 Fix thread safety issue: use thread-local storage
        self._thread_local = threading.local()
        # Save original print function
        self._original_print = builtins.print
        
        # Terminal exclusive locking mechanism
        self._terminal_lock = threading.RLock()  # Use reentrant lock
        self._is_streaming = False
        self._streaming_owner = None
        self._pending_messages = []
        self._pending_lock = threading.Lock()
    
    def _format_message(self, prefix: str, *args, **kwargs) -> tuple:
        """Format message with prefix"""
        if args:
            # Add prefix to the first argument
            formatted_args = list(args)
            formatted_args[0] = f"{prefix} {formatted_args[0]}"
            return tuple(formatted_args), kwargs
        else:
            return (prefix,), kwargs
    
    def _safe_print(self, prefix: str, *args, category: str = 'general', 
                   importance: MessageImportance = MessageImportance.INFO, **kwargs):
        """Safe print considering terminal exclusive state and log routing"""
        try:
            thread_id = threading.get_ident()
            
            # Format the complete message for logging
            if args:
                message = str(args[0])
                for arg in args[1:]:
                    message += " " + str(arg)
            else:
                message = ""
            
            # Log to file and check if should show in terminal
            output_manager = get_output_manager()
            should_show_in_terminal = output_manager.log_message(
                f"{prefix} {message}", category, importance
            )
            
            # Only proceed with terminal output if determined necessary
            if not should_show_in_terminal:
                return
            
            # If currently in streaming mode and not the streaming owner thread
            if self._is_streaming and self._streaming_owner != thread_id:
                # Add message to pending queue
                formatted_args, formatted_kwargs = self._format_message(prefix, *args, **kwargs)
                try:
                    with self._pending_lock:
                        self._pending_messages.append((formatted_args, formatted_kwargs))
                except RuntimeError:
                    # Handle case where lock cannot be acquired during shutdown
                    return
                return
            
            # Normal print to terminal
            try:
                with self._terminal_lock:
                    formatted_args, formatted_kwargs = self._format_message(prefix, *args, **kwargs)
                    self._original_print(*formatted_args, **formatted_kwargs)
            except (RuntimeError, OSError, ValueError):
                # Handle cases where terminal lock fails during shutdown
                # or stdout is no longer available
                pass
        except Exception:
            # Catch any other exceptions during shutdown to prevent crashes
            pass
    
    def system_print(self, *args, category: str = 'system', 
                    importance: MessageImportance = MessageImportance.INFO, **kwargs):
        """System message print"""
        self._safe_print("[system]", *args, category=category, importance=importance, **kwargs)

    def manager_print(self, *args, category: str = 'execution', 
                     importance: MessageImportance = MessageImportance.IMPORTANT, **kwargs):
        """Main AGIBot message print"""
        #self._safe_print("[manager]", *args, category=category, importance=importance, **kwargs)
        self._safe_print("", *args, category=category, importance=importance, **kwargs)

    def agent_print(self, agent_id: str, *args, category: str = 'execution',
                   importance: MessageImportance = MessageImportance.IMPORTANT, **kwargs):
        """Specified AGIBot message print"""
        prefix = f"[{agent_id}]"
        self._safe_print(prefix, *args, category=category, importance=importance, **kwargs)
    
    def set_agent_id(self, agent_id: Optional[str]):
        """Set current AGIBot ID (thread-local)"""
        self._thread_local.agent_id = agent_id
    
    def get_agent_id(self) -> Optional[str]:
        """Get current AGIBot ID (thread-local)"""
        return getattr(self._thread_local, 'agent_id', None)
    
    @contextmanager
    def streaming_context(self, show_start_message: bool = True):
        """Streaming output context manager, ensures terminal exclusive during streaming"""
        thread_id = threading.get_ident()
        
        # Wait to acquire terminal control
        acquired = False
        while not acquired:
            with self._terminal_lock:
                if not self._is_streaming:
                    # Acquire control
                    self._is_streaming = True
                    self._streaming_owner = thread_id
                    acquired = True
                    
                    if show_start_message:
                        # Show streaming start message
                        current_id = self.get_agent_id()
                        if current_id is None or current_id == "manager":
                            prefix = "" # "[manager]"
                        else:
                            prefix = f"[{current_id}]"
                        self._original_print(f"{prefix} 🔄 LLM is thinking:")
                    break
            
            if not acquired:
                # If control not acquired, wait briefly and retry
                time.sleep(0.1)
        
        try:
            yield StreamingPrinter(self)
        finally:
            # Release terminal control
            with self._terminal_lock:
                self._is_streaming = False
                self._streaming_owner = None
                
                # Add empty line after streaming completes
                self._original_print("")
                
                # Process pending messages
                self._flush_pending_messages()
    
    def _flush_pending_messages(self):
        """Process all pending messages"""
        with self._pending_lock:
            if not self._pending_messages:
                return
                
            # Print separator to indicate pending messages
            self._original_print("")  # Empty line separator
            
            for args, kwargs in self._pending_messages:
                self._original_print(*args, **kwargs)
            
            self._pending_messages.clear()
    
    def force_print(self, *args, **kwargs):
        """Force print, ignoring streaming state (for emergency messages)"""
        with self._terminal_lock:
            self._original_print(*args, **kwargs)


class StreamingPrinter:
    """Streaming printer for use within streaming_context"""
    
    def __init__(self, print_system: PrintSystem):
        self.print_system = print_system
        self.content_buffer = ""
    
    def write(self, text: str):
        """Write streaming content"""
        if text:
            # Output directly to terminal without prefix
            print(text, end="", flush=True)
            self.content_buffer += text
    
    def newline(self):
        """Add newline"""
        print()
    
    def get_content(self) -> str:
        """Get accumulated content"""
        return self.content_buffer


# Global print system instance
_print_system = None

def get_print_system() -> PrintSystem:
    """Get print system instance"""
    global _print_system
    if _print_system is None:
        _print_system = PrintSystem()
    return _print_system


# Convenience functions
def print_system(*args, **kwargs):
    """Print system message"""
    get_print_system().system_print(*args, **kwargs)


def print_manager(*args, **kwargs):
    """Print main AGIBot message"""
    get_print_system().manager_print(*args, **kwargs)


def print_agent(agent_id: str, *args, **kwargs):
    """Print specified AGIBot message"""
    get_print_system().agent_print(agent_id, *args, **kwargs)


def set_agent_id(agent_id: Optional[str]):
    """Set current AGIBot ID"""
    get_print_system().set_agent_id(agent_id)


def get_agent_id() -> Optional[str]:
    """Get current AGIBot ID"""
    return get_print_system().get_agent_id()


def print_current(*args, category: str = 'execution', 
                 importance: MessageImportance = MessageImportance.IMPORTANT, **kwargs):
    """Print message based on current agent ID setting"""
    ps = get_print_system()
    current_id = ps.get_agent_id()
    
    if current_id is None:
        # If no agent ID set, use manager print
        ps.manager_print(*args, category=category, importance=importance, **kwargs)
    elif current_id == "manager":
        # If manager, use manager print
        ps.manager_print(*args, category=category, importance=importance, **kwargs)
    else:
        # Other cases use agent print
        ps.agent_print(current_id, *args, category=category, importance=importance, **kwargs)


# Specialized logging functions
def print_llm_response(*args, **kwargs):
    """Print LLM response (always shown in terminal)"""
    print_current(*args, category='llm', importance=MessageImportance.IMPORTANT, **kwargs)


def print_tool_call(*args, **kwargs):
    """Print tool call information (always shown in terminal)"""
    print_current(*args, category='tools', importance=MessageImportance.IMPORTANT, **kwargs)


def print_tool_result(*args, **kwargs):
    """Print tool result (always shown in terminal)"""
    print_current(*args, category='tools', importance=MessageImportance.IMPORTANT, **kwargs)


def print_debug(*args, **kwargs):
    """Print debug information (log only, not shown in terminal)"""
    print_current(*args, category='debug', importance=MessageImportance.DEBUG, **kwargs)


def print_system_info(*args, **kwargs):
    """Print system information (log only, not shown in terminal)"""
    print_current(*args, category='system', importance=MessageImportance.INFO, **kwargs)


def print_error(*args, **kwargs):
    """Print error message (always shown in terminal)"""
    print_current(*args, category='general', importance=MessageImportance.CRITICAL, **kwargs)


def streaming_context(show_start_message: bool = True):
    """Streaming output context manager"""
    return get_print_system().streaming_context(show_start_message)


def force_print(*args, **kwargs):
    """Force print (for emergency messages)"""
    get_print_system().force_print(*args, **kwargs)


# Enhanced global print function (optional)
def enhanced_print(*args, **kwargs):
    """Enhanced print function with automatic prefix"""
    ps = get_print_system()
    current_id = ps.get_agent_id()
    if current_id:
        ps.agent_print(current_id, *args, **kwargs)
    else:
        ps.manager_print(*args, **kwargs)


def install_enhanced_print():
    """Install enhanced print system (replace global print)"""
    builtins.print = enhanced_print


def restore_original_print():
    """Restore original print system"""
    ps = get_print_system()
    builtins.print = ps._original_print


# Context managers
class with_system_print:
    """System print context manager"""
    def __init__(self):
        self.ps = get_print_system()
        self.original_print = None
    
    def __enter__(self):
        self.original_print = builtins.print
        builtins.print = self.ps.system_print
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        builtins.print = self.original_print


class with_manager_print:
    """Main AGIBot print context manager"""
    def __init__(self):
        self.ps = get_print_system()
        self.original_print = None
    
    def __enter__(self):
        self.original_print = builtins.print
        builtins.print = self.ps.manager_print
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        builtins.print = self.original_print


class with_agent_print:
    """Specified AGIBot print context manager"""
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.ps = get_print_system()
        self.original_print = None
    
    def __enter__(self):
        self.original_print = builtins.print
        builtins.print = lambda *args, **kwargs: self.ps.agent_print(self.agent_id, *args, **kwargs)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        builtins.print = self.original_print 