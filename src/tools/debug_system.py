#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from .print_system import print_system, print_current
"""
Debug System Module - Used for diagnosing program freezing issues

This module provides the following features:
1. Capture signals (like Ctrl+C) and display current stack
2. Real-time monitoring of program status
3. Record key execution points
4. Provide debugging reports
"""

import signal
import sys
import traceback
import threading
import time
import os
import psutil
from datetime import datetime
from typing import Dict, Any, List, Optional
import json

class DebugSystem:
    """Main debug system class"""
    
    def __init__(self, enable_stack_trace: bool = True, 
                 enable_memory_monitor: bool = True,
                 enable_execution_tracker: bool = True,
                 show_activation_message: bool = False):
        """
        Initialize debug system
        
        Args:
            enable_stack_trace: Whether to enable stack tracing
            enable_memory_monitor: Whether to enable memory monitoring
            enable_execution_tracker: Whether to enable execution tracking
            show_activation_message: Whether to show debug system activation message
        """
        self.enable_stack_trace = enable_stack_trace
        self.enable_memory_monitor = enable_memory_monitor
        self.enable_execution_tracker = enable_execution_tracker
        self.show_activation_message = show_activation_message
        
        # Execution tracker
        self.execution_stack = []
        self.current_operation = "Program startup"
        self.operation_start_time = time.time()
        
        # Memory monitoring
        self.memory_snapshots = []
        self.memory_monitor_thread = None
        self.memory_monitor_active = False
        self.shutdown_flag = False  # Add shutdown flag
        
        # Signal handlers
        self.original_sigint_handler = None
        self.original_sigterm_handler = None
        
        # Debug information collection
        self.debug_log = []
        
    def install_signal_handlers(self):
        """Install signal handlers"""
        if self.enable_stack_trace:
            # Save original handlers
            self.original_sigint_handler = signal.signal(signal.SIGINT, self._signal_handler)
            self.original_sigterm_handler = signal.signal(signal.SIGTERM, self._signal_handler)
            if self.show_activation_message:
                print_current("🔧 Debug system activated - Press Ctrl+C to show debug information")
    
    def _signal_handler(self, signum, frame):
        """Signal handler - Display thread stacks then exit"""
        print_current("\n🚨 Interrupt signal received, displaying thread stacks:")
        
        # Only show stack trace
        self._show_stack_trace(frame)
        
        # Exit directly
        sys.exit(1)
    
    def _show_current_status(self):
        """Display current execution status"""
        print_current(f"📍 Current operation: {self.current_operation}")
        current_time = time.time()
        duration = current_time - self.operation_start_time
        print_current(f"⏱️ Execution time: {duration:.2f} seconds")
        
        if self.execution_stack:
            print_current(f"📚 Execution stack depth: {len(self.execution_stack)}")
            print_current(f"🔄 Execution path: {' -> '.join(self.execution_stack[-5:])}")  # Show last 5 operations
    
    def _show_stack_trace(self, frame):
        """Display Python stack trace"""
        print_current("\n📋 Python stack trace:")
        print_current("-" * 60)
        
        # Get stacks for all threads
        for thread_id, frame_dict in sys._current_frames().items():
            thread_name = "Unknown thread"
            for thread in threading.enumerate():
                if thread.ident == thread_id:
                    thread_name = thread.name
                    break
            
            print_current(f"\n🧵 Thread: {thread_name} (ID: {thread_id})")
            print_current("Stack:")
            
            # Format stack information
            stack_lines = traceback.format_stack(frame_dict)
            for i, line in enumerate(stack_lines[-10:]):  # Show last 10 stack frames
                lines = line.strip().split('\n')
                for sub_line in lines:
                    if sub_line.strip():
                        print_current(f"  {sub_line}")
        
        print_current("-" * 60)
    
    def _show_execution_history(self):
        """Display execution history"""
        print_current("\n📈 Execution history (last 10 operations):")
        print_current("-" * 60)
        
        recent_ops = self.execution_stack[-10:] if len(self.execution_stack) > 10 else self.execution_stack
        for i, op in enumerate(recent_ops, 1):
            print_current(f"  {i}. {op}")
        
        if not self.execution_stack:
            print_current("  (No execution history)")
        
        print_current("-" * 60)
    
    def _show_memory_status(self):
        """Display memory status"""
        print_current("\n💾 Memory status:")
        print_current("-" * 60)
        
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            memory_percent = process.memory_percent()
            
            print_current(f"  RSS memory: {memory_info.rss / 1024 / 1024:.2f} MB")
            print_current(f"  VMS memory: {memory_info.vms / 1024 / 1024:.2f} MB")
            print_current(f"  Memory usage: {memory_percent:.2f}%")
            
            # Show memory trend
            if len(self.memory_snapshots) > 1:
                current_memory = memory_info.rss
                previous_memory = self.memory_snapshots[-1]['rss']
                trend = current_memory - previous_memory
                trend_mb = trend / 1024 / 1024
                
                if trend > 0:
                    print_current(f"  Memory trend: ↗️ +{trend_mb:.2f} MB")
                elif trend < 0:
                    print_current(f"  Memory trend: ↘️ {trend_mb:.2f} MB")
                else:
                    print_current(f"  Memory trend: ➡️ Stable")
        
        except Exception as e:
            print_current(f"  ❌ Cannot get memory information: {e}")
        
        print_current("-" * 60)
    
    def _show_process_info(self):
        """Display process information"""
        print_current("\n🔍 Process information:")
        print_current("-" * 60)
        
        try:
            process = psutil.Process()
            print_current(f"  Process ID: {process.pid}")
            print_current(f"  Process name: {process.name()}")
            print_current(f"  Start time: {datetime.fromtimestamp(process.create_time()).strftime('%Y-%m-%d %H:%M:%S')}")
            print_current(f"  CPU usage: {process.cpu_percent():.2f}%")
            
            # Show number of open files
            try:
                open_files = len(process.open_files())
                print_current(f"  Open files: {open_files}")
            except:
                pass
            
            # Show thread count
            print_current(f"  Thread count: {process.num_threads()}")
            
        except Exception as e:
            print_current(f"  ❌ Cannot get process information: {e}")
        
        print_current("-" * 60)
    
    def _save_debug_report(self):
        """Save debug report to file"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_file = f"debug_report_{timestamp}.json"
            
            report = {
                "timestamp": timestamp,
                "current_operation": self.current_operation,
                "operation_duration": time.time() - self.operation_start_time,
                "execution_stack": self.execution_stack.copy(),
                "debug_log": self.debug_log.copy(),
                "memory_snapshots": self.memory_snapshots.copy(),
                "process_info": self._get_process_info_dict(),
                "python_version": sys.version,
                "platform": sys.platform
            }
            
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            
            print_current(f"📄 Debug report saved to: {report_file}")
            
        except Exception as e:
            print_current(f"❌ Failed to save debug report: {e}")
    
    def _save_debug_report_silent(self):
        """Save debug report to file silently (without displaying any information)"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_file = f"debug_report_{timestamp}.json"
            
            report = {
                "timestamp": timestamp,
                "current_operation": self.current_operation,
                "operation_duration": time.time() - self.operation_start_time,
                "execution_stack": self.execution_stack.copy(),
                "debug_log": self.debug_log.copy(),
                "memory_snapshots": self.memory_snapshots.copy(),
                "process_info": self._get_process_info_dict(),
                "python_version": sys.version,
                "platform": sys.platform
            }
            
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            
        except Exception:
            pass  # Handle errors silently
    
    def _get_process_info_dict(self) -> Dict[str, Any]:
        """Get process information dictionary"""
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            
            return {
                "pid": process.pid,
                "name": process.name(),
                "cpu_percent": process.cpu_percent(),
                "memory_rss": memory_info.rss,
                "memory_vms": memory_info.vms,
                "memory_percent": process.memory_percent(),
                "num_threads": process.num_threads(),
                "create_time": process.create_time()
            }
        except:
            return {}
    
    def _enter_debug_mode(self):
        """Enter interactive debug mode"""
        print_current("\n🔍 Debug mode - Enter 'help' to see available commands")
        
        while True:
            try:
                cmd = input("debug> ").strip()
                
                if cmd == 'help':
                    print_current("Available commands:")
                    print_current("  stack - Show current stack")
                    print_current("  memory - Show memory information")
                    print_current("  threads - Show thread information")
                    print_current("  history - Show execution history")
                    print_current("  vars - Show local variables")
                    print_current("  continue/c - Continue execution")
                    print_current("  quit/q - Exit program")
                    
                elif cmd == 'stack':
                    self._show_stack_trace(None)
                    
                elif cmd == 'memory':
                    self._show_memory_status()
                    
                elif cmd == 'threads':
                    self._show_thread_info()
                    
                elif cmd == 'history':
                    self._show_execution_history()
                    
                elif cmd == 'vars':
                    self._show_variables()
                    
                elif cmd in ['continue', 'c']:
                    print_current("▶️ Continuing execution...")
                    break
                    
                elif cmd in ['quit', 'q']:
                    print_current("👋 Exiting program")
                    sys.exit(1)
                    
                else:
                    print_current(f"Unknown command: {cmd}, enter 'help' to see available commands")
                    
            except (EOFError, KeyboardInterrupt):
                print_current("\n👋 Exiting debug mode")
                break
    
    def _show_thread_info(self):
        """Display thread information"""
        print_current("\n🧵 Thread information:")
        print_current("-" * 60)
        
        for thread in threading.enumerate():
            status = "Active" if thread.is_alive() else "Stopped"
            daemon = "Daemon thread" if thread.daemon else "Normal thread"
            print_current(f"  {thread.name} (ID: {thread.ident}) - {status} - {daemon}")
        
        print_current("-" * 60)
    
    def _show_variables(self):
        """Display variables in current scope"""
        print_current("\n📋 Variable information:")
        print_current("-" * 60)
        print_current("  (Local variables can only be shown at breakpoints)")
        print_current("-" * 60)
    
    def start_memory_monitor(self, interval: float = 5.0):
        """Start memory monitoring"""
        if not self.enable_memory_monitor or self.memory_monitor_active:
            return
        
        self.memory_monitor_active = True
        self.memory_monitor_thread = threading.Thread(
            target=self._memory_monitor_loop,
            args=(interval,),
            name="MemoryMonitor",
            daemon=True
        )
        self.memory_monitor_thread.start()
    
    def _memory_monitor_loop(self, interval: float):
        """Memory monitoring loop"""
        while self.memory_monitor_active and not self.shutdown_flag:
            try:
                process = psutil.Process()
                memory_info = process.memory_info()
                
                snapshot = {
                    "timestamp": time.time(),
                    "rss": memory_info.rss,
                    "vms": memory_info.vms,
                    "percent": process.memory_percent()
                }
                
                self.memory_snapshots.append(snapshot)
                
                # Keep last 100 snapshots
                if len(self.memory_snapshots) > 100:
                    self.memory_snapshots = self.memory_snapshots[-100:]
                
                # Check shutdown condition more frequently (every 0.5 seconds)
                sleep_time = min(interval, 0.5)
                for _ in range(int(interval / sleep_time)):
                    if not self.memory_monitor_active or self.shutdown_flag:
                        return
                    time.sleep(sleep_time)
                
            except Exception:
                break
    
    def stop_memory_monitor(self):
        """Stop memory monitoring"""
        self.memory_monitor_active = False
        if self.memory_monitor_thread:
            self.memory_monitor_thread.join(timeout=1.0)
    
    def track_operation(self, operation_name: str):
        """Track operation"""
        if self.enable_execution_tracker:
            self.execution_stack.append(operation_name)
            self.current_operation = operation_name
            self.operation_start_time = time.time()
            
            # Record to debug log
            self.debug_log.append({
                "timestamp": time.time(),
                "operation": operation_name,
                "type": "operation_start"
            })
            
            # Keep execution stack at reasonable size
            if len(self.execution_stack) > 1000:
                self.execution_stack = self.execution_stack[-500:]  # Keep last 500
    
    def finish_operation(self, operation_name: Optional[str] = None):
        """Finish operation"""
        if self.enable_execution_tracker and self.execution_stack:
            # 🔧 MULTI-INSTANCE FIX: Disable strict operation name matching in multi-instance/concurrent environments
            # because multiple AGIBot instances may share the same debug system, causing execution stack confusion
            if operation_name and self.execution_stack[-1] != operation_name:
                # Only record debug log, don't print warning to avoid false positives in normal multi-instance execution
                self.log_event("operation_mismatch", 
                    f"Operation name mismatch: expected {operation_name}, actual {self.execution_stack[-1]}")
            
            finished_op = self.execution_stack.pop()
            duration = time.time() - self.operation_start_time
            
            # Record to debug log
            self.debug_log.append({
                "timestamp": time.time(),
                "operation": finished_op,
                "duration": duration,
                "type": "operation_finish"
            })
            
            # Update current operation
            if self.execution_stack:
                self.current_operation = self.execution_stack[-1]
            else:
                self.current_operation = "Idle"
            
            self.operation_start_time = time.time()
    
    def log_event(self, event_type: str, message: str, **kwargs):
        """Log event"""
        event = {
            "timestamp": time.time(),
            "type": event_type,
            "message": message,
            **kwargs
        }
        self.debug_log.append(event)
        
        # Keep log at reasonable size
        if len(self.debug_log) > 10000:
            self.debug_log = self.debug_log[-5000:]  # Keep last 5000 entries
    
    def restore_signal_handlers(self):
        """Restore original signal handlers"""
        if self.original_sigint_handler:
            signal.signal(signal.SIGINT, self.original_sigint_handler)
        if self.original_sigterm_handler:
            signal.signal(signal.SIGTERM, self.original_sigterm_handler)
    
    def cleanup(self):
        """Clean up debug system"""
        self.shutdown_flag = True  # Set shutdown flag first
        self.stop_memory_monitor()
        self.restore_signal_handlers()


# Global debug system instance
_debug_system = None

def get_debug_system() -> DebugSystem:
    """Get global debug system instance"""
    global _debug_system
    if _debug_system is None:
        _debug_system = DebugSystem()
    return _debug_system

def install_debug_system(enable_stack_trace: bool = True,
                        enable_memory_monitor: bool = True,
                        enable_execution_tracker: bool = True,
                        show_activation_message: bool = False):
    """Install debug system"""
    global _debug_system
    _debug_system = DebugSystem(
        enable_stack_trace=enable_stack_trace,
        enable_memory_monitor=enable_memory_monitor,
        enable_execution_tracker=enable_execution_tracker,
        show_activation_message=show_activation_message
    )
    _debug_system.install_signal_handlers()
    if enable_memory_monitor:
        _debug_system.start_memory_monitor()
    return _debug_system

def track_operation(operation_name: str):
    """Track operation - convenience function"""
    debug_system = get_debug_system()
    debug_system.track_operation(operation_name)

def finish_operation(operation_name: Optional[str] = None):
    """Finish operation - convenience function"""
    debug_system = get_debug_system()
    debug_system.finish_operation(operation_name)

def log_debug_event(event_type: str, message: str, **kwargs):
    """Log debug event - convenience function"""
    debug_system = get_debug_system()
    debug_system.log_event(event_type, message, **kwargs) 