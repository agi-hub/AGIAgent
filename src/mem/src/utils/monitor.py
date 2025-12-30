#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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

Performance Monitoring Module
"""

import time
import psutil
import threading
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
import json
import os


def _load_monitor_config() -> Dict[str, Any]:
    """
    Simple configuration loader for monitor (avoid circular import)
    
    Returns:
        Configuration dictionary
    """
    config = {}
    
    # Default config files in priority order
    config_files = [
        "config/config.txt",
        "config/config_memory.txt"
    ]
    
    # Try to find project root and config files
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_roots = [
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))),  # 4 levels up
        os.path.dirname(os.path.dirname(os.path.dirname(current_dir))),  # 3 levels up
        os.path.dirname(os.path.dirname(current_dir)),  # 2 levels up
        os.path.dirname(current_dir),  # 1 level up
        current_dir,  # current dir
        os.getcwd()  # working directory
    ]
    
    config_files_found = []
    for root in project_roots:
        for config_file in config_files:
            full_path = os.path.join(root, config_file)
            if os.path.exists(full_path) and full_path not in config_files_found:
                config_files_found.append(full_path)
    
    # Also try the simple config.txt in various locations
    simple_config_files = ["config.txt", "config/config.txt"]
    for root in project_roots:
        for config_file in simple_config_files:
            full_path = os.path.join(root, config_file)
            if os.path.exists(full_path) and full_path not in config_files_found:
                config_files_found.append(full_path)
    
    # Load configuration from files (later files override earlier ones)
    for config_file in config_files_found:
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    # Skip comments and empty lines
                    if not line or line.startswith('#'):
                        continue
                    # Parse key=value lines
                    if '=' in line:
                        key, value = line.split('=', 1)
                        key = key.strip()
                        value = value.strip()
                        config[key] = value
        except Exception:
            # Continue reading other files even if one fails
            pass
    
    return config


@dataclass
class PerformanceMetrics:
    """Performance metrics data class"""
    timestamp: float
    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    disk_usage_percent: float
    active_threads: int
    operation_count: int = 0
    error_count: int = 0
    avg_response_time: float = 0.0


class PerformanceMonitor:
    """Performance monitor"""

    def __init__(self, log_file: str = "logs/performance.log", enabled: Optional[bool] = None):
        """
        Initialize performance monitor

        Args:
            log_file: Performance log file path
            enabled: Whether to enable monitoring (None to auto-detect from config)
        """
        self.log_file = log_file
        self.metrics_history: List[PerformanceMetrics] = []
        self.max_history_size = 1000
        self.lock = threading.Lock()
        self.monitor_thread = None
        
        # Load configuration to determine if monitoring is enabled
        if enabled is None:
            config = _load_monitor_config()
            # Check for performance monitoring enable/disable flags
            performance_enabled_str = config.get("performance_monitor_enabled", 
                                                config.get("monitor_enabled", "True"))
            self.enabled = performance_enabled_str.lower() in ('true', '1', 'yes', 'on')
        else:
            self.enabled = enabled

        # Operation statistics
        self.operation_stats = {
            "total_operations": 0,
            "successful_operations": 0,
            "failed_operations": 0,
            "avg_response_time": 0.0
        }

        # Only start monitoring if enabled
        if self.enabled:
            # Ensure log directory exists
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            
            # Start monitoring thread
            self.monitoring = True
            self.monitor_thread = threading.Thread(
                target=self._monitor_loop, daemon=True)
            self.monitor_thread.start()
        else:
            self.monitoring = False

    def _monitor_loop(self):
        """Monitoring loop"""
        while self.monitoring and self.enabled:
            try:
                metrics = self._collect_metrics()
                if metrics is not None:
                    self._add_metrics(metrics)
                time.sleep(60)  # Collect every minute
            except Exception as e:
                # Only print error if monitoring is enabled
                if self.enabled:
                    print(f"Performance monitoring error: {e}")

    def _collect_metrics(self) -> Optional[PerformanceMetrics]:
        """Collect performance metrics"""
        if not self.enabled:
            return None
            
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')

        return PerformanceMetrics(
            timestamp=time.time(),
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            memory_used_mb=memory.used / 1024 / 1024,
            disk_usage_percent=disk.percent,
            active_threads=threading.active_count(),
            operation_count=self.operation_stats["total_operations"],
            error_count=self.operation_stats["failed_operations"],
            avg_response_time=self.operation_stats["avg_response_time"]
        )

    def _add_metrics(self, metrics: PerformanceMetrics):
        """Add metrics to history and log"""
        if not self.enabled or metrics is None:
            return
            
        with self.lock:
            self.metrics_history.append(metrics)
            if len(self.metrics_history) > self.max_history_size:
                self.metrics_history.pop(0)

        # Write to log file
        self._write_to_log(metrics)

    def _write_to_log(self, metrics: PerformanceMetrics):
        """Write metrics to log file"""
        if not self.enabled:
            return
            
        try:
            log_entry = {
                "timestamp": datetime.fromtimestamp(metrics.timestamp).isoformat(),
                "cpu_percent": metrics.cpu_percent,
                "memory_percent": metrics.memory_percent,
                "memory_used_mb": metrics.memory_used_mb,
                "disk_usage_percent": metrics.disk_usage_percent,
                "active_threads": metrics.active_threads,
                "operation_count": metrics.operation_count,
                "error_count": metrics.error_count,
                "avg_response_time": metrics.avg_response_time
            }

            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(log_entry) + '\n')

        except Exception as e:
            pass  # Silently ignore log write errors

    def record_operation(self, operation_name: str, duration: float, success: bool = True):
        """Record operation metrics"""
        if not self.enabled:
            return
            
        with self.lock:
            self.operation_stats["total_operations"] += 1
            if success:
                self.operation_stats["successful_operations"] += 1
            else:
                self.operation_stats["failed_operations"] += 1

            # Update average response time
            total_time = (self.operation_stats["avg_response_time"] * 
                         (self.operation_stats["total_operations"] - 1) + duration)
            self.operation_stats["avg_response_time"] = total_time / self.operation_stats["total_operations"]

    def get_metrics(self) -> List[PerformanceMetrics]:
        """Get metrics history"""
        if not self.enabled:
            return []
            
        with self.lock:
            return self.metrics_history.copy()

    def get_latest_metrics(self) -> Optional[PerformanceMetrics]:
        """Get latest metrics"""
        if not self.enabled:
            return None
            
        with self.lock:
            return self.metrics_history[-1] if self.metrics_history else None

    def get_operation_stats(self) -> Dict[str, Any]:
        """Get operation statistics"""
        if not self.enabled:
            return {"enabled": False}
            
        with self.lock:
            stats = self.operation_stats.copy()
            stats["enabled"] = True
            return stats

    def stop_monitoring(self):
        """Stop monitoring"""
        self.monitoring = False
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=2)


class OperationTimer:
    """Operation timer context manager"""

    def __init__(self, monitor: PerformanceMonitor, operation_name: str):
        self.monitor = monitor
        self.operation_name = operation_name
        self.start_time = None

    def __enter__(self):
        if self.monitor.enabled:
            self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.monitor.enabled and self.start_time:
            duration = time.time() - self.start_time
            success = exc_type is None
            self.monitor.record_operation(self.operation_name, duration, success)


# Global performance monitor instance
_performance_monitor = None


def get_performance_monitor() -> PerformanceMonitor:
    """
    Get performance monitor instance

    Returns:
        Performance monitor instance
    """
    global _performance_monitor

    if _performance_monitor is None:
        _performance_monitor = PerformanceMonitor()

    return _performance_monitor


def monitor_operation(operation_name: str):
    """
    Operation monitoring decorator

    Args:
        operation_name: Operation name
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            monitor = get_performance_monitor()
            # Only use timer if monitoring is enabled
            if monitor.enabled:
                with OperationTimer(monitor, operation_name):
                    return func(*args, **kwargs)
            else:
                # Just execute the function without monitoring
                return func(*args, **kwargs)
        return wrapper
    return decorator
