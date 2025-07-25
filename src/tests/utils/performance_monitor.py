#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
性能监控工具
提供系统资源使用监控和性能指标收集功能
"""

import psutil
import time
import threading
from typing import Dict, List, Any, Optional
import statistics
import json
from dataclasses import dataclass
from datetime import datetime

@dataclass
class PerformanceSnapshot:
    """性能快照数据类"""
    timestamp: float
    cpu_percent: float
    memory_mb: float
    disk_io_read: int
    disk_io_write: int
    network_io_sent: int
    network_io_recv: int

class PerformanceMonitor:
    """性能监控器"""
    
    def __init__(self, interval: float = 0.5):
        """
        初始化性能监控器
        
        Args:
            interval: 监控间隔（秒）
        """
        self.interval = interval
        self.snapshots: List[PerformanceSnapshot] = []
        self.monitoring = False
        self.monitor_thread = None
        self.start_time = None
        self.initial_memory = None
        
    def __enter__(self):
        """上下文管理器入口"""
        self.start_monitoring()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.stop_monitoring()
    
    def start_monitoring(self):
        """开始监控"""
        if self.monitoring:
            return
        
        self.monitoring = True
        self.start_time = time.time()
        self.initial_memory = psutil.virtual_memory().used
        self.snapshots.clear()
        
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        """停止监控"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)
    
    def _monitor_loop(self):
        """监控循环"""
        while self.monitoring:
            try:
                # 获取系统指标
                cpu_percent = psutil.cpu_percent()
                memory = psutil.virtual_memory()
                disk_io = psutil.disk_io_counters()
                network_io = psutil.net_io_counters()
                
                snapshot = PerformanceSnapshot(
                    timestamp=time.time(),
                    cpu_percent=cpu_percent,
                    memory_mb=memory.used / 1024 / 1024,
                    disk_io_read=disk_io.read_bytes if disk_io else 0,
                    disk_io_write=disk_io.write_bytes if disk_io else 0,
                    network_io_sent=network_io.bytes_sent if network_io else 0,
                    network_io_recv=network_io.bytes_recv if network_io else 0
                )
                
                self.snapshots.append(snapshot)
                
            except Exception as e:
                print(f"Performance monitoring error: {e}")
            
            time.sleep(self.interval)
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取性能统计信息"""
        if not self.snapshots:
            return {}
        
        cpu_values = [s.cpu_percent for s in self.snapshots]
        memory_values = [s.memory_mb for s in self.snapshots]
        
        # 计算I/O差值
        if len(self.snapshots) > 1:
            disk_read_diff = self.snapshots[-1].disk_io_read - self.snapshots[0].disk_io_read
            disk_write_diff = self.snapshots[-1].disk_io_write - self.snapshots[0].disk_io_write
            network_sent_diff = self.snapshots[-1].network_io_sent - self.snapshots[0].network_io_sent
            network_recv_diff = self.snapshots[-1].network_io_recv - self.snapshots[0].network_io_recv
        else:
            disk_read_diff = disk_write_diff = network_sent_diff = network_recv_diff = 0
        
        duration = self.snapshots[-1].timestamp - self.snapshots[0].timestamp if len(self.snapshots) > 1 else 0
        
        return {
            "duration_seconds": duration,
            "sample_count": len(self.snapshots),
            "peak_cpu_percent": max(cpu_values),
            "avg_cpu_percent": statistics.mean(cpu_values),
            "min_cpu_percent": min(cpu_values),
            "peak_memory_mb": max(memory_values),
            "avg_memory_mb": statistics.mean(memory_values),
            "min_memory_mb": min(memory_values),
            "total_disk_read_mb": disk_read_diff / 1024 / 1024,
            "total_disk_write_mb": disk_write_diff / 1024 / 1024,
            "total_network_sent_mb": network_sent_diff / 1024 / 1024,
            "total_network_recv_mb": network_recv_diff / 1024 / 1024
        }
    
    def get_max_memory_usage(self) -> int:
        """获取最大内存使用量（字节）"""
        if not self.snapshots:
            return 0
        return int(max(s.memory_mb for s in self.snapshots) * 1024 * 1024)
    
    def get_memory_growth(self) -> int:
        """获取内存增长量（字节）"""
        if not self.snapshots or not self.initial_memory:
            return 0
        current_memory = int(self.snapshots[-1].memory_mb * 1024 * 1024)
        return max(0, current_memory - self.initial_memory)
    
    def export_to_json(self, file_path: str):
        """导出性能数据到JSON文件"""
        data = {
            "metadata": {
                "start_time": self.start_time,
                "duration": time.time() - self.start_time if self.start_time else 0,
                "interval": self.interval,
                "sample_count": len(self.snapshots)
            },
            "statistics": self.get_statistics(),
            "snapshots": [
                {
                    "timestamp": s.timestamp,
                    "cpu_percent": s.cpu_percent,
                    "memory_mb": s.memory_mb,
                    "disk_io_read": s.disk_io_read,
                    "disk_io_write": s.disk_io_write,
                    "network_io_sent": s.network_io_sent,
                    "network_io_recv": s.network_io_recv
                }
                for s in self.snapshots
            ]
        }
        
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)

class ResourceTracker:
    """资源跟踪器"""
    
    def __init__(self):
        self.process = psutil.Process()
        self.start_stats = None
        self.peak_stats = {}
        self.monitoring = False
        self.monitor_thread = None
        
    def __enter__(self):
        """上下文管理器入口"""
        self.start_tracking()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.stop_tracking()
    
    def start_tracking(self):
        """开始跟踪"""
        if self.monitoring:
            return
        
        self.monitoring = True
        self.start_stats = self._get_current_stats()
        self.peak_stats = self.start_stats.copy()
        
        self.monitor_thread = threading.Thread(target=self._tracking_loop, daemon=True)
        self.monitor_thread.start()
    
    def stop_tracking(self):
        """停止跟踪"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)
    
    def _get_current_stats(self) -> Dict[str, float]:
        """获取当前进程统计信息"""
        try:
            memory_info = self.process.memory_info()
            cpu_percent = self.process.cpu_percent()
            
            return {
                "cpu_percent": cpu_percent,
                "memory_rss": memory_info.rss / 1024 / 1024,  # MB
                "memory_vms": memory_info.vms / 1024 / 1024,  # MB
                "num_threads": self.process.num_threads(),
                "num_fds": self.process.num_fds() if hasattr(self.process, 'num_fds') else 0
            }
        except Exception:
            return {
                "cpu_percent": 0.0,
                "memory_rss": 0.0,
                "memory_vms": 0.0,
                "num_threads": 0,
                "num_fds": 0
            }
    
    def _tracking_loop(self):
        """跟踪循环"""
        while self.monitoring:
            try:
                current_stats = self._get_current_stats()
                
                # 更新峰值统计
                for key, value in current_stats.items():
                    if value > self.peak_stats.get(key, 0):
                        self.peak_stats[key] = value
                
            except Exception as e:
                print(f"Resource tracking error: {e}")
            
            time.sleep(0.1)  # 更频繁的监控
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取资源统计信息"""
        current_stats = self._get_current_stats()
        
        if not self.start_stats:
            return current_stats
        
        return {
            "peak_cpu_percent": self.peak_stats.get("cpu_percent", 0),
            "peak_memory_mb": self.peak_stats.get("memory_rss", 0),
            "peak_virtual_memory_mb": self.peak_stats.get("memory_vms", 0),
            "peak_threads": self.peak_stats.get("num_threads", 0),
            "peak_file_descriptors": self.peak_stats.get("num_fds", 0),
            "current_cpu_percent": current_stats.get("cpu_percent", 0),
            "current_memory_mb": current_stats.get("memory_rss", 0),
            "avg_cpu_percent": (self.start_stats.get("cpu_percent", 0) + current_stats.get("cpu_percent", 0)) / 2,
            "avg_memory_mb": (self.start_stats.get("memory_rss", 0) + current_stats.get("memory_rss", 0)) / 2,
            "memory_growth_mb": current_stats.get("memory_rss", 0) - self.start_stats.get("memory_rss", 0)
        }

class BenchmarkRunner:
    """基准测试运行器"""
    
    def __init__(self):
        self.results = []
    
    def run_benchmark(self, name: str, func: callable, *args, **kwargs) -> Dict[str, Any]:
        """运行基准测试"""
        monitor = PerformanceMonitor(interval=0.1)
        tracker = ResourceTracker()
        
        with monitor, tracker:
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                success = True
                error = None
            except Exception as e:
                result = None
                success = False
                error = str(e)
            
            end_time = time.time()
        
        # 收集性能数据
        performance_stats = monitor.get_statistics()
        resource_stats = tracker.get_statistics()
        
        benchmark_result = {
            "name": name,
            "success": success,
            "error": error,
            "execution_time": end_time - start_time,
            "timestamp": datetime.now().isoformat(),
            "performance": performance_stats,
            "resources": resource_stats,
            "result": result
        }
        
        self.results.append(benchmark_result)
        return benchmark_result
    
    def get_all_results(self) -> List[Dict[str, Any]]:
        """获取所有基准测试结果"""
        return self.results
    
    def get_summary(self) -> Dict[str, Any]:
        """获取基准测试摘要"""
        if not self.results:
            return {}
        
        successful_tests = [r for r in self.results if r["success"]]
        failed_tests = [r for r in self.results if not r["success"]]
        
        execution_times = [r["execution_time"] for r in successful_tests]
        
        return {
            "total_tests": len(self.results),
            "successful_tests": len(successful_tests),
            "failed_tests": len(failed_tests),
            "success_rate": len(successful_tests) / len(self.results) if self.results else 0,
            "avg_execution_time": statistics.mean(execution_times) if execution_times else 0,
            "min_execution_time": min(execution_times) if execution_times else 0,
            "max_execution_time": max(execution_times) if execution_times else 0,
            "total_execution_time": sum(execution_times) if execution_times else 0
        }
    
    def export_results(self, file_path: str):
        """导出基准测试结果"""
        export_data = {
            "summary": self.get_summary(),
            "results": self.results,
            "export_time": datetime.now().isoformat()
        }
        
        with open(file_path, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)

class MemoryProfiler:
    """内存分析器"""
    
    def __init__(self):
        self.snapshots = []
        self.tracking = False
    
    def start_profiling(self):
        """开始内存分析"""
        self.tracking = True
        self.snapshots.clear()
        self._take_snapshot("start")
    
    def stop_profiling(self):
        """停止内存分析"""
        if self.tracking:
            self._take_snapshot("stop")
            self.tracking = False
    
    def take_snapshot(self, label: str = None):
        """手动获取内存快照"""
        if self.tracking:
            self._take_snapshot(label or f"snapshot_{len(self.snapshots)}")
    
    def _take_snapshot(self, label: str):
        """内部快照方法"""
        try:
            import gc
            gc.collect()  # 强制垃圾回收
            
            process = psutil.Process()
            memory_info = process.memory_info()
            
            snapshot = {
                "label": label,
                "timestamp": time.time(),
                "rss_mb": memory_info.rss / 1024 / 1024,
                "vms_mb": memory_info.vms / 1024 / 1024,
                "percent": process.memory_percent(),
                "available_mb": psutil.virtual_memory().available / 1024 / 1024
            }
            
            self.snapshots.append(snapshot)
            
        except Exception as e:
            print(f"Memory profiling error: {e}")
    
    def get_memory_growth(self) -> Dict[str, float]:
        """获取内存增长情况"""
        if len(self.snapshots) < 2:
            return {}
        
        start = self.snapshots[0]
        end = self.snapshots[-1]
        
        return {
            "rss_growth_mb": end["rss_mb"] - start["rss_mb"],
            "vms_growth_mb": end["vms_mb"] - start["vms_mb"],
            "percent_growth": end["percent"] - start["percent"],
            "peak_rss_mb": max(s["rss_mb"] for s in self.snapshots),
            "peak_vms_mb": max(s["vms_mb"] for s in self.snapshots)
        }
    
    def detect_memory_leaks(self, threshold_mb: float = 10.0) -> bool:
        """检测内存泄漏"""
        growth = self.get_memory_growth()
        return growth.get("rss_growth_mb", 0) > threshold_mb
    
    def export_profile(self, file_path: str):
        """导出内存分析结果"""
        profile_data = {
            "snapshots": self.snapshots,
            "growth_analysis": self.get_memory_growth(),
            "memory_leak_detected": self.detect_memory_leaks(),
            "export_time": datetime.now().isoformat()
        }
        
        with open(file_path, 'w') as f:
            json.dump(profile_data, f, indent=2) 