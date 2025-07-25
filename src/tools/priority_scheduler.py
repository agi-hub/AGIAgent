#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Priority-based Multi-Agent Scheduler

This module implements a fair scheduling system for multi-agent execution
using priority queues and resource management.
"""

import threading
import time
import queue
from typing import Dict, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import statistics
from collections import deque, defaultdict
import uuid

from .print_system import print_current, print_agent


@dataclass
class AgentTask:
    """Agent task data structure"""
    task_id: str
    agent_id: str
    task_func: Callable
    priority: float
    submit_time: float
    estimated_duration: float = 30.0  # Estimated execution time (seconds)
    max_retries: int = 3
    retry_count: int = 0
    dependencies: list = field(default_factory=list)
    
    def __lt__(self, other):
        """For priority queue sorting (lower values have higher priority)"""
        return self.priority < other.priority


@dataclass 
class AgentMetrics:
    """Agent performance metrics"""
    agent_id: str
    total_executions: int = 0
    total_execution_time: float = 0.0
    avg_execution_time: float = 0.0
    success_count: int = 0
    failure_count: int = 0
    last_execution_time: Optional[float] = None
    recent_execution_times: deque = field(default_factory=lambda: deque(maxlen=10))
    resource_usage_score: float = 0.0
    fairness_score: float = 1.0  # Fairness score, higher values have higher priority
    
    def update_execution(self, execution_time: float, success: bool):
        """Update execution metrics"""
        self.total_executions += 1
        self.total_execution_time += execution_time
        self.avg_execution_time = self.total_execution_time / self.total_executions
        self.recent_execution_times.append(execution_time)
        self.last_execution_time = time.time()
        
        if success:
            self.success_count += 1
        else:
            self.failure_count += 1
            
        # Calculate resource usage score (average of recent execution times)
        if self.recent_execution_times:
            self.resource_usage_score = statistics.mean(self.recent_execution_times)
    
    def calculate_dynamic_priority(self, current_time: float, base_priority: float = 5.0, 
                                  avg_executions: float = 0.0, min_executions: int = 0) -> float:
        """
        Calculate dynamic priority with aggressive fairness enforcement
        
        Args:
            current_time: Current timestamp
            base_priority: Base priority value
            avg_executions: Average executions across all agents
            min_executions: Minimum executions among all agents
        """
        priority = base_priority
        
        # Aggressive fairness adjustment: heavily penalize agents ahead of average
        if avg_executions > 0:
            execution_ratio = self.total_executions / max(avg_executions, 1)
            if execution_ratio > 1.1:  # Penalize if more than 10% above average
                fairness_penalty = (execution_ratio - 1.0) * 5.0
                priority += fairness_penalty
            elif execution_ratio < 0.9:  # Reward if more than 10% below average
                fairness_bonus = (1.0 - execution_ratio) * 4.0
                priority -= fairness_bonus
        
        # Aggressive wait time adjustment
        if self.last_execution_time:
            wait_time = current_time - self.last_execution_time
            if wait_time > 3:  # Boost priority if waited more than 3 seconds
                time_bonus = min(wait_time / 3, 8.0)
                priority -= time_bonus
                
                # Extra bonus for very long waits
                if wait_time > 10:
                    extra_bonus = min((wait_time - 10) / 5, 5.0)
                    priority -= extra_bonus
        else:
            # Agents never executed get highest priority
            priority -= 15.0  # Increased from 10.0 to 15.0
        
        # Aggressive execution gap penalty
        if min_executions >= 0 and self.total_executions > min_executions + 1:
            gap_penalty = (self.total_executions - min_executions) * 3.0
            priority += gap_penalty
            
            # Extra penalty for large gaps
            if self.total_executions > min_executions + 2:
                extra_penalty = (self.total_executions - min_executions - 2) * 2.0
                priority += extra_penalty
        
        # Success rate adjustment (less aggressive than fairness)
        if self.total_executions > 0:
            success_rate = self.success_count / self.total_executions
            priority -= (success_rate - 0.5) * 0.2  # Reduced influence
        
        # Apply fairness score multiplier
        priority *= (2.0 - self.fairness_score)  # fairness_score 1.0 = no change, 2.0 = half priority
        
        # Ensure minimum priority for extremely penalized agents
        return max(priority, 0.01)

    def emergency_restart(self):
        """Emergency restart the scheduler to recover from a blocked state"""
        print_current("🚨 EMERGENCY RESTART: Attempting to recover from deadlock...")
        
        try:
            # Stop the scheduler
            old_active = self.scheduler_active
            self.stop()
            
            # Wait a second for threads to clean up
            time.sleep(1)
            
            # Clean up potentially stuck tasks in the queue
            with self.metrics_lock:
                queue_size = self.task_queue.qsize()
                if queue_size > 0:
                    print_current(f"🚨 Clearing {queue_size} potentially stuck tasks from queue")
                    old_queue = self.task_queue
                    self.task_queue = queue.PriorityQueue()
                    
                    # Try to transfer up to 10 non-stuck tasks
                    transferred = 0
                    while not old_queue.empty() and transferred < 10:
                        try:
                            task = old_queue.get_nowait()
                            self.task_queue.put(task)
                            transferred += 1
                        except queue.Empty:
                            break
                    
                    print_current(f"🚨 Transferred {transferred} tasks to new queue")
                
                # Clear tracking info
                self.active_task_start_times.clear()
                self.worker_last_activity.clear()
            
            # Restart the scheduler
            if old_active:
                self.start()
                print_current("🚨 Emergency restart completed")
            
            return True
            
        except Exception as e:
            print_current(f"🚨 Emergency restart failed: {e}")
            return False


class ResourceMonitor:
    """Resource monitor"""
    
    def __init__(self, max_concurrent_agents: int = 5):
        self.max_concurrent_agents = max_concurrent_agents
        self.active_agents = set()
        self.agent_start_times = {}
        self.lock = threading.Lock()
        
    def can_start_agent(self, agent_id: str) -> bool:
        """Check if a new agent can be started"""
        with self.lock:
            return len(self.active_agents) < self.max_concurrent_agents
    
    def register_agent_start(self, agent_id: str):
        """Register agent start execution"""
        with self.lock:
            self.active_agents.add(agent_id)
            self.agent_start_times[agent_id] = time.time()
            print_current(f"🟢 Agent {agent_id} started execution ({len(self.active_agents)}/{self.max_concurrent_agents} slots used)")
    
    def register_agent_finish(self, agent_id: str):
        """Register agent finish execution"""
        with self.lock:
            if agent_id in self.active_agents:
                self.active_agents.remove(agent_id)
                execution_time = time.time() - self.agent_start_times.get(agent_id, time.time())
                print_current(f"🔴 Agent {agent_id} finished execution (took {execution_time:.1f}s, {len(self.active_agents)}/{self.max_concurrent_agents} slots used)")
                
                if agent_id in self.agent_start_times:
                    del self.agent_start_times[agent_id]
    
    def get_active_count(self) -> int:
        """Get current number of active agents"""
        with self.lock:
            return len(self.active_agents)
    
    def get_agent_execution_time(self, agent_id: str) -> float:
        """Get current execution time for an agent"""
        with self.lock:
            if agent_id in self.agent_start_times:
                return time.time() - self.agent_start_times[agent_id]
            return 0.0


@dataclass
class RoundExecutionRequest:
    """Represents a request for a new round of execution."""
    agent_id: str
    current_round: int
    next_round: int
    priority: float
    request_time: float

    def __lt__(self, other):
        """For priority queue sorting (lower values have higher priority)"""
        return self.priority < other.priority


class PriorityAgentScheduler:
    """Priority-based agent scheduler"""
    
    def __init__(self, max_workers: int = 5, fairness_interval: float = 2.0):
        """
        Initialize priority agent scheduler
        
        Args:
            max_workers: Maximum number of worker threads
            fairness_interval: Fairness adjustment interval (seconds)
        """
        self.max_workers = max_workers
        self.fairness_interval = fairness_interval  # Fairness adjustment interval (seconds)
        
        # Core components
        self.task_queue = queue.PriorityQueue()
        self.agent_metrics = {}  # agent_id -> AgentMetrics
        self.resource_monitor = ResourceMonitor(max_workers)
        
        # Scheduler state
        self.scheduler_active = False
        self.worker_threads = []
        self.metrics_lock = threading.RLock()  # Use RLock to allow reentrant locking
        self.last_lock_release_time = time.time()
        self.lock_timeout_threshold = 15.0  # Lowered timeout threshold to 15s for faster deadlock detection
        self.fairness_thread = None
        
        # Statistics
        self.total_tasks_submitted = 0
        self.total_tasks_completed = 0
        self.total_tasks_failed = 0
        
        # Round scheduling control
        self.round_execution_counts = {}  # agent_id -> current_round_count
        self.round_request_queue = queue.PriorityQueue()  # Round execution request queue
        self.round_scheduler_active = False
        self.round_scheduler_thread = None
        
        # Thread monitoring and safety mechanisms
        self.worker_health_check_interval = 3.0  # Health check interval
        self.max_task_execution_time = 120.0  # Max task execution time
        self.worker_last_activity = {}  # Track worker activity time
        self.active_task_start_times = {}  # Track task start time
        self.health_monitor_thread = None
        
        # Lock contention configuration
        self.lock_contention_threshold = 10  # Lock contention threshold
        self.lock_acquire_timeout = 2.0  # Lowered lock acquire timeout
        
        print_current("🎯 Priority Agent Scheduler initialized with aggressive deadlock protection")
        
        
    def start(self):
        """Start the scheduler"""
        if self.scheduler_active:
            return
            
        self.scheduler_active = True
        
        # Start worker threads for initial agent spawning
        for i in range(self.max_workers):
            worker = threading.Thread(
                target=self._worker_loop,
                name=f"AgentWorker-{i}",
                daemon=True
            )
            worker.start()
            self.worker_threads.append(worker)
            
        # Start fairness adjustment thread
        self.fairness_thread = threading.Thread(
            target=self._fairness_adjustment_loop,
            name="FairnessAdjuster",
            daemon=True
        )
        self.fairness_thread.start()
        
        # Start round scheduler
        self.round_scheduler_active = True
        self.round_scheduler_thread = threading.Thread(
            target=self._round_scheduler_loop,
            name="RoundScheduler", 
            daemon=True
        )
        self.round_scheduler_thread.start()
        
        # Start health monitor
        if getattr(self, "hung_task_detection", False):
            self.health_monitor_thread = threading.Thread(
                target=self._health_monitor_loop,
                name="HealthMonitor",
                daemon=True
            )
            self.health_monitor_thread.start()
        
        print_current(f"🚀 Priority scheduler started with {self.max_workers} workers and round-level fairness control")
    
    def stop(self):
        """Stop the scheduler"""
        self.scheduler_active = False
        self.round_scheduler_active = False
        
        # Wait for all threads to finish
        for worker in self.worker_threads:
            if worker.is_alive():
                worker.join(timeout=2.0)
        
        if self.fairness_thread and self.fairness_thread.is_alive():
            self.fairness_thread.join(timeout=2.0)
        
        if self.round_scheduler_thread and self.round_scheduler_thread.is_alive():
            self.round_scheduler_thread.join(timeout=2.0)
            
        if self.health_monitor_thread and self.health_monitor_thread.is_alive():
            self.health_monitor_thread.join(timeout=2.0)
        
    
    def submit_agent_task(self, agent_id: str, task_func: Callable, 
                         estimated_duration: float = 30.0, 
                         base_priority: float = 5.0) -> str:
        """Submit agent task to scheduling queue"""
        
        # Generate task ID
        task_id = f"task_{agent_id}_{int(time.time())}_{uuid.uuid4().hex[:8]}"
        
        # Get or create agent metrics
        with self.metrics_lock:
            if agent_id not in self.agent_metrics:
                self.agent_metrics[agent_id] = AgentMetrics(agent_id=agent_id)
            
            # Calculate statistical info for fairness
            current_time = time.time()
            all_executions = [m.total_executions for m in self.agent_metrics.values()]
            avg_executions = sum(all_executions) / len(all_executions) if all_executions else 0
            min_executions = min(all_executions) if all_executions else 0
            
            # Calculate dynamic priority with fairness context
            priority = self.agent_metrics[agent_id].calculate_dynamic_priority(
                current_time, base_priority, avg_executions, min_executions
            )
        
        # Create task
        task = AgentTask(
            task_id=task_id,
            agent_id=agent_id,
            task_func=task_func,
            priority=priority,
            submit_time=current_time,
            estimated_duration=estimated_duration
        )
        
        # Add to priority queue
        self.task_queue.put(task)
        self.total_tasks_submitted += 1
        
        print_agent(agent_id, f"📋 Task {task_id} submitted with priority {priority:.2f}")
        
        # Auto-start scheduler if not already running
        if not self.scheduler_active:
            self.start()
        
        return task_id
    
    def _worker_loop(self):
        """Worker thread main loop"""
        worker_name = threading.current_thread().name
        print_current(f"👷 Worker {worker_name} started")
        
        # Initialize worker activity time
        with self.metrics_lock:
            self.worker_last_activity[worker_name] = time.time()
        
        # Reduce lock acquisition frequency: batch update activity time
        last_activity_update = time.time()
        activity_update_interval = 5.0  # Update every 5 seconds
        
        while self.scheduler_active:
            try:
                # Only update activity time when needed
                current_time = time.time()
                if current_time - last_activity_update > activity_update_interval:
                    with self.metrics_lock:
                        self.worker_last_activity[worker_name] = current_time
                    last_activity_update = current_time
                
                # Get task from queue (set timeout to avoid blocking)
                try:
                    task = self.task_queue.get(timeout=1.0)
                except queue.Empty:
                    continue
                
                # Check if resources are available
                if not self.resource_monitor.can_start_agent(task.agent_id):
                    print_current(f"⏳ No available slots for {task.agent_id}, requeueing...")
                    time.sleep(0.5)
                    self.task_queue.put(task)
                    continue
                
                # Optimize fairness check: separate data fetch and computation
                should_delay = False
                delay_info = None
                waiting_agents = []
                
                # Step 1: Quickly fetch necessary data
                with self.metrics_lock:
                    if len(self.agent_metrics) > 1:
                        current_executions = self.agent_metrics[task.agent_id].total_executions
                        execution_data = {}
                        for agent_id, metrics in self.agent_metrics.items():
                            execution_data[agent_id] = {
                                'executions': metrics.total_executions,
                                'last_execution_time': metrics.last_execution_time
                            }
                
                # Step 2: Do complex computation outside the lock
                if len(execution_data) > 1:
                    all_executions = [data['executions'] for data in execution_data.values()]
                    min_executions = min(all_executions)
                    avg_executions = sum(all_executions) / len(all_executions)
                    
                    execution_gap = current_executions - min_executions
                    execution_ratio = current_executions / max(avg_executions, 1)
                    
                    # Delay if gap is large (relaxed threshold)
                    if execution_gap > 3 or execution_ratio > 2.0:
                        should_delay = True
                        delay_time = min(execution_gap * 0.2, 1.0)
                        priority_penalty = execution_gap * 0.5
                        delay_info = {
                            'delay_time': delay_time,
                            'priority_penalty': priority_penalty,
                            'gap': execution_gap,
                            'ratio': execution_ratio
                        }
                    
                    # Check for waiting agents (non-blocking)
                    elif execution_gap > 1 and len(all_executions) > 2:
                        for agent_id, data in execution_data.items():
                            if (agent_id != task.agent_id and 
                                data['executions'] <= min_executions and
                                data['last_execution_time'] and
                                current_time - data['last_execution_time'] > 15):
                                waiting_agents.append(agent_id)
                
                # Step 3: Act on computation results (outside lock)
                if should_delay and delay_info:
                    print_current(f"⚖️ Moderate fairness control: delaying {task.agent_id} by {delay_info['delay_time']:.1f}s (gap: {delay_info['gap']}, ratio: {delay_info['ratio']:.2f})")
                    time.sleep(delay_info['delay_time'])
                    task.priority += delay_info['priority_penalty']
                    self.task_queue.put(task)
                    continue
                
                if waiting_agents:
                    print_current(f"⚖️ Note: agents {waiting_agents} are waiting, but allowing {task.agent_id} to proceed")
                
                # Execute task
                self._execute_task(task)
                
                # Mark task as done
                self.task_queue.task_done()
                
            except Exception as e:
                print_current(f"❌ Worker {worker_name} error: {e}")
                try:
                    self.task_queue.task_done()
                except:
                    pass
        
        # Clean up worker activity record on stop
        with self.metrics_lock:
            if worker_name in self.worker_last_activity:
                del self.worker_last_activity[worker_name]
        
        print_current(f"👷 Worker {worker_name} stopped")
    
    def _execute_task(self, task: AgentTask):
        """Execute a single task with timeout protection"""
        agent_id = task.agent_id
        task_id = task.task_id
        start_time = time.time()
        success = False
        
        try:
            # Record task start time for timeout detection
            worker_name = threading.current_thread().name
            with self.metrics_lock:
                self.worker_last_activity[worker_name] = start_time
                self.active_task_start_times[task_id] = start_time
            
            # Register agent start execution
            self.resource_monitor.register_agent_start(agent_id)
            
            print_agent(agent_id, f"▶️ Starting task {task_id} on {worker_name}")
            
            # Execute task function with timeout awareness
            result = task.task_func()
            
            success = result.get('success', False) if isinstance(result, dict) else True
            execution_time = time.time() - start_time
            
            # Update metrics
            with self.metrics_lock:
                self.agent_metrics[agent_id].update_execution(execution_time, success)
            
            if success:
                self.total_tasks_completed += 1
                print_agent(agent_id, f"✅ Task {task_id} completed successfully in {execution_time:.1f}s")
            else:
                self.total_tasks_failed += 1
                print_agent(agent_id, f"⚠️ Task {task_id} completed with issues in {execution_time:.1f}s")
                
        except Exception as e:
            execution_time = time.time() - start_time
            self.total_tasks_failed += 1
            
            # Update metrics
            with self.metrics_lock:
                self.agent_metrics[agent_id].update_execution(execution_time, False)
            
            print_agent(agent_id, f"❌ Task {task_id} failed after {execution_time:.1f}s: {e}")
            
            # Consider retry
            if task.retry_count < task.max_retries:
                task.retry_count += 1
                task.priority += 1.0  # Lower priority
                print_agent(agent_id, f"🔄 Retrying task {task_id} (attempt {task.retry_count}/{task.max_retries})")
                self.task_queue.put(task)
            
        finally:
            # Clean up task tracking info
            with self.metrics_lock:
                if task_id in self.active_task_start_times:
                    del self.active_task_start_times[task_id]
                self.worker_last_activity[worker_name] = time.time()
            
            # Register agent finish execution
            self.resource_monitor.register_agent_finish(agent_id)
    
    def _fairness_adjustment_loop(self):
        """Fairness adjustment loop"""
        print_current("⚖️ Fairness adjustment thread started")
        
        while self.scheduler_active:
            try:
                time.sleep(self.fairness_interval)
                
                if not self.scheduler_active:
                    break
                
                self._adjust_fairness_scores()
                
            except Exception as e:
                print_current(f"⚠️ Fairness adjustment error: {e}")
        
        print_current("⚖️ Fairness adjustment thread stopped")
    
    def _health_monitor_loop(self):
        """Health monitor main loop with aggressive deadlock detection"""
        print_current("🏥 Health monitor started with aggressive deadlock detection")
        
        stalled_check_counter = 0
        deadlock_check_counter = 0
        
        while self.scheduler_active:
            try:
                time.sleep(1.0)  # More frequent checks (every second)
                
                if not self.scheduler_active:
                    break
                
                # More frequent deadlock detection (every 3 checks)
                deadlock_check_counter += 1
                if deadlock_check_counter >= 3:
                    if self.detect_and_recover_deadlock():
                        print_current("🚨 Deadlock recovery performed by health monitor")
                        stalled_check_counter = 0
                        deadlock_check_counter = 0
                        time.sleep(2)  # Let system stabilize
                        continue
                    deadlock_check_counter = 0
                
                self._check_hung_tasks()
                self._check_worker_health()
                
                # Check for stalled agents (every 10 seconds)
                stalled_check_counter += 1
                if stalled_check_counter >= 10:
                    reset_count = self.reset_stalled_agents()
                    if reset_count > 0:
                        print_current(f"🔄 Reset {reset_count} stalled agents for better fairness")
                    stalled_check_counter = 0
                
                # Diagnose lock contention
                if deadlock_check_counter == 1:
                    self._diagnose_system_health()
                
            except Exception as e:
                print_current(f"⚠️ Health monitor error: {e}")
        
        print_current("🏥 Health monitor thread stopped")
    
    def _diagnose_system_health(self):
        """Diagnose system health status"""
        try:
            current_time = time.time()
            
            # Check lock status
            lock_age = current_time - self.last_lock_release_time
            if lock_age > 5.0:
                print_current(f"⚠️ Lock not released for {lock_age:.1f}s")
            
            # Check queue sizes
            task_queue_size = self.task_queue.qsize()
            round_queue_size = self.round_request_queue.qsize()
            
            if task_queue_size > 10 or round_queue_size > 10:
                print_current(f"📊 Queue status: tasks={task_queue_size}, rounds={round_queue_size}")
            
            # Check active worker count
            active_workers = len([t for t in self.worker_threads if t.is_alive()])
            if active_workers != self.max_workers:
                print_current(f"⚠️ Worker count mismatch: {active_workers}/{self.max_workers}")
            
        except Exception as e:
            print_current(f"⚠️ Health diagnosis error: {e}")
    
    def _check_hung_tasks(self):
        """Check for hung tasks"""
        current_time = time.time()
        hung_tasks = []
        
        with self.metrics_lock:
            for task_id, start_time in self.active_task_start_times.items():
                execution_time = current_time - start_time
                if execution_time > self.max_task_execution_time:
                    hung_tasks.append((task_id, execution_time))
        
        for task_id, execution_time in hung_tasks:
            print_current(f"🚨 Detected hung task {task_id} running for {execution_time:.1f}s (limit: {self.max_task_execution_time}s)")
            # Note: We cannot forcibly kill tasks, but can log and alert
    
    def _check_worker_health(self):
        """Check worker health status"""
        current_time = time.time()
        inactive_workers = []
        stale_workers = []
        
        with self.metrics_lock:
            for worker_name, last_activity in self.worker_last_activity.items():
                inactive_time = current_time - last_activity
                
                # Only alert for truly long-unresponsive workers
                if inactive_time > 300:  # 5 minutes of inactivity
                    inactive_workers.append((worker_name, inactive_time))
                elif inactive_time > 120:  # 2 minutes of inactivity, possibly stuck
                    stale_workers.append((worker_name, inactive_time))
        
        # Distinguish real problems from normal waiting
        if inactive_workers:
            print_current(f"🚨 Found {len(inactive_workers)} potentially hung workers")
            for worker_name, inactive_time in inactive_workers:
                print_current(f"🚨 Worker {worker_name} potentially hung: {inactive_time:.1f}s without activity")
        
        # Only show possibly stuck workers in debug mode
        if stale_workers and hasattr(self, 'debug_mode') and getattr(self, 'debug_mode', False):
            print_current(f"🔍 Debug: {len(stale_workers)} workers with extended idle time")
            for worker_name, inactive_time in stale_workers:
                print_current(f"🔍 Debug: Worker {worker_name} idle for {inactive_time:.1f}s")
    
    def _adjust_fairness_scores(self):
        """Ultra-aggressive fairness score adjustment for maximum balance"""
        # Separate data fetch and computation to avoid long lock holding
        agent_data = {}
        current_time = time.time()
        
        # Step 1: Quickly fetch necessary data
        with self.metrics_lock:
            if len(self.agent_metrics) < 2:
                return
            
            for agent_id, metrics in self.agent_metrics.items():
                agent_data[agent_id] = {
                    'total_executions': metrics.total_executions,
                    'last_execution_time': metrics.last_execution_time,
                    'fairness_score': metrics.fairness_score
                }
        
        # Step 2: Do complex computation outside the lock
        if len(agent_data) < 2:
            return
            
        execution_counts = [data['total_executions'] for data in agent_data.values()]
        avg_executions = statistics.mean(execution_counts)
        max_executions = max(execution_counts)
        min_executions = min(execution_counts)
        execution_gap = max_executions - min_executions
        
        # If perfectly balanced, no adjustment needed
        if execution_gap == 0:
            return
        
        # Aggressive adjustment strategy
        adjustments_to_apply = {}
        adjustments_made = 0
        
        for agent_id, data in agent_data.items():
            old_score = data['fairness_score']
            execution_deviation = data['total_executions'] - avg_executions
            
            # Multi-dimensional unfairness factor
            execution_factor = execution_deviation / max(avg_executions, 1)
            
            # Wait time factor
            wait_factor = 0
            if data['last_execution_time']:
                wait_time = current_time - data['last_execution_time']
                if wait_time > 5:
                    wait_factor = min(wait_time / 30, 2.0)
            
            new_score = old_score
            
            # Aggressive score adjustment
            if execution_deviation > 0.5:
                penalty = execution_factor * 2.0 + min(execution_deviation * 0.5, 2.0)
                new_score = max(old_score - penalty, 0.05)
                adjustments_made += 1
            elif execution_deviation < -0.3:
                bonus = abs(execution_factor) * 2.5 + wait_factor
                new_score = min(old_score + bonus, 5.0)
                adjustments_made += 1
            elif execution_deviation < 0:
                bonus = abs(execution_deviation) * 0.3 + wait_factor * 0.5
                new_score = min(old_score + bonus, 3.0)
                adjustments_made += 1
            
            if new_score != old_score:
                adjustments_to_apply[agent_id] = new_score
        
        # Step 3: Quickly apply results
        if adjustments_to_apply:
            with self.metrics_lock:
                for agent_id, new_score in adjustments_to_apply.items():
                    if agent_id in self.agent_metrics:
                        self.agent_metrics[agent_id].fairness_score = new_score
    
    def _immediate_priority_boost(self):
        """Immediately boost priority for agents waiting too long"""
        current_time = time.time()
        boosted_count = 0
        
        # Find tasks in the queue and adjust priority
        temp_queue = queue.PriorityQueue()
        
        while not self.task_queue.empty():
            try:
                task = self.task_queue.get_nowait()
                
                # Check if immediate boost is needed
                if task.agent_id in self.agent_metrics:
                    metrics = self.agent_metrics[task.agent_id]
                    
                    if (metrics.last_execution_time and 
                        current_time - metrics.last_execution_time > 8):
                        
                        wait_time = current_time - metrics.last_execution_time
                        boost = min(wait_time / 5, 3.0)
                        old_priority = task.priority
                        task.priority = max(task.priority - boost, 0.01)
                        
                        print_current(f"🚀 Immediate priority boost for {task.agent_id}: {old_priority:.2f} → {task.priority:.2f} (waiting {wait_time:.1f}s)")
                        boosted_count += 1
                
                temp_queue.put(task)
                
            except queue.Empty:
                break
        
        # Put tasks back to the original queue
        while not temp_queue.empty():
            self.task_queue.put(temp_queue.get())
        
        if boosted_count > 0:
            print_current(f"🚀 Immediate priority boost applied to {boosted_count} waiting tasks")
    
    def request_next_round(self, agent_id: str, current_round: int, 
                          max_rounds: int, wait_timeout: float = 30.0) -> bool:
        """
        Request permission to execute the next round
        
        Args:
            agent_id: Agent ID
            current_round: Current round
            max_rounds: Maximum rounds
            wait_timeout: Wait timeout
            
        Returns:
            Whether permission was granted
        """
        if current_round >= max_rounds:
            return False
        
        priority = None
        current_time = time.time()
        
        # First lock: only for initialization and priority calculation
        with self.metrics_lock:
            if agent_id not in self.agent_metrics:
                self.agent_metrics[agent_id] = AgentMetrics(agent_id=agent_id)
            
            if agent_id not in self.round_execution_counts:
                self.round_execution_counts[agent_id] = 0
            
            all_round_counts = [count for count in self.round_execution_counts.values()]
            avg_rounds = sum(all_round_counts) / len(all_round_counts) if all_round_counts else 0
            min_rounds = min(all_round_counts) if all_round_counts else 0
            current_round_count = self.round_execution_counts[agent_id]
            last_execution_time = self.agent_metrics[agent_id].last_execution_time
        
        # Calculate round priority (fewer rounds = higher priority)
        round_gap = current_round_count - min_rounds
        round_ratio = current_round_count / max(avg_rounds, 1)
        
        priority = 5.0 + round_gap * 2.0
        
        if round_ratio > 1.2:
            priority += (round_ratio - 1.0) * 5.0
        elif round_ratio < 0.8:
            priority -= (1.0 - round_ratio) * 3.0
        
        # Wait time bonus
        if last_execution_time:
            wait_time = current_time - last_execution_time
            if wait_time > 5:
                time_bonus = min(wait_time / 5, 6.0)
                priority -= time_bonus
        
        priority = max(priority, 0.01)
        
        # Create round request
        round_request = RoundExecutionRequest(
            agent_id=agent_id,
            current_round=current_round,
            next_round=current_round + 1,
            priority=priority,
            request_time=current_time
        )
        
        # Submit to round scheduling queue
        self.round_request_queue.put(round_request)
        print_agent(agent_id, f"🎫 Round {current_round+1} execution request submitted (priority: {priority:.2f})")
        
        # Wait for permission using short lock acquisition
        start_wait = time.time()
        while time.time() - start_wait < wait_timeout:
            permission_granted = False
            with self.metrics_lock:
                permission_granted = self.round_execution_counts.get(agent_id, 0) > current_round
            
            if permission_granted:
                return True
                
            time.sleep(0.1)
        
        print_agent(agent_id, f"⏰ Round {current_round+1} request timeout after {wait_timeout}s")
        return False
    
    def _round_scheduler_loop(self):
        """Round scheduler main loop"""
        print_current("🎮 Round scheduler started")
        
        request_retry_counts = {}  # request_id -> retry_count
        max_retries_per_request = 5
        yield_timeout_threshold = 10.0
        
        while self.round_scheduler_active:
            try:
                try:
                    request = self.round_request_queue.get(timeout=1.0)
                except queue.Empty:
                    continue
                
                request_id = f"{request.agent_id}_{request.current_round}_{request.next_round}"
                
                retry_count = request_retry_counts.get(request_id, 0)
                if retry_count >= max_retries_per_request:
                    print_current(f"🚨 Request {request_id} exceeded max retries ({max_retries_per_request}), forcing execution")
                    del request_retry_counts[request_id]
                    self._grant_round_permission(request)
                    continue
                
                with self.metrics_lock:
                    all_round_counts = [count for count in self.round_execution_counts.values()]
                    if len(all_round_counts) <= 1:
                        self._grant_round_permission(request)
                        if request_id in request_retry_counts:
                            del request_retry_counts[request_id]
                        continue
                    
                    current_rounds = self.round_execution_counts.get(request.agent_id, 0)
                    min_rounds = min(all_round_counts)
                    avg_rounds = sum(all_round_counts) / len(all_round_counts)
                    
                    round_gap = current_rounds - min_rounds
                    round_ratio = current_rounds / max(avg_rounds, 1)
                    
                    if round_gap > 3 or round_ratio > 2.0:
                        if retry_count < 2:
                            delay_time = min(round_gap * 0.3, 1.5)
                            print_current(f"⚖️ Moderate round fairness control: delaying {request.agent_id} by {delay_time:.1f}s (gap: {round_gap}, ratio: {round_ratio:.2f}, retry: {retry_count})")
                            time.sleep(delay_time)
                        
                        request.priority += round_gap * 1.0
                        request_retry_counts[request_id] = retry_count + 1
                        self.round_request_queue.put(request)
                        continue
                    
                    waiting_agents = [
                        agent_id for agent_id, count in self.round_execution_counts.items()
                        if count <= min_rounds and agent_id != request.agent_id
                    ]
                    
                    should_yield = False
                    if waiting_agents and round_gap > 1:
                        active_waiting_agents = []
                        queue_snapshot = []
                        
                        temp_queue = queue.PriorityQueue()
                        found_waiting_requests = False
                        
                        try:
                            while not self.round_request_queue.empty():
                                other_request = self.round_request_queue.get_nowait()
                                temp_queue.put(other_request)
                                if other_request.agent_id in waiting_agents:
                                    active_waiting_agents.append(other_request.agent_id)
                                    found_waiting_requests = True
                        except queue.Empty:
                            pass
                        
                        while not temp_queue.empty():
                            self.round_request_queue.put(temp_queue.get_nowait())
                        
                        if found_waiting_requests and retry_count < 2 and round_gap > 2:
                            should_yield = True
                            print_current(f"⚖️ Yielding to active waiting agents: {active_waiting_agents}, requeueing {request.agent_id} (retry: {retry_count})")
                        elif retry_count >= 2:
                            print_current(f"🚨 Stop yielding for {request.agent_id} after {retry_count} retries, forcing execution")
                        elif not found_waiting_requests:
                            print_current(f"⚖️ No active requests from waiting agents {waiting_agents}, allowing {request.agent_id} to proceed")
                        elif round_gap <= 2:
                            print_current(f"⚖️ Gap is small ({round_gap}), allowing {request.agent_id} to proceed")
                    
                    if should_yield:
                        request.priority += 1.0
                        request_retry_counts[request_id] = retry_count + 1
                        self.round_request_queue.put(request)
                        time.sleep(0.1)
                        continue
                    
                    self._grant_round_permission(request)
                    if request_id in request_retry_counts:
                        del request_retry_counts[request_id]
                
            except Exception as e:
                print_current(f"❌ Round scheduler error: {e}")
        
        print_current("🎮 Round scheduler stopped")
    
    def _grant_round_permission(self, request):
        """Grant round execution permission"""
        def grant_operation():
            self.round_execution_counts[request.agent_id] = request.next_round
            
            if request.agent_id in self.agent_metrics:
                self.agent_metrics[request.agent_id].total_executions += 1
                self.agent_metrics[request.agent_id].last_execution_time = time.time()
            return True
        
        result = self._safe_lock_operation(grant_operation, timeout=1.0)
        
        if result:
            print_agent(request.agent_id, f"✅ Round {request.next_round} execution granted")
        else:
            print_current(f"⚠️ Failed to grant round permission for {request.agent_id}")
            self.round_execution_counts[request.agent_id] = request.next_round

    def get_status(self) -> Dict[str, Any]:
        """Get scheduler status"""
        with self.metrics_lock:
            agent_stats = {}
            for agent_id, metrics in self.agent_metrics.items():
                agent_stats[agent_id] = {
                    "total_executions": metrics.total_executions,
                    "avg_execution_time": metrics.avg_execution_time,
                    "success_rate": metrics.success_count / max(metrics.total_executions, 1),
                    "fairness_score": metrics.fairness_score,
                    "last_execution_time": metrics.last_execution_time
                }
        
        return {
            "scheduler_active": self.scheduler_active,
            "queue_size": self.task_queue.qsize(),
            "active_agents": self.resource_monitor.get_active_count(),
            "max_workers": self.max_workers,
            "total_submitted": self.total_tasks_submitted,
            "total_completed": self.total_tasks_completed,
            "total_failed": self.total_tasks_failed,
            "agent_stats": agent_stats
        }
    
    def print_status(self):
        """Print scheduler status"""
        status = self.get_status()
        
        print_current("📊 ===========================================")
        print_current("📊 Priority Agent Scheduler Status")
        print_current("📊 ===========================================")
        print_current(f"📊 Scheduler Active: {status['scheduler_active']}")
        print_current(f"📊 Queue Size: {status['queue_size']}")
        print_current(f"📊 Active Agents: {status['active_agents']}/{status['max_workers']}")
        print_current(f"📊 Tasks Submitted: {status['total_submitted']}")
        print_current(f"📊 Tasks Completed: {status['total_completed']}")
        print_current(f"📊 Tasks Failed: {status['total_failed']}")
        
        if status['agent_stats']:
            print_current("📊 Agent Performance:")
            for agent_id, stats in status['agent_stats'].items():
                print_current(f"📊   {agent_id}: {stats['total_executions']} execs, "
                            f"{stats['success_rate']:.1%} success, "
                            f"fairness: {stats['fairness_score']:.2f}")
        
        print_current("📊 ===========================================")

    def detect_and_recover_deadlock(self) -> bool:
        """
        Detect and recover from deadlock state
        
        Returns:
            True if deadlock was detected and recovery attempted
        """
        current_time = time.time()
        
        # Check for potential lock timeout
        if current_time - self.last_lock_release_time > self.lock_timeout_threshold:
            print_current(f"🚨 DEADLOCK DETECTED: Lock held for {current_time - self.last_lock_release_time:.1f}s")
            
            try:
                # Try to acquire lock (short timeout)
                if self.metrics_lock.acquire(timeout=1.0):
                    try:
                        self.last_lock_release_time = current_time
                        print_current("🔧 Lock acquired successfully, deadlock may have resolved")
                        return False
                    finally:
                        self.metrics_lock.release()
                else:
                    print_current("🚨 CONFIRMED DEADLOCK: Cannot acquire lock within timeout")
                    return self._emergency_deadlock_recovery()
                    
            except Exception as e:
                print_current(f"🚨 DEADLOCK DETECTION ERROR: {e}")
                return self._emergency_deadlock_recovery()
        
        return False
    
    def _emergency_deadlock_recovery(self) -> bool:
        """
        Emergency deadlock recovery
        
        Returns:
            True if recovery was attempted
        """
        print_current("🚨 INITIATING EMERGENCY DEADLOCK RECOVERY...")
        
        try:
            # Stop all background threads
            old_active = self.scheduler_active
            self.scheduler_active = False
            self.round_scheduler_active = False
            
            print_current("🚨 Waiting for threads to terminate...")
            time.sleep(2)
            
            print_current("🚨 Force cleaning task queues...")
            
            old_task_queue = self.task_queue
            old_round_queue = self.round_request_queue
            
            self.task_queue = queue.PriorityQueue()
            self.round_request_queue = queue.PriorityQueue()
            self.metrics_lock = threading.RLock()
            self.last_lock_release_time = time.time()
            
            transferred_tasks = 0
            transferred_rounds = 0
            
            try:
                while not old_task_queue.empty() and transferred_tasks < 10:
                    task = old_task_queue.get_nowait()
                    self.task_queue.put(task)
                    transferred_tasks += 1
            except:
                pass
                
            try:
                while not old_round_queue.empty() and transferred_rounds < 10:
                    request = old_round_queue.get_nowait()
                    self.round_request_queue.put(request)
                    transferred_rounds += 1
            except:
                pass
            
            print_current(f"🚨 Transferred {transferred_tasks} tasks and {transferred_rounds} round requests")
            
            with self.metrics_lock:
                self.worker_last_activity.clear()
                self.active_task_start_times.clear()
            
            if old_active:
                print_current("🚨 Restarting scheduler after deadlock recovery...")
                self.start()
            
            print_current("🚨 DEADLOCK RECOVERY COMPLETED")
            return True
            
        except Exception as e:
            print_current(f"🚨 DEADLOCK RECOVERY FAILED: {e}")
            return False
    
    def _safe_lock_operation(self, operation, timeout: float = 2.0):
        """
        Safe lock operation with timeout protection
        
        Args:
            operation: Function to execute under lock
            timeout: Lock acquire timeout
        """
        start_time = time.time()
        try:
            if self.metrics_lock.acquire(timeout=timeout):
                try:
                    result = operation()
                    self.last_lock_release_time = time.time()
                    return result
                finally:
                    self.metrics_lock.release()
            else:
                elapsed = time.time() - start_time
                print_current(f"⚠️ Lock acquisition timeout after {elapsed:.1f}s")
                if elapsed > 5.0:
                    print_current("🚨 Possible deadlock detected, attempting recovery...")
                    self.detect_and_recover_deadlock()
                return None
        except Exception as e:
            print_current(f"⚠️ Safe lock operation error: {e}")
            return None
    
    def _fast_metrics_read(self, agent_id: str = None):
        """
        Fast metrics read with timeout protection
        
        Returns:
            Copied metrics data dict
        """
        def read_operation():
            if agent_id:
                if agent_id in self.agent_metrics:
                    metrics = self.agent_metrics[agent_id]
                    return {
                        'total_executions': metrics.total_executions,
                        'last_execution_time': metrics.last_execution_time,
                        'fairness_score': metrics.fairness_score
                    }
                return None
            else:
                result = {}
                for aid, metrics in self.agent_metrics.items():
                    result[aid] = {
                        'total_executions': metrics.total_executions,
                        'last_execution_time': metrics.last_execution_time,
                        'fairness_score': metrics.fairness_score
                    }
                return result
        
        return self._safe_lock_operation(read_operation, timeout=1.0)
    
    def _fast_metrics_update(self, updates_dict):
        """
        Fast metrics update with timeout protection
        
        Args:
            updates_dict: {agent_id: {field: value, ...}, ...}
        """
        def update_operation():
            for agent_id, updates in updates_dict.items():
                if agent_id in self.agent_metrics:
                    metrics = self.agent_metrics[agent_id]
                    for field, value in updates.items():
                        if hasattr(metrics, field):
                            setattr(metrics, field, value)
            return True
        
        return self._safe_lock_operation(update_operation, timeout=1.0)

    def emergency_stop_and_restart(self) -> bool:
        """
        Emergency stop and restart the entire scheduling system.
        Use when infinite loops or system hangs are detected.
        
        Returns:
            True if restart was successful
        """
        print_current("🚨 EMERGENCY STOP AND RESTART INITIATED...")
        
        try:
            # Step 1: Stop all activity
            print_current("🚨 Step 1: Stopping all scheduler activities...")
            old_scheduler_active = self.scheduler_active
            old_round_scheduler_active = self.round_scheduler_active
            
            self.scheduler_active = False
            self.round_scheduler_active = False
            
            # Step 2: Wait for threads to terminate
            print_current("🚨 Step 2: Waiting for threads to terminate...")
            time.sleep(3)
            
            # Step 3: Force clean all queues and state
            print_current("🚨 Step 3: Force cleaning all queues and states...")
            
            old_submitted = self.total_tasks_submitted
            old_completed = self.total_tasks_completed
            old_failed = self.total_tasks_failed
            
            while not self.task_queue.empty():
                try:
                    self.task_queue.get_nowait()
                except queue.Empty:
                    break
            
            while not self.round_request_queue.empty():
                try:
                    self.round_request_queue.get_nowait()
                except queue.Empty:
                    break
            
            with self.metrics_lock:
                for agent_id, metrics in self.agent_metrics.items():
                    metrics.fairness_score = 1.0
                    
                self.round_execution_counts.clear()
                self.worker_last_activity.clear()
                self.active_task_start_times.clear()
            
            # Step 4: Recreate core components
            print_current("🚨 Step 4: Recreating core components...")
            self.task_queue = queue.PriorityQueue()
            self.round_request_queue = queue.PriorityQueue()
            
            self.metrics_lock = threading.RLock()
            self.last_lock_release_time = time.time()
            
            self.total_tasks_submitted = old_submitted
            self.total_tasks_completed = old_completed
            self.total_tasks_failed = old_failed
            
            # Step 5: Restart scheduler
            if old_scheduler_active or old_round_scheduler_active:
                print_current("🚨 Step 5: Restarting scheduler...")
                self.start()
            
            print_current("🚨 EMERGENCY RESTART COMPLETED SUCCESSFULLY")
            return True
            
        except Exception as e:
            print_current(f"🚨 EMERGENCY RESTART FAILED: {e}")
            return False
    
    def force_deadlock_break(self):
        """
        Force break deadlock, for external calls
        """
        print_current("🚨 FORCE DEADLOCK BREAK CALLED")
        
        if self.detect_and_recover_deadlock():
            print_current("🚨 Deadlock recovery successful")
            return True
        
        print_current("🚨 Deadlock recovery failed, performing emergency restart...")
        return self.emergency_stop_and_restart()

    def reset_stalled_agents(self):
        """
        Reset stalled agents to ensure all agents get execution opportunities
        """
        current_time = time.time()
        reset_count = 0
        
        with self.metrics_lock:
            if len(self.agent_metrics) < 2:
                return reset_count
                
            all_executions = [m.total_executions for m in self.agent_metrics.values()]
            min_executions = min(all_executions)
            max_executions = max(all_executions)
            
            if max_executions - min_executions > 5:
                print_current(f"🔄 Large execution gap detected ({max_executions - min_executions}), resetting stalled agents")
                
                for agent_id, metrics in self.agent_metrics.items():
                    if metrics.total_executions <= min_executions + 1:
                        metrics.fairness_score = max(metrics.fairness_score - 2.0, 0.01)
                        print_current(f"🚀 Boosting priority for stalled agent {agent_id}: fairness_score = {metrics.fairness_score:.2f}")
                        reset_count += 1
                    
                    elif metrics.total_executions >= max_executions - 1:
                        metrics.fairness_score = min(metrics.fairness_score + 1.0, 10.0)
                        print_current(f"⚡ Reducing priority for ahead agent {agent_id}: fairness_score = {metrics.fairness_score:.2f}")
                
                if self.round_execution_counts:
                    round_counts = list(self.round_execution_counts.values())
                    min_rounds = min(round_counts)
                    max_rounds = max(round_counts)
                    
                    if max_rounds - min_rounds > 3:
                        print_current(f"🔄 Resetting extreme round counts (gap: {max_rounds - min_rounds})")
                        avg_rounds = sum(round_counts) / len(round_counts)
                        for agent_id in self.round_execution_counts:
                            current_rounds = self.round_execution_counts[agent_id]
                            if current_rounds > avg_rounds + 2:
                                self.round_execution_counts[agent_id] = int(avg_rounds + 1)
                                print_current(f"📉 Reduced rounds for {agent_id}: {current_rounds} -> {self.round_execution_counts[agent_id]}")
                            elif current_rounds < avg_rounds - 2:
                                self.round_execution_counts[agent_id] = int(avg_rounds - 1)
                                print_current(f"📈 Increased rounds for {agent_id}: {current_rounds} -> {self.round_execution_counts[agent_id]}")
        
        return reset_count


# Global scheduler instance
_global_scheduler = None
_scheduler_lock = threading.Lock()

def get_priority_scheduler(max_workers: int = 5, auto_start: bool = False) -> PriorityAgentScheduler:
    """
    Get global priority scheduler instance
    
    Args:
        max_workers: Maximum number of worker threads
        auto_start: Whether to automatically start the scheduler (default False for lazy loading)
    
    Returns:
        PriorityAgentScheduler instance
    """
    global _global_scheduler
    
    with _scheduler_lock:
        if _global_scheduler is None:
            _global_scheduler = PriorityAgentScheduler(max_workers=max_workers)
            
            # Improved: Default to lazy loading, only start when explicitly requested
            if auto_start:
                _global_scheduler.start()
                print_current(f"🚀 Priority scheduler auto-started with {max_workers} workers")
            else:
                print_current(f"🏗️ Priority scheduler created (will start when first task is submitted)")
        
        return _global_scheduler

def cleanup_scheduler():
    """Clean up global scheduler"""
    global _global_scheduler
    
    with _scheduler_lock:
        if _global_scheduler is not None:
            _global_scheduler.stop()
            _global_scheduler = None 