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

多应用自动重启监控程序
监控多个GUI/app.py实例，每个实例使用不同的端口和应用名称
"""

import subprocess
import time
import socket
import os
import sys
import signal
import logging
import json
import threading
from datetime import datetime

class AppInstance:
    """单个应用实例的监控器"""
    def __init__(self, name, port, app_name, description, app_path, base_dir, check_interval, max_startup_attempts, startup_retry_delay, logger):
        self.name = name
        self.port = port
        self.app_name = app_name
        self.description = description
        self.app_path = app_path
        self.base_dir = base_dir
        self.check_interval = check_interval
        self.max_startup_attempts = max_startup_attempts
        self.startup_retry_delay = startup_retry_delay
        self.process = None
        self.startup_attempts = 0
        self.logger = logger.getChild(f"[{self.name}]")
        self.running = False
        
    def is_port_in_use(self, port):
        """检查端口是否被占用"""
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.settimeout(1)
                result = sock.connect_ex(('127.0.0.1', port))
                return result == 0
        except Exception as e:
            self.logger.error(f"检查端口时发生错误: {e}")
            return False
    
    def is_app_running(self):
        """检查应用是否在运行并监听指定端口"""
        try:
            return self.is_port_in_use(self.port)
        except Exception as e:
            self.logger.error(f"检查进程时发生错误: {e}")
            return False
    
    def start_app(self):
        """启动应用"""
        try:
            if not os.path.exists(self.app_path):
                self.logger.error(f"找不到应用文件: {self.app_path}")
                return False
            
            self.logger.info(f"正在启动应用 (端口: {self.port}, 应用名: {self.app_name})...")
            
            # 切换到GUI目录
            gui_dir = os.path.dirname(self.app_path)
            
            # 为每个应用创建独立的日志文件
            log_dir = os.path.join(self.base_dir, "logs")
            os.makedirs(log_dir, exist_ok=True)
            stdout_log = os.path.join(log_dir, f"app_{self.name}_stdout.log")
            stderr_log = os.path.join(log_dir, f"app_{self.name}_stderr.log")
            
            # 构建启动命令
            cmd = [sys.executable, "app.py", "--port", str(self.port), "--app", self.app_name]
            
            # 启动进程
            # 注意：日志文件使用绝对路径，避免工作目录问题
            stdout_file = open(stdout_log, "a", encoding="utf-8")
            stderr_file = open(stderr_log, "a", encoding="utf-8")
            
            self.process = subprocess.Popen(
                cmd,
                cwd=gui_dir,
                stdout=stdout_file,
                stderr=stderr_file,
                preexec_fn=os.setsid  # 创建新的进程组
            )
            
            # 检查进程是否成功启动
            if self.process.poll() is not None:
                self.logger.error(f"应用启动失败，退出码: {self.process.returncode}")
                return False
            
            self.logger.info(f"应用进程已启动，PID: {self.process.pid}，等待端口监听...")
            
            # 等待应用启动并监听端口，最多等待30秒
            max_wait_time = 30
            check_interval = 1
            waited_time = 0
            
            while waited_time < max_wait_time:
                time.sleep(check_interval)
                waited_time += check_interval
                
                # 检查进程是否还在运行
                if self.process.poll() is not None:
                    self.logger.error(f"应用进程意外退出，退出码: {self.process.returncode}")
                    return False
                
                # 检查端口是否开始监听
                if self.is_port_in_use(self.port):
                    self.logger.info(f"应用已启动并监听端口 {self.port}，PID: {self.process.pid}，等待时间: {waited_time}秒")
                    return True
            
            # 如果30秒后端口仍未监听，但进程还在运行，可能是启动时间较长
            if self.process.poll() is None:
                self.logger.warning(f"应用进程仍在运行，但端口 {self.port} 在 {max_wait_time} 秒内未开始监听")
                # 继续运行，让监控循环继续检查
                return True
            else:
                self.logger.error(f"应用启动失败，进程已退出")
                return False
                
        except Exception as e:
            self.logger.error(f"启动应用时发生错误: {e}")
            return False
    
    def kill_existing_processes(self):
        """杀死现有的应用进程"""
        try:
            # 查找匹配的应用进程（通过端口和应用名）
            pattern = f"python.*app.py.*--port {self.port}.*--app {self.app_name}"
            result = subprocess.run(
                ['pgrep', '-f', pattern],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0 and result.stdout.strip():
                pids = result.stdout.strip().split('\n')
                for pid_str in pids:
                    try:
                        pid = int(pid_str.strip())
                        self.logger.info(f"终止进程 PID: {pid}")
                        os.kill(pid, signal.SIGTERM)
                        time.sleep(1)
                        # 如果进程仍然存在，强制杀死
                        try:
                            os.kill(pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass  # 进程已经不存在
                    except (ValueError, ProcessLookupError):
                        continue
                        
        except Exception as e:
            self.logger.error(f"终止现有进程时发生错误: {e}")
    
    def stop(self):
        """停止应用"""
        self.running = False
        if self.process and self.process.poll() is None:
            try:
                # 终止子进程组
                os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
                self.process.wait(timeout=5)
            except (ProcessLookupError, subprocess.TimeoutExpired):
                try:
                    os.killpg(os.getpgid(self.process.pid), signal.SIGKILL)
                except ProcessLookupError:
                    pass
    
    def monitor_loop(self):
        """监控循环"""
        self.logger.info(f"开始监控应用 (端口: {self.port}, 应用名: {self.app_name}, 描述: {self.description})")
        self.running = True
        
        while self.running:
            try:
                if not self.is_app_running():
                    # 检查进程是否还在运行（可能进程在但端口未监听）
                    process_running = False
                    if self.process and self.process.poll() is None:
                        process_running = True
                        # 进程在运行但端口未监听，可能是启动中，等待一下
                        self.logger.debug(f"进程运行中但端口未监听，等待启动完成 (端口: {self.port})")
                        time.sleep(2)
                        # 再次检查端口
                        if self.is_app_running():
                            continue
                    
                    if not process_running:
                        self.logger.warning(f"检测到应用未运行 (端口: {self.port})")
                    
                    # 清理可能存在的僵尸进程
                    self.kill_existing_processes()
                    # 如果进程还在运行但端口未监听，强制终止
                    if self.process and self.process.poll() is None:
                        try:
                            self.logger.warning(f"终止未监听端口的进程 PID: {self.process.pid}")
                            os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
                            time.sleep(2)
                            if self.process.poll() is None:
                                os.killpg(os.getpgid(self.process.pid), signal.SIGKILL)
                        except (ProcessLookupError, OSError):
                            pass
                        self.process = None
                    
                    time.sleep(2)
                    
                    # 尝试启动应用
                    if self.start_app():
                        self.logger.info(f"应用重启成功 (端口: {self.port})")
                        self.startup_attempts = 0
                    else:
                        self.startup_attempts += 1
                        self.logger.error(f"应用启动失败 (端口: {self.port}, 尝试 {self.startup_attempts}/{self.max_startup_attempts})")
                        
                        if self.startup_attempts >= self.max_startup_attempts:
                            self.logger.error(f"达到最大启动尝试次数，等待{self.startup_retry_delay}秒后重试")
                            time.sleep(self.startup_retry_delay)
                            self.startup_attempts = 0
                        else:
                            time.sleep(5)
                else:
                    # 应用正在运行，重置启动尝试计数器
                    if self.startup_attempts > 0:
                        self.startup_attempts = 0
                
                time.sleep(self.check_interval)
                
            except Exception as e:
                self.logger.error(f"监控循环中发生错误: {e}")
                time.sleep(5)


class MultiAppMonitor:
    """多应用监控器"""
    def __init__(self, config_path=None):
        # 设置工作目录为脚本所在目录
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.app_path = os.path.join(self.base_dir, "../", "app.py")
        
        # 加载配置
        if config_path is None:
            config_path = os.path.join(self.base_dir, "monitor_config.json")
        
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"配置文件不存在: {config_path}")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = json.load(f)
        
        # 设置日志
        log_dir = os.path.join(self.base_dir, "logs")
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, "monitor.log")
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # 创建应用实例列表
        self.app_instances = []
        self.monitor_threads = []
        
        check_interval = self.config.get('check_interval', 1)
        max_startup_attempts = self.config.get('max_startup_attempts', 3)
        startup_retry_delay = self.config.get('startup_retry_delay', 60)
        
        for app_config in self.config.get('apps', []):
            app_instance = AppInstance(
                name=app_config['name'],
                port=app_config['port'],
                app_name=app_config['app_name'],
                description=app_config.get('description', ''),
                app_path=self.app_path,
                base_dir=self.base_dir,
                check_interval=check_interval,
                max_startup_attempts=max_startup_attempts,
                startup_retry_delay=startup_retry_delay,
                logger=self.logger
            )
            self.app_instances.append(app_instance)
        
        # 注册信号处理器，优雅退出
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
        self.running = False
    
    def signal_handler(self, signum, frame):
        """处理退出信号"""
        self.logger.info(f"收到信号 {signum}，正在退出监控程序...")
        self.stop_all()
        sys.exit(0)
    
    def stop_all(self):
        """停止所有应用监控"""
        self.running = False
        self.logger.info("正在停止所有应用监控...")
        
        # 停止所有应用实例
        for app_instance in self.app_instances:
            app_instance.stop()
        
        # 等待所有监控线程结束
        for thread in self.monitor_threads:
            thread.join(timeout=5)
        
        self.logger.info("所有应用监控已停止")
    
    def run(self):
        """启动所有应用的监控"""
        self.logger.info("=" * 60)
        self.logger.info("AGI Agent GUI 多应用监控程序启动")
        self.logger.info(f"共监控 {len(self.app_instances)} 个应用")
        self.logger.info("=" * 60)
        
        for app_instance in self.app_instances:
            self.logger.info(f"  - {app_instance.name}: 端口 {app_instance.port}, 应用名 {app_instance.app_name}")
        
        self.logger.info("=" * 60)
        
        self.running = True
        
        # 为每个应用启动一个监控线程
        for app_instance in self.app_instances:
            thread = threading.Thread(
                target=app_instance.monitor_loop,
                name=f"Monitor-{app_instance.name}",
                daemon=False
            )
            thread.start()
            self.monitor_threads.append(thread)
        
        # 等待所有线程（主线程保持运行）
        try:
            for thread in self.monitor_threads:
                thread.join()
        except KeyboardInterrupt:
            self.logger.info("用户中断，退出监控")
            self.stop_all()


def main():
    """主函数"""
    print("AGI Agent GUI 多应用监控程序")
    print("=" * 60)
    print("此程序将监控多个 GUI/app.py 实例")
    print("每个实例使用不同的端口和应用名称")
    print("如果检测到程序未运行，将自动重启")
    print("按 Ctrl+C 停止监控")
    print("=" * 60)
    
    try:
        monitor = MultiAppMonitor()
        monitor.run()
    except FileNotFoundError as e:
        print(f"错误: {e}")
        print("请确保 monitor_config.json 文件存在")
        sys.exit(1)
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
