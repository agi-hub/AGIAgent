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

自动重启监控程序
监控GUI/app.py是否在5002端口运行，如果没有则自动启动
"""

import subprocess
import time
import socket
import os
import sys
import signal
import logging
from datetime import datetime

class AppMonitor:
    def __init__(self):
        # 设置工作目录为脚本所在目录
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.app_path = os.path.join(self.base_dir, "../", "app.py")
        self.port = 5002
        self.check_interval = 1  # 每秒检测一次
        self.process = None
        
        # 设置日志
        log_dir = os.path.join(self.base_dir, "../../logs")
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, "monitor.log")
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # 注册信号处理器，优雅退出
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
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
        """检查app.py进程是否在运行并监听5002端口"""
        try:
            # 检查端口是否被占用
            if self.is_port_in_use(self.port):
                return True
            return False
            
            
        except Exception as e:
            self.logger.error(f"检查进程时发生错误: {e}")
            return False
    
    def start_app(self):
        """启动app.py"""
        try:
            if not os.path.exists(self.app_path):
                self.logger.error(f"找不到应用文件: {self.app_path}")
                return False
            
            self.logger.info("正在启动 GUI/app.py...")
            
            # 切换到GUI目录
            gui_dir = os.path.dirname(self.app_path)
            stdout_log = "logs/app_stdout.log"  # stdout日志文件
            stderr_log = "logs/app_stderr.log"  # stderr日志文件
            # 启动进程
            self.process = subprocess.Popen(
                [sys.executable, "app.py"],
                cwd=gui_dir,
                #stdout=subprocess.PIPE,
                #stderr=subprocess.PIPE,
                stdout=open(stdout_log, "a", encoding="utf-8"),  # "a"表示追加模式，"w"表示覆盖模式
                stderr=open(stderr_log, "a", encoding="utf-8"),
                preexec_fn=os.setsid  # 创建新的进程组
            )
            
            # 等待一小段时间让应用启动
            time.sleep(3)
            
            # 检查进程是否成功启动
            if self.process.poll() is None:
                self.logger.info(f"GUI/app.py 已启动，PID: {self.process.pid}")
                return True
            else:
                stdout, stderr = self.process.communicate()
                self.logger.error(f"GUI/app.py 启动失败:")
                if stdout:
                    self.logger.error(f"STDOUT: {stdout.decode('utf-8', errors='ignore')}")
                if stderr:
                    self.logger.error(f"STDERR: {stderr.decode('utf-8', errors='ignore')}")
                return False
                
        except Exception as e:
            self.logger.error(f"启动应用时发生错误: {e}")
            return False
    
    def kill_existing_processes(self):
        """杀死现有的app.py进程"""
        try:
            result = subprocess.run(
                ['pgrep', '-f', 'python.*GUI/app.py'],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0 and result.stdout.strip():
                pids = result.stdout.strip().split('\n')
                for pid in pids:
                    try:
                        pid = int(pid.strip())
                        self.logger.info(f"终止进程 PID: {pid}")
                        os.kill(pid, signal.SIGTERM)
                        time.sleep(1)
                        # 如果进程仍然存在，强制杀死
                        try:
                            os.kill(pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass  # 进程已经不存在
                    except (ValueError, ProcessLookupError) as e:
                        continue
                        
        except Exception as e:
            self.logger.error(f"终止现有进程时发生错误: {e}")
    
    def signal_handler(self, signum, frame):
        """处理退出信号"""
        self.logger.info(f"收到信号 {signum}，正在退出监控程序...")
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
        sys.exit(0)
    
    def run(self):
        """主监控循环"""
        self.logger.info("开始监控 GUI/app.py...")
        self.logger.info(f"目标端口: {self.port}")
        self.logger.info(f"检测间隔: {self.check_interval}秒")
        self.logger.info(f"应用路径: {self.app_path}")
        
        startup_attempts = 0
        max_startup_attempts = 3
        
        while True:
            try:
                if not self.is_app_running():
                    self.logger.warning("检测到 GUI/app.py 未运行")
                    
                    # 清理可能存在的僵尸进程
                    self.kill_existing_processes()
                    time.sleep(2)
                    
                    # 尝试启动应用
                    if self.start_app():
                        self.logger.info("GUI/app.py 重启成功")
                        startup_attempts = 0
                    else:
                        startup_attempts += 1
                        self.logger.error(f"GUI/app.py 启动失败 (尝试 {startup_attempts}/{max_startup_attempts})")
                        
                        if startup_attempts >= max_startup_attempts:
                            self.logger.error("达到最大启动尝试次数，等待60秒后重试")
                            time.sleep(60)
                            startup_attempts = 0
                        else:
                            time.sleep(5)
                else:
                    # 应用正在运行，重置启动尝试计数器
                    if startup_attempts > 0:
                        startup_attempts = 0
                
                time.sleep(self.check_interval)
                
            except KeyboardInterrupt:
                self.logger.info("用户中断，退出监控")
                break
            except Exception as e:
                self.logger.error(f"监控循环中发生错误: {e}")
                time.sleep(5)

def main():
    """主函数"""
    print("AGI Agent GUI 监控程序")
    print("=" * 50)
    print("此程序将监控 GUI/app.py 是否在5002端口运行")
    print("如果检测到程序未运行，将自动重启")
    print("按 Ctrl+C 停止监控")
    print("=" * 50)
    
    monitor = AppMonitor()
    monitor.run()

if __name__ == "__main__":
    main()

