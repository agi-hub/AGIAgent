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

import subprocess
import time
import queue
import threading
import re
import os
from typing import Dict, Any
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config_loader import get_auto_fix_interactive_commands


class TerminalTools:
    def __init__(self, workspace_root: str = None):
        self.workspace_root = workspace_root or "."
    
    def _fix_html_entities(self, text: str) -> str:
        """
        Auto-correct HTML entities in text using Python's html.unescape().
        This handles all standard HTML entities.
        
        Args:
            text: Text that might contain HTML entities
            
        Returns:
            Text with HTML entities corrected
        """
        import html
        
        original_text = text
        
        # Use Python's built-in html.unescape() for comprehensive entity decoding
        decoded_text = html.unescape(text)
        
        # Log if any changes were made
        if original_text != decoded_text:
            # Count common entities for logging
            common_entities = {
                '&lt;': '<',
                '&gt;': '>',
                '&amp;': '&',
                '&quot;': '"',
                '&#x27;': "'",
                '&#39;': "'"
            }
            
            corrections = []
            for entity, char in common_entities.items():
                count = original_text.count(entity)
                if count > 0:
                    corrections.append(f"{entity} → {char} ({count} times)")
            
            # If there are other entities not in our common list, mention them generically
            if corrections:
                print(f"🔧 Auto-corrected HTML entities in command: {', '.join(corrections)}")
            else:
                print(f"🔧 Auto-corrected HTML entities in command (various types found)")
        
        return decoded_text
    
    def _read_process_output_with_timeout(self, process, timeout_inactive=300, max_total_time=600):
        """
        Asynchronously read process output with smart timeout, while displaying real-time output to user
        """
        stdout_lines = []
        stderr_lines = []
        last_output_time = time.time()
        start_time = time.time()
        
        stdout_queue = queue.Queue()
        stderr_queue = queue.Queue()
        
        def read_stdout():
            try:
                for line in iter(process.stdout.readline, ''):
                    if line:
                        stdout_queue.put(('stdout', line, time.time()))
                process.stdout.close()
            except:
                pass
        
        def read_stderr():
            try:
                for line in iter(process.stderr.readline, ''):
                    if line:
                        stderr_queue.put(('stderr', line, time.time()))
                process.stderr.close()
            except:
                pass
        
        stdout_thread = threading.Thread(target=read_stdout)
        stderr_thread = threading.Thread(target=read_stderr)
        stdout_thread.daemon = True
        stderr_thread.daemon = True
        stdout_thread.start()
        stderr_thread.start()
        
        timed_out = False
        
        pass  # Separator line removed
        print("🚀 Command execution started, real-time output as follows:")
        pass  # Separator line removed
        
        try:
            while process.poll() is None:
                current_time = time.time()
                
                got_output = False
                
                try:
                    while True:
                        output_type, line, timestamp = stdout_queue.get_nowait()
                        line_clean = line.rstrip()
                        stdout_lines.append(line_clean)
                        if line_clean:
                            print(f"📤 {line_clean}")
                        last_output_time = timestamp
                        got_output = True
                except queue.Empty:
                    pass
                
                try:
                    while True:
                        output_type, line, timestamp = stderr_queue.get_nowait()
                        line_clean = line.rstrip()
                        stderr_lines.append(line_clean)
                        if line_clean:
                            print(f"⚠️  {line_clean}")
                        last_output_time = timestamp
                        got_output = True
                except queue.Empty:
                    pass
                
                time_since_last_output = current_time - last_output_time
                total_time = current_time - start_time
                
                if total_time > max_total_time:
                    print(f"\n⏰ Process execution exceeded maximum time limit of {max_total_time} seconds, force terminating")
                    timed_out = True
                    break
                elif time_since_last_output > timeout_inactive:
                    print(f"\n⏰ Process has no output for more than {timeout_inactive} seconds, may be stuck, force terminating")
                    timed_out = True
                    break
                
                time.sleep(0.1)
            
            if timed_out:
                try:
                    process.terminate()
                    print("🔄 Attempting graceful process termination...")
                    try:
                        process.wait(timeout=5)
                        print("✅ Process terminated gracefully")
                    except subprocess.TimeoutExpired:
                        print("💀 Force killing process...")
                        process.kill()
                        process.wait()
                        print("✅ Process force terminated")
                except:
                    pass
            
            try:
                while True:
                    output_type, line, timestamp = stdout_queue.get_nowait()
                    line_clean = line.rstrip()
                    stdout_lines.append(line_clean)
                    if line_clean:
                        print(f"📤 {line_clean}")
            except queue.Empty:
                pass
            
            try:
                while True:
                    output_type, line, timestamp = stderr_queue.get_nowait()
                    line_clean = line.rstrip()
                    stderr_lines.append(line_clean)
                    if line_clean:
                        print(f"⚠️  {line_clean}")
            except queue.Empty:
                pass
                
        except KeyboardInterrupt:
            print("\n⏰ User interrupted, terminating process")
            timed_out = True
            try:
                process.terminate()
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            except:
                pass
        
        return_code = process.returncode if process.returncode is not None else -1
        
        pass  # Separator line removed
        if timed_out:
            print("⏰ Command execution timed out")
        elif return_code == 0:
            print("✅ Command execution completed successfully")
        else:
            print(f"❌ Command execution failed, exit code: {return_code}")
        pass  # Separator line removed
        
        return '\n'.join(stdout_lines), '\n'.join(stderr_lines), return_code, timed_out

    def _detect_interactive_command(self, command: str) -> bool:
        """检测命令是否可能需要交互式输入"""
        interactive_patterns = [
            r'\bsudo\b(?!\s+(-n|--non-interactive))',  # sudo without -n flag
            r'\bapt\s+(?:install|upgrade|update)\b(?!.*-y)',  # apt without -y flag
            r'\byum\s+(?:install|update)\b(?!.*-y)',  # yum without -y flag
            r'\bdnf\s+(?:install|update)\b(?!.*-y)',  # dnf without -y flag
            r'\bpip\s+install\b(?!.*--quiet)',  # pip install without --quiet
            r'\bgit\s+(?:push|pull)\b',  # git operations that might need credentials
            r'\bssh\b',  # ssh connections
            r'\bmysql\b',  # mysql client
            r'\bpsql\b',  # postgresql client
        ]
        
        return any(re.search(pattern, command, re.IGNORECASE) for pattern in interactive_patterns)
    
    def _make_command_non_interactive(self, command: str) -> str:
        """将交互式命令转换为非交互式版本"""
        original_command = command
        
        # sudo 命令添加 -n 标志（非交互式）
        if re.search(r'\bsudo\b(?!\s+(-n|--non-interactive))', command, re.IGNORECASE):
            command = re.sub(r'\bsudo\b', 'sudo -n', command, flags=re.IGNORECASE)
        
        # apt 命令添加 -y 标志
        if re.search(r'\bapt\s+(?:install|upgrade|update)\b(?!.*-y)', command, re.IGNORECASE):
            command = re.sub(r'\b(apt\s+(?:install|upgrade|update))\b', r'\1 -y', command, flags=re.IGNORECASE)
        
        # yum 命令添加 -y 标志
        if re.search(r'\byum\s+(?:install|update)\b(?!.*-y)', command, re.IGNORECASE):
            command = re.sub(r'\b(yum\s+(?:install|update))\b', r'\1 -y', command, flags=re.IGNORECASE)
        
        # dnf 命令添加 -y 标志
        if re.search(r'\bdnf\s+(?:install|update)\b(?!.*-y)', command, re.IGNORECASE):
            command = re.sub(r'\b(dnf\s+(?:install|update))\b', r'\1 -y', command, flags=re.IGNORECASE)
        
        # pip 命令添加 --quiet 标志
        if re.search(r'\bpip\s+install\b(?!.*--quiet)', command, re.IGNORECASE):
            command = re.sub(r'\b(pip\s+install)\b', r'\1 --quiet', command, flags=re.IGNORECASE)
        
        return command
    
    def _provide_command_suggestions(self, command: str) -> str:
        """为交互式命令提供建议"""
        suggestions = []
        
        if 'sudo' in command.lower() and '-n' not in command:
            suggestions.append("💡 建议：使用 'sudo -n' 进行非交互式执行，或预先配置sudo免密")
        
        if re.search(r'\bapt\s+(?:install|upgrade|update)\b', command, re.IGNORECASE) and '-y' not in command:
            suggestions.append("💡 建议：使用 'apt -y' 自动确认安装")
        
        if 'git push' in command.lower() or 'git pull' in command.lower():
            suggestions.append("💡 建议：配置SSH密钥或使用个人访问令牌避免密码输入")
        
        if 'ssh' in command.lower():
            suggestions.append("💡 建议：使用SSH密钥认证避免密码输入")
        
        return "\n".join(suggestions)

    def run_terminal_cmd(self, command: str, is_background: bool = False, 
                        timeout_inactive: int = 300, max_total_time: int = 600, 
                        auto_fix_interactive: bool = None, **kwargs) -> Dict[str, Any]:
        """
        Run a terminal command with smart timeout handling and interactive command detection.
        
        Args:
            command: 要执行的命令
            is_background: 是否后台执行
            timeout_inactive: 无输出超时时间
            max_total_time: 最大执行时间
            auto_fix_interactive: 是否自动修复交互式命令（如果为None，从配置文件读取）
        """
        print("Running terminal command")
        
        # 如果没有指定auto_fix_interactive参数，从配置文件读取
        if auto_fix_interactive is None:
            auto_fix_interactive = get_auto_fix_interactive_commands()
        
        # Ignore additional parameters
        if kwargs:
            print(f"⚠️  Ignoring additional parameters: {list(kwargs.keys())}")
        
        # Auto-correct HTML entities in command
        original_command = command
        command = self._fix_html_entities(command)
        
        # 检测是否为交互式命令
        is_interactive = self._detect_interactive_command(command)
        
        if is_interactive:
            if auto_fix_interactive:
                command = self._make_command_non_interactive(command)
            else:
                suggestions = self._provide_command_suggestions(command)
                if suggestions:
                    print(suggestions)
        
        print(f"Command: {command}")
        print(f"Working directory: {self.workspace_root}")
        print(f"Absolute working directory: {os.path.abspath(self.workspace_root)}")
        
        try:
            if is_background:
                process = subprocess.Popen(
                    command,
                    shell=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    cwd=self.workspace_root
                )
                return {
                    'status': 'started_background',
                    'command': command,
                    'pid': process.pid,
                    'working_directory': self.workspace_root
                }
            else:
                gui_indicators = [
                    'open ', 'start ', 'xdg-open', 'gnome-open', 'kde-open',
                    'firefox', 'chrome', 'safari', 'explorer',
                    'notepad', 'gedit', 'nano', 'vim', 'emacs',
                    'python -m', 'pygame', 'tkinter', 'qt', 'gtk'
                ]
                
                is_potentially_interactive = any(indicator in command.lower() for indicator in gui_indicators)
                
                if is_potentially_interactive:
                    timeout_inactive = min(timeout_inactive, 60)
                    max_total_time = min(max_total_time, 180)
                    print(f"🖥️ 检测到可能的交互式/GUI程序，使用较短超时: {timeout_inactive}s无输出超时, {max_total_time}s最大执行时间")
                
                long_running_indicators = [
                    'git clone', 'git fetch', 'git pull', 'git push',
                    'npm install', 'pip install', 'yarn install',
                    'docker build', 'docker pull', 'docker push',
                    'wget', 'curl -O', 'scp', 'rsync',
                    'make', 'cmake', 'gcc', 'g++', 'javac',
                    'python setup.py', 'python -m pip'
                ]
                
                is_potentially_long_running = any(indicator in command.lower() for indicator in long_running_indicators)
                
                if is_potentially_long_running:
                    timeout_inactive = max(timeout_inactive, 600)
                    max_total_time = max(max_total_time, 1800)
                    print(f"⏳ 检测到可能的长时间运行命令，使用更长超时: {timeout_inactive}s无输出超时, {max_total_time}s最大执行时间")
                
                # 对于交互式命令，使用特殊的环境变量
                env = None
                if is_interactive:
                    env = os.environ.copy()
                    env['DEBIAN_FRONTEND'] = 'noninteractive'  # 对于apt命令
                    env['NEEDRESTART_MODE'] = 'a'  # 自动重启服务
                
                process = subprocess.Popen(
                    command,
                    shell=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    bufsize=0,
                    universal_newlines=True,
                    cwd=self.workspace_root,
                    env=env
                )
                
                stdout, stderr, return_code, timed_out = self._read_process_output_with_timeout(
                    process, timeout_inactive, max_total_time
                )
                
                status = 'completed'
                if timed_out:
                    status = 'timeout'
                elif return_code != 0:
                    status = 'error'
                
                result = {
                    'status': status,
                    'command': command,
                    'original_command': original_command,
                    'stdout': stdout,
                    'stderr': stderr,
                    'return_code': return_code,
                    'working_directory': self.workspace_root,
                    'timeout_inactive': timeout_inactive,
                    'max_total_time': max_total_time,
                    'was_interactive': is_interactive
                }
                
                if timed_out:
                    result['timeout_reason'] = 'Process timed out due to inactivity or maximum time limit'
                
                # 如果是交互式命令且失败了，提供额外的帮助信息
                if is_interactive and return_code != 0:
                    suggestions = self._provide_command_suggestions(original_command)
                    if suggestions:
                        result['suggestions'] = suggestions
                        print("\n" + suggestions)
                
                return result
                
        except Exception as e:
            return {
                'status': 'error',
                'command': command,
                'original_command': original_command,
                'error': str(e),
                'working_directory': self.workspace_root,
                'was_interactive': is_interactive
            } 