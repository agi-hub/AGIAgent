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
    
    def _read_process_output_with_timeout(self, process, timeout_inactive=180, max_total_time=300):
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
        """Detect if command might require interactive input"""
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
        """Convert interactive commands to non-interactive versions"""
        original_command = command
        
        # Add -n flag to sudo commands (non-interactive)
        if re.search(r'\bsudo\b(?!\s+(-n|--non-interactive))', command, re.IGNORECASE):
            command = re.sub(r'\bsudo\b', 'sudo -n', command, flags=re.IGNORECASE)
        
        # Add -y flag to apt commands
        if re.search(r'\bapt\s+(?:install|upgrade|update)\b(?!.*-y)', command, re.IGNORECASE):
            command = re.sub(r'\b(apt\s+(?:install|upgrade|update))\b', r'\1 -y', command, flags=re.IGNORECASE)
        
        # Add -y flag to yum commands
        if re.search(r'\byum\s+(?:install|update)\b(?!.*-y)', command, re.IGNORECASE):
            command = re.sub(r'\b(yum\s+(?:install|update))\b', r'\1 -y', command, flags=re.IGNORECASE)
        
        # Add -y flag to dnf commands
        if re.search(r'\bdnf\s+(?:install|update)\b(?!.*-y)', command, re.IGNORECASE):
            command = re.sub(r'\b(dnf\s+(?:install|update))\b', r'\1 -y', command, flags=re.IGNORECASE)
        
        # Add --quiet flag to pip commands
        if re.search(r'\bpip\s+install\b(?!.*--quiet)', command, re.IGNORECASE):
            command = re.sub(r'\b(pip\s+install)\b', r'\1 --quiet', command, flags=re.IGNORECASE)
        
        return command
    
    def _provide_command_suggestions(self, command: str) -> str:
        """Provide suggestions for interactive commands"""
        suggestions = []
        
        if 'sudo' in command.lower() and '-n' not in command:
            suggestions.append("💡 Suggestion: Use 'sudo -n' for non-interactive execution, or configure passwordless sudo")
        
        if re.search(r'\bapt\s+(?:install|upgrade|update)\b', command, re.IGNORECASE) and '-y' not in command:
            suggestions.append("💡 Suggestion: Use 'apt -y' to automatically confirm installation")
        
        if 'git push' in command.lower() or 'git pull' in command.lower():
            suggestions.append("💡 Suggestion: Configure SSH keys or use personal access tokens to avoid password input")
        
        if 'ssh' in command.lower():
            suggestions.append("💡 Suggestion: Use SSH key authentication to avoid password input")
        
        return "\n".join(suggestions)

    def run_terminal_cmd(self, command: str, is_background: bool = False, 
                        timeout_inactive: int = 180, max_total_time: int = 300, 
                        auto_fix_interactive: bool = None, **kwargs) -> Dict[str, Any]:
        """
        Run a terminal command with smart timeout handling and interactive command detection.
        
        Args:
            command: Command to execute
            is_background: Whether to run in background
            timeout_inactive: Timeout for no output
            max_total_time: Maximum execution time
            auto_fix_interactive: Whether to auto-fix interactive commands (if None, read from config file)
        """
        print("Running terminal command")
        
        # If auto_fix_interactive parameter is not specified, read from config file
        if auto_fix_interactive is None:
            auto_fix_interactive = get_auto_fix_interactive_commands()
        
        # Ignore additional parameters
        if kwargs:
            print(f"⚠️  Ignoring additional parameters: {list(kwargs.keys())}")
        
        # Auto-correct HTML entities in command
        original_command = command
        command = self._fix_html_entities(command)
        
        # Detect if it's an interactive command
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
                    print(f"🖥️ Detected potential interactive/GUI program, using shorter timeout: {timeout_inactive}s no output timeout, {max_total_time}s maximum execution time")
                
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
                    print(f"⏳ Detected potential long-running command, using longer timeout: {timeout_inactive}s no output timeout, {max_total_time}s maximum execution time")
                
                # For interactive commands, use special environment variables
                env = None
                if is_interactive:
                    env = os.environ.copy()
                    env['DEBIAN_FRONTEND'] = 'noninteractive'  # For apt commands
                    env['NEEDRESTART_MODE'] = 'a'  # Auto restart services
                
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
                
                # If it's an interactive command and it failed, provide additional help information
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