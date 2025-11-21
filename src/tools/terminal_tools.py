#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from .print_system import print_system, print_current
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
"""

import subprocess
import time
import queue
import threading
import re
import os
from typing import Dict, Any
import sys
import signal
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
                print_current(f"🔧 Auto-corrected HTML entities in command: {', '.join(corrections)}")
            else:
                print_current(f"🔧 Auto-corrected HTML entities in command (various types found)")
        
        return decoded_text
    
    def _read_process_output_with_timeout_and_input(self, process, timeout_inactive=180, max_total_time=300):
        """
        Asynchronously read process output with smart timeout, while displaying real-time output to user
        """
        stdout_lines = []
        stderr_lines = []
        last_output_time = time.time()
        start_time = time.time()
        last_progress_line = None  # Track last progress line to avoid duplicates
        
        stdout_queue = queue.Queue()
        stderr_queue = queue.Queue()
        
        def read_stdout():
            try:
                # Use buffered reading to detect \r characters before readline processes them
                buffer = ''
                chunk_size = 1024
                while True:
                    chunk = process.stdout.read(chunk_size)
                    if not chunk:
                        break
                    # Debug: track that we're receiving data
                    if chunk.strip():
                        stdout_queue.put(('debug', f"[DEBUG] Received stdout chunk: {len(chunk)} chars", time.time(), False))
                    buffer += chunk
                    
                    # Process buffer for \r (carriage return) and \n (newline)
                    while True:
                        # Check for \r first (progress bar updates)
                        if '\r' in buffer:
                            # Find the position of \r
                            cr_pos = buffer.find('\r')
                            # Extract content before \r
                            line = buffer[:cr_pos]
                            # Remove any trailing \n
                            line = line.rstrip('\n')
                            if line.strip():
                                # This is a progress bar update
                                stdout_queue.put(('stdout', line + '\n', time.time(), True))
                            # Remove processed part including \r
                            buffer = buffer[cr_pos + 1:]
                            continue
                        
                        # Check for \n (regular line ending)
                        if '\n' in buffer:
                            nl_pos = buffer.find('\n')
                            line = buffer[:nl_pos]
                            # Always process the line, even if it's empty (to preserve formatting)
                            is_progress = any(indicator in line for indicator in ['%', '|', '#', '/', 'it/s', 's/it', 'ETA', '进度', 'MB', 'KB', 'GB', 'kB/s', 'MB/s', 'GB/s', '━', '█', 'eta']) if line.strip() else False
                            stdout_queue.put(('stdout', line + '\n', time.time(), is_progress))
                            buffer = buffer[nl_pos + 1:]
                            continue
                        
                        # No more complete lines in buffer
                        break
                
                # Process remaining buffer
                if buffer.strip():
                    # Check if remaining buffer looks like progress bar
                    is_progress = any(indicator in buffer for indicator in ['%', '|', '#', '/', 'it/s', 's/it', 'ETA', '进度', 'MB', 'KB', 'GB', 'kB/s', 'MB/s', 'GB/s', '━', '█', 'eta'])
                    stdout_queue.put(('stdout', buffer + '\n', time.time(), is_progress))
                process.stdout.close()
            except:
                pass
        
        def read_stderr():
            try:
                # Use buffered reading to detect \r characters
                buffer = ''
                chunk_size = 1024
                while True:
                    chunk = process.stderr.read(chunk_size)
                    if not chunk:
                        break
                    # Debug: track that we're receiving data
                    if chunk.strip():
                        stderr_queue.put(('debug', f"[DEBUG] Received stderr chunk: {len(chunk)} chars", time.time(), False))
                    buffer += chunk
                    
                    # Process buffer for \r and \n
                    while True:
                        # Check for \r first (progress bar updates)
                        if '\r' in buffer:
                            cr_pos = buffer.find('\r')
                            line = buffer[:cr_pos].rstrip('\n')
                            if line.strip():
                                stderr_queue.put(('stderr', line + '\n', time.time(), True))
                            buffer = buffer[cr_pos + 1:]
                            continue
                        
                        # Check for \n (regular line ending)
                        if '\n' in buffer:
                            nl_pos = buffer.find('\n')
                            line = buffer[:nl_pos]
                            # Always process the line, even if it's empty (to preserve formatting)
                            is_progress = any(indicator in line for indicator in ['%', '|', '#', '/', 'it/s', 's/it', 'ETA', '进度', 'MB', 'KB', 'GB', 'kB/s', 'MB/s', 'GB/s', '━', '█', 'eta']) if line.strip() else False
                            stderr_queue.put(('stderr', line + '\n', time.time(), is_progress))
                            buffer = buffer[nl_pos + 1:]
                            continue
                        
                        break
                
                # Process remaining buffer
                if buffer.strip():
                    is_progress = any(indicator in buffer for indicator in ['%', '|', '#', '/', 'it/s', 's/it', 'ETA', '进度', 'MB', 'KB', 'GB', 'kB/s', 'MB/s', 'GB/s', '━', '█', 'eta'])
                    stderr_queue.put(('stderr', buffer + '\n', time.time(), is_progress))
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
        
        
        try:
            while process.poll() is None:
                current_time = time.time()
                
                got_output = False
                
                # No GUI input handling in terminal mode
                
                try:
                    while True:
                        item = stdout_queue.get_nowait()
                        # Handle both old format (3 items) and new format (4 items with is_update)
                        if len(item) == 4:
                            output_type, line, timestamp, is_update = item
                        else:
                            output_type, line, timestamp = item
                            is_update = False
                        
                        # Handle debug messages
                        if output_type == 'debug':
                            print_current(line)  # Debug messages already formatted
                            last_output_time = timestamp
                            got_output = True
                            continue
                        
                        # Clean line: remove \r and trailing whitespace
                        line_clean = line.replace('\r', '').rstrip()
                        stdout_lines.append(line_clean)
                        # Always display the line, even if empty (to preserve formatting)
                        if is_update:
                            # For progress bar updates, avoid showing duplicate progress lines
                            if line_clean != last_progress_line:
                                print_current(f"📤 {line_clean}")
                                last_progress_line = line_clean
                        else:
                            print_current(f"📤 {line_clean}")
                            # Reset progress line tracking for non-progress output
                            last_progress_line = None
                        last_output_time = timestamp
                        got_output = True
                except queue.Empty:
                    pass
                
                try:
                    while True:
                        item = stderr_queue.get_nowait()
                        # Handle both old format (3 items) and new format (4 items with is_update)
                        if len(item) == 4:
                            output_type, line, timestamp, is_update = item
                        else:
                            output_type, line, timestamp = item
                            is_update = False
                        
                        # Handle debug messages
                        if output_type == 'debug':
                            print_current(line)  # Debug messages already formatted
                            last_output_time = timestamp
                            got_output = True
                            continue
                        
                        # Clean line: remove \r and trailing whitespace
                        line_clean = line.replace('\r', '').rstrip()
                        stderr_lines.append(line_clean)
                        # Always display the line, even if empty (to preserve formatting)
                        if is_update:
                            # For progress bar updates in GUI, don't use \r as it causes display issues
                            # Just print normally to avoid overwriting previous lines
                            print_current(f"⚠️  {line_clean}")
                        else:
                            print_current(f"⚠️  {line_clean}")
                        last_output_time = timestamp
                        got_output = True
                except queue.Empty:
                    pass
                
                time_since_last_output = current_time - last_output_time
                total_time = current_time - start_time
                
                if total_time > max_total_time:
                    print_current(f"\n⏰ Process execution exceeded maximum time limit of {max_total_time} seconds, force terminating")
                    timed_out = True
                    break
                elif time_since_last_output > timeout_inactive:
                    print_current(f"\n⏰ Process has no output for more than {timeout_inactive} seconds, may be stuck, force terminating")
                    timed_out = True
                    break
                
                time.sleep(0.1)
            
            if timed_out:
                try:
                    process.terminate()
                    print_current("🔄 Attempting graceful process termination...")
                    try:
                        process.wait(timeout=5)
                        print_current("✅ Process terminated gracefully")
                    except subprocess.TimeoutExpired:
                        print_current("💀 Force killing process...")
                        process.kill()
                        process.wait()
                        print_current("✅ Process force terminated")
                except:
                    pass
            
            try:
                while True:
                    item = stdout_queue.get_nowait()
                    # Handle both old format (3 items) and new format (4 items with is_update)
                    if len(item) == 4:
                        output_type, line, timestamp, is_update = item
                    else:
                        output_type, line, timestamp = item
                        is_update = False
                    
                    # Clean line: remove \r and trailing whitespace
                    line_clean = line.replace('\r', '').rstrip()
                    stdout_lines.append(line_clean)
                    # Always display the line, even if empty
                    if is_update:
                        # For progress bar updates in GUI, don't use \r as it causes display issues
                        print_current(f"📤 {line_clean}")
                    else:
                        print_current(f"📤 {line_clean}")
            except queue.Empty:
                pass
            
            try:
                while True:
                    item = stderr_queue.get_nowait()
                    # Handle both old format (3 items) and new format (4 items with is_update)
                    if len(item) == 4:
                        output_type, line, timestamp, is_update = item
                    else:
                        output_type, line, timestamp = item
                        is_update = False
                    
                    # Clean line: remove \r and trailing whitespace
                    line_clean = line.replace('\r', '').rstrip()
                    stderr_lines.append(line_clean)
                    # Always display the line, even if empty
                    if is_update:
                        # For progress bar updates in GUI, don't use \r as it causes display issues
                        print_current(f"⚠️  {line_clean}")
                    else:
                        print_current(f"⚠️  {line_clean}")
            except queue.Empty:
                pass
                
        except KeyboardInterrupt:
            print_current("\n⏰ User interrupted, terminating process")
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
        

        if timed_out:
            print_current("⏰ Command execution timed out")
        elif return_code == 0:
            pass
            #print_current("✅ Command execution completed successfully")
        # Removed the failure status print - no longer printing failure messages

        
        return '\n'.join(stdout_lines), '\n'.join(stderr_lines), return_code, timed_out

    def _detect_interactive_command(self, command: str) -> bool:
        """Detect if command might require interactive input"""
        interactive_patterns = [
            r'\bsudo\b(?!\s+(-n|--non-interactive))',  # sudo without -n flag
            r'\bapt\s+(?:install|upgrade|update)\b(?!.*-y)',  # apt without -y flag
            r'\byum\s+(?:install|update)\b(?!.*-y)',  # yum without -y flag
            r'\bdnf\s+(?:install|update)\b(?!.*-y)',  # dnf without -y flag
            # Note: pip install is not interactive by default, so we don't detect it here
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
        
        # Note: pip install is not truly interactive and doesn't need --quiet flag
        # Removing --quiet allows users to see detailed installation progress
        # If users want quiet mode, they can add --quiet flag explicitly
        
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

    def talk_to_user(self, query: str, timeout: int = 10) -> Dict[str, Any]:
        """
        Display a question to the user and wait for keyboard input with timeout.
        
        Args:
            query: The question to display to the user
            timeout: Maximum time to wait for user response (default: 10 seconds, -1 to disable timeout)
            
        Returns:
            Dict containing the user's response or timeout indication
        """
        print_current(f"❓ {query}")
        
        # Check if timeout is disabled
        if timeout == -1:
            print_current("⏱️  Waiting for user reply (no timeout)...")
        else:
            print_current(f"⏱️  Waiting for user reply ({timeout}seconds)...")
        
        # Create a queue to communicate between threads
        response_queue = queue.Queue()
        
        def get_user_input():
            """Thread function to get user input"""
            try:
                user_input = input("👤 Please enter your reply: ")
                response_queue.put(('success', user_input.strip()))
            except EOFError:
                # Handle Ctrl+D or end of input
                response_queue.put(('error', 'EOF'))
            except KeyboardInterrupt:
                # Handle Ctrl+C
                response_queue.put(('error', 'KeyboardInterrupt'))
            except Exception as e:
                response_queue.put(('error', str(e)))
        
        # Start input thread
        input_thread = threading.Thread(target=get_user_input)
        input_thread.daemon = True
        input_thread.start()
        
        # Wait for response or timeout
        try:
            if timeout == -1:
                # No timeout - wait indefinitely
                status, response = response_queue.get()
            else:
                # Normal timeout behavior
                status, response = response_queue.get(timeout=timeout)
            
            if status == 'success':
                print_current(f"✅ User reply: {response}")
                return {
                    'status': 'success',
                    'query': query,
                    'user_response': response,
                    'timeout': timeout,
                    'response_time': 'within_timeout' if timeout != -1 else 'no_timeout'
                }
            else:
                print_current(f"❌ Input error: {response}")
                return {
                    'status': 'failed',
                    'query': query,
                    'user_response': 'no user response',
                    'timeout': timeout,
                    'response_time': 'error',
                    'error': response
                }
                
        except queue.Empty:
            # Timeout occurred (only possible when timeout != -1)
            print_current("⏰ User did not reply within specified time")
            return {
                'status': 'failed',
                'query': query,
                'user_response': 'no user response',
                'timeout': timeout,
                'response_time': 'timeout'
            }
        except Exception as e:
            print_current(f"❌ Error occurred while waiting for user input: {e}")
            return {
                'status': 'failed',
                'query': query,
                'user_response': 'no user response',
                'timeout': timeout,
                'response_time': 'error',
                'error': str(e)
            }

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
        # If auto_fix_interactive parameter is not specified, read from config file
        if auto_fix_interactive is None:
            auto_fix_interactive = get_auto_fix_interactive_commands()
        
        # Ignore additional parameters
        if kwargs:
            print_current(f"⚠️  Ignoring additional parameters: {list(kwargs.keys())}")
        
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

        
        try:
            if is_background:
                process = subprocess.Popen(
                    command,
                    shell=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    errors='ignore',
                    cwd=self.workspace_root
                )
                return {
                    'status': 'started_background',
                    'command': command,
                    'pid': process.pid,
                    'working_directory': self.workspace_root
                }
            else:
                # Initialize env variable to None - will be set if needed
                env = None
                
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
                    print_current(f"🖥️ Detected potential interactive/GUI program, using shorter timeout: {timeout_inactive}s no output timeout, {max_total_time}s maximum execution time")
                
                # Special handling for pip install - use 2 minutes timeout when no output
                is_pip_install = 'pip install' in command.lower() or 'python -m pip install' in command.lower()
                
                if is_pip_install:
                    # Use 2 minutes (120 seconds) timeout when no output
                    timeout_inactive = 120
                    # Keep max_total_time at default or use a reasonable value
                    if max_total_time < 600:
                        max_total_time = 600  # 10 minutes maximum execution time
                    print_current(f"⏱️  Detected pip install command, using timeout: {timeout_inactive}s no output timeout, {max_total_time}s maximum execution time")
                    
                    # Ensure pip uses unbuffered output for better visibility
                    # Set environment variables for unbuffered Python output
                    if env is None:
                        env = os.environ.copy()
                    env['PIP_PROGRESS_BAR'] = 'on'
                    env['FORCE_COLOR'] = '1'
                    env['PIP_DISABLE_PIP_VERSION_CHECK'] = '1'  # Reduce noise
                
                # Special handling for Python programs - ensure output is visible
                is_python_program = command.lower().startswith('python ') or 'python' in command.lower()
                
                if is_python_program:
                    print_current(f"🐍 Detected Python program, ensuring unbuffered output")
                    # Use shorter timeout for Python programs as they should produce output
                    if timeout_inactive > 60:
                        timeout_inactive = 60  # 1 minute timeout for Python programs
                    print_current(f"⏱️  Using timeout: {timeout_inactive}s no output timeout, {max_total_time}s maximum execution time")
                
                long_running_indicators = [
                    'git clone', 'git fetch', 'git pull', 'git push',
                    'npm install', 'yarn install',
                    'docker build', 'docker pull', 'docker push',
                    'wget', 'curl -O', 'scp', 'rsync',
                    'make', 'cmake', 'gcc', 'g++', 'javac',
                    'python setup.py'
                ]
                
                # Only apply long-running timeout if not pip install
                is_potentially_long_running = not is_pip_install and any(indicator in command.lower() for indicator in long_running_indicators)
                
                if is_potentially_long_running:
                    timeout_inactive = max(timeout_inactive, 600)
                    max_total_time = max(max_total_time, 1800)
                    #print_current(f"⏳ Detected potential long-running command, using longer timeout: {timeout_inactive}s no output timeout, {max_total_time}s maximum execution time")
                
                # For interactive commands, use special environment variables
                # Initialize env if needed (for interactive commands or if already set by pip install)
                if is_interactive:
                    # Always set noninteractive mode for automated execution
                    if env is None:
                        env = os.environ.copy()
                    env['DEBIAN_FRONTEND'] = 'noninteractive'  # For apt commands
                    env['NEEDRESTART_MODE'] = 'a'  # Auto restart services
                    #print_current("🔧 DEBUG: Set noninteractive environment for interactive command")
                
                # Always ensure proper encoding for all commands
                if env is None:
                    env = os.environ.copy()
                # Force UTF-8 encoding and unbuffered output for all Python programs
                env['PYTHONUNBUFFERED'] = '1'
                env['PYTHONIOENCODING'] = 'utf-8'
                # Set console encoding for Windows
                if os.name == 'nt':  # Windows
                    env['PYTHONLEGACYWINDOWSSTDIO'] = '1'
                
                # Ensure env is initialized before subprocess.Popen
                # If env is still None, use current process environment (default behavior)
                popen_kwargs = {
                    'shell': True,
                    'stdout': subprocess.PIPE,
                    'stderr': subprocess.PIPE,
                    'stdin': subprocess.PIPE,  # Enable stdin for interactive input
                    'text': True,
                    'encoding': 'utf-8',  # Explicitly set encoding
                    'errors': 'replace',  # Replace invalid characters instead of ignoring
                    'bufsize': 0,  # Unbuffered
                    'universal_newlines': True,
                    'cwd': self.workspace_root,
                    'env': env  # Always pass env (now always initialized)
                }
                
                process = subprocess.Popen(command, **popen_kwargs)
                
                # Add debug info for output capture
                print_current(f"🔍 Process started with PID: {process.pid}")
                
                stdout, stderr, return_code, timed_out = self._read_process_output_with_timeout_and_input(
                    process, timeout_inactive, max_total_time
                )
                
                status = 'success'
                if timed_out:
                    status = 'failed'
                elif return_code != 0:
                    # Special handling for 'which' command - exit code 1 means command not found, which is normal
                    if command.strip().startswith('which ') and return_code == 1:
                        status = 'success'  # which command returning 1 is normal when command not found
                    else:
                        status = 'failed'
                
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
                        print_current("\n" + suggestions)
                
                return result
                
        except Exception as e:
            return {
                'status': 'failed',
                'command': command,
                'original_command': original_command,
                'error': str(e),
                'working_directory': self.workspace_root,
                'was_interactive': is_interactive
            } 