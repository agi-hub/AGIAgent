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
        has_received_output = False  # Track if we've ever received output
        
        stdout_queue = queue.Queue()
        stderr_queue = queue.Queue()
        
        # Shared state for buffer flushing (for interactive programs)
        stdout_buffer_lock = threading.Lock()
        stdout_buffer_state = {'content': '', 'last_update': time.time()}
        stderr_buffer_lock = threading.Lock()
        stderr_buffer_state = {'content': '', 'last_update': time.time()}
        
        def flush_buffers_periodically(flush_interval=0.3):
            """Periodically flush buffer content even without newline for interactive programs"""
            while process.poll() is None:
                time.sleep(flush_interval)
                current_time = time.time()
                
                # Flush stdout buffer if it has content and hasn't been updated recently
                with stdout_buffer_lock:
                    if stdout_buffer_state['content']:
                        time_since_update = current_time - stdout_buffer_state['last_update']
                        # If buffer hasn't been updated for a while, flush it (program might be waiting for input)
                        if time_since_update >= flush_interval:
                            content = stdout_buffer_state['content']
                            if content.strip():
                                is_progress = any(indicator in content for indicator in ['%', '|', '#', '/', 'it/s', 's/it', 'ETA', '进度', 'MB', 'KB', 'GB', 'kB/s', 'MB/s', 'GB/s', '━', '█', 'eta'])
                                stdout_queue.put(('stdout', content + '\n', current_time, is_progress))
                            stdout_buffer_state['content'] = ''
                
                # Flush stderr buffer if it has content and hasn't been updated recently
                with stderr_buffer_lock:
                    if stderr_buffer_state['content']:
                        time_since_update = current_time - stderr_buffer_state['last_update']
                        if time_since_update >= flush_interval:
                            content = stderr_buffer_state['content']
                            if content.strip():
                                is_progress = any(indicator in content for indicator in ['%', '|', '#', '/', 'it/s', 's/it', 'ETA', '进度', 'MB', 'KB', 'GB', 'kB/s', 'MB/s', 'GB/s', '━', '█', 'eta'])
                                stderr_queue.put(('stderr', content + '\n', current_time, is_progress))
                            stderr_buffer_state['content'] = ''
        
        def read_stdout():
            try:
                # Use more aggressive reading for interactive programs
                # Try to use read1() from underlying buffer for non-blocking reads
                # Fallback to read(1) for immediate response
                buffer = ''
                # Check if we can access underlying buffer's read1() method
                use_read1 = False
                read1_func = None
                try:
                    if hasattr(process.stdout, 'buffer') and hasattr(process.stdout.buffer, 'read1'):
                        read1_func = process.stdout.buffer.read1
                        use_read1 = True
                    elif hasattr(process.stdout, 'read1'):
                        read1_func = process.stdout.read1
                        use_read1 = True
                except:
                    pass
                
                while True:
                    # Try to read available data (non-blocking approach)
                    try:
                        if use_read1 and read1_func:
                            # read1() reads at least 1 byte if available, doesn't wait for full buffer
                            raw_chunk = read1_func(8192)  # Max bytes to read, but reads whatever is available
                            # Decode the raw bytes if needed
                            if isinstance(raw_chunk, bytes):
                                chunk = raw_chunk.decode('utf-8', errors='replace')
                            else:
                                chunk = raw_chunk
                        else:
                            # Fallback: use read() with smaller chunk size for better responsiveness
                            chunk = process.stdout.read(1)  # Read 1 byte at a time for immediate response
                    except (OSError, ValueError, AttributeError):
                        # Stream might be closed or in invalid state
                        chunk = ''
                    
                    if not chunk:
                        # No data available, check if process has ended
                        if process.poll() is not None:
                            break
                        # Store current buffer for periodic flushing
                        if buffer:
                            with stdout_buffer_lock:
                                stdout_buffer_state['content'] = buffer
                                stdout_buffer_state['last_update'] = time.time()
                        # Small sleep to avoid busy waiting
                        time.sleep(0.01)
                        continue
                    
                    # Clear buffer state when new data arrives
                    with stdout_buffer_lock:
                        stdout_buffer_state['content'] = ''
                    
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
                        # Store remaining buffer for periodic flushing (for interactive programs)
                        if buffer:
                            with stdout_buffer_lock:
                                stdout_buffer_state['content'] = buffer
                                stdout_buffer_state['last_update'] = time.time()
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
                # Use more aggressive reading for interactive programs
                # Try to use read1() from underlying buffer for non-blocking reads
                # Fallback to read(1) for immediate response
                buffer = ''
                # Check if we can access underlying buffer's read1() method
                use_read1 = False
                read1_func = None
                try:
                    if hasattr(process.stderr, 'buffer') and hasattr(process.stderr.buffer, 'read1'):
                        read1_func = process.stderr.buffer.read1
                        use_read1 = True
                    elif hasattr(process.stderr, 'read1'):
                        read1_func = process.stderr.read1
                        use_read1 = True
                except:
                    pass
                
                while True:
                    # Try to read available data (non-blocking approach)
                    try:
                        if use_read1 and read1_func:
                            # read1() reads at least 1 byte if available, doesn't wait for full buffer
                            raw_chunk = read1_func(8192)  # Max bytes to read, but reads whatever is available
                            # Decode the raw bytes if needed
                            if isinstance(raw_chunk, bytes):
                                chunk = raw_chunk.decode('utf-8', errors='replace')
                            else:
                                chunk = raw_chunk
                        else:
                            # Fallback: use read() with smaller chunk size for better responsiveness
                            chunk = process.stderr.read(1)  # Read 1 byte at a time for immediate response
                    except (OSError, ValueError, AttributeError):
                        # Stream might be closed or in invalid state
                        chunk = ''
                    
                    if not chunk:
                        # No data available, check if process has ended
                        if process.poll() is not None:
                            break
                        # Store current buffer for periodic flushing
                        if buffer:
                            with stderr_buffer_lock:
                                stderr_buffer_state['content'] = buffer
                                stderr_buffer_state['last_update'] = time.time()
                        # Small sleep to avoid busy waiting
                        time.sleep(0.01)
                        continue
                    
                    # Clear buffer state when new data arrives
                    with stderr_buffer_lock:
                        stderr_buffer_state['content'] = ''
                    
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
                        
                        # No more complete lines in buffer
                        # Store remaining buffer for periodic flushing (for interactive programs)
                        if buffer:
                            with stderr_buffer_lock:
                                stderr_buffer_state['content'] = buffer
                                stderr_buffer_state['last_update'] = time.time()
                        break
                
                # Process remaining buffer
                if buffer.strip():
                    is_progress = any(indicator in buffer for indicator in ['%', '|', '#', '/', 'it/s', 's/it', 'ETA', '进度', 'MB', 'KB', 'GB', 'kB/s', 'MB/s', 'GB/s', '━', '█', 'eta'])
                    stderr_queue.put(('stderr', buffer + '\n', time.time(), is_progress))
                process.stderr.close()
            except:
                pass
        
        # Thread to handle user input for interactive programs
        stdin_closed = threading.Event()
        
        def read_stdin():
            """Read user input from terminal and send to process stdin"""
            try:
                # Only handle stdin if it's a TTY (interactive terminal)
                if sys.stdin.isatty() and process.stdin:
                    while process.poll() is None and not stdin_closed.is_set():
                        try:
                            # Check if stdin is available (non-blocking check)
                            # Use select for Unix, or just try readline for Windows
                            if sys.platform != 'win32':
                                import select
                                # Check if stdin has data available (timeout 0.1 seconds)
                                if select.select([sys.stdin], [], [], 0.1)[0]:
                                    user_input = sys.stdin.readline()
                                else:
                                    # No input available, continue loop
                                    continue
                            else:
                                # Windows: use readline with timeout simulation
                                # For Windows, we'll just use readline which will block
                                # but the daemon thread will be killed when process ends
                                user_input = sys.stdin.readline()
                            
                            if user_input:
                                # Send input to process stdin
                                if process.stdin and not process.stdin.closed:
                                    process.stdin.write(user_input)
                                    process.stdin.flush()
                        except (OSError, ValueError, BrokenPipeError):
                            # Process stdin might be closed
                            break
                        except EOFError:
                            # User pressed Ctrl+D or stdin closed
                            break
                        except Exception:
                            # Other errors, continue
                            pass
            except Exception:
                pass
            finally:
                # Close stdin when done
                try:
                    if process.stdin and not process.stdin.closed:
                        process.stdin.close()
                except:
                    pass
        
        stdout_thread = threading.Thread(target=read_stdout)
        stderr_thread = threading.Thread(target=read_stderr)
        flush_thread = threading.Thread(target=flush_buffers_periodically)
        stdin_thread = threading.Thread(target=read_stdin)
        stdout_thread.daemon = True
        stderr_thread.daemon = True
        flush_thread.daemon = True
        stdin_thread.daemon = True
        stdout_thread.start()
        stderr_thread.start()
        flush_thread.start()  # Start periodic buffer flushing for interactive programs
        stdin_thread.start()  # Start stdin reading thread for interactive programs
        
        timed_out = False
        
        
        try:
            while process.poll() is None:
                current_time = time.time()
                
                got_output = False
                
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
                            has_received_output = True  # Mark that we've received output
                            continue
                        
                        # For progress bar updates (is_update=True), use \r for single-line updates
                        if is_update:
                            # Remove trailing \n and whitespace, but use \r at end to overwrite current line
                            line_clean = line.rstrip('\n').rstrip()
                            stdout_lines.append(line_clean)  # Store clean version
                            # Use \r at end to overwrite current line (tqdm uses \r to return to line start)
                            if line_clean != last_progress_line:
                                print_current(f"📤 {line_clean}", end='\r')
                                last_progress_line = line_clean
                        else:
                            # For regular output, remove \r and trailing whitespace
                            line_clean = line.replace('\r', '').rstrip()
                            stdout_lines.append(line_clean)
                            print_current(f"📤 {line_clean}")
                            # Reset progress line tracking for non-progress output
                            last_progress_line = None
                        last_output_time = timestamp
                        got_output = True
                        has_received_output = True  # Mark that we've received output
                        has_received_output = True  # Mark that we've received output
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
                            has_received_output = True  # Mark that we've received output
                            continue
                        
                        # For progress bar updates (is_update=True), use \r for single-line updates
                        if is_update:
                            # Remove trailing \n and whitespace, but use \r at end to overwrite current line
                            line_clean = line.rstrip('\n').rstrip()
                            stderr_lines.append(line_clean)  # Store clean version
                            # Use \r at end to overwrite current line (tqdm uses \r to return to line start)
                            print_current(f"⚠️  {line_clean}", end='\r')
                        else:
                            # For regular output, remove \r and trailing whitespace
                            line_clean = line.replace('\r', '').rstrip()
                            stderr_lines.append(line_clean)
                            print_current(f"⚠️  {line_clean}")
                        last_output_time = timestamp
                        got_output = True
                        has_received_output = True  # Mark that we've received output
                except queue.Empty:
                    pass
                
                # If we've received output before, use longer timeout (3x the original)
                # This gives more time for processes that are actively producing output
                effective_timeout = timeout_inactive * 3 if has_received_output else timeout_inactive
                
                time_since_last_output = current_time - last_output_time
                total_time = current_time - start_time
                
                if total_time > max_total_time:
                    print_current(f"\n⏰ Process execution exceeded maximum time limit of {max_total_time} seconds, force terminating")
                    timed_out = True
                    break
                elif time_since_last_output > effective_timeout:
                    print_current(f"\n⏰ Process has no output for more than {effective_timeout} seconds, may be stuck, force terminating")
                    timed_out = True
                    break
                
                time.sleep(0.1)
            
            # Signal stdin thread to stop
            stdin_closed.set()
            
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
            
            # Ensure stdin is closed
            try:
                if process.stdin and not process.stdin.closed:
                    process.stdin.close()
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
                    
                    # For progress bar updates (is_update=True), use \r for single-line updates
                    if is_update:
                        # Remove trailing \n and whitespace, but use \r at end to overwrite current line
                        line_clean = line.rstrip('\n').rstrip()
                        stdout_lines.append(line_clean)  # Store clean version
                        # Use \r at end to overwrite current line (tqdm uses \r to return to line start)
                        print_current(f"📤 {line_clean}", end='\r')
                    else:
                        # For regular output, remove \r and trailing whitespace
                        line_clean = line.replace('\r', '').rstrip()
                        stdout_lines.append(line_clean)
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
                    
                    # For progress bar updates (is_update=True), use \r for single-line updates
                    if is_update:
                        # Remove trailing \n and whitespace, but use \r at end to overwrite current line
                        line_clean = line.rstrip('\n').rstrip()
                        stderr_lines.append(line_clean)  # Store clean version
                        # Use \r at end to overwrite current line (tqdm uses \r to return to line start)
                        print_current(f"⚠️  {line_clean}", end='\r')
                    else:
                        # For regular output, remove \r and trailing whitespace
                        line_clean = line.replace('\r', '').rstrip()
                        stderr_lines.append(line_clean)
                        print_current(f"⚠️  {line_clean}")
            except queue.Empty:
                pass
                
        except KeyboardInterrupt:
            print_current("\n⏰ User interrupted, terminating process")
            timed_out = True
            stdin_closed.set()  # Signal stdin thread to stop
            try:
                process.terminate()
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            except:
                pass
            finally:
                # Ensure stdin is closed
                try:
                    if process.stdin and not process.stdin.closed:
                        process.stdin.close()
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
        Supports both terminal and GUI modes.
        
        Args:
            query: The question to display to the user
            timeout: Maximum time to wait for user response (default: 10 seconds, -1 to disable timeout)
            
        Returns:
            Dict containing the user's response or timeout indication
        """
        import sys
        import os
        
        # Detect GUI mode: check environment variable or if stdin is not a TTY
        gui_mode = os.environ.get('AGIA_GUI_MODE', '').lower() == 'true' or not sys.stdin.isatty()
        
        if gui_mode:
            # GUI mode: notify GUI and read from input queue
            # Print special marker for GUI to detect
            # IMPORTANT: Print messages in a specific order and flush immediately
            # The GUI will look for these markers in the output stream
            # Note: We don't print user-friendly messages in GUI mode as the GUI will show a popup
            try:
                print_current("🔔 GUI_USER_INPUT_REQUEST")
                sys.stdout.flush()
                # Small delay to ensure message is processed
                import time
                time.sleep(0.01)
                
                print_current(f"QUERY: {query}")
                sys.stdout.flush()
                time.sleep(0.01)
                
                print_current(f"TIMEOUT: {timeout}")
                sys.stdout.flush()
                time.sleep(0.01)
            except Exception as e:
                # Fallback: if printing fails, try to send directly to queue
                print_current(f"⚠️ Error in GUI mode message sending: {e}")
                sys.stdout.flush()
            
            # Try to get input queue from main module
            input_queue = None
            try:
                main_module = sys.modules.get('__main__', None)
                if main_module and hasattr(main_module, '_agia_gui_input_queue'):
                    input_queue = main_module._agia_gui_input_queue
            except:
                pass
            
            # Read from input queue if available, otherwise fall back to stdin
            try:
                if input_queue:
                    # Use queue with timeout
                    if timeout == -1:
                        # No timeout - wait indefinitely
                        user_input = input_queue.get()
                    else:
                        # Use queue.get with timeout
                        user_input = input_queue.get(timeout=timeout)
                    response = user_input.strip() if isinstance(user_input, str) else str(user_input).strip()
                else:
                    # Fall back to stdin reading (for compatibility)
                    if timeout == -1:
                        user_input = sys.stdin.readline()
                    else:
                        import select
                        if sys.platform != 'win32' and hasattr(select, 'select'):
                            if select.select([sys.stdin], [], [], timeout)[0]:
                                user_input = sys.stdin.readline()
                            else:
                                print_current("⏰ User did not reply within specified time")
                                return {
                                    'status': 'failed',
                                    'query': query,
                                    'user_response': 'no user response',
                                    'timeout': timeout,
                                    'response_time': 'timeout'
                                }
                        else:
                            response_queue = queue.Queue()
                            
                            def read_input():
                                try:
                                    user_input = sys.stdin.readline()
                                    response_queue.put(('success', user_input))
                                except Exception as e:
                                    response_queue.put(('error', str(e)))
                            
                            input_thread = threading.Thread(target=read_input)
                            input_thread.daemon = True
                            input_thread.start()
                            
                            try:
                                status, user_input = response_queue.get(timeout=timeout)
                                if status == 'error':
                                    raise Exception(user_input)
                            except queue.Empty:
                                print_current("⏰ User did not reply within specified time")
                                return {
                                    'status': 'failed',
                                    'query': query,
                                    'user_response': 'no user response',
                                    'timeout': timeout,
                                    'response_time': 'timeout'
                                }
                    response = user_input.strip() if user_input else ""
                
                print_current(f"✅ User reply: {response}")
                return {
                    'status': 'success',
                    'query': query,
                    'user_response': response,
                    'timeout': timeout,
                    'response_time': 'within_timeout' if timeout != -1 else 'no_timeout'
                }
            except queue.Empty:
                # Timeout occurred (for queue.get)
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
        else:
            # Terminal mode: use input() as before
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
                    if max_total_time < 3000:
                        max_total_time = 3000  # 50 minutes maximum execution time
                    print_current(f"⏱️  Detected pip install command, using timeout: {timeout_inactive}s no output timeout, {max_total_time}s maximum execution time")
                    
                    # Ensure pip uses unbuffered output for better visibility
                    # Set environment variables for unbuffered Python output
                    if env is None:
                        env = os.environ.copy()
                    env['PIP_PROGRESS_BAR'] = 'on'
                    env['FORCE_COLOR'] = '1'
                    env['PIP_DISABLE_PIP_VERSION_CHECK'] = '1'  # Reduce noise
                
                # Special handling for Python programs - ensure output is visible
                is_python_program = command.lower().startswith('python ') or command.lower().startswith('python3 ') or 'python' in command.lower()
                
                if is_python_program:
                    print_current(f"🐍 Detected Python program, ensuring unbuffered output")
                    # Add -u flag for unbuffered mode if not already present
                    # This ensures immediate output even for interactive programs
                    if '-u' not in command and '--unbuffered' not in command:
                        # Insert -u flag after python/python3
                        import re
                        # Match python or python3 at the start or after whitespace
                        pattern = r'(\bpython3?\b)'
                        if re.search(pattern, command, re.IGNORECASE):
                            command = re.sub(pattern, r'\1 -u', command, count=1, flags=re.IGNORECASE)
                            print_current(f"🔧 Added -u flag for unbuffered output")
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
    
    def run_claude(self, prompt: str, work_dir: str = None, **kwargs) -> Dict[str, Any]:
        """
        Run claude command using claude_shell.py as a separate process.
        
        Args:
            prompt: Prompt to send to claude
            work_dir: Working directory path for claude_shell.py execution (if None, uses workspace_root)
        
        Returns:
            Dictionary with execution results including status, stdout, stderr, and return_code
        """
        # Ignore additional parameters
        if kwargs:
            print_current(f"⚠️  Ignoring additional parameters: {list(kwargs.keys())}")
        
        # Determine working directory
        if work_dir is None or work_dir == "./" or work_dir == ".":
            # Use workspace_root when work_dir is None, "./", or "."
            work_dir = self.workspace_root
        else:
            # Resolve the path
            work_dir = os.path.expanduser(work_dir)
            # If it's a relative path, resolve it relative to workspace_root
            if not os.path.isabs(work_dir):
                work_dir = os.path.join(self.workspace_root, work_dir)
        
        # Always convert to absolute path before passing to claude_shell.py
        work_dir = os.path.abspath(work_dir)
        
        if not os.path.exists(work_dir):
            return {
                'status': 'failed',
                'error': f'Working directory does not exist: {work_dir}',
                'prompt': prompt,
                'work_dir': work_dir
            }
        if not os.path.isdir(work_dir):
            return {
                'status': 'failed',
                'error': f'Path is not a directory: {work_dir}',
                'prompt': prompt,
                'work_dir': work_dir
            }
        
        # Find claude_shell.py path (should be in src/utils/)
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        claude_shell_path = os.path.join(project_root, 'src', 'utils', 'claude_shell.py')
        
        if not os.path.exists(claude_shell_path):
            return {
                'status': 'failed',
                'error': f'claude_shell.py not found at: {claude_shell_path}',
                'prompt': prompt,
                'work_dir': work_dir
            }
        
        # Build command
        cmd = [sys.executable, claude_shell_path, prompt]
        if work_dir:
            cmd.extend(['-d', work_dir])
        
        try:
            print_current(f"🤖 Running claude with prompt: {prompt[:100]}..." if len(prompt) > 100 else f"🤖 Running claude with prompt: {prompt}")
            print_current(f"📁 Working directory: {work_dir}")
            
            # Run claude_shell.py as subprocess
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding='utf-8',
                errors='replace',
                cwd=work_dir
            )
            
            # Read output with timeout (claude commands can take a while)
            timeout_inactive = 300  # 5 minutes for no output
            max_total_time = 600    # 10 minutes maximum
            
            stdout, stderr, return_code, timed_out = self._read_process_output_with_timeout_and_input(
                process, timeout_inactive, max_total_time
            )
            
            status = 'success'
            if timed_out:
                status = 'failed'
            elif return_code != 0:
                status = 'failed'
            
            result = {
                'status': status,
                'prompt': prompt,
                'stdout': stdout,
                'stderr': stderr,
                'return_code': return_code,
                'work_dir': work_dir,
                'timeout_inactive': timeout_inactive,
                'max_total_time': max_total_time
            }
            
            if timed_out:
                result['timeout_reason'] = 'Process timed out due to inactivity or maximum time limit'
            
            return result
            
        except FileNotFoundError:
            return {
                'status': 'failed',
                'error': f'Python interpreter not found: {sys.executable}',
                'prompt': prompt,
                'work_dir': work_dir
            }
        except Exception as e:
            return {
                'status': 'failed',
                'error': str(e),
                'prompt': prompt,
                'work_dir': work_dir
            } 