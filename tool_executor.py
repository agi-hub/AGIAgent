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

import os
import json
import re
import argparse
import sys
import datetime
import platform
import subprocess
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
import requests
import hashlib
from openai import OpenAI
from tools import Tools
from config_loader import get_api_key, get_api_base, get_model, get_max_tokens, get_streaming, get_language, get_truncation_length, get_history_truncation_length, get_web_content_truncation_length, get_summary_history, get_summary_max_length, get_summary_trigger_length, get_simplified_search_output

# Check if the model is a Claude model
def is_claude_model(model: str) -> bool:
    """Check if the model name is a Claude model"""
    return model.lower().startswith('claude')

# Dynamically import Anthropic
def get_anthropic_client():
    """Dynamically import and return Anthropic client class"""
    try:
        from anthropic import Anthropic
        return Anthropic
    except ImportError:
        print("❌ Anthropic library not installed, please run: pip install anthropic")
        raise ImportError("Anthropic library not installed")

def get_ip_location_info():
    """
    Get IP geolocation information
    
    Returns:
        Dict containing IP and location information
    """
    try:
        # Try to get public IP and location info
        response = requests.get('http://ipapi.co/json/', timeout=5)
        if response.status_code == 200:
            data = response.json()
            return {
                'ip': data.get('ip', 'Unknown'),
                'city': data.get('city', 'Unknown'),
                'region': data.get('region', 'Unknown'),
                'country': data.get('country_name', 'Unknown'),
                'country_code': data.get('country_code', 'Unknown'),
                'timezone': data.get('timezone', 'Unknown')
            }
    except Exception as e:
        print(f"Warning: Could not retrieve IP location info: {e}")
    
    # Fallback: try alternative service
    try:
        response = requests.get('https://ipinfo.io/json', timeout=5)
        if response.status_code == 200:
            data = response.json()
            location = data.get('loc', '').split(',')
            return {
                'ip': data.get('ip', 'Unknown'),
                'city': data.get('city', 'Unknown'),
                'region': data.get('region', 'Unknown'),
                'country': data.get('country', 'Unknown'),
                'country_code': data.get('country', 'Unknown'),
                'timezone': data.get('timezone', 'Unknown')
            }
    except Exception as e:
        print(f"Warning: Could not retrieve IP location info from fallback: {e}")
    
    # Final fallback: return unknown values
    return {
        'ip': 'Unknown',
        'city': 'Unknown',
        'region': 'Unknown',
        'country': 'Unknown',
        'country_code': 'Unknown',
        'timezone': 'Unknown'
    }

class ToolExecutor:
    def __init__(self, api_key: str = None, 
                 model: str = None, 
                 api_base: str = None, 
                 workspace_dir: str = None,
                 debug_mode: bool = False,
                 logs_dir: str = "logs",
                 session_timestamp: str = None,
                 streaming: bool = None,
                 interactive_mode: bool = False):
        """
        Initialize the ToolExecutor
        
        Args:
            api_key: API key for LLM service
            model: Model name to use
            api_base: Base URL for the API service
            workspace_dir: Directory for workspace files
            debug_mode: Whether to enable debug logging
            logs_dir: Directory for log files
            session_timestamp: Timestamp for this session (used for log organization)
            streaming: Whether to use streaming output (None to use config.txt)
            interactive_mode: Whether to enable interactive mode
        """
        # Load API key from config.txt if not provided
        if api_key is None:
            api_key = get_api_key()
            if api_key is None:
                raise ValueError("API key not found. Please provide api_key parameter or set it in config.txt")
        self.api_key = api_key
        
        # Load model from config.txt if not provided
        if model is None:
            model = get_model()
            if model is None:
                raise ValueError("Model not found. Please provide model parameter or set it in config.txt")
        self.model = model
        
        # Load API base from config.txt if not provided
        if api_base is None:
            api_base = get_api_base()
            if api_base is None:
                raise ValueError("API base URL not found. Please provide api_base parameter or set it in config.txt")
        
        # Load streaming configuration from config.txt if not provided
        if streaming is None:
            streaming = get_streaming()
        self.streaming = streaming
        
        # Load language configuration from config.txt
        self.language = get_language()
        
        # Load history summarization configuration from config.txt
        self.summary_history = get_summary_history()
        self.summary_max_length = get_summary_max_length()
        self.summary_trigger_length = get_summary_trigger_length()
        
        # Load simplified search output configuration from config.txt
        self.simplified_search_output = get_simplified_search_output()
        
        self.workspace_dir = workspace_dir or os.getcwd()
        self.debug_mode = debug_mode
        self.logs_dir = logs_dir
        self.session_timestamp = session_timestamp
        self.interactive_mode = interactive_mode
        
        # Check if it's a Claude model, automatically adjust api_base
        self.is_claude = is_claude_model(model)
        
        self.api_base = api_base
        
        # print(f"🤖 LLM Configuration:")  # Commented out to reduce terminal noise
        # print(f"   Model: {self.model}")  # Commented out to reduce terminal noise
        # print(f"   API Base: {self.api_base}")  # Commented out to reduce terminal noise
        # print(f"   API Key: {self.api_key[:20]}...{self.api_key[-10:]}")  # Commented out to reduce terminal noise
        # print(f"   Workspace: {self.workspace_dir}")  # Commented out to reduce terminal noise
        # print(f"   Language: {'中文' if self.language == 'zh' else 'English'} ({self.language})")  # Commented out to reduce terminal noise
        # print(f"   Streaming: {'✅ Enabled' if self.streaming else '❌ Disabled (Batch mode)'}")  # Commented out to reduce terminal noise
        # print(f"   Cache Optimization: ✅ Enabled (All rounds use combined prompts for maximum cache hits)")  # Commented out to reduce terminal noise
        # print(f"   History Summarization: {'✅ Enabled' if self.summary_history else '❌ Disabled'} (Trigger: {self.summary_trigger_length} chars, Max: {self.summary_max_length} chars)")  # Commented out to reduce terminal noise
        # print(f"   Simplified Search Output: {'✅ Enabled' if self.simplified_search_output else '❌ Disabled'} (Affects codebase_search and web_search terminal display)")  # Commented out to reduce terminal noise
        # if debug_mode:
        #     print(f"   Debug Mode: Enabled (Log directory: {logs_dir})")  # Commented out to reduce terminal noise
        
        # Set up LLM client
        self._setup_llm_client()
        
        # Initialize tools with LLM configuration for web search filtering
        from tools import Tools
        
        # Get the parent directory of workspace (typically the output directory)
        out_dir = os.path.dirname(self.workspace_dir) if self.workspace_dir else os.getcwd()
        
        self.tools = Tools(
            workspace_root=self.workspace_dir,
            llm_api_key=self.api_key,
            llm_model=self.model,
            llm_api_base=self.api_base,
            enable_llm_filtering=False,  # Disable LLM filtering by default for faster responses
            out_dir=out_dir
        )
        
        # Initialize summary generator for conversation history summarization
        if self.summary_history:
            try:
                from multi_round_executor.summary_generator import SummaryGenerator
                self.conversation_summarizer = SummaryGenerator(self, detailed_summary=True)
                # print(f"✅ Conversation history summarizer initialized")
            except ImportError as e:
                print(f"⚠️ Failed to import SummaryGenerator: {e}, history summarization disabled")
                self.conversation_summarizer = None
                self.summary_history = False  # Disable summarization if import fails
        else:
            self.conversation_summarizer = None
        
        # Map of tool names to their implementation methods
        self.tool_map = {
            "codebase_search": self.tools.codebase_search,
            "read_file": self.tools.read_file,
            "run_terminal_cmd": self.tools.run_terminal_cmd,
            "list_dir": self.tools.list_dir,
            "grep_search": self.tools.grep_search,
            "edit_file": self.tools.edit_file,
            "file_search": self.tools.file_search,
            "delete_file": self.tools.delete_file,
            "reapply": self.tools.reapply,
            "web_search": self.tools.web_search,
            "diff_history": self.tools.diff_history,
            "tool_help": self.tools.tool_help
        }
        
        # Add plugin tools if available
        if hasattr(self.tools, 'kb_search'):
            self.tool_map.update({
                "kb_search": self.tools.kb_search,
                "kb_content": self.tools.kb_content,
                "kb_body": self.tools.kb_body
            })
            # print("🔌 Plugin tools registered: kb_search, kb_content, kb_body")
        
        # Log related settings
        # Fix logs directory path construction - ensure it's relative to parent of workspace, not workspace itself
        if workspace_dir and session_timestamp:
            # Get the parent directory of workspace (typically the output directory)
            parent_dir = os.path.dirname(workspace_dir) if workspace_dir else os.getcwd()
            self.logs_dir = os.path.join(parent_dir, logs_dir, session_timestamp)
        elif workspace_dir:
            parent_dir = os.path.dirname(workspace_dir) if workspace_dir else os.getcwd()
            self.logs_dir = os.path.join(parent_dir, logs_dir)
        else:
            self.logs_dir = os.path.join(os.getcwd(), logs_dir, session_timestamp) if session_timestamp else os.path.join(os.getcwd(), logs_dir)
        
        self.llm_logs_dir = self.logs_dir  # LLM call logs directory
        self.llm_call_counter = 0  # LLM call counter
        
        # Ensure log directory exists
        os.makedirs(self.llm_logs_dir, exist_ok=True)
        
        # If DEBUG mode is enabled, initialize CSV logger
        if self.debug_mode:
            # print(f"🐛 DEBUG mode enabled, LLM call records will be saved to: {self.llm_logs_dir}/llmcall.csv")
            pass
    
    def _setup_llm_client(self):
        """
        Set up the LLM client based on the model type.
        """
        if self.is_claude:
            # print(f"🧠 Detected Claude model, using Anthropic protocol")
            # Adjust api_base for Claude models
            if not self.api_base.endswith('/anthropic'):
                if self.api_base.endswith('/v1'):
                    self.api_base = self.api_base[:-3] + '/anthropic'
                else:
                    self.api_base = self.api_base.rstrip('/') + '/anthropic'
            
            # print(f" Claude API Base: {self.api_base}")
            
            # Initialize Anthropic client
            Anthropic = get_anthropic_client()
            self.client = Anthropic(
                api_key=self.api_key,
                base_url=self.api_base
            )
        else:
            # print(f"🤖 Using OpenAI protocol")
            # Initialize OpenAI client
            self.client = OpenAI(
                api_key=self.api_key,
                base_url=self.api_base
            )
    
    def _get_max_tokens_for_model(self, model: str) -> int:
        """
        Get the appropriate max_tokens for the given model.
        First tries to read from config.txt, then falls back to model defaults.
        
        Args:
            model: Model name
            
        Returns:
            Max tokens for the model
        """
        # First try to get max_tokens from configuration file
        config_max_tokens = get_max_tokens()
        if config_max_tokens is not None:
            # print(f"🔧 Using max_tokens from config: {config_max_tokens}")
            return config_max_tokens
        
        # Fallback to model-specific defaults
        model_limits = {
            # Claude models
            'claude-3-haiku-20240307': 4096,
            'claude-3-5-haiku-20241022': 8192,
            'claude-3-sonnet-20240229': 4096,
            'claude-3-5-sonnet-20240620': 8192,
            'claude-3-5-sonnet-20241022': 8192,
            'claude-3-opus-20240229': 4096,
            'claude-3-7-sonnet-latest': 8192,
            # OpenAI models (generally have higher limits)
            'gpt-4': 8192,
            'gpt-4o': 16384,
            'gpt-4o-mini': 16384,
            'gpt-3.5-turbo': 4096,
            # Qwen models (SiliconFlow)
            'Qwen/Qwen2.5-7B-Instruct': 8192,
            'Qwen/Qwen3-32B': 8192,
            'Qwen/Qwen3-30B-A3B': 8192,
        }
        
        # Get model-specific limit or default to 8192 for unknown models
        max_tokens = model_limits.get(model, 8192)
        
        # Extra safety check for Claude models
        if 'claude' in model.lower() and 'haiku' in model.lower():
            max_tokens = min(max_tokens, 4096)
        elif 'claude' in model.lower():
            max_tokens = min(max_tokens, 8192)
        
        print(f"🔧 Using default max_tokens for model {model}: {max_tokens}")
        return max_tokens
    
    def _check_command_available(self, command: str) -> bool:
        """
        Check if a command is available in the system.
        
        Args:
            command: Command name to check
            
        Returns:
            True if command is available, False otherwise
        """
        try:
            import subprocess
            # Use 'where' on Windows, 'which' on Unix-like systems
            check_cmd = "where" if platform.system().lower() == "windows" else "which"
            result = subprocess.run([check_cmd, command], 
                                  capture_output=True, 
                                  text=True, 
                                  timeout=5)
            return result.returncode == 0
        except Exception:
            return False
    
    def load_system_prompt(self, prompt_file: str = "prompts.txt") -> str:
        """
        Load the system prompt from modular files or single file.
        
        Args:
            prompt_file: Path to the prompt file (legacy support)
            
        Returns:
            The system prompt text
        """
        try:
            # Try to load from modular files first
            # Core files that must exist for modular loading
            core_files = ["prompts/system_prompt.txt", "prompts/rules_prompt.txt", "prompts/tool_prompt.txt"]
            # Optional files that will be included if they exist
            optional_files = ["prompts/user_rules.txt", "prompts/plugin_tool_prompts.txt"]
            
            system_prompt = ""
            
            # Check if all core modular files exist
            all_core_files_exist = all(os.path.exists(f) for f in core_files)
            
            if all_core_files_exist:
                # Load and combine modular prompt files
                loaded_files = []
                
                # Load core files first
                for file_path in core_files:
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read().strip()
                            if content:
                                system_prompt += content + "\n\n"
                                loaded_files.append(file_path)
                    except Exception as e:
                        print(f"Warning: Could not load core file {file_path}: {e}")
                        # Fall back to single file if any core file fails
                        all_core_files_exist = False
                        break
                
                # Load optional files if core files loaded successfully
                if all_core_files_exist:
                    for file_path in optional_files:
                        if os.path.exists(file_path):
                            try:
                                with open(file_path, 'r', encoding='utf-8') as f:
                                    content = f.read().strip()
                                    if content:
                                        system_prompt += content + "\n\n"
                                        loaded_files.append(file_path)
                            except Exception as e:
                                print(f"Warning: Could not load optional file {file_path}: {e}")
                                # Continue loading other files, don't fail completely
                    
                    print("✅ Loaded modular system prompts from: " + ", ".join(loaded_files))
                else:
                    # Fall back to single file
                    with open(prompt_file, 'r', encoding='utf-8') as f:
                        system_prompt = f.read()
                    print(f"⚠️ Fallback to single prompt file: {prompt_file}")
            else:
                # Fall back to single file approach
                with open(prompt_file, 'r', encoding='utf-8') as f:
                    system_prompt = f.read()
                print(f"📄 Loaded single prompt file: {prompt_file}")
            
            # Add operating system information
            try:
                system_name = platform.system()
                system_release = platform.release()
                
                # Get Python version
                python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
                
                # Check if pip is available
                pip_available = "Available" if self._check_command_available("pip") else "Not Available"
                
                os_instruction = f"""

**Operating System Information**:
- Operating System: {system_name} {system_release}
- Python Version: {python_version}
- pip: {pip_available}
"""
                
                # Add system-specific information
                if system_name.lower() == "linux":
                    # For Linux: check gdb and shell type
                    gdb_available = "Available" if self._check_command_available("gdb") else "Not Available"
                    shell_type = os.environ.get('SHELL', 'Unknown').split('/')[-1] if os.environ.get('SHELL') else 'Unknown'
                    
                    os_instruction += f"""- gdb: {gdb_available}
- Shell Type: {shell_type}
- Please use Linux-compatible commands and forward slashes for paths
"""
                
                elif system_name.lower() == "windows":
                    # For Windows: check PowerShell
                    powershell_available = "Available" if self._check_command_available("powershell") else "Not Available"
                    
                    os_instruction += f"""- PowerShell: {powershell_available}
- Please use Windows-compatible commands and backslashes for paths
"""
                
                elif system_name.lower() == "darwin":  # macOS
                    # For macOS: check shell type
                    shell_type = os.environ.get('SHELL', 'Unknown').split('/')[-1] if os.environ.get('SHELL') else 'Unknown'
                    
                    os_instruction += f"""- Shell Type: {shell_type}
- Please use macOS-compatible commands and forward slashes for paths
"""
                
                os_instruction += "\n"
                
            except Exception as e:
                print(f"Warning: Could not retrieve OS information: {e}")
                os_instruction = ""
            
            # Add language instruction based on configuration
            if self.language == 'zh':
                language_instruction = """

**重要的语言设置指令**:
- 系统语言配置为中文，请尽量使用中文进行回复和生成报告
- 当生成分析报告、总结文档或其他输出文件时，请使用中文
- 代码注释和说明文档也请尽量使用中文
- 只有在涉及英文专业术语或代码本身时才使用英语
- 报告标题、章节名称等都应使用中文

"""
            else:
                language_instruction = """

**Language Configuration**:
- System language is set to English
- Please respond and generate reports in English
- Code comments and documentation should be in English

"""
            
            # Add current date information
            current_date = datetime.datetime.now()
            date_instruction = f"""

**Current Date Information**:
- Current Date: {current_date.strftime('%Y-%m-%d')}
- Current Time: {current_date.strftime('%Y-%m-%d %H:%M:%S')}

"""
            
            # Add IP geolocation information
            location_info = get_ip_location_info()
            location_instruction = f"""

**Location Information**:
- City: {location_info['city']}
- Country: {location_info['country']}

"""
            
            return system_prompt + os_instruction + language_instruction + date_instruction + location_instruction
        except Exception as e:
            print(f"Error loading system prompt: {e}")
            return "You are a helpful AI assistant that can use tools to accomplish tasks."
    
    def execute_subtask(self, user_prompt: str, system_prompt_file: str = "prompts.txt", task_history: List[Dict[str, Any]] = None) -> str:
        """
        Execute a single subtask using an LLM with tool capabilities.
        
        Args:
            user_prompt: The prompt for the subtask
            system_prompt_file: Path to the system prompt file
            task_history: Previous task execution history for multi-round continuity
            
        Returns:
            Text result from executing the subtask
        """
        try:
            # Load system prompt
            system_prompt = self.load_system_prompt(system_prompt_file)
            
            # Add workspace directory information to the system prompt
            workspace_instruction = f"""

**Important Workspace Information**:
- Workspace Directory: {self.workspace_dir}
- Please save all created code files and project files in this directory
- When creating or editing files, please use filenames directly, do not add "workspace/" prefix to paths
- The system has automatically set the correct working directory, you only need to use relative filenames

"""
            
            # Get existing code context from workspace
            workspace_context = self._get_workspace_context()
            
            # Combine prompts with workspace context
            enhanced_system_prompt = system_prompt + workspace_instruction + workspace_context
            
            # Prepare messages for the LLM
            # For cache optimization, combine history with current prompt instead of using chat history
            combined_user_prompt = self._build_combined_user_prompt(user_prompt, task_history)
            
            messages = [
                {"role": "system", "content": enhanced_system_prompt},
                {"role": "user", "content": combined_user_prompt}
            ]
            
            # Generate cache key for potential cache hit identification
            cache_key = self._generate_cache_key(enhanced_system_prompt, combined_user_prompt)
            print(f"🔑 Request cache key: {cache_key[:16]}... (for cache hit tracking)")
            
            # Mark this as cache optimized since we're using the combined prompt format for all rounds
            is_cache_optimized_format = True
            
            # Start tool calling loop, support multiple rounds of tool calls
            max_tool_rounds = 5  # Maximum tool calling rounds to prevent infinite loops
            current_round = 0
            final_response = ""
            
            # Initialize current round tool tracking for edit_file hallucination detection
            current_round_tools = []
            
            while current_round < max_tool_rounds:
                current_round += 1
                
                # Reset current round tool tracking for new round
                current_round_tools = []
                
                # Call the LLM
                print(f"🤖 Calling LLM (Round {current_round}): {user_prompt if current_round == 1 else 'Processing tool call results'}")
                
                # Save debug log for this round's input (before LLM call)
                if self.debug_mode and current_round == 1:
                    try:
                        initial_tool_calls_info = {
                            "is_initial_call": True,
                            "round_type": "initial_user_prompt",
                            "user_prompt": user_prompt
                        }
                        
                        # Don't call _save_llm_call_debug_log here yet, as we don't have the response
                        # This will be handled after we get the LLM response
                    except Exception as e:
                        print(f"⚠️ Initial debug preparation failed: {e}")
                
                if self.is_claude:
                    # Use Anthropic Claude API
                    # Claude needs to separate system and other messages
                    system_message = ""
                    claude_messages = []
                    
                    for msg in messages:
                        if msg["role"] == "system":
                            system_message = msg["content"]
                        else:
                            claude_messages.append(msg)
                    
                    if self.streaming:
                        print("🔄 Starting streaming generation...")
                    else:
                        print("🔄 Starting batch generation...")
                    
                    # Add retry mechanism
                    max_retries = 3
                    for retry_count in range(max_retries):
                        try:
                            if self.streaming:
                                # Streaming mode with hallucination detection
                                with self.client.messages.stream(
                                    model=self.model,
                                    max_tokens=self._get_max_tokens_for_model(self.model),
                                    system=system_message,
                                    messages=claude_messages,
                                    temperature=0.7
                                ) as stream:
                                    content = ""
                                    
                                    # Get hallucination detection configuration
                                    hallucination_config = self._get_hallucination_detection_config(current_round_tools)
                                    detection_buffer = ""
                                    generation_stopped = False
                                    
                                    # Calculate buffer size based on longest trigger
                                    if hallucination_config.get("enabled", True):
                                        max_trigger_length = max(len(trigger) for trigger in hallucination_config.get("triggers", [""]))
                                        buffer_size = max_trigger_length * hallucination_config.get("buffer_size_multiplier", 2)
                                    else:
                                        buffer_size = 0
                                    
                                    for text in stream.text_stream:
                                        # Add current text to detection buffer BEFORE printing
                                        if hallucination_config.get("enabled", True):
                                            detection_buffer += text
                                            
                                            # Keep buffer size manageable
                                            if len(detection_buffer) > buffer_size:
                                                # Remove excess from beginning
                                                detection_buffer = detection_buffer[-buffer_size:]
                                            
                                            # Update current round tools tracking FIRST (before hallucination detection)
                                            self._update_current_round_tools(detection_buffer, current_round_tools)
                                            
                                            # Get updated hallucination detection configuration with current tools
                                            updated_config = self._get_hallucination_detection_config(current_round_tools)
                                            
                                            # Then check for hallucination triggers BEFORE printing
                                            detected, trigger_found = self._detect_hallucination_in_stream(detection_buffer, updated_config)
                                            if detected:
                                                generation_stopped = True
                                                
                                                # Remove the hallucination content from already accumulated content
                                                original_content = content
                                                content = self._remove_hallucination_from_content(content, trigger_found, updated_config)
                                                
                                                # Print the completion part to terminal if content was extended
                                                if len(content) > len(original_content):
                                                    completion_part = content[len(original_content):]
                                                    print(completion_part, end="", flush=True)
                                                
                                                break
                                        
                                        # Only print and accumulate content if no hallucination detected
                                        print(text, end="", flush=True)
                                        content += text
                                    
                                    if generation_stopped:
                                        pass  # Generation was stopped due to hallucination detection
                                    else:
                                        print("\n✅ Streaming completed")
                            else:
                                # Batch mode (original implementation)
                                response = self.client.messages.create(
                                    model=self.model,
                                    max_tokens=self._get_max_tokens_for_model(self.model),
                                    system=system_message,
                                    messages=claude_messages,
                                    temperature=0.7
                                )
                                
                                # Get complete response content
                                content = response.content[0].text
                                print("\n🤖 LLM response:")
                                print("✅ Generation completed")
                            
                            # If content generated successfully, break out of retry loop
                            break
                            
                        except Exception as api_error:
                            print(f"\n⚠️ Claude API call failed (attempt {retry_count + 1}/{max_retries}): {api_error}")
                            
                            # If it's the last retry, raise exception
                            if retry_count == max_retries - 1:
                                print(f"❌ Claude API call finally failed, please check:")
                                print(f"   1. API key validity")
                                print(f"   2. Network connection")
                                print(f"   3. Claude service availability")
                                print(f"   4. Message content compliance with API requirements")
                                raise api_error
                            else:
                                print(f"🔄 Waiting {2 ** retry_count} seconds before retry...")
                                import time
                                time.sleep(2 ** retry_count)  # Exponential backoff
                                continue
                
                else:
                    # Use OpenAI API
                    if self.streaming:
                        print("🔄 Starting streaming generation...")
                        # Streaming mode with hallucination detection
                        response = self.client.chat.completions.create(
                            model=self.model,
                            messages=messages,
                            max_tokens=self._get_max_tokens_for_model(self.model),
                            temperature=0.7,
                            top_p=0.8,
                            stream=True
                        )
                        
                        content = ""
                        
                        # Get hallucination detection configuration
                        hallucination_config = self._get_hallucination_detection_config(current_round_tools)
                        detection_buffer = ""
                        generation_stopped = False
                        
                        # Calculate buffer size based on longest trigger
                        if hallucination_config.get("enabled", True):
                            max_trigger_length = max(len(trigger) for trigger in hallucination_config.get("triggers", [""]))
                            buffer_size = max_trigger_length * hallucination_config.get("buffer_size_multiplier", 2)
                        else:
                            buffer_size = 0
                        
                        for chunk in response:
                            if chunk.choices and len(chunk.choices) > 0 and chunk.choices[0].delta.content is not None:
                                chunk_content = chunk.choices[0].delta.content
                                
                                # Add current chunk to detection buffer BEFORE printing
                                if hallucination_config.get("enabled", True):
                                    detection_buffer += chunk_content
                                    
                                    # Keep buffer size manageable
                                    if len(detection_buffer) > buffer_size:
                                        # Remove excess from beginning
                                        detection_buffer = detection_buffer[-buffer_size:]
                                    
                                    # Update current round tools tracking FIRST (before hallucination detection)
                                    self._update_current_round_tools(detection_buffer, current_round_tools)
                                    
                                    # Get updated hallucination detection configuration with current tools
                                    updated_config = self._get_hallucination_detection_config(current_round_tools)
                                    
                                    # Then check for hallucination triggers BEFORE printing
                                    detected, trigger_found = self._detect_hallucination_in_stream(detection_buffer, updated_config)
                                    if detected:
                                        generation_stopped = True
                                        
                                        # Remove the hallucination content from already accumulated content
                                        original_content = content
                                        content = self._remove_hallucination_from_content(content, trigger_found, updated_config)
                                        
                                        # Print the completion part to terminal if content was extended
                                        if len(content) > len(original_content):
                                            completion_part = content[len(original_content):]
                                            print(completion_part, end="", flush=True)
                                        
                                        break
                                
                                # Only print and accumulate content if no hallucination detected
                                print(chunk_content, end="", flush=True)
                                content += chunk_content
                        
                        if generation_stopped:
                            pass  # Generation was stopped due to hallucination detection
                        else:
                            print("\n✅ Streaming completed")
                    else:
                        # Batch mode (original implementation)
                        print("🔄 Starting batch generation...")
                        response = self.client.chat.completions.create(
                            model=self.model,
                            messages=messages,
                            max_tokens=self._get_max_tokens_for_model(self.model),
                            temperature=0.7,
                            top_p=0.8
                        )
                        
                        # Get complete response content
                        if response.choices and len(response.choices) > 0:
                            content = response.choices[0].message.content
                        else:
                            content = ""
                            print("⚠️ Warning: OpenAI API returned empty choices list")
                        print("\n🤖 LLM response:")
                        print("✅ Generation completed")
                
                print(f"\n📝 Response content length: {len(content)} characters")
                
                # Add LLM response to conversation history
                messages.append({"role": "assistant", "content": content})
                
                # 为当前轮次保存调试日志（在解析工具调用之前）
                if self.debug_mode:
                    try:
                        current_round_info = {
                            "round_type": "llm_response",
                            "response_length": len(content),
                            "round_summary": f"Round {current_round} LLM response received"
                        }
                        
                        # 如果是第一轮，添加初始调用信息
                        if current_round == 1:
                            current_round_info.update({
                                "is_initial_call": True,
                                "user_prompt": user_prompt
                            })
                        
                        # 暂时保存，稍后根据是否有工具调用来决定如何完善这个信息
                        temp_tool_calls_info = current_round_info
                    except Exception as e:
                        print(f"⚠️ Response logging preparation failed: {e}")
                        temp_tool_calls_info = {}
                
                # Parse the content for tool calls first - tool calls have priority
                tool_calls = self.parse_tool_calls(content)
                
                # Check for TASK_COMPLETED flag and detect conflicts
                has_task_completed = "TASK_COMPLETED:" in content
                has_tool_calls = len(tool_calls) > 0
                
                # CONFLICT DETECTION: Both tool calls and TASK_COMPLETED present
                conflict_detected = has_tool_calls and has_task_completed
                if conflict_detected:
                    print(f"⚠️ CONFLICT DETECTED: Both tool calls and TASK_COMPLETED flag found!")
                    print(f"🔧 Prioritizing tool execution, TASK_COMPLETED signal will be ignored.")
                    # We'll add a warning message in the next round
                
                # If TASK_COMPLETED but no tool calls, complete the task
                if has_task_completed and not has_tool_calls:
                    print(f"🎉 TASK_COMPLETED flag detected in content, task completed!")
                    # Extract the completion message
                    task_completed_match = re.search(r'TASK_COMPLETED:\s*(.+)', content)
                    if task_completed_match:
                        completion_message = task_completed_match.group(1).strip()
                        print(f"🎉 Task completion flag detected: {completion_message}")
                    
                    # Save final debug log
                    if self.debug_mode:
                        # 为任务完成的情况保存调试日志
                        try:
                            completion_round_info = temp_tool_calls_info.copy() if 'temp_tool_calls_info' in locals() else {}
                            completion_round_info.update({
                                "has_tool_calls": False,
                                "task_completed": True,
                                "completion_detected": True,
                                "round_result": "task_completed_flag"
                            })
                            
                            self._save_llm_call_debug_log(messages, f"Round {current_round}: Task completed with TASK_COMPLETED flag", current_round, completion_round_info)
                        except Exception as log_error:
                            print(f"❌ Completion debug log save failed: {log_error}")
                            # 降级处理：使用基本信息保存
                            self._save_llm_call_debug_log(messages, f"Round {current_round}: Task completed with TASK_COMPLETED flag", current_round)
                    
                    return content
                
                if tool_calls:
                    print(f"🔧 Found {len(tool_calls)} tool calls, starting execution...")
                    
                    # Debug: print tool call details
                    if self.debug_mode:
                        for i, call in enumerate(tool_calls, 1):
                            print(f"   Parameters: {call['arguments']}")
                    
                    # Execute all tool calls and collect results
                    all_tool_results = []
                    successful_executions = 0
                    
                    for i, tool_call in enumerate(tool_calls, 1):
                        # Print tool execution start tag BEFORE any other output
                        print(f"<tool_execute tool_name=\"{tool_call['name']}\" tool_number=\"{i}\">")
                        
                        print(f"   - Executing tool {i}: {tool_call['name']}")
                        
                        try:
                            tool_result = self.execute_tool(tool_call)
                            all_tool_results.append({
                                'tool_name': tool_call['name'],
                                'tool_params': tool_call['arguments'],  # 恢复工具参数记录
                                'tool_result': tool_result
                            })
                            successful_executions += 1
                            
                            # Real-time print of each tool's execution result
                            # print(f"🛠️ Tool {i} execution result ({tool_call['name']}):")

                            
                            if isinstance(tool_result, dict):
                                # Use simplified formatting for search tools if enabled in config
                                if (self.simplified_search_output and 
                                    tool_call['name'] in ['codebase_search', 'web_search']):
                                    formatted_result = self._format_search_result_for_terminal(tool_result, tool_call['name'])
                                else:
                                    formatted_result = self._format_dict_as_text(tool_result)
                                print(formatted_result)
                            else:
                                print(str(tool_result))
                            
                            # print(f"   ✅ Tool execution completed: {tool_call['name']}")
                            # print()
                            
                        except Exception as e:
                            error_msg = f"Tool {tool_call['name']} execution failed: {str(e)}"
                            print(f"❌ {error_msg}")
                            all_tool_results.append({
                                'tool_name': tool_call['name'],
                                'tool_params': tool_call['arguments'],
                                'tool_result': f"Error: {error_msg}"
                            })
                        
                        # Print tool execution end tag - moved to after all output is complete
                        print(f"</tool_execute>")
                    
                    # Validate tool execution results
                    if not all_tool_results:
                        print("❌ Warning: No tool execution results, there may be tool call parsing or execution issues")
                    else:
                        # print(f"✅ Successfully executed {successful_executions}/{len(tool_calls)} tools")
                        pass
                    
                    # Perform incremental codebase update
                    try:
                        self.tools.perform_incremental_update()
                    except Exception as e:
                        print(f"⚠️ Codebase incremental update failed: {e}")
                    
                    # Interactive mode: Ask user confirmation after tool execution
                    continue_execution, user_input = self.ask_user_confirmation("Tool execution completed. Continue to next round?")
                    if not continue_execution:
                        print("❌ User chose to stop execution after tool execution")
                        return "USER_INTERRUPTED: User chose to stop execution after tool execution"
                    
                    # For cache optimization: rebuild messages with combined context instead of traditional chat history
                    try:
                        tool_results_message = self._format_tool_results_for_llm(all_tool_results)
                        
                        # Rebuild the entire conversation history as a single user prompt for cache optimization
                        conversation_history = self._build_conversation_history_for_cache(messages, content, tool_results_message)
                        
                        # Add conflict warning if both tool calls and TASK_COMPLETED were detected
                        if conflict_detected:
                            conflict_warning = self._generate_conflict_warning()
                            conversation_history += "\n\n" + conflict_warning
                            print(f"⚠️ Added conflict warning to next round conversation")
                        
                        # If user provided custom input, add it to the conversation history
                        if user_input:
                            user_guidance_msg = f"\n\nThe information from user in interactive mode (important): {user_input}"
                            conversation_history += user_guidance_msg
                            # print(f"📝 User guidance added to conversation context")
                        
                        # Reset messages to cache-optimized format: system + combined_user_prompt
                        messages = [
                            {"role": "system", "content": enhanced_system_prompt},
                            {"role": "user", "content": conversation_history}
                        ]
                        
                        # print(f"📤 Rebuilt conversation with cache optimization for round {current_round + 1}")
                        
                        # Force save debug log - ensure tool results are recorded
                        if self.debug_mode:
                            try:
                                # 构建完整的工具调用信息用于日志记录
                                tool_calls_info = {
                                    "parsed_tool_calls": tool_calls,
                                    "tool_results": all_tool_results,
                                    "formatted_tool_results": tool_results_message,
                                    "conversation_history": conversation_history,
                                    "successful_executions": successful_executions,
                                    "total_tool_calls": len(tool_calls),
                                    "conflict_detected": conflict_detected,
                                    "user_input": user_input if user_input else None
                                }
                                
                                self._save_llm_call_debug_log(messages, f"Round {current_round}: Cache-optimized rebuild after tool execution", current_round, tool_calls_info)
                            except Exception as log_error:
                                print(f"❌ Debug log save failed: {log_error}")
                        
                    except Exception as tool_result_error:
                        error_msg = f"Tool result processing failed: {str(tool_result_error)}"
                        print(f"❌ {error_msg}")
                        
                        # Even if error occurs, save log to record the problem
                        if self.debug_mode:
                            try:
                                messages.append({"role": "user", "content": f"Tool execution error: {error_msg}"})
                                
                                # 即使出错也要记录工具调用信息
                                error_tool_calls_info = {
                                    "parsed_tool_calls": tool_calls,
                                    "tool_results": all_tool_results,
                                    "error": error_msg,
                                    "processing_failed": True
                                }
                                
                                self._save_llm_call_debug_log(messages, f"Round {current_round}: Tool execution error - {error_msg}", current_round, error_tool_calls_info)
                            except Exception as error_log_error:
                                print(f"❌ Error log save also failed: {error_log_error}")
                        
                        # Try to continue even with error
                        print("⚠️ Attempting to continue to next round...")
                        
                        # If user provided custom input but cache optimization failed, add it directly
                        if user_input:
                            user_guidance_msg = f"The information from user in interactive mode (important): {user_input}"
                            messages.append({"role": "user", "content": user_guidance_msg})
                            # print(f"📝 User guidance added to conversation (fallback mode)")
                    
                    # Continue to next round, let LLM see tool results
                    continue
                else:
                    # No tool calls found, check if this might be an error or intended completion
                    should_have_tools = self._should_expect_tool_calls(content, current_round, user_prompt)
                    
                    if should_have_tools and current_round < max_tool_rounds:
                        # Generate warning message for the LLM about missing tool calls
                        warning_message = self._generate_tool_call_warning(content)
                        print(f"⚠️ Expected tool calls but none found, sending warning to LLM (Round {current_round})")
                        
                        # Add warning message to conversation history
                        messages.append({"role": "user", "content": warning_message})
                        
                        # 为工具调用警告情况保存调试日志
                        if self.debug_mode:
                            try:
                                warning_round_info = temp_tool_calls_info.copy() if 'temp_tool_calls_info' in locals() else {}
                                # 使用历史截断长度限制警告消息的长度
                                warning_truncation_length = get_history_truncation_length()
                                warning_round_info.update({
                                    "has_tool_calls": False,
                                    "should_have_tools": True,
                                    "warning_sent": True,
                                    "warning_message": warning_message[:warning_truncation_length] + "..." if len(warning_message) > warning_truncation_length else warning_message,
                                    "round_result": "tool_call_warning_sent"
                                })
                                
                                self._save_llm_call_debug_log(messages, f"Round {current_round}: Tool call warning sent", current_round, warning_round_info)
                            except Exception as log_error:
                                print(f"❌ Warning debug log save failed: {log_error}")
                        
                        # Continue to next round with the warning
                        continue
                    else:
                        # No tool calls expected or max rounds reached, save final response and exit loop
                        final_response = content
                        if should_have_tools and current_round >= max_tool_rounds:
                            print(f"⚠️ Expected tool calls but reached max rounds ({max_tool_rounds}), completing task")
                        else:
                            print("📝 No tool calls expected, task completed")
                        
                        # Save final debug log
                        if self.debug_mode:
                            # 为没有工具调用的情况保存调试日志
                            try:
                                final_round_info = temp_tool_calls_info.copy() if 'temp_tool_calls_info' in locals() else {}
                                final_round_info.update({
                                    "has_tool_calls": False,
                                    "task_completed": has_task_completed,
                                    "should_have_tools": should_have_tools,
                                    "round_result": "final_response_no_tools"
                                })
                                
                                self._save_llm_call_debug_log(messages, f"Round {current_round}: Final response, no tool calls", current_round, final_round_info)
                            except Exception as log_error:
                                print(f"❌ Final debug log save failed: {log_error}")
                                # 降级处理：使用基本信息保存
                                self._save_llm_call_debug_log(messages, f"Round {current_round}: Final response, no tool calls", current_round)
                        break
            
            # If maximum rounds limit reached
            if current_round >= max_tool_rounds:
                print(f"⚠️ Reached maximum tool call rounds limit ({max_tool_rounds}), ending task")
                final_response = content
            
            return final_response
            
        except Exception as e:
            error_msg = f"❌ Error executing subtask: {str(e)}"
            print(error_msg)
            return error_msg
    
    def parse_tool_calls(self, content: str) -> List[Dict[str, Any]]:
        """
        Parse multiple tool calls from the model's response.
        
        Args:
            content: The model's response text
            
        Returns:
            List of dictionaries with tool name and parameters
        """

        
        # Debug mode, save raw content for analysis
        if self.debug_mode:
            # Check for common tool call format markers
            has_function_calls = '<function_calls>' in content
            has_invoke = '<invoke' in content
            has_function_call = '<function_call>' in content
            has_json_block = '```json' in content
        
        all_tool_calls = []
        
        # First try to parse individual <function_call> tags (single format)
        function_call_pattern = r'<function_call>\s*\{(.*?)\}\s*</function_call>'
        function_call_matches = re.findall(function_call_pattern, content, re.DOTALL)
        if function_call_matches:
            for match in function_call_matches:
                try:
                    # Parse the JSON content inside function_call tags
                    json_str = '{' + match + '}'
                    tool_data = json.loads(json_str)
                    
                    if isinstance(tool_data, dict):
                        # Handle different JSON structures
                        if 'name' in tool_data and 'parameters' in tool_data:
                            all_tool_calls.append({
                                "name": tool_data["name"],
                                "arguments": tool_data["parameters"]
                            })
                        elif 'name' in tool_data and 'content' in tool_data:
                            all_tool_calls.append({
                                "name": tool_data["name"],
                                "arguments": tool_data["content"]
                            })
                except json.JSONDecodeError as e:
                    continue
            
            # If we found function_call format tool calls, return them
            if all_tool_calls:
                return all_tool_calls
        
        # Try to parse XML format with function_calls wrapper
        function_calls_matches = re.findall(r'<function_calls>(.*?)</function_calls>', content, re.DOTALL)
        if function_calls_matches:
            for i, function_calls_text in enumerate(function_calls_matches, 1):
                # Parse the function calls in this block
                function_calls = self.parse_function_calls(function_calls_text)
                if function_calls:
                    all_tool_calls.extend(function_calls)
            
            # If we found function_calls wrapper tool calls, return directly, avoid duplicate parsing
            if all_tool_calls:
                return all_tool_calls
        
        # Only try to parse individual invoke tags if no function_calls wrapper was found
        invoke_pattern = r'<invoke name="([^"]+)">(.*?)</invoke>'
        invoke_matches = re.findall(invoke_pattern, content, re.DOTALL)
        if invoke_matches:
            for tool_name, args_text in invoke_matches:
                args = self.parse_arguments(args_text)
                all_tool_calls.append({"name": tool_name, "arguments": args})
        
        # If we found tool calls through XML parsing, return them
        if all_tool_calls:
            return all_tool_calls
        
        # Fallback: try to parse Python function call format
        python_tool_calls = self.parse_python_function_calls(content)
        if python_tool_calls:
            all_tool_calls.extend(python_tool_calls)
            return all_tool_calls
        
        # Fallback: try to parse JSON format with nested content structure (like in the logs)
        json_pattern = r'```json\s*(.*?)\s*```'
        json_match = re.search(json_pattern, content, re.DOTALL)
        if json_match:
            try:
                json_str = json_match.group(1).strip()
                tool_data = json.loads(json_str)
                
                # Handle nested structure like {"name": "edit_file", "content": {...}}
                if isinstance(tool_data, dict):
                    if 'name' in tool_data and 'content' in tool_data:
                        return [{
                            "name": tool_data["name"],
                            "arguments": tool_data["content"]
                        }]
                    # Check if it's a valid tool call format with name and parameters
                    elif 'name' in tool_data and 'parameters' in tool_data:
                        return [{
                            "name": tool_data["name"],
                            "arguments": tool_data["parameters"]
                        }]
                    # Check if it's a direct parameter format, try to infer tool name from context
                    else:
                        # Look for tool name mentioned in the text before JSON
                        text_before_json = content[:content.find('```json')]
                        
                        # Common tool names to look for
                        tool_names = list(self.tool_map.keys())
                        inferred_tool = None
                        
                        for tool_name in tool_names:
                            if tool_name in text_before_json.lower() or tool_name.replace('_', ' ') in text_before_json.lower():
                                inferred_tool = tool_name
                                break
                        
                        # If no tool found in text, try to infer from parameters
                        if not inferred_tool:
                            if 'target_file' in tool_data and ('should_read_entire_file' in tool_data or 'start_line' in tool_data):
                                inferred_tool = 'read_file'
                            elif 'relative_workspace_path' in tool_data:
                                inferred_tool = 'list_dir'
                            elif 'query' in tool_data and 'target_directories' in tool_data:
                                inferred_tool = 'codebase_search'
                            elif 'query' in tool_data and ('include_pattern' in tool_data or 'exclude_pattern' in tool_data):
                                inferred_tool = 'grep_search'
                            elif 'command' in tool_data and 'is_background' in tool_data:
                                inferred_tool = 'run_terminal_cmd'
                            elif 'target_file' in tool_data and ('instructions' in tool_data or 'code_edit' in tool_data):
                                inferred_tool = 'edit_file'
                        
                        if inferred_tool:
                            return [{
                                "name": inferred_tool,
                                "arguments": tool_data
                            }]
            except json.JSONDecodeError as e:
                pass
        
        # Try to parse plain JSON without code blocks
        try:
            # Look for JSON-like structure in the content
            json_start = content.find('{')
            json_end = content.rfind('}') + 1
            if json_start != -1 and json_end > json_start:
                json_str = content[json_start:json_end]
                tool_data = json.loads(json_str)
                
                # Handle nested structure
                if isinstance(tool_data, dict):
                    if 'name' in tool_data and 'content' in tool_data:
                        return [{
                            "name": tool_data["name"],
                            "arguments": tool_data["content"]
                        }]
                    elif 'name' in tool_data and 'parameters' in tool_data:
                        return [{
                            "name": tool_data["name"],
                            "arguments": tool_data["parameters"]
                        }]
        except json.JSONDecodeError:
            pass
        

        

        
        return []

    def parse_tool_call(self, content: str) -> Optional[Dict[str, Any]]:
        """
        Parse a single tool call from the model's response (backward compatibility).
        
        Args:
            content: The model's response text
            
        Returns:
            Dictionary with tool name and parameters, or None if no tool call found
        """
        tool_calls = self.parse_tool_calls(content)
        return tool_calls[0] if tool_calls else None


    
    def parse_python_function_calls(self, content: str) -> List[Dict[str, Any]]:
        """
        Parse Python-style function calls from the model's response.
        This serves as a fallback for when the model doesn't use the correct XML format.
        
        Args:
            content: The model's response text
            
        Returns:
            List of dictionaries, each representing a function call
        """

        
        function_calls = []
        
        # Pattern to match function calls like: tool_name({"param": "value", ...})
        # Updated pattern to better handle multiline strings and nested braces
        python_pattern = r'(\w+)\s*\(\s*(\{(?:[^{}]|(?:\{[^{}]*\}))*\})\s*\)'
        
        matches = re.findall(python_pattern, content, re.DOTALL)
        
        for tool_name, params_str in matches:
            # Check if this is a valid tool name
            if tool_name in self.tool_map:
                try:
                    # Try to parse the parameters as JSON directly
                    # Fix common JSON issues
                    params_json = params_str.replace("'", '"')  # Replace single quotes with double quotes
                    
                    params = json.loads(params_json)
                    
                    function_calls.append({
                        "name": tool_name,
                        "arguments": params
                    })
                    
                except json.JSONDecodeError as e:
                    # Try to extract individual parameters manually
                    try:
                        params = self._parse_python_params_manually(params_str)
                        if params:
                            function_calls.append({
                                "name": tool_name,
                                "arguments": params
                            })
                    except Exception as e2:
                        continue
        
        return function_calls
    
    def _parse_python_params_manually(self, params_str: str) -> Dict[str, Any]:
        """
        Manually parse Python function parameters when JSON parsing fails.
        
        Args:
            params_str: Parameter string from Python function call
            
        Returns:
            Dictionary of parameters
        """
        params = {}
        
        # Remove the outer braces if present
        if params_str.startswith('{') and params_str.endswith('}'):
            params_str = params_str[1:-1].strip()
        
        # Split by commas, but be careful about commas inside strings
        param_parts = []
        current_part = ""
        in_quotes = False
        quote_char = None
        brace_depth = 0
        
        for char in params_str:
            if char in ('"', "'") and not in_quotes:
                in_quotes = True
                quote_char = char
                current_part += char
            elif char == quote_char and in_quotes:
                in_quotes = False
                quote_char = None
                current_part += char
            elif char == '{' and not in_quotes:
                brace_depth += 1
                current_part += char
            elif char == '}' and not in_quotes:
                brace_depth -= 1
                current_part += char
            elif char == ',' and not in_quotes and brace_depth == 0:
                param_parts.append(current_part.strip())
                current_part = ""
            else:
                current_part += char
        
        if current_part.strip():
            param_parts.append(current_part.strip())
        
        # Parse each parameter
        for part in param_parts:
            # Look for key: value pattern
            if ':' in part:
                key_value = part.split(':', 1)
                if len(key_value) == 2:
                    key = key_value[0].strip().strip('"\'')
                    value = key_value[1].strip()
                    
                    # Remove quotes from value if present
                    if value.startswith('"') and value.endswith('"'):
                        value = value[1:-1]
                    elif value.startswith("'") and value.endswith("'"):
                        value = value[1:-1]
                    
                    # Convert boolean values
                    if value.lower() in ('true', 'false'):
                        value = value.lower() == 'true'
                    # Convert numeric values
                    elif value.isdigit():
                        value = int(value)
                    
                    params[key] = value
        
        return params
    
    def parse_function_calls(self, function_calls_text: str) -> List[Dict[str, Any]]:
        """
        Parse function calls from the given text.
        
        Args:
            function_calls_text: Text containing function calls
            
        Returns:
            List of dictionaries, each representing a function call
        """
        function_calls = []
        # Look for individual function calls
        invoke_pattern = r'<invoke name="([^"]+)">(.*?)</invoke>'
        invokes = re.findall(invoke_pattern, function_calls_text, re.DOTALL)
        for name, args_text in invokes:
            # Parse the arguments
            args = self.parse_arguments(args_text)
            function_calls.append({"name": name, "arguments": args})
        return function_calls
    
    def parse_arguments(self, args_text: str) -> Dict[str, Any]:
        """
        Parse arguments from the given text.
        
        Args:
            args_text: Text containing arguments
            
        Returns:
            Dictionary of argument names and values
        """
        args = {}
        
        # Method 1: Try the traditional <parameter name="...">value</parameter> format
        arg_pattern = r'<parameter name="([^"]+)">(.*?)</parameter>'
        arg_matches = re.findall(arg_pattern, args_text, re.DOTALL)
        for name, value in arg_matches:
            value = value.strip()
            args[name] = self._convert_parameter_value(value)
        
        # Method 2: Try the direct tag format <tag_name>value</tag_name>
        # This supports the more intuitive XML format that models often generate
        if not args:  # Only try this if the traditional format didn't work
            # Find all XML tags and their content
            direct_tag_pattern = r'<([^/][^>]*?)>(.*?)</\1>'
            direct_matches = re.findall(direct_tag_pattern, args_text, re.DOTALL)
            
            for tag_name, value in direct_matches:
                # Clean up the tag name (remove any attributes)
                tag_name = tag_name.split()[0]
                value = value.strip()
                
                # Handle special cases for array-like structures
                if tag_name == 'target_directories':
                    # Handle <target_directories><item>value1</item><item>value2</item></target_directories>
                    item_pattern = r'<item[^>]*>(.*?)</item>'
                    items = re.findall(item_pattern, value, re.DOTALL)
                    if items:
                        args[tag_name] = [item.strip() for item in items]
                    else:
                        args[tag_name] = self._convert_parameter_value(value)
                else:
                    args[tag_name] = self._convert_parameter_value(value)
        
        return args
    
    def _convert_parameter_value(self, value: str) -> Any:
        """
        Convert parameter value to appropriate type.
        
        Args:
            value: String value to convert
            
        Returns:
            Converted value (string, int, bool, list, etc.)
        """
        # For certain parameters that may contain meaningful whitespace/formatting,
        # don't strip the value
        value_stripped = value.strip()
        
        # Handle boolean values
        if value_stripped.lower() in ('true', 'false'):
            return value_stripped.lower() == 'true'
        
        # Handle integers
        if value_stripped.isdigit():
            return int(value_stripped)
        
        # Handle negative integers
        if value_stripped.startswith('-') and value_stripped[1:].isdigit():
            return int(value_stripped)
        
        # Handle JSON arrays/objects
        if (value_stripped.startswith('[') and value_stripped.endswith(']')) or (value_stripped.startswith('{') and value_stripped.endswith('}')):
            try:
                return json.loads(value_stripped)
            except json.JSONDecodeError:
                pass
        
        # Return original value (not stripped) for string parameters to preserve formatting
        return value
    
    def execute_tool(self, tool_call: Dict[str, Any]) -> Any:
        """
        Execute a tool with the given parameters.
        
        Args:
            tool_call: Dictionary containing tool name and parameters
            
        Returns:
            Result of executing the tool
        """
        tool_name = tool_call["name"]
        params = tool_call["arguments"]
        
        print(f"Executing tool: {tool_name} with params: {list(params.keys())}")
        
        if tool_name in self.tool_map:
            tool_func = self.tool_map[tool_name]
            try:
                # Filter out None values and empty strings for optional parameters
                filtered_params = {k: v for k, v in params.items() if v is not None and v != ""}
                
                # Special handling for read_file to map end_line_one_indexed to end_line_one_indexed_inclusive
                if tool_name == "read_file" and "end_line_one_indexed" in filtered_params:
                    # Map end_line_one_indexed to end_line_one_indexed_inclusive
                    filtered_params["end_line_one_indexed_inclusive"] = filtered_params.pop("end_line_one_indexed")
                    print("Mapped end_line_one_indexed parameter to end_line_one_indexed_inclusive")
                
                # Robustness handling: auto-correct wrong parameter names for edit_file and read_file
                if tool_name in ["edit_file", "read_file"]:
                    # Map relative_workspace_path to target_file
                    if "relative_workspace_path" in filtered_params:
                        filtered_params["target_file"] = filtered_params.pop("relative_workspace_path")
                        print(f"🔧 Auto-corrected parameter: relative_workspace_path -> target_file for {tool_name}")
                    # Map file_path to target_file
                    if "file_path" in filtered_params:
                        filtered_params["target_file"] = filtered_params.pop("file_path")
                        print(f"🔧 Auto-corrected parameter: file_path -> target_file for {tool_name}")
                    # Map filename to target_file (for edit_file)
                    if "filename" in filtered_params:
                        filtered_params["target_file"] = filtered_params.pop("filename")
                        print(f"🔧 Auto-corrected parameter: filename -> target_file for {tool_name}")
                
                # Robustness handling for edit_file: auto-correct content to code_edit
                if tool_name == "edit_file" and "content" in filtered_params:
                    # Map content to code_edit
                    filtered_params["code_edit"] = filtered_params.pop("content")
                    print(f"🔧 Auto-corrected parameter: content -> code_edit for {tool_name}")
                
                # Robustness handling for codebase_search: auto-correct search_term to query
                if tool_name == "codebase_search" and "search_term" in filtered_params:
                    # Map search_term to query
                    filtered_params["query"] = filtered_params.pop("search_term")
                    print(f"🔧 Auto-corrected parameter: search_term -> query for {tool_name}")
                
                # No special handling needed for run_terminal_cmd anymore
                
                result = tool_func(**filtered_params)
                
                # Enhanced error handling for edit_file and other tools
                if isinstance(result, dict) and result.get('status') == 'error':
                    # For terminal commands, preserve stdout and stderr information
                    if tool_name == 'run_terminal_cmd':
                        # Keep the original result with all details for terminal commands
                        return result
                    else:
                        # Return detailed error information for other failed tool executions
                        error_msg = result.get('error', 'Unknown error occurred')
                        return {
                            'tool': tool_name,
                            'status': 'error', 
                            'error': error_msg,
                            'parameters': filtered_params
                        }
                
                return result
            except TypeError as e:
                # Handle parameter mismatch - include correct usage example
                usage_example = self._get_tool_usage_example(tool_name)
                error_result = {
                    'tool': tool_name,
                    'status': 'error',
                    'error': f"Parameter mismatch: {str(e)}",
                    'parameters': params,
                    'correct_usage': usage_example
                }
                print(f"❌ Tool execution failed: {error_result}")
                return error_result
            except Exception as e:
                # General exception handling - include correct usage example
                usage_example = self._get_tool_usage_example(tool_name)
                error_result = {
                    'tool': tool_name,
                    'status': 'error', 
                    'error': f"Execution failed: {str(e)}",
                    'parameters': params,
                    'correct_usage': usage_example
                }
                print(f"❌ Tool execution failed: {error_result}")
                return error_result
        else:
            # Unknown tool - provide list of available tools with brief usage info
            available_tools_list = list(self.tool_map.keys())
            tools_help_summary = []
            
            # Get brief help for each available tool
            for available_tool in available_tools_list[:5]:  # Limit to first 5 tools to avoid overwhelming output
                try:
                    help_info = self.tools.tool_help(available_tool)
                    if 'description' in help_info and 'error' not in help_info:
                        # Get first sentence of description
                        desc = help_info['description'].split('.')[0].split('\n')[0][:100]
                        tools_help_summary.append(f"- {available_tool}: {desc}")
                    else:
                        tools_help_summary.append(f"- {available_tool}")
                except:
                    tools_help_summary.append(f"- {available_tool}")
            
            if len(available_tools_list) > 5:
                tools_help_summary.append(f"... and {len(available_tools_list) - 5} more tools")
            
            available_tools_help = "\n".join(tools_help_summary)
            
            error_result = {
                'tool': tool_name,
                'status': 'error',
                'error': f"Unknown tool: {tool_name}",
                'available_tools': available_tools_list,
                'available_tools_help': f"Available tools:\n{available_tools_help}\n\nUse tool_help('<tool_name>') to get detailed usage for any specific tool."
            }
            print(f"❌ Tool execution failed: {error_result}")
            return error_result
    
    def _format_dict_as_text(self, data: Dict[str, Any]) -> str:
        """
        Format a dictionary result as readable text.
        
        Args:
            data: Dictionary to format
            
        Returns:
            Formatted text string
        """
        if not isinstance(data, dict):
            return str(data)
        
        lines = []
        
        # Handle common result patterns
        if 'error' in data:
            error_msg = f"Error: {data['error']}"
            if 'tool' in data:
                error_msg = f"Tool '{data['tool']}' failed: {data['error']}"
            if 'parameters' in data:
                error_msg += f"\nParameters used: {data['parameters']}"
            if 'available_tools' in data:
                error_msg += f"\nAvailable tools: {', '.join(data['available_tools'])}"
            if 'available_tools_help' in data:
                error_msg += f"\n\n{data['available_tools_help']}"
            if 'correct_usage' in data:
                error_msg += f"\n\n{data['correct_usage']}"
            return error_msg
        
        if 'status' in data:
            lines.append(f"Status: {data['status']}")
        
        if 'file' in data:
            lines.append(f"File: {data['file']}")
        
        if 'content' in data:
            lines.append(f"Content:\n{data['content']}")
        
        # Special handling for web search results
        if 'search_term' in data and 'results' in data:
            lines.append(f"Search Term: {data['search_term']}")
            if 'timestamp' in data:
                lines.append(f"Search Time: {data['timestamp']}")
            
            results = data['results']
            if isinstance(results, list):
                lines.append(f"\nSearch Results ({len(results)} items):")
                for i, result in enumerate(results[:10], 1):  # Limit to first 10
                    if isinstance(result, dict):
                        lines.append(f"\n{i}. {result.get('title', 'No Title')}")
                        
                        # URL field removed from display to reduce clutter
                        
                        # Handle content with priority: full_content > content > content_summary > snippet
                        content_shown = False
                        if result.get('full_content'):
                            content = result['full_content']
                            if len(content) > get_truncation_length():
                                lines.append(f"   Content: {content[:get_truncation_length()]}...\n   [Content truncated - showing first {get_truncation_length()} characters]")
                            else:
                                lines.append(f"   Content: {content}")
                            content_shown = True
                        elif result.get('content'):
                            content = result['content']
                            if len(content) > get_truncation_length():
                                lines.append(f"   Content: {content[:get_truncation_length()]}...\n   [Content truncated - showing first {get_truncation_length()} characters]")
                            else:
                                lines.append(f"   Content: {content}")
                            content_shown = True
                        elif result.get('content_summary'):
                            lines.append(f"   Content Summary: {result['content_summary']}")
                            content_shown = True
                        
                        # If no content, show snippet
                        if not content_shown and result.get('snippet'):
                            lines.append(f"   Summary: {result['snippet'][:get_truncation_length()]}...")
                        
                        # Show source
                        if result.get('source'):
                            lines.append(f"   Source: {result['source']}")
                        
                        # Show content status for results without content
                        if result.get('content_status'):
                            lines.append(f"   Content Status: {result['content_status']}")
            
            # Add additional web search metadata
            if 'content_fetched' in data:
                lines.append(f"\nContent Fetched: {data['content_fetched']}")
            if 'total_results' in data:
                lines.append(f"Total Results: {data['total_results']}")
            if 'results_with_content' in data:
                lines.append(f"Results with Content: {data['results_with_content']}")
            
            return '\n'.join(lines)
        
        # Handle other types of results
        if 'results' in data:
            results = data['results']
            if isinstance(results, list):
                lines.append(f"Results ({len(results)} items):")
                for i, result in enumerate(results[:10]):  # Limit to first 10
                    if isinstance(result, dict):
                        if 'file' in result and 'line_number' in result:
                            lines.append(f"  {i+1}. {result['file']}:{result['line_number']} - {result.get('line', '')}")
                        elif 'file' in result and 'snippet' in result:
                            lines.append(f"  {i+1}. {result['file']} - {result['snippet'][:get_truncation_length()]}...")
                        else:
                            lines.append(f"  {i+1}. {str(result)[:get_truncation_length()]}...")
                    else:
                        lines.append(f"  {i+1}. {str(result)}")
        
        if 'output' in data:
            lines.append(f"Output:\n{data['output']}")
        
        if 'stdout' in data:
            lines.append(f"Output:\n{data['stdout']}")
        
        if 'stderr' in data and data['stderr']:
            lines.append(f"Error Output:\n{data['stderr']}")
        
        # If no specific formatting applied, show all key-value pairs
        if not lines:
            for key, value in data.items():
                if isinstance(value, (list, dict)):
                    lines.append(f"{key}: {json.dumps(value, indent=2, ensure_ascii=False)}")
                else:
                    lines.append(f"{key}: {value}")
        
        return '\n'.join(lines)

    def _get_tool_usage_example(self, tool_name: str) -> str:
        """
        Get correct usage example for a tool from help_tools.
        
        Args:
            tool_name: Name of the tool to get usage example for
            
        Returns:
            Formatted usage example string
        """
        try:
            # Get tool help information
            help_info = self.tools.tool_help(tool_name)
            
            if 'error' in help_info:
                return f"Tool '{tool_name}' not found in help system."
            
            # Format the usage example
            usage_parts = []
            usage_parts.append(f"## Correct Usage for '{tool_name}':")
            usage_parts.append("")
            
            # Add description
            if 'description' in help_info:
                usage_parts.append(f"**Description:** {help_info['description']}")
                usage_parts.append("")
            
            # Add parameters information
            if 'parameters' in help_info:
                params_info = help_info['parameters']
                properties = params_info.get('properties', {})
                required_params = params_info.get('required', [])
                
                usage_parts.append("**Parameters:**")
                for param_name, param_info in properties.items():
                    param_type = param_info.get('type', 'string')
                    description = param_info.get('description', '')
                    is_required = param_name in required_params
                    required_marker = " **(REQUIRED)**" if is_required else " (optional)"
                    
                    usage_parts.append(f"- `{param_name}` ({param_type}){required_marker}: {description}")
                usage_parts.append("")
            
            # Add usage example
            usage_parts.append("**Correct XML Format Example:**")
            usage_parts.append("```xml")
            usage_parts.append("<function_calls>")
            usage_parts.append(f'<invoke name="{tool_name}">')
            
            # Generate example parameters
            if 'parameters' in help_info:
                properties = help_info['parameters'].get('properties', {})
                required_params = help_info['parameters'].get('required', [])
                
                for param_name, param_info in properties.items():
                    param_type = param_info.get('type', 'string')
                    
                    # Generate example values based on type and tool
                    if param_type == "boolean":
                        example_value = "false"
                    elif param_type == "integer":
                        if param_name in ['start_line_one_indexed', 'end_line_one_indexed_inclusive']:
                            example_value = "1" if 'start' in param_name else "50"
                        else:
                            example_value = "1"
                    elif param_type == "array":
                        if tool_name == "codebase_search" and param_name == "target_directories":
                            example_value = '["src/*", "lib/*"]'
                        else:
                            example_value = '["example1", "example2"]'
                    else:  # string
                        if param_name == "target_file":
                            example_value = "src/main.py"
                        elif param_name == "query":
                            example_value = "search term or code pattern"
                        elif param_name == "command":
                            example_value = "ls -la"
                        elif param_name == "relative_workspace_path":
                            example_value = "src"
                        elif param_name == "instructions":
                            example_value = "Brief description of the change you are making"
                        elif param_name == "code_edit":
                            example_value = "# ... existing code ...\nnew_code_here\n# ... existing code ..."
                        elif param_name == "search_term":
                            example_value = "Python best practices"

                        else:
                            example_value = f"your_{param_name}_here"
                    
                    usage_parts.append(f'<parameter name="{param_name}">{example_value}</parameter>')
            
            usage_parts.append("</invoke>")
            usage_parts.append("</function_calls>")
            usage_parts.append("```")
            usage_parts.append("")
            
            # Add alternative direct XML format
            usage_parts.append("**Alternative Direct XML Format:**")
            usage_parts.append("```xml")
            usage_parts.append("<function_calls>")
            usage_parts.append(f'<invoke name="{tool_name}">')
            
            if 'parameters' in help_info:
                properties = help_info['parameters'].get('properties', {})
                for param_name, param_info in properties.items():
                    param_type = param_info.get('type', 'string')
                    
                    if param_type == "boolean":
                        example_value = "false"
                    elif param_type == "integer":
                        example_value = "1"
                    elif param_type == "array":
                        example_value = '["item1", "item2"]'
                    else:
                        example_value = "value"
                    
                    usage_parts.append(f'<{param_name}>{example_value}</{param_name}>')
            
            usage_parts.append("</invoke>")
            usage_parts.append("</function_calls>")
            usage_parts.append("```")
            
            return '\n'.join(usage_parts)
            
        except Exception as e:
            return f"Failed to get usage example for '{tool_name}': {str(e)}"

    def _build_combined_user_prompt(self, user_prompt: str, task_history: List[Dict[str, Any]] = None) -> str:
        """
        Build a combined user prompt that includes task history for cache optimization.
        Instead of using chat history, we combine all context into a single user message.
        
        Args:
            user_prompt: Current user prompt
            task_history: Previous task execution history
            
        Returns:
            Combined user prompt string
        """
        prompt_parts = []
        
        # Add task history context if provided
        if task_history:
            prompt_parts.append("## Previous Task Context:")
            prompt_parts.append("Below is the context from previous tasks in this session:\n")
            
            # 从配置文件读取历史记录截断长度
            history_truncation_length = get_history_truncation_length()
            
            for i, record in enumerate(task_history, 1):
                if record.get("role") == "system":
                    # Skip system messages as they're already in system prompt
                    continue
                elif "prompt" in record and "result" in record:
                    # Convert task history record to context
                    prompt_parts.append(f"### Previous Task {i}:")
                    prompt_parts.append(f"**User Request:** {record['prompt']}")
                    prompt_parts.append(f"**Assistant Response:** {record['result'][:history_truncation_length]}..." if len(record['result']) > history_truncation_length else f"**Assistant Response:** {record['result']}")
                    prompt_parts.append("")  # Empty line for separation
                elif record.get("role") == "user":
                    prompt_parts.append(f"### Previous User Request {i}:")
                    prompt_parts.append(record.get("content", ""))
                    prompt_parts.append("")
                elif record.get("role") == "assistant":
                    prompt_parts.append(f"### Previous Assistant Response {i}:")
                    content = record.get("content", "")
                    # Truncate very long assistant responses for cache efficiency
                    if len(content) > history_truncation_length:
                        content = content[:history_truncation_length] + "... [truncated for cache optimization]"
                    prompt_parts.append(content)
                    prompt_parts.append("")
            
            prompt_parts.append("## Current Task:")
            prompt_parts.append("Based on the above context, please handle the following current request:\n")
        
        # Add current user prompt
        prompt_parts.append(user_prompt)
        
        combined_prompt = "\n".join(prompt_parts)
        
        if self.debug_mode:
            print(f"🔄 Cache optimization: Combined prompt length: {len(combined_prompt)} characters")
            if task_history:
                print(f"🔄 Combined {len(task_history)} historical records into current prompt")
            else:
                print(f"🔄 No history, using direct prompt for optimal caching")
        
        return combined_prompt

    def _build_conversation_history_for_cache(self, messages: List[Dict[str, Any]], assistant_response: str, tool_results: str) -> str:
        """
        Build conversation history for cache optimization by combining all context into a single user prompt.
        When summary_history is enabled, summarizes previous conversation history except the latest tool result.
        
        Args:
            messages: Current message array (system + user + assistant)
            assistant_response: Latest assistant response 
            tool_results: Tool execution results
            
        Returns:
            Combined conversation history as user prompt
        """
        history_parts = []
        
        # Extract the original user prompt from messages (skip system message)
        original_user_content = ""
        for msg in messages:
            if msg["role"] == "user":
                original_user_content = msg["content"]
                break
        
        # Check if we should use history summarization
        if self.summary_history and self.conversation_summarizer and len(messages) > 2:
            # Calculate total conversation history length to decide if summarization is needed
            total_conversation_length = 0
            conversation_history = []
            for msg in messages:
                if msg["role"] in ["user", "assistant"]:
                    content = msg.get("content", "")
                    total_conversation_length += len(content)
                    conversation_history.append(msg)
            
            # Also add assistant response and tool results to total length calculation
            total_conversation_length += len(assistant_response) + len(tool_results)
            
            print(f"📊 Total conversation length: {total_conversation_length} chars (trigger threshold: {self.summary_trigger_length} chars)")
            
            # Only summarize if total length exceeds trigger threshold
            if total_conversation_length > self.summary_trigger_length:
                print("📋 Conversation length exceeds threshold, using history summarization for context control...")
                
                # Generate summary of previous conversation (excluding latest tool results)
                try:
                    conversation_summary = self.conversation_summarizer.generate_conversation_history_summary(
                        conversation_history, 
                        latest_tool_result=tool_results
                    )
                    
                    # Add summarized context
                    history_parts.append("## Previous Conversation Summary:")
                    history_parts.append(conversation_summary)
                    
                    # Add latest assistant response
                    history_parts.append("## Latest Assistant Response:")
                    history_parts.append(assistant_response)
                    
                    # Add latest tool execution results (full text)
                    history_parts.append("## Latest Tool Execution Results:")
                    history_parts.append(tool_results)
                    
                    print(f"✅ Using summarized context ({len(conversation_summary)} chars) + latest tool results ({len(tool_results)} chars)")
                    
                except Exception as e:
                    print(f"⚠️ Failed to generate conversation summary: {e}, falling back to full history")
                    # Fall back to original behavior
                    return self._build_full_conversation_history(original_user_content, assistant_response, tool_results)
            else:
                print("📝 Conversation length below threshold, using full conversation history")
                # Use full history when below threshold
                return self._build_full_conversation_history(original_user_content, assistant_response, tool_results)
        else:
            # Original behavior when history summarization is disabled
            return self._build_full_conversation_history(original_user_content, assistant_response, tool_results)
        
        # Add continuation prompt
        history_parts.append("## Current Request:")
        history_parts.append("Please continue processing based on the above context and tool execution results. Provide your next response or tool calls as needed.")
        
        return "\n\n".join(history_parts)
    
    def _build_full_conversation_history(self, original_user_content: str, assistant_response: str, tool_results: str) -> str:
        """
        Build full conversation history without summarization (original behavior)
        
        Args:
            original_user_content: Original user prompt content
            assistant_response: Latest assistant response
            tool_results: Tool execution results
            
        Returns:
            Full conversation history as user prompt
        """
        history_parts = []
        
        # Add original context
        history_parts.append("## Previous Conversation Context:")
        history_parts.append(original_user_content)
        
        # Add assistant response
        history_parts.append("## Assistant Response:")
        history_parts.append(assistant_response)
        
        # Add tool execution results
        history_parts.append("## Tool Execution Results:")
        history_parts.append(tool_results)
        
        # Add continuation prompt
        history_parts.append("## Current Request:")
        history_parts.append("Please continue processing based on the above context and tool execution results. Provide your next response or tool calls as needed.")
        
        return "\n\n".join(history_parts)

    def _generate_cache_key(self, system_prompt: str, user_prompt: str) -> str:
        """
        Generate a cache key for the current LLM request to help identify potential cache hits.
        
        Args:
            system_prompt: System prompt content
            user_prompt: User prompt content
            
        Returns:
            Cache key string (hash of the prompt content)
        """
        import hashlib
        
        # Combine system and user prompts
        combined_content = f"SYSTEM:{system_prompt}\nUSER:{user_prompt}\nMODEL:{self.model}"
        
        # Generate MD5 hash for cache key
        cache_key = hashlib.md5(combined_content.encode('utf-8')).hexdigest()
        
        if self.debug_mode:
            print(f"🔑 Cache key generated: {cache_key}")
            print(f"🔑 Content hash based on: system({len(system_prompt)} chars) + user({len(user_prompt)} chars) + model({self.model})")
        
        return cache_key

    def analyze_cache_potential(self, logs_dir: str = None) -> Dict[str, Any]:
        """
        Analyze cache potential by examining LLM call logs.
        
        Args:
            logs_dir: Directory containing LLM call logs
            
        Returns:
            Dictionary with cache analysis results
        """
        if logs_dir is None:
            logs_dir = self.llm_logs_dir
        
        print(f"🔍 Analyzing cache potential in: {logs_dir}")
        
        cache_keys = {}
        total_calls = 0
        cache_optimized_calls = 0
        
        try:
            for filename in os.listdir(logs_dir):
                if filename.startswith('llm_call_') and filename.endswith('.json'):
                    file_path = os.path.join(logs_dir, filename)
                    
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            log_data = json.load(f)
                        
                        call_info = log_data.get('call_info', {})
                        cache_key = call_info.get('cache_key', 'unknown')
                        is_cache_optimized = call_info.get('cache_optimized', False)
                        
                        total_calls += 1
                        if is_cache_optimized:
                            cache_optimized_calls += 1
                        
                        if cache_key in cache_keys:
                            cache_keys[cache_key]['count'] += 1
                            cache_keys[cache_key]['files'].append(filename)
                        else:
                            cache_keys[cache_key] = {
                                'count': 1,
                                'files': [filename],
                                'cache_optimized': is_cache_optimized
                            }
                    except Exception as e:
                        print(f"⚠️ Error reading log file {filename}: {e}")
                        continue
            
            # Calculate potential savings
            potential_cache_hits = sum(1 for key_data in cache_keys.values() if key_data['count'] > 1)
            total_repeated_calls = sum(max(0, key_data['count'] - 1) for key_data in cache_keys.values())
            
            analysis = {
                'total_calls': total_calls,
                'cache_optimized_calls': cache_optimized_calls,
                'unique_cache_keys': len(cache_keys),
                'potential_cache_hits': total_repeated_calls,
                'cache_hit_rate': (total_repeated_calls / total_calls * 100) if total_calls > 0 else 0,
                'repeated_requests': {k: v for k, v in cache_keys.items() if v['count'] > 1}
            }
            
            print(f"📊 Cache Analysis Results:")
            print(f"   Total LLM calls: {total_calls}")
            print(f"   Cache-optimized calls: {cache_optimized_calls}")
            print(f"   Unique cache keys: {len(cache_keys)}")
            print(f"   Potential cache hits: {total_repeated_calls}")
            print(f"   Potential cache hit rate: {analysis['cache_hit_rate']:.1f}%")
            
            if analysis['repeated_requests']:
                print(f"🔄 Repeated requests (potential cache hits):")
                for cache_key, data in analysis['repeated_requests'].items():
                    print(f"   Cache key {cache_key[:16]}...: {data['count']} calls")
            
            return analysis
            
        except Exception as e:
            print(f"❌ Cache analysis failed: {e}")
            return {'error': str(e)}

    def _get_workspace_context(self) -> str:
        """
        Get basic workspace context without detailed information that could cause hallucination.
        
        Returns:
            String representation of workspace context
        """
        if not os.path.exists(self.workspace_dir):
            return ""
        
        context_parts = ["\n**Current Workspace Information:**\n"]
        
        # Define code file extensions to include
        code_extensions = {'.py', '.js', '.ts', '.jsx', '.tsx', '.java', '.c', '.cpp', '.h', '.css', '.html', '.php', '.rb', '.go', '.rs', '.swift', '.kt', '.scala', '.sql', '.sh', '.bat', '.ps1', '.yaml', '.yml', '.json', '.xml', '.md', '.txt'}
        
        # Find all code files in workspace
        code_files = []
        total_files = 0
        total_size = 0
        
        for root, dirs, files in os.walk(self.workspace_dir):
            # Skip hidden directories and common non-code directories
            dirs[:] = [d for d in dirs if not d.startswith('.') and d not in {'node_modules', '__pycache__', 'venv', 'env', 'build', 'dist', 'target'}]
            
            for file in files:
                if any(file.endswith(ext) for ext in code_extensions) and not file.startswith('.'):
                    file_path = os.path.join(root, file)
                    rel_path = os.path.relpath(file_path, self.workspace_dir)
                    
                    try:
                        # Get file size and modification time
                        stat_info = os.stat(file_path)
                        file_size = stat_info.st_size
                        
                        code_files.append({
                            'path': rel_path,
                            'size': file_size,
                        })
                        
                        total_files += 1
                        total_size += file_size
                        
                    except Exception as e:
                        print(f"⚠️   Unable to get file information {rel_path}: {e}")
                        continue
        
        if not code_files:
            context_parts.append("No files found. Use list_dir tool to explore the workspace.\n")
            return ''.join(context_parts)
        
        # Add basic summary statistics only
        context_parts.append(f"📊 **Basic Statistics**: {total_files} files, total size {self._format_file_size(total_size)}\n")
        context_parts.append("⚠️ **Important**: File names and statistics shown above are for reference only.\n")
        context_parts.append("**You MUST use tools (list_dir, read_file, codebase_search) to get actual file contents before making any analysis or conclusions.**\n")
        
        return ''.join(context_parts)
    
    def _format_file_size(self, size_bytes: int) -> str:
        """
        Format file size in human readable format.
        
        Args:
            size_bytes: Size in bytes
            
        Returns:
            Formatted size string
        """
        if size_bytes < 1024:
            return f"{size_bytes}B"
        elif size_bytes < 1024 * 1024:
            return f"{size_bytes / 1024:.1f}KB"
        else:
            return f"{size_bytes / (1024 * 1024):.1f}MB"
    
    def _get_file_language(self, file_path: str) -> str:
        """
        Get the programming language for syntax highlighting based on file extension.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Language identifier for syntax highlighting
        """
        ext = os.path.splitext(file_path)[1].lower()
        
        language_map = {
            '.py': 'python',
            '.js': 'javascript',
            '.ts': 'typescript',
            '.jsx': 'jsx',
            '.tsx': 'tsx',
            '.java': 'java',
            '.c': 'c',
            '.cpp': 'cpp',
            '.h': 'c',
            '.css': 'css',
            '.html': 'html',
            '.php': 'php',
            '.rb': 'ruby',
            '.go': 'go',
            '.rs': 'rust',
            '.swift': 'swift',
            '.kt': 'kotlin',
            '.scala': 'scala',
            '.sql': 'sql',
            '.sh': 'bash',
            '.bat': 'batch',
            '.ps1': 'powershell',
            '.yaml': 'yaml',
            '.yml': 'yaml',
            '.json': 'json',
            '.xml': 'xml',
            '.md': 'markdown'
        }
        
        return language_map.get(ext, 'text')

    def _save_llm_call_debug_log(self, messages: List[Dict[str, Any]], content: str, current_round: int = 0, tool_calls_info: Dict[str, Any] = None) -> None:
        """
        Save detailed debug log for LLM call.
        
        Args:
            messages: Complete messages sent to LLM
            content: LLM response content
            current_round: Current round number
            tool_calls_info: Additional tool call information for better logging
        """
        try:
            # Increment call counter
            self.llm_call_counter += 1
            
            # Create timestamp
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # microseconds to milliseconds
            
            # Create log filename
            log_filename = f"llm_call_{self.llm_call_counter:03d}_{timestamp}.json"
            log_path = os.path.join(self.llm_logs_dir, log_filename)
            
            # Generate cache key for this call
            if len(messages) >= 2:
                system_content = messages[0].get('content', '') if messages[0].get('role') == 'system' else ''
                user_content = messages[1].get('content', '') if messages[1].get('role') == 'user' else ''
                call_cache_key = self._generate_cache_key(system_content, user_content)
            else:
                call_cache_key = "unknown"
            
            # Prepare debug data - including detailed tool call information
            # All rounds use cache optimization format (system + combined_user_prompt)
            is_cache_optimized = True
            
            debug_data = {
                "call_info": {
                    "call_number": self.llm_call_counter,
                    "timestamp": datetime.datetime.now().isoformat(),
                    "model": self.model,
                    "cache_key": call_cache_key,
                    "cache_optimized": is_cache_optimized,  # True for initial cache-optimized requests
                    "round": current_round  # Track which round this is
                },
                "messages": messages,
                "response_content": content
            }
            
            # Add tool call information if available
            if tool_calls_info:
                debug_data["tool_calls_info"] = tool_calls_info
                
                # Add detailed breakdown for better debugging
                if "parsed_tool_calls" in tool_calls_info:
                    debug_data["call_info"]["tool_calls_count"] = len(tool_calls_info["parsed_tool_calls"])
                    debug_data["call_info"]["tool_names"] = [tc.get("name", "unknown") for tc in tool_calls_info["parsed_tool_calls"]]
                
                if "tool_results" in tool_calls_info:
                    debug_data["call_info"]["tool_results_count"] = len(tool_calls_info["tool_results"])
                    
                if "formatted_tool_results" in tool_calls_info:
                    debug_data["call_info"]["formatted_results_length"] = len(tool_calls_info["formatted_tool_results"])
            
            # Save to JSON file
            with open(log_path, 'w', encoding='utf-8') as f:
                json.dump(debug_data, f, ensure_ascii=False, indent=2)
            
            print(f"🐛 Debug log saved: {log_filename}")
            
        except Exception as e:
            print(f"⚠️ Debug log save failed: {e}")

    def _format_tool_results_for_llm(self, tool_results: List[Dict[str, Any]]) -> str:
        """
        Format tool execution results for the LLM to understand.
        
        Args:
            tool_results: List of tool execution results
            
        Returns:
            Formatted message string for the LLM
        """
        if not tool_results:
            return "No tool results to report."
        
        # 从配置文件读取截断长度
        truncation_length = get_truncation_length()
        
        message_parts = ["Tool execution results:\n"]
        
        for i, result in enumerate(tool_results, 1):
            tool_name = result.get('tool_name', 'unknown')
            tool_params = result.get('tool_params', {})
            tool_result = result.get('tool_result', '')
            
            # Format the tool result section
            message_parts.append(f"## Tool {i}: {tool_name}")
            
            # Add parameters if meaningful
            if tool_params:
                key_params = []
                for key, value in tool_params.items():
                    if key in ['target_file', 'query', 'command', 'relative_workspace_path', 'search_term', 'instructions']:
                        # Truncate long values for readability
                        if isinstance(value, str) and len(value) > truncation_length:
                            value = value[:truncation_length] + "..."
                        key_params.append(f"{key}={value}")
                if key_params:
                    message_parts.append(f"**Parameters:** {', '.join(key_params)}")
            
            # Check if this is a read_file operation with should_read_entire_file=true
            is_read_entire_file = (tool_name == 'read_file' and 
                                 tool_params.get('should_read_entire_file', False) is True)
            
            # Format the result
            message_parts.append("**Result:**")
            if isinstance(tool_result, dict):
                # Handle different types of tool results
                if 'error' in tool_result:
                    message_parts.append(f"❌ Error: {tool_result['error']}")
                elif 'status' in tool_result:
                    status = tool_result['status']
                    if status == 'completed' or status == 'success':
                        message_parts.append(f"✅ {status.title()}")
                    elif status == 'error':
                        message_parts.append(f"❌ {status.title()}")
                    else:
                        message_parts.append(f"ℹ️ {status.title()}")
                    
                    # Add additional result details
                    for key, value in tool_result.items():
                        if key not in ['status', 'command', 'working_directory']:
                            # For read_entire_file operations, don't truncate content
                            if is_read_entire_file and key == 'content':
                                message_parts.append(f"- {key}: {value}")
                                print(f"📄 Full file content passed to LLM, length: {len(value) if isinstance(value, str) else 'N/A'} characters")
                            elif isinstance(value, str) and len(value) > truncation_length:
                                # Truncate very long content for non-read-entire-file operations
                                value = value[:truncation_length] + f"... [Content truncated, total length: {len(value)} characters]"
                                message_parts.append(f"- {key}: {value}")
                            else:
                                message_parts.append(f"- {key}: {value}")
                else:
                    # Fallback formatting
                    formatted_result = self._format_dict_as_text(tool_result)
                    # For read_entire_file operations, don't truncate the formatted result
                    if is_read_entire_file:
                        message_parts.append(formatted_result)
                        print(f"📄 Full file content formatted and passed to LLM")
                    elif len(formatted_result) > truncation_length:
                        formatted_result = formatted_result[:truncation_length] + "... [Content truncated]"
                        message_parts.append(formatted_result)
                    else:
                        message_parts.append(formatted_result)
            else:
                # Handle non-dict results
                result_str = str(tool_result)
                # For read_entire_file operations, don't truncate
                if is_read_entire_file:
                    message_parts.append(result_str)
                elif len(result_str) > truncation_length:
                    result_str = result_str[:truncation_length] + "... [Content truncated]"
                    message_parts.append(result_str)
                else:
                    message_parts.append(result_str)
            
            # Add separator between tools
            if i < len(tool_results):
                message_parts.append("")  # Empty line for separation
        
        return '\n'.join(message_parts)

    def _should_expect_tool_calls(self, content: str, current_round: int, user_prompt: str) -> bool:
        """
        Determine if we should expect tool calls based on the content and context.
        
        Args:
            content: LLM response content
            current_round: Current conversation round
            user_prompt: Original user prompt
            
        Returns:
            True if tool calls should be expected, False otherwise
        """
        # PRIORITY CHECK: Don't expect tools if TASK_COMPLETED flag is present
        if "TASK_COMPLETED:" in content:
            print(f"🎉 TASK_COMPLETED flag detected in content, no further tool calls expected")
            return False
        
        # Don't expect tools if this is clearly a final response
        final_response_indicators = [
            "task completed", "work is done", "finished", "完成了", "任务完成",
            "here's the summary", "in conclusion", "to summarize", "总结一下",
            "no further action needed", "all set", "done!", "finished!",
            "以上就是", "这样就完成了", "任务执行完毕"
        ]
        
        content_lower = content.lower()
        for indicator in final_response_indicators:
            if indicator in content_lower:
                return False
        
        # STRICT CHECK: For analysis/report tasks, if no tool calls in first round, it's likely hallucination
        analysis_keywords = [
            "analyze", "analysis", "report", "summary", "examine", "inspect",
            "review", "evaluate", "assess", "parse", "investigate",
            "分析", "报告", "总结", "检查", "评估", "解析", "调查"
        ]
        
        user_wants_analysis = any(keyword in user_prompt.lower() for keyword in analysis_keywords)
        if user_wants_analysis and current_round == 1:
            # Check if content contains detailed analysis without tool calls
            analysis_content_indicators = [
                "function", "class", "method", "variable", "import", "def ",
                "code structure", "implementation", "algorithm", "logic",
                "技术栈", "代码结构", "实现", "算法", "逻辑", "函数", "类", "方法"
            ]
            
            has_detailed_analysis = any(indicator in content_lower for indicator in analysis_content_indicators)
            if has_detailed_analysis:
                print(f"⚠️ Analysis content detected without tool calls - likely hallucination")
                return True
        
        # Check if the content mentions tools or actions that typically require tools
        tool_action_keywords = [
            # File operations
            "create file", "read file", "edit file", "modify file", "write to file",
            "check file", "look at file", "examine file", "analyze file",
            "创建文件", "读取文件", "编辑文件", "修改文件", "写入文件", "查看文件",
            
            # Search operations  
            "search for", "find", "look for", "grep", "search in", "查找", "搜索",
            
            # Terminal operations
            "run command", "execute", "install", "npm", "pip", "bash", "shell",
            "运行命令", "执行", "安装",
            
            # Directory operations
            "list directory", "check directory", "explore", "ls", "dir",
            "列出目录", "查看目录", "浏览",
            
            # Analysis operations
            "analyze code", "examine", "investigate", "debug", "trace",
            "分析代码", "检查", "调试",
            
            # Web operations
            "search web", "look up", "find information", "research",
            "网络搜索", "查找信息", "研究"
        ]
        
        for keyword in tool_action_keywords:
            if keyword in content_lower:
                return True
        
        # Check if user prompt suggests tool usage is needed
        user_prompt_lower = user_prompt.lower()
        user_tool_indicators = [
            "create", "build", "implement", "code", "file", "script", "program",
            "search", "find", "analyze", "check", "debug", "fix", "modify",
            "install", "setup", "configure", "run", "execute", "test",
            "write", "generate", "develop", "make", "add", "edit",
            "创建", "构建", "实现", "编程", "文件", "脚本", "程序", 
            "搜索", "查找", "分析", "检查", "调试", "修复", "修改",
            "安装", "设置", "配置", "运行", "执行", "测试",
            "写", "生成", "开发", "制作", "添加", "编辑"
        ]
        
        for indicator in user_tool_indicators:
            if indicator in user_prompt_lower:
                return True
        
        # For first round responses that are very short, likely missing tool calls
        if current_round == 1 and len(content.strip()) < 200:
            # But exclude obvious conversational responses
            conversational_patterns = [
                "hello", "hi", "thanks", "thank you", "sorry", "please", "yes", "no",
                "你好", "谢谢", "对不起", "请", "是的", "不是",
                "i understand", "i see", "got it", "okay", "sure",
                "我明白", "我知道", "好的", "确定", "当然"
            ]
            
            is_conversational = any(pattern in content_lower for pattern in conversational_patterns)
            if not is_conversational:
                return True
        
        return False
    
    def _generate_conflict_warning(self) -> str:
        """
        Generate a warning message for the LLM about tool call and TASK_COMPLETED conflict.
        
        Returns:
            Warning message string to send to the LLM
        """
        warning_parts = []
        
        warning_parts.append("⚠️ **CONFLICT DETECTED - IMPORTANT NOTICE** ⚠️")
        warning_parts.append("")
        warning_parts.append("Your previous response contained both tool calls and a TASK_COMPLETED signal.")
        warning_parts.append("- Use tools first, then in a SEPARATE response, send TASK_COMPLETED if the task is truly finished")

        
        return "\n".join(warning_parts)
    
    def _generate_tool_call_warning(self, content: str) -> str:
        """
        Generate a warning message for the LLM about missing tool calls.
        
        Args:
            content: LLM response content that didn't contain tool calls
            
        Returns:
            Warning message string to send back to the LLM
        """
        warning_parts = []
        
        warning_parts.append("🚨 **CRITICAL ERROR: HALLUCINATION DETECTED** 🚨")
        warning_parts.append("")
        warning_parts.append("Your previous response appears to contain fabricated or assumed information without using tools to gather actual data.")
        warning_parts.append("")
        warning_parts.append("**CRITICAL ISSUES DETECTED:**")
        warning_parts.append("1. You provided analysis or content without reading actual files")
        warning_parts.append("2. You made assumptions about code structure, functions, or file contents")
        warning_parts.append("3. You generated reports based on file names or guesses rather than actual data")
        warning_parts.append("")
        warning_parts.append("**MANDATORY CORRECTION REQUIRED:**")
        warning_parts.append("1. You MUST use tools to gather ACTUAL information before responding")
        warning_parts.append("2. NEVER assume file contents, directory structure, or code functionality")
        warning_parts.append("3. Always read files using read_file tool before making any claims about their contents")
        warning_parts.append("4. Use list_dir to explore directories before making assumptions")
        warning_parts.append("5. Use codebase_search to find specific code patterns")
        warning_parts.append("")
        warning_parts.append("**Correct Tool Call Formats:**")
        warning_parts.append("")
        warning_parts.append("**Format 1 - XML Function Calls (Recommended):**")
        warning_parts.append("```")
        warning_parts.append("<function_calls>")
        warning_parts.append('<invoke name="tool_name">')
        warning_parts.append('<parameter name="param1">value1</parameter>')
        warning_parts.append('<parameter name="param2">value2</parameter>')
        warning_parts.append('</invoke>')
        warning_parts.append("</function_calls>")
        warning_parts.append("```")
        warning_parts.append("")
        warning_parts.append("**Format 2 - Direct XML Tags:**")
        warning_parts.append("```")
        warning_parts.append("<function_calls>")
        warning_parts.append('<invoke name="tool_name">')
        warning_parts.append('<param1>value1</param1>')
        warning_parts.append('<param2>value2</param2>')
        warning_parts.append('</invoke>')
        warning_parts.append("</function_calls>")
        warning_parts.append("```")
        warning_parts.append("")
        warning_parts.append("**Available Tools:**")
        available_tools = list(self.tool_map.keys())
        for tool in available_tools:
            warning_parts.append(f"- {tool}")
        warning_parts.append("")
        warning_parts.append("**Your previous response was:**")
        warning_parts.append(f"```")
        # Truncate very long responses for readability
        history_truncation_length = get_history_truncation_length()
        if len(content) > history_truncation_length:
            warning_parts.append(f"{content[:history_truncation_length]}... [truncated]")
        else:
            warning_parts.append(content)
        warning_parts.append("```")
        warning_parts.append("")
        warning_parts.append("Please respond again with the proper tool calls to complete the task. Do not just describe what you would do - actually make the tool calls!")
        
        return "\n".join(warning_parts)

    def test_api_connection(self) -> bool:
        """
        Testing API connection
        
        Returns:
            bool: True if connection is successful, False otherwise
        """
        print("🔍 Testing API connection...")
        
        try:
            if self.is_claude:
                # Claude API
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=50,
                    messages=[{"role": "user", "content": "Hello, please respond with 'Connection test successful'"}],
                    temperature=0.1
                )
                
                if hasattr(response, 'content') and response.content:
                    content = response.content[0].text if response.content else ""
                    print(f"✅ Claude API connection successful! Response: {content[:50]}...")
                    return True
                else:
                    print("❌ Claude API response format exception")
                    return False
            else:
                # OpenAI API
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": "Hello, please respond with 'Connection test successful'"}],
                    max_tokens=50,
                    temperature=0.1
                )
                
                if response.choices and response.choices[0].message:
                    content = response.choices[0].message.content
                    print(f"✅ OpenAI API connection successful! Response: {content[:50]}...")
                    return True
                else:
                    print("❌ OpenAI API response format exception")
                    return False
                    
        except Exception as e:
            print(f"❌ API connection test failed: {e}")
            print(f"🔧 Please check the following items:")
            print(f"   1. API key validity")
            print(f"   2. Network connection")
            print(f"   3. API service endpoint accessibility")
            print(f"   4. API quota")
            return False

    def ask_user_confirmation(self, message: str, default_yes: bool = True) -> tuple:
        """
        Ask user for confirmation in interactive mode
        
        Args:
            message: Confirmation message to display
            default_yes: Whether to default to 'yes' if user just presses Enter
            
        Returns:
            Tuple of (continue: bool, user_input: str or None)
            - continue: True if user confirms, False otherwise
            - user_input: User's custom input if provided, None otherwise
        """
        if not self.interactive_mode:
            return True, None  # In non-interactive mode, always continue with no user input
        
        try:
            default_hint = "(Y/n/custom)" if default_yes else "(y/N/custom)"
            print(f"\n🤝 {message} {default_hint}")
            print("💡 You can type 'y' for yes, 'n' for no, or provide your own instructions to guide the next step:")
            response = input(">>> ").strip()
            
            if not response:  # Empty response, use default
                return default_yes, None
            
            response_lower = response.lower()
            
            # Check for explicit yes/no responses
            if response_lower in ['y', 'yes', '是', '确定']:
                return True, None
            elif response_lower in ['n', 'no', '否', '取消']:
                return False, None
            else:
                # User provided custom input
                print(f"📝 Custom instruction received: {response}")
                return True, response
            
        except (KeyboardInterrupt, EOFError):
            print("\n❌ User cancelled operation")
            return False, None

    def analyze_debug_logs_completeness(self, logs_dir: str = None) -> Dict[str, Any]:
        """
        分析调试日志的完整性，检查工具调用信息是否完整记录
        
        Args:
            logs_dir: 日志目录路径
            
        Returns:
            包含分析结果的字典
        """
        if logs_dir is None:
            logs_dir = self.llm_logs_dir
        
        print(f"🔍 分析调试日志完整性: {logs_dir}")
        
        analysis_result = {
            "total_log_files": 0,
            "logs_with_tool_calls": 0,
            "logs_with_complete_tool_info": 0,
            "logs_with_tool_results": 0,
            "logs_with_formatted_results": 0,
            "incomplete_logs": [],
            "error_logs": [],
            "summary": {}
        }
        
        try:
            if not os.path.exists(logs_dir):
                return {"error": f"日志目录不存在: {logs_dir}"}
            
            log_files = [f for f in os.listdir(logs_dir) if f.startswith('llm_call_') and f.endswith('.json')]
            analysis_result["total_log_files"] = len(log_files)
            
            for filename in sorted(log_files):
                file_path = os.path.join(logs_dir, filename)
                
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        log_data = json.load(f)
                    
                    call_info = log_data.get("call_info", {})
                    tool_calls_info = log_data.get("tool_calls_info", {})
                    
                    # 检查是否有工具调用
                    has_tool_calls = bool(tool_calls_info.get("parsed_tool_calls"))
                    has_tool_results = bool(tool_calls_info.get("tool_results"))
                    has_formatted_results = bool(tool_calls_info.get("formatted_tool_results"))
                    
                    file_analysis = {
                        "filename": filename,
                        "call_number": call_info.get("call_number"),
                        "round": call_info.get("round"),
                        "has_tool_calls": has_tool_calls,
                        "has_tool_results": has_tool_results,
                        "has_formatted_results": has_formatted_results,
                        "tool_calls_count": call_info.get("tool_calls_count", 0),
                        "tool_results_count": call_info.get("tool_results_count", 0)
                    }
                    
                    if has_tool_calls:
                        analysis_result["logs_with_tool_calls"] += 1
                        
                        # 检查工具调用信息的完整性
                        if has_tool_results and has_formatted_results:
                            analysis_result["logs_with_complete_tool_info"] += 1
                        else:
                            analysis_result["incomplete_logs"].append(file_analysis)
                    
                    if has_tool_results:
                        analysis_result["logs_with_tool_results"] += 1
                    
                    if has_formatted_results:
                        analysis_result["logs_with_formatted_results"] += 1
                    
                    # 检查是否有错误信息
                    if tool_calls_info.get("processing_failed") or tool_calls_info.get("error"):
                        analysis_result["error_logs"].append(file_analysis)
                
                except Exception as e:
                    error_info = {
                        "filename": filename,
                        "error": str(e)
                    }
                    analysis_result["error_logs"].append(error_info)
                    print(f"❌ 读取日志文件失败 {filename}: {e}")
            
            # 生成总结
            total_files = analysis_result["total_log_files"]
            tool_call_files = analysis_result["logs_with_tool_calls"]
            complete_files = analysis_result["logs_with_complete_tool_info"]
            
            analysis_result["summary"] = {
                "completeness_rate": (complete_files / tool_call_files * 100) if tool_call_files > 0 else 0,
                "tool_call_coverage": (tool_call_files / total_files * 100) if total_files > 0 else 0,
                "incomplete_count": len(analysis_result["incomplete_logs"]),
                "error_count": len(analysis_result["error_logs"])
            }
            
            # 打印分析结果
            print(f"📊 调试日志完整性分析结果:")
            print(f"   总日志文件数: {total_files}")
            print(f"   包含工具调用的日志: {tool_call_files}")
            print(f"   工具调用信息完整的日志: {complete_files}")
            print(f"   完整性比率: {analysis_result['summary']['completeness_rate']:.1f}%")
            print(f"   工具调用覆盖率: {analysis_result['summary']['tool_call_coverage']:.1f}%")
            
            if analysis_result["incomplete_logs"]:
                print(f"⚠️ 发现 {len(analysis_result['incomplete_logs'])} 个不完整的工具调用日志")
            
            if analysis_result["error_logs"]:
                print(f"❌ 发现 {len(analysis_result['error_logs'])} 个包含错误的日志")
            
            return analysis_result
            
        except Exception as e:
            print(f"❌ 日志分析失败: {e}")
            return {"error": str(e)}

    def _get_hallucination_detection_config(self, current_round_tools: List[str] = None) -> Dict[str, Any]:
        """
        Get hallucination detection configuration.
        
        Args:
            current_round_tools: List of tool names called in current round
        
        Returns:
            Dictionary with hallucination detection settings
        """
        # Default hallucination triggers that the system uses for formatting
        default_triggers = [
            "## Tool Execution Results:",
            "Tool execution results:",
            "## Tool Results:",
            "**Tool Results:**",
            "**Tool Execution Results:**"
        ]
        
        # Edit file specific triggers - detect when edit_file is called after other tools in same round
        edit_file_triggers = [
            r'<invoke name="edit_file">',
            r'<invoke name="edit_file"',
            r'<invoke name="edit_file',  # Handle incomplete patterns
            r'name="edit_file"',
            r'name="edit_file',         # Handle incomplete patterns
            r'"edit_file"',
            r'"edit_file',              # Handle incomplete patterns
            r'edit_file\(',
            r'edit_file \(',
            r'<function_call>\s*\{\s*"name"\s*:\s*"edit_file"',
            r'<edit_file>',
            r'</edit_file>'
        ]
        
        config = {
            "enabled": True,  # Enable by default to prevent hallucination
            "triggers": default_triggers,
            "buffer_size_multiplier": 2,  # Buffer size = max_trigger_length * multiplier
            "case_sensitive": False,  # Case-insensitive detection for better coverage
            "log_detection": True,  # Log when detection occurs
            "edit_file_triggers": edit_file_triggers,
            "current_round_tools": current_round_tools or []
        }
        
        return config
    
    def _detect_hallucination_in_stream(self, detection_buffer: str, config: Dict[str, Any]) -> tuple:
        """
        Detect hallucination triggers in streaming content.
        
        Args:
            detection_buffer: Current detection buffer content
            config: Hallucination detection configuration
            
        Returns:
            Tuple of (detected: bool, trigger_found: str or None)
        """
        if not config.get("enabled", True):
            return False, None
        
        # First check for standard hallucination triggers
        triggers = config.get("triggers", [])
        case_sensitive = config.get("case_sensitive", False)
        
        # Prepare content for comparison
        content_to_check = detection_buffer if case_sensitive else detection_buffer.lower()
        
        for trigger in triggers:
            trigger_to_check = trigger if case_sensitive else trigger.lower()
            
            if trigger_to_check in content_to_check:
                if config.get("log_detection", True):
                    print(f"🚨 HALLUCINATION DETECTED: Found '{trigger}' in output!")
                return True, trigger
        
        # Check for edit_file hallucination (edit_file called after other tools in same round)
        current_round_tools = config.get("current_round_tools", [])
        edit_file_triggers = config.get("edit_file_triggers", [])
        
        # Check if current buffer contains edit_file call
        import re
        current_buffer_has_edit_file = False
        edit_file_trigger_found = None
        
        for edit_trigger in edit_file_triggers:
            if re.search(edit_trigger, content_to_check, re.IGNORECASE if not case_sensitive else 0):
                current_buffer_has_edit_file = True
                edit_file_trigger_found = edit_trigger
                break
        
        # Only trigger hallucination if:
        # 1. Current buffer contains edit_file call AND
        # 2. There are already other tools called in this round (excluding edit_file itself)
        other_tools = [tool for tool in current_round_tools if tool != 'edit_file']
        if current_buffer_has_edit_file and other_tools and len(other_tools) > 0:
            return True, f"edit_file_after_tools:{edit_file_trigger_found}"
        
        return False, None
    
    def _update_current_round_tools(self, detection_buffer: str, current_round_tools: List[str]) -> None:
        """
        Update current round tools tracking by detecting tool calls in the buffer.
        
        Args:
            detection_buffer: Current detection buffer content
            current_round_tools: List to update with detected tool names
        """
        import re
        
        # Tool call patterns to detect
        tool_patterns = [
            r'<invoke name="([^"]+)">',
            r'<invoke name="([^"]+)"',
            r'"name"\s*:\s*"([^"]+)"',
            r'<function_call>\s*{\s*"name"\s*:\s*"([^"]+)"',
            r'<([a-zA-Z_][a-zA-Z0-9_]*)\s*>',  # Direct XML tags like <edit_file>
        ]
        
        # Common tool names to look for
        known_tools = [
            'read_file', 'edit_file', 'list_dir', 'codebase_search', 'grep_search',
            'run_terminal_cmd', 'web_search', 'file_search', 'delete_file', 'reapply',
            'diff_history', 'tool_help', 'kb_search', 'kb_content', 'kb_body'
        ]
        
        for pattern in tool_patterns:
            matches = re.findall(pattern, detection_buffer, re.IGNORECASE)
            for match in matches:
                tool_name = match.strip()
                # Only add known tools and avoid duplicates
                if tool_name in known_tools and tool_name not in current_round_tools:
                    current_round_tools.append(tool_name)
    
    def _remove_hallucination_from_content(self, content: str, trigger: str, config: Dict[str, Any]) -> str:
        """
        Remove hallucination content from the generated text.
        
        Args:
            content: Generated content
            trigger: The trigger that was detected
            config: Detection configuration
            
        Returns:
            Cleaned content with hallucination removed
        """
        case_sensitive = config.get("case_sensitive", False)
        
        # Check if this is an edit_file hallucination
        is_edit_file_hallucination = trigger.startswith("edit_file_after_tools:")
        
        if is_edit_file_hallucination:
            # For edit_file hallucination, we need to handle XML completion differently
            return self._handle_edit_file_hallucination(content, trigger, config)
        
        # Handle standard hallucination triggers
        if case_sensitive:
            trigger_pos = content.rfind(trigger)
        else:
            # Case-insensitive search
            content_lower = content.lower()
            trigger_lower = trigger.lower()
            trigger_pos_lower = content_lower.rfind(trigger_lower)
            
            if trigger_pos_lower != -1:
                # Find the actual position in the original content
                trigger_pos = trigger_pos_lower
            else:
                trigger_pos = -1
        
        if trigger_pos != -1:
            # Remove everything from the trigger onwards
            cleaned_content = content[:trigger_pos].rstrip()
            
            if config.get("log_detection", True):
                removed_length = len(content) - len(cleaned_content)
                print(f"🧹 Removed {removed_length} characters of hallucination content")
                print(f"📝 Clean content length: {len(cleaned_content)} characters")
            
            return cleaned_content
        
        return content
    
    def _handle_edit_file_hallucination(self, content: str, trigger: str, config: Dict[str, Any]) -> str:
        """
        Handle edit_file hallucination by completing the XML structure with a dummy call.
        
        Args:
            content: Generated content
            trigger: The edit_file trigger that was detected
            config: Detection configuration
            
        Returns:
            Content with completed dummy edit_file XML structure
        """
        import re
        
        # Extract the actual trigger pattern from the trigger string
        actual_trigger = trigger.split(":", 1)[1] if ":" in trigger else trigger
        
        # Find where the edit_file call starts
        case_sensitive = config.get("case_sensitive", False)
        
        # Try to find the edit_file trigger position
        trigger_pos = -1
        
        # Search for various edit_file patterns, including incomplete ones
        edit_file_patterns = [
            r'<invoke name="edit_file">',
            r'<invoke name="edit_file"',
            r'<invoke name="edit_file',  # Handle incomplete patterns
            r'name="edit_file"',
            r'name="edit_file',         # Handle incomplete patterns
            r'"edit_file"',
            r'"edit_file',              # Handle incomplete patterns
            r'<function_call>\s*\{\s*"name"\s*:\s*"edit_file"',
            r'<edit_file>'
        ]
        
        for pattern in edit_file_patterns:
            match = re.search(pattern, content, re.IGNORECASE if not case_sensitive else 0)
            if match:
                trigger_pos = match.start()
                break
        
        if trigger_pos != -1:
            # Remove everything from the edit_file call onwards
            cleaned_content = content[:trigger_pos].rstrip()
            
            # Check if we need to complete an existing edit_file call
            if "<function_calls>" in cleaned_content and "</function_calls>" not in cleaned_content[cleaned_content.rfind("<function_calls>"):]:
                # We're inside an existing function_calls block, complete the edit_file call
                dummy_edit_file = '''
<parameter name="target_file">dummy_file_placeholder.txt</parameter>
</invoke>
</function_calls>'''
            else:
                # Fallback: just add a comment
                dummy_edit_file = '''
<!-- Hallucination detected: edit_file call was prevented -->'''
            
            # Combine cleaned content with dummy structure
            final_content = cleaned_content + dummy_edit_file
            

            
            return final_content
        else:
            # If we can't find the trigger position, just remove from the end
            # This is a fallback case

            
            # Add dummy structure at the end that won't be parsed as a tool call
            dummy_edit_file = '''
<!-- Hallucination detected: edit_file call was prevented -->
<function_calls_disabled>
<invoke name="edit_file">
<parameter name="target_file">dummy_file_placeholder.txt</parameter>
<parameter name="instructions">Dummy edit call - hallucination detected and prevented</parameter>
<parameter name="code_edit"># This is a dummy edit call created by hallucination detection
# The actual edit_file call was prevented to avoid hallucination
# No actual file operations will be performed</parameter>
</invoke>
</function_calls_disabled>'''
            
            return content.rstrip() + dummy_edit_file

    def test_hallucination_detection(self, test_content: str = None) -> Dict[str, Any]:
        """
        Test the hallucination detection functionality.
        
        Args:
            test_content: Optional test content to check. If None, uses default test cases.
            
        Returns:
            Dictionary with test results
        """
        print("🧪 Testing hallucination detection functionality...")
        
        # Get configuration
        config = self._get_hallucination_detection_config()
        
        # Default test cases if no content provided
        if test_content is None:
            test_cases = [
                "This is normal content without any triggers.",
                "Let me analyze the code. ## Tool Execution Results: Here are the results...",
                "The analysis shows that **Tool Results:** indicate success.",
                "Normal content here. Tool execution results: Everything looks good.",
                "## Tool Results:\n- File created successfully\n- Code analyzed",
                "**Tool Execution Results:**\nThe following tools were executed:",
                "This content has no triggers at all."
            ]
        else:
            test_cases = [test_content]
        
        results = {
            "config": config,
            "test_results": [],
            "detection_summary": {
                "total_tests": len(test_cases),
                "detections": 0,
                "false_positives": 0,
                "false_negatives": 0
            }
        }
        
        print(f"📋 Testing {len(test_cases)} cases...")
        
        for i, content in enumerate(test_cases, 1):
            print(f"\n🔍 Test case {i}:")
            print(f"Content preview: {content[:80]}{'...' if len(content) > 80 else ''}")
            
            # Test detection
            detected, trigger_found = self._detect_hallucination_in_stream(content, config)
            
            # Test content cleaning if detected
            cleaned_content = None
            if detected:
                cleaned_content = self._remove_hallucination_from_content(content, trigger_found, config)
                results["detection_summary"]["detections"] += 1
            
            test_result = {
                "case_number": i,
                "original_content": content,
                "detected": detected,
                "trigger_found": trigger_found,
                "cleaned_content": cleaned_content,
                "original_length": len(content),
                "cleaned_length": len(cleaned_content) if cleaned_content else len(content)
            }
            
            results["test_results"].append(test_result)
            
            # Print result
            if detected:
                print(f"   ✅ DETECTED: '{trigger_found}'")
                print(f"   🧹 Content cleaned: {len(content)} → {len(cleaned_content)} chars")
            else:
                print(f"   ✅ No hallucination detected")
        
        # Print summary
        print(f"\n📊 Test Summary:")
        print(f"   Total tests: {results['detection_summary']['total_tests']}")
        print(f"   Detections: {results['detection_summary']['detections']}")
        print(f"   Detection rate: {results['detection_summary']['detections'] / results['detection_summary']['total_tests'] * 100:.1f}%")
        
        return results

    def test_edit_file_hallucination_detection(self) -> Dict[str, Any]:
        """
        Test the edit_file hallucination detection functionality.
        
        Returns:
            Dictionary with test results
        """
        print("🧪 Testing edit_file hallucination detection functionality...")
        
        # Test cases for edit_file hallucination
        test_cases = [
            {
                "name": "Normal edit_file call (no other tools)",
                "current_round_tools": [],
                "content": '<invoke name="edit_file"><parameter name="target_file">test.py</parameter></invoke>',
                "should_detect": False
            },
            {
                "name": "Edit_file after read_file (should detect)",
                "current_round_tools": ["read_file"],
                "content": '<invoke name="edit_file"><parameter name="target_file">test.py</parameter></invoke>',
                "should_detect": True
            },
            {
                "name": "Edit_file after multiple tools (should detect)",
                "current_round_tools": ["read_file", "codebase_search"],
                "content": '<invoke name="edit_file"><parameter name="target_file">test.py</parameter></invoke>',
                "should_detect": True
            },
            {
                "name": "Multiple tools without edit_file (should not detect)",
                "current_round_tools": ["read_file", "codebase_search"],
                "content": '<invoke name="list_dir"><parameter name="relative_workspace_path">.</parameter></invoke>',
                "should_detect": False
            },
            {
                "name": "Edit_file with JSON format after other tools (should detect)",
                "current_round_tools": ["web_search"],
                "content": '{"name": "edit_file", "parameters": {"target_file": "test.py"}}',
                "should_detect": True
            }
        ]
        
        results = {
            "test_results": [],
            "summary": {
                "total_tests": len(test_cases),
                "correct_detections": 0,
                "false_positives": 0,
                "false_negatives": 0
            }
        }
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"\n🔍 Test case {i}: {test_case['name']}")
            print(f"Current round tools: {test_case['current_round_tools']}")
            print(f"Content: {test_case['content'][:100]}{'...' if len(test_case['content']) > 100 else ''}")
            
            # Get configuration with current round tools
            config = self._get_hallucination_detection_config(test_case['current_round_tools'])
            
            # Test detection
            detected, trigger_found = self._detect_hallucination_in_stream(test_case['content'], config)
            
            # Check if detection result matches expectation
            is_correct = (detected == test_case['should_detect'])
            
            if is_correct:
                results["summary"]["correct_detections"] += 1
                print(f"   ✅ CORRECT: {'Detected' if detected else 'Not detected'} as expected")
            else:
                if detected and not test_case['should_detect']:
                    results["summary"]["false_positives"] += 1
                    print(f"   ❌ FALSE POSITIVE: Detected when shouldn't have")
                elif not detected and test_case['should_detect']:
                    results["summary"]["false_negatives"] += 1
                    print(f"   ❌ FALSE NEGATIVE: Should have detected but didn't")
            
            if detected:
                print(f"   🚨 Trigger found: {trigger_found}")
            
            test_result = {
                "case_number": i,
                "name": test_case['name'],
                "current_round_tools": test_case['current_round_tools'],
                "content": test_case['content'],
                "should_detect": test_case['should_detect'],
                "detected": detected,
                "trigger_found": trigger_found,
                "is_correct": is_correct
            }
            
            results["test_results"].append(test_result)
        
        # Print summary
        summary = results["summary"]
        accuracy = (summary["correct_detections"] / summary["total_tests"] * 100) if summary["total_tests"] > 0 else 0
        
        print(f"\n📊 Edit_file Hallucination Detection Test Summary:")
        print(f"   Total tests: {summary['total_tests']}")
        print(f"   Correct detections: {summary['correct_detections']}")
        print(f"   False positives: {summary['false_positives']}")
        print(f"   False negatives: {summary['false_negatives']}")
        print(f"   Accuracy: {accuracy:.1f}%")
        
        return results

    def _format_search_result_for_terminal(self, data: Dict[str, Any], tool_name: str) -> str:
        """
        Format search results (codebase_search and web_search) for simplified terminal display.
        Only shows brief summary with limited characters to reduce terminal clutter.
        
        Args:
            data: Dictionary result from search tools
            tool_name: Name of the tool that generated this result
            
        Returns:
            Simplified formatted text string for terminal display
        """
        if not isinstance(data, dict):
            return str(data)
        
        lines = []
        
        # Handle error cases first
        if 'error' in data:
            return f"❌ {tool_name} failed: {data['error']}"
        
        # Handle codebase_search results
        if tool_name == 'codebase_search':
            query = data.get('query', 'unknown')
            results = data.get('results', [])
            total_results = len(results)
            
            lines.append(f"🔍 Code search for '{query}': Found {total_results} results")
            
            # Show only first 3 results with very brief info
            for i, result in enumerate(results[:3], 1):
                if isinstance(result, dict):
                    file_path = result.get('file', 'unknown')
                    start_line = result.get('start_line', '')
                    # Show only first 100 characters of snippet
                    snippet = result.get('snippet', '')[:100].replace('\n', ' ').strip()
                    if len(result.get('snippet', '')) > 100:
                        snippet += "..."
                    
                    lines.append(f"  {i}. {file_path}:{start_line} - {snippet}")
            
            if total_results > 3:
                lines.append(f"  ... and {total_results - 3} more results")
            
            # Add repository stats briefly
            stats = data.get('repository_stats', {})
            if stats:
                lines.append(f"📊 Repository: {stats.get('total_files', 0)} files, {stats.get('total_segments', 0)} segments")
        
        # Handle web_search results
        elif tool_name == 'web_search':
            search_term = data.get('search_term', 'unknown')
            results = data.get('results', [])
            total_results = len(results)
            
            lines.append(f"🌐 Web search for '{search_term}': Found {total_results} results")
            
            # Show only first 3 results with very brief info
            for i, result in enumerate(results[:3], 1):
                if isinstance(result, dict):
                    title = result.get('title', 'No Title')[:80]  # Limit title length
                    if len(result.get('title', '')) > 80:
                        title += "..."
                    
                    # Show brief snippet or content summary
                    content_preview = ""
                    if result.get('snippet'):
                        content_preview = result['snippet'][:100].replace('\n', ' ').strip()
                    elif result.get('content_summary'):
                        content_preview = result['content_summary'][:100].replace('\n', ' ').strip()
                    elif result.get('content'):
                        content_preview = result['content'][:100].replace('\n', ' ').strip()
                    
                    if content_preview and len(content_preview) >= 100:
                        content_preview += "..."
                    
                    lines.append(f"  {i}. {title}")
                    if content_preview:
                        lines.append(f"     {content_preview}")
            
            if total_results > 3:
                lines.append(f"  ... and {total_results - 3} more results")
            
            # Add metadata briefly
            if data.get('content_fetched'):
                lines.append(f"📄 Content fetched: {data['content_fetched']}")
        
        # For other tools or unrecognized search results, fall back to original formatting
        else:
            return self._format_dict_as_text(data)
        
        return '\n'.join(lines)


def test_search_result_formatting():
    """Test the new search result formatting functionality"""
    print("=== Testing Search Result Formatting ===")
    
    # Create a mock ToolExecutor instance for testing
    executor = ToolExecutor(
        api_key="test_key",
        model="test_model", 
        api_base="test_base"
    )
    
    # Test codebase_search result formatting
    print("\n1. Testing codebase_search formatting:")
    codebase_result = {
        'query': 'function definition',
        'results': [
            {
                'file': 'src/main.py',
                'snippet': 'def process_data(input_data):\n    """Process input data and return results"""\n    # Implementation here\n    return processed_data',
                'start_line': 15,
                'end_line': 20,
                'score': 0.95,
                'search_type': 'hybrid'
            },
            {
                'file': 'utils/helpers.py', 
                'snippet': 'def validate_input(data):\n    if not data:\n        raise ValueError("Data cannot be empty")\n    return True',
                'start_line': 5,
                'end_line': 8,
                'score': 0.87,
                'search_type': 'vector'
            },
            {
                'file': 'config/settings.py',
                'snippet': 'def load_config():\n    with open("config.json") as f:\n        return json.load(f)',
                'start_line': 12,
                'end_line': 15,
                'score': 0.75,
                'search_type': 'keyword'
            }
        ],
        'total_results': 3,
        'repository_stats': {
            'total_files': 25,
            'total_segments': 150
        },
        'search_method': 'hybrid_vector_keyword'
    }
    
    formatted_codebase = executor._format_search_result_for_terminal(codebase_result, 'codebase_search')
    print("Formatted codebase_search result:")
    print(formatted_codebase)
    
    # Test web_search result formatting
    print("\n2. Testing web_search formatting:")
    web_result = {
        'search_term': 'Python best practices',
        'results': [
            {
                'title': 'Python Best Practices: A Complete Guide for Developers',
                'snippet': 'Learn the essential Python best practices that every developer should know. This comprehensive guide covers coding standards, performance optimization, and more.',
                'content': 'Python is a powerful programming language that offers many features for developers. Following best practices ensures your code is maintainable, readable, and efficient...',
                'source': 'developer.com'
            },
            {
                'title': 'Top 10 Python Coding Standards You Should Follow',
                'snippet': 'Discover the most important Python coding standards and conventions that will make your code more professional and easier to maintain.',
                'content': 'Writing clean, readable Python code is essential for any developer. Here are the top 10 coding standards you should follow...',
                'source': 'python.org'
            }
        ],
        'timestamp': '2024-01-15T10:30:00',
        'content_fetched': 2,
        'total_results': 2
    }
    
    formatted_web = executor._format_search_result_for_terminal(web_result, 'web_search')
    print("Formatted web_search result:")
    print(formatted_web)
    
    # Test with many results to show truncation
    print("\n3. Testing with many results (truncation):")
    many_results = codebase_result.copy()
    many_results['results'] = codebase_result['results'] * 3  # 9 results total
    
    formatted_many = executor._format_search_result_for_terminal(many_results, 'codebase_search')
    print("Formatted result with many items:")
    print(formatted_many)
    
    print("\n=== Test Complete ===")


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description='Execute a subtask using LLM with tools')
    parser.add_argument('prompt', nargs='?', help='The prompt for the subtask')
    parser.add_argument('--api-key', '-k', help='API key for the LLM service')
    parser.add_argument('--model', '-m', default="Qwen/Qwen3-30B-A3B", help='Model to use')
    parser.add_argument('--system-prompt', '-s', default="prompts.txt", help='System prompt file')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode with detailed logging')
    parser.add_argument('--logs-dir', default="logs", help='Directory for saving debug logs')
    parser.add_argument('--workspace-dir', help='Working directory for code files and project output')
    parser.add_argument('--streaming', action='store_true', help='Enable streaming output mode')
    parser.add_argument('--no-streaming', action='store_true', help='Disable streaming output mode (force batch)')
    parser.add_argument('--test-edit-hallucination', action='store_true', help='Test edit_file hallucination detection')
    
    args = parser.parse_args()
    
    # Handle streaming configuration
    streaming = None
    if args.streaming and args.no_streaming:
        print("Warning: Both --streaming and --no-streaming specified, using config.txt default")
    elif args.streaming:
        streaming = True
    elif args.no_streaming:
        streaming = False
    # If neither specified, streaming=None will use config.txt value
    
    # Create executor
    executor = ToolExecutor(
        api_key=args.api_key, 
        model=args.model,
        workspace_dir=args.workspace_dir,
        debug_mode=args.debug,
        logs_dir=args.logs_dir,
        streaming=streaming
    )
    
    # Handle test mode
    if args.test_edit_hallucination:
        print("🧪 Running edit_file hallucination detection tests...")
        test_results = executor.test_edit_file_hallucination_detection()
        return
    
    # Check if prompt is provided for normal execution
    if not args.prompt:
        parser.error("prompt is required unless using --test-edit-hallucination")
    
    # Execute subtask
    result = executor.execute_subtask(args.prompt, args.system_prompt)
    
    print(result)


if __name__ == "__main__":
    main()
