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
"""

import os
import json
import argparse
import datetime
import threading
import logging
import time
import warnings

# Suppress asyncio warnings that occur during FastMCP cleanup
warnings.filterwarnings("ignore", message=".*Event loop is closed.*")
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*coroutine.*was never awaited.*")

from typing import Dict, Any, List, Optional, Union, Tuple
from openai import OpenAI
from src.tools.print_system import *
from src.tools.agent_context import get_current_agent_id
from src.tools.debug_system import track_operation, finish_operation
from src.tools.cli_mcp_wrapper import get_cli_mcp_wrapper, initialize_cli_mcp_wrapper, safe_cleanup_cli_mcp_wrapper
from src.config_loader import *
from src.tools.message_system import get_message_router
from src.multi_round_executor.task_checker import TaskChecker
from src.api_callers import (
    call_claude_with_chat_based_tools_streaming,
    call_openai_with_chat_based_tools_streaming,
    call_claude_with_chat_based_tools_non_streaming,
    call_openai_with_chat_based_tools_non_streaming,
    call_openai_with_standard_tools,
    call_claude_with_standard_tools,
)

# Initialize logger
logger = logging.getLogger(__name__)

# Note: JSON parsing utilities now imported below with other utils

# Import get_info utilities
from utils.get_info import (
    get_system_environment_info,
    get_workspace_info
)

# Import parse utilities
from utils.parse import (
    generate_tools_prompt_from_json,
    generate_tools_prompt_from_xml,
    parse_tool_calls_from_json,
    parse_tool_calls_from_xml,
    parse_python_function_calls,
)

# Note: test_api_connection imported dynamically to avoid circular imports

# Check if the API base uses Anthropic format
def is_anthropic_api(api_base: str) -> bool:
    """Check if the API base URL uses Anthropic format"""
    return api_base.lower().endswith('/anthropic') if api_base else False


# Backward compatibility function (deprecated)
def is_claude_model(model: str) -> bool:
    """
    Deprecated: Check if the model name is a Claude model
    This function is kept for backward compatibility only.
    Use is_anthropic_api(api_base) instead to check API format.
    """
    return model.lower().startswith('claude')


# Dynamically import Anthropic
def get_anthropic_client():
    """Dynamically import and return Anthropic client class"""
    try:
        from anthropic import Anthropic
        return Anthropic
    except ImportError:
        print_current("Anthropic library not installed, please run: pip install anthropic")
        raise ImportError("Anthropic library not installed")

class ToolExecutor:
    def __init__(self, api_key: Optional[str] = None,
                 model: Optional[str] = None,
                 api_base: Optional[str] = None,
                 workspace_dir: Optional[str] = None,
                 debug_mode: bool = False,
                 logs_dir: str = "logs",
                 session_timestamp: Optional[str] = None,
                 streaming: Optional[bool] = None,
                 interactive_mode: bool = False,
                 MCP_config_file: Optional[str] = None,
                 prompts_folder: Optional[str] = None,
                 user_id: Optional[str] = None,
                 subtask_loops: Optional[int] = None,
                 plan_mode: bool = False,
                 enable_thinking: Optional[bool] = None):
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
            MCP_config_file: Custom MCP configuration file path (optional, defaults to 'config/mcp_servers.json')
            prompts_folder: Custom prompts folder path (optional, defaults to 'prompts')
            user_id: User ID for MCP knowledge base tools
            subtask_loops: Number of subtask loops (-1 for infinite loop)
            enable_thinking: Whether to enable thinking mode (None to use config.txt)
        """
        # Load API key from config/config.txt if not provided
        if api_key is None:
            api_key = get_api_key()
            if api_key is None:
                raise ValueError("API key not found. Please provide api_key parameter or set it in config/config.txt")
        self.api_key = api_key
        
        # Load model from config/config.txt if not provided
        if model is None:
            model = get_model()
            if model is None:
                raise ValueError("Model not found. Please provide model parameter or set it in config/config.txt")
        self.model = model
        
        # Load API base from config/config.txt if not provided
        if api_base is None:
            api_base = get_api_base()
            if api_base is None:
                raise ValueError("API base URL not found. Please provide api_base parameter or set it in config/config.txt")
        
        # Load streaming configuration from config/config.txt
        if streaming is None:
            streaming = get_streaming()
        self.streaming = streaming
        
        # Load language configuration from config/config.txt
        self.language = get_language()
        
        self.summary_max_length = get_summary_max_length()
        self.summary_trigger_length = get_summary_trigger_length()
        
        # Store subtask loops information for infinite loop mode detection
        self.subtask_loops = subtask_loops
        
        # Store plan mode flag
        self.plan_mode = plan_mode
        
        # Load simplified search output configuration from config/config.txt
        self.simplified_search_output = get_simplified_search_output()
        
        # Load multi-agent mode configuration from config/config.txt
        self.multi_agent = get_multi_agent()
        
        self.workspace_dir = workspace_dir or os.getcwd()
        self.debug_mode = debug_mode
        self.logs_dir = logs_dir
        self.session_timestamp = session_timestamp
        self.interactive_mode = interactive_mode
        
        # Store custom file paths
        self.MCP_config_file = MCP_config_file or "config/mcp_servers.json"
        self.prompts_folder = prompts_folder or "prompts"
        
        # Store user ID for MCP knowledge base tools
        self.user_id = user_id
        
        # Set api_base first
        self.api_base = api_base
        
        # Check if using Anthropic API based on api_base
        self.is_claude = is_anthropic_api(self.api_base)
        
        # Load tool calling format configuration from config/config.txt
        # True = standard tool calling, False = chat-based tool calling
        tool_calling_format = get_tool_calling_format()
        
        # Set use_chat_based_tools based on configuration
        # Note: use_chat_based_tools is the inverse of tool_calling_format
        self.use_chat_based_tools = not tool_calling_format
        
        # Load thinking support configuration from config/config.txt
        # True = enable thinking, False = disable thinking
        # Parameter overrides config.txt value if provided
        if enable_thinking is not None:
            self.enable_thinking = enable_thinking
        else:
            self.enable_thinking = get_enable_thinking()
        
        # Load LLM output control parameters from config/config.txt
        self.temperature = get_temperature()
        self.top_p = get_top_p()
        
        # Load tool call parsing format configuration from config/config.txt
        # "json" or "xml" - determines which format to try first when parsing tool calls
        self.tool_call_parse_format = get_tool_call_parse_format()
        
        # Print system is ready to use
        
        # Display tool calling method
        if self.use_chat_based_tools:
            print_system(f"🔧 Tool calling method: Chat-based (tools described in messages)")
        else:
            print_system(f"🔧 Tool calling method: Standard API tool calling")
        
        # History summarization cache
        self.history_summary_cache = {}
        self.last_summarized_history_length = 0
        
        # Long-term memory update control
        self.memory_update_counter = 0
        self.memory_update_interval = 10
        
        # Tool definitions cache to avoid repeated loading
        self._tool_definitions_cache = None
        self._tool_definitions_cache_timestamp = None
        
        # print_system(f"🤖 LLM Configuration:")  # Commented out to reduce terminal noise
        # print_system(f"   Model: {self.model}")  # Commented out to reduce terminal noise
        # print_system(f"   API Base: {self.api_base}")  # Commented out to reduce terminal noise
        # print_system(f"   API Key: {self.api_key[:20]}...{self.api_key[-10:]}")  # Commented out to reduce terminal noise
        # print_system(f"   Workspace: {self.workspace_dir}")  # Commented out to reduce terminal noise
        # print_system(f"   Language: {'中文' if self.language == 'zh' else 'English'} ({self.language})")  # Commented out to reduce terminal noise
        # print_system(f"   Streaming: {'✅ Enabled' if self.streaming else '❌ Disabled (Batch mode)'}")  # Commented out to reduce terminal noise
        # print_system(f"   Cache Optimization: ✅ Enabled (All rounds use combined prompts for maximum cache hits)")  # Commented out to reduce terminal noise
        # print_system(f"   History Summarization: {'✅ Enabled' if self.summary_history else '❌ Disabled'} (Trigger: {self.summary_trigger_length} chars, Max: {self.summary_max_length} chars)")  # Commented out to reduce terminal noise
        # print_system(f"   Simplified Search Output: {'✅ Enabled' if self.simplified_search_output else '❌ Disabled'} (Affects workspace_search and web_search terminal display)")  # Commented out to reduce terminal noise
        # if debug_mode:
        #     print_system(f"   Debug Mode: Enabled (Log directory: {logs_dir})")  # Commented out to reduce terminal noise
        
        # Set up LLM client
        self._setup_llm_client()
        
        # Initialize tools with LLM configuration for web search filtering
        from tools import Tools
        
        # Get the parent directory of workspace (typically the output directory)
        out_dir = os.path.dirname(self.workspace_dir) if self.workspace_dir else os.getcwd()
        
        # Store the project root directory for image path processing
        self.project_root_dir = out_dir
        
        self.tools = Tools(
            workspace_root=self.workspace_dir,
            llm_api_key=self.api_key,
            llm_model=self.model,
            llm_api_base=self.api_base,
            enable_llm_filtering=False,  # Disable LLM filtering by default for faster responses
            enable_summary=get_web_search_summary(),  # Load web search summary setting from config
            out_dir=out_dir,
            user_id=self.user_id
        )
        
        # Initialize long-term memory system
        try:
            # Check if long-term memory is enabled via environment variable
            if os.environ.get('AGIBOT_LONG_TERM_MEMORY', '').lower() in ('false', '0', 'no', 'off'):
                print_current("⚠️ Long-term memory is disabled via environment variable AGIBOT_LONG_TERM_MEMORY")
                self.long_term_memory = None
            else:
                from tools.long_term_memory import LongTermMemoryTools
                # Long-term memory is now stored in the project root directory
                # Configuration files will be automatically loaded from the project root directory
                self.long_term_memory = LongTermMemoryTools(
                    workspace_root=self.workspace_dir  # For compatibility only; actual storage is in the project root directory
                )
                #print_current("✅ Long-term memory system initialized successfully (global shared storage)")
        except ImportError as e:
            print_current(f"⚠️ Long-term memory module import failed: {e}")
            self.long_term_memory = None
        except Exception as e:
            print_current(f"⚠️ Long-term memory system initialization failed: {e}")
            self.long_term_memory = None
        
        # Initialize multi-agent tools directly if enabled
        if self.multi_agent:
            from tools.multiagents import MultiAgentTools
            self.multi_agent_tools = MultiAgentTools(self.workspace_dir, debug_mode=self.debug_mode)
        else:
            self.multi_agent_tools = None
        
        # Initialize enhanced history compressor
        try:
            from tools.enhanced_history_compressor import EnhancedHistoryCompressor
            # EnhancedHistoryCompressor will automatically load trigger_length from config if None
            self.simple_compressor = EnhancedHistoryCompressor(
                trigger_length=None,  # Will be loaded from config automatically
                keep_recent_rounds=2  # Keep last 2 rounds uncompressed
            )
        except ImportError as e:
            print_system(f"⚠️ Failed to import EnhancedHistoryCompressor: {e}, enhanced compression disabled")
            self.simple_compressor = None
        
        # Initialize history compression tools
        try:
            from tools.history_compression_tools import HistoryCompressionTools
            self.history_compression_tools = HistoryCompressionTools(tool_executor=self)
        except ImportError as e:
            print_system(f"⚠️ Failed to import HistoryCompressionTools: {e}")
            self.history_compression_tools = None
        
        # Initialize planning tools for dynamic tool loading
        try:
            from tools.planning_tools import PlanningTools
            self.planning_tools = PlanningTools(workspace_root=self.workspace_dir)
        except ImportError as e:
            print_system(f"⚠️ Failed to import PlanningTools: {e}")
            self.planning_tools = None
        except Exception as e:
            print_system(f"⚠️ Failed to initialize PlanningTools: {e}")
            self.planning_tools = None
        
        # Store current task history reference for history compression tool
        self._current_task_history = None
        
        # Helper function for disabled multi-agent tools
        def _multi_agent_disabled_error(*args, **kwargs):
            return {"status": "error", "message": "Multi-agent functionality is disabled. Enable it in config/config.txt by setting multi_agent=True"}
        
        # Map of tool names to their implementation methods
        self.tool_map = {
            "workspace_search": self.tools.workspace_search,
            "read_file": self.tools.read_file,
            "run_terminal_cmd": self.tools.run_terminal_cmd,
            "run_claude": self.tools.run_claude,
            "list_dir": self.tools.list_dir,
            "grep_search": self.tools.grep_search,
            "edit_file": self.tools.edit_file,
            "file_search": self.tools.file_search,
            "web_search": self.tools.web_search,
            "search_img": self.tools.search_img, 
            "tool_help": self.enhanced_tool_help,
            "fetch_webpage_content": self.tools.fetch_webpage_content,
            "get_background_update_status": self.tools.get_background_update_status,
            "talk_to_user": self.tools.talk_to_user,
            "idle": self.tools.idle,
            "get_sensor_data": self.tools.get_sensor_data,
            "read_img": self.tools.read_img,
            "merge_file": self.tools.merge_file,
            "parse_doc_to_md": self.tools.parse_doc_to_md,
            "convert_docs_to_markdown": self.tools.convert_docs_to_markdown,
        }
        
        # Add history compression tool if available
        if self.history_compression_tools:
            self.tool_map["compress_history"] = self.history_compression_tools.compress_history
        
        # Add planning tool if available
        if self.planning_tools:
            self.tool_map["plan_tools"] = self.planning_tools.plan_tools
        
        # Add long-term memory tools if available
        if self.long_term_memory:
            self.tool_map.update({
                "recall_memories": self.long_term_memory.recall_memories,
                "recall_memories_by_time": self.long_term_memory.recall_memories_by_time,
                "get_memory_summary": self.long_term_memory.get_memory_summary,
            })
            print_system("🧠 Long-term memory tools registered")
        else:
            # Add error handlers for disabled memory tools
            def _memory_disabled_error(*args, **kwargs):
                return {"status": "error", "message": "Long-term memory feature not enabled or initialization failed"}
            
            self.tool_map.update({
                "recall_memories": _memory_disabled_error,
                "recall_memories_by_time": _memory_disabled_error,
                "get_memory_summary": _memory_disabled_error,
            })
        
        # Initialize skill tools if long-term memory is enabled
        self.skill_tools = None
        if self.long_term_memory:
            try:
                from src.skill_evolve.skill_tools import SkillTools
                self.skill_tools = SkillTools(
                    workspace_root=self.workspace_dir,
                    user_id=self.user_id
                )
                # Register skill tools
                self.tool_map.update({
                    "query_skill": self.skill_tools.query_skill,
                    "rate_skill": self.skill_tools.rate_skill,
                    "edit_skill": self.skill_tools.edit_skill,
                    "delete_skill": self.skill_tools.delete_skill,
                    "copy_skill_files": self.skill_tools.copy_skill_files,
                })
                print_system("🎯 Skill tools registered")
            except ImportError as e:
                print_current(f"⚠️ Skill tools module import failed: {e}")
                self.skill_tools = None
            except Exception as e:
                print_current(f"⚠️ Skill tools initialization failed: {e}")
                self.skill_tools = None
        
        # Add error handlers for disabled skill tools
        if not self.skill_tools:
            def _skill_disabled_error(*args, **kwargs):
                return {"status": "error", "message": "Skill tools are only available when long-term memory is enabled"}
            
            self.tool_map.update({
                "query_skill": _skill_disabled_error,
                "rate_skill": _skill_disabled_error,
                "edit_skill": _skill_disabled_error,
                "delete_skill": _skill_disabled_error,
                "copy_skill_files": _skill_disabled_error,
            })
        
        # Add multi-agent tools if enabled, otherwise add error handlers
        if self.multi_agent_tools:
            self.tool_map.update({
                "spawn_agent": self.multi_agent_tools.spawn_agent,
                "send_P2P_message": self.multi_agent_tools.send_P2P_message,
                "read_received_messages": self.multi_agent_tools.read_received_messages,
                "send_status_update_to_manager": self.multi_agent_tools.send_status_update_to_manager,
                "send_broadcast_message": self.multi_agent_tools.send_broadcast_message,
                "get_agent_session_info": self.multi_agent_tools.get_agent_session_info,
                "terminate_agent": self.multi_agent_tools.terminate_agent
            })
        else:
            # Add error handlers for disabled multi-agent tools
            self.tool_map.update({
                "spawn_agent": _multi_agent_disabled_error,
                "send_P2P_message": _multi_agent_disabled_error,
                "read_received_messages": _multi_agent_disabled_error,
                "send_status_update_to_manager": _multi_agent_disabled_error,
                "send_broadcast_message": _multi_agent_disabled_error,
                "get_agent_session_info": _multi_agent_disabled_error,
                "terminate_agent": _multi_agent_disabled_error,
            })
        
        # Initialize custom tool
        try:
            from tools.custom_tool import CustomTool
            self.custom_tool = CustomTool(workspace_root=self.workspace_dir)
            # Register custom command tool
            self.tool_map["custom_command"] = self.custom_tool.execute_command
            print_system("🎮 Custom tool registered")
        except ImportError as e:
            print_current(f"⚠️ Custom tool module import failed: {e}")
            self.custom_tool = None
            # Add error handler for disabled custom tool
            def _custom_tool_disabled_error(*args, **kwargs):
                return {"status": "error", "message": "Custom tool not available"}
            self.tool_map["custom_command"] = _custom_tool_disabled_error
        except Exception as e:
            print_current(f"⚠️ Custom tool initialization failed: {e}")
            self.custom_tool = None
            def _custom_tool_disabled_error(*args, **kwargs):
                return {"status": "error", "message": f"Custom tool initialization failed: {str(e)}"}
            self.tool_map["custom_command"] = _custom_tool_disabled_error
        

        # Initialize MCP clients
        self.cli_mcp_client = get_cli_mcp_wrapper(self.MCP_config_file)
        self.cli_mcp_initialized = False
        
        import asyncio
        try:
            loop = asyncio.get_running_loop()
            if loop and loop.is_running():
                # We're in an async context, use thread pool for initialization
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    try:
                        future_cli = executor.submit(asyncio.run, initialize_cli_mcp_wrapper(self.MCP_config_file))
                        self.cli_mcp_initialized = future_cli.result(timeout=10)
                        if self.cli_mcp_initialized:
                            print_system(f"✅ cli-mcp client initialized during startup with config: {self.MCP_config_file}")
                    except Exception as e:
                        print_system(f"⚠️ cli-mcp client startup initialization failed: {e}")
                        self.cli_mcp_initialized = False
            else:
                # Safe to run async initialization directly
                try:
                    self.cli_mcp_initialized = asyncio.run(initialize_cli_mcp_wrapper(self.MCP_config_file))
                    if self.cli_mcp_initialized:
                        print_system(f"✅ cli-mcp client initialized during startup with config: {self.MCP_config_file}")
                except Exception as e:
                    print_system(f"⚠️ cli-mcp client startup initialization failed: {e}")
                    self.cli_mcp_initialized = False
        except RuntimeError:
            # No event loop, safe to create one
            try:
                self.cli_mcp_initialized = asyncio.run(initialize_cli_mcp_wrapper(self.MCP_config_file))
                if self.cli_mcp_initialized:
                    print_system(f"✅ cli-mcp client initialized during startup with config: {self.MCP_config_file}")
            except Exception as e:
                print_system(f"⚠️ cli-mcp client startup initialization failed: {e}")
                self.cli_mcp_initialized = False
        
        # Add MCP tools to tool_map after successful initialization
        if self.cli_mcp_initialized:
            self._add_mcp_tools_to_map()
            #print_current(f"🔧 MCP tools loaded successfully during startup")

        
        # Log related settings
        # Only create logs directory if we have a valid workspace_dir
        if workspace_dir:
            # Get the parent directory of workspace (typically the output directory)
            parent_dir = os.path.dirname(workspace_dir) if workspace_dir else os.getcwd()
            self.logs_dir = os.path.join(parent_dir, "logs")  # Simplified: directly use "logs"
        else:
            # Don't create logs directory in project root when no workspace_dir is specified
            print_current("⚠️ No workspace_dir specified, skipping logs directory creation")
            self.logs_dir = None
        
        self.llm_logs_dir = self.logs_dir  # LLM call logs directory
        
        # Initialize debug recorder
        from src.multi_round_executor.debug_recorder import DebugRecorder
        self.debug_recorder = DebugRecorder(
            debug_mode=debug_mode,
            llm_logs_dir=self.llm_logs_dir,
            model=self.model,
        )
        
        # Ensure log directory exists only if logs_dir is set
        if self.llm_logs_dir:
            os.makedirs(self.llm_logs_dir, exist_ok=True)
        
    async def _initialize_mcp_async(self):
        """Initialize both MCP clients asynchronously"""
        try:
            # Initialize cli-mcp wrapper
            if not self.cli_mcp_initialized:
                self.cli_mcp_initialized = await initialize_cli_mcp_wrapper(self.MCP_config_file)
                if self.cli_mcp_initialized:
                    print_system("✅ cli-mcp client initialized successfully")
                else:
                    print_current("⚠️ cli-mcp client initialization failed")
            
            # Add MCP tools to tool_map after initialization
            if self.cli_mcp_initialized:
                self._add_mcp_tools_to_map()
                
        except Exception as e:
            print_current(f"⚠️ MCP client async initialization failed: {e}")
    
    def _should_initialize_cli_mcp(self) -> bool:
        """Check if cli-mcp should be initialized"""
        # Always initialize cli-mcp if config exists
        import os
        cli_config_path = self.MCP_config_file if self.MCP_config_file else "mcp.json"
        return os.path.exists(cli_config_path)

    def _add_mcp_tools_to_map(self):
        """Add MCP tools to the tool mapping"""
        # Clear tool definitions cache since tools are being updated
        if hasattr(self, '_clear_tool_definitions_cache'):
            self._clear_tool_definitions_cache()
        
        # Create tool source mapping table
        if not hasattr(self, 'tool_source_map'):
            self.tool_source_map = {}
        
        # FastMCP tools have the highest priority - register FastMCP tools first
        fastmcp_tools_added = []
        fastmcp_server_names = []  # Record the server names handled by FastMCP
        
        try:
            # Directly check the FastMCP wrapper, avoid relying on the MCP client state
            from tools.fastmcp_wrapper import get_fastmcp_wrapper, is_fastmcp_initialized
            
            # Directly get the FastMCP wrapper, do not rely on is_fastmcp_initialized()
            try:
                # Use the same config file that was used for initialization
                fastmcp_wrapper = get_fastmcp_wrapper(config_path=self.MCP_config_file, workspace_dir=self.workspace_dir)
                if fastmcp_wrapper and getattr(fastmcp_wrapper, 'initialized', False):
                    # Get FastMCP tools
                    fastmcp_tools = fastmcp_wrapper.get_available_tools()

                    if fastmcp_tools:
                        # Get the server names handled by FastMCP
                        if hasattr(fastmcp_wrapper, 'servers'):
                            fastmcp_server_names = list(fastmcp_wrapper.servers.keys())
                        
                        for tool_name in fastmcp_tools:
                            
                            # Create a wrapper function for each FastMCP tool
                            def create_fastmcp_tool_wrapper(tool_name=tool_name, wrapper=fastmcp_wrapper):
                                def sync_fastmcp_tool_wrapper(**kwargs):
                                    try:
                                        # Use the synchronous call_tool_sync method instead of async call_tool
                                        return wrapper.call_tool_sync(tool_name, kwargs)
                                    except Exception as e:
                                        return {"error": f"FastMCP tool {tool_name} call failed: {e}"}
                                
                                return sync_fastmcp_tool_wrapper
                            
                            # Add to tool mapping with FastMCP priority
                            self.tool_map[tool_name] = create_fastmcp_tool_wrapper()
                            self.tool_source_map[tool_name] = 'fastmcp'
                            fastmcp_tools_added.append(tool_name)
                        
                        logger.info(f"Added {len(fastmcp_tools)} FastMCP tools: {', '.join(fastmcp_tools)}")
                    else:
                        logger.debug("FastMCP wrapper found but no tools available")
                elif fastmcp_wrapper:
                    logger.debug("FastMCP wrapper found but not initialized yet")
                else:
                    logger.debug("FastMCP wrapper is None")
            except Exception as e:
                logger.error(f"Error getting FastMCP wrapper: {e}")
                
        except Exception as e:
            logger.error(f"Failed to add FastMCP tools to mapping: {e}")
        
        # Add cli-mcp tools (NPX/NPM format) - but skip tools already handled by FastMCP
        if self.cli_mcp_initialized and self.cli_mcp_client:
            try:
                # Get available MCP tools from cli-mcp wrapper
                cli_mcp_tools = self.cli_mcp_client.get_available_tools()
                
                if cli_mcp_tools:
                    cli_mcp_tools_added = []
                    for tool_name in cli_mcp_tools:
                        # Intelligently check if the tool is already handled by FastMCP
                        should_skip = False
                        skip_reason = ""
                        
                        # CLI-MCP tool format: server_name_tool_name (split by the first underscore)
                        if '_' in tool_name:
                            # Find the position of the first underscore for splitting
                            first_underscore = tool_name.find('_')
                            server_part = tool_name[:first_underscore]        # Server name part
                            actual_tool_name = tool_name[first_underscore+1:] # Actual tool name
                            
                            # Check 1: Is the server handled by FastMCP?
                            for fastmcp_server in fastmcp_server_names:
                                if (server_part == fastmcp_server.replace('-', '_') or 
                                    server_part.replace('_', '-') == fastmcp_server or
                                    server_part == fastmcp_server):
                                    should_skip = True
                                    skip_reason = f"server {fastmcp_server} handled by FastMCP"
                                    break
                            
                            # Check 2: Is the tool name already registered by FastMCP?
                            if not should_skip and actual_tool_name in fastmcp_tools_added:
                                should_skip = True
                                skip_reason = f"tool {actual_tool_name} already in FastMCP"
                        
                        # Check 3: Is the full tool name already registered by FastMCP?
                        if not should_skip and tool_name in fastmcp_tools_added:
                            should_skip = True
                            skip_reason = f"exact tool name in FastMCP"
                        
                        if should_skip:
                            logger.debug(f"Skipping cli-mcp tool {tool_name} ({skip_reason})")
                            continue
                            
                        # Create a wrapper function for each cli-mcp tool
                        def create_cli_mcp_tool_wrapper(tool_name=tool_name):
                            def sync_cli_mcp_tool_wrapper(**kwargs):
                                import asyncio
                                try:
                                    # Call the cli-mcp wrapper
                                    return asyncio.run(self.cli_mcp_client.call_tool(tool_name, kwargs))
                                except Exception as e:
                                    return {"error": f"cli-mcp tool {tool_name} call failed: {e}"}
                            
                            return sync_cli_mcp_tool_wrapper
                        
                        # Add to tool mapping WITHOUT prefix
                        self.tool_map[tool_name] = create_cli_mcp_tool_wrapper()
                        self.tool_source_map[tool_name] = 'cli_mcp'
                        cli_mcp_tools_added.append(tool_name)
                    
                    if cli_mcp_tools_added:
                        logger.info(f"Added {len(cli_mcp_tools_added)} cli-mcp tools: {', '.join(cli_mcp_tools_added)}")
                    else:
                        logger.debug("No cli-mcp tools added (all handled by FastMCP)")
            except Exception as e:
                logger.error(f"Failed to add cli-mcp tools to mapping: {e}")
                self.cli_mcp_initialized = False
    
    def cleanup(self):
        """Clean up all resources and threads"""
        try:
            
            # Cleanup cli-mcp client
            if hasattr(self, 'cli_mcp_client') and self.cli_mcp_client:
                try:
                    safe_cleanup_cli_mcp_wrapper()
                    # print_current("🔌 cli-mcp client cleanup completed")
                except Exception as e:
                    print_current(f"⚠️ cli-mcp client cleanup failed: {e}")
            
            # Cleanup long-term memory
            if hasattr(self, 'long_term_memory') and self.long_term_memory:
                try:
                    self.long_term_memory.cleanup()
                    # print_current("🧠 Long-term memory cleanup completed")
                except Exception as e:
                    print_current(f"⚠️ Long-term memory cleanup failed: {e}")
            
            # Cleanup tools
            if hasattr(self, 'tools') and self.tools:
                try:
                    self.tools.cleanup()
                except Exception as e:
                    print_current(f"⚠️ Tools cleanup failed: {e}")
            
            # Close LLM client connections if needed
            if hasattr(self, 'client'):
                try:
                    if hasattr(self.client, 'close'):
                        self.client.close()
                except:
                    pass
            
            # print_system(f"✅ ToolExecutor cleanup completed")
            
        except Exception as e:
            print_system(f"⚠️ Error during ToolExecutor cleanup: {e}")
    
    def _handle_task_completion(self, prompt: str, result: str, task_history: List[Dict[str, Any]], 
                                messages: List[Dict[str, Any]], round_counter: int,
                                has_tool_calls: bool, tool_calls_count: int = 0, 
                                successful_executions: int = 0, history_was_optimized: bool = False) -> Union[str, Tuple[str, List[Dict[str, Any]]]]:
        """
        Unified method to handle task completion: save debug log, store memory, and return result
        
        Args:
            prompt: Original task prompt
            result: Task execution result
            task_history: Task history
            messages: LLM messages for debug log
            round_counter: Current execution round
            has_tool_calls: Whether tools were called
            tool_calls_count: Number of tool calls (if any)
            successful_executions: Number of successful tool executions (if any)
            history_was_optimized: Whether history was optimized
            
        Returns:
            Result string or tuple (result, optimized_history)
        """
        # Determine completion method
        if has_tool_calls:
            completion_method = "task_completed_with_tools"
            execution_result = "task_completed_with_tools"
        else:
            completion_method = "TASK_COMPLETED_flag"
            execution_result = "task_completed_flag"
        
        # Save debug log
        if self.debug_mode:
            try:
                completion_info = {
                    "has_tool_calls": has_tool_calls,
                    "task_completed": True,
                    "completion_detected": True,
                    "execution_result": execution_result,
                }
                if has_tool_calls:
                    completion_info.update({
                        "tool_calls_count": tool_calls_count,
                        "successful_executions": successful_executions
                    })
                
                log_message = f"Task completed with TASK_COMPLETED flag" + (" after tool execution" if has_tool_calls else "")
                self.debug_recorder.save_llm_call_debug_log(messages, log_message, 1, completion_info)
            except Exception as log_error:
                print_current(f"❌ Completion debug log save failed: {log_error}")
        
        # Store task completion in long-term memory
        metadata = {
            "task_completed": True,
            "completion_method": completion_method,
            "execution_round": round_counter,
            "model_used": self.model
        }
        if has_tool_calls:
            metadata.update({
                "tool_calls_count": tool_calls_count,
                "successful_executions": successful_executions
            })
        
        self._store_task_completion_memory(prompt, result, metadata, force_update=True)
        
        finish_operation(f"executing task (round {round_counter})")
        
        # Return optimized history if available
        if history_was_optimized:
            return (result, task_history)
        return result
    
    def _store_task_completion_memory(self, task_prompt: str, task_result: str, metadata: Dict[str, Any] = None, force_update: bool = False):
        """
        Store task completion in long-term memory
        
        Args:
            task_prompt: The original task prompt
            task_result: The task execution result
            metadata: Additional metadata about the execution
            force_update: Force update regardless of interval (for task completion)
        """
        try:
            if not hasattr(self, 'long_term_memory') or not self.long_term_memory:
                # Long-term memory not available, skip silently
                return
            
            # Increment the counter
            self.memory_update_counter += 1

            # Check if update is needed: every 10 rounds or forced update
            should_update = (self.memory_update_counter % self.memory_update_interval == 0) or force_update

            if not should_update:
                # Skip this update, but log in debug mode
                return

            # Store the memory
            result = self.long_term_memory.memory_manager.store_task_memory(
                task_prompt=task_prompt,
                task_result=task_result,
                execution_metadata=metadata
            )

            if result.get("status") == "success":
                action = result.get("action", "stored")
                memory_id = result.get("memory_id", "unknown")
                # Only print for new memories, not updates
                #if action == "added":
                #    print_current(f"🧠 Task memory stored (ID: {memory_id})")
                #elif action == "updated":
                #    print_current(f"🧠 Task memory updated (ID: {memory_id})")
            else:
                # Only print errors in debug mode to avoid cluttering output
                if self.debug_mode:
                    print_current(f"⚠️ Failed to store task memory: {result.get('error', 'Unknown error')}")

        except Exception as e:
            # Only print errors in debug mode
            if self.debug_mode:
                print_current(f"⚠️ Exception occurred while storing task memory: {e}")
                
    def _setup_llm_client(self):
        """
        Set up the LLM client based on the API base URL.
        """
        if self.is_claude:
            # print_current(f"🧠 Detected Anthropic API, using Anthropic protocol")
            # print_current(f" Anthropic API Base: {self.api_base}")
            
            # Initialize Anthropic client
            Anthropic = get_anthropic_client()
            self.client = Anthropic(
                api_key=self.api_key,
                base_url=self.api_base
            )
        else:
            # print_current(f"🤖 Using OpenAI protocol")
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
            # print_current(f"🔧 Using max_tokens from config: {config_max_tokens}")
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
        
        print_current(f"🔧 Using default max_tokens for model {model}: {max_tokens}")
        return max_tokens
    

    
    def load_system_prompt(self, prompt_file: str = "prompts.txt") -> str:
        """
        Load only the core system prompt (system_prompt.txt).
        Other prompt files are loaded separately for user message construction.
        
        Args:
            prompt_file: Path to the prompt file (legacy support)
            
        Returns:
            The core system prompt text from system_prompt.txt, modified for infinite loop mode if applicable
        """
        try:
            # Choose prompt file based on plan mode
            if self.plan_mode:
                prompt_filename = "system_plan_prompt.txt"
            else:
                prompt_filename = "system_prompt.txt"
            
            # Try to load prompt file from custom prompts folder
            system_prompt_file = os.path.join(self.prompts_folder, prompt_filename)
            
            if os.path.exists(system_prompt_file):
                with open(system_prompt_file, 'r', encoding='utf-8') as f:
                    system_prompt = f.read().strip()
            else:
                # Fall back to single file approach
                with open(prompt_file, 'r', encoding='utf-8') as f:
                    system_prompt = f.read()
            
            # Add system language information in plan mode
            if self.plan_mode:
                # Map language code to language name
                lang_map = {
                    'zh': 'Chinese (中文)',
                    'en': 'English'
                }
                lang_name = lang_map.get(self.language, self.language)
                language_info = f"\n\nThe current system language is: **{lang_name}**\n\nPlease use {lang_name} for all conversations, questions, and content generation.\n"
                # Insert language info after "Language Setting" section
                if "## Language Setting" in system_prompt:
                    # Find the position after "## Language Setting" section header and content
                    # Look for the next section header (## ) after Language Setting
                    lang_section_pos = system_prompt.find("## Language Setting")
                    # Find the next section header
                    next_section_pos = system_prompt.find("\n## ", lang_section_pos + len("## Language Setting"))
                    if next_section_pos != -1:
                        # Insert before the next section
                        system_prompt = system_prompt[:next_section_pos] + language_info + system_prompt[next_section_pos:]
                    else:
                        # No next section found, append at the end
                        system_prompt = system_prompt + language_info
                else:
                    # Insert at the beginning if Language Setting section doesn't exist
                    system_prompt = language_info + system_prompt
            
            # Modify system prompt for infinite loop mode
            infinite_loop_mode = (self.subtask_loops == -1)
            if infinite_loop_mode:
                # Replace the task completion section for infinite loop mode
                task_completion_section = """## Task Completion Signal
When you've fully completed a task and believe no further iterations are needed, you MUST send this singal to exit the task execution:
TASK_COMPLETED: [Brief description of what was accomplished]
Note: Don't send TASK_COMPLETED signal if you calls tools in the current round, you should wait and check the tool executing result in the next round and then send this signal.
Do not do more than what the user requests, and do not do less than what the user requests. When a task is completed, stop immediately without unnecessary iterations or improvements.
If the user is just greeting you, asking simple questions, or not assigning a specific task, please respond directly and finish the task using TASK_COMPLETED."""
                
                infinite_loop_section = """## Infinite Autonomous Loop Mode
You are currently operating in INFINITE AUTONOMOUS LOOP MODE. In this mode:
- The system will continue executing until the task is naturally completed
- DO NOT use TASK_COMPLETED signal - it will not stop the execution in this mode
- Focus on making continuous progress towards the goal through iterative improvements
- When you have truly completed the task, use the talk_to_user tool to notify the user:
  talk_to_user(query="TASK_COMPLETED: [Brief description of what was accomplished]", timeout=-1)
- The timeout=-1 parameter disables the timeout, allowing the user to acknowledge completion
- Continue working autonomously until you achieve the objective
- Use tools and make changes as needed to move closer to the goal
- Each iteration should build upon previous work and make meaningful progress"""
                
                system_prompt = system_prompt.replace(task_completion_section, infinite_loop_section)
            
            # Add skill query feature if long-term memory is enabled
            if self.skill_tools:
                skill_query_section = """
## Skill Query Feature
For complex tasks, you can use the `query_skill` tool to search for relevant historical experiences and skills that might help you complete the task more efficiently. This is especially useful when you encounter similar problems or need to follow established patterns.

When you use skills from `query_skill`, make sure to:
1. Keep the skill_id in your conversation history for reference
2. Explicitly document which skills you referenced in plan.md
3. After task completion, use `rate_skill` to update the quality index of skills you used

The skill system helps you learn from past experiences and improve over time. Use it proactively for complex tasks!
"""
                # Insert skill query section before "Task Execution Approach" or at the end
                if "## Task Execution Approach" in system_prompt:
                    task_exec_pos = system_prompt.find("## Task Execution Approach")
                    system_prompt = system_prompt[:task_exec_pos] + skill_query_section + "\n" + system_prompt[task_exec_pos:]
                else:
                    system_prompt = system_prompt + skill_query_section
            
            return system_prompt
                
        except Exception as e:
            print_current(f"Error loading system prompt: {e}")
            return "You are a helpful AI assistant that can use tools to accomplish tasks."
    
    def load_user_prompt_components(self) -> Dict[str, str]:
        """
        Load all prompt components that go into the user message.
        
        Returns:
            Dictionary containing different prompt components
        """
        components = {
            'rules_and_tools': '',
            'system_environment': '',
            'workspace_info': '',
        }
        
        try:
            # For chat-based tools, generate tool descriptions from JSON instead of loading files
            if self.use_chat_based_tools:
                # Generate tools prompt from JSON definitions
                # Force reload to ensure FastMCP tools are included if they were initialized after first load
                tool_definitions = self._load_tool_definitions_from_file(force_reload=True)
                
                # Choose format based on tool_call_parse_format configuration
                if self.tool_call_parse_format == "xml":
                    tools_prompt = generate_tools_prompt_from_xml(tool_definitions, self.language)
                else:
                    # Default to JSON format
                    tools_prompt = generate_tools_prompt_from_json(tool_definitions, self.language)
                
                # Load only rules and plugin prompts (excluding deprecated tool files)
                rules_tool_files = [
                    os.path.join(self.prompts_folder, "rules_prompt.txt"), 
                    os.path.join(self.prompts_folder, "mcp_kb_tool_prompts.txt"),
                    os.path.join(self.prompts_folder, "user_rules.txt")
                ]
                
                rules_parts = []
                if tools_prompt:
                    rules_parts.append(tools_prompt)
                
                loaded_files = []
                
                for file_path in rules_tool_files:
                    if os.path.exists(file_path):
                        try:
                            with open(file_path, 'r', encoding='utf-8') as f:
                                content = f.read().strip()
                                if content:
                                    rules_parts.append(content)
                                    loaded_files.append(file_path)
                        except Exception as e:
                            print_current(f"Warning: Could not load file {file_path}: {e}")
                
                # Load multiagent_prompt.txt if multi-agent is enabled
                if self.multi_agent:
                    multiagent_file_path = os.path.join(self.prompts_folder, "multiagent_prompt.txt")
                    if os.path.exists(multiagent_file_path):
                        try:
                            with open(multiagent_file_path, 'r', encoding='utf-8') as f:
                                content = f.read().strip()
                                if content:
                                    rules_parts.append(content)
                                    loaded_files.append(multiagent_file_path)
                        except Exception as e:
                            print_current(f"Warning: Could not load file {multiagent_file_path}: {e}")
                
                if rules_parts:
                    components['rules_and_tools'] = "\n\n".join(rules_parts)
                

            else:
                # For standard tool calling, load only rules (no tool descriptions needed)
                rules_tool_files = [
                    os.path.join(self.prompts_folder, "rules_prompt.txt"), 
                    os.path.join(self.prompts_folder, "mcp_kb_tool_prompts.txt"),
                    os.path.join(self.prompts_folder, "user_rules.txt")
                ]
                
                rules_parts = []
                loaded_files = []
                
                for file_path in rules_tool_files:
                    if os.path.exists(file_path):
                        try:
                            with open(file_path, 'r', encoding='utf-8') as f:
                                content = f.read().strip()
                                if content:
                                    rules_parts.append(content)
                                    loaded_files.append(file_path)
                        except Exception as e:
                            print_current(f"Warning: Could not load file {file_path}: {e}")
                
                # Load multiagent_prompt.txt if multi-agent is enabled
                if self.multi_agent:
                    multiagent_file_path = os.path.join(self.prompts_folder, "multiagent_prompt.txt")
                    if os.path.exists(multiagent_file_path):
                        try:
                            with open(multiagent_file_path, 'r', encoding='utf-8') as f:
                                content = f.read().strip()
                                if content:
                                    rules_parts.append(content)
                                    loaded_files.append(multiagent_file_path)
                        except Exception as e:
                            print_current(f"Warning: Could not load file {multiagent_file_path}: {e}")
                
                if rules_parts:
                    components['rules_and_tools'] = "\n\n".join(rules_parts)
                
                print_debug("✅ Using standard tool calling (tool descriptions provided via API)")
            
            # Inject agent role information into rules_prompt content
            if components['rules_and_tools']:
                from src.tools.agent_context import get_current_agent_id
                current_agent_id = get_current_agent_id()
                
                # Determine role based on agent_id
                if current_agent_id and current_agent_id.startswith('agent_'):
                    # This is an executor agent
                    role_info = f"\n\n<agent_role_info>\nYOUR ROLE: You are an EXECUTOR agent. Your agent ID is: {current_agent_id}\n- You are NOT the manager\n- As an executor, you MUST NOT edit the plan.md file\n- You should follow instructions from the manager\n</agent_role_info>\n"
                elif current_agent_id == 'manager' or not current_agent_id:
                    # This is the manager (or no agent_id means manager by default)
                    role_info = f"\n\n<agent_role_info>\nYOUR ROLE: You are the MANAGER agent. Your agent ID is: {current_agent_id or 'manager'}\n- You are responsible for planning and updating plan.md\n- You can spawn executor agents and assign tasks to them\n</agent_role_info>\n"
                else:
                    role_info = ""
                
                # Inject role info after rules_prompt content
                if role_info:
                    components['rules_and_tools'] = components['rules_and_tools'] + role_info
                
            # Note: Removed loading of deprecated files:
            # - prompts/tool_prompt.txt
            # - prompts/tool_prompt_for_chat.txt  
            # - prompts/multiagent_prompt.txt
            # These are now replaced by JSON-generated tool descriptions
            
            # Load system environment information
            components['system_environment'] = get_system_environment_info(
                language=self.language, 
                model=self.model, 
                api_base=self.api_base
            )
            
            # Load workspace information
            components['workspace_info'] = get_workspace_info(self.workspace_dir)
            
        except Exception as e:
            print_current(f"Warning: Error loading user prompt components: {e}")
        
        return components
    
    
    def _build_alternating_history_messages(self, task_history: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """
        Convert the task history to a list of alternating messages.
        Format: assistant reply -> user tool execution result -> assistant reply -> user tool execution result...
        
        Args:
            task_history: Previous task execution history
            
        Returns:
            List of messages in alternating format, each element is {"role": "assistant"|"user", "content": "..."}
        """
        if not task_history:
            return []
        
        messages = []
        processed_history = task_history

        # Keep only the most recent temporary error feedback (if any), remove earlier ones
        error_feedback_indices = [i for i, record in enumerate(processed_history) 
                                  if record.get("temporary_error_feedback", False)]
        if len(error_feedback_indices) > 1:
            indices_to_remove = error_feedback_indices[:-1]
            # Remove from last to first to avoid index issues
            for idx in sorted(indices_to_remove, reverse=True):
                processed_history.pop(idx)
            print_debug(f"🧹 Cleaned {len(indices_to_remove)} old temporary error feedback records")
        
        for i, record in enumerate(processed_history, 1):
            if record.get("role") == "system":
                continue

            elif "result" in record:
                # Check if this is an error feedback record
                is_error_feedback = record.get("error", False)

                if is_error_feedback:
                    # Error feedback: add as user message
                    error_response = record['result'].strip()
                    messages.append({
                        "role": "user",
                        "content": error_response
                    })
                    continue

                # Parse the result field to separate assistant reply and tool execution results
                assistant_response = record['result'].strip()

                # Check if response contains tool calls and/or tool execution results

                # Extract tool calls section if present
                #if "--- Tool Calls ---" in assistant_response:
                #    parts = assistant_response.split("--- Tool Calls ---", 1)
                #    main_content = parts[0].strip()
                #   remaining_content = parts[1] if len(parts) > 1 else ""
                
                
                # Check if there are tool execution results after tool calls
                #if "--- Tool Execution Results ---" in remaining_content:
                #    tool_parts = remaining_content.split("--- Tool Execution Results ---", 1)
                #    tool_calls_section = tool_parts[0].strip()
                #    tool_results_section = tool_parts[1].strip() if len(tool_parts) > 1 else ""
                #else:
                #    tool_calls_section = remaining_content.strip()

                # If no tool call section but have tool execution results
                if "--- Tool Execution Results ---" in assistant_response:
                    parts = assistant_response.split("--- Tool Execution Results ---", 1)
                    assistant_content_parts = parts[0].strip()
                    tool_results_section = parts[1].strip() if len(parts) > 1 else ""
                else:
                    assistant_content_parts = assistant_response.strip()
                    tool_results_section = None

                # If first round, add user's request as user message
                # if "prompt" in record:
                #     user_request = record['prompt'].strip()
                #     messages.append({
                #         "role": "user",
                #         "content": user_request
                #     })
                
                # Add tool calls section to assistant message if present
                #if tool_calls_section:
                #    assistant_content_parts.append(f"\n**Tool Calls:**\n{tool_calls_section}")

                # Add assistant message
                if assistant_content_parts:
                    messages.append({
                        "role": "assistant",
                        "content": assistant_content_parts
                    })
                #print_debug('ROLE ASSISTANT:\n' + assistant_content_parts.replace("\\n", "\n"))
                
                # If tool execution results present, add as user message
                if tool_results_section:
                    # Standardize tool result format for more consistent caching
                    standardized_tool_results = self._standardize_tool_results_for_llm_input(tool_results_section)
                    messages.append({
                        "role": "user",
                        "content": f"{standardized_tool_results}"
                    })
                #print_debug('ROLE USER:\n' + standardized_tool_results.replace("\\n", "\n"))

        return messages
    
    def _build_execution_instructions(self, execution_round: int, prompt: str) -> str:
        """
        Build the execution instructions section and add it to the message list.

        Args:
            execution_round: The current execution round number
            prompt: The user's original task prompt/requirement
        """
        messages = []
        # Execution instructions
        messages.append("## Execution Instructions:")
        
        # Add user's original requirement/prompt
        if prompt:
            messages.append(f"**User's Original Requirement:**\n{prompt}\n")
        
        # Check if we're in infinite loop mode
        infinite_loop_mode = (self.subtask_loops == -1)
        
        if infinite_loop_mode:
            messages.append(f"This is round {execution_round} of task execution in **INFINITE AUTONOMOUS LOOP MODE**.")
            messages.append("DO NOT use TASK_COMPLETED in this mode - it will not stop the execution")
            messages.append("When task is truly completed, use: talk_to_user(query=\"TASK_COMPLETED: [description]\", timeout=-1)")
        else:
            messages.append(f"This is round {execution_round} of task execution. Please continue with the task based on the above context and requirements.")
            messages.append("When finished, use TASK_COMPLETED: [description] to finish the task")
        return "\n".join(messages)

    def _add_history_to_llm_input_message(self, message_parts: List[str], task_history: List[Dict[str, Any]]) -> None:
        """
        Add task history context to LLM input message parts with intelligent summarization and standardized formatting.
        This function handles history length checking and adds history to the message with better cache hits.
        
        Args:
            message_parts: List to append history content to (will be sent to LLM as input)
            task_history: Previous task execution history
        """
        if not task_history:
            return
        
        # Use task_history directly
        processed_history = task_history

        # Only keep the most recent temporary error feedback (if any), remove earlier ones
        error_feedback_indices = [i for i, record in enumerate(processed_history) 
                                 if record.get("temporary_error_feedback", False)]
        if len(error_feedback_indices) > 1:
            # Keep only the last error feedback, remove the others
            indices_to_remove = error_feedback_indices[:-1]
            # Remove from the end to avoid index issues
            for idx in sorted(indices_to_remove, reverse=True):
                processed_history.pop(idx)
            print_debug(f"🧹 Cleaned {len(indices_to_remove)} old temporary error feedback records")
        
        message_parts.append("## Previous Rounds Context ")
        message_parts.append("Below is the context from previous tasks in this session:")
        message_parts.append("")
        
        for i, record in enumerate(processed_history, 1):
            if record.get("role") == "system":
                continue
            
            elif "result" in record:
                # Check if this is an error feedback record
                is_error_feedback = record.get("error", False)
                
                if is_error_feedback:
                    # Error feedback: prominently display at the beginning
                    message_parts.append("CRITICAL ERROR FEEDBACK FROM PREVIOUS ROUND:")
                    error_response = record['result'].strip()
                    message_parts.append(error_response)
                    message_parts.append("")
                    continue  # Skip the rest, error feedback already fully displayed
                
                # Optimization: Only show the user request in the first round, show round info in subsequent rounds
                if "prompt" in record:
                    # First round: display the full user request
                    user_request = record['prompt'].strip()
                    message_parts.append(f"**User Request:**")
                    message_parts.append(user_request)
                    message_parts.append("")
                else:
                    # Subsequent rounds: display round information
                    task_round = record.get('task_round', 'N/A')
                    message_parts.append(f"**Round {task_round} Execution Result:**")
                    message_parts.append("")
                
                # Format assistant response with consistent line breaks and standardized tool result formatting
                assistant_response = record['result'].strip()
                
                # Check if response contains tool calls and/or execution results
                tool_calls_section = ""
                tool_results_section = ""
                main_content = assistant_response
                
                # Extract tool calls section if present
                if "--- Tool Calls ---" in assistant_response:
                    parts = assistant_response.split("--- Tool Calls ---", 1)
                    main_content = parts[0].strip()
                    remaining_content = parts[1] if len(parts) > 1 else ""
                    
                    # Check if there are also tool execution results after tool calls
                    if "--- Tool Execution Results ---" in remaining_content:
                        tool_parts = remaining_content.split("--- Tool Execution Results ---", 1)
                        tool_calls_section = tool_parts[0].strip()
                        tool_results_section = tool_parts[1].strip() if len(tool_parts) > 1 else ""
                    else:
                        tool_calls_section = remaining_content.strip()
                
                # If no tool calls section but has tool execution results
                elif "--- Tool Execution Results ---" in assistant_response:
                    parts = assistant_response.split("--- Tool Execution Results ---", 1)
                    main_content = parts[0].strip()
                    tool_results_section = parts[1].strip() if len(parts) > 1 else ""
                    
                # Display the main assistant response
                message_parts.append(f"**LLM Response:**")
                message_parts.append(main_content)
                message_parts.append("")
                
                # Display tool calls if present
                if tool_calls_section:
                    message_parts.append("**LLM Called Following Tools (It is a reply from the environment, if you want to calling tools, you should fill in the tool calls section, not here!!!), you should not print this section in your response**")
                    message_parts.append(tool_calls_section)
                
                # Display tool execution results if present
                if tool_results_section:
                    message_parts.append("**Tool Execution Results:**")
                    # Standardize tool results format for better cache consistency
                    tool_results_section = self._standardize_tool_results_for_llm_input(tool_results_section)
                    message_parts.append(tool_results_section)
                
                message_parts.append("")  # Extra space after separator
    
    def _standardize_tool_results_for_llm_input(self, tool_results: str) -> str:
        """
        Standardize tool results format for LLM input message to improve cache consistency.
        
        Args:
            tool_results: Raw tool results string
            
        Returns:
            Standardized tool results string ready for LLM input
        """
        lines = tool_results.split('\n')
        standardized_lines = []
        
        for line in lines:
            # Remove trailing whitespace from each line
            line = line.rstrip()
            
            # Skip empty lines at the beginning
            if not standardized_lines and not line:
                continue
                
            # Standardize tool execution markers

            if line.startswith('Executing tool:'):
                # Standardize executing tool message format
                parts = line.split(' with params: ')
                if len(parts) == 2:
                    tool_info = parts[0].replace('Executing tool: ', '')
                    params_info = parts[1]
                    standardized_lines.append(f'Executing tool: {tool_info} with params: {params_info}')
                else:
                    standardized_lines.append(line)
            elif line.startswith('<tool_execute'):
                # Extract tool name and number, standardize format
                import re
                match = re.search(r'tool_name="([^"]+)".*tool_number="(\d+)"', line)
                if match:
                    tool_name, tool_number = match.groups()
                    standardized_lines.append(f'<tool_execute tool_name="{tool_name}" tool_number="{tool_number}">')
                else:
                    standardized_lines.append(line)
            elif line.startswith('</tool_execute>'):
                standardized_lines.append('</tool_execute>')
            else:
                standardized_lines.append(line)
        
        # Join lines and ensure consistent line ending
        result = '\n'.join(standardized_lines)
        
        # Remove trailing newlines and add a single one
        result = result.rstrip() + '\n' if result.strip() else ''
        
        return result
    
    def _get_recent_history_subset(self, task_history: List[Dict[str, Any]], max_length: int) -> List[Dict[str, Any]]:
        """
        Get a subset of recent history that doesn't exceed the maximum length.
        
        Args:
            task_history: Full task history
            max_length: Maximum allowed character length
            
        Returns:
            Subset of recent history records
        """
        if not task_history:
            return []
        
        # Start from the most recent records and work backwards
        recent_history = []
        current_length = 0
        
        for record in reversed(task_history):
            # Calculate the length of this record
            record_length = len(str(record.get("content", ""))) + len(str(record.get("result", ""))) + len(str(record.get("prompt", "")))
            
            # Check if adding this record would exceed the limit
            if current_length + record_length > max_length and recent_history:
                break
            
            recent_history.insert(0, record)
            current_length += record_length
        
        return recent_history

    def _add_error_feedback_to_history(self, task_history: Optional[List[Dict[str, Any]]] = None, 
                                       error_type: str = "", error_message: str = "", 
                                       execution_round: int = None) -> None:
        """
        将错误反馈添加到task_history中，以便在下一轮次中显眼地显示给大模型
        
        Args:
            task_history: 任务历史记录列表（如果为None，则从self._current_task_history获取）
            error_type: 错误类型 ('json_parse_error', 'hallucination_detected', 'multiple_tools_detected')
            error_message: 错误消息
            execution_round: 当前执行轮次（如果为None，则从self._current_execution_round获取）
        """
        # 如果没有提供task_history，尝试从self获取
        if task_history is None:
            task_history = getattr(self, '_current_task_history', None)
        
        if task_history is None:
            return
        
        # 如果没有提供execution_round，尝试从self获取
        if execution_round is None:
            execution_round = getattr(self, '_current_execution_round', 1)
        
        # 构建显眼的错误反馈消息
        if error_type == 'json_parse_error':
            feedback_message = f"""
CRITICAL ERROR FEEDBACK - JSON PARSING FAILED

⚠️⚠️⚠️ YOUR PREVIOUS TOOL CALL JSON FORMAT WAS INVALID ⚠️⚠️⚠️

The system failed to parse your tool call JSON format in the previous round.
This means your tool call was NOT executed successfully.

Error details: {error_message}

🔴 IMPORTANT REMINDERS:
1. Ensure your JSON format is valid and properly escaped
2. Check that all string values are properly quoted
3. Verify that all brackets and braces are properly matched
4. Do NOT include markdown code block markers (```json) inside JSON string values
5. Make sure boolean values are lowercase (true/false, not True/False)
6. Ensure numeric values are not quoted

Please regenerate your tool call with correct JSON format.

END OF ERROR FEEDBACK
"""
        elif error_type == 'hallucination_detected':
            feedback_message = f"""
CRITICAL ERROR FEEDBACK - HALLUCINATION DETECTED

⚠️⚠️⚠️ YOUR PREVIOUS RESPONSE CONTAINED HALLUCINATED CONTENT ⚠️⚠️⚠️

The system detected that yed to output tool execution results or tool call sections
that were NOT actually executed. This is a hallucination and was stopped.

Error details: {error_message}

🔴 IMPORTANT REMINDERS:
1. NEVER output "**LLM Called Following Tools in this round" or similar text
2. NEVER output "**Tool Execution Results:**" or similar text
3. Do NOT fabricate tool execution results - wait for actual tool execution results
4. Your response was truncated at the hallucination point, 
   the tools you called before this point should be executed successfully.

Please regenerate your response without hallucinated content.

END OF ERROR FEEDBACK
"""
        elif error_type == 'multiple_tools_detected':
            feedback_message = f"""
CRITICAL ERROR FEEDBACK - MULTIPLE TOOL CALLS DETECTED

⚠️⚠️⚠️ YOUR PREVIOUS RESPONSE CONTAINED MULTIPLE TOOL CALLS ⚠️⚠️⚠️

The system detected multipl calls in your response, but only ONE tool call
is allowed per round. Your response was truncated after the first tool call.

Error details: {error_message}

🔴 IMPORTANT REMINDERS:
1. You can ONLY call ONE tool per round
2. If you need to call multiple tools, call them sequentially in separate rounds
3. Wait for the tool execution result before calling the next tool
4. Do NOT include multiple ```json blocks with tool calls in a single response
5. Your response was truncated after the first tool call

Please regenerate your response with only ONE tool call.

END OF ERROR FEEDBACK
"""
        else:
            feedback_message = f"""
CRITICAL ERROR FEEDBACK

⚠️⚠️⚠️ AN ERROR OCCURRED IN THE PREVIOUS ROUND ⚠️⚠️⚠️

Error type: {error_type}
Error details: {error_message}

Please review the error and adjust your response accordingly.

END OF ERROR FEEDBACK
"""
        
        # 添加错误反馈记录到历史
        # 注意：添加一个标志表示这是临时的错误反馈，应该在下一轮成功后被清理
        error_record = {
            "task_round": execution_round,
            "result": feedback_message,
            "error": True,
            "error_type": error_type,
            "task_completed": False,
            "timestamp": datetime.datetime.now().isoformat(),
            "temporary_error_feedback": True  # 标记为临时错误反馈
        }
        task_history.append(error_record)

    def execute_subtask(self, prompt: str, prompts_file: str = "", 
                       task_history: Optional[List[Dict[str, Any]]] = None, 
                       execution_round: int = 1) -> Union[str, Tuple[str, List[Dict[str, Any]]]]:
        """
        Execute a subtask with potential multiple rounds (if tools need to be called)
        
        Args:
            prompt: User prompt
            prompts_file: Prompt file to load (currently not used, loads default system prompt)
            task_history: Historical messages from previous rounds (to maintain conversation context)
            execution_round: Current execution round number
            
        Returns:
            Execution result (str) or tuple (result, optimized_task_history) if history was optimized
        """
        track_operation(f"executing task (prompt length: {len(prompt)})")
        
        # Clear streaming tool execution flags and results for new task
        self._tools_executed_in_stream = False
        self._streaming_tool_results = []
        
        # Initialize task history if not provided
        if task_history is None:
            task_history = []
        
        # Store current task history reference for history compression tool
        self._current_task_history = task_history
        self._current_execution_round = execution_round
        
        history_was_optimized = False
        
        round_counter = execution_round
        
        try:
            # Check for terminate messages before each tool call in every round
            terminate_signal = self._check_terminate_messages()
            if terminate_signal:
                return terminate_signal
            
            # Load system prompt (only core system_prompt.txt content)
            system_prompt = self.load_system_prompt()
            tool_and_env_message = self._build_tool_and_env_message(prompt)
            history_messages = self._build_alternating_history_messages(task_history) if task_history else []
            execution_messages = self._build_execution_instructions(round_counter, prompt)

            # Compose the messages list by flattening history_messages if needed
            messages = []
            messages.append({"role": "user", "content": tool_and_env_message})
            messages.extend(history_messages)
            messages.append({"role": "user", "content": execution_messages})


            # Old method: Add task history to the message (at the end) - COMMENTED OUT
            #if task_history:
            #     message_parts = [prompt]
            #     self._add_history_to_llm_input_message(message_parts, task_history)
            #     user_message = "\n".join(message_parts)
             
            # Prepare messages for the LLM with proper system/user separation
            #messages = [
            #     {"role": "user", "content": user_message}
            #]
            
            # Execute LLM call with standard tools
            content, tool_calls = self._call_llm_with_standard_tools(messages, system_prompt)
            
            # Execute tools if present (regardless of completion flag)
            if tool_calls:
                # Always format tool calls for history (needed for final response)
                tool_calls_formatted = self._format_tool_calls_for_history(tool_calls)
                
                # Check if tools were already executed during streaming
                tools_already_executed = (self.streaming and 
                                        (not self.use_chat_based_tools) and
                                        hasattr(self, '_tools_executed_in_stream') and 
                                        getattr(self, '_tools_executed_in_stream', False))

                # Print tool calls for terminal display with better formatting
                #if tool_calls_formatted:
                #    # Remove the "**Tool Calls:**" header since we already printed our own
                #    display_content = tool_calls_formatted.replace("**Tool Calls:**\n", "").strip()
                #    print_current(display_content)
                    
                if tools_already_executed:
                    all_tool_results = getattr(self, '_streaming_tool_results', [])
                    successful_executions = len(all_tool_results)

                else:
                    if self.use_chat_based_tools:
                        print_current("")
                    
                    # Execute all tool calls and collect results
                    all_tool_results = []
                    successful_executions = 0

                    for i, tool_call in enumerate(tool_calls, 1):
                        # Handle standard format tool calls (both OpenAI and Anthropic)
                        try:
                            tool_name = self._get_tool_name_from_call(tool_call)
                            tool_params = self._get_tool_params_from_call(tool_call)
                            #print_current(f"🔧 Executing tool {tool_name}")
                        except Exception as e:
                            print_current(f"❌ Failed to extract tool name/params from tool_call {i}: {e}")
                            continue
                        
                        # Let streaming_output handle all tool execution display
                        try:
                            # Convert to standard format for execute_tool
                            standard_tool_call = {
                                "name": tool_name,
                                "arguments": tool_params
                            }

                            tool_result = self.execute_tool(standard_tool_call)
                            
                            all_tool_results.append({
                                'tool_name': tool_name,
                                'tool_params': tool_params,
                                'tool_result': tool_result
                            })
                            
                            # Only count as successful if the tool didn't return an error
                            if not (isinstance(tool_result, dict) and tool_result.get('status') in ['error', 'failed']):
                                successful_executions += 1
                            
                        except Exception as e:
                            error_msg = f"Tool {tool_name} execution failed: {str(e)}"
                            print_current(f"❌ {error_msg}")
                            all_tool_results.append({
                                'tool_name': tool_name,
                                'tool_params': tool_params,
                                'tool_result': f"Error: {error_msg}"
                            })
                
                # Format tool results
                tool_results_message = self._format_tool_results_for_llm(all_tool_results)
                
                # Check for TASK_COMPLETED flag in original content (after tool execution)
                completion_message = TaskChecker.extract_completion_info(content)
                has_task_completed = completion_message is not None
                
                # Save debug log with tool execution info
                if self.debug_mode:
                    try:
                        tool_execution_info = {
                            "has_tool_calls": True,
                            "parsed_tool_calls": tool_calls,
                            "tool_results": all_tool_results,
                            "formatted_tool_results": tool_results_message,
                            "successful_executions": successful_executions,
                            "total_tool_calls": len(tool_calls),
                            "has_task_completed": has_task_completed
                        }
                        self.debug_recorder.save_llm_call_debug_log(messages, f"Single execution with {len(tool_calls)} tool calls", 1, tool_execution_info)
                    except Exception as log_error:
                        print_current(f"❌ Debug log save failed: {log_error}")
                
                # Build combined result with tool calls and tool results
                result_parts = [content]
                #if tool_calls_formatted:
                #    result_parts.append("\n\n--- Tool Calls ---\n" + tool_calls_formatted)
                
                # If there was a TASK_COMPLETED flag, re-add it before tool execution results
                # This ensures task_checker can detect it (it only checks content before "--- Tool Execution Results ---")
                if has_task_completed:
                    completion_msg = completion_message or ""
                    if completion_msg:
                        result_parts.append(f"\n\nTASK_COMPLETED: {completion_msg}")
                    else:
                        result_parts.append("\n\nTASK_COMPLETED")
                
                result_parts.append("\n\n--- Tool Execution Results ---\n" + tool_results_message)
                combined_result = "".join(result_parts)
                
                # Handle task completion or normal tool execution
                if has_task_completed:
                    return self._handle_task_completion(
                        prompt=prompt,
                        result=combined_result,
                        task_history=task_history,
                        messages=messages,
                        round_counter=round_counter,
                        has_tool_calls=True,
                        tool_calls_count=len(tool_calls),
                        successful_executions=successful_executions,
                        history_was_optimized=history_was_optimized
                    )
                
                # Normal tool execution (no completion flag)
                self._store_task_completion_memory(prompt, combined_result, {
                    "task_completed": False,
                    "completion_method": "tool_execution",
                    "execution_round": round_counter,
                    "tool_calls_count": len(tool_calls),
                    "successful_executions": successful_executions,
                    "model_used": self.model
                }, force_update=False)

                finish_operation(f"executing task (round {round_counter})")
                if history_was_optimized:
                    return (combined_result, task_history)
                return combined_result
            
            else:
                # No tool calls: check for TASK_COMPLETED flag
                completion_message = TaskChecker.extract_completion_info(content)
                
                if completion_message is not None:
                    # Task completed without tool calls
                    return self._handle_task_completion(
                        prompt=prompt,
                        result=content,
                        task_history=task_history,
                        messages=messages,
                        round_counter=round_counter,
                        has_tool_calls=False,
                        history_was_optimized=history_was_optimized
                    )
                
                # No tool calls and no TASK_COMPLETED flag, return LLM response directly
                if self.debug_mode:
                    try:
                        no_tools_info = {
                            "has_tool_calls": False,
                            "task_completed": False,
                            "execution_result": "llm_response_only"
                        }
                        self.debug_recorder.save_llm_call_debug_log(messages, f"Single execution, no tool calls", 1, no_tools_info)
                    except Exception as log_error:
                        print_current(f"❌ Final debug log save failed: {log_error}")
                
                finish_operation(f"executing task (round {round_counter})")
                if history_was_optimized:
                    return (content, task_history)
                return content
            
        except json.JSONDecodeError as e:
            error_msg = f"JSON parsing error in tool call: {str(e)}"
            print_debug(error_msg)
            
            self._add_error_feedback_to_history(
                task_history=task_history,
                error_type='json_parse_error',
                error_message=error_msg,
                execution_round=round_counter
            )
            
            finish_operation(f"executing task (round {round_counter})")
            return error_msg
        except Exception as e:
            error_msg = f"Error executing subtask: {str(e)}"
            print_current(error_msg)
            self._add_error_feedback_to_history(
                task_history=task_history,
                error_type='general_error',
                error_message=error_msg,
                execution_round=round_counter
            )
            
            finish_operation(f"executing task (round {round_counter})")
            return error_msg
            
        return ""
    
    def _check_terminate_messages(self) -> Optional[str]:
        """
        Check if the agent has received a terminate signal.

        Returns:
            If a terminate signal is received, return the termination message; otherwise, return None.
        """
        try:
            # Only check in multi-agent mode
            if not self.multi_agent_tools:
                return None


            current_agent_id = get_current_agent_id()
            # If no agent_id is set, treat as manager
            if not current_agent_id:
                current_agent_id = "manager"

            try:
                router = get_message_router(self.multi_agent_tools.workspace_root, cleanup_on_init=False)
                mailbox = router.get_mailbox(current_agent_id)

                if not mailbox:
                    return None

                # Directly get unread messages, do not mark as read automatically
                unread_messages = mailbox.get_unread_messages()

                # Check if there is a terminate signal
                for message in unread_messages:
                    if hasattr(message, 'message_type') and hasattr(message, 'content'):
                        message_type = message.message_type
                        content = message.content

                        # Check if it is a system message and contains a terminate signal
                        if (message_type.value == "system" and
                            isinstance(content, dict) and
                            content.get("signal") == "terminate"):

                            reason = content.get("reason", "Terminated by request")
                            sender = content.get("sender", "unknown")

                            terminate_msg = f"AGENT_TERMINATED: Agent {current_agent_id} received terminate signal from {sender}. Reason: {reason}"
                            print_current(f"🛑 {terminate_msg}")

                            # Only mark the message as read after confirming the terminate signal
                            try:
                                mailbox.mark_as_read(message.message_id)
                            except Exception as e:
                                print_current(f"⚠️ Warning: Could not mark terminate message as read: {e}")

                            return terminate_msg

                return None

            except Exception as e:
                if self.debug_mode:
                    print_current(f"⚠️ Warning: Error accessing mailbox directly: {e}")
                return None

        except Exception as e:
            # If checking terminate messages fails, normal execution should not be interrupted
            if self.debug_mode:
                print_current(f"⚠️ Warning: Failed to check terminate messages: {e}")
            return None
    
    def parse_tool_calls(self, content: str) -> List[Dict[str, Any]]:
        """
        Parse multiple tool calls from the model's response.
        
        Args:
            content: The model's response text
            
        Returns:
            List of dictionaries with tool name and parameters
        """
        
        all_tool_calls = []
        
        # Try to parse from configured format first (JSON or XML)
        if self.tool_call_parse_format == "json":
            # Try JSON format first
            json_tool_calls = parse_tool_calls_from_json(content)
            if json_tool_calls:
                all_tool_calls.extend(json_tool_calls)
                if self.debug_mode:
                    print_debug(f"✅ Successfully parsed {len(json_tool_calls)} tool calls from JSON")
            else:
                # If JSON format didn't find any tool calls, try XML format as fallback
                xml_tool_calls = parse_tool_calls_from_xml(content)
                if xml_tool_calls:
                    all_tool_calls.extend(xml_tool_calls)
                    if self.debug_mode:
                        print_debug(f"✅ Successfully parsed {len(xml_tool_calls)} tool calls from XML (fallback from JSON)")
        elif self.tool_call_parse_format == "xml":
            # Try XML format first
            xml_tool_calls = parse_tool_calls_from_xml(content)
            if xml_tool_calls:
                all_tool_calls.extend(xml_tool_calls)
                if self.debug_mode:
                    print_debug(f"✅ Successfully parsed {len(xml_tool_calls)} tool calls from XML")
            else:
                # If XML format didn't find any tool calls, try JSON format as fallback
                json_tool_calls = parse_tool_calls_from_json(content)
                if json_tool_calls:
                    all_tool_calls.extend(json_tool_calls)
                    if self.debug_mode:
                        print_debug(f"✅ Successfully parsed {len(json_tool_calls)} tool calls from JSON (fallback from XML)")
        
        # If no tool calls found from configured format or fallback, try Python format
        if not all_tool_calls:
            python_tool_calls = parse_python_function_calls(content, self.tool_map)
            if python_tool_calls:
                all_tool_calls.extend(python_tool_calls)
                if self.debug_mode:
                    print_debug(f"✅ Successfully parsed {len(python_tool_calls)} tool calls from Python format")
        
        # Convert to standard format: use "input" instead of "arguments"
        standardized_tool_calls = []
        for tool_call in all_tool_calls:
            if isinstance(tool_call, dict) and "name" in tool_call:
                standardized_tool_call = {
                    "name": tool_call["name"],
                    "input": tool_call.get("arguments") or tool_call.get("input", {})
                }
                standardized_tool_calls.append(standardized_tool_call)
            else:
                standardized_tool_calls.append(tool_call)
        
        return standardized_tool_calls
    
    def _run_async_safe(self, coro):
        """
        Safely run an asynchronous coroutine, handling various event loop scenarios.

        Args:
            coro: The asynchronous coroutine object to execute.

        Returns:
            The result of the coroutine execution.
        """
        import asyncio

        try:
            # Try to run directly (when no event loop is running).
            return asyncio.run(coro)
        except RuntimeError as e:
            # If an event loop is already running in the current thread, run in a new thread.
            if "cannot be called from a running event loop" in str(e):
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, coro)
                    return future.result(timeout=60)
            else:
                # For other RuntimeErrors, try creating and running a new event loop.
                try:
                    new_loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(new_loop)
                    try:
                        return new_loop.run_until_complete(coro)
                    finally:
                        new_loop.close()
                except Exception as loop_e:
                    raise RuntimeError(f"Failed to run async coroutine: {loop_e}") from e

    def execute_tool(self, tool_call: Dict[str, Any], streaming_output: bool = False) -> Any:
        """
        Execute a tool with the given parameters, optionally with streaming output.
        
        Args:
            tool_call: Dictionary containing tool name and parameters
            streaming_output: Whether to enable streaming output (currently not used, kept for compatibility)
            
        Returns:
            Result of executing the tool
        """
        tool_name = tool_call["name"]
        params = tool_call.get("input") or tool_call.get("arguments")
        
        tool_source = getattr(self, 'tool_source_map', {}).get(tool_name, 'regular')
        
        # Handle FastMCP tools
        if tool_source == 'fastmcp':
            current_thread = threading.current_thread().name
            print_debug(f"🚀 [Thread: {current_thread}] Calling FastMCP tool: {tool_name}")
            
            try:
                if tool_name in self.tool_map:
                    # Execute the tool function
                    result = self.tool_map[tool_name](**params)
                    return result
                else:
                    error_msg = f"FastMCP tool {tool_name} not found in tool map"
                    print_debug(f"❌ [Thread: {current_thread}] {error_msg}")
                    return {"error": error_msg}
                    
            except Exception as e:
                error_msg = f"FastMCP tool call failed: {e}"
                print_current(f"❌ [Thread: {current_thread}] {error_msg}")
                return {"error": error_msg}
        
        # Handle cli-mcp tools
        if tool_source == 'cli_mcp':
            current_thread = threading.current_thread().name
            
            if not self.cli_mcp_initialized:
                from tools.cli_mcp_wrapper import get_cli_mcp_status, initialize_cli_mcp_wrapper
                
                global_status = get_cli_mcp_status(self.MCP_config_file)
                if global_status.get("initialized", False):
                    self.cli_mcp_initialized = True
                    self._add_mcp_tools_to_map()
                else:
                    try:
                        self.cli_mcp_initialized = self._run_async_safe(
                            initialize_cli_mcp_wrapper(self.MCP_config_file)
                        )
                        if self.cli_mcp_initialized:
                            self._add_mcp_tools_to_map()
                            print_debug(f"✅ [Thread: {current_thread}] cli-mcp client initialized successfully")
                        else:
                            error_msg = f"cli-mcp client initialization failed in thread {current_thread}"
                            print_debug(f"❌ {error_msg}")
                            return {"error": error_msg}
                    except Exception as e:
                        error_msg = f"cli-mcp client initialization failed in thread {current_thread}: {e}"
                        print_debug(f"❌ {error_msg}")
                        return {"error": error_msg}
            
            try:
                print_debug(f"🔧 [Thread: {current_thread}] Calling cli-mcp tool: {tool_name}")
                actual_tool_name = tool_name.replace("cli_mcp_", "")
                result = self._run_async_safe(
                    self.cli_mcp_client.call_tool(actual_tool_name, params)
                )
                print_debug(f"✅ [Thread: {current_thread}] cli-mcp tool call completed successfully")
                return result
                
            except Exception as e:
                error_msg = f"cli-mcp tool call failed in thread {current_thread}: {e}"
                print_debug(f"❌ {error_msg}")
                return {"error": error_msg}
        
        # Handle regular tools
        if tool_name in self.tool_map:
            tool_func = self.tool_map[tool_name]
            try:
                # Filter out None values and empty strings for optional parameters
                # Special handling: for edit_file, preserve empty string for code_edit parameter
                if tool_name == "edit_file":
                    filtered_params = {k: v for k, v in params.items() if v is not None and not (v == "" and k != "code_edit")}
                else:
                    filtered_params = {k: v for k, v in params.items() if v is not None and v != ""}
                # Execute the tool function
                result = tool_func(**filtered_params)
                return result
                
            except TypeError as e:
                # Handle parameter mismatch with helpful guidance
                error_msg = f"Parameter mismatch: {str(e)}"
                error_result = {
                    'tool': tool_name,
                    'status': 'error',
                    'error': error_msg,
                    'parameters': params
                }
                print_debug(f"❌ Tool execution failed: {error_result}")
                return error_result
            except Exception as e:
                # General exception handling
                error_result = {
                    'tool': tool_name,
                    'status': 'error', 
                    'error': f"Execution failed: {str(e)}",
                    'parameters': params
                }
                print_debug(f"❌ Tool execution failed: {error_result}")
                return error_result
        else:
            # Unknown tool
            error_result = {
                'tool': tool_name,
                'status': 'error',
                'error': f"Unknown tool: {tool_name}"
            }
            print_debug(f"❌ Tool execution failed: {error_result}")
            return error_result
    
    def _format_dict_as_text(self, data: Dict[str, Any], for_terminal_display: bool = False, tool_name: str = None, tool_params: Dict[str, Any] = None) -> str:
        """
        Format a dictionary result as readable text.
        
        Args:
            data: Dictionary to format
            for_terminal_display: If True, skip stdout/stderr for terminal commands to avoid duplication
            tool_name: Name of the tool that generated this result (for special handling)
            tool_params: Parameters of the tool that generated this result (for special handling)
            
        Returns:
            Formatted text string
        """
        if not isinstance(data, dict):
            return str(data)
        
        lines = []
        
        # Handle error cases first
        if 'error' in data:
            error_msg = f"Error: {data['error']}"
            if 'tool' in data:
                error_msg = f"Tool '{data['tool']}' failed: {data['error']}"
            # Add available tools help if present (for unknown tool errors)
            if 'available_tools_help' in data:
                error_msg += f"\n\n{data['available_tools_help']}"
            elif 'available_tools' in data:
                tools_list = data['available_tools']
                if isinstance(tools_list, list) and len(tools_list) > 0:
                    error_msg += f"\n\nAvailable tools: {', '.join(tools_list[:10])}"
                    if len(tools_list) > 10:
                        error_msg += f", and {len(tools_list) - 10} more..."
            return error_msg
        
        # Initialize processed_keys set first
        processed_keys = {'error', 'status', 'success'}
        
        # Show status if present
        if 'status' in data:
            if data['status'] == 'success':
                lines.append("✅ Success")
            elif data['status'] == 'error':
                lines.append("❌ Failed")
            else:
                lines.append(f"Status: {data['status']}")
        elif 'success' in data:
            status = "✅ Success" if data['success'] else "❌ Failed"
            lines.append(status)
        
        # Special handling for idle tool with early_exit (message detected)
        if tool_name == 'idle' and data.get('early_exit') and data.get('infinite_sleep'):
            # Prioritize showing the new message content prominently
            if 'content' in data:
                lines.append("🔔 URGENT: NEW MESSAGE DETECTED!")
                lines.append("=" * 60)
                lines.append(self._format_content_field(data['content']))
                lines.append("=" * 60)
                processed_keys.add('content')
            elif 'new_message_content' in data:
                lines.append("🔔 URGENT: NEW MESSAGE DETECTED!")
                lines.append("=" * 60)
                lines.append(f"Message from {data.get('new_message_sender', 'unknown')}: {data['new_message_content']}")
                lines.append("=" * 60)
                processed_keys.add('new_message_content')
                processed_keys.add('new_message_sender')
            if 'description' in data:
                lines.append(f"\n⚠️ {data['description']}")
                processed_keys.add('description')
        
        # Handle key fields in priority order
        field_handlers = [
            ('result', lambda v: f"Result: {v}"),
            ('content', self._format_content_field),
            ('file', lambda v: f"File: {v}"),
            ('output', lambda v: f"Output:\n{v}"),
            ('message', lambda v: f"Message: {v}"),
        ]
        
        # Process high-priority fields
        for field_name, handler in field_handlers:
            if field_name in data and field_name not in processed_keys:
                try:
                    formatted_value = handler(data[field_name])
                    if formatted_value:
                        lines.append(formatted_value)
                except Exception as e:
                    lines.append(f"{field_name}: {data[field_name]}")
                processed_keys.add(field_name)
        
        # Handle stdout/stderr (avoid duplication for terminal commands)
        if not (for_terminal_display and 'command' in data):
            if 'stdout' in data and data['stdout']:
                lines.append(f"Output:\n{data['stdout']}")
                processed_keys.add('stdout')
            if 'stderr' in data and data['stderr']:
                lines.append(f"Error Output:\n{data['stderr']}")
                processed_keys.add('stderr')
        else:
            processed_keys.update(['stdout', 'stderr', 'command', 'working_directory'])
        
        # Handle remaining fields generically
        remaining = {k: v for k, v in data.items() if k not in processed_keys}
        for key, value in remaining.items():
            lines.append(self._format_generic_field(key, value))
        
        return '\n'.join(lines) if lines else str(data)
    
    def _format_content_field(self, content: Any) -> str:
        """Format content field with basic truncation info if available."""
        if not isinstance(content, str):
            return f"Content: {content}"
        
        # Simple content formatting - let the content speak for itself
        return f"Content:\n{content}"
    
    def _format_generic_field(self, key: str, value: Any) -> str:
        """Format a generic field with reasonable truncation for large data."""
        if isinstance(value, (list, dict)):
            if isinstance(value, list) and len(value) > 10:
                return f"{key}: [List with {len(value)} items - first few: {value[:3]}...]"
            elif isinstance(value, dict) and len(str(value)) > 1000:
                return f"{key}: [Dict with {len(value)} keys - too large to display fully]"
            else:
                return f"{key}: {json.dumps(value, indent=2, ensure_ascii=False)}"
        elif isinstance(value, str) and len(value) > 1000:
            # Handle large strings (possibly base64 data)
            if key == 'data' and len(value) > 500:
                preview = value[:50] + f"... [Total: {len(value)} chars]"
                return f"{key}: {preview}"
            else:
                preview = value[:200] + "..." if len(value) > 200 else value
                return f"{key}: {preview}"
        else:
            return f"{key}: {value}"

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
                # Check if tool_params is a dictionary before calling .items()
                if isinstance(tool_params, dict):
                    for key, value in tool_params.items():
                        if key in ['target_file', 'query', 'command', 'relative_workspace_path', 'search_term', 'instructions']:
                            # Show full parameter values without truncation
                            key_params.append(f"{key}={value}")
                    if key_params:
                        message_parts.append(f"**Parameters:** {', '.join(key_params)}")
                else:
                    # If tool_params is not a dict (e.g., string), just show it as is
                    message_parts.append(f"**Parameters:** {tool_params}")
            
            # Format the result using _format_dict_as_text for all cases
            message_parts.append("**Result:**")
            if isinstance(tool_result, dict):
                formatted_result = self._format_dict_as_text(tool_result, for_terminal_display=False, tool_name=tool_name, tool_params=tool_params)
                message_parts.append(formatted_result)
            else:
                # Handle non-dict results
                result_str = str(tool_result)
                message_parts.append(result_str)
                # Check if this is a get_sensor_data operation for logging
                is_sensor_data = (tool_name == 'get_sensor_data')
                if is_sensor_data:
                    print_current(f"📸 Full sensor data (non-dict) passed to LLM, length: {len(result_str)} characters")
            
            # Add separator between tools
            if i < len(tool_results):
                message_parts.append("")  # Empty line for separation
        
        return '\n'.join(message_parts)
  

    def _format_search_result_for_terminal(self, data: Dict[str, Any], tool_name: str) -> str:
        """
        Format search results (workspace_search and web_search) for simplified terminal display.
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
        
        # Handle workspace_search results
        if tool_name == 'workspace_search':
            query = data.get('query', 'unknown')
            results = data.get('results', [])
            total_results = len(results)
            
            # Show all results (up to 10) with brief info
            for i, result in enumerate(results[:10], 1):
                if isinstance(result, dict):
                    file_path = result.get('file', 'unknown')
                    start_line = result.get('start_line', '')
                    # Show only first 100 characters of snippet
                    snippet = result.get('snippet', '')[:100].replace('\n', ' ').strip()
                    if len(result.get('snippet', '')) > 100:
                        snippet += "..."
                    
                    lines.append(f"  {i}. {file_path}:{start_line} - {snippet}")
            
            if total_results > 10:
                lines.append(f"  ... and {total_results - 10} more results")
            
        # Handle web_search results
        elif tool_name == 'web_search':
            search_term = data.get('search_term', 'unknown')
            results = data.get('results', [])
            
            # Get total results count from various possible fields
            total_results = data.get('total_results')  # First try the direct field
            if total_results is None:
                # Handle cases where results were replaced with summary
                if data.get('detailed_results_replaced_with_summary'):
                    total_results = data.get('total_results_processed', 0)
                    simplified_results = data.get('simplified_results', [])
                    # Use simplified results for display if original results were removed
                    if not results and simplified_results:
                        results = simplified_results
                else:
                    total_results = len(results)
            
            lines.append(f"🌐 Web search for '{search_term}': Found {total_results} results")
            
            # Show only first 3 results with very brief info
            for i, result in enumerate(results[:3], 1):
                if isinstance(result, dict):
                    title = result.get('title', 'No Title')[:200]  # Limit title length
                    if len(result.get('title', '')) > 200:
                        title += "..."
                    
                    # Show brief snippet or content summary
                    content_preview = ""
                    if result.get('snippet'):
                        content_preview = result['snippet'][:200].replace('\n', ' ').strip()
                    elif result.get('content_summary'):
                        content_preview = result['content_summary'][:200].replace('\n', ' ').strip()
                    elif result.get('content'):
                        content_preview = result['content'][:200].replace('\n', ' ').strip()
                    
                    if content_preview and len(content_preview) >= 200:
                        content_preview += "..."
                    
                    lines.append(f"  {i}. {title}")
                    if content_preview:
                        lines.append(f"     {content_preview}")
            
            if total_results > 3:
                lines.append(f"...")
            
        
        # For other tools or unrecognized search results, fall back to original formatting
        else:
            return self._format_dict_as_text(data, for_terminal_display=True, tool_name=tool_name)
        
        return '\n'.join(lines)

    def _convert_tools_to_standard_format(self, provider="openai"):
        """
        Convert current tool_map to standard tool calling format.
        
        Args:
            provider: "openai" or "anthropic"
            
        Returns:
            List of tools in standard format
        """
        standard_tools = []
        
        # Load tool definitions from JSON file
        tool_definitions = self._load_tool_definitions_from_file()
        
        # Get tool source mapping
        tool_source_map = getattr(self, 'tool_source_map', {})
        
        # Convert to standard format based on provider
        for tool_name in self.tool_map.keys():
            tool_source = tool_source_map.get(tool_name, 'regular')
            
            # Handle FastMCP tools
            if tool_source == 'fastmcp':
                try:
                    from tools.fastmcp_wrapper import get_fastmcp_wrapper

                    fastmcp_wrapper = get_fastmcp_wrapper(config_path=self.MCP_config_file, workspace_dir=self.workspace_dir)
                    if fastmcp_wrapper and getattr(fastmcp_wrapper, 'initialized', False):
                        # Get tool definition from FastMCP wrapper
                        fastmcp_tool_def = fastmcp_wrapper.get_tool_definition(tool_name)
                        if fastmcp_tool_def:
                            if provider == "openai":
                                # OpenAI format for FastMCP tools
                                standard_tool = {
                                    "type": "function",
                                    "function": {
                                        "name": tool_name,
                                        "description": fastmcp_tool_def.get("description", f"FastMCP tool: {tool_name}"),
                                        "parameters": fastmcp_tool_def.get("input_schema", {
                                            "type": "object",
                                            "properties": {},
                                            "required": []
                                        })
                                    }
                                }
                            elif provider == "anthropic":
                                # Anthropic format for FastMCP tools
                                standard_tool = {
                                    "name": tool_name,
                                    "description": fastmcp_tool_def.get("description", f"FastMCP tool: {tool_name}"),
                                    "input_schema": fastmcp_tool_def.get("input_schema", {
                                        "type": "object",
                                        "properties": {},
                                        "required": []
                                    })
                                }
                            
                            standard_tools.append(standard_tool)
                except Exception as e:
                    print_current(f"⚠️ Failed to get FastMCP tool {tool_name} definition for standard format: {e}")
            
            # Handle cli-mcp tools
            elif tool_source == 'cli_mcp':
                if self.cli_mcp_client and self.cli_mcp_initialized:
                    try:
                        # Use tool name directly (no prefix for cli-mcp tools now)
                        cli_mcp_tool_def = self.cli_mcp_client.get_tool_definition(tool_name)
                        if cli_mcp_tool_def:
                            if provider == "openai":
                                # OpenAI format for cli-mcp tools
                                standard_tool = {
                                    "type": "function",
                                    "function": {
                                        "name": tool_name,  # Use original name (no prefix)
                                        "description": cli_mcp_tool_def.get("description", f"cli-mcptool: {tool_name}"),
                                        "parameters": cli_mcp_tool_def.get("input_schema", {
                                            "type": "object",
                                            "properties": {},
                                            "required": []
                                        })
                                    }
                                }
                            elif provider == "anthropic":
                                # Anthropic format for cli-mcp tools
                                standard_tool = {
                                    "name": tool_name,  # Use original name (no prefix)
                                    "description": cli_mcp_tool_def.get("description", f"cli-mcp tool: {tool_name}"),
                                    "input_schema": cli_mcp_tool_def.get("input_schema", {
                                        "type": "object",
                                        "properties": {},
                                        "required": []
                                    })
                                }
                            
                            standard_tools.append(standard_tool)
                    except Exception as e:
                        print_current(f"⚠️ Failed to get SSE MCP tool {tool_name} definition: {e}")
            # Handle regular tools from JSON definitions
            elif tool_name in tool_definitions:
                tool_def = tool_definitions[tool_name]
                
                if provider == "openai":
                    # OpenAI format
                    standard_tool = {
                        "type": "function",
                        "function": {
                            "name": tool_name,
                            "description": tool_def["description"],
                            "parameters": tool_def["parameters"]
                        }
                    }
                elif provider == "anthropic":
                    # Anthropic format (uses input_schema instead of parameters)
                    standard_tool = {
                        "name": tool_name,
                        "description": tool_def["description"],
                        "input_schema": tool_def["parameters"]
                    }
                
                standard_tools.append(standard_tool)
 
        
        return standard_tools

    def _call_llm_with_standard_tools(self, messages, system_message):
        """
        Call LLM with either standard tool calling format or chat-based tool calling.
        
        Args:
            messages: Messages including system, user, history, and execution instructions
            
        Returns:
            Tuple of (content, tool_calls)
        """
        if self.use_chat_based_tools:
            if self.streaming:
                if self.is_claude:
                    return call_claude_with_chat_based_tools_streaming(self, messages, system_message)
                else:
                    return call_openai_with_chat_based_tools_streaming(self, messages, system_message)
            else:
                if self.is_claude:
                    return call_claude_with_chat_based_tools_non_streaming(self, messages, system_message)
                else:
                    return call_openai_with_chat_based_tools_non_streaming(self, messages, system_message)
        elif self.is_claude:
            return call_claude_with_standard_tools(self, messages, system_message)
        else:
            return call_openai_with_standard_tools(self, messages, system_message)

    def _get_tool_name_from_call(self, tool_call):
        """
        Extract tool name from different tool call formats.
        
        Args:
            tool_call: Tool call in various formats (OpenAI, Anthropic, or chat-based)
            
        Returns:
            Tool name string
        """
        if isinstance(tool_call, dict):
            # OpenAI format: {"id": "...", "type": "function", "function": {"name": "...", "arguments": "..."}}
            if "function" in tool_call and isinstance(tool_call["function"], dict):
                return tool_call["function"]["name"]
            # Anthropic/Chat-based format: {"id": "...", "name": "...", "input": {...}}
            elif "name" in tool_call:
                return tool_call["name"]
        else:
            # Handle OpenAI API raw object format as fallback
            if hasattr(tool_call, 'function') and hasattr(tool_call.function, 'name'):
                return tool_call.function.name
            elif hasattr(tool_call, 'name'):
                return tool_call.name
        
        raise ValueError(f"Unknown tool call format: {tool_call}")

    def _get_tool_params_from_call(self, tool_call):
        """
        Extract tool parameters from different tool call formats.
        
        Args:
            tool_call: Tool call in various formats (OpenAI, Anthropic, or chat-based)
            
        Returns:
            Tool parameters dict
        """
        if isinstance(tool_call, dict):
            # OpenAI format: {"id": "...", "type": "function", "function": {"name": "...", "arguments": "..."}}
            if "function" in tool_call and isinstance(tool_call["function"], dict):
                return tool_call["function"]["arguments"]
            # Anthropic/Chat-based format: {"id": "...", "name": "...", "input": {...}}
            elif "input" in tool_call:
                return tool_call["input"]
        else:
            # Handle OpenAI API raw object format as fallback
            if hasattr(tool_call, 'function') and hasattr(tool_call.function, 'arguments'):
                return tool_call.function.arguments
            elif hasattr(tool_call, 'input'):
                return tool_call.input
        
        raise ValueError(f"Unknown tool call format: {tool_call}")

    def _format_tool_calls_for_history(self, tool_calls: List[Dict[str, Any]]) -> str:
        """
        Format tool calls for inclusion in history records.
        
        Args:
            tool_calls: List of tool calls in standard format
            
        Returns:
            Formatted string representation of tool calls
        """
        if not tool_calls:
            return ""
        
        formatted_calls = []
        
        for i, tool_call in enumerate(tool_calls, 1):
            tool_name = self._get_tool_name_from_call(tool_call)
            tool_params = self._get_tool_params_from_call(tool_call)
            
            formatted_calls.append(f"")
            formatted_calls.append(f"Tool {i}: {tool_name}")
            
            # Format parameters in a readable way
            if tool_params:
                formatted_calls.append("Parameters:")
                # Check if tool_params is a dictionary before calling .items()
                if isinstance(tool_params, dict):
                    for key, value in tool_params.items():
                        # Special handling for edit_file tool's code parameters
                        if tool_name == "edit_file" and key in ["old_code", "code_edit"]:
                            display_value = self._truncate_code_parameter(str(value))
                        # Special handling for talk_to_user: skip query parameter (will be printed by the tool itself)
                        elif tool_name == "talk_to_user" and key == "query":
                            # Skip printing query content to avoid duplication
                            continue
                        else:
                            # Show complete tool calls without truncation for better debugging
                            display_value = value
                            formatted_calls.append(f"  - {key}: {display_value}")
                else:
                    # If tool_params is not a dict (e.g., JSON string), show it as is
                    formatted_calls.append(f"  {tool_params}")
            else:
                formatted_calls.append("Parameters: None")
        
        return "\n".join(formatted_calls)

    def _truncate_code_parameter(self, code_content: str, max_lines: int = 1) -> str:
        """
        Truncate code content to show only the first few lines with ellipsis.

        Args:
            code_content: The code content to truncate
            max_lines: Maximum number of lines to show (default: 1)

        Returns:
            Truncated code content with ellipsis if needed
        """
        if not code_content:
            return code_content

        lines = code_content.split('\n')

        if len(lines) <= max_lines:
            return code_content

        # Show first max_lines lines
        truncated_lines = lines[:max_lines]
        result = '\n'.join(truncated_lines)

        # Add ellipsis to indicate truncation (on the same line)
        if result:
            result += '...'

        return result

    def _load_tool_definitions_from_file(self, json_file_path: str = None, force_reload: bool = False) -> Dict[str, Any]:
        """
        Load tool definitions from JSON file with caching to avoid repeated loading.
        
        Args:
            json_file_path: Path to the JSON file containing tool definitions
            force_reload: Whether to force reload even if cache exists
            
        Returns:
            Dictionary containing tool definitions
        """
        try:
            import json
            
            # Check cache first (unless force_reload is True)
            if not force_reload and self._tool_definitions_cache is not None:
                # Check if cache is still valid (within 60 seconds)
                current_time = time.time()
                if (self._tool_definitions_cache_timestamp is not None and 
                    current_time - self._tool_definitions_cache_timestamp < 60):
                    # Using cached tool definitions (avoiding repeated FastMCP loading)
                    return self._tool_definitions_cache
            
            # Load basic tool definitions
            tool_definitions = {}
            
            # Use default path if none provided
            if json_file_path is None:
                json_file_path = os.path.join(self.prompts_folder, "tool_prompt.json")
            
            # Try to load from the provided path
            if os.path.exists(json_file_path):
                with open(json_file_path, 'r', encoding='utf-8') as f:
                    tool_definitions = json.load(f)
            else:
                # No fallback definitions available
                tool_definitions = {}
            
            # Load memory tool definitions
            memory_tools_file = os.path.join(self.prompts_folder, "memory_tools.json")
            if os.path.exists(memory_tools_file):
                try:
                    with open(memory_tools_file, 'r', encoding='utf-8') as f:
                        memory_tools = json.load(f)
                        tool_definitions.update(memory_tools)
                except Exception as e:
                    print_current(f"⚠️ Error loading memory tools: {e}")
            else:
                # print_current(f"⚠️ Memory tools file not found: {memory_tools_file}")
                pass
            
            # Check if multi-agent mode is enabled
            multi_agent_enabled = self._is_multi_agent_enabled()
            
            if multi_agent_enabled:
                # Load multi-agent tool definitions from custom prompts folder
                multiagent_file_path = os.path.join(self.prompts_folder, "multiagent_tool_prompt.json")
                if os.path.exists(multiagent_file_path):
                    with open(multiagent_file_path, 'r', encoding='utf-8') as f:
                        multiagent_tools = json.load(f)
                        tool_definitions.update(multiagent_tools)
                        # print_current(f"✅ Loaded multi-agent tool definitions from {multiagent_file_path}")
                else:
                    # print_current(f"⚠️  Multi-agent tool definitions file not found: {multiagent_file_path}")
                    pass
            else:
                # print_current("🔒 Multi-agent mode disabled - skipping multi-agent tool definitions")
                pass
            
            # Load FastMCP tool definitions dynamically
            try:
                from tools.fastmcp_wrapper import get_fastmcp_wrapper

                fastmcp_wrapper = get_fastmcp_wrapper(config_path=self.MCP_config_file, workspace_dir=self.workspace_dir)
                if fastmcp_wrapper and getattr(fastmcp_wrapper, 'initialized', False):
                    fastmcp_tools = fastmcp_wrapper.get_available_tools()
                    if fastmcp_tools:
                        # Only print loading info on first load or force reload
                        should_print = force_reload or not hasattr(self, '_fastmcp_loaded_before')
                        
                        if should_print:
                            print_debug(f"🔧 Loading {len(fastmcp_tools)} FastMCP tool definitions")
                        
                        for tool_name in fastmcp_tools:
                            try:
                                # Get tool definition from FastMCP wrapper
                                tool_def = fastmcp_wrapper.get_tool_definition(tool_name)
                                if tool_def:
                                    # Convert to the format expected by our tool definitions
                                    tool_definitions[tool_name] = {
                                        "description": tool_def.get("description", f"FastMCP tool: {tool_name}"),
                                        "parameters": {
                                            "type": "object",
                                            "properties": tool_def.get("input_schema", {}).get("properties", {}),
                                            "required": tool_def.get("input_schema", {}).get("required", [])
                                        }
                                    }
                                    if should_print:
                                        print_debug(f"✅ Added FastMCP tool: {tool_name}")
                            except Exception as e:
                                print_debug(f"⚠️ Failed to load FastMCP tool definition for {tool_name}: {e}")
                                continue
                        
                        if should_print:
                            print_debug(f"✅ FastMCP tool definitions loaded successfully")
                        
                        # Mark that we've loaded FastMCP tools before
                        self._fastmcp_loaded_before = True
                else:
                    # If FastMCP is not initialized yet, invalidate cache to force reload later
                    if not force_reload:
                        print_debug("⚠️ FastMCP not initialized yet, will retry on next tool definition load")
                        self._tool_definitions_cache = None
                        self._tool_definitions_cache_timestamp = None
                    
            except Exception as e:
                print_debug(f"⚠️ Failed to load FastMCP tool definitions: {e}")
            
            # Load cli-mcp tool definitions dynamically
            try:
                if hasattr(self, 'tool_source_map'):
                    cli_mcp_tools = [tool_name for tool_name, source in self.tool_source_map.items() 
                                   if source == 'cli_mcp']
                    
                    if cli_mcp_tools and self.cli_mcp_initialized and self.cli_mcp_client:
                        print_debug(f"🔧 Loading {len(cli_mcp_tools)} cli-mcp tool definitions")
                        
                        for tool_name in cli_mcp_tools:
                            try:
                                # Get tool definition from cli-mcp wrapper
                                tool_def = self.cli_mcp_client.get_tool_definition(tool_name)
                                if tool_def:
                                    # Convert to the format expected by our tool definitions
                                    tool_definitions[tool_name] = {
                                        "description": tool_def.get("description", f"cli-mcp tool: {tool_name}"),
                                        "parameters": {
                                            "type": "object",
                                            "properties": tool_def.get("input_schema", {}).get("properties", {}),
                                            "required": tool_def.get("input_schema", {}).get("required", [])
                                        }
                                    }
                                    print_debug(f"✅ Added cli-mcp tool definition: {tool_name}")
                            except Exception as e:
                                print_debug(f"⚠️ Failed to load cli-mcp tool definition for {tool_name}: {e}")
                                continue
                        
                        print_debug(f"✅ cli-mcp tool definitions loaded successfully")
                    
            except Exception as e:
                print_debug(f"⚠️ Failed to load cli-mcp tool definitions: {e}")
            
            # Load dynamic tools from current_tool_list.json
            try:
                current_tool_list_path = os.path.join(self.workspace_dir, "current_tool_list.json")
                if os.path.exists(current_tool_list_path):
                    try:
                        with open(current_tool_list_path, 'r', encoding='utf-8') as f:
                            current_tool_list = json.load(f)
                        
                        # Check if file is not empty
                        if current_tool_list and isinstance(current_tool_list, dict) and len(current_tool_list) > 0:
                            # Merge dynamic tools into tool definitions
                            tool_definitions.update(current_tool_list)
                            # Clear the file after loading
                            with open(current_tool_list_path, 'w', encoding='utf-8') as f:
                                json.dump({}, f)
                    except json.JSONDecodeError as e:
                        print_debug(f"⚠️ Failed to parse current_tool_list.json: {e}")
                    except Exception as e:
                        print_debug(f"⚠️ Failed to load current_tool_list.json: {e}")
            except Exception as e:
                print_debug(f"⚠️ Error processing current_tool_list.json: {e}")
            
            # Cache the loaded tool definitions
            self._tool_definitions_cache = tool_definitions
            self._tool_definitions_cache_timestamp = time.time()
            
            return tool_definitions
                
        except json.JSONDecodeError as e:
            print_current(f"❌ Error parsing JSON in {json_file_path}: {e}")
        except Exception as e:
            print_current(f"❌ Error loading tool definitions from {json_file_path}: {e}")
        
        # Return empty definitions if file loading fails
        # print_current("🔄 No fallback tool definitions available")
        return {}
    
    def _clear_tool_definitions_cache(self):
        """Clear the tool definitions cache to force reload on next access."""
        self._tool_definitions_cache = None
        self._tool_definitions_cache_timestamp = None
    
    def _is_multi_agent_enabled(self) -> bool:
        """
        Check if multi-agent mode is enabled from configuration or environment variable.
        This method checks environment variable first (for GUI override), then falls back to config file.
        
        Returns:
            True if multi-agent mode is enabled, False otherwise
        """
        try:
            # Use get_multi_agent() which checks environment variable first, then config file
            from config_loader import get_multi_agent
            return get_multi_agent()
                
        except Exception as e:
            print_current(f"⚠️  Error checking multi-agent configuration: {e}")
            # Default to True if configuration cannot be read
            return True
    
    # Tool prompt generation function moved to utils/parse.py
    
    def _build_tool_and_env_message(self, user_prompt: str) -> Any:
        """
        Build user message with new architecture:
        1. Pure user requirement (first)
        2. Rules and tools prompts  
        3. System environment info
        4. Workspace info
        5. Execution instructions (last)
        
        Note: Task history is handled separately in execute_subtask by calling _add_history_to_llm_input_message.
        
        Args:
            user_prompt: Current user prompt (pure requirement)
            task_history: Previous task execution history (deprecated, kept for compatibility but not used)
            execution_round: Current execution round number
            
        Returns:
            Structured user message (string)
        """
        # Read unread inbox messages and merge into user requirement
        inbox_messages_content = self._get_and_read_inbox_messages()
        
        message_parts = []
        
        # 1. Pure user requirement (first) - merge with inbox messages
        # IMPORTANT: Inbox messages should be prioritized and clearly marked as NEW incoming messages
        if inbox_messages_content:
            # If there are inbox messages, display them prominently as NEW INCOMING MESSAGES
            message_parts.append("=" * 60)
            message_parts.append("🔔 URGENT: NEW MESSAGES RECEIVED FROM INBOX!")
            message_parts.append("=" * 60)
            message_parts.append("⚠️ ATTENTION: You have just received the following messages.")
            message_parts.append("⚠️ These are REAL-TIME messages that require your IMMEDIATE attention and action.")
            message_parts.append("")
            message_parts.append(inbox_messages_content)
            message_parts.append("")
            message_parts.append("=" * 60)
            message_parts.append("👆 END OF NEW MESSAGES - Please respond to the above messages immediately!")
            message_parts.append("=" * 60)
            message_parts.append("")
            # Original prompt follows as additional context
            if user_prompt.strip():
                message_parts.append("Original Task Context:")
                message_parts.append(user_prompt)
        else:
            message_parts.append(user_prompt)
        message_parts.append("")  # Empty line for separation
        
        # 2. Load and add rules and tools prompts
        prompt_components = self.load_user_prompt_components()
        
        if prompt_components['rules_and_tools']:
            message_parts.append("---")
            message_parts.append("")
            message_parts.append(prompt_components['rules_and_tools'])
            message_parts.append("")
        
        # 3. System environment information
        if prompt_components['system_environment']:
            message_parts.append("---")
            message_parts.append("")
            message_parts.append(prompt_components['system_environment'])
            message_parts.append("")
        
        # 3.5. Mailbox status information removed - inbox messages are now merged into user requirement above
        
        # 4. Workspace information
        if prompt_components['workspace_info']:
            message_parts.append("---")
            message_parts.append("")
            message_parts.append(prompt_components['workspace_info'])
            message_parts.append("")
            
        # Build final message
        final_message = "\n".join(message_parts)

        return final_message

    def _get_and_read_inbox_messages(self) -> Optional[str]:
        """
        Get unread inbox messages content and mark them as read.
        Returns formatted message content string, or None if no messages.
        
        Returns:
            Formatted message content string if there are unread messages, None otherwise
        """
        try:

            # Get current agent ID
            from src.tools.agent_context import get_current_agent_id
            current_agent_id = get_current_agent_id()
            
            # If no agent_id is set, treat as manager
            if not current_agent_id:
                current_agent_id = "manager"
            
            if self.debug_mode:
                print_current(f"🔍 [DEBUG] Current agent ID: {current_agent_id}")
            
            # Get message router and mailbox
            from src.tools.message_system import get_message_router
            router = get_message_router(self.workspace_dir, cleanup_on_init=False)
            if not router:
                if self.debug_mode:
                    print_current(f"⚠️ [DEBUG] Failed to get message router")
                return None
            
            mailbox = router.get_mailbox(current_agent_id)
            if not mailbox:
                # Try to register the agent
                try:
                    mailbox = router.register_agent(current_agent_id)
                    if not mailbox:
                        if self.debug_mode:
                            print_current(f"⚠️ [DEBUG] Failed to get mailbox for agent {current_agent_id}")
                        return None
                except Exception as e:
                    if self.debug_mode:
                        import traceback
                        print_current(f"⚠️ [DEBUG] Exception while registering mailbox for {current_agent_id}: {e}")
                        print_current(f"⚠️ [DEBUG] Traceback: {traceback.format_exc()}")
                    return None
            
            # Get unread messages
            unread_messages = mailbox.get_unread_messages()
            if self.debug_mode:
                print_current(f"🔍 [DEBUG] Found {len(unread_messages) if unread_messages else 0} unread messages")
            if not unread_messages:
                return None
            
            # Format messages content - extract only the text content
            message_contents = []
            for i, message in enumerate(unread_messages, 1):
                # Extract content text
                content = message.content
                if isinstance(content, dict):
                    # Try common content fields
                    text_content = content.get('text') or content.get('message') or content.get('content')
                    if text_content:
                        # Format message based on sender - user messages are primary requirements
                        if message.sender_id == "user":
                            message_contents.append(f"**USER REQUIREMENT {i}:** {text_content}")
                        else:
                            message_contents.append(f"**Message {i} from {message.sender_id}:** {text_content}")
                    else:
                        # Fallback: format the whole content dict
                        message_contents.append(f"**Message {i} from {message.sender_id}:** {str(content)}")
                else:
                    message_contents.append(f"**Message {i} from {message.sender_id}:** {str(content)}")
                
                # Mark message as read
                try:
                    mailbox.mark_as_read(message.message_id)
                    print_current(f"✅ Marked inbox message {message.message_id} as read")
                except Exception as e:
                    print_current(f"⚠️ Warning: Could not mark message {message.message_id} as read: {e}")
            
            if message_contents:
                formatted_content = "\n\n".join(message_contents)
                print_current(f"📬 Read {len(unread_messages)} unread message(s) from inbox and merged into user requirement")
                return formatted_content
            
            return None
            
        except Exception as e:
            # Silently fail to avoid disrupting normal operation
            if self.debug_mode:
                print_current(f"⚠️ Error reading inbox messages: {e}")
            return None

    def enhanced_tool_help(self, tool_name: str, **kwargs) -> Dict[str, Any]:
        """
        Enhanced tool_help that supports both built-in tools and MCP tools.
        
        Args:
            tool_name: The tool name to get help for
            
        Returns:
            Dictionary containing comprehensive tool usage information
        """
        # Ignore additional parameters
        if kwargs:
            print_current(f"⚠️ Ignoring additional parameters: {list(kwargs.keys())}")
        
        # First check if it's a built-in tool
        try:
            builtin_help = self.tools.tool_help(tool_name)
            if 'error' not in builtin_help:
                builtin_help['tool_type'] = 'built-in'
                return builtin_help
        except Exception as e:
            print_current(f"⚠️ Error getting built-in tool help: {e}")
        
        # Check if it's an MCP tool
        mcp_tool_def = self._get_mcp_tool_definition(tool_name)
        if mcp_tool_def:
            help_info = {
                "tool_name": tool_name,
                "tool_type": mcp_tool_def.get("tool_type", "mcp"),
                "description": mcp_tool_def["description"],
                "parameters": mcp_tool_def["parameters"],
                "usage_example": self._generate_mcp_usage_example(tool_name, mcp_tool_def),
                "parameter_template": self._generate_parameter_template(mcp_tool_def["parameters"]),
                "notes": mcp_tool_def.get("notes", "This is an MCP (Model Context Protocol) tool."),
                "mcp_format_warning": "⚠️ MCP tools typically use camelCase parameter format (e.g. entityType) rather than snake_case (e.g. entity_type). Please refer to the usage_example for the correct format."
            }
            
            return help_info
        
        # Tool not found - get all available tools including MCP tools
        all_tools = self._get_all_available_tools()
        available_tools = list(all_tools.keys())
        
        return {
            "error": f"Tool '{tool_name}' not found",
            "available_tools": available_tools,
            "all_tools_with_descriptions": all_tools,
            "message": f"Available tools are: {', '.join(available_tools)}",
            "suggestion": "Use list_available_tools() to see all available tools with descriptions"
        }

    def _get_mcp_tool_definition(self, tool_name: str) -> Optional[Dict[str, Any]]:
        """Get MCP tool definition from MCP clients"""
        try:
            # Check if it's a cli-mcp tool
            if hasattr(self, 'cli_mcp_client') and self.cli_mcp_client and self.cli_mcp_initialized:
                cli_mcp_tools = self.cli_mcp_client.get_available_tools()
                if tool_name in cli_mcp_tools:
                    tool_def = self.cli_mcp_client.get_tool_definition(tool_name)
                    if tool_def:
                        return {
                            "description": tool_def.get("description", f"cli-mcp tool: {tool_name}"),
                            "parameters": tool_def.get("input_schema", {}),
                            "notes": f"MCP tool (cli-mcp): {tool_name}. Please note to use the correct parameter format (usually camelCase).",
                            "tool_type": "cli-mcp"
                        }
        except Exception as e:
            print_current(f"⚠️ Error getting MCP tool definition: {e}")
        
        return None

    def _get_all_available_tools(self) -> Dict[str, str]:
        """Get all available tools including MCP tools"""
        all_tools = {}
        
        # Add built-in tools
        for tool_name in self.tool_map.keys():
            # Skip MCP tools here, we'll add them separately
            tool_source = getattr(self, 'tool_source_map', {}).get(tool_name, 'regular')
            if tool_source == 'regular':
                try:
                    help_info = self.tools.tool_help(tool_name)
                    if 'error' not in help_info:
                        description = help_info["description"]
                        first_sentence = description.split(".")[0] + "." if "." in description else description
                        if len(first_sentence) > 100:
                            first_sentence = first_sentence[:97] + "..."
                        all_tools[tool_name] = f"[Built-in] {first_sentence}"
                    else:
                        all_tools[tool_name] = f"[Built-in] {tool_name}"
                except:
                    all_tools[tool_name] = f"[Built-in] {tool_name}"
        
        # Add MCP tools
        try:
            # Add cli-mcp tools
            if hasattr(self, 'cli_mcp_client') and self.cli_mcp_client and self.cli_mcp_initialized:
                cli_mcp_tools = self.cli_mcp_client.get_available_tools()
                for tool_name in cli_mcp_tools:
                    try:
                        tool_def = self.cli_mcp_client.get_tool_definition(tool_name)
                        description = tool_def.get("description", f"cli-mcp tool: {tool_name}") if tool_def else f"cli-mcp tool: {tool_name}"
                        first_sentence = description.split(".")[0] + "." if "." in description else description
                        if len(first_sentence) > 100:
                            first_sentence = first_sentence[:97] + "..."
                        all_tools[tool_name] = f"[MCP/CLI] {first_sentence}"
                    except Exception as e:
                        all_tools[tool_name] = f"[MCP/CLI] {tool_name} (error getting definition)"
        
        except Exception as e:
            print_current(f"⚠️ Error getting MCP tool list: {e}")
        
        return all_tools

    def _generate_mcp_usage_example(self, tool_name: str, tool_def: Dict[str, Any]) -> str:
        """Generate usage example for MCP tools"""
        parameters = tool_def.get("parameters", {})
        properties = parameters.get("properties", {})
        required = parameters.get("required", [])
        
        # Build example arguments
        example_args = {}
        for param_name, param_info in properties.items():
            param_type = param_info.get("type", "string")
            
            # Generate appropriate example values
            if param_type == "array":
                if "entities" in param_name.lower():
                    example_args[param_name] = [{
                        "name": "Example Entity",
                        "entityType": "Person", 
                        "observations": ["Relevant observation info"]
                    }]
                else:
                    example_args[param_name] = ["example1", "example2"]
            elif param_type == "boolean":
                example_args[param_name] = True
            elif param_type == "integer":
                example_args[param_name] = 1
            elif param_type == "object":
                if "parameters" in param_name.lower():
                    example_args[param_name] = {"query": "search keywords"}
                else:
                    example_args[param_name] = {"key": "value"}
            else:
                # String type
                if "query" in param_name.lower():
                    example_args[param_name] = "search keywords"
                elif "path" in param_name.lower() or "file" in param_name.lower():
                    example_args[param_name] = "/path/to/file.txt"
                elif "content" in param_name.lower():
                    example_args[param_name] = "file content"
                elif "name" in param_name.lower():
                    example_args[param_name] = "example name"
                elif "type" in param_name.lower():
                    example_args[param_name] = "Person"
                else:
                    example_args[param_name] = "example value"
        
        # Special handling for known MCP tools
        if tool_name == "create_entities":
            example_args = {
                "entities": [{
                    "name": "user",
                    "entityType": "Person",
                    "observations": ["likes eating ice pops"]
                }]
            }
        elif tool_name == "write_file" or "write" in tool_name.lower():
            example_args = {
                "path": "/home/user/example.txt",
                "content": "This is example file content\nwith multiple lines"
            }
        elif tool_name == "read_file" or "read" in tool_name.lower():
            example_args = {
                "path": "/home/user/example.txt"
            }
        elif "search" in tool_name.lower():
            example_args = {
                "query": "search keywords",
                "language": "en",
                "num_results": 10
            }
        
        import json
        example_json = json.dumps(example_args, ensure_ascii=False, indent=2)
        
        return f'''{{
  "name": "{tool_name}",
  "arguments": {example_json}
}}

📝 MCP Tool Call Format Notes:
- Parameter names typically use camelCase format (e.g. entityType, numResults)
- Avoid using snake_case format (e.g. entity_type, num_results)
- Ensure parameter types match the tool definition correctly'''

    def _generate_parameter_template(self, parameters: Dict[str, Any]) -> str:
        """Generate a parameter template showing how to call the tool."""
        template_lines = []
        properties = parameters.get("properties", {})
        required_params = parameters.get("required", [])
        
        for param_name, param_info in properties.items():
            param_type = param_info.get("type", "string")
            description = param_info.get("description", "")
            is_required = param_name in required_params
            
            # Generate appropriate example values
            if param_type == "array":
                example_value = '["example1", "example2"]'
            elif param_type == "boolean":
                example_value = "true"
            elif param_type == "integer":
                example_value = "1"
            else:
                if "path" in param_name.lower() or "file" in param_name.lower():
                    example_value = "path/to/file.py"
                elif "command" in param_name.lower():
                    example_value = "ls -la"
                elif "query" in param_name.lower() or "search" in param_name.lower():
                    example_value = "search query"
                elif "url" in param_name.lower():
                    example_value = "https://example.com"
                elif "edit_mode" in param_name.lower():
                    example_value = '"replace_lines"'
                elif "start_line" in param_name.lower():
                    example_value = "10"
                elif "end_line" in param_name.lower():
                    example_value = "15"
                elif "position" in param_name.lower():
                    example_value = "15"
                else:
                    example_value = "value"
            
            required_marker = " (REQUIRED)" if is_required else " (OPTIONAL)"
            template_lines.append(f'"{param_name}": {example_value}  // {description}{required_marker}')
        
        return "{\n  " + ",\n  ".join(template_lines) + "\n}"

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
    
    args = parser.parse_args()
    
    # Handle streaming configuration
    streaming = None
    if args.streaming and args.no_streaming:
        print_current("Warning: Both --streaming and --no-streaming specified, using config.txt default")
    elif args.streaming:
        streaming = True
    elif args.no_streaming:
        streaming = False
    # If neither specified, streaming=None will use config.txt value
    
    # Check if prompt is provided for normal execution
    if not args.prompt:
        parser.error("prompt is required")
    
    # Create executor
    executor = ToolExecutor(
        api_key=args.api_key, 
        model=args.model,
        workspace_dir=args.workspace_dir,
        debug_mode=args.debug,
        logs_dir=args.logs_dir,
        streaming=streaming
    )
    
    # Execute subtask
    result = executor.execute_subtask(args.prompt, args.system_prompt)
    
    print(result)

if __name__ == "__main__":
    main()