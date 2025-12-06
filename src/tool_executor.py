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
import re
import argparse
import datetime
import sys
import base64
import mimetypes
import threading
import logging
import time
import warnings

# Suppress asyncio warnings that occur during FastMCP cleanup
warnings.filterwarnings("ignore", message=".*Event loop is closed.*")
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*coroutine.*was never awaited.*")

from typing import Dict, Any, List, Optional, Union, Tuple
from openai import OpenAI
from src.tools.print_system import (print_system, print_current, streaming_context,
                                print_debug, print_system, print_current, print_current,
                                print_error)
from src.tools.agent_context import get_current_agent_id
from src.tools.debug_system import track_operation, finish_operation
# 🚀 延迟导入优化：MCP 相关模块延迟加载，避免启动时加载 FastMCP 框架（~3秒）
# from src.tools.cli_mcp_wrapper import get_cli_mcp_wrapper, initialize_cli_mcp_wrapper, safe_cleanup_cli_mcp_wrapper
# 这些函数将在实际使用时才导入
from src.tools.mcp_client import safe_cleanup_mcp_client
from src.config_loader import get_api_key, get_api_base, get_model, get_max_tokens, get_streaming, get_language, get_truncation_length, get_simplified_search_output, get_web_search_summary, get_multi_agent, get_tool_calling_format, get_compression_min_length, get_compression_head_length, get_compression_tail_length
from src.tools.message_system import get_message_router

# Initialize logger
logger = logging.getLogger(__name__)

# Note: JSON parsing utilities now imported below with other utils

# Import get_info utilities
from utils.get_info import (
    get_system_environment_info,
    get_workspace_info,
    format_file_size,
    get_file_language,
)

# Import cache efficiency utilities
from utils.cacheeff import (
    analyze_cache_potential,
    estimate_token_count,
)

# Import parse utilities
from utils.parse import (
    fix_json_escapes,
    smart_escape_quotes_in_json_values,
    fix_json_string_values_robust,
    rebuild_json_structure,
    parse_python_params_manually,
    convert_parameter_value,
    generate_tools_prompt_from_json,
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

def _fix_json_boolean_values(json_str: str) -> str:
    """
    Fix boolean value formatting issues in a JSON string.
    Replace :True with :true and :False with :false.

    Args:
        json_str: The original JSON string.

    Returns:
        The corrected JSON string.
    """
    json_str = re.sub(r':\s*True\b', ': true', json_str)
    json_str = re.sub(r':\s*False\b', ': false', json_str)
    return json_str


def validate_tool_call_json(json_str: str, tool_name: str = "") -> Tuple[bool, Optional[Dict], str]:
    """
    Validate and parse the JSON parameters for a tool call, with auto-fix support for multiple formats.

    Args:
        json_str: The JSON string.
        tool_name: Name of the tool (for logging).

    Returns:
        Tuple of (is_valid, parsed_data, error_message)
    """
    if not json_str or not json_str.strip():
        return False, None, "Empty JSON string"

    # Try parsing directly first
    try:
        data = json.loads(json_str)
        return True, data, ""
    except json.JSONDecodeError as e:
        error_msg = f"JSON parsing failed"
        if tool_name:
            error_msg += f" (tool: {tool_name})"
        error_msg += f": {e.msg} at line {e.lineno} column {e.colno}"

        # Add detailed error context
        lines = json_str.split('\n')
        if 0 <= e.lineno - 1 < len(lines):
            error_line = lines[e.lineno - 1]
            error_msg += f"\nError line: {error_line}"
            if e.colno > 0:
                error_msg += f"\nPosition: {' ' * (e.colno - 1)}^"

        return False, None, error_msg
    except Exception as e:
        return False, None, f"Unexpected error: {str(e)}"


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
                 plan_mode: bool = False):
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
        
        # Check if using GLM model with Anthropic API
        self.is_glm = 'glm-' in self.model.lower() and 'anthropic' in self.api_base.lower()
        
        # Load tool calling format configuration from config/config.txt
        # True = standard tool calling, False = chat-based tool calling
        tool_calling_format = get_tool_calling_format()
        
        # Set use_chat_based_tools based on configuration
        # Note: use_chat_based_tools is the inverse of tool_calling_format
        self.use_chat_based_tools = not tool_calling_format
        
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
        self.memory_update_counter = 0  # 记忆更新计数器
        self.memory_update_interval = 10  # 每10轮更新一次记忆
        
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
        # print_system(f"   Simplified Search Output: {'✅ Enabled' if self.simplified_search_output else '❌ Disabled'} (Affects workspace_search and web_search terminal display)")  # Commented out to reduce terminal noise
        # if debug_mode:
        #     print_system(f"   Debug Mode: Enabled (Log directory: {logs_dir})")  # Commented out to reduce terminal noise
        
        # 🚀 LLM客户端延迟初始化：不在__init__中创建客户端
        self.client = None
        self._llm_client_initialized = False
        
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
        
        # 🚀 长期记忆系统延迟初始化：只在首次使用时才初始化
        self.long_term_memory = None
        self._long_term_memory_initialized = False
        
        # 🚀 Prompt缓存优化：避免重复读取文件和生成文本
        self._prompt_file_cache = {}  # {file_path: (content, mtime)}
        self._prompt_components_cache = None  # 缓存load_user_prompt_components的结果
        self._tools_prompt_cache = {}  # {(lang, def_hash): generated_prompt}
        
        # Initialize history optimizer for image data optimization
        try:
            from tools.history_optimizer import HistoryOptimizer
            self.history_optimizer = HistoryOptimizer(workspace_root=self.workspace_dir)
            # print_current("✅ History optimizer initialized for image data optimization")
        except ImportError as e:
            print_current(f"⚠️ Failed to import HistoryOptimizer: {e}")
            self.history_optimizer = None
        
        # Initialize multi-agent tools directly if enabled
        if self.multi_agent:
            from tools.multiagents import MultiAgentTools
            self.multi_agent_tools = MultiAgentTools(self.workspace_dir, debug_mode=self.debug_mode)
        else:
            self.multi_agent_tools = None
        
        # Store current round's image data for next round vision API
        self.current_round_images = []
        
        # Store previous messages for cache analysis
        self.previous_messages = []
        
        # Initialize simple history compressor for conversation history compression
        try:
            from tools.simple_history_compressor import SimpleHistoryCompressor
            # Load compression settings from config
            min_length = get_compression_min_length()
            head_length = get_compression_head_length()
            tail_length = get_compression_tail_length()
            self.simple_compressor = SimpleHistoryCompressor(
                min_length=min_length,
                head_length=head_length,
                tail_length=tail_length
            )
        except ImportError as e:
            print_system(f"⚠️ Failed to import SimpleHistoryCompressor: {e}, simple compression disabled")
            self.simple_compressor = None
        else:
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
            "list_dir": self.tools.list_dir,
            "grep_search": self.tools.grep_search,
            "edit_file": self.tools.edit_file,
            "file_search": self.tools.file_search,
            "web_search": self.tools.web_search,
            "search_img": self.tools.search_img,  # Add image search tool
            "tool_help": self.enhanced_tool_help,  # Use enhanced version that supports MCP tools
            "fetch_webpage_content": self.tools.fetch_webpage_content,
            "get_background_update_status": self.tools.get_background_update_status,
            "talk_to_user": self.tools.talk_to_user,
            "idle": self.tools.idle,
            "get_sensor_data": self.tools.get_sensor_data,
            "merge_file": self.tools.merge_file,  # Add file merging tool
            "parse_doc_to_md": self.tools.parse_doc_to_md,  # Add document parsing tool
            "convert_docs_to_markdown": self.tools.convert_docs_to_markdown,  # Add document conversion tool
            "mouse_control": self.tools.mouse_control,  # Add mouse control tool
        }
        
        # Add history compression tool if available
        if self.history_compression_tools:
            self.tool_map["compress_history"] = self.history_compression_tools.compress_history
        
        # Add planning tool if available
        if self.planning_tools:
            self.tool_map["plan_tools"] = self.planning_tools.plan_tools
        
        # 🚀 长期记忆工具延迟初始化包装器（仅在启用时注册）
        if self._is_long_term_memory_enabled():
            def _create_long_term_memory_wrapper(tool_name):
                """创建长期记忆工具的延迟初始化包装器"""
                def wrapper(*args, **kwargs):
                    # 确保长期记忆系统已初始化
                    self._ensure_long_term_memory_initialized()
                    
                    # 检查初始化是否成功
                    if self.long_term_memory is None:
                        return {"status": "error", "message": "Long-term memory system initialization failed"}
                    
                    # 调用实际的工具方法
                    if tool_name == "recall_memories":
                        return self.long_term_memory.recall_memories(*args, **kwargs)
                    elif tool_name == "recall_memories_by_time":
                        return self.long_term_memory.recall_memories_by_time(*args, **kwargs)
                    elif tool_name == "get_memory_summary":
                        return self.long_term_memory.get_memory_summary(*args, **kwargs)
                    else:
                        return {"status": "error", "message": f"Unknown long-term memory tool: {tool_name}"}
                return wrapper
            
            # 注册长期记忆工具（使用延迟初始化包装器）
            self.tool_map.update({
                "recall_memories": _create_long_term_memory_wrapper("recall_memories"),
                "recall_memories_by_time": _create_long_term_memory_wrapper("recall_memories_by_time"),
                "get_memory_summary": _create_long_term_memory_wrapper("get_memory_summary"),
            })
            print_debug("🧠 Long-term memory tools registered (lazy initialization)")
        else:
            print_debug("ℹ️ Long-term memory is disabled, skipping tool registration")
        
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
        
        # 🚀 MCP智能加载：检查是否配置了MCP服务器
        # 检查是否配置了MCP服务器
        has_mcp_servers = self._check_mcp_servers_configured()
        
        # 初始化MCP相关变量
        self.cli_mcp_client = None
        self.direct_mcp_client = None
        self.cli_mcp_initialized = False
        self.direct_mcp_initialized = False
        self.mcp_initialization_attempted = False
        
        if has_mcp_servers:
            # If MCP servers are configured, initialize at startup
            print_debug("🔌 MCP server configuration detected, initializing MCP at startup...")
            # Immediately initialize MCP (pass is_startup=True)
            self._ensure_mcp_initialized(is_startup=True)
        else:
            # If no MCP servers are configured, enable lazy loading
            print_debug("⏭️ No MCP server configuration detected, enabling lazy loading")
        
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
        self.llm_call_counter = 0  # LLM call counter
        
        # Ensure log directory exists only if logs_dir is set
        if self.llm_logs_dir:
            os.makedirs(self.llm_logs_dir, exist_ok=True)

    
    def _check_mcp_servers_configured(self) -> bool:
        """
        检查MCP配置文件中是否配置了MCP服务器
        
        Returns:
            True if MCP servers are configured, False otherwise
        """
        try:
            config_path = self.MCP_config_file
            
            # 检查配置文件是否存在
            if not os.path.exists(config_path):
                return False
            
            # 读取配置文件
            import json
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            # 检查是否有mcpServers配置
            mcp_servers = config.get("mcpServers", {})
            
            if not mcp_servers:
                return False
            
            # 检查是否有启用的服务器
            for server_name, server_config in mcp_servers.items():
                # 跳过禁用的服务器
                if server_config.get("enabled", True) is False:
                    continue
                
                # 检查是否有command或url配置（表示这是一个有效的服务器配置）
                if server_config.get("command") or server_config.get("url"):
                    return True
            
            # 没有找到有效的服务器配置
            return False
            
        except Exception as e:
            print_debug(f"⚠️ 检查MCP配置时出错: {e}")
            return False
    
    def _ensure_mcp_initialized(self, is_startup: bool = False):
        """
        确保MCP已初始化 - 延迟加载实现
        在第一次使用MCP工具时调用此方法，或在启动时检测到配置时调用
        
        Args:
            is_startup: 是否在启动时调用（True=启动时，False=首次使用时）
        """
        # 如果已经尝试过初始化,直接返回
        if self.mcp_initialization_attempted:
            return
        
        self.mcp_initialization_attempted = True
        
        if is_startup:
            print_system("🔄 检测到MCP服务器配置,开始初始化MCP客户端...")
        else:
            print_system("🔄 首次使用MCP工具,开始初始化MCP客户端...")
        
        # 🚀 在首次使用时创建MCP客户端实例
        try:
            from src.tools.cli_mcp_wrapper import get_cli_mcp_wrapper
            self.cli_mcp_client = get_cli_mcp_wrapper(self.MCP_config_file)
            
            from tools.mcp_client import MCPClient
            self.direct_mcp_client = MCPClient(self.MCP_config_file, workspace_dir=self.workspace_dir)
            
            print_debug("✅ MCP客户端实例创建成功")
        except Exception as e:
            print_current(f"⚠️ 创建MCP客户端实例失败: {e}")
            return
        
        # Initialize MCP clients with proper order: FastMCP first, then cli-mcp
        import asyncio
        try:
            loop = asyncio.get_running_loop()
            if loop and loop.is_running():
                # We're in an async context, use thread pool for initialization
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    # Initialize direct MCP client (FastMCP) FIRST
                    try:
                        future_direct = executor.submit(asyncio.run, self.direct_mcp_client.initialize())
                        self.direct_mcp_initialized = future_direct.result(timeout=10)
                        if self.direct_mcp_initialized:
                            print_system(f"✅ SSE MCP client (FastMCP) initialized with config: {self.MCP_config_file}")
                    except Exception as e:
                        print_current(f"⚠️ SSE MCP client initialization failed: {e}")
                        self.direct_mcp_initialized = False
                    
                    # Only initialize cli-mcp if FastMCP is not handling servers
                    should_init_cli_mcp = self._should_initialize_cli_mcp()
                    if should_init_cli_mcp:
                        try:
                            # 🚀 延迟导入：只在实际使用 MCP 时才加载
                            from src.tools.cli_mcp_wrapper import initialize_cli_mcp_wrapper
                            future_cli = executor.submit(asyncio.run, initialize_cli_mcp_wrapper(self.MCP_config_file))
                            self.cli_mcp_initialized = future_cli.result(timeout=10)
                            if self.cli_mcp_initialized:
                                print_system(f"✅ cli-mcp client initialized with config: {self.MCP_config_file}")
                        except Exception as e:
                            print_system(f"⚠️ cli-mcp client initialization failed: {e}")
                            self.cli_mcp_initialized = False
                    else:
                        print_system("⏭️ Skipping cli-mcp initialization (servers handled by FastMCP)")
                        self.cli_mcp_initialized = False
            else:
                # Safe to run async initialization directly
                # Initialize direct MCP client (FastMCP) FIRST
                try:
                    self.direct_mcp_initialized = asyncio.run(self.direct_mcp_client.initialize())
                    if self.direct_mcp_initialized:
                        print_system(f"✅ SSE MCP client (FastMCP) initialized with config: {self.MCP_config_file}")
                except Exception as e:
                    print_current(f"⚠️ SSE MCP client initialization failed: {e}")
                    self.direct_mcp_initialized = False
                
                # Only initialize cli-mcp if needed
                should_init_cli_mcp = self._should_initialize_cli_mcp()
                if should_init_cli_mcp:
                    try:
                        # 🚀 延迟导入：只在实际使用 MCP 时才加载
                        from src.tools.cli_mcp_wrapper import initialize_cli_mcp_wrapper
                        self.cli_mcp_initialized = asyncio.run(initialize_cli_mcp_wrapper(self.MCP_config_file))
                        if self.cli_mcp_initialized:
                            print_system(f"✅ cli-mcp client initialized with config: {self.MCP_config_file}")
                    except Exception as e:
                        print_current(f"⚠️ cli-mcp client initialization failed: {e}")
                        self.cli_mcp_initialized = False
                else:
                    print_system("⏭️ Skipping cli-mcp initialization (servers handled by FastMCP)")
                    self.cli_mcp_initialized = False
        except RuntimeError:
            # No event loop, safe to create one
            # Initialize direct MCP client (FastMCP) FIRST
            try:
                self.direct_mcp_initialized = asyncio.run(self.direct_mcp_client.initialize())
                if self.direct_mcp_initialized:
                    print_system(f"✅ SSE MCP client (FastMCP) initialized with config: {self.MCP_config_file}")
            except Exception as e:
                print_current(f"⚠️ SSE MCP client initialization failed: {e}")
                self.direct_mcp_initialized = False
            
            # Only initialize cli-mcp if needed
            should_init_cli_mcp = self._should_initialize_cli_mcp()
            if should_init_cli_mcp:
                try:
                    # 🚀 延迟导入：只在实际使用 MCP 时才加载
                    from src.tools.cli_mcp_wrapper import initialize_cli_mcp_wrapper
                    self.cli_mcp_initialized = asyncio.run(initialize_cli_mcp_wrapper(self.MCP_config_file))
                    if self.cli_mcp_initialized:
                        print_system(f"✅ cli-mcp client initialized with config: {self.MCP_config_file}")
                except Exception as e:
                    print_current(f"⚠️ cli-mcp client initialization failed: {e}")
                    self.cli_mcp_initialized = False
            else:
                print_system("⏭️ Skipping cli-mcp initialization (servers handled by FastMCP)")
                self.cli_mcp_initialized = False
        
        # Add MCP tools to tool_map after successful initialization
        if self.cli_mcp_initialized or self.direct_mcp_initialized:
            self._add_mcp_tools_to_map()
            print_system(f"🔧 MCP tools loaded successfully")
    
    async def _initialize_mcp_async(self):
        """Initialize both MCP clients asynchronously"""
        try:
            # Initialize cli-mcp wrapper
            if not self.cli_mcp_initialized:
                # 🚀 延迟导入：只在实际使用 MCP 时才加载
                from src.tools.cli_mcp_wrapper import initialize_cli_mcp_wrapper
                self.cli_mcp_initialized = await initialize_cli_mcp_wrapper(self.MCP_config_file)
                if self.cli_mcp_initialized:
                    print_system("✅ cli-mcp client initialized successfully")
                else:
                    print_current("⚠️ cli-mcp client initialization failed")
            
            # Initialize direct MCP client
            if not self.direct_mcp_initialized:
                self.direct_mcp_initialized = await self.direct_mcp_client.initialize()
                if self.direct_mcp_initialized:
                    print_current("✅ Direct MCP client initialized successfully")
                else:
                    print_current("⚠️ Direct MCP client initialization failed")
            
            # Add MCP tools to tool_map after initialization
            if self.cli_mcp_initialized or self.direct_mcp_initialized:
                self._add_mcp_tools_to_map()
                
        except Exception as e:
            print_current(f"⚠️ MCP client async initialization failed: {e}")
    
    def _ensure_long_term_memory_initialized(self):
        """
        确保长期记忆系统已初始化 - 延迟加载实现
        在第一次使用长期记忆工具时调用此方法
        """
        if self._long_term_memory_initialized:
            return
        
        self._long_term_memory_initialized = True
        
        # Check if long-term memory is disabled
        if os.environ.get('AGIBOT_LONG_TERM_MEMORY', '').lower() in ('false', '0', 'no', 'off'):
            print_debug("ℹ️ Long-term memory is disabled via environment variable")
            return
        
        print_system("🔄 首次使用长期记忆工具，开始初始化长期记忆系统...")
        
        try:
            from tools.long_term_memory import LongTermMemoryTools
            self.long_term_memory = LongTermMemoryTools(
                workspace_root=self.workspace_dir
            )
            print_system("✅ 长期记忆系统初始化成功")
        except ImportError as e:
            print_current(f"⚠️ Long-term memory module import failed: {e}")
            self.long_term_memory = None
        except Exception as e:
            print_current(f"⚠️ Long-term memory system initialization failed: {e}")
            self.long_term_memory = None
    
    def _should_initialize_cli_mcp(self) -> bool:
        """Check if cli-mcp should be initialized based on FastMCP status"""
        try:
            # If FastMCP is not available or not initialized, use cli-mcp
            if not self.direct_mcp_initialized:
                return True
            
            # Check if FastMCP has servers configured
            if hasattr(self.direct_mcp_client, 'fastmcp_wrapper') and self.direct_mcp_client.fastmcp_wrapper:
                fastmcp_wrapper = self.direct_mcp_client.fastmcp_wrapper
                
                # If FastMCP has no servers, allow cli-mcp
                if not hasattr(fastmcp_wrapper, 'servers') or not fastmcp_wrapper.servers:
                    return True
                
                # Check if there are servers that FastMCP cannot handle
                # Load cli-mcp config to compare
                import json
                import os
                
                cli_config_path = self.MCP_config_file if self.MCP_config_file else "mcp.json"
                if not os.path.exists(cli_config_path):
                    return False  # No cli-mcp config, no need to initialize
                
                try:
                    with open(cli_config_path, 'r', encoding='utf-8') as f:
                        cli_config = json.load(f)
                    
                    cli_servers = cli_config.get("mcpServers", {})
                    fastmcp_servers = set(fastmcp_wrapper.servers.keys())
                    
                    # Check if there are CLI servers not handled by FastMCP
                    for server_name in cli_servers.keys():
                        # Normalize server names for comparison
                        normalized_cli = server_name.replace('-', '_').replace('_', '-')
                        
                        is_handled = False
                        for fastmcp_server in fastmcp_servers:
                            normalized_fast = fastmcp_server.replace('-', '_').replace('_', '-')
                            if (server_name == fastmcp_server or 
                                normalized_cli == normalized_fast or
                                server_name.replace('-', '_') == fastmcp_server.replace('-', '_')):
                                is_handled = True
                                break
                        
                        if not is_handled:
                            print_current(f"📋 Found server {server_name} not handled by FastMCP, will initialize cli-mcp")
                            return True
                    
                    #print_current("📋 All servers handled by FastMCP, skipping cli-mcp")
                    return False
                    
                except Exception as e:
                    print_current(f"⚠️ Error reading cli-mcp config: {e}, defaulting to initialize cli-mcp")
                    return True
            
            return True  # Default to allowing cli-mcp if unsure
            
        except Exception as e:
            print_current(f"⚠️ Error checking cli-mcp initialization status: {e}, defaulting to initialize")
            return True

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
        
        # Add direct MCP client tools (SSE format) - but skip tools already handled
        if self.direct_mcp_initialized and self.direct_mcp_client:
            try:
                # Get available MCP tools from the direct MCP client
                direct_mcp_tools = self.direct_mcp_client.get_available_tools()
                
                if direct_mcp_tools:
                    direct_mcp_tools_added = []
                    for tool_name in direct_mcp_tools:
                        # Intelligently check if the SSE MCP tool is already handled by FastMCP
                        should_skip = False
                        skip_reason = ""
                        
                        # SSE MCP tool names are usually the server name
                        # Check 1: Is the server name handled by FastMCP?
                        for fastmcp_server in fastmcp_server_names:
                            if (tool_name == fastmcp_server or 
                                tool_name.replace('-', '_') == fastmcp_server.replace('-', '_')):
                                should_skip = True
                                skip_reason = f"server {fastmcp_server} handled by FastMCP"
                                break
                        
                        # Check 2: Is the tool name directly in the FastMCP tool list?
                        if not should_skip and tool_name in fastmcp_tools_added:
                            should_skip = True
                            skip_reason = f"exact tool name in FastMCP"
                        
                        # Check 3: If there are FastMCP tools and the server name matches the pattern
                        if not should_skip and fastmcp_tools_added:
                            for fastmcp_tool in fastmcp_tools_added:
                                # Check if the FastMCP tool is from the same server
                                if (tool_name in fastmcp_tool or 
                                    fastmcp_tool.startswith(tool_name.replace('-', '_')) or 
                                    tool_name.replace('-', '_') in fastmcp_tool):
                                    should_skip = True
                                    skip_reason = f"FastMCP tool {fastmcp_tool} from same server"
                                    break
                        
                        if should_skip:
                            logger.debug(f"Skipping SSE MCP tool {tool_name} ({skip_reason})")
                            continue
                            
                        # Create a wrapper function for each direct MCP tool
                        def create_direct_mcp_tool_wrapper(tool_name=tool_name):
                            def sync_direct_mcp_tool_wrapper(**kwargs):
                                import asyncio
                                try:
                                    # Call the direct MCP client
                                    return asyncio.run(self.direct_mcp_client.call_tool(tool_name, kwargs))
                                except Exception as e:
                                    return {"error": f"Direct MCP tool {tool_name} call failed: {e}"}
                            
                            return sync_direct_mcp_tool_wrapper
                        
                        # Add to tool mapping WITHOUT prefix for SSE tools
                        self.tool_map[tool_name] = create_direct_mcp_tool_wrapper()
                        self.tool_source_map[tool_name] = 'direct_mcp'
                        direct_mcp_tools_added.append(tool_name)
                    
                    if direct_mcp_tools_added:
                        logger.info(f"Added {len(direct_mcp_tools_added)} SSE MCP tools: {', '.join(direct_mcp_tools_added)}")
            except Exception as e:
                logger.error(f"Failed to add direct MCP tools to mapping: {e}")
                self.direct_mcp_initialized = False
    
    def cleanup(self):
        """Clean up all resources and threads"""
        try:
            import sys
            
            # Cleanup cli-mcp client
            # 🎯 性能优化：只有在 cli_mcp_wrapper 已被导入时才清理
            if hasattr(self, 'cli_mcp_client') and self.cli_mcp_client:
                # 检查模块是否已加载（避免在清理时延迟导入）
                if 'src.tools.cli_mcp_wrapper' in sys.modules:
                    try:
                        from src.tools.cli_mcp_wrapper import safe_cleanup_cli_mcp_wrapper
                        safe_cleanup_cli_mcp_wrapper()
                        # print_current("🔌 cli-mcp client cleanup completed")
                    except Exception as e:
                        print_current(f"⚠️ cli-mcp client cleanup failed: {e}")
            
            # Cleanup direct MCP client
            # 🎯 性能优化：只有在 mcp_client 已被导入时才清理
            if hasattr(self, 'direct_mcp_client') and self.direct_mcp_client:
                if 'src.tools.mcp_client' in sys.modules:
                    try:
                        safe_cleanup_mcp_client()
                        # print_current("🔌 Direct MCP client cleanup completed")
                    except Exception as e:
                        print_current(f"⚠️ Direct MCP client cleanup failed: {e}")
            
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
        确保LLM客户端已初始化 - 延迟加载实现
        在第一次调用LLM时才创建客户端
        """
        # 如果已经初始化，直接返回
        if self._llm_client_initialized:
            return
        
        self._llm_client_initialized = True
        print_debug("🔄 首次调用LLM，开始初始化LLM客户端...")
        
        if self.is_claude:
            print_debug(f"🧠 Detected Anthropic API, using Anthropic protocol")
            
            # 延迟导入 Anthropic 库
            Anthropic = get_anthropic_client()
            self.client = Anthropic(
                api_key=self.api_key,
                base_url=self.api_base
            )
        else:
            print_debug(f"🤖 Using OpenAI protocol")
            
            # 延迟导入 OpenAI 库
            from openai import OpenAI
            self.client = OpenAI(
                api_key=self.api_key,
                base_url=self.api_base
            )
        
        print_debug("✅ LLM客户端初始化成功")
    
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
            
            return system_prompt
                
        except Exception as e:
            print_current(f"Error loading system prompt: {e}")
            return "You are a helpful AI assistant that can use tools to accomplish tasks."
    
    def _load_prompt_file_cached(self, file_path: str) -> str:
        """
        缓存加载提示文件，使用文件修改时间检测变化
        
        Args:
            file_path: 文件路径
            
        Returns:
            文件内容，如果文件不存在返回空字符串
        """
        if not os.path.exists(file_path):
            return ""
        
        try:
            # 检查缓存
            current_mtime = os.path.getmtime(file_path)
            
            if file_path in self._prompt_file_cache:
                cached_content, cached_mtime = self._prompt_file_cache[file_path]
                if cached_mtime == current_mtime:
                    return cached_content  # 缓存命中
            
            # 缓存未命中，读取文件
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
            
            # 更新缓存
            self._prompt_file_cache[file_path] = (content, current_mtime)
            return content
            
        except Exception as e:
            print_debug(f"Warning: Could not load file {file_path}: {e}")
            return ""
    
    def load_user_prompt_components(self) -> Dict[str, str]:
        """
        Load all prompt components that go into the user message.
        使用缓存优化，避免重复读取文件
        
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
                # 使用缓存加载工具定义（不强制重新加载）
                tool_definitions = self._load_tool_definitions_from_file(force_reload=False)
                
                # 缓存生成的工具提示
                import hashlib
                def_hash = hashlib.md5(str(sorted(tool_definitions.items())).encode()).hexdigest()
                cache_key = (self.language, def_hash)
                
                if cache_key in self._tools_prompt_cache:
                    json_tools_prompt = self._tools_prompt_cache[cache_key]
                else:
                    json_tools_prompt = generate_tools_prompt_from_json(tool_definitions, self.language)
                    self._tools_prompt_cache[cache_key] = json_tools_prompt
                
                # Load only rules and plugin prompts (使用缓存)
                rules_tool_files = [
                    os.path.join(self.prompts_folder, "rules_prompt.txt"), 
                    os.path.join(self.prompts_folder, "mcp_kb_tool_prompts.txt"),
                    os.path.join(self.prompts_folder, "user_rules.txt")
                ]
                
                rules_parts = []
                if json_tools_prompt:
                    rules_parts.append(json_tools_prompt)
                
                # 使用缓存方法加载文件
                for file_path in rules_tool_files:
                    content = self._load_prompt_file_cached(file_path)
                    if content:
                        rules_parts.append(content)
                
                if rules_parts:
                    components['rules_and_tools'] = "\n\n".join(rules_parts)
                
                # Log the approach used
                #if json_tools_prompt:
                #    print_current("✅ Using JSON-generated tool descriptions for chat-based model")
                #else:
                #    print_current("⚠️  Failed to generate JSON tool descriptions, falling back to file-based approach")
                    
            else:
                # For standard tool calling, load only rules (使用缓存)
                rules_tool_files = [
                    os.path.join(self.prompts_folder, "rules_prompt.txt"), 
                    os.path.join(self.prompts_folder, "mcp_kb_tool_prompts.txt"),
                    os.path.join(self.prompts_folder, "user_rules.txt")
                ]
                
                rules_parts = []
                
                # 使用缓存方法加载文件
                for file_path in rules_tool_files:
                    content = self._load_prompt_file_cached(file_path)
                    if content:
                        rules_parts.append(content)
                
                if rules_parts:
                    components['rules_and_tools'] = "\n\n".join(rules_parts)
                
                print_debug("✅ Using standard tool calling (tool descriptions provided via API)")
                
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
    
    
    def _add_full_history_to_message(self, message_parts: List[str], task_history: List[Dict[str, Any]]) -> None:
        """
        Add full task history to message parts with standardized formatting for better cache hits.
        
        Args:
            message_parts: List to append history content to
            task_history: Previous task execution history
        """
        message_parts.append("## Previous Round Context:")
        message_parts.append("Below is the context from previous tasks in this session:")
        message_parts.append("")
        
        for i, record in enumerate(task_history, 1):
            if record.get("role") == "system":
                continue
            
            elif "result" in record:
                '''
                # Add clear separator line for each round with improved labeling
                if record.get("is_summary", False):
                    message_parts.append(f"### Summary of Earlier Rounds:")
                else:
                    # Calculate position relative to end for recent rounds
                    position_from_end = len(task_history) - i
                    if position_from_end == 0:
                        message_parts.append(f"### Most Recent Round:")
                    elif position_from_end == 1:
                        message_parts.append(f"### Second Most Recent Round:")
                    else:
                        message_parts.append(f"### Recent Round (T-{position_from_end}):")
                message_parts.append("")
                '''
                
                # 🔧 优化：只在第一轮显示用户请求，后续轮次只显示轮次信息
                if "prompt" in record:
                    # 第一轮：显示完整的用户请求
                    user_request = record['prompt'].strip()
                    message_parts.append(f"**User Request:**")
                    message_parts.append(user_request)
                    message_parts.append("")
                else:
                    # 后续轮次：显示轮次信息
                    task_round = record.get('task_round', 'N/A')
                    message_parts.append(f"**Round {task_round} Execution:**")
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
                    message_parts.append("**LLM Called Following Tools in this round (It is a reply from the environment, if you want to calling tools, you should fill in the tool calls section, not here!!!), you should not print this section in your response**")
                    message_parts.append(tool_calls_section)
                    #message_parts.append("")
                
                # Display tool execution results if present
                if tool_results_section:
                    message_parts.append("**Tool Execution Results:**")
                    # Standardize tool results format for better cache consistency
                    tool_results_section = self._standardize_tool_results_format(tool_results_section)
                    message_parts.append(tool_results_section)
                    message_parts.append("")
                
                message_parts.append("")  # Extra space after separator
    
    def _standardize_tool_results_format(self, tool_results: str) -> str:
        """
        Standardize tool results format for better cache consistency.
        
        Args:
            tool_results: Raw tool results string
            
        Returns:
            Standardized tool results string
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
            if line.startswith('<tool_execute'):
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
            elif line.startswith('Executing tool:'):
                # Standardize executing tool message format
                parts = line.split(' with params: ')
                if len(parts) == 2:
                    tool_info = parts[0].replace('Executing tool: ', '')
                    params_info = parts[1]
                    standardized_lines.append(f'Executing tool: {tool_info} with params: {params_info}')
                else:
                    standardized_lines.append(line)
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

    def _compute_history_hash(self, task_history: List[Dict[str, Any]]) -> str:
        """
        Compute a hash for the task history to enable caching.
        
        Args:
            task_history: Task history to hash
            
        Returns:
            Hash string for the history
        """
        import hashlib
        
        # Create a string representation of the history
        history_str = ""
        for record in task_history:
            history_str += str(record.get("prompt", "")) + str(record.get("result", "")) + str(record.get("content", ""))
        
        # Create SHA256 hash
        return hashlib.sha256(history_str.encode('utf-8')).hexdigest()

    def get_history_summary_cache_info(self) -> Dict[str, Any]:
        """
        Get information about the history summary cache.
        
        Returns:
            Dictionary containing cache information
        """
        cache_size = len(self.history_summary_cache) if hasattr(self, 'history_summary_cache') else 0
        last_length = getattr(self, 'last_summarized_history_length', 0)
        
        return {
            'cache_size': cache_size,
            'last_summarized_length': last_length,
            'cache_keys': list(self.history_summary_cache.keys()) if hasattr(self, 'history_summary_cache') else []
        }

    def clear_history_summary_cache(self) -> None:
        """
        Clear the history summary cache.
        """
        if hasattr(self, 'history_summary_cache'):
            self.history_summary_cache.clear()
        if hasattr(self, 'last_summarized_history_length'):
            self.last_summarized_history_length = 0

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
        
        original_history_id = id(task_history)  # Track if we modify the history
        history_was_optimized = False
        
        # Initialize current_round_images if not exists
        if not hasattr(self, 'current_round_images'):
            self.current_round_images = []
        
        # Track if get_sensor_data was called in current round
        current_round_has_sensor_data = False
        round_counter = execution_round
        
        try:
            # Enhancement: Check for terminate messages before each tool call in every round
            terminate_signal = self._check_terminate_messages()
            if terminate_signal:
                return terminate_signal
            
            # Load system prompt (only core system_prompt.txt content)
            system_prompt = self.load_system_prompt()
            
            # Build user message with new architecture
            user_message = self._build_new_user_message(prompt, task_history, round_counter)
            
            # Prepare messages for the LLM with proper system/user separation
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ]
            
            # Save debug log for this call's input (before LLM call)
            if self.debug_mode:
                try:
                    initial_call_info = {
                        "is_single_call": True,
                        "call_type": "standard_tools_execution",
                        "user_prompt": prompt
                    }
                except Exception as e:
                    print_current(f"⚠️ Debug preparation failed: {e}")
            
            # Execute LLM call with standard tools
            content, tool_calls = self._call_llm_with_standard_tools(messages, user_message, system_prompt)
            
            # 显示LLM的message内容（在工具执行之前）
            if content and not self.streaming:
                print_current("")
                print_current("💬 LLM Response:")
                print_current(content)
            
            
            # After vision analysis is complete, immediately optimize the history to remove any analyzed base64 image data
            print_debug(f"🔍 Checking optimization conditions: task_history={len(task_history) if task_history else 0}, history_optimizer={hasattr(self, 'history_optimizer') and self.history_optimizer is not None}")
            if task_history and hasattr(self, 'history_optimizer') and self.history_optimizer:
                try:
                    # Immediately optimize the history, removing all image data (keep_recent_images=0)
                    # Since the vision API has already provided a text description, the original image data is no longer needed
                    #print_current(f"🔍 Starting optimization with {len(task_history)} history records...")
                    optimized_history = self.history_optimizer.optimize_history_for_context(
                        task_history, keep_recent_images=0
                    )
                    # Update the history reference to ensure subsequent rounds use the optimized history
                    original_count = len(task_history)
                    task_history.clear()
                    task_history.extend(optimized_history)
                    history_was_optimized = True
                    #print_current(f"✅ History optimization complete: {original_count} → {len(optimized_history)} records")
                except Exception as e:
                    print_current(f"❌ History optimization failed: {e}")
            else:
                print_debug("⚠️ Skipping history optimization - conditions not met")
            # Show raw model response for debugging
            #if self.debug_mode and content:
            #    print_current("🤖 Raw model response:")
            #    print_current(content)
            
            # Calculate and display token and character statistics
            #self._display_llm_statistics(messages, content, tool_calls)
            
            # Store current messages for next round cache analysis
            self.previous_messages = messages.copy()
            
            # Check for TASK_COMPLETED flag and detect conflicts
            has_task_completed = "TASK_COMPLETED:" in content
            has_tool_calls = len(tool_calls) > 0

            # Remember if there was originally a TASK_COMPLETED flag
            original_has_task_completed = has_task_completed
            # Save the completion message before removing it (for conflict case)
            original_completion_message = None
            if has_task_completed:
                task_completed_match = re.search(r'TASK_COMPLETED:\s*(.+)', content)
                if task_completed_match:
                    original_completion_message = task_completed_match.group(1).strip()

            # CONFLICT DETECTION: Both tool calls and TASK_COMPLETED present
            conflict_detected = has_tool_calls and has_task_completed
            if conflict_detected:
                #print_current(f"⚠️ CONFLICT DETECTED: Both tool calls and TASK_COMPLETED flag found, executing tools first then completing task")
                # Remove the TASK_COMPLETED flag from the content to ensure tool execution proceeds
                content = re.sub(r'TASK_COMPLETED:.*', '', content).strip()
                has_task_completed = False # Ensure the flag is updated after removal
            
            # If TASK_COMPLETED but no tool calls, complete the task
            if has_task_completed and not has_tool_calls:
                # Remove unnecessary log output
                # Extract the completion message
                task_completed_match = re.search(r'TASK_COMPLETED:\s*(.+)', content)
                if task_completed_match:
                    completion_message = task_completed_match.group(1).strip()
                
                # Save final debug log
                if self.debug_mode:
                    try:
                        completion_info = {
                            "has_tool_calls": False,
                            "task_completed": True,
                            "completion_detected": True,
                            "execution_result": "task_completed_flag"
                        }
                        
                        self._save_llm_call_debug_log(messages, f"Task completed with TASK_COMPLETED flag", 1, completion_info)
                    except Exception as log_error:
                        print_current(f"❌ Completion debug log save failed: {log_error}")
                
                # Store task completion in long-term memory
                self._store_task_completion_memory(prompt, content, {
                    "task_completed": True,
                    "completion_method": "TASK_COMPLETED_flag",
                    "execution_round": round_counter,
                    "model_used": self.model
                }, force_update=True)
                
                finish_operation(f"executing task (round {round_counter})")
                # Return optimized history if available
                if history_was_optimized:
                    return (content, task_history)
                return content
            
            # Execute tools if present
            if tool_calls:
                # Always format tool calls for history (needed for final response)
                tool_calls_formatted = self._format_tool_calls_for_history(tool_calls)
                
                # 🔧 NEW: Track if get_sensor_data was called in current round
                current_round_has_sensor_data = False
                
                # Check if tools were already executed during streaming
                tools_already_executed = (self.streaming and 
                                        (not self.use_chat_based_tools) and
                                        hasattr(self, '_tools_executed_in_stream') and 
                                        getattr(self, '_tools_executed_in_stream', False))
                
                if tools_already_executed:
                    #print_current("✅ Tools were already executed during streaming - collecting results for response formatting")
                    # For streaming execution, we still need to format the response properly
                    # but skip actual execution since it was done during streaming
                    all_tool_results = getattr(self, '_streaming_tool_results', [])
                    successful_executions = len(all_tool_results)
                    
                    # Show tool call information
                    #print_current(f"🔧 Model decided to call {len(tool_calls)} tools:")
                    if tool_calls_formatted:
                        # Remove the "**Tool Calls:**" header since we already printed our own
                        display_content = tool_calls_formatted.replace("**Tool Calls:**\n", "").strip()
                        print_current(display_content)
                else:
                    pass
                    #print_current(f"🔧 Model decided to call {len(tool_calls)} tools:")

                    # 添加换行（仅限chat based接口，在工具执行之前）
                    if self.use_chat_based_tools:
                        print_current("")
                    
                    # Print tool calls for terminal display with better formatting
                    if tool_calls_formatted:
                        # Remove the "**Tool Calls:**" header since we already printed our own
                        display_content = tool_calls_formatted.replace("**Tool Calls:**\n", "").strip()
                        print_current(display_content)
                    
                    #print_current("=" * 50)
                    #print_current("🚀 Starting tool execution...")
                    
                    # Execute all tool calls and collect results
                    all_tool_results = []
                    successful_executions = 0
                
                # Only execute tools if they weren't already executed during streaming
                if not tools_already_executed:
                    #print_current(f"🚀 Starting execution of {len(tool_calls)} tool calls...")
                    for i, tool_call in enumerate(tool_calls, 1):
                        # Handle standard format tool calls (both OpenAI and Anthropic)
                        try:
                            tool_name = self._get_tool_name_from_call(tool_call)
                            tool_params = self._get_tool_params_from_call(tool_call)
                            print_current(f"🔧 Executing tool {tool_name}")
                        except Exception as e:
                            print_current(f"❌ Failed to extract tool name/params from tool_call {i}: {e}")
                            print_current(f"Tool call structure: {tool_call}")
                            continue
                        
                        # 🔧 NEW: Track if get_sensor_data was called in current round
                        if tool_name == 'get_sensor_data':
                            current_round_has_sensor_data = True
                        
                        # Let streaming_output handle all tool execution display
                        try:
                            # Convert to standard format for execute_tool
                            standard_tool_call = {
                                "name": tool_name,
                                "arguments": tool_params
                            }
                            # Pass streaming_output=True for real-time tool execution display
                            tool_result = self.execute_tool(standard_tool_call, streaming_output=True)
                            
                            all_tool_results.append({
                                'tool_name': tool_name,
                                'tool_params': tool_params,
                                'tool_result': tool_result
                            })
                            successful_executions += 1
                            
                            # Tool result is already displayed by streaming output, no need to duplicate
                            
                        except Exception as e:
                            error_msg = f"Tool {tool_name} execution failed: {str(e)}"
                            print_current(f"❌ {error_msg}")
                            all_tool_results.append({
                                'tool_name': tool_name,
                                'tool_params': tool_params,
                                'tool_result': f"Error: {error_msg}"
                            })
                        
                        # Separator is handled by streaming output, no need to duplicate
                
                # 🔧 MODIFIED: Store image data but don't use vision API
                self._extract_current_round_images(all_tool_results)
                
                # 🔧 NEW: Format tool results with base64 data detection
                tool_results_message = self._format_tool_results_for_llm(all_tool_results, include_base64_info=current_round_has_sensor_data)
                
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
                            "conflict_detected": conflict_detected
                        }
                        
                        self._save_llm_call_debug_log(messages, f"Single execution with {len(tool_calls)} tool calls", 1, tool_execution_info)
                    except Exception as log_error:
                        print_current(f"❌ Debug log save failed: {log_error}")
                
                # Return combined response with tool calls and tool results
                result_parts = [content]
                if tool_calls_formatted:
                    result_parts.append("\n\n--- Tool Calls ---\n" + tool_calls_formatted)
                result_parts.append("\n\n--- Tool Execution Results ---\n" + tool_results_message)
                
                # Check if this was originally intended to be task completion after tool execution
                if original_has_task_completed:
                    # Use saved completion message or extract from content
                    completion_message = original_completion_message
                    if not completion_message:
                        task_completed_match = re.search(r'TASK_COMPLETED:\s*(.+)', content)
                        if task_completed_match:
                            completion_message = task_completed_match.group(1).strip()
                    
                    # Re-add TASK_COMPLETED flag to the result so task_checker can detect it
                    if completion_message:
                        combined_result = "".join(result_parts) + f"\n\nTASK_COMPLETED: {completion_message}"
                    else:
                        combined_result = "".join(result_parts) + "\n\nTASK_COMPLETED"
                    
                    # Save final debug log
                    if self.debug_mode:
                        try:
                            completion_info = {
                                "has_tool_calls": True,
                                "task_completed": True,
                                "completion_detected": True,
                                "execution_result": "task_completed_with_tools",
                                "tool_calls_count": len(tool_calls),
                                "successful_executions": successful_executions
                            }

                            self._save_llm_call_debug_log(messages, f"Task completed with TASK_COMPLETED flag after tool execution", 1, completion_info)
                        except Exception as log_error:
                            print_current(f"❌ Completion debug log save failed: {log_error}")

                    # Store task completion in long-term memory
                    self._store_task_completion_memory(prompt, combined_result, {
                        "task_completed": True,
                        "completion_method": "task_completed_with_tools",
                        "execution_round": round_counter,
                        "tool_calls_count": len(tool_calls),
                        "successful_executions": successful_executions,
                        "model_used": self.model
                    }, force_update=True)

                    finish_operation(f"executing task (round {round_counter})")
                    # Return optimized history if available
                    if history_was_optimized:
                        return (combined_result, task_history)
                    return combined_result

                # Store task completion in long-term memory (normal tool execution)
                combined_result = "".join(result_parts)
                self._store_task_completion_memory(prompt, combined_result, {
                    "task_completed": False,  # Not task completion, just tool execution
                    "completion_method": "tool_execution",
                    "execution_round": round_counter,
                    "tool_calls_count": len(tool_calls),
                    "successful_executions": successful_executions,
                    "model_used": self.model
                }, force_update=False)

                finish_operation(f"executing task (round {round_counter})")
                # Return optimized history if available, even with tool calls
                if history_was_optimized:
                    return (combined_result, task_history)
                return combined_result
            
            else:
                # No tool calls, return LLM response directly
                # print_current("📝 No tool calls found, returning LLM response")
                
                # 🔧 NEW: Add base64 data status information when no tools are called
                #base64_status_info = "\n\n## Base64 Data Status\n❌ No base64 encoded image data acquired in this round (no get_sensor_data tool called)."
                #content = content + base64_status_info
                
                # Save debug log for response without tools
                if self.debug_mode:
                    try:
                        no_tools_info = {
                            "has_tool_calls": False,
                            "task_completed": has_task_completed,
                            "execution_result": "llm_response_only"
                        }
                        
                        self._save_llm_call_debug_log(messages, f"Single execution, no tool calls", 1, no_tools_info)
                    except Exception as log_error:
                        print_current(f"❌ Final debug log save failed: {log_error}")
                
                # Store task completion in long-term memory
                self._store_task_completion_memory(prompt, content, {
                    "task_completed": True,
                    "completion_method": "llm_response_only",
                    "execution_round": round_counter,
                    "model_used": self.model
                }, force_update=False)
                
                finish_operation(f"executing task (round {round_counter})")
                # Return optimized history if available
                if history_was_optimized:
                    return (content, task_history)
                return content
            
        except json.JSONDecodeError as e:
            error_msg = f"JSON parsing error in tool call: {str(e)}"
            print_debug(error_msg)
            print_debug(f"📄 This usually means the model generated invalid JSON in tool arguments")
            print_debug(f"💡 Try regenerating the response or check the model's tool calling format")
            finish_operation(f"executing task (round {round_counter})")
            return error_msg
        except Exception as e:
            error_msg = f"Error executing subtask: {str(e)}"
            print_debug(error_msg)
            
            # Add more specific error information for common issues
            if "Expecting ',' delimiter" in str(e):
                print_current(f"💡 This is likely a JSON formatting issue in tool arguments")
                print_current(f"🔧 The model may have generated malformed JSON - try regenerating")
            elif "json.loads" in str(e) or "JSONDecodeError" in str(e):
                print_current(f"💡 JSON parsing error detected - check tool argument formatting")
            
            finish_operation(f"executing task (round {round_counter})")
            return error_msg
            
            # Add current result to accumulated results
            #accumulated_results += content
            
            # Add tool calls to conversation history
            #conversation_history.append({"role": "assistant", "content": content})
            
            # Add tool calls to tool_calls list
            #tool_calls_list.extend(tool_calls)
            
            #round_counter += 1
        
        # If we reach here, it means we've completed all tool calls
        # print_current("🎉 All tool calls completed successfully!")  # Reduced verbose output

        # Return final accumulated results
        return "Error: Task execution failed - reached unexpected code path"
    
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
            if not current_agent_id:
                return None

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
    
    def _has_complete_json_tool_call(self, content: str) -> bool:
        """
        检测content中是否包含完整的工具调用（支持带```json标记和不带标记的纯JSON格式）
        
        Args:
            content: 累积的响应内容
            
        Returns:
            bool: 如果检测到完整的工具调用JSON返回True
        """
        # 首先检查是否有```json标记的格式
        json_block_marker = '```json'
        if json_block_marker in content:
            first_pos = content.find(json_block_marker)
            json_start = first_pos + len(json_block_marker)
            first_block_end = content.find('```', json_start)
            
            # 如果第一个块没有闭合，不要停止（可能还在接收中）
            if first_block_end == -1:
                return False
            
            # 查找第二个```json块（必须在第一个块之后）
            second_pos = content.find(json_block_marker, first_block_end + 3)
            # 如果找到第二个```json块，且第一个块已完整，说明有多个工具调用，需要截断
            return second_pos != -1
        
        # 如果没有```json标记，检查是否是纯JSON格式的工具调用
        # 查找 "tool_name" 和 "parameters" 字段
        if '"tool_name"' in content and '"parameters"' in content:
            # 尝试找到第一个完整的JSON对象（以{开始，以}结束）
            try:
                brace_start = content.find('{')
                if brace_start != -1:
                    # 尝试找到匹配的闭合括号
                    brace_count = 0
                    in_string = False
                    escape_next = False
                    brace_end = -1
                    
                    for i in range(brace_start, len(content)):
                        char = content[i]
                        if escape_next:
                            escape_next = False
                            continue
                        if char == '\\':
                            escape_next = True
                            continue
                        if char == '"' and not escape_next:
                            in_string = not in_string
                            continue
                        if not in_string:
                            if char == '{':
                                brace_count += 1
                            elif char == '}':
                                brace_count -= 1
                                if brace_count == 0:
                                    brace_end = i + 1
                                    break
                    
                    if brace_end > brace_start:
                        # 检查是否包含tool_name和parameters
                        json_str = content[brace_start:brace_end]
                        if '"tool_name"' in json_str and '"parameters"' in json_str:
                            # 尝试解析JSON验证其有效性
                            try:
                                import json
                                tool_data = json.loads(json_str)
                                if isinstance(tool_data, dict) and 'tool_name' in tool_data and 'parameters' in tool_data:
                                    # 检查是否有第二个工具调用
                                    remaining_content = content[brace_end:]
                                    if '"tool_name"' in remaining_content and '"parameters"' in remaining_content:
                                        return True  # 有多个工具调用
                                    return True  # 至少有一个完整的工具调用
                            except:
                                pass
            except:
                pass
        
        return False
    
    def _is_complete_json_tool_call(self, content: str) -> bool:
        """
        检测content中是否包含完整的工具调用（兼容性方法，调用新的检测方法）
        """
        return self._has_complete_json_tool_call(content)
    
    def _find_second_json_block_start(self, content: str) -> int:
        """
        找到第二个```json块的起始位置
        确保第一个JSON块已经完整闭合后再查找第二个
        
        Args:
            content: 响应内容
            
        Returns:
            int: 第二个```json块的起始位置，如果没找到返回-1
        """
        json_block_marker = '```json'
        first_pos = content.find(json_block_marker)
        if first_pos == -1:
            return -1
        
        # 检查第一个JSON块是否已经完整闭合
        json_start = first_pos + len(json_block_marker)
        first_block_end = content.find('```', json_start)
        
        # 如果第一个块没有闭合，不要查找第二个块（可能还在接收中）
        if first_block_end == -1:
            return -1
        
        # 在第一个块闭合之后查找第二个```json块
        second_pos = content.find(json_block_marker, first_block_end + 3)
        return second_pos
    
    def _ensure_first_json_block_complete(self, content: str) -> str:
        """
        简化版本：确保第一个```json块有闭合的```标记
        不做复杂的修复，只做基本检查，将修复交给parse_tool_calls处理
        
        Args:
            content: 响应内容
            
        Returns:
            str: 确保第一个JSON块有闭合标记的内容（如果缺失则补全），否则返回原内容
        """
        json_block_marker = '```json'
        first_pos = content.find(json_block_marker)
        
        # 如果没有找到```json标记，直接返回原内容，让parse_tool_calls处理
        if first_pos == -1:
            return content
        
        # 找到第一个```json块的开始位置
        json_start = first_pos + len(json_block_marker)
        
        # 查找第一个```json块的结束位置（闭合的```）
        first_block_end = content.find('```', json_start)
        
        # 如果第一个块已经完整闭合，直接返回
        if first_block_end != -1:
            return content
        
        # 如果第一个块没有闭合，简单补全闭合标记（不做复杂的JSON验证）
        # 只查找第一个匹配的```来闭合，如果找不到就原样返回让parse_tool_calls处理
        return content + '\n```'
    
    def _get_content_before_second_json(self, content: str) -> str:
        """
        获取第二个```json块之前的内容
        确保包含完整的第一个JSON块
        
        Args:
            content: 完整的响应内容
            
        Returns:
            str: 第二个```json块之前的内容，确保第一个JSON块完整
        """
        second_json_pos = self._find_second_json_block_start(content)
        if second_json_pos == -1:
            # 没有找到第二个```json，尝试确保第一个块完整
            return self._ensure_first_json_block_complete(content)
        
        # 截取到第二个```json之前的内容
        content_before_second = content[:second_json_pos].rstrip()
        
        # 确保第一个块是完整的
        result = self._ensure_first_json_block_complete(content_before_second)
        
        return result
    
    def _extract_json_block_robust(self, content: str, start_marker: str = '```json') -> Optional[str]:
        """
        更健壮地提取JSON块，处理嵌套的```标记和不完整的JSON块。
        特别优化了对超长JSON块（如包含大量文本的code_edit字段）的处理。
        
        Args:
            content: 要提取的内容
            start_marker: JSON块开始标记，默认为'```json'
            
        Returns:
            提取的JSON字符串，如果提取失败返回None
        """
        json_start = content.find(start_marker)
        if json_start == -1:
            return None
        
        # 从标记后开始查找JSON内容
        json_content_start = json_start + len(start_marker)
        
        # 策略1: 先尝试找到结束的```标记
        # 对于超长JSON块，使用更智能的查找策略
        json_content_end = -1
        
        # 查找JSON块的结束标记```
        # 优化：对于tool_name/parameters格式，可以利用结尾的 }\n} 模式
        # 先尝试找到最后一个```（JSON块的真正结束）
        last_triple_backtick = content.rfind('```', json_content_start)
        if last_triple_backtick > json_content_start:
            # 检查这个位置之前是否有 }\n} 模式（说明这是JSON的结束）
            # 对于超长内容，扩大检查范围
            check_range = 50 if len(content) > 10000 else 20
            before_marker = content[max(0, last_triple_backtick-check_range):last_triple_backtick]
            # 检查多种可能的结尾模式
            if ('}\n}' in before_marker or '}\n  }' in before_marker or 
                before_marker.rstrip().endswith('}') or
                content[last_triple_backtick-1:last_triple_backtick] in ['\n', '\r', ' ']):
                # 这很可能是JSON块的结束标记
                json_content_end = last_triple_backtick
            else:
                # 继续使用原来的逻辑查找
                i = current_pos = json_content_start
                while i < len(content) - 2:
                    if content[i:i+3] == '```':
                        # 检查这是否是开始标记（前面没有内容或是换行）
                        # 对于超长内容，放宽检查条件
                        if i == json_content_start or content[i-1] in ['\n', '\r', ' ']:
                            # 检查这个位置之前是否有JSON结束模式
                            check_before = content[max(0, i-50):i]
                            if ('}\n}' in check_before or '}\n  }' in check_before or 
                                check_before.rstrip().endswith('}')):
                                # 这是结束标记
                                json_content_end = i
                                break
                    i += 1
        
        # 策略2: 如果没有找到结束标记，使用括号匹配来找到完整的JSON对象
        if json_content_end == -1:
            # 没有找到结束标记，可能JSON块不完整或超长
            # 尝试找到最后一个完整的JSON对象或数组
            remaining = content[json_content_start:]
            
            # 使用括号匹配来找到完整的JSON对象
            # 对于超长内容，使用更高效的算法
            brace_count = 0
            bracket_count = 0
            in_string = False
            escape_next = False
            last_valid_pos = -1
            i = 0
            
            # 跳过开头的空白
            while i < len(remaining) and remaining[i] in ' \t\n\r':
                i += 1
            
            # 如果第一个字符是{，开始计数
            if i < len(remaining) and remaining[i] == '{':
                brace_count = 1
                i += 1
                
                # 对于超长内容，优化性能：批量处理字符
                while i < len(remaining):
                    char = remaining[i]
                    
                    if escape_next:
                        escape_next = False
                        i += 1
                        continue
                        
                    if char == '\\':
                        escape_next = True
                        i += 1
                        continue
                        
                    if char == '"' and not escape_next:
                        in_string = not in_string
                        i += 1
                        continue
                        
                    if not in_string:
                        if char == '{':
                            brace_count += 1
                        elif char == '}':
                            brace_count -= 1
                            if brace_count == 0 and bracket_count == 0:
                                last_valid_pos = i + 1
                                break
                        elif char == '[':
                            bracket_count += 1
                        elif char == ']':
                            bracket_count -= 1
                            if brace_count == 0 and bracket_count == 0:
                                last_valid_pos = i + 1
                                break
                    
                    i += 1
            
            if last_valid_pos > 0:
                return remaining[:last_valid_pos].strip()
            # 如果找不到完整的JSON，返回剩余内容（让JSON解析器尝试处理）
            # 但尝试找到最后一个可能的结束位置
            # 对于包含code_edit的超长JSON，尝试找到最后一个}
            if 'code_edit' in remaining:
                last_brace = remaining.rfind('}')
                if last_brace > 0:
                    # 检查这个位置是否合理（前面应该有匹配的{）
                    test_json = remaining[:last_brace+1].strip()
                    # 简单验证：检查是否以{开头
                    if test_json.startswith('{'):
                        return test_json
            return remaining.strip()
        
        # 提取JSON内容
        json_content = content[json_content_start:json_content_end].strip()
        return json_content
    
    def parse_tool_calls(self, content: str) -> List[Dict[str, Any]]:
        """
        Parse multiple tool calls from the model's response.
        
        Args:
            content: The model's response text
            
        Returns:
            List of dictionaries with tool name and parameters
        """
        
        # Debug mode, save raw content for analysis
        debug_info = []
        if self.debug_mode:
            # Check for common tool call format markers
            has_function_calls = '<function_calls>' in content
            has_invoke = '<invoke' in content
            has_function_call = '<function_call>' in content
            has_json_block = '```json' in content
            has_tool_calls_json = '"tool_calls"' in content
            debug_info.append(f"Markers: json_block={has_json_block}, tool_calls_json={has_tool_calls_json}, function_calls={has_function_calls}")
        
        all_tool_calls = []
        

        # Try OpenAI-style tool calls format with improved JSON extraction
        openai_json_pattern = r'```json\s*\{\s*"tool_calls"\s*:\s*\[(.*?)\]\s*\}\s*```'
        openai_json_match = re.search(openai_json_pattern, content, re.DOTALL)
        if openai_json_match or '```json' in content and '"tool_calls"' in content:
            try:
                # Use robust extraction method
                json_block = self._extract_json_block_robust(content, '```json')
                if not json_block:
                    if self.debug_mode:
                        debug_info.append("Failed to extract JSON block with ```json marker")
                    raise ValueError("Could not extract JSON block")
                
                # Try to parse JSON directly without complex escape fixing
                try:
                    tool_calls_data = json.loads(json_block)
                    
                    if isinstance(tool_calls_data, dict) and 'tool_calls' in tool_calls_data:
                        for i, tool_call in enumerate(tool_calls_data['tool_calls']):
                            if isinstance(tool_call, dict) and 'function' in tool_call:
                                function_data = tool_call['function']
                                if 'name' in function_data and 'arguments' in function_data:
                                    arguments = function_data['arguments']
                                    # If arguments is a string (JSON), parse it
                                    if isinstance(arguments, str):
                                        try:
                                            arguments = json.loads(arguments)
                                        except json.JSONDecodeError:
                                            pass
                                    
                                    all_tool_calls.append({
                                        "name": function_data['name'],
                                        "arguments": arguments
                                    })
                    
                    # If we found OpenAI-style tool calls, return them
                    if all_tool_calls:
                        return all_tool_calls
                except json.JSONDecodeError as e:
                    error_msg = f"Failed to parse JSON block: {str(e)[:200]}"
                    if self.debug_mode:
                        debug_info.append(f"OpenAI JSON parse error: {error_msg}")
                        debug_info.append(f"JSON block length: {len(json_block)}, first 200 chars: {json_block[:200]}")
                    print_current(error_msg)
                    # Try to fix JSON containing SVG/XML content or other complex content
                    try:
                        # Try fixing with smart quote escaping (handles SVG XML with quotes)
                        fixed_json = smart_escape_quotes_in_json_values(json_block)
                        if fixed_json == json_block:
                            # If fix didn't change anything, try other methods
                            if self.debug_mode:
                                debug_info.append("smart_escape_quotes_in_json_values didn't modify JSON")
                        tool_calls_data = json.loads(fixed_json)
                        
                        if isinstance(tool_calls_data, dict) and 'tool_calls' in tool_calls_data:
                            for i, tool_call in enumerate(tool_calls_data['tool_calls']):
                                    if isinstance(tool_call, dict) and 'function' in tool_call:
                                        function_data = tool_call['function']
                                        if 'name' in function_data and 'arguments' in function_data:
                                            arguments = function_data['arguments']
                                            if isinstance(arguments, str):
                                                try:
                                                    arguments = json.loads(arguments)
                                                except json.JSONDecodeError:
                                                    pass
                                            
                                            all_tool_calls.append({
                                                "name": function_data['name'],
                                                "arguments": arguments
                                            })
                            
                        if all_tool_calls:
                            print_current("✅ Successfully parsed JSON after applying quote escaping fixes")
                            return all_tool_calls
                    except (json.JSONDecodeError, Exception) as fix_error:
                        error_msg = f"⚠️ All JSON fix attempts failed. Original: {str(e)[:200]}, Fix: {str(fix_error)[:200]}"
                        if self.debug_mode:
                            debug_info.append(error_msg)
                            print_current(f"⚠️ {error_msg}")
                        else:
                            print_current(f"⚠️ {error_msg}")

            except (json.JSONDecodeError, ValueError, Exception) as e:
                error_msg = f"Failed to parse OpenAI-style JSON tool calls: {str(e)[:200]}"
                if self.debug_mode:
                    debug_info.append(error_msg)
                    print_current(error_msg)
        
        # Also try to parse direct JSON tool calls without ```json wrapper
        direct_json_pattern = r'\{\s*"tool_calls"\s*:\s*\[(.*?)\]\s*\}'
        direct_json_match = re.search(direct_json_pattern, content, re.DOTALL)
        if direct_json_match and not all_tool_calls:
            try:
                json_str = direct_json_match.group(0)
                tool_calls_data = json.loads(json_str)
                if isinstance(tool_calls_data, dict) and 'tool_calls' in tool_calls_data:
                    for tool_call in tool_calls_data['tool_calls']:
                        if isinstance(tool_call, dict) and 'function' in tool_call:
                            function_data = tool_call['function']
                            if 'name' in function_data and 'arguments' in function_data:
                                arguments = function_data['arguments']
                                # If arguments is a string (JSON), parse it
                                if isinstance(arguments, str):
                                    try:
                                        arguments = json.loads(arguments)
                                    except json.JSONDecodeError:
                                        pass
                                
                                all_tool_calls.append({
                                    "name": function_data['name'],
                                    "arguments": arguments
                                })
                    
                    # If we found direct JSON tool calls, return them
                    if all_tool_calls:
                        return all_tool_calls
            except json.JSONDecodeError as e:
                if self.debug_mode:
                    print_current(f"Failed to parse direct JSON tool calls: {e}")
                # Try to fix JSON containing SVG/XML content or other complex content
                try:
                    # Try fixing with smart quote escaping (handles SVG XML with quotes)
                    fixed_json = smart_escape_quotes_in_json_values(json_str)
                    tool_calls_data = json.loads(fixed_json)
                    
                    if isinstance(tool_calls_data, dict) and 'tool_calls' in tool_calls_data:
                        for tool_call in tool_calls_data['tool_calls']:
                            if isinstance(tool_call, dict) and 'function' in tool_call:
                                function_data = tool_call['function']
                                if 'name' in function_data and 'arguments' in function_data:
                                    arguments = function_data['arguments']
                                    if isinstance(arguments, str):
                                        try:
                                            arguments = json.loads(arguments)
                                        except json.JSONDecodeError:
                                            pass
                                    
                                    all_tool_calls.append({
                                        "name": function_data['name'],
                                        "arguments": arguments
                                    })
                    
                    if all_tool_calls:
                        print_current("✅ Successfully parsed direct JSON after applying quote escaping fixes")
                        return all_tool_calls
                except (json.JSONDecodeError, Exception) as fix_error:
                    if self.debug_mode:
                        print_current(f"⚠️ All direct JSON fix attempts failed. Original error: {str(e)[:200]}, Fix error: {str(fix_error)[:200]}")
        
        # Continue with existing XML parsing logic...
        # Try to parse individual <function_call> tags (single format)
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
        
        # NEW: Support for multiple independent JSON tool calls (like our new format)
        # Look for multiple ```json blocks with tool_name format
        # Use improved extraction for each block
        if '```json' in content:
            # Try to extract all JSON blocks using robust method
            json_blocks = []
            search_start = 0
            while True:
                remaining_content = content[search_start:]
                json_block = self._extract_json_block_robust(remaining_content, '```json')
                if json_block:
                    json_blocks.append(json_block)
                    # Find the position of this block in original content
                    block_start = content.find('```json', search_start)
                    if block_start == -1:
                        break
                    block_end = content.find('```', block_start + 7)
                    if block_end == -1:
                        # No closing marker, break
                        break
                    search_start = block_end + 3
                else:
                    break
            
            if json_blocks:
                # Process each extracted JSON block
                for json_block in json_blocks:
                    try:
                        json_str = json_block.strip()
                        # 对于超长JSON块，先尝试直接解析
                        tool_data = json.loads(json_str)
                        
                        if isinstance(tool_data, dict):
                            # Check if it's our new tool_name format
                            if 'tool_name' in tool_data and 'parameters' in tool_data:
                                all_tool_calls.append({
                                    "name": tool_data["tool_name"],
                                    "arguments": tool_data["parameters"]
                                })
                            # Check if it's the old name format (backward compatibility)
                            elif 'name' in tool_data and 'parameters' in tool_data:
                                all_tool_calls.append({
                                    "name": tool_data["name"],
                                    "arguments": tool_data["parameters"]
                                })
                            # Check if it's content format
                            elif 'name' in tool_data and 'content' in tool_data:
                                all_tool_calls.append({
                                    "name": tool_data["name"],
                                    "arguments": tool_data["content"]
                                })
                    except json.JSONDecodeError as e:
                        # Try to fix JSON containing unescaped newlines or quotes
                        # 对于包含code_edit等超长字段的JSON，优先使用robust方法
                        if 'code_edit' in json_str or len(json_str) > 5000:
                            # 超长JSON，优先使用robust方法
                            try:
                                fixed_json_robust = fix_json_string_values_robust(json_str)
                                if fixed_json_robust != json_str:
                                    tool_data = json.loads(fixed_json_robust)
                                    if isinstance(tool_data, dict):
                                        if 'tool_name' in tool_data and 'parameters' in tool_data:
                                            all_tool_calls.append({
                                                "name": tool_data["tool_name"],
                                                "arguments": tool_data["parameters"]
                                            })
                                        elif 'name' in tool_data and 'parameters' in tool_data:
                                            all_tool_calls.append({
                                                "name": tool_data["name"],
                                                "arguments": tool_data["parameters"]
                                            })
                                        elif 'name' in tool_data and 'content' in tool_data:
                                            all_tool_calls.append({
                                                "name": tool_data["name"],
                                                "arguments": tool_data["content"]
                                            })
                            except (json.JSONDecodeError, Exception) as robust_fix_e:
                                # 如果robust方法也失败，尝试正则方法
                                try:
                                    fixed_json = smart_escape_quotes_in_json_values(json_str)
                                    tool_data = json.loads(fixed_json)
                                    
                                    if isinstance(tool_data, dict):
                                        if 'tool_name' in tool_data and 'parameters' in tool_data:
                                            all_tool_calls.append({
                                                "name": tool_data["tool_name"],
                                                "arguments": tool_data["parameters"]
                                            })
                                        elif 'name' in tool_data and 'parameters' in tool_data:
                                            all_tool_calls.append({
                                                "name": tool_data["name"],
                                                "arguments": tool_data["parameters"]
                                            })
                                        elif 'name' in tool_data and 'content' in tool_data:
                                            all_tool_calls.append({
                                                "name": tool_data["name"],
                                                "arguments": tool_data["content"]
                                            })
                                except (json.JSONDecodeError, Exception) as fix_e:
                                    if self.debug_mode:
                                        debug_info.append(f"Failed to parse JSON block (length: {len(json_str)}): {str(e)[:100]}, robust fix failed: {str(robust_fix_e)[:100]}, regex fix failed: {str(fix_e)[:100]}")
                                    else:
                                        pass  # 静默失败，继续尝试其他方法
                        else:
                            # 普通长度JSON，先尝试正则方法
                            try:
                                # 首先尝试使用正则表达式方法
                                fixed_json = smart_escape_quotes_in_json_values(json_str)
                                tool_data = json.loads(fixed_json)
                                
                                if isinstance(tool_data, dict):
                                    if 'tool_name' in tool_data and 'parameters' in tool_data:
                                        all_tool_calls.append({
                                            "name": tool_data["tool_name"],
                                            "arguments": tool_data["parameters"]
                                        })
                                    elif 'name' in tool_data and 'parameters' in tool_data:
                                        all_tool_calls.append({
                                            "name": tool_data["name"],
                                            "arguments": tool_data["parameters"]
                                        })
                                    elif 'name' in tool_data and 'content' in tool_data:
                                        all_tool_calls.append({
                                            "name": tool_data["name"],
                                            "arguments": tool_data["content"]
                                        })
                            except (json.JSONDecodeError, Exception) as fix_e:
                                # 如果正则方法失败，尝试使用更可靠的字符级解析方法
                                try:
                                    fixed_json_robust = fix_json_string_values_robust(json_str)
                                    if fixed_json_robust != json_str:
                                        tool_data = json.loads(fixed_json_robust)
                                        if isinstance(tool_data, dict):
                                            if 'tool_name' in tool_data and 'parameters' in tool_data:
                                                all_tool_calls.append({
                                                    "name": tool_data["tool_name"],
                                                    "arguments": tool_data["parameters"]
                                                })
                                            elif 'name' in tool_data and 'parameters' in tool_data:
                                                all_tool_calls.append({
                                                    "name": tool_data["name"],
                                                    "arguments": tool_data["parameters"]
                                                })
                                            elif 'name' in tool_data and 'content' in tool_data:
                                                all_tool_calls.append({
                                                    "name": tool_data["name"],
                                                    "arguments": tool_data["content"]
                                                })
                                except (json.JSONDecodeError, Exception) as robust_fix_e:
                                    if self.debug_mode:
                                        debug_info.append(f"Failed to parse JSON block: {str(fix_e)[:100]}, robust fix also failed: {str(robust_fix_e)[:100]}")
                                    else:
                                        pass  # 静默失败，继续尝试其他方法
            
                # If we found any tool calls through multiple JSON blocks, return them
                if all_tool_calls:
                    if self.debug_mode:
                        print_current(f"✅ Successfully parsed {len(all_tool_calls)} tool calls from JSON blocks")
                    return all_tool_calls
        
        # Fallback: try to parse single JSON format with nested content structure (like in the logs)
        # Use robust extraction method
        if '```json' in content and not all_tool_calls:
            json_block = self._extract_json_block_robust(content, '```json')
            if json_block:
                json_str = json_block.strip()
                # 对于包含code_edit字段的超长JSON，优先使用robust修复方法
                if 'code_edit' in json_str or len(json_str) > 5000:
                    # 超长JSON，优先使用robust方法
                    try:
                        fixed_json_robust = fix_json_string_values_robust(json_str)
                        if fixed_json_robust != json_str:
                            tool_data = json.loads(fixed_json_robust)
                            if isinstance(tool_data, dict):
                                if 'tool_name' in tool_data and 'parameters' in tool_data:
                                    return [{
                                        "name": tool_data["tool_name"],
                                        "arguments": tool_data["parameters"]
                                    }]
                                elif 'name' in tool_data and 'parameters' in tool_data:
                                    return [{
                                        "name": tool_data["name"],
                                        "arguments": tool_data["parameters"]
                                    }]
                                elif 'name' in tool_data and 'content' in tool_data:
                                    return [{
                                        "name": tool_data["name"],
                                        "arguments": tool_data["content"]
                                    }]
                    except (json.JSONDecodeError, Exception) as robust_e:
                        if self.debug_mode:
                            debug_info.append(f"Robust fix failed for long JSON: {str(robust_e)[:100]}")
                        # 继续尝试其他方法
                        pass
                
                # 尝试直接解析
                try:
                    tool_data = json.loads(json_str)
                    
                    # Handle nested structure like {"name": "edit_file", "content": {...}}
                    if isinstance(tool_data, dict):
                        if 'name' in tool_data and 'content' in tool_data:
                            return [{
                                "name": tool_data["name"],
                                "arguments": tool_data["content"]
                            }]
                        # Check if it's a valid tool call format with tool_name and parameters (new JSON format)
                        elif 'tool_name' in tool_data and 'parameters' in tool_data:
                            return [{
                                "name": tool_data["tool_name"],
                                "arguments": tool_data["parameters"]
                            }]
                        # Check if it's a valid tool call format with name and parameters (compatible with old format)
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
                                    inferred_tool = 'workspace_search'
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
                    # Try to fix JSON containing unescaped newlines or quotes
                    # 对于超长JSON，优先使用robust方法
                    if 'code_edit' in json_str or len(json_str) > 5000:
                        try:
                            fixed_json_robust = fix_json_string_values_robust(json_str)
                            if fixed_json_robust != json_str:
                                tool_data = json.loads(fixed_json_robust)
                                if isinstance(tool_data, dict):
                                    if 'tool_name' in tool_data and 'parameters' in tool_data:
                                        return [{
                                            "name": tool_data["tool_name"],
                                            "arguments": tool_data["parameters"]
                                        }]
                                    elif 'name' in tool_data and 'parameters' in tool_data:
                                        return [{
                                            "name": tool_data["name"],
                                            "arguments": tool_data["parameters"]
                                        }]
                                    elif 'name' in tool_data and 'content' in tool_data:
                                        return [{
                                            "name": tool_data["name"],
                                            "arguments": tool_data["content"]
                                        }]
                        except (json.JSONDecodeError, Exception) as robust_e:
                            if self.debug_mode:
                                debug_info.append(f"Robust fix failed: {str(robust_e)[:100]}")
                            # 继续尝试正则方法
                            pass
                    
                    # 尝试使用正则表达式方法
                    try:
                        fixed_json = smart_escape_quotes_in_json_values(json_str)
                        tool_data = json.loads(fixed_json)
                        
                        if isinstance(tool_data, dict):
                            if 'name' in tool_data and 'content' in tool_data:
                                return [{
                                    "name": tool_data["name"],
                                    "arguments": tool_data["content"]
                                }]
                            elif 'tool_name' in tool_data and 'parameters' in tool_data:
                                return [{
                                    "name": tool_data["tool_name"],
                                    "arguments": tool_data["parameters"]
                                }]
                            elif 'name' in tool_data and 'parameters' in tool_data:
                                return [{
                                    "name": tool_data["name"],
                                    "arguments": tool_data["parameters"]
                                }]
                    except (json.JSONDecodeError, Exception) as regex_e:
                        if self.debug_mode:
                            debug_info.append(f"Regex fix failed: {str(regex_e)[:100]}")
                        # 如果正则方法也失败，再次尝试robust方法（可能之前没执行）
                        if 'code_edit' not in json_str and len(json_str) <= 5000:
                            try:
                                fixed_json_robust = fix_json_string_values_robust(json_str)
                                if fixed_json_robust != json_str:
                                    tool_data = json.loads(fixed_json_robust)
                                    if isinstance(tool_data, dict):
                                        if 'tool_name' in tool_data and 'parameters' in tool_data:
                                            return [{
                                                "name": tool_data["tool_name"],
                                                "arguments": tool_data["parameters"]
                                            }]
                                        elif 'name' in tool_data and 'parameters' in tool_data:
                                            return [{
                                                "name": tool_data["name"],
                                                "arguments": tool_data["parameters"]
                                            }]
                                        elif 'name' in tool_data and 'content' in tool_data:
                                            return [{
                                                "name": tool_data["name"],
                                                "arguments": tool_data["content"]
                                            }]
                            except (json.JSONDecodeError, Exception):
                                pass  # 继续到下一个修复尝试
                except Exception as e:
                    if self.debug_mode:
                        debug_info.append(f"Single JSON fallback failed: {str(e)[:100]}")
                    # Try to fix it with robust method first
                    try:
                        if 'code_edit' in json_str or len(json_str) > 5000:
                            fixed_json_robust = fix_json_string_values_robust(json_str)
                            if fixed_json_robust != json_str:
                                tool_data = json.loads(fixed_json_robust)
                                if isinstance(tool_data, dict):
                                    if 'tool_name' in tool_data and 'parameters' in tool_data:
                                        return [{
                                            "name": tool_data["tool_name"],
                                            "arguments": tool_data["parameters"]
                                        }]
                                    elif 'name' in tool_data and 'parameters' in tool_data:
                                        return [{
                                            "name": tool_data["name"],
                                            "arguments": tool_data["parameters"]
                                        }]
                                    elif 'name' in tool_data and 'content' in tool_data:
                                        return [{
                                            "name": tool_data["name"],
                                            "arguments": tool_data["content"]
                                        }]
                        
                        # 如果robust方法失败或未执行，尝试正则方法
                        fixed_json = smart_escape_quotes_in_json_values(json_str)
                        tool_data = json.loads(fixed_json)
                        if isinstance(tool_data, dict):
                            if 'name' in tool_data and ('parameters' in tool_data or 'content' in tool_data):
                                if 'name' in tool_data and 'content' in tool_data:
                                    return [{
                                        "name": tool_data["name"],
                                        "arguments": tool_data["content"]
                                    }]
                                elif 'tool_name' in tool_data and 'parameters' in tool_data:
                                    return [{
                                        "name": tool_data["tool_name"],
                                        "arguments": tool_data["parameters"]
                                    }]
                                elif 'name' in tool_data and 'parameters' in tool_data:
                                    return [{
                                        "name": tool_data["name"],
                                        "arguments": tool_data["parameters"]
                                    }]
                    except Exception as fix_e:
                        if self.debug_mode:
                            debug_info.append(f"JSON fix attempt also failed: {str(fix_e)[:100]}")
        
        # Try to parse JSON array format (AGIAgent's chat-based tool calls)
        try:
            # Look for JSON array structure in the content
            array_start = content.find('[')
            array_end = content.rfind(']') + 1
            if array_start != -1 and array_end > array_start:
                json_str = content[array_start:array_end]
                tool_array = json.loads(json_str)
                
                # Handle JSON array of tool calls
                if isinstance(tool_array, list):
                    parsed_tools = []
                    for tool_item in tool_array:
                        if isinstance(tool_item, dict):
                            if 'tool_name' in tool_item and 'parameters' in tool_item:
                                parsed_tools.append({
                                    "name": tool_item["tool_name"],
                                    "arguments": tool_item["parameters"]
                                })
                            elif 'name' in tool_item and 'parameters' in tool_item:
                                parsed_tools.append({
                                    "name": tool_item["name"],
                                    "arguments": tool_item["parameters"]
                                })
                    
                    if parsed_tools:
                        return parsed_tools
        except json.JSONDecodeError:
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
                    # Support for AGIAgent's chat-based tool calling format
                    elif 'tool_name' in tool_data and 'parameters' in tool_data:
                        return [{
                            "name": tool_data["tool_name"],
                            "arguments": tool_data["parameters"]
                        }]
        except json.JSONDecodeError as e:
            if self.debug_mode:
                debug_info.append(f"JSON array parse failed: {str(e)[:100]}")
        
        # If all parsing attempts failed, log detailed debug info
        if not all_tool_calls:
            # Check if we have JSON blocks but couldn't parse them
            has_json_markers = '```json' in content or ('{' in content and '}' in content)
            if has_json_markers and self.debug_mode:
                debug_msg = f"⚠️ Failed to parse tool calls. Debug info: {'; '.join(debug_info)}"
                debug_msg += f"\nContent length: {len(content)}"
                debug_msg += f"\nContent preview (first 500 chars): {content[:500]}"
                debug_msg += f"\nContent preview (last 500 chars): {content[-500:] if len(content) > 500 else content}"
                # 检查是否有code_edit字段
                if 'code_edit' in content:
                    code_edit_start = content.find('code_edit')
                    debug_msg += f"\nFound 'code_edit' field at position {code_edit_start}"
                    # 显示code_edit字段周围的内容
                    preview_start = max(0, code_edit_start - 100)
                    preview_end = min(len(content), code_edit_start + 200)
                    debug_msg += f"\nAround 'code_edit': {content[preview_start:preview_end]}"
                print_current(debug_msg)
            elif has_json_markers:
                # Even in non-debug mode, log a warning if we expected to find JSON
                warning_msg = f"⚠️ Warning: Found JSON markers but failed to parse tool calls. Content length: {len(content)}"
                # 检查是否有code_edit字段（可能是超长内容导致解析失败）
                if 'code_edit' in content:
                    warning_msg += f" (Contains 'code_edit' field, may be due to very long content)"
                print_current(warning_msg)
                # Try one last aggressive attempt: look for any JSON-like structure
                try:
                    # Try to find and extract any dictionary-like structure
                    brace_start = content.find('{')
                    if brace_start != -1:
                        # Try to find matching closing brace
                        brace_count = 0
                        in_string = False
                        escape_next = False
                        brace_end = -1
                        for i in range(brace_start, len(content)):
                            char = content[i]
                            if escape_next:
                                escape_next = False
                                continue
                            if char == '\\':
                                escape_next = True
                                continue
                            if char == '"' and not escape_next:
                                in_string = not in_string
                                continue
                            if not in_string:
                                if char == '{':
                                    brace_count += 1
                                elif char == '}':
                                    brace_count -= 1
                                    if brace_count == 0:
                                        brace_end = i + 1
                                        break
                        
                        if brace_end > brace_start:
                            potential_json = content[brace_start:brace_end]
                            try:
                                fixed = smart_escape_quotes_in_json_values(potential_json)
                                tool_data = json.loads(fixed)
                                if isinstance(tool_data, dict):
                                    # Try to infer what this is
                                    if 'tool_name' in tool_data and 'parameters' in tool_data:
                                        return [{
                                            "name": tool_data["tool_name"],
                                            "arguments": tool_data["parameters"]
                                        }]
                                    elif 'name' in tool_data and ('parameters' in tool_data or 'content' in tool_data):
                                        return [{
                                            "name": tool_data["name"],
                                            "arguments": tool_data.get("parameters", tool_data.get("content", {}))
                                        }]
                            except Exception:
                                pass
                except Exception:
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
                        params = parse_python_params_manually(params_str)
                        if params:
                            function_calls.append({
                                "name": tool_name,
                                "arguments": params
                            })
                    except Exception as e2:
                        continue
        
        return function_calls
    

    
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
            args[name] = convert_parameter_value(value)
        
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
                        args[tag_name] = convert_parameter_value(value)
                else:
                    args[tag_name] = convert_parameter_value(value)
        
        return args
    

    
    def execute_tool(self, tool_call: Dict[str, Any], streaming_output: bool = False) -> Any:
        """
        Execute a tool with the given parameters, optionally with streaming output.
        
        Args:
            tool_call: Dictionary containing tool name and parameters
            streaming_output: Whether to stream the tool execution output
            
        Returns:
            Result of executing the tool
        """
        tool_name = tool_call["name"]
        params = tool_call["arguments"]
        
        # 🔧 Error handling: If the large model calls the send_message_to_manager tool, correct it to send_message_to_agent_or_manager
        if tool_name == "send_message_to_manager":
            print_current(f"🔧 Auto-correcting tool call: {tool_name} -> send_message_to_agent_or_manager")
            tool_name = "send_message_to_agent_or_manager"
            
            # If receiver_id parameter is missing, set it to manager
            if "receiver_id" not in params:
                params["receiver_id"] = "manager"
                print_current(f"🔧 Added missing receiver_id parameter: manager")
            
            # Update tool_call object to reflect the correction
            tool_call["name"] = tool_name
        
        # Check tool source from mapping table
        tool_source = getattr(self, 'tool_source_map', {}).get(tool_name, 'regular')
        
        # 🔄 延迟加载MCP: 在以下情况下初始化MCP
        # 1. 工具来源已标记为MCP工具
        # 2. 工具不在tool_map中,且工具名称符合MCP命名规则(包含下划线或特定前缀)
        should_init_mcp = False
        if not self.mcp_initialization_attempted:
            if tool_source in ['fastmcp', 'cli_mcp']:
                # 情况1: 已知是MCP工具
                should_init_mcp = True
            elif tool_name not in self.tool_map:
                # 情况2: 工具不存在,且可能是MCP工具(根据命名规则判断)
                # MCP工具通常包含下划线,如: taobao_search, filesystem_read_file
                # 排除已知的非MCP工具
                known_regular_tools = {
                    'list_files', 'read_file', 'write_file', 'edit_file', 'delete_file',
                    'execute_bash', 'search_files', 'search_content', 'create_directory',
                    'list_directory', 'move_file', 'copy_file'
                }
                if tool_name not in known_regular_tools and '_' in tool_name:
                    should_init_mcp = True
        
        if should_init_mcp:
            self._ensure_mcp_initialized()
            # 重新检查工具来源,因为初始化后可能会改变
            tool_source = getattr(self, 'tool_source_map', {}).get(tool_name, 'regular')
        
        # Handle FastMCP tools
        if tool_source == 'fastmcp':
            current_thread = threading.current_thread().name
            print_debug(f"🚀 [Thread: {current_thread}] Calling FastMCP tool: {tool_name}")
            
            try:

                if tool_name in self.tool_map:
                    # Execute the tool function with streaming awareness
                    if streaming_output:
                        # For streaming output, show execution in real-time
                        self._stream_tool_execution(tool_name, params, self.tool_map[tool_name])
                        result = self.tool_map[tool_name](**params)
                        # Stream the result output immediately after execution
                        self._stream_tool_result(tool_name, result, params)
                    else:
                        result = self.tool_map[tool_name](**params)
                    
                    print_debug(f"✅ [Thread: {current_thread}] FastMCP tool call successful: {tool_name}")
                    return result
                else:
                    error_msg = f"FastMCP tool {tool_name} not found in tool map"
                    print_current(f"❌ [Thread: {current_thread}] {error_msg}")
                    return {"error": error_msg}
                    
            except Exception as e:
                error_msg = f"FastMCP tool call failed: {e}"
                print_current(f"❌ [Thread: {current_thread}] {error_msg}")
                return {"error": error_msg}
        
        # Handle cli-mcp tools
        if tool_source == 'cli_mcp':
            # Enhanced multi-thread initialization for cli-mcp client
            current_thread = threading.current_thread().name
            
            # First check: instance-level initialization status
            if not self.cli_mcp_initialized:
                print_debug(f"🔄 [Thread: {current_thread}] cli-mcp client not initialized, attempting initialization for tool {tool_name}...")
                
                # Second check: global cli-mcp wrapper status
                from tools.cli_mcp_wrapper import get_cli_mcp_status, is_cli_mcp_initialized, initialize_cli_mcp_wrapper
                
                global_status = get_cli_mcp_status(self.MCP_config_file)
                print_debug(f"🔍 [Thread: {current_thread}] Global cli-mcp status: {global_status}")
                
                # If globally initialized but not locally, sync the status
                if global_status.get("initialized", False) and not self.cli_mcp_initialized:
                    print_debug(f"🔄 [Thread: {current_thread}] Global cli-mcp is initialized, syncing local status...")
                    self.cli_mcp_initialized = True
                    self._add_mcp_tools_to_map()
                    print_debug(f"✅ [Thread: {current_thread}] Local cli-mcp status synced successfully")
                
                # If still not initialized, attempt initialization with enhanced retry
                if not self.cli_mcp_initialized:
                    retry_count = 0
                    max_retries = 3
                    
                    while retry_count < max_retries and not self.cli_mcp_initialized:
                        try:
                            retry_count += 1
                            print_debug(f"🔄 [Thread: {current_thread}] cli-mcp initialization attempt {retry_count}/{max_retries}")
                            
                            import asyncio
                            
                            # Enhanced async handling for different thread contexts
                            try:
                                loop = asyncio.get_running_loop()
                                if loop.is_running():
                                    # We're in an async context, use thread pool for initialization
                                    import concurrent.futures
                                    with concurrent.futures.ThreadPoolExecutor() as executor:
                                        future = executor.submit(asyncio.run, initialize_cli_mcp_wrapper(self.MCP_config_file))
                                        init_result = future.result(timeout=20)  # Increased timeout
                                        self.cli_mcp_initialized = init_result
                                else:
                                    # We can run the async function directly
                                    self.cli_mcp_initialized = asyncio.run(initialize_cli_mcp_wrapper(self.MCP_config_file))
                            except RuntimeError as re:
                                # No event loop or other runtime issues
                                print_debug(f"⚠️ [Thread: {current_thread}] Runtime error during async init: {re}")
                                # Try creating new event loop
                                try:
                                    new_loop = asyncio.new_event_loop()
                                    asyncio.set_event_loop(new_loop)
                                    self.cli_mcp_initialized = new_loop.run_until_complete(initialize_cli_mcp_wrapper(self.MCP_config_file))
                                    new_loop.close()
                                except Exception as loop_e:
                                    print_debug(f"❌ [Thread: {current_thread}] Failed to create new event loop: {loop_e}")
                                    self.cli_mcp_initialized = False
                            
                            # Verify initialization and add tools to mapping
                            if self.cli_mcp_initialized:
                                self._add_mcp_tools_to_map()
                                print_debug(f"✅ [Thread: {current_thread}] cli-mcp client initialized successfully with config: {self.MCP_config_file}")
                                
                                # Double-check the tool mapping
                                if tool_name in self.tool_map:
                                    print_debug(f"✅ [Thread: {current_thread}] Tool {tool_name} found in tool mapping")
                                else:
                                    print_debug(f"⚠️ [Thread: {current_thread}] Tool {tool_name} NOT found in tool mapping after initialization")
                                break
                            else:
                                print_debug(f"⚠️ [Thread: {current_thread}] cli-mcp initialization attempt {retry_count} failed")
                                if retry_count < max_retries:
                                                time.sleep(3)  # Wait 3 seconds before retry
                                    
                        except Exception as e:
                            print_debug(f"⚠️ [Thread: {current_thread}] cli-mcp client initialization attempt {retry_count} failed: {e}")
                            if retry_count < max_retries:
                                            time.sleep(3)  # Wait 3 seconds before retry
                            else:
                                error_msg = f"cli-mcp client initialization failed after {max_retries} attempts in thread {current_thread}: {e}"
                                print_debug(f"❌ {error_msg}")
                                return {"error": error_msg}
                    
                    # Final check
                    if not self.cli_mcp_initialized:
                        error_msg = f"cli-mcp client failed to initialize after all attempts in thread {current_thread}"
                        print_debug(f"❌ {error_msg}")
                        return {"error": error_msg}
            
            # Call cli-mcp tool with enhanced error handling
            try:
                print_debug(f"🔧 [Thread: {current_thread}] Calling cli-mcp tool: {tool_name}")
                
                import asyncio
                
                # Enhanced async execution handling
                try:
                    loop = asyncio.get_running_loop()
                    if loop.is_running():
                        # We're in an async context, use thread pool for execution
                        import concurrent.futures
                        with concurrent.futures.ThreadPoolExecutor() as executor:
                            # Remove prefix from tool name if present
                            actual_tool_name = tool_name.replace("cli_mcp_", "")
                            print_debug(f"🎯 [Thread: {current_thread}] Calling actual tool: {actual_tool_name} with params: {params}")
                            future = executor.submit(asyncio.run, self.cli_mcp_client.call_tool(actual_tool_name, params))
                            result = future.result(timeout=35)  # Increased timeout
                            print_debug(f"✅ [Thread: {current_thread}] cli-mcp tool call completed successfully")
                            return result
                except RuntimeError:
                    # No event loop, safe to run asyncio.run
                    pass
                
                # Synchronous execution
                actual_tool_name = tool_name.replace("cli_mcp_", "")
                print_debug(f"🎯 [Thread: {current_thread}] Calling actual tool (sync): {actual_tool_name} with params: {params}")
                result = asyncio.run(self.cli_mcp_client.call_tool(actual_tool_name, params))
                print_debug(f"✅ [Thread: {current_thread}] cli-mcp tool call completed successfully (sync)")
                return result
                
            except Exception as e:
                error_msg = f"cli-mcp tool call failed in thread {current_thread}: {e}"
                print_debug(f"❌ {error_msg}")
                return {"error": error_msg}
        
        # Handle direct MCP tools (SSE)
        elif tool_source == 'direct_mcp':
            # Ensure direct MCP client is initialized
            if not self.direct_mcp_initialized:
                print_debug(f"🔄 Attempting to initialize direct MCP client for tool {tool_name}...")
                import asyncio
                
                retry_count = 0
                max_retries = 3
                
                while retry_count < max_retries and not self.direct_mcp_initialized:
                    try:
                        retry_count += 1
                        print_debug(f"🔄 Direct MCP initialization attempt {retry_count}/{max_retries}")
                        
                        loop = asyncio.get_running_loop()
                        if loop.is_running():
                            # We're in an async context, use thread pool for initialization
                            import concurrent.futures
                            with concurrent.futures.ThreadPoolExecutor() as executor:
                                future = executor.submit(asyncio.run, self.direct_mcp_client.initialize())
                                self.direct_mcp_initialized = future.result(timeout=15)  # Increased timeout
                        else:
                            # We can run the async function directly
                            self.direct_mcp_initialized = asyncio.run(self.direct_mcp_client.initialize())
                        
                        # Add MCP tools to tool_map after successful initialization
                        if self.direct_mcp_initialized:
                            self._add_mcp_tools_to_map()
                            print_debug(f"✅ Direct MCP client initialized with config: {self.MCP_config_file}")
                            break
                        else:
                            print_debug(f"⚠️ Direct MCP initialization attempt {retry_count} failed")
                            if retry_count < max_retries:
                                            time.sleep(2)  # Wait 2 seconds before retry
                                
                    except Exception as e:
                        print_debug(f"⚠️ Direct MCP client initialization attempt {retry_count} failed: {e}")
                        if retry_count < max_retries:
                                        time.sleep(2)  # Wait 2 seconds before retry
                        else:
                            return {"error": f"Direct MCP client initialization failed after {max_retries} attempts: {e}"}
            
            # Call direct MCP tool
            try:
                import asyncio
                
                # Check if in async environment
                try:
                    loop = asyncio.get_running_loop()
                    if loop.is_running():
                        # In async environment, use thread pool for execution
                        import concurrent.futures
                        with concurrent.futures.ThreadPoolExecutor() as executor:
                            # Use tool name directly (no prefix removal needed for SSE tools)
                            future = executor.submit(asyncio.run, self.direct_mcp_client.call_tool(tool_name, params))
                            result = future.result(timeout=300)  # 300 seconds timeout for long-running tasks like image generation
                            return result
                except RuntimeError:
                    # No event loop, safe to run asyncio.run
                    pass
                
                # Run async function in sync environment
                result = asyncio.run(self.direct_mcp_client.call_tool(tool_name, params))
                return result
                
            except Exception as e:
                print_debug(f"❌ SSE MCP tool call failed: {e}")
                return {"error": f"SSE MCP tool call failed: {e}"}
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
                
                # Special handling for read_file to map end_line_one_indexed to end_line_one_indexed_inclusive
                if tool_name == "read_file" and "end_line_one_indexed" in filtered_params:
                    # Map end_line_one_indexed to end_line_one_indexed_inclusive
                    filtered_params["end_line_one_indexed_inclusive"] = filtered_params.pop("end_line_one_indexed")
                    #print_current("Mapped end_line_one_indexed parameter to end_line_one_indexed_inclusive")
                
                # Robustness handling: auto-correct wrong parameter names for edit_file and read_file
                if tool_name in ["edit_file", "read_file"]:
                    # Map relative_workspace_path to target_file
                    if "relative_workspace_path" in filtered_params:
                        filtered_params["target_file"] = filtered_params.pop("relative_workspace_path")
                        print_current(f"🔧 Auto-corrected parameter: relative_workspace_path -> target_file for {tool_name}")
                    # Map file_path to target_file
                    if "file_path" in filtered_params:
                        filtered_params["target_file"] = filtered_params.pop("file_path")
                        print_current(f"🔧 Auto-corrected parameter: file_path -> target_file for {tool_name}")
                    # Map filename to target_file (for edit_file)
                    if "filename" in filtered_params:
                        filtered_params["target_file"] = filtered_params.pop("filename")
                        print_current(f"🔧 Auto-corrected parameter: filename -> target_file for {tool_name}")
                
                # Robustness handling for edit_file: auto-correct content to code_edit
                if tool_name == "edit_file" and "content" in filtered_params:
                    # Map content to code_edit
                    filtered_params["code_edit"] = filtered_params.pop("content")
                    print_current(f"🔧 Auto-corrected parameter: content -> code_edit for {tool_name}")
                
                # Robustness handling for workspace_search: auto-correct search_term to query
                if tool_name == "workspace_search" and "search_term" in filtered_params:
                    # Map search_term to query
                    filtered_params["query"] = filtered_params.pop("search_term")
                    print_current(f"🔧 Auto-corrected parameter: search_term -> query for {tool_name}")
                
                # 🔧 Robustness handling for multi-agent messaging tools: auto-add missing content parameter
                if tool_name in ["send_message_to_agent_or_manager", "broadcast_message_to_agents"]:
                    if "content" not in filtered_params:
                        # Provide default content based on message context
                        if tool_name == "send_message_to_agent_or_manager":
                            receiver_id = filtered_params.get("receiver_id", "unknown")
                            message_type = filtered_params.get("message_type", "status_update")
                            filtered_params["content"] = {
                                "message": f"Automated message to {receiver_id}",
                                "type": message_type,
                                "status": "active"
                            }
                        elif tool_name == "broadcast_message_to_agents":
                            message_type = filtered_params.get("message_type", "broadcast")
                            filtered_params["content"] = {
                                "message": "Broadcast message to all agents",
                                "type": message_type,
                                "status": "active"
                            }
                        print_current(f"🔧 Auto-added missing content parameter for {tool_name}: {filtered_params['content']}")
                
                # No special handling needed for run_terminal_cmd anymore
                
                # Execute the tool function with streaming awareness
                if streaming_output:
                    # For streaming output, show execution in real-time
                    self._stream_tool_execution(tool_name, filtered_params, tool_func)
                    result = tool_func(**filtered_params)
                    # Stream the result output immediately after execution
                    self._stream_tool_result(tool_name, result, filtered_params)
                else:
                    result = tool_func(**filtered_params)
                
                # Enhanced error handling for edit_file and other tools
                if isinstance(result, dict) and result.get('status') == 'error':
                    # For terminal commands, preserve stdout and stderr information
                    if tool_name == 'run_terminal_cmd':
                        # Keep the original result with all details for terminal commands
                        return result
                    else:
                        # Return detailed error information for other failed tool executions
                        error_msg = result.get('error', result.get('message', 'Unknown error occurred'))
                        return {
                            'tool': tool_name,
                            'status': 'error', 
                            'error': error_msg,
                            'parameters': filtered_params,
                            'details': result  # Include original result for debugging
                        }
                
                return result
            except TypeError as e:
                # Handle parameter mismatch with helpful guidance
                error_msg = f"Parameter mismatch: {str(e)}"
                
                # Add specific guidance for common parameter issues
                if tool_name == 'edit_file' and 'code_edit' in str(e):
                    error_msg += "\n💡 HINT: edit_file requires 'code_edit' parameter. Example: \"code_edit\": \"your code content here\""
                elif tool_name == 'edit_file' and any(param in str(e) for param in ['start_line', 'end_line']):
                    error_msg += "\n💡 HINT: edit_file replace_lines mode requires 'start_line_one_indexed' and 'end_line_one_indexed_inclusive' parameters"
                elif tool_name == 'read_file' and any(param in str(e) for param in ['start_line', 'end_line', 'should_read']):
                    error_msg += "\n💡 HINT: read_file requires 'target_file' and 'should_read_entire_file'. Line parameters are optional when should_read_entire_file=true"
                elif tool_name == 'run_terminal_cmd' and 'is_background' in str(e):
                    error_msg += "\n💡 HINT: run_terminal_cmd requires 'command' and 'is_background' parameters"
                
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
        
        # Handle key fields in priority order
        field_handlers = [
            ('result', lambda v: f"Result: {v}"),
            ('content', self._format_content_field),
            ('file', lambda v: f"File: {v}"),
            ('output', lambda v: f"Output:\n{v}"),
            ('message', lambda v: f"Message: {v}"),
        ]
        
        # Process high-priority fields
        processed_keys = {'error', 'status', 'success'}
        for field_name, handler in field_handlers:
            if field_name in data:
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

    def _save_llm_call_debug_log(self, messages: List[Dict[str, Any]], content: str, tool_call_round: int = 0, tool_calls_info: Optional[Dict[str, Any]] = None) -> None:
        """
        Save detailed debug log for LLM call.
        
        Args:
            messages: Complete messages sent to LLM
            content: LLM response content
            tool_call_round: Current tool call round number
            tool_calls_info: Additional tool call information for better logging
        """
        try:
            # Increment call counter
            self.llm_call_counter += 1
            
            # Create timestamp
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # microseconds to milliseconds
            
            current_agent_id = get_current_agent_id()
            if current_agent_id:
                log_filename = f"llm_call_{current_agent_id}_{self.llm_call_counter:03d}_{timestamp}.json"
            else:
                log_filename = f"llm_call_{self.llm_call_counter:03d}_{timestamp}.json"
            
            # Only create log path if logs directory is available
            if self.llm_logs_dir:
                log_path = os.path.join(self.llm_logs_dir, log_filename)
            else:
                log_path = None
            
            # 🔧 Apply message optimization to remove base64 data from logs
            optimized_messages = self._optimize_messages_for_logging(messages)
            
            # Prepare debug data - including detailed tool call information
            debug_data = {
                "call_info": {
                    "call_number": self.llm_call_counter,
                    "timestamp": datetime.datetime.now().isoformat(),
                    "model": self.model,
                    "tool_call_round": tool_call_round  # Track which tool call round this is
                },
                "messages": optimized_messages,
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
            
            # Save to JSON file only if log_path is available
            if log_path:
                with open(log_path, 'w', encoding='utf-8') as f:
                    # Convert escaped newlines to actual newlines in response_content
                    if "response_content" in debug_data:
                        debug_data["response_content"] = debug_data["response_content"].replace('\\n', '\n')

                    # Convert escaped newlines in messages content
                    if "messages" in debug_data:
                        for message in debug_data["messages"]:
                            if "content" in message:
                                message["content"] = message["content"].replace('\\n', '\n')

                    # Convert escaped newlines in tool_calls_info
                    if "tool_calls_info" in debug_data:
                        tool_calls_info = debug_data["tool_calls_info"]
                        if "formatted_tool_results" in tool_calls_info:
                            tool_calls_info["formatted_tool_results"] = tool_calls_info["formatted_tool_results"].replace('\\n', '\n')

                    json.dump(debug_data, f, ensure_ascii=False, indent=2)
            
            
        except Exception as e:
            print_current(f"⚠️ Debug log save failed: {e}")

    def _optimize_messages_for_logging(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Optimize messages by replacing base64 data with references for logging purposes.
        
        Args:
            messages: Original messages list
            
        Returns:
            Optimized messages list with base64 data replaced by references
        """
        if not hasattr(self, 'history_optimizer') or not self.history_optimizer:
            return messages
        
        optimized_messages = []
        
        for message in messages:
            optimized_message = message.copy()
            
            # Check and optimize content field
            if 'content' in message and isinstance(message['content'], str):
                optimized_content = self._optimize_text_for_logging(message['content'])
                optimized_message['content'] = optimized_content
            
            optimized_messages.append(optimized_message)
        
        return optimized_messages
    
    def _optimize_text_for_logging(self, text: str) -> str:
        """
        Optimize text content by replacing base64 data with lightweight references for logging.
        
        Args:
            text: Original text content
            
        Returns:
            Optimized text with base64 data replaced by references
        """
        if not text or not isinstance(text, str):
            return text
        
        import re
        import hashlib
        
        # Detect base64 image data patterns
        base64_pattern = r'[A-Za-z0-9+/]{500,}={0,2}'
        matches = list(re.finditer(base64_pattern, text))
        
        if not matches:
            return text
        
        optimized_text = text
        offset = 0
        
        for match in matches:
            base64_data = match.group()
            
            # Calculate image hash for reference
            image_hash = hashlib.md5(base64_data.encode()).hexdigest()[:16]
            
            # Extract file path info if present
            file_marker_pattern = r'\[FILE_(?:SOURCE|SAVED):([^\]]+)\]'
            file_match = re.search(file_marker_pattern, base64_data)
            file_info = f"|{file_match.group(1)}" if file_match else ""
            
            # Estimate size
            estimated_size_kb = len(base64_data) * 3 // 4 // 1024
            
            # Create compact reference
            reference_text = f"[IMAGE_DATA_REF:{image_hash}|{estimated_size_kb}KB{file_info}]"
            
            # Calculate position in adjusted text
            start_pos = match.start() + offset
            end_pos = match.end() + offset
            
            # Replace base64 data with reference
            optimized_text = (optimized_text[:start_pos] + 
                            reference_text + 
                            optimized_text[end_pos:])
            
            # Update offset
            offset += len(reference_text) - len(base64_data)
        
        return optimized_text

    def _display_llm_statistics(self, messages: List[Dict[str, Any]], response_content: str, tool_calls: Optional[List[Dict[str, Any]]] = None) -> None:
        """
        Display LLM input/output statistics including token count and character count.
        
        Args:
            messages: Input messages sent to LLM
            response_content: Response content from LLM
            tool_calls: Tool calls from LLM response (optional)
        """
        try:
            # Calculate input statistics
            input_text = ""
            for message in messages:
                role = message.get("role", "")
                content = message.get("content", "")
                input_text += f"[{role}] {content}\n"
            
            # Detect if input contains images (base64 data)
            import re
            has_images = bool(re.search(r'[A-Za-z0-9+/]{100,}={0,2}', input_text))
            
            # Estimate token counts for response content, including image tokens
            input_tokens_est = estimate_token_count(input_text, has_images=has_images, model=self.model)
            output_tokens_est = estimate_token_count(response_content, has_images=False, model=self.model)
            
            # Estimate token counts for tool calls if present
            tool_calls_tokens = 0
            if tool_calls:
                tool_calls_text = self._format_tool_calls_for_token_estimation(tool_calls)
                tool_calls_tokens = estimate_token_count(tool_calls_text)
            
            # Total output tokens including tool calls
            total_output_tokens = output_tokens_est + tool_calls_tokens
            
            # Calculate cache-related statistics
            cache_stats = analyze_cache_potential(messages, self.previous_messages)
            
            # Display simplified statistics in one line
            cached_tokens = cache_stats['estimated_cache_tokens']
            new_input_tokens = cache_stats['new_tokens']

            # print_current(f"📊 Input cached tokens: {cached_tokens:,}, Input new tokens: {new_input_tokens:,}, Output tokens: {total_output_tokens:,}")  # Reduced verbose output
            
        except Exception as e:
            print_current(f"⚠️ Statistics calculation failed: {e}")

    def _format_tool_calls_for_token_estimation(self, tool_calls: List[Dict[str, Any]]) -> str:
        """
        Format tool calls into text for token estimation.
        
        Args:
            tool_calls: List of tool calls
            
        Returns:
            Formatted text representation of tool calls
        """
        if not tool_calls:
            return ""
        
        formatted_parts = []
        for tool_call in tool_calls:
            # Handle different tool call formats
            if isinstance(tool_call, dict):
                # Extract tool name
                tool_name = ""
                if "name" in tool_call:
                    tool_name = tool_call["name"]
                elif "function" in tool_call and isinstance(tool_call["function"], dict):
                    tool_name = tool_call["function"].get("name", "")
                
                # Extract parameters/arguments
                params = {}
                if "arguments" in tool_call:
                    params = tool_call["arguments"]
                elif "input" in tool_call:
                    params = tool_call["input"]
                elif "function" in tool_call and isinstance(tool_call["function"], dict):
                    if "arguments" in tool_call["function"]:
                        try:
                            import json
                            params = json.loads(tool_call["function"]["arguments"]) if isinstance(tool_call["function"]["arguments"], str) else tool_call["function"]["arguments"]
                        except:
                            params = tool_call["function"]["arguments"]
                
                # Format tool call as text
                tool_text = f"Tool: {tool_name}\n"
                if params:
                    import json
                    try:
                        params_text = json.dumps(params, ensure_ascii=False)
                        tool_text += f"Parameters: {params_text}\n"
                    except:
                        tool_text += f"Parameters: {str(params)}\n"
                
                formatted_parts.append(tool_text)
        
        return "\n".join(formatted_parts)
    
    # Cache analysis functions moved to utils/cacheeff.py
    


    def _format_tool_results_for_llm(self, tool_results: List[Dict[str, Any]], include_base64_info: bool = False) -> str:
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
                for key, value in tool_params.items():
                    if key in ['target_file', 'query', 'command', 'relative_workspace_path', 'search_term', 'instructions']:
                        # Show full parameter values without truncation
                        key_params.append(f"{key}={value}")
                if key_params:
                    message_parts.append(f"**Parameters:** {', '.join(key_params)}")
            
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
        
        # Add base64 data detection information
        if include_base64_info:
            message_parts.append("")
            message_parts.append("## Base64 Data Status")
            message_parts.append("✅ Base64 encoded image data has been successfully acquired in this round.")
        
        return '\n'.join(message_parts)
    
    def _format_tool_results_with_vision(self, tool_results: List[Dict[str, Any]], vision_images: List[Dict[str, Any]]) -> Any:
        """
        Format tool results that contain vision data for LLM.
        Returns the proper format for vision-capable models.
        
        Args:
            tool_results: List of tool execution results
            vision_images: List of vision image data
            
        Returns:
            Properly formatted content for vision models (content array format)
        """
        truncation_length = get_truncation_length()
        
        # Build text content first
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
                        # Show full parameter values without truncation
                        key_params.append(f"{key}={value}")
                if key_params:
                    message_parts.append(f"**Parameters:** {', '.join(key_params)}")
            
            # Format the result
            message_parts.append("**Result:**")
            if isinstance(tool_result, dict):
                if tool_result.get('success') is not None:
                    # Structured result format
                    status = "✅ Success" if tool_result.get('success') else "❌ Failed"
                    message_parts.append(status)
                    
                    for key, value in tool_result.items():
                        # For image data in get_sensor_data, show metadata but reference image below
                        if (tool_name == 'get_sensor_data' and key == 'data' and 
                            any(img['tool_index'] == i for img in vision_images)):
                            message_parts.append(f"- {key}: [IMAGE DATA - See image below]")
                            print_current(f"📸 Image data formatted for vision API, tool {i}")
                        elif key not in ['status', 'command', 'working_directory']:
                            # Show full content without truncation
                            message_parts.append(f"- {key}: {value}")
            
            # Add separator between tools
            if i < len(tool_results):
                message_parts.append("")  # Empty line for separation
        
        # Build content array with text and images
        text_content = '\n'.join(message_parts)
        
        # Create content array format for vision models
        content_parts = []
        
        # Add text part
        content_parts.append({
            "type": "text",
            "text": text_content
        })
        
        # Add image parts
        for img_data in vision_images:
            if self.is_claude:
                # Claude format
                content_parts.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": img_data['mime_type'],
                        "data": img_data['data']
                    }
                })
            else:
                # OpenAI format
                content_parts.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{img_data['mime_type']};base64,{img_data['data']}"
                    }
                })
        
        print_current(f"🖼️ Formatted {len(vision_images)} images for vision API ({self.model})")
        return content_parts


    

    









    

    

    

    




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
            
            #lines.append(f"🔍 Code search for '{query}': Found {total_results} results")
            
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
            
            # Add repository stats briefly
            #stats = data.get('repository_stats', {})
            #if stats:
            #    lines.append(f"📊 Repository: {stats.get('total_files', 0)} files, {stats.get('total_segments', 0)} segments")
        
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
            
            # Add metadata briefly
            #if data.get('content_fetched'):
            #    lines.append(f"📄 Content fetched: {data['content_fetched']}")
        
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
            
            # Handle direct MCP tools (SSE)
            elif tool_source == 'direct_mcp':
                if self.direct_mcp_client and self.direct_mcp_initialized:
                    try:
                        # Use tool name directly (no prefix for SSE tools)
                        direct_mcp_tool_def = self.direct_mcp_client.get_tool_definition(tool_name)
                        if direct_mcp_tool_def:
                            if provider == "openai":
                                # OpenAI format for direct MCP tools
                                standard_tool = {
                                    "type": "function",
                                    "function": {
                                        "name": tool_name,  # No prefix for SSE tools
                                        "description": direct_mcp_tool_def.get("description", f"SSE MCP tools: {tool_name}"),
                                        "parameters": direct_mcp_tool_def.get("inputSchema", direct_mcp_tool_def.get("input_schema", {
                                            "type": "object",
                                            "properties": {},
                                            "required": []
                                        }))
                                    }
                                }
                            elif provider == "anthropic":
                                # Anthropic format for direct MCP tools
                                standard_tool = {
                                    "name": tool_name,  # No prefix for SSE tools
                                    "description": direct_mcp_tool_def.get("description", f"SSE MCP tools: {tool_name}"),
                                    "input_schema": direct_mcp_tool_def.get("inputSchema", direct_mcp_tool_def.get("input_schema", {
                                        "type": "object",
                                        "properties": {},
                                        "required": []
                                    }))
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

    def _call_llm_with_standard_tools(self, messages, user_message, system_message):
        """
        Call LLM with either standard tool calling format or chat-based tool calling.
        
        Args:
            messages: Message history for the LLM
            user_message: Current user message
            system_message: System message
            
        Returns:
            Tuple of (content, tool_calls)
        """
        # 🚀 确保LLM客户端已初始化（延迟加载）
        self._setup_llm_client()
        
        if self.use_chat_based_tools:
            return self._call_llm_with_chat_based_tools(messages, user_message, system_message)
        elif self.is_glm:
            return self._call_glm_with_standard_tools(messages, user_message, system_message)
        elif self.is_claude:
            return self._call_claude_with_standard_tools(messages, user_message, system_message)
        else:
            return self._call_openai_with_standard_tools(messages, user_message, system_message)

    def _call_llm_with_chat_based_tools(self, messages, user_message, system_message):
        """
        Call LLM with chat-based tool calling (no standard tool calling format).
        Tools are described in the message and responses are parsed from content.
        
        Args:
            messages: Message history for the LLM
            user_message: Current user message
            system_message: System message
            
        Returns:
            Tuple of (content, tool_calls)
        """
        # Retry logic for retryable errors
        max_retries = 3
        for attempt in range(max_retries + 1):  # 0, 1, 2, 3 (4 total attempts)
            try:
                if self.is_claude:
                    # Use Anthropic Claude API for chat-based tool calling
                    claude_messages = [{"role": "user", "content": user_message}]
                    
                    if self.streaming:
                        with streaming_context(show_start_message=False) as printer:
                            # 显示LLM开始说话的emoji
                            printer.write(f"\n💬 ")
                            
                            with self.client.messages.stream(
                                model=self.model,
                                max_tokens=self._get_max_tokens_for_model(self.model),
                                system=system_message,
                                messages=claude_messages,
                                temperature=0.7
                            ) as stream:
                                content = ""
                                hallucination_detected = False
                                json_block_detected = False
                                stream_error_occurred = False
                                stream_error_message = ""
                                
                                # 缓冲打印机制：至少缓冲100个字符
                                buffer = ""
                                min_buffer_size = 100
                                total_printed = 0
                                
                                try:
                                    for text in stream.text_stream:
                                        buffer += text
                                        content += text
                                        
                                        # Check for hallucination patterns - strict match (检查整个 content，避免打印幻觉字符串)
                                        hallucination_patterns = [
                                            "LLM Called Following Tools in this round",
                                            "**Tool Execution Results:**"
                                        ]
                                        hallucination_detected_flag = False
                                        hallucination_start = -1
                                        for pattern in hallucination_patterns:
                                            if pattern in content:
                                                hallucination_start = content.find(pattern)
                                                hallucination_detected_flag = True
                                                break
                                        
                                        if hallucination_detected_flag:
                                            print_debug("\n🚨 Hallucination Detected, stop chat")
                                            hallucination_detected = True
                                            # 截断 buffer 和 content 到幻觉位置之前，避免打印幻觉字符串
                                            if hallucination_start > 0:
                                                content = content[:hallucination_start].rstrip()
                                                # 检查 buffer 中是否已经包含了幻觉字符串
                                                if len(buffer) > len(content) - total_printed:
                                                    # buffer 中包含幻觉字符串，需要截断
                                                    buffer = content[total_printed:] if len(content) > total_printed else ""
                                                else:
                                                    # buffer 还没有包含幻觉字符串，保持原样
                                                    pass
                                            else:
                                                # 幻觉字符串在开头，清空 buffer 和 content
                                                buffer = ""
                                                content = ""
                                            break
                                        
                                        # 当缓冲区达到最小大小时，打印缓冲区内容
                                        # 但需要检查是否已经包含了第二个```json（如果检测到工具调用）
                                        if len(buffer) >= min_buffer_size:
                                            # 检查是否应该提前截断（如果已经检测到第二个工具调用）
                                            if self._is_complete_json_tool_call(content):
                                                # 找到第二个```json的位置
                                                content_to_print = self._get_content_before_second_json(content)
                                                # 只打印到第二个```json之前的内容
                                                if len(content_to_print) < total_printed + len(buffer):
                                                    # 需要截断
                                                    to_print = content_to_print[total_printed:]
                                                    if to_print:
                                                        printer.write(to_print)
                                                        total_printed = len(content_to_print)
                                                    buffer = ""
                                                    json_block_detected = True
                                                    break
                                            else:
                                                # 正常打印
                                                printer.write(buffer)
                                                total_printed += len(buffer)
                                                buffer = ""
                                        
                                        # 检测工具调用：检查是否包含完整的工具调用JSON（支持带```json标记和不带标记的纯JSON）
                                        if self._has_complete_json_tool_call(content):
                                            json_block_detected = True
                                            # 在break之前，先打印buffer中工具调用之前的内容
                                            # 找到第二个```json的位置，只打印到那里之前的内容
                                            content_to_print = self._get_content_before_second_json(content)
                                            # 计算需要打印的内容：从total_printed到content_to_print的末尾
                                            remaining_to_print = content_to_print[total_printed:]
                                            if remaining_to_print:
                                                printer.write(remaining_to_print)
                                                total_printed = len(content_to_print)
                                            # 清空buffer，因为已经打印了所有应该打印的内容
                                            buffer = ""
                                            break
                                        
                                        # 额外检查：如果检测到纯JSON格式的工具调用（不带```json标记），也停止接收
                                        if '"tool_name"' in content and '"parameters"' in content:
                                            # 检查是否已经有一个完整的JSON对象
                                            try:
                                                brace_start = content.find('{')
                                                if brace_start != -1:
                                                    brace_count = 0
                                                    in_string = False
                                                    escape_next = False
                                                    brace_end = -1
                                                    
                                                    for i in range(brace_start, len(content)):
                                                        char = content[i]
                                                        if escape_next:
                                                            escape_next = False
                                                            continue
                                                        if char == '\\':
                                                            escape_next = True
                                                            continue
                                                        if char == '"' and not escape_next:
                                                            in_string = not in_string
                                                            continue
                                                        if not in_string:
                                                            if char == '{':
                                                                brace_count += 1
                                                            elif char == '}':
                                                                brace_count -= 1
                                                                if brace_count == 0:
                                                                    brace_end = i + 1
                                                                    break
                                                    
                                                    if brace_end > brace_start:
                                                        json_str = content[brace_start:brace_end]
                                                        if '"tool_name"' in json_str and '"parameters"' in json_str:
                                                            try:
                                                                import json
                                                                tool_data = json.loads(json_str)
                                                                if isinstance(tool_data, dict) and 'tool_name' in tool_data and 'parameters' in tool_data:
                                                                    json_block_detected = True
                                                                    # 在break之前，先打印buffer中工具调用之前的内容
                                                                    # 找到JSON对象开始的位置，只打印到那里之前的内容
                                                                    text_before_json = content[:brace_start].rstrip()
                                                                    # 计算需要打印的内容：从total_printed到brace_start之前
                                                                    remaining_to_print = text_before_json[total_printed:]
                                                                    if remaining_to_print:
                                                                        printer.write(remaining_to_print)
                                                                        total_printed = len(text_before_json)
                                                                    # 清空buffer，因为已经打印了所有应该打印的内容
                                                                    buffer = ""
                                                                    break
                                                            except:
                                                                pass
                                            except:
                                                pass
                                except Exception as e:
                                    # 捕获流式处理中的异常
                                    stream_error_occurred = True
                                    stream_error_message = f"Streaming error: {type(e).__name__}: {str(e)}"
                                    print_debug(f"⚠️ {stream_error_message}")
                                    print_current(f"⚠️ Claude API streaming error: {str(e)}")
                                    # 继续处理已接收的内容
                                finally:
                                    # 确保流被正确关闭
                                    try:
                                        if hasattr(stream, 'close'):
                                            stream.close()
                                    except Exception as close_error:
                                        print_debug(f"⚠️ Error closing Anthropic stream: {close_error}")
                                
                                # 如果发生流错误，记录并继续处理
                                if stream_error_occurred:
                                    print_current(f"⚠️ 流式响应中断，已处理内容长度: {len(content)} 字符")
                                    if not content:
                                        # 如果没有接收到任何内容，重新抛出异常
                                        raise Exception(f"Anthropic API streaming failed: {stream_error_message}")
                                
                                # 处理剩余缓冲区和截断逻辑
                                if json_block_detected:
                                    # 检查是否有```json标记
                                    has_json_block = '```json' in content
                                    if has_json_block:
                                        # 找到第二个```json的位置
                                        content_to_print = self._get_content_before_second_json(content)
                                        
                                        # 打印缓冲区中还没打印的部分（但不超过第二个```json之前）
                                        remaining_buffer = content_to_print[total_printed:]
                                        if remaining_buffer:
                                            printer.write(remaining_buffer)
                                            total_printed = len(content_to_print)
                                    else:
                                        # 纯JSON格式：找到第一个JSON对象开始的位置
                                        try:
                                            brace_start = content.find('{')
                                            if brace_start != -1:
                                                text_before_json = content[:brace_start].rstrip()
                                                remaining_buffer = text_before_json[total_printed:]
                                                if remaining_buffer:
                                                    printer.write(remaining_buffer)
                                                    total_printed = len(text_before_json)
                                        except:
                                            pass
                                    
                                    # 不打印buffer中工具调用之后的内容
                                    buffer = ""
                                else:
                                    # 没有检测到工具调用，打印剩余缓冲区
                                    if buffer:
                                        printer.write(buffer)
                                
                                # If hallucination was detected, return early
                                if hallucination_detected:
                                    # 添加换行（仅限chat接口）
                                    if not content.endswith('\n'):
                                        content += '\n'
                                    return content, []
                                
                                # 检查是否有工具调用（即使只有一个）
                                # 查找第一个```json块或纯JSON格式的工具调用
                                has_json_block = '```json' in content
                                # 也检查纯JSON格式（不带```json标记）
                                has_plain_json_tool_call = ('"tool_name"' in content and '"parameters"' in content) and not has_json_block
                                
                                if json_block_detected:
                                    # 检测到第二个工具调用，只解析第一个
                                    # 确保用于解析的content包含完整的第一个工具调用
                                    # 即使被截断了，也要确保第一个JSON块是完整的
                                    if has_json_block:
                                        content_for_parsing = self._ensure_first_json_block_complete(content)
                                    else:
                                        # 纯JSON格式：找到第一个完整的JSON对象并截断
                                        try:
                                            brace_start = content.find('{')
                                            if brace_start != -1:
                                                brace_count = 0
                                                in_string = False
                                                escape_next = False
                                                brace_end = -1
                                                
                                                for i in range(brace_start, len(content)):
                                                    char = content[i]
                                                    if escape_next:
                                                        escape_next = False
                                                        continue
                                                    if char == '\\':
                                                        escape_next = True
                                                        continue
                                                    if char == '"' and not escape_next:
                                                        in_string = not in_string
                                                        continue
                                                    if not in_string:
                                                        if char == '{':
                                                            brace_count += 1
                                                        elif char == '}':
                                                            brace_count -= 1
                                                            if brace_count == 0:
                                                                brace_end = i + 1
                                                                break
                                                
                                                if brace_end > brace_start:
                                                    content_for_parsing = content[:brace_end]
                                                else:
                                                    content_for_parsing = content
                                            else:
                                                content_for_parsing = content
                                        except:
                                            content_for_parsing = content
                                    
                                    # Parse tool calls from the accumulated content
                                    tool_calls = self.parse_tool_calls(content_for_parsing)
                                    
                                    # 🎯 关键修改：只保留第一个工具调用，符合"每轮只能调用一个工具"的规则
                                    if tool_calls and len(tool_calls) > 1:
                                        tool_calls = [tool_calls[0]]
                                    
                                    # Convert tool calls to standard format for compatibility
                                    standardized_tool_calls = []
                                    for tool_call in tool_calls:
                                        if isinstance(tool_call, dict) and "name" in tool_call and "arguments" in tool_call:
                                            standardized_tool_calls.append({
                                                "name": tool_call["name"],
                                                "input": tool_call["arguments"]  # Use "input" format like Anthropic
                                            })
                                    
                                    # 调试：检查解析结果
                                    if not standardized_tool_calls:
                                        print_current(f"⚠️ Warning: After detecting multiple tool calls, failed to parse any valid tool call. Content length: {len(content_for_parsing)}")
                                        print_current(f"Content for parsing: {content_for_parsing[:500]}...")
                                    
                                    # 添加换行（仅限chat接口）
                                    if not content_for_parsing.endswith('\n'):
                                        content_for_parsing += '\n'
                                    return content_for_parsing, standardized_tool_calls
                                elif has_json_block or has_plain_json_tool_call:
                                    # 只有一个工具调用，正常解析并返回
                                    # 确保JSON块完整（如果使用```json标记）
                                    if has_json_block:
                                        content_for_parsing = self._ensure_first_json_block_complete(content)
                                    else:
                                        # 纯JSON格式，直接使用content
                                        content_for_parsing = content
                                    
                                    # Parse tool calls from the accumulated content
                                    tool_calls = self.parse_tool_calls(content_for_parsing)
                                    
                                    # 只保留第一个工具调用
                                    if tool_calls and len(tool_calls) > 1:
                                        tool_calls = [tool_calls[0]]
                                    
                                    # Convert tool calls to standard format for compatibility
                                    standardized_tool_calls = []
                                    for tool_call in tool_calls:
                                        if isinstance(tool_call, dict) and "name" in tool_call and "arguments" in tool_call:
                                            standardized_tool_calls.append({
                                                "name": tool_call["name"],
                                                "input": tool_call["arguments"]  # Use "input" format like Anthropic
                                            })
                                        else:
                                            print_current(f"⚠️ Warning: Tool call format invalid: {tool_call}")
                                    
                                    # 调试：检查转换结果
                                    if not standardized_tool_calls:
                                        print_current(f"⚠️ Warning: Failed to convert tool calls to standard format. Parsed tool_calls: {tool_calls}")
                                        print_current(f"Content for parsing length: {len(content_for_parsing)}")
                                        if self.debug_mode:
                                            print_current(f"Content snippet: {content_for_parsing[:500]}...")
                                    
                                    # 添加换行（仅限chat接口）
                                    if not content_for_parsing.endswith('\n'):
                                        content_for_parsing += '\n'
                                    return content_for_parsing, standardized_tool_calls
                                
                                # 没有工具调用，返回空列表
                                # 添加换行（仅限chat接口）
                                if not content.endswith('\n'):
                                    content += '\n'
                                return content, []
                    else:
                        # print_current("🔄 LLM is thinking:")
                        response = self.client.messages.create(
                            model=self.model,
                            max_tokens=self._get_max_tokens_for_model(self.model),
                            system=system_message,
                            messages=claude_messages,
                            temperature=0.7
                        )
                        
                        content = ""
                        for content_block in response.content:
                            if content_block.type == "text":
                                content += content_block.text

                        # Check for hallucination patterns in non-streaming response - strict match
                        if "LLM Called Following Tools in this round" in content or "**Tool Execution Results:**" in content:
                            # print_current("\n🚨 Hallucination Detected, stop chat")  # Reduced verbose output
                            # 添加换行（仅限chat接口）
                            if not content.endswith('\n'):
                                content += '\n'
                            return content, []
                        
                else:
                    # Use OpenAI API for chat-based tool calling
                    api_messages = [
                        {"role": "system", "content": system_message},
                        {"role": "user", "content": user_message}
                    ]
                    
                    if self.streaming:
                        with streaming_context(show_start_message=False) as printer:
                            # 显示LLM开始说话的emoji
                            printer.write(f"\n💬 ")
                            
                            response = self.client.chat.completions.create(
                                model=self.model,
                                messages=api_messages,
                                max_tokens=self._get_max_tokens_for_model(self.model),
                                temperature=0.7,
                                top_p=0.8,
                                stream=True
                            )

                            content = ""
                            hallucination_detected = False
                            json_block_detected = False
                            stream_error_occurred = False
                            stream_error_message = ""
                            
                            # 缓冲打印机制：至少缓冲100个字符
                            buffer = ""
                            min_buffer_size = 100
                            total_printed = 0
                            
                            try:
                                for chunk in response:
                                    if chunk.choices and len(chunk.choices) > 0:
                                        delta = chunk.choices[0].delta
                                        if delta.content is not None:
                                            buffer += delta.content
                                            content += delta.content
                                            
                                            # 在缓冲区逻辑中统一检测幻觉模式和工具调用
                                            # 检测幻觉模式（优先于工具调用检测）
                                            # 严格匹配两个幻觉标志：
                                            # 1. "LLM Called Following Tools in this round" (8个单词)
                                            # 2. "**Tool Execution Results:**"
                                            hallucination_patterns = [
                                                "LLM Called Following Tools in this round",
                                                "**Tool Execution Results:**"
                                            ]
                                            hallucination_detected_flag = False
                                            hallucination_start = -1
                                            for pattern in hallucination_patterns:
                                                if pattern in content:
                                                    hallucination_start = content.find(pattern)
                                                    hallucination_detected_flag = True
                                                    break
                                            
                                            if hallucination_detected_flag:
                                                print_debug("\n🚨 Hallucination Detected, stop chat")
                                                hallucination_detected = True
                                                # 截断内容到幻觉开始位置，避免打印幻觉字符串
                                                if hallucination_start > 0:
                                                    content_to_print = content[:hallucination_start].rstrip()
                                                    # 打印幻觉之前的内容（如果还有未打印的）
                                                    remaining_to_print = content_to_print[total_printed:]
                                                    if remaining_to_print:
                                                        printer.write(remaining_to_print)
                                                    content = content_to_print
                                                buffer = ""
                                                break
                                            
                                            # 检测工具调用：检查是否已经包含了第二个```json
                                            if self._is_complete_json_tool_call(content):
                                                # 找到第二个```json的位置
                                                content_to_print = self._get_content_before_second_json(content)
                                                # 打印到第二个```json之前的内容（如果还有未打印的）
                                                if len(content_to_print) > total_printed:
                                                    remaining_to_print = content_to_print[total_printed:]
                                                    if remaining_to_print:
                                                        printer.write(remaining_to_print)
                                                    total_printed = len(content_to_print)
                                                # 更新content为截断后的内容
                                                content = content_to_print
                                                buffer = ""
                                                json_block_detected = True
                                                break
                                            
                                            # 当缓冲区达到最小大小时，打印缓冲区内容
                                            if len(buffer) >= min_buffer_size:
                                                printer.write(buffer)
                                                total_printed += len(buffer)
                                                buffer = ""
                            except Exception as e:
                                # 捕获流式处理中的异常
                                stream_error_occurred = True
                                stream_error_message = f"Streaming error: {type(e).__name__}: {str(e)}"
                                print_debug(f"⚠️ {stream_error_message}")
                                print_current(f"⚠️ OpenAI API streaming error: {str(e)}")
                                # 继续处理已接收的内容
                            finally:
                                # 显式关闭streaming连接（无论是否发生错误都要关闭）
                                try:
                                    if hasattr(response, 'close'):
                                        response.close()
                                    elif hasattr(response, '__aexit__'):
                                        # 异步上下文管理器
                                        pass
                                except Exception as close_error:
                                    print_debug(f"⚠️ Error closing OpenAI stream: {close_error}")
                            
                            # 如果发生流错误，记录并继续处理
                            if stream_error_occurred:
                                print_current(f"⚠️ 流式响应中断，已处理内容长度: {len(content)} 字符")
                                if not content:
                                    # 如果没有接收到任何内容，重新抛出异常
                                    raise Exception(f"OpenAI API streaming failed: {stream_error_message}")
                            
                            # 处理剩余缓冲区（如果循环内没有检测到特殊模式）
                            if not hallucination_detected and not json_block_detected:
                                # 没有检测到幻觉或工具调用，打印剩余缓冲区
                                if buffer:
                                    printer.write(buffer)
                            
                            # If hallucination was detected, return early
                            if hallucination_detected:
                                # 添加换行（仅限chat接口）
                                if not content.endswith('\n'):
                                    content += '\n'
                                return content, []
                            
                            # 检查是否有工具调用（即使只有一个）
                            # 查找第一个```json块
                            has_json_block = '```json' in content
                            
                            if json_block_detected:
                                # 检测到第二个工具调用，只解析第一个
                                # content已经在循环内部被截断到第二个JSON块之前
                                # 确保用于解析的content包含完整的第一个工具调用
                                content_for_parsing = self._ensure_first_json_block_complete(content)
                                
                                # Parse tool calls from the accumulated content
                                tool_calls = self.parse_tool_calls(content_for_parsing)
                                
                                # 🎯 关键修改：只保留第一个工具调用，符合"每轮只能调用一个工具"的规则
                                if tool_calls and len(tool_calls) > 1:
                                    tool_calls = [tool_calls[0]]
                                
                                # Convert tool calls to standard format for compatibility
                                standardized_tool_calls = []
                                for tool_call in tool_calls:
                                    if isinstance(tool_call, dict) and "name" in tool_call and "arguments" in tool_call:
                                        standardized_tool_calls.append({
                                            "name": tool_call["name"],
                                            "input": tool_call["arguments"]  # Use "input" format like Anthropic
                                        })
                                    else:
                                        print_current(f"⚠️ Warning: Tool call format invalid: {tool_call}")
                                
                                # 调试：检查转换结果
                                if not standardized_tool_calls and tool_calls:
                                    print_current(f"⚠️ Warning: Failed to convert tool calls to standard format. Parsed tool_calls: {tool_calls}")
                                    print_current(f"Content for parsing length: {len(content_for_parsing)}")
                                    #print_current(f"Content snippet: {content_for_parsing[:500]}...")
                                
                                # 添加换行（仅限chat接口）
                                if not content_for_parsing.endswith('\n'):
                                    content_for_parsing += '\n'
                                return content_for_parsing, standardized_tool_calls
                            elif has_json_block:
                                # 只有一个工具调用，正常解析并返回
                                # 确保JSON块完整
                                content_for_parsing = self._ensure_first_json_block_complete(content)
                                
                                # Parse tool calls from the accumulated content
                                tool_calls = self.parse_tool_calls(content_for_parsing)
                                
                                # 只保留第一个工具调用
                                if tool_calls and len(tool_calls) > 1:
                                    tool_calls = [tool_calls[0]]
                                
                                # Convert tool calls to standard format for compatibility
                                standardized_tool_calls = []
                                for tool_call in tool_calls:
                                    if isinstance(tool_call, dict) and "name" in tool_call and "arguments" in tool_call:
                                        standardized_tool_calls.append({
                                            "name": tool_call["name"],
                                            "input": tool_call["arguments"]  # Use "input" format like Anthropic
                                        })
                                    else:
                                        print_current(f"⚠️ Warning: Tool call format invalid: {tool_call}")
                                
                                # 调试：检查转换结果
                                if not standardized_tool_calls:
                                    print_current(f"⚠️ Warning: Failed to convert tool calls to standard format. Parsed tool_calls: {tool_calls}")
                                    print_current(f"Content for parsing length: {len(content_for_parsing)}")
                                    #print_current(f"Content snippet: {content_for_parsing[:500]}...")
                                
                                # 添加换行（仅限chat接口）
                                if not content_for_parsing.endswith('\n'):
                                    content_for_parsing += '\n'
                                return content_for_parsing, standardized_tool_calls
                            
                            # 没有工具调用，返回空列表
                            # 添加换行（仅限chat接口）
                            if not content.endswith('\n'):
                                content += '\n'
                            return content, []
                        
                    else:
                        # print_current("🔄 LLM is thinking:")
                        response = self.client.chat.completions.create(
                            model=self.model,
                            messages=api_messages,
                            max_tokens=self._get_max_tokens_for_model(self.model),
                            temperature=0.7,
                            top_p=0.8,
                            stream=False
                    )

                    # Check if response is a Stream object (should not happen with stream=False)
                    if hasattr(response, '__iter__') and not hasattr(response, 'choices'):
                        # If we got a Stream object, consume it to get the actual response
                        print_current("⚠️ Warning: Received Stream object despite stream=False. Converting to regular response...")
                        content = ""
                        json_block_detected = False
                        stream_error_occurred = False
                        stream_error_message = ""
                        
                        try:
                            for chunk in response:
                                if hasattr(chunk, 'choices') and chunk.choices and len(chunk.choices) > 0:
                                    delta = chunk.choices[0].delta
                                    if hasattr(delta, 'content') and delta.content is not None:
                                        content += delta.content
                                        
                                        # 检测完整的JSON工具调用块
                                        if self._is_complete_json_tool_call(content):
                                            json_block_detected = True
                                            break
                        except Exception as e:
                            # 捕获流式处理中的异常
                            stream_error_occurred = True
                            stream_error_message = f"Streaming error: {type(e).__name__}: {str(e)}"
                            print_debug(f"⚠️ {stream_error_message}")
                            print_current(f"⚠️ OpenAI API streaming error (unexpected stream mode): {str(e)}")
                            # 继续处理已接收的内容
                        finally:
                            # 显式关闭streaming连接（无论是否发生错误都要关闭）
                            try:
                                if hasattr(response, 'close'):
                                    response.close()
                                    print_debug("🔌 已显式关闭streaming连接")
                            except Exception as close_error:
                                print_debug(f"⚠️ 关闭streaming连接时出错: {close_error}")
                        
                        # 如果发生流错误，记录并继续处理
                        if stream_error_occurred:
                            print_current(f"⚠️ 流式响应中断，已处理内容长度: {len(content)} 字符")
                            if not content:
                                # 如果没有接收到任何内容，重新抛出异常
                                raise Exception(f"OpenAI API streaming failed (unexpected stream mode): {stream_error_message}")
                        
                        # Return early with parsed content
                        tool_calls = self.parse_tool_calls(content)
                        standardized_tool_calls = []
                        for tool_call in tool_calls:
                            if isinstance(tool_call, dict) and "name" in tool_call and "arguments" in tool_call:
                                standardized_tool_calls.append({
                                    "name": tool_call["name"],
                                    "input": tool_call["arguments"]
                                })
                        # 添加换行（仅限chat接口）
                        if not content.endswith('\n'):
                            content += '\n'
                        return content, standardized_tool_calls

                    # Extract content and thinking field from OpenAI response
                    message = response.choices[0].message
                    content = message.content or ""

                    # Handle thinking field for OpenAI o1 models and other reasoning models
                    thinking = getattr(message, 'thinking', None)
                    if thinking:
                        # Combine thinking and content with clear separation
                        content = f"## Thinking Process\n\n{thinking}\n\n## Final Answer\n\n{content}"

                    # Check for hallucination patterns in non-streaming response - strict match
                    if "LLM Called Following Tools in this round" in content or "**Tool Execution Results:**" in content:
                        # print_current("\n🚨 Hallucination Detected, stop chat")  # Reduced verbose output
                        # 添加换行（仅限chat接口）
                        if not content.endswith('\n'):
                            content += '\n'
                        return content, []

                # Parse tool calls from the response content
                tool_calls = self.parse_tool_calls(content)
                
                # 🎯 关键修改：只保留第一个工具调用，符合"每轮只能调用一个工具"的规则
                if tool_calls and len(tool_calls) > 1:
                    tool_calls = [tool_calls[0]]
                
                # Convert tool calls to standard format for compatibility
                standardized_tool_calls = []
                for tool_call in tool_calls:
                    if isinstance(tool_call, dict) and "name" in tool_call and "arguments" in tool_call:
                        standardized_tool_calls.append({
                            "name": tool_call["name"],
                            "input": tool_call["arguments"]  # Use "input" format like Anthropic
                        })
                
                # 添加换行（仅限chat接口）
                if not content.endswith('\n'):
                    content += '\n'
                return content, standardized_tool_calls
                
            except Exception as e:
                error_str = str(e).lower()
                
                # Check if this is a retryable error
                retryable_errors = [
                    'overloaded', 'rate limit', 'too many requests',
                    'service unavailable', 'timeout', 'temporary failure',
                    'server error', '429', '503', '502', '500',
                    'peer closed connection', 'incomplete chunked read'
                ]
                
                # Find which error keyword matched
                matched_error_keyword = None
                for error_keyword in retryable_errors:
                    if error_keyword in error_str:
                        matched_error_keyword = error_keyword
                        break
                
                is_retryable = matched_error_keyword is not None
                
                if is_retryable and attempt < max_retries:
                    # Calculate retry delay with exponential backoff
                    retry_delay = min(2 ** attempt, 10)  # 1, 2, 4 seconds, max 10
                    
                    api_type = "Claude API" if self.is_claude else "OpenAI API"
                    print_current(f"⚠️ {api_type} {matched_error_keyword} error (attempt {attempt + 1}/{max_retries + 1}): {e}")
                    print_current(f"💡 Consider switching to a different model or trying again later")
                    print_current(f"🔄 You can change the model in config.txt and restart AGIAgent")
                    print_current(f"🔄 Retrying in {retry_delay} seconds...")
                    
                    # Wait before retry
                    time.sleep(retry_delay)
                    continue  # Retry the loop
                    
                else:
                    # Non-retryable error or max retries exceeded
                    api_type = "Claude API" if self.is_claude else "OpenAI API"
                    if is_retryable:
                        print_current(f"❌ {api_type} {matched_error_keyword} error: Maximum retries ({max_retries}) exceeded")
                        print_current(f"💡 Consider switching to a different model or trying again later")
                        print_current(f"🔄 You can change the model in config.txt and restart AGIAgent")
                    else:
                        print_current(f"❌ Chat-based LLM API call failed: {e}")
                    
                    raise e

    def _call_glm_with_standard_tools(self, messages, user_message, system_message):
        """
        Call GLM with standard tool calling format.
        """
        # Get standard tools for Anthropic
        tools = self._convert_tools_to_standard_format("anthropic")
        
        # Check if we have stored image data for vision API
        if hasattr(self, 'current_round_images') and self.current_round_images:
            print_current(f"🖼️ Using vision API with {len(self.current_round_images)} stored images")
            # Build vision message with stored images
            vision_user_message = self._build_vision_message(user_message if isinstance(user_message, str) else user_message.get("text", ""))
            claude_messages = [{"role": "user", "content": vision_user_message}]
            # Clear image data after using it for vision API to prevent reuse in subsequent rounds
            print_current("🧹 Clearing image data after vision API usage")
            self.current_round_images = []
        else:
            # Prepare messages for Claude - user_message can be string or content array
            claude_messages = [{"role": "user", "content": user_message}]
        

        
        # Retry logic for retryable errors
        max_retries = 3
        for attempt in range(max_retries + 1):  # 0, 1, 2, 3 (4 total attempts)
            try:
                if self.streaming:
                    # Simplified streaming logic - only handle message streaming, read tool calls from final messages in other parts
                    content = ""
                    tool_calls = []

                    with streaming_context(show_start_message=True) as printer:
                        # 显示LLM开始说话的emoji
                        printer.write(f"\n💬 ")

                        hallucination_detected = False
                        stream_error_occurred = False
                        last_event_type = None
                        error_details = None

                        with self.client.messages.stream(
                            model=self.model,
                            max_tokens=self._get_max_tokens_for_model(self.model),
                            system=system_message,
                            messages=claude_messages,
                            tools=tools,
                            temperature=0.7
                        ) as stream:
                            try:
                                for event in stream:
                                    try:
                                        event_type = getattr(event, 'type', None)
                                        last_event_type = event_type

                                        # Only handle text content streaming events
                                        if event_type == "content_block_delta":
                                            try:
                                                delta = getattr(event, 'delta', None)

                                                if delta:
                                                    delta_type = getattr(delta, 'type', None)

                                                    if delta_type == "text_delta":
                                                        # 文本内容流式输出
                                                        text = getattr(delta, 'text', '')
                                                        # 检测幻觉模式 - 严格匹配两个标志
                                                        if "LLM Called Following Tools in this round" in text or "**Tool Execution Results:**" in text:
                                                            print_current("\n🚨 Hallucination detected, stopping conversation")
                                                            hallucination_detected = True
                                                            break
                                                        printer.write(text)
                                                        content += text
                                            except Exception as e:
                                                print_debug(f"⚠️ Error processing content_block_delta: {type(e).__name__}: {str(e)}")
                                                # 继续处理其他事件

                                        # 处理消息统计信息
                                        elif event_type == "message_delta":
                                            try:
                                                delta = getattr(event, 'delta', None)
                                                if delta:
                                                    usage = getattr(delta, 'usage', None) or getattr(event, 'usage', None)
                                                    if usage:
                                                        input_tokens = getattr(usage, 'input_tokens', 0) or 0
                                                        output_tokens = getattr(usage, 'output_tokens', 0) or 0
                                                        cache_creation_tokens = getattr(usage, 'cache_creation_input_tokens', 0) or 0
                                                        cache_read_tokens = getattr(usage, 'cache_read_input_tokens', 0) or 0

                                                        if cache_creation_tokens > 0 or cache_read_tokens > 0:
                                                            print_debug(f"\n📊 Token Usage - Input: {input_tokens}, Output: {output_tokens}, Cache Creation: {cache_creation_tokens}, Cache Read: {cache_read_tokens}")
                                            except Exception as e:
                                                print_debug(f"⚠️ Error processing message_delta: {type(e).__name__}: {str(e)}")

                                    except Exception as event_error:
                                        # Single event processing failure should not interrupt the entire stream
                                        print_debug(f"⚠️ Error processing event {last_event_type}: {type(event_error).__name__}: {str(event_error)}")
                                        # Do not use continue, let the loop continue naturally

                            except Exception as e:
                                # Check if it's a JSON parsing error, if so ignore and continue streaming inference
                                error_str = str(e)
                                if "expected value at line 1 column" in error_str and "ValueError" in str(type(e)):
                                    # JSON parsing error, ignore and continue processing other events
                                    print_debug(f"⚠️ JSON parsing error ignored for event_type={last_event_type}: {type(e).__name__}: {str(e)}")
                                    continue  # 继续处理下一个事件
                                else:
                                    # 其他类型的错误，使用增强的错误处理
                                    stream_error_occurred = True
                                    error_details = f"Streaming failed at event_type={last_event_type}: {type(e).__name__}: {str(e)}"
                                    print_debug(error_details)

                                    # 尝试回退到text_stream
                                    try:
                                        for text in stream.text_stream:
                                            if "LLM Called Following Tools in this round" in text or "**Tool Execution Results:**" in text:
                                                print_current("\n🚨 Hallucination detected, stopping conversation")
                                                hallucination_detected = True
                                                break
                                            printer.write(text)
                                            content += text
                                    except Exception as fallback_error:
                                        print_error(f"Text streaming also failed: {fallback_error}")
                                        break

                            # 如果检测到幻觉，提前返回
                            if hallucination_detected:
                                return content, []

                        print_current("")

                        # Read tool calls directly from final message
                        if not stream_error_occurred:
                            try:
                                final_message = stream.get_final_message()

                                for content_block in final_message.content:
                                    if content_block.type == "tool_use":
                                        # 验证工具调用input
                                        tool_input = content_block.input
                                        tool_name = content_block.name

                                        # input should already be dict, but check for safety
                                        if isinstance(tool_input, str):
                                            # Fix boolean format issues
                                            tool_input = _fix_json_boolean_values(tool_input)
                                            is_valid, parsed_input, error_msg = validate_tool_call_json(tool_input, tool_name)
                                            if not is_valid:
                                                # Only show failure info for non-empty string errors
                                                if error_msg != "Empty JSON string":
                                                    print_error(f"❌ Final message tool call validation failed: {error_msg}")
                                                else:
                                                    print_debug(f"⚠️ Empty tool input for {tool_name}, skipping")
                                                continue
                                            tool_input = parsed_input

                                        tool_calls.append({
                                            "id": content_block.id,
                                            "name": tool_name,
                                            "input": tool_input
                                        })

                            except Exception as e:
                                print_error(f"Failed to get final message: {type(e).__name__}: {str(e)}")

                    # Execute tool calls
                    if tool_calls:
                        for tool_call_data in tool_calls:
                            try:
                                tool_name = tool_call_data['name']

                                # Convert to standard format
                                standard_tool_call = {
                                    "name": tool_name,
                                    "arguments": tool_call_data['input']
                                }

                                tool_result = self.execute_tool(standard_tool_call, streaming_output=True)

                                # 存储结果
                                if not hasattr(self, '_streaming_tool_results'):
                                    self._streaming_tool_results = []

                                self._streaming_tool_results.append({
                                    'tool_name': tool_name,
                                    'tool_params': tool_call_data['input'],
                                    'tool_result': tool_result
                                })

                                self._tools_executed_in_stream = True

                            except Exception as e:
                                print_error(f"❌ Tool {tool_name} execution failed: {str(e)}")

                        print_debug("✅ All tool executions completed")

                    # If an error occurred during streaming, append error details to content for feedback to the LLM
                    if stream_error_occurred and error_details is not None:
                        error_feedback = f"\n\n⚠️ **Streaming Error Feedback**: There was a problem parsing the previous response: {error_details}\nPlease regenerate a correct response based on this error message."
                        content += error_feedback

                    return content, tool_calls
                else:
                    # print_current("🔄 LLM is thinking: ")
                    response = self.client.messages.create(
                        model=self.model,
                        max_tokens=self._get_max_tokens_for_model(self.model),
                        system=system_message,
                        messages=claude_messages,
                        tools=tools,
                        temperature=0.7
                    )
                    
                    content = ""
                    tool_calls = []
                    
                    # Extract content and tool use blocks
                    for content_block in response.content:
                        if content_block.type == "text":
                            content += content_block.text
                        elif content_block.type == "tool_use":
                            tool_calls.append({
                                "id": content_block.id,
                                "name": content_block.name,
                                "input": content_block.input
                            })
                    
                    return content, tool_calls
                    
            except Exception as e:
                error_str = str(e).lower()
                
                # Check if this is a retryable error
                retryable_errors = [
                    'overloaded', 'rate limit', 'too many requests',
                    'service unavailable', 'timeout', 'temporary failure',
                    'server error', '429', '503', '502', '500',
                    'peer closed connection', 'incomplete chunked read'
                ]
                
                # Find which error keyword matched
                matched_error_keyword = None
                for error_keyword in retryable_errors:
                    if error_keyword in error_str:
                        matched_error_keyword = error_keyword
                        break
                
                is_retryable = matched_error_keyword is not None
                
                if is_retryable and attempt < max_retries:
                    # Calculate retry delay with exponential backoff
                    retry_delay = 1
                    
                    print_current(f"⚠️ GLM API {matched_error_keyword} error (attempt {attempt + 1}/{max_retries + 1}): {e}")
                    print_current(f"💡 Consider switching to a different model or trying again later")
                    print_current(f"🔄 You can change the model in config.txt and restart AGIAgent")
                    print_current(f"🔄 Retrying in {retry_delay} seconds...")
                    
                    # Wait before retry
                    time.sleep(retry_delay)
                    continue  # Retry the loop
                    
                else:
                    # Non-retryable error or max retries exceeded
                    if is_retryable:
                        print_current(f"❌ GLM API {matched_error_keyword} error: Maximum retries ({max_retries}) exceeded")
                        print_current(f"💡 Consider switching to a different model or trying again later")
                        print_current(f"🔄 You can change the model in config.txt and restart AGIAgent")
                    else:
                        print_current(f"❌ GLM API call failed: {e}")
                    
                    raise e

    def _call_openai_with_standard_tools(self, messages, user_message, system_message):
        """
        Call OpenAI with standard tool calling format.
        """
        # Get standard tools for OpenAI
        tools = self._convert_tools_to_standard_format("openai")
        
        # Check if we have stored image data for vision API
        if hasattr(self, 'current_round_images') and self.current_round_images:
            print_current(f"🖼️ Using vision API with {len(self.current_round_images)} stored images")
            # Build vision message with stored images
            vision_user_message = self._build_vision_message(user_message if isinstance(user_message, str) else user_message.get("text", ""))
            api_messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": vision_user_message}
            ]
            # Clear image data after using it for vision API to prevent reuse in subsequent rounds
            print_current("🧹 Clearing image data after vision API usage")
            self.current_round_images = []
        else:
            # Prepare messages - user_message can be string or content array
            api_messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ]
        
        # Retry logic for retryable errors
        max_retries = 3
        for attempt in range(max_retries + 1):  # 0, 1, 2, 3 (4 total attempts)
            try:
                if self.streaming:
                    # 流式处理逻辑 - 在接收到完整工具调用块后立即停止
                    content = ""
                    tool_calls = []
                    tool_calls_buffer = {}  # 用于收集增量式的工具调用信息
                    
                    with streaming_context(show_start_message=False) as printer:
                        # 显示LLM开始说话的emoji
                        printer.write(f"\n💬 ")
                        hallucination_detected = False

                        response = self.client.chat.completions.create(
                            model=self.model,
                            messages=api_messages,
                            tools=tools,
                            max_tokens=self._get_max_tokens_for_model(self.model),
                            temperature=0.7,
                            top_p=0.8,
                            stream=True
                        )

                        try:
                            tool_calls_completed = False
                            empty_chunks_after_tool_calls = 0
                            max_empty_chunks = 3  # 允许在工具调用完成后最多接收3个空chunk来捕获后续文本
                            
                            for chunk in response:
                                if chunk.choices and len(chunk.choices) > 0:
                                    delta = chunk.choices[0].delta
                                    finish_reason = chunk.choices[0].finish_reason
                                    
                                    # 处理文本内容的流式输出
                                    if delta.content is not None:
                                        printer.write(delta.content)
                                        content += delta.content
                                        # 如果工具调用已完成但仍有文本内容，重置空chunk计数
                                        if tool_calls_completed:
                                            empty_chunks_after_tool_calls = 0
                                    
                                    # 处理工具调用的增量更新
                                    if delta.tool_calls:
                                        for tool_call_delta in delta.tool_calls:
                                            idx = tool_call_delta.index
                                            if idx not in tool_calls_buffer:
                                                tool_calls_buffer[idx] = {
                                                    "id": "",
                                                    "type": "function",
                                                    "function": {
                                                        "name": "",
                                                        "arguments": ""
                                                    }
                                                }
                                            
                                            # 累积工具调用信息
                                            if tool_call_delta.id:
                                                tool_calls_buffer[idx]["id"] = tool_call_delta.id
                                            if tool_call_delta.function:
                                                if tool_call_delta.function.name:
                                                    tool_calls_buffer[idx]["function"]["name"] = tool_call_delta.function.name
                                                if tool_call_delta.function.arguments:
                                                    tool_calls_buffer[idx]["function"]["arguments"] += tool_call_delta.function.arguments
                                    
                                    # 检查finish_reason
                                    if finish_reason is not None:
                                        if finish_reason == "tool_calls":
                                            # 工具调用完成，但可能还有后续文本，继续处理
                                            tool_calls_completed = True
                                            print_debug("🔧 工具调用完成，继续接收可能的后续文本...")
                                        else:
                                            # 其他结束原因（如"stop"），正常结束
                                            print_debug(f"✅ 流式响应结束: {finish_reason}")
                                            break
                                    else:
                                        # 如果没有finish_reason，检查是否在工具调用完成后收到空chunk
                                        if tool_calls_completed:
                                            # 检查当前chunk是否为空（没有内容和工具调用）
                                            has_content = delta.content is not None and len(delta.content.strip()) > 0
                                            has_tool_calls = delta.tool_calls is not None and len(delta.tool_calls) > 0
                                            
                                            if not has_content and not has_tool_calls:
                                                empty_chunks_after_tool_calls += 1
                                                # 如果连续收到多个空chunk，可能流已结束
                                                if empty_chunks_after_tool_calls >= max_empty_chunks:
                                                    print_debug(f"🔚 工具调用完成后收到{max_empty_chunks}个空chunk，结束接收")
                                                    break
                                            else:
                                                # 有内容，重置计数
                                                empty_chunks_after_tool_calls = 0
                        finally:
                            # 显式关闭streaming连接，通知服务器停止生成
                            # 这确保了服务器端能够感知到客户端已停止接收
                            if hasattr(response, 'close'):
                                try:
                                    response.close()
                                    print_debug("🔌 已显式关闭streaming连接")
                                except Exception as e:
                                    print_debug(f"⚠️ 关闭streaming连接时出错: {e}")
                        
                        print_current("")
                    
                    # 将缓冲区中的工具调用转换为列表
                    if tool_calls_buffer:
                        for idx in sorted(tool_calls_buffer.keys()):
                            tool_calls.append(tool_calls_buffer[idx])
                    
                    # Execute tool calls
                    if tool_calls:
                        for i, tool_call in enumerate(tool_calls):
                            try:
                                tool_name = tool_call["function"]["name"]
                                tool_params_str = tool_call["function"]["arguments"]
                                
                                # 使用增强的JSON验证和解析
                                # 修复布尔值格式问题
                                # tool_params_str = _fix_json_boolean_values(tool_params_str)
                                is_valid, tool_params, error_msg = validate_tool_call_json(tool_params_str, tool_name)
                                
                                if not is_valid:
                                    print_error(f"❌ Tool {i + 1} ({tool_name}) JSON解析失败:")
                                    print_error(f"   {error_msg}")
                                    print_debug(f"   Raw arguments: {tool_params_str[:200]}...")
                                    continue
                                
                                #print_current(f"🎯 Tool {i + 1}: {tool_name}")
                                
                                # Convert to standard format
                                standard_tool_call = {
                                    "name": tool_name,
                                    "arguments": tool_params
                                }
                                
                                tool_result = self.execute_tool(standard_tool_call, streaming_output=True)
                                
                                # 存储结果
                                if not hasattr(self, '_streaming_tool_results'):
                                    self._streaming_tool_results = []
                                
                                self._streaming_tool_results.append({
                                    'tool_name': tool_name,
                                    'tool_params': tool_params,
                                    'tool_result': tool_result
                                })
                                
                                self._tools_executed_in_stream = True
                                
                            except Exception as e:
                                print_error(f"❌ Tool {i + 1} execution failed: {type(e).__name__}: {str(e)}")
                                print_debug(f"   Tool: {tool_call.get('function', {}).get('name', 'unknown')}")
                        
                        print_debug("✅ All tool executions completed")

                    # If hallucination was detected, return early with empty tool calls
                    if hallucination_detected:
                        return content, []

                    # print_current("\n✅ Streaming completed")
                    return content, tool_calls
                else:
                    # print_current("🔄 Starting batch generation with standard tools...")
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=api_messages,
                        tools=tools,
                        max_tokens=self._get_max_tokens_for_model(self.model),
                        temperature=0.7,
                        top_p=0.8,
                        stream=False
                    )

                    # Check if response is a Stream object (should not happen with stream=False)
                    if hasattr(response, '__iter__') and not hasattr(response, 'choices'):
                        # If we got a Stream object, consume it to get the actual response
                        print_current("⚠️ Warning: Received Stream object despite stream=False. Converting to regular response...")
                        content = ""
                        tool_calls = []
                        for chunk in response:
                            if hasattr(chunk, 'choices') and chunk.choices and len(chunk.choices) > 0:
                                delta = chunk.choices[0].delta
                                if hasattr(delta, 'content') and delta.content is not None:
                                    content += delta.content
                                # Collect tool calls from chunks if present
                                if hasattr(delta, 'tool_calls') and delta.tool_calls:
                                    for tc in delta.tool_calls:
                                        tool_calls.append(tc)
                        return content, tool_calls

                    # Extract content and thinking field from OpenAI response
                    message = response.choices[0].message
                    content = message.content or ""

                    # Handle thinking field for OpenAI o1 models and other reasoning models
                    thinking = getattr(message, 'thinking', None)
                    if thinking:
                        # Combine thinking and content with clear separation
                        content = f"## Thinking Process\n\n{thinking}\n\n## Final Answer\n\n{content}"

                    # Check for hallucination patterns in non-streaming response - strict match
                    if "LLM Called Following Tools in this round" in content or "**Tool Execution Results:**" in content:
                        print_debug("\n🚨 Hallucination Detected, stop chat")
                        # 截断内容到幻觉位置之前，避免打印幻觉字符串
                        hallucination_patterns = [
                            "LLM Called Following Tools in this round",
                            "**Tool Execution Results:**"
                        ]
                        hallucination_start = len(content)
                        for pattern in hallucination_patterns:
                            if pattern in content:
                                hallucination_start = min(hallucination_start, content.find(pattern))
                        if hallucination_start > 0:
                            content = content[:hallucination_start].rstrip()
                        else:
                            content = ""
                        return content, []

                    raw_tool_calls = response.choices[0].message.tool_calls or []
                    
                    # Convert OpenAI tool_calls objects to dictionary format
                    tool_calls = []
                    for tool_call in raw_tool_calls:
                        tool_calls.append({
                            "id": tool_call.id,
                            "type": tool_call.type,
                            "function": {
                                "name": tool_call.function.name,
                                "arguments": tool_call.function.arguments
                            }
                        })
                    
                    # print_current("✅ Generation completed")
                    return content, tool_calls
                    
            except Exception as e:
                error_str = str(e).lower()
                
                # Check if this is a retryable error
                retryable_errors = [
                    'overloaded', 'rate limit', 'too many requests',
                    'service unavailable', 'timeout', 'temporary failure',
                    'server error', '429', '503', '502', '500',
                    'peer closed connection', 'incomplete chunked read'
                ]
                
                # Find which error keyword matched
                matched_error_keyword = None
                for error_keyword in retryable_errors:
                    if error_keyword in error_str:
                        matched_error_keyword = error_keyword
                        break
                
                is_retryable = matched_error_keyword is not None
                
                if is_retryable and attempt < max_retries:
                    # Calculate retry delay with exponential backoff
                    retry_delay = min(2 ** attempt, 10)  # 1, 2, 4 seconds, max 10
                    
                    print_current(f"⚠️ OpenAI API {matched_error_keyword} error (attempt {attempt + 1}/{max_retries + 1}): {e}")
                    print_current(f"💡 Consider switching to a different model or trying again later")
                    print_current(f"🔄 You can change the model in config.txt and restart AGIAgent")
                    print_current(f"🔄 Retrying in {retry_delay} seconds...")
                    
                    # Wait before retry
                    time.sleep(retry_delay)
                    continue  # Retry the loop
                    
                else:
                    # Non-retryable error or max retries exceeded
                    if is_retryable:
                        print_current(f"❌ OpenAI API {matched_error_keyword} error: Maximum retries ({max_retries}) exceeded")
                        print_current(f"💡 Consider switching to a different model or trying again later")
                        print_current(f"🔄 You can change the model in config.txt and restart AGIAgent")
                    else:
                        print_current(f"❌ OpenAI API call failed: {e}")
                    
                    raise e

    def _execute_tool_immediately(self, tool_call, tool_index):
        """
        Execute a tool call immediately during streaming.
        
        Args:
            tool_call: The complete tool call object
            tool_index: The tool index for display purposes
        """
        try:
            tool_name = tool_call["function"]["name"]
            tool_params_str = tool_call["function"]["arguments"]
            
            # Parse parameters
            import json
            tool_params = json.loads(tool_params_str)
            
            print_current(f"⚡ Executing tool {tool_index} immediately: {tool_name}")
            print_current(f"   Parameters: {tool_params}")
            
            # Convert to standard format for execute_tool
            standard_tool_call = {
                "name": tool_name,
                "arguments": tool_params
            }
            
            tool_result = self.execute_tool(standard_tool_call, streaming_output=True)
            
            # Store result for later response formatting
            if not hasattr(self, '_streaming_tool_results'):
                self._streaming_tool_results = []
            
            self._streaming_tool_results.append({
                'tool_name': tool_name,
                'tool_params': tool_params,
                'tool_result': tool_result
            })
            
            # Set flag indicating tools were executed during streaming
            self._tools_executed_in_stream = True
            
            # Tool result is already displayed by streaming output, no need to duplicate
            
        except Exception as e:
            print_current(f"   ❌ Tool {tool_index} execution failed: {str(e)}")

    def _call_claude_with_standard_tools(self, messages, user_message, system_message):
        """
        Call Claude with standard tool calling format.
        """
        # Get standard tools for Anthropic
        tools = self._convert_tools_to_standard_format("anthropic")
        
        # Check if we have stored image data for vision API
        if hasattr(self, 'current_round_images') and self.current_round_images:
            print_current(f"🖼️ Using vision API with {len(self.current_round_images)} stored images")
            # Build vision message with stored images
            vision_user_message = self._build_vision_message(user_message if isinstance(user_message, str) else user_message.get("text", ""))
            claude_messages = [{"role": "user", "content": vision_user_message}]
            # Clear image data after using it for vision API to prevent reuse in subsequent rounds
            print_current("🧹 Clearing image data after vision API usage")
            self.current_round_images = []
        else:
            # Prepare messages for Claude - user_message can be string or content array
            claude_messages = [{"role": "user", "content": user_message}]
        

        
        # Retry logic for retryable errors
        max_retries = 3
        for attempt in range(max_retries + 1):  # 0, 1, 2, 3 (4 total attempts)
            try:
                if self.streaming:
                    # 完善的流式处理逻辑 - 处理所有事件类型包括工具调用
                    content = ""
                    tool_calls = []
                    
                    # 工具调用缓冲区 - 用于逐步构建工具调用JSON
                    tool_call_buffers = {}  # {block_index: {"id": str, "name": str, "input_json": str}}
                    current_block_index = None
                    current_block_type = None

                    with streaming_context(show_start_message=True) as printer:
                        # 显示LLM开始说话的emoji
                        printer.write(f"\n💬 ")
                        
                        hallucination_detected = False
                        stream_error_occurred = False
                        last_event_type = None
                        error_details = None

                        with self.client.messages.stream(
                            model=self.model,
                            max_tokens=self._get_max_tokens_for_model(self.model),
                            system=system_message,
                            messages=claude_messages,
                            tools=tools,
                            temperature=0.7
                        ) as stream:
                            try:
                                for event in stream:
                                    # 在每个事件处理前检查是否已经检测到幻觉
                                    if hallucination_detected:
                                        print_current("\n🛑 Stopping stream due to hallucination detection")
                                        break
                                        
                                    try:
                                        event_type = getattr(event, 'type', None)
                                        last_event_type = event_type
                                        
                                        # 详细记录事件的原始数据用于调试
                                        try:
                                            # 尝试获取事件的所有属性
                                            event_dict = {}
                                            if hasattr(event, '__dict__'):
                                                event_dict = event.__dict__
                                            elif hasattr(event, 'model_dump'):
                                                event_dict = event.model_dump()
                                        except:
                                            pass

                                        # 处理内容块开始事件
                                        if event_type == "content_block_start":
                                            try:
                                                
                                                # 安全地获取index属性
                                                try:
                                                    block_index = getattr(event, 'index', None)
                                                except Exception as idx_err:
                                                    print_error(f"   Error getting index: {type(idx_err).__name__}: {str(idx_err)}")
                                                    block_index = None
                                                
                                                # 安全地获取content_block属性
                                                try:
                                                    content_block = getattr(event, 'content_block', None)

                                                    # 尝试序列化content_block以查看其内容
                                                    if content_block:
                                                        try:
                                                            if hasattr(content_block, '__dict__'):
                                                                pass
                                                        except Exception as dump_err:
                                                            pass
                                                except Exception as cb_err:
                                                    print_error(f"   Error getting content_block: {type(cb_err).__name__}: {str(cb_err)}")
                                                    import traceback
                                                    content_block = None
                                                
                                                if content_block:
                                                    try:
                                                        block_type = getattr(content_block, 'type', None)
                                                        current_block_index = block_index
                                                        current_block_type = block_type
                                                        
                                                        if block_type == "tool_use":
                                                            # 开始一个新的工具调用
                                                            tool_id = getattr(content_block, 'id', '')
                                                            tool_name = getattr(content_block, 'name', '')
                                                            
                                                            tool_call_buffers[block_index] = {
                                                                "id": tool_id,
                                                                "name": tool_name,
                                                                "input_json": ""
                                                            }
                                                    except Exception as type_err:
                                                        print_error(f"   Error processing block_type: {type(type_err).__name__}: {str(type_err)}")
                                            except Exception as e:
                                                print_error(f"⚠️ Error processing content_block_start: {type(e).__name__}: {str(e)}")
                                                import traceback
                                                print_error(f"   Full traceback:\n{traceback.format_exc()}")
                                                # 继续处理其他事件，不中断流

                                        # 处理内容块增量事件
                                        elif event_type == "content_block_delta":
                                            try:
                                                delta = getattr(event, 'delta', None)
                                                block_index = getattr(event, 'index', None)
                                                
                                                if delta:
                                                    delta_type = getattr(delta, 'type', None)
                                                    
                                                    if delta_type == "text_delta":
                                                        # 文本内容流式输出
                                                        text = getattr(delta, 'text', '')
                                                        # 检测幻觉模式 - 严格匹配两个标志
                                                        if "LLM Called Following Tools in this round" in text or "**Tool Execution Results:**" in text:
                                                            print_current("\n🚨 Hallucination detected, stopping conversation")
                                                            hallucination_detected = True
                                                            break
                                                        printer.write(text)
                                                        content += text
                                                    
                                                    elif delta_type == "input_json_delta":
                                                        partial_json = getattr(delta, 'partial_json', '')

                                                        if block_index in tool_call_buffers and partial_json:
                                                            current_json = tool_call_buffers[block_index]["input_json"]
                                                            
                                                            tool_call_buffers[block_index]["input_json"] += partial_json
                                            except Exception as e:
                                                print_debug(f"⚠️ Error processing content_block_delta: {type(e).__name__}: {str(e)}")

                                        # 处理内容块结束事件
                                        elif event_type == "content_block_stop":
                                            try:
                                                block_index = getattr(event, 'index', None)
                                                
                                                if block_index in tool_call_buffers:
                                                    # 工具调用块结束，验证并保存
                                                    buffer = tool_call_buffers[block_index]
                                                    tool_name = buffer["name"]
                                                    json_str = buffer["input_json"]
                                                    
                                                    # 验证JSON完整性
                                                    # 修复布尔值格式问题
                                                    # json_str = _fix_json_boolean_values(json_str)
                                                    is_valid, parsed_input, error_msg = validate_tool_call_json(json_str, tool_name)
                                                    
                                                    if is_valid:
                                                        tool_calls.append({
                                                            "id": buffer["id"],
                                                            "name": tool_name,
                                                            "input": parsed_input
                                                        })
                                                        print_debug(f"✅ Tool call validated: {tool_name}")
                                                    else:
                                                        print_error(f"❌ Tool call JSON validation failed for {tool_name}:")
                                                        print_error(f"   {error_msg}")
                                                        print_debug(f"   Raw JSON: {json_str[:200]}...")
                                            except Exception as e:
                                                print_debug(f"⚠️ Error processing content_block_stop: {type(e).__name__}: {str(e)}")

                                        # 处理消息统计信息
                                        elif event_type == "message_delta":
                                            try:
                                                delta = getattr(event, 'delta', None)
                                                if delta:
                                                    usage = getattr(delta, 'usage', None) or getattr(event, 'usage', None)
                                                    if usage:
                                                        input_tokens = getattr(usage, 'input_tokens', 0) or 0
                                                        output_tokens = getattr(usage, 'output_tokens', 0) or 0
                                                        cache_creation_tokens = getattr(usage, 'cache_creation_input_tokens', 0) or 0
                                                        cache_read_tokens = getattr(usage, 'cache_read_input_tokens', 0) or 0

                                                        if cache_creation_tokens > 0 or cache_read_tokens > 0:
                                                            print_debug(f"\n📊 Token Usage - Input: {input_tokens}, Output: {output_tokens}, Cache Creation: {cache_creation_tokens}, Cache Read: {cache_read_tokens}")
                                            except Exception as e:
                                                print_debug(f"⚠️ Error processing message_delta: {type(e).__name__}: {str(e)}")
                                    
                                    except Exception as event_error:
                                        # Single event processing failure should not interrupt the entire stream
                                        print_debug(f"⚠️ Error processing event {last_event_type}: {type(event_error).__name__}: {str(event_error)}")
                                        # Do not use continue, let the loop continue naturally

                            except Exception as e:
                                # 增强的错误处理
                                stream_error_occurred = True
                                error_details = f"Streaming failed at event_type={last_event_type}: {type(e).__name__}: {str(e)}"
                                print_error(error_details)
                                # 如果有部分工具调用数据，尝试保存
                                if tool_call_buffers:
                                    pass

                                # 尝试回退到text_stream
                                try:
                                    for text in stream.text_stream:
                                        if "LLM Called Following Tools in this round" in text or "**Tool Execution Results:**" in text:
                                            print_current("\n🚨 Hallucination detected, stopping conversation")
                                            hallucination_detected = True
                                            break
                                        printer.write(text)
                                        content += text
                                except Exception as fallback_error:
                                    print_error(f"Text streaming also failed: {fallback_error}")
                                    break

                            # 如果检测到幻觉，提前返回
                            if hallucination_detected:
                                return content, []

                        print_current("")

                        # 如果流式处理中没有获取到完整的工具调用，尝试从final message获取
                        if not tool_calls and not stream_error_occurred:
                            try:
                                final_message = stream.get_final_message()
                                
                                for content_block in final_message.content:
                                    if content_block.type == "tool_use":
                                        # 验证工具调用input
                                        tool_input = content_block.input
                                        tool_name = content_block.name
                                        
                                        # input应该已经是dict，但为安全起见检查
                                        if isinstance(tool_input, str):
                                            #tool_input = _fix_json_boolean_values(tool_input)
                                            is_valid, parsed_input, error_msg = validate_tool_call_json(tool_input, tool_name)
                                            if not is_valid:
                                                # Only show failure info for non-empty string errors
                                                if error_msg != "Empty JSON string":
                                                    print_error(f"❌ Final message tool call validation failed: {error_msg}")
                                                else:
                                                    print_debug(f"⚠️ Empty tool input for {tool_name}, skipping")
                                                continue
                                            tool_input = parsed_input
                                        
                                        tool_calls.append({
                                            "id": content_block.id,
                                            "name": tool_name,
                                            "input": tool_input
                                        })

                                if tool_calls:
                                    pass
                            except Exception as e:
                                print_error(f"Failed to get final message: {type(e).__name__}: {str(e)}")

                    # Execute tool calls
                    if tool_calls:
                        for tool_call_data in tool_calls:
                            try:
                                tool_name = tool_call_data['name']

                                # Convert to standard format
                                standard_tool_call = {
                                    "name": tool_name,
                                    "arguments": tool_call_data['input']
                                }

                                tool_result = self.execute_tool(standard_tool_call, streaming_output=True)

                                # 存储结果
                                if not hasattr(self, '_streaming_tool_results'):
                                    self._streaming_tool_results = []

                                self._streaming_tool_results.append({
                                    'tool_name': tool_name,
                                    'tool_params': tool_call_data['input'],
                                    'tool_result': tool_result
                                })

                                self._tools_executed_in_stream = True

                            except Exception as e:
                                print_error(f"❌ Tool {tool_name} execution failed: {str(e)}")

                        print_debug("✅ All tool executions completed")

                    # If an error occurred during streaming, append error details to content for feedback to the LLM
                    if stream_error_occurred and error_details is not None:
                        error_feedback = f"\n\n⚠️ **Streaming Error Feedback**: There was a problem parsing the previous response: {error_details}\nPlease regenerate a correct response based on this error message."
                        content += error_feedback

                    return content, tool_calls
                else:
                    # print_current("🔄 LLM is thinking: ")
                    response = self.client.messages.create(
                        model=self.model,
                        max_tokens=self._get_max_tokens_for_model(self.model),
                        system=system_message,
                        messages=claude_messages,
                        tools=tools,
                        temperature=0.7
                    )
                    
                    content = ""
                    tool_calls = []
                    
                    # Extract content and tool use blocks
                    for content_block in response.content:
                        if content_block.type == "text":
                            content += content_block.text
                        elif content_block.type == "tool_use":
                            tool_calls.append({
                                "id": content_block.id,
                                "name": content_block.name,
                                "input": content_block.input
                            })

                    # Check for hallucination patterns in non-streaming response - strict match
                    if "LLM Called Following Tools in this round" in content or "**Tool Execution Results:**" in content:
                        print_debug("\n🚨 Hallucination Detected, stop chat")
                        # 截断内容到幻觉位置之前，避免打印幻觉字符串
                        hallucination_patterns = [
                            "LLM Called Following Tools in this round",
                            "**Tool Execution Results:**"
                        ]
                        hallucination_start = len(content)
                        for pattern in hallucination_patterns:
                            if pattern in content:
                                hallucination_start = min(hallucination_start, content.find(pattern))
                        if hallucination_start > 0:
                            content = content[:hallucination_start].rstrip()
                        else:
                            content = ""
                        return content, []

                    return content, tool_calls

            except Exception as e:
                error_str = str(e).lower()

                # Check if this is a retryable error
                retryable_errors = [
                    'overloaded', 'rate limit', 'too many requests',
                    'service unavailable', 'timeout', 'temporary failure',
                    'server error', '429', '503', '502', '500'
                ]

                # Find which error keyword matched
                matched_error_keyword = None
                for error_keyword in retryable_errors:
                    if error_keyword in error_str:
                        matched_error_keyword = error_keyword
                        break

                is_retryable = matched_error_keyword is not None

                if is_retryable and attempt < max_retries:
                    # Calculate retry delay with exponential backoff
                    retry_delay = 1

                    print_current(f"⚠️ Claude API {matched_error_keyword} error (attempt {attempt + 1}/{max_retries + 1}): {e}")
                    print_current(f"💡 Consider switching to a different model or trying again later")
                    print_current(f"🔄 You can change the model in config.txt and restart AGIAgent")
                    print_current(f"🔄 Retrying in {retry_delay} seconds...")

                    # Wait before retry
                    time.sleep(retry_delay)
                    continue  # Retry the loop
                    
                else:
                    # Non-retryable error or max retries exceeded
                    if is_retryable:
                        print_current(f"❌ Claude API {matched_error_keyword} error: Maximum retries ({max_retries}) exceeded")
                        print_current(f"💡 Consider switching to a different model or trying again later")
                        print_current(f"🔄 You can change the model in config.txt and restart AGIAgent")
                    else:
                        print_current(f"❌ Claude API call failed: {e}")
                    
                    raise e

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
            Tool parameters dictionary
        """
        if isinstance(tool_call, dict):
            # OpenAI format: {"id": "...", "type": "function", "function": {"name": "...", "arguments": "..."}}
            if "function" in tool_call and isinstance(tool_call["function"], dict):
                arguments = tool_call["function"]["arguments"]
                if isinstance(arguments, str):
                    import json
                    try:
                        return json.loads(arguments)
                    except json.JSONDecodeError as e:
                        # JSON parsing failed, return empty dict to avoid crashes
                        print_current(f"Failed to parse JSON arguments: {arguments}")
                        return {}
                return arguments
            # Anthropic/Chat-based format: {"id": "...", "name": "...", "input": {...}}
            elif "input" in tool_call:
                return tool_call["input"]
            # Legacy/Chat-based format: {"name": "...", "arguments": {...}}
            elif "arguments" in tool_call:
                return tool_call["arguments"]
        else:
            # Handle OpenAI API raw object format as fallback
            if hasattr(tool_call, 'function') and hasattr(tool_call.function, 'arguments'):
                arguments = tool_call.function.arguments
                if isinstance(arguments, str):
                    import json
                    try:
                        return json.loads(arguments)
                    except json.JSONDecodeError as e:
                        # JSON parsing failed, return empty dict to avoid crashes
                        print_current(f"Failed to parse JSON arguments: {arguments}")
                        return {}
                return arguments
            elif hasattr(tool_call, 'input'):
                return tool_call.input
            elif hasattr(tool_call, 'arguments'):
                return tool_call.arguments
        
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
        #formatted_calls.append("**Tool Calls:**")
        
        for i, tool_call in enumerate(tool_calls, 1):
            tool_name = self._get_tool_name_from_call(tool_call)
            tool_params = self._get_tool_params_from_call(tool_call)
            
            formatted_calls.append(f"")
            formatted_calls.append(f"Tool {i}: {tool_name}")
            
            # Format parameters in a readable way
            if tool_params:
                formatted_calls.append("Parameters:")
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
            
            # Use default path if none provided
            if json_file_path is None:
                json_file_path = os.path.join(self.prompts_folder, "tool_prompt.json")
            
            # 🚀 优化：使用文件修改时间检测缓存是否有效
            if not force_reload and self._tool_definitions_cache is not None:
                # 检查所有相关文件的修改时间
                cache_valid = True
                for cached_file, cached_mtime in getattr(self, '_tool_defs_file_mtimes', {}).items():
                    if os.path.exists(cached_file):
                        current_mtime = os.path.getmtime(cached_file)
                        if current_mtime != cached_mtime:
                            cache_valid = False
                            break
                    else:
                        cache_valid = False
                        break
                
                if cache_valid:
                    print_debug("✅ 工具定义缓存命中（文件未修改）")
                    return self._tool_definitions_cache
            
            # 缓存未命中或强制重新加载，记录文件修改时间
            if not hasattr(self, '_tool_defs_file_mtimes'):
                self._tool_defs_file_mtimes = {}
            
            # Load basic tool definitions
            tool_definitions = {}
            
            # Try to load from the provided path
            if os.path.exists(json_file_path):
                with open(json_file_path, 'r', encoding='utf-8') as f:
                    tool_definitions = json.load(f)
                    self._tool_defs_file_mtimes[json_file_path] = os.path.getmtime(json_file_path)
                    print_debug(f"📝 加载工具定义: {json_file_path}")
            else:
                # No fallback definitions available
                tool_definitions = {}
                print_debug(f"⚠️ 工具定义文件不存在: {json_file_path}")
            
            # Load memory tool definitions (only if long-term memory is enabled)
            if self._is_long_term_memory_enabled():
                memory_tools_file = os.path.join(self.prompts_folder, "memory_tools.json")
                if os.path.exists(memory_tools_file):
                    try:
                        with open(memory_tools_file, 'r', encoding='utf-8') as f:
                            memory_tools = json.load(f)
                            tool_definitions.update(memory_tools)
                            self._tool_defs_file_mtimes[memory_tools_file] = os.path.getmtime(memory_tools_file)
                            print_debug("✅ Long-term memory tool definitions loaded")
                    except Exception as e:
                        print_current(f"⚠️ Error loading memory tools: {e}")
            else:
                print_debug("ℹ️ Long-term memory is disabled, skipping memory tool definitions")
            
            # Check if multi-agent mode is enabled
            multi_agent_enabled = self._is_multi_agent_enabled()
            
            if multi_agent_enabled:
                # Load multi-agent tool definitions from custom prompts folder
                multiagent_file_path = os.path.join(self.prompts_folder, "multiagent_tool_prompt.json")
                if os.path.exists(multiagent_file_path):
                    with open(multiagent_file_path, 'r', encoding='utf-8') as f:
                        multiagent_tools = json.load(f)
                        tool_definitions.update(multiagent_tools)
                        self._tool_defs_file_mtimes[multiagent_file_path] = os.path.getmtime(multiagent_file_path)
            
            # 🔧 NEW: Load FastMCP tool definitions dynamically
            # 🚀 优化：只在direct_mcp_client已初始化时才加载FastMCP工具定义
            try:
                # 检查MCP客户端是否已初始化
                if self.direct_mcp_client is not None:
                    fastmcp_tools = self.direct_mcp_client.get_available_tools()
                    if fastmcp_tools:
                        # Only print loading info on first load or force reload
                        should_print = force_reload or not hasattr(self, '_fastmcp_loaded_before')
                        
                        if should_print:
                            print_debug(f"🔧 Loading {len(fastmcp_tools)} FastMCP tool definitions")
                        
                        for tool_name in fastmcp_tools:
                            try:
                                # Get tool definition from direct MCP client
                                tool_def = self.direct_mcp_client.get_tool_definition(tool_name)
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
                    # If FastMCP is not initialized yet, skip loading (will be loaded when MCP is actually used)
                    if not force_reload and not hasattr(self, '_fastmcp_skip_warned'):
                        print_debug("ℹ️ FastMCP not initialized yet, skipping tool definitions (will load when MCP is used)")
                        self._fastmcp_skip_warned = True
                    
            except Exception as e:
                print_debug(f"⚠️ Failed to load FastMCP tool definitions: {e}")
            
            # 🔧 NEW: Load cli-mcp tool definitions dynamically
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
            
            # 🔧 NEW: Load dynamic tools from current_tool_list.json
            # Only load if we're actually reloading (not using cache)
            # This prevents clearing the file if plan_tools just wrote to it in the same round
            is_using_cache = not force_reload and self._tool_definitions_cache is not None and (
                self._tool_definitions_cache_timestamp is not None and 
                time.time() - self._tool_definitions_cache_timestamp < 60
            )
            
            if not is_using_cache:
                # We're reloading, so it's safe to load and clear current_tool_list.json
                # But check if file was just written (within last 5 seconds) to avoid clearing it
                try:
                    current_tool_list_path = os.path.join(self.workspace_dir, "current_tool_list.json")
                    if os.path.exists(current_tool_list_path):
                        # Check file modification time - if modified within last 5 seconds, skip to avoid clearing
                        # a file that was just written by plan_tools
                        file_mtime = os.path.getmtime(current_tool_list_path)
                        time_since_modification = time.time() - file_mtime
                        
                        if time_since_modification < 5:
                            print_debug(f"ℹ️ current_tool_list.json was modified {time_since_modification:.2f} seconds ago, skipping to preserve it (likely just written by plan_tools)")
                        else:
                            try:
                                with open(current_tool_list_path, 'r', encoding='utf-8') as f:
                                    current_tool_list = json.load(f)
                                
                                # Check if file is not empty
                                if current_tool_list and isinstance(current_tool_list, dict) and len(current_tool_list) > 0:
                                    # Merge dynamic tools into tool definitions
                                    tool_definitions.update(current_tool_list)
                                    print_debug(f"✅ Loaded {len(current_tool_list)} dynamic tools from current_tool_list.json")
                                    
                                    # Clear the file after loading
                                    with open(current_tool_list_path, 'w', encoding='utf-8') as f:
                                        json.dump({}, f)
                                    print_debug("✅ Cleared current_tool_list.json after loading")
                                else:
                                    # File is empty, skip
                                    print_debug("ℹ️ current_tool_list.json is empty, skipping")
                            except json.JSONDecodeError as e:
                                print_debug(f"⚠️ Failed to parse current_tool_list.json: {e}")
                            except Exception as e:
                                print_debug(f"⚠️ Failed to load current_tool_list.json: {e}")
                    else:
                        # File doesn't exist, skip silently
                        pass
                except Exception as e:
                    print_debug(f"⚠️ Error processing current_tool_list.json: {e}")
            else:
                # Using cache, skip loading current_tool_list.json to preserve it for next round
                print_debug("ℹ️ Using cached tool definitions, skipping current_tool_list.json to preserve it for next round")
            
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
        #print_current("🔄 Tool definitions cache cleared")
    
    def _is_multi_agent_enabled(self) -> bool:
        """
        Check if multi-agent mode is enabled from configuration.
        
        Returns:
            True if multi-agent mode is enabled, False otherwise
        """
        try:
            from config_loader import get_config_value
            multi_agent_config = get_config_value("multi_agent", "True")
            
            # Handle different possible values
            if isinstance(multi_agent_config, str):
                return multi_agent_config.lower() in ["true", "1", "yes", "on"]
            elif isinstance(multi_agent_config, bool):
                return multi_agent_config
            else:
                return bool(multi_agent_config)
                
        except Exception as e:
            print_current(f"⚠️  Error checking multi-agent configuration: {e}")
            # Default to True if configuration cannot be read
            return True
    
    def _is_long_term_memory_enabled(self) -> bool:
        """
        Check if long-term memory is enabled from configuration or environment variable.
        
        Returns:
            True if long-term memory is enabled, False otherwise
        """
        # First check environment variable (GUI setting takes precedence)
        env_value = os.environ.get('AGIBOT_LONG_TERM_MEMORY', '').lower()
        if env_value:
            return env_value in ('true', '1', 'yes', 'on')
        
        # Then check config file
        try:
            from config_loader import get_config_value
            long_term_memory_config = get_config_value("enable_long_term_memory", "False")
            
            # Handle different possible values
            if isinstance(long_term_memory_config, str):
                return long_term_memory_config.lower() in ["true", "1", "yes", "on"]
            elif isinstance(long_term_memory_config, bool):
                return long_term_memory_config
            else:
                return bool(long_term_memory_config)
                
        except Exception as e:
            print_debug(f"⚠️  Error checking long-term memory configuration: {e}")
            # Default to False if configuration cannot be read
            return False
    
    # Tool prompt generation function moved to utils/parse.py
    
    def _parse_image_tags(self, text: str) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Parse image tags in user input [img=path_to_image_file]
        
        Args:
            text: User input text
            
        Returns:
            Tuple of (processed_text, image_data_list)
            - processed_text: Text with image tags removed
            - image_data_list: List of image data, each containing {'path': str, 'data': str, 'mime_type': str}
        """
        # Regular expression to match image tags
        image_pattern = r'\[img=([^\]]+)\]'
        matches = re.findall(image_pattern, text)
        
        if not matches:
            return text, []
        
        # Process each image file
        image_data_list = []
        for image_path in matches:
            try:
                # Normalize path
                if not os.path.isabs(image_path):
                    # Relative path, relative to project root directory
                    full_path = os.path.join(self.project_root_dir, image_path)
                else:
                    full_path = image_path
                
                # Check if file exists
                if not os.path.exists(full_path):
                    print_current(f"⚠️ Image file does not exist: {full_path}")
                    continue
                
                # Read image file and encode as base64
                with open(full_path, 'rb') as f:
                    image_data = f.read()
                    base64_data = base64.b64encode(image_data).decode('utf-8')
                
                # Get MIME type
                mime_type, _ = mimetypes.guess_type(full_path)
                if not mime_type or not mime_type.startswith('image/'):
                    print_current(f"⚠️ Unsupported image format: {full_path}")
                    continue
                
                image_data_list.append({
                    'path': image_path,
                    'full_path': full_path,
                    'data': base64_data,
                    'mime_type': mime_type
                })
                
                print_current(f"📸 Successfully loaded image: {image_path} ({mime_type})")
                
            except Exception as e:
                print_current(f"❌ Failed to load image {image_path}: {e}")
                continue
        
        # Remove image tags from text
        processed_text = re.sub(image_pattern, '', text).strip()
        
        return processed_text, image_data_list
    
    def _build_message_with_images(self, text_content: str, image_data_list: List[Dict[str, Any]], is_claude: bool = False) -> Any:
        """
        Build message content with images
        
        Args:
            text_content: Text content
            image_data_list: List of image data
            is_claude: Whether using Claude model
            
        Returns:
            Message content built according to model type
        """
        if not image_data_list:
            return text_content
        
        if is_claude:
            # Claude format: using content array
            content_parts = []
            
            # Add text part
            if text_content.strip():
                content_parts.append({
                    "type": "text",
                    "text": text_content
                })
            
            # Add image parts
            for image_data in image_data_list:
                content_parts.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": image_data['mime_type'],
                        "data": image_data['data']
                    }
                })
            
            return content_parts
        else:
            # OpenAI format: using content array
            content_parts = []
            
            # Add text part
            if text_content.strip():
                content_parts.append({
                    "type": "text",
                    "text": text_content
                })
            
            # Add image parts
            for image_data in image_data_list:
                content_parts.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{image_data['mime_type']};base64,{image_data['data']}"
                    }
                })
            
            return content_parts
    
    def _build_new_user_message(self, user_prompt: str, task_history: List[Dict[str, Any]] = None, execution_round: int = 1) -> Any:
        """
        Build user message with new architecture:
        1. Pure user requirement (first)
        2. Rules and tools prompts  
        3. System environment info
        4. Workspace info
        5. History context (with intelligent summarization)
        6. Execution instructions (last)
        
        Args:
            user_prompt: Current user prompt (pure requirement)
            task_history: Previous task execution history
            execution_round: Current execution round number
            
        Returns:
            Structured user message (string or content array with images)
        """
        # Check and process image tags in first iteration
        processed_prompt = user_prompt
        image_data_list = []
        
        if execution_round == 1:
            processed_prompt, image_data_list = self._parse_image_tags(user_prompt)
        
        message_parts = []
        
        # 1. Pure user requirement (first)
        message_parts.append(processed_prompt)
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
        
        # 3.5. Mailbox status information (if multi-agent mode is enabled)
        mailbox_status = self._get_mailbox_status_info()
        if mailbox_status:
            message_parts.append("---")
            message_parts.append("")
            message_parts.append(mailbox_status)
            message_parts.append("")
        
        # 4. Workspace information
        if prompt_components['workspace_info']:
            message_parts.append("---")
            message_parts.append("")
            message_parts.append(prompt_components['workspace_info'])
            message_parts.append("")
        
        # 5. Add task history context if provided
        if task_history:
            message_parts.append("---")
            message_parts.append("")
            
            # Use task_history directly since image optimization is now handled after vision API analysis
            processed_history = task_history
            
            # 🔧 优化：计算历史记录长度时，只计算result字段，因为prompt字段只在第一轮存在
            total_history_length = sum(len(str(record.get("result", ""))) for record in processed_history)
            
            # Check if we need to trim the history when it's too long
            max_history_length = 50000  # Default maximum history length
            if total_history_length > max_history_length:
                print_system(f"📊 History length ({total_history_length} chars) exceeds maximum ({max_history_length} chars)")
                print_system("⚠️ History is very long. Using recent history subset to keep context manageable.")
                
                # Use recent history subset as fallback when history is still too long
                recent_history = self._get_recent_history_subset(processed_history, max_length=max_history_length // 2)
                self._add_full_history_to_message(message_parts, recent_history)
                print_system(f"📋 Using recent history subset: {len(recent_history)} records instead of {len(processed_history)} records")
            else:
                # History is manageable, use processed history
                self._add_full_history_to_message(message_parts, processed_history)
        
        # 6. Execution instructions (last)
        message_parts.append("---")
        message_parts.append("")
        message_parts.append("## Execution Instructions:")
        
        # Check if we're in infinite loop mode
        infinite_loop_mode = (self.subtask_loops == -1)
        
        if infinite_loop_mode:
            message_parts.append(f"This is round {execution_round} of task execution in **INFINITE AUTONOMOUS LOOP MODE**.")
            message_parts.append("")
            message_parts.append("**IMPORTANT - INFINITE LOOP MODE INSTRUCTIONS:**")
            message_parts.append("- You are currently running in infinite autonomous loop mode")
            message_parts.append("- The system will continue executing until the task is naturally completed")
            message_parts.append("- DO NOT use TASK_COMPLETED flag in this mode - it will not stop the execution")
            message_parts.append("- When task is truly completed, use: talk_to_user(query=\"TASK_COMPLETED: [description]\", timeout=-1)")
            message_parts.append("- The timeout=-1 parameter disables timeout for task completion notification")
            message_parts.append("- Focus on making continuous progress towards the goal")
            message_parts.append("- Use tools and make changes as needed to achieve the objective")
            message_parts.append("")
            message_parts.append("Please continue with the task based on the above context and requirements.")
        else:
            message_parts.append(f"This is round {execution_round} of task execution. Please continue with the task based on the above context and requirements.")
            message_parts.append("")
            message_parts.append("**TASK COMPLETION:** When you've fully completed the task, use TASK_COMPLETED: [description] to signal completion.")
        
        # Build final message
        combined_message = "\n".join(message_parts)
        
        # If there is image data, build message format with images
        if image_data_list:
            final_message = self._build_message_with_images(combined_message, image_data_list, self.is_claude)
            print_current(f"📸 First iteration contains {len(image_data_list)} images")
        else:
            final_message = combined_message
        
        if self.debug_mode:
            if task_history:
                pass
        
        return final_message

    def _get_mailbox_status_info(self) -> Optional[str]:
        """
        Get mailbox status information for the current agent.
        
        Returns:
            Mailbox status information string if there are unread messages, None otherwise
        """
        try:
            # Only check mailbox status in multi-agent mode
            if not self.multi_agent:
                return None
            
            # Get current agent ID
            from src.tools.agent_context import get_current_agent_id
            current_agent_id = get_current_agent_id()
            if not current_agent_id:
                return None
            
            # Get message router and mailbox
            from src.tools.message_system import get_message_router
            router = get_message_router(self.workspace_dir, cleanup_on_init=False)
            if not router:
                return None
            
            mailbox = router.get_mailbox(current_agent_id)
            if not mailbox:
                return None
            
            # Get unread messages count
            unread_messages = mailbox.get_unread_messages()
            unread_count = len(unread_messages)
            
            if unread_count > 0:
                # Format mailbox status information
                status_info = f"## 📬 Mailbox Status\n"
                status_info += f"**Agent {current_agent_id}** has **{unread_count} unread message(s)** in mailbox.\n\n"
                status_info += f"💡 **Action Required**: You can use the `read_received_messages` tool to read these messages.\n"
                status_info += f"   - Use `read_received_messages(include_read=False)` to read only unread messages\n"
                status_info += f"   - Use `read_received_messages(include_read=True)` to read all messages (including already read ones)\n\n"
                
                # Add priority information if available
                if unread_messages:
                    priority_counts = {}
                    for msg in unread_messages:
                        if hasattr(msg, 'priority') and hasattr(msg.priority, 'value'):
                            priority = msg.priority.value
                        else:
                            priority = 'normal'
                        priority_counts[priority] = priority_counts.get(priority, 0) + 1
                    
                    if priority_counts:
                        status_info += f"📊 **Message Priority Breakdown**:\n"
                        for priority, count in sorted(priority_counts.items(), reverse=True):
                            # Ensure priority is a string before calling upper()
                            priority_str = str(priority).upper()
                            status_info += f"   - {priority_str}: {count} message(s)\n"
                        status_info += "\n"
                
                return status_info
            
            return None
            
        except Exception as e:
            # Silently fail to avoid disrupting normal operation
            if self.debug_mode:
                print_current(f"⚠️ Error checking mailbox status: {e}")
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
            
            # Check if it's a direct MCP tool
            if hasattr(self, 'direct_mcp_client') and self.direct_mcp_client and self.direct_mcp_initialized:
                direct_mcp_tools = self.direct_mcp_client.get_available_tools()
                if tool_name in direct_mcp_tools:
                    tool_def = self.direct_mcp_client.get_tool_definition(tool_name)
                    if tool_def:
                        return {
                            "description": tool_def.get("description", f"SSE MCP tool: {tool_name}"),
                            "parameters": tool_def.get("inputSchema", tool_def.get("input_schema", {})),
                            "notes": f"MCP tool (SSE): {tool_name}. This is an MCP tool connected via SSE protocol.",
                            "tool_type": "direct-mcp"
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
            
            # Add direct MCP tools
            if hasattr(self, 'direct_mcp_client') and self.direct_mcp_client and self.direct_mcp_initialized:
                direct_mcp_tools = self.direct_mcp_client.get_available_tools()
                for tool_name in direct_mcp_tools:
                    try:
                        tool_def = self.direct_mcp_client.get_tool_definition(tool_name)
                        description = tool_def.get("description", f"SSE MCP tool: {tool_name}") if tool_def else f"SSE MCP tool: {tool_name}"
                        first_sentence = description.split(".")[0] + "." if "." in description else description
                        if len(first_sentence) > 100:
                            first_sentence = first_sentence[:97] + "..."
                        all_tools[tool_name] = f"[MCP/SSE] {first_sentence}"
                    except Exception as e:
                        all_tools[tool_name] = f"[MCP/SSE] {tool_name} (error getting definition)"
        
        except Exception as e:
            print_current(f"⚠️ Error getting MCP tool list: {e}")
        
        return all_tools

    def _stream_tool_execution(self, tool_name: str, params: Dict[str, Any], tool_func) -> None:
        """
        Show streaming output for tool execution progress.
        
        Args:
            tool_name: Name of the tool being executed
            params: Parameters passed to the tool
            tool_func: The tool function to execute
        """
        try:
            # Log detailed parameters to file
            detailed_params = str(params)
            print_debug(f"Tool {tool_name} with parameters: {detailed_params}")
            print_debug(f"   Working directory: {os.getcwd()}")
            print_debug(f"   Status: Executing...")
            
            # For certain tools, show important parameters in terminal
            if tool_name == 'run_terminal_cmd':
                command = params.get('command', 'unknown command')
                print_debug(f"🚀 Command execution started, real-time output as follows:")
                print_debug(f"   Working directory: {os.getcwd()}")
                
            elif tool_name == 'read_file':
                target_file = params.get('target_file', 'unknown file')
                print_debug(f"🎯 Requested to read file: {target_file}")
                
            elif tool_name == 'edit_file':
                target_file = params.get('target_file', 'unknown file')
                edit_mode = params.get('edit_mode', 'unknown mode')
                print_debug(f"📝 Editing file: {target_file} (mode: {edit_mode})")
                
            elif tool_name == 'web_search':
                search_term = params.get('search_term', 'unknown query')
                print_debug(f"🔍 Starting web search: {search_term}")
                
            elif tool_name == 'workspace_search':
                query = params.get('query', 'unknown query')
                print_debug(f"🔍 Searching workspace: {query}")
            
        except Exception as e:
            print_error(f"⚠️ Error showing tool execution progress: {e}")

    def _stream_tool_result(self, tool_name: str, result: Any, tool_params: dict = None) -> None:
        """
        Stream tool execution result in real-time.
        
        Args:
            tool_name: Name of the tool that was executed
            result: Result from tool execution
            tool_params: Parameters passed to the tool (optional)
        """
        try:
            # Log full result to file
            full_result_str = str(result)
            print_debug(f"Tool {tool_name} full result: {full_result_str}")
            
            # Show results for various tool types
            if isinstance(result, dict):
                # Use simplified formatting for search tools if enabled in config
                if (self.simplified_search_output and 
                    tool_name in ['workspace_search', 'web_search']):
                    formatted_result = self._format_search_result_for_terminal(result, tool_name)
                    print_current(formatted_result)
                
                # Handle MCP tools (FastMCP, CLI-MCP, Direct MCP) results
                elif tool_name in getattr(self, 'tool_source_map', {}) or any(prefix in tool_name for prefix in ['jina_', 'cli_mcp_']):
                    tool_source = getattr(self, 'tool_source_map', {}).get(tool_name, 'unknown')
                    
                    if result.get('status') == 'success' and 'result' in result:
                        tool_result_content = result['result']
                        
                        # Format the result content appropriately - NO TRUNCATION
                        if isinstance(tool_result_content, str):
                            # For string results, show directly without truncation
                            print_debug(f"✅ {tool_source.upper()} Tool Result:\n{tool_result_content}")
                        elif isinstance(tool_result_content, dict):
                            # For dict results, format as text without truncation
                            formatted_result = self._format_dict_as_text(tool_result_content, for_terminal_display=True, tool_name=tool_name, tool_params=tool_params)
                            print_debug(f"✅ {tool_source.upper()} Tool Result:\n{formatted_result}")
                        else:
                            print_debug(f"✅ {tool_source.upper()} Tool Result: {str(tool_result_content)}")
                    elif result.get('status') == 'error':
                        error_msg = result.get('error', 'Unknown error')
                        print_debug(f"❌ {tool_source.upper()} Tool Error: {error_msg}")
                    else:
                        print_debug(f"ℹ️  {tool_source.upper()} Tool Status: {result.get('status', 'unknown')}")
                
                # For file operations, show results for edit_file tool and errors for others
                elif 'status' in result:
                    if tool_name == 'edit_file':
                        # Always show edit_file results (success or error)
                        status = result.get('status', 'unknown')
                        file_path = result.get('file', 'unknown file')
                        action = result.get('action', 'processed')

                        if status == 'success':
                            print_debug(f"✅ File Operation Succeed: {action} {file_path}")
                        elif status in ['error', 'failed']:
                            error_msg = result.get('error')
                            print_debug(f"❌ File Operation Failed: {file_path} - {error_msg}")
                    elif result.get('status') in ['error', 'failed']:
                        status = result.get('status', 'unknown')
                        print_debug(f"Status: {status}")
            
        except Exception as e:
            print_error(f"⚠️ Error streaming tool result: {e}")

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

    def _extract_current_round_images(self, tool_results: List[Dict[str, Any]]) -> None:
        """
        Extract image data from current round tool results for next round vision API.
        
        Args:
            tool_results: List of tool execution results
        """
        # Only clear if we actually have new image data to process
        new_images = []
        
        for result in tool_results:
            tool_name = result.get('tool_name', '')
            tool_result = result.get('tool_result', {})
            
            # Check if this is get_sensor_data with image data
            if tool_name == 'get_sensor_data' and isinstance(tool_result, dict):
                data_field = tool_result.get('data', '')
                dataformat = tool_result.get('dataformat', '')
                
                # Check if it's image data
                if (isinstance(data_field, str) and 
                    len(data_field) > 1000 and  # Likely base64 data
                    'base64 encoded image/' in dataformat):
                    
                    # Clean the base64 data (remove any file markers)
                    import re
                    clean_base64 = re.sub(r'\[FILE_(?:SOURCE|SAVED):[^\]]+\]', '', data_field)
                    
                    # Extract MIME type from dataformat
                    if 'image/jpeg' in dataformat:
                        mime_type = 'image/jpeg'
                    elif 'image/png' in dataformat:
                        mime_type = 'image/png'
                    else:
                        mime_type = 'image/jpeg'  # default
                    
                    # Store image data for next round
                    new_images.append({
                        'data': clean_base64,
                        'mime_type': mime_type
                    })
                    
                    print_current(f"🖼️ Stored image data for next round vision API (MIME: {mime_type}, size: {len(clean_base64)} chars)")
        
        # Only update the array if we found new images
        if new_images:
            self.current_round_images = new_images
    
    def _build_vision_message(self, text_content: str) -> List[Dict[str, Any]]:
        """
        Build vision message content array from text and stored images.
        
        Args:
            text_content: Text content to include
            
        Returns:
            Content array for vision API
        """
        content_parts = []
        
        # Add text part
        content_parts.append({
            "type": "text",
            "text": text_content
        })
        
        # Add image parts
        for img_data in self.current_round_images:
            if self.is_claude:
                # Claude format
                content_parts.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": img_data['mime_type'],
                        "data": img_data['data']
                    }
                })
            else:
                # OpenAI format
                content_parts.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{img_data['mime_type']};base64,{img_data['data']}"
                    }
                })
        
        return content_parts
    
    def _perform_vision_analysis(self, vision_content: List[Dict[str, Any]], original_content: str) -> str:
        """
        Perform immediate vision analysis using the vision-capable model.
        
        Args:
            vision_content: Content array with text and images for vision API
            original_content: Original LLM response content
            
        Returns:
            Vision analysis result as string
        """
        try:
            # Prepare system prompt for vision analysis
            vision_system_prompt = "You are an AI assistant with vision capabilities. Analyze the images provided and give detailed descriptions of what you see."
            
            # Prepare messages for vision analysis
            vision_messages = [
                {"role": "system", "content": vision_system_prompt},
                {"role": "user", "content": vision_content}
            ]
            
            print_current("🔍 Performing vision analysis...")
            
            # Call LLM with vision data
            if self.is_claude:
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=self._get_max_tokens_for_model(self.model),
                    system=vision_system_prompt,
                    messages=[{"role": "user", "content": vision_content}],
                    temperature=0.7
                )
                
                vision_analysis = ""
                for content_block in response.content:
                    if content_block.type == "text":
                        vision_analysis += content_block.text
                        
            else:
                # OpenAI format
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=vision_messages,
                    max_tokens=self._get_max_tokens_for_model(self.model),
                    temperature=0.7,
                    top_p=0.8,
                    stream=False
                )

                # Extract content and thinking field from OpenAI response
                message = response.choices[0].message
                vision_analysis = message.content or ""

                # Handle thinking field for OpenAI o1 models and other reasoning models
                thinking = getattr(message, 'thinking', None)
                if thinking:
                    # Combine thinking and content with clear separation
                    vision_analysis = f"## Thinking Process\n\n{thinking}\n\n## Analysis Result\n\n{vision_analysis}"
            
            print_current(f"✅ Vision analysis completed: {len(vision_analysis)} characters")
            return f"## Vision Analysis Results:\n\n{vision_analysis}"
            
        except Exception as e:
            print_current(f"❌ Vision analysis failed: {e}")
            # Fall back to text description
            text_content = ""
            for item in vision_content:
                if item.get("type") == "text":
                    text_content = item.get("text", "")
                    break
            return f"## Tool Results (Vision analysis failed):\n\n{text_content}"


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