"""
Unified Memory Management Interface
"""

import os
import json
import time
import requests
import threading
import queue
from typing import List, Dict, Any, Optional, Tuple, Callable
from dataclasses import dataclass
import functools
from functools import lru_cache

from ..utils.exceptions import MemorySystemError, ConfigError, LLMClientError
from ..utils.logger import get_logger
from ..utils.monitor import monitor_operation
from ..utils.cache_strategy import cache_result
from ..clients.llm_client import LLMClient
from .preliminary import PreliminaryMemoryManager
from .memoir import MemoirManager

logger = get_logger(__name__)


@dataclass
class WriteRequest:
    """Data structure for write requests"""
    text: str
    update_memoir_all: bool = True
    max_days_back: Optional[int] = None
    callback: Optional[Callable[[Dict[str, Any]], None]] = None
    request_id: Optional[str] = None
    timestamp: float = None
    priority: int = 0  # Priority, the higher the number, the higher the priority
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()
        if self.request_id is None:
            self.request_id = f"req_{int(self.timestamp * 1000)}"


class PriorityQueue:
    """Priority queue implementation"""
    
    def __init__(self):
        self.queue = queue.PriorityQueue()
    
    def put(self, priority: int, item: Any):
        """Add item to the queue"""
        # Use negative priority because PriorityQueue is a min-heap
        self.queue.put((-priority, time.time(), item))
    
    def get(self, timeout: Optional[float] = None):
        """Get item from the queue"""
        try:
            priority, timestamp, item = self.queue.get(timeout=timeout)
            return item
        except queue.Empty:
            raise queue.Empty
    
    def qsize(self) -> int:
        """Get queue size"""
        return self.queue.qsize()
    
    def empty(self) -> bool:
        """Check if the queue is empty"""
        return self.queue.empty()
    
    def task_done(self):
        """Mark task as done"""
        self.queue.task_done()
    
    def join(self):
        """Wait for all tasks to be completed"""
        self.queue.join()


def handle_exceptions(func: Callable) -> Callable:
    """
    Exception handling decorator

    Args:
        func: Function to be decorated

    Returns:
        Decorated function that automatically handles exceptions and returns unified format
    """
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        try:
            return func(self, *args, **kwargs)
        except Exception as e:
            logger.error(f"Exception occurred in {func.__name__}: {e}")
            return {
                "success": False,
                "error": str(e),
                "timestamp": time.time()
            }
    return wrapper


@dataclass
class ToolCall:
    """Tool call definition"""
    name: str
    arguments: Dict[str, Any]


class ToolCallExtractor:
    """Tool call parser, responsible for extracting tool calls from LLM responses"""

    @staticmethod
    def extract_from_response(response: Dict[str, Any], logger=None) -> List[ToolCall]:
        """
        Extract tool calls from LLM response

        Args:
            response: LLM API response
            logger: Logger

        Returns:
            List of tool calls
        """
        tool_calls = []
        try:
            choices = response.get('choices', [])
            if not choices:
                if logger:
                    logger.warning("No choices field in response")
                return tool_calls

            choice = choices[0]
            message = choice.get('message', {})

            if not message:
                if logger:
                    logger.warning("No message field in response")
                return tool_calls

            # Check for tool calls (new format)
            if 'tool_calls' in message:
                tool_calls_data = message.get('tool_calls', [])
                if logger:
                    logger.info(f"Detected {len(tool_calls_data)} tool calls")

                for tool_call in tool_calls_data:
                    function_call = tool_call.get('function', {})
                    name = function_call.get('name', '')
                    arguments_str = function_call.get('arguments', '{}')

                    if not name:
                        if logger:
                            logger.warning("Tool call missing name field")
                        continue

                    try:
                        arguments = json.loads(arguments_str)
                        tool_calls.append(
                            ToolCall(name=name, arguments=arguments))
                        if logger:
                            logger.info(f"Successfully extracted tool call: {name} - {arguments}")
                    except json.JSONDecodeError as e:
                        if logger:
                            logger.warning(
                                f"Failed to parse tool arguments: {e}, args: {arguments_str}")
                        # Try to extract arguments from string
                        arguments = ToolCallExtractor._extract_arguments_from_string(
                            arguments_str)
                        tool_calls.append(
                            ToolCall(name=name, arguments=arguments))
                        if logger:
                            logger.info(f"Extracted tool call using string parsing: {name} - {arguments}")

            # Check for function calls (old format, compatibility)
            elif 'function_call' in message:
                function_call = message['function_call']
                name = function_call.get('name', '')
                arguments_str = function_call.get('arguments', '{}')

                if not name:
                    if logger:
                        logger.warning("Function call missing name field")
                    return tool_calls

                try:
                    arguments = json.loads(arguments_str)
                    tool_calls.append(ToolCall(name=name, arguments=arguments))
                    if logger:
                        logger.info(f"Successfully extracted function call: {name} - {arguments}")
                except json.JSONDecodeError as e:
                    if logger:
                        logger.warning(f"Failed to parse function arguments: {e}, args: {arguments_str}")
                    # Try to extract arguments from string
                    arguments = ToolCallExtractor._extract_arguments_from_string(
                        arguments_str)
                    tool_calls.append(ToolCall(name=name, arguments=arguments))
                    if logger:
                        logger.info(f"Extracted function call using string parsing: {name} - {arguments}")

            # If no tool calls, try to extract from content
            elif 'content' in message:
                content = message.get('content', '')
                if content:
                    if logger:
                        logger.info("Attempting to extract tool calls from content")
                    content_tool_calls = ToolCallExtractor.extract_from_content(
                        content, logger)
                    tool_calls.extend(content_tool_calls)

        except Exception as e:
            if logger:
                logger.error(f"Failed to extract tool calls: {e}")
                logger.error(f"Response content: {response}")

        return tool_calls

    @staticmethod
    def extract_from_content(content: str, logger=None) -> List[ToolCall]:
        """
        Extract tool calls from text content

        Args:
            content: Text content
            logger: Logger

        Returns:
            List of tool calls
        """
        tool_calls = []
        try:
            if not content or not content.strip():
                return tool_calls

            # Try to parse JSON format tool calls
            content = content.strip()

            # Find JSON blocks
            json_start = content.find('{')
            json_end = content.rfind('}')

            if json_start != -1 and json_end != -1 and json_end > json_start:
                json_content = content[json_start:json_end + 1]
                try:
                    data = json.loads(json_content)

                    # Check if it's a single tool call
                    if 'name' in data and 'arguments' in data:
                        name = data.get('name', '')
                        arguments = data.get('arguments', {})
                        if name:
                            tool_calls.append(
                                ToolCall(name=name, arguments=arguments))
                            if logger:
                                logger.info(
                                    f"Extracted tool call from JSON: {name} - {arguments}")

                    # Check if it's an array of tool calls
                    elif 'tool_calls' in data:
                        for tool_call_data in data['tool_calls']:
                            name = tool_call_data.get('name', '')
                            arguments = tool_call_data.get('arguments', {})
                            if name:
                                tool_calls.append(
                                    ToolCall(name=name, arguments=arguments))
                                if logger:
                                    logger.info(
                                        f"Extracted tool call from JSON array: {name} - {arguments}")

                except json.JSONDecodeError as e:
                    if logger:
                        logger.warning(f"JSON parsing failed: {e}")
                    # Continue to try other methods

            # If JSON parsing fails, try to extract from natural language
            if not tool_calls:
                tool_calls = ToolCallExtractor.extract_from_natural_language(
                    content, logger)

        except Exception as e:
            if logger:
                logger.error(f"Failed to extract tool calls from content: {e}")

        return tool_calls

    @staticmethod
    def extract_from_natural_language(content: str, logger=None) -> List[ToolCall]:
        """
        Extract tool calls from natural language

        Args:
            content: Natural language content
            logger: Logger

        Returns:
            List of tool calls
        """
        tool_calls = []
        try:
            if not content or not content.strip():
                return tool_calls

            lines = content.split('\n')
            current_tool = None
            current_args = {}

            for line in lines:
                line = line.strip()
                if not line:
                    continue

                # Check if it's a tool name (multiple formats)
                tool_patterns = [
                    r'^Tool:\s*(.+)$',
                    r'^Function:\s*(.+)$',
                    r'^Search Method:\s*(.+)$'
                ]

                tool_found = False
                for pattern in tool_patterns:
                    import re
                    match = re.match(pattern, line, re.IGNORECASE)
                    if match:
                        if current_tool:
                            tool_calls.append(
                                ToolCall(name=current_tool, arguments=current_args))
                            if logger:
                                logger.info(
                                    f"Extracted tool call from natural language: {current_tool} - {current_args}")

                        current_tool = match.group(1).strip()
                        current_args = {}
                        tool_found = True
                        break

                if tool_found:
                    continue

                # Check if it's a parameter (multiple formats)
                if current_tool and ':' in line:
                    try:
                        key, value = line.split(':', 1)
                        key = key.strip()
                        value = value.strip()

                        # Try to parse JSON value
                        try:
                            parsed_value = json.loads(value)
                            current_args[key] = parsed_value
                        except json.JSONDecodeError:
                            # If not JSON, use string value directly
                            current_args[key] = value
                    except ValueError:
                        # If split fails, skip this line
                        continue

            # Add the last tool
            if current_tool:
                tool_calls.append(
                    ToolCall(name=current_tool, arguments=current_args))
                if logger:
                    logger.info(
                        f"Extracted tool call from natural language: {current_tool} - {current_args}")

            # If no tool calls found, try to extract from the whole content
            if not tool_calls:
                tool_calls = ToolCallExtractor._extract_from_whole_content(
                    content, logger)

        except Exception as e:
            if logger:
                logger.error(f"Failed to extract tool calls from natural language: {e}")

        return tool_calls

    @staticmethod
    def _extract_from_whole_content(content: str, logger=None) -> List[ToolCall]:
        """
        Attempt to extract tool calls from the whole content (fallback method)

        Args:
            content: Content
            logger: Logger

        Returns:
            List of tool calls
        """
        tool_calls = []
        try:
            # Try to match common search patterns
            search_patterns = [
                (r'Search.*?memory', 'search_preliminary_memories_by_query'),
                (r'Query.*?memory', 'search_preliminary_memories_by_query'),
                (r'Find.*?memory', 'search_preliminary_memories_by_query'),
                (r'Time.*?search', 'search_preliminary_memories_by_time'),
                (r'Date.*?search', 'search_preliminary_memories_by_time'),
                (r'Summary.*?search', 'search_memoir_memories_by_query'),
                (r'Trend.*?search', 'search_memoir_memories_by_query'),
                (r'Pattern.*?search', 'search_memoir_memories_by_query')
            ]

            import re
            for pattern, tool_name in search_patterns:
                if re.search(pattern, content, re.IGNORECASE):
                    # Extract query content
                    query_match = re.search(r'["""]([^"""]+)["""]', content)
                    if query_match:
                        query = query_match.group(1)
                    else:
                        # If no quotes, try to extract keywords
                        words = content.split()
                        query = ' '.join(words[:3])  # Take the first 3 words as query

                    tool_calls.append(ToolCall(name=tool_name, arguments={
                                      "query": query, "top_k": 5}))
                    if logger:
                        logger.info(
                            f"Extracted tool call from content pattern match: {tool_name} - {{'query': '{query}', 'top_k': 5}}")
                    break

        except Exception as e:
            if logger:
                logger.error(f"Failed to extract tool calls from whole content: {e}")

        return tool_calls

    @staticmethod
    def _extract_arguments_from_string(arguments_str: str) -> Dict[str, Any]:
        """
        Extract arguments from string (fallback method)

        Args:
            arguments_str: Argument string

        Returns:
            Argument dictionary
        """
        arguments = {}
        try:
            # Try to parse JSON
            return json.loads(arguments_str)
        except json.JSONDecodeError:
            # If not JSON, try to parse key-value pairs
            try:
                # Remove possible quotes and parentheses
                clean_str = arguments_str.strip().strip('"{}')
                if ':' in clean_str:
                    parts = clean_str.split(',')
                    for part in parts:
                        if ':' in part:
                            key, value = part.split(':', 1)
                            key = key.strip().strip('"')
                            value = value.strip().strip('"')
                            arguments[key] = value
            except Exception:
                # If all else fails, return the original string as query parameter
                arguments = {"query": arguments_str.strip()}

        return arguments


class MemManagerAgent:
    """
    Intelligent Memory Management Agent based on Large Models

    Responsible for deciding whether to add or update memories and automatically generating summaries and embeddings.
    Reconstruction version: call the preliminary_memory and memoir_manager modules separately.

    Main interfaces:
    1. write_memory_auto() - Highly encapsulated intelligent input processing
    2. read_memory_auto() - Highly encapsulated intelligent search
    3. get_status_auto() - Get full system status information
    4. get_status_summary() - Get system status summary
    5. health_check() - System health check

    Usage example:
        agent = MemManagerAgent()
        agent.write_memory_auto("Today I learned Python")
        results = agent.read_memory_auto("Python")
    """

    def __init__(
        self,
        storage_path: str = "memory",
        config_file: str = "config.txt",
        similarity_threshold: Optional[float] = None,
        max_tokens: Optional[int] = None,
        enable_async: bool = True,
        max_queue_size: int = 1000,
        worker_threads: int = 2
    ) -> None:
        """
        Initialize MemManagerAgent

        Args:
            storage_path: Storage path
            config_file: Configuration file path
            similarity_threshold: Similarity threshold
            max_tokens: Maximum tokens
            enable_async: Whether to enable async processing
            max_queue_size: Maximum queue size for async processing
            worker_threads: Number of worker threads for async processing
        """
        self.storage_path = storage_path
        self.config_file = config_file
        self.similarity_threshold = similarity_threshold
        self.max_tokens = max_tokens
        
        # Async processing configuration
        self.enable_async = enable_async
        self.max_queue_size = max_queue_size
        self.worker_threads = worker_threads
        
        # Async processing related
        self.write_queue = PriorityQueue() if self.enable_async else queue.Queue(maxsize=self.max_queue_size)
        self.workers = []
        self.is_running = False
        self.async_stats = {
            "total_requests": 0,
            "processed_requests": 0,
            "failed_requests": 0,
            "retried_requests": 0,
            "queue_size": 0,
            "start_time": time.time(),
            "last_processed_time": None,
            "average_processing_time": 0.0,
            "total_processing_time": 0.0
        }
        
        # Request status tracking
        self.request_status = {}  # request_id -> status info

        # Load configuration
        self.config = self._load_config(config_file)

        # Initialize submodules
        self._init_submodules()

        # Start background worker threads
        if self.enable_async:
            self._start_workers()

    def _load_config(self, config_file: str) -> Dict[str, Any]:
        """Load configuration from file"""
        config = {
            "default_top_k": 5,
            "default_max_days_back": 30,
            "similarity_threshold": 0.7,
            "max_tokens": 4096
        }

        try:
            # Use ConfigLoader for multi-file configuration support
            from ..utils.config import ConfigLoader
            
            # Create ConfigLoader instance - it will use default config files if config_file is None
            config_loader = ConfigLoader(config_file)
            
            # Extract relevant configuration values
            config.update({
                "default_top_k": int(config_loader.get("default_top_k", config["default_top_k"])),
                "default_max_days_back": int(config_loader.get("default_max_days_back", config["default_max_days_back"])),
                "similarity_threshold": float(config_loader.get("similarity_threshold", config["similarity_threshold"])),
                "max_tokens": int(config_loader.get("max_tokens", config["max_tokens"]))
            })
            
            # Copy all other config values from ConfigLoader
            for key, value in config_loader.config.items():
                if key not in config:
                    config[key] = value
            
            logger.info(f"Configuration loaded using ConfigLoader with files: {config_loader.config_files}")
            
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            logger.warning("Using default configuration")

        # Apply constructor parameters
        if self.similarity_threshold is not None:
            config["similarity_threshold"] = self.similarity_threshold
        if self.max_tokens is not None:
            config["max_tokens"] = self.max_tokens

        return config

    def _init_submodules(self):
        """Initialize submodules"""
        try:
            # Use a default config file path for submodules
            # They will use ConfigLoader internally which supports multiple files
            # Calculate absolute path to config file
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_dir))))
            default_config = os.path.join(project_root, "config", "config.txt")
            
            # Initialize LLM client
            self.llm_client = LLMClient(config_file=default_config)
            
            # Initialize preliminary memory manager
            preliminary_path = os.path.join(self.storage_path, "preliminary_memory")
            self.preliminary_memory = PreliminaryMemoryManager(
                storage_path=preliminary_path,
                config_file=default_config,
                max_tokens=self.config.get("max_tokens", 4096)
            )

            # Initialize memoir manager
            memoir_path = os.path.join(self.storage_path, "memoir")
            self.memoir_manager = MemoirManager(
                storage_path=memoir_path,
                preliminary_memory=self.preliminary_memory,
                config_file=default_config
            )

            # Set default values
            self.default_top_k = self.config.get("default_top_k", 5)
            self.default_max_days_back = self.config.get("default_max_days_back", 30)

            logger.info("Submodules initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize submodules: {e}")
            raise

    def _start_workers(self):
        """Start background worker threads"""
        if self.is_running:
            return
            
        self.is_running = True
        logger.info(f"Starting {self.worker_threads} worker threads")
        
        for i in range(self.worker_threads):
            worker = threading.Thread(
                target=self._worker_loop,
                name=f"AsyncMemoryWorker-{i}",
                daemon=True
            )
            worker.start()
            self.workers.append(worker)
    
    def _worker_loop(self):
        """Worker thread main loop"""
        logger.info(f"Worker thread {threading.current_thread().name} started")
        
        while self.is_running:
            try:
                # Get request from queue, set timeout to check for stop signal
                request = self.write_queue.get(timeout=1.0)
                
                # Process request
                self._process_write_request(request)
                
                # Mark task as done
                self.write_queue.task_done()
                
            except queue.Empty:
                # Queue is empty, continue loop
                continue
            except Exception as e:
                logger.error(f"Error processing request in worker thread: {e}")
                # Mark task as done to avoid queue blocking
                try:
                    self.write_queue.task_done()
                except:
                    pass
        
        logger.info(f"Worker thread {threading.current_thread().name} stopped")
    
    def _process_write_request(self, request: WriteRequest):
        """Process write request"""
        start_time = time.time()
        retry_count = 0
        max_retries = 3
        
        # Update request status to processing
        if request.request_id in self.request_status:
            self.request_status[request.request_id].update({
                "status": "processing",
                "start_time": start_time
            })
        
        while retry_count <= max_retries:
            try:
                logger.info(f"Processing write request {request.request_id}: {request.text[:50]}...")
                
                # Call synchronous write method
                result = self._write_memory_sync(
                    text=request.text,
                    update_memoir_all=request.update_memoir_all,
                    max_days_back=request.max_days_back
                )
                
                # Update statistics
                processing_time = time.time() - start_time
                self.async_stats["processed_requests"] += 1
                self.async_stats["last_processed_time"] = time.time()
                self.async_stats["total_processing_time"] += processing_time
                self.async_stats["average_processing_time"] = (
                    self.async_stats["total_processing_time"] / self.async_stats["processed_requests"]
                )
                
                # Update request status to completed
                if request.request_id in self.request_status:
                    self.request_status[request.request_id].update({
                        "status": "completed",
                        "result": result,
                        "processing_time": processing_time,
                        "completion_time": time.time()
                    })
                
                # Execute callback
                if request.callback:
                    try:
                        request.callback(result)
                    except Exception as e:
                        logger.error(f"Error executing callback function: {e}")
                
                logger.info(f"Request {request.request_id} processed, time: {processing_time:.2f} seconds")
                return  # Successfully processed, exit retry loop
                
            except Exception as e:
                retry_count += 1
                logger.error(f"Failed to process write request {request.request_id} (attempt {retry_count}/{max_retries + 1}): {e}")
                
                # Update request status to retrying
                if request.request_id in self.request_status:
                    self.request_status[request.request_id].update({
                        "status": "retrying",
                        "retry_count": retry_count,
                        "error": str(e)
                    })
                
                if retry_count <= max_retries:
                    # Still retries available
                    self.async_stats["retried_requests"] += 1
                    time.sleep(1.0)  # Retry delay
                else:
                    # Retry attempts exhausted
                    self.async_stats["failed_requests"] += 1
                    
                    # Update request status to failed
                    if request.request_id in self.request_status:
                        self.request_status[request.request_id].update({
                            "status": "failed",
                            "final_error": str(e),
                            "completion_time": time.time()
                        })
                    
                    # Execute error callback
                    if request.callback:
                        try:
                            error_result = {
                                "success": False,
                                "error": str(e),
                                "request_id": request.request_id,
                                "retry_count": retry_count,
                                "timestamp": time.time()
                            }
                            request.callback(error_result)
                        except Exception as callback_error:
                            logger.error(f"Error executing error callback: {callback_error}")

    def _write_memory_sync(
        self,
        text: str,
        update_memoir_all: bool = True,
        max_days_back: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Synchronous memory writing (internal method)
        
        Args:
            text: Text content to process
            update_memoir_all: Whether to update advanced memory (default True)
            max_days_back: Maximum days for batch update (default 30)

        Returns:
            Writing results
        """
        try:
            # Write to preliminary memory
            preliminary_result = self.preliminary_memory.write_memory_auto(
                text)

            # Update advanced memory
            memoir_update_result = {
                "success": False, "updated_dates": [], "new_memories_processed": 0}

            # Check if memoir should be updated
            should_update_memoir = (
                update_memoir_all and
                preliminary_result.get("success", False) and
                preliminary_result.get("action") in ["added", "updated"]  # Explicitly specify valid action types
            )

            # Add detailed logging for memoir update decision
            logger.info(f"Memoir update decision:")
            logger.info(f"  update_memoir_all: {update_memoir_all}")
            logger.info(f"  preliminary_success: {preliminary_result.get('success', False)}")
            logger.info(f"  preliminary_action: {preliminary_result.get('action', 'unknown')}")
            logger.info(f"  should_update_memoir: {should_update_memoir}")

            if should_update_memoir:
                logger.info("Starting Memoir update...")
                max_days = max_days_back if max_days_back is not None else self.default_max_days_back
                memoir_update_result = self.memoir_manager.update_memoir_all(
                    force_update=True,  # Force completion of all unprocessed memories
                    max_days_back=max_days
                )
                logger.info(f"Memoir update completed: {memoir_update_result}")
            else:
                logger.info(
                    f"Skipping Memoir update: update_memoir_all={update_memoir_all}, preliminary_result={preliminary_result}")

            return {
                "success": True,
                "preliminary_result": preliminary_result,
                "memoir_update_result": memoir_update_result,
                "timestamp": time.time()
            }

        except Exception as e:
            logger.error(f"Intelligent memory writing failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "preliminary_result": {
                    "success": False,
                    "action": "error",
                    "error": str(e)
                },
                "memoir_update_result": {
                    "success": False,
                    "updated_dates": [],
                    "new_memories_processed": 0
                },
                "timestamp": time.time()
            }

    @handle_exceptions
    @monitor_operation("get_status_auto")
    def get_status_auto(self, include_details: bool = True) -> Dict[str, Any]:
        """
        Get comprehensive system status information

        Args:
            include_details: Whether to include detailed information

        Returns:
            System status dictionary
        """
        try:
            # Get basic status
            status = {
                "success": True,
                "timestamp": time.time(),
                "storage_path": self.storage_path,
                "config_file": self.config_file,
                "similarity_threshold": self.similarity_threshold,
                "max_tokens": self.max_tokens,
                "default_top_k": self.default_top_k,
                "default_max_days_back": self.default_max_days_back
            }

            if include_details:
                # Get preliminary memory status
                try:
                    prelim_status = self.preliminary_memory.get_memory_stats()
                    status["preliminary_memory"] = prelim_status
                except Exception as e:
                    status["preliminary_memory"] = {"error": str(e)}

                # Get memoir status
                try:
                    memoir_status = self.memoir_manager.get_status_summary()
                    status["memoir"] = memoir_status
                except Exception as e:
                    status["memoir"] = {"error": str(e)}

                # Get configuration details
                status["config"] = self.config

            # Add async processing status
            async_status = {
                "async_enabled": self.enable_async,
                "queue_size": self.write_queue.qsize(),
                "max_queue_size": self.max_queue_size,
                "worker_threads": self.worker_threads,
                "active_workers": len([w for w in self.workers if w.is_alive()]),
                "is_running": self.is_running,
                "stats": self.async_stats.copy()
            }
            
            status["async_status"] = async_status

            return status

        except Exception as e:
            logger.error(f"Failed to get system status: {e}")
            return {
                "success": False,
                "error": str(e),
                "timestamp": time.time()
            }

    @handle_exceptions
    @monitor_operation("get_status_summary")
    def get_status_summary(self) -> Dict[str, Any]:
        """
        Get system status summary

        Returns:
            Status summary dictionary
        """
        try:
            # Get basic summary
            summary = {
                "success": True,
                "timestamp": time.time(),
                "storage_path": self.storage_path,
                "similarity_threshold": self.similarity_threshold,
                "max_tokens": self.max_tokens
            }

            # Get preliminary memory summary
            try:
                prelim_summary = self.preliminary_memory.get_memory_stats()
                summary["preliminary_memory"] = {
                    "memory_count": prelim_summary.get("memory_count", 0),
                    "total_size_mb": prelim_summary.get("total_size_mb", 0)
                }
            except Exception as e:
                summary["preliminary_memory"] = {"error": str(e)}

            # Get memoir summary
            try:
                memoir_summary = self.memoir_manager.get_status_summary()
                summary["memoir"] = {
                    "total_memoirs": memoir_summary.get("total_memoirs", 0),
                    "total_size_mb": memoir_summary.get("total_size_mb", 0)
                }
            except Exception as e:
                summary["memoir"] = {"error": str(e)}

            # Add async processing summary
            async_summary = {
                "async_enabled": self.enable_async,
                "queue_size": self.write_queue.qsize(),
                "total_requests": self.async_stats["total_requests"],
                "processed_requests": self.async_stats["processed_requests"],
                "failed_requests": self.async_stats["failed_requests"],
                "retried_requests": self.async_stats["retried_requests"],
                "success_rate": (
                    self.async_stats["processed_requests"] / max(self.async_stats["total_requests"], 1)
                ) * 100,
                "average_processing_time": self.async_stats["average_processing_time"]
            }
            
            summary["async_summary"] = async_summary

            return summary

        except Exception as e:
            logger.error(f"Failed to get status summary: {e}")
            return {
                "success": False,
                "error": str(e),
                "timestamp": time.time()
            }

    # --- TF-IDF related methods ---
    def update_preliminary_tfidf_model(self) -> Dict[str, Any]:
        """Manually update preliminary memory TF-IDF model"""
        try:
            return self.preliminary_memory.update_tfidf_model()
        except Exception as e:
            logger.error(f"Failed to update preliminary memory TF-IDF model: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    def get_preliminary_tfidf_stats(self) -> Dict[str, Any]:
        """Get preliminary memory TF-IDF model statistics"""
        try:
            return self.preliminary_memory.get_tfidf_stats()
        except Exception as e:
            logger.error(f"Failed to get preliminary memory TF-IDF statistics: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    @handle_exceptions
    @monitor_operation("health_check")
    def health_check(self) -> Dict[str, Any]:
        """
        System health check

        Returns:
            Health check results
        """
        try:
            health_status = {
                "success": True,
                "timestamp": time.time(),
                "overall_status": "healthy",
                "checks": {}
            }

            # Check preliminary memory
            try:
                prelim_stats = self.preliminary_memory.get_memory_stats()
                health_status["checks"]["preliminary_memory"] = {
                    "status": "healthy",
                    "memory_count": prelim_stats.get("memory_count", 0),
                    "storage_size_mb": prelim_stats.get("total_size_mb", 0)
                }
            except Exception as e:
                health_status["checks"]["preliminary_memory"] = {
                    "status": "unhealthy",
                    "error": str(e)
                }
                health_status["overall_status"] = "degraded"

            # Check memoir
            try:
                memoir_stats = self.memoir_manager.get_status_summary()
                health_status["checks"]["memoir"] = {
                    "status": "healthy",
                    "total_memoirs": memoir_stats.get("total_memoirs", 0),
                    "total_size_mb": memoir_stats.get("total_size_mb", 0)
                }
            except Exception as e:
                health_status["checks"]["memoir"] = {
                    "status": "unhealthy",
                    "error": str(e)
                }
                health_status["overall_status"] = "degraded"

            # Check async processing health status
            async_health = {
                "queue_healthy": self.write_queue.qsize() < self.max_queue_size * 0.8,
                "workers_healthy": all(w.is_alive() for w in self.workers),
                "queue_size": self.write_queue.qsize(),
                "active_workers": len([w for w in self.workers if w.is_alive()]),
                "success_rate": (
                    self.async_stats["processed_requests"] / max(self.async_stats["total_requests"], 1)
                ) * 100
            }
            
            # Determine async processing status
            if not async_health["queue_healthy"]:
                async_health["status"] = "queue_full"
                health_status["overall_status"] = "degraded"
            elif not async_health["workers_healthy"]:
                async_health["status"] = "workers_unhealthy"
                health_status["overall_status"] = "degraded"
            elif async_health["success_rate"] < 90:
                async_health["status"] = "low_success_rate"
                health_status["overall_status"] = "degraded"
            else:
                async_health["status"] = "healthy"
            
            health_status["checks"]["async_processing"] = async_health

            # Check storage
            try:
                if os.path.exists(self.storage_path):
                    health_status["checks"]["storage"] = {
                        "status": "healthy",
                        "path": self.storage_path,
                        "exists": True
                    }
                else:
                    health_status["checks"]["storage"] = {
                        "status": "unhealthy",
                        "path": self.storage_path,
                        "exists": False
                    }
                    health_status["overall_status"] = "degraded"
            except Exception as e:
                health_status["checks"]["storage"] = {
                    "status": "unhealthy",
                    "error": str(e)
                }
                health_status["overall_status"] = "degraded"

            return health_status

        except Exception as e:
            logger.error(f"System health check failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "timestamp": time.time(),
                "overall_status": "error"
            }

    @handle_exceptions
    @monitor_operation("write_memory_auto")
    def write_memory_auto(
        self,
        text: str,
        update_memoir_all: bool = True,
        max_days_back: Optional[int] = None,
        callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        priority: int = 0
    ) -> Dict[str, Any]:
        """
        Intelligent memory writing interface (Async by default)

        Args:
            text: Text content to process
            update_memoir_all: Whether to update advanced memory (default True)
            max_days_back: Maximum days for batch update (default 30)
            callback: Optional callback function to call when the write operation is complete
            priority: Request priority (higher number = higher priority, default 0)

        Returns:
            Writing results (immediate response for async mode)
        """
        if not self.enable_async:
            # Synchronous mode, call directly
            return self._write_memory_sync(text, update_memoir_all, max_days_back)

        try:
            # Create write request
            request = WriteRequest(
                text=text,
                update_memoir_all=update_memoir_all,
                max_days_back=max_days_back,
                callback=callback,
                priority=priority
            )
            
            # Try to add request to queue
            if isinstance(self.write_queue, PriorityQueue):
                self.write_queue.put(priority, request)
            else:
                self.write_queue.put_nowait(request)
            
            self.async_stats["total_requests"] += 1
            self.async_stats["queue_size"] = self.write_queue.qsize()
            
            # Record request status
            self.request_status[request.request_id] = {
                "status": "queued",
                "priority": priority,
                "timestamp": time.time()
            }

            logger.info(f"Write request {request.request_id} added to queue, priority: {priority}, current queue size: {self.async_stats['queue_size']}")
            
            return {
                "success": True,
                "status": "queued",
                "request_id": request.request_id,
                "priority": priority,
                "queue_size": self.async_stats["queue_size"],
                "message": "Request added to queue, will be processed in the background",
                "timestamp": time.time(),
                "async_mode": True,
                "estimated_wait_time": self._estimate_wait_time(),
                "queue_position": self.async_stats["queue_size"],
                "text_preview": request.text[:50] + "..." if len(request.text) > 50 else request.text
            }
            
        except queue.Full:
            logger.warning(f"Write queue is full, rejecting request {request.request_id}")
            return {
                "success": False,
                "status": "queue_full",
                "request_id": request.request_id,
                "error": "Write queue is full, please try again later",
                "timestamp": time.time()
            }
        except Exception as e:
            logger.error(f"Failed to enqueue write request: {e}")
            return {
                "success": False,
                "error": str(e),
                "timestamp": time.time()
            }

    def wait_for_completion(self, timeout: Optional[float] = None) -> bool:
        """
        Wait for all requests in the queue to be processed
        
        Args:
            timeout: Timeout in seconds
            
        Returns:
            Whether waiting was successful
        """
        try:
            self.write_queue.join()
            return True
        except Exception as e:
            logger.error(f"Error waiting for queue completion: {e}")
            return False

    def shutdown(self, wait: bool = True, timeout: Optional[float] = None):
        """
        Shutdown the asynchronous memory manager
        
        Args:
            wait: Whether to wait for all requests to be processed
            timeout: Wait timeout
        """
        logger.info("Starting shutdown of asynchronous memory manager...")
        
        # Stop accepting new requests
        self.is_running = False
        
        if wait:
            # Wait for requests in the queue to be processed
            try:
                self.write_queue.join()
                logger.info("All queue requests processed")
            except Exception as e:
                logger.error(f"Error waiting for queue completion: {e}")
        
        # Wait for worker threads to end
        for worker in self.workers:
            if worker.is_alive():
                worker.join(timeout=timeout or 30.0)
        
        logger.info("Asynchronous memory manager shut down")

    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.shutdown(wait=True)

    @handle_exceptions
    @monitor_operation("read_memory_auto")
    def read_memory_auto(self, query: str, top_k: Optional[int] = None) -> Dict[str, Any]:
        """
        Intelligent memory reading, automatically selects the most appropriate search method.

        Args:
            query (str): Search query content.
            top_k (Optional[int]): Number of results to return, defaults to configuration value.

        Returns:
            Dict[str, Any]: Search result dictionary.

        Example:
            >>> manager = MemManagerAgent()
            >>> result = manager.read_memory_auto("What did I learn about Python today?")
            >>> print(result)
        """
        try:
            if top_k is None:
                top_k = self.default_top_k

            # Validate query
            validation_result = self._validate_query(query, top_k)
            if validation_result:
                return validation_result

            # Try direct time search first
            time_search_result = self._try_direct_time_search(query, top_k)
            if time_search_result:
                return time_search_result

            # Use LLM to determine search method
            return self._perform_llm_search(query, top_k)

        except Exception as e:
            logger.error(f"Intelligent memory search failed: {e}")
            return {
                'success': False,
                'search_type': 'error',
                'query': query,
                'top_k': top_k,
                'results': [],
                'timestamp': time.time(),
                'error': str(e)
            }

    def get_request_status(self, request_id: str) -> Dict[str, Any]:
        """
        Query request status
        
        Args:
            request_id: Request ID
            
        Returns:
            Request status information
        """
        if request_id not in self.request_status:
            return {
                "success": False,
                "error": f"Request {request_id} does not exist",
                "timestamp": time.time()
            }
        
        status_info = self.request_status[request_id].copy()
        status_info["success"] = True
        status_info["timestamp"] = time.time()
        
        return status_info
    
    def get_all_request_status(self) -> Dict[str, Any]:
        """
        Get all request statuses
        
        Returns:
            All request status information
        """
        return {
            "success": True,
            "total_requests": len(self.request_status),
            "requests": self.request_status.copy(),
            "timestamp": time.time()
        }
    
    def cleanup_completed_requests(self, max_age_hours: int = 24):
        """
        Clean up completed request statuses (to avoid memory leaks)
        
        Args:
            max_age_hours: Maximum retention time (hours)
        """
        current_time = time.time()
        max_age_seconds = max_age_hours * 3600
        
        completed_statuses = ["completed", "failed"]
        to_remove = []
        
        for request_id, status_info in self.request_status.items():
            if status_info.get("status") in completed_statuses:
                completion_time = status_info.get("completion_time", 0)
                if current_time - completion_time > max_age_seconds:
                    to_remove.append(request_id)
        
        for request_id in to_remove:
            del self.request_status[request_id]
        
        if to_remove:
            logger.info(f"Cleaned up {len(to_remove)} completed request statuses")

    def _validate_query(self, query: str, top_k: int) -> Optional[Dict[str, Any]]:
        """Validate query parameters"""
        if not query or not query.strip():
            logger.warning("Query is empty, returning empty results")
            return {
                "success": True,
                "search_type": "empty_query",
                "query": query,
                "top_k": top_k,
                "results": [],
                "timestamp": time.time()
            }

        if top_k <= 0:
            return {
                "success": False,
                "error": "top_k must be greater than 0",
                "timestamp": time.time()
            }

        return None

    def _perform_llm_search(self, query: str, top_k: int) -> Dict[str, Any]:
        """Execute LLM intelligent search"""
        try:
            logger.info(f"Starting LLM intelligent search: query='{query}', top_k={top_k}")

            # Build search prompts
            user_prompt = self._build_search_user_prompt(query, top_k)
            system_prompt = self._create_optimized_search_system_prompt()
            tools = self._define_search_tools()

            logger.info(
                f"Built prompt length: user prompt={len(user_prompt)}, system prompt={len(system_prompt)}")
            logger.info(f"Available tools count: {len(tools)}")

            # Call LLM
            llm_response = self._call_llm_with_retry(
                user_prompt, system_prompt, tools)

            if llm_response:
                logger.info("LLM call successful, starting tool call extraction")
                logger.debug(f"LLM response: {llm_response}")

                # Process LLM response
                tool_calls = ToolCallExtractor.extract_from_response(
                    llm_response, logger)

                logger.info(f"Extracted {len(tool_calls)} tool calls")
                if tool_calls:
                    for i, tool_call in enumerate(tool_calls):
                        logger.info(
                            f"Tool call {i+1}: {tool_call.name} - {tool_call.arguments}")

                return self._process_search_results(tool_calls, query, top_k)
            else:
                logger.warning("LLM call failed, using fallback search")
                # Fallback to direct search
                return self._handle_search_fallback(query, top_k)

        except Exception as e:
            logger.error(f"LLM search failed: {e}")
            logger.exception("LLM search exception details:")
            return self._handle_search_fallback(query, top_k)

    def _build_search_user_prompt(self, query: str, top_k: int) -> str:
        """Build search user prompt"""
        return f"""User query: {query}
Return result count: {top_k}

Please analyze the above query, select the most appropriate search tool, and return the tool call in the following JSON format:

```json
{{
  "name": "tool_name",
  "arguments": {{
    "parameter_name": "parameter_value"
  }}
}}
```

**Important requirements:**
1. Must select a tool call, cannot return empty results
2. Must use the above JSON format, do not add any other content
3. All required parameters must be provided
4. Time parameters must use standard format (e.g.: 2025年, 2025年7月, 2025年7月8日)

Please return the tool call JSON immediately, do not add any explanatory text."""

    def _call_llm_with_retry(self, user_prompt: str, system_prompt: str, tools: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Call LLM with retry"""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                return self._call_llm_with_tools(user_prompt, system_prompt, tools)
            except Exception as e:
                logger.warning(
                    f"LLM call failed (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt == max_retries - 1:
                    return None
                time.sleep(1)  # Brief delay before retry

    def _handle_search_fallback(self, query: str, top_k: int) -> Dict[str, Any]:
        """Search fallback handling"""
        logger.warning("LLM call failed, using default preliminary search")
        results = self.preliminary_memory.search_memories_by_query(
            query, top_k)
        return {
            "success": True,
            "search_type": "fallback_preliminary_search",
            "query": query,
            "top_k": top_k,
            "results": results,
            "timestamp": time.time(),
            "error": "LLM call failed, using fallback search"
        }

    def _try_direct_time_search(self, query: str, top_k: int) -> Optional[Dict[str, Any]]:
        """Try direct time search (pre-check)"""
        try:
            import re

            # More strict time pattern matching to ensure query mainly contains time information
            time_patterns = [
                # Full date: 2025年7月8日
                r'^(\d{4}年\d{1,2}月\d{1,2}日)$',
                # Year-month: 2025年7月
                r'^(\d{4}年\d{1,2}月)$',
                # Year: 2025年
                r'^(\d{4}年)$',
                # Fuzzy time expressions (as standalone queries)
                r'^(今天|昨天|明天|这个月|上个月|今年|去年|前年)$'
            ]

            # Check if query mainly contains time information
            query_stripped = query.strip()
            
            # First check if it's a pure time query
            for pattern in time_patterns:
                match = re.search(pattern, query_stripped)
                if match:
                    time_expr = match.group(1)
                    logger.info(f"Detected pure time expression: {time_expr}")
                    
                    # For pure time queries, default to memoir search
                    normalized_time = self._normalize_time_expression(time_expr)
                    try:
                        results = self.memoir_manager.search_memoir_by_time(normalized_time, top_k)
                        return {
                            "success": True,
                            "search_type": "search_memoir_memories_by_time",
                            "query": query,
                            "top_k": top_k,
                            "results": results,
                            "timestamp": time.time(),
                            "detected_time": time_expr,
                            "normalized_time": normalized_time
                        }
                    except Exception as e:
                        logger.warning(f"Direct memoir time search failed: {e}")
                        break

            # Check if query contains embedded time information (as part of the query)
            embedded_time_patterns = [
                # Embedded full date
                r'(\d{4}年\d{1,2}月\d{1,2}日)',
                # Embedded year-month
                r'(\d{4}年\d{1,2}月)',
                # Embedded year
                r'(\d{4}年)',
                # Embedded fuzzy time
                r'(今天|昨天|明天|这个月|上个月|今年|去年|前年)'
            ]

            for pattern in embedded_time_patterns:
                match = re.search(pattern, query_stripped)
                if match:
                    time_expr = match.group(1)
                    logger.info(f"Detected embedded time expression: {time_expr} in query: {query}")

                    # Check if query contains general vocabulary, if so use LLM search
                    summary_keywords = ['总结', '发展', '趋势', '回顾', '概况', '整体', '模式', '进展', '经验', '分享', '内容', '分析']
                    if any(keyword in query for keyword in summary_keywords):
                        logger.info(f"Query contains general vocabulary, using LLM search: {query}")
                        return None

                    # Check if query asks for specific content
                    specific_keywords = ['具体', '详细', '内容', '记录', '原始', '完整', '做了什么', '有什么']
                    if any(keyword in query for keyword in specific_keywords):
                        logger.info(f"Query asks for specific content, using preliminary time search: {query}")
                        normalized_time = self._normalize_time_expression(time_expr)
                        try:
                            results = self.preliminary_memory.search_memories_by_time(normalized_time, top_k)
                            return {
                                "success": True,
                                "search_type": "search_preliminary_memories_by_time",
                                "query": query,
                                "top_k": top_k,
                                "results": results,
                                "timestamp": time.time(),
                                "detected_time": time_expr,
                                "normalized_time": normalized_time
                            }
                        except Exception as e:
                            logger.warning(f"Direct preliminary time search failed: {e}")
                            break

                    # For other queries containing time information, default to memoir search
                    logger.info(f"Time query defaulting to memoir search: {query}")
                    normalized_time = self._normalize_time_expression(time_expr)
                    try:
                        results = self.memoir_manager.search_memoir_by_time(normalized_time, top_k)
                        return {
                            "success": True,
                            "search_type": "search_memoir_memories_by_time",
                            "query": query,
                            "top_k": top_k,
                            "results": results,
                            "timestamp": time.time(),
                            "detected_time": time_expr,
                            "normalized_time": normalized_time
                        }
                    except Exception as e:
                        logger.warning(f"Direct memoir time search failed: {e}")
                        break

            # If no time information detected, return None for LLM processing
            logger.info(f"No time information detected in query: {query}")
            return None

        except Exception as e:
            logger.warning(f"Time pre-check failed: {e}")
            return None

    @cache_result(ttl=300)  # Cache for 5 minutes
    def _create_optimized_search_system_prompt(self) -> str:
        """Create optimized search system prompt"""
        return """You are an intelligent memory search assistant. Your task is to analyze user queries, select the most appropriate search method and return tool calls in standard JSON format.

## Core Decision Rules

### 1. Time Query Priority Judgment
**Time query characteristics:**
- Contains specific dates: 2025年, 2025年7月, 2025年7月8日
- Contains fuzzy time: today, yesterday, tomorrow, this month, last month, this year, last year
- Contains time-related vocabulary: date, time, when, which day

**Time query selection:**
- If query contains time information → use time search tools
- **Key judgment criteria**: choose preliminary or memoir time search based on query content

### 2. Time Standardization Rules (Important!)

**Fuzzy time expressions must be standardized to specific times:**
- "today" → "2025年7月11日" (current date)
- "yesterday" → "2025年7月10日" (current date - 1 day)
- "tomorrow" → "2025年7月12日" (current date + 1 day)
- "this month" → "2025年7月" (current year-month)
- "last month" → "2025年6月" (current year-month - 1 month)
- "this year" → "2025年" (current year)
- "last year" → "2024年" (current year - 1 year)
- "year before last" → "2023年" (current year - 2 years)
- "this week" → "2025年7月" (current year-month)
- "last week" → "2025年7月" (current year-month, simplified processing)

**Important:** All fuzzy time expressions must be converted to standard format when passed to tools!

### 3. Time Search Type Judgment (Important!)

#### 3.1 Use preliminary time search when:
- Asking what was done specifically: **"做了什么", "有什么", "具体内容", "详细记录"**
- Asking about specific events, conversations, notes
- Asking about specific people, places, technical details
- Asking about what happened at specific time points

#### 3.2 Use memoir time search when:
- Asking for summary content: **"总结", "发展", "趋势", "回顾", "概况"**
- Asking about overall situation: **"整体", "模式", "进展", "状况"**
- Asking for general descriptions: **"怎么样", "如何", "什么情况"**
- Asking about themes, trends, patterns within time periods

### 4. Content Type Judgment (CRITICAL!)

**Specific content query characteristics:**
- Specific people names, place names, technical terms
- Specific conversations, notes, thoughts
- Specific events, projects, tasks
- Contains "什么", "如何", "为什么" and other specific questions
- Asking for specific details, implementations, or concrete information

**General/Summary content query characteristics (USE MEMOIR):**
- Themes, trends, patterns, developments
- Summary content, overall situation, experiences
- Contains "趋势", "总结", "整体", "模式", "发展", "经验", "分享", "内容" and other vocabulary
- Asking for overview, trends, patterns, or general experiences
- Questions about "how things are going", "what's the trend", "what's the experience"

### 5. Search Method Selection Matrix

| Query Type | Specific Content | General/Summary Content |
|-----------|-----------------|-------------------------|
| Time Query | search_preliminary_memories_by_time | search_memoir_memories_by_time |
| Content Query | search_preliminary_memories_by_query | search_memoir_memories_by_query |

## Decision Process

1. **Step 1: Check if it contains time information**
   - Yes → proceed to step 2
   - No → proceed to step 3

2. **Step 2: Judge time query type**
   - Asking for specific content (做了什么, 有什么, 具体记录) → search_preliminary_memories_by_time
   - Asking for summary content (总结, 发展, 趋势, 回顾) → search_memoir_memories_by_time
   - Unclear → judge based on other keywords

3. **Step 3: Judge content type**
   - Specific content → choose preliminary search
   - General/Summary content → choose memoir search

## Example Analysis

**Time query examples:**
- "2025年7月8日做了什么" → search_preliminary_memories_by_time (specific content)
- "2025年的学习总结" → search_memoir_memories_by_time (summary content)
- "今天的工作内容" → search_preliminary_memories_by_time (specific content)
- "这个月的发展趋势" → search_memoir_memories_by_time (trend content)
- "2024年的技术发展" → search_memoir_memories_by_time (development content)
- "去年做了什么" → search_preliminary_memories_by_time (specific content, target_date: "2024年")
- "去年的总结" → search_memoir_memories_by_time (summary content, target_date: "2024年")

**Content query examples:**
- "Python编程" → search_preliminary_memories_by_query (specific technology)
- "机器学习" → search_preliminary_memories_by_query (specific technology)
- "Python编程的发展趋势" → search_memoir_memories_by_query (trend content)
- "机器学习项目经验" → search_memoir_memories_by_query (experience content)
- "技术会议分享内容" → search_memoir_memories_by_query (summary content)
- "学习趋势" → search_memoir_memories_by_query (trend content)
- "技术发展模式" → search_memoir_memories_by_query (pattern content)
- "项目经验总结" → search_memoir_memories_by_query (summary content)

## Parameter Requirements

**Time search parameters:**
- target_date: Must be in standard format (2025年, 2025年7月, 2025年7月8日)
- top_k: Number of results to return

**Content search parameters:**
- query: Extract core keywords from user query
- top_k: Number of results to return

## Important Reminders

1. **Must select a tool call** - cannot return empty results
2. **Parameters must be complete and correct** - all required parameters must be provided
3. **Time format must be standardized** - use standard Chinese time format
4. **Prioritize the most matching search method** - choose the most appropriate tool based on query characteristics
5. **Key to time queries is judging content type** - specific content uses preliminary, general content uses memoir
6. **For uncertain queries, prioritize preliminary search** - safer choice
7. **Fuzzy time expressions must be standardized** - "去年"→"2024年", "今天"→"2025年7月11日", etc.
8. **General/Summary queries ALWAYS use memoir** - if query asks for trends, experiences, patterns, or summaries, use memoir search

## Output Format

You must return strictly in the following JSON format, do not add any other content:

```json
{
  "name": "tool_name",
  "arguments": {
    "parameter_name": "parameter_value"
  }
}
```

**Note:** This is the standard format for tool calls, must be returned exactly in this format, do not add any explanatory text or additional formatting."""

    @cache_result(ttl=60)  # Cache for 1 minute
    def _define_search_tools(self) -> List[Dict[str, Any]]:
        """Define search tools"""
        return [
            {
                "type": "function",
                "function": {
                    "name": "search_preliminary_memories_by_query",
                    "description": "Search preliminary memories (raw text memories). Suitable for: searching specific content, person names, place names, technical terms, specific conversations, notes, thoughts. Use when the user asks for specific details.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Search query content, extract core keywords from the user's question"
                            },
                            "top_k": {
                                "type": "integer",
                                "description": "Number of results to return",
                                "default": 5
                            }
                        },
                        "required": ["query"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "search_preliminary_memories_by_time",
                    "description": "Search preliminary memories by time. Suitable for: when the user mentions specific dates, years, or months, or asks for memories within specific time periods.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "target_date": {
                                "type": "string",
                                "description": "Target date, must be in standard format: 2025年, 2025年7月, 2025年7月8日"
                            },
                            "top_k": {
                                "type": "integer",
                                "description": "Number of results to return",
                                "default": 5
                            }
                        },
                        "required": ["target_date"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "search_memoir_memories_by_query",
                    "description": "Search advanced memories (memoir summary memories). Suitable for: searching themes, trends, patterns, summary content. Use when the user asks for general content.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Search query content, extract core keywords from the user's question"
                            },
                            "top_k": {
                                "type": "integer",
                                "description": "Number of results to return",
                                "default": 5
                            }
                        },
                        "required": ["query"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "search_memoir_memories_by_time",
                    "description": "Search advanced memories by time. Suitable for: when the user asks for summaries, trends, or patterns within specific time periods.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "target_date": {
                                "type": "string",
                                "description": "Target date, must be in standard format: 2025年, 2025年7月, 2025年7月8日"
                            },
                            "top_k": {
                                "type": "integer",
                                "description": "Number of results to return",
                                "default": 5
                            }
                        },
                        "required": ["target_date"]
                    }
                }
            }
        ]

    def _call_llm_with_tools(self, prompt: str, system_prompt: str, tools: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Call LLM with tools"""
        try:
            # Use the tool call method of the LLM client
            return self.llm_client.generate_response_with_tools(
                prompt=prompt,
                system_prompt=system_prompt,
                tools=tools,
                temperature=0.0,
                max_tokens=1000,
                tool_choice="auto"
            )

        except Exception as e:
            logger.error(f"Failed to call LLM with tools: {e}")
            raise LLMClientError(f"Failed to call LLM with tools: {e}")

    def _process_search_results(self, tool_calls: List[ToolCall], query: str, top_k: int) -> Dict[str, Any]:
        """Process search results"""
        try:
            if tool_calls:
                # Execute the first tool call
                tool_call = tool_calls[0]
                logger.info(
                    f"Executing tool call: {tool_call.name} - {tool_call.arguments}")

                result = self._execute_search_tool(tool_call)

                return {
                    "success": True,
                    "search_type": tool_call.name,
                    "query": query,
                    "top_k": top_k,
                    "results": result,
                    "tool_arguments": tool_call.arguments,
                    "timestamp": time.time()
                }
            else:
                # If there is no tool call, use the default preliminary search
                logger.warning("No tool calls detected, using default preliminary search")
                logger.info(f"Query: '{query}', top_k: {top_k}")

                results = self.preliminary_memory.search_memories_by_query(
                    query, top_k)

                return {
                    "success": True,
                    "search_type": "default_preliminary_search",
                    "query": query,
                    "top_k": top_k,
                    "results": results,
                    "timestamp": time.time(),
                    "warning": "No tool calls detected, using fallback search method"
                }
        except Exception as e:
            logger.error(f"Failed to process search results: {e}")
            # Fallback to preliminary search
            try:
                results = self.preliminary_memory.search_memories_by_query(
                    query, top_k)
                return {
                    "success": True,
                    "search_type": "error_fallback_preliminary_search",
                    "query": query,
                    "top_k": top_k,
                    "results": results,
                    "timestamp": time.time(),
                    "error": str(e),
                    "warning": "Tool call execution failed, using fallback search method"
                }
            except Exception as fallback_error:
                logger.error(f"Fallback search also failed: {fallback_error}")
                return {
                    "success": False,
                    "search_type": "error",
                    "query": query,
                    "top_k": top_k,
                    "results": [],
                    "timestamp": time.time(),
                    "error": f"Search failed: {e}, fallback search also failed: {fallback_error}"
                }

    def _execute_search_tool(self, tool_call: ToolCall) -> List[Dict[str, Any]]:
        """Execute search tool"""
        try:
            name = tool_call.name
            args = tool_call.arguments

            if name == "search_preliminary_memories_by_query":
                return self.preliminary_memory.search_memories_by_query(
                    args.get("query", ""),
                    args.get("top_k", 5)
                )
            elif name == "search_preliminary_memories_by_time":
                target_date = args.get("target_date", "")
                # Normalize time expression before searching
                normalized_date = self._normalize_time_expression(target_date)
                return self.preliminary_memory.search_memories_by_time(
                    normalized_date,
                    args.get("top_k", 5)
                )
            elif name == "search_memoir_memories_by_query":
                return self.memoir_manager.search_memoir_by_query(
                    args.get("query", ""),
                    args.get("top_k", 5)
                )
            elif name == "search_memoir_memories_by_time":
                target_date = args.get("target_date", "")
                # Normalize time expression before searching
                normalized_date = self._normalize_time_expression(target_date)
                return self.memoir_manager.search_memoir_by_time(
                    normalized_date,
                    args.get("top_k", 5)
                )
            else:
                logger.warning(f"Unknown search tool: {name}")
                return []

        except Exception as e:
            logger.error(f"Failed to execute search tool: {e}")
            return []

    def _normalize_time_expression(self, time_expr: str) -> str:
        """Normalize time expressions"""
        try:
            from datetime import datetime, timedelta

            time_expr = time_expr.strip()

            # Handle "today"
            if time_expr in ["今天", "today"]:
                today = datetime.now()
                result = f"{today.year}年{today.month}月{today.day}日"
                return result.replace(' ', '').replace('\n', '').strip()

            # Handle "yesterday"
            elif time_expr in ["昨天", "yesterday"]:
                yesterday = datetime.now() - timedelta(days=1)
                result = f"{yesterday.year}年{yesterday.month}月{yesterday.day}日"
                return result.replace(' ', '').replace('\n', '').strip()

            # Handle "tomorrow"
            elif time_expr in ["明天", "tomorrow"]:
                tomorrow = datetime.now() + timedelta(days=1)
                result = f"{tomorrow.year}年{tomorrow.month}月{tomorrow.day}日"
                return result.replace(' ', '').replace('\n', '').strip()

            # Handle "this week"
            elif time_expr in ["这周", "本周", "this week"]:
                today = datetime.now()
                result = f"{today.year}年{today.month}月"
                return result.replace(' ', '').replace('\n', '').strip()

            # Handle "last week"
            elif time_expr in ["上周", "last week"]:
                last_week = datetime.now() - timedelta(weeks=1)
                result = f"{last_week.year}年{last_week.month}月"
                return result.replace(' ', '').replace('\n', '').strip()

            # Handle "this month"
            elif time_expr in ["这个月", "本月", "this month"]:
                today = datetime.now()
                result = f"{today.year}年{today.month}月"
                return result.replace(' ', '').replace('\n', '').strip()

            # Handle "last month"
            elif time_expr in ["上个月", "last month"]:
                today = datetime.now()
                if today.month == 1:
                    result = f"{today.year-1}年12月"
                else:
                    result = f"{today.year}年{today.month-1}月"
                return result.replace(' ', '').replace('\n', '').strip()

            # Handle "this year"
            elif time_expr in ["今年", "this year"]:
                today = datetime.now()
                result = f"{today.year}年"
                return result.replace(' ', '').replace('\n', '').strip()

            # Handle "last year"
            elif time_expr in ["去年", "last year"]:
                today = datetime.now()
                result = f"{today.year-1}年"
                return result.replace(' ', '').replace('\n', '').strip()

            # Handle "year before last"
            elif time_expr in ["前年", "year before last"]:
                today = datetime.now()
                result = f"{today.year-2}年"
                return result.replace(' ', '').replace('\n', '').strip()

            # If already in standard format, return directly
            else:
                return time_expr.replace(' ', '').replace('\n', '').strip()

        except Exception as e:
            logger.warning(f"Failed to normalize time expression: {e}, original expression: {time_expr}")
            return time_expr.replace(' ', '').replace('\n', '').strip()

    # Proxy methods - directly call submodules
    def search_preliminary_memories_by_query(self, query: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """Search preliminary memories"""
        if top_k is None:
            top_k = self.default_top_k
        return self.preliminary_memory.search_memories_by_query(query, top_k)

    def search_preliminary_memories_by_time(self, target_date: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """Search preliminary memories by time"""
        if top_k is None:
            top_k = self.default_top_k
        return self.preliminary_memory.search_memories_by_time(target_date, top_k)

    def search_memoir_memories_by_query(self, query: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """Search advanced memories"""
        if top_k is None:
            top_k = self.default_top_k
        return self.memoir_manager.search_memoir_by_query(query, top_k)

    def search_memoir_memories_by_time(self, target_date: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """Search advanced memories by time"""
        if top_k is None:
            top_k = self.default_top_k
        return self.memoir_manager.search_memoir_by_time(target_date, top_k)

    def _estimate_wait_time(self) -> float:
        """
        Estimate wait time (seconds)
        
        Returns:
            Estimated wait time
        """
        if self.async_stats["processed_requests"] == 0:
            return 5.0  # Default estimate 5 seconds
        
        # Estimate based on average processing time and queue size
        avg_processing_time = self.async_stats["average_processing_time"]
        queue_size = self.async_stats["queue_size"]
        worker_count = len([w for w in self.workers if w.is_alive()])
        
        if worker_count == 0:
            return float('inf')  # No worker threads
        
        # Estimate formula: queue size * average processing time / worker count
        estimated_time = (queue_size * avg_processing_time) / worker_count
        
        # Limit to a reasonable range
        return min(max(estimated_time, 1.0), 300.0)  # 1 second to 5 minutes
