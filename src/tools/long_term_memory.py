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

Long-term Memory Management Module
Implements AGIAgent's long-term memory functionality based on the mem project architecture.
"""

import os
import sys
import json
import time
import hashlib
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
from .print_system import print_current, print_system, print_error, print_debug

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    # Correct import for mem modules
    from mem.src.core.memory_manager import MemManagerAgent
    from mem.src.models.memory_cell import MemCell
    from mem.src.utils.logger import get_logger

    _MEM_AVAILABLE = True
except ImportError as e:
    print_debug(f"Warning: Failed to import mem modules: {e}")
    # Fallback to simplified implementation
    MemManagerAgent = None
    MemCell = None
    get_logger = lambda name: None
    _MEM_AVAILABLE = False




class LongTermMemoryManager:
    """
    Long-term Memory Manager
    Integrates mem project's memory management into AGIAgent
    """

    def __init__(self, workspace_root: str, memory_config_file: str = None):
        """
        Initialize the long-term memory manager

        Args:
            workspace_root: Root directory of the workspace
            memory_config_file: Path to memory config file
        """
        self.workspace_root = workspace_root

        # Store long-term memory in the project root for global sharing
        # Get project root (AGIAgent directory)
        # From src/tools/long_term_memory.py -> AGIAgent/
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(current_dir))
        self.memory_dir = os.path.join(project_root, "long_term_memory")
        os.makedirs(self.memory_dir, exist_ok=True)

        # Set memory config file path
        if memory_config_file is None:
            # Use AGIAgent's config_memory.txt file
            memory_config_file = os.path.join(project_root, "config", "config_memory.txt")
            # Fallback to config.txt if config_memory.txt does not exist
            if not os.path.exists(memory_config_file):
                memory_config_file = os.path.join(project_root, "config", "config.txt")

        self.memory_config_file = memory_config_file
        self.initialized = False
        self.memory_manager = None

        # Try to initialize memory manager
        self._initialize_memory_manager()

    def _initialize_memory_manager(self):
        """Initialize the memory manager"""
        try:
            # Check if long-term memory is enabled via environment variable
            if os.environ.get('AGIBOT_LONG_TERM_MEMORY', '').lower() in ('false', '0', 'no', 'off'):
                print_system("⚠️ Long-term memory is disabled via environment variable AGIBOT_LONG_TERM_MEMORY")
                self.initialized = False
                return
            
            if MemManagerAgent is None:
                print_system("⚠️ mem module not properly imported, long-term memory will use a simplified implementation")
                self.initialized = False
                return

            # Create memory manager instance
            # Pass None to use main config.txt with memory config overlay
            self.memory_manager = MemManagerAgent(
                storage_path=self.memory_dir,
                config_file=None
            )

            # Perform health check
            health_status = self.memory_manager.health_check()
            if health_status.get("success", False):
                self.initialized = True
                print_system("✅ Long-term memory manager initialized successfully")
            else:
                print_system(f"⚠️ Long-term memory manager health check failed: {health_status.get('error', 'Unknown error')}")
                self.initialized = False

        except Exception as e:
            print_error(f"❌ Failed to initialize long-term memory manager: {e}")
            self.initialized = False

    def is_available(self) -> bool:
        """Check if long-term memory is available"""
        return self.initialized and self.memory_manager is not None

    def store_task_memory(self, task_prompt: str, task_result: str, execution_metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Store task memory

        Args:
            task_prompt: Task prompt
            task_result: Task result
            execution_metadata: Execution metadata

        Returns:
            Storage result
        """
        if not self.is_available():
            return {
                "status": "failed",
                "error": "Long-term memory is not available",
                "fallback_used": True
            }

        try:
            # Build memory text
            memory_text = self._build_memory_text(task_prompt, task_result, execution_metadata)

            # Write memory
            result = self.memory_manager.write_memory_auto(memory_text)

            if result.get("success", False):
                #print_current(f"✅ Task memory stored: {result.get('action', 'unknown')} (ID: {result.get('mem_id', 'unknown')})")
                return {
                    "status": "success",
                    "action": result.get("action"),
                    "memory_id": result.get("mem_id"),
                    "similarity_score": result.get("similarity_score"),
                    "was_updated": result.get("action") == "updated"
                }
            else:
                print_debug(f"❌ Failed to store task memory: {result.get('error', 'Unknown error')}")
                return {
                    "status": "failed",
                    "error": result.get("error", "Storage failed")
                }

        except Exception as e:
            print_debug(f"❌ Exception occurred while storing task memory: {e}")
            return {
                "status": "failed",
                "error": str(e)
            }

    def recall_relevant_memories(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Recall relevant memories

        Args:
            query: Query string
            top_k: Number of memories to return (default 5)

        Returns:
            Recall result, including formatted memory list
        """
        if not self.is_available():
            return {
                "status": "failed",
                "error": "Long-term memory is not available",
                "memories": []
            }

        try:
            # Perform memory search
            result = self.memory_manager.read_memory_auto(query, top_k=top_k)

            if result.get("success", False):
                memories = result.get("results", [])  # Use 'results' instead of 'memories'

                formatted_memories = []
                for memory in memories:
                    formatted_memory = self._format_memory_for_context(memory)
                    if formatted_memory:
                        formatted_memories.append(formatted_memory)

                return {
                    "status": "success",
                    "memories": formatted_memories,
                    "total_found": len(memories),
                    "search_method": result.get("search_method", "auto")
                }
            else:
                return {
                    "status": "failed",
                    "error": result.get("error", "Search failed"),
                    "memories": []
                }

        except Exception as e:
            print_current(f"❌ Exception occurred while recalling memories: {e}")
            return {
                "status": "failed",
                "error": str(e),
                "memories": []
            }

    def search_memories_by_time(self, time_query: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Search memories by time

        Args:
            time_query: Time query (e.g. "yesterday", "last week", "2024-01-01")
            top_k: Number of memories to return

        Returns:
            Search result
        """
        if not self.is_available():
            return {
                "status": "failed",
                "error": "Long-term memory is not available",
                "memories": []
            }

        try:
            # Use memory manager's time search function if available
            if hasattr(self.memory_manager, 'search_preliminary_memories_by_time'):
                memories = self.memory_manager.search_preliminary_memories_by_time(
                    target_date=time_query,
                    top_k=top_k
                )

                formatted_memories = []
                for memory in memories:
                    formatted_memory = self._format_memory_for_context(memory)
                    if formatted_memory:
                        formatted_memories.append(formatted_memory)

                return {
                    "status": "success",
                    "memories": formatted_memories,
                    "time_query": time_query,
                    "total_found": len(memories)
                }
            else:
                # Fallback to normal search
                return self.recall_relevant_memories(f"Time: {time_query}", top_k)

        except Exception as e:
            print_current(f"❌ Exception occurred while searching memories by time: {e}")
            return {
                "status": "failed",
                "error": str(e),
                "memories": []
            }

    def get_memory_stats(self) -> Dict[str, Any]:
        """
        Get memory statistics

        Returns:
            Statistics
        """
        if not self.is_available():
            return {
                "status": "failed",
                "error": "Long-term memory is not available"
            }

        try:
            # Get system status
            status = self.memory_manager.get_status_auto(include_details=True)

            if status.get("success", False):
                stats = status.get("status", {})
                return {
                    "status": "success",
                    "total_memories": stats.get("total_memories", 0),
                    "preliminary_memories": stats.get("preliminary_memories", 0),
                    "memoir_entries": stats.get("memoir_entries", 0),
                    "storage_path": self.memory_dir,
                    "last_update": stats.get("last_update"),
                    "health_status": stats.get("health_status", "unknown")
                }
            else:
                return {
                    "status": "failed",
                    "error": status.get("error", "Failed to get status")
                }

        except Exception as e:
            print_debug(f"❌ Exception occurred while getting memory statistics: {e}")
            return {
                "status": "failed",
                "error": str(e)
            }

    def _build_memory_text(self, task_prompt: str, task_result: str, metadata: Dict[str, Any] = None) -> str:
        """
        Build memory text

        Args:
            task_prompt: Task prompt
            task_result: Task result
            metadata: Metadata

        Returns:
            Formatted memory text
        """
        lines = []

        # Add timestamp
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        lines.append(f"Time: {timestamp}")
        lines.append("")

        # Add task prompt
        lines.append("Task Requirement:")
        lines.append(task_prompt.strip())
        lines.append("")

        # Add task result
        lines.append("Execution Result:")
        # Truncate overly long result
        result_text = task_result.strip()
        if len(result_text) > 2000:
            result_text = result_text[:2000] + "...[Result Truncated]"
        lines.append(result_text)
        lines.append("")

        # Add metadata
        if metadata:
            lines.append("Execution Info:")
            for key, value in metadata.items():
                if key in ["execution_time", "tool_calls_count", "model_used", "success"]:
                    lines.append(f"- {key}: {value}")
            lines.append("")

        return "\n".join(lines)

    def _format_memory_for_context(self, memory: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Format memory for context

        Args:
            memory: Raw memory data

        Returns:
            Formatted memory data
        """
        try:
            summary = None
            create_time = None
            mem_id = "unknown"

            # Handle different memory formats
            if "mem_cell" in memory:
                mem_cell = memory["mem_cell"]

                if hasattr(mem_cell, 'summary'):
                    summary = mem_cell.summary
                    create_time = getattr(mem_cell, 'create_time', None)
                    mem_id = getattr(mem_cell, 'mem_id', 'unknown')
                else:
                    # Dictionary format
                    summary = mem_cell.get("summary", "")
                    create_time = mem_cell.get("create_time")
                    mem_id = mem_cell.get("mem_id", "unknown")
            else:
                # Direct format
                summary = memory.get("summary", "")
                create_time = memory.get("create_time")
                mem_id = memory.get("mem_id", "unknown")

            # Enhanced summary check to avoid filtering out empty strings
            if not summary or (isinstance(summary, str) and summary.strip() == ""):
                # Try to get content from other fields
                alternative_content = None
                if "mem_cell" in memory:
                    mem_cell = memory["mem_cell"]
                    # Try to get from text field
                    if hasattr(mem_cell, 'text') and mem_cell.text:
                        if isinstance(mem_cell.text, list) and mem_cell.text:
                            alternative_content = mem_cell.text[0][:200] + "..." if len(mem_cell.text[0]) > 200 else mem_cell.text[0]
                        elif isinstance(mem_cell.text, str):
                            alternative_content = mem_cell.text[:200] + "..." if len(mem_cell.text) > 200 else mem_cell.text

                if alternative_content:
                    summary = f"[Extracted from text] {alternative_content}"
                else:
                    return None

            # Format time
            time_str = "Unknown time"
            if create_time:
                try:
                    if isinstance(create_time, (int, float)):
                        time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(create_time))
                    else:
                        time_str = str(create_time)
                except:
                    time_str = str(create_time)

            formatted_result = {
                "memory_id": mem_id,
                "summary": summary,
                "create_time": time_str,
                "similarity_score": memory.get("similarity_score", 0.0)
            }

            return formatted_result

        except Exception as e:
            print_debug(f"❌ Error formatting memory: {e}")
            return None

    def cleanup(self):
        """Clean up resources"""
        try:
            if self.memory_manager:
                # If memory manager has a cleanup method, call it
                if hasattr(self.memory_manager, 'cleanup'):
                    self.memory_manager.cleanup()
        except Exception as e:
            pass


class LongTermMemoryTools:
    """
    Long-term Memory Tools
    Provides tool methods for tool_executor
    """

    def __init__(self, workspace_root: str, memory_config_file: str = None):
        """
        Initialize long-term memory tools

        Args:
            workspace_root: Root directory of the workspace
            memory_config_file: Path to memory config file
        """
        self.memory_manager = LongTermMemoryManager(workspace_root, memory_config_file)

    def recall_memories(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Recall relevant memories tool

        Args:
            query: Query string
            top_k: Number of memories to return

        Returns:
            Recall result
        """
        try:
            result = self.memory_manager.recall_relevant_memories(query, top_k)

            if result.get("success", False):
                memories = result.get("memories", [])

                # Format output
                if not memories:
                    return {
                        "status": "success",  # Use 'success' field to match tool_executor logic
                        "message": "No relevant memories found",
                        "memories_count": 0,
                        "memories": []
                    }

                formatted_output = []
                formatted_output.append(f"Found {len(memories)} relevant memories:")
                formatted_output.append("")

                for i, memory in enumerate(memories, 1):
                    # Memories are already formatted, use correct field names
                    formatted_output.append(f"Memory {i}:")
                    formatted_output.append(f"  ID: {memory.get('memory_id', 'Unknown')}")
                    formatted_output.append(f"  Time: {memory.get('create_time', 'Unknown')}")
                    formatted_output.append(f"  Similarity: {memory.get('similarity_score', 0.0):.3f}")
                    formatted_output.append(f"  Content: {memory.get('summary', 'No summary')}")
                    formatted_output.append("")

                final_message = "\n".join(formatted_output)

                return {
                    "status": "success",
                    "message": final_message,
                    "memories_count": len(memories),
                    "memories": memories,
                    "search_method": result.get("search_method", "auto")
                }
            else:
                return {
                    "status": "failed",
                    "message": f"Memory recall failed: {result.get('error', 'Unknown error')}",
                    "memories_count": 0,
                    "memories": []
                }

        except Exception as e:
            print_current(f"❌ Exception in recall_memories tool: {e}")
            return {
                "status": "failed",
                "message": f"Exception occurred while recalling memories: {e}",
                "memories_count": 0,
                "memories": []
            }

    def recall_memories_by_time(self, time_query: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Recall memories by time tool

        Args:
            time_query: Time query
            top_k: Number of memories to return

        Returns:
            Recall result
        """
        try:
            result = self.memory_manager.search_memories_by_time(time_query, top_k)

            if result.get("success", False):
                memories = result.get("memories", [])

                if not memories:
                    return {
                        "status": "success",
                        "message": f"No relevant memories found for time '{time_query}'",
                        "memories_count": 0,
                        "memories": []
                    }

                formatted_output = []
                formatted_output.append(f"Found {len(memories)} memories for time '{time_query}':")
                formatted_output.append("")

                for i, memory in enumerate(memories, 1):
                    formatted_output.append(f"Memory {i}:")
                    formatted_output.append(f"  Time: {memory.get('create_time', 'Unknown')}")
                    formatted_output.append(f"  Content: {memory.get('summary', 'No summary')}")
                    formatted_output.append("")

                return {
                    "status": "success",
                    "message": "\n".join(formatted_output),
                    "memories_count": len(memories),
                    "memories": memories,
                    "time_query": time_query
                }
            else:
                return {
                    "status": "failed",
                    "message": f"Time memory search failed: {result.get('error', 'Unknown error')}",
                    "memories_count": 0,
                    "memories": []
                }

        except Exception as e:
            return {
                "status": "failed",
                "message": f"Exception occurred while recalling memories by time: {e}",
                "memories_count": 0,
                "memories": []
            }

    def get_memory_summary(self) -> Dict[str, Any]:
        """
        Get memory system summary

        Returns:
            Memory system summary
        """
        try:
            result = self.memory_manager.get_memory_stats()

            if result.get("success", False):
                stats = result

                summary_lines = []
                summary_lines.append("📊 Long-term Memory System Status:")
                summary_lines.append("")
                summary_lines.append(f"  Total memories: {stats.get('total_memories', 0)}")
                summary_lines.append(f"  Preliminary memories: {stats.get('preliminary_memories', 0)}")
                summary_lines.append(f"  Memoir entries: {stats.get('memoir_entries', 0)}")
                summary_lines.append(f"  Storage path: {stats.get('storage_path', 'Unknown')}")
                summary_lines.append(f"  Health status: {stats.get('health_status', 'Unknown')}")

                if stats.get('last_update'):
                    summary_lines.append(f"  Last update: {stats.get('last_update')}")

                return {
                    "status": "success",
                    "message": "\n".join(summary_lines),
                    "stats": stats
                }
            else:
                return {
                    "status": "error",
                    "message": f"Failed to get memory statistics: {result.get('error', 'Unknown error')}",
                    "stats": {}
                }

        except Exception as e:
            return {
                "status": "error",
                "message": f"Exception occurred while getting memory summary: {e}",
                "stats": {}
            }

    def cleanup(self):
        """Clean up resources"""
        if self.memory_manager:
            self.memory_manager.cleanup()