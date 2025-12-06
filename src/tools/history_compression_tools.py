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

from typing import Dict, Any, List, Optional
from datetime import datetime
from .print_system import print_current, print_debug


class HistoryCompressionTools:
    """Tools for compressing conversation history on demand"""
    
    def __init__(self, tool_executor=None):
        """Initialize history compression tools with reference to tool executor"""
        self.tool_executor = tool_executor
    
    def compress_history(self, keep_recent_rounds: int = 2, **kwargs) -> Dict[str, Any]:
        """
        Compress conversation history using simple compression to reduce context length.
        This tool allows the model to actively request history compression when needed.
        
        Args:
            keep_recent_rounds: Number of recent rounds to keep uncompressed (default: 2)
            
        Returns:
            Dictionary containing compression result and statistics
        """
        # Ignore additional parameters
        if kwargs:
            print_current(f"⚠️ Ignoring additional parameters: {list(kwargs.keys())}")
        
        if not self.tool_executor:
            return {
                "status": "error",
                "message": "Tool executor not available. This tool can only be used during task execution."
            }
        
        # Get current task history from executor
        if not hasattr(self.tool_executor, '_current_task_history') or not self.tool_executor._current_task_history:
            return {
                "status": "error",
                "message": "No task history available. History compression can only be performed during active task execution."
            }
        
        task_history = self.tool_executor._current_task_history
        
        # Filter history records that have results (similar to executor logic)
        history_for_llm = [record for record in task_history 
                          if "result" in record or "error" in record]
        
        if len(history_for_llm) <= keep_recent_rounds:
            return {
                "status": "skipped",
                "message": f"History is too short ({len(history_for_llm)} records). Need more than {keep_recent_rounds} records to compress.",
                "current_records": len(history_for_llm),
                "keep_recent_rounds": keep_recent_rounds
            }
        
        # Split history: older records to summarize, recent records to keep
        records_to_summarize = history_for_llm[:-keep_recent_rounds] if len(history_for_llm) > keep_recent_rounds else []
        recent_records = history_for_llm[-keep_recent_rounds:] if len(history_for_llm) > keep_recent_rounds else history_for_llm
        
        if not records_to_summarize:
            return {
                "status": "skipped",
                "message": "No records to compress. All records are within the keep_recent_rounds range.",
                "current_records": len(history_for_llm),
                "keep_recent_rounds": keep_recent_rounds
            }
        
        # Calculate lengths
        records_to_summarize_length = sum(len(str(record.get("result", ""))) 
                                          for record in records_to_summarize)
        recent_records_length = sum(len(str(record.get("result", ""))) 
                                   for record in recent_records)
        total_history_length = records_to_summarize_length + recent_records_length
        
        # Note: Active compression (when called by the model) does not check trigger_length
        # The model can request compression at any time, regardless of content length
        
        # Use simple compression for history management
        if hasattr(self.tool_executor, 'simple_compressor') and self.tool_executor.simple_compressor:
                try:
                    print_current(f"🗜️ Using simple compression for {len(records_to_summarize)} older records ({records_to_summarize_length} chars)...")
                    
                    # Print content before compression
                    print_debug("=" * 80)
                    print_debug("📋 CONTENT BEFORE COMPRESSION (Simple):")
                    print_debug("=" * 80)
                    print_debug(f"Total records in history: {len(history_for_llm)}")
                    print_debug(f"Records to compress: {len(records_to_summarize)}")
                    print_debug(f"Recent records to keep uncompressed: {len(recent_records)}")
                    print_debug("\n--- Records TO COMPRESS ---")
                    for i, record in enumerate(records_to_summarize, 1):
                        print_debug(f"\n--- Record {i} (Round {record.get('task_round', 'N/A')}) ---")
                        result_content = str(record.get("result", ""))
                        print_debug(f"Length: {len(result_content)} characters")
                        print_debug(f"Content:\n{result_content}")
                    print_debug("\n--- Recent Records TO KEEP UNCOMPRESSED ---")
                    for i, record in enumerate(recent_records, 1):
                        print_debug(f"\n--- Recent Record {i} (Round {record.get('task_round', 'N/A')}) ---")
                        result_content = str(record.get("result", ""))
                        print_debug(f"Length: {len(result_content)} characters")
                        print_debug(f"Content:\n{result_content}")
                    print_debug("=" * 80)
                    
                    # Active compression: no trigger_length check, compress immediately
                    compressed_older_records = self.tool_executor.simple_compressor.compress_history(records_to_summarize)
                    
                    # Print content after compression
                    print_debug("=" * 80)
                    print_debug("📋 CONTENT AFTER COMPRESSION (Simple):")
                    print_debug("=" * 80)
                    for i, record in enumerate(compressed_older_records, 1):
                        print_debug(f"\n--- Compressed Record {i} ---")
                        result_content = str(record.get("result", ""))
                        print_debug(f"Length: {len(result_content)} characters")
                        print_debug(f"Content:\n{result_content}")
                    print_debug("=" * 80)
                    
                    # Combine compressed older records with uncompressed recent records
                    compressed_history = compressed_older_records + recent_records
                    
                    # Update task history
                    non_llm_records = [record for record in task_history 
                                     if not ("result" in record) or record.get("error")]
                    task_history.clear()
                    task_history.extend(non_llm_records + compressed_history)
                    
                    # Calculate compression stats
                    compressed_length = sum(len(str(r.get("result", ""))) for r in compressed_older_records)
                    new_total_length = compressed_length + recent_records_length
                    
                    return {
                        "status": "success",
                        "compression_method": "simple",
                        "message": f"History compressed using simple compression",
                        "original_records": len(records_to_summarize),
                        "compressed_records": len(compressed_older_records),
                        "recent_records_kept": len(recent_records),
                        "original_length": records_to_summarize_length,
                        "compressed_length": compressed_length,
                        "recent_length": recent_records_length,
                        "total_before": total_history_length,
                        "total_after": new_total_length,
                        "compression_ratio": f"{(1 - new_total_length/total_history_length)*100:.1f}%"
                    }
                except Exception as e:
                    return {
                        "status": "error",
                        "message": f"Simple compression failed: {e}"
                    }
        else:
            return {
                "status": "error",
                "message": "Simple compressor is not available."
            }

