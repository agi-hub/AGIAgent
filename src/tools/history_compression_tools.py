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
        
        # Use enhanced compression for history management
        # Compression will only occur if total_history_length exceeds summary_trigger_length
        # EnhancedHistoryCompressor checks trigger_length internally
        if hasattr(self.tool_executor, 'simple_compressor') and self.tool_executor.simple_compressor:
                try:
                    # Check if using EnhancedHistoryCompressor (new method)
                    if hasattr(self.tool_executor.simple_compressor, 'compress_history') and \
                       hasattr(self.tool_executor.simple_compressor, '_truncation_compress'):
                        # New enhanced compression: simple + truncation
                        # Will skip compression if total_history_length <= summary_trigger_length
                        print_current(f"🗜️ Using enhanced compression (simple + truncation) for {len(history_for_llm)} records...")
                        
                        # Print content before compression
                        print_debug("=" * 80)
                        print_debug("📋 CONTENT BEFORE COMPRESSION (Enhanced):")
                        print_debug("=" * 80)
                        print_debug(f"Total records in history: {len(task_history)}")
                        print_debug(f"LLM records: {len(history_for_llm)}")
                        print_debug(f"Records to compress: {len(records_to_summarize)}")
                        print_debug(f"Recent records to keep uncompressed: {len(recent_records)}")
                        print_debug(f"Total history length: {total_history_length}")
                        print_debug("=" * 80)
                        
                        # Execute enhanced compression (simple + truncation)
                        # EnhancedHistoryCompressor will check trigger_length and skip if below threshold
                        compressed_history, compression_stats = self.tool_executor.simple_compressor.compress_history(task_history)
                        
                        # Extract LLM records from compressed history
                        compressed_llm_records = [r for r in compressed_history 
                                                  if "result" in r or "error" in r]
                        
                        # Update task history
                        non_llm_records = [record for record in task_history 
                                         if not ("result" in record or "error" in record)]
                        task_history.clear()
                        task_history.extend(non_llm_records + compressed_llm_records)
                        
                        # Print content after compression
                        print_debug("=" * 80)
                        print_debug("📋 CONTENT AFTER COMPRESSION (Enhanced):")
                        print_debug("=" * 80)
                        print_debug(f"Final records: {len(compressed_history)}")
                        print_debug(f"Final LLM records: {len(compressed_llm_records)}")
                        print_debug(f"Compression stats: {compression_stats}")
                        print_debug("=" * 80)
                        
                        # Calculate final stats
                        final_length = sum(len(str(r.get("result", ""))) for r in compressed_llm_records)
                        
                        return {
                            "status": "success",
                            "compression_method": "enhanced",
                            "message": f"History compressed using enhanced compression (simple + truncation)",
                            "original_records": len(history_for_llm),
                            "compressed_records": len(compressed_llm_records),
                            "recent_records_kept": compression_stats.get('simple_compression', {}).get('recent_rounds_kept', 0),
                            "original_length": total_history_length,
                            "compressed_length": final_length,
                            "total_before": total_history_length,
                            "total_after": final_length,
                            "compression_ratio": f"{(1 - final_length/total_history_length)*100:.1f}%" if total_history_length > 0 else "0%",
                            "truncation_stats": compression_stats.get('truncation_compression', {}),
                            "simple_stats": compression_stats.get('simple_compression', {})
                        }
                    else:
                        # Fallback to old compression method (backward compatibility)
                        # Check trigger_length: only compress if total_history_length exceeds threshold
                        try:
                            from config_loader import get_summary_trigger_length
                            trigger_length = get_summary_trigger_length()
                        except (ImportError, Exception):
                            trigger_length = 100000  # Default fallback
                        
                        if total_history_length <= trigger_length:
                            return {
                                "status": "skipped",
                                "message": f"History length {total_history_length} <= trigger_length {trigger_length}, skipping compression",
                                "current_records": len(history_for_llm),
                                "total_history_length": total_history_length,
                                "trigger_length": trigger_length
                            }
                        
                        print_current(f"🗜️ Using simple compression for {len(records_to_summarize)} older records ({records_to_summarize_length} chars)...")
                        
                        # Print content before compression
                        print_debug("=" * 80)
                        print_debug("📋 CONTENT BEFORE COMPRESSION (Simple):")
                        print_debug("=" * 80)
                        print_debug(f"Total records in history: {len(history_for_llm)}")
                        print_debug(f"Records to compress: {len(records_to_summarize)}")
                        print_debug(f"Recent records to keep uncompressed: {len(recent_records)}")
                        print_debug(f"Total history length: {total_history_length}, trigger_length: {trigger_length}")
                        print_debug("=" * 80)
                        
                        # Compress with trigger_length check
                        compressed_older_records = self.tool_executor.simple_compressor.compress_history(records_to_summarize, trigger_length=trigger_length)
                        
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
                    import traceback
                    print_debug(f"⚠️ Compression failed: {e}")
                    traceback.print_exc()
                    return {
                        "status": "error",
                        "message": f"Compression failed: {e}"
                    }
        else:
            return {
                "status": "error",
                "message": "Simple compressor is not available."
            }

