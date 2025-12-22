#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright (c) 2025 AGI Agent Research Group.

Enhanced History Compressor - Direct record deletion compression:
1. When history exceeds trigger_length, compress to target_length (preserve last N rounds)
2. Insert deletion note at the position where records were deleted
3. Using target_length (usually smaller than trigger_length) prevents repeated compression cycles
"""

from typing import Dict, Any, List, Tuple, Optional
from .print_system import print_current, print_debug


class EnhancedHistoryCompressor:
    """
    Enhanced History Compressor

    Implements direct deletion compression:
    1. When the history exceeds the trigger length, compress to target length (keep the last N rounds)
    2. Insert a note at the position where records were deleted
    3. Using target_length (usually smaller than trigger_length) prevents repeated compression and cache misses
    """

    def __init__(
        self,
        trigger_length: Optional[int] = None,
        target_length: Optional[int] = None,
        keep_recent_rounds: int = 2,
    ):
        """
        Initialize the enhanced compressor

        Args:
            trigger_length: The total length threshold for triggering compression (default loads summary_trigger_length from config, or 100000 chars if not set)
            target_length: The target length after compression (default loads compression_target_length from config, or use 70% of trigger_length if not set)
            keep_recent_rounds: Number of most recent rounds to keep (default 2)
        """
        # Lazy import to avoid circular imports
        if trigger_length is None:
            try:
                from config_loader import get_summary_trigger_length

                trigger_length = get_summary_trigger_length()
            except (ImportError, Exception) as e:
                # Fallback to default if config loading fails
                print_debug(
                    f"⚠️ Failed to load summary_trigger_length from config: {e}, using default 100000"
                )
                trigger_length = 100000

        if target_length is None:
            try:
                from config_loader import get_compression_target_length

                target_length = get_compression_target_length()
            except (ImportError, Exception) as e:
                # Fallback to 70% of trigger_length if config loading fails
                target_length = int(trigger_length * 0.7)
                print_debug(
                    f"⚠️ Failed to load compression_target_length from config: {e}, using default {target_length} (70% of trigger_length)"
                )

        self.trigger_length = trigger_length
        self.target_length = target_length
        self.keep_recent_rounds = keep_recent_rounds

    def compress_history(
        self, task_history: List[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Perform compression process: directly delete old records

        Args:
            task_history: Original history records

        Returns:
            (compressed_history, stats): The compressed history and statistics
        """
        if not task_history:
            return task_history, {
                "simple_compression": {
                    "original_records": 0,
                    "compressed_records": 0,
                    "compressed": False,
                },
                "truncation_compression": {"truncated": False, "records_deleted": 0},
                "final": {"total_records": 0},
            }

        # Step 1: Check total length. If less than trigger_length, do not compress
        # Calculate length for all records, no distinction between LLM and others
        total_length = self._calculate_total_length(task_history)
        if total_length <= self.trigger_length:
            print_debug(
                f"🗜️ History length {total_length} <= trigger_length {self.trigger_length}, skipping compression"
            )
            return task_history, {
                "simple_compression": {
                    "original_records": len(task_history),
                    "compressed_records": len(task_history),
                    "compressed": False,
                },
                "truncation_compression": {"truncated": False, "records_deleted": 0},
                "final": {"total_records": len(task_history)},
            }

        # Step 2: Apply truncation compression for all records (delete older records, keep last N, insert deletion note)
        # No separation of LLM/non-LLM records, process all together  
        final_history, truncation_stats = self._truncation_compress(task_history)

        # Step 3: Build stats
        stats = {
            "simple_compression": {
                "original_records": len(task_history),
                "compressed_records": len(task_history),
                "compressed": False,
            },
            "truncation_compression": truncation_stats,
            "final": {"total_records": len(final_history)},
        }

        return final_history, stats

    def _truncation_compress(
        self, history: List[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Truncation compression: delete oldest records until target length is met (keeping last N rounds).
        Insert a note at the deletion point.

        Args:
            history: List of history records

        Returns:
            (final_history, stats): Compressed history and statistics
        """
        # Calculate current total length
        current_length = self._calculate_total_length(history)

        # If not exceeding trigger, return as is
        if current_length <= self.trigger_length:
            return history, {
                "truncated": False,
                "original_length": current_length,
                "final_length": current_length,
                "records_deleted": 0,
                "original_records": len(history),
                "final_records": len(history),
            }

        # Ensure at least keep_recent_rounds are kept
        if len(history) <= self.keep_recent_rounds:
            # Not enough records to remove
            return history, {
                "truncated": False,
                "original_length": current_length,
                "final_length": current_length,
                "records_deleted": 0,
                "original_records": len(history),
                "final_records": len(history),
            }

        # Split to records to keep (recent N) and records to delete (older)
        records_to_keep = history[-self.keep_recent_rounds :]
        records_to_delete = history[: -self.keep_recent_rounds]

        # Calculate kept records' length
        keep_length = self._calculate_total_length(records_to_keep)

        # If just the recent kept records already exceed the target, warn, but still delete all older records
        if keep_length > self.target_length:
            print_debug(
                f"⚠️ Warning: Even keeping only {self.keep_recent_rounds} recent rounds ({keep_length:,} chars) exceeds target_length ({self.target_length:,} chars). Will delete all older records to minimize total length."
            )

        # Remove from old records until target is met
        final_history = records_to_delete.copy()
        records_deleted = 0
        original_length = current_length
        original_records = len(history)

        # Calculate allowance for older history (target - length of new/kept records)
        # If keep_length > target, available_length < 0, so no old records are retained
        available_length = self.target_length - keep_length

        # Walk from oldest preserved forward, keeping until limit
        kept_older_records = []
        kept_older_length = 0

        for record in reversed(records_to_delete):
            record_length = self._calculate_record_length(record)
            if kept_older_length + record_length <= available_length:
                kept_older_records.insert(0, record)  # maintain order
                kept_older_length += record_length
            else:
                break

        # Count number of records deleted
        records_deleted = len(records_to_delete) - len(kept_older_records)

        # Build final history: kept old records + deletion note (if needed) + kept recent N records
        final_history = kept_older_records.copy()

        if records_deleted > 0:
            deletion_note = self._create_deletion_note(records_deleted)
            final_history.append(deletion_note)

        final_history.extend(records_to_keep)

        # Calculate final length
        final_length = self._calculate_total_length(final_history)

        stats = {
            "truncated": True,
            "original_length": original_length,
            "final_length": final_length,
            "records_deleted": records_deleted,
            "original_records": original_records,
            "final_records": len(final_history),
            "recent_rounds_kept": len(records_to_keep),
        }

        return final_history, stats

    def _calculate_total_length(self, history: List[Dict[str, Any]]) -> int:
        """
        Calculate the total character count of all history records.

        Only counts main fields: prompt, result, content, response, output, data

        Args:
            history: List of history records

        Returns:
            Total characters
        """
        total = 0
        for record in history:
            total += self._calculate_record_length(record)
        return total

    def _calculate_record_length(self, record: Dict[str, Any]) -> int:
        """
        Calculate the character count of a single record.

        Args:
            record: Single history record

        Returns:
            Character count
        """
        total = 0
        fields_to_count = [
            "prompt",
            "result",
            "content",
            "response",
            "output",
            "data",
        ]
        for field in fields_to_count:
            if field in record:
                total += len(str(record[field]))
        return total

    def _create_deletion_note(self, records_deleted: int) -> Dict[str, Any]:
        """
        Create a deletion note record.

        Args:
            records_deleted: Number of records deleted

        Returns:
            Note record (dict)
        """
        return {
            "result": f"older records are deleted due to context length limit ({records_deleted} records deleted)",
            "type": "compression_note",
        }

    def get_compression_stats(
        self,
        original_history: List[Dict[str, Any]],
        compressed_history: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Get compression statistics

        Args:
            original_history: The original history
            compressed_history: The compressed history

        Returns:
            Compression stats
        """
        original_length = self._calculate_total_length(original_history)
        compressed_length = self._calculate_total_length(compressed_history)

        compression_ratio = (
            (1 - compressed_length / original_length) * 100 if original_length > 0 else 0
        )
        saved_chars = original_length - compressed_length

        # Estimate tokens saved (rough estimate: 1 token ≈ 4 chars)
        estimated_token_savings = saved_chars // 4

        return {
            "original_chars": original_length,
            "compressed_chars": compressed_length,
            "saved_chars": saved_chars,
            "compression_ratio": compression_ratio,
            "estimated_token_savings": estimated_token_savings,
            "original_records": len(original_history),
            "compressed_records": len(compressed_history),
        }

