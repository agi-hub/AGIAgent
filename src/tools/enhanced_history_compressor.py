#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright (c) 2025 AGI Agent Research Group.

Enhanced History Compressor - Two-stage compression:
1. Simple compression: truncate long fields (head+tail, exclude last 2 rounds)
2. Truncation compression: delete oldest records until within truncation_length
"""

import json
from typing import Dict, Any, List, Tuple, Optional
from .print_system import print_current, print_debug


class EnhancedHistoryCompressor:
    """
    增强的历史压缩器
    
    实现两级压缩：
    1. 简单压缩：字段级别的头尾保留压缩（排除最后2轮）
    2. 限定压缩：记录级别的删除压缩（全部记录都可删除）
    """
    
    def __init__(self, 
                 min_length: int = 500,
                 head_length: int = 100,
                 tail_length: int = 100,
                 trigger_length: Optional[int] = None,
                 keep_recent_rounds: int = 2,
                 ellipsis: str = "\n...[omitted {} chars]...\n"):
        """
        初始化增强压缩器
        
        Args:
            min_length: 触发字段压缩的最小长度（默认500字符）
            head_length: 字段压缩时保留的开头字符数（默认100）
            tail_length: 字段压缩时保留的结尾字符数（默认100）
            trigger_length: 触发压缩的历史记录总长度阈值（默认从配置文件读取summary_trigger_length，如果未配置则使用100000字符）
            keep_recent_rounds: 简单压缩时保留的最近轮次数（默认2）
            ellipsis: 省略标记格式
        """
        # Lazy import to avoid circular imports
        if trigger_length is None:
            try:
                from config_loader import get_summary_trigger_length
                trigger_length = get_summary_trigger_length()
            except (ImportError, Exception) as e:
                # Fallback to default if config loading fails
                print_debug(f"⚠️ Failed to load summary_trigger_length from config: {e}, using default 100000")
                trigger_length = 100000
        
        self.min_length = min_length
        self.head_length = head_length
        self.tail_length = tail_length
        self.trigger_length = trigger_length
        self.keep_recent_rounds = keep_recent_rounds
        self.ellipsis = ellipsis
    
    def compress_history(self, 
                        task_history: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        执行完整压缩流程：简单压缩 + 限定压缩
        
        Args:
            task_history: 原始历史记录
        
        Returns:
            (compressed_history, stats): 压缩后的历史记录和统计信息
        """
        if not task_history:
            return task_history, {
                "simple_compression": {"original_records": 0, "compressed_records": 0},
                "truncation_compression": {"truncated": False, "records_deleted": 0},
                "final": {"total_records": 0}
            }
        
        # 步骤1：分离非LLM记录和LLM记录
        non_llm_records = [r for r in task_history 
                          if not ("result" in r or "error" in r)]
        llm_records = [r for r in task_history 
                      if "result" in r or "error" in r]
        
        if not llm_records:
            return task_history, {
                "simple_compression": {"original_records": 0, "compressed_records": 0},
                "truncation_compression": {"truncated": False, "records_deleted": 0},
                "final": {"total_records": len(task_history)}
            }
        
        # 步骤1.5：检查总长度，如果小于trigger_length则不进行任何压缩
        total_length = self._calculate_total_length(llm_records)
        if total_length <= self.trigger_length:
            print_debug(f"🗜️ History length {total_length} <= trigger_length {self.trigger_length}, skipping compression")
            return task_history, {
                "simple_compression": {"original_records": len(llm_records), "compressed_records": len(llm_records), "compressed": False},
                "truncation_compression": {"truncated": False, "records_deleted": 0},
                "final": {
                    "total_records": len(task_history),
                    "llm_records": len(llm_records),
                    "non_llm_records": len(non_llm_records)
                }
            }
        
        # 步骤2：简单压缩（排除最后2轮）
        compressed_llm_records, simple_stats = self._simple_compress(llm_records)
        
        # 步骤3：限定压缩（全部记录都可删除，使用trigger_length作为限制）
        final_llm_records, truncation_stats = self._truncation_compress(compressed_llm_records)
        
        # 步骤4：合并非LLM记录和压缩后的LLM记录
        final_history = non_llm_records + final_llm_records
        
        # 步骤5：生成统计信息
        stats = {
            "simple_compression": simple_stats,
            "truncation_compression": truncation_stats,
            "final": {
                "total_records": len(final_history),
                "llm_records": len(final_llm_records),
                "non_llm_records": len(non_llm_records)
            }
        }
        
        return final_history, stats
    
    def _simple_compress(self, history: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        简单压缩：字段级别的头尾保留压缩（排除最后N轮）
        
        Args:
            history: LLM历史记录
        
        Returns:
            (compressed_history, stats): 压缩后的记录和统计信息
        """
        if len(history) <= self.keep_recent_rounds:
            # 记录数不足，不进行压缩
            return history, {
                "original_records": len(history),
                "compressed_records": len(history),
                "recent_rounds_kept": len(history),
                "compressed": False
            }
        
        # 分离：最后N轮 vs 其他轮次
        older_records = history[:-self.keep_recent_rounds]
        recent_records = history[-self.keep_recent_rounds:]
        
        # 对旧记录进行字段压缩
        compressed_older_records = []
        for record in older_records:
            compressed_record = self._compress_record_fields(record.copy())
            compressed_older_records.append(compressed_record)
        
        # 合并：压缩的旧记录 + 未压缩的新记录
        compressed_history = compressed_older_records + recent_records
        
        stats = {
            "original_records": len(history),
            "compressed_records": len(compressed_history),
            "recent_rounds_kept": len(recent_records),
            "older_records_compressed": len(compressed_older_records),
            "compressed": True
        }
        
        return compressed_history, stats
    
    def _truncation_compress(self, history: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        限定压缩：删除最旧的记录直到满足长度限制
        
        Args:
            history: 简单压缩后的历史记录
        
        Returns:
            (final_history, stats): 满足长度限制的历史记录和统计信息
        """
        # 计算当前总长度
        current_length = self._calculate_total_length(history)
        
        # 如果未超过限制，直接返回
        if current_length <= self.trigger_length:
            return history, {
                "truncated": False,
                "original_length": current_length,
                "final_length": current_length,
                "records_deleted": 0,
                "original_records": len(history),
                "final_records": len(history)
            }
        
        # 循环删除最旧的记录
        final_history = history.copy()
        records_deleted = 0
        original_length = current_length
        original_records = len(history)
        
        print_debug(f"🗜️ Truncation compression: original length {original_length} exceeds trigger_length {self.trigger_length}")
        
        while current_length > self.trigger_length and len(final_history) > 0:
            # 删除最旧的记录（第一条）
            deleted_record = final_history.pop(0)
            records_deleted += 1
            
            # 重新计算长度
            current_length = self._calculate_total_length(final_history)
            
            print_debug(f"🗜️ Deleted record {records_deleted}, current length: {current_length}, remaining records: {len(final_history)}")
            
            # 安全检查：至少保留1条记录（如果可能）
            if len(final_history) == 0:
                print_current(f"⚠️ All records deleted, but still exceeds trigger_length")
                break
        
        stats = {
            "truncated": True,
            "original_length": original_length,
            "final_length": current_length,
            "records_deleted": records_deleted,
            "original_records": original_records,
            "final_records": len(final_history)
        }
        
        if records_deleted > 0:
            print_current(f"🗜️ Truncation compression: deleted {records_deleted} oldest records, "
                         f"length reduced from {original_length} to {current_length}")
        
        return final_history, stats
    
    def _compress_record_fields(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """
        压缩单条记录的字段（头尾保留，中间删除）
        
        Args:
            record: 单条历史记录
        
        Returns:
            压缩后的记录
        """
        # 需要压缩的字段列表
        fields_to_check = ['prompt', 'result', 'content', 'response', 'output', 'data']
        
        for field in fields_to_check:
            if field in record:
                record[field] = self._compress_field_content(record[field])
        
        return record
    
    def _compress_field_content(self, content: Any) -> Any:
        """
        压缩字段内容（递归处理字符串、字典、列表）
        
        Args:
            content: 字段内容
        
        Returns:
            压缩后的内容
        """
        if isinstance(content, str):
            return self._compress_string(content)
        elif isinstance(content, dict):
            return {k: self._compress_field_content(v) for k, v in content.items()}
        elif isinstance(content, list):
            return [self._compress_field_content(item) for item in content]
        else:
            return content
    
    def _compress_string(self, text: str) -> str:
        """
        压缩字符串：头尾保留，中间删除
        
        Args:
            text: 原始字符串
        
        Returns:
            压缩后的字符串
        """
        if not text or len(text) <= self.min_length:
            return text
        
        # 检查是否是JSON格式
        if self._looks_like_json(text):
            return self._compress_json_string(text)
        
        # 普通字符串：头尾保留，中间删除
        return self._truncate_string(text)
    
    def _looks_like_json(self, text: str) -> bool:
        """检查字符串是否看起来像JSON格式"""
        text = text.strip()
        return (text.startswith('{') and text.endswith('}')) or \
               (text.startswith('[') and text.endswith(']'))
    
    def _compress_json_string(self, text: str) -> str:
        """压缩JSON格式的字符串"""
        try:
            json_data = json.loads(text)
            compressed_json = self._compress_field_content(json_data)
            return json.dumps(compressed_json, ensure_ascii=False, indent=2)
        except (json.JSONDecodeError, Exception):
            return self._truncate_string(text)
    
    def _truncate_string(self, text: str) -> str:
        """
        截断字符串：保留头尾，删除中间
        
        Args:
            text: 原始字符串
        
        Returns:
            截断后的字符串
        """
        if len(text) <= self.min_length:
            return text
        
        # 特殊处理：如果包含 "Tool execution results:"，对标记前后分别压缩
        marker = "Tool execution results:"
        if marker in text:
            return self._truncate_string_with_marker(text, marker)
        
        # 计算省略的字符数
        omitted_chars = len(text) - self.head_length - self.tail_length
        
        # 确保不为负数
        if omitted_chars <= 0:
            return text
        
        # 获取头尾部分
        head_part = text[:self.head_length]
        tail_part = text[-self.tail_length:]
        
        # 创建省略标记
        ellipsis_text = self.ellipsis.format(omitted_chars)
        
        return head_part + ellipsis_text + tail_part
    
    def _truncate_string_with_marker(self, text: str, marker: str) -> str:
        """
        截断包含标记的字符串：对标记前后部分分别进行压缩
        
        Args:
            text: 包含标记的原始字符串
            marker: 标记字符串（如 "Tool execution results:"）
        
        Returns:
            截断后的字符串（保留标记）
        """
        # 查找标记位置
        marker_pos = text.find(marker)
        if marker_pos == -1:
            # 不应该发生，但回退到普通截断（避免递归）
            omitted_chars = len(text) - self.head_length - self.tail_length
            if omitted_chars <= 0:
                return text
            head_part = text[:self.head_length]
            tail_part = text[-self.tail_length:]
            ellipsis_text = self.ellipsis.format(omitted_chars)
            return head_part + ellipsis_text + tail_part
        
        # 分为三部分：标记前、标记本身、标记后
        before_marker = text[:marker_pos]
        marker_text = marker
        after_marker = text[marker_pos + len(marker):]
        
        # 压缩标记前的部分（如果足够长）
        if len(before_marker) > self.min_length:
            omitted_before = len(before_marker) - self.head_length - self.tail_length
            if omitted_before > 0:
                before_head = before_marker[:self.head_length]
                before_tail = before_marker[-self.tail_length:]
                before_ellipsis = self.ellipsis.format(omitted_before)
                compressed_before = before_head + before_ellipsis + before_tail
            else:
                compressed_before = before_marker
        else:
            compressed_before = before_marker
        
        # 压缩标记后的部分（如果足够长）
        if len(after_marker) > self.min_length:
            omitted_after = len(after_marker) - self.head_length - self.tail_length
            if omitted_after > 0:
                after_head = after_marker[:self.head_length]
                after_tail = after_marker[-self.tail_length:]
                after_ellipsis = self.ellipsis.format(omitted_after)
                compressed_after = after_head + after_ellipsis + after_tail
            else:
                compressed_after = after_marker
        else:
            compressed_after = after_marker
        
        # 组合：压缩的前部分 + 标记 + 压缩的后部分
        return compressed_before + marker_text + compressed_after
    
    def _calculate_total_length(self, history: List[Dict[str, Any]]) -> int:
        """
        计算历史记录的总字符数
        
        只计算主要字段：prompt, result, content, response, output, data
        
        Args:
            history: 历史记录列表
        
        Returns:
            总字符数
        """
        total = 0
        fields_to_count = ['prompt', 'result', 'content', 'response', 'output', 'data']
        
        for record in history:
            for field in fields_to_count:
                if field in record:
                    total += len(str(record[field]))
        
        return total
    
    def get_compression_stats(self, 
                            original_history: List[Dict[str, Any]], 
                            compressed_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        获取压缩统计信息
        
        Args:
            original_history: 原始历史记录
            compressed_history: 压缩后的历史记录
        
        Returns:
            压缩统计信息
        """
        original_length = self._calculate_total_length(original_history)
        compressed_length = self._calculate_total_length(compressed_history)
        
        compression_ratio = (1 - compressed_length / original_length) * 100 if original_length > 0 else 0
        saved_chars = original_length - compressed_length
        
        # 估算token节省（粗略估算：1 token ≈ 4 chars）
        estimated_token_savings = saved_chars // 4
        
        return {
            'original_chars': original_length,
            'compressed_chars': compressed_length,
            'saved_chars': saved_chars,
            'compression_ratio': compression_ratio,
            'estimated_token_savings': estimated_token_savings,
            'original_records': len(original_history),
            'compressed_records': len(compressed_history)
        }

