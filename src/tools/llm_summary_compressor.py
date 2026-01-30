#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright (c) 2025 AGI Agent Research Group.

LLM Summary Compressor - Use LLM to summarize old conversation history:
1. When history exceeds trigger_length, use LLM to summarize old records
2. Keep the most recent N rounds uncompressed
3. Replace old records with a concise summary
"""

from typing import Dict, Any, List, Tuple, Optional
from .print_system import print_current, print_debug, streaming_context


class LLMSummaryCompressor:
    """
    LLM Summary Compressor
    
    Uses LLM to generate summaries of old conversation history instead of deleting them.
    This preserves important context while reducing token consumption.
    """
    
    def __init__(
        self,
        trigger_length: Optional[int] = None,
        target_length: Optional[int] = None,
        keep_recent_rounds: int = 2,
        api_client=None,
        model: str = None,
        api_key: str = None,
        api_base: str = None,
    ):
        """
        Initialize the LLM summary compressor
        
        Args:
            trigger_length: Length threshold for triggering compression (default: from config or 100000)
            target_length: Target length after compression (default: from config or 50000)
            keep_recent_rounds: Number of most recent rounds to keep uncompressed (default: 2)
            api_client: Optional pre-configured API client
            model: Model name for LLM calls
            api_key: API key for LLM calls
            api_base: API base URL for LLM calls
        """
        # Load trigger_length from config
        if trigger_length is None:
            try:
                from config_loader import get_summary_trigger_length
                trigger_length = get_summary_trigger_length()
            except (ImportError, Exception) as e:
                print_debug(f"⚠️ Failed to load summary_trigger_length from config: {e}, using default 100000")
                trigger_length = 100000
        
        # Load target_length from config
        if target_length is None:
            try:
                from config_loader import get_compression_target_length
                target_length = get_compression_target_length()
            except (ImportError, Exception) as e:
                target_length = int(trigger_length * 0.5)
                print_debug(f"⚠️ Failed to load compression_target_length from config: {e}, using default {target_length}")
        
        self.trigger_length = trigger_length
        self.target_length = target_length
        self.keep_recent_rounds = keep_recent_rounds
        self.api_client = api_client
        self.model = model
        self.api_key = api_key
        self.api_base = api_base
        
        # Load max_tokens from config
        try:
            from config_loader import get_max_tokens
            self.max_tokens = get_max_tokens() or 16384
        except (ImportError, Exception):
            self.max_tokens = 16384
        
        # Load summary_streaming from config
        try:
            from config_loader import get_summary_streaming
            self.streaming = get_summary_streaming()
        except (ImportError, Exception):
            self.streaming = True  # Default to streaming enabled
    
    def compress_history(
        self, task_history: List[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Perform compression using LLM summarization
        
        Args:
            task_history: Original history records
            
        Returns:
            (compressed_history, stats): The compressed history and statistics
        """
        if not task_history:
            return task_history, {
                "compression_method": "llm_summary",
                "compressed": False,
                "original_records": 0,
                "final_records": 0,
            }
        
        # Calculate total length
        total_length = self._calculate_total_length(task_history)
        
        # Check if compression is needed
        if total_length <= self.trigger_length:
            print_debug(f"🗜️ [LLM Summary] History length {total_length:,} <= trigger_length {self.trigger_length:,}, skipping compression")
            return task_history, {
                "compression_method": "llm_summary",
                "compressed": False,
                "original_records": len(task_history),
                "final_records": len(task_history),
                "original_length": total_length,
                "final_length": total_length,
            }
        
        # Filter records with results (LLM records)
        history_for_llm = [r for r in task_history if "result" in r or "error" in r]
        non_llm_records = [r for r in task_history if not ("result" in r or "error" in r)]
        
        # Not enough records to compress (need at least 1 record when keep_recent_rounds=0)
        min_records = max(1, self.keep_recent_rounds + 1)
        if len(history_for_llm) < min_records:
            return task_history, {
                "compression_method": "llm_summary",
                "compressed": False,
                "original_records": len(task_history),
                "final_records": len(task_history),
                "reason": "not_enough_records"
            }
        
        # Split into old records (to summarize) and recent records (to keep)
        # Handle keep_recent_rounds=0 specially (Python slice [:-0] returns empty list)
        if self.keep_recent_rounds == 0:
            records_to_summarize = history_for_llm  # Compress all records
            recent_records = []  # Keep nothing
        else:
            records_to_summarize = history_for_llm[:-self.keep_recent_rounds]
            recent_records = history_for_llm[-self.keep_recent_rounds:]
        
        if not records_to_summarize:
            return task_history, {
                "compression_method": "llm_summary",
                "compressed": False,
                "original_records": len(task_history),
                "final_records": len(task_history),
                "reason": "no_records_to_summarize"
            }
        
        # Check for existing summary in records to summarize (for incremental update)
        existing_summary = None
        new_records = []
        for record in records_to_summarize:
            if record.get("type") == "llm_summary":
                # Extract existing summary content (remove the reminder header)
                result = record.get("result", "")
                # Find the actual summary content after the reminder box
                if "╚" in result:
                    existing_summary = result.split("╚")[1].split("╝")[-1].strip()
                elif "历史对话摘要" in result:
                    existing_summary = result.split("\n\n", 2)[-1] if "\n\n" in result else result
                else:
                    existing_summary = result
                print_current(f"🗜️ [LLM Summary] Found existing summary ({len(existing_summary):,} chars), will perform incremental update")
            else:
                new_records.append(record)
        
        # Use new_records (without old summary) for summarization
        records_for_summary = new_records if new_records else records_to_summarize
        
        # Calculate original length of records to summarize
        original_summary_length = self._calculate_total_length(records_for_summary)
        recent_length = self._calculate_total_length(recent_records)
        
        # Calculate target summary length
        # target_length is the overall target, minus the length of recent records we're keeping
        target_summary_length = max(self.target_length - recent_length, int(original_summary_length * 0.2))
        
        print_current(f"🗜️ [LLM Summary] Compressing {len(records_for_summary)} records ({original_summary_length:,} chars) using LLM summarization...")
        print_current(f"🗜️ [LLM Summary] Target summary length: ~{target_summary_length:,} chars")
        
        try:
            # Generate LLM summary with target length (pass existing summary for incremental update)
            summary = self._generate_summary(records_for_summary, target_summary_length, existing_summary)
            
            if not summary:
                print_debug("⚠️ [LLM Summary] Failed to generate summary, keeping original history")
                return task_history, {
                    "compression_method": "llm_summary",
                    "compressed": False,
                    "error": "summary_generation_failed"
                }
            
            # Create summary record with file reading reminder - VERY PROMINENT
            file_reading_reminder = """
╔════════════════════════════════════════════════════════════════════╗
║  ⚠️⚠️⚠️ 【强制要求 - 必须先读取文件再继续编码】 ⚠️⚠️⚠️           ║
╠════════════════════════════════════════════════════════════════════╣
║  此摘要是压缩版本，代码细节不完整！                                ║
║                                                                    ║
║  在执行任何编码操作之前，你必须：                                  ║
║  1. 立即使用 read_file 工具读取下方 FILES TO READ 中的文件        ║
║  2. 确认现有代码的变量名、函数签名、类结构                        ║
║  3. 确保新代码与现有代码兼容                                      ║
║                                                                    ║
║  ❌ 不读取文件直接编码 = 代码不兼容、变量名错误、导入缺失         ║
║  ✅ 先读取关键文件 = 代码正确、无冲突                              ║
╚════════════════════════════════════════════════════════════════════╝

"""
            summary_record = {
                "result": f"[📋 历史对话摘要 - 已压缩 {len(records_to_summarize)} 条记录]\n\n{file_reading_reminder}{summary}",
                "type": "llm_summary",
                "compressed_records_count": len(records_to_summarize),
                "original_length": original_summary_length,
            }
            
            # Build final history: non-LLM records + summary + recent records
            final_history = non_llm_records + [summary_record] + recent_records
            
            # Calculate final length
            final_length = self._calculate_total_length(final_history)
            summary_length = len(summary)
            
            print_current(f"🗜️ [LLM Summary] Compression complete: {original_summary_length:,} chars → {summary_length:,} chars (summary), kept {len(recent_records)} recent rounds ({recent_length:,} chars)")
            
            # Print summary content to log for debugging
            print_current(f"🗜️ [LLM Summary] ========== 摘要内容开始 ==========")
            print_current(summary)
            print_current(f"🗜️ [LLM Summary] ========== 摘要内容结束 (共 {summary_length:,} 字符) ==========")
            
            stats = {
                "compression_method": "llm_summary",
                "compressed": True,
                "original_records": len(task_history),
                "final_records": len(final_history),
                "records_summarized": len(records_to_summarize),
                "recent_rounds_kept": len(recent_records),
                "original_length": total_length,
                "final_length": final_length,
                "summary_length": summary_length,
                "compression_ratio": f"{(1 - final_length/total_length)*100:.1f}%" if total_length > 0 else "0%",
            }
            
            return final_history, stats
            
        except Exception as e:
            print_debug(f"⚠️ [LLM Summary] Compression failed: {e}")
            import traceback
            traceback.print_exc()
            return task_history, {
                "compression_method": "llm_summary",
                "compressed": False,
                "error": str(e)
            }
    
    def _generate_summary(self, records: List[Dict[str, Any]], target_length: int = None, existing_summary: str = None) -> str:
        """
        Generate summary using LLM (supports incremental update)
        
        Args:
            records: Records to summarize
            target_length: Target length of the summary in characters
            existing_summary: Previous summary to update (for incremental compression)
            
        Returns:
            Summary text
        """
        # Prepare content to summarize - no truncation, keep all content
        content_parts = []
        original_length = 0
        for i, record in enumerate(records, 1):
            result = record.get("result", "")
            if result:
                result_str = str(result)
                original_length += len(result_str)
                content_parts.append(f"[记录 {i}]\n{result_str}")
        
        content_to_summarize = "\n\n---\n\n".join(content_parts)
        
        # Calculate target length if not provided
        if target_length is None:
            target_length = int(original_length * 0.2)  # Default to 20% of original
        
        # Build prompts with structured format
        min_length = int(target_length * 0.8)
        
        # Determine if this is an incremental update
        is_incremental = existing_summary is not None and len(existing_summary) > 100
        
        if is_incremental:
            # Calculate existing summary length for compression guidance
            existing_len = len(existing_summary) if existing_summary else 0
            max_length = int(target_length * 1.2)
            
            system_prompt = f"""你是一个软件工程对话摘要器，用于 IDE 场景。你需要对已有的"项目状态摘要"进行**增量更新与压缩**。

⚠️ **【关键：控制长度，防止膨胀】** ⚠️
- 当前已有摘要：{existing_len:,} 字符
- **目标长度：{target_length:,} 字符**
- **最大长度：{max_length:,} 字符（严禁超过）**

如果合并后超过目标长度，你必须**压缩旧内容**：

========================
压缩策略（重要）
========================

1. **合并相似 TASK**：
   - 同一模块/功能的多个 TASK 可以合并为一个
   - 例如：TASK 3（创建模型）+ TASK 5（更新模型）→ 合并为一个"模型开发"TASK

2. **精简 DETAILS**：
   - 只保留关键决策和最终结果
   - 删除中间过程和重复描述
   - 例如："创建了文件并添加了代码" → "创建 xxx.py"

3. **精简 CODE SNIPPETS**：
   - 只保留**函数签名**和**关键逻辑**
   - 删除完整代码实现，保留结构

4. **必须保留**（不可删除）：
   - 所有 FILEPATHS（文件路径列表）
   - ERRORS AND SOLUTIONS
   - CONFIGURATION
   - FILES TO READ

========================
增量更新规则
========================

A. 任务合并
- 新对话属于已有 TASK → 更新该 TASK
- 新对话是新任务 → 创建新 TASK（考虑是否可与已有 TASK 合并）

B. 状态更新
- done：已完成
- in-progress：进行中

C. 长度控制
- 输出长度必须在 **{min_length:,} ~ {max_length:,} 字符之间**
- 如果超过 {max_length:,}，必须进一步压缩旧 TASK 的 DETAILS"""
        else:
            system_prompt = f"""你是一个软件工程对话摘要器，用于 IDE 场景。你需要将多轮对话历史整理成结构化的"项目状态摘要（Project Summary）"。

========================
输出格式（必须严格遵守）
========================

TASK <编号>: <任务名>
STATUS: done | in-progress | abandoned
USER QUERIES: <轮次范围，如 1-5>
DETAILS:
- <要点1>
- <要点2>
- <关键决策、变更、结果>
FILEPATHS:
- <创建/修改的文件路径1>
- <创建/修改的文件路径2>
CODE SNIPPETS:
```<语言>
<关键代码片段>
```

（可以有多个 TASK，按时间顺序编号）

ERRORS AND SOLUTIONS:
- <错误1>: <解决方案1>
- <错误2>: <解决方案2>

USER CORRECTIONS AND INSTRUCTIONS:
- <用户的硬性要求清单>

CONFIGURATION:
- <配置项1>: <值1>
- <配置项2>: <值2>

NEXT STEPS:
- <下一步动作>

⚠️ FILES TO READ (重要):
- <后续编码前必须读取的文件1>
- <后续编码前必须读取的文件2>
（列出所有包含关键代码结构的文件，后续模型需要读取这些文件来了解代码上下文）

========================
摘要规则（严格执行）
========================

A. 任务识别与组织
1) 识别对话中的不同任务/目标，每个独立目标创建一个 TASK
2) 同一目标/同一模块/同一文件的操作归入同一个 TASK
3) 按时间顺序编号：TASK 1, TASK 2, ...

B. 状态判定
- done：目标已实现或用户确认完成
- in-progress：已开始但仍需后续工作

C. 内容详细度要求
- DETAILS 必须包含具体的技术细节，不要泛泛而谈
- FILEPATHS 必须列出所有涉及的文件完整路径
- CODE SNIPPETS 必须保留关键代码（函数签名、类定义、配置内容）
- 每个 TASK 的内容要充分展开

D. 错误与问题记录
- 所有异常、报错、失败必须记录到 ERRORS AND SOLUTIONS
- 必须同时记录解决方案

E. 不要编造
- 没有在对话中出现的信息不要写
- 不确定的内容不要下结论

F. FILES TO READ（必须填写）
- 列出所有包含关键代码结构的文件（如定义了类、函数、配置的文件）
- 这些文件是后续编码时必须先读取的，以确保代码兼容性
- 优先列出：主程序文件、模型定义、配置文件、API 路由文件

G. 长度要求
- 输出必须达到 **{min_length:,} 字符以上**
- 目标长度：**{target_length:,} 字符**
- 通过详细展开每个 TASK 的 DETAILS 和 CODE SNIPPETS 来达到长度要求"""

        if is_incremental:
            user_prompt = f"""请对以下项目状态摘要进行**增量更新与压缩**。

⚠️ **【长度控制 - 严格执行】** ⚠️
- 已有摘要：{existing_len:,} 字符
- 目标长度：**{target_length:,} 字符**
- 最大长度：**{max_length:,} 字符（严禁超过）**

如果合并后超长，请压缩旧 TASK 的 DETAILS（精简描述），但保留所有 FILEPATHS。

========================
已有摘要（可以压缩 DETAILS，但保留 FILEPATHS）
========================

{existing_summary}

========================
新增对话片段
========================

{content_to_summarize}

========================

请输出**更新后的摘要**（合并新旧内容，控制在 {target_length:,} 字符左右，最多 {max_length:,} 字符）："""
        else:
            user_prompt = f"""请将以下 {len(records)} 条对话历史整理成结构化的项目状态摘要。

【长度要求】输出必须达到 {min_length:,} 字符以上，目标 {target_length:,} 字符

以下是需要整理的 {len(records)} 条记录：

{content_to_summarize}

请按照指定格式输出项目状态摘要（包含 TASK、STATUS、DETAILS、FILEPATHS、CODE SNIPPETS 等）："""

        # Call LLM
        return self._call_llm(system_prompt, user_prompt)
    
    def _call_llm(self, system_prompt: str, user_prompt: str) -> str:
        """
        Call LLM to generate response (supports streaming output)
        
        Args:
            system_prompt: System prompt
            user_prompt: User prompt
            
        Returns:
            LLM response text
        """
        try:
            # Method 1: Use provided API client (non-streaming for backward compatibility)
            if self.api_client:
                # Check if it's Anthropic client
                if hasattr(self.api_client, 'messages'):
                    response = self.api_client.messages.create(
                        model=self.model,
                        max_tokens=self.max_tokens,
                        system=system_prompt,
                        messages=[{"role": "user", "content": user_prompt}]
                    )
                    if hasattr(response, 'content') and response.content:
                        if isinstance(response.content, list):
                            return response.content[0].text
                        return str(response.content)
                # Check if it's OpenAI-compatible client
                elif hasattr(self.api_client, 'chat'):
                    response = self.api_client.chat.completions.create(
                        model=self.model,
                        max_tokens=self.max_tokens,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt}
                        ]
                    )
                    return response.choices[0].message.content
            
            # Method 2: Create new client using config
            from config_loader import get_api_key, get_api_base, get_model
            
            api_key = self.api_key or get_api_key()
            api_base = self.api_base or get_api_base()
            model = self.model or get_model()
            
            if not api_key or not api_base or not model:
                raise ValueError("Missing API configuration for LLM summary compression")
            
            # Determine API type based on api_base
            is_anthropic = api_base.lower().endswith('/anthropic') if api_base else False
            
            print_current(f"🗜️ [LLM Summary] Calling LLM with max_tokens={self.max_tokens}, model={model}, streaming={self.streaming}")
            
            if is_anthropic:
                return self._call_anthropic(api_key, api_base, model, system_prompt, user_prompt)
            else:
                return self._call_openai(api_key, api_base, model, system_prompt, user_prompt)
            
        except Exception as e:
            print_debug(f"⚠️ [LLM Summary] LLM call failed: {e}")
            raise e
    
    def _call_anthropic(self, api_key: str, api_base: str, model: str, 
                        system_prompt: str, user_prompt: str) -> str:
        """
        Call Anthropic API (supports streaming)
        """
        from anthropic import Anthropic
        
        client = Anthropic(api_key=api_key, base_url=api_base)
        
        if self.streaming:
            # Streaming mode
            content = ""
            with streaming_context(show_start_message=False) as printer:
                printer.write("\n🗜️ [摘要生成中] ")
                
                with client.messages.stream(
                    model=model,
                    max_tokens=self.max_tokens,
                    system=system_prompt,
                    messages=[{"role": "user", "content": user_prompt}]
                ) as stream:
                    stop_reason = None
                    input_tokens = 0
                    output_tokens = 0
                    
                    for event in stream:
                        event_type = getattr(event, 'type', None)
                        
                        # Handle content_block_delta event (text content)
                        if event_type == "content_block_delta":
                            delta = getattr(event, 'delta', None)
                            if delta:
                                delta_type = getattr(delta, 'type', None)
                                if delta_type == "text_delta":
                                    text = getattr(delta, 'text', '')
                                    content += text
                                    printer.write(text)
                        
                        # Handle message_delta event (usage stats)
                        elif event_type == "message_delta":
                            delta = getattr(event, 'delta', None)
                            if delta:
                                stop_reason = getattr(delta, 'stop_reason', stop_reason)
                            usage = getattr(event, 'usage', None)
                            if usage:
                                output_tokens = getattr(usage, 'output_tokens', output_tokens)
                        
                        # Handle message_start event (input tokens)
                        elif event_type == "message_start":
                            message = getattr(event, 'message', None)
                            if message:
                                usage = getattr(message, 'usage', None)
                                if usage:
                                    input_tokens = getattr(usage, 'input_tokens', input_tokens)
                
                printer.write("\n")
            
            # Print API response metadata
            print_current(f"🗜️ [LLM Summary] API Response - stop_reason: {stop_reason}, input_tokens: {input_tokens}, output_tokens: {output_tokens}")
            
            if not content:
                raise ValueError("Empty response from Anthropic API (streaming)")
            return content
        else:
            # Non-streaming mode
            response = client.messages.create(
                model=model,
                max_tokens=self.max_tokens,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}]
            )
            
            stop_reason = getattr(response, 'stop_reason', 'unknown')
            usage = getattr(response, 'usage', None)
            if usage:
                print_current(f"🗜️ [LLM Summary] API Response - stop_reason: {stop_reason}, input_tokens: {getattr(usage, 'input_tokens', 'N/A')}, output_tokens: {getattr(usage, 'output_tokens', 'N/A')}")
            else:
                print_current(f"🗜️ [LLM Summary] API Response - stop_reason: {stop_reason}")
            
            if hasattr(response, 'content') and response.content:
                if isinstance(response.content, list):
                    return response.content[0].text
                return str(response.content)
            raise ValueError("Empty response from Anthropic API")
    
    def _call_openai(self, api_key: str, api_base: str, model: str,
                     system_prompt: str, user_prompt: str) -> str:
        """
        Call OpenAI-compatible API (supports streaming)
        """
        from openai import OpenAI
        
        client = OpenAI(api_key=api_key, base_url=api_base)
        
        if self.streaming:
            # Streaming mode
            content = ""
            with streaming_context(show_start_message=False) as printer:
                printer.write("\n🗜️ [摘要生成中] ")
                
                response = client.chat.completions.create(
                    model=model,
                    max_tokens=self.max_tokens,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    stream=True
                )
                
                finish_reason = None
                for chunk in response:
                    if chunk.choices and chunk.choices[0].delta:
                        delta = chunk.choices[0].delta
                        if hasattr(delta, 'content') and delta.content:
                            content += delta.content
                            printer.write(delta.content)
                        # Capture finish_reason from the last chunk
                        if chunk.choices[0].finish_reason:
                            finish_reason = chunk.choices[0].finish_reason
                
                printer.write("\n")
            
            # Print API response metadata
            print_current(f"🗜️ [LLM Summary] API Response - finish_reason: {finish_reason}")
            
            if not content:
                raise ValueError("Empty response from OpenAI API (streaming)")
            return content
        else:
            # Non-streaming mode
            response = client.chat.completions.create(
                model=model,
                max_tokens=self.max_tokens,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
            )
            
            if response and response.choices:
                finish_reason = response.choices[0].finish_reason
                usage = getattr(response, 'usage', None)
                if usage:
                    print_current(f"🗜️ [LLM Summary] API Response - finish_reason: {finish_reason}, prompt_tokens: {getattr(usage, 'prompt_tokens', 'N/A')}, completion_tokens: {getattr(usage, 'completion_tokens', 'N/A')}")
                else:
                    print_current(f"🗜️ [LLM Summary] API Response - finish_reason: {finish_reason}")
            
            if response and response.choices and response.choices[0].message:
                return response.choices[0].message.content
            raise ValueError("Empty response from OpenAI API")
    
    def _calculate_total_length(self, history: List[Dict[str, Any]]) -> int:
        """
        Calculate total character count of history records
        
        Args:
            history: List of history records
            
        Returns:
            Total character count
        """
        total = 0
        fields_to_count = ["prompt", "result", "content", "response", "output", "data"]
        for record in history:
            for field in fields_to_count:
                if field in record:
                    total += len(str(record[field]))
        return total
    
    def get_compression_stats(
        self,
        original_history: List[Dict[str, Any]],
        compressed_history: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Get compression statistics
        
        Args:
            original_history: Original history
            compressed_history: Compressed history
            
        Returns:
            Compression stats
        """
        original_length = self._calculate_total_length(original_history)
        compressed_length = self._calculate_total_length(compressed_history)
        
        compression_ratio = (1 - compressed_length / original_length) * 100 if original_length > 0 else 0
        saved_chars = original_length - compressed_length
        estimated_token_savings = saved_chars // 4
        
        return {
            "compression_method": "llm_summary",
            "original_chars": original_length,
            "compressed_chars": compressed_length,
            "saved_chars": saved_chars,
            "compression_ratio": compression_ratio,
            "estimated_token_savings": estimated_token_savings,
            "original_records": len(original_history),
            "compressed_records": len(compressed_history),
        }
