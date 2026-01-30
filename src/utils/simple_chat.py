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

简单的流式问答程序，支持 OpenAI 和 Claude 接口。
- 无工具调用
- 无系统提示词
- 仅支持流式输出
"""

import sys
import os
import time
from pathlib import Path
from typing import List, Optional

# 添加项目根目录到 sys.path，以便导入 src 模块
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.config_loader import (
    get_api_base,
    get_api_key,
    get_model,
    get_temperature,
    get_max_tokens,
)


def is_claude_model(model: str, api_base: Optional[str] = None) -> bool:
    """判断是否为 Claude 模型
    
    Args:
        model: 模型名称
        api_base: API base URL（可选）
    
    Returns:
        如果模型名称包含 'claude' 或 'anthropic'，或者 api_base 以 '/anthropic' 结尾，返回 True
    """
    # 首先检查 api_base（优先级更高）
    if api_base:
        if api_base.lower().endswith('/anthropic') or 'anthropic' in api_base.lower():
            return True
    
    # 然后检查模型名称
    if model:
        if 'claude' in model.lower() or 'anthropic' in model.lower():
            return True
    
    return False


def chat_openai_streaming(client, model: str, messages: List[dict], temperature: Optional[float] = None, max_tokens: Optional[int] = None):
    """OpenAI 流式调用，返回内容和统计信息"""
    params = {
        "model": model,
        "messages": messages,
        "stream": True,
    }
    
    if temperature is not None:
        params["temperature"] = temperature
    if max_tokens is not None:
        params["max_tokens"] = max_tokens
    
    debug_mode = os.getenv("DEBUG") or os.getenv("AGIBOT_DEBUG")
    
    try:
        # 开始计时
        start_time = time.perf_counter()
        first_token_time = None
        
        response = client.chat.completions.create(**params)
        
        content = ""
        chunk_count = 0
        has_content = False
        prompt_tokens = 0
        completion_tokens = 0
        total_tokens = 0
        
        for chunk in response:
            chunk_count += 1
            
            # 调试模式：打印 chunk 信息
            if debug_mode and chunk_count <= 3:
                print(f"\n[DEBUG] Chunk {chunk_count}: {type(chunk).__name__}", file=sys.stderr)
                if hasattr(chunk, 'choices'):
                    print(f"[DEBUG]   choices: {len(chunk.choices) if chunk.choices else 0}", file=sys.stderr)
            
            # 检查是否有错误
            if hasattr(chunk, 'error'):
                raise Exception(f"API 错误: {chunk.error}")
            
            # 获取 token 使用信息（通常在最后一个 chunk 中）
            if hasattr(chunk, 'usage') and chunk.usage:
                prompt_tokens = getattr(chunk.usage, 'prompt_tokens', 0) or 0
                completion_tokens = getattr(chunk.usage, 'completion_tokens', 0) or 0
                total_tokens = getattr(chunk.usage, 'total_tokens', 0) or 0
            
            if chunk.choices and len(chunk.choices) > 0:
                choice = chunk.choices[0]
                
                # 调试模式：打印 choice 信息
                if debug_mode and chunk_count <= 3:
                    print(f"[DEBUG]   choice.delta exists: {hasattr(choice, 'delta')}", file=sys.stderr)
                    if hasattr(choice, 'delta') and choice.delta:
                        print(f"[DEBUG]   delta.content: {getattr(choice.delta, 'content', None)}", file=sys.stderr)
                
                # 检查 finish_reason
                if hasattr(choice, 'finish_reason') and choice.finish_reason:
                    if choice.finish_reason != 'stop':
                        # 非正常结束，可能是长度限制等
                        if debug_mode:
                            print(f"[DEBUG] finish_reason: {choice.finish_reason}", file=sys.stderr)
                
                # 获取 delta（流式响应中的增量内容）
                delta = choice.delta if hasattr(choice, 'delta') else None
                if delta:
                    # 检查 content 字段
                    if hasattr(delta, 'content') and delta.content is not None:
                        text = delta.content
                        # 记录第一个 token 的时间
                        if first_token_time is None:
                            first_token_time = time.perf_counter()
                        print(text, end="", flush=True)
                        content += text
                        has_content = True
        
        # 结束计时
        end_time = time.perf_counter()
        total_time = end_time - start_time
        ttft = (first_token_time - start_time) if first_token_time else None
        
        # 如果没有收到任何内容
        if not has_content:
            if chunk_count == 0:
                raise Exception("未收到任何响应数据块")
            else:
                raise Exception(f"收到了 {chunk_count} 个数据块，但没有任何文本内容。请设置 DEBUG=1 查看详细信息。")
        
        print()  # 换行
        
        # 返回内容和统计信息
        stats = {
            'total_time': total_time,
            'ttft': ttft,
            'prompt_tokens': prompt_tokens,
            'completion_tokens': completion_tokens,
            'total_tokens': total_tokens,
        }
        return content, stats
    except Exception as e:
        print(f"\n[流式调用错误] {type(e).__name__}: {e}")
        if debug_mode:
            import traceback
            traceback.print_exc()
        raise


def chat_claude_streaming(client, model: str, messages: List[dict], temperature: Optional[float] = None, max_tokens: Optional[int] = None):
    """Claude 流式调用，返回内容和统计信息"""
    params = {
        "model": model,
        "messages": messages,
    }
    
    if temperature is not None:
        params["temperature"] = temperature
    if max_tokens is not None:
        params["max_tokens"] = max_tokens
    
    try:
        # 开始计时
        start_time = time.perf_counter()
        first_token_time = None
        
        content = ""
        event_count = 0
        prompt_tokens = 0
        completion_tokens = 0
        total_tokens = 0
        
        with client.messages.stream(**params) as stream:
            for event in stream:
                event_count += 1
                event_type = getattr(event, 'type', None)
                
                # 获取 token 使用信息
                if event_type == "message_delta":
                    delta = getattr(event, 'delta', None)
                    if delta:
                        usage = getattr(delta, 'usage', None) or getattr(event, 'usage', None)
                        if usage:
                            prompt_tokens = getattr(usage, 'input_tokens', 0) or 0
                            completion_tokens = getattr(usage, 'output_tokens', 0) or 0
                            total_tokens = prompt_tokens + completion_tokens
                
                if event_type == "content_block_delta":
                    delta = getattr(event, 'delta', None)
                    if delta:
                        delta_type = getattr(delta, 'type', None)
                        if delta_type == "text_delta":
                            text = getattr(delta, 'text', '')
                            # 记录第一个 token 的时间
                            if first_token_time is None:
                                first_token_time = time.perf_counter()
                            print(text, end="", flush=True)
                            content += text
            
            # 尝试从最终消息获取完整的 token 信息
            try:
                final_message = stream.get_final_message()
                if hasattr(final_message, 'usage') and final_message.usage:
                    prompt_tokens = getattr(final_message.usage, 'input_tokens', 0) or 0
                    completion_tokens = getattr(final_message.usage, 'output_tokens', 0) or 0
                    total_tokens = prompt_tokens + completion_tokens
            except Exception:
                pass  # 如果无法获取最终消息，使用流中获取的值
        
        # 结束计时
        end_time = time.perf_counter()
        total_time = end_time - start_time
        ttft = (first_token_time - start_time) if first_token_time else None
        
        # 如果没有收到任何内容，可能是错误
        if not content and event_count == 0:
            raise Exception("未收到任何响应数据")
        
        print()  # 换行
        
        # 返回内容和统计信息
        stats = {
            'total_time': total_time,
            'ttft': ttft,
            'prompt_tokens': prompt_tokens,
            'completion_tokens': completion_tokens,
            'total_tokens': total_tokens,
        }
        return content, stats
    except Exception as e:
        print(f"\n[流式调用错误] {type(e).__name__}: {e}")
        raise


def main():
    """主函数"""
    # 从配置读取参数
    api_key = get_api_key()
    api_base = get_api_base()
    model = get_model()
    temperature = get_temperature()
    max_tokens = get_max_tokens()
    
    if not api_key:
        print("错误: 未找到 API key，请在 config/config.txt 中设置或设置环境变量 AGIBOT_API_KEY")
        sys.exit(1)
    
    if not model:
        print("错误: 未找到模型名称，请在 config/config.txt 中设置或设置环境变量 AGIBOT_MODEL")
        sys.exit(1)
    
    # 判断使用哪个接口
    is_claude = is_claude_model(model, api_base)
    
    # 初始化客户端
    try:
        if is_claude:
            from anthropic import Anthropic
            client = Anthropic(api_key=api_key, base_url=api_base)
            print(f"使用 Claude 接口，模型: {model}")
        else:
            from openai import OpenAI
            client = OpenAI(api_key=api_key, base_url=api_base)
            print(f"使用 OpenAI 接口，模型: {model}")
    except Exception as e:
        print(f"错误: 初始化客户端失败: {e}")
        sys.exit(1)
    
    print("开始聊天（输入 'exit' 或 'quit' 退出，空行发送）\n")
    
    # 消息历史
    messages: List[dict] = []
    
    # 交互循环
    while True:
        try:
            # 读取用户输入
            user_input = input("你: ").strip()
            
            if not user_input or user_input.lower() in {'exit', 'quit', 'q'}:
                print("再见！")
                break
            
            # 添加到消息历史
            messages.append({"role": "user", "content": user_input})
            
            # 调用 API
            print("助手: ", end="", flush=True)
            try:
                if is_claude:
                    response, stats = chat_claude_streaming(client, model, messages, temperature, max_tokens)
                else:
                    response, stats = chat_openai_streaming(client, model, messages, temperature, max_tokens)
                
                # 显示统计信息
                total_time = stats['total_time']
                ttft = stats['ttft']
                prompt_tokens = stats['prompt_tokens']
                completion_tokens = stats['completion_tokens']
                total_tokens = stats['total_tokens']
                
                # 计算输出 tokens/s
                output_tokens_per_sec = 0.0
                if completion_tokens > 0 and ttft is not None:
                    # 输出时间 = 总时间 - TTFT
                    output_time = total_time - ttft
                    if output_time > 0:
                        output_tokens_per_sec = completion_tokens / output_time
                
                print(f"\n[统计] 总时间: {total_time:.2f}s | ", end="")
                if ttft is not None:
                    print(f"TTFT: {ttft*1000:.0f}ms | ", end="")
                print(f"总tokens: {total_tokens} (输入: {prompt_tokens}, 输出: {completion_tokens})", end="")
                if output_tokens_per_sec > 0:
                    print(f" | 输出速度: {output_tokens_per_sec:.1f} tokens/s")
                else:
                    print()
                
                # 将助手回复添加到消息历史
                if response:
                    messages.append({"role": "assistant", "content": response})
                else:
                    print("警告: 收到空响应")
                    messages.pop()  # 移除用户消息，因为没有得到有效回复
            except Exception as e:
                import traceback
                error_msg = f"\n错误: API 调用失败: {type(e).__name__}: {e}"
                print(error_msg)
                # 在调试模式下显示完整错误信息
                if os.getenv("DEBUG") or os.getenv("AGIBOT_DEBUG"):
                    print("\n详细错误信息:")
                    traceback.print_exc()
                # 移除失败的用户消息
                messages.pop()
            
            print()  # 空行分隔
            
        except KeyboardInterrupt:
            print("\n\n再见！")
            break
        except EOFError:
            print("\n\n再见！")
            break
        except Exception as e:
            print(f"\n错误: {e}")


if __name__ == "__main__":
    main()

