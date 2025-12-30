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

Minimal Anthropic chat-based demo (streaming).

- Reads api_key/model/api_base from config/config.txt (same as ToolExecutor)
- Enables thinking output when supported by the model
- Uses the streaming interface; no tool calling / JSON / hallucination checks
- Prints the response directly to the terminal
"""

import sys
from typing import List

from anthropic import Anthropic

from src.config_loader import (
    get_api_base,
    get_api_key,
    get_enable_thinking,
    get_max_tokens,
    get_model,
    get_temperature,
    get_top_p,
)


def build_client() -> Anthropic:
    """Create an Anthropic client using the shared config."""
    api_key = get_api_key()
    api_base = get_api_base()

    if not api_key:
        raise RuntimeError("api_key is missing. Set it in config/config.txt or AGIBOT_API_KEY.")

    return Anthropic(api_key=api_key, base_url=api_base)


def collect_text_blocks(content) -> (str, str):
    """
    Split Anthropic response content into thinking text and final answer text.
    Returns (thinking_text, answer_text).
    """
    thinking_chunks: List[str] = []
    answer_chunks: List[str] = []

    for block in content:
        block_type = getattr(block, "type", None)
        if block_type == "thinking":
            # Anthropic thinking blocks expose either .text or .thinking depending on SDK version
            text = getattr(block, "text", None) or getattr(block, "thinking", None)
            if text:
                thinking_chunks.append(text)
        elif block_type == "text":
            answer_chunks.append(getattr(block, "text", ""))

    thinking_text = "\n".join(chunk for chunk in thinking_chunks if chunk).strip()
    answer_text = "".join(chunk for chunk in answer_chunks if chunk).strip()
    return thinking_text, answer_text


def chat_once(prompt: str) -> None:
    """
    Stream one user prompt to Claude and print thinking + answer.
    
    Uses the Anthropic streaming interface so long generations do not timeout.
    """
    client = build_client()

    model = get_model()
    max_tokens = get_max_tokens() or 4096
    temperature = get_temperature()
    top_p = get_top_p()
    enable_thinking = get_enable_thinking()

    # Enable thinking for reasoning-capable models
    # Note: When thinking is enabled, temperature MUST be 1.0
    if enable_thinking:
        temperature = 1.0
    
    params = {
        "model": model,
        "max_tokens": max_tokens,
        "system": "You are a helpful AI assistant.",  # Match tool_executor pattern
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
    }

    # Include top_p only when defined to avoid overriding provider defaults
    if top_p is not None:
        params["top_p"] = top_p

    if enable_thinking:
        params["thinking"] = {"type": "enabled", "budget_tokens": 10000}

    # Use event-based streaming to capture both thinking and text content
    answer_text = ""
    thinking_text = ""
    thinking_printed = False
    answer_started = False
    
    with client.messages.stream(**params) as stream:
        # Process events to capture thinking blocks separately
        for event in stream:
            event_type = getattr(event, 'type', None)
            
            # Handle content block start to identify the type
            if event_type == "content_block_start":
                content_block = getattr(event, 'content_block', None)
                if content_block:
                    block_type = getattr(content_block, 'type', None)
                    if block_type == "thinking":
                        print("🧠 Thinking...\n")
                        thinking_printed = True
                    elif block_type == "text":
                        if thinking_printed:
                            print("\n\n💬 ")  # Extra newline after thinking
                        else:
                            print("💬 ")
                        answer_started = True
            
            # Handle content block deltas (streaming content)
            elif event_type == "content_block_delta":
                delta = getattr(event, 'delta', None)
                if delta:
                    delta_type = getattr(delta, 'type', None)
                    
                    # Thinking content - print immediately
                    if delta_type == "thinking_delta":
                        thinking_chunk = getattr(delta, 'thinking', '')
                        print(thinking_chunk, end="", flush=True)
                        thinking_text += thinking_chunk
                    
                    # Text content (final answer) - print immediately
                    elif delta_type == "text_delta":
                        text = getattr(delta, 'text', '')
                        print(text, end="", flush=True)
                        answer_text += text
        
        # Get final message
        final_response = stream.get_final_message()

    print("\n")


def main() -> None:
    print("Simple Anthropic chat (chat-based, thinking enabled).")
    print("Press Enter on an empty line or type 'exit' to quit.\n")

    for line in sys.stdin:
        prompt = line.strip()
        if not prompt or prompt.lower() in {"exit", "quit"}:
            print("Bye!")
            return

        try:
            chat_once(prompt)
        except Exception as exc:  # Keep the demo simple
            print(f"Error: {exc}")

        print("\n---\n")


if __name__ == "__main__":
    main()

