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

from typing import Dict, Any, List


def estimate_token_count(text: str, has_images: bool = False, model: str = "gpt-4") -> int:
    """
    Estimate token count for given text, including image tokens if present.
    This is a rough approximation based on character count and common tokenization patterns.
    
    Args:
        text: Input text to estimate tokens for
        has_images: Whether the input contains images
        model: Model name (affects token calculation)
        
    Returns:
        Estimated token count
    """
    if not text:
        return 0
    
    # Basic estimation rules:
    # - English: ~4 characters per token
    # - Chinese: ~1.5 characters per token (since Chinese characters are more dense)
    # - Code: ~3.5 characters per token (due to symbols and keywords)
    
    # Detect text type
    chinese_chars = sum(1 for char in text if '\u4e00' <= char <= '\u9fff')
    total_chars = len(text)
    
    if chinese_chars > total_chars * 0.3:  # More than 30% Chinese characters
        # Primarily Chinese text
        estimated_tokens = int(total_chars / 1.5)
    elif any(keyword in text.lower() for keyword in ['def ', 'class ', 'import ', 'function', '{', '}', '()', '=>']):
        # Likely contains code
        estimated_tokens = int(total_chars / 3.5)
    else:
        # Primarily English/Latin text
        estimated_tokens = int(total_chars / 4)
    
    # Add image tokens if present
    if has_images:
        import re
        # Look for base64 image data patterns
        base64_pattern = r'[A-Za-z0-9+/]{100,}={0,2}'
        base64_matches = re.findall(base64_pattern, text)
        
        if base64_matches:
            base64_chars = sum(len(match) for match in base64_matches)
            image_count = len(base64_matches)
            
            if "claude" in model.lower():
                # Claude: base64 encoded images use approximately 1.4 tokens per character
                image_tokens = int(base64_chars * 1.4)
            elif "gpt-4" in model.lower():
                # GPT-4 Vision: estimate based on image size and detail
                # For base64 images, estimate ~1.2 tokens per character
                image_tokens = int(base64_chars * 1.2)
            else:
                # For other models, use conservative estimate
                image_tokens = int(base64_chars * 1.0)
            
            estimated_tokens += image_tokens
            
            # Optional debug info (commented out to avoid import issues)
            # print(f"🖼️ Added {image_tokens} tokens for {image_count} images ({base64_chars} base64 chars)")
    
    # Ensure minimum of 1 token for non-empty text
    return max(1, estimated_tokens)


def analyze_cache_potential(messages: List[Dict[str, Any]], previous_messages: List[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Analyze cache potential for the current request by comparing with previous messages.
    
    Args:
        messages: Input messages sent to LLM
        previous_messages: Previous messages from last round (for cache comparison)
        
    Returns:
        Dictionary containing cache analysis results
    """
    import re
    
    try:
        # For safety, import print_current here to avoid circular imports
        try:
            from tools.print_system import print_current
        except ImportError:
            def print_current(msg):
                print(msg)
        
        # If no previous messages, no cache potential
        if not previous_messages:
            total_content = ""
            for message in messages:
                content = message.get("content", "")
                total_content += content
            
            has_images = bool(re.search(r'[A-Za-z0-9+/]{100,}={0,2}', total_content))
            total_tokens = estimate_token_count(total_content, has_images=has_images)
            
            return {
                'has_history': False,
                'total_tokens': total_tokens,
                'history_tokens': 0,
                'new_tokens': total_tokens,
                'cache_hit_potential': 0,
                'estimated_cache_tokens': 0,
                'cache_efficiency': 0
            }
        
        # Compare current messages with previous messages to find common prefix
        # This simulates how LLM API caching works: exact prefix matching
        cached_content = ""
        new_content = ""
        
        # Serialize both message arrays to strings for exact comparison
        current_serialized = ""
        previous_serialized = ""
        
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            current_serialized += f"[{role}]{content}"
        
        for msg in previous_messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            previous_serialized += f"[{role}]{content}"
        
        # Find the longest common prefix at character level
        common_chars = 0
        min_length = min(len(current_serialized), len(previous_serialized))
        
        for i in range(min_length):
            if current_serialized[i] == previous_serialized[i]:
                common_chars += 1
            else:
                break
        
        # Split content based on common prefix
        if common_chars > 0:
            cached_content = current_serialized[:common_chars]
            new_content = current_serialized[common_chars:]
        else:
            cached_content = ""
            new_content = current_serialized
        
        # Detect if content contains images (base64 data)
        has_cached_images = bool(re.search(r'[A-Za-z0-9+/]{100,}={0,2}', cached_content))
        has_new_images = bool(re.search(r'[A-Za-z0-9+/]{100,}={0,2}', new_content))
        
        # Calculate token estimates including image tokens
        cached_tokens = estimate_token_count(cached_content, has_images=has_cached_images)
        new_tokens = estimate_token_count(new_content, has_images=has_new_images)
        total_tokens = cached_tokens + new_tokens
        
        # Estimate cache hit potential based on cached ratio
        cache_hit_potential = cached_tokens / total_tokens if total_tokens > 0 else 0
        
        # Estimate cache efficiency (how likely the cache will actually be used)
        cache_efficiency = 0.9 if cache_hit_potential > 0.5 else 0.7  # High efficiency for significant cache hits
        estimated_cache_tokens = int(cached_tokens * cache_efficiency)
        
        return {
            'has_history': cached_tokens > 0,
            'total_tokens': total_tokens,
            'history_tokens': cached_tokens,  # Renamed for clarity
            'new_tokens': new_tokens,
            'cache_hit_potential': cache_hit_potential,
            'estimated_cache_tokens': estimated_cache_tokens,
            'cache_efficiency': cache_efficiency
        }
        
    except Exception as e:
        # For safety, import print_current here to avoid circular imports
        try:
            from tools.print_system import print_current
        except ImportError:
            def print_current(msg):
                print(msg)
                
        print_current(f"⚠️ Cache analysis failed: {e}")
        return {
            'has_history': False,
            'total_tokens': 0,
            'history_tokens': 0,
            'new_tokens': 0,
            'cache_hit_potential': 0,
            'estimated_cache_tokens': 0,
            'cache_efficiency': 0
        } 