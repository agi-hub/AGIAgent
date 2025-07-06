#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright (c) 2025 AGI Bot Research Group.

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


def estimate_token_count(text: str) -> int:
    """
    Estimate token count for given text.
    This is a rough approximation based on character count and common tokenization patterns.
    
    Args:
        text: Input text to estimate tokens for
        
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
    
    # Ensure minimum of 1 token for non-empty text
    return max(1, estimated_tokens)


def estimate_cache_efficiency(content: str) -> float:
    """
    Estimate cache efficiency based on content standardization.
    
    Args:
        content: Content to analyze for cache efficiency
        
    Returns:
        Cache efficiency ratio (0.0 to 1.0)
    """
    if not content:
        return 0.0
    
    efficiency_score = 0.0
    
    # Check for standardized timestamps
    if "[STANDARDIZED_FOR_CACHE]" in content:
        efficiency_score += 0.3
    
    # Check for standardized tool execution markers
    if "Tool execution results:" in content:
        efficiency_score += 0.2
    
    # Check for consistent formatting (separators)
    separator_count = content.count("=" * 60)
    if separator_count > 0:
        efficiency_score += 0.2
    
    # Check for standardized tool result formatting
    if "## Tool" in content and "**Parameters:**" in content:
        efficiency_score += 0.2
    
    # Check for minimal dynamic content
    dynamic_indicators = ["timestamp", "time:", "date:", "ms", "seconds"]
    dynamic_count = sum(1 for indicator in dynamic_indicators if indicator.lower() in content.lower())
    if dynamic_count == 0:
        efficiency_score += 0.1
    
    return min(1.0, efficiency_score)


def analyze_cache_potential(messages: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Analyze cache potential for the current request.
    
    Args:
        messages: Input messages sent to LLM
        
    Returns:
        Dictionary containing cache analysis results
    """
    try:
        # For safety, import print_current here to avoid circular imports
        try:
            from tools.print_system import print_current
        except ImportError:
            def print_current(msg):
                print(msg)
        
        total_content = ""
        history_content = ""
        new_content = ""
        
        for message in messages:
            content = message.get("content", "")
            total_content += content
            
            # Detect history sections (marked by separators)
            if "=" * 60 in content:  # Our history separator
                # Split by history separator
                parts = content.split("=" * 60)
                if len(parts) > 1:
                    # Everything before the last separator is likely history
                    history_content += "=" * 60 + ("=" * 60).join(parts[:-1])
                    new_content += parts[-1]
                else:
                    new_content += content
            else:
                new_content += content
        
        # Calculate token estimates
        total_tokens = estimate_token_count(total_content)
        history_tokens = estimate_token_count(history_content)
        new_tokens = estimate_token_count(new_content)
        
        # Estimate cache hit potential based on history ratio
        cache_hit_potential = history_tokens / total_tokens if total_tokens > 0 else 0
        
        # Estimate how many tokens might be cached
        # Assume cache hit if history ratio is high and content is standardized
        estimated_cache_tokens = 0
        cache_efficiency = 0
        if cache_hit_potential > 0.5:  # More than 50% is history
            # Estimate cache efficiency based on content standardization
            cache_efficiency = estimate_cache_efficiency(history_content)
            estimated_cache_tokens = int(history_tokens * cache_efficiency)
        
        return {
            'has_history': history_tokens > 0,
            'total_tokens': total_tokens,
            'history_tokens': history_tokens,
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