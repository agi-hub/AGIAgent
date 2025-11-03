#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Routine Utilities - Helper functions for handling routine files
"""

import os
from typing import Optional

def read_routine_content(routine_file: str) -> Optional[str]:
    """
    Read routine file content
    
    Args:
        routine_file: Path to routine file
        
    Returns:
        Routine content string or None if file doesn't exist or error occurs
    """
    if not routine_file:
        return None
        
    try:
        if os.path.exists(routine_file):
            with open(routine_file, 'r', encoding='utf-8') as f:
                content = f.read().strip()
            #print(f"📋 Loaded routine file: {routine_file}")
            return content
        else:
            #print(f"⚠️ Warning: Routine file not found: {routine_file}")
            return None
    except Exception as e:
        #print(f"❌ Error reading routine file {routine_file}: {e}")
        return None

def format_routine_for_single_task(routine_content: str) -> str:
    """
    Format routine content for single task mode
    
    Args:
        routine_content: Raw routine content
        
    Returns:
        Formatted routine content with recommended prefix
    """
    if not routine_content:
        return ""
    
    formatted_content = f"""

This is the recommended routine you should follow for this task:

{routine_content}"""
    
    return formatted_content

def format_routine_for_todo_mode(routine_content: str) -> str:
    """
    Format routine content for todo decomposition mode
    
    Args:
        routine_content: Raw routine content
        
    Returns:
        Formatted routine content for system prompt
    """
    if not routine_content:
        return ""
    
    formatted_content = f"""

## Routine Guidelines

Please follow these routine guidelines when planning and decomposing tasks:

{routine_content}

These guidelines should be considered alongside the standard task decomposition principles below."""
    
    return formatted_content

def append_routine_to_requirement(user_requirement: str, routine_file: str) -> str:
    """
    Append routine content to user requirement for single task mode
    
    Args:
        user_requirement: Original user requirement
        routine_file: Path to routine file
        
    Returns:
        Enhanced requirement with routine content appended
    """
    routine_content = read_routine_content(routine_file)
    if not routine_content:
        return user_requirement
    
    formatted_routine = format_routine_for_single_task(routine_content)
    enhanced_requirement = user_requirement + formatted_routine
    
    return enhanced_requirement