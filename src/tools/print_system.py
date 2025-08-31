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

"""
AGIAgent Print System
----------------------------------
Features:
1. print_current   Write to <agent_id>.out according to agent_id, manager/None → terminal.
2. print_debug     Write to <agent_id>.log according to agent_id; manager/None → manager.log.
3. print_system    Write to agia.log.
4. streaming_context   For streaming writes (no automatic newline).
"""

import os
import builtins
import re
from contextlib import contextmanager
from typing import Optional, List
from src.tools.agent_context import get_current_agent_id, get_current_log_dir, set_current_log_dir
from src.config_loader import get_emoji_disabled 

# Emoji remove 

# Log directory (now managed via agent_context)
# _LOG_DIR: Optional[str] = None  # Removed global variable

_EMOJI_DISABLED: Optional[bool] = None

def _emoji_disabled() -> bool:
    """Detect whether to remove emoji (with cache)."""
    global _EMOJI_DISABLED
    if _EMOJI_DISABLED is None:
        try:
            _EMOJI_DISABLED = bool(get_emoji_disabled())
        except Exception:
            _EMOJI_DISABLED = False
    return _EMOJI_DISABLED

def remove_emoji(text: str) -> str:
    """Remove only emoji, keep other Unicode (such as Chinese)."""
    if not isinstance(text, str):
        return text  # type: ignore[return-value]

    emoji_pattern = (
        r'[\U00002600-\U000026FF]'   # Symbols
        r'|[\U00002700-\U000027BF]'  # Dingbats
        r'|[\U0001F600-\U0001F64F]'  # Emoticons
        r'|[\U0001F300-\U0001F5FF]'  # Misc symbols & pictographs
        r'|[\U0001F680-\U0001F6FF]'  # Transport & map
        r'|[\U0001F1E0-\U0001F1FF]'  # Regional indicators
        r'|[\U00002702-\U000027B0]'  # Dingbats (dup for legacy)
        r'|[\U0001F900-\U0001F9FF]'  # Supplemental symbols & pictographs
        r'|[\U0001FA70-\U0001FAFF]'  # Symbols & pictographs ext-A
        r'|\U0000FE0F'                # VS-16
        r'|\U0000200D'                # ZWJ
    )
    return re.sub(emoji_pattern, '', text)


def _join_message(*args: object, sep: str = ' ') -> str:
    """Join any objects into a string."""
    try:
        return sep.join(str(a) for a in args)
    except Exception:
        return sep.join([str(a) for a in args])


def _process_newlines_for_terminal(text: str) -> str:
    """Convert \n in text to real line breaks for terminal output"""
    if not isinstance(text, str):
        return text
    return text.replace('\\n', '\n')


def _write_to_file(file_path: str, content: str, newline: bool = True) -> None:
    """Append to file inside configured LOG_DIR (if any), auto-create dirs."""
    # Get LOG_DIR from context instead of global variable
    log_dir = get_current_log_dir()
    final_path = os.path.join(log_dir, file_path) if log_dir else file_path

    dir_name = os.path.dirname(final_path)
    if dir_name and not os.path.exists(dir_name):
        os.makedirs(dir_name, exist_ok=True)

    with open(final_path, 'a', encoding='utf-8', buffering=1) as fh:
        fh.write(content)
        if newline:
            fh.write('\n')
        fh.flush()


# ---------------------------------------------------------------------------
# Public helper to set log directory (called by main / multiagents)
# ---------------------------------------------------------------------------


def set_output_directory(out_dir: str) -> None:
    """Configure global log directory as <out_dir>/logs."""
    log_dir = os.path.join(out_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    set_current_log_dir(log_dir)


# Distinguish keywords allowed to pass through to builtins.print (for compatibility)
_PRINT_KWARGS = {'sep', 'end', 'file', 'flush'}

def print_current(*args: object, **kwargs) -> None:  # noqa: D401
    """Output to corresponding .out file or terminal according to current agent_id."""
    current_id = get_current_agent_id()
    message = _join_message(*args)
    if _emoji_disabled():
        message = remove_emoji(message)

    # Extract print-compatible kwargs
    print_kwargs = {k: v for k, v in kwargs.items() if k in _PRINT_KWARGS}
    end_char = print_kwargs.get('end', '\n')

    if current_id is None or current_id == 'manager':
        # Handle line breaks when outputting to terminal
        processed_message = _process_newlines_for_terminal(message)
        builtins.print(processed_message, **print_kwargs)
    else:
        _write_to_file(f"{current_id}.out", message, newline=(end_char != ''))


def print_debug(*args: object, **kwargs) -> None:  # noqa: D401
    """Write to <agent_id>.log or manager.log (not output to terminal)."""
    current_id = get_current_agent_id()
    file_name = 'manager.log' if current_id in (None, 'manager') else f"{current_id}.log"
    message = _join_message(*args)
    if _emoji_disabled():
        message = remove_emoji(message)
    end_char = kwargs.get('end', '\n')
    _write_to_file(file_name, message, newline=(end_char != ''))


def print_system(*args: object, **kwargs) -> None:  # noqa: D401
    """Write to agia.log."""
    message = _join_message(*args)
    if _emoji_disabled():
        message = remove_emoji(message)
    end_char = kwargs.get('end', '\n')
    _write_to_file('agia.log', message, newline=(end_char != ''))


class _StreamWriter:
    """Simplified streaming writer."""

    def __init__(self, agent_id: Optional[str]):
        self.agent_id = agent_id or 'manager'
        self.buffer: List[str] = []

    def write(self, text: str) -> None: 
        if not text:
            return
        processed = remove_emoji(text) if _emoji_disabled() else text
        if self.agent_id == 'manager':
            # Handle line breaks when outputting to terminal
            processed_for_terminal = _process_newlines_for_terminal(processed)
            builtins.print(processed_for_terminal, end='', flush=True)
        else:
            _write_to_file(f"{self.agent_id}.out", processed, newline=False)
        self.buffer.append(processed)

    def get_content(self) -> str:
        """Return written content (no newline)."""
        return ''.join(self.buffer)


@contextmanager
def streaming_context(show_start_message: bool = True):
    _ = show_start_message
    writer = _StreamWriter(get_current_agent_id())
    try:
        yield writer
    finally:
        pass

@contextmanager
def with_agent_print(agent_id: str):
    """Context manager to set agent context for print operations."""
    from src.tools.agent_context import set_current_agent_id, set_current_log_dir
    import os
    
    # Set the agent ID for this context
    set_current_agent_id(agent_id)
    
    # Set log directory for this agent
    # Create agent-specific log directory
    agent_log_dir = os.path.join(os.getcwd(), 'logs', agent_id)
    os.makedirs(agent_log_dir, exist_ok=True)
    set_current_log_dir(agent_log_dir)
    
    try:
        yield
    finally:
        # Reset agent ID and log directory when context exits
        set_current_agent_id(None)
        set_current_log_dir(None)

def print_error(*args, **kwargs): 
    print_current(*args, **kwargs)

# ---------------------------------------------------------------------------
# Compatibility: print_agent (used by multiagents etc.)
# ---------------------------------------------------------------------------


def print_agent(agent_id: str, *args, **kwargs):  # pragma: no cover
    """Write message directly to <agent_id>.out (emoji-handled)."""
    message = _join_message(*args)
    if _emoji_disabled():
        message = remove_emoji(message)
    end_char = kwargs.get('end', '\n')
    _write_to_file(f"{agent_id}.out", message, newline=(end_char != ''))
