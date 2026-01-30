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

Claude Shell - A shell wrapper for formatting claude command JSON stream output

Usage:
    python src/utils/claude_shell.py "your prompt"
    or
    ./src/utils/claude_shell.py "your prompt"
"""

import sys
import json
import subprocess
import argparse
import os
from pathlib import Path
from typing import Optional, Dict, Any


def extract_compact_content(obj: Dict[str, Any]) -> Optional[str]:
    """
    Extract compact content from JSON object
    
    Args:
        obj: Parsed JSON object
    
    Returns:
        Extracted compact content string, or None if no relevant content
    """
    obj_type = obj.get("type", "")
    output_parts = []
    text_parts = []  # For collecting consecutive text content
    
    # Handle assistant type messages
    if obj_type == "assistant":
        message = obj.get("message", {})
        content = message.get("content", [])
        
        for item in content:
            if isinstance(item, dict):
                item_type = item.get("type", "")
                
                # Extract text content
                if item_type == "text":
                    text = item.get("text", "")
                    if text:
                        # Convert \n escape sequences to actual newlines (safe for UTF-8)
                        text = text.replace('\\n', '\n').replace('\\t', '\t').replace('\\r', '\r')
                        text_parts.append(text)
                
                # Extract tool call information
                elif item_type == "tool_use":
                    # Output previously collected text first
                    if text_parts:
                        output_parts.append("".join(text_parts))
                        text_parts = []
                    
                    tool_name = item.get("name", "")
                    tool_input = item.get("input", {})
                    
                    if tool_name:
                        output_parts.append(f"\nTool Call: {tool_name}")
                        if tool_input:
                            # Format tool input
                            input_str = json.dumps(tool_input, indent=2, ensure_ascii=False, sort_keys=False)
                            output_parts.append(f"Input:\n{input_str}")
        
        # Output remaining text content
        if text_parts:
            output_parts.append("".join(text_parts))
    
    # Handle user type messages (tool results)
    elif obj_type == "user":
        message = obj.get("message", {})
        content = message.get("content", [])
        
        for item in content:
            if isinstance(item, dict):
                item_type = item.get("type", "")
                
                # Extract tool result
                if item_type == "tool_result":
                    result_content = item.get("content", "")
                    tool_use_id = item.get("tool_use_id", "")
                    
                    if result_content:
                        output_parts.append(f"\nTool Result: {tool_use_id[:8]}...")
                        # If result is string, output directly; if object, format output
                        if isinstance(result_content, str):
                            # Convert \n escape sequences to actual newlines (safe for UTF-8)
                            result_content = result_content.replace('\\n', '\n').replace('\\t', '\t').replace('\\r', '\r')
                            output_parts.append(result_content)
                        else:
                            result_str = json.dumps(result_content, indent=2, ensure_ascii=False, sort_keys=False)
                            output_parts.append(result_str)
    
    if output_parts:
        return "\n".join(output_parts)
    return None


def format_json_line(line: str, indent: int = 2, compact: bool = True) -> Optional[str]:
    """
    Format single-line JSON string
    
    Args:
        line: Single-line JSON string
        indent: JSON indentation spaces (only used in non-compact mode)
        compact: Whether to use compact mode
    
    Returns:
        Formatted string, or None if parsing fails or no relevant content
    """
    line = line.strip()
    if not line:
        return None
    
    try:
        # Parse JSON
        obj = json.loads(line)
        
        # Compact mode: extract only key content
        if compact:
            compact_content = extract_compact_content(obj)
            return compact_content
        
        # Full mode: format entire JSON object
        formatted = json.dumps(obj, indent=indent, ensure_ascii=False, sort_keys=False)
        return formatted
        
    except json.JSONDecodeError:
        # If not valid JSON, show in full mode, ignore in compact mode
        if not compact:
            return f"# Non-JSON content: {line}"
        return None


def run_claude(prompt: str, indent: int = 2, compact: bool = True, work_dir: Optional[str] = None) -> int:
    """
    Run claude command and format output
    
    Args:
        prompt: Prompt to send to claude
        indent: JSON indentation spaces (only used in non-compact mode)
        compact: Whether to use compact mode (default True)
        work_dir: Working directory path, if specified, command will be executed in that directory
    
    Returns:
        Process exit code
    """
    # Build command
    cmd = [
        "claude",
        "-p", prompt,
        "--output-format", "stream-json",
        "--verbose",
        "--dangerously-skip-permissions"
    ]
    
    # Handle working directory
    cwd = None
    if work_dir:
        work_path = Path(work_dir).expanduser().resolve()
        if not work_path.exists():
            print(f"Error: Specified directory does not exist: {work_dir}", file=sys.stderr)
            return 1
        if not work_path.is_dir():
            print(f"Error: Specified path is not a directory: {work_dir}", file=sys.stderr)
            return 1
        cwd = str(work_path)
    
    try:
        # Start process
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # Merge stderr to stdout
            text=True,
            bufsize=1,  # Line buffering
            universal_newlines=True,
            cwd=cwd  # Set working directory
        )
        
        # Read and format output line by line
        line_count = 0
        for line in process.stdout:
            formatted = format_json_line(line, indent, compact)
            if formatted:
                print(formatted)
                line_count += 1
        
        # Wait for process to finish
        return_code = process.wait()
        
        if line_count == 0 and not compact:
            print("# Warning: No JSON output received", file=sys.stderr)
        
        return return_code
        
    except FileNotFoundError:
        print(f"Error: 'claude' command not found. Please ensure claude is installed and in PATH.", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\n# User interrupted", file=sys.stderr)
        process.terminate()
        return 130
    except Exception as e:
        print(f"Error: Exception occurred while executing claude command: {e}", file=sys.stderr)
        return 1


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Claude Shell - Format claude command JSON stream output",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s "write an svd demo"
  %(prog)s "explain Python decorators" --full
  %(prog)s "explain Python decorators" --full --indent 4
  %(prog)s "write an svd demo" -d /path/to/directory
  %(prog)s "write an svd demo" -d ~/projects/myproject
        """
    )
    
    parser.add_argument(
        "prompt",
        nargs="?",
        help="Prompt to send to claude (if not provided, will read from stdin)"
    )
    
    parser.add_argument(
        "--indent",
        type=int,
        default=2,
        help="JSON indentation spaces (default: 2, only used in full mode)"
    )
    
    parser.add_argument(
        "--compact",
        action="store_true",
        default=True,
        help="Enable compact output mode (default enabled), only output assistant's text, tool_use's name/input and tool_result's content"
    )
    
    parser.add_argument(
        "--full",
        action="store_false",
        dest="compact",
        help="Disable compact mode, output full JSON format"
    )
    
    parser.add_argument(
        "-d", "--directory",
        type=str,
        default=None,
        help="Specify working directory, program will execute claude command in that directory (supports relative and absolute paths, supports ~ expansion)"
    )
    
    args = parser.parse_args()
    
    # Get prompt
    if args.prompt:
        prompt = args.prompt
    else:
        # Read from stdin
        if sys.stdin.isatty():
            print("Error: Please provide prompt as argument, or input via pipe", file=sys.stderr)
            parser.print_help()
            sys.exit(1)
        prompt = sys.stdin.read().strip()
        if not prompt:
            print("Error: Prompt cannot be empty", file=sys.stderr)
            sys.exit(1)
    
    # Run claude command
    exit_code = run_claude(prompt, args.indent, args.compact, args.directory)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()

