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

from typing import Dict, Any


class HelpTools:
    def __init__(self):
        """Initialize help tools with current tool definitions."""
        # Updated tool definitions matching current AGIBot tool structure
        self.tool_definitions = {
            # Core Tools
            "run_terminal_cmd": {
                "description": "Execute terminal commands directly on the system. Can run commands in background for GUI applications and web servers.",
                "parameters": {
                    "properties": {
                        "command": {
                            "description": "The terminal command to execute",
                            "type": "string"
                        },
                        "is_background": {
                            "description": "Whether to run in background (true/false). Use true for GUI apps and web servers.",
                            "type": "boolean"
                        }
                    },
                    "required": ["command", "is_background"],
                    "type": "object"
                },
                "notes": "Append '| cat' for interactive commands (git, less, more, etc.)"
            },
            "read_file": {
                "description": "Read the contents of a file with specified line range or entire file. Returns file contents with line numbers and context.",
                "parameters": {
                    "properties": {
                        "target_file": {
                            "description": "File path to read (relative to workspace or absolute path)",
                            "type": "string"
                        },
                        "start_line_one_indexed": {
                            "description": "Starting line number (1-indexed, inclusive)",
                            "type": "integer"
                        },
                        "end_line_one_indexed_inclusive": {
                            "description": "Ending line number (1-indexed, inclusive)",
                            "type": "integer"
                        },
                        "should_read_entire_file": {
                            "description": "Whether to read entire file (true/false)",
                            "type": "boolean"
                        }
                    },
                    "required": ["target_file", "start_line_one_indexed", "end_line_one_indexed_inclusive", "should_read_entire_file"],
                    "type": "object"
                },
                "notes": "Can view at most 250 lines at a time"
            },
            "edit_file": {
                "description": "Edit an existing file or create a new file. Supports multiple editing modes including precise line-based editing. SAFETY: Always read file first before editing existing files.",
                "parameters": {
                    "properties": {
                        "target_file": {
                            "description": "File path to modify (relative to workspace or absolute path)",
                            "type": "string"
                        },
                        "instructions": {
                            "description": "Brief description of the edit being performed",
                            "type": "string"
                        },
                        "code_edit": {
                            "description": "Content to write, replace, insert, or append",
                            "type": "string"
                        },
                        "edit_mode": {
                            "description": "Editing mode: 'append' (SAFEST - add to end), 'replace_lines' (replace specific lines), 'insert_lines' (insert at position), 'auto' (USE WITH CAUTION - requires existing code markers), 'full_replace' (DANGEROUS - replace entire file)",
                            "type": "string"
                        },
                        "start_line": {
                            "description": "Starting line number for replace_lines mode (1-indexed, inclusive)",
                            "type": "integer"
                        },
                        "end_line": {
                            "description": "Ending line number for replace_lines mode (1-indexed, inclusive)",
                            "type": "integer"
                        },
                        "insert_position": {
                            "description": "Line number to insert before for insert_lines mode (1-indexed)",
                            "type": "integer"
                        }
                    },
                    "required": ["target_file", "instructions", "code_edit"],
                    "type": "object"
                },
                "notes": "SAFETY FIRST: Read file before editing. Use 'append' mode as default. Avoid 'auto' mode without existing code markers.",
                "warning": "NEVER use auto mode on existing files without proper existing code markers (// ... existing code ...)"
            },
            "list_dir": {
                "description": "List contents of a directory. Useful for exploring file structure before using more targeted tools.",
                "parameters": {
                    "properties": {
                        "relative_workspace_path": {
                            "description": "Directory path to list (relative to workspace root)",
                            "type": "string"
                        }
                    },
                    "required": ["relative_workspace_path"],
                    "type": "object"
                }
            },
            
            # Search Tools
            "codebase_search": {
                "description": "Semantic search to find relevant code snippets from the codebase. Uses AI-powered understanding to match query intent.",
                "parameters": {
                    "properties": {
                        "query": {
                            "description": "Search query for semantic code search",
                            "type": "string"
                        },
                        "target_directories": {
                            "description": "Array of directory patterns to search (optional)",
                            "type": "array",
                            "items": {"type": "string"}
                        }
                    },
                    "required": ["query"],
                    "type": "object"
                }
            },
            "grep_search": {
                "description": "Fast regex-based text search using ripgrep engine. Finds exact pattern matches in files.",
                "parameters": {
                    "properties": {
                        "query": {
                            "description": "Regex pattern to search for",
                            "type": "string"
                        },
                        "include_pattern": {
                            "description": "File pattern to include (e.g., '*.py') - optional",
                            "type": "string"
                        },
                        "exclude_pattern": {
                            "description": "File pattern to exclude - optional",
                            "type": "string"
                        },
                        "case_sensitive": {
                            "description": "Whether search is case sensitive - optional",
                            "type": "boolean"
                        }
                    },
                    "required": ["query"],
                    "type": "object"
                },
                "notes": "Results capped at 50 matches to avoid overwhelming output"
            },
            "file_search": {
                "description": "Fuzzy search for filenames. Use when you know part of a filename but not its exact location.",
                "parameters": {
                    "properties": {
                        "query": {
                            "description": "Fuzzy filename to search for",
                            "type": "string"
                        }
                    },
                    "required": ["query"],
                    "type": "object"
                },
                "notes": "Results capped to 10 matches. Make query more specific if needed."
            },
            "web_search": {
                "description": "Search the web for real-time information. Useful for current events, technology updates, or recent information.",
                "parameters": {
                    "properties": {
                        "search_term": {
                            "description": "Search term for web search. Be specific and include relevant keywords.",
                            "type": "string"
                        }
                    },
                    "required": ["search_term"],
                    "type": "object"
                },
                "notes": "Include version numbers or dates for technical queries"
            },
            
            # Utility Tools
            "tool_help": {
                "description": "Get detailed usage information for any available tool. Essential for self-diagnosis and learning.",
                "parameters": {
                    "properties": {
                        "tool_name": {
                            "description": "Name of the tool to get help for",
                            "type": "string"
                        }
                    },
                    "required": ["tool_name"],
                    "type": "object"
                },
                "notes": "Returns detailed parameters, examples, and usage guidelines"
            },
            
            # Extended Web Tools
            "fetch_webpage_content": {
                "description": "Fetch content from a specific URL. Use for direct access to user-provided links or detailed webpage analysis.",
                "parameters": {
                    "properties": {
                        "url": {
                            "description": "URL to fetch content from",
                            "type": "string"
                        },
                        "search_term": {
                            "description": "Search term to highlight in content - optional",
                            "type": "string"
                        }
                    },
                    "required": ["url"],
                    "type": "object"
                },
                "notes": "Returns webpage content with metadata"
            }
        }

    def tool_help(self, tool_name: str, **kwargs) -> Dict[str, Any]:
        """
        Provides detailed usage information for a specific tool.
        
        Args:
            tool_name: The tool name to get help for
            
        Returns:
            Dictionary containing comprehensive tool usage information
        """
        # Ignore additional parameters
        if kwargs:
            print(f"⚠️  Ignoring additional parameters: {list(kwargs.keys())}")
        
        if tool_name not in self.tool_definitions:
            available_tools = list(self.tool_definitions.keys())
            return {
                "error": f"Tool '{tool_name}' not found",
                "available_tools": available_tools,
                "message": f"Available tools are: {', '.join(available_tools)}",
                "suggestion": "Use list_available_tools() to see all available tools with descriptions"
            }
        
        tool_def = self.tool_definitions[tool_name]
        
        # Format the comprehensive help information
        help_info = {
            "tool_name": tool_name,
            "description": tool_def["description"],
            "parameters": tool_def["parameters"],
            "usage_example": self._generate_usage_example(tool_name),
            "parameter_template": self._generate_parameter_template(tool_def["parameters"])
        }
        
        # Add additional information if available
        if "notes" in tool_def:
            help_info["notes"] = tool_def["notes"]
        if "warning" in tool_def:
            help_info["warning"] = tool_def["warning"]
        
        return help_info
    
    def _generate_parameter_template(self, parameters: Dict[str, Any]) -> str:
        """Generate a parameter template showing how to call the tool."""
        template_lines = []
        properties = parameters.get("properties", {})
        required_params = parameters.get("required", [])
        
        for param_name, param_info in properties.items():
            param_type = param_info.get("type", "string")
            description = param_info.get("description", "")
            is_required = param_name in required_params
            
            # Generate appropriate example values
            if param_type == "array":
                example_value = '["example1", "example2"]'
            elif param_type == "boolean":
                example_value = "true"
            elif param_type == "integer":
                example_value = "1"
            else:
                if "path" in param_name.lower() or "file" in param_name.lower():
                    example_value = "path/to/file.py"
                elif "command" in param_name.lower():
                    example_value = "ls -la"
                elif "query" in param_name.lower() or "search" in param_name.lower():
                    example_value = "search query"
                elif "url" in param_name.lower():
                    example_value = "https://example.com"
                elif "edit_mode" in param_name.lower():
                    example_value = '"replace_lines"'
                elif "line" in param_name.lower() and "position" not in param_name.lower():
                    example_value = "10"
                elif "position" in param_name.lower():
                    example_value = "15"
                else:
                    example_value = "value"
            
            required_marker = " (REQUIRED)" if is_required else " (OPTIONAL)"
            template_lines.append(f'"{param_name}": {example_value}  // {description}{required_marker}')
        
        return "{\n  " + ",\n  ".join(template_lines) + "\n}"
    
    def _generate_usage_example(self, tool_name: str) -> str:
        """Generate a practical usage example for the tool."""
        examples = {
            "run_terminal_cmd": '{\n  "name": "run_terminal_cmd",\n  "arguments": {\n    "command": "python script.py",\n    "is_background": false\n  }\n}',
            "read_file": '{\n  "name": "read_file",\n  "arguments": {\n    "target_file": "src/main.py",\n    "start_line_one_indexed": 1,\n    "end_line_one_indexed_inclusive": 50,\n    "should_read_entire_file": false\n  }\n}',
            "edit_file": '// ALWAYS save content to files, NEVER output to chat!\n{\n  "name": "edit_file",\n  "arguments": {\n    "target_file": "data_analysis.py",\n    "instructions": "Create data analysis script",\n    "code_edit": "import pandas as pd\\nimport matplotlib.pyplot as plt\\n\\ndef analyze_data(filename):\\n    df = pd.read_csv(filename)\\n    return df.describe()",\n    "edit_mode": "append"\n  }\n}\n\n// Example: Creating a report file instead of chat output\n{\n  "name": "edit_file",\n  "arguments": {\n    "target_file": "project_report.md",\n    "instructions": "Create comprehensive project analysis report",\n    "code_edit": "# Project Analysis Report\\n\\n## Executive Summary\\n\\nThis report analyzes...",\n    "edit_mode": "full_replace"\n  }\n}',
            "list_dir": '{\n  "name": "list_dir",\n  "arguments": {\n    "relative_workspace_path": "src"\n  }\n}',
            "codebase_search": '{\n  "name": "codebase_search",\n  "arguments": {\n    "query": "function that handles user authentication",\n    "target_directories": ["src", "lib"]\n  }\n}',
            "grep_search": '{\n  "name": "grep_search",\n  "arguments": {\n    "query": "def.*authenticate",\n    "include_pattern": "*.py",\n    "case_sensitive": false\n  }\n}',
            "file_search": '{\n  "name": "file_search",\n  "arguments": {\n    "query": "config.py"\n  }\n}',
            "web_search": '{\n  "name": "web_search",\n  "arguments": {\n    "search_term": "Python asyncio best practices 2024"\n  }\n}',
            "tool_help": '{\n  "name": "tool_help",\n  "arguments": {\n    "tool_name": "edit_file"\n  }\n}',
            "fetch_webpage_content": '{\n  "name": "fetch_webpage_content",\n  "arguments": {\n    "url": "https://example.com/article",\n    "search_term": "important topic"\n  }\n}'
        }
        
        return examples.get(tool_name, f'{{\n  "name": "{tool_name}",\n  "arguments": {{\n    // See parameter template above\n  }}\n}}')
    
    def list_available_tools(self, **kwargs) -> Dict[str, Any]:
        """List all available tools with brief descriptions and categories."""
        # Ignore additional parameters
        if kwargs:
            print(f"⚠️  Ignoring additional parameters: {list(kwargs.keys())}")
        
        # Organize tools by category
        categories = {
            "Core Tools": ["run_terminal_cmd", "read_file", "edit_file", "list_dir"],
            "Search Tools": ["codebase_search", "grep_search", "file_search", "web_search"],
            "Utility Tools": ["tool_help"],
            "Extended Web Tools": ["fetch_webpage_content"]
        }
        
        tools_by_category = {}
        for category, tool_names in categories.items():
            tools_by_category[category] = {}
            for tool_name in tool_names:
                if tool_name in self.tool_definitions:
                    # Get the first sentence of the description
                    description = self.tool_definitions[tool_name]["description"]
                    first_sentence = description.split(".")[0] + "." if "." in description else description
                    if len(first_sentence) > 100:
                        first_sentence = first_sentence[:97] + "..."
                    tools_by_category[category][tool_name] = first_sentence
        
        return {
            "tools_by_category": tools_by_category,
            "total_count": len(self.tool_definitions),
            "available_tools": list(self.tool_definitions.keys()),
            "message": "Use tool_help('<tool_name>') to get detailed information about any specific tool",
            "categories": list(categories.keys())
        }
