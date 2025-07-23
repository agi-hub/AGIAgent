#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from .print_system import print_system, print_current
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

from typing import Dict, Any, Optional


class HelpTools:
    def __init__(self, tool_executor=None):
        """Initialize help tools with current tool definitions."""
        # Store reference to tool executor for MCP tool access
        self.tool_executor = tool_executor
        
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
                "description": "Read the contents of a file with specified line range or entire file. When should_read_entire_file=true, line parameters are ignored and entire file is read.",
                "parameters": {
                    "properties": {
                        "target_file": {
                            "description": "File path to read (relative to workspace or absolute path)",
                            "type": "string"
                        },
                        "should_read_entire_file": {
                            "description": "Whether to read entire file. When true, line parameters are ignored. When false, line parameters are required.",
                            "type": "boolean"
                        },
                        "start_line_one_indexed": {
                            "description": "Starting line number (1-indexed, inclusive). Required when should_read_entire_file=false, ignored when should_read_entire_file=true.",
                            "type": "integer"
                        },
                        "end_line_one_indexed_inclusive": {
                            "description": "Ending line number (1-indexed, inclusive). Required when should_read_entire_file=false, ignored when should_read_entire_file=true.",
                            "type": "integer"
                        }
                    },
                    "required": ["target_file", "should_read_entire_file"],
                    "type": "object"
                },
                "notes": "When reading partial file, can view at most 250 lines at a time. When reading entire file, limited to 500 lines - if file is larger, only first 500 lines are shown with clear truncation indication including total lines count."
            },
            "edit_file": {
                "description": "Edit a file or create a new file using three modes. SAFETY: For existing files, it is recommended to read the file content first to understand the context.",
                "parameters": {
                    "properties": {
                        "target_file": {
                            "description": "The path of the file to modify (relative to workspace or absolute path)",
                            "type": "string"
                        },
                        "edit_mode": {
                            "description": "Edit mode: 'lines_replace' (intelligent mode - can use existing code markers for precise merging, or directly replace entire content), 'append' (safest - append to the end), 'full_replace' (completely replace the file)",
                            "type": "string"
                        },
                        "code_edit": {
                            "description": "Content to edit. lines_replace mode supports two usages: 1) Use '// ... existing code ...' markers for smart merging 2) Provide complete content for direct replacement. append mode: provide content to append. full_replace mode: provide complete new file content.",
                            "type": "string"
                        },
                        "instructions": {
                            "description": "A brief description of the edit operation being performed",
                            "type": "string"
                        }
                    },
                    "required": ["target_file", "edit_mode", "code_edit", "instructions"],
                    "type": "object"
                },
                "notes": "Important: lines_replace mode is most flexible, supporting both smart merging and full replacement. Prefer append mode for safety."
            },
            "list_dir": {
                "description": "List the contents of a directory. Returns files and subdirectories with basic information.",
                "parameters": {
                    "properties": {
                        "relative_workspace_path": {
                            "description": "Path to list contents of, relative to the workspace root",
                            "type": "string"
                        }
                    },
                    "required": ["relative_workspace_path"],
                    "type": "object"
                },
                "notes": "Use '.' for current directory or empty string for workspace root"
            },
            "codebase_search": {
                "description": "Semantic search that finds code by meaning, not exact text. Best for exploring unfamiliar codebases and understanding code behavior.",
                "parameters": {
                    "properties": {
                        "query": {
                            "description": "Complete question about what you want to understand (e.g., 'How does user authentication work?')",
                            "type": "string"
                        },
                        "target_directories": {
                            "description": "Single directory path to limit search scope, or empty array for entire codebase",
                            "type": "array",
                            "items": {"type": "string"}
                        }
                    },
                    "required": ["query", "target_directories"],
                    "type": "object"
                },
                "notes": "Start with broad queries like 'authentication flow' then narrow down. Ask complete questions, not just keywords."
            },
            "grep_search": {
                "description": "Fast regex-based text search using ripgrep engine. Finds exact pattern matches in files. **PERFORMANCE BEST PRACTICES**: 1) Use simple queries with 2-5 terms maximum, 2) Split complex multi-term searches into separate calls, 3) Always use exclude_pattern for better performance, 4) Prefer specific searches over broad ones.",
                "parameters": {
                    "properties": {
                        "query": {
                            "description": "Regex pattern to search for. AVOID complex patterns with 8+ '|' operators",
                            "type": "string"
                        },
                        "include_pattern": {
                            "description": "File pattern to include (e.g., '*.py') - optional",
                            "type": "string"
                        },
                        "exclude_pattern": {
                            "description": "File pattern to exclude - RECOMMENDED: use 'output_*/*|__pycache__/*|*.egg-info/*|cache/*' for better performance",
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
                "notes": "Results capped at 50 matches. For 1000+ results, split query into smaller, focused searches."
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
            "search_img": {
                "description": "通过输入的query获取多张相关图片（最多5张），图片保存到本地文件，返回图片文件列表的JSON格式。使用多搜索引擎策略：Google可用时按 Google->百度->Bing 顺序搜索，Google不可用时按 百度->Bing 顺序搜索。",
                "parameters": {
                    "properties": {
                        "query": {
                            "description": "图片搜索查询字符串，描述要查找的图片内容。请使用具体、清晰的描述词。",
                            "type": "string"
                        }
                    },
                    "required": ["query"],
                    "type": "object"
                },
                "notes": "一次搜索可保存多张相关图片，文件名包含序号区分。图片保存在workspace/web_search_result/images/目录下，返回包含所有图片信息的JSON列表。多引擎备份策略确保搜索成功率。"
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
            "spawn_agibot": {
                "description": "Spawn a new AGIBot instance to handle a specific task asynchronously. Useful for complex task decomposition and parallel execution. TIP: For most cases, omit the 'model' parameter to use the same model as the current instance automatically.",
                "parameters": {
                    "task_description": {
                        "type": "string",
                        "description": "Description of the task for the new AGIBot instance",
                        "required": True
                    },
                    "agent_id": {
                        "type": "string", 
                        "description": "Custom agent ID (optional, will auto-generate if not provided). Must match format 'agent_XXX' where XXX is a 3-digit number (e.g., 'agent_001')",
                        "required": False
                    },
                    "output_directory": {
                        "type": "string",
                        "description": "Directory where the new AGIBot should save its output (optional, will use parent's if not provided)",
                        "required": False
                    },
                    "api_key": {
                        "type": "string",
                        "description": "API key for the new instance (optional, will use current if not provided)",
                        "required": False
                    },
                    "model": {
                        "type": "string",
                        "description": "Model name for the new instance. RECOMMENDED: Leave this parameter empty to automatically use the same model as the current instance. Only specify if you need a different model.",
                        "required": False
                    },
                    "max_loops": {
                        "type": "integer",
                        "description": "Maximum number of loops for the new instance (optional, defaults to 30)",
                        "required": False
                    },
                    "shared_workspace": {
                        "type": "boolean",
                        "description": "Whether to share workspace with parent (optional, defaults to true)",
                        "required": False
                    },
                    "single_task_mode": {
                        "type": "boolean",
                        "description": "Whether to run in single task mode (optional, defaults to true)",
                        "required": False
                    },
                    "wait_for_completion": {
                        "type": "boolean",
                        "description": "Whether to wait for completion before returning (optional, defaults to false)",
                        "required": False
                    }
                },
                "notes": "Returns task_id for tracking. Use wait_for_agibot_spawns to check completion status."
            },
            "wait_for_agibot_spawns": {
                "description": "Wait for multiple spawned AGIBot instances to complete. Useful for synchronizing after launching multiple parallel tasks.",
                "parameters": {
                    "properties": {
                        "task_ids": {
                            "description": "List of task IDs to wait for (optional, will auto-discover if not provided)",
                            "type": "array",
                            "items": {"type": "string"}
                        },
                        "output_directories": {
                            "description": "List of output directories to check (optional)",
                            "type": "array",
                            "items": {"type": "string"}
                        },
                        "check_interval": {
                            "description": "Interval in seconds between status checks (default: 5)",
                            "type": "integer"
                        },
                        "max_wait_time": {
                            "description": "Maximum time to wait in seconds (default: 3600)",
                            "type": "integer"
                        }
                    },
                    "required": [],
                    "type": "object"
                },
                "notes": "If no task_ids or output_directories are provided, the tool will auto-discover all AGIBot spawn tasks in the current directory tree. Returns detailed completion status for all tasks."
            },
            "debug_thread_status": {
                "description": "Debug function: Check the status of currently active threads. Use to diagnose thread states created by spawn_agibot, especially when the program is stuck or threads terminate unexpectedly.",
                "parameters": {
                    "properties": {},
                    "required": [],
                    "type": "object"
                },
                "notes": "Returns detailed status information for all tracked threads, including thread ID, alive status, daemon thread status, etc. Primarily used for debugging thread management issues in spawn_agibot."
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
            },
            
            # User Interaction Tools
            "talk_to_user": {
                "description": "Display a question to the user and wait for keyboard input with timeout. This tool allows interactive communication with the user.",
                "parameters": {
                    "properties": {
                        "query": {
                            "description": "The question to display to the user",
                            "type": "string"
                        },
                        "timeout": {
                            "description": "Maximum time to wait for user response in seconds (default: 10 seconds)",
                            "type": "integer"
                        }
                    },
                    "required": ["query"],
                    "type": "object"
                },
                "notes": "Returns user response or 'no user response' if timeout occurs. Useful for getting user input during automated tasks."
            }
        }

    def _get_mcp_tool_definition(self, tool_name: str) -> Optional[Dict[str, Any]]:
        """Get MCP tool definition from tool executor"""
        if not self.tool_executor:
            return None
        
        try:
            # Check if it's a cli-mcp tool
            if hasattr(self.tool_executor, 'cli_mcp_client') and self.tool_executor.cli_mcp_client:
                cli_mcp_tools = self.tool_executor.cli_mcp_client.get_available_tools()
                if tool_name in cli_mcp_tools:
                    tool_def = self.tool_executor.cli_mcp_client.get_tool_definition(tool_name)
                    if tool_def:
                        return {
                            "description": tool_def.get("description", f"cli-mcp tool: {tool_name}"),
                            "parameters": tool_def.get("input_schema", {}),
                            "notes": f"MCP工具 (cli-mcp): {tool_name}. 请注意使用正确的参数格式（通常为camelCase）。",
                            "tool_type": "cli-mcp"
                        }
            
            # Check if it's a direct MCP tool
            if hasattr(self.tool_executor, 'direct_mcp_client') and self.tool_executor.direct_mcp_client:
                direct_mcp_tools = self.tool_executor.direct_mcp_client.get_available_tools()
                if tool_name in direct_mcp_tools:
                    tool_def = self.tool_executor.direct_mcp_client.get_tool_definition(tool_name)
                    if tool_def:
                        return {
                            "description": tool_def.get("description", f"SSE MCP tool: {tool_name}"),
                            "parameters": tool_def.get("inputSchema", tool_def.get("input_schema", {})),
                            "notes": f"MCP工具 (SSE): {tool_name}. 这是通过SSE协议连接的MCP工具。",
                            "tool_type": "direct-mcp"
                        }
            
        except Exception as e:
            print_current(f"⚠️ 获取MCP工具定义时出错: {e}")
        
        return None

    def _get_all_available_tools(self) -> Dict[str, str]:
        """Get all available tools including MCP tools"""
        all_tools = {}
        
        # Add built-in tools
        for tool_name, tool_def in self.tool_definitions.items():
            description = tool_def["description"]
            first_sentence = description.split(".")[0] + "." if "." in description else description
            if len(first_sentence) > 100:
                first_sentence = first_sentence[:97] + "..."
            all_tools[tool_name] = f"[内置] {first_sentence}"
        
        # Add MCP tools if tool_executor is available
        if self.tool_executor:
            try:
                # Add cli-mcp tools
                if hasattr(self.tool_executor, 'cli_mcp_client') and self.tool_executor.cli_mcp_client:
                    cli_mcp_tools = self.tool_executor.cli_mcp_client.get_available_tools()
                    for tool_name in cli_mcp_tools:
                        try:
                            tool_def = self.tool_executor.cli_mcp_client.get_tool_definition(tool_name)
                            description = tool_def.get("description", f"cli-mcp tool: {tool_name}") if tool_def else f"cli-mcp tool: {tool_name}"
                            first_sentence = description.split(".")[0] + "." if "." in description else description
                            if len(first_sentence) > 100:
                                first_sentence = first_sentence[:97] + "..."
                            all_tools[tool_name] = f"[MCP/CLI] {first_sentence}"
                        except Exception as e:
                            all_tools[tool_name] = f"[MCP/CLI] {tool_name} (error getting definition)"
                
                # Add direct MCP tools
                if hasattr(self.tool_executor, 'direct_mcp_client') and self.tool_executor.direct_mcp_client:
                    direct_mcp_tools = self.tool_executor.direct_mcp_client.get_available_tools()
                    for tool_name in direct_mcp_tools:
                        try:
                            tool_def = self.tool_executor.direct_mcp_client.get_tool_definition(tool_name)
                            description = tool_def.get("description", f"SSE MCP tool: {tool_name}") if tool_def else f"SSE MCP tool: {tool_name}"
                            first_sentence = description.split(".")[0] + "." if "." in description else description
                            if len(first_sentence) > 100:
                                first_sentence = first_sentence[:97] + "..."
                            all_tools[tool_name] = f"[MCP/SSE] {first_sentence}"
                        except Exception as e:
                            all_tools[tool_name] = f"[MCP/SSE] {tool_name} (error getting definition)"
            
            except Exception as e:
                print_current(f"⚠️ 获取MCP工具列表时出错: {e}")
        
        return all_tools

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
            print_current(f"⚠️ Ignoring additional parameters: {list(kwargs.keys())}")
        
        # Check if it's a built-in tool
        if tool_name in self.tool_definitions:
            tool_def = self.tool_definitions[tool_name]
            
            # Format the comprehensive help information
            help_info = {
                "tool_name": tool_name,
                "tool_type": "built-in",
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
        
        # Check if it's an MCP tool
        mcp_tool_def = self._get_mcp_tool_definition(tool_name)
        if mcp_tool_def:
            help_info = {
                "tool_name": tool_name,
                "tool_type": mcp_tool_def.get("tool_type", "mcp"),
                "description": mcp_tool_def["description"],
                "parameters": mcp_tool_def["parameters"],
                "usage_example": self._generate_mcp_usage_example(tool_name, mcp_tool_def),
                "parameter_template": self._generate_parameter_template(mcp_tool_def["parameters"]),
                "notes": mcp_tool_def.get("notes", "这是一个MCP (Model Context Protocol) 工具。"),
                "mcp_format_warning": "⚠️ MCP工具通常使用camelCase参数格式（如 entityType），而不是snake_case（如 entity_type）。请参考usage_example中的正确格式。"
            }
            
            return help_info
        
        # Tool not found
        all_tools = self._get_all_available_tools()
        available_tools = list(all_tools.keys())
        
        return {
            "error": f"Tool '{tool_name}' not found",
            "available_tools": available_tools,
            "all_tools_with_descriptions": all_tools,
            "message": f"Available tools are: {', '.join(available_tools)}",
            "suggestion": "Use list_available_tools() to see all available tools with descriptions"
        }

    def _generate_mcp_usage_example(self, tool_name: str, tool_def: Dict[str, Any]) -> str:
        """Generate usage example for MCP tools"""
        parameters = tool_def.get("parameters", {})
        properties = parameters.get("properties", {})
        required = parameters.get("required", [])
        
        # Build example arguments
        example_args = {}
        for param_name, param_info in properties.items():
            param_type = param_info.get("type", "string")
            
            # Generate appropriate example values
            if param_type == "array":
                if "entities" in param_name.lower():
                    example_args[param_name] = [{
                        "name": "示例实体",
                        "entityType": "Person",
                        "observations": ["相关观察信息"]
                    }]
                else:
                    example_args[param_name] = ["example1", "example2"]
            elif param_type == "boolean":
                example_args[param_name] = True
            elif param_type == "integer":
                example_args[param_name] = 1
            elif param_type == "object":
                if "parameters" in param_name.lower():
                    example_args[param_name] = {"query": "搜索关键词"}
                else:
                    example_args[param_name] = {"key": "value"}
            else:
                # String type
                if "query" in param_name.lower():
                    example_args[param_name] = "搜索关键词"
                elif "path" in param_name.lower() or "file" in param_name.lower():
                    example_args[param_name] = "/path/to/file.txt"
                elif "content" in param_name.lower():
                    example_args[param_name] = "文件内容"
                elif "name" in param_name.lower():
                    example_args[param_name] = "示例名称"
                elif "type" in param_name.lower():
                    example_args[param_name] = "Person"
                else:
                    example_args[param_name] = "示例值"
        
        # Special handling for known MCP tools
        if tool_name == "create_entities":
            example_args = {
                "entities": [{
                    "name": "用户",
                    "entityType": "Person",
                    "observations": ["喜欢吃xieo冰棍"]
                }]
            }
        elif tool_name == "write_file" or "write" in tool_name.lower():
            example_args = {
                "path": "/home/user/example.txt",
                "content": "这是一个示例文件内容\n包含多行文本"
            }
        elif tool_name == "read_file" or "read" in tool_name.lower():
            example_args = {
                "path": "/home/user/example.txt"
            }
        elif "search" in tool_name.lower():
            example_args = {
                "query": "搜索关键词",
                "language": "zh",
                "num_results": 10
            }
        
        import json
        example_json = json.dumps(example_args, ensure_ascii=False, indent=2)
        
        return f'''{{
  "name": "{tool_name}",
  "arguments": {example_json}
}}

📝 MCP工具调用格式注意事项:
- 参数名通常使用camelCase格式 (如: entityType, numResults)
- 避免使用snake_case格式 (如: entity_type, num_results)
- 确保参数类型正确匹配工具定义'''

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
                elif "start_line" in param_name.lower():
                    example_value = "10"
                elif "end_line" in param_name.lower():
                    example_value = "15"
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
            "read_file": '// Example 1: Read entire file\n{\n  "name": "read_file",\n  "arguments": {\n    "target_file": "src/main.py",\n    "should_read_entire_file": true\n  }\n}\n\n// Example 2: Read specific line range\n{\n  "name": "read_file",\n  "arguments": {\n    "target_file": "src/main.py",\n    "should_read_entire_file": false,\n    "start_line_one_indexed": 1,\n    "end_line_one_indexed_inclusive": 50\n  }\n}',
            "edit_file": '// ALWAYS save content to files, NEVER output to chat!\n// Example 1: Append mode (SAFEST)\n{\n  "name": "edit_file",\n  "arguments": {\n    "target_file": "data_analysis.py",\n    "edit_mode": "append",\n    "code_edit": "import pandas as pd\\nimport matplotlib.pyplot as plt\\n\\ndef analyze_data(filename):\\n    df = pd.read_csv(filename)\\n    return df.describe()",\n    "instructions": "Add data analysis functions to script"\n  }\n}\n\n// Example 2a: Lines replace mode with existing code markers (SMART MERGE)\n{\n  "name": "edit_file",\n  "arguments": {\n    "target_file": "calculator.py",\n    "edit_mode": "lines_replace",\n    "code_edit": "def add(a, b):\\n    return a + b\\n\\ndef multiply(a, b):\\n    return a * b\\n\\n// ... existing code ...\\n\\ndef main():\\n    print(\\"Calculator started\\")\\n    result = add(5, 3)\\n    product = multiply(4, 7)\\n    print(f\\"Results: {result}, {product}\\")\\n    print(\\"Calculator finished\\")",\n    "instructions": "Add multiply function and update main function using existing code markers"\n  }\n}\n\n// Example 2b: Lines replace mode without markers (FULL REPLACEMENT)\n{\n  "name": "edit_file",\n  "arguments": {\n    "target_file": "simple_script.py",\n    "edit_mode": "lines_replace",\n    "code_edit": "#!/usr/bin/env python3\\n\\ndef hello_world():\\n    print(\\"Hello, World!\\")\\n\\nif __name__ == \\"__main__\\":\\n    hello_world()",\n    "instructions": "Replace entire file with simple hello world script"\n  }\n}\n\n// Example 3: Full replace mode (DANGEROUS - use with caution)\n{\n  "name": "edit_file",\n  "arguments": {\n    "target_file": "project_report.md",\n    "edit_mode": "full_replace",\n    "code_edit": "# Project Analysis Report\\n\\n## Executive Summary\\n\\nThis report analyzes the project performance and provides recommendations...\\n\\n## Key Findings\\n\\n1. Performance improvements needed\\n2. Code quality is good\\n3. Documentation requires updates",\n    "instructions": "Create comprehensive project analysis report"\n  }\n}',
            "list_dir": '{\n  "name": "list_dir",\n  "arguments": {\n    "relative_workspace_path": "src"\n  }\n}',
            "codebase_search": '{\n  "name": "codebase_search",\n  "arguments": {\n    "query": "function that handles user authentication",\n    "target_directories": ["src", "lib"]\n  }\n}',
            "grep_search": '// RECOMMENDED: Split complex searches\n{\n  "name": "grep_search",\n  "arguments": {\n    "query": "openai|gpt",\n    "include_pattern": "*.py",\n    "exclude_pattern": "output_*/*|__pycache__/*|*.egg-info/*|cache/*",\n    "case_sensitive": false\n  }\n}\n\n// Follow with separate searches for other terms:\n{\n  "name": "grep_search",\n  "arguments": {\n    "query": "llm|llm_model",\n    "include_pattern": "*.py",\n    "exclude_pattern": "output_*/*|__pycache__/*|*.egg-info/*|cache/*",\n    "case_sensitive": false\n  }\n}',
            "file_search": '{\n  "name": "file_search",\n  "arguments": {\n    "query": "config.py"\n  }\n}',
            "web_search": '{\n  "name": "web_search",\n  "arguments": {\n    "search_term": "Python asyncio best practices 2024"\n  }\n}',
            "tool_help": '{\n  "name": "tool_help",\n  "arguments": {\n    "tool_name": "edit_file"\n  }\n}',
            "spawn_agibot": '{\n  "name": "spawn_agibot",\n  "arguments": {\n    "task_description": "Create a Python web scraper that extracts data from e-commerce websites",\n    "agent_id": "agent_010",\n    "max_loops": 15,\n    "shared_workspace": true,\n    "single_task_mode": true,\n    "wait_for_completion": false\n  }\n}',
            "wait_for_agibot_spawns": '{\n  "name": "wait_for_agibot_spawns",\n  "arguments": {\n    "check_interval": 10,\n    "max_wait_time": 1800\n  }\n}',
            "debug_thread_status": '{\n  "name": "debug_thread_status",\n  "arguments": {}\n}',
            "fetch_webpage_content": '{\n  "name": "fetch_webpage_content",\n  "arguments": {\n    "url": "https://example.com/article",\n    "search_term": "important topic"\n  }\n}',
            "talk_to_user": '{\n  "name": "talk_to_user",\n  "arguments": {\n    "query": "What is the capital of France?",\n    "timeout": 5\n  }\n}'
        }
        
        return examples.get(tool_name, f'{{\n  "name": "{tool_name}",\n  "arguments": {{\n    // See parameter template above\n  }}\n}}')

    def list_available_tools(self, **kwargs) -> Dict[str, Any]:
        """List all available tools with brief descriptions and categories."""
        # Ignore additional parameters
        if kwargs:
            print_current(f"⚠️ Ignoring additional parameters: {list(kwargs.keys())}")
        
        # Get all available tools
        all_tools = self._get_all_available_tools()
        
        # Organize tools by category
        categories = {
            "Core Tools": ["run_terminal_cmd", "read_file", "edit_file", "list_dir"],
            "Search Tools": ["codebase_search", "grep_search", "file_search", "web_search"],
            "Utility Tools": ["tool_help", "spawn_agibot", "wait_for_agibot_spawns", "debug_thread_status"],
            "Extended Web Tools": ["fetch_webpage_content"],
            "User Interaction Tools": ["talk_to_user"]
        }
        
        tools_by_category = {}
        
        # Add built-in tools
        for category, tool_names in categories.items():
            tools_by_category[category] = {}
            for tool_name in tool_names:
                if tool_name in all_tools:
                    tools_by_category[category][tool_name] = all_tools[tool_name]
        
        # Add MCP tools as separate categories
        mcp_cli_tools = {}
        mcp_sse_tools = {}
        
        for tool_name, description in all_tools.items():
            if "[MCP/CLI]" in description:
                mcp_cli_tools[tool_name] = description
            elif "[MCP/SSE]" in description:
                mcp_sse_tools[tool_name] = description
        
        if mcp_cli_tools:
            tools_by_category["MCP Tools (CLI)"] = mcp_cli_tools
        if mcp_sse_tools:
            tools_by_category["MCP Tools (SSE)"] = mcp_sse_tools
        
        # Calculate totals
        builtin_count = sum(len(tools) for category, tools in tools_by_category.items() if "MCP" not in category)
        mcp_count = len(mcp_cli_tools) + len(mcp_sse_tools)
        
        return {
            "tools_by_category": tools_by_category,
            "total_count": len(all_tools),
            "builtin_count": builtin_count,
            "mcp_count": mcp_count,
            "available_tools": list(all_tools.keys()),
            "message": "Use tool_help('<tool_name>') to get detailed information about any specific tool",
            "categories": list(tools_by_category.keys()),
            "mcp_note": "MCP工具支持动态加载的外部工具。请注意MCP工具通常使用camelCase参数格式。"
        }
