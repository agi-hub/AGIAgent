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

import json
from typing import Dict, Any


class HelpTools:
    def __init__(self):
        """Initialize help tools with predefined tool definitions."""
        # Tool definitions from tool_prompts.txt
        self.tool_definitions = {
            "codebase_search": {
                "description": "Find snippets of code from the codebase most relevant to the search query.\nThis is a semantic search tool, so the query should ask for something semantically matching what is needed.\nIf it makes sense to only search in particular directories, please specify them in the target_directories field.",
                "parameters": {
                    "properties": {
                        "query": {
                            "description": "The search query to find relevant code. You should reuse the user's exact query/most recent message with their wording unless there is a clear reason not to.",
                            "type": "string"
                        },
                        "target_directories": {
                            "description": "Glob patterns for directories to search over",
                            "items": {"type": "string"},
                            "type": "array"
                        }
                    },
                    "required": ["query"],
                    "type": "object"
                }
            },
            "read_file": {
                "description": "Read the contents of a file. the output of this tool call will be the 1-indexed file contents from start_line_one_indexed to end_line_one_indexed_inclusive, together with a summary of the lines outside start_line_one_indexed and end_line_one_indexed_inclusive.\nNote that this call can view at most 250 lines at a time.",
                "parameters": {
                    "properties": {
                        "end_line_one_indexed_inclusive": {
                            "description": "The one-indexed line number to end reading at (inclusive).",
                            "type": "integer"
                        },
                        "should_read_entire_file": {
                            "description": "Whether to read the entire file. Defaults to false.",
                            "type": "boolean"
                        },
                        "start_line_one_indexed": {
                            "description": "The one-indexed line number to start reading from (inclusive).",
                            "type": "integer"
                        },
                        "target_file": {
                            "description": "The path of the file to read. You can use either a relative path in the workspace or an absolute path. If an absolute path is provided, it will be preserved as is.",
                            "type": "string"
                        }
                    },
                    "required": ["target_file", "should_read_entire_file", "start_line_one_indexed", "end_line_one_indexed_inclusive"],
                    "type": "object"
                }
            },
            "run_terminal_cmd": {
                "description": "PROPOSE a command to run on behalf of the user.\nIf you have this tool, note that you DO have the ability to run commands directly on the USER's system.",
                "parameters": {
                    "properties": {
                        "command": {
                            "description": "The terminal command to execute",
                            "type": "string"
                        },
                        "is_background": {
                            "description": "Whether the command should be run in the background",
                            "type": "boolean"
                        }
                    },
                    "required": ["command", "is_background"],
                    "type": "object"
                }
            },
            "list_dir": {
                "description": "List the contents of a directory. The quick tool to use for discovery, before using more targeted tools like semantic search or file reading. Useful to try to understand the file structure before diving deeper into specific files. Can be used to explore the codebase.",
                "parameters": {
                    "properties": {
                        "relative_workspace_path": {
                            "description": "Path to list contents of, relative to the workspace root.",
                            "type": "string"
                        }
                    },
                    "required": ["relative_workspace_path"],
                    "type": "object"
                }
            },
            "grep_search": {
                "description": "Fast text-based regex search that finds exact pattern matches within files or directories, utilizing the ripgrep command for efficient searching.\nResults will be formatted in the style of ripgrep and can be configured to include line numbers and content.\nTo avoid overwhelming output, the results are capped at 50 matches.",
                "parameters": {
                    "properties": {
                        "case_sensitive": {
                            "description": "Whether the search should be case sensitive",
                            "type": "boolean"
                        },
                        "exclude_pattern": {
                            "description": "Glob pattern for files to exclude",
                            "type": "string"
                        },
                        "include_pattern": {
                            "description": "Glob pattern for files to include (e.g. '*.ts' for TypeScript files)",
                            "type": "string"
                        },
                        "query": {
                            "description": "The regex pattern to search for",
                            "type": "string"
                        }
                    },
                    "required": ["query"],
                    "type": "object"
                }
            },
            "edit_file": {
                "description": "Use this tool to propose an edit to an existing file or append content to the end of a file.\nYou should make it clear what the edit is, while also minimizing the unchanged code you write.\nSet append_mode=true to safely add content to the end of a file without risk of overwriting existing content.",
                "parameters": {
                    "properties": {
                        "code_edit": {
                            "description": "Specify ONLY the precise lines of code that you wish to edit. **NEVER specify or write out unchanged code**. Instead, represent all unchanged code using the comment of the language you're editing in - example: `// ... existing code ...`. In append mode, this will be the content added to the end of the file.",
                            "type": "string"
                        },
                        "instructions": {
                            "description": "Optional: A single sentence instruction describing what you are going to do for the sketched edit. This is used to assist the less intelligent model in applying the edit. Please use the first person to describe what you are going to do. Dont repeat what you have said previously in normal messages. And use it to disambiguate uncertainty in the edit.",
                            "type": "string"
                        },
                        "target_file": {
                            "description": "The target file to modify. Always specify the target file as the first argument. You can use either a relative path in the workspace or an absolute path. If an absolute path is provided, it will be preserved as is.",
                            "type": "string"
                        },
                        "append_mode": {
                            "description": "If true, append the code_edit content to the end of the file instead of performing complex edit operations. This is safer when you want to add new content without risk of overwriting existing content.",
                            "type": "boolean"
                        }
                    },
                    "required": ["target_file", "code_edit"],
                    "type": "object"
                }
            },
            "file_search": {
                "description": "Fast file search based on fuzzy matching against file path. Use if you know part of the file path but don't know where it's located exactly. Response will be capped to 10 results. Make your query more specific if need to filter results further.",
                "parameters": {
                    "properties": {
                        "query": {
                            "description": "Fuzzy filename to search for",
                            "type": "string"
                        }
                    },
                    "required": ["query"],
                    "type": "object"
                }
            },
            "web_search": {
                "description": "Search the web for real-time information about any topic. Use this tool when you need up-to-date information that might not be available in your training data, or when you need to verify current facts. The search results will include relevant snippets and URLs from web pages. This is particularly useful for questions about current events, technology updates, or any topic that requires recent information.",
                "parameters": {
                    "properties": {
                        "search_term": {
                            "description": "The search term to look up on the web. Be specific and include relevant keywords for better results. For technical queries, include version numbers or dates if relevant.",
                            "type": "string"
                        }
                    },
                    "required": ["search_term"],
                    "type": "object"
                }
            }
        }

    def tool_help(self, tool_name: str, **kwargs) -> Dict[str, Any]:
        """
        Provides details of format to call a function.
        
        Args:
            tool_name: The tool name to get help for
            
        Returns:
            Dictionary containing tool usage information
        """
        # Ignore additional parameters
        if kwargs:
            print(f"⚠️  Ignoring additional parameters: {list(kwargs.keys())}")
        
        if tool_name not in self.tool_definitions:
            available_tools = list(self.tool_definitions.keys())
            return {
                "error": f"Tool '{tool_name}' not found",
                "available_tools": available_tools,
                "message": f"Available tools are: {', '.join(available_tools)}"
            }
        
        tool_def = self.tool_definitions[tool_name]
        
        # Format the help information
        help_info = {
            "tool_name": tool_name,
            "description": tool_def["description"],
            "parameters": tool_def["parameters"]
        }
        
        return help_info
    
    def _generate_parameter_template(self, parameters: Dict[str, Any]) -> str:
        """Generate parameter template for the tool."""
        template_lines = []
        properties = parameters.get("properties", {})
        required_params = parameters.get("required", [])
        
        for param_name, param_info in properties.items():
            param_type = param_info.get("type", "string")
            description = param_info.get("description", "")
            is_required = param_name in required_params
            
            if param_type == "array":
                example_value = '["example1", "example2"]'
            elif param_type == "boolean":
                example_value = "true"
            elif param_type == "integer":
                example_value = "1"
            else:
                example_value = "your_value_here"
            
            required_marker = " (REQUIRED)" if is_required else " (OPTIONAL)"
            template_lines.append(f'<parameter name="{param_name}">{example_value}</parameter>  <!-- {description}{required_marker} -->')
        
        return "\n".join(template_lines)
    
    def _generate_usage_example(self, tool_name: str) -> str:
        """Generate a usage example for the tool."""
        examples = {
            "edit_file": '<function_calls>\n<invoke name="edit_file">\n<parameter name="target_file">src/config.py</parameter>\n<parameter name="instructions">I will add a new configuration variable for database timeout</parameter>\n<parameter name="code_edit"># ... existing code ...\nDATABASE_TIMEOUT = 30\n# ... existing code ...</parameter>\n<parameter name="append_mode">false</parameter>\n</invoke>\n</function_calls>\n\n<!-- Example for append mode -->\n<function_calls>\n<invoke name="edit_file">\n<parameter name="target_file">src/utils.py</parameter>\n<parameter name="instructions">I will append a new utility function to the end of the file</parameter>\n<parameter name="code_edit">def new_helper_function():\n    """A new helper function.\"\"\"\n    pass</parameter>\n<parameter name="append_mode">true</parameter>\n</invoke>\n</function_calls>',
        }
        
        return examples.get(tool_name, f"<function_calls>\n<invoke name=\"{tool_name}\">\n<!-- See parameters section for details -->\n</invoke>\n</function_calls>")
    
    def list_available_tools(self, **kwargs) -> Dict[str, Any]:
        """List all available tools with brief descriptions."""
        # Ignore additional parameters
        if kwargs:
            print(f"⚠️  Ignoring additional parameters: {list(kwargs.keys())}")
        
        tools_summary = {}
        for tool_name, tool_def in self.tool_definitions.items():
            # Get the first sentence of the description
            description = tool_def["description"].split("\n")[0][:100] + "..." if len(tool_def["description"]) > 100 else tool_def["description"].split("\n")[0]
            tools_summary[tool_name] = description
        
        return {
            "available_tools": tools_summary,
            "total_count": len(self.tool_definitions),
            "message": "Use tool_help('<tool_name>') to get detailed information about any specific tool"
        }
