#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from .print_system import print_system, print_current
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

import os
import json
from typing import Dict, Any, Optional


class HelpTools:
    def __init__(self, tool_executor=None):
        """Initialize help tools with current tool definitions."""
        # Store reference to tool executor for MCP tool access
        self.tool_executor = tool_executor
        
        # Load tool definitions from tool_prompt.json
        self.tool_definitions = self._load_tool_definitions()

    def _load_tool_definitions(self) -> Dict[str, Any]:
        """Load tool definitions from tool_prompt.json file."""
        try:
            # Get the directory where this file is located
            current_dir = os.path.dirname(os.path.abspath(__file__))
            # Navigate to the prompts directory
            prompts_dir = os.path.join(os.path.dirname(current_dir), '..', 'prompts')
            tool_prompt_path = os.path.join(prompts_dir, 'tool_prompt.json')
            
            with open(tool_prompt_path, 'r', encoding='utf-8') as f:
                tool_definitions = json.load(f)
            
            return tool_definitions
        except Exception as e:
            print_current(f"⚠️ Failed to load tool definitions from tool_prompt.json: {e}")
            return {}

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
                            "notes": f"MCP tool (cli-mcp): {tool_name}. Please note to use the correct parameter format (usually camelCase).",
                            "tool_type": "cli-mcp"
                        }
            
        except Exception as e:
            print_current(f"⚠️ Error getting MCP tool definition: {e}")
        
        return None

    def _get_all_available_tools(self) -> Dict[str, str]:
        """Get all available tools including MCP tools"""
        all_tools = {}
        
        # Add built-in tools from tool_prompt.json
        for tool_name, tool_def in self.tool_definitions.items():
            description = tool_def.get("description", "")
            first_sentence = description.split(".")[0] + "." if "." in description else description
            if len(first_sentence) > 100:
                first_sentence = first_sentence[:97] + "..."
            all_tools[tool_name] = f"[Built-in] {first_sentence}"
        
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
                
            
            except Exception as e:
                print_current(f"⚠️ Error getting MCP tool list: {e}")
        
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
                "parameter_template": self._generate_parameter_template(tool_def["parameters"])
            }
            

            
            return help_info
        
        # Check if it's an MCP tool
        mcp_tool_def = self._get_mcp_tool_definition(tool_name)
        if mcp_tool_def:
            help_info = {
                "tool_name": tool_name,
                "tool_type": mcp_tool_def.get("tool_type", "mcp"),
                "description": mcp_tool_def["description"],
                "parameters": mcp_tool_def["parameters"],
                "parameter_template": self._generate_parameter_template(mcp_tool_def["parameters"]),
                "notes": mcp_tool_def.get("notes", "This is an MCP (Model Context Protocol) tool."),
                "mcp_format_warning": "⚠️ MCP tools usually use camelCase parameter format (like entityType)"
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

    def list_available_tools(self, **kwargs) -> Dict[str, Any]:
        """List all available tools with brief descriptions and categories."""
        # Ignore additional parameters
        if kwargs:
            print_current(f"⚠️ Ignoring additional parameters: {list(kwargs.keys())}")
        
        # Get all available tools
        all_tools = self._get_all_available_tools()
        
        # Organize tools by category dynamically
        tools_by_category = {}
        
        # Add all built-in tools to a single category
        builtin_tools = {}
        for tool_name, description in all_tools.items():
            if "[Built-in]" in description:
                builtin_tools[tool_name] = description
        
        if builtin_tools:
            tools_by_category["Built-in Tools"] = builtin_tools
        
        # Add MCP tools as separate categories
        mcp_cli_tools = {}
        
        for tool_name, description in all_tools.items():
            if "[MCP/CLI]" in description:
                mcp_cli_tools[tool_name] = description
        
        if mcp_cli_tools:
            tools_by_category["MCP Tools (CLI)"] = mcp_cli_tools
        
        # Calculate totals
        builtin_count = sum(len(tools) for category, tools in tools_by_category.items() if "MCP" not in category)
        mcp_count = len(mcp_cli_tools)
        
        return {
            "tools_by_category": tools_by_category,
            "total_count": len(all_tools),
            "builtin_count": builtin_count,
            "mcp_count": mcp_count,
            "available_tools": list(all_tools.keys()),
            "message": "Use tool_help('<tool_name>') to get detailed information about any specific tool",
            "categories": list(tools_by_category.keys()),
            "mcp_note": "MCP tools support dynamically loaded external tools. Please note that MCP tools usually use camelCase parameter format."
        }
