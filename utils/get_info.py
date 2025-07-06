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

import os
import sys
import datetime
import platform
import subprocess
import requests
from typing import Dict, Any


def get_ip_location_info():
    """
    Get IP geolocation information
    
    Returns:
        Dict containing IP and location information
    """
    try:
        # For safety, import print_current here to avoid circular imports
        from tools.print_system import print_current
    except ImportError:
        # Fallback if print_current is not available
        def print_current(msg):
            print(msg)
    
    try:
        # Try to get public IP and location info
        response = requests.get('http://ipapi.co/json/', timeout=5)
        if response.status_code == 200:
            data = response.json()
            return {
                'ip': data.get('ip', 'Unknown'),
                'city': data.get('city', 'Unknown'),
                'region': data.get('region', 'Unknown'),
                'country': data.get('country_name', 'Unknown'),
                'country_code': data.get('country_code', 'Unknown'),
                'timezone': data.get('timezone', 'Unknown')
            }
    except Exception as e:
        print_current(f"Warning: Could not retrieve IP location info: {e}")
    
    # Fallback: try alternative service
    try:
        response = requests.get('https://ipinfo.io/json', timeout=5)
        if response.status_code == 200:
            data = response.json()
            location = data.get('loc', '').split(',')
            return {
                'ip': data.get('ip', 'Unknown'),
                'city': data.get('city', 'Unknown'),
                'region': data.get('region', 'Unknown'),
                'country': data.get('country', 'Unknown'),
                'country_code': data.get('country', 'Unknown'),
                'timezone': data.get('timezone', 'Unknown')
            }
    except Exception as e:
        print_current(f"Warning: Could not retrieve IP location info from fallback: {e}")
    
    # Final fallback: return unknown values
    return {
        'ip': 'Unknown',
        'city': 'Unknown',
        'region': 'Unknown',
        'country': 'Unknown',
        'country_code': 'Unknown',
        'timezone': 'Unknown'
    }


def check_command_available(command: str) -> bool:
    """
    Check if a command is available in the system.
    
    Args:
        command: Command name to check
        
    Returns:
        True if command is available, False otherwise
    """
    try:
        # Use 'where' on Windows, 'which' on Unix-like systems
        check_cmd = "where" if platform.system().lower() == "windows" else "which"
        result = subprocess.run([check_cmd, command], 
                              capture_output=True, 
                              text=True, 
                              timeout=5)
        return result.returncode == 0
    except Exception:
        return False


def get_system_environment_info(language: str = 'en', model: str = None, api_base: str = None) -> str:
    """
    Get system environment information.
    
    Args:
        language: Language preference ('zh' for Chinese, 'en' for English)
        model: Current model name (optional)
        api_base: API base URL (optional)
        
    Returns:
        Formatted system environment information
    """
    try:
        # For safety, import print_current here to avoid circular imports
        from tools.print_system import print_current
    except ImportError:
        # Fallback if print_current is not available
        def print_current(msg):
            print(msg)
    
    try:
        system_name = platform.system()
        system_release = platform.release()
        
        # Get Python version
        python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        
        # Check if pip is available
        pip_available = "Available" if check_command_available("pip") else "Not Available"
        
        os_instruction = f"""**Operating System Information**:
- Operating System: {system_name} {system_release}
- Python Version: {python_version}
- pip: {pip_available}"""
        
        # Add system-specific information
        if system_name.lower() == "linux":
            # For Linux: check gdb and shell type
            gdb_available = "Available" if check_command_available("gdb") else "Not Available"
            shell_type = os.environ.get('SHELL', 'Unknown').split('/')[-1] if os.environ.get('SHELL') else 'Unknown'
            
            os_instruction += f"""
- gdb: {gdb_available}
- Shell Type: {shell_type}
- Please use Linux-compatible commands and forward slashes for paths"""
        
        elif system_name.lower() == "windows":
            # For Windows: check PowerShell
            powershell_available = "Available" if check_command_available("powershell") else "Not Available"
            
            os_instruction += f"""
- PowerShell: {powershell_available}
- Please use Windows-compatible commands and backslashes for paths"""
        
        elif system_name.lower() == "darwin":  # macOS
            # For macOS: check shell type
            shell_type = os.environ.get('SHELL', 'Unknown').split('/')[-1] if os.environ.get('SHELL') else 'Unknown'
            
            os_instruction += f"""
- Shell Type: {shell_type}
- Please use macOS-compatible commands and forward slashes for paths"""
        
        # Add language instruction based on configuration
        if language == 'zh':
            language_instruction = """

**重要的语言设置指令**:
- 系统语言配置为中文，请尽量使用中文进行回复和生成报告
- 当生成分析报告、总结文档或其他输出文件时，请使用中文
- 代码注释和说明文档也请尽量使用中文
- 只有在涉及英文专业术语或代码本身时才使用英语
- 报告标题、章节名称等都应使用中文"""
        else:
            language_instruction = """

**Language Configuration**:
- System language is set to English
- Please respond and generate reports in English
- Code comments and documentation should be in English"""
        
        # Add current date information (standardized for cache consistency)
        current_date = datetime.datetime.now()
        date_instruction = f"""

**Current Date Information**:
- Current Date: {current_date.strftime('%Y-%m-%d')}
- Current Time: [STANDARDIZED_FOR_CACHE]"""
        
        # Add IP geolocation information
        location_info = get_ip_location_info()
        location_instruction = f"""

**Location Information**:
- City: {location_info['city']}
- Country: {location_info['country']}"""
        
        # Add current model information if provided
        if model and api_base:
            model_instruction = f"""

**AI Model Information**:
- Current Model: {model}
- API Base: {api_base}
- When spawning new agents, you can omit the 'model' parameter to use the same model ({model}), or specify a different model if needed"""
        else:
            model_instruction = ""
        
        return os_instruction + language_instruction + date_instruction + location_instruction + model_instruction
        
    except Exception as e:
        print_current(f"Warning: Could not retrieve system environment information: {e}")
        return ""


def format_file_size(size_bytes: int) -> str:
    """
    Format file size in human readable format.
    
    Args:
        size_bytes: Size in bytes
        
    Returns:
        Formatted size string
    """
    if size_bytes < 1024:
        return f"{size_bytes}B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f}KB"
    else:
        return f"{size_bytes / (1024 * 1024):.1f}MB"


def get_workspace_context(workspace_dir: str) -> str:
    """
    Get basic workspace context without detailed information that could cause hallucination.
    
    Args:
        workspace_dir: Path to the workspace directory
        
    Returns:
        String representation of workspace context
    """
    if not workspace_dir or not os.path.exists(workspace_dir):
        return ""
    
    try:
        # For safety, import print_current here to avoid circular imports
        from tools.print_system import print_current
    except ImportError:
        # Fallback if print_current is not available
        def print_current(msg):
            print(msg)
    
    context_parts = ["\n**Current Workspace Information:**\n"]
    
    # Define code file extensions to include
    code_extensions = {'.py', '.js', '.ts', '.jsx', '.tsx', '.java', '.c', '.cpp', '.h', '.css', '.html', '.php', '.rb', '.go', '.rs', '.swift', '.kt', '.scala', '.sql', '.sh', '.bat', '.ps1', '.yaml', '.yml', '.json', '.xml', '.md', '.txt'}
    
    # Find all code files in workspace
    code_files = []
    total_files = 0
    total_size = 0
    
    for root, dirs, files in os.walk(workspace_dir):
        # Skip hidden directories and common non-code directories
        dirs[:] = [d for d in dirs if not d.startswith('.') and d not in {'node_modules', '__pycache__', 'venv', 'env', 'build', 'dist', 'target'}]
        
        for file in files:
            if any(file.endswith(ext) for ext in code_extensions) and not file.startswith('.'):
                file_path = os.path.join(root, file)
                rel_path = os.path.relpath(file_path, workspace_dir)
                
                try:
                    # Get file size and modification time
                    stat_info = os.stat(file_path)
                    file_size = stat_info.st_size
                    
                    code_files.append({
                        'path': rel_path,
                        'size': file_size,
                    })
                    
                    total_files += 1
                    total_size += file_size
                    
                except Exception as e:
                    print_current(f"⚠️   Unable to get file information {rel_path}: {e}")
                    continue
    
    if not code_files:
        context_parts.append("No files found. Use list_dir tool to explore the workspace.\n")
        return ''.join(context_parts)
    
    # Add basic summary statistics only
    context_parts.append(f"📊 **Basic Statistics**: {total_files} files, total size {format_file_size(total_size)}\n")
    context_parts.append("⚠️ **Important**: File names and statistics shown above are for reference only.\n")
    context_parts.append("**You MUST use tools (list_dir, read_file, codebase_search) to get actual file contents before making any analysis or conclusions.**\n")
    
    return ''.join(context_parts)


def get_file_language(file_path: str) -> str:
    """
    Get the programming language for syntax highlighting based on file extension.
    
    Args:
        file_path: Path to the file
        
    Returns:
        Language identifier for syntax highlighting
    """
    ext = os.path.splitext(file_path)[1].lower()
    
    language_map = {
        '.py': 'python',
        '.js': 'javascript',
        '.ts': 'typescript',
        '.jsx': 'jsx',
        '.tsx': 'tsx',
        '.java': 'java',
        '.c': 'c',
        '.cpp': 'cpp',
        '.h': 'c',
        '.css': 'css',
        '.html': 'html',
        '.php': 'php',
        '.rb': 'ruby',
        '.go': 'go',
        '.rs': 'rust',
        '.swift': 'swift',
        '.kt': 'kotlin',
        '.scala': 'scala',
        '.sql': 'sql',
        '.sh': 'bash',
        '.bat': 'batch',
        '.ps1': 'powershell',
        '.yaml': 'yaml',
        '.yml': 'yaml',
        '.json': 'json',
        '.xml': 'xml',
        '.md': 'markdown'
    }
    
    return language_map.get(ext, 'text')


def get_workspace_info(workspace_dir: str) -> str:
    """
    Get workspace directory and context information.
    
    Args:
        workspace_dir: Path to the workspace directory
        
    Returns:
        Formatted workspace information
    """
    workspace_instruction = f"""**Important Workspace Information**:
- Workspace Directory: {workspace_dir}
- Please save all created code files and project files in this directory
- When creating or editing files, please use filenames directly, do not add prefix to paths
- The system has automatically set the correct working directory, you only need to use relative filenames. Example: If workspace is "output_xxx/workspace", use "./your_file_name" not "output_xxx/workspace/your_file_name/your_file_name"
"""
    
    # Get existing code context from workspace
    workspace_context = get_workspace_context(workspace_dir)
    
    return workspace_instruction + "\n\n" + workspace_context 