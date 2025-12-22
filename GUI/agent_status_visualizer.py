#!/usr/bin/env python3
"""
Agent Status Visualizer - Real-time web-based visualization of agent execution status
Displays agents as UML entities and messages as communication arrows between them
"""

import os
import json
import glob
import re
import hashlib
from pathlib import Path
from datetime import datetime
from flask import Flask, jsonify, send_from_directory
import argparse

# Import Mermaid processor
try:
    from src.tools.mermaid_processor import mermaid_processor
    MERMAID_PROCESSOR_AVAILABLE = True
except ImportError:
    try:
        import sys
        # Try to add parent directory to path
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
        from src.tools.mermaid_processor import mermaid_processor
        MERMAID_PROCESSOR_AVAILABLE = True
    except ImportError:
        MERMAID_PROCESSOR_AVAILABLE = False
        print("⚠️ Mermaid processor not available")

app = Flask(__name__)

# Global variable to store the output directory path
OUTPUT_DIR = None


def find_status_files(output_dir):
    """Find all agent status files in the output directory"""
    pattern = os.path.join(output_dir, '.agia_spawn_*_status.json')
    status_files = glob.glob(pattern)
    return status_files


def load_status_file(filepath):
    """Load and parse a status file"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading status file {filepath}: {e}")
        return None


def find_message_files(output_dir):
    """Find all message files in mailboxes - optimized version"""
    messages = []
    mailboxes_dir = os.path.join(output_dir, 'mailboxes')
    
    if not os.path.exists(mailboxes_dir):
        return messages
    
    # Search in all inbox directories
    inbox_pattern = os.path.join(mailboxes_dir, '*', 'inbox', '*.json')
    try:
        message_files = glob.glob(inbox_pattern)
    except Exception as e:
        print(f"Error searching for message files: {e}")
        return messages
    
    # Limit the number of files to process to avoid timeout
    max_files = 10000  # Reasonable limit
    if len(message_files) > max_files:
        print(f"Warning: Too many message files ({len(message_files)}), processing first {max_files}")
        message_files = message_files[:max_files]
    
    # Use list comprehension and batch processing for better performance
    # Read files in parallel would be ideal, but for simplicity, we'll optimize the current approach
    for msg_file in message_files:
        try:
            # Use smaller buffer size for faster reads on small JSON files
            with open(msg_file, 'r', encoding='utf-8', buffering=8192) as f:
                msg_data = json.load(f)
                # Only add essential fields to reduce memory usage
                if msg_data:
                    messages.append(msg_data)
        except (json.JSONDecodeError, IOError, OSError) as e:
            # Skip problematic files silently to avoid log spam
            continue
    
    return messages


def find_status_updates(output_dir):
    """Find status updates from agent status files"""
    status_updates = []
    status_files = find_status_files(output_dir)
    
    for status_file in status_files:
        try:
            status_data = load_status_file(status_file)
            if not status_data:
                continue
            
            agent_id = status_data.get('agent_id', 'unknown')
            status = status_data.get('status', 'unknown')
            
            # Get timestamp for status update
            # Use completion_time if available, otherwise use last_loop_update or start_time
            timestamp = None
            if status_data.get('completion_time'):
                timestamp = status_data.get('completion_time')
            elif status_data.get('last_loop_update'):
                timestamp = status_data.get('last_loop_update')
            elif status_data.get('start_time'):
                timestamp = status_data.get('start_time')
            
            # Only add status updates for non-running states (success, completed, failed, etc.)
            if status and status != 'running' and timestamp:
                status_updates.append({
                    'agent_id': agent_id,
                    'status': status,
                    'timestamp': timestamp,
                    'completion_time': status_data.get('completion_time'),
                    'current_loop': status_data.get('current_loop', 0)
                })
        except Exception as e:
            print(f"Error processing status file {status_file}: {e}")
            continue
    
    # Sort by timestamp
    status_updates.sort(key=lambda x: x.get('timestamp', ''))
    return status_updates


def find_tool_calls_from_logs(output_dir):
    """Find all tool calls from agent log files (.out and .log files)"""
    tool_calls = []
    logs_dir = os.path.join(output_dir, 'logs')
    
    if not os.path.exists(logs_dir):
        return tool_calls
    
    # Pattern to match both formats:
    # "Tool {tool_name} at {timestamp} with parameters: {params}" (old text format)
    # "Tool {tool_name} with parameters: {params}" (old text format, fallback)
    tool_pattern_with_timestamp = re.compile(r'Tool\s+(\w+)\s+at\s+([^\s]+)\s+with\s+parameters:\s+(.+)$')
    tool_pattern_without_timestamp = re.compile(r'Tool\s+(\w+)\s+with\s+parameters:\s+(.+)$')
    
    # Find all .out files (where print_current writes tool calls as JSON)
    agent_out_files = glob.glob(os.path.join(logs_dir, 'agent_*.out'))
    manager_out_file = os.path.join(logs_dir, 'manager.out')
    out_files = agent_out_files.copy()
    if os.path.exists(manager_out_file):
        out_files.append(manager_out_file)
    
    # Also find .log files for backward compatibility
    agent_log_files = glob.glob(os.path.join(logs_dir, 'agent_*.log'))
    manager_log_file = os.path.join(logs_dir, 'manager.log')
    log_files = agent_log_files.copy()
    if os.path.exists(manager_log_file):
        log_files.append(manager_log_file)
    
    # Process .out files (JSON format)
    for out_file in out_files:
        # Extract agent_id from filename (e.g., agent_001.out -> agent_001, manager.out -> manager)
        filename = os.path.basename(out_file)
        if filename == 'manager.out':
            agent_id = 'manager'
        else:
            agent_id = filename.replace('.out', '')
        
        try:
            # Get file modification time as approximate timestamp
            file_mtime = os.path.getmtime(out_file)
            file_mtime_iso = datetime.fromtimestamp(file_mtime).isoformat()
            
            with open(out_file, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                # Look for JSON code blocks containing tool calls
                # Pattern: ```json ... {"tool_name": "...", "parameters": {...}} ... ```
                json_block_pattern = re.compile(r'```json\s*\n(.*?)\n```', re.DOTALL)
                json_blocks = list(json_block_pattern.finditer(content))
                
                # Track position of each JSON block in file for better ordering
                for block_idx, json_block_match in enumerate(json_blocks):
                    json_str = json_block_match.group(1).strip()
                    try:
                        tool_call_data = json.loads(json_str)
                        if isinstance(tool_call_data, dict) and 'tool_name' in tool_call_data:
                            tool_name = tool_call_data.get('tool_name', 'unknown')
                            tool_params = tool_call_data.get('parameters', {})
                            
                            # Convert parameters dict to string for display
                            if isinstance(tool_params, dict):
                                # Remove code_edit parameter for edit_file to reduce size
                                params_for_display = tool_params.copy()
                                if tool_name == 'edit_file' and 'code_edit' in params_for_display:
                                    params_for_display.pop('code_edit')
                                params_str = json.dumps(params_for_display, ensure_ascii=False)
                            else:
                                params_str = str(tool_params)
                            
                            # Use file modification time as timestamp
                            # For multiple tools in same file, add small offset based on block position
                            # to maintain correct ordering
                            timestamp_str = file_mtime_iso
                            if block_idx > 0:
                                # Add microseconds offset: block_idx * 1000 microseconds
                                # This ensures tools in the same file are ordered correctly
                                try:
                                    base_dt = datetime.fromisoformat(file_mtime_iso.replace('Z', '+00:00'))
                                    from datetime import timedelta
                                    offset_dt = base_dt + timedelta(microseconds=block_idx * 1000)
                                    timestamp_str = offset_dt.isoformat()
                                except:
                                    # Fallback: just use base timestamp
                                    pass
                            
                            tool_calls.append({
                                'agent_id': agent_id,
                                'tool_name': tool_name,
                                'timestamp': timestamp_str,
                                'parameters': params_str,
                                'line_number': 0  # JSON blocks span multiple lines
                            })
                    except json.JSONDecodeError:
                        # Skip invalid JSON blocks
                        continue
        except Exception as e:
            print(f"Error reading out file {out_file}: {e}")
            continue
    
    # Process .log files (text format, for backward compatibility)
    for log_file in log_files:
        # Extract agent_id from filename (e.g., agent_001.log -> agent_001, manager.log -> manager)
        filename = os.path.basename(log_file)
        if filename == 'manager.log':
            agent_id = 'manager'
        else:
            agent_id = filename.replace('.log', '')
        
        try:
            with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                for line_num, line in enumerate(f, 1):
                    # Try new format first (with timestamp)
                    match = tool_pattern_with_timestamp.search(line)
                    if match:
                        tool_name = match.group(1)
                        timestamp_str = match.group(2)
                        params_str = match.group(3)
                    else:
                        # Fallback to old format (without timestamp)
                        match = tool_pattern_without_timestamp.search(line)
                        if match:
                            tool_name = match.group(1)
                            timestamp_str = ''  # No timestamp in old format
                            params_str = match.group(2)
                        else:
                            continue  # No match, skip this line
                    
                    # Process the matched tool call
                    # Parse parameters (remove code_edit for edit_file)
                    try:
                        # Try to parse as JSON-like dict
                        params_str_clean = params_str.strip()
                        if params_str_clean.startswith('{') and params_str_clean.endswith('}'):
                            # Remove code_edit parameter for edit_file
                            if tool_name == 'edit_file':
                                # Use regex to remove code_edit parameter (handles both single and double quotes)
                                # Match: 'code_edit': ... or "code_edit": ...
                                params_str_clean = re.sub(r"['\"]code_edit['\"]\s*:\s*[^,}]+(?:,\s*)?", "", params_str_clean)
                                # Also handle multi-line code_edit values
                                params_str_clean = re.sub(r"['\"]code_edit['\"]\s*:\s*\"[^\"]*\"(?:,\s*)?", "", params_str_clean)
                                params_str_clean = re.sub(r"['\"]code_edit['\"]\s*:\s*'[^']*'(?:,\s*)?", "", params_str_clean)
                                # Clean up double commas and trailing/leading commas
                                params_str_clean = re.sub(r",\s*,", ",", params_str_clean)  # Remove double commas
                                params_str_clean = re.sub(r",\s*}", "}", params_str_clean)  # Remove trailing comma
                                params_str_clean = re.sub(r"{\s*,", "{", params_str_clean)  # Remove leading comma
                        
                        tool_calls.append({
                            'agent_id': agent_id,
                            'tool_name': tool_name,
                            'timestamp': timestamp_str,
                            'parameters': params_str_clean,
                            'line_number': line_num
                        })
                    except Exception as e:
                        # If parsing fails, still add the tool call with raw params (but try to remove code_edit)
                        params_str_clean = params_str
                        if tool_name == 'edit_file':
                            # Simple removal of code_edit line
                            params_str_clean = re.sub(r"['\"]code_edit['\"]\s*:\s*[^,}]+(?:,\s*)?", "", params_str_clean)
                        tool_calls.append({
                            'agent_id': agent_id,
                            'tool_name': tool_name,
                            'timestamp': timestamp_str,
                            'parameters': params_str_clean,
                            'line_number': line_num
                        })
        except Exception as e:
            print(f"Error reading log file {log_file}: {e}")
            continue
    
    # Sort by timestamp (empty timestamps will be sorted first, then by agent_id)
    tool_calls.sort(key=lambda x: (x.get('timestamp', '') or '', x.get('agent_id', '')))
    return tool_calls


def find_mermaid_figures_from_plan(output_dir):
    """Find all mermaid figure paths from plan.md
    
    This function:
    1. First tries to find existing image references in plan.md (legacy support)
    2. If no images found, parses mermaid code blocks and generates images
    """
    figures = []
    workspace_dir = os.path.join(output_dir, 'workspace')
    plan_md_path = os.path.join(workspace_dir, 'plan.md')
    
    if not os.path.exists(plan_md_path):
        return figures
    
    try:
        with open(plan_md_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
            
            # First, try to find existing image references (legacy support)
            # Pattern to match: ![Figure X](path/to/image.svg)
            # Matches both SVG and PNG formats
            figure_pattern = re.compile(r'!\[Figure\s+(\d+)\]\(([^)]+\.(?:svg|png))\)', re.IGNORECASE)
            existing_figures = list(figure_pattern.finditer(content))
            
            if existing_figures:
                # Use existing image references
                for match in existing_figures:
                    figure_num = match.group(1)
                    image_path = match.group(2)
                    
                    # Convert relative path to absolute path
                    if not os.path.isabs(image_path):
                        # If path starts with images/, it's relative to workspace
                        if image_path.startswith('images/'):
                            abs_image_path = os.path.join(workspace_dir, image_path)
                        else:
                            abs_image_path = os.path.join(workspace_dir, image_path)
                    else:
                        abs_image_path = image_path
                    
                    # Check if file exists
                    if os.path.exists(abs_image_path):
                        # Use relative path from output_dir for serving
                        rel_path = os.path.relpath(abs_image_path, output_dir)
                        figures.append({
                            'figure_number': figure_num,
                            'path': rel_path,
                            'absolute_path': abs_image_path,
                            'filename': os.path.basename(image_path)
                        })
            else:
                # No existing images found, parse mermaid code blocks and generate images
                # Pattern to match mermaid code blocks: ```mermaid ... ```
                mermaid_pattern = re.compile(r'```mermaid\s*\n(.*?)\n```', re.DOTALL)
                mermaid_blocks = list(mermaid_pattern.finditer(content))
                
                if mermaid_blocks and MERMAID_PROCESSOR_AVAILABLE:
                    # Create images directory if it doesn't exist
                    images_dir = os.path.join(workspace_dir, 'images')
                    os.makedirs(images_dir, exist_ok=True)
                    
                    # Process each mermaid block
                    for idx, match in enumerate(mermaid_blocks, start=1):
                        mermaid_code = match.group(1).strip()
                        
                        if not mermaid_code:
                            continue
                        
                        # Generate filename for the image
                        # Use hash of mermaid code for unique filename
                        hash_object = hashlib.sha256(mermaid_code.encode('utf-8'))
                        hash_hex = hash_object.hexdigest()[:16]
                        filename_base = f"mermaid_plan_{hash_hex}"
                        
                        svg_path = os.path.join(images_dir, f"{filename_base}.svg")
                        png_path = os.path.join(images_dir, f"{filename_base}.png")
                        
                        # Generate images using mermaid processor
                        try:
                            svg_success, png_success = mermaid_processor._generate_mermaid_image(
                                mermaid_code,
                                Path(svg_path),
                                Path(png_path)
                            )
                            
                            # Prefer SVG, fallback to PNG
                            if svg_success and os.path.exists(svg_path):
                                rel_path = os.path.relpath(svg_path, output_dir)
                                figures.append({
                                    'figure_number': str(idx),
                                    'path': rel_path,
                                    'absolute_path': svg_path,
                                    'filename': os.path.basename(svg_path)
                                })
                            elif png_success and os.path.exists(png_path):
                                rel_path = os.path.relpath(png_path, output_dir)
                                figures.append({
                                    'figure_number': str(idx),
                                    'path': rel_path,
                                    'absolute_path': png_path,
                                    'filename': os.path.basename(png_path)
                                })
                        except Exception as e:
                            print(f"Error generating mermaid image {idx}: {e}")
                            continue
                elif mermaid_blocks and not MERMAID_PROCESSOR_AVAILABLE:
                    print("⚠️ Mermaid code blocks found in plan.md but mermaid processor is not available")
                    
    except Exception as e:
        print(f"Error reading plan.md: {e}")
        import traceback
        traceback.print_exc()
    
    # Sort by figure number
    figures.sort(key=lambda x: int(x.get('figure_number', 0)))
    return figures


def get_agent_round(agent_id, status_data):
    """Get the current round/loop number for an agent"""
    if status_data and 'current_loop' in status_data:
        return status_data.get('current_loop', 0)
    return 0


def organize_messages_by_round(messages, agent_statuses):
    """Organize messages by round based on agent loop numbers and timestamps"""
    # Group messages by approximate round
    # We'll use timestamps and agent loop numbers to estimate rounds
    rounds = {}
    
    # Sort messages by timestamp
    sorted_messages = sorted(messages, key=lambda x: x.get('timestamp', ''))
    
    # Track message sequence to better estimate rounds
    message_sequence = []
    
    for msg in sorted_messages:
        sender_id = msg.get('sender_id', 'unknown')
        receiver_id = msg.get('receiver_id', 'unknown')
        timestamp = msg.get('timestamp', '')
        
        # Try to determine round from sender's current loop
        sender_status = agent_statuses.get(sender_id, {})
        sender_round = get_agent_round(sender_id, sender_status)
        
        # Use receiver's round if sender is manager (manager might not have status file)
        if sender_id == 'manager' or sender_round == 0:
            receiver_status = agent_statuses.get(receiver_id, {})
            receiver_round = get_agent_round(receiver_id, receiver_status)
            round_num = receiver_round
        else:
            round_num = sender_round
        
        # If we can't determine round from status, try to infer from message sequence
        if round_num == 0 and message_sequence:
            # Use the round of the previous message in the sequence
            round_num = message_sequence[-1].get('estimated_round', 1)
        
        # Ensure round_num is at least 1
        round_num = max(1, round_num)
        
        # Store estimated round with message
        msg['estimated_round'] = round_num
        message_sequence.append(msg)
        
        if round_num not in rounds:
            rounds[round_num] = []
        
        rounds[round_num].append(msg)
    
    return rounds


@app.route('/api/reload', methods=['POST'])
def reload_directory():
    """API endpoint to reload and find the latest output directory"""
    global OUTPUT_DIR
    
    # Search in the script's directory (where agent_status_visualizer.py is located)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    new_output_dir = find_latest_output_dir(script_dir)
    
    if new_output_dir and os.path.exists(new_output_dir):
        OUTPUT_DIR = os.path.abspath(new_output_dir)
        return jsonify({
            'success': True,
            'output_directory': OUTPUT_DIR,
            'message': f'Reloaded: {os.path.basename(OUTPUT_DIR)}'
        })
    else:
        return jsonify({
            'success': False,
            'message': 'No output directory found',
            'output_directory': OUTPUT_DIR or 'Not set'
        }), 404

@app.route('/api/status')
def get_status():
    """API endpoint to get current agent statuses and messages"""
    try:
        # Always return output_directory, even if not set
        output_dir = OUTPUT_DIR if OUTPUT_DIR else None
        
        if not output_dir or not os.path.exists(output_dir):
            return jsonify({
                'error': 'Output directory not found',
                'agents': {},
                'messages': [],
                'agent_ids': [],
                'output_directory': output_dir or '未设置',
                'timestamp': datetime.now().isoformat()
            }), 404
        
        # Load all agent statuses
        status_files = find_status_files(output_dir)
        agent_statuses = {}
        
        for status_file in status_files:
            status_data = load_status_file(status_file)
            if status_data:
                agent_id = status_data.get('agent_id', 'unknown')
                agent_statuses[agent_id] = status_data
        
        # Also add manager if not present (manager might not have status file)
        if 'manager' not in agent_statuses:
            agent_statuses['manager'] = {
                'agent_id': 'manager',
                'status': 'running',
                'current_loop': 0
            }
        
        # Load all messages (this might take time if there are many files)
        messages = find_message_files(output_dir)
        
        # Sort messages by timestamp (chronological order)
        sorted_messages = sorted(messages, key=lambda x: x.get('timestamp', '') or '')
        
        # Load tool calls from log files
        tool_calls = find_tool_calls_from_logs(output_dir)
        print(f"Found {len(tool_calls)} tool calls from log files")
        
        # Load mermaid figures from plan.md
        mermaid_figures = find_mermaid_figures_from_plan(output_dir)
        print(f"Found {len(mermaid_figures)} mermaid figures from plan.md")
        
        # Load status updates from status files
        status_updates = find_status_updates(output_dir)
        print(f"Found {len(status_updates)} status updates")
        
        # Get all unique agent IDs
        agent_ids = set(agent_statuses.keys())
        for msg in messages:
            agent_ids.add(msg.get('sender_id', ''))
            agent_ids.add(msg.get('receiver_id', ''))
        agent_ids = sorted([aid for aid in agent_ids if aid])
        
        return jsonify({
            'agents': agent_statuses,
            'messages': sorted_messages,  # Send sorted messages instead of by round
            'tool_calls': tool_calls,  # Add tool calls from logs
            'status_updates': status_updates,  # Add status updates from status files
            'mermaid_figures': mermaid_figures,  # Add mermaid figures from plan.md
            'agent_ids': agent_ids,
            'output_directory': output_dir,
            'timestamp': datetime.now().isoformat(),
            'message_count': len(sorted_messages)
        })
    except Exception as e:
        import traceback
        error_msg = str(e)
        traceback.print_exc()
        return jsonify({
            'error': f'Error loading status: {error_msg}',
            'agents': {},
            'messages': [],
            'agent_ids': [],
            'output_directory': OUTPUT_DIR or '未设置',
            'timestamp': datetime.now().isoformat()
        }), 500


@app.route('/')
def index():
    """Serve the main HTML page"""
    html_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'agent_status_visualizer.html')
    if not os.path.exists(html_path):
        return f"Error: HTML file not found at {html_path}", 404
    try:
        return send_from_directory(os.path.dirname(os.path.abspath(__file__)), 'agent_status_visualizer.html')
    except Exception as e:
        return f"Error serving HTML file: {str(e)}", 500


@app.route('/<path:path>')
def serve_static(path):
    """Serve static files"""
    return send_from_directory(os.path.dirname(os.path.abspath(__file__)), path)


@app.route('/api/files/<path:path>')
@app.route('/api/agent-status-files/<path:path>')
def serve_output_file(path):
    """Serve files from output directory (for mermaid images)"""
    if not OUTPUT_DIR:
        return jsonify({'error': 'Output directory not set'}), 404
    
    # Construct full path
    file_path = os.path.join(OUTPUT_DIR, path)
    
    # Security check: ensure path is within OUTPUT_DIR
    if not os.path.abspath(file_path).startswith(os.path.abspath(OUTPUT_DIR)):
        return jsonify({'error': 'Invalid path'}), 403
    
    if not os.path.exists(file_path):
        return jsonify({'error': 'File not found'}), 404
    
    return send_from_directory(OUTPUT_DIR, path)


def find_latest_output_dir(search_dir=None):
    """Find the latest output directory matching output_* pattern
    
    Looks for directories matching output_YYYYMMDD_HHMMSS format
    """
    if search_dir is None:
        search_dir = os.getcwd()
    
    # Look for directories matching output_* pattern
    output_dirs = []
    
    # Search in current directory
    if os.path.exists(search_dir):
        for item in os.listdir(search_dir):
            item_path = os.path.join(search_dir, item)
            if os.path.isdir(item_path) and item.startswith('output_'):
                # Check if it contains mailboxes or status files (indicating it's a valid output dir)
                has_mailboxes = os.path.exists(os.path.join(item_path, 'mailboxes'))
                has_status = len(glob.glob(os.path.join(item_path, '.agia_spawn_*_status.json'))) > 0
                # Also check for manager status file
                has_manager_status = os.path.exists(os.path.join(item_path, '.agia_spawn_manager_status.json'))
                if has_mailboxes or has_status or has_manager_status:
                    output_dirs.append(item_path)
    
    if not output_dirs:
        return None
    
    # Try to sort by timestamp in directory name first (output_YYYYMMDD_HHMMSS)
    # If timestamp parsing fails, fall back to modification time
    def get_sort_key(dir_path):
        dir_name = os.path.basename(dir_path)
        # Try to extract timestamp from directory name (output_YYYYMMDD_HHMMSS)
        if dir_name.startswith('output_'):
            timestamp_str = dir_name[7:]  # Remove 'output_' prefix
            try:
                # Parse YYYYMMDD_HHMMSS format
                if '_' in timestamp_str:
                    date_part, time_part = timestamp_str.split('_', 1)
                    if len(date_part) == 8 and len(time_part) == 6:
                        # Convert to sortable format: YYYYMMDDHHMMSS
                        sortable_timestamp = date_part + time_part
                        return (sortable_timestamp, os.path.getmtime(dir_path))
            except Exception:
                pass
        # Fall back to modification time
        return ('0', os.path.getmtime(dir_path))
    
    # Sort by timestamp (newest first), fallback to modification time
    output_dirs.sort(key=get_sort_key, reverse=True)
    latest_dir = output_dirs[0]
    print(f"Found {len(output_dirs)} output directory(ies), using latest: {os.path.basename(latest_dir)}")
    return latest_dir


def main():
    global OUTPUT_DIR
    
    parser = argparse.ArgumentParser(description='Agent Status Visualizer')
    parser.add_argument('-d', '--output-dir', type=str, default=None,
                       dest='output_dir',
                       help='Path to the output directory containing status files and mailboxes. If not provided, automatically searches for the latest output_* directory.')
    parser.add_argument('--host', type=str, default='0.0.0.0',
                       help='Host to bind the server to (default: 0.0.0.0, accessible from all network interfaces)')
    parser.add_argument('--port', type=int, default=5000,
                       help='Port to bind the server to (default: 5000)')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug mode')
    
    args = parser.parse_args()
    
    # If output_dir not provided, try to find latest automatically
    if args.output_dir:
        OUTPUT_DIR = os.path.abspath(args.output_dir)
    else:
        print("No output directory specified, searching for latest output_* directory...")
        OUTPUT_DIR = find_latest_output_dir()
        if OUTPUT_DIR:
            print(f"Using latest output directory: {OUTPUT_DIR}")
        else:
            print("Error: Could not find any output_* directory.")
            print("Please specify a directory using -d or --output-dir")
            print("Example: python agent_status_visualizer.py -d output_20251211_111545")
            return 1
    
    if not os.path.exists(OUTPUT_DIR):
        print(f"Error: Output directory does not exist: {OUTPUT_DIR}")
        return 1
    
    print(f"Starting Agent Status Visualizer...")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Output directory exists: {os.path.exists(OUTPUT_DIR)}")
    if args.host == '0.0.0.0':
        # Get local IP address for better user experience
        import socket
        try:
            # Connect to a remote address to get local IP
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            local_ip = s.getsockname()[0]
            s.close()
            print(f"Server URL: http://{local_ip}:{args.port} (accessible from network)")
            print(f"Local URL: http://127.0.0.1:{args.port} (local access)")
        except Exception:
            print(f"Server URL: http://0.0.0.0:{args.port} (accessible from network)")
    else:
        print(f"Server URL: http://{args.host}:{args.port}")
    print(f"Open http://{args.host if args.host != '0.0.0.0' else '127.0.0.1'}:{args.port} in your browser to view the visualization")
    
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == '__main__':
    main()

