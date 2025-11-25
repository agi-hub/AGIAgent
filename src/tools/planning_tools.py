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

import os
import json
from typing import Dict, Any, List, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from .print_system import print_current, print_debug, print_error


class PlanningTools:
    """Tools for dynamic tool planning and loading"""
    
    def __init__(self, workspace_root: str = None):
        """
        Initialize PlanningTools
        
        Args:
            workspace_root: Root directory of the workspace
        """
        self.workspace_root = workspace_root or os.getcwd()
    
    def plan_tools(self, query: str) -> Dict[str, Any]:
        """
        Plan and select tools based on query using TFIDF matching.
        Loads tools from workspace/tool_pool.json, finds TOP3 matching tools,
        and writes them to workspace/current_tool_list.json.
        
        Args:
            query: Search query string describing the needed tools
            
        Returns:
            Dictionary containing selected tool names and status
        """
        try:
            # Validate query
            if not query or not str(query).strip():
                return {
                    "status": "error",
                    "message": "Query parameter is required and cannot be empty"
                }
            
            query = str(query).strip()
            
            # Path to tool pool file
            tool_pool_path = os.path.join(self.workspace_root, "tool_pool.json")
            
            # Check if tool_pool.json exists
            if not os.path.exists(tool_pool_path):
                return {
                    "status": "error",
                    "message": f"tool_pool.json not found at {tool_pool_path}"
                }
            
            # Load tool pool
            try:
                with open(tool_pool_path, 'r', encoding='utf-8') as f:
                    tool_pool = json.load(f)
            except json.JSONDecodeError as e:
                return {
                    "status": "error",
                    "message": f"Failed to parse tool_pool.json: {e}"
                }
            except Exception as e:
                return {
                    "status": "error",
                    "message": f"Failed to load tool_pool.json: {e}"
                }
            
            if not tool_pool or not isinstance(tool_pool, dict):
                return {
                    "status": "error",
                    "message": "tool_pool.json is empty or invalid format"
                }
            
            # Extract tool names and descriptions
            tool_names = []
            tool_descriptions = []
            
            for tool_name, tool_info in tool_pool.items():
                if isinstance(tool_info, dict):
                    # Extract description from tool info
                    description = tool_info.get("description", "")
                    if not description:
                        # Try to get description from parameters or other fields
                        description = str(tool_info)
                else:
                    description = str(tool_info)
                
                tool_names.append(tool_name)
                tool_descriptions.append(description)
            
            if not tool_names:
                return {
                    "status": "error",
                    "message": "No tools found in tool_pool.json"
                }
            
            # Use TFIDF to find matching tools
            try:
                # Create TFIDF vectorizer
                vectorizer = TfidfVectorizer(
                    max_features=1000,
                    ngram_range=(1, 2),
                    min_df=1,
                    max_df=0.95
                )
                
                # Fit and transform tool descriptions
                tool_vectors = vectorizer.fit_transform(tool_descriptions)
                
                # Transform query
                query_vector = vectorizer.transform([query])
                
                # Calculate cosine similarity
                similarities = cosine_similarity(query_vector, tool_vectors)[0]
                
                # Get top 3 indices
                top_indices = similarities.argsort()[-3:][::-1]
                
                # Select top 3 tools (even if similarity is 0, we still want top matches)
                selected_tools = {}
                selected_names = []
                
                for idx in top_indices:
                    tool_name = tool_names[idx]
                    similarity_score = similarities[idx]
                    selected_tools[tool_name] = tool_pool[tool_name]
                    selected_names.append(tool_name)
                    print_debug(f"🔍 Selected tool: {tool_name} (similarity: {similarity_score:.4f})")
                
                # Ensure we have at least one tool
                if not selected_tools and tool_names:
                    # Return the first tool as fallback
                    first_tool = tool_names[0]
                    selected_tools[first_tool] = tool_pool[first_tool]
                    selected_names = [first_tool]
                    print_debug(f"⚠️ No matching tools found, using fallback: {first_tool}")
                
            except Exception as e:
                print_error(f"TFIDF matching failed: {e}")
                # Fallback: return first tool
                if tool_names:
                    first_tool = tool_names[0]
                    selected_tools = {first_tool: tool_pool[first_tool]}
                    selected_names = [first_tool]
                else:
                    return {
                        "status": "error",
                        "message": f"TFIDF matching failed: {e}"
                    }
            
            # Write selected tools to current_tool_list.json
            current_tool_list_path = os.path.join(self.workspace_root, "current_tool_list.json")
            
            # Debug: Print selected tools before writing
            print_debug(f"📝 Preparing to write {len(selected_tools)} tools to {current_tool_list_path}")
            print_debug(f"📝 Selected tools: {list(selected_tools.keys())}")
            
            # Ensure selected_tools is not empty
            if not selected_tools:
                print_error(f"❌ selected_tools is empty! Cannot write to file.")
                return {
                    "status": "error",
                    "message": "No tools were selected. selected_tools dictionary is empty."
                }
            
            try:
                # Ensure directory exists (if path has a directory component)
                dir_path = os.path.dirname(current_tool_list_path)
                if dir_path and not os.path.exists(dir_path):
                    os.makedirs(dir_path, exist_ok=True)
                
                with open(current_tool_list_path, 'w', encoding='utf-8') as f:
                    json.dump(selected_tools, f, ensure_ascii=False, indent=2)
                
                # Verify file was written correctly
                if os.path.exists(current_tool_list_path):
                    file_size = os.path.getsize(current_tool_list_path)
                    print_debug(f"✅ Wrote {len(selected_tools)} tools to current_tool_list.json (file size: {file_size} bytes)")
                    
                    # Double-check file content
                    try:
                        with open(current_tool_list_path, 'r', encoding='utf-8') as f:
                            verify_content = json.load(f)
                        if verify_content and len(verify_content) > 0:
                            print_debug(f"✅ Verified: File contains {len(verify_content)} tools")
                        else:
                            print_error(f"❌ Verification failed: File is empty after write!")
                    except Exception as e:
                        print_error(f"❌ Failed to verify file content: {e}")
                else:
                    print_error(f"❌ File was not created at {current_tool_list_path}")
                    return {
                        "status": "error",
                        "message": f"File was not created at {current_tool_list_path}"
                    }
            except Exception as e:
                print_error(f"❌ Failed to write current_tool_list.json: {e}")
                import traceback
                print_error(traceback.format_exc())
                return {
                    "status": "error",
                    "message": f"Failed to write current_tool_list.json: {e}"
                }
            
            # Return success with selected tool names
            return {
                "status": "success",
                "selected_tools": selected_names,
                "count": len(selected_names),
                "message": f"Selected {len(selected_names)} tools: {', '.join(selected_names)}"
            }
            
        except Exception as e:
            print_error(f"plan_tools failed: {e}")
            return {
                "status": "error",
                "message": f"plan_tools execution failed: {e}"
            }

