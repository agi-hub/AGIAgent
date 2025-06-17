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

import glob
import os
from typing import List, Dict, Any

class CodeSearchTools:
    def codebase_search(self, query: str, target_directories: List[str] = None, **kwargs) -> Dict[str, Any]:
        """
        Use vector database and keyword database for semantic search to find the most relevant code snippets in the codebase.
        
        Args:
            query: The search query to find relevant code
            target_directories: Glob patterns for directories to search over (temporarily ignored, use entire codebase)

        Returns:
            Dictionary with search results
        """
        print(f"🔍 Codebase semantic search: {query}")
        # Ignore additional parameters
        if kwargs:
            print(f"⚠️  Ignoring additional parameters: {list(kwargs.keys())}")
        
        if not self.code_parser:
            print(f"❌ Code parser not initialized, using basic search")
            return self._fallback_codebase_search(query, target_directories)
        
        try:
            # Perform hybrid search (vector search + keyword search)
            search_results = self.code_parser.hybrid_search(
                query=query, 
                vector_top_k=8,  # Vector search returns 8 results
                keyword_top_k=5  # Keyword search returns 5 results
            )
            
            # Convert search results format
            results = []
            for result in search_results[:10]:  # Limit to first 10 results
                segment = result.segment
                
                # Find matching position of query content in segment
                matched_line_num, matched_snippet = self._find_best_match_in_segment(
                    segment, query
                )
                
                # If specific matching position is found, use more precise snippet
                if matched_line_num > 0:
                    display_snippet = matched_snippet
                    actual_start_line = matched_line_num
                else:
                    # Use the first part of the original segment
                    lines = segment.content.split('\n')
                    display_snippet = '\n'.join(lines[:min(10, len(lines))])
                    actual_start_line = segment.start_line
                
                results.append({
                    'file': segment.file_path,
                    'snippet': display_snippet,
                    'start_line': actual_start_line,
                    'end_line': segment.end_line,
                    'score': result.score,
                    'search_type': result.search_type,
                    'segment_id': segment.segment_id,
                    'segment_range': f"{segment.start_line}-{segment.end_line}"
                })
            
            # Get repository statistics
            stats = self.code_parser.get_repository_stats()
            
            print(f"✅ Search completed, found {len(results)} relevant code snippets")
            print(f"📊 Codebase statistics: {stats.get('total_files', 0)} files, {stats.get('total_segments', 0)} code segments")
            
            return {
                'query': query,
                'results': results,
                'total_results': len(results),
                'repository_stats': stats,
                'search_method': 'hybrid_vector_keyword'
            }
            
        except Exception as e:
            print(f"❌ Semantic search failed: {e}, using basic search")
            return self._fallback_codebase_search(query, target_directories)
    
    def _find_best_match_in_segment(self, segment, query: str):
        """
        Find the best matching position in a code segment
        
        Args:
            segment: Code segment object
            query: Search query
            
        Returns:
            (matched_line_num, matched_snippet): Matched line number and code snippet
        """
        lines = segment.content.split('\n')
        query_lower = query.lower()
        
        best_match_line = -1
        best_match_score = 0
        
        # Look for the line that contains the most query content
        for i, line in enumerate(lines):
            line_lower = line.lower()
            score = 0
            
            # Check if entire query string is in line
            if query_lower in line_lower:
                score += 10
            
            # Check each word in query
            for word in query_lower.split():
                if len(word) > 2 and word in line_lower:
                    score += 2
            
            if score > best_match_score:
                best_match_score = score
                best_match_line = i
        
        # If good match is found, return snippet with context
        if best_match_line >= 0 and best_match_score > 0:
            start_context = max(0, best_match_line - 3)
            end_context = min(len(lines), best_match_line + 7)
            
            matched_snippet = '\n'.join(lines[start_context:end_context])
            actual_line_num = segment.start_line + best_match_line
            
            return actual_line_num, matched_snippet
        
        return -1, ""

    def _fallback_codebase_search(self, query: str, target_directories: List[str] = None) -> Dict[str, Any]:
        """
        Basic search implementation (fallback option)
        """
        print(f"🔍 Using basic text search: {query}")
        
        results = []
        search_dirs = []
        
        if target_directories:
            for dir_pattern in target_directories:
                matching_dirs = glob.glob(os.path.join(self.workspace_root, dir_pattern))
                search_dirs.extend(matching_dirs)
        else:
            search_dirs = [self.workspace_root]
        
        # Walk through directories and search files
        for directory in search_dirs:
            for root, _, files in os.walk(directory):
                for file in files:
                    if file.endswith(('.py', '.js', '.ts', '.jsx', '.tsx', '.css', '.java', '.c', '.cpp', '.h', '.md', '.txt')):
                        file_path = os.path.join(root, file)
                        try:
                            with open(file_path, 'r', encoding='utf-8') as f:
                                content = f.read()
                                
                                # Simple relevance scoring based on keyword presence
                                score = 0
                                for keyword in query.lower().split():
                                    if keyword in content.lower():
                                        score += content.lower().count(keyword)
                                
                                if score > 0:
                                    # Find relevant code snippets
                                    lines = content.split('\n')
                                    for i, line in enumerate(lines):
                                        if any(keyword in line.lower() for keyword in query.lower().split()):
                                            start = max(0, i - 5)
                                            end = min(len(lines), i + 6)
                                            snippet = '\n'.join(lines[start:end])
                                            
                                            results.append({
                                                'file': os.path.relpath(file_path, self.workspace_root),
                                                'snippet': snippet,
                                                'start_line': start + 1,
                                                'end_line': end,
                                                'score': score,
                                                'search_type': 'text_matching'
                                            })
                        except Exception as e:
                            print(f"Error reading file {file_path}: {e}")
        
        # Sort results by relevance
        results.sort(key=lambda x: x['score'], reverse=True)
        
        return {
            'query': query,
            'results': results[:10],  # Limit to top 10 results
            'search_method': 'basic_text_matching'
        }
