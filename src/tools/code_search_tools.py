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

import glob
import os
from typing import List, Dict, Any, Optional

class CodeSearchTools:
    def __init__(self):
        """Initialize code search tools"""
        # These attributes will be available from BaseTools through multiple inheritance
        # Only define them if they don't already exist (to avoid overriding BaseTools initialization)
        if not hasattr(self, 'code_parser'):
            self.code_parser = None
        if not hasattr(self, 'workspace_root'):
            self.workspace_root = None
    
    def workspace_search(self, query: str, target_directories: Optional[List[str]] = None, **kwargs) -> Dict[str, Any]:
        """
        Use vector database and keyword database for semantic search to find the most relevant code snippets in the codebase.
        
        Args:
            query: The search query to find relevant code
            target_directories: Glob patterns for directories to search over (temporarily ignored, use entire codebase)

        Returns:
            Dictionary with search results
        """
        print_current(f"🔍 Codebase semantic search: {query}")
        # Ignore additional parameters
        if kwargs:
            print_current(f"⚠️  Ignoring additional parameters: {list(kwargs.keys())}")
        
        if not self.code_parser:
            print_current(f"❌ Code parser not initialized, using basic search")
            return self._fallback_workspace_search(query, target_directories)
        
        try:
            # Expand query with both Chinese and English terms for better search
            expanded_query = self._expand_query_for_search(query)
            
            # Perform hybrid search (vector search + keyword search)
            search_results = self.code_parser.hybrid_search(
                query=expanded_query, 
                vector_top_k=10,  # Increased for better recall
                keyword_top_k=8   # Increased for better recall
            )
            
            # Filter out irrelevant results and improve scoring
            filtered_results = self._filter_and_score_results(search_results, query)
            
            # Convert search results format
            results = []
            for result in filtered_results[:10]:  # Limit to first 10 results
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
            
            print_current(f"✅ Search completed, found {len(results)} relevant code snippets")
            print_current(f"📊 Codebase statistics: {stats.get('total_files', 0)} files, {stats.get('total_segments', 0)} code segments")
            
            return {
                'query': query,
                'results': results,
                'total_results': len(results),
                'repository_stats': stats,
                'search_method': 'hybrid_vector_keyword'
            }
            
        except Exception as e:
            print_current(f"❌ Semantic search failed: {e}, using basic search")
            return self._fallback_workspace_search(query, target_directories)
    
    def get_background_update_status(self) -> Dict[str, Any]:
        """
        Get the status information of the background incremental update thread for code repository
        
        Returns:
            Dictionary containing background update status and statistics
        """
        if not self.code_parser:
            return {
                'status': 'error',
                'message': 'Code parser not initialized',
                'background_update_enabled': False,
                'thread_running': False
            }
        
        try:
            # Check if background update is enabled
            background_enabled = getattr(self.code_parser, '_background_update_enabled', False)
            
            # Get background update statistics
            stats = self.code_parser.get_background_update_stats()
            
            # Check if thread is running
            thread_running = False
            if hasattr(self.code_parser, 'background_update_thread') and self.code_parser.background_update_thread:
                thread_running = self.code_parser.background_update_thread.is_running()
            
            status_info = {
                'status': 'success',
                'background_update_enabled': background_enabled,
                'thread_running': thread_running,
                'update_interval': getattr(self.code_parser, 'update_interval', 1.0),
                'statistics': stats
            }
            
            # If statistics available, add friendly status description
            if stats:
                total_checks = stats.get('total_checks', 0)
                total_updates = stats.get('total_updates', 0)
                successful_updates = stats.get('successful_updates', 0)
                failed_updates = stats.get('failed_updates', 0)
                last_update_time = stats.get('last_update_time')
                last_error = stats.get('last_error')
                
                status_info['status_summary'] = {
                    'total_checks': total_checks,
                    'total_updates': total_updates,
                    'successful_updates': successful_updates,
                    'failed_updates': failed_updates,
                    'success_rate': f"{(successful_updates / max(total_updates, 1) * 100):.1f}%" if total_updates > 0 else "100%",
                    'last_update_time': last_update_time,
                    'last_error': last_error
                }
                
                # Generate status description
                if thread_running:
                    status_info['description'] = f"Background update thread is running, performed {total_checks} checks, completed {successful_updates} updates"
                else:
                    status_info['description'] = "Background update thread is not running"
            else:
                status_info['description'] = "Background update thread status information unavailable"
            
            return status_info
            
        except Exception as e:
            return {
                'status': 'error',
                'message': f'Failed to get background update status: {e}',
                'background_update_enabled': False,
                'thread_running': False
            }
    
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

    def _expand_query_for_search(self, query: str) -> str:
        """
        Expand query with related terms for better search
        
        Args:
            query: Original search query
            
        Returns:
            Expanded query string
        """
        # Map Chinese terms to English equivalents for better code search
        chinese_to_english = {
            'Multi-agent': 'multi agent multiagent',
            'Establish': 'create build establish setup',
            'Communication': 'communication message communicate',
            'Implementation': 'implement implementation',
            'Method': 'method way approach',
            'System': 'system',
            'Management': 'manage management',
            'Processing': 'process handle',
            'Configuration': 'config configuration',
            'Module': 'module',
            'Class': 'class',
            'Function': 'function',
            'Method': 'method',
            'Interface': 'interface',
            'Service': 'service',
            'Client': 'client',
            'Server': 'server',
            'Network': 'network',
            'Protocol': 'protocol'
        }
        
        expanded_terms = [query]
        
        # Add English equivalents for Chinese terms
        for chinese, english in chinese_to_english.items():
            if chinese in query:
                expanded_terms.append(english)
        
        # Add related programming terms
        if 'agent' in query.lower():
            expanded_terms.extend(['agent', 'spawn', 'worker', 'process', 'thread'])
        if 'Communication' in query or 'communication' in query.lower():
            expanded_terms.extend(['message', 'queue', 'channel', 'socket', 'rpc'])
        if 'Establish' in query or 'create' in query.lower():
            expanded_terms.extend(['initialize', 'setup', 'spawn', 'start'])
        
        return ' '.join(expanded_terms)

    def _filter_and_score_results(self, search_results: List, query: str) -> List:
        """
        Filter and improve scoring of search results
        
        Args:
            search_results: Raw search results
            query: Original query
            
        Returns:
            Filtered and re-scored results
        """
        if not search_results:
            return []
        
        # Filter out dictionary files and irrelevant results
        filtered_results = []
        for result in search_results:
            segment = result.segment
            file_path = segment.file_path.lower()
            
            # Skip dictionary files and other irrelevant files
            if any(pattern in file_path for pattern in ['ppocr_keys', 'dict', 'vocab', 'stopwords']):
                continue
                
            # Skip files with very high non-ASCII ratio (likely data files)
            #content = segment.content
            #if content:
            #    non_ascii_count = sum(1 for char in content if ord(char) > 127)
            #    if len(content) > 0 and non_ascii_count / len(content) > 0.7:
            #        continue
            
            # Boost score for code files
            #if any(ext in file_path for ext in ['.py', '.js', '.ts', '.java', '.cpp', '.c', '.h']):
            #    result.score *= 1.2
            
            # Boost score for files with relevant keywords
            #if any(keyword in content.lower() for keyword in ['agent', 'communication', 'spawn', #'process']):
            #    result.score *= 1.1
            
            filtered_results.append(result)
        
        # Sort by score
        filtered_results.sort(key=lambda x: x.score, reverse=True)
        
        return filtered_results

    def _fallback_workspace_search(self, query: str, target_directories: List[str] = None) -> Dict[str, Any]:
        """
        Basic search implementation (fallback option)
        """
        print_current(f"🔍 Using basic text search: {query}")
        
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
                            print_current(f"Error reading file {file_path}: {e}")
        
        # Sort results by relevance
        results.sort(key=lambda x: x['score'], reverse=True)
        
        return {
            'query': query,
            'results': results[:10],  # Limit to top 10 results
            'search_method': 'basic_text_matching'
        }
