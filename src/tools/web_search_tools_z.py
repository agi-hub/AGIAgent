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

import requests
import json
import datetime
from typing import Dict, Any, List
import time
import os


class ZhipuWebSearchTools:
    """
    Zhipu AI Web Search Tools
    A web search utility class based on the Zhipu AI Web Search API.
    """

    def __init__(self, api_key: str, search_engine: str = "search_std", default_count: int = 10, workspace_root: str = None):
        """
        Initialize the Zhipu Web Search Tool

        Args:
            api_key: Zhipu AI API key
            search_engine: Default search engine, options: search_std, search_pro, search_pro_sogou, search_pro_quark
            default_count: Default number of search results (1-50)
            workspace_root: Root directory for saving search results
        """
        self.api_key = api_key
        self.search_engine = search_engine
        self.default_count = min(max(default_count, 1), 50)  # Ensure within 1-50
        self.api_url = "https://open.bigmodel.cn/api/paas/v4/web_search"

        # Set up workspace and result directory
        if workspace_root:
            self.workspace_root = workspace_root
            self.web_result_dir = os.path.join(self.workspace_root, "web_search_result")
        else:
            self.workspace_root = os.getcwd()
            self.web_result_dir = os.path.join(self.workspace_root, "web_search_result")

        # Ensure result directory exists
        os.makedirs(self.web_result_dir, exist_ok=True)

    def web_search(self, search_term: str, fetch_content: bool = False, max_content_results: int = 10, **kwargs) -> Dict[str, Any]:
        """
        Perform web search using Zhipu AI

        Args:
            search_term: Query keywords
            fetch_content: Whether to fetch webpage content (not implemented in this version)
            max_content_results: Maximum number of content results (not used in this version)
            **kwargs: Other arguments
                - search_intent: Whether to perform search intent recognition (default False)
                - count: Number of results (default is self.default_count)
                - search_domain_filter: Domain filtering
                - search_recency_filter: Time filter (oneDay, oneWeek, oneMonth, oneYear, noLimit)
                - content_size: Content size (medium, high)
                - request_id: Request ID
                - user_id: User ID

        Returns:
            A dictionary of search results, compatible with the output format of web_search_tools.py
        """
        try:
            # Prepare request parameters
            request_data = {
                "search_query": search_term,
                "search_engine": self.search_engine,
                "search_intent": kwargs.get("search_intent", False),
                "count": min(max(kwargs.get("count", self.default_count), 1), 50)
            }

            # Add optional parameters
            optional_params = [
                "search_domain_filter",
                "search_recency_filter",
                "content_size",
                "request_id",
                "user_id"
            ]

            for param in optional_params:
                if param in kwargs:
                    request_data[param] = kwargs[param]

            # Set headers
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }

            # Make POST request
            response = requests.post(
                self.api_url,
                headers=headers,
                json=request_data,
                timeout=30
            )

            # Check response status
            response.raise_for_status()

            # Parse response
            api_response = response.json()

            # Convert API results to compatible format
            results = self._convert_api_results(api_response.get("search_result", []))

            # Build return data
            result_data = {
                'status': 'success',
                'search_term': search_term,
                'results': results,
                'timestamp': datetime.datetime.now().isoformat(),
                'total_results': len(results),
                'content_fetched': False,  # Content fetching is not supported in this version
                'results_with_content': 0,
                'saved_html_files': 0,
                'saved_txt_files': 0,
                'total_txt_files_in_directory': 0,
                'search_engine': self.search_engine,
                'api_request_id': api_response.get("id"),
                'api_created': api_response.get("created")
            }

            # If search intent info is present, include it
            if "search_intent" in api_response:
                result_data['search_intent'] = api_response["search_intent"]

            # Print search result details
            if results:
                for i, result in enumerate(results, 1):
                    title = result.get('title', 'No Title')
                    content = result.get('snippet', 'No snippet')
                    url = result.get('url', 'No URL')
                    source = result.get('source', '')

                    print(f"\n[{i}] {title}")
                    if source:
                        print(f"{url}")

            # Save search results to file
            if results:
                self._save_search_results_to_file(search_term, results)

            return result_data

        except requests.exceptions.Timeout:
            print("❌ Search request timed out")
            return self._create_error_result(search_term, "Search request timed out")

        except requests.exceptions.RequestException as e:
            print(f"❌ Search request failed: {e}")
            return self._create_error_result(search_term, f"API request failed: {str(e)}")

        except json.JSONDecodeError as e:
            print(f"❌ Failed to parse response: {e}")
            return self._create_error_result(search_term, f"Failed to parse response: {str(e)}")

        except Exception as e:
            print(f"❌ Unknown error occurred during search: {e}")
            return self._create_error_result(search_term, f"Unknown error: {str(e)}")

    def _save_search_results_to_file(self, search_term: str, results: List[Dict]) -> None:
        """
        Save search results to a file in the web_search_result directory.

        Args:
            search_term: The search query used
            results: List of search results to save
        """
        try:
            # Create a safe filename from search term
            safe_filename = "".join(c for c in search_term if c.isalnum() or c in (' ', '-', '_')).rstrip()
            safe_filename = safe_filename.replace(' ', '_')
            if not safe_filename:
                safe_filename = "search_results"

            # Limit filename length
            if len(safe_filename) > 100:
                safe_filename = safe_filename[:100]

            filename = f"{safe_filename}.txt"
            filepath = os.path.join(self.web_result_dir, filename)

            # Format the content
            content_lines = []
            content_lines.append(f"Search Query: {search_term}")
            content_lines.append(f"Search Time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            content_lines.append(f"Total Results: {len(results)}")
            content_lines.append("=" * 80)
            content_lines.append("")

            for i, result in enumerate(results, 1):
                title = result.get('title', 'No Title')
                url = result.get('url', 'No URL')
                snippet = result.get('snippet', 'No snippet available')
                source = result.get('source', '')

                content_lines.append(f"[{i}] {title}")
                if source:
                    content_lines.append(f"Source: {source}")
                content_lines.append(f"URL: {url}")
                content_lines.append(f"Summary: {snippet}")
                content_lines.append("-" * 80)
                content_lines.append("")

            # Write to file
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write('\n'.join(content_lines))

            print(f"💾 Search results saved to: {filepath}")

        except Exception as e:
            print(f"❌ Failed to save search results: {e}")

    def _convert_api_results(self, api_results: List[Dict]) -> List[Dict]:
        """
        Convert Zhipu API search results to a format compatible with original implementation.

        Args:
            api_results: List of search results returned by Zhipu API

        Returns:
            Converted result list
        """
        converted_results = []

        for i, result in enumerate(api_results):
            converted_result = {
                'title': result.get('title', 'No Title'),
                'url': result.get('link', ''),
                'snippet': result.get('content', ''),
                'content': '',  # Full content not fetched in this version
                'rank': i + 1,
                'source': result.get('media', ''),
                'icon': result.get('icon', ''),
                'refer': result.get('refer', ''),
                'publish_date': result.get('publish_date', ''),
                'has_full_content': False
            }
            converted_results.append(converted_result)

        return converted_results

    def _create_error_result(self, search_term: str, error_message: str) -> Dict[str, Any]:
        """
        Create an error result

        Args:
            search_term: Search keyword
            error_message: Error message

        Returns:
            Dictionary of error result
        """
        return {
            'status': 'failed',
            'search_term': search_term,
            'results': [{
                'title': f'Search Failed: {search_term}',
                'url': '',
                'snippet': error_message,
                'content': error_message,
                'rank': 1,
                'source': '',
                'icon': '',
                'refer': '',
                'publish_date': '',
                'has_full_content': False
            }],
            'timestamp': datetime.datetime.now().isoformat(),
            'total_results': 1,
            'content_fetched': False,
            'results_with_content': 0,
            'saved_html_files': 0,
            'saved_txt_files': 0,
            'total_txt_files_in_directory': 0,
            'error': error_message
        }
