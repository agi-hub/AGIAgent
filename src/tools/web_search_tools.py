#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from .print_system import print_system, print_current, print_system, print_error, print_debug
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

import re
import time
import requests
import urllib.parse
from typing import List, Dict, Any
import os
import signal
import platform
from concurrent.futures import ThreadPoolExecutor, as_completed
import concurrent.futures
import datetime
import base64
import io

# Import config_loader to get truncation length configuration
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config_loader import get_web_content_truncation_length, get_truncation_length


class TimeoutError(Exception):
    """Custom timeout exception"""
    pass


def timeout_handler(signum, frame):
    """Signal handler for timeout"""
    raise TimeoutError("Operation timed out")


def is_windows():
    """Check if running on Windows"""
    return platform.system().lower() == 'windows'


def is_main_thread():
    """Check if running in main thread"""
    import threading
    return threading.current_thread() is threading.main_thread()


# Check if the model is a Claude model
def is_claude_model(model: str) -> bool:
    """Check if the model name is a Claude model"""
    return "claude" in model.lower() or "anthropic" in model.lower()


# Check if Playwright is available
def _check_playwright_available():
    """Check if playwright is available for browser automation"""
    try:
        import playwright
        # Also check if browser is installed by trying to import sync_api
        from playwright.sync_api import sync_playwright
        return True
    except ImportError:
        return False
    except Exception:
        # Handle other errors (like GLIBC issues)
        return False

# Cache the playwright availability check result
_PLAYWRIGHT_AVAILABLE = None

def is_playwright_available():
    """Check if playwright is available (cached result)"""
    global _PLAYWRIGHT_AVAILABLE
    if _PLAYWRIGHT_AVAILABLE is None:
        _PLAYWRIGHT_AVAILABLE = _check_playwright_available()
    return _PLAYWRIGHT_AVAILABLE


# Dynamically import Anthropic
def get_anthropic_client():
    """Dynamically import and return Anthropic client class"""
    try:
        from anthropic import Anthropic
        return Anthropic
    except ImportError:
        print_current("Anthropic library not installed, please run: pip install anthropic")
        raise ImportError("Anthropic library not installed")


class WebSearchTools:
    def __init__(self, llm_api_key: str = None, llm_model: str = None, llm_api_base: str = None, enable_llm_filtering: bool = False, enable_summary: bool = True, out_dir: str = None, verbose: bool = True):
        self._google_connectivity_checked = False
        self._google_available = True
        self._last_google_request = 0  # Track last Google request time for rate limiting
        self.verbose = verbose  # Control verbose debug output
        
        # LLM configuration for content filtering and summarization
        self.enable_llm_filtering = enable_llm_filtering
        self.enable_summary = enable_summary
        self.llm_client = None
        self.llm_model = llm_model
        self.is_claude = False
        
        # Initialize web search result directory path but don't create it yet
        self.web_result_dir = os.path.join(out_dir, "workspace", "web_search_result")
        
        if (enable_llm_filtering or enable_summary) and llm_api_key and llm_model and llm_api_base:
            try:
                self._setup_llm_client(llm_api_key, llm_model, llm_api_base)
                features = []
                if enable_llm_filtering:
                    features.append("content filtering")
                if enable_summary:
                    features.append("search results summarization")
                print_system(f"🤖 LLM features enabled with model {llm_model}: {', '.join(features)}")
            except Exception as e:
                print_error(f"⚠️ Failed to setup LLM client, disabling LLM features: {e}")
                self.enable_llm_filtering = False
                self.enable_summary = False
        elif enable_llm_filtering or enable_summary:
            self.enable_llm_filtering = False
            self.enable_summary = False
    
    def _verbose_print(self, message: str):
        """Print message only if verbose mode is enabled"""
        if self.verbose:
            print(message)
    
    def _ensure_result_directory(self):
        """Ensure the web search result directory exists"""
        try:
            os.makedirs(self.web_result_dir, exist_ok=True)
        except Exception as e:
            print_current(f"⚠️ Failed to create result directory: {e}")
            self.web_result_dir = None
    
    def _count_txt_files_in_result_dir(self) -> int:
        """Count the number of txt files in the web search result directory"""
        try:
            if not self.web_result_dir or not os.path.exists(self.web_result_dir):
                return 0
            
            txt_files = [f for f in os.listdir(self.web_result_dir) 
                        if f.endswith('.txt') and os.path.isfile(os.path.join(self.web_result_dir, f))]
            return len(txt_files)
        except Exception as e:
            print_current(f"⚠️ Failed to count txt files: {e}")
            return 0
    
    def _save_webpage_html(self, page, url: str, title: str, search_term: str = "") -> str:
        """
        Save webpage HTML content to file
        
        Args:
            page: Playwright page object
            url: Original URL
            title: Page title
            search_term: Search term for filename context
            
        Returns:
            Path to saved file or empty string if failed
        """
        # Ensure the web search result directory exists when needed
        self._ensure_result_directory()
        
        if not self.web_result_dir:
            return ""
        
        try:
            # Get HTML content
            html_content = page.content()
            
            # Check for special pages that shouldn't be saved as HTML
            if ("当前环境异常，完成验证后即可继续访问。" in html_content or 
                "豆丁网" in html_content or "docin.com" in html_content or
                "百度学术搜索" in html_content or "xueshu.baidu.com" in html_content or
                "百度学术" in html_content or "- 百度学术" in title or
                "相关论文" in html_content or "获取方式" in html_content):
                if "当前环境异常，完成验证后即可继续访问。" in html_content:
                    print_current("⚠️ Skipping HTML save for verification page")
                elif "豆丁网" in html_content or "docin.com" in html_content:
                    print_current("⚠️ Skipping HTML save for DocIn embedded document page")
                elif ("百度学术搜索" in html_content or "xueshu.baidu.com" in html_content or 
                      "百度学术" in html_content or "- 百度学术" in title or
                      "相关论文" in html_content or "获取方式" in html_content):
                    print_current("⚠️ Skipping HTML save for Baidu Scholar search page")

                return ""  # Return empty string to indicate no file was saved
            
            # Generate filename
            safe_title = re.sub(r'[^\w\s-]', '', title)[:50]  # Remove special chars, limit length
            safe_title = re.sub(r'[-\s]+', '_', safe_title)  # Replace spaces and hyphens with underscore
            
            # Add timestamp for uniqueness
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Create filename
            if search_term:
                safe_search = re.sub(r'[^\w\s-]', '', search_term)[:30]
                safe_search = re.sub(r'[-\s]+', '_', safe_search)
                filename = f"{safe_search}_{safe_title}_{timestamp}.html"
            else:
                filename = f"{safe_title}_{timestamp}.html"
            
            # Remove double underscores and ensure filename is not empty
            filename = re.sub(r'_+', '_', filename).strip('_')
            if not filename or filename == '.html' or len(filename) < 8:  # .html is 5 chars, so minimum valid name is 3+ chars
                filename = f"webpage_{timestamp}.html"
            
            filepath = os.path.join(self.web_result_dir, filename)
            
            # Save HTML content
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            return filepath
            
        except Exception as e:
            print_current(f"⚠️ Failed to save webpage HTML: {e}")
            return ""
    
    def _save_webpage_content(self, page, url: str, title: str, content: str, search_term: str = "") -> tuple:
        """
        Save both webpage HTML and extracted text content to files
        
        Args:
            page: Playwright page object
            url: Original URL
            title: Page title
            content: Extracted text content
            search_term: Search term for filename context
            
        Returns:
            Tuple of (html_filepath, txt_filepath) or empty strings if failed
        """
        # Ensure the web search result directory exists when needed
        self._ensure_result_directory()
        
        if not self.web_result_dir:
            return "", ""
        
        html_filepath = ""
        txt_filepath = ""
        
        try:
            # Generate base filename
            safe_title = re.sub(r'[^\w\s-]', '', title)[:50]  # Remove special chars, limit length
            safe_title = re.sub(r'[-\s]+', '_', safe_title)  # Replace spaces and hyphens with underscore
            
            # Add timestamp for uniqueness
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Create base filename
            if search_term:
                safe_search = re.sub(r'[^\w\s-]', '', search_term)[:30]
                safe_search = re.sub(r'[-\s]+', '_', safe_search)
                base_filename = f"{safe_search}_{safe_title}_{timestamp}"
            else:
                base_filename = f"{safe_title}_{timestamp}"
            
            # Remove double underscores and ensure filename is not empty
            base_filename = re.sub(r'_+', '_', base_filename).strip('_')
            if not base_filename:
                base_filename = f"webpage_{timestamp}"
            
            # Ensure base_filename is not empty and has valid characters
            if len(base_filename) < 3:
                base_filename = f"webpage_{timestamp}"
            
            # Save HTML content (but check for special pages first)
            try:
                html_content = page.content()
                
                # Check for special pages that shouldn't be saved as HTML
                should_skip_html = False
                if ("当前环境异常，完成验证后即可继续访问。" in html_content or 
                    "豆丁网" in html_content or "docin.com" in html_content or
                    "百度学术搜索" in html_content or "xueshu.baidu.com" in html_content or
                    "百度学术" in html_content or "- 百度学术" in title or
                    "相关论文" in html_content or "获取方式" in html_content):
                    should_skip_html = True
                    if "当前环境异常，完成验证后即可继续访问。" in html_content:
                        print_current("⚠️ Skipping HTML save for verification page")
                    elif "豆丁网" in html_content or "docin.com" in html_content:
                        print_current("⚠️ Skipping HTML save for DocIn embedded document page")
                    elif ("百度学术搜索" in html_content or "xueshu.baidu.com" in html_content or
                          "百度学术" in html_content or "- 百度学术" in title or
                          "相关论文" in html_content or "获取方式" in html_content):
                        print_current("⚠️ Skipping HTML save for Baidu Scholar search page")
                        
                
                if not should_skip_html:
                    # Ensure the HTML file has .html extension
                    html_filename = f"{base_filename}.html"
                    html_filepath = os.path.join(self.web_result_dir, html_filename)
                    with open(html_filepath, 'w', encoding='utf-8') as f:
                        f.write(html_content)

                    
            except Exception as e:
                print_current(f"⚠️ Failed to save webpage HTML: {e}")
            
            # Save text content
            try:
                if content and content.strip():
                    # Clean the content thoroughly for saving
                    cleaned_content = self._clean_text_for_saving(content)
                    
                    if cleaned_content and len(cleaned_content.strip()) > 50:
                        # Ensure the txt file has .txt extension
                        txt_filename = f"{base_filename}.txt"
                        txt_filepath = os.path.join(self.web_result_dir, txt_filename)
                        
                        # Create a formatted text file with metadata
                        # Special handling for verification page and DocIn embedded documents
                        if cleaned_content.strip() == "当前环境异常，完成验证后即可继续访问。":
                            formatted_content = f"""Title: {title}
URL: {url}
Search Term: {search_term}
Timestamp: {datetime.datetime.now().isoformat()}
Original Content Length: {len(content)} characters
Cleaned Content Length: {len(cleaned_content)} characters


{cleaned_content}
"""
                        elif cleaned_content.strip() == "正文为嵌入式文档，不可阅读":
                            formatted_content = f"""Title: {title}
URL: {url}
Search Term: {search_term}
Timestamp: {datetime.datetime.now().isoformat()}
Original Content Length: {len(content)} characters
Cleaned Content Length: {len(cleaned_content)} characters


{cleaned_content}
"""
                        elif cleaned_content.strip() == "结果无可用数据":
                            formatted_content = f"""Title: {title}
URL: {url}
Search Term: {search_term}
Timestamp: {datetime.datetime.now().isoformat()}
Original Content Length: {len(content)} characters
Cleaned Content Length: {len(cleaned_content)} characters


{cleaned_content}
"""
                        else:
                            formatted_content = f"""Title: {title}
URL: {url}
Search Term: {search_term}
Timestamp: {datetime.datetime.now().isoformat()}
Original Content Length: {len(content)} characters
Cleaned Content Length: {len(cleaned_content)} characters


{cleaned_content}
"""
                        
                        with open(txt_filepath, 'w', encoding='utf-8') as f:
                            f.write(formatted_content)
                else:
                    print_current(f"⚠️ No text content to save for: {title}")
            except Exception as e:
                print_current(f"⚠️ Failed to save text content: {e}")
            
            return html_filepath, txt_filepath
            
        except Exception as e:
            print_current(f"⚠️ Failed to save webpage content: {e}")
            return "", ""
    
    def _setup_llm_client(self, api_key: str, model: str, api_base: str):
        """Setup LLM client for content filtering"""
        self.is_claude = is_claude_model(model)
        
        if self.is_claude:
            # Adjust api_base for Claude models
            if not api_base.endswith('/anthropic'):
                if api_base.endswith('/v1'):
                    api_base = api_base[:-3] + '/anthropic'
                else:
                    api_base = api_base.rstrip('/') + '/anthropic'
            
            # Initialize Anthropic client
            Anthropic = get_anthropic_client()
            self.llm_client = Anthropic(
                api_key=api_key,
                base_url=api_base
            )
        else:
            # Initialize OpenAI client
            from openai import OpenAI
            self.llm_client = OpenAI(
                api_key=api_key,
                base_url=api_base
            )

    def _check_google_connectivity(self) -> bool:
        """
        Check if Google is accessible by trying to download Google homepage
        """
        if self._google_connectivity_checked:
            return self._google_available
        
        print_debug("🌐 Checking Google connectivity for the first time...")
        try:
            # Try to download Google homepage with 3 second timeout
            response = requests.get('https://www.google.com', 
                                  timeout=3, 
                                  headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'})
            
            if response.status_code == 200 and len(response.text) > 100:
                print_debug("✅ Google connectivity test passed")
                self._google_available = True
            else:
                print_debug(f"❌ Google connectivity test failed: status {response.status_code}")
                self._google_available = False
                
        except requests.exceptions.Timeout:
            print_debug("⚠️ Google connectivity test timeout (>3s)")
            self._google_available = False
        except requests.exceptions.RequestException as e:
            print_debug(f"⚠️ Google connectivity test error: {e}")
            self._google_available = False
        except Exception as e:
            print_debug(f"⚠️ Google connectivity test error: {e}")
            self._google_available = False
        
        self._google_connectivity_checked = True
        return self._google_available
    
    def _extract_relevant_content_with_llm(self, content: str, search_term: str, title: str = "") -> str:
        """
        Use LLM to extract relevant information from webpage content
        
        Args:
            content: Raw webpage content
            search_term: Original search term
            title: Page title for context
            
        Returns:
            Filtered relevant content
        """
        if not self.enable_llm_filtering or not self.llm_client or not content.strip():
            return content
        
        # Check for verification page and skip LLM processing
        if "当前环境异常，完成验证后即可继续访问。" in content:
            print_current("⚠️ Detected verification page in LLM filtering, skipping LLM processing")
            return "当前环境异常，完成验证后即可继续访问。"
        
        # Check for DocIn embedded document page and skip LLM processing
        if "豆丁网" in content or "docin.com" in content:
            print_current("⚠️ Detected DocIn embedded document page in LLM filtering, skipping LLM processing")
            return "正文为嵌入式文档，不可阅读"
        
        # Check for Baidu Scholar search page and skip LLM processing
        if ("百度学术搜索" in content or "百度学术" in content or
            "相关论文" in content or "获取方式" in content or
            "按相关性按相关性按被引量按时间降序" in content):
            print_current("⚠️ Detected Baidu Scholar search page in LLM filtering, skipping LLM processing")
            return "结果无可用数据"
        
        # Skip processing if content is too short
        if len(content.strip()) < 100:
            return content
        
        try:
            print_current(f"🧠 Using LLM to extract relevant information for: {search_term}")
            
            # Construct system prompt for content filtering
            system_prompt = """You are an expert at extracting relevant information from web content. Your task is to:

1. Extract ONLY the information that is directly relevant to the search query
2. Remove navigation menus, advertisements, cookie notices, footer information, sidebar content, and other webpage UI elements
3. Remove repetitive or promotional content
4. Keep the main article content, key facts, data, and relevant details
5. Maintain the original language and important formatting
6. If the content doesn't contain relevant information, return "No relevant content found"

Focus on providing clean, useful information that directly answers or relates to what the user was searching for."""

            # Construct user prompt
            user_prompt = f"""Search Query: "{search_term}"
Page Title: "{title}"

Please extract only the relevant information from the following webpage content. Remove navigation elements, ads, and irrelevant text, keeping only content that relates to the search query:

---
{content[:8000]}  
---

Please provide the extracted relevant content:"""

            # Call LLM based on type
            if self.is_claude:
                # Claude API call
                response = self.llm_client.messages.create(
                    model=self.llm_model,
                    max_tokens=min(4000, 8192),  # Use safe limit for web search content filtering
                    system=system_prompt,
                    messages=[{"role": "user", "content": user_prompt}],
                    temperature=0.1
                )
                
                if hasattr(response, 'content') and response.content:
                    filtered_content = response.content[0].text if response.content else ""
                else:
                    print_current("⚠️ Claude API response format unexpected")
                    return content
            else:
                # OpenAI API call
                response = self.llm_client.chat.completions.create(
                    model=self.llm_model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    max_tokens=min(4000, 8192),  # Use safe limit for web search content filtering
                    temperature=0.1
                )
                
                if response.choices and response.choices[0].message:
                    filtered_content = response.choices[0].message.content
                else:
                    print_current("⚠️ OpenAI API response format unexpected")
                    return content
            
            # Validate filtered content
            if filtered_content and filtered_content.strip():
                if "No relevant content found" in filtered_content or "no relevant content" in filtered_content.lower():
                    print_current("🔍 LLM determined no relevant content found")
                    return "No relevant content found in this webpage for the search query."
                elif len(filtered_content.strip()) > 50:  # Ensure we got substantial content
                    print_current(f"✅ LLM filtering completed: {len(content)} → {len(filtered_content)} characters")
                    return filtered_content.strip()
            
            print_current("⚠️ LLM filtering produced insufficient content, using original")
            return content
            
        except Exception as e:
            print_current(f"❌ LLM content filtering failed: {e}")
            return content

    def _summarize_search_results_with_llm(self, results: List[Dict], search_term: str) -> str:
        """
        Use LLM to summarize all search results with detailed individual analysis
        
        Args:
            results: List of search results with content
            search_term: Original search term
            
        Returns:
            Comprehensive summary with individual webpage analysis
        """
        if not self.enable_summary or not self.llm_client or not results:
            return ""
        
        # Filter results with meaningful content and prepare individual result details
        valid_results = []
        for i, result in enumerate(results):
            content = result.get('content', '')
            if content and len(content.strip()) > 100:
                # Skip error messages and non-content
                if not any(error_phrase in content.lower() for error_phrase in [
                    'content fetch error', 'processing error', 'timeout', 
                    'video or social media link', 'non-webpage link',
                    '当前环境异常', '正文为嵌入式文档', '结果无可用数据'
                ]):
                    # Add file path information to the result
                    result_with_files = result.copy()
                    result_with_files['result_index'] = i + 1
                    valid_results.append(result_with_files)
        
        if not valid_results:
            print_current("⚠️ No valid content found for summarization")
            return ""
        
        print_current(f"📝 Using LLM to summarize {len(valid_results)} search results for: {search_term}")
        
        try:
            # Construct system prompt for detailed individual analysis
            system_prompt = """You are an expert at analyzing and summarizing web search results. Your task is to create a comprehensive summary that processes each webpage result individually while maintaining focus on the search query. Follow these guidelines:

CONTENT REQUIREMENTS:
1. Focus specifically on information related to the search query: "{search_term}"
2. Process each webpage result individually with detailed analysis
3. Extract key information, facts, data, dates, names, numbers, and concrete details from each source
4. Maintain objectivity and note different perspectives when they exist
5. Use the original language of the content (Chinese/English) as appropriate
6. Preserve important details rather than providing superficial summaries

STRUCTURE REQUIREMENTS:
- Start with a brief overview of the search topic
- Analyze each webpage result individually in separate sections
- For each webpage, include:
  * Title and main findings
  * Key facts, data, and specific details
  * Relevant quotes or important information
  * How it relates to the search query
  * The corresponding saved HTML file location (if available)
- End with a synthesis of key findings across all sources

INDIVIDUAL RESULT ANALYSIS:
For each webpage result, provide:
- Clear section heading with the webpage title
- Detailed extraction of relevant information
- Specific facts, statistics, quotes, and examples
- Analysis of how the content answers the search query
- File location information for reference

TECHNICAL NOTE:
Always end your summary with this important notice:
"📁 **Original Content Storage**: Complete HTML source files and plain text versions of all webpages have been automatically saved to the `web_search_result` folder. For more detailed original content or in-depth analysis, you can use the `read_file` tool to directly access these files, or use the `workspace_search` tool to search for specific information within the folder. File names include search keywords, webpage titles, and timestamps for easy identification and retrieval."

Create a detailed, informative summary that provides substantial value by analyzing each webpage individually."""

            # Prepare content from results with file information
            results_content = []
            for i, result in enumerate(valid_results[:10], 1):  # Limit to top 10 results to avoid token limits
                title = result.get('title', f'Result {i}')
                content = result.get('content', '')[:4000]  # Increased limit for more detailed analysis
                source = result.get('source', 'Unknown')
                
                # Get file path information
                html_file = result.get('saved_html_path', '')
                txt_file = result.get('saved_txt_path', '')
                
                file_info = ""
                if html_file:
                    file_info += f"HTML File: {os.path.basename(html_file)}\n"
                if txt_file:
                    file_info += f"Text File: {os.path.basename(txt_file)}\n"
                
                result_section = f"=== Webpage {i}: {title} (Source: {source}) ===\n"
                if file_info:
                    result_section += f"Saved Files:\n{file_info}\n"
                result_section += f"Content:\n{content}\n"
                
                results_content.append(result_section)
            
            combined_content = "\n".join(results_content)
            
            # Construct user prompt with focus on individual analysis
            user_prompt = f"""Search Query: "{search_term}"

Please provide a comprehensive analysis of the following search results. Focus on extracting information specifically related to the search query "{search_term}". 

Analyze each webpage result individually and provide:
1. A brief overview of the search topic
2. Individual analysis of each webpage result with:
   - Title and main findings related to the search query
   - Key facts, data, statistics, and specific details
   - Important quotes or information
   - How the content answers or relates to the search query
   - Reference to the saved HTML file location
3. A synthesis of key findings across all sources

Search Results to Analyze:
{combined_content}

Please create a detailed, structured analysis that preserves important information from each webpage while focusing on the search query."""

            # Call LLM based on type
            if self.is_claude:
                # Claude API call
                response = self.llm_client.messages.create(
                    model=self.llm_model,
                    max_tokens=6000,  # Increased limit for detailed individual analysis
                    system=system_prompt.format(search_term=search_term),
                    messages=[{"role": "user", "content": user_prompt}],
                    temperature=0.2  # Lower temperature for more structured analysis
                )
                
                if hasattr(response, 'content') and response.content:
                    summary = response.content[0].text if response.content else ""
                else:
                    print_current("⚠️ Claude API response format unexpected")
                    return ""
            else:
                # OpenAI API call
                response = self.llm_client.chat.completions.create(
                    model=self.llm_model,
                    messages=[
                        {"role": "system", "content": system_prompt.format(search_term=search_term)},
                        {"role": "user", "content": user_prompt}
                    ],
                    max_tokens=6000,  # Increased limit for detailed individual analysis
                    temperature=0.2  # Lower temperature for more structured analysis
                )
                
                if response.choices and response.choices[0].message:
                    summary = response.choices[0].message.content
                else:
                    print_current("⚠️ OpenAI API response format unexpected")
                    return ""
            
            # Validate summary
            if summary and summary.strip() and len(summary.strip()) > 100:
                print_current(f"✅ Search results detailed analysis completed: {len(summary)} characters")
                return summary.strip()
            else:
                print_current("⚠️ LLM produced insufficient summary")
                return ""
                
        except Exception as e:
            print_current(f"❌ Search results summarization failed: {e}")
            return ""

    def web_search(self, search_term: str, fetch_content: bool = True, max_content_results: int = 3, **kwargs) -> Dict[str, Any]:
        """
        Search the web for real-time information using Playwright.
        """
        # Check if Playwright is available before proceeding
        if not is_playwright_available():
            print_current("❌ Playwright is not installed or not available")
            print_current("💡 Install with: pip install playwright && playwright install chromium")
            return {
                'status': 'failed',
                'search_term': search_term,
                'results': [{
                    'title': 'Playwright not available',
                    'url': '',
                    'snippet': self._clean_snippet('Playwright library is not installed. Run: pip install playwright && playwright install chromium'),
                    'content': ''
                }],
                'timestamp': datetime.datetime.now().isoformat(),
                'error': 'playwright_not_installed'
            }
        
        # Store current search term for LLM filtering
        self._current_search_term = search_term
        
        
        # Ignore additional parameters
        if kwargs:
            print_current(f"⚠️  Ignoring additional parameters: {list(kwargs.keys())}")
        
        print_debug(f"🔍 Search keywords: {search_term}")
        if fetch_content:
            print_debug(f"📄 Will automatically fetch webpage content for the first {max_content_results} results")
        else:
            print_current(f"📝 Only get search result summaries, not webpage content")
        
        # Set global timeout of 30 seconds for the entire search operation
        old_handler = None
        if not is_windows() and is_main_thread():
            try:
                old_handler = signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(30)
            except ValueError as e:
                print_current(f"⚠️ Cannot set signal handler (not in main thread): {e}")
                old_handler = None
        
        browser = None
        try:
            # Import Playwright (already checked availability above)
            from playwright.sync_api import sync_playwright
            import urllib.parse
            
            results = []
            
            with sync_playwright() as p:
                # Ensure DISPLAY is unset to prevent X11 usage
                import os
                original_display = os.environ.get('DISPLAY')
                if 'DISPLAY' in os.environ:
                    del os.environ['DISPLAY']
                
                try:
                    browser = p.chromium.launch(
                        headless=True,
                        args=[
                            '--no-sandbox',
                            '--disable-setuid-sandbox',
                            '--disable-dev-shm-usage',
                            '--disable-web-security',
                            '--disable-features=VizDisplayCompositor,TranslateUI,AudioServiceOutOfProcess',
                            '--disable-gpu',
                            '--disable-gpu-sandbox',
                            '--disable-software-rasterizer',
                            '--disable-background-timer-throttling',
                            '--disable-renderer-backgrounding',
                            '--disable-backgrounding-occluded-windows',
                            '--disable-extensions',
                            '--disable-default-apps',
                            '--disable-sync',
                            '--disable-background-networking',
                            '--disable-component-update',
                            '--disable-client-side-phishing-detection',
                            '--disable-hang-monitor',
                            '--disable-popup-blocking',
                            '--disable-prompt-on-repost',
                            '--disable-domain-reliability',
                            '--no-first-run',
                            '--no-default-browser-check',
                            '--no-pings',
                            '--disable-remote-debugging',
                            '--disable-http2',
                            '--disable-quic',
                            '--ignore-ssl-errors',
                            '--ignore-certificate-errors',
                            '--disable-dev-shm-usage',
                            '--disable-background-mode',
                            '--disable-features=TranslateUI',
                            '--force-color-profile=srgb',
                            '--disable-ipc-flooding-protection',
                            '--disable-blink-features=AutomationControlled',
                            '--exclude-switches=enable-automation',
                            '--disable-plugins-discovery',
                            '--allow-running-insecure-content',
                            '--disable-web-security',
                            '--disable-features=VizDisplayCompositor'
                        ]
                    )
                    
                    # Create context with more realistic configuration to avoid bot detection
                    context = browser.new_context(
                        user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36',
                        viewport={'width': 1366, 'height': 768},
                        ignore_https_errors=True,
                        java_script_enabled=True,
                        bypass_csp=True,
                        locale='en-US',
                        timezone_id='America/New_York',
                        extra_http_headers={
                            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
                            'Accept-Language': 'en-US,en;q=0.9',
                            'Accept-Encoding': 'gzip, deflate, br',
                            'DNT': '1',
                            'Connection': 'keep-alive',
                            'Upgrade-Insecure-Requests': '1',
                            'Sec-Fetch-Dest': 'document',
                            'Sec-Fetch-Mode': 'navigate',
                            'Sec-Fetch-Site': 'none',
                            'Sec-Fetch-User': '?1',
                            'sec-ch-ua': '"Chromium";v="122", "Not(A:Brand";v="24", "Google Chrome";v="122"',
                            'sec-ch-ua-mobile': '?0',
                            'sec-ch-ua-platform': '"Windows"'
                        }
                    )
                finally:
                    # Restore original DISPLAY if it existed
                    if original_display is not None:
                        os.environ['DISPLAY'] = original_display
                page = context.new_page()
                
                # Set shorter page timeout to prevent hanging
                page.set_default_timeout(8000)  # 8 seconds
                page.set_default_navigation_timeout(12000)  # 12 seconds
                
                # Add stealth script to avoid detection
                page.add_init_script("""
                    // Pass the Webdriver Test
                    Object.defineProperty(navigator, 'webdriver', {
                        get: () => undefined,
                    });
                    
                    // Pass the Chrome Test
                    window.chrome = {
                        runtime: {},
                    };
                    
                    // Pass the Permissions Test
                    const originalQuery = window.navigator.permissions.query;
                    window.navigator.permissions.query = (parameters) => (
                        parameters.name === 'notifications' ?
                            Promise.resolve({ state: Notification.permission }) :
                            originalQuery(parameters)
                    );
                    
                    // Pass the Plugins Length Test
                    Object.defineProperty(navigator, 'plugins', {
                        get: () => [1, 2, 3, 4, 5],
                    });
                    
                    // Pass the Languages Test
                    Object.defineProperty(navigator, 'languages', {
                        get: () => ['en-US', 'en'],
                    });
                """)
                
                if search_term.startswith(('http://', 'https://')):
                    print_current(f"🔗 Direct URL detected, attempting to access: {search_term}")
                    try:
                        page.goto(search_term, timeout=10000, wait_until='domcontentloaded')
                        page.wait_for_timeout(1000)
                        
                        try:
                            title = page.title() or "Untitled page"
                        except:
                            title = "Untitled page"
                        
                        content = ""
                        saved_html_path = ""
                        saved_txt_path = ""
                        
                        # Check if this is a DocIn page by URL or Baidu Scholar page
                        current_url = page.url
                        if "docin.com" in current_url or "豆丁网" in title:
                            print_current("⚠️ Detected DocIn embedded document page by URL/title")
                            content = "正文为嵌入式文档，不可阅读"
                        elif ("百度学术搜索" in title or "xueshu.baidu.com" in current_url or 
                              "百度学术" in title or "- 百度学术" in title or
                              "search.baidu.com" in current_url):
                            print_current("⚠️ Detected Baidu Scholar search page by URL/title")
                            content = "结果无可用数据"
                        else:
                            if fetch_content:
                                content = self._extract_main_content(page)
                            print_current(f"📄 Extracted webpage content length: {len(content)} characters")
                            
                            # Apply LLM filtering if enabled
                            if content and self.enable_llm_filtering:
                                content = self._extract_relevant_content_with_llm(content, search_term, title)
                            
                            # Save both HTML and text content to files
                            if content and len(content.strip()) > 100:
                                saved_html_path, saved_txt_path = self._save_webpage_content(page, search_term, title, content, search_term)
                        
                        # Clean content for better LLM processing
                        cleaned_content = self._clean_text_for_saving(content) if content else ""
                        
                        result_dict = {
                            'title': title,
                            'snippet': self._clean_snippet(f'Direct URL access successful: {title}'),
                            'source': 'direct_access',
                            'content': cleaned_content if cleaned_content else (content if content else "Unable to get webpage content"),
                            'content_length': len(cleaned_content if cleaned_content else content),
                            'access_status': 'success'
                        }
                        
                        # Add saved file paths if available
                        if saved_html_path:
                            result_dict['saved_html_path'] = saved_html_path
                        if saved_txt_path:
                            result_dict['saved_txt_path'] = saved_txt_path
                        
                        results.append(result_dict)
                        
                        print_current(f"✅ Successfully accessed URL directly, got {len(content)} characters of content")
                        
                    except Exception as e:
                        print_current(f"❌ Direct URL access failed: {e}")
                        results.append({
                            'title': f'Access failed: {search_term}',
                            'url': search_term,
                            'snippet': self._clean_snippet(f'Direct URL access failed: {str(e)}'),
                            'source': 'direct_access_failed',
                            'content': f'Access failed: {str(e)}',
                            'error': str(e),
                            'access_status': 'failed'
                        })
                else:
                    # Initialize available search engines
                    search_engines = []
                    
                    # Check Google connectivity first and prioritize if available
                    if self._check_google_connectivity():
                        # Primary Google search
                        search_engines.append({
                            'name': 'Google',
                            'url': 'https://www.google.com/search?q={}&gl=us&hl=en&safe=off',
                            'result_selector': 'h3 a, h1 a, .g a h3, .yuRUbf a h3, .LC20lb, .DKV0Md, [data-ved] h3',
                            'container_selector': '.g, .tF2Cxc, [data-ved]',
                            'snippet_selectors': ['.VwiC3b', '.s', '.st', 'span', '.IsZvec', '.aCOpRe', '.yXK7lf'],
                            'anti_bot_indicators': ['Our systems have detected unusual traffic', 'g-recaptcha', 'captcha', 'verify you are human', 'blocked', 'unusual activity']
                        })
                        print_debug("🔍 Google search engine added as primary option (connectivity confirmed)")
                    else:
                        print_debug("⚠️ Google connectivity test failed, will use alternative search engines")
                    
                    # Always add Baidu as fallback or secondary option
                    search_engines.append({
                        'name': 'Baidu',
                        'url': 'https://www.baidu.com/s?wd={}',
                        'result_selector': 'h3.t a',
                        'container_selector': '.result',
                        'snippet_selectors': ['.c-abstract', '.c-span9', 'span', 'div']
                    })
                    print_debug("🔍 Baidu search engine added to available options")
                    
                    if not search_engines:
                        print_current("No search engines available")
                        return {
                            'status': 'failed',
                            'search_term': search_term,
                            'results': [{
                                'title': 'No search engines available',
                                'url': '',
                                'snippet': self._clean_snippet('All search engines are unavailable due to connectivity issues.'),
                                'content': 'No search engines available'
                            }],
                            'timestamp': datetime.datetime.now().isoformat(),
                            'error': 'no_search_engines_available'
                        }
                    
                    optimized_search_term = self._optimize_search_term(search_term)
                    encoded_term = urllib.parse.quote_plus(optimized_search_term)
                    
                    for engine in search_engines:
                        try:
                            print_debug(f"🔍 Trying to search with {engine['name']}...")
                            
                            # Add rate limiting for Google to avoid being blocked
                            if engine['name'].startswith('Google'):
                                current_time = time.time()
                                time_since_last_request = current_time - self._last_google_request
                                # Delay 1-3 seconds to avoid anti-bot detection
                                import random
                                min_delay = 1.0
                                max_delay = 3.0
                                required_delay = random.uniform(min_delay, max_delay)
                                if time_since_last_request < required_delay:
                                    wait_time = required_delay - time_since_last_request
                                    self._verbose_print(f"⏱️ Rate limiting: waiting {wait_time:.1f} seconds before Google request")
                                    time.sleep(wait_time)
                                self._last_google_request = time.time()
                            
                            search_url = engine['url'].format(encoded_term)
                            
                            # Use very short timeout for search engines
                            page.goto(search_url, timeout=6000, wait_until='domcontentloaded')
                            
                            # Add random delay to mimic human behavior (500-1500ms)
                            import random
                            human_delay = random.randint(500, 1500)
                            page.wait_for_timeout(human_delay)
                            
                            # Add some mouse movement to mimic human behavior
                            try:
                                page.mouse.move(random.randint(100, 500), random.randint(100, 400))
                                page.wait_for_timeout(random.randint(100, 300))
                            except:
                                pass
                            
                            # Check for anti-bot mechanisms (especially for Google)
                            if engine['name'] == 'Google' and 'anti_bot_indicators' in engine:
                                page_content = page.content()
                                for indicator in engine['anti_bot_indicators']:
                                    if indicator.lower() in page_content.lower():
                                        self._verbose_print(f"⚠️ {engine['name']} detected anti-bot mechanism: {indicator}")
                                        self._verbose_print(f"🔄 Skipping {engine['name']} due to anti-bot protection")
                                        raise Exception(f"Anti-bot protection detected: {indicator}")
                            
                            # Get search results with error handling
                            try:
                                result_elements = page.query_selector_all(engine['result_selector'])
                            except Exception as selector_error:
                                print_debug(f"⚠️ Selector error for {engine['name']}: {selector_error}")
                                # Fallback to basic result selector
                                try:
                                    result_elements = page.query_selector_all('a[href], h3, .result, .g, .rc')
                                except Exception as fallback_error:
                                    print_debug(f"❌ Fallback selector also failed: {fallback_error}")
                                    result_elements = []
                            
                            if result_elements:
                                print_debug(f"✅ {engine['name']} search successful, found {len(result_elements)} results")
                                
                                for i, elem in enumerate(result_elements[:10]):  # Top 5 results
                                    try:
                                        title = ""
                                        url = ""
                                        snippet = ""
                                        
                                        # Extract title and URL based on engine type
                                        if engine['name'].startswith('Google'):
                                            # For Google results, handle different element types
                                            tag_name = 'unknown'
                                            try:
                                                tag_name = elem.evaluate('element => element.tagName.toLowerCase()')
                                            except Exception as evaluate_error:
                                                print_debug(f"⚠️ Element evaluate error: {evaluate_error}")
                                                # Fallback: try to determine tag type from element properties
                                                try:
                                                    if hasattr(elem, 'tag_name'):
                                                        tag_name = elem.tag_name.lower()
                                                except:
                                                    pass
                                            
                                            if tag_name == 'h3':
                                                title = elem.text_content().strip()
                                                # Find parent link element with error handling
                                                parent_link = None
                                                try:
                                                    parent_link = elem.query_selector('xpath=ancestor::a[1]')
                                                except Exception as parent_error:
                                                    print_debug(f"⚠️ Parent link selector error: {parent_error}")
                                                
                                                if parent_link:
                                                    url = parent_link.get_attribute('href') or ""
                                                else:
                                                    # Try to find sibling or nearby link with error handling
                                                    nearby_link = None
                                                    try:
                                                        nearby_link = elem.query_selector('xpath=../a | xpath=../../a | xpath=../../../a')
                                                    except Exception as nearby_error:
                                                        print_debug(f"⚠️ Nearby link selector error: {nearby_error}")
                                                    
                                                    url = nearby_link.get_attribute('href') if nearby_link else ""
                                            else:
                                                # For link elements
                                                title = elem.text_content().strip()
                                                url = elem.get_attribute('href') or ""
                                        else:
                                            # For other search engines
                                            title = elem.text_content().strip()
                                            url = elem.get_attribute('href') or ""
                                        
                                        if url.startswith('/'):
                                            if engine['name'] == 'Zhihu':
                                                url = 'https://www.zhihu.com' + url
                                            elif engine['name'] == 'Wikipedia':
                                                url = 'https://en.wikipedia.org' + url
                                        
                                        snippet = self._extract_snippet_from_search_result(elem, engine)
                                        
                                        # Handle Google URL format
                                        if url and url.startswith('/url?q='):
                                            url = urllib.parse.unquote(url.split('&')[0][7:])
                                        
                                        # Handle Baidu redirect URLs - mark for special processing
                                        elif url and 'baidu.com/link?url=' in url:
                                            pass  # Baidu redirect URL detected, using extended timeout
                                            # Mark this as a redirect URL for special handling during content fetch
                                            pass
                                        
                                        if title and len(title) > 5:
                                            # Clean snippet before truncating
                                            cleaned_snippet = self._clean_snippet(snippet) if snippet else f'Search result from {engine["name"]}'
                                            results.append({
                                                'title': title,
                                                'snippet': cleaned_snippet[:get_truncation_length()] if cleaned_snippet else f'Search result from {engine["name"]}',
                                                'source': engine['name'],
                                                'content': '',
                                                '_internal_url': url  # Keep URL internally for content fetching
                                            })
                                    
                                    except Exception as e:
                                        print_current(f"Error extracting result {i}: {e}")
                                        continue
                                
                                if results:
                                    break
                            else:
                                print_current(f"❌ {engine['name']} found no search results")
                        
                        except Exception as e:
                            print_current(f"❌ {engine['name']} search failed: {e}")
                            continue
                    
                    if fetch_content and results:
                        print_debug(f"\n🚀 Starting to fetch webpage content for first {min(max_content_results, len(results))} results using parallel processing...")
                        
                        # Use parallel processing for better efficiency
                        try:
                            # Close the shared page since parallel processing will create its own browsers
                            page = None
                            
                            # Use parallel content fetching with 2 concurrent workers to avoid anti-bot detection
                            self._fetch_webpage_content_parallel(results[:max_content_results], max_workers=2)
                            
                        except Exception as e:
                            print_current(f"⚠️ Parallel content fetching failed: {e}")
                            print_current(f"🔄 Falling back to sequential content fetching...")
                            # Fallback to sequential method if parallel fails
                            try:
                                # Need to recreate page for fallback
                                context = browser.new_context(
                                    user_agent='Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                                    viewport={'width': 1024, 'height': 768},
                                    ignore_https_errors=True,
                                    java_script_enabled=True,
                                    bypass_csp=True
                                )
                                page = context.new_page()
                                
                                self._fetch_webpage_content_with_timeout(results[:max_content_results], page, timeout_seconds=20)
                            except Exception as fallback_e:
                                print_current(f"⚠️ Fallback sequential content fetching also failed: {fallback_e}")
                                # Final fallback - try basic method
                                try:
                                    self._fetch_webpage_content(results[:max_content_results], page)
                                except Exception as final_e:
                                    print_current(f"⚠️ All content fetching methods failed: {final_e}")
                        
                        valid_results = []
                        for result in results:
                            # Keep results with good content, or results without content but with snippets
                            if result.get('content') and len(result['content'].strip()) > 100:
                                valid_results.append(result)
                            elif not fetch_content:
                                valid_results.append(result)
                            elif result.get('snippet') and len(result['snippet'].strip()) > 20:
                                # Keep results with useful snippets even if content fetch failed
                                valid_results.append(result)
                        
                        if valid_results:
                            results = valid_results
                            print_debug(f"✅ Successfully got {len(results)} search results with valid content")
                        else:
                            print_current("⚠️ All search results failed to get valid webpage content, returning search results only")
                            # Return search results even without content
                            for result in results:
                                if not result.get('content'):
                                    result['content'] = 'Content not available - search result only'
                
                # Ensure browser is closed
                try:
                    browser.close()
                except:
                    pass
            
            if not results:
                print_current("🔄 All search engines failed, providing fallback result...")
                results = [{
                    'title': f'Search: {search_term}',
                    'snippet': self._clean_snippet(f'Failed to get results from search engines. Possible reasons: network connection issues, search engine structure changes, or access restrictions. Recommend manual search: {search_term}'),
                    'source': 'fallback',
                    'content': ''
                }]
            
            if fetch_content:
                optimized_results = []
                for result in results:
                    if result.get('content') and len(result['content'].strip()) > 100:
                        content = result['content']
                        summary_truncation_length = get_truncation_length() // 5  # 使用截断长度的1/5
                        content_summary = content[:summary_truncation_length] + "..." if len(content) > summary_truncation_length else content
                        
                        optimized_result = {
                            'title': result['title'],
                            'content': content,
                            'content_summary': content_summary,
                            'full_content': content,
                            'source': result['source'],
                            'has_full_content': True,
                            'content_length': result.get('content_length', len(content)),
                            'access_status': result.get('access_status', 'success')
                        }
                        # URL removed from results as requested
                        
                        if 'snippet' in result:
                            optimized_result['snippet'] = self._clean_snippet(result['snippet'])
                        
                        optimized_results.append(optimized_result)
                    else:
                        result_copy = result.copy()
                        result_copy.update({
                            'has_full_content': False,
                            'content_status': 'Unable to get webpage content'
                        })
                        optimized_results.append(result_copy)
                
                if optimized_results:
                    results = optimized_results
                    print_debug(f"✅ Optimized search result format, {len([r for r in results if r.get('has_full_content')])} results contain full content")
            
            # Count saved files
            saved_html_count = len([r for r in results if r.get('saved_html_path')])
            saved_txt_count = len([r for r in results if r.get('saved_txt_path')])
            
            # Generate summary if enabled and content was fetched
            summary = ""
            if fetch_content and self.enable_summary and results:
                summary = self._summarize_search_results_with_llm(results, search_term)
            
            # Check total txt files in web_search_result directory
            total_txt_files = self._count_txt_files_in_result_dir()
            
            # Clean all snippets in results to remove excessive whitespace
            for result in results:
                if 'snippet' in result and result['snippet']:
                    result['snippet'] = self._clean_snippet(result['snippet'])
            
            result_data = {
                'status': 'success',
                'search_term': search_term,
                'results': results,
                'timestamp': datetime.datetime.now().isoformat(),
                'total_results': len(results),
                'content_fetched': fetch_content,
                'results_with_content': len([r for r in results if r.get('has_full_content')]) if fetch_content else 0,
                'saved_html_files': saved_html_count,
                'saved_txt_files': saved_txt_count,
                'total_txt_files_in_directory': total_txt_files
            }
            
            # Add warning if there are too many txt files
            if total_txt_files > 10:
                result_data['search_material_warning'] = f"⚠️ Enough materials have been collected ({total_txt_files} text files). Please do not call the search again in the next round."
            
            # Add summary to result data if available
            if summary:
                result_data['summary'] = summary
                result_data['summary_available'] = True
                
                # Replace detailed results with simplified summary-focused results for LLM
                simplified_results = []
                for i, result in enumerate(results[:5], 1):  # Keep only top 5 for reference
                    simplified_result = {
                        'title': result.get('title', f'Result {i}'),
                        'source': result.get('source', 'Unknown'),
                        'content_available': bool(result.get('content') and len(result.get('content', '').strip()) > 50)
                    }
                    simplified_results.append(simplified_result)
                
                # Replace the detailed results with simplified ones for LLM
                result_data['detailed_results_replaced_with_summary'] = True
                result_data['simplified_results'] = simplified_results
                result_data['total_results_processed'] = len(results)
                
                # Remove the detailed results array to avoid overwhelming LLM
                del result_data['results']
                
                print_current(f"📋 Generated comprehensive summary ({len(summary)} characters)")
                print_current(f"\n🎯 Final Summary for Search Term: '{search_term}'")
                print_current(f"{'='*60}")
                print(summary)
                print_current(f"{'='*60}\n")
            else:
                result_data['summary_available'] = False
            
            # Add helpful message about saved files and original content access
            file_notice_parts = []
            if saved_html_count > 0 or saved_txt_count > 0:
                files_info = []
                if saved_html_count > 0:
                    files_info.append(f"{saved_html_count} HTML files")
                if saved_txt_count > 0:
                    files_info.append(f"{saved_txt_count} text files")
                
                files_str = " and ".join(files_info)
                file_notice_parts.append(f"📁 {files_str} saved to folder: {self.web_result_dir}/")
                file_notice_parts.append("💡 You can use workspace_search or grep_search tools to search within these files")
                
                #print_current(f"\n📁 {files_str} saved to folder: {self.web_result_dir}/")
                #print_current(f"💡 You can use workspace_search or grep_search tools to search within these files")
            
            # Always add notice about original content access
            if summary:
                file_notice_parts.append("📄 Complete original webpage content is stored in the saved files above")
                file_notice_parts.append("🔍 Use the saved files to access full details beyond this summary")
            
            if file_notice_parts:
                result_data['files_notice'] = "\n".join(file_notice_parts)
            
            return result_data
        
        except Exception as playwright_error:
            # Handle Playwright browser launch errors (including GLIBC issues)
            error_str = str(playwright_error)
            if 'GLIBC' in error_str or 'version' in error_str or 'not found' in error_str:
                print_current(f"❌ Playwright system compatibility error: {playwright_error}")
                print_current("⚠️  Your system GLIBC version is too old for Playwright")
                print_current("💡 Suggestion: Try using requests-based fallback or upgrade your system")
                return {
                    'status': 'failed',
                    'search_term': search_term,
                    'results': [{
                        'title': 'System compatibility issue',
                        'url': '',
                        'snippet': self._clean_snippet(f'Playwright requires GLIBC 2.28+ but your system has an older version. Error: {error_str}'),
                        'content': 'Please upgrade your system or use alternative search method'
                    }],
                    'timestamp': datetime.datetime.now().isoformat(),
                    'error': 'glibc_compatibility_error'
                }
            elif 'PlaywrightContextManager' in error_str or '_playwright' in error_str:
                print_current(f"❌ Playwright initialization error: {playwright_error}")
                print_current("💡 This might be due to browser installation issues")
                return {
                    'status': 'failed',
                    'search_term': search_term,
                    'results': [{
                        'title': 'Playwright initialization failed',
                        'url': '',
                        'snippet': self._clean_snippet(f'Playwright failed to initialize properly. Error: {error_str}'),
                        'content': 'Browser initialization failed, please check Playwright installation'
                    }],
                    'timestamp': datetime.datetime.now().isoformat(),
                    'error': 'playwright_init_error'
                }
            else:
                print_current(f"❌ Playwright error: {playwright_error}")
                return {
                    'status': 'failed',
                    'search_term': search_term,
                    'results': [{
                        'title': f'Playwright error: {search_term}',
                        'url': '',
                        'snippet': self._clean_snippet(f'Playwright failed with error: {str(playwright_error)}'),
                        'content': f'Playwright error: {str(playwright_error)}'
                    }],
                    'timestamp': datetime.datetime.now().isoformat(),
                    'error': str(playwright_error)
                }
        
        except TimeoutError:
            return {
                'status': 'failed',
                'search_term': search_term,
                'results': [{
                    'title': f'Search timeout: {search_term}',
                    'url': '',
                    'snippet': self._clean_snippet('Web search operation timed out after 60 seconds.'),
                    'content': 'Search operation timed out'
                }],
                'timestamp': datetime.datetime.now().isoformat(),
                'error': 'search_timeout'
            }
        
        except Exception as e:
            print_current(f"❌ Web search failed: {e}")
            return {
                'status': 'failed',
                'search_term': search_term,
                'results': [{
                    'title': f'Search error: {search_term}',
                    'url': '',
                    'snippet': self._clean_snippet(f'Web search failed with error: {str(e)}'),
                    'content': f'Search error: {str(e)}'
                }],
                'timestamp': datetime.datetime.now().isoformat(),
                'error': str(e)
            }
        
        finally:
            # Reset the alarm and restore the original signal handler
            if not is_windows() and is_main_thread() and old_handler is not None:
                try:
                    signal.alarm(0)
                    signal.signal(signal.SIGALRM, old_handler)
                except ValueError:
                    # Already not in main thread, nothing to clean up
                    pass
            
            # Emergency browser cleanup
            if browser:
                try:
                    browser.close()
                except:
                    pass

    def _clean_snippet(self, snippet: str) -> str:
        """
        Clean snippet text by removing excessive whitespace characters
        Removes all types of whitespace (spaces, newlines, tabs, etc.) and normalizes to single spaces
        """
        import re
        
        if not snippet:
            return ""
        
        # Remove all types of whitespace characters (spaces, newlines, tabs, etc.)
        # Replace multiple consecutive whitespace characters with a single space
        cleaned = re.sub(r'\s+', ' ', snippet)
        
        # Strip leading and trailing whitespace
        cleaned = cleaned.strip()
        
        return cleaned
    
    def _extract_snippet_from_search_result(self, elem, engine) -> str:
        """
        Extract snippet/description from search result element
        """
        snippet = ""
        
        try:
            container = None
            if engine['container_selector']:
                try:
                    container = elem.query_selector(f'xpath=ancestor::*[contains(@class, "{engine["container_selector"].replace(".", "")}")]')
                except Exception as xpath_error:
                    print_debug(f"⚠️ XPath selector error: {xpath_error}")
            
            if not container:
                try:
                    container = elem.query_selector('xpath=ancestor::div[2]')
                except Exception as ancestor_error:
                    print_debug(f"⚠️ Ancestor selector error: {ancestor_error}")
            
            if container:
                for selector in engine['snippet_selectors']:
                    try:
                        snippet_elem = container.query_selector(selector)
                        if snippet_elem:
                            text = snippet_elem.text_content().strip()
                            if text and len(text) > 20 and not text.startswith('http') and '...' not in text[:10]:
                                snippet = text
                                break
                    except Exception as snippet_error:
                        print_debug(f"⚠️ Snippet selector error: {snippet_error}")
                        continue
                
                if not snippet:
                    try:
                        all_text_elems = container.query_selector_all('span, div, p')
                        for text_elem in all_text_elems:
                            try:
                                text = text_elem.text_content().strip()
                                if text and len(text) > 30 and len(text) < 200:
                                    if any(char in text for char in '，。？！、；：') or ' ' in text:
                                        snippet = text
                                        break
                            except Exception as text_error:
                                continue
                    except Exception as text_elements_error:
                        print_debug(f"⚠️ Text elements selector error: {text_elements_error}")
        
        except Exception as e:
            print_current(f"Error extracting snippet: {e}")
        
        # Clean the snippet to remove excessive whitespace
        return self._clean_snippet(snippet)

    def _fetch_webpage_content_with_timeout(self, results: List[Dict], page, timeout_seconds: int = 25) -> None:
        """
        Fetch webpage content with additional timeout control
        """
        start_time = time.time()
        
        for i, result in enumerate(results):
            if time.time() - start_time > timeout_seconds:
                print_current(f"⏰ Overall content fetching timeout reached ({timeout_seconds}s), stopping")
                break
                
            try:
                print_current(f"📖 Getting webpage content for result {i+1}: {result['title'][:50]}...")
                
                target_url = result.get('_internal_url') or result.get('url', '')
                
                # Handle Baidu redirect URLs with extended timeout
                is_baidu_redirect = 'baidu.com/link?url=' in target_url
                if is_baidu_redirect:
                    
                    # Try to decode the redirect URL first
                    decoded_url = self._decode_baidu_redirect_url(target_url)
                    if decoded_url != target_url:
                        target_url = decoded_url
                        print_current(f"🎯 Using decoded URL instead of redirect")
                        is_baidu_redirect = False  # No longer need special handling
                
                # Quick domain check
                problematic_domains = [
                    'douyin.com', 'tiktok.com', 'youtube.com', 'youtu.be',
                    'bilibili.com', 'b23.tv', 'instagram.com', 'facebook.com',
                    'twitter.com', 'x.com', 'linkedin.com'
                ]
                
                if any(domain in target_url.lower() for domain in problematic_domains):
                    print_current(f"⏭️ Skip video/social media link: {target_url}")
                    result['content'] = "Video or social media link, skip content fetch"
                    continue
                
                if target_url.startswith(('javascript:', 'mailto:')):
                    print_current(f"⏭️ Skip non-webpage link: {target_url}")
                    result['content'] = "Non-webpage link, skip content fetch"
                    continue
                
                # Navigate to page with short timeout
                remaining_time = max(5000, int((timeout_seconds - (time.time() - start_time)) * 1000))
                if remaining_time <= 5000:
                    # print_current("⏰ Insufficient remaining time, skip this result")
                    result['content'] = "Insufficient time, skip content fetch"
                    continue
                
                try:
                    # Use optimized timeout for faster processing
                    timeout_ms = min(remaining_time, 10000)
                    
                    # Reduce retry attempts to 1 (no retry) for faster processing
                    max_retries = 1
                    success = False
                    last_error = None
                    
                    for attempt in range(max_retries):
                        try:
                            if attempt > 0:
                                # print_current(f"🔄 Retry attempt {attempt + 1} for: {result['title'][:30]}...")
                                page.wait_for_timeout(500)  # Reduced wait before retry for faster processing
                            
                            # Skip special header settings for faster processing
                            
                            page.goto(target_url, timeout=timeout_ms, wait_until='domcontentloaded')
                            
                            # Optimized wait time for faster processing
                            initial_wait = 1000
                            page.wait_for_timeout(initial_wait)
                            
                            # Check if we've been redirected to an error page
                            current_url = page.url
                            if 'chrome-error://' in current_url or 'about:blank' in current_url:
                                raise Exception(f"Redirected to error page: {current_url}")
                            
                            # Skip additional wait for faster processing
                            # Additional wait removed to optimize performance
                            
                            success = True
                            break
                            
                        except Exception as nav_error:
                            last_error = nav_error
                            error_msg = str(nav_error).lower()
                            
                            # Special handling for common Baidu redirect errors
                            if is_baidu_redirect and ('timeout' in error_msg or 'net::err' in error_msg):
                                # print_current(f"⚠️ Navigation attempt {attempt + 1} failed (Baidu redirect timeout): {nav_error}")  # Commented out to reduce terminal noise
                                if attempt < max_retries - 1:
                                    continue
                            elif attempt < max_retries - 1:
                                # print_current(f"⚠️ Navigation attempt {attempt + 1} failed: {nav_error}")  # Commented out to reduce terminal noise
                                continue
                            else:
                                raise nav_error
                    
                    if success:
                        content = self._extract_main_content(page)
                        
                        if content and len(content.strip()) > 100:
                            search_term = getattr(self, '_current_search_term', '')
                            title = result.get('title', 'Untitled')
                            
                            # Apply LLM filtering if enabled
                            if self.enable_llm_filtering:
                                content = self._extract_relevant_content_with_llm(content, search_term, title)
                            
                            # Save both HTML and text content to files
                            saved_html_path, saved_txt_path = self._save_webpage_content(page, target_url, title, content, search_term)
                            if saved_html_path:
                                result['saved_html_path'] = saved_html_path
                            if saved_txt_path:
                                result['saved_txt_path'] = saved_txt_path
                            
                            # Clean content for better LLM processing
                            cleaned_content = self._clean_text_for_saving(content)
                            result['content'] = cleaned_content if cleaned_content else content
                            result['final_url'] = page.url
                            print_current(f"✅ Successfully got {len(result['content'])} characters of useful content")
                            if is_baidu_redirect:
                                print_current(f"🎯 Final redirected URL: {page.url}")
                        else:
                            result['content'] = "Content too short or unable to extract"
                
                except Exception as e:
                    error_msg = str(e)
                    error_type = "unknown"
                    
                    # Enhanced error classification for better debugging
                    if "ERR_HTTP2_PROTOCOL_ERROR" in error_msg:
                        error_msg = "HTTP2 protocol error (server connectivity issue)"
                        error_type = "protocol_error"
                    elif "ERR_NETWORK_CHANGED" in error_msg:
                        error_msg = "Network connection changed during request"
                        error_type = "network_changed"
                    elif "ERR_CONNECTION_REFUSED" in error_msg:
                        error_msg = "Connection refused by server"
                        error_type = "connection_refused"
                    elif "ERR_CONNECTION_TIMED_OUT" in error_msg:
                        error_msg = "Connection timed out"
                        error_type = "connection_timeout"
                    elif "interrupted by another navigation" in error_msg:
                        error_msg = "Navigation interrupted (redirect loop or server issue)"
                        error_type = "navigation_interrupted"
                    elif "chrome-error://" in error_msg:
                        error_msg = "Browser error page (server unreachable)"
                        error_type = "error_page"
                    elif "timeout" in error_msg.lower() or "timed out" in error_msg.lower():
                        if is_baidu_redirect:
                            error_msg = "Baidu redirect timeout (server too slow, consider using longer timeout)"
                        else:
                            error_msg = "Page load timeout (server too slow or unreachable)"
                        error_type = "timeout"
                    elif "net::" in error_msg:
                        error_msg = "Network connectivity error"
                        error_type = "network_error"
                    elif "SSL" in error_msg or "TLS" in error_msg:
                        error_msg = "SSL/TLS certificate error"
                        error_type = "ssl_error"
                    
                    # print_current(f"❌ Failed to get webpage content: {error_msg}")  # Commented out to reduce terminal noise
                    result['content'] = f"Content fetch error: {error_msg}"
                    result['error_type'] = error_type
                    
                    # Special message for Baidu redirects
                    if is_baidu_redirect:
                        pass  # print_current(f"💡 Baidu redirect troubleshooting: URL={target_url[:100]}...")  # Commented out to reduce terminal noise
                
            except Exception as e:
                # print_current(f"❌ Error processing result {i+1}: {e}")  # Commented out to reduce terminal noise
                result['content'] = f"Processing error: {str(e)}"

    def _fetch_webpage_content_parallel(self, results: List[Dict], max_workers: int = 2) -> None:
        """
        Fetch webpage content for multiple results in parallel using ThreadPoolExecutor

        Args:
            results: List of search results to fetch content for
            max_workers: Maximum number of concurrent workers (default: 2)
        """
        if not results:
            return
        
        print_debug(f"🚀 Starting parallel content fetching for {len(results)} results with {max_workers} workers")
        
        # Filter results that need content fetching
        results_to_fetch = []
        for i, result in enumerate(results):
            target_url = result.get('_internal_url') or result.get('url', '')
            if not target_url:
                result['content'] = "No URL available"
                continue
            
            # Quick domain check - skip problematic domains
            problematic_domains = [
                'douyin.com', 'tiktok.com', 'youtube.com', 'youtu.be',
                'bilibili.com', 'b23.tv', 'instagram.com', 'facebook.com',
                'twitter.com', 'x.com', 'linkedin.com'
            ]
            
            if any(domain in target_url.lower() for domain in problematic_domains):
                # print_current(f"⏭️ Skip video/social media link: {result['title'][:30]}...")
                result['content'] = "Video or social media link, skip content fetch"
                continue
            
            if target_url.startswith(('javascript:', 'mailto:')):
                # print_current(f"⏭️ Skip non-webpage link: {result['title'][:30]}...")
                result['content'] = "Non-webpage link, skip content fetch"
                continue
            
            results_to_fetch.append((i, result))
        
        if not results_to_fetch:
            print_current("⚠️ No valid URLs to fetch content from")
            return
        
        print_debug(f"📊 Processing {len(results_to_fetch)} valid URLs for content fetching")
        
        # Use ThreadPoolExecutor for parallel processing with improved timeout handling
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks with delays to avoid triggering anti-bot detection
            future_to_result = {}
            import random
            for idx, (i, result) in enumerate(results_to_fetch):
                # Add progressive delay: first request immediately, then 1-3 seconds between requests
                if idx > 0:
                    delay = random.uniform(1.0, 3.0)  # Random delay between 1-3 seconds
                    print_debug(f"⏱️ Waiting {delay:.1f} seconds before submitting request {idx+1}/{len(results_to_fetch)}")
                    time.sleep(delay)
                
                future = executor.submit(self._fetch_single_webpage_content, result, i)
                future_to_result[future] = (i, result)
            
            # Collect results as they complete with progressive timeout
            completed = 0
            timeout_per_batch = max(30, len(results_to_fetch) * 8)  # Adaptive timeout
            
            try:
                for future in as_completed(future_to_result, timeout=timeout_per_batch):
                    try:
                        i, original_result = future_to_result[future]
                        updated_result = future.result(timeout=5)  # Quick result extraction
                        
                        # Update the original result with fetched content
                        if updated_result:
                            original_result.update(updated_result)
                        
                        completed += 1
                        
                    except Exception as e:
                        i, original_result = future_to_result[future]
                        # print_current(f"❌ Error fetching content for result {i+1}: {e}")  # Commented out to reduce terminal noise
                        original_result['content'] = f"Parallel fetch error: {str(e)}"
                        completed += 1
                        
            except concurrent.futures.TimeoutError:
                print_current(f"⏰ Parallel processing timeout after {timeout_per_batch}s, handling remaining tasks...")
                
                # Handle remaining futures that didn't complete
                for future in future_to_result:
                    if not future.done():
                        try:
                            future.cancel()
                        except:
                            pass
                        
                        i, original_result = future_to_result[future]
                        if 'content' not in original_result or not original_result.get('content'):
                            original_result['content'] = "Parallel fetch timeout"
                        completed += 1
        
        print_debug(f"🎉 Parallel content fetching completed for all {len(results_to_fetch)} results")

    def _fetch_single_webpage_content(self, result: Dict, index: int) -> Dict[str, Any]:
        """
        Fetch content for a single webpage (used by parallel processing)
        
        Args:
            result: Single search result to fetch content for
            index: Index of the result for logging
            
        Returns:
            Updated result dictionary with content
        """
        # Check if Playwright is available before proceeding
        if not is_playwright_available():
            result['content'] = "Playwright not available. Install with: pip install playwright && playwright install chromium"
            return result
        
        from playwright.sync_api import sync_playwright
        
        target_url = result.get('_internal_url') or result.get('url', '')
        
        try:
            # Add small random delay before starting to fetch (1-3 seconds) to avoid triggering anti-bot detection
            import random
            pre_delay = random.uniform(1.0, 3.0)
            time.sleep(pre_delay)
            
            print_debug(f"🔗 [{index+1}] Fetching content: {result['title'][:40]}...")
            
            # Handle Baidu redirect URLs
            is_baidu_redirect = 'baidu.com/link?url=' in target_url
            if is_baidu_redirect:
                
                decoded_url = self._decode_baidu_redirect_url(target_url)
                if decoded_url != target_url:
                    target_url = decoded_url
                    # print_current(f"🎯 [{index+1}] Using decoded URL")
                    is_baidu_redirect = False
            
            with sync_playwright() as p:
                # Ensure DISPLAY is unset to prevent X11 usage
                import os
                original_display = os.environ.get('DISPLAY')
                if 'DISPLAY' in os.environ:
                    del os.environ['DISPLAY']
                
                try:
                    browser = p.chromium.launch(
                        headless=True,
                        args=[
                            '--no-sandbox',
                            '--disable-setuid-sandbox',
                            '--disable-dev-shm-usage',
                            '--disable-web-security',
                            '--disable-features=VizDisplayCompositor',
                            '--disable-gpu',
                            '--disable-gpu-sandbox',
                            '--disable-software-rasterizer',
                            '--disable-background-timer-throttling',
                            '--disable-renderer-backgrounding',
                            '--disable-extensions',
                            '--disable-default-apps',
                            '--disable-sync',
                            '--no-first-run',
                            '--no-default-browser-check',
                            '--no-pings',
                            '--disable-remote-debugging',
                            '--disable-http2',
                            '--disable-quic',
                            '--ignore-ssl-errors',
                            '--ignore-certificate-errors',
                            '--disable-background-mode',
                            '--disable-features=TranslateUI',
                            '--force-color-profile=srgb',
                            '--disable-ipc-flooding-protection'
                        ]
                    )
                    
                    context = browser.new_context(
                        user_agent='Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                        viewport={'width': 1024, 'height': 768},
                        ignore_https_errors=True,
                        java_script_enabled=True,
                        bypass_csp=True
                    )
                finally:
                    # Restore original DISPLAY if it existed
                    if original_display is not None:
                        os.environ['DISPLAY'] = original_display
                
                page = context.new_page()
                
                # Use optimized timeout for parallel processing - reduced for faster processing
                timeout_ms = 10000
                max_retries = 1
                
                success = False
                for attempt in range(max_retries):
                    try:
                        if attempt > 0:
                            print_current(f"🔄 [{index+1}] Retry attempt {attempt + 1}")
                            page.wait_for_timeout(500)
                        
                        # Skip special header settings for faster processing
                        
                        page.goto(target_url, timeout=timeout_ms, wait_until='domcontentloaded')
                        
                        # Optimized wait times for faster processing
                        wait_time = 500
                        page.wait_for_timeout(wait_time)
                        
                        # Check for error pages
                        current_url = page.url
                        if 'chrome-error://' in current_url or 'about:blank' in current_url:
                            raise Exception(f"Redirected to error page: {current_url}")
                        
                        success = True
                        break
                        
                    except Exception as nav_error:
                        if attempt < max_retries - 1:
                            print_current(f"⚠️ [{index+1}] Navigation attempt {attempt + 1} failed: {nav_error}")
                            continue
                        else:
                            raise nav_error
                
                if success:
                    content = self._extract_main_content(page)
                    
                    if content and len(content.strip()) > 100:
                        search_term = getattr(self, '_current_search_term', '')
                        title = result.get('title', 'Untitled')
                        
                        # Apply LLM filtering if enabled
                        if self.enable_llm_filtering:
                            content = self._extract_relevant_content_with_llm(content, search_term, title)
                        
                        # Save both HTML and text content to files
                        saved_html_path, saved_txt_path = self._save_webpage_content(page, target_url, title, content, search_term)
                        
                        print_debug(f"✅ [{index+1}] Successfully fetched {len(content)} characters")
                        browser.close()
                        
                        # Clean content for better LLM processing
                        cleaned_content = self._clean_text_for_saving(content)
                        
                        result_data = {
                            'content': cleaned_content if cleaned_content else content,
                            'content_length': len(cleaned_content if cleaned_content else content),
                            'final_url': page.url if 'page' in locals() else None
                        }
                        if saved_html_path:
                            result_data['saved_html_path'] = saved_html_path
                        if saved_txt_path:
                            result_data['saved_txt_path'] = saved_txt_path
                        
                        return result_data
                    else:
    
                        browser.close()
                        return {'status': 'failed', 'content': "Content too short or unable to extract"}
                else:
                    browser.close()
                    return {'status': 'failed', 'content': "Failed to load page after retries"}
        
        except Exception as e:
            error_msg = str(e)
            # print_current(f"❌ [{index+1}] Fetch failed: {error_msg}")  # Commented out to reduce terminal noise
            return {'status': 'failed', 'content': f"Content fetch error: {error_msg}"}

    def _fetch_webpage_content(self, results: List[Dict], page) -> None:
        """
        Fetch actual webpage content for the search results (fallback method)
        """
        for i, result in enumerate(results):
            start_time = time.time()
            try:
                print_current(f"📖 Getting webpage content for result {i+1}: {result['title'][:50]}...")
                
                target_url = result.get('_internal_url') or result.get('url', '')
                
                # Handle Baidu redirect URLs with extended timeout
                is_baidu_redirect = 'baidu.com/link?url=' in target_url
                if is_baidu_redirect:
                    
                    # Try to decode the redirect URL first
                    decoded_url = self._decode_baidu_redirect_url(target_url)
                    if decoded_url != target_url:
                        target_url = decoded_url
                        # print_current(f"🎯 Using decoded URL instead of redirect (fallback)")
                        is_baidu_redirect = False  # No longer need special handling
                
                problematic_domains = [
                    'douyin.com', 'tiktok.com',
                    'youtube.com', 'youtu.be',
                    'bilibili.com', 'b23.tv',
                    'instagram.com', 'facebook.com',
                    'twitter.com', 'x.com',
                    'linkedin.com'
                ]
                
                if any(domain in target_url.lower() for domain in problematic_domains):
                    # print_current(f"⏭️ Skip video/social media link: {target_url}")
                    result['content'] = "Video or social media link, skip content fetch"
                    continue
                
                if target_url.startswith('javascript:') or target_url.startswith('mailto:'):
                    # print_current(f"⏭️ Skip non-webpage link: {target_url}")
                    result['content'] = "Non-webpage link, skip content fetch"
                    continue
                
                if time.time() - start_time > 3:
                    # print_current("⏰ Processing time exceeded 3 seconds, skip this result")
                    result['content'] = "Processing timeout"
                    continue
                
                try:
                    # Use optimized timeout for faster processing
                    timeout_ms = 10000
                    
                    # Reduce retry attempts to 1 (no retry) for faster processing
                    max_retries = 1
                    success = False
                    
                    for attempt in range(max_retries):
                        try:
                            if attempt > 0:
                                # print_current(f"🔄 Retry attempt {attempt + 1} (fallback) for: {result['title'][:30]}...")
                                page.wait_for_timeout(500)
                            
                            page.goto(target_url, timeout=timeout_ms, wait_until='domcontentloaded')
                            page.wait_for_timeout(500)
                            
                            # Check for error pages
                            current_url = page.url
                            if 'chrome-error://' in current_url or 'about:blank' in current_url:
                                raise Exception(f"Redirected to error page: {current_url}")
                            
                            success = True
                            break
                            
                        except Exception as nav_error:
                            if attempt < max_retries - 1:
                                continue
                            else:
                                raise nav_error
                    
                    if success:
                        content = self._extract_main_content(page)
                        
                        if content and len(content.strip()) > 100:
                            search_term = getattr(self, '_current_search_term', '')
                            title = result.get('title', 'Untitled')
                            
                            # Apply LLM filtering if enabled
                            if self.enable_llm_filtering:
                                content = self._extract_relevant_content_with_llm(content, search_term, title)
                            
                            # Save both HTML and text content to files
                            saved_html_path, saved_txt_path = self._save_webpage_content(page, target_url, title, content, search_term)
                            if saved_html_path:
                                result['saved_html_path'] = saved_html_path
                            if saved_txt_path:
                                result['saved_txt_path'] = saved_txt_path
                            
                            # Clean content for better LLM processing
                            cleaned_content = self._clean_text_for_saving(content)
                            result['content'] = cleaned_content if cleaned_content else content
                            elapsed_time = time.time() - start_time
                            # print_current(f"✅ Successfully got {len(result['content'])} characters of useful content (time: {elapsed_time:.2f}s)")
                        else:
                            result['content'] = ""
                            # print_current(f"⚠️ Webpage content too short or unable to get, skip this result")
                
                except Exception as extract_error:
                    error_msg = str(extract_error)
                    if "ERR_HTTP2_PROTOCOL_ERROR" in error_msg:
                        error_msg = "HTTP2 protocol error"
                    elif "interrupted by another navigation" in error_msg:
                        error_msg = "Navigation interrupted"
                    
                    # print_current(f"⚠️ Content extraction failed: {error_msg}")  # Commented out to reduce terminal noise
                    result['content'] = ""
                
            except Exception as e:
                elapsed_time = time.time() - start_time
                # print_current(f"❌ Failed to get webpage content (time: {elapsed_time:.2f}s): {e}")  # Commented out to reduce terminal noise
                result['content'] = ""
                
                if "timeout" in str(e).lower() or elapsed_time > 2:
                    result['content'] = "Webpage access timeout"

    def _extract_main_content(self, page) -> str:
        """
        Extract main content from a webpage with improved CSS and formatting handling
        """
        content = ""
        
        try:
            content_selectors = [
                '.article_content', '.article-content', '.content-detail', '.text-detail',
                '.news-detail', '.detail-content', '.article-detail', '.story-detail',
                '.article_text', '.news_content', '.post-text', '.entry-text',
                '.story-content', '.article-body', '.post-body', '.entry-content',
                '.news-content', '.article-text', '.story-text', '.content-body',
                '.zhengwen', '.neirong', '.wenzhang', '.content_txt', '.txt_content',
                '.article_txt', '.news_txt', '.detail_txt', '.main_txt',
                'article', 'main', 
                '.content', '.main-content', '.post-content', '#content',
                '.text', '.txt', '.article', '.news', '.detail',
                '.markdown-body',
                '.wiki-content',
                '.documentation',
                '.docs-content',
                '[role="main"]',
                '.container .content', '.container main',
                '#main-content', '#article-content', '#post-content',
                '.page-content', '.single-content', '.primary-content',
                'body'
            ]
            
            for selector in content_selectors:
                try:
                    elements = page.query_selector_all(selector)
                    if elements:
                        for elem in elements:
                            try:
                                text = elem.text_content().strip()
                                if text and len(text) > 100:
                                    # Check for verification page early
                                    if "当前环境异常，完成验证后即可继续访问。" in text:
                                        return "当前环境异常，完成验证后即可继续访问。"
                                    
                                    # Check for DocIn embedded document page
                                    if "豆丁网" in text or "docin.com" in text:
                                        return "正文为嵌入式文档，不可阅读"
                                    
                                    # Check for Baidu Scholar search page
                                    if ("百度学术搜索" in text or "百度学术" in text or 
                                        "相关论文" in text or "获取方式" in text or
                                        "按相关性按相关性按被引量按时间降序" in text):
                                        return "结果无可用数据"
                                    
                                    if self._is_quality_content(text):
                                        content = text
                                        # print_current(f"✅ Successfully extracted content with selector '{selector}'")
                                        break
                            except Exception as elem_error:
                                continue
                        if content:
                            break
                except Exception as selector_error:
                    print_debug(f"⚠️ Content selector error for '{selector}': {selector_error}")
                    continue
            
            if not content:
                try:
                    # print_current("⚠️ Selector method found no content, trying to extract full body text")
                    body_elem = None
                    try:
                        body_elem = page.query_selector('body')
                    except Exception as body_selector_error:
                        print_debug(f"⚠️ Body selector error: {body_selector_error}")
                    
                    body_text = ""
                    if body_elem:
                        try:
                            body_text = body_elem.text_content()
                        except Exception as body_text_error:
                            print_debug(f"⚠️ Body text extraction error: {body_text_error}")
                    
                    # Check for verification page in body text
                    if body_text and "当前环境异常，完成验证后即可继续访问。" in body_text:
                        return "当前环境异常，完成验证后即可继续访问。"
                    
                    # Check for DocIn embedded document page in body text
                    if body_text and ("豆丁网" in body_text or "docin.com" in body_text):
                        return "正文为嵌入式文档，不可阅读"
                    
                    # Check for Baidu Scholar search page in body text
                    if body_text and ("百度学术搜索" in body_text or "百度学术" in body_text or
                                      "相关论文" in body_text or "获取方式" in body_text or
                                      "按相关性按相关性按被引量按时间降序" in body_text):
                        return "结果无可用数据"
                    
                    if body_text and len(body_text) > 300:
                        cleaned_body = self._clean_body_content(body_text)
                        if cleaned_body and len(cleaned_body) > 200:
                            content = cleaned_body
                            # print_current("✅ Successfully extracted using body content")
                except Exception as body_extraction_error:
                    print_debug(f"⚠️ Body extraction error: {body_extraction_error}")
            
            if content:
                # Check for verification page again before post-processing
                if "当前环境异常，完成验证后即可继续访问。" in content:
                    return "当前环境异常，完成验证后即可继续访问。"
                
                # Check for DocIn embedded document page again before post-processing
                if "豆丁网" in content or "docin.com" in content:
                    return "正文为嵌入式文档，不可阅读"
                
                # Check for Baidu Scholar search page again before post-processing
                if ("百度学术搜索" in content or "百度学术" in content or
                    "相关论文" in content or "获取方式" in content or
                    "按相关性按相关性按被引量按时间降序" in content):
                    return "结果无可用数据"
                
                # Post-process extracted content to handle common issues
                content = self._post_process_extracted_content(content)
                
                if len(content) < 100:
    
                    return ""
                
                # print_current(f"📄 Successfully extracted content, total length: {len(content)} characters")
        
        except Exception as e:
            # print_current(f"Error extracting webpage content: {e}")  # Commented out to reduce terminal noise
            pass
        
        return content

    def _post_process_extracted_content(self, content: str) -> str:
        """
        Post-process extracted content to handle common formatting issues
        """
        import re
        
        # Remove CSS rules at the beginning of content
        if content.startswith('.') and '{' in content:
            # Find the end of CSS block(s)
            css_pattern = r'^\s*\.[^}]*\}\s*'
            content = re.sub(css_pattern, '', content)
        
        # Remove inline CSS rules that might appear anywhere
        content = re.sub(r'\.[a-zA-Z][\w\-]*\s*\{[^}]*\}\s*', '', content)
        
        # Add line breaks for better structure
        # Add breaks after common sentence endings
        content = re.sub(r'([。！？])\s*([1-9]\d*\.|\([1-9]\d*\)|（[1-9]\d*）)', r'\1\n\2', content)
        content = re.sub(r'([.!?])\s*([1-9]\d*\.|\([1-9]\d*\))', r'\1\n\2', content)
        
        # Add breaks after numbered items
        content = re.sub(r'([1-9]\d*\.[^1-9\n]{10,}?)(\s+[1-9]\d*\.)', r'\1\n\2', content)
        
        # Add breaks after typical news article patterns
        content = re.sub(r'(【[^】]*】[^【]{10,}?)(\s+【)', r'\1\n\2', content)  # 【标题】content 【下一个标题】
        content = re.sub(r'(（[1-9]\d*）[^（]{10,}?)(\s+（[1-9]\d*）)', r'\1\n\2', content)  # （1）content （2）
        
        # Clean up excessive whitespace but preserve intentional formatting
        content = re.sub(r' {3,}', '  ', content)  # Reduce multiple spaces to max 2
        content = re.sub(r'\n\s*\n\s*\n', '\n\n', content)  # Max 2 consecutive newlines
        
        # Remove obvious navigation/UI text at the beginning - but be more conservative
        lines = content.split('\n')
        cleaned_lines = []
        skip_initial_nav = True
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Skip initial navigation-like content - but be more lenient for news content
            if skip_initial_nav:
                # Check if line looks like main content (has Chinese characters and reasonable length)
                chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', line))
                
                # More lenient criteria for news content
                if (chinese_chars > 3 and len(line) > 10) or \
                   ('月' in line and '日' in line) or \
                   ('新闻' in line or '内容' in line or '报道' in line or '消息' in line) or \
                   ('国际' in line or '外交' in line or '政治' in line or '经济' in line):
                    skip_initial_nav = False
                    cleaned_lines.append(line)
                elif line.startswith(('1.', '2.', '3.', '一、', '二、', '三、')) and chinese_chars > 2:
                    skip_initial_nav = False
                    cleaned_lines.append(line)
                # Skip obvious navigation but be more conservative
                elif len(line) < 30 and any(nav in line for nav in ['首页', '登录', '注册', 'APP下载', '分享到']):
                    continue
                # Include news-like content even if short
                elif chinese_chars > 5 or ('丨' in line) or ('｜' in line) or ('：' in line and chinese_chars > 3):
                    cleaned_lines.append(line)
                else:
                    # If we haven't found main content yet but this looks substantial, include it
                    if chinese_chars > 5 or len(line) > 20:
                        cleaned_lines.append(line)
            else:
                cleaned_lines.append(line)
        
        if cleaned_lines:
            content = '\n'.join(cleaned_lines)
        
        return content.strip()

    def _is_quality_content(self, text: str) -> bool:
        """
        Check if text is high-quality main content
        """
        navigation_keywords = [
            'login', 'register', 'home', 'navigation', 'menu', 'search', 'share',
            'copyright', 'contact us', 'about us', 'privacy policy', 'terms of use',
            'login', 'register', 'home', 'menu', 'search', 'share',
            'copyright', 'contact us', 'about us', 'privacy', 'terms'
        ]
        
        # Check for Chinese content (likely news content)
        import re
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
        if chinese_chars > 10:  # If has substantial Chinese content, likely good
            return True
        
        nav_count = sum(1 for keyword in navigation_keywords if keyword.lower() in text.lower())
        words_count = len(text.split())
        
        # Be more lenient with navigation keyword ratio for news content
        if words_count > 0 and nav_count / words_count > 0.15:  # Increased threshold
            return False
        
        # Be more lenient with sentence structure requirement for news titles
        sentence_endings = text.count('。') + text.count('.') + text.count('!') + text.count('？') + text.count('?')
        news_markers = text.count('丨') + text.count('｜') + text.count('：') + text.count('——')
        
        # If has news markers or is short (likely title), don't require sentence endings
        if words_count > 30 and sentence_endings == 0 and news_markers == 0:
            return False
        
        return True

    def _clean_body_content(self, body_text: str) -> str:
        """
        Clean content extracted from body tag
        """
        import re
        
        # Remove common navigation and footer content
        lines = body_text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            if len(line) < 5:  # More lenient minimum length
                continue
            
            # Skip lines that look like navigation but be more conservative
            if any(keyword in line.lower() for keyword in ['home', 'about', 'contact', 'login', 'register', 'menu']):
                if len(line) < 30:  # Only skip short navigation lines
                    continue
            
            # Skip lines with too many links but be more lenient
            if line.count('http') > 5:  # Increased threshold
                continue
            
            # Keep lines with Chinese content (likely news content)
            chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', line))
            if chinese_chars > 0:
                cleaned_lines.append(line)
            # Keep meaningful English content
            elif len(line) >= 10 and len(line.split()) >= 2:
                cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines[:100])  # Increased limit to preserve more content

    def _clean_extracted_content(self, content: str) -> str:
        """
        Clean and format extracted content
        """
        import re
        
        # Remove excessive whitespace
        content = re.sub(r'\s+', ' ', content)
        
        # Remove common unwanted patterns
        content = re.sub(r'Share\s+Tweet\s+Pin\s+Email', '', content)
        content = re.sub(r'Follow us on.*?(?=\.|$)', '', content)
        content = re.sub(r'Subscribe.*?newsletter', '', content)
        
        # Limit content length to prevent excessive output
        web_truncation_length = get_web_content_truncation_length()
        if len(content) > web_truncation_length:
            content = content[:web_truncation_length] + "... [Content truncated]"
        
        return content.strip()

    def _clean_text_for_saving(self, content: str) -> str:
        """
        Clean text content for saving to txt files, preserving meaningful content
        """
        import re
        
        if not content or not content.strip():
            return ""
        
        # Check for verification page early and return as-is
        if "当前环境异常，完成验证后即可继续访问。" in content:
            print_current("⚠️ Detected verification page in cleaning, returning verification message only")
            return "当前环境异常，完成验证后即可继续访问。"
        
        # Check for DocIn embedded document page early and return as-is
        if "豆丁网" in content or "docin.com" in content:
            print_current("⚠️ Detected DocIn embedded document page in cleaning, returning message only")
            return "正文为嵌入式文档，不可阅读"
        
        # Check for Baidu Scholar search page early and return as-is
        if ("百度学术搜索" in content or "百度学术" in content or
            "相关论文" in content or "获取方式" in content or
            "按相关性按相关性按被引量按时间降序" in content):
            print_current("⚠️ Detected Baidu Scholar search page in cleaning, returning message only")
            return "结果无可用数据"
        
        # Remove HTML tags
        content = re.sub(r'<[^>]+>', '', content)
        
        # Remove CSS blocks
        content = re.sub(r'\{[^{}]*\}', '', content, flags=re.DOTALL)
        
        # Remove JavaScript function blocks
        content = re.sub(r'function\s*\w*\s*\([^)]*\)\s*\{[^}]*\}', '', content, flags=re.DOTALL)
        content = re.sub(r'(var|let|const)\s+\w+\s*=.*?;', '', content)
        content = re.sub(r'\$\([^)]+\)\.[^;]+;?', '', content)
        
        # Remove JSON-format image embedding information (e.g., Baidu Baijiahao format)
        # Remove image objects: {"type":"img","link":"...","imgHeight":...,"imgWidth":...}
        # Handle both single-line and multi-line JSON objects
        content = re.sub(r'\{"type"\s*:\s*"img"[^}]*\}', '', content, flags=re.DOTALL)
        # Remove image objects in JSON arrays: ,{"type":"img",...}
        content = re.sub(r',\s*\{\s*"type"\s*:\s*"img"[^}]*\}', '', content, flags=re.DOTALL)
        # Remove image-related JSON fields (standalone or in objects)
        content = re.sub(r'"imgHeight"\s*:\s*\d+[,\s]*', '', content)
        content = re.sub(r'"imgWidth"\s*:\s*\d+[,\s]*', '', content)
        content = re.sub(r'"gifsrc"\s*:\s*"[^"]*"[,\s]*', '', content)
        content = re.sub(r'"gifsize"\s*:\s*"[^"]*"[,\s]*', '', content)
        content = re.sub(r'"gifbytes"\s*:\s*"[^"]*"[,\s]*', '', content)
        content = re.sub(r'"caption"\s*:\s*"[^"]*"[,\s]*', '', content)
        content = re.sub(r'"text-align"\s*:\s*"[^"]*"[,\s]*', '', content)
        content = re.sub(r'"image-align"\s*:\s*"[^"]*"[,\s]*', '', content)
        content = re.sub(r'"img_combine"\s*:\s*"[^"]*"[,\s]*', '', content)
        # Remove image links in JSON format: "link":"https://..." (but preserve text content links if needed)
        # Only remove if it's clearly an image link (contains image-related patterns)
        content = re.sub(r'"link"\s*:\s*"https?://[^"]*\.(jpg|jpeg|png|gif|webp|bmp|svg)[^"]*"[,\s]*', '', content, flags=re.IGNORECASE)
        content = re.sub(r'"link"\s*:\s*"https?://[^"]*(pics|image|img|photo|pic)[^"]*"[,\s]*', '', content, flags=re.IGNORECASE)
        # Remove data_html fields that contain HTML (often includes image references)
        content = re.sub(r'"data_html"\s*:\s*"[^"]*"[,\s]*', '', content)
        # Remove JSON objects that are primarily image metadata (contain imgHeight/imgWidth but no meaningful text)
        content = re.sub(r'\{[^{}]*"imgHeight"[^{}]*"imgWidth"[^{}]*\}', '', content, flags=re.DOTALL)
        
        # Remove URLs and data strings
        content = re.sub(r'https?://[^\s]+', '', content)
        content = re.sub(r'data:[^;]+;[^,]+,[^\s]+', '', content)
        
        # Remove basic CSS properties
        content = re.sub(r'-webkit-[^:]+:[^;]+;?', '', content, flags=re.IGNORECASE)
        content = re.sub(r'-moz-[^:]+:[^;]+;?', '', content, flags=re.IGNORECASE)
        content = re.sub(r'-ms-[^:]+:[^;]+;?', '', content, flags=re.IGNORECASE)
        content = re.sub(r'\w+\s*:\s*[^;{}\n]+[;}]', '', content)  # Generic property:value patterns
        
        # Process line by line with more lenient filtering for news content
        lines = content.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            
            # Skip empty lines and very short lines
            if not line or len(line) < 2:
                continue
            
            # Skip lines containing JSON-format image embedding information
            if (re.search(r'"type"\s*:\s*"img"', line) or
                re.search(r'"imgHeight"', line) or
                re.search(r'"imgWidth"', line) or
                re.search(r'"gifsrc"', line) or
                re.search(r'"gifsize"', line) or
                re.search(r'"gifbytes"', line) or
                (re.search(r'"link"\s*:\s*"https?://', line) and 
                 re.search(r'\.(jpg|jpeg|png|gif|webp|bmp|svg)', line, re.IGNORECASE)) or
                (re.search(r'"link"\s*:\s*"https?://', line) and 
                 re.search(r'(pics|image|img|photo|pic)', line, re.IGNORECASE))):
                continue
            
            # Skip obvious code lines but be more conservative
            if any([
                line.startswith('function '),
                line.startswith('var '),
                line.startswith('let '),
                line.startswith('const '),
                line.startswith('window.'),
                line.startswith('document.'),
                line.startswith('$.'),
                line.startswith('bds.'),
                line.startswith('ct.'),
                re.match(r'^-webkit-', line),
                re.match(r'^-moz-', line),
                re.match(r'^-ms-', line),
                re.match(r'^[A-Za-z0-9]{20,}$', line),  # Very long technical strings
            ]):
                continue
            
            # Skip CSS-like lines but be more specific
            if re.search(r':\s*[^;]+;', line) and not any(marker in line for marker in ['：', '丨', '｜', '新闻', '报道']):
                continue
            
            # Skip lines that are mostly punctuation but preserve news separators
            non_punct_chars = re.sub(r'[^\w\s\u4e00-\u9fff]', '', line)
            if len(non_punct_chars) < len(line) * 0.3 and len(line) > 15:
                # But keep lines with news-like separators
                if not any(sep in line for sep in ['丨', '｜', '：', '——', '【', '】']):
                    continue
            
            # Keep meaningful content with more lenient criteria
            chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', line))
            word_count = len(line.split())
            
            # Prioritize Chinese content (news titles, content)
            if chinese_chars > 0:
                cleaned_lines.append(line)
            # English content needs structure but be more lenient
            elif word_count >= 2:
                has_sentence_structure = any(punct in line for punct in '.!?。！？')
                has_meaningful_words = len(re.findall(r'\b[a-zA-Z]{2,}\b', line)) >= 1
                
                if has_sentence_structure or has_meaningful_words or len(line) >= 15:
                    cleaned_lines.append(line)
            # Keep longer titles and meaningful short content
            elif len(line) >= 5 and word_count >= 1:
                if not re.match(r'^[a-zA-Z0-9._-]+$', line):
                    cleaned_lines.append(line)
        
        # Join cleaned lines
        cleaned_content = '\n'.join(cleaned_lines)
        
        # Basic cleanup
        cleaned_content = re.sub(r'\n\s*\n\s*\n+', '\n\n', cleaned_content)  # Max 2 consecutive newlines
        cleaned_content = re.sub(r' {3,}', '  ', cleaned_content)  # Max 2 spaces
        cleaned_content = cleaned_content.strip()
        
        return cleaned_content

    def fetch_webpage_content(self, url: str, search_term: str = None, **kwargs) -> Dict[str, Any]:
        """
        Directly fetch content from a specific webpage URL.
        """
        
        # Ignore additional parameters
        if kwargs:
            print_current(f"⚠️  Ignoring additional parameters: {list(kwargs.keys())}")
        
        print_debug(f"Fetching content from: {url}")
        
        # Set timeout for this operation
        old_handler = None
        if not is_windows() and is_main_thread():
            try:
                old_handler = signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(30)  # 30 second timeout
            except ValueError as e:
                print_current(f"⚠️ Cannot set signal handler (not in main thread): {e}")
                old_handler = None
        
        # Check if Playwright is available before proceeding
        if not is_playwright_available():
            print_current("❌ Playwright is not installed or not available")
            print_current("💡 Install with: pip install playwright && playwright install chromium")
            return {
                'status': 'failed',
                'url': url,
                'content': 'Playwright not available. Install with: pip install playwright && playwright install chromium',
                'error': 'playwright_not_installed',
                'timestamp': datetime.datetime.now().isoformat()
            }
        
        try:
            from playwright.sync_api import sync_playwright
            
            with sync_playwright() as p:
                # Ensure DISPLAY is unset to prevent X11 usage
                import os
                original_display = os.environ.get('DISPLAY')
                if 'DISPLAY' in os.environ:
                    del os.environ['DISPLAY']
                
                try:
                    browser = p.chromium.launch(
                        headless=True,
                        args=[
                            '--no-sandbox',
                            '--disable-setuid-sandbox',
                            '--disable-dev-shm-usage',
                            '--disable-web-security',
                            '--disable-features=VizDisplayCompositor',
                            '--disable-gpu',
                            '--disable-gpu-sandbox',
                            '--disable-software-rasterizer',
                            '--disable-background-timer-throttling',
                            '--disable-renderer-backgrounding',
                            '--disable-extensions',
                            '--disable-default-apps',
                            '--disable-sync',
                            '--no-first-run',
                            '--no-default-browser-check',
                            '--no-pings',
                            '--disable-remote-debugging',
                            '--disable-http2',
                            '--disable-quic',
                            '--ignore-ssl-errors',
                            '--ignore-certificate-errors',
                            '--disable-background-mode',
                            '--disable-features=TranslateUI',
                            '--force-color-profile=srgb',
                            '--disable-ipc-flooding-protection'
                        ]
                    )
                    
                    context = browser.new_context(
                        user_agent='Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                        viewport={'width': 1024, 'height': 768},
                        ignore_https_errors=True,
                        java_script_enabled=True,
                        bypass_csp=True
                    )
                finally:
                    # Restore original DISPLAY if it existed
                    if original_display is not None:
                        os.environ['DISPLAY'] = original_display
                page = context.new_page()
                
                # Use optimized timeout for faster processing
                final_timeout = 10000
                
                # Check if this is a Baidu redirect URL
                is_baidu_redirect = 'baidu.com/link?url=' in url
                
                if is_baidu_redirect:
                    print_current(f"🔄 Detected Baidu redirect URL, using extended timeout")
                    # Try to decode first
                    decoded_url = self._decode_baidu_redirect_url(url)
                    if decoded_url != url:
                        url = decoded_url
                        is_baidu_redirect = False  # No longer need special handling
                        print_current(f"🎯 Using decoded URL: {url[:100]}...")
                
                page.goto(url, timeout=final_timeout, wait_until='domcontentloaded')
                
                # Optimized wait time for faster processing
                wait_time = 1000
                page.wait_for_timeout(wait_time)
                
                # Skip additional wait for faster processing
                
                try:
                    title = page.title() or "Untitled page"
                except:
                    title = "Untitled page"
                
                content = self._extract_main_content(page)
                
                # Apply LLM filtering if enabled and search term provided
                if content and self.enable_llm_filtering and search_term:
                    content = self._extract_relevant_content_with_llm(content, search_term, title)
                
                # Save both HTML and text content to files
                saved_html_path = ""
                saved_txt_path = ""
                if content and len(content.strip()) > 100:
                    saved_html_path, saved_txt_path = self._save_webpage_content(page, url, title, content, search_term or "")
                
                final_url = page.url
                
                browser.close()
                
                # Clean content for better LLM processing
                cleaned_content = self._clean_text_for_saving(content)
                
                # Check total txt files in web_search_result directory
                total_txt_files = self._count_txt_files_in_result_dir()
                
                result_data = {
                    'title': title,
                    'content': cleaned_content if cleaned_content else content,
                    'content_length': len(cleaned_content if cleaned_content else content),
                    'timestamp': datetime.datetime.now().isoformat(),
                    'status': 'success',
                    'total_txt_files_in_directory': total_txt_files
                }
                
                # Add warning if there are too many txt files
                if total_txt_files > 10:
                    result_data['search_material_warning'] = f"⚠️ Enough materials have been collected ({total_txt_files} text files). Please do not call the search again in the next round."
                
                if saved_html_path or saved_txt_path:
                    if saved_html_path:
                        result_data['saved_html_path'] = saved_html_path
                    if saved_txt_path:
                        result_data['saved_txt_path'] = saved_txt_path
                    
                    result_data['file_notice'] = f"📁 Webpage content saved to folder: {self.web_result_dir}/\n💡 You can use workspace_search or grep_search tools to search within these files"
                    print_current(f"\n📁 Webpage content saved to folder: {self.web_result_dir}/")
                    print_current(f"💡 You can use workspace_search or grep_search tools to search within these files")
                
                return result_data
                
        except ImportError:
            return {
                'error': 'Playwright not installed. Run: pip install playwright && playwright install chromium',
                'status': 'error'
            }
        
        except TimeoutError:
            return {
                'error': 'Operation timed out after 30 seconds',
                'status': 'timeout'
            }
        
        except Exception as e:
            return {
                'error': str(e),
                'status': 'error'
            }
        
        finally:
            # Reset the alarm and restore the original signal handler
            if not is_windows() and is_main_thread() and old_handler is not None:
                try:
                    signal.alarm(0)
                    signal.signal(signal.SIGALRM, old_handler)
                except ValueError:
                    # Already not in main thread, nothing to clean up
                    pass

    def _decode_baidu_redirect_url(self, baidu_url: str) -> str:
        """
        Try to decode Baidu redirect URL to get the real destination URL
        """
        try:
            # Baidu URL format: http://www.baidu.com/link?url=encoded_url
            if 'baidu.com/link?url=' in baidu_url:
                # Extract the encoded part
                url_part = baidu_url.split('baidu.com/link?url=')[1]
                # Remove any additional parameters
                url_part = url_part.split('&')[0]
                
                # Try multiple decoding methods
                decoding_methods = [
                    # Basic URL decoding
                    lambda x: urllib.parse.unquote(x),
                    # Double URL decoding (sometimes URLs are double-encoded)
                    lambda x: urllib.parse.unquote(urllib.parse.unquote(x)),
                    # Replace plus signs with spaces then decode
                    lambda x: urllib.parse.unquote(x.replace('+', ' ')),
                    # Try to decode as UTF-8 bytes
                    lambda x: urllib.parse.unquote(x, encoding='utf-8'),
                ]
                
                for i, decode_method in enumerate(decoding_methods):
                    try:
                        decoded = decode_method(url_part)
                        if decoded.startswith(('http://', 'https://')):
                            print_current(f"✅ Successfully decoded Baidu redirect URL using method {i+1}")
                            print_current(f"🎯 Decoded URL: {decoded[:100]}...")
                            return decoded
                    except Exception as decode_error:
                        continue
                
                # Try base64 decoding (sometimes Baidu uses base64)
                try:
                    import base64
                    # Remove URL-safe characters and try base64 decode
                    clean_url = url_part.replace('-', '+').replace('_', '/')
                    # Add padding if needed
                    while len(clean_url) % 4:
                        clean_url += '='
                    decoded_bytes = base64.b64decode(clean_url)
                    decoded = decoded_bytes.decode('utf-8')
                    if decoded.startswith(('http://', 'https://')):
                        print_current(f"✅ Successfully decoded Baidu redirect URL using base64")
                        print_current(f"🎯 Decoded URL: {decoded[:100]}...")
                        return decoded
                except:
                    pass
                
                # If none of the decoding methods worked, the URL might be using Baidu's custom encoding
                # In that case, we can't easily decode it, so we'll keep the redirect URL
                # but still try to access it with extended timeout
                
                    
        except Exception as e:
            print_current(f"⚠️ Failed to decode Baidu redirect URL: {e}")
        
        # Return original URL if decoding fails
        return baidu_url

    def _optimize_search_term(self, search_term: str) -> str:
        """
        Optimize search terms, especially for time-related searches
        """
        import datetime
        import re
        
        current_date = datetime.datetime.now()
        current_year = current_date.year
        
        optimized_term = search_term.lower()
        
        for year_match in re.finditer(r'\b(\d{4})\b', optimized_term):
            year = int(year_match.group(1))
            if year > current_year:
                print_current(f"🔄 Found future year {year}, replacing with current year {current_year}")
                optimized_term = optimized_term.replace(str(year), str(current_year))
        
        today_keywords = ['today', 'latest', 'current', 'recent', 'breaking']
        if any(keyword in optimized_term for keyword in today_keywords):
            date_str = current_date.strftime('%B %d %Y')
            
            if not re.search(r'\b\d{4}\b', optimized_term):
                optimized_term = f"{optimized_term} {date_str}"
        
        if 'news' in optimized_term:
            news_sources = ['reuters', 'ap news', 'bbc', 'cnn', 'npr']
            if not any(source in optimized_term for source in news_sources):
                optimized_term = f"{optimized_term} breaking news headlines"
        
        return optimized_term

    def search_img(self, query: str, **kwargs) -> Dict[str, Any]:
        """
        Get multiple related images through input query, save to local files, return image file list.
        
        Args:
            query: Image search query string
            **kwargs: Other parameters (ignored)
            
        Returns:
            Dictionary containing multiple image information, with images field as JSON list format
        """
        # Define MD5 hashes of images to filter out
        FILTERED_IMAGE_HASHES = [
            "f7581bb6ed68eec740feb1e9931f22d6",  
            "923e31f20669ef6cc6b86c48cdcad1f0",  
            "901093ca6d9ffbb484f2e92abbf83fba",
            "5b2e0a4206c7b08e609d5d705d22b16e",  # Linear_Attention_Models_applic_20250915_114527_01.webp
            "7b4d6f66b4a09740307aef24d246554a"   # Linear_Attention_Models_applic_20250915_114527_02.webp
        ]
        # Ignore extra parameters
        if kwargs:
            print_current(f"⚠️ Ignoring extra parameters: {list(kwargs.keys())}")
        
        print_current(f"🔍 Image search: {query}")
        
        # Set timeout handling
        old_handler = None
        if not is_windows() and is_main_thread():
            try:
                old_handler = signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(30)  # 30 second timeout
            except ValueError as e:
                print_current(f"⚠️ Cannot set signal handler (not in main thread): {e}")
                old_handler = None
        
        # Check if Playwright is available before proceeding
        if not is_playwright_available():
            print_current("❌ Playwright is not installed or not available")
            print_current("💡 Install with: pip install playwright && playwright install chromium")
            return {
                'status': 'failed',
                'query': query,
                'error': 'Playwright not installed',
                'suggestion': 'Install with: pip install playwright && playwright install chromium',
                'timestamp': datetime.datetime.now().isoformat()
            }
        
        browser = None
        try:
            # 导入必要的库
            try:
                from playwright.sync_api import sync_playwright
                import urllib.parse, re, os, io
                from PIL import Image
            except ImportError as e:
                return {
                    'status': 'failed',
                    'query': query,
                    'error': f'缺少必要的库: {e}',
                    'suggestion': '请安装: pip install playwright pillow',
                    'timestamp': datetime.datetime.now().isoformat()
                }
            except Exception as e:
                return {
                    'status': 'failed',
                    'query': query,
                    'error': f'导入库时出错: {e}',
                    'timestamp': datetime.datetime.now().isoformat()
                }
            
            # 确保图片保存目录存在
            self._ensure_result_directory()
            if not self.web_result_dir:
                return {
                    'status': 'failed',
                    'query': query,
                    'error': '无法创建图片保存目录',
                    'timestamp': datetime.datetime.now().isoformat()
                }
            
            # 创建images子目录
            images_dir = os.path.join(self.web_result_dir, "images")
            try:
                if not os.path.exists(images_dir):
                    os.makedirs(images_dir)
            except Exception as e:
                return {
                    'status': 'failed',
                    'query': query,
                    'error': f'无法创建图片目录: {e}',
                    'timestamp': datetime.datetime.now().isoformat()
                }
            
            with sync_playwright() as p:
                # 确保DISPLAY未设置以防止X11使用
                original_display = os.environ.get('DISPLAY')
                if 'DISPLAY' in os.environ:
                    del os.environ['DISPLAY']
                
                try:
                    browser = p.chromium.launch(
                        headless=True,
                        args=[
                            '--no-sandbox',
                            '--disable-setuid-sandbox',
                            '--disable-dev-shm-usage',
                            '--disable-web-security',
                            '--disable-features=VizDisplayCompositor,TranslateUI',
                            '--disable-gpu',
                            '--disable-gpu-sandbox',
                            '--disable-software-rasterizer',
                            '--disable-background-timer-throttling',
                            '--disable-renderer-backgrounding',
                            '--disable-extensions',
                            '--disable-default-apps',
                            '--disable-sync',
                            '--no-first-run',
                            '--no-default-browser-check',
                            '--no-pings',
                            '--disable-remote-debugging',
                            '--ignore-ssl-errors',
                            '--ignore-certificate-errors',
                            '--disable-background-mode',
                            '--force-color-profile=srgb',
                            '--disable-ipc-flooding-protection'
                        ]
                    )
                    
                    context = browser.new_context(
                        user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36',
                        viewport={'width': 1366, 'height': 768},
                        ignore_https_errors=True,
                        java_script_enabled=True,
                        bypass_csp=True
                    )
                finally:
                    # Restore original DISPLAY
                    if original_display is not None:
                        os.environ['DISPLAY'] = original_display
                
                page = context.new_page()
                page.set_default_timeout(10000)  # 10 second timeout
                
                # Build image search URL
                encoded_query = urllib.parse.quote_plus(query)
                
                # Build search engine list, priority order: Google -> Baidu -> Bing
                search_engines = []
                
                # Check Google connectivity to decide whether to include Google
                if self._check_google_connectivity():
                    print_debug("✅ Google available, using Google -> Baidu -> Bing search order")
                    search_engines = [
                        {
                            'name': 'Google Images',
                            'url': f'https://images.google.com/search?q={encoded_query}&tbm=isch&safe=off&tbs=isz:l&imgsz=large',
                            'image_selector': 'img[data-iurl], img[data-ou], img[data-src], img[src], img',
                            'container_selector': '.rg_bx, .isv-r, .ivg-i',
                            'supports_original': True,  # 支持获取原图
                            'click_selector': '.rg_bx, .isv-r, .ivg-i',  # 点击选择器
                            'original_image_selector': 'img[data-ou], img[data-iurl], img[src]',  # 原图选择器
                            'back_button_selector': 'button[aria-label="Close"], .close-button, .back-button'  # 返回按钮
                        },
                        {
                            'name': 'Baidu Images',
                            'url': f'https://image.baidu.com/search/index?tn=baiduimage&ps=1&ct=201326592&lm=-1&cl=2&nc=1&ie=utf-8&z=3&word={encoded_query}',
                            'image_selector': 'img',
                            'container_selector': '.imgitem, .card-wrap'
                        },
                        {
                            'name': 'Bing Images', 
                            'url': f'https://www.bing.com/images/search?q={encoded_query}&form=HDRSC2&qft=+filterui:imagesize-wallpaper+filterui:aspect-wide',
                            'image_selector': 'img.mimg, img[data-src], img[src], .iusc img, .richImgLnk img, img',
                            'container_selector': '.imgpt, .iusc'
                        }
                    ]
                else:
                    print_debug("⚠️ Google unavailable, using Baidu -> Bing search order")
                    search_engines = [
                        {
                            'name': 'Baidu Images',
                            'url': f'https://image.baidu.com/search/index?tn=baiduimage&ps=1&ct=201326592&lm=-1&cl=2&nc=1&ie=utf-8&z=3&word={encoded_query}',
                            'image_selector': 'img',
                            'container_selector': '.imgitem, .card-wrap'
                        },
                        {
                            'name': 'Bing Images', 
                            'url': f'https://www.bing.com/images/search?q={encoded_query}&form=HDRSC2&qft=+filterui:imagesize-wallpaper+filterui:aspect-wide',
                            'image_selector': 'img.mimg, img[data-src], img[src], .iusc img, .richImgLnk img, img',
                            'container_selector': '.imgpt, .iusc'
                        }
                    ]
                
                image_found = False
                result_data = {
                    'status': 'success',
                    'query': query,
                    'timestamp': datetime.datetime.now().isoformat(),
                    'search_engine_used': None,
                    'image_found': False
                }
                
                for engine in search_engines:
                    try:
                        print_debug(f"🔍 Attempting to use {engine['name']} for image search...")
                        
                        # Visit search page with improved waiting strategy
                        try:
                            page.goto(engine['url'], timeout=6000, wait_until='domcontentloaded')
                            # Wait for page to stabilize
                            page.wait_for_timeout(1000)
                            # Try to wait for images to load
                            try:
                                page.wait_for_selector('img', timeout=2000)
                            except:
                                pass  # Continue even if no images found
                        except Exception as page_error:
                            print_debug(f"⚠️ Page loading error for {engine['name']}: {page_error}")
                            continue
                        
                        # 根据搜索引擎类型使用不同的图片提取方法
                        if engine['name'] == 'Google Images':
                            # Google Images：使用改进的JSON元数据解析方法
                            valid_images = self._extract_google_images_metadata(page)
                            processed_count = len(valid_images)
                            skipped_reasons = {}
                            print_debug(f"🔍 Google Images extracted {len(valid_images)} images from JSON metadata")
                        else:
                            # 其他搜索引擎：使用原有的元素查找方法
                            valid_images, processed_count, skipped_reasons = self._extract_other_engines_images(page, engine)
                        
                        # 输出统计信息
                        self._print_extraction_stats(engine, valid_images, processed_count, skipped_reasons)
                        
                        print_debug(f"✅ {engine['name']} found {len(valid_images)} valid images")
                        
                        # 显示有效图片的详细信息
                        if valid_images:
                            print_debug(f"📋 Valid images details:")
                            for i, img_info in enumerate(valid_images[:5]):  # 只显示前5个
                                print_debug(f"   Image {i+1}: {img_info['src'][:60]}...")
                                if img_info.get('original_src') and img_info['original_src'] != img_info['src']:
                                    print_debug(f"     Original: {img_info['original_src'][:60]}...")
                        
                        if valid_images:
                            # Save multiple valid images (max 20)
                            max_images = min(20, len(valid_images))
                            saved_images = []
                            saved_count = 0  # 添加实际保存的图片计数器

                            # Generate unified timestamp for all images in this batch
                            batch_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

                            print_current(f"📥 Downloading {max_images} images...")

                            for i, selected_image in enumerate(valid_images[:max_images]):
                                # 优先使用原图URL，如果没有则使用缩略图URL
                                image_url = selected_image.get('original_src', selected_image['src'])
                                thumbnail_url = selected_image['src']
                                
                                
                                import time
                                image_start_time = time.time()
                                max_image_time = 3.0  # 增加图片处理时间到3秒 
                                
                                # Download and save image
                                try:
                                    # Get image data
                                    image_data = None
                                    
                                    # Special handling for data:image format base64 images
                                    if image_url.startswith('data:image'):
                                        try:
                                            # Parse data:image format: data:image/jpeg;base64,<base64_data>
                                            header, base64_data = image_url.split(',', 1)
                                            image_data = base64.b64decode(base64_data)
                                            print_debug(f"✅ Successfully parsed base64 image data, size: {len(image_data)} bytes")
                                        except Exception as e:
                                            print_debug(f"⚠️ Failed to parse base64 image: {e}")
                                            continue
                                    else:
                                        
                                        import requests
                                        from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
                                        
                                        start_time = time.time()
                                        max_wait_time = 2.0  # 增加超时时间到2秒
                                        
                                        def download_with_requests(url):
                                            headers = {
                                                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36',
                                                'Accept': 'image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8',
                                                'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
                                                'Referer': 'https://images.google.com/' if 'google' in url else 'https://image.baidu.com/'
                                            }
                                            response = requests.get(url, headers=headers, timeout=1.5, stream=True)  # 增加请求超时到1.5秒
                                            if response.status_code == 200:
                                                # 限制下载大小，避免下载超大文件
                                                content = b''
                                                max_size = 10 * 1024 * 1024
                                                for chunk in response.iter_content(chunk_size=8192):
                                                    content += chunk
                                                    if len(content) > max_size:
                                                        raise Exception(f"Image too large: {len(content)} bytes")
                                                return response.status_code, content
                                            else:
                                                return response.status_code, None
                                        
                                        try:
                                            
                                            with ThreadPoolExecutor(max_workers=1) as executor:
                                                future = executor.submit(download_with_requests, image_url)
                                                try:
                                                    status_code, image_data = future.result(timeout=max_wait_time)
                                                    
                                                    if status_code == 200 and image_data:
                                                        print_debug(f"✅ Successfully downloaded HTTP image, size: {len(image_data)} bytes")
                                                    else:
                                                        continue
                                                except FutureTimeoutError:
                                                    future.cancel()  # 取消任务
                                                    continue
                                        except Exception as download_error:
                                            print_debug(f"❌ Download error for image {i+1}: {download_error}")
                                            continue
                                    
                                    # Validate if it's a valid image and get format (unified processing for all image data)
                                    if image_data:
                                        try:
                                            # 验证阶段也检查时间
                                            validation_start = time.time()
                                            remaining_time = max_image_time - (validation_start - image_start_time)
                                            if remaining_time < 0.2:  # 如果剩余时间不足0.2秒，跳过验证
                                                continue
                                                
                                            # Check if this image should be filtered out by computing its MD5 hash
                                            import hashlib
                                            image_md5 = hashlib.md5(image_data).hexdigest()
                                            if image_md5 in FILTERED_IMAGE_HASHES:
                                                skipped_reasons['md5_filtered'] = skipped_reasons.get('md5_filtered', 0) + 1
                                                print_debug(f"🚫 Image {i+1} filtered out (matches excluded image MD5: {image_md5})")
                                                continue
                                                
                                            with io.BytesIO(image_data) as img_buffer:
                                                img = Image.open(img_buffer)
                                                img.verify()  # Verify image format
                                                
                                                # Reopen to get info (cannot use after verify)
                                                img_buffer.seek(0)
                                                img = Image.open(img_buffer)
                                                
                                                # 增加实际保存的图片计数器
                                                saved_count += 1
                                                
                                                # Generate filename (including sequence number)
                                                safe_query = re.sub(r'[^\w\s-]', '', query)[:30]
                                                safe_query = re.sub(r'[-\s]+', '_', safe_query)
                                                
                                                # 统一转换为JPG格式以确保一致性
                                                # 原始格式信息仍保留在返回数据中
                                                original_format = img.format.lower() if img.format else 'unknown'
                                                
                                                # 统一使用jpg扩展名
                                                filename = f"{safe_query}_{batch_timestamp}_{saved_count:02d}.jpg"
                                                filepath = os.path.join(images_dir, filename)
                                                
                                                # 如果原图不是JPG格式，则转换为JPG保存
                                                if original_format not in ['jpg', 'jpeg']:
                                                    # 转换为RGB模式（JPG不支持透明度）
                                                    if img.mode in ('RGBA', 'LA', 'P'):
                                                        # 创建白色背景
                                                        background = Image.new('RGB', img.size, (255, 255, 255))
                                                        if img.mode == 'P':
                                                            img = img.convert('RGBA')
                                                        background.paste(img, mask=img.split()[-1] if img.mode in ('RGBA', 'LA') else None)
                                                        img = background
                                                    elif img.mode != 'RGB':
                                                        img = img.convert('RGB')
                                                    
                                                    # 保存为JPG格式
                                                    img.save(filepath, 'JPEG', quality=95, optimize=True)
                                                    print_debug(f"💾 Converted {original_format.upper()} to JPG and saved")
                                                else:
                                                    # 原本就是JPG，直接保存原始数据
                                                    with open(filepath, 'wb') as f:
                                                        f.write(image_data)
                                                    print_debug(f"💾 Saved original JPG format")
                                                
                                                # Get relative path (relative to workspace_root)
                                                relative_path = os.path.relpath(filepath, self.workspace_root or os.getcwd())
                                                
                                                # Add to saved images list
                                                saved_images.append({
                                                    'original_image_url': image_url,
                                                    'thumbnail_url': thumbnail_url,  # 添加缩略图URL
                                                    'is_original_image': image_url != thumbnail_url,  # 标记是否为原图
                                                    'local_image_path': filepath,
                                                    'relative_image_path': relative_path,
                                                    'original_format': original_format,  # 原始图片格式
                                                    'saved_format': 'jpg',  # 统一保存为JPG格式
                                                    'image_format': 'jpg',  # 向后兼容，统一为jpg
                                                    'image_size_bytes': len(image_data),
                                                    'image_dimensions': f"{img.width}x{img.height}",
                                                    'alt_text': selected_image['alt'],
                                                    'width': img.width,
                                                    'height': img.height,
                                                    'filename': filename,
                                                    'index': saved_count  # 使用saved_count确保索引连续
                                                })
                                                
                                                print_debug(f"✅ Image {saved_count} saved: {relative_path} ({img.width}x{img.height}, {len(image_data)} bytes)")
                                                
                                        except Exception as e:
                                            print_debug(f"⚠️ Image {i+1} validation or save failed: {e}")
                                            continue
                                        
                                except Exception as e:
                                    print_debug(f"⚠️ Error downloading image {i+1}: {e}")
                                    continue
                                
                                # 检查整体图片处理时间
                                total_elapsed = time.time() - image_start_time
                                if total_elapsed > max_image_time:
                                    continue
                            
                            # If images were successfully saved, update results
                            if saved_images:
                                result_data.update({
                                    'search_engine_used': engine['name'],
                                    'image_found': True,
                                    'images': saved_images,
                                    'total_images_saved': len(saved_images),
                                    'total_images_available': len(valid_images)
                                })
                                image_found = True
                                print_current(f"✅ Saved {len(saved_images)} images to web_search_result/images/")
                                break
                        else:
                            print_debug(f"❌ {engine['name']} found no valid images")
                            
                    except Exception as e:
                        print_debug(f"❌ {engine['name']} search failed: {e}")
                        continue
                
                browser.close()
                
                if not image_found:
                    result_data.update({
                        'error': 'No valid images found',
                        'suggestion': 'Please try using more specific search keywords, or check your network connection'
                    })
                    print_current(f"❌ Image search failed: {query}")
                else:
                    print_debug(f"🎉 Image search completed successfully: {query}")
                
                return result_data
                
        except ImportError as import_error:
            return {
                'status': 'failed',
                'query': query,
                'error': f'Playwright not installed: {import_error}',
                'suggestion': 'Install command: pip install playwright && playwright install chromium',
                'timestamp': datetime.datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                'status': 'failed',
                'query': query,
                'error': f'Image search failed: {str(e)}',
                'timestamp': datetime.datetime.now().isoformat()
            }
            
        finally:
            # Reset timeout signal
            if not is_windows() and is_main_thread() and old_handler is not None:
                try:
                    signal.alarm(0)
                    signal.signal(signal.SIGALRM, old_handler)
                except ValueError:
                    pass
            
            # Ensure browser is closed
            if browser:
                try:
                    browser.close()
                except:
                    pass
    
    def _get_google_image_detail_original(self, page, detail_url: str, engine: dict) -> str:
        """
        从Google Images详情页获取原图URL
        
        Args:
            page: Playwright页面对象
            detail_url: 详情页URL
            engine: 搜索引擎配置
            
        Returns:
            原图URL，如果获取失败则返回空字符串
        """
        try:
            # 保存当前页面状态
            original_url = page.url
            
            # 访问详情页
            page.goto(detail_url, timeout=5000, wait_until='domcontentloaded')
            page.wait_for_timeout(1000)
            
            # 尝试多种选择器来获取原图
            original_selectors = [
                'img[data-ou]',  # Google Images原图属性
                'img[data-iurl]',  # Google Images大图属性
                'img[src*="gstatic.com"]',  # Google静态资源
                'img[src*="googleusercontent.com"]',  # Google用户内容
                'img[data-src]',  # 延迟加载的图片
                'img[src]'  # 普通图片
            ]
            
            original_src = ""
            for selector in original_selectors:
                try:
                    img_element = page.query_selector(selector)
                    if img_element:
                        src = img_element.get_attribute('src') or img_element.get_attribute('data-src')
                        if src and (src.startswith('http') or src.startswith('//')):
                            # 处理协议相对URL
                            if src.startswith('//'):
                                src = 'https:' + src
                            
                            # 验证是否为高质量图片URL
                            if any(domain in src.lower() for domain in ['gstatic.com', 'googleusercontent.com', 'google.com']):
                                original_src = src
                                print_debug(f"✅ Found original image from detail page: {src[:80]}...")
                                break
                except Exception as e:
                    print_debug(f"⚠️ Selector {selector} failed: {e}")
                    continue
            
            # 返回原页面
            page.goto(original_url, timeout=5000, wait_until='domcontentloaded')
            
            return original_src
            
        except Exception as e:
            print_debug(f"⚠️ Failed to get original from detail page: {e}")
            # 尝试返回原页面
            try:
                page.goto(original_url, timeout=5000, wait_until='domcontentloaded')
            except:
                pass
            return ""
    
    def _extract_google_images_metadata(self, page) -> list:
        """
        从Google Images页面提取JSON元数据，基于参考代码实现
        
        Args:
            page: Playwright页面对象
            
        Returns:
            图片信息列表
        """
        valid_images = []
        
        try:
            # 获取页面HTML内容
            html_content = page.content()
            
            # 查找所有包含图片元数据的JSON对象
            import re
            import json
            
            print_debug("🔍 Searching for Google Images JSON metadata...")
            
            # 查找 'class="rg_meta notranslate"' 标签内的JSON数据
            # 这是参考代码中使用的方法
            pattern = r'class="rg_meta[^"]*"[^>]*>(\{[^}]*\})'
            matches = re.findall(pattern, html_content)
            
            if not matches:
                # 尝试更宽泛的匹配模式
                pattern = r'"rg_meta[^"]*"[^>]*>(\{[^<]*\})'
                matches = re.findall(pattern, html_content)
            
            if not matches:
                # 尝试另一种模式：查找JavaScript中的图片数据
                pattern = r'\["(https?://[^"]*\.(?:jpg|jpeg|png|gif|webp|bmp))"[^\]]*\]'
                url_matches = re.findall(pattern, html_content, re.IGNORECASE)
                
                if url_matches:
                    print_debug(f"📸 Found {len(url_matches)} image URLs in JavaScript")
                    for i, url in enumerate(url_matches[:20]):  # 限制最多20张
                        valid_images.append({
                            'src': url,
                            'original_src': url,
                            'width': 'unknown',
                            'height': 'unknown',
                            'alt': f'Google Images result {i+1}',
                            'source': 'javascript_pattern'
                        })
                return valid_images
            
            print_debug(f"📸 Found {len(matches)} JSON metadata objects")
            
            for i, match in enumerate(matches[:20]):  # 限制处理最多20个对象
                try:
                    # 清理和解码JSON字符串
                    json_str = match.strip()
                    
                    # 移除转义字符
                    json_str = json_str.replace('\\u003d', '=')
                    json_str = json_str.replace('\\u0026', '&')
                    json_str = json_str.replace('\\"', '"')
                    json_str = json_str.replace('\\/', '/')
                    
                    # 尝试解析JSON
                    try:
                        metadata = json.loads(json_str)
                    except json.JSONDecodeError:
                        # 如果直接解析失败，尝试使用bytes解码（参考代码的方法）
                        try:
                            decoded = bytes(json_str, "utf-8").decode("unicode_escape")
                            metadata = json.loads(decoded)
                        except:
                            print_debug(f"⚠️ Failed to parse JSON for image {i+1}")
                            continue
                    
                    # 提取图片信息（参考原代码的字段映射）
                    image_info = self._format_google_image_object(metadata, i+1)
                    
                    if image_info and image_info.get('original_src'):
                        valid_images.append(image_info)
                        print_debug(f"✅ Extracted image {i+1}: {image_info['original_src'][:80]}...")
                    
                except Exception as e:
                    print_debug(f"⚠️ Error processing JSON object {i+1}: {e}")
                    continue
            
            print_debug(f"🎯 Successfully extracted {len(valid_images)} images from Google Images metadata")
            
        except Exception as e:
            print_debug(f"❌ Error extracting Google Images metadata: {e}")
            
        return valid_images
    
    def _format_google_image_object(self, metadata: dict, index: int) -> dict:
        """
        格式化Google Images的JSON元数据对象
        基于参考代码的format_object方法
        
        Args:
            metadata: 原始JSON元数据
            index: 图片索引
            
        Returns:
            格式化后的图片信息字典
        """
        try:
            # 参考代码中的字段映射：
            # 'ity' -> image_format (图片格式)
            # 'oh' -> image_height (原图高度)
            # 'ow' -> image_width (原图宽度) 
            # 'ou' -> image_link (原图URL) ⭐ 这是最重要的字段
            # 'pt' -> image_description (图片描述)
            # 'rh' -> image_host (图片托管站点)
            # 'ru' -> image_source (源页面URL)
            # 'tu' -> image_thumbnail_url (缩略图URL)
            
            # 获取原图URL（最重要）
            original_url = metadata.get('ou', '')
            thumbnail_url = metadata.get('tu', '')
            
            if not original_url:
                # 如果没有原图URL，尝试其他字段
                original_url = metadata.get('murl', '') or metadata.get('url', '')
            
            if not original_url:
                return None
            
            # 处理协议相对URL
            if original_url.startswith('//'):
                original_url = 'https:' + original_url
            if thumbnail_url.startswith('//'):
                thumbnail_url = 'https:' + thumbnail_url
            
            # 构建图片信息
            image_info = {
                'src': thumbnail_url or original_url,  # 缩略图URL，如果没有则用原图URL
                'original_src': original_url,  # 原图URL ⭐ 关键字段
                'width': metadata.get('ow', 'unknown'),  # 原图宽度
                'height': metadata.get('oh', 'unknown'),  # 原图高度
                'alt': metadata.get('pt', '') or metadata.get('s', '') or f'Google Images result {index}',
                'image_format': metadata.get('ity', ''),
                'image_host': metadata.get('rh', ''),
                'image_source': metadata.get('ru', ''),
                'source': 'google_json_metadata'
            }
            
            # 验证URL有效性
            if not (original_url.startswith('http') or original_url.startswith('//')):
                return None
                
            # 过滤掉明显的非图片URL
            if any(keyword in original_url.lower() for keyword in ['logo', 'favicon', 'icon', 'sprite']):
                return None
            
            print_debug(f"📋 Formatted image {index}: {image_info['width']}x{image_info['height']} - {original_url[:60]}...")
            
            return image_info
            
        except Exception as e:
            print_debug(f"⚠️ Error formatting image object {index}: {e}")
            return None
    
    def _extract_other_engines_images(self, page, engine: dict) -> tuple:
        """
        从其他搜索引擎（非Google Images）提取图片信息
        
        Args:
            page: Playwright页面对象
            engine: 搜索引擎配置
            
        Returns:
            (valid_images, processed_count, skipped_reasons) 的元组
        """
        valid_images = []
        processed_count = 0
        skipped_reasons = {}
        
        try:
            # Find image elements with error handling
            try:
                image_elements = page.query_selector_all(engine['image_selector'])
                print_debug(f"🔍 {engine['name']} found {len(image_elements)} image elements")
            except Exception as selector_error:
                print_debug(f"⚠️ Selector error for {engine['name']}: {selector_error}")
                # Fallback to basic img selector
                try:
                    image_elements = page.query_selector_all('img')
                    print_debug(f"🔍 {engine['name']} fallback found {len(image_elements)} image elements")
                except Exception as fallback_error:
                    print_debug(f"❌ Fallback selector also failed: {fallback_error}")
                    return valid_images, processed_count, skipped_reasons
            
            # Process all images
            for i, img in enumerate(image_elements[:25]):  # Check up to 25 images
                try:
                    # Validate image element
                    if not img or not hasattr(img, 'get_attribute'):
                        skipped_reasons['invalid_element'] = skipped_reasons.get('invalid_element', 0) + 1
                        continue
                    
                    processed_count += 1
                    
                    # Get image URL based on search engine
                    if engine['name'] == 'Baidu Images':
                        src = img.get_attribute('data-imgurl') or img.get_attribute('src')
                    else:
                        src = img.get_attribute('data-src') or img.get_attribute('src')
                    
                    if not src:
                        skipped_reasons['no_src'] = skipped_reasons.get('no_src', 0) + 1
                        continue
                        
                    # Validate URL format
                    if not (src.startswith('http') or src.startswith('//') or src.startswith('data:image')):
                        skipped_reasons['not_http'] = skipped_reasons.get('not_http', 0) + 1
                        continue
                    
                    # Handle protocol-relative URLs
                    if src.startswith('//'):
                        src = 'https:' + src
                        
                    if src.endswith('.svg'):
                        skipped_reasons['svg_format'] = skipped_reasons.get('svg_format', 0) + 1
                        continue
                    
                    # Get image metadata
                    width = img.get_attribute('width') or 'unknown'
                    height = img.get_attribute('height') or 'unknown' 
                    alt = img.get_attribute('alt') or ''
                    
                    # Filter by keywords
                    src_lower = src.lower()
                    alt_lower = alt.lower()
                    
                    skip_keywords = [
                        'logo', 'favicon', 'watermark', 'advertisement', 'banner', 'button',
                        'sprite', 'avatar_default', 'placeholder', 'icon'
                    ]
                    
                    if any(keyword in src_lower or keyword in alt_lower for keyword in skip_keywords):
                        skipped_reasons['keyword_filter'] = skipped_reasons.get('keyword_filter', 0) + 1
                        continue
                    
                    # Size filtering for non-Google engines
                    if width != 'unknown' and height != 'unknown':
                        try:
                            w, h = int(width), int(height)
                            min_size = 150
                            if w < min_size or h < min_size:
                                skipped_reasons['size_too_small'] = skipped_reasons.get('size_too_small', 0) + 1
                                continue
                            # Aspect ratio limits
                            ratio = max(w, h) / min(w, h)
                            if ratio > 6:
                                skipped_reasons['aspect_ratio'] = skipped_reasons.get('aspect_ratio', 0) + 1
                                continue
                        except:
                            pass
                    
                    # Baidu-specific filtering
                    if engine['name'] == 'Baidu Images':
                        if 'baidu.com' in src_lower and ('static' in src_lower or 'logo' in src_lower):
                            skipped_reasons['baidu_static'] = skipped_reasons.get('baidu_static', 0) + 1
                            continue
                    
                    # Add valid image
                    valid_images.append({
                        'src': src,
                        'original_src': src,  # For non-Google engines, original = src
                        'width': width,
                        'height': height,
                        'alt': alt
                    })
                    
                except Exception as e:
                    skipped_reasons['exception'] = skipped_reasons.get('exception', 0) + 1
                    continue
        
        except Exception as e:
            print_debug(f"❌ Error extracting images from {engine['name']}: {e}")
            
        return valid_images, processed_count, skipped_reasons
    
    def _print_extraction_stats(self, engine: dict, valid_images: list, processed_count: int, skipped_reasons: dict):
        """
        打印图片提取统计信息
        
        Args:
            engine: 搜索引擎配置
            valid_images: 有效图片列表
            processed_count: 处理的图片数量
            skipped_reasons: 跳过的原因统计
        """
        try:
            if engine['name'] == 'Google Images':
                print_debug(f"📊 Google Images: extracted {len(valid_images)} images from JSON metadata")
            else:
                print_debug(f"📊 {engine['name']}: checked {processed_count} elements, found {len(valid_images)} valid images")
            
            if skipped_reasons:
                skip_descriptions = {
                    'invalid_element': 'Invalid elements',
                    'no_src': 'No image URL',
                    'not_http': 'Non-HTTP URL',
                    'svg_format': 'SVG format',
                    'keyword_filter': 'Keyword filtered',
                    'size_too_small': 'Size too small',
                    'aspect_ratio': 'Abnormal aspect ratio',
                    'baidu_static': 'Baidu static resources',
                    'exception': 'Processing exception'
                }
                for reason, count in skipped_reasons.items():
                    desc = skip_descriptions.get(reason, reason)
                    print_debug(f"   - {desc}: {count} items")
                    
            # Show valid images details
            if valid_images:
                print_debug(f"📋 Valid images sample (first 3):")
                for i, img_info in enumerate(valid_images[:3]):
                    print_debug(f"   Image {i+1}: {img_info['src'][:60]}...")
                    if img_info.get('original_src') and img_info['original_src'] != img_info['src']:
                        print_debug(f"     Original: {img_info['original_src'][:60]}...")
        except Exception as e:
            print_debug(f"⚠️ Error printing extraction stats: {e}")