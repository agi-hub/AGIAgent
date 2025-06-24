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

# 导入config_loader以获取截断长度配置
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


# Check if the model is a Claude model
def is_claude_model(model: str) -> bool:
    """Check if the model name is a Claude model"""
    return "claude" in model.lower() or "anthropic" in model.lower()


# Dynamically import Anthropic
def get_anthropic_client():
    """Dynamically import and return Anthropic client class"""
    try:
        from anthropic import Anthropic
        return Anthropic
    except ImportError:
        print("❌ Anthropic library not installed, please run: pip install anthropic")
        raise ImportError("Anthropic library not installed")


class WebSearchTools:
    def __init__(self, llm_api_key: str = None, llm_model: str = None, llm_api_base: str = None, enable_llm_filtering: bool = False, enable_summary: bool = True, out_dir: str = None):
        self._google_connectivity_checked = False
        self._google_available = True
        self._last_google_request = 0  # Track last Google request time for rate limiting
        
        # LLM configuration for content filtering and summarization
        self.enable_llm_filtering = enable_llm_filtering
        self.enable_summary = enable_summary
        self.llm_client = None
        self.llm_model = llm_model
        self.is_claude = False
        
        # Initialize web search result directory
        if out_dir:
            self.web_result_dir = os.path.join(out_dir, "workspace", "web_search_result")
        else:
            # Fallback to default if no out_dir provided
            self.web_result_dir = os.path.join("workspace", "web_search_result")
        self._ensure_result_directory()
        
        if (enable_llm_filtering or enable_summary) and llm_api_key and llm_model and llm_api_base:
            try:
                self._setup_llm_client(llm_api_key, llm_model, llm_api_base)
                features = []
                if enable_llm_filtering:
                    features.append("content filtering")
                if enable_summary:
                    features.append("search results summarization")
                print(f"🤖 LLM features enabled with model {llm_model}: {', '.join(features)}")
            except Exception as e:
                print(f"⚠️ Failed to setup LLM client, disabling LLM features: {e}")
                self.enable_llm_filtering = False
                self.enable_summary = False
        elif enable_llm_filtering or enable_summary:
            print("⚠️ LLM features requested but missing configuration, disabling")
            self.enable_llm_filtering = False
            self.enable_summary = False
        else:
            print("📝 LLM features disabled")
    
    def _ensure_result_directory(self):
        """Ensure the web search result directory exists"""
        try:
            if not os.path.exists(self.web_result_dir):
                os.makedirs(self.web_result_dir)
                print(f"📁 Created web search result directory: {self.web_result_dir}")
            else:
                print(f"📁 Using existing web search result directory: {self.web_result_dir}")
        except Exception as e:
            print(f"⚠️ Failed to create result directory: {e}")
            self.web_result_dir = None
    
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
                    print("⚠️ Skipping HTML save for verification page")
                elif "豆丁网" in html_content or "docin.com" in html_content:
                    print("⚠️ Skipping HTML save for DocIn embedded document page")
                elif ("百度学术搜索" in html_content or "xueshu.baidu.com" in html_content or 
                      "百度学术" in html_content or "- 百度学术" in title or
                      "相关论文" in html_content or "获取方式" in html_content):
                    print("⚠️ Skipping HTML save for Baidu Scholar search page")
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
            
            print(f"💾 Saved webpage HTML: {filename} ({len(html_content)} bytes)")
            return filepath
            
        except Exception as e:
            print(f"⚠️ Failed to save webpage HTML: {e}")
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
                        print("⚠️ Skipping HTML save for verification page")
                    elif "豆丁网" in html_content or "docin.com" in html_content:
                        print("⚠️ Skipping HTML save for DocIn embedded document page")
                    elif ("百度学术搜索" in html_content or "xueshu.baidu.com" in html_content or
                          "百度学术" in html_content or "- 百度学术" in title or
                          "相关论文" in html_content or "获取方式" in html_content):
                        print("⚠️ Skipping HTML save for Baidu Scholar search page")
                
                if not should_skip_html:
                    # Ensure the HTML file has .html extension
                    html_filename = f"{base_filename}.html"
                    html_filepath = os.path.join(self.web_result_dir, html_filename)
                    with open(html_filepath, 'w', encoding='utf-8') as f:
                        f.write(html_content)
                    print(f"💾 Saved webpage HTML: {html_filename} ({len(html_content)} bytes)")
                else:
                    print(f"⏭️ Skipped HTML save for special page: {title[:50]}...")
                    
            except Exception as e:
                print(f"⚠️ Failed to save webpage HTML: {e}")
            
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
                        print(f"📝 Saved cleaned text: {txt_filename} ({len(cleaned_content)} characters, cleaned from {len(content)})")
                    else:
                        print(f"⚠️ Content too short or low quality after cleaning for: {title}")
                else:
                    print(f"⚠️ No text content to save for: {title}")
            except Exception as e:
                print(f"⚠️ Failed to save text content: {e}")
            
            return html_filepath, txt_filepath
            
        except Exception as e:
            print(f"⚠️ Failed to save webpage content: {e}")
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
        
        print("🌐 Checking Google connectivity for the first time...")
        try:
            # Try to download Google homepage with 3 second timeout
            response = requests.get('https://www.google.com', 
                                  timeout=3, 
                                  headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'})
            
            if response.status_code == 200 and len(response.text) > 100:
                print("✅ Google connectivity test passed")
                self._google_available = True
            else:
                print(f"❌ Google connectivity test failed: status {response.status_code}")
                self._google_available = False
                
        except requests.exceptions.Timeout:
            print("⚠️ Google connectivity test timeout (>3s)")
            self._google_available = False
        except requests.exceptions.RequestException as e:
            print(f"⚠️ Google connectivity test error: {e}")
            self._google_available = False
        except Exception as e:
            print(f"⚠️ Google connectivity test error: {e}")
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
            print("⚠️ Detected verification page in LLM filtering, skipping LLM processing")
            return "当前环境异常，完成验证后即可继续访问。"
        
        # Check for DocIn embedded document page and skip LLM processing
        if "豆丁网" in content or "docin.com" in content:
            print("⚠️ Detected DocIn embedded document page in LLM filtering, skipping LLM processing")
            return "正文为嵌入式文档，不可阅读"
        
        # Check for Baidu Scholar search page and skip LLM processing
        if ("百度学术搜索" in content or "百度学术" in content or
            "相关论文" in content or "获取方式" in content or
            "按相关性按相关性按被引量按时间降序" in content):
            print("⚠️ Detected Baidu Scholar search page in LLM filtering, skipping LLM processing")
            return "结果无可用数据"
        
        # Skip processing if content is too short
        if len(content.strip()) < 100:
            return content
        
        try:
            print(f"🧠 Using LLM to extract relevant information for: {search_term}")
            
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
                    print("⚠️ Claude API response format unexpected")
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
                    print("⚠️ OpenAI API response format unexpected")
                    return content
            
            # Validate filtered content
            if filtered_content and filtered_content.strip():
                if "No relevant content found" in filtered_content or "no relevant content" in filtered_content.lower():
                    print("🔍 LLM determined no relevant content found")
                    return "No relevant content found in this webpage for the search query."
                elif len(filtered_content.strip()) > 50:  # Ensure we got substantial content
                    print(f"✅ LLM filtering completed: {len(content)} → {len(filtered_content)} characters")
                    return filtered_content.strip()
            
            print("⚠️ LLM filtering produced insufficient content, using original")
            return content
            
        except Exception as e:
            print(f"❌ LLM content filtering failed: {e}")
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
            print("⚠️ No valid content found for summarization")
            return ""
        
        print(f"📝 Using LLM to summarize {len(valid_results)} search results for: {search_term}")
        
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
"📁 **Original Content Storage**: Complete HTML source files and plain text versions of all webpages have been automatically saved to the `web_search_result` folder. For more detailed original content or in-depth analysis, you can use the `read_file` tool to directly access these files, or use the `codebase_search` tool to search for specific information within the folder. File names include search keywords, webpage titles, and timestamps for easy identification and retrieval."

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
                    print("⚠️ Claude API response format unexpected")
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
                    print("⚠️ OpenAI API response format unexpected")
                    return ""
            
            # Validate summary
            if summary and summary.strip() and len(summary.strip()) > 100:
                print(f"✅ Search results detailed analysis completed: {len(summary)} characters")
                return summary.strip()
            else:
                print("⚠️ LLM produced insufficient summary")
                return ""
                
        except Exception as e:
            print(f"❌ Search results summarization failed: {e}")
            return ""

    def web_search(self, search_term: str, fetch_content: bool = True, max_content_results: int = 30, **kwargs) -> Dict[str, Any]:
        """
        Search the web for real-time information using Playwright.
        """
        # Store current search term for LLM filtering
        self._current_search_term = search_term
        
        
        # Ignore additional parameters
        if kwargs:
            print(f"⚠️  Ignoring additional parameters: {list(kwargs.keys())}")
        
        print(f"🔍 Search keywords: {search_term}")
        if fetch_content:
            print(f"📄 Will automatically fetch webpage content for the first {max_content_results} results")
        else:
            print(f"📝 Only get search result summaries, not webpage content")
        
        # Set global timeout of 30 seconds for the entire search operation
        old_handler = None
        if not is_windows():
            old_handler = signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(30)
        
        browser = None
        try:
            # Try to import Playwright with better error handling
            try:
                from playwright.sync_api import sync_playwright
                import urllib.parse
            except ImportError as e:
                raise ImportError(f"Failed to import Playwright: {e}")
            except Exception as e:
                # Handle any other import-related errors (like GLIBC issues during import)
                raise Exception(f"Playwright import/initialization error: {e}")
            
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
                    print(f"🔗 Direct URL detected, attempting to access: {search_term}")
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
                            print("⚠️ Detected DocIn embedded document page by URL/title")
                            content = "正文为嵌入式文档，不可阅读"
                        elif ("百度学术搜索" in title or "xueshu.baidu.com" in current_url or 
                              "百度学术" in title or "- 百度学术" in title or
                              "search.baidu.com" in current_url):
                            print("⚠️ Detected Baidu Scholar search page by URL/title")
                            content = "结果无可用数据"
                        else:
                            if fetch_content:
                                content = self._extract_main_content(page)
                            print(f"📄 Extracted webpage content length: {len(content)} characters")
                            
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
                            'snippet': f'Direct URL access successful: {title}',
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
                        
                        print(f"✅ Successfully accessed URL directly, got {len(content)} characters of content")
                        
                    except Exception as e:
                        print(f"❌ Direct URL access failed: {e}")
                        results.append({
                            'title': f'Access failed: {search_term}',
                            'url': search_term,
                            'snippet': f'Direct URL access failed: {str(e)}',
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
                        print("🔍 Google search engine added as primary option (connectivity confirmed)")
                        
                        # Add backup Google search with different approach
                        search_engines.append({
                            'name': 'Google_Backup',
                            'url': 'https://www.google.com/search?q={}&num=20&hl=en&lr=lang_en&cr=countryUS&safe=off&tbs=lr:lang_1en',
                            'result_selector': 'h3, .LC20lb, .DKV0Md, [data-ved] h3, .yuRUbf h3',
                            'container_selector': '.g, .tF2Cxc, [data-ved], .yuRUbf',
                            'snippet_selectors': ['.VwiC3b', '.s', '.st', 'span', '.IsZvec', '.aCOpRe', '.yXK7lf', '[data-sncf]'],
                            'anti_bot_indicators': ['Our systems have detected unusual traffic', 'g-recaptcha', 'captcha', 'verify you are human', 'blocked', 'unusual activity']
                        })
                        print("🔍 Google backup search engine added")
                    else:
                        print("⚠️ Google connectivity test failed, will use alternative search engines")
                    
                    # Always add Baidu as fallback or secondary option
                    search_engines.append({
                        'name': 'Baidu',
                        'url': 'https://www.baidu.com/s?wd={}',
                        'result_selector': 'h3.t a',
                        'container_selector': '.result',
                        'snippet_selectors': ['.c-abstract', '.c-span9', 'span', 'div']
                    })
                    print("🔍 Baidu search engine added to available options")
                    
                    # Add DuckDuckGo as additional fallback
                    search_engines.append({
                        'name': 'DuckDuckGo',
                        'url': 'https://duckduckgo.com/?q={}',
                        'result_selector': '[data-testid="result-title-a"], .result__title a, h2.result__title a',
                        'container_selector': '[data-testid="result"], .result, .web-result',
                        'snippet_selectors': ['.result__snippet', '[data-testid="result-snippet"]', '.result-snippet']
                    })
                    print("🔍 DuckDuckGo search engine added as additional fallback")
                    
                    if not search_engines:
                        print("❌ No search engines available")
                        return {
                            'search_term': search_term,
                            'results': [{
                                'title': 'No search engines available',
                                'url': '',
                                'snippet': 'All search engines are unavailable due to connectivity issues.',
                                'content': 'No search engines available'
                            }],
                            'timestamp': datetime.datetime.now().isoformat(),
                            'error': 'no_search_engines_available'
                        }
                    
                    optimized_search_term = self._optimize_search_term(search_term)
                    encoded_term = urllib.parse.quote_plus(optimized_search_term)
                    
                    for engine in search_engines:
                        try:
                            print(f"🔍 Trying to search with {engine['name']}...")
                            
                            # Add rate limiting for Google to avoid being blocked
                            if engine['name'].startswith('Google'):
                                current_time = time.time()
                                time_since_last_request = current_time - self._last_google_request
                                if time_since_last_request < 3:  # Wait at least 3 seconds between Google requests
                                    wait_time = 3 - time_since_last_request
                                    print(f"⏱️ Rate limiting: waiting {wait_time:.1f} seconds before Google request")
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
                                        print(f"⚠️ {engine['name']} detected anti-bot mechanism: {indicator}")
                                        print(f"🔄 Skipping {engine['name']} due to anti-bot protection")
                                        raise Exception(f"Anti-bot protection detected: {indicator}")
                            
                            result_elements = page.query_selector_all(engine['result_selector'])
                            
                            if result_elements:
                                print(f"✅ {engine['name']} search successful, found {len(result_elements)} results")
                                
                                for i, elem in enumerate(result_elements[:30]):  # Top 30 results
                                    try:
                                        title = ""
                                        url = ""
                                        snippet = ""
                                        
                                        # Extract title and URL based on engine type
                                        if engine['name'].startswith('Google'):
                                            # For Google results, handle different element types
                                            try:
                                                tag_name = elem.evaluate('element => element.tagName.toLowerCase()')
                                            except:
                                                tag_name = 'unknown'
                                            
                                            if tag_name == 'h3':
                                                title = elem.text_content().strip()
                                                # Find parent link element
                                                parent_link = elem.query_selector('xpath=ancestor::a[1]')
                                                if parent_link:
                                                    url = parent_link.get_attribute('href') or ""
                                                else:
                                                    # Try to find sibling or nearby link
                                                    nearby_link = elem.query_selector('xpath=../a | xpath=../../a | xpath=../../../a')
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
                                            results.append({
                                                'title': title,
                                                'snippet': snippet[:get_truncation_length()] if snippet else f'Search result from {engine["name"]}',
                                                'source': engine['name'],
                                                'content': '',
                                                '_internal_url': url  # Keep URL internally for content fetching
                                            })
                                    
                                    except Exception as e:
                                        print(f"Error extracting result {i}: {e}")
                                        continue
                                
                                if results:
                                    break
                            else:
                                print(f"❌ {engine['name']} found no search results")
                        
                        except Exception as e:
                            print(f"❌ {engine['name']} search failed: {e}")
                            continue
                    
                    if fetch_content and results:
                        print(f"\n🚀 Starting to fetch webpage content for first {min(max_content_results, len(results))} results using parallel processing...")
                        
                        # Use parallel processing for better efficiency
                        try:
                            # Close the shared page since parallel processing will create its own browsers
                            page = None
                            
                            # Use parallel content fetching with 8 concurrent workers for better performance
                            self._fetch_webpage_content_parallel(results[:max_content_results], max_workers=8)
                            
                        except Exception as e:
                            print(f"⚠️ Parallel content fetching failed: {e}")
                            print(f"🔄 Falling back to sequential content fetching...")
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
                                print(f"⚠️ Fallback sequential content fetching also failed: {fallback_e}")
                                # Final fallback - try basic method
                                try:
                                    self._fetch_webpage_content(results[:max_content_results], page)
                                except Exception as final_e:
                                    print(f"⚠️ All content fetching methods failed: {final_e}")
                        
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
                            print(f"✅ Successfully got {len(results)} search results with valid content")
                        else:
                            print("⚠️ All search results failed to get valid webpage content, returning search results only")
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
                print("🔄 All search engines failed, providing fallback result...")
                results = [{
                    'title': f'Search: {search_term}',
                    'snippet': f'Failed to get results from search engines. Possible reasons: network connection issues, search engine structure changes, or access restrictions. Recommend manual search: {search_term}',
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
                            optimized_result['snippet'] = result['snippet']
                        
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
                    print(f"✅ Optimized search result format, {len([r for r in results if r.get('has_full_content')])} results contain full content")
            
            # Count saved files
            saved_html_count = len([r for r in results if r.get('saved_html_path')])
            saved_txt_count = len([r for r in results if r.get('saved_txt_path')])
            
            # Generate summary if enabled and content was fetched
            summary = ""
            if fetch_content and self.enable_summary and results:
                summary = self._summarize_search_results_with_llm(results, search_term)
            
            result_data = {
                'search_term': search_term,
                'results': results,
                'timestamp': datetime.datetime.now().isoformat(),
                'total_results': len(results),
                'content_fetched': fetch_content,
                'results_with_content': len([r for r in results if r.get('has_full_content')]) if fetch_content else 0,
                'saved_html_files': saved_html_count,
                'saved_txt_files': saved_txt_count
            }
            
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
                
                print(f"📋 Generated comprehensive summary ({len(summary)} characters)")
                print(f"\n🎯 Final Summary for Search Term: '{search_term}'")
                print(f"{'='*60}")
                print(summary)
                print(f"{'='*60}\n")
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
                file_notice_parts.append("💡 You can use codebase_search or grep_search tools to search within these files")
                
                print(f"\n📁 {files_str} saved to folder: {self.web_result_dir}/")
                print(f"💡 You can use codebase_search or grep_search tools to search within these files")
            
            # Always add notice about original content access
            if summary:
                file_notice_parts.append("📄 Complete original webpage content is stored in the saved files above")
                file_notice_parts.append("🔍 Use the saved files to access full details beyond this summary")
            
            if file_notice_parts:
                result_data['files_notice'] = "\n".join(file_notice_parts)
            
            return result_data
            
        except ImportError as import_error:
            print(f"Playwright not installed: {import_error}")
            print("Install with: pip install playwright && playwright install chromium")
            return {
                'search_term': search_term,
                'results': [{
                    'title': 'Playwright not available',
                    'url': '',
                    'snippet': 'Playwright library is not installed. Run: pip install playwright && playwright install chromium',
                    'content': ''
                }],
                'timestamp': datetime.datetime.now().isoformat(),
                'error': 'playwright_not_installed'
            }
        
        except Exception as playwright_error:
            # Handle Playwright browser launch errors (including GLIBC issues)
            error_str = str(playwright_error)
            if 'GLIBC' in error_str or 'version' in error_str or 'not found' in error_str:
                print(f"❌ Playwright system compatibility error: {playwright_error}")
                print("⚠️  Your system GLIBC version is too old for Playwright")
                print("💡 Suggestion: Try using requests-based fallback or upgrade your system")
                return {
                    'search_term': search_term,
                    'results': [{
                        'title': 'System compatibility issue',
                        'url': '',
                        'snippet': f'Playwright requires GLIBC 2.28+ but your system has an older version. Error: {error_str}',
                        'content': 'Please upgrade your system or use alternative search method'
                    }],
                    'timestamp': datetime.datetime.now().isoformat(),
                    'error': 'glibc_compatibility_error'
                }
            elif 'PlaywrightContextManager' in error_str or '_playwright' in error_str:
                print(f"❌ Playwright initialization error: {playwright_error}")
                print("💡 This might be due to browser installation issues")
                return {
                    'search_term': search_term,
                    'results': [{
                        'title': 'Playwright initialization failed',
                        'url': '',
                        'snippet': f'Playwright failed to initialize properly. Error: {error_str}',
                        'content': 'Browser initialization failed, please check Playwright installation'
                    }],
                    'timestamp': datetime.datetime.now().isoformat(),
                    'error': 'playwright_init_error'
                }
            else:
                print(f"❌ Playwright error: {playwright_error}")
                return {
                    'search_term': search_term,
                    'results': [{
                        'title': f'Playwright error: {search_term}',
                        'url': '',
                        'snippet': f'Playwright failed with error: {str(playwright_error)}',
                        'content': f'Playwright error: {str(playwright_error)}'
                    }],
                    'timestamp': datetime.datetime.now().isoformat(),
                    'error': str(playwright_error)
                }
        
        except TimeoutError:
            # print("❌ Web search timed out after 60 seconds")  # Commented out to reduce terminal noise
            return {
                'search_term': search_term,
                'results': [{
                    'title': f'Search timeout: {search_term}',
                    'url': '',
                    'snippet': 'Web search operation timed out after 60 seconds.',
                    'content': 'Search operation timed out'
                }],
                'timestamp': datetime.datetime.now().isoformat(),
                'error': 'search_timeout'
            }
        
        except Exception as e:
            print(f"❌ Web search failed: {e}")
            return {
                'search_term': search_term,
                'results': [{
                    'title': f'Search error: {search_term}',
                    'url': '',
                    'snippet': f'Web search failed with error: {str(e)}',
                    'content': f'Search error: {str(e)}'
                }],
                'timestamp': datetime.datetime.now().isoformat(),
                'error': str(e)
            }
        
        finally:
            # Reset the alarm and restore the original signal handler
            if not is_windows() and old_handler is not None:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)
            
            # Emergency browser cleanup
            if browser:
                try:
                    browser.close()
                except:
                    pass

    def _extract_snippet_from_search_result(self, elem, engine) -> str:
        """
        Extract snippet/description from search result element
        """
        snippet = ""
        
        try:
            container = None
            if engine['container_selector']:
                container = elem.query_selector(f'xpath=ancestor::*[contains(@class, "{engine["container_selector"].replace(".", "")}")]')
            
            if not container:
                container = elem.query_selector('xpath=ancestor::div[2]')
            
            if container:
                for selector in engine['snippet_selectors']:
                    snippet_elem = container.query_selector(selector)
                    if snippet_elem:
                        text = snippet_elem.text_content().strip()
                        if text and len(text) > 20 and not text.startswith('http') and '...' not in text[:10]:
                            snippet = text
                            break
                
                if not snippet:
                    all_text_elems = container.query_selector_all('span, div, p')
                    for text_elem in all_text_elems:
                        text = text_elem.text_content().strip()
                        if text and len(text) > 30 and len(text) < 200:
                            if any(char in text for char in '，。？！、；：') or ' ' in text:
                                snippet = text
                                break
        
        except Exception as e:
            print(f"Error extracting snippet: {e}")
        
        return snippet

    def _fetch_webpage_content_with_timeout(self, results: List[Dict], page, timeout_seconds: int = 25) -> None:
        """
        Fetch webpage content with additional timeout control
        """
        start_time = time.time()
        
        for i, result in enumerate(results):
            if time.time() - start_time > timeout_seconds:
                print(f"⏰ Overall content fetching timeout reached ({timeout_seconds}s), stopping")
                break
                
            try:
                print(f"📖 Getting webpage content for result {i+1}: {result['title'][:50]}...")
                
                target_url = result.get('_internal_url') or result.get('url', '')
                
                # Handle Baidu redirect URLs with extended timeout
                is_baidu_redirect = 'baidu.com/link?url=' in target_url
                if is_baidu_redirect:
                    
                    # Try to decode the redirect URL first
                    decoded_url = self._decode_baidu_redirect_url(target_url)
                    if decoded_url != target_url:
                        target_url = decoded_url
                        print(f"🎯 Using decoded URL instead of redirect")
                        is_baidu_redirect = False  # No longer need special handling
                
                # Quick domain check
                problematic_domains = [
                    'douyin.com', 'tiktok.com', 'youtube.com', 'youtu.be',
                    'bilibili.com', 'b23.tv', 'instagram.com', 'facebook.com',
                    'twitter.com', 'x.com', 'linkedin.com'
                ]
                
                if any(domain in target_url.lower() for domain in problematic_domains):
                    print(f"⏭️ Skip video/social media link: {target_url}")
                    result['content'] = "Video or social media link, skip content fetch"
                    continue
                
                if target_url.startswith(('javascript:', 'mailto:')):
                    print(f"⏭️ Skip non-webpage link: {target_url}")
                    result['content'] = "Non-webpage link, skip content fetch"
                    continue
                
                # Navigate to page with short timeout
                remaining_time = max(5000, int((timeout_seconds - (time.time() - start_time)) * 1000))
                if remaining_time <= 5000:
                    # print("⏰ Insufficient remaining time, skip this result")
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
                                # print(f"🔄 Retry attempt {attempt + 1} for: {result['title'][:30]}...")
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
                                # print(f"⚠️ Navigation attempt {attempt + 1} failed (Baidu redirect timeout): {nav_error}")  # Commented out to reduce terminal noise
                                if attempt < max_retries - 1:
                                    continue
                            elif attempt < max_retries - 1:
                                # print(f"⚠️ Navigation attempt {attempt + 1} failed: {nav_error}")  # Commented out to reduce terminal noise
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
                            print(f"✅ Successfully got {len(result['content'])} characters of useful content")
                            if is_baidu_redirect:
                                print(f"🎯 Final redirected URL: {page.url}")
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
                    
                    # print(f"❌ Failed to get webpage content: {error_msg}")  # Commented out to reduce terminal noise
                    result['content'] = f"Content fetch error: {error_msg}"
                    result['error_type'] = error_type
                    
                    # Special message for Baidu redirects
                    if is_baidu_redirect:
                        pass  # print(f"💡 Baidu redirect troubleshooting: URL={target_url[:100]}...")  # Commented out to reduce terminal noise
                
            except Exception as e:
                # print(f"❌ Error processing result {i+1}: {e}")  # Commented out to reduce terminal noise
                result['content'] = f"Processing error: {str(e)}"

    def _fetch_webpage_content_parallel(self, results: List[Dict], max_workers: int = 8) -> None:
        """
        Fetch webpage content for multiple results in parallel using ThreadPoolExecutor
        
        Args:
            results: List of search results to fetch content for
            max_workers: Maximum number of concurrent workers (default: 4)
        """
        if not results:
            return
        
        print(f"🚀 Starting parallel content fetching for {len(results)} results with {max_workers} workers")
        
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
                # print(f"⏭️ Skip video/social media link: {result['title'][:30]}...")
                result['content'] = "Video or social media link, skip content fetch"
                continue
            
            if target_url.startswith(('javascript:', 'mailto:')):
                # print(f"⏭️ Skip non-webpage link: {result['title'][:30]}...")
                result['content'] = "Non-webpage link, skip content fetch"
                continue
            
            results_to_fetch.append((i, result))
        
        if not results_to_fetch:
            print("⚠️ No valid URLs to fetch content from")
            return
        
        print(f"📊 Processing {len(results_to_fetch)} valid URLs for content fetching")
        
        # Use ThreadPoolExecutor for parallel processing with improved timeout handling
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_result = {}
            for i, result in results_to_fetch:
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
                        print(f"✅ Completed {completed}/{len(results_to_fetch)} content fetching tasks")
                        
                    except Exception as e:
                        i, original_result = future_to_result[future]
                        # print(f"❌ Error fetching content for result {i+1}: {e}")  # Commented out to reduce terminal noise
                        original_result['content'] = f"Parallel fetch error: {str(e)}"
                        completed += 1
                        
            except concurrent.futures.TimeoutError:
                print(f"⏰ Parallel processing timeout after {timeout_per_batch}s, handling remaining tasks...")
                
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
        
        print(f"🎉 Parallel content fetching completed for all {len(results_to_fetch)} results")

    def _fetch_single_webpage_content(self, result: Dict, index: int) -> Dict[str, Any]:
        """
        Fetch content for a single webpage (used by parallel processing)
        
        Args:
            result: Single search result to fetch content for
            index: Index of the result for logging
            
        Returns:
            Updated result dictionary with content
        """
        from playwright.sync_api import sync_playwright
        
        target_url = result.get('_internal_url') or result.get('url', '')
        
        try:
            print(f"🔗 [{index+1}] Fetching content: {result['title'][:40]}...")
            
            # Handle Baidu redirect URLs
            is_baidu_redirect = 'baidu.com/link?url=' in target_url
            if is_baidu_redirect:
                
                decoded_url = self._decode_baidu_redirect_url(target_url)
                if decoded_url != target_url:
                    target_url = decoded_url
                    # print(f"🎯 [{index+1}] Using decoded URL")
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
                            print(f"🔄 [{index+1}] Retry attempt {attempt + 1}")
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
                            print(f"⚠️ [{index+1}] Navigation attempt {attempt + 1} failed: {nav_error}")
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
                        
                        print(f"✅ [{index+1}] Successfully fetched {len(content)} characters")
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
                        return {'content': "Content too short or unable to extract"}
                else:
                    browser.close()
                    return {'content': "Failed to load page after retries"}
        
        except Exception as e:
            error_msg = str(e)
            # print(f"❌ [{index+1}] Fetch failed: {error_msg}")  # Commented out to reduce terminal noise
            return {'content': f"Content fetch error: {error_msg}"}

    def _fetch_webpage_content(self, results: List[Dict], page) -> None:
        """
        Fetch actual webpage content for the search results (fallback method)
        """
        for i, result in enumerate(results):
            start_time = time.time()
            try:
                print(f"📖 Getting webpage content for result {i+1}: {result['title'][:50]}...")
                
                target_url = result.get('_internal_url') or result.get('url', '')
                
                # Handle Baidu redirect URLs with extended timeout
                is_baidu_redirect = 'baidu.com/link?url=' in target_url
                if is_baidu_redirect:
                    
                    # Try to decode the redirect URL first
                    decoded_url = self._decode_baidu_redirect_url(target_url)
                    if decoded_url != target_url:
                        target_url = decoded_url
                        # print(f"🎯 Using decoded URL instead of redirect (fallback)")
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
                    # print(f"⏭️ Skip video/social media link: {target_url}")
                    result['content'] = "Video or social media link, skip content fetch"
                    continue
                
                if target_url.startswith('javascript:') or target_url.startswith('mailto:'):
                    # print(f"⏭️ Skip non-webpage link: {target_url}")
                    result['content'] = "Non-webpage link, skip content fetch"
                    continue
                
                if time.time() - start_time > 3:
                    # print("⏰ Processing time exceeded 3 seconds, skip this result")
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
                                # print(f"🔄 Retry attempt {attempt + 1} (fallback) for: {result['title'][:30]}...")
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
                            # print(f"✅ Successfully got {len(result['content'])} characters of useful content (time: {elapsed_time:.2f}s)")
                        else:
                            result['content'] = ""
                            # print(f"⚠️ Webpage content too short or unable to get, skip this result")
                
                except Exception as extract_error:
                    error_msg = str(extract_error)
                    if "ERR_HTTP2_PROTOCOL_ERROR" in error_msg:
                        error_msg = "HTTP2 protocol error"
                    elif "interrupted by another navigation" in error_msg:
                        error_msg = "Navigation interrupted"
                    
                    # print(f"⚠️ Content extraction failed: {error_msg}")  # Commented out to reduce terminal noise
                    result['content'] = ""
                
            except Exception as e:
                elapsed_time = time.time() - start_time
                # print(f"❌ Failed to get webpage content (time: {elapsed_time:.2f}s): {e}")  # Commented out to reduce terminal noise
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
                                    # print(f"✅ Successfully extracted content with selector '{selector}'")
                                    break
                        except:
                            continue
                    if content:
                        break
            
            if not content:
                try:
                    # print("⚠️ Selector method found no content, trying to extract full body text")
                    body_text = page.query_selector('body').text_content() if page.query_selector('body') else ""
                    
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
                            # print("✅ Successfully extracted using body content")
                except:
                    pass
            
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
                
                # print(f"📄 Successfully extracted content, total length: {len(content)} characters")
        
        except Exception as e:
            # print(f"Error extracting webpage content: {e}")  # Commented out to reduce terminal noise
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
            print("⚠️ Detected verification page in cleaning, returning verification message only")
            return "当前环境异常，完成验证后即可继续访问。"
        
        # Check for DocIn embedded document page early and return as-is
        if "豆丁网" in content or "docin.com" in content:
            print("⚠️ Detected DocIn embedded document page in cleaning, returning message only")
            return "正文为嵌入式文档，不可阅读"
        
        # Check for Baidu Scholar search page early and return as-is
        if ("百度学术搜索" in content or "百度学术" in content or
            "相关论文" in content or "获取方式" in content or
            "按相关性按相关性按被引量按时间降序" in content):
            print("⚠️ Detected Baidu Scholar search page in cleaning, returning message only")
            return "结果无可用数据"
        
        # Remove HTML tags
        content = re.sub(r'<[^>]+>', '', content)
        
        # Remove CSS blocks
        content = re.sub(r'\{[^{}]*\}', '', content, flags=re.DOTALL)
        
        # Remove JavaScript function blocks
        content = re.sub(r'function\s*\w*\s*\([^)]*\)\s*\{[^}]*\}', '', content, flags=re.DOTALL)
        content = re.sub(r'(var|let|const)\s+\w+\s*=.*?;', '', content)
        content = re.sub(r'\$\([^)]+\)\.[^;]+;?', '', content)
        
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
            print(f"⚠️  Ignoring additional parameters: {list(kwargs.keys())}")
        
        print(f"Fetching content from: {url}")
        
        # Set timeout for this operation
        old_handler = None
        if not is_windows():
            old_handler = signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(30)  # 30 second timeout
        
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
                
                if is_baidu_redirect:
                    print(f"🔄 Detected Baidu redirect URL, using extended timeout")
                    # Try to decode first
                    decoded_url = self._decode_baidu_redirect_url(url)
                    if decoded_url != url:
                        url = decoded_url
                        is_baidu_redirect = False  # No longer need special handling
                        print(f"🎯 Using decoded URL: {url[:100]}...")
                
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
                
                result_data = {
                    'title': title,
                    'content': cleaned_content if cleaned_content else content,
                    'content_length': len(cleaned_content if cleaned_content else content),
                    'timestamp': datetime.datetime.now().isoformat(),
                    'status': 'success'
                }
                
                if saved_html_path or saved_txt_path:
                    if saved_html_path:
                        result_data['saved_html_path'] = saved_html_path
                    if saved_txt_path:
                        result_data['saved_txt_path'] = saved_txt_path
                    
                    result_data['file_notice'] = f"📁 Webpage content saved to folder: {self.web_result_dir}/\n💡 You can use codebase_search or grep_search tools to search within these files"
                    print(f"\n📁 Webpage content saved to folder: {self.web_result_dir}/")
                    print(f"💡 You can use codebase_search or grep_search tools to search within these files")
                
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
            if not is_windows() and old_handler is not None:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)

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
                            print(f"✅ Successfully decoded Baidu redirect URL using method {i+1}")
                            print(f"🎯 Decoded URL: {decoded[:100]}...")
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
                        print(f"✅ Successfully decoded Baidu redirect URL using base64")
                        print(f"🎯 Decoded URL: {decoded[:100]}...")
                        return decoded
                except:
                    pass
                
                # If none of the decoding methods worked, the URL might be using Baidu's custom encoding
                # In that case, we can't easily decode it, so we'll keep the redirect URL
                # but still try to access it with extended timeout
                
                    
        except Exception as e:
            print(f"⚠️ Failed to decode Baidu redirect URL: {e}")
        
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
                print(f"🔄 Found future year {year}, replacing with current year {current_year}")
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