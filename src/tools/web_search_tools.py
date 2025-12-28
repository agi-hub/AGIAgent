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
from concurrent.futures import ThreadPoolExecutor
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
    def __init__(self, llm_api_key: str = None, llm_model: str = None, llm_api_base: str = None, enable_llm_filtering: bool = False, enable_summary: bool = True, workspace_root: str = None, out_dir: str = None, verbose: bool = True):
        self.verbose = verbose  # Control verbose debug output
        
        # LLM configuration for content filtering and summarization
        self.enable_llm_filtering = enable_llm_filtering
        self.enable_summary = enable_summary
        self.llm_client = None
        self.llm_model = llm_model
        self.is_claude = False
        
        # Track failed search engines to skip them in future attempts
        self.failed_engines = set()
        
        # Track downloaded URLs to avoid duplicate downloads
        self.downloaded_urls = set()
        
        # Store workspace root for relative path calculation
        # workspace_root should point to the workspace directory, not the parent out_dir
        if workspace_root:
            self.workspace_root = workspace_root
            # Calculate out_dir from workspace_root if not provided
            if not out_dir:
                # If workspace_root ends with 'workspace', out_dir is its parent
                if os.path.basename(workspace_root) == 'workspace':
                    out_dir = os.path.dirname(workspace_root)
                else:
                    out_dir = workspace_root
        else:
            # Fallback: use out_dir or current working directory
            if out_dir:
                # If out_dir is provided, workspace_root should be out_dir/workspace
                self.workspace_root = os.path.join(out_dir, "workspace")
            else:
                self.workspace_root = os.getcwd()
                out_dir = os.getcwd()
        
        # Initialize web search result directory path but don't create it yet
        # web_result_dir should be in workspace/web_search_result
        self.web_result_dir = os.path.join(self.workspace_root, "web_search_result")
        
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
    
    def _detect_special_page(self, content: str, title: str = "", url: str = "") -> tuple:
        """
        统一检测特殊页面（验证页面、豆丁网嵌入文档、百度文库、百度移动端、百度学术搜索页面、DuckDuckGo帮助页面等）
        
        Args:
            content: 页面内容（HTML或文本）
            title: 页面标题（可选）
            url: 页面URL（可选）
            
        Returns:
            tuple: (is_special, message_type, message)
            - is_special: 是否是特殊页面
            - message_type: 特殊页面类型 ("verification", "docin", "baidu_wenku", "baidu_mbd", "baidu_scholar", "duckduckgo_help", None)
            - message: 返回的消息内容
        """
        if not content:
            # 即使没有内容，也检查URL（用于快速过滤）
            if url:
                url_lower = url.lower()
                if "wenku.baidu.com" in url_lower:
                    return True, "baidu_wenku", "Baidu Wenku page, unable to download correct text, automatically filtered"
            return False, None, ""
        
        # 检测 DuckDuckGo 帮助页面和广告页面（优先通过 URL 检测）
        if url:
            url_lower = url.lower()
            if ("duckduckgo.com/duckduckgo-help-pages" in url_lower or
                "duckduckgo-help-pages" in url_lower or
                "ads-by-microsoft" in url_lower or
                "ads-by-yelp" in url_lower or
                "ads-by-tripadvisor" in url_lower):
                return True, "duckduckgo_help", "DuckDuckGo Ads, filtered"
            
            # 检测百度文库（优先通过 URL 检测）
            if "wenku.baidu.com" in url_lower:
                return True, "baidu_wenku", "Baidu Wenku page, unable to download correct text, automatically filtered"
            
            # 检测百度移动端（通常是视频，无法下载正确文字）
            if "mbd.baidu.com" in url_lower:
                return True, "baidu_mbd", "Baidu mobile page (usually video), unable to download correct text, automatically filtered"
        
        # 通过内容检测 DuckDuckGo 帮助页面
        if ("duckduckgo-help-pages" in content.lower() or
            "Ads By Microsoft on DuckDuckGo" in content or
            "Ads By Yelp on DuckDuckGo" in content or
            "Ads By Tripadvisor on DuckDuckGo" in content or
            "Ads by Microsoft on DuckDuckGo Private Search" in content):
            return True, "duckduckgo_help", "DuckDuckGo Ads, filtered"
        
        # 检测验证页面
        if "当前环境异常，完成验证后即可继续访问。" in content:
            return True, "verification", "当前环境异常，完成验证后即可继续访问。"
        
        # 检测豆丁网嵌入式文档页面
        if "豆丁网" in content or "docin.com" in content:
            return True, "docin", "正文为嵌入式文档，不可阅读"
        
        # 检测百度文库（通过内容检测，作为URL检测的备用）
        if "wenku.baidu.com" in content.lower() or "百度文库" in content:
            return True, "baidu_wenku", "Baidu Wenku page, unable to download correct text, automatically filtered"
        
        # 检测百度移动端（通过内容检测，作为URL检测的备用）
        if "mbd.baidu.com" in content.lower():
            return True, "baidu_mbd", "Baidu mobile page (usually video), unable to download correct text, automatically filtered"
        
        # 检测百度学术搜索页面
        if ("百度学术搜索" in content or "xueshu.baidu.com" in content or
            "百度学术" in content or "- 百度学术" in title or
            "相关论文" in content or "获取方式" in content or
            "按相关性按相关性按被引量按时间降序" in content):
            return True, "baidu_scholar", "结果无可用数据"
        
        return False, None, ""
    
    def _ensure_result_directory(self):
        """Ensure the web search result directory exists"""
        if not self.web_result_dir:
            # If web_result_dir is not set, try to initialize it using workspace_root
            if self.workspace_root:
                self.web_result_dir = os.path.join(self.workspace_root, "web_search_result")
            else:
                # Fallback to current working directory
                self.web_result_dir = os.path.join(os.getcwd(), "workspace", "web_search_result")
        
        try:
            os.makedirs(self.web_result_dir, exist_ok=True)
            # Verify directory was created successfully
            if not os.path.exists(self.web_result_dir):
                raise Exception(f"Directory creation failed: {self.web_result_dir}")
        except Exception as e:
            print_current(f"⚠️ Failed to create result directory: {e}")
            print_current(f"⚠️ Attempted path: {self.web_result_dir}")
            # Don't set to None, keep the path for debugging
            # But log the error so user knows files won't be saved
    
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
            print_current(f"⚠️ Cannot save files: web_result_dir is not set")
            return "", ""
        
        # Verify directory exists before attempting to save
        if not os.path.exists(self.web_result_dir):
            print_current(f"⚠️ Cannot save files: directory does not exist: {self.web_result_dir}")
            return "", ""
        
        # 首先通过 URL 检测特殊页面（快速检测，避免加载页面内容）
        is_special_by_url = False
        page_type_by_url = None
        message_by_url = ""
        if url:
            _, page_type_by_url, message_by_url = self._detect_special_page("", title, url)
            if page_type_by_url:
                is_special_by_url = True
        
        # 如果通过 URL 检测到特殊页面，直接返回，不保存
        if is_special_by_url:
            print_debug(f"⚠️ {message_by_url}: {url[:80]}...")
            return "", ""
        
        # 规范化URL并检查是否已下载（去除查询参数、锚点等，只保留基础URL）
        normalized_url = self._normalize_url_for_dedup(url) if url else ""
        if normalized_url and normalized_url in self.downloaded_urls:
            print_debug(f"⏭️ 跳过重复URL: {url[:80]}... (已下载)")
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
                
                # 确保html_content是字符串类型（Playwright通常返回UTF-8字符串）
                if isinstance(html_content, bytes):
                    # 如果是字节，尝试检测编码并解码
                    detected_encoding = self._detect_html_encoding(html_content)
                    try:
                        html_content = html_content.decode(detected_encoding)
                    except (UnicodeDecodeError, LookupError):
                        try:
                            html_content = html_content.decode('utf-8')
                        except UnicodeDecodeError:
                            html_content = html_content.decode('utf-8', errors='replace')
                            print_debug(f"⚠️ Used error replacement for HTML encoding")
                
                # 确保HTML中的charset声明是UTF-8
                if re.search(r'<meta[^>]*charset', html_content, re.IGNORECASE):
                    html_content = re.sub(
                        r'(<meta[^>]*charset\s*=\s*["\']?)[^"\'\s>]+',
                        r'\1utf-8',
                        html_content,
                        flags=re.IGNORECASE,
                        count=1
                    )
                elif '<head>' in html_content.lower():
                    html_content = re.sub(
                        r'(<head[^>]*>)',
                        r'\1\n<meta http-equiv="Content-Type" content="text/html; charset=UTF-8" />',
                        html_content,
                        flags=re.IGNORECASE,
                        count=1
                    )
                
                # 使用统一的特殊页面检测（传递 URL 以便检测 DuckDuckGo 帮助页面）
                is_special, page_type, message = self._detect_special_page(html_content, title, url)
                
                if is_special:
                    print_debug(f"⚠️ {message}: {url[:80]}...")
                    return "", ""  # 检测到特殊页面，不保存
                
                if not is_special:
                    # Ensure the HTML file has .html extension
                    html_filename = f"{base_filename}.html"
                    html_filepath = os.path.join(self.web_result_dir, html_filename)
                    with open(html_filepath, 'w', encoding='utf-8') as f:
                        f.write(html_content)
                    # 成功保存HTML后，记录URL以避免重复下载
                    if normalized_url:
                        self.downloaded_urls.add(normalized_url)

                    
            except Exception as e:
                print_current(f"⚠️ Failed to save webpage HTML: {e}")
            
            # Save text content
            try:
                if content and content.strip():
                    # For very large content, truncate before cleaning to avoid timeout
                    MAX_CONTENT_LENGTH_FOR_CLEANING = 500000  # 500KB limit
                    content_to_clean = content
                    if len(content) > MAX_CONTENT_LENGTH_FOR_CLEANING:
                        print_current(f"⚠️ Content too large ({len(content)} chars), truncating to {MAX_CONTENT_LENGTH_FOR_CLEANING} chars before cleaning")
                        content_to_clean = content[:MAX_CONTENT_LENGTH_FOR_CLEANING]
                    
                    # Clean the content thoroughly for saving with timeout protection
                    cleaned_content = None
                    try:
                        # Use threading timeout for cleaning operation (10 seconds max)
                        from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
                        with ThreadPoolExecutor(max_workers=1) as executor:
                            future = executor.submit(self._clean_text_for_saving, content_to_clean)
                            try:
                                cleaned_content = future.result(timeout=10.0)  # 10 second timeout
                            except FutureTimeoutError:
                                print_current(f"⚠️ Content cleaning timeout (10s), using truncated content without full cleaning")
                                # Fallback: use simplified cleaning for timeout case
                                cleaned_content = self._clean_text_for_saving_simple(content_to_clean)
                    except Exception as clean_error:
                        print_current(f"⚠️ Content cleaning failed: {clean_error}, using simplified cleaning")
                        # Fallback to simplified cleaning
                        try:
                            cleaned_content = self._clean_text_for_saving_simple(content_to_clean)
                        except Exception as simple_clean_error:
                            print_current(f"⚠️ Simplified cleaning also failed: {simple_clean_error}, saving original content")
                            cleaned_content = content_to_clean
                    
                    # Log if content was filtered out during cleaning
                    if not cleaned_content or not cleaned_content.strip():
                        print_debug(f"⚠️ Content was filtered out during cleaning for: {title[:50]}... (original length: {len(content)})")
                    elif len(cleaned_content.strip()) <= 50:
                        print_debug(f"⚠️ Content too short after cleaning for: {title[:50]}... (original: {len(content)}, cleaned: {len(cleaned_content.strip())})")
                    
                    if cleaned_content and len(cleaned_content.strip()) > 50:
                        # Ensure the txt file has .txt extension
                        txt_filename = f"{base_filename}.txt"
                        txt_filepath = os.path.join(self.web_result_dir, txt_filename)
                        
                        # Create a formatted text file with metadata (统一格式)
                        formatted_content = f"""Title: {title}
URL: {url}
Search Term: {search_term}
Timestamp: {datetime.datetime.now().isoformat()}
Original Content Length: {len(content)} characters
Cleaned Content Length: {len(cleaned_content)} characters


{cleaned_content}
"""
                        
                        try:
                            with open(txt_filepath, 'w', encoding='utf-8') as f:
                                f.write(formatted_content)
                            # 成功保存后，记录URL以避免重复下载
                            if normalized_url:
                                self.downloaded_urls.add(normalized_url)
                        except Exception as write_error:
                            print_current(f"⚠️ Failed to write text file: {write_error}")
                            txt_filepath = ""  # Reset filepath on write failure
                    else:
                        # Content exists but was filtered out or too short
                        if content and content.strip():
                            print_debug(f"⚠️ Skipped saving text file for: {title[:50]}... (content length: {len(content)}, cleaned length: {len(cleaned_content.strip()) if cleaned_content else 0})")
                else:
                    print_debug(f"⚠️ No text content to save for: {title[:50]}... (content is empty or whitespace only)")
            except TimeoutError as timeout_err:
                print_current(f"⚠️ Save text content timeout: {timeout_err}, continuing without txt file")
                txt_filepath = ""  # Ensure we return empty string on timeout
            except Exception as e:
                print_current(f"⚠️ Failed to save text content: {e}, continuing without txt file")
                txt_filepath = ""  # Ensure we return empty string on error
            
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
        
        # 使用统一的特殊页面检测并跳过LLM处理
        is_special, page_type, message = self._detect_special_page(content)
        if is_special:
            if page_type == "verification":
                print_current("⚠️ Detected verification page in LLM filtering, skipping LLM processing")
            elif page_type == "docin":
                print_current("⚠️ Detected DocIn embedded document page in LLM filtering, skipping LLM processing")
            elif page_type == "baidu_wenku":
                print_debug("⚠️ Detected Baidu Wenku page in LLM filtering, skipping LLM processing")
            elif page_type == "baidu_mbd":
                print_debug("⚠️ Detected Baidu MBD page in LLM filtering, skipping LLM processing")
            elif page_type == "baidu_scholar":
                print_current("⚠️ Detected Baidu Scholar search page in LLM filtering, skipping LLM processing")
            elif page_type == "duckduckgo_help":
                print_current("⚠️ Detected DuckDuckGo help/ad page in LLM filtering, skipping LLM processing")
            return message
        
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
            print_current(f"LLM content filtering failed: {e}")
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
        # Minimum 200 characters after filtering
        valid_results = []
        for i, result in enumerate(results):
            content = result.get('content', '')
            if content and len(content.strip()) > 200:
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
            print_current(f"Search results summarization failed: {e}")
            return ""

    def web_search(self, search_term: str, fetch_content: bool = True, max_content_results: int = 5, **kwargs) -> Dict[str, Any]:
        """
        Search the web for real-time information using Playwright.
        """
        # Check if Playwright is available before proceeding
        if not is_playwright_available():
            print_current("Playwright is not installed or not available")
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
        
        # Set global timeout of 90 seconds for the entire search operation
        # Increased from 60s to accommodate multiple search engine attempts
        old_handler = None
        if not is_windows() and is_main_thread():
            try:
                old_handler = signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(90)  # 90 seconds timeout
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
                    
                    # 搜索结果页面使用桌面版以确保正确的DOM结构
                    # Search results page uses desktop version for proper DOM structure
                    context = browser.new_context(
                        user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36',
                        viewport={'width': 1366, 'height': 768},
                        ignore_https_errors=True,
                        java_script_enabled=True,
                        bypass_csp=True,
                        locale='en-US',
                        timezone_id='America/New_York',
                        extra_http_headers={
                            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
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
                
                # 🚀 性能优化：搜索结果页面不拦截资源，确保页面正常加载
                # Performance optimization: No blocking on search pages to ensure proper loading
                # Note: We only optimize content download pages, not search result pages
                print_debug("🔍 Search results page: no resource blocking to ensure compatibility")
                
                # Set longer page timeout to prevent connection issues
                page.set_default_timeout(15000)  # 15 seconds (increased from 8s)
                page.set_default_navigation_timeout(20000)  # 20 seconds (increased from 12s)
                
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
                        
                        
                        content = self._extract_main_content(page)
                            
                        # Apply LLM filtering if enabled
                        if content and self.enable_llm_filtering:
                            content = self._extract_relevant_content_with_llm(content, search_term, title)
                            
                        # Always try to save HTML file (even if text content is short)
                        # Text file will only be saved if content is long enough
                        if content:
                            print_debug(f"💾 Attempting to save content for: {title[:50]}... (content length: {len(content)})")
                            saved_html_path, saved_txt_path = self._save_webpage_content(page, search_term, title, content, search_term)
                            if saved_html_path or saved_txt_path:
                                print_debug(f"✅ Successfully saved files - HTML: {saved_html_path}, TXT: {saved_txt_path}")
                            else:
                                print_current(f"⚠️ Failed to save files for: {title[:50]}...")
                        else:
                            # Even if no content extracted, try to save HTML
                            print_debug(f"⚠️ No content extracted, but attempting to save HTML for: {title[:50]}...")
                            saved_html_path, saved_txt_path = self._save_webpage_content(page, search_term, title, "", search_term)
                            if saved_html_path:
                                print_debug(f"✅ Saved HTML file even without text content: {saved_html_path}")
                        
                        # Clean content for better LLM processing
                        cleaned_content = self._clean_text_for_saving(content) if content else ""
                        
                        result_dict = {
                            'title': title,
                            'url': search_term,  # The search_term is the direct URL in this case
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
                        print_current(f"Direct URL access failed: {e}")
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
                    # Try using requests first for faster search (no browser needed)
                    print_debug("🔍 Trying requests-based search (faster, no browser)")
                    
                    try:
                        results = self._search_with_requests(search_term)
                        if results:
                            print_debug(f"✅ Requests-based search successful, found {len(results)} results")
                        else:
                            print_debug("⚠️ Requests-based search returned no results, falling back to Playwright")
                            results = []
                    except Exception as e:
                        print_debug(f"⚠️ Requests-based search failed: {e}, falling back to Playwright")
                        results = []
                    
                    # If requests search failed, fall back to Playwright
                    if not results:
                        print_debug("🔍 Using Playwright for search")
                        # Initialize available search engines
                        # DuckDuckGo always as primary search engine (most bot-friendly, no anti-bot measures)
                        print_debug("🔍 Using DuckDuckGo as primary search engine")
                        
                        search_engines = []
                        
                        # Add DuckDuckGo first (always first regardless of language)
                        search_engines.append({
                            'name': 'DuckDuckGo',
                            'url': 'https://html.duckduckgo.com/html/?q={}',
                            'result_selector': '.result__a, .web-result__a, a.result__a, .result a, .links_main a',
                            'container_selector': '.result, .web-result, .links_main',
                            'snippet_selectors': ['.result__snippet', '.result__body', '.snippet', '.result__description']
                        })
                        print_debug("🔍 DuckDuckGo search engine added as primary option")
                        
                        # Check if query contains Chinese characters for secondary options
                        has_chinese = bool(re.search(r'[\u4e00-\u9fff]', search_term))
                        
                        if has_chinese:
                            # For Chinese queries, add Baidu as secondary option
                            search_engines.append({
                                'name': 'Baidu',
                                'url': 'https://www.baidu.com/s?wd={}',
                                'result_selector': 'h3.t a, h3 a, .t a, .c-title-text, .c-title a, .result h3 a, .c-container h3 a, .c-title a, a[data-click], .result-op h3 a, .c-row h3 a, .c-gap-top-small h3 a',
                                'container_selector': '.result, .c-container, .result-op, .c-row, .c-gap-top-small',
                                'snippet_selectors': ['.c-abstract', '.c-span9', '.c-abstract-text', '.c-color-text', 'span', 'div', '.c-span12']
                            })
                            print_debug("🔍 Baidu search engine added as secondary option (Chinese query)")
                        
                        # Add Google as fallback option
                        search_engines.append({
                            'name': 'Google',
                            'url': 'https://www.google.com/search?q={}&gl=us&hl=en&safe=off',
                            'result_selector': 'h3 a, h1 a, .g a h3, .yuRUbf a h3, .LC20lb, .DKV0Md, [data-ved] h3',
                            'container_selector': '.g, .tF2Cxc, [data-ved]',
                            'snippet_selectors': ['.VwiC3b', '.s', '.st', 'span', '.IsZvec', '.aCOpRe', '.yXK7lf'],
                            'anti_bot_indicators': ['Our systems have detected unusual traffic', 'g-recaptcha', 'captcha', 'verify you are human', 'blocked', 'unusual activity']
                        })
                        print_debug("🔍 Google search engine added as fallback option")
                        
                        # If not Chinese query, also add Baidu as last resort
                        if not has_chinese:
                            search_engines.append({
                                'name': 'Baidu',
                                'url': 'https://www.baidu.com/s?wd={}',
                                'result_selector': 'h3.t a, h3 a, .t a, .c-title-text, .c-title a, .result h3 a, .c-container h3 a, .c-title a, a[data-click], .result-op h3 a, .c-row h3 a, .c-gap-top-small h3 a',
                                'container_selector': '.result, .c-container, .result-op, .c-row, .c-gap-top-small',
                                'snippet_selectors': ['.c-abstract', '.c-span9', '.c-abstract-text', '.c-color-text', 'span', 'div', '.c-span12']
                            })
                            print_debug("🔍 Baidu search engine added as last resort option")
                        
                        optimized_search_term = self._optimize_search_term(search_term)
                        encoded_term = urllib.parse.quote_plus(optimized_search_term)
                        
                        for engine_idx, engine in enumerate(search_engines):
                            try:
                                # Skip this engine if it has failed before
                                if engine['name'] in self.failed_engines:
                                    print_debug(f"⏭️ Skipping {engine['name']} (failed in previous attempt)")
                                    continue
                                
                                print_debug(f"🔍 Trying to search with {engine['name']}")
                                
                                search_url = engine['url'].format(encoded_term)
                                
                                # Use longer timeout for all search engines to ensure reliability
                                # DuckDuckGo: 12s, Baidu: 20s (longer for better results), Google: 12s
                                timeout_ms = 20000 if engine['name'] == 'Baidu' else 12000
                                page.goto(search_url, timeout=timeout_ms, wait_until='domcontentloaded')
                                
                                # Wait longer for page to stabilize and load results
                                # Baidu needs more time to load all results
                                wait_time = 2500 if engine['name'] == 'Baidu' else 1500
                                page.wait_for_timeout(wait_time)
                                
                                # For Baidu, try to scroll down to load more results (lazy loading)
                                if engine['name'] == 'Baidu':
                                    try:
                                        # Scroll down multiple times to trigger lazy loading
                                        # Increase scroll attempts to load more results
                                        for scroll_attempt in range(5):  # Increased from 3 to 5
                                            # Scroll to bottom
                                            page.evaluate('window.scrollTo(0, document.body.scrollHeight)')
                                            page.wait_for_timeout(1000)  # Increased wait time from 800ms to 1000ms
                                            # Also try scrolling incrementally
                                            scroll_position = (scroll_attempt + 1) * 1000
                                            page.evaluate(f'window.scrollTo(0, {scroll_position})')
                                            page.wait_for_timeout(500)
                                        # Scroll back to top
                                        page.evaluate('window.scrollTo(0, 0)')
                                        page.wait_for_timeout(1000)  # Increased wait time
                                        print_debug(f"📜 Scrolled Baidu page {5} times to load more results")
                                    except Exception as scroll_error:
                                        print_debug(f"⚠️ Baidu scroll failed: {scroll_error}")
                                
                                # Get search results with error handling
                                # For Baidu, query multiple times after scrolling to ensure we get all loaded results
                                result_elements = []
                                try:
                                    # First attempt
                                    result_elements = page.query_selector_all(engine['result_selector'])
                                    if engine['name'] == 'Baidu':
                                        # Wait a bit more and query again (lazy loading may need more time)
                                        page.wait_for_timeout(1000)
                                        # Query again to get newly loaded results
                                        additional_elements = page.query_selector_all(engine['result_selector'])
                                        if len(additional_elements) > len(result_elements):
                                            result_elements = additional_elements
                                            print_debug(f"📜 Baidu: Found {len(result_elements)} results after additional wait")
                                except Exception as selector_error:
                                    print_debug(f"⚠️ Selector error for {engine['name']}: {selector_error}")
                                    # Fallback to basic result selector
                                    try:
                                        result_elements = page.query_selector_all('a[href], h3, .result, .g, .rc')
                                    except Exception as fallback_error:
                                        print_debug(f"Fallback selector also failed: {fallback_error}")
                                        result_elements = []
                                
                                if result_elements:
                                    print_debug(f"✅ {engine['name']} search successful, found {len(result_elements)} result elements")
                                    
                                    # Collect all results first (not just first 10)
                                    all_raw_results = []
                                    valid_count = 0
                                    skipped_count = 0
                                    for i, elem in enumerate(result_elements):
                                        try:
                                            title = ""
                                            url = ""
                                            snippet = ""

                                            # Safely get text content
                                            try:
                                                title = elem.text_content().strip()
                                            except Exception as text_error:
                                                # If text_content fails (e.g., "Execution context was destroyed"), try inner_text
                                                try:
                                                    title = elem.inner_text().strip()
                                                except:
                                                    print_debug(f"⚠️ Failed to extract title for result {i}: {text_error}")
                                                    continue  # Skip this result
                                            
                                            url = elem.get_attribute('href') or ""
                                            
                                            # For Baidu, if URL is not found on the element itself, try to find it in parent/container
                                            if engine['name'] == 'Baidu' and (not url or not url.startswith(('http://', 'https://', '/', 'javascript:', 'mailto:'))):
                                                try:
                                                    # Try to find link in parent container
                                                    parent = elem.query_selector('xpath=ancestor::*[contains(@class, "result") or contains(@class, "c-container") or contains(@class, "c-row")][1]')
                                                    if parent:
                                                        # Look for link in parent
                                                        link_in_parent = parent.query_selector('a[href]')
                                                        if link_in_parent:
                                                            url = link_in_parent.get_attribute('href') or url
                                                except Exception as parent_error:
                                                    print_debug(f"⚠️ Failed to find URL in parent for Baidu result {i}: {parent_error}")
                                            
                                            snippet = self._extract_snippet_from_search_result(elem, engine)
                                            
                                            # Handle Google URL format
                                            if url and url.startswith('/url?q='):
                                                url = urllib.parse.unquote(url.split('&')[0][7:])
                                            
                                            # Handle Baidu URL format (may be relative or absolute)
                                            if url and not url.startswith(('http://', 'https://', 'javascript:', 'mailto:')):
                                                # Baidu may use relative URLs, try to construct absolute URL
                                                if url.startswith('/'):
                                                    url = 'https://www.baidu.com' + url
                                                elif url.startswith('//'):
                                                    url = 'https:' + url
                                                # If still not absolute, try to get from href attribute again
                                                if not url.startswith(('http://', 'https://')):
                                                    # Try to get the actual href
                                                    try:
                                                        actual_href = elem.get_attribute('href')
                                                        if actual_href and actual_href.startswith(('http://', 'https://')):
                                                            url = actual_href
                                                    except:
                                                        pass
                                            
                                            # More lenient filtering for Baidu (minimum 3 chars for title)
                                            min_title_length = 3 if engine['name'] == 'Baidu' else 5
                                            if title and len(title) >= min_title_length and url and url.startswith(('http://', 'https://')):
                                                all_raw_results.append({
                                                    'title': title,
                                                    'snippet': snippet,
                                                    'url': url,
                                                    'source': engine['name']
                                                })
                                                valid_count += 1
                                            else:
                                                skipped_count += 1
                                                if engine['name'] == 'Baidu' and i < 5:  # Debug first 5 skipped results
                                                    print_debug(f"⚠️ Baidu result {i+1} skipped: title_len={len(title) if title else 0}, url_valid={url.startswith(('http://', 'https://')) if url else False}")
                                        
                                        except Exception as e:
                                            print_debug(f"Error extracting result {i}: {e}")
                                            skipped_count += 1
                                            continue
                                    
                                    print_debug(f"📊 {engine['name']} extraction stats: {valid_count} valid, {skipped_count} skipped out of {len(result_elements)} elements")
                                    
                                    # Parse and deduplicate URLs before adding to results
                                    if all_raw_results:
                                        # Parse URLs (decode DuckDuckGo/Baidu redirects) and deduplicate
                                        seen_urls = set()
                                        deduplicated_results = []
                                        
                                        for raw_result in all_raw_results:
                                            url = raw_result['url']
                                            
                                            # Filter out Baidu Wenku and MBD links early (cannot download correct text)
                                            if 'wenku.baidu.com' in url.lower():
                                                continue
                                            if 'mbd.baidu.com' in url.lower():
                                                continue
                                            
                                            # Decode redirect URLs to get real destination
                                            if 'duckduckgo.com/l/' in url.lower() and 'uddg=' in url.lower():
                                                decoded_url = self._decode_duckduckgo_redirect_url(url)
                                                if decoded_url != url:
                                                    url = decoded_url
                                            
                                            if 'baidu.com/link?url=' in url.lower():
                                                decoded_url = self._decode_baidu_redirect_url(url)
                                                if decoded_url != url:
                                                    url = decoded_url
                                            
                                            # Check again after decoding (decoded URL might be Baidu Wenku or MBD)
                                            if 'wenku.baidu.com' in url.lower():
                                                continue
                                            if 'mbd.baidu.com' in url.lower():
                                                continue
                                            
                                            # Normalize URL for deduplication
                                            normalized_url = self._normalize_url_for_dedup(url)
                                            
                                            # Skip if already seen
                                            if normalized_url in seen_urls:
                                                continue
                                            
                                            seen_urls.add(normalized_url)
                                            
                                            # Clean snippet before truncating
                                            cleaned_snippet = self._clean_snippet(raw_result['snippet']) if raw_result['snippet'] else f'Search result from {raw_result["source"]}'
                                            deduplicated_results.append({
                                                'title': raw_result['title'],
                                                'snippet': cleaned_snippet[:get_truncation_length()] if cleaned_snippet else f'Search result from {raw_result["source"]}',
                                                'source': raw_result['source'],
                                                'content': '',
                                                '_internal_url': url,  # Use decoded/normalized URL
                                                'url': url  # Also store in url field for compatibility
                                            })
                                        
                                        print_debug(f"📋 {engine['name']} deduplication: {len(all_raw_results)} raw → {len(deduplicated_results)} unique results")
                                        results.extend(deduplicated_results)
                                    
                                    if results:
                                        break
                                else:
                                    print_debug(f"{engine['name']} found no search results")
                            
                            except Exception as e:
                                print_debug(f"{engine['name']} search failed: {e}")
                                # Mark this engine as failed so it won't be retried
                                self.failed_engines.add(engine['name'])
                                print_debug(f"🚫 {engine['name']} marked as failed, will be skipped in future attempts")
                                continue
                    
                    if fetch_content and results:
                        # Show which URLs will be downloaded
                        
                        
                        # Download results until we have enough valid ones (max_content_results)
                        downloaded_indices = set()  # Track which results have been downloaded
                        valid_results = []
                        batch_size = max_content_results
                        max_attempts = min(len(results), max_content_results * 3)  # Try up to 3x the target to find enough valid results
                        
                        # Print header once before all batches
                        #print_current(f"📖 Fetching webpage content")
                        
                        attempt = 0
                        while len(valid_results) < max_content_results and attempt < max_attempts and len(downloaded_indices) < len(results):
                            # Find next batch of results to download
                            batch_to_download = []
                            for idx, result in enumerate(results):
                                if idx not in downloaded_indices:
                                    batch_to_download.append((idx, result))
                                    if len(batch_to_download) >= batch_size:
                                        break
                            
                            if not batch_to_download:
                                break
                            
                            # Download this batch
                            batch_results = [result for _, result in batch_to_download]
                            batch_indices = [idx for idx, _ in batch_to_download]
                            
                            # Print batch details with continuous numbering
                            current_offset = len(downloaded_indices)
                            for idx, result in enumerate(batch_results):
                                url = result.get('_internal_url') or result.get('url', 'N/A')
                                title = result.get('title', 'Untitled')[:60]
                                print_debug(f"  [{current_offset + idx + 1}] {title} -> {url[:100]}...")
                            
                            try:
                                self._fetch_webpage_content_with_timeout(batch_results, page, timeout_seconds=60)
                            except Exception as e:
                                print_debug(f"⚠️ Content fetching failed: {e}")
                                # Try basic method as fallback
                                try:
                                    self._fetch_webpage_content(batch_results, page)
                                except Exception as final_e:
                                    print_debug(f"⚠️ All content fetching methods failed: {final_e}")
                            
                            # Mark these as downloaded
                            for idx in batch_indices:
                                downloaded_indices.add(idx)
                            
                            # Check which results in this batch are valid
                            # Valid content must be at least 200 characters after filtering
                            for result in batch_results:
                                # Keep results with good content (minimum 200 characters)
                                content = result.get('content', '')
                                if content and len(content.strip()) > 200:
                                    # Skip error messages and filtered pages
                                    if not any(error_phrase in content.lower() for error_phrase in [
                                        'content fetch error', 'processing error', 'timeout',
                                        '无法下载正确文字', '已自动过滤', 'skip content fetch',
                                        'video or social media link', 'advertisement page'
                                    ]):
                                        if result not in valid_results:  # Avoid duplicates
                                            valid_results.append(result)
                                elif not fetch_content:
                                    if result not in valid_results:
                                        valid_results.append(result)
                                elif result.get('snippet') and len(result['snippet'].strip()) > 20:
                                    # Keep results with useful snippets even if content fetch failed
                                    if result not in valid_results:
                                        valid_results.append(result)
                            
                            attempt += 1
                            
                            # If we have enough valid results, stop
                            if len(valid_results) >= max_content_results:
                                break
                        
                        # Use valid results, or fall back to all downloaded results
                        if valid_results:
                            results = valid_results[:max_content_results]  # Limit to max_content_results
                            # Count results with actual content (not just snippets)
                            results_with_content = [r for r in results if r.get('content') and len(r.get('content', '').strip()) > 200]
                            results_with_snippet_only = len(results) - len(results_with_content)
                        else:
                            print_debug("⚠️ No results with valid content found, returning search results only")
                            # Return downloaded results even without content
                            downloaded_results = [results[idx] for idx in sorted(downloaded_indices)][:max_content_results]
                            for result in downloaded_results:
                                if not result.get('content'):
                                    result['content'] = 'Content not available - search result only'
                            results = downloaded_results
                
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
                    # Minimum 200 characters to be considered valid content
                    if result.get('content') and len(result['content'].strip()) > 200:
                        content = result['content']
                        summary_truncation_length = get_truncation_length() // 5  # Use 1/5 of truncation length
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
                            'url': result.get('_internal_url') or result.get('url', ''),
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
                        'url': result.get('_internal_url') or result.get('url', ''),
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
                print_current(f"Playwright system compatibility error: {playwright_error}")
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
                print_current(f"Playwright initialization error: {playwright_error}")
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
                print_current(f"Playwright error: {playwright_error}")
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
                    'snippet': self._clean_snippet('Web search operation timed out after 90 seconds. This may be due to slow network or search engines blocking requests. Please try again or use a different search term.'),
                    'content': 'Search operation timed out'
                }],
                'timestamp': datetime.datetime.now().isoformat(),
                'error': 'search_timeout'
            }
        
        except Exception as e:
            print_current(f"Web search failed: {e}")
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
        Handles navigation errors gracefully
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
                            try:
                                text = snippet_elem.text_content().strip()
                            except Exception as text_error:
                                # If text_content fails (e.g., context destroyed), try inner_text
                                try:
                                    text = snippet_elem.inner_text().strip()
                                except:
                                    continue
                            
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
                                try:
                                    text = text_elem.text_content().strip()
                                except:
                                    try:
                                        text = text_elem.inner_text().strip()
                                    except:
                                        continue
                                
                                if text and len(text) > 30 and len(text) < 200:
                                    if any(char in text for char in '，。？！、；：') or ' ' in text:
                                        snippet = text
                                        break
                            except Exception as text_error:
                                continue
                    except Exception as text_elements_error:
                        print_debug(f"⚠️ Text elements selector error: {text_elements_error}")
        
        except Exception as e:
            # Check if it's a navigation/context error
            if "Execution context was destroyed" in str(e) or "navigation" in str(e).lower():
                print_debug(f"⚠️ Page navigation during snippet extraction: {e}")
            else:
                print_current(f"Error extracting snippet: {e}")
        
        # Clean the snippet to remove excessive whitespace
        return self._clean_snippet(snippet)

    def _search_with_requests(self, search_term: str) -> List[Dict]:
        """
        使用 requests 进行搜索（不需要浏览器，更快）
        目前只支持 DuckDuckGo HTML 版本
        
        Args:
            search_term: 搜索词
            
        Returns:
            搜索结果列表
        """
        try:
            import requests
            from bs4 import BeautifulSoup
            import urllib.parse
            
            # 禁用 SSL 警告
            import urllib3
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
            
            # 使用 DuckDuckGo HTML 版本（对爬虫友好）
            encoded_term = urllib.parse.quote_plus(search_term)
            search_url = f'https://html.duckduckgo.com/html/?q={encoded_term}'
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.9',
                'Accept-Encoding': 'gzip, deflate, br',
                'Connection': 'keep-alive'
            }
            
            # 发送搜索请求
            response = requests.get(search_url, headers=headers, timeout=10, verify=False)
            
            if response.status_code != 200:
                print_debug(f"⚠️ Search request failed with status {response.status_code}")
                return []
            
            # 解析 HTML
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # 查找搜索结果
            # DuckDuckGo HTML 使用 .result 类
            result_divs = soup.find_all('div', class_='result')
            
            if not result_divs:
                # 尝试其他可能的选择器
                result_divs = soup.find_all('div', class_='web-result')
            
            if not result_divs:
                print_debug(f"⚠️ No results found in HTML")
                return []
            
            print_debug(f"🔍 Found {len(result_divs)} result containers")
            
            results = []
            for i, result_div in enumerate(result_divs[:30]):  # 限制最多 30 个结果
                try:
                    # 提取标题和链接
                    title_link = result_div.find('a', class_='result__a')
                    if not title_link:
                        # 尝试其他可能的类名
                        title_link = result_div.find('a', class_='web-result__a')
                    if not title_link:
                        continue
                    
                    title = title_link.get_text(strip=True)
                    url = title_link.get('href', '')
                    
                    if not title or not url:
                        continue
                    
                    # 提取摘要
                    snippet = ""
                    snippet_elem = result_div.find('a', class_='result__snippet')
                    if snippet_elem:
                        snippet = snippet_elem.get_text(strip=True)
                    
                    if not snippet:
                        # 尝试其他可能的选择器
                        snippet_elem = result_div.find('div', class_='result__snippet')
                        if snippet_elem:
                            snippet = snippet_elem.get_text(strip=True)
                    
                    results.append({
                        'title': title,
                        'snippet': snippet or f'Search result from DuckDuckGo',
                        'url': url,
                        'source': 'DuckDuckGo'
                    })
                    
                except Exception as e:
                    print_debug(f"⚠️ Error parsing result {i}: {e}")
                    continue
            
            print_debug(f"✅ Extracted {len(results)} results from DuckDuckGo")
            
            # 解析和去重 URL
            if results:
                seen_urls = set()
                deduplicated_results = []
                
                for raw_result in results:
                    url = raw_result['url']
                    
                    # 过滤百度文库等
                    if 'wenku.baidu.com' in url.lower():
                        continue
                    if 'mbd.baidu.com' in url.lower():
                        continue
                    
                    # 解码 DuckDuckGo 重定向 URL
                    if 'duckduckgo.com/l/' in url.lower() and 'uddg=' in url.lower():
                        decoded_url = self._decode_duckduckgo_redirect_url(url)
                        if decoded_url != url:
                            url = decoded_url
                    
                    # 再次检查解码后的 URL
                    if 'wenku.baidu.com' in url.lower():
                        continue
                    if 'mbd.baidu.com' in url.lower():
                        continue
                    
                    # 规范化 URL 用于去重
                    normalized_url = self._normalize_url_for_dedup(url)
                    
                    # 跳过重复
                    if normalized_url in seen_urls:
                        continue
                    
                    seen_urls.add(normalized_url)
                    
                    # 清理摘要
                    cleaned_snippet = self._clean_snippet(raw_result['snippet']) if raw_result['snippet'] else f'Search result from {raw_result["source"]}'
                    
                    deduplicated_results.append({
                        'title': raw_result['title'],
                        'snippet': cleaned_snippet[:get_truncation_length()] if cleaned_snippet else f'Search result from {raw_result["source"]}',
                        'source': raw_result['source'],
                        'content': '',
                        '_internal_url': url,
                        'url': url
                    })
                
                return deduplicated_results
            
            return []
            
        except Exception as e:
            print_debug(f"⚠️ Requests-based search failed: {e}")
            return []
    
    def _download_single_webpage(self, result: Dict, global_index: int, target_url: str, search_term: str) -> Dict:
        """
        下载单个网页（用于并行下载）
        
        Args:
            result: 搜索结果字典
            global_index: 全局索引
            target_url: 目标 URL
            search_term: 搜索词
            
        Returns:
            更新后的 result 字典，包含 success 标志
        """
        try:
            # 使用 requests 下载网页
            html_content, final_url, title = self._download_webpage_with_requests(target_url, timeout=5.0)
            
            if not html_content:
                result['content'] = "Failed to download webpage"
                result['success'] = False
                return result
            
            # 从 HTML 提取文本内容
            content = self._extract_content_from_html(html_content)
            
            if not content or len(content.strip()) < 100:
                result['content'] = "Content too short or unable to extract"
                result['success'] = False
                return result
            
            # 更新标题（如果从 HTML 获取的更好）
            if title and title != "Untitled":
                result['title'] = title
            else:
                title = result.get('title', 'Untitled')
            
            # Apply LLM filtering if enabled and content exists
            if content and self.enable_llm_filtering:
                content = self._extract_relevant_content_with_llm(content, search_term, title)
            
            # 使用最终 URL（处理重定向后的）
            actual_url = final_url if final_url else target_url
            
            # 保存文件
            saved_html_path, saved_txt_path = self._save_webpage_content_from_html(
                html_content, actual_url, title, content or "", search_term
            )
            
            if saved_html_path:
                result['saved_html_path'] = saved_html_path
            if saved_txt_path:
                result['saved_txt_path'] = saved_txt_path
            
            # Clean content for better LLM processing
            cleaned_content = self._clean_text_for_saving(content) if content else ""
            result['content'] = cleaned_content if cleaned_content else (content if content else "Content too short or unable to extract")
            result['final_url'] = actual_url
            result['success'] = True
            
            return result
            
        except Exception as e:
            print_debug(f"[{global_index+1}] Error processing result: {e}")
            result['content'] = f"Processing error: {str(e)}"
            result['success'] = False
            return result

    def _fetch_webpage_content_with_timeout(self, results: List[Dict], page, timeout_seconds: int = 60) -> None:
        """
        Fetch webpage content with additional timeout control
        Uses parallel page loading: opens multiple pages at once, then processes them sequentially
        """
        start_time = time.time()
        
        # Get browser context from the page (只用于搜索，不用于下载内容)
        context = page.context
        
        # Phase 1: 准备要下载的URL列表（不使用 Playwright）
        
        urls_to_download = []  # List of (result, index, target_url)
        
        # 准备所有需要下载的 URL
        for i, result in enumerate(results):
            # Prepare URL
            target_url = result.get('_internal_url') or result.get('url', '')
            target_url = self._normalize_url(target_url)
            
            # Decode DuckDuckGo redirect URLs to get the real destination URL
            if 'duckduckgo.com/l/' in target_url.lower() and 'uddg=' in target_url.lower():
                decoded_url = self._decode_duckduckgo_redirect_url(target_url)
                if decoded_url != target_url:
                    target_url = self._normalize_url(decoded_url)
            
            # Handle Baidu redirect URLs
            if 'baidu.com/link?url=' in target_url:
                decoded_url = self._decode_baidu_redirect_url(target_url)
                if decoded_url != target_url:
                    target_url = self._normalize_url(decoded_url)
            
            # Skip problematic domains and Baidu Wenku/MBD (cannot download correct text)
            problematic_domains = [
                'douyin.com', 'tiktok.com', 'youtube.com', 'youtu.be',
                'bilibili.com', 'b23.tv', 'instagram.com', 'facebook.com',
                'twitter.com', 'x.com', 'linkedin.com',
                'wenku.baidu.com',  # 百度文库，无法下载正确文字
                'mbd.baidu.com'  # 百度移动端，通常是视频，无法下载正确文字
            ]
            if any(domain in target_url.lower() for domain in problematic_domains):
                if 'wenku.baidu.com' in target_url.lower():
                    result['content'] = "Baidu Wenku page, unable to download correct text, automatically filtered"
                elif 'mbd.baidu.com' in target_url.lower():
                    result['content'] = "Baidu mobile page (usually video), unable to download correct text, automatically filtered"
                else:
                    result['content'] = "Video or social media link, skip content fetch"
                continue
            
            if target_url.startswith(('javascript:', 'mailto:')):
                result['content'] = "Non-webpage link, skip content fetch"
                continue
            
            # 添加到下载列表
            urls_to_download.append((result, i, target_url))
        
        
        # Phase 2: 使用 requests 并行下载内容（更快更轻量）
        
        search_term = getattr(self, '_current_search_term', '')
        
        # 使用 ThreadPoolExecutor 并行下载
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import threading
        
        # 最多同时下载 5 个页面
        max_workers = min(5, len(urls_to_download))
        
        print_debug(f"🚀 Starting parallel download with {max_workers} workers for {len(urls_to_download)} pages...")
        
        # 线程安全的计数器
        valid_index_lock = threading.Lock()
        valid_index = [0]  # 使用列表以便在闭包中修改
        
        # 提交所有下载任务
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 创建 future -> (result, global_index, target_url) 的映射
            future_to_info = {}
            
            for result, global_index, target_url in urls_to_download:
                if time.time() - start_time > timeout_seconds:
                    print_current(f"⏰ Overall content fetching timeout reached ({timeout_seconds}s), stopping")
                    break
                
                future = executor.submit(
                    self._download_single_webpage,
                    result, global_index, target_url, search_term
                )
                future_to_info[future] = (result, global_index, target_url)
            
            # 处理完成的下载任务（按完成顺序）
            completed_count = 0
            
            for future in as_completed(future_to_info):
                completed_count += 1
                result, global_index, target_url = future_to_info[future]
                
                try:
                    # 获取下载结果（result 已经被更新）
                    updated_result = future.result()
                    
                    # 打印进度（线程安全）
                    if updated_result.get('success') and updated_result.get('content'):
                        if len(updated_result['content'].strip()) > 100:
                            with valid_index_lock:
                                valid_index[0] += 1
                                current_index = valid_index[0]
                            
                            title = updated_result.get('title', 'Untitled')
                            self._print_webpage_summary(current_index, title, target_url, updated_result['content'])
                    
                except Exception as e:
                    print_debug(f"[{global_index+1}] Failed to get download result: {e}")
                    result['content'] = f"Download failed: {str(e)}"
        
        print_debug(f"✅ Parallel download completed: {completed_count} pages processed, {valid_index[0]} pages with valid content")

    def _fetch_webpage_content(self, results: List[Dict], page) -> None:
        """
        Fetch actual webpage content for the search results (fallback method)
        """
        valid_index = 0  # Counter for valid results that will be printed (starts from 0, will be 1-indexed when printing)
        
        for i, result in enumerate(results):
            start_time = time.time()
            try:
                print_current(f"📖 Getting webpage content for result {i+1}: {result['title'][:50]}...")
                
                target_url = result.get('_internal_url') or result.get('url', '')
                
                # Normalize URL (add protocol if missing)
                target_url = self._normalize_url(target_url)
                
                # Handle Baidu redirect URLs with extended timeout
                is_baidu_redirect = 'baidu.com/link?url=' in target_url
                if is_baidu_redirect:
                    
                    # Try to decode the redirect URL first
                    decoded_url = self._decode_baidu_redirect_url(target_url)
                    if decoded_url != target_url:
                        target_url = self._normalize_url(decoded_url)  # Normalize decoded URL too
                        # print_current(f"🎯 Using decoded URL instead of redirect (fallback)")
                        is_baidu_redirect = False  # No longer need special handling
                
                problematic_domains = [
                    'douyin.com', 'tiktok.com',
                    'youtube.com', 'youtu.be',
                    'bilibili.com', 'b23.tv',
                    'instagram.com', 'facebook.com',
                    'twitter.com', 'x.com',
                    'linkedin.com',
                    'wenku.baidu.com',  # 百度文库，无法下载正确文字
                    'mbd.baidu.com'  # 百度移动端，通常是视频，无法下载正确文字
                ]
                
                if any(domain in target_url.lower() for domain in problematic_domains):
                    if 'wenku.baidu.com' in target_url.lower():
                        result['content'] = "Baidu Wenku page, unable to download correct text, skip content fetch"
                    elif 'mbd.baidu.com' in target_url.lower():
                        result['content'] = "Baidu mobile page (usually video), unable to download correct text, skip content fetch"
                    else:
                        # print_current(f"⏭️ Skip video/social media link: {target_url}")
                        result['content'] = "Video or social media link, skip content fetch"
                    continue
                
                # Skip ads-by pages
                if 'ads-by' in target_url.lower():
                    result['content'] = "Advertisement page, skip content fetch"
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
                    
                    # No retry - try once and fail fast if unsuccessful
                    page.goto(target_url, timeout=timeout_ms, wait_until='domcontentloaded')
                    page.wait_for_timeout(500)
                    
                    # Check for error pages
                    current_url = page.url
                    if 'chrome-error://' in current_url or 'about:blank' in current_url:
                        raise Exception(f"Redirected to error page: {current_url}")
                    
                    # Check if redirected to ads-by page
                    if 'ads-by' in current_url.lower():
                        result['content'] = "Advertisement page, skip content fetch"
                        continue
                    
                    # Success - extract content
                    content = self._extract_main_content(page)
                    title = result.get('title', 'Untitled')
                    search_term = getattr(self, '_current_search_term', '')
                    
                    # Log content extraction result
                    if not content or not content.strip():
                        print_debug(f"⚠️ No content extracted from: {title[:50]}...")
                    elif len(content.strip()) <= 100:
                        print_debug(f"⚠️ Content too short ({len(content.strip())} chars) from: {title[:50]}...")
                    
                    # Always try to save HTML file (even if text content is short)
                    # Apply LLM filtering if enabled and content exists
                    if content and self.enable_llm_filtering:
                        content = self._extract_relevant_content_with_llm(content, search_term, title)
                    
                    # Save both HTML and text content to files (HTML will be saved even if text is short)
                    saved_html_path, saved_txt_path = self._save_webpage_content(page, target_url, title, content or "", search_term)
                    if saved_html_path:
                        result['saved_html_path'] = saved_html_path
                    if saved_txt_path:
                        result['saved_txt_path'] = saved_txt_path
                    
                    # Clean content for better LLM processing
                    cleaned_content = self._clean_text_for_saving(content) if content else ""
                    result['content'] = cleaned_content if cleaned_content else (content if content else "")
                    
                    # Print webpage summary - only increment valid_index for results that are actually printed
                    if content:
                        valid_index += 1
                        self._print_webpage_summary(valid_index, title, target_url, result['content'])
                        elapsed_time = time.time() - start_time
                        # print_current(f"✅ Successfully got {len(result['content'])} characters of useful content (time: {elapsed_time:.2f}s)")
                    else:
                        print_debug(f"⚠️ No content extracted, but HTML may have been saved for: {title[:50]}...")
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
                # print_current(f"Failed to get webpage content (time: {elapsed_time:.2f}s): {e}")  # Commented out to reduce terminal noise
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
                # Medium-specific selectors (add first for priority)
                'article section[data-name="body"]',
                'article [data-testid="post-content"]',
                'article .postArticle-content',
                'article .section-content',
                'article [name="body"]',
                'article .postArticle',
                # Common article selectors
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
                    # Add timeout protection using Playwright's built-in timeout mechanism
                    # Temporarily set shorter timeout for this selector query
                    try:
                        page.set_default_timeout(3000)  # 3 seconds timeout per selector
                        elements = page.query_selector_all(selector)
                    except Exception as query_error:
                        # Playwright will raise TimeoutError if timeout occurs
                        error_msg = str(query_error)
                        if "timeout" in error_msg.lower():
                            print_debug(f"⏰ Selector '{selector}' query timeout (3s), trying next selector")
                        else:
                            print_debug(f"⚠️ Selector '{selector}' query error: {query_error}")
                        elements = None
                    finally:
                        # Restore default timeout (8 seconds as used elsewhere in the code)
                        page.set_default_timeout(8000)
                    
                    if elements:
                        for elem in elements:
                            try:
                                text = elem.text_content().strip()
                                if text and len(text) > 100:
                                    # 使用统一的特殊页面检测
                                    is_special, page_type, message = self._detect_special_page(text)
                                    if is_special:
                                        return message
                                    
                                    if self._is_quality_content(text):
                                        # 如果内容足够长（>500字符），使用它；否则继续尝试其他selector
                                        if len(text) > 500:
                                            content = text
                                            # print_current(f"✅ Successfully extracted content with selector '{selector}'")
                                            break
                                        elif not content or len(text) > len(content):
                                            # 保存当前找到的最长内容，但继续尝试其他selector
                                            content = text
                            except Exception as elem_error:
                                continue
                        # 如果找到了足够长的内容（>500字符），停止尝试其他selector
                        if content and len(content) > 500:
                            break
                except Exception as selector_error:
                    print_debug(f"⚠️ Content selector error for '{selector}': {selector_error}")
                    continue
            
            # If no content found or content is too short, try extracting from body
            # Lower threshold to 200 to be more lenient
            if not content or len(content) < 200:
                try:
                    # print_current("⚠️ Selector method found no content or content too short, trying to extract full body text")
                    body_elem = None
                    try:
                        # Add timeout protection using Playwright's built-in timeout mechanism
                        try:
                            page.set_default_timeout(3000)  # 3 seconds timeout
                            body_elem = page.query_selector('body')
                        except Exception as body_query_error:
                            error_msg = str(body_query_error)
                            if "timeout" in error_msg.lower():
                                print_debug(f"⏰ Body selector query timeout (3s)")
                            else:
                                print_debug(f"⚠️ Body selector query error: {body_query_error}")
                            body_elem = None
                        finally:
                            # Restore default timeout (8 seconds as used elsewhere in the code)
                            page.set_default_timeout(8000)
                    except Exception as body_selector_error:
                        print_debug(f"⚠️ Body selector error: {body_selector_error}")
                    
                    body_text = ""
                    if body_elem:
                        try:
                            body_text = body_elem.text_content()
                        except Exception as body_text_error:
                            print_debug(f"⚠️ Body text extraction error: {body_text_error}")
                    
                    # 使用统一的特殊页面检测
                    if body_text:
                        is_special, page_type, message = self._detect_special_page(body_text)
                        if is_special:
                            return message
                    
                    if body_text and len(body_text) > 300:
                        cleaned_body = self._clean_body_content(body_text)
                        if cleaned_body and len(cleaned_body) > 200:
                            # Use body content if it's longer than current content
                            if not content or len(cleaned_body) > len(content):
                                content = cleaned_body
                                # print_current("✅ Successfully extracted using body content")
                except Exception as body_extraction_error:
                    print_debug(f"⚠️ Body extraction error: {body_extraction_error}")
            
            if content:
                # 在后处理前使用统一的特殊页面检测
                is_special, page_type, message = self._detect_special_page(content)
                if is_special:
                    return message
                
                # Post-process extracted content to handle common issues
                content = self._post_process_extracted_content(content)
                
                # Lower threshold to 50 characters to be more lenient
                # Some pages may have valid but short content
                if len(content) < 50:
                    print_debug(f"⚠️ Content too short after post-processing ({len(content)} chars), returning empty")
                    return ""
                
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
                # For English content (like Medium articles), be more lenient
                elif len(line) > 30 and not any(nav in line.lower() for nav in ['home', 'login', 'register', 'menu', 'navigation']):
                    # If line is substantial and doesn't look like navigation, include it
                    skip_initial_nav = False
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
        # Only reject if navigation keywords are very high (>25%) and content is short
        if words_count > 0 and nav_count / words_count > 0.25 and words_count < 200:
            return False
        
        # Be more lenient with sentence structure requirement
        # For longer content (>500 chars), don't require sentence endings
        sentence_endings = text.count('。') + text.count('.') + text.count('!') + text.count('？') + text.count('?')
        news_markers = text.count('丨') + text.count('｜') + text.count('：') + text.count('——')
        
        # Only require sentence endings for medium-length content without news markers
        # Very short content (<100 chars) or very long content (>500 chars) don't need sentence endings
        if 100 < len(text) < 500 and words_count > 30 and sentence_endings == 0 and news_markers == 0:
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
            if len(line) < 3:  # More lenient minimum length (reduced from 5 to 3)
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
            # Keep meaningful English content - more lenient criteria
            elif len(line) >= 5:  # Reduced from 10 to 5, removed word count requirement
                # Allow single words if they're meaningful (like course codes, titles)
                word_count = len(line.split())
                if word_count >= 1:  # Allow single words (reduced from 2)
                    cleaned_lines.append(line)
        
        # Increased limit significantly to preserve more content (from 100 to 2000)
        # Return more lines to preserve comprehensive content
        return '\n'.join(cleaned_lines[:2000])

    def _detect_html_encoding(self, html_bytes: bytes, default_encoding: str = 'utf-8') -> str:
        """
        检测HTML内容的编码
        
        Args:
            html_bytes: HTML字节内容
            default_encoding: 默认编码
            
        Returns:
            检测到的编码名称
        """
        try:
            # 方法1: 从HTTP Content-Type头检测（如果可用）
            # 这里我们主要从HTML内容本身检测
            
            # 方法2: 从HTML meta标签检测charset
            # 检查前10KB内容（通常charset在开头）
            content_preview = html_bytes[:10240].decode('latin-1', errors='ignore')
            
            # 查找charset声明（多种格式）
            charset_patterns = [
                r'<meta[^>]*charset\s*=\s*["\']?([^"\'\s>]+)',
                r'<meta[^>]*content\s*=\s*["\'][^"\']*charset\s*=\s*([^"\'\s;]+)',
                r'charset\s*=\s*["\']?([^"\'\s>]+)',
            ]
            
            for pattern in charset_patterns:
                match = re.search(pattern, content_preview, re.IGNORECASE)
                if match:
                    detected_charset = match.group(1).strip().lower()
                    # 规范化编码名称
                    encoding_map = {
                        'gb2312': 'gbk',
                        'gb_2312': 'gbk',
                        'gb-2312': 'gbk',
                        'gbk': 'gbk',
                        'utf-8': 'utf-8',
                        'utf8': 'utf-8',
                        'big5': 'big5',
                        'big-5': 'big5',
                    }
                    detected_charset = encoding_map.get(detected_charset, detected_charset)
                    if detected_charset:
                        return detected_charset
            
            # 方法3: 使用chardet库检测（如果可用）
            try:
                import chardet
                detected = chardet.detect(html_bytes[:50000])  # 检测前50KB
                if detected and detected.get('encoding'):
                    detected_encoding = detected['encoding'].lower()
                    # 规范化编码名称
                    if detected_encoding in ['gb2312', 'gb_2312', 'gb-2312']:
                        detected_encoding = 'gbk'
                    if detected_encoding and detected.get('confidence', 0) > 0.7:
                        return detected_encoding
            except ImportError:
                pass  # chardet不可用，跳过
            
            return default_encoding
            
        except Exception as e:
            print_debug(f"⚠️ Encoding detection failed: {e}, using default {default_encoding}")
            return default_encoding
    
    def _download_webpage_with_requests(self, url: str, timeout: float = 5.0) -> tuple:
        """
        使用 requests 下载网页，返回 HTML 内容和最终 URL
        
        Args:
            url: 目标 URL
            timeout: 超时时间（秒）
            
        Returns:
            (html_content, final_url, title) 或 (None, None, None) 如果失败
        """
        try:
            import requests
            from bs4 import BeautifulSoup
            # 禁用 SSL 警告
            import urllib3
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
                'Accept-Encoding': 'gzip, deflate, br',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1'
            }
            
            response = requests.get(url, headers=headers, timeout=timeout, allow_redirects=True, verify=False)
            
            if response.status_code != 200:
                return None, None, None
            
            # 获取最终 URL（处理重定向）
            final_url = response.url
            
            # 使用字节内容而不是text，以便正确检测编码
            html_bytes = response.content
            
            # 检测编码
            detected_encoding = self._detect_html_encoding(html_bytes)
            
            # 尝试使用检测到的编码解码
            try:
                html_content = html_bytes.decode(detected_encoding)
            except (UnicodeDecodeError, LookupError):
                # 如果检测的编码失败，尝试UTF-8
                try:
                    html_content = html_bytes.decode('utf-8')
                    detected_encoding = 'utf-8'
                except UnicodeDecodeError:
                    # 如果UTF-8也失败，尝试GBK（常见中文编码）
                    try:
                        html_content = html_bytes.decode('gbk')
                        detected_encoding = 'gbk'
                    except UnicodeDecodeError:
                        # 最后尝试使用errors='replace'来避免完全失败
                        html_content = html_bytes.decode('utf-8', errors='replace')
                        detected_encoding = 'utf-8'
                        print_debug(f"⚠️ Used error replacement for encoding, some characters may be lost")
            
            # 解析 HTML 获取标题（使用正确解码的内容）
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # 提取标题
            title = None
            if soup.title:
                title = soup.title.string
            if not title:
                title = "Untitled"
            
            # 确保HTML内容使用UTF-8编码（统一编码）
            # 如果原始编码不是UTF-8，需要更新HTML中的charset声明
            if detected_encoding.lower() not in ['utf-8', 'utf8']:
                # 更新或添加charset声明
                if re.search(r'<meta[^>]*charset', html_content, re.IGNORECASE):
                    html_content = re.sub(
                        r'(<meta[^>]*charset\s*=\s*["\']?)[^"\'\s>]+',
                        r'\1utf-8',
                        html_content,
                        flags=re.IGNORECASE,
                        count=1
                    )
                elif '<head>' in html_content.lower():
                    html_content = re.sub(
                        r'(<head[^>]*>)',
                        r'\1\n<meta http-equiv="Content-Type" content="text/html; charset=UTF-8" />',
                        html_content,
                        flags=re.IGNORECASE,
                        count=1
                    )
            
            return html_content, final_url, title
            
        except Exception as e:
            print_debug(f"⚠️ Requests download failed: {e}")
            return None, None, None
    
    def _extract_content_from_html(self, html_content: str) -> str:
        """
        从 HTML 内容中提取主要文本内容
        
        Args:
            html_content: HTML 字符串（应该是UTF-8编码）
            
        Returns:
            提取的文本内容
        """
        try:
            from bs4 import BeautifulSoup
            
            # 确保html_content是字符串类型
            if isinstance(html_content, bytes):
                # 如果是字节，尝试检测编码并解码
                detected_encoding = self._detect_html_encoding(html_content)
                try:
                    html_content = html_content.decode(detected_encoding)
                except (UnicodeDecodeError, LookupError):
                    try:
                        html_content = html_content.decode('utf-8')
                    except UnicodeDecodeError:
                        html_content = html_content.decode('utf-8', errors='replace')
            
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # 移除 script 和 style 标签
            for tag in soup(['script', 'style', 'nav', 'header', 'footer', 'aside']):
                tag.decompose()
            
            # 尝试找到主要内容区域
            content_selectors = [
                'article', 'main', '.article-content', '.content', '.post-content',
                '.entry-content', '.article-body', '#content', '.main-content'
            ]
            
            content_text = ""
            for selector in content_selectors:
                if selector.startswith('.'):
                    elements = soup.find_all(class_=selector[1:])
                elif selector.startswith('#'):
                    elements = [soup.find(id=selector[1:])] if soup.find(id=selector[1:]) else []
                else:
                    elements = soup.find_all(selector)
                
                if elements:
                    for elem in elements:
                        if elem:
                            text = elem.get_text(separator='\n', strip=True)
                            if len(text) > len(content_text):
                                content_text = text
                    if len(content_text) > 500:
                        break
            
            # 如果没找到主要内容，使用 body
            if not content_text or len(content_text) < 200:
                body = soup.find('body')
                if body:
                    content_text = body.get_text(separator='\n', strip=True)
            
            return content_text
            
        except Exception as e:
            print_debug(f"⚠️ Content extraction failed: {e}")
            return ""

    def _save_webpage_content_from_html(self, html_content: str, url: str, title: str, content: str, search_term: str = "") -> tuple:
        """
        从 HTML 字符串保存网页内容（不需要 Playwright page 对象）
        
        Args:
            html_content: HTML 内容字符串
            url: 原始 URL
            title: 页面标题
            content: 提取的文本内容
            search_term: 搜索词
            
        Returns:
            Tuple of (html_filepath, txt_filepath) or empty strings if failed
        """
        # Ensure the web search result directory exists when needed
        self._ensure_result_directory()
        
        if not self.web_result_dir:
            print_current(f"⚠️ Cannot save files: web_result_dir is not set")
            return "", ""
        
        # Verify directory exists before attempting to save
        if not os.path.exists(self.web_result_dir):
            print_current(f"⚠️ Cannot save files: directory does not exist: {self.web_result_dir}")
            return "", ""
        
        # 检测特殊页面（快速检测，避免保存无用内容）
        is_special_by_url = False
        if url:
            _, page_type_by_url, message_by_url = self._detect_special_page("", title, url)
            if page_type_by_url:
                is_special_by_url = True
        
        # 如果通过 URL 检测到特殊页面，直接返回，不保存
        if is_special_by_url:
            print_debug(f"⚠️ {message_by_url}: {url[:80]}...")
            return "", ""
        
        # 规范化URL并检查是否已下载
        normalized_url = self._normalize_url_for_dedup(url) if url else ""
        if normalized_url and normalized_url in self.downloaded_urls:
            print_debug(f"⏭️ 跳过重复URL: {url[:80]}... (已下载)")
            return "", ""
        
        html_filepath = ""
        txt_filepath = ""
        
        try:
            # Generate base filename
            safe_title = re.sub(r'[^\w\s-]', '', title)[:50]
            safe_title = re.sub(r'[-\s]+', '_', safe_title)
            
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
            
            # Save HTML content
            try:
                # 确保html_content是字符串类型
                if isinstance(html_content, bytes):
                    # 如果是字节，尝试检测编码并解码
                    detected_encoding = self._detect_html_encoding(html_content)
                    try:
                        html_content = html_content.decode(detected_encoding)
                    except (UnicodeDecodeError, LookupError):
                        try:
                            html_content = html_content.decode('utf-8')
                        except UnicodeDecodeError:
                            html_content = html_content.decode('utf-8', errors='replace')
                            print_debug(f"⚠️ Used error replacement for HTML encoding")
                
                # 确保HTML中的charset声明是UTF-8
                if re.search(r'<meta[^>]*charset', html_content, re.IGNORECASE):
                    html_content = re.sub(
                        r'(<meta[^>]*charset\s*=\s*["\']?)[^"\'\s>]+',
                        r'\1utf-8',
                        html_content,
                        flags=re.IGNORECASE,
                        count=1
                    )
                elif '<head>' in html_content.lower():
                    html_content = re.sub(
                        r'(<head[^>]*>)',
                        r'\1\n<meta http-equiv="Content-Type" content="text/html; charset=UTF-8" />',
                        html_content,
                        flags=re.IGNORECASE,
                        count=1
                    )
                
                # 检测HTML中的特殊页面
                is_special, page_type, message = self._detect_special_page(html_content, title, url)
                
                if is_special:
                    print_debug(f"⚠️ {message}: {url[:80]}...")
                    return "", ""
                
                if not is_special:
                    html_filename = f"{base_filename}.html"
                    html_filepath = os.path.join(self.web_result_dir, html_filename)
                    with open(html_filepath, 'w', encoding='utf-8') as f:
                        f.write(html_content)
                    # 成功保存HTML后，记录URL以避免重复下载
                    if normalized_url:
                        self.downloaded_urls.add(normalized_url)
                    
            except Exception as e:
                print_current(f"⚠️ Failed to save webpage HTML: {e}")
            
            # Save text content
            try:
                if content and content.strip():
                    # For very large content, truncate before cleaning
                    MAX_CONTENT_LENGTH_FOR_CLEANING = 500000
                    content_to_clean = content
                    if len(content) > MAX_CONTENT_LENGTH_FOR_CLEANING:
                        print_current(f"⚠️ Content too large ({len(content)} chars), truncating to {MAX_CONTENT_LENGTH_FOR_CLEANING} chars before cleaning")
                        content_to_clean = content[:MAX_CONTENT_LENGTH_FOR_CLEANING]
                    
                    # Clean the content
                    cleaned_content = None
                    try:
                        from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
                        with ThreadPoolExecutor(max_workers=1) as executor:
                            future = executor.submit(self._clean_text_for_saving, content_to_clean)
                            try:
                                cleaned_content = future.result(timeout=10.0)
                            except FutureTimeoutError:
                                print_current(f"⚠️ Content cleaning timeout (10s), using simplified cleaning")
                                cleaned_content = self._clean_text_for_saving_simple(content_to_clean)
                    except Exception as clean_error:
                        print_current(f"⚠️ Content cleaning failed: {clean_error}, using simplified cleaning")
                        try:
                            cleaned_content = self._clean_text_for_saving_simple(content_to_clean)
                        except Exception as simple_clean_error:
                            print_current(f"⚠️ Simplified cleaning also failed: {simple_clean_error}, saving original content")
                            cleaned_content = content_to_clean
                    
                    if cleaned_content and len(cleaned_content.strip()) > 50:
                        txt_filename = f"{base_filename}.txt"
                        txt_filepath = os.path.join(self.web_result_dir, txt_filename)
                        
                        # Create a formatted text file with metadata
                        formatted_content = f"""Title: {title}
URL: {url}
Search Term: {search_term}
Timestamp: {datetime.datetime.now().isoformat()}
Original Content Length: {len(content)} characters
Cleaned Content Length: {len(cleaned_content)} characters


{cleaned_content}
"""
                        
                        try:
                            with open(txt_filepath, 'w', encoding='utf-8') as f:
                                f.write(formatted_content)
                            # 成功保存后，记录URL
                            if normalized_url:
                                self.downloaded_urls.add(normalized_url)
                        except Exception as write_error:
                            print_current(f"⚠️ Failed to write text file: {write_error}")
                            txt_filepath = ""
                            
            except Exception as e:
                print_current(f"⚠️ Failed to save text content: {e}")
                txt_filepath = ""
            
            return html_filepath, txt_filepath
            
        except Exception as e:
            print_current(f"⚠️ Failed to save webpage content: {e}")
            return "", ""
    
    def _print_webpage_summary(self, index: int, title: str, url: str, content: str) -> None:
        """
        Print a summary of the downloaded webpage (title and URL, no content preview)
        
        Args:
            index: Index of the result (1-based)
            title: Webpage title
            url: Webpage URL
            content: Webpage content (not printed, only used for validation)
        """
        try:
            # Skip if content is too short or is an error message
            if not content or len(str(content)) < 50:
                return
            
            # Skip ads-by pages
            if 'ads-by' in url.lower():
                return
            
            # Remove newlines and carriage returns from title, replace with space
            import re
            display_title = re.sub(r'[\n\r]+', ' ', str(title)).strip()
            
            # Truncate title if too long
            if len(display_title) > 100:
                display_title = display_title[:97] + '...'
            
            # Print title and URL (URL on new line)
            print_current(f"[{index}] {display_title}")
            print_current(f"    {url}")
            
        except Exception as e:
            # If summary generation fails, just print basic info
            try:
                import re
                clean_title = re.sub(r'[\n\r]+', ' ', str(title)).strip() if title else 'Untitled'
                clean_title = clean_title[:100]
                print_current(f"[{index}] {clean_title}")
                if url:
                    print_current(f"    {url}")
            except:
                pass
    
    def _clean_text_for_saving_simple(self, content: str) -> str:
        """
        Simplified text cleaning for timeout fallback - only basic operations
        """
        import re
        
        if not content or not content.strip():
            return ""
        
        # 使用统一的特殊页面检测
        is_special, page_type, message = self._detect_special_page(content)
        if is_special:
            return message
        
        # Remove HTML tags (basic)
        content = re.sub(r'<[^>]+>', '', content)
        
        # Remove obvious JSON blocks (simplified patterns only)
        content = re.sub(r'\{"@context"[^}]*\}', '', content, flags=re.DOTALL)
        content = re.sub(r'\{"__typename"[^}]{100,}\}', '', content, flags=re.DOTALL)
        
        # Remove data URIs
        content = re.sub(r'data:[^;]+;[^,]+,[^\s]+', '', content)
        
        # Basic cleanup
        content = re.sub(r'\n\s*\n\s*\n+', '\n\n', content)  # Max 2 consecutive newlines
        content = re.sub(r' {3,}', '  ', content)  # Max 2 spaces
        
        return content.strip()
    
    def _clean_text_for_saving(self, content: str) -> str:
        """
        Clean text content for saving to txt files, preserving meaningful content
        """
        import re
        
        if not content or not content.strip():
            return ""
        
        # For very large content, use simplified cleaning to avoid timeout
        MAX_CONTENT_LENGTH = 300000  # 300KB - use simplified cleaning for larger content
        if len(content) > MAX_CONTENT_LENGTH:
            print_debug(f"⚠️ Content too large ({len(content)} chars), using simplified cleaning")
            return self._clean_text_for_saving_simple(content)
        
        # 使用统一的特殊页面检测
        is_special, page_type, message = self._detect_special_page(content)
        if is_special:
            if page_type == "verification":
                print_current("⚠️ Detected verification page in cleaning, returning verification message only")
            elif page_type == "docin":
                print_debug("⚠️ Detected DocIn embedded document page in cleaning, returning message only")
            elif page_type == "baidu_wenku":
                print_debug("⚠️ Detected Baidu Wenku page in cleaning, returning message only")
            elif page_type == "baidu_mbd":
                print_debug("⚠️ Detected Baidu MBD page in cleaning, returning message only")
            elif page_type == "baidu_scholar":
                print_debug("⚠️ Detected Baidu Scholar search page in cleaning, returning message only")
            elif page_type == "duckduckgo_help":
                print_current("⚠️ Detected DuckDuckGo help/ad page in cleaning, returning message only")
            return message
        
        # Remove HTML tags
        content = re.sub(r'<[^>]+>', '', content)
        
        # Remove large JSON/GraphQL/JavaScript blocks BEFORE word separation
        # This prevents code from being split into words and then partially preserved
        
        # Remove JSON-LD blocks
        content = re.sub(r'\{"@context"[^}]*\}', '', content, flags=re.DOTALL)
        
        # Remove GraphQL cache blocks (e.g., "PremiumProductCollections_learningProduct:degree~...")
        content = re.sub(r'"[A-Za-z0-9_]+~[A-Za-z0-9_]+"\s*:\s*\{[^}]*"__typename"[^}]*\}', '', content, flags=re.DOTALL)
        
        # Remove large JSON objects with __typename (GraphQL responses)
        content = re.sub(r'\{"__typename"[^}]{100,}\}', '', content, flags=re.DOTALL)
        
        # Remove JSON objects containing common browser/system properties (often in JSON data)
        # Pattern: {"system":"...","platform":"...","browser":{...}}
        content = re.sub(r'\{"system"\s*:\s*"[^"]*"\s*,\s*"platform"\s*:\s*"[^"]*"[^}]*\}', '', content, flags=re.DOTALL)
        content = re.sub(r'\{"browser"\s*:\s*\{[^}]*"name"\s*:\s*"[^"]*"[^}]*\}', '', content, flags=re.DOTALL)
        
        # Remove JSON objects with "userData", "appName", "NaptimeStore" etc. (Apollo cache)
        content = re.sub(r'\{"userData"\s*:\s*\{[^}]*\}', '', content, flags=re.DOTALL)
        content = re.sub(r'\{"appName"\s*:\s*"[^"]*"\s*\}', '', content)
        content = re.sub(r'\{"NaptimeStore"\s*:\s*\{[^}]*\}', '', content, flags=re.DOTALL)
        
        # Remove JSON objects with "responseCache", "elementsToUrlMapping" etc.
        content = re.sub(r'\{"responseCache"\s*:\s*[^}]*\}', '', content, flags=re.DOTALL)
        content = re.sub(r'\{"elementsToUrlMapping"\s*:\s*[^}]*\}', '', content, flags=re.DOTALL)
        
        # Remove JavaScript code blocks - more comprehensive patterns
        content = re.sub(r'if\s*\(typeof\s+\w+\s*===\s*[\'"]undefined[\'"]\)\s*return[^;]*;', '', content, flags=re.IGNORECASE)
        content = re.sub(r'window\.\w+\s*=\s*window\.\w+\s*\|\|\s*\[\];', '', content)
        content = re.sub(r'window\.\w+\.push\([^)]+\);', '', content)
        content = re.sub(r'window\.localStorage\.(setItem|getItem)\([^)]+\);', '', content)
        content = re.sub(r'\baddEventListener\s*\([^)]+\);?', '', content, flags=re.IGNORECASE)  # More specific pattern
        content = re.sub(r'console\.(log|error|warn|debug)\([^)]*\);', '', content)
        content = re.sub(r'Sentry\.(browserTracingIntegration|setTag|init)\([^)]*\);?', '', content, flags=re.IGNORECASE | re.DOTALL)
        content = re.sub(r'dataLayer\.push\(\{[^}]*\}\);?', '', content, flags=re.DOTALL)
        
        # Remove lines that are mostly JavaScript/JSON (before word separation)
        # This catches patterns like "addEventListener" that might not be caught above
        lines_before_sep = content.split('\n')
        filtered_lines = []
        for line in lines_before_sep:
            line_stripped = line.strip()
            # Skip lines that are clearly code
            if (re.search(r'\b(addEventListener|typeof|window\.|console\.|Sentry\.|dataLayer\.)', line_stripped, re.IGNORECASE) and
                (line_stripped.count('(') + line_stripped.count(')') + line_stripped.count(';')) > 2):
                continue
            # Skip lines that are mostly JSON structure
            if len(line_stripped) > 30:
                json_chars = line_stripped.count('"') + line_stripped.count(':') + line_stripped.count('{') + line_stripped.count('}')
                if json_chars > len(line_stripped) * 0.3:  # More than 30% JSON characters
                    text_chars = len(re.findall(r'[A-Za-z\s]', line_stripped))
                    if text_chars < len(line_stripped) * 0.2:  # Less than 20% text
                        continue
            filtered_lines.append(line)
        content = '\n'.join(filtered_lines)
        
        # Fix words stuck together after removing HTML tags
        # Rule 1: Add space between lowercase letter followed by uppercase letter (e.g., "ForIndividuals" -> "For Individuals")
        # This handles most cases like "ForIndividuals", "BusinessesFor", "FreeMIT"
        content = re.sub(r'([a-z])([A-Z])', r'\1 \2', content)
        
        # Rule 2: Add space between uppercase letter followed by uppercase letter and then lowercase (e.g., "MITMedia" -> "MIT Media")
        # This handles acronyms followed by words like "MITMedia", "AILearning"
        content = re.sub(r'([A-Z])([A-Z][a-z])', r'\1 \2', content)
        
        # Rule 3: Add space between digit and letter (e.g., "2024MIT" -> "2024 MIT")
        content = re.sub(r'(\d)([A-Za-z])', r'\1 \2', content)
        
        # Rule 4: Add space between letter and digit when it starts a new word (e.g., "MIT2024" -> "MIT 2024")
        # But be careful not to break things like "CS101" or "Room123"
        # Only add space if the digit is followed by more digits or a space
        content = re.sub(r'([A-Za-z])(\d{4,})', r'\1 \2', content)  # Years and long numbers
        
        # Remove CSS blocks (IMPROVED: More precise to avoid deleting legitimate braces)
        # Only remove blocks that contain CSS properties (colon+semicolon pattern)
        # This preserves mathematical sets {1,2,3}, code parameters {x,y}, etc.
        content = re.sub(r'\{[^{}]*:[^{}]*;[^{}]*\}', '', content, flags=re.DOTALL)
        
        # Additional: Remove empty braces that might be left over
        content = re.sub(r'\{\s*\}', '', content)
        
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
        
        # Remove URLs and data strings (IMPROVED: Replace with placeholder instead of deleting)
        # This preserves the information that a link existed, useful for references, citations, etc.
        # Preserve important academic/technical domains
        important_domains = r'(?:doi\.org|arxiv\.org|github\.com|wikipedia\.org|scholar\.google)'
        
        # First, protect important domain URLs by replacing them temporarily
        protected_urls = []
        def protect_url(match):
            url = match.group(0)
            if re.search(important_domains, url, re.IGNORECASE):
                protected_urls.append(url)
                return f'__PROTECTED_URL_{len(protected_urls)-1}__'
            return match.group(0)
        
        content = re.sub(r'https?://[^\s]+', protect_url, content)
        
        # Replace remaining URLs with placeholder
        content = re.sub(r'https?://[^\s]+', '[链接]', content)
        
        # Restore protected URLs
        for i, url in enumerate(protected_urls):
            content = content.replace(f'__PROTECTED_URL_{i}__', url)
        
        # Remove data URIs (these are typically inline images)
        content = re.sub(r'data:[^;]+;[^,]+,[^\s]+', '', content)
        
        # Remove basic CSS properties
        content = re.sub(r'-webkit-[^:]+:[^;]+;?', '', content, flags=re.IGNORECASE)
        content = re.sub(r'-moz-[^:]+:[^;]+;?', '', content, flags=re.IGNORECASE)
        content = re.sub(r'-ms-[^:]+:[^;]+;?', '', content, flags=re.IGNORECASE)
        
        # Generic CSS property:value patterns (IMPROVED: More precise to avoid deleting metadata)
        # Only match typical CSS value patterns, not arbitrary text
        # CSS values are usually: numbers+units, colors, or specific keywords
        css_value_patterns = [
            # Number + unit (but only common CSS properties to avoid false positives)
            r'(?:width|height|margin|padding|font-size|line-height|border|top|left|right|bottom|max-width|min-width|max-height|min-height)\s*:\s*\d+(?:px|em|rem|vh|vw|pt|cm|mm|in|pc|ex|ch|vmin|vmax)\s*[;}]',
            # Percentage for CSS properties only
            r'(?:width|height|margin|padding|opacity|font-size|line-height|top|left|right|bottom)\s*:\s*\d+%\s*[;}]',
            r'\w+\s*:\s*#[0-9a-fA-F]{3,8}\s*[;}]',  # Hex color
            r'\w+\s*:\s*rgba?\([^)]+\)\s*[;}]',  # RGB/RGBA color
            r'\w+\s*:\s*hsla?\([^)]+\)\s*[;}]',  # HSL/HSLA color
            r'\w+\s*:\s*(?:none|auto|inherit|initial|unset|normal|bold|italic|solid|dashed|dotted|hidden|visible|flex|grid|block|inline|absolute|relative|fixed|sticky|left|right|center|top|bottom|middle)\s*[;}]',  # Common CSS keywords
        ]
        
        for pattern in css_value_patterns:
            content = re.sub(pattern, '', content, flags=re.IGNORECASE)
        
        # DO NOT use the generic '\w+\s*:\s*[^;{}\n]+[;}]' pattern as it's too broad
        # This preserves important metadata like "作者: 张三;", "时间: 2024年;", etc.
        
        # Process line by line with more lenient filtering for news content
        # Note: Large JSON/JS blocks have already been removed before word separation
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
            
            # Skip lines with excessive JSON/GraphQL data (e.g., Apollo cache, GraphQL responses)
            # These typically contain patterns like "__typename", "PremiumProductCollections", etc.
            json_indicators = [
                r'"__typename"\s*:',  # GraphQL typename
                r'PremiumProductCollections',  # Specific GraphQL cache keys
                r'[A-Za-z0-9_]+~[A-Za-z0-9_]+"\s*:\s*\{',  # GraphQL cache key patterns
                r'\{"__ref"\s*:',  # Apollo cache references
                r'"__typename"[^}]*"__ref"',  # Apollo cache patterns
            ]
            if any(re.search(pattern, line) for pattern in json_indicators):
                continue
            
            # Skip lines that are mostly JSON structure (multiple quotes, colons, braces)
            # Count JSON-like characters
            json_chars = line.count('"') + line.count(':') + line.count('{') + line.count('}') + line.count(',')
            if len(line) > 30 and json_chars > len(line) * 0.25:  # More than 25% JSON characters (lowered threshold)
                # But allow if it contains meaningful text (more than 30% letters/spaces, raised threshold)
                text_chars = len(re.findall(r'[A-Za-z\s]', line))
                if text_chars < len(line) * 0.3:  # Less than 30% text (raised threshold)
                    continue
            
            # Skip lines that start with JSON-like patterns (even if incomplete)
            # These are often fragments of JSON objects
            if re.match(r'^[,{]\s*"[^"]+"\s*:\s*"[^"]*"', line):  # Starts with ,{"key":"value" or {"key":"value"
                # Check if it's mostly JSON structure
                if json_chars > len(line) * 0.2:  # More than 20% JSON characters
                    continue
            
            # Skip lines containing JSON-like key-value pairs with common technical keys
            json_key_patterns = [
                r'"system"\s*:\s*"[^"]*"',
                r'"platform"\s*:\s*"[^"]*"',
                r'"browser"\s*:\s*\{',
                r'"userData"\s*:\s*\{',
                r'"appName"\s*:\s*"[^"]*"',
                r'"NaptimeStore"\s*:\s*\{',
                r'"responseCache"\s*:\s*',
                r'"elementsToUrlMapping"\s*:\s*',
                r'"isMobileBrowser"\s*:\s*(true|false)',
                r'"isAndroid"\s*:\s*(true|false)',
                r'"isMobile"\s*:\s*(true|false)',
                r'"isIOS"\s*:\s*(true|false)',
            ]
            if any(re.search(pattern, line, re.IGNORECASE) for pattern in json_key_patterns):
                # If line contains these patterns and has high JSON character density, skip it
                if json_chars > len(line) * 0.2:  # More than 20% JSON characters
                    continue
            
            # Skip lines that look like JavaScript code fragments (after word separation)
            # These patterns catch code that was split by word separation rules
            js_fragment_patterns = [
                r'\b(sentry|config|json|parse|typeof|window|dataLayer|console|addEventListener)\s+(config|json|parse|typeof|window|dataLayer|console|addEventListener)',  # Split variable names
                r'\bJSON\.parse\s*\(',
                r'\btypeof\s+\w+\s*===',
                r'\bnew\s+RegExp\s*\(',
                r'\breturn\s+\w+\.(elements|test|push)\s*',
                r'\bif\s*\([^)]*\.(indexOf|test|push)',  # if statements with method calls
                r'\b\w+\s*=\s*\{[^}]*\}\s*;',  # Object assignments ending with semicolon
                r'/\*\s*(globals|eslint)',  # Comment markers
                r'\b\w+\s*=\s*\([^)]*\)\s*;',  # Function assignments ending with semicolon
            ]
            if any(re.search(pattern, line, re.IGNORECASE) for pattern in js_fragment_patterns):
                # Check if line has high code density
                code_density = (line.count('(') + line.count(')') + line.count(';') + line.count('{') + line.count('}') + line.count('=')) / max(len(line), 1)
                if code_density > 0.1:  # More than 10% code characters
                    continue
            
            # Skip lines containing code-like patterns with high code character density
            # This catches fragments like "if (sentry Config && sentry Config.public Dsn)"
            if re.search(r'\bif\s*\([^)]*&&', line, re.IGNORECASE):  # if statements with &&
                code_density = (line.count('(') + line.count(')') + line.count(';') + line.count('{') + line.count('}') + line.count('=') + line.count('&&') + line.count('.')) / max(len(line), 1)
                if code_density > 0.1:  # More than 10% code characters
                    continue
            
            # Skip lines that contain "Config" or "Config." followed by property access (common in JS code fragments)
            if re.search(r'\b\w+\s+Config\s*\.', line, re.IGNORECASE):  # e.g., "sentry Config.public"
                code_density = (line.count('(') + line.count(')') + line.count(';') + line.count('{') + line.count('}') + line.count('=') + line.count('.')) / max(len(line), 1)
                if code_density > 0.08:  # More than 8% code characters (lowered threshold)
                    continue
            
            # Skip lines containing common JavaScript library patterns (after word separation)
            # These patterns catch code that was split by word separation rules
            js_library_patterns = [
                r'\bgform\s*\.',  # Gravity Forms
                r'\bj\s+Query\s*\(',  # jQuery (split)
                r'\bset\s+Timeout\s*\(',  # setTimeout (split)
                r'\bclear\s+Timeout\s*\(',  # clearTimeout (split)
                r'\bwindow\s*\[\s*[\'"]',  # window['...']
                r'\bsession\s+Storage\s*\.',  # sessionStorage (split)
                r'\bJSON\s*\.\s*stringify',  # JSON.stringify (split)
                r'\bobserver\s*\.',  # observer.
                r'\bdocument\s*\.\s*body',  # document.body (split)
                r'\bvisibility\s+Test\s+Div',  # visibilityTestDiv (split)
                r'\bgform\s+Wrapper\s+Div',  # gformWrapperDiv (split)
                r'\btrigger\s+Post\s+Render',  # triggerPostRender (split)
                r'\bpost\s+Render\s+Fired',  # postRenderFired (split)
                r'\bwp\s*\.\s*i18\s+n',  # wp.i18n (split)
                r'\bsource\s+URL\s*=',  # sourceURL (split)
                r'\bNREUM\s*\|\|',  # NREUM||
                r'\beverything\s+Except\s+Flag',  # everythingExceptFlag (split)
            ]
            if any(re.search(pattern, line, re.IGNORECASE) for pattern in js_library_patterns):
                code_density = (line.count('(') + line.count(')') + line.count(';') + line.count('{') + line.count('}') + line.count('=') + line.count('.') + line.count('[') + line.count(']') + line.count('||') + line.count('&&')) / max(len(line), 1)
                if code_density > 0.05:  # More than 5% code characters
                    continue
            
            # Skip lines that look like code fragments with parentheses and property access
            # Pattern: word space word dot word (e.g., "sentry Config.public Dsn")
            if re.search(r'\b\w+\s+\w+\s*\.\s*\w+', line):  # Property access pattern
                code_density = (line.count('(') + line.count(')') + line.count(';') + line.count('{') + line.count('}') + line.count('=') + line.count('.') + line.count('&&')) / max(len(line), 1)
                if code_density > 0.1:  # More than 10% code characters
                    continue
            
            # Skip short lines that contain code-like patterns (common in code fragments)
            # These are often remnants of JavaScript code after word separation
            if len(line) < 100:  # Short lines
                code_indicators = [
                    r'\b(sentry|config|json|parse|typeof|window|dataLayer|console)\s+(config|json|parse|typeof|window|dataLayer|console)',  # Split variable names
                    r'\bJSON\.parse',
                    r'\bnew\s+RegExp',
                    r'\breturn\s+\w+\.',
                    r'/\*\s*(globals|eslint)',
                    r'\bis\s+New\s+Visit',  # Split variable names like "isNewVisit"
                    r'\breturn\s+arkose',  # Code fragments
                    r'\breturn\s+epic',  # Code fragments
                    r'\ballow\s+Urls',  # Split variable names like "allowUrls"
                ]
                if any(re.search(pattern, line, re.IGNORECASE) for pattern in code_indicators):
                    # Check code density
                    code_density = (line.count('(') + line.count(')') + line.count(';') + line.count('{') + line.count('}') + line.count('=') + line.count('.') + line.count('&&') + line.count('[') + line.count(']') + line.count('||')) / max(len(line), 1)
                    if code_density > 0.08:  # More than 8% code characters
                        continue
                
                # Skip lines with multiple code indicators (return, ||, ;, /*, etc.)
                code_keywords = ['return', '||', ';', '/*', '*/', 'new', 'RegExp', 'epic', 'Response', 'elements', 'arkose', 'Traces', 'Pattern', 'globals', 'eslint']
                keyword_count = sum(1 for keyword in code_keywords if keyword.lower() in line.lower())
                if keyword_count >= 1:  # At least 1 code keyword (lowered threshold)
                    code_density = (line.count('(') + line.count(')') + line.count(';') + line.count('{') + line.count('}') + line.count('=') + line.count('.') + line.count('&&') + line.count('[') + line.count(']') + line.count('||')) / max(len(line), 1)
                    if code_density > 0.05:  # More than 5% code characters (lowered threshold)
                        continue
                
                # Skip lines that are clearly code comments or fragments
                if re.search(r'/\*\s*(globals|eslint)', line, re.IGNORECASE) or line.strip().startswith('/*') or line.strip().endswith('*/'):
                    continue
                
                # Skip lines starting with //# (source map comments)
                if line.strip().startswith('//#'):
                    continue
                
                # Skip lines containing sourceURL patterns (even if split)
                if re.search(r'source\s+URL\s*=', line, re.IGNORECASE):
                    continue
                
                # Skip lines ending with code operators (&&, ||, ;, etc.)
                if re.search(r'[&|;]\s*$', line):  # Ends with &&, ||, or ;
                    code_density = (line.count('(') + line.count(')') + line.count(';') + line.count('{') + line.count('}') + line.count('=') + line.count('.') + line.count('&&') + line.count('[') + line.count(']') + line.count('||')) / max(len(line), 1)
                    if code_density > 0.05:  # More than 5% code characters
                        continue
            
            # Skip lines ending with code-like patterns (e.g., "}';", ");", etc.)
            if re.search(r'[{}();]\s*[\'"]?\s*$', line):  # Ends with code characters
                code_density = (line.count('(') + line.count(')') + line.count(';') + line.count('{') + line.count('}')) / max(len(line), 1)
                if code_density > 0.12:  # More than 12% code characters
                    continue
            
            # Skip lines with excessive escaped Unicode (e.g., \u002F, \u002F)
            unicode_escapes = len(re.findall(r'\\u[0-9a-fA-F]{4}', line))
            if unicode_escapes > 3:  # More than 3 Unicode escapes
                continue
            
            # Skip lines that look like JavaScript code (containing common JS patterns)
            js_patterns = [
                r'typeof\s+\w+\s*===',  # typeof checks
                r'window\.\w+\s*=',  # window assignments
                r'\.push\(',  # array.push
                r'\.setItem\(',  # localStorage.setItem
                r'\.getItem\(',  # localStorage.getItem
                r'addEventListener\(',  # event listeners
                r'querySelector',  # DOM queries
                r'dataLayer\.push',  # Google Tag Manager
                r'Sentry\.',  # Sentry error tracking
                r'console\.(log|error|warn)',  # console statements
                r'JSON\.parse',  # JSON parsing
                r'new RegExp',  # RegExp creation
                r'return\s+[^;]+;',  # return statements (if line is mostly code)
            ]
            # Only skip if line is mostly code (has JS patterns AND high code density)
            if any(re.search(pattern, line) for pattern in js_patterns):
                code_density = (line.count('(') + line.count(')') + line.count(';') + line.count('{') + line.count('}')) / max(len(line), 1)
                if code_density > 0.15:  # More than 15% code characters
                    continue
            
            # Skip CSS-like lines but be more specific (IMPROVED: Only skip real CSS, not metadata)
            # Only skip if it looks like actual CSS (has CSS-specific value patterns)
            if re.search(r':\s*[^;]+;', line):
                # Check if it's actual CSS (contains CSS-specific patterns)
                is_css = any([
                    re.search(r':\s*\d+(?:px|em|rem|%|vh|vw|pt|cm|mm)', line),  # CSS units
                    re.search(r':\s*#[0-9a-fA-F]{3,8}', line),  # Hex colors
                    re.search(r':\s*rgba?\(', line),  # RGB colors
                    re.search(r':\s*hsla?\(', line),  # HSL colors
                    # CSS color keywords (expanded list)
                    re.search(r':\s*(?:red|blue|green|white|black|gray|grey|yellow|orange|purple|pink|brown|cyan|magenta|lime|navy|teal|olive|maroon|aqua|fuchsia|silver)\s*[;}]', line, re.IGNORECASE),
                    # CSS common keywords
                    re.search(r':\s*(?:none|auto|inherit|initial|flex|grid|block|inline|absolute|relative|fixed|hidden|visible|bold|italic|normal|solid|dashed|dotted)\s*[;}]', line, re.IGNORECASE),
                ])
                # If it's CSS, skip it
                # If it's not CSS (e.g., metadata like "作者: 张三;"), keep it
                if is_css:
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
            # English content - much more lenient criteria to preserve course codes, titles, etc.
            elif word_count >= 1:  # Reduced from 2 to 1 to allow single words
                has_sentence_structure = any(punct in line for punct in '.!?。！？')
                has_meaningful_words = len(re.findall(r'\b[a-zA-Z]{2,}\b', line)) >= 1
                # Allow course codes like "CS221", "248A", etc.
                has_course_code = bool(re.search(r'\b[A-Z]{2,}\s*\d+[A-Z]?\b', line))
                # Allow lines with numbers (course numbers, dates, etc.)
                has_numbers = bool(re.search(r'\d+', line))
                
                # More lenient: keep if it has sentence structure, meaningful words, course codes, numbers, or reasonable length
                if (has_sentence_structure or has_meaningful_words or has_course_code or 
                    has_numbers or len(line) >= 10):  # Reduced from 15 to 10
                    cleaned_lines.append(line)
            # Keep even shorter content if it looks meaningful (course codes, single important words)
            elif len(line) >= 3:  # Reduced from 5 to 3
                # Allow course codes, numbers, or lines with letters
                has_course_code = bool(re.search(r'\b[A-Z]{2,}\s*\d+[A-Z]?\b', line))
                has_numbers = bool(re.search(r'\d+', line))
                has_letters = bool(re.search(r'[A-Za-z]', line))
                
                if has_course_code or (has_numbers and has_letters) or (has_letters and len(line) >= 5):
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
            print_current("Playwright is not installed or not available")
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
                    
                    # 使用移动版 User Agent 获取更轻量的页面
                    context = browser.new_context(
                        user_agent='Mozilla/5.0 (iPhone; CPU iPhone OS 16_6 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.6 Mobile/15E148 Safari/604.1',
                        viewport={'width': 414, 'height': 896},  # iPhone 尺寸
                        ignore_https_errors=True,
                        java_script_enabled=True,
                        bypass_csp=True
                    )
                finally:
                    # Restore original DISPLAY if it existed
                    if original_display is not None:
                        os.environ['DISPLAY'] = original_display
                page = context.new_page()
                
                # 🚀 性能优化：阻止加载图片、CSS、字体、媒体等资源
                def block_resources(route):
                    """阻止加载非必要资源以加快页面加载速度"""
                    resource_type = route.request.resource_type
                    if resource_type in ['image', 'stylesheet', 'font', 'media', 'other']:
                        route.abort()
                    else:
                        route.continue_()
                
                page.route('**/*', block_resources)
                print_debug("🚀 Performance optimization: blocking images, CSS, fonts, and media")
                
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
                # Minimum 200 characters to save content
                if content and len(content.strip()) > 200:
                    saved_html_path, saved_txt_path = self._save_webpage_content(page, url, title, content, search_term or "")
                
                final_url = page.url
                
                browser.close()
                
                # Clean content for better LLM processing
                cleaned_content = self._clean_text_for_saving(content)
                
                # Check total txt files in web_search_result directory
                total_txt_files = self._count_txt_files_in_result_dir()
                
                result_data = {
                    'title': title,
                    'url': url,
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

    def _normalize_url(self, url: str) -> str:
        """
        Normalize URL by adding protocol if missing
        
        Args:
            url: URL string that may be missing protocol
            
        Returns:
            Normalized URL with protocol
        """
        if not url:
            return url
        
        # If URL starts with //, add https://
        if url.startswith('//'):
            url = 'https:' + url
        # If URL doesn't start with http:// or https://, add https://
        elif not url.startswith(('http://', 'https://')):
            # Only add https:// if it looks like a domain (contains a dot)
            if '.' in url and not url.startswith(('javascript:', 'mailto:', 'data:')):
                url = 'https://' + url
        
        return url
    
    def _decode_duckduckgo_redirect_url(self, ddg_url: str) -> str:
        """
        Decode DuckDuckGo redirect URL to get the real destination URL
        
        Args:
            ddg_url: DuckDuckGo redirect URL (format: https://duckduckgo.com/l/?uddg=...)
            
        Returns:
            Real destination URL, or original URL if decoding fails
        """
        try:
            import urllib.parse
            
            # DuckDuckGo URL format: https://duckduckgo.com/l/?uddg=encoded_url
            if 'duckduckgo.com/l/' not in ddg_url.lower() or 'uddg=' not in ddg_url.lower():
                return ddg_url
            
            # Parse the URL
            parsed = urllib.parse.urlparse(ddg_url)
            query_params = urllib.parse.parse_qs(parsed.query)
            
            # Get the uddg parameter
            if 'uddg' in query_params:
                encoded_url = query_params['uddg'][0]
                # Decode the URL
                decoded_url = urllib.parse.unquote(encoded_url)
                return decoded_url
            
            return ddg_url
            
        except Exception as e:
            print_debug(f"⚠️ Failed to decode DuckDuckGo redirect URL: {e}")
            return ddg_url
    
    def _normalize_url_for_dedup(self, url: str) -> str:
        """
        Normalize URL for deduplication by removing query parameters, fragments, and tracking parameters
        This helps identify duplicate URLs even if they have different tracking parameters
        
        Args:
            url: URL string
            
        Returns:
            Normalized URL without query parameters and fragments
        """
        if not url:
            return ""
        
        import urllib.parse
        
        # First, decode DuckDuckGo redirect URLs to get the real destination URL
        # This is important because DuckDuckGo redirect links all look similar
        # but point to different destinations
        if 'duckduckgo.com/l/' in url.lower() and 'uddg=' in url.lower():
            real_url = self._decode_duckduckgo_redirect_url(url)
            if real_url != url:
                # Use the real destination URL for deduplication
                url = real_url
        
        # Also decode Baidu redirect URLs
        # IMPORTANT: If Baidu redirect URL decoding fails, we must keep the full URL
        # (including query parameters) for deduplication, because different query parameters
        # may point to different destination URLs
        is_baidu_redirect = 'baidu.com/link?url=' in url.lower()
        baidu_decode_failed = False
        
        if is_baidu_redirect:
            original_url = url
            real_url = self._decode_baidu_redirect_url(url)
            if real_url != url:
                url = real_url
            else:
                # Decoding failed - keep original URL with full query parameters
                baidu_decode_failed = True
                url = original_url
        
        # First normalize the URL (add protocol if missing)
        normalized = self._normalize_url(url)
        
        try:
            # Parse the URL
            parsed = urllib.parse.urlparse(normalized)
            
            # For Baidu redirect URLs that failed to decode, keep the full URL including query
            # This ensures different Baidu redirect links are not incorrectly identified as duplicates
            if baidu_decode_failed:
                # Keep the full URL (including query parameters) for Baidu redirects that failed to decode
                return normalized.lower()
            
            # Remove query parameters and fragments for deduplication
            # This helps catch duplicates like:
            # - https://example.com/page?utm_source=google
            # - https://example.com/page?utm_source=bing
            # - https://example.com/page#section
            
            # Reconstruct URL without query and fragment
            dedup_url = urllib.parse.urlunparse((
                parsed.scheme,
                parsed.netloc,
                parsed.path,
                parsed.params,  # Keep params (rarely used)
                '',  # Remove query
                ''   # Remove fragment
            ))
            
            # Remove trailing slash for consistency (unless it's the root path)
            if dedup_url.endswith('/') and parsed.path != '/':
                dedup_url = dedup_url.rstrip('/')
            
            return dedup_url.lower()  # Use lowercase for case-insensitive comparison
            
        except Exception as e:
            # If parsing fails, return normalized URL as-is
            print_debug(f"⚠️ URL normalization failed for {url[:80]}: {e}")
            return normalized.lower()
    
    def _optimize_search_term(self, search_term: str) -> str:
        """
        Optimize search terms, especially for time-related searches and academic identifiers
        """
        import datetime
        import re
        
        # Check for DOI pattern (e.g., 10.1145/3712003)
        doi_pattern = r'\b10\.\d{4,}/[^\s]+\b'
        doi_match = re.search(doi_pattern, search_term)
        if doi_match:
            doi = doi_match.group(0)
            print_debug(f"🔬 Detected DOI: {doi}, using quoted search for exact match")
            # For DOI searches, wrap the DOI in quotes for exact matching
            # Keep other keywords outside quotes
            search_term_without_doi = search_term.replace(doi, '').strip()
            if search_term_without_doi:
                return f'"{doi}" {search_term_without_doi}'
            else:
                return f'"{doi}"'
        
        # Check for arXiv ID pattern (e.g., 2301.12345 or arXiv:2301.12345)
        arxiv_pattern = r'\b(?:arXiv:)?(\d{4}\.\d{4,5})\b'
        arxiv_match = re.search(arxiv_pattern, search_term, re.IGNORECASE)
        if arxiv_match:
            arxiv_id = arxiv_match.group(0)
            print_debug(f"📚 Detected arXiv ID: {arxiv_id}, using quoted search for exact match")
            search_term_without_arxiv = search_term.replace(arxiv_id, '').strip()
            if search_term_without_arxiv:
                return f'"{arxiv_id}" {search_term_without_arxiv}'
            else:
                return f'"{arxiv_id}"'
        
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
            print_current("Playwright is not installed or not available")
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
                # Google Images added directly without connectivity check
                print_debug("🔍 Using Google -> Baidu -> Bing search order for images")
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
                        # Skip this engine if it has failed before
                        if engine['name'] in self.failed_engines:
                            print_debug(f"⏭️ Skipping {engine['name']} (failed in previous attempt)")
                            continue
                        
                        print_debug(f"🔍 Attempting to use {engine['name']} for image search...")
                        
                        # Visit search page with improved waiting strategy
                        # Use 5 seconds timeout for Google Images, 20 seconds for Baidu Images (Baidu needs more time)
                        try:
                            if engine['name'] == 'Baidu Images':
                                engine_timeout = 20000  # 20 seconds for Baidu Images (no timeout limit)
                            elif engine['name'] == 'Google Images':
                                engine_timeout = 5000  # 5 seconds for Google Images
                            else:
                                engine_timeout = 6000  # 6 seconds for other engines
                            page.goto(engine['url'], timeout=engine_timeout, wait_until='domcontentloaded')
                            # Wait for page to stabilize
                            page.wait_for_timeout(500)
                            # Try to wait for images to load
                            try:
                                page.wait_for_selector('img', timeout=1000)
                            except:
                                pass  # Continue even if no images found
                        except Exception as page_error:
                            print_debug(f"⚠️ Page loading error for {engine['name']}: {page_error}")
                            # Mark this engine as failed
                            self.failed_engines.add(engine['name'])
                            print_debug(f"🚫 {engine['name']} marked as failed, will be skipped in future attempts")
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
                                            print_debug(f"Download error for image {i+1}: {download_error}")
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
                            print_debug(f"{engine['name']} found no valid images")
                            
                    except Exception as e:
                        print_debug(f"{engine['name']} search failed: {e}")
                        # Mark this engine as failed so it won't be retried
                        self.failed_engines.add(engine['name'])
                        print_debug(f"🚫 {engine['name']} marked as failed, will be skipped in future attempts")
                        continue
                
                browser.close()
                
                if not image_found:
                    result_data.update({
                        'error': 'No valid images found',
                        'suggestion': 'Please try using more specific search keywords, or check your network connection'
                    })
                    print_current(f"Image search failed: {query}")
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
            print_debug(f"Error extracting Google Images metadata: {e}")
            
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
                    print_debug(f"Fallback selector also failed: {fallback_error}")
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
            print_debug(f"Error extracting images from {engine['name']}: {e}")
            
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