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

Mermaid Chart Processor Utility

This module provides functionality to process markdown files containing Mermaid charts,
convert them to SVG images using multiple methods (CLI, Playwright, Python library, or online API),
and then convert SVG to PNG using the enhanced SVG to PNG converter.
"""

import re
import os
import subprocess
import tempfile
import requests
import base64
import hashlib
import urllib.request
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

# 导入ForeignObject转换工具
try:
    from src.utils.foreign_object_converter import convert_mermaid_foreign_objects, has_foreign_objects
    FOREIGN_OBJECT_CONVERTER_AVAILABLE = True
except ImportError:
    try:
        from ..utils.foreign_object_converter import convert_mermaid_foreign_objects, has_foreign_objects
        FOREIGN_OBJECT_CONVERTER_AVAILABLE = True
    except ImportError:
        FOREIGN_OBJECT_CONVERTER_AVAILABLE = False
        print("⚠️ ForeignObject converter not available")

# 导入PNG裁剪工具
try:
    from src.utils.png_cropper import PNGCropper
    PNG_CROPPER_AVAILABLE = True
except ImportError:
    try:
        from ..utils.png_cropper import PNGCropper
        PNG_CROPPER_AVAILABLE = True
    except ImportError:
        PNG_CROPPER_AVAILABLE = False
        print("⚠️ PNG cropper not available")

# 导入HSL颜色转换工具
try:
    from src.utils.hsl_color_converter import convert_svg_hsl_colors_optimized
    HSL_CONVERTER_AVAILABLE = True
except ImportError:
    try:
        from ..utils.hsl_color_converter import convert_svg_hsl_colors_optimized
        HSL_CONVERTER_AVAILABLE = True
    except ImportError:
        HSL_CONVERTER_AVAILABLE = False
        print("⚠️ HSL color converter not available")

from .print_system import print_current, print_system, print_debug

# Import the enhanced SVG to PNG converter
try:
    from .svg_to_png import EnhancedSVGToPNGConverter
    SVG_TO_PNG_AVAILABLE = True
except ImportError:
    # Fallback: try to import from temp directory
    try:
        import sys
        sys.path.append('temp')
        from svg_to_png import EnhancedSVGToPNGConverter
        SVG_TO_PNG_AVAILABLE = True
    except ImportError:
        SVG_TO_PNG_AVAILABLE = False
        print_debug("⚠️ Enhanced SVG to PNG converter not available")

# Check for multiple mermaid rendering methods
def _check_playwright():
    """Check if playwright is available for rendering"""
    try:
        import playwright
        return True
    except ImportError:
        return False

def _ensure_local_mermaid_library():
    """
    Ensure local Mermaid library is available.
    Downloads if not present.
    
    Returns:
        Path to local mermaid.min.js file
    """
    # Create assets directory if it doesn't exist
    assets_dir = Path(__file__).parent.parent.parent / "assets" / "mermaid"
    assets_dir.mkdir(parents=True, exist_ok=True)
    
    mermaid_js_path = assets_dir / "mermaid.min.js"
    
    # Check if local file exists
    if mermaid_js_path.exists():
        print_debug(f"✅ Local Mermaid library found: {mermaid_js_path}")
        return mermaid_js_path
    
    # Download Mermaid library
    print_debug(f"📥 Downloading Mermaid library...")
    mermaid_url = "https://unpkg.com/mermaid@10/dist/mermaid.min.js"
    
    try:
        # Download the file
        urllib.request.urlretrieve(mermaid_url, mermaid_js_path)
        
        # Verify file was downloaded successfully
        if mermaid_js_path.exists() and mermaid_js_path.stat().st_size > 0:
            print_debug(f"✅ Mermaid library downloaded successfully: {mermaid_js_path}")
            return mermaid_js_path
        else:
            print_debug(f"❌ Failed to download Mermaid library")
            return None
            
    except Exception as e:
        print_debug(f"❌ Error downloading Mermaid library: {e}")
        return None

# Check available methods
PLAYWRIGHT_AVAILABLE = _check_playwright()

def _is_error_svg(svg_path: Path) -> bool:
    """
    Check if the generated SVG contains error indicators.
    
    Args:
        svg_path: Path to the SVG file
        
    Returns:
        True if SVG contains errors, False otherwise
    """
    try:
        if not svg_path.exists() or svg_path.stat().st_size == 0:
            return True
            
        with open(svg_path, 'r', encoding='utf-8') as f:
            svg_content = f.read()
        
        # Check for common error indicators in mermaid-generated SVGs
        # Look for actual error content, not just CSS class definitions
        error_indicators = [
            'Syntax error in text',
            'Parse error on line',
            'Error parsing',
            '<text class="error-text"',  # Actual error text elements
            '<path class="error-icon"'   # Actual error icon elements
        ]
        
        # If SVG contains any error indicators, it's an error SVG
        for indicator in error_indicators:
            if indicator in svg_content:
                return True
                
        # Check if SVG contains only error content and no actual chart elements
        if ('mermaid version' in svg_content and 
            'text-anchor: middle;">Syntax error' in svg_content):
            return True
                
        # Additional check: if SVG is very small (< 500 chars) it might be malformed
        if len(svg_content.strip()) < 500:
            return True
            
        return False
        
    except Exception as e:
        print_debug(f"❌ Error checking SVG content: {e}")
        return True  # Assume error if we can't read the file

def _generate_smart_filename(mermaid_code: str, following_content: str = "", fallback_index: int = 1) -> str:
    """
    Generate a smart filename for mermaid charts based on figure caption comment or content hash.
    
    Args:
        mermaid_code: The mermaid code content
        following_content: Content following the mermaid block (to extract figure caption)
        fallback_index: Index to use if no caption found and hash fails
        
    Returns:
        A clean filename without extension
    """
    def sanitize_filename(name: str) -> str:
        """Sanitize filename by removing invalid characters but keeping Chinese characters"""
        # Remove or replace invalid characters for filenames
        # Keep Chinese characters, letters, numbers, spaces, hyphens, and underscores
        import re
        # First, replace common problematic characters
        name = name.replace('/', '_').replace('\\', '_').replace(':', '_')
        name = name.replace('*', '_').replace('?', '_').replace('"', '_')
        name = name.replace('<', '_').replace('>', '_').replace('|', '_')
        
        # Remove any other non-printable or problematic characters but keep Chinese
        # This regex keeps: Chinese characters, letters, numbers, spaces, hyphens, underscores, dots
        name = re.sub(r'[^\u4e00-\u9fff\w\s\-\.]', '_', name)
        
        # Replace multiple spaces/underscores with single underscore
        name = re.sub(r'[\s_]+', '_', name)
        
        # Remove leading/trailing underscores and dots
        name = name.strip('_.')
        
        # Limit length to reasonable size (100 characters)
        if len(name) > 100:
            name = name[:100]
        
        return name
    
    def extract_caption_from_comment(content: str) -> Optional[str]:
        """Extract figure caption from comment following mermaid block"""
        # Look for the figure caption comment pattern: <!-- the_figure_caption -->
        # The content after mermaid block might contain: <!-- Loan approval decision tree -->
        caption_match = re.search(r'<!--\s*([^-]+?)\s*-->', content.strip(), re.IGNORECASE)
        if caption_match:
            caption = caption_match.group(1).strip()
            # Filter out common system comments that shouldn't be used as captions
            system_comments = ['the_figure_caption', 'Available formats', 'Source code file']
            if not any(sys_comment in caption for sys_comment in system_comments):
                return caption
        return None
    
    try:
        # First, try to extract caption from comment following mermaid block
        caption = extract_caption_from_comment(following_content)
        
        if caption:
            sanitized_caption = sanitize_filename(caption)
            if sanitized_caption and len(sanitized_caption) >= 2:  # At least 2 characters
                return sanitized_caption
        
        # If no valid caption found, generate SHA256 hash
        hash_object = hashlib.sha256(mermaid_code.encode('utf-8'))
        hash_hex = hash_object.hexdigest()
        # Use first 16 characters of hash for reasonable filename length
        return f"mermaid_sha{hash_hex[:16]}"
        
    except Exception as e:
        print_debug(f"⚠️ Error generating smart filename: {e}")
        # Fallback to old naming scheme
        return f"mermaid_{fallback_index}"

# Determine best available method
PREFERRED_METHOD = "playwright" if PLAYWRIGHT_AVAILABLE else "none"


class MermaidProcessor:
    """
    Processor for converting Mermaid charts in markdown files to images using multiple methods.
    
    Supports the following rendering methods (in order of preference):
    1. Mermaid CLI (mmdc) with default theme - requires npm install -g @mermaid-js/mermaid-cli
    2. Playwright - requires pip install playwright && playwright install chromium
    3. Mermaid CLI (mmdc) with neutral theme - fallback using same CLI tool
    4. Online API fallback - uses mermaid.ink (requires internet connection)
    """
    
    def __init__(self, silent_init: bool = False):
        """Initialize the Mermaid processor."""
        self.preferred_method = PREFERRED_METHOD
        self.mermaid_available = (PLAYWRIGHT_AVAILABLE)
        
        # 初始化PNG裁剪器
        if PNG_CROPPER_AVAILABLE:
            self.png_cropper = PNGCropper()
        else:
            self.png_cropper = None
        
        # Initialize SVG to PNG converter
        if SVG_TO_PNG_AVAILABLE:
            self.svg_to_png_converter = EnhancedSVGToPNGConverter()
        else:
            self.svg_to_png_converter = None
        
        # 浏览器实例缓存，用于性能优化
        self._browser = None
        self._page = None
        self._browser_initialized = False
    
    def _init_browser(self):
        """初始化浏览器实例（如果还没有初始化）"""
        if not self._browser_initialized and PLAYWRIGHT_AVAILABLE:
            try:
                from playwright.sync_api import sync_playwright
                playwright = sync_playwright().start()
                self._browser = playwright.chromium.launch(
                    headless=True,
                    args=[
                        '--no-sandbox',
                        '--disable-dev-shm-usage',
                        '--disable-gpu',
                        '--disable-web-security',
                        '--disable-features=VizDisplayCompositor',
                        '--disable-extensions',
                        '--disable-plugins',
                        '--disable-images',
                        '--disable-javascript-harmony-shipping',
                        '--disable-background-timer-throttling',
                        '--disable-backgrounding-occluded-windows',
                        '--disable-renderer-backgrounding',
                        '--disable-field-trial-config',
                        '--disable-ipc-flooding-protection',
                        '--memory-pressure-off',
                        '--max_old_space_size=4096'
                    ]
                )
                # 创建高分辨率页面上下文
                self._context = self._browser.new_context(
                    device_scale_factor=2.0,  # 2倍像素密度
                    viewport={"width": 1200, "height": 800}
                )
                self._page = self._context.new_page()
                
                # 设置页面优化选项
                self._page.set_extra_http_headers({
                    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                    'Accept-Language': 'en-US,en;q=0.5',
                    'Accept-Encoding': 'gzip, deflate',
                    'Connection': 'keep-alive',
                })
                
                # 禁用不必要的功能（但保留 SVG 和字体）
                self._page.route("**/*.{png,jpg,jpeg,gif,woff,woff2,ttf,eot}", lambda route: route.abort())
                self._page.route("**/analytics/**", lambda route: route.abort())
                self._page.route("**/tracking/**", lambda route: route.abort())
                
                self._browser_initialized = True
                print(f"🔧 Browser initialized for reuse")
            except Exception as e:
                print(f"⚠️ Failed to initialize browser: {e}")
                self._browser_initialized = False
    
    def _cleanup_browser(self):
        """清理浏览器实例"""
        if self._browser_initialized:
            try:
                if self._page:
                    self._page.close()
                if self._browser:
                    self._browser.close()
                self._browser = None
                self._page = None
                self._browser_initialized = False
                print(f"🔧 Browser cleaned up")
            except Exception as e:
                print(f"⚠️ Error cleaning up browser: {e}")
    
    def __del__(self):
        """析构函数，确保浏览器被清理"""
        self._cleanup_browser()
    
    
    def process_markdown_file(self, md_file_path: str, output_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Process markdown file to convert Mermaid charts to images.
        
        Args:
            md_file_path: Path to the markdown file
            output_dir: Output directory for images (optional)
            
        Returns:
            Dictionary with processing results
        """
        try:
            #print_current(f"🎨 Processing Mermaid charts in: {md_file_path}")
            
            md_path = Path(md_file_path).absolute()
            md_dir = md_path.parent
            
            # Check if this is a plan.md file (case-insensitive)
            is_plan_file = md_path.name.lower() == 'plan.md'
            
            # Skip processing for plan.md files - keep mermaid source code as-is
            if is_plan_file:
                print_debug(f"📝 Skipping Mermaid processing for plan.md (keeping source code)")
                return {
                    'status': 'skipped',
                    'file': md_file_path,
                    'message': 'Skipped processing for plan.md - mermaid source code preserved',
                    'charts_found': 0,
                    'charts_processed': 0
                }
            
            # Use markdown file directory if no output dir specified
            if not output_dir:
                output_dir = md_dir
            else:
                output_dir = Path(output_dir)
            
            # Create images directory
            img_dir = output_dir / "images"
            img_dir.mkdir(exist_ok=True)
            
            # Read markdown file
            with open(md_file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Extract all Mermaid code blocks (enhanced to handle incomplete blocks)
            content_modified = False
            
            # First, try to find complete Mermaid blocks
            complete_pattern = re.compile(r'```mermaid\n(.*?)\n```', re.DOTALL)
            complete_matches = list(complete_pattern.finditer(content))
            
            # Then, find incomplete Mermaid blocks (missing closing ```)
            # Use a more precise pattern: look for ```mermaid that is NOT followed by a closing ``` 
            # We'll find all ```mermaid blocks and then filter out the complete ones
            incomplete_matches = []
            
            # Split content by ```mermaid to find potential incomplete blocks
            mermaid_starts = []
            start_idx = 0
            while True:
                idx = content.find('```mermaid\n', start_idx)
                if idx == -1:
                    break
                mermaid_starts.append(idx)
                start_idx = idx + 1
            
            # For each ```mermaid start, check if it has a proper closing ```
            for start_idx in mermaid_starts:
                # Check if this start position is already part of a complete match
                is_part_of_complete = False
                for complete_match in complete_matches:
                    if (start_idx >= complete_match.start() and 
                        start_idx < complete_match.end()):
                        is_part_of_complete = True
                        break
                
                if not is_part_of_complete:
                    # Find the content after ```mermaid\n
                    content_start = start_idx + len('```mermaid\n')
                    # Look for the closing ``` after this position
                    remaining_content = content[content_start:]
                    closing_idx = remaining_content.find('\n```')
                    
                    if closing_idx == -1:
                        # No closing ``` found, this is incomplete
                        # Find where the Mermaid content actually ends
                        # Look for the first double newline (paragraph break) or end of meaningful content
                        lines = remaining_content.split('\n')
                        mermaid_lines = []
                        
                        for i, line in enumerate(lines):
                            line_stripped = line.strip()
                            if not line_stripped:
                                # Empty line - check if this is end of mermaid content
                                # Look ahead to see if there's non-mermaid content
                                rest_lines = lines[i+1:]
                                has_non_mermaid_content = False
                                for future_line in rest_lines:
                                    future_stripped = future_line.strip()
                                    if future_stripped and not future_stripped.startswith('```'):
                                        # This looks like regular markdown content, not mermaid
                                        has_non_mermaid_content = True
                                        break
                                
                                if has_non_mermaid_content:
                                    # Stop here, this empty line separates mermaid from regular content
                                    break
                                else:
                                    # Include this empty line as it might be part of mermaid formatting
                                    mermaid_lines.append(line)
                            else:
                                mermaid_lines.append(line)
                        
                        # Join the mermaid content and calculate end position
                        mermaid_content = '\n'.join(mermaid_lines).rstrip()
                        mermaid_char_count = len(mermaid_content)
                        if mermaid_content.endswith('\n'):
                            end_pos = content_start + mermaid_char_count
                        else:
                            # Add one more character for the newline that should come after mermaid content
                            end_pos = content_start + mermaid_char_count + 1
                        
                        block_content = mermaid_content
                        
                        if block_content:  # Only consider it incomplete if there's actual content
                            # Create a match-like object for compatibility
                            class IncompleteMatch:
                                def __init__(self, start, end, content):
                                    self._start = start
                                    self._end = end
                                    self._content = content
                                
                                def start(self):
                                    return self._start
                                
                                def end(self):
                                    return self._end
                                
                                def group(self, num):
                                    if num == 1:
                                        return self._content
                                    return None
                            
                            incomplete_matches.append(IncompleteMatch(start_idx, end_pos, block_content))
            
            # Auto-fix incomplete Mermaid blocks
            if incomplete_matches:
                print_debug(f"🔧 Found {len(incomplete_matches)} incomplete Mermaid block(s), auto-fixing...")
                content_backup = content
                
                # Process incomplete matches from end to beginning to avoid position shifts
                for match in reversed(incomplete_matches):
                    code = match.group(1).strip()
                    if code:  # Only fix if there's actual content
                        # Replace the incomplete block with a complete one
                        start_pos = match.start()
                        end_pos = match.end()
                        
                        # Create properly formatted Mermaid block
                        replacement = f"```mermaid\n{code}\n```"
                        content = content[:start_pos] + replacement + content[end_pos:]
                        content_modified = True
                        print_debug(f"✅ Auto-fixed incomplete Mermaid block")
                
                # Write the corrected content back to file if modifications were made
                if content_modified:
                    with open(md_file_path, 'w', encoding='utf-8') as f:
                        f.write(content)
                    print_debug(f"📝 Auto-corrected {len(incomplete_matches)} incomplete Mermaid block(s) in file")
            
            # Now extract all Mermaid code blocks with optional following caption from the (potentially corrected) content
            # Match: ```mermaid\n{code}\n``` followed by optional whitespace, newline, and comment
            # The comment can be on the same line or on a new line after ```
            pattern = re.compile(r'```mermaid\n(.*?)\n```(\s*(?:\n\s*)?<!--[^>]*-->)?', re.DOTALL)
            matches = list(pattern.finditer(content))
            
            if not matches:
                print_debug("📝 No Mermaid charts found")
                return {
                    'status': 'success',
                    'file': md_file_path,
                    'charts_found': 0,
                    'charts_processed': 0,
                    'message': 'No Mermaid charts found',
                    'auto_fixed': len(incomplete_matches) if incomplete_matches else 0
                }
            
            print_debug(f"📊 Found {len(matches)} Mermaid chart(s)")
            
            processed_count = 0
            
            # Prepare batch processing tasks
            mermaid_tasks = []
            chart_info = []  # Store chart information for later processing
            
            for i, match in enumerate(reversed(matches)):
                try:
                    code = match.group(1).strip()
                    following_comment = match.group(2) if match.group(2) else ""
                    
                    # Generate smart filename based on caption from comment or hash
                    base_filename = _generate_smart_filename(code, following_comment, len(matches)-i)
                    
                    # Generate image filenames for both SVG and PNG formats
                    svg_name = f"{base_filename}.svg"
                    png_name = f"{base_filename}.png"
                    svg_path = img_dir / svg_name
                    png_path = img_dir / png_name
                    rel_svg_path = f"images/{svg_name}"
                    rel_png_path = f"images/{png_name}"
                    
                    # Generate corresponding mermaid code filename
                    mermaid_code_name = f"{base_filename}.mmd"
                    mermaid_code_path = img_dir / mermaid_code_name
                    rel_mermaid_path = f"images/{mermaid_code_name}"
                    
                    # Save original Mermaid code to separate file
                    try:
                        with open(mermaid_code_path, 'w', encoding='utf-8') as f:
                            f.write(code)
                    except Exception as e:
                        print_debug(f"❌ Failed to save Mermaid code: {e}")
                    
                    # Add to batch processing tasks
                    mermaid_tasks.append((code, svg_path, png_path))
                    
                    # Store chart information for later processing
                    chart_info.append({
                        'match': match,
                        'base_filename': base_filename,
                        'svg_name': svg_name,
                        'png_name': png_name,
                        'svg_path': svg_path,
                        'png_path': png_path,
                        'rel_svg_path': rel_svg_path,
                        'rel_png_path': rel_png_path,
                        'rel_mermaid_path': rel_mermaid_path,
                        'following_comment': following_comment,
                        'index': len(matches)-i
                    })
                    
                except Exception as e:
                    print_debug(f"❌ Error preparing Mermaid chart: {e}")
            
            # Batch generate all images using a single browser instance
            if mermaid_tasks:
                batch_results = self._generate_mermaid_images_batch(mermaid_tasks)
            else:
                batch_results = []
            
            # Process results and update content
            for i, (chart_data, (svg_success, png_success)) in enumerate(zip(chart_info, batch_results)):
                try:
                    # Check if the generated SVG contains errors
                    if svg_success:
                        is_error_svg = _is_error_svg(chart_data['svg_path'])
                        if is_error_svg:
                            print_debug(f"❌ Generated SVG contains errors, treating as failed")
                            svg_success = False
                            png_success = False
                    
                    # If SVG generation successful, replace content
                    if svg_success:
                        # Generate appropriate alt text based on caption from comment
                        def extract_caption_from_comment_for_alt(content: str) -> Optional[str]:
                            if not content:
                                return None
                            # Strip whitespace and newlines, then search for comment
                            content_cleaned = content.strip()
                            caption_match = re.search(r'<!--\s*([^-]+?)\s*-->', content_cleaned, re.IGNORECASE | re.DOTALL)
                            if caption_match:
                                caption = caption_match.group(1).strip()
                                # Filter out system comments and placeholder text
                                system_comments = ['the_figure_caption', 'Available formats', 'Source code file']
                                # Also filter out placeholder patterns like "the_diagram_caption", "the_example_diagram_caption"
                                placeholder_patterns = ['the_diagram_caption', 'the_example_diagram_caption', 'the_figure_caption']
                                if caption and not any(sys_comment in caption for sys_comment in system_comments):
                                    # Check if it's a placeholder (contains "the_" and "caption")
                                    if 'the_' in caption.lower() and 'caption' in caption.lower():
                                        # It's a placeholder, don't use it
                                        return None
                                    return caption
                            return None
                        
                        caption = extract_caption_from_comment_for_alt(chart_data['following_comment'])
                        if caption:
                            alt_text = caption
                        elif chart_data['base_filename'].startswith('mermaid_sha'):
                            # For SHA-based filenames, use empty string when no caption is provided
                            alt_text = ""
                        else:
                            # For named files, use the filename as title
                            alt_text = chart_data['base_filename'].replace('_', ' ').title()
                        
                        # Use SVG for display in markdown if available, otherwise use PNG
                        display_path = chart_data['rel_svg_path'] if svg_success else chart_data['rel_png_path']
                        format_info = ""
                        if svg_success and png_success:
                            format_info = f"<!-- Available formats: PNG={chart_data['rel_png_path']}, SVG={chart_data['rel_svg_path']} -->\n"
                        elif svg_success:
                            format_info = f"<!-- Available formats: SVG={chart_data['rel_svg_path']} -->\n"
                        
                        # Get complete Mermaid code block positions
                        start_pos = chart_data['match'].start()
                        end_pos = chart_data['match'].end()
                        
                        # Replace mermaid code block with image reference
                        replacement = f"\n![{alt_text}]({display_path})\n\n{format_info}<!-- Source code file: {chart_data['rel_mermaid_path']} -->\n"
                        
                        # Replace original content
                        content = content[:start_pos] + replacement + content[end_pos:]
                        
                        #if svg_success and png_success:
                        #    print_debug(f"✅ Successfully generated: {chart_data['svg_name']} and {chart_data['png_name']}")
                        #elif svg_success:
                        #    print_debug(f"✅ Successfully generated: {chart_data['svg_name']}")
                        processed_count += 1
                    else:
                        # Mermaid compilation failed, replace with error comment
                        print_debug(f"❌ Mermaid compilation failed, replacing with error comment")
                        
                        # Extract caption for error message
                        def extract_caption_from_comment_for_error(content: str) -> Optional[str]:
                            caption_match = re.search(r'<!--\s*([^-]+?)\s*-->', content.strip(), re.IGNORECASE)
                            if caption_match:
                                caption = caption_match.group(1).strip()
                                system_comments = ['the_figure_caption', 'Available formats', 'Source code file']
                                if not any(sys_comment in caption for sys_comment in system_comments):
                                    return caption
                            return None
                        
                        caption = extract_caption_from_comment_for_error(chart_data['following_comment'])
                        if caption:
                            error_replacement = f"\n<!-- ❌ Mermaid chart compilation failed: {caption} -->\n<!-- Source code file: {chart_data['rel_mermaid_path']} -->\n"
                        else:
                            error_replacement = f"\n<!-- ❌ Mermaid chart compilation failed (Figure {chart_data['index']}) -->\n<!-- Source code file: {chart_data['rel_mermaid_path']} -->\n"
                        
                        # Get complete Mermaid code block positions
                        start_pos = chart_data['match'].start()
                        end_pos = chart_data['match'].end()
                        
                        # Replace original content with error comment
                        content = content[:start_pos] + error_replacement + content[end_pos:]
                        
                except Exception as e:
                    print_debug(f"❌ Error processing Mermaid chart result: {e}")
            
            # Write updated file
            with open(md_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            #print_debug(f"✅ Processing complete. Modified file saved to: {md_path}")
            #print_debug(f"📁 Generated images saved in: {img_dir}")
            
            return {
                'status': 'success',
                'file': md_file_path,
                'charts_found': len(matches),
                'charts_processed': processed_count,
                'images_dir': str(img_dir),
                'message': f'Successfully processed {processed_count}/{len(matches)} Mermaid charts'
            }
            
        except Exception as e:
            print_debug(f"❌ Error processing markdown file: {e}")
            return {
                'status': 'failed',
                'file': md_file_path,
                'error': str(e),
                'message': f'Failed to process markdown file: {e}'
            }
    
    def _generate_mermaid_image(self, mermaid_code: str, svg_path: Path, png_path: Path = None) -> Tuple[bool, bool]:
        """
        Generate Mermaid SVG and PNG images using the best available method.
        
        Args:
            mermaid_code: Mermaid chart code
            svg_path: Output SVG image path
            png_path: Output PNG image path (optional)
            
        Returns:
            Tuple of (svg_success, png_success)
        """
        # Check if Playwright is available before proceeding
        if not PLAYWRIGHT_AVAILABLE:
            print(f"❌ Playwright not available")
            return False, False
        
        # 直接使用 Playwright 方法
        return self._generate_mermaid_image_playwright(mermaid_code, svg_path, png_path)
    
    def _generate_mermaid_images_batch(self, mermaid_tasks: List[tuple]) -> List[tuple]:
        """
        Generate multiple Mermaid images using a single browser instance and single page for better performance.
        
        Args:
            mermaid_tasks: List of tuples (mermaid_code, svg_path, png_path)
            
        Returns:
            List of tuples (svg_success, png_success) for each task
        """
        if not PLAYWRIGHT_AVAILABLE:
            print(f"❌ Playwright not available")
            return [(False, False)] * len(mermaid_tasks)
        
        if not mermaid_tasks:
            return []
        
        results = []
        
        try:
            import tempfile
            from playwright.sync_api import sync_playwright
            
            # Get local Mermaid library path
            local_mermaid_path = _ensure_local_mermaid_library()
            if local_mermaid_path:
                mermaid_script_src = f"file://{local_mermaid_path}"
                print_debug(f"🔧 Using local Mermaid library: {local_mermaid_path}")
            else:
                mermaid_script_src = "https://unpkg.com/mermaid@10/dist/mermaid.min.js"
                print_debug(f"⚠️ Using remote Mermaid library as fallback")
            
            # Create HTML with all mermaid charts
            mermaid_divs = []
            for i, (mermaid_code, _, _) in enumerate(mermaid_tasks):
                mermaid_divs.append(f'<div class="mermaid" id="chart_{i}">{mermaid_code}</div>')
            
            html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Mermaid Charts Batch</title>
    <style>
        body {{
            margin: 0;
            padding: 10px;
            background: white;
            font-family: "Microsoft YaHei", "SimHei", "SimSun", "Arial", sans-serif;
        }}
        .mermaid {{
            background: white;
            text-align: center;
            margin-bottom: 20px;
            page-break-inside: avoid;
        }}
        .mermaid svg {{
            max-width: 100%;
            height: auto;
        }}
    </style>
</head>
<body>
    {chr(10).join(mermaid_divs)}
    <script src="{mermaid_script_src}"></script>
    <script>
        mermaid.initialize({{
            startOnLoad: true,
            theme: 'default',
            fontFamily: 'Microsoft YaHei, SimHei, SimSun, Arial, sans-serif',
            flowchart: {{
                useMaxWidth: false,
                htmlLabels: true,
                curve: 'basis',
                padding: 15,
                nodeSpacing: 30,
                rankSpacing: 40,
                diagramPadding: 20
            }},
            gantt: {{
                useMaxWidth: false,
                leftPadding: 75,
                rightPadding: 50
            }},
            sequence: {{
                useMaxWidth: false,
                diagramMarginX: 30,
                diagramMarginY: 30,
                boxMargin: 10,
                boxTextMargin: 5,
                noteMargin: 10,
                messageMargin: 35
            }},
            journey: {{
                useMaxWidth: false,
                diagramMarginX: 30,
                diagramMarginY: 30
            }},
            timeline: {{
                useMaxWidth: false,
                diagramMarginX: 30,
                diagramMarginY: 30
            }},
            securityLevel: 'loose',
            maxWidth: 700
        }});
    </script>
</body>
</html>"""
            
            with sync_playwright() as p:
                browser = p.chromium.launch(
                    headless=True,
                    args=[
                        '--no-sandbox',
                        '--disable-dev-shm-usage',
                        '--disable-gpu',
                        '--disable-web-security',
                        '--disable-extensions',
                        '--disable-plugins',
                        '--disable-images',
                        '--disable-background-timer-throttling',
                        '--disable-backgrounding-occluded-windows',
                        '--disable-renderer-backgrounding',
                        '--memory-pressure-off',
                        '--max_old_space_size=4096'
                    ]
                )
                # 创建高分辨率页面上下文
                context = browser.new_context(
                    device_scale_factor=2.0,  # 2倍像素密度
                    viewport={"width": 1200, "height": 800}
                )
                page = context.new_page()
                
                # Create temporary HTML file with all charts
                with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False, encoding='utf-8') as temp_file:
                    temp_file.write(html_content)
                    temp_html_path = temp_file.name
                
                try:
                    # Load HTML file with all charts
                    file_url = f"file://{temp_html_path}"
                    page.goto(file_url, wait_until="domcontentloaded", timeout=5000)
                    
                    # Wait for all SVG elements to appear and have content
                    try:
                        # Wait for all SVG elements to appear
                        page.wait_for_selector(".mermaid svg", timeout=5000)
                        # Wait for all SVGs to have actual content
                        page.wait_for_function(
                            "Array.from(document.querySelectorAll('.mermaid svg')).every(svg => svg.innerHTML.length > 0)",
                            timeout=5000
                        )
                    except Exception as e:
                        print(f"⚠️ Waiting for SVG content failed: {e}")
                        # Fallback: just wait a bit
                        page.wait_for_timeout(500)
                    
                    # Process each chart
                    for i, (mermaid_code, svg_path, png_path) in enumerate(mermaid_tasks):
                        try:
                            
                            svg_success = False
                            png_success = False
                            
                            # Extract SVG content for this specific chart
                            svg_element = page.locator(f"#chart_{i} svg").first
                            if svg_element:
                                svg_content = svg_element.evaluate("el => el.outerHTML")
                                if svg_content:
                                    # Fix XML issues: convert <br> to <br/> for proper XML formatting
                                    svg_content = svg_content.replace('<br>', '<br/>')
                                    
                                    # Add XML declaration
                                    full_svg = f'<?xml version="1.0" encoding="UTF-8"?>\n{svg_content}'
                                    
                                    # 转换HSL颜色为标准RGB颜色（如果可用）
                                    if HSL_CONVERTER_AVAILABLE:
                                        try:
                                            converted_svg = convert_svg_hsl_colors_optimized(full_svg)
                                            if converted_svg != full_svg:
                                                print_debug(f"🎨 Converted HSL colors to RGB for better compatibility")
                                                full_svg = converted_svg
                                        except Exception as e:
                                            print_debug(f"⚠️ HSL color conversion failed: {e}")
                                    
                                    # 转换foreignObject为原生SVG text元素（如果可用）
                                    if FOREIGN_OBJECT_CONVERTER_AVAILABLE and has_foreign_objects(full_svg):
                                        try:
                                            converted_svg = convert_mermaid_foreign_objects(full_svg)
                                            if converted_svg != full_svg:
                                                print_debug(f"🔧 Converted foreignObject elements to native SVG text for better PDF compatibility")
                                                full_svg = converted_svg
                                        except Exception as e:
                                            print_debug(f"⚠️ ForeignObject conversion failed: {e}")
                                    
                                    # Save SVG
                                    with open(svg_path, 'w', encoding='utf-8') as f:
                                        f.write(full_svg)
                                    svg_success = True
                            
                            # Generate PNG if requested
                            if png_path and svg_success:
                                try:
                                    mermaid_element = page.locator(f"#chart_{i}").first
                                    if mermaid_element:
                                        mermaid_element.screenshot(
                                            type="png",
                                            path=str(png_path),
                                            omit_background=True
                                        )
                                        
                                        # 自动裁剪PNG图片，去除空白区域
                                        if self.png_cropper:
                                            try:
                                                self.png_cropper.crop_png(png_path, padding=15, verbose=False)
                                            except Exception:
                                                pass  # 静默处理裁剪错误
                                        
                                        png_success = True
                                except Exception as e:
                                    print(f"⚠️ PNG screenshot failed for chart {i+1}: {e}")
                            
                            results.append((svg_success, png_success))
                            
                        except Exception as e:
                            print(f"❌ Error processing chart {i+1}: {e}")
                            results.append((False, False))
                
                finally:
                    # Clean up temporary file
                    try:
                        import os
                        os.unlink(temp_html_path)
                    except OSError:
                        pass
                
                browser.close()
                
            return results
            
        except Exception as e:
            print(f"❌ Single-page batch processing failed: {e}")
            import traceback
            traceback.print_exc()
            return [(False, False)] * len(mermaid_tasks)
    

    
    def _generate_mermaid_image_playwright(self, mermaid_code: str, svg_path: Path, png_path: Path = None) -> Tuple[bool, bool]:
        """
        Generate Mermaid SVG and PNG images using Playwright browser automation.
        
        Args:
            mermaid_code: Mermaid chart code
            svg_path: Output SVG image path
            png_path: Output PNG image path (optional)
            
        Returns:
            Tuple of (svg_success, png_success)
        """
        if not PLAYWRIGHT_AVAILABLE:
            print(f"❌ Playwright not available")
            return False, False
            
        try:
            import tempfile
            from playwright.sync_api import sync_playwright
            
            print(f"🌐 Using Playwright to generate image...")
            
            # Get local Mermaid library path
            local_mermaid_path = _ensure_local_mermaid_library()
            if local_mermaid_path:
                mermaid_script_src = f"file://{local_mermaid_path}"
                print_debug(f"🔧 Using local Mermaid library: {local_mermaid_path}")
            else:
                mermaid_script_src = "https://unpkg.com/mermaid@10/dist/mermaid.min.js"
                print_debug(f"⚠️ Using remote Mermaid library as fallback")
            
            # Create HTML with Mermaid chart - 优化版本
            html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Mermaid Chart</title>
    <style>
        body {{
            margin: 0;
            padding: 10px;
            background: white;
            font-family: "Microsoft YaHei", "SimHei", "SimSun", "Arial", sans-serif;
        }}
        .mermaid {{
            background: white;
            text-align: center;
            max-width: 700px;
            margin: 0 auto;
        }}
        .mermaid svg {{
            max-width: 100%;
            height: auto;
        }}
    </style>
</head>
<body>
    <div class="mermaid">{mermaid_code}</div>
    <script src="{mermaid_script_src}"></script>
    <script>
        mermaid.initialize({{
            startOnLoad: true,
            theme: 'default',
            fontFamily: 'Microsoft YaHei, SimHei, SimSun, Arial, sans-serif',
            flowchart: {{
                useMaxWidth: false,
                htmlLabels: true,
                curve: 'basis',
                padding: 15,
                nodeSpacing: 30,
                rankSpacing: 40,
                diagramPadding: 20
            }},
            gantt: {{
                useMaxWidth: false,
                leftPadding: 75,
                rightPadding: 50
            }},
            sequence: {{
                useMaxWidth: false,
                diagramMarginX: 30,
                diagramMarginY: 30,
                boxMargin: 10,
                boxTextMargin: 5,
                noteMargin: 10,
                messageMargin: 35
            }},
            journey: {{
                useMaxWidth: false,
                diagramMarginX: 30,
                diagramMarginY: 30
            }},
            timeline: {{
                useMaxWidth: false,
                diagramMarginX: 30,
                diagramMarginY: 30
            }},
            securityLevel: 'loose',
            maxWidth: 700
        }});
    </script>
</body>
</html>"""
            
            # Create temporary HTML file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False, encoding='utf-8') as temp_file:
                temp_file.write(html_content)
                temp_html_path = temp_file.name
            
            try:
                print(f"🔧 Launching browser...")
                with sync_playwright() as p:
                    browser = p.chromium.launch(
                        headless=True,
                        args=[
                            '--no-sandbox',
                            '--disable-dev-shm-usage',
                            '--disable-gpu',
                            '--disable-web-security',
                            '--disable-extensions',
                            '--disable-plugins',
                            '--disable-images',
                            '--disable-background-timer-throttling',
                            '--disable-backgrounding-occluded-windows',
                            '--disable-renderer-backgrounding',
                            '--memory-pressure-off',
                            '--max_old_space_size=4096'
                        ]
                    )
                    # 创建高分辨率页面上下文
                    context = browser.new_context(
                        device_scale_factor=2.0,  # 2倍像素密度
                        viewport={"width": 800, "height": 600}
                    )
                    page = context.new_page()
                    
                    print(f"🔧 Loading HTML file...")
                    # Load HTML file with optimized settings
                    file_url = f"file://{temp_html_path}"
                    print(f"🔧 File URL: {file_url}")
                    page.goto(file_url, wait_until="domcontentloaded", timeout=10000)
                    
                    print(f"🔧 Waiting for Mermaid to render...")
                    # Wait for Mermaid to render with more precise conditions
                    try:
                        # Wait for SVG element to appear
                        page.wait_for_selector(".mermaid svg", timeout=5000)
                        # Wait for SVG to have actual content (not empty)
                        page.wait_for_function(
                            "document.querySelector('.mermaid svg').innerHTML.length > 0",
                            timeout=5000
                        )
                    except Exception as e:
                        print(f"⚠️ Waiting for SVG content failed: {e}")
                        # Fallback: just wait a bit
                        page.wait_for_timeout(200)
                    
                    print(f"🔧 Extracting SVG content and generating PNG...")
                    
                    svg_success = False
                    png_success = False
                    
                    # Get the complete SVG element (not just innerHTML)
                    svg_element = page.locator(".mermaid svg").first
                    if svg_element:
                        # Get the outer HTML (complete SVG tag)
                        svg_content = svg_element.evaluate("el => el.outerHTML")
                        if svg_content:
                            # Fix XML issues: convert <br> to <br/> for proper XML formatting
                            svg_content = svg_content.replace('<br>', '<br/>')
                            
                            # Add XML declaration
                            full_svg = f'<?xml version="1.0" encoding="UTF-8"?>\n{svg_content}'
                            
                            # 转换HSL颜色为标准RGB颜色（如果可用）
                            if HSL_CONVERTER_AVAILABLE:
                                try:
                                    converted_svg = convert_svg_hsl_colors_optimized(full_svg)
                                    if converted_svg != full_svg:
                                        print_debug(f"🎨 Converted HSL colors to RGB for better compatibility")
                                        full_svg = converted_svg
                                except Exception as e:
                                    print_debug(f"⚠️ HSL color conversion failed: {e}")
                            
                            # 转换foreignObject为原生SVG text元素（如果可用）
                            if FOREIGN_OBJECT_CONVERTER_AVAILABLE and has_foreign_objects(full_svg):
                                try:
                                    converted_svg = convert_mermaid_foreign_objects(full_svg)
                                    if converted_svg != full_svg:
                                        print_debug(f"🔧 Converted foreignObject elements to native SVG text for better PDF compatibility")
                                        full_svg = converted_svg
                                except Exception as e:
                                    print_debug(f"⚠️ ForeignObject conversion failed: {e}")
                            
                            # Save SVG
                            with open(svg_path, 'w', encoding='utf-8') as f:
                                f.write(full_svg)
                            print(f"✅ SVG content saved to {svg_path}")
                            svg_success = True
                        else:
                            print(f"❌ No SVG content found in mermaid element")
                    else:
                        print(f"❌ No SVG element found")
                    
                    # Generate PNG if requested
                    if png_path and svg_success:
                        try:
                            # Take screenshot of the mermaid element
                            print(f"🔧 Taking PNG screenshot...")
                            # Get the mermaid container element
                            mermaid_element = page.locator(".mermaid").first
                            if mermaid_element:
                                # Take screenshot with padding
                                png_bytes = mermaid_element.screenshot(
                                    type="png",
                                    path=str(png_path),
                                    omit_background=True
                                )
                                print(f"✅ PNG screenshot saved to {png_path}")
                                
                                # 自动裁剪PNG图片，去除空白区域
                                if self.png_cropper:
                                    try:
                                        self.png_cropper.crop_png(png_path, padding=15, verbose=False)
                                    except Exception:
                                        pass  # 静默处理裁剪错误
                                
                                png_success = True
                            else:
                                print(f"❌ No mermaid element found for PNG screenshot")
                        except Exception as e:
                            print(f"⚠️ PNG screenshot failed: {e}")
                    
                    browser.close()
                
                # Check if files were created successfully
                if svg_success and (not png_path or png_success):
                    print(f"✅ Playwright generation successful")
                    return svg_success, png_success
                else:
                    print(f"❌ Playwright generation failed: SVG={svg_success}, PNG={png_success}")
                    return svg_success, png_success
                    
            finally:
                # Clean up temporary file
                try:
                    import os
                    os.unlink(temp_html_path)
                except OSError:
                    pass
                
        except Exception as e:
            print(f"❌ Playwright generation failed: {e}")
            import traceback
            traceback.print_exc()
            return False, False
    
    def has_mermaid_charts(self, md_file_path: str) -> bool:
        """
        Check if a markdown file contains Mermaid charts (including incomplete ones).
        
        Args:
            md_file_path: Path to the markdown file
            
        Returns:
            True if Mermaid charts are found, False otherwise
        """
        try:
            with open(md_file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Check for complete Mermaid blocks
            complete_pattern = re.compile(r'```mermaid\n(.*?)\n```', re.DOTALL)
            complete_matches = complete_pattern.findall(content)
            
            if complete_matches:
                return True
            
            # Check for incomplete Mermaid blocks (missing closing ```)
            # Use the same improved logic as in process_markdown_file
            mermaid_starts = []
            start_idx = 0
            while True:
                idx = content.find('```mermaid\n', start_idx)
                if idx == -1:
                    break
                mermaid_starts.append(idx)
                start_idx = idx + 1
            
            # For each ```mermaid start, check if it has a proper closing ```
            for start_idx in mermaid_starts:
                # Check if this start position is already part of a complete match
                is_part_of_complete = False
                complete_matches_iter = list(re.compile(r'```mermaid\n(.*?)\n```', re.DOTALL).finditer(content))
                for complete_match in complete_matches_iter:
                    if (start_idx >= complete_match.start() and 
                        start_idx < complete_match.end()):
                        is_part_of_complete = True
                        break
                
                if not is_part_of_complete:
                    # Find the content after ```mermaid\n
                    content_start = start_idx + len('```mermaid\n')
                    # Look for the closing ``` after this position
                    remaining_content = content[content_start:]
                    closing_idx = remaining_content.find('\n```')
                    
                    if closing_idx == -1:
                        # No closing ``` found, this is incomplete
                        # Find where this incomplete block ends (end of file or next ```)
                        next_backticks = remaining_content.find('```')
                        if next_backticks == -1:
                            # Goes to end of file
                            block_content = remaining_content.strip()
                        else:
                            # Ends before next backticks
                            block_content = remaining_content[:next_backticks].strip()
                        
                        if block_content:  # Only consider it incomplete if there's actual content
                            return True
            
            return False
            
        except Exception as e:
            print_debug(f"❌ Error checking for Mermaid charts: {e}")
            return False
    
    def scan_and_process_directory(self, directory_path: str) -> Dict[str, Any]:
        """
        Scan directory for markdown files and process Mermaid charts.
        
        Args:
            directory_path: Directory to scan
            
        Returns:
            Dictionary with processing results
        """
        try:
            print_debug(f"📂 Scanning directory: {directory_path}")
            
            # Find markdown files in root directory only (not recursive)
            markdown_files = []
            try:
                for file in os.listdir(directory_path):
                    file_path = os.path.join(directory_path, file)
                    if os.path.isfile(file_path) and file.endswith('.md'):
                        markdown_files.append(file_path)
            except Exception as e:
                print_debug(f"❌ Failed to scan directory: {e}")
                return {
                    'status': 'failed',
                    'error': str(e),
                    'message': f'Failed to scan directory: {e}'
                }
            
            if not markdown_files:
                print_debug("❌ No markdown files found")
                return {
                    'status': 'success',
                    'files_found': 0,
                    'files_processed': 0,
                    'message': 'No markdown files found'
                }
            
            print_debug(f"📄 Found {len(markdown_files)} markdown file(s):")
            for file in markdown_files:
                print_debug(f"   - {file}")
            
            # Process each markdown file
            processed_count = 0
            total_charts = 0
            total_processed_charts = 0
            
            for markdown_file in markdown_files:
                print_debug(f"\n🔧 Processing file: {markdown_file}")
                try:
                    result = self.process_markdown_file(markdown_file)
                    if result['status'] == 'success':
                        processed_count += 1
                        total_charts += result['charts_found']
                        total_processed_charts += result['charts_processed']
                except Exception as e:
                    print_debug(f"❌ Failed to process file: {markdown_file}, error: {e}")
            
            print_debug(f"\n✅ Mermaid processing complete! Successfully processed {processed_count}/{len(markdown_files)} files")
            print_debug(f"📊 Total charts processed: {total_processed_charts}/{total_charts}")
            print_debug(f"📁 Images saved in images directories alongside markdown files")
            
            return {
                'status': 'success',
                'files_found': len(markdown_files),
                'files_processed': processed_count,
                'total_charts_found': total_charts,
                'total_charts_processed': total_processed_charts,
                'message': f'Successfully processed {processed_count}/{len(markdown_files)} files'
            }
            
        except Exception as e:
            print_debug(f"❌ Error during directory processing: {e}")
            return {
                'status': 'failed',
                'error': str(e),
                'message': f'Failed to process directory: {e}'
            }


# Create a global instance for easy access (silent initialization to avoid early logging)
mermaid_processor = MermaidProcessor(silent_init=True)