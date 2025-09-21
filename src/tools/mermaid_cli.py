#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
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
from pathlib import Path
from typing import List, Dict, Any, Optional

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
def _check_mermaid_cli():
    """Check if mermaid-cli (mmdc) is available"""
    try:
        result = subprocess.run(['mmdc', '--version'], 
                              capture_output=True, text=True, timeout=10)
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False

def _check_playwright():
    """Check if playwright is available for rendering"""
    try:
        import playwright
        return True
    except ImportError:
        return False

def _check_mermaid_py():
    """Check if mmdc CLI is available (fallback method)"""
    # This is now just a fallback that uses the same CLI as the primary method
    return MERMAID_CLI_AVAILABLE

# Check available methods
MERMAID_CLI_AVAILABLE = _check_mermaid_cli()
PLAYWRIGHT_AVAILABLE = _check_playwright()
MERMAID_PY_AVAILABLE = _check_mermaid_py()
ONLINE_FALLBACK = True  # Always available as fallback

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
if MERMAID_CLI_AVAILABLE:
    PREFERRED_METHOD = "cli"
elif PLAYWRIGHT_AVAILABLE:
    PREFERRED_METHOD = "playwright"
elif MERMAID_PY_AVAILABLE:
    PREFERRED_METHOD = "python"
else:
    PREFERRED_METHOD = "online"


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
        self.mermaid_available = (MERMAID_CLI_AVAILABLE or PLAYWRIGHT_AVAILABLE or 
                                MERMAID_PY_AVAILABLE or ONLINE_FALLBACK)
        
        # Initialize SVG to PNG converter
        if SVG_TO_PNG_AVAILABLE:
            self.svg_to_png_converter = EnhancedSVGToPNGConverter()
        else:
            self.svg_to_png_converter = None
        
    
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
                print_current(f"🔧 Found {len(incomplete_matches)} incomplete Mermaid block(s), auto-fixing...")
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
                        print_current(f"✅ Auto-fixed incomplete Mermaid block")
                
                # Write the corrected content back to file if modifications were made
                if content_modified:
                    with open(md_file_path, 'w', encoding='utf-8') as f:
                        f.write(content)
                    print_current(f"📝 Auto-corrected {len(incomplete_matches)} incomplete Mermaid block(s) in file")
            
            # Now extract all Mermaid code blocks with optional following caption from the (potentially corrected) content
            # Match: ```mermaid\n{code}\n``` followed by optional whitespace and comment
            pattern = re.compile(r'```mermaid\n(.*?)\n```(\s*<!--[^>]*-->)?', re.DOTALL)
            matches = list(pattern.finditer(content))
            
            if not matches:
                print_current("📝 No Mermaid charts found")
                return {
                    'status': 'success',
                    'file': md_file_path,
                    'charts_found': 0,
                    'charts_processed': 0,
                    'message': 'No Mermaid charts found',
                    'auto_fixed': len(incomplete_matches) if incomplete_matches else 0
                }
            
            print_current(f"📊 Found {len(matches)} Mermaid chart(s)")
            
            processed_count = 0
            
            # Process from end to beginning to avoid position shifts
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
                    
                    #print_current(f"🔧 Processing Mermaid chart {len(matches)-i}...")
                    
                    # Save original Mermaid code to separate file
                    try:
                        with open(mermaid_code_path, 'w', encoding='utf-8') as f:
                            f.write(code)
                        #print_current(f"📝 Saved Mermaid code: {mermaid_code_name}")
                    except Exception as e:
                        print_current(f"❌ Failed to save Mermaid code: {e}")
                    
                    # Generate SVG image using the best available method
                    svg_success = self._generate_mermaid_image(code, svg_path)
                    png_success = False
                    
                    # Check if the generated SVG contains errors
                    if svg_success:
                        is_error_svg = _is_error_svg(svg_path)
                        if is_error_svg:
                            print_current(f"❌ Generated SVG contains errors, treating as failed")
                            svg_success = False
                    
                    # If SVG generation successful, try to convert to PNG using enhanced converter
                    if svg_success and self.svg_to_png_converter:
                        try:
                            success, message = self.svg_to_png_converter.convert(svg_path, png_path, enhance_chinese=True)
                            if success:
                                png_success = True
                                #print_current(f"✅ SVG to PNG conversion successful: {message}")
                            else:
                                print_current(f"⚠️ SVG generated successfully but PNG conversion failed: {message}")
                        except Exception as e:
                            print_current(f"⚠️ SVG to PNG conversion error: {e}")
                    elif svg_success:
                        print_current(f"⚠️ SVG generated successfully but PNG converter not available")
                    
                    # If SVG generation successful, replace content
                    if svg_success:
                        # Generate appropriate alt text based on caption from comment
                        # Extract caption from comment for alt text
                        def extract_caption_from_comment_for_alt(content: str) -> Optional[str]:
                            caption_match = re.search(r'<!--\s*([^-]+?)\s*-->', content.strip(), re.IGNORECASE)
                            if caption_match:
                                caption = caption_match.group(1).strip()
                                system_comments = ['the_figure_caption', 'Available formats', 'Source code file']
                                if not any(sys_comment in caption for sys_comment in system_comments):
                                    return caption
                            return None
                        
                        caption = extract_caption_from_comment_for_alt(following_comment)
                        if caption:
                            alt_text = caption
                        elif base_filename.startswith('mermaid_sha'):
                            # For SHA-based filenames, use figure number
                            alt_text = f"Figure {len(matches)-i}"
                        else:
                            # For named files, use the filename as title
                            alt_text = base_filename.replace('_', ' ').title()
                        
                        # Use SVG for display in markdown if available, otherwise use PNG
                        display_path = rel_svg_path if svg_success else rel_png_path
                        format_info = ""
                        if svg_success and png_success:
                            format_info = f"<!-- Available formats: PNG={rel_png_path}, SVG={rel_svg_path} -->\n"
                        elif svg_success:
                            format_info = f"<!-- Available formats: SVG={rel_svg_path} -->\n"
                        
                        # NEW: Use standard markdown image format for better pandoc compatibility
                        replacement = f"\n![{alt_text}]({display_path})\n\n{format_info}<!-- Source code file: {rel_mermaid_path} -->\n"
                        
                        # OLD: HTML format (commented out for pandoc compatibility)
                        # Determine appropriate size based on chart type
                        # chart_type = code.strip().split('\n')[0] if '\n' in code else code.strip()
                        # Use consistent styling without max-height to prevent distortion
                        # size_style = "max-width: 80%; height: auto; width: 80%;"
                        # replacement = f"\n<div align=\"center\">\n\n<img src=\"{display_path}\" alt=\"{alt_text}\" style=\"{size_style}\" />\n\n</div>\n\n{format_info}<!-- Source code file: {rel_mermaid_path} -->\n"
                        
                        # Get complete Mermaid code block positions
                        start_pos = match.start()
                        end_pos = match.end()
                        
                        # Replace original content
                        content = content[:start_pos] + replacement + content[end_pos:]
                        
                        if svg_success and png_success:
                            print_current(f"✅ Successfully generated: {svg_name} and converted to {png_name}")
                        elif svg_success:
                            print_current(f"✅ Successfully generated: {svg_name}")
                        processed_count += 1
                    else:
                        # Mermaid compilation failed, replace with error comment
                        print_current(f"❌ Mermaid compilation failed, replacing with error comment")
                        
                        # Extract caption for error message
                        def extract_caption_from_comment_for_error(content: str) -> Optional[str]:
                            caption_match = re.search(r'<!--\s*([^-]+?)\s*-->', content.strip(), re.IGNORECASE)
                            if caption_match:
                                caption = caption_match.group(1).strip()
                                system_comments = ['the_figure_caption', 'Available formats', 'Source code file']
                                if not any(sys_comment in caption for sys_comment in system_comments):
                                    return caption
                            return None
                        
                        caption = extract_caption_from_comment_for_error(following_comment)
                        if caption:
                            error_replacement = f"\n<!-- ❌ Mermaid chart compilation failed: {caption} -->\n<!-- Source code file: {rel_mermaid_path} -->\n"
                        else:
                            error_replacement = f"\n<!-- ❌ Mermaid chart compilation failed (Figure {len(matches)-i}) -->\n<!-- Source code file: {rel_mermaid_path} -->\n"
                        
                        # Get complete Mermaid code block positions
                        start_pos = match.start()
                        end_pos = match.end()
                        
                        # Replace original content with error comment
                        content = content[:start_pos] + error_replacement + content[end_pos:]
                        
                except Exception as e:
                    print_current(f"❌ Error processing Mermaid chart: {e}")
            
            # Write updated file
            with open(md_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            #print_current(f"✅ Processing complete. Modified file saved to: {md_path}")
            #print_current(f"📁 Generated images saved in: {img_dir}")
            
            return {
                'status': 'success',
                'file': md_file_path,
                'charts_found': len(matches),
                'charts_processed': processed_count,
                'images_dir': str(img_dir),
                'message': f'Successfully processed {processed_count}/{len(matches)} Mermaid charts'
            }
            
        except Exception as e:
            print_current(f"❌ Error processing markdown file: {e}")
            return {
                'status': 'failed',
                'file': md_file_path,
                'error': str(e),
                'message': f'Failed to process markdown file: {e}'
            }
    
    def _generate_mermaid_image(self, mermaid_code: str, output_path: Path) -> bool:
        """
        Generate Mermaid SVG image using the best available method.
        
        Args:
            mermaid_code: Mermaid chart code
            output_path: Output SVG image path
            
        Returns:
            True if successful, False otherwise
        """
        # Try methods in order of preference
        methods = [
            ("cli", self._generate_mermaid_image_cli),
            ("playwright", self._generate_mermaid_image_playwright),
            ("python", self._generate_mermaid_image_python),
            ("online", self._generate_mermaid_image_online)
        ]
        
        # Try preferred method first
        for method_name, method_func in methods:
            if method_name == self.preferred_method:
                try:
                    if method_func(mermaid_code, output_path):
                        print_debug(f"✅ Successfully generated using {method_name} method")
                        return True
                except Exception as e:
                    print_debug(f"❌ {method_name} method failed: {e}")
        
        # Try other methods as fallbacks
        for method_name, method_func in methods:
            if method_name != self.preferred_method:
                try:
                    if method_func(mermaid_code, output_path):
                        print_debug(f"✅ Successfully generated using fallback {method_name} method")
                        return True
                except Exception as e:
                    print_debug(f"❌ Fallback {method_name} method failed: {e}")
        
        print_debug("❌ All methods failed to generate image")
        return False
    

    
    def _generate_mermaid_image_cli(self, mermaid_code: str, output_path: Path) -> bool:
        """
        Generate Mermaid SVG image using mermaid CLI (mmdc).
        
        Args:
            mermaid_code: Mermaid chart code
            output_path: Output SVG image path
            
        Returns:
            True if successful, False otherwise
        """
        if not MERMAID_CLI_AVAILABLE:
            return False
            
        try:
            print_debug(f"🔧 Using Mermaid CLI to generate image...")
            
            # Create temporary mermaid file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.mmd', delete=False, encoding='utf-8') as temp_file:
                temp_file.write(mermaid_code)
                temp_mermaid_path = temp_file.name
            
            try:
                # Use mermaid CLI to generate high-quality SVG image
                cmd = [
                    'mmdc',
                    '-i', temp_mermaid_path,
                    '-o', str(output_path),
                    '-b', 'transparent',  # transparent background
                    '--quiet',  # suppress output
                    '-e', 'svg'  # explicitly set SVG format
                ]
                
                # Only add theme if the mermaid code doesn't have custom theme configuration
                if not ('%%{init:' in mermaid_code and 'theme' in mermaid_code):
                    cmd.extend(['-t', 'default'])  # use default theme only if no custom theme
                
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
                
                # Check if command was successful and file was created
                if result.returncode == 0 and output_path.exists() and output_path.stat().st_size > 0:
                    print_debug(f"✅ Mermaid CLI generation successful")
                    return True
                else:
                    print_debug(f"❌ Mermaid CLI generation failed: {result.stderr}")
                    return False
                    
            finally:
                # Clean up temporary file
                try:
                    os.unlink(temp_mermaid_path)
                except OSError:
                    pass
                
        except subprocess.TimeoutExpired:
            print_debug(f"❌ Mermaid CLI generation timed out")
            return False
        except Exception as e:
            print_debug(f"❌ Mermaid CLI generation failed: {e}")
            return False
    
    def _generate_mermaid_image_playwright(self, mermaid_code: str, output_path: Path) -> bool:
        """
        Generate Mermaid SVG image using Playwright browser automation.
        
        Args:
            mermaid_code: Mermaid chart code
            output_path: Output SVG image path
            
        Returns:
            True if successful, False otherwise
        """
        if not PLAYWRIGHT_AVAILABLE:
            return False
            
        try:
            from playwright.sync_api import sync_playwright
            
            print_debug(f"🌐 Using Playwright to generate image...")
            
            # Create HTML with Mermaid chart
            html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
    <style>
        body {{ margin: 0; padding: 20px; background: transparent; }}
        .mermaid {{ background: transparent; }}
    </style>
</head>
<body>
    <div class="mermaid">{mermaid_code}</div>
    <script>
        mermaid.initialize({{ 
            startOnLoad: true, 
            theme: 'default',
            flowchart: {{
                useMaxWidth: false,
                nodeSpacing: 30,
                rankSpacing: 40,
                diagramPadding: 20
            }},
            maxWidth: 700
        }});
    </script>
</body>
</html>"""
            
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True)
                # 创建高分辨率页面上下文
                context = browser.new_context(
                    device_scale_factor=2.0,  # 2倍像素密度
                    viewport={"width": 1200, "height": 800}
                )
                page = context.new_page()
                
                # Load HTML content
                page.set_content(html_content)
                
                # Wait for Mermaid to render
                page.wait_for_selector(".mermaid svg", timeout=10000)
                
                # Extract SVG content from the mermaid element
                svg_content = page.locator(".mermaid svg").inner_html()
                if svg_content:
                    # Get the complete SVG element with proper XML declaration
                    full_svg = f'<?xml version="1.0" encoding="UTF-8"?>\n<svg xmlns="http://www.w3.org/2000/svg">{svg_content}</svg>'
                    
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
                    with open(output_path, 'w', encoding='utf-8') as f:
                        f.write(full_svg)
                else:
                    print_debug("❌ No SVG content found in mermaid element")
                    return False
                
                browser.close()
            
            # Check if file was created successfully
            if output_path.exists() and output_path.stat().st_size > 0:
                print_debug(f"✅ Playwright generation successful")
                return True
            else:
                print_debug(f"❌ Playwright generation failed: output file not created or empty")
                return False
                
        except Exception as e:
            print_debug(f"❌ Playwright generation failed: {e}")
            return False
    
    def _generate_mermaid_image_python(self, mermaid_code: str, output_path: Path) -> bool:
        """
        Generate Mermaid SVG image using mmdc CLI (fallback method).
        
        Args:
            mermaid_code: Mermaid chart code
            output_path: Output SVG image path
            
        Returns:
            True if successful, False otherwise
        """
        if not MERMAID_PY_AVAILABLE:
            return False
            
        try:
            import tempfile
            import subprocess
            import os
            
            print_debug(f"🐍 Using mmdc CLI (fallback method) to generate image...")
            
            # Create temporary mermaid file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.mmd', delete=False, encoding='utf-8') as temp_file:
                temp_file.write(mermaid_code)
                temp_mermaid_path = temp_file.name
            
            try:
                # Use mermaid CLI with neutral theme for fallback
                cmd = [
                    'mmdc',
                    '-i', temp_mermaid_path,
                    '-o', str(output_path),
                    '-b', 'transparent',  # transparent background
                    '--quiet',  # suppress output
                    '-e', 'svg'  # explicitly set SVG format
                ]
                
                # Only add theme if the mermaid code doesn't have custom theme configuration
                if not ('%%{init:' in mermaid_code and 'theme' in mermaid_code):
                    cmd.extend(['-t', 'neutral'])  # use neutral theme for fallback only if no custom theme
                
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
                
                # Check if command was successful and file was created
                if result.returncode == 0 and output_path.exists() and output_path.stat().st_size > 0:
                    print_debug(f"✅ CLI fallback generation successful")
                    return True
                else:
                    print_debug(f"❌ CLI fallback generation failed: {result.stderr}")
                    return False
                    
            finally:
                # Clean up temporary file
                try:
                    os.unlink(temp_mermaid_path)
                except OSError:
                    pass
                
        except Exception as e:
            print_debug(f"❌ CLI fallback generation failed: {e}")
            return False
    
    def _generate_mermaid_image_online(self, mermaid_code: str, output_path: Path) -> bool:
        """
        Generate Mermaid SVG image using online mermaid.ink API.
        
        Args:
            mermaid_code: Mermaid chart code
            output_path: Output SVG image path
            
        Returns:
            True if successful, False otherwise
        """
        try:
            print_debug(f"🌐 Using online mermaid.ink API to generate image...")
            
            # Encode Mermaid code to base64
            encoded_code = base64.b64encode(mermaid_code.encode('utf-8')).decode('utf-8')
            # Use SVG endpoint for vector graphics
            api_url = f"https://mermaid.ink/svg/{encoded_code}"
            
            # Download SVG
            response = requests.get(api_url, timeout=30)
            if response.status_code == 200:
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(response.text)
                print_debug(f"✅ Online API generation successful")
                return True
            else:
                print_debug(f"❌ Online API failed, status code: {response.status_code}")
                return False
                
        except requests.exceptions.RequestException as e:
            print_debug(f"❌ Network request failed: {e}")
            return False
        except Exception as e:
            print_debug(f"❌ Online API generation failed: {e}")
            return False
    
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
            print_current(f"📂 Scanning directory: {directory_path}")
            
            # Find markdown files in root directory only (not recursive)
            markdown_files = []
            try:
                for file in os.listdir(directory_path):
                    file_path = os.path.join(directory_path, file)
                    if os.path.isfile(file_path) and file.endswith('.md'):
                        markdown_files.append(file_path)
            except Exception as e:
                print_current(f"❌ Failed to scan directory: {e}")
                return {
                    'status': 'failed',
                    'error': str(e),
                    'message': f'Failed to scan directory: {e}'
                }
            
            if not markdown_files:
                print_current("❌ No markdown files found")
                return {
                    'status': 'success',
                    'files_found': 0,
                    'files_processed': 0,
                    'message': 'No markdown files found'
                }
            
            print_current(f"📄 Found {len(markdown_files)} markdown file(s):")
            for file in markdown_files:
                print_current(f"   - {file}")
            
            # Process each markdown file
            processed_count = 0
            total_charts = 0
            total_processed_charts = 0
            
            for markdown_file in markdown_files:
                print_current(f"\n🔧 Processing file: {markdown_file}")
                try:
                    result = self.process_markdown_file(markdown_file)
                    if result['status'] == 'success':
                        processed_count += 1
                        total_charts += result['charts_found']
                        total_processed_charts += result['charts_processed']
                except Exception as e:
                    print_current(f"❌ Failed to process file: {markdown_file}, error: {e}")
            
            print_current(f"\n✅ Mermaid processing complete! Successfully processed {processed_count}/{len(markdown_files)} files")
            print_current(f"📊 Total charts processed: {total_processed_charts}/{total_charts}")
            print_current(f"📁 Images saved in images directories alongside markdown files")
            
            return {
                'status': 'success',
                'files_found': len(markdown_files),
                'files_processed': processed_count,
                'total_charts_found': total_charts,
                'total_charts_processed': total_processed_charts,
                'message': f'Successfully processed {processed_count}/{len(markdown_files)} files'
            }
            
        except Exception as e:
            print_current(f"❌ Error during directory processing: {e}")
            return {
                'status': 'failed',
                'error': str(e),
                'message': f'Failed to process directory: {e}'
            }


# Create a global instance for easy access (silent initialization to avoid early logging)
mermaid_processor = MermaidProcessor(silent_init=True)