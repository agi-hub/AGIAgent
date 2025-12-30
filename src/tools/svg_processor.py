#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SVG Processor for Markdown Files

This module processes SVG code blocks in markdown files and converts them to PNG images.
It detects ```svg code blocks, generates separate SVG files, converts them to PNG,
and updates the markdown with image links.

Copyright (c) 2025 AGI Agent Research Group.
Licensed under the Apache License, Version 2.0
"""

import os
import re
import html
import hashlib
import tempfile
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from .print_system import print_system, print_current, print_debug


class SVGProcessor:
    """SVG processor for markdown files"""
    
    def __init__(self, workspace_root: Optional[str] = None):
        """Initialize the SVG processor"""
        self.workspace_root = workspace_root or os.getcwd()
        self.svg_output_dir = "images"  # Directory for generated images
        self._check_dependencies()
    
    def set_workspace_root(self, workspace_root: str):
        """Set the workspace root directory"""
        self.workspace_root = workspace_root
    
    def _check_dependencies(self):
        """Check if required dependencies are available"""
        self.inkscape_available = self._check_command_available('inkscape')
        self.rsvg_convert_available = self._check_command_available('rsvg-convert')
        self.cairosvg_available = self._check_python_package('cairosvg')
        
        if self.inkscape_available:
            print_debug("🎨 Inkscape detected for SVG to PNG conversion")
        elif self.rsvg_convert_available:
            print_debug("🎨 rsvg-convert detected for SVG to PNG conversion")
        elif self.cairosvg_available:
            print_debug("🎨 CairoSVG detected for SVG to PNG conversion")
        else:
            print_debug("⚠️ No SVG conversion tools available. Please install inkscape, rsvg-convert, or cairosvg")
    
    def _check_command_available(self, command: str) -> bool:
        """Check if a command is available in the system"""
        import platform
        try:
            # On Windows, try both with and without .exe extension
            if platform.system().lower() == "windows":
                # First try the command as-is
                try:
                    result = subprocess.run([command, '--version'],
                                          capture_output=True, text=True, encoding='utf-8', errors='ignore', timeout=5)
                    if result.returncode == 0:
                        return True
                except:
                    pass
                
                # Then try with .exe extension
                try:
                    result = subprocess.run([command + '.exe', '--version'],
                                          capture_output=True, text=True, encoding='utf-8', errors='ignore', timeout=5)
                    if result.returncode == 0:
                        return True
                except:
                    pass
                
                # Finally, use 'where' command to check if it exists in PATH
                try:
                    result = subprocess.run(['where', command], 
                                          capture_output=True, text=True, timeout=5)
                    return result.returncode == 0
                except:
                    return False
            else:
                # Unix-like systems
                result = subprocess.run([command, '--version'],
                                      capture_output=True, text=True, encoding='utf-8', errors='ignore', timeout=5)
                return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.CalledProcessError):
            return False
    
    def _check_python_package(self, package: str) -> bool:
        """
        Check if a Python package is available (without importing it)
        
        🚀 性能优化：使用 importlib.util.find_spec 检查模块存在性
        避免在启动时导入 cairosvg (~2.4秒) 和 cairocffi (~1.5秒)
        """
        try:
            import importlib.util
            spec = importlib.util.find_spec(package)
            return spec is not None
        except (ImportError, ValueError, AttributeError):
            # ValueError: can occur for invalid module names
            # AttributeError: can occur for edge cases
            return False
    
    def has_svg_blocks(self, markdown_file: str) -> bool:
        """
        Check if a markdown file contains SVG code blocks (including malformed ones)
        
        Args:
            markdown_file: Path to the markdown file
            
        Returns:
            True if SVG blocks are found, False otherwise
        """
        try:
            with open(markdown_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Pattern to match standard ```svg code blocks
            standard_pattern = r'```svg\s*\n(.*?)\n```'
            standard_matches = re.findall(standard_pattern, content, re.DOTALL | re.IGNORECASE)
            
            # Pattern to match malformed ```svg code blocks (missing closing ```)
            malformed_pattern = r'```svg\s*\n(.*?</svg>)(?!\s*\n```)'
            malformed_matches = re.findall(malformed_pattern, content, re.DOTALL | re.IGNORECASE)
            
            total_matches = len(standard_matches) + len(malformed_matches)
            
            if total_matches > 0:
                print_debug(f"📊 Found {len(standard_matches)} standard + {len(malformed_matches)} malformed SVG blocks")
            
            return total_matches > 0
            
        except Exception as e:
            print_debug(f"❌ Error checking SVG blocks in {markdown_file}: {e}")
            return False
    
    def extract_svg_blocks(self, content: str) -> List[Dict[str, Any]]:
        """
        Extract SVG code blocks from markdown content with error tolerance
        
        Args:
            content: Markdown content
            
        Returns:
            List of dictionaries containing SVG block information
        """
        svg_blocks = []
        
        # First try standard pattern
        standard_pattern = r'```svg\s*\n(.*?)\n```'
        
        for match in re.finditer(standard_pattern, content, re.DOTALL | re.IGNORECASE):
            svg_code = match.group(1).strip()
            # Decode HTML entities (e.g., &lt; -> <, &gt; -> >, &amp; -> &)
            svg_code = html.unescape(svg_code)
            start_pos = match.start()
            end_pos = match.end()
            full_block = match.group(0)
            
            # Generate a unique ID for this SVG block based on content hash
            svg_hash = hashlib.md5(svg_code.encode('utf-8')).hexdigest()[:8]
            
            # Extract caption from following comment
            following_content = content[end_pos:end_pos+200]  # Check next 200 chars
            caption = self._extract_caption_from_comment(following_content)
            
            svg_blocks.append({
                'id': svg_hash,
                'svg_code': svg_code,
                'full_block': full_block,
                'start_pos': start_pos,
                'end_pos': end_pos,
                'caption': caption,
                'is_corrected': False
            })
        
        # Apply error tolerance for malformed SVG blocks
        corrected_content = self._apply_svg_error_tolerance(content)
        
        # If content was corrected, re-extract blocks
        if corrected_content != content:
            print_debug("🔧 Applied SVG error tolerance corrections")
            
            # Clear previous blocks and re-extract from corrected content
            svg_blocks = []
            
            for match in re.finditer(standard_pattern, corrected_content, re.DOTALL | re.IGNORECASE):
                svg_code = match.group(1).strip()
                # Decode HTML entities (e.g., &lt; -> <, &gt; -> >, &amp; -> &)
                svg_code = html.unescape(svg_code)
                start_pos = match.start()
                end_pos = match.end()
                full_block = match.group(0)
                
                # Generate a unique ID for this SVG block based on content hash
                svg_hash = hashlib.md5(svg_code.encode('utf-8')).hexdigest()[:8]
                
                # Extract caption from following comment
                following_content = corrected_content[end_pos:end_pos+200]  # Check next 200 chars
                caption = self._extract_caption_from_comment(following_content)
                
                svg_blocks.append({
                    'id': svg_hash,
                    'svg_code': svg_code,
                    'full_block': full_block,
                    'start_pos': start_pos,
                    'end_pos': end_pos,
                    'caption': caption,
                    'is_corrected': True,
                    'original_content': content,
                    'corrected_content': corrected_content
                })
        
        print_debug(f"📊 Found {len(svg_blocks)} SVG code blocks")
        return svg_blocks
    
    def _extract_caption_from_comment(self, content: str) -> Optional[str]:
        """
        Extract figure caption from comment following SVG block
        
        Args:
            content: Content following the SVG block
            
        Returns:
            Caption text if found, None otherwise
        """
        # Look for the figure caption comment pattern: <!-- caption -->
        # Match the first HTML comment that doesn't contain system keywords
        caption_match = re.search(r'<!--\s*([^-]+?)\s*-->', content.strip(), re.IGNORECASE)
        if caption_match:
            caption = caption_match.group(1).strip()
            # Filter out common system comments that shouldn't be used as captions
            system_comments = ['the_figure_caption', 'Available formats', 'Source code file', 'SVG processing failed']
            if not any(sys_comment in caption for sys_comment in system_comments):
                return caption
        return None
    
    def _apply_svg_error_tolerance(self, content: str) -> str:
        """
        Apply error tolerance to fix malformed SVG code blocks

        This method looks for SVG blocks that start with ```svg but are malformed,
        and fixes them by ensuring proper formatting. It handles cases where:
        - SVG blocks end with </svg> without a proper closing ```
        - AI adds extra ``` markers after </svg> that should be removed

        Args:
            content: Original markdown content

        Returns:
            Corrected markdown content
        """
        corrected_content = content
        corrections_made = 0

        # Simple approach: replace malformed patterns with corrected versions
        # Pattern 1: ```svg\n...<svg> (missing closing ```)
        pattern1 = r'```svg\s*\n(.*?)</svg>(?![\s]*```)'
        corrected_content = re.sub(
            pattern1,
            lambda m: f"```svg\n{m.group(1)}</svg>\n```",
            corrected_content,
            flags=re.DOTALL | re.IGNORECASE
        )
        if re.search(pattern1, content, re.DOTALL | re.IGNORECASE):
            corrections_made += 1
            print_debug(f"🔧 Fixed malformed SVG blocks with missing closing markers")

        if corrections_made > 0:
            print_debug(f"✅ Applied {corrections_made} SVG error tolerance corrections")

        return corrected_content
    
    def _fix_svg_xml_entities(self, svg_code: str) -> str:
        """
        自动修复SVG中未转义的XML特殊字符
        
        在XML/SVG中，以下字符必须转义：
        - & → &amp;
        - < → &lt;
        - > → &gt;
        - " → &quot;
        - ' → &apos;
        
        Args:
            svg_code: 原始SVG代码
            
        Returns:
            修正后的SVG代码
        """
        try:
            # 在文本内容中查找未转义的 & 字符
            # 匹配 >...& ...< 之间的内容（文本节点）
            def replace_unescaped_ampersand(match):
                text = match.group(0)
                # 只替换未转义的 & (不是 &amp; &lt; &gt; &quot; &apos; &#数字; &#x十六进制;)
                text = re.sub(r'&(?!(amp|lt|gt|quot|apos|#\d+|#x[0-9a-fA-F]+);)', '&amp;', text)
                return text
            
            # 在标签的文本内容中替换（匹配 >文本< 的模式）
            fixed_code = re.sub(r'>([^<>]*)<', replace_unescaped_ampersand, svg_code)
            
            if fixed_code != svg_code:
                print_debug("🔧 Fixed unescaped XML entities in SVG code")
                # 显示修正的数量
                original_ampersands = len(re.findall(r'&(?!(amp|lt|gt|quot|apos|#\d+|#x[0-9a-fA-F]+);)', svg_code))
                if original_ampersands > 0:
                    print_debug(f"   ✓ Fixed {original_ampersands} unescaped '&' character(s)")
            
            return fixed_code
            
        except Exception as e:
            print_debug(f"⚠️ Error fixing XML entities: {e}")
            return svg_code  # 如果修复失败，返回原始代码
    
    def _convert_css_background_to_svg(self, svg_code: str) -> str:
        """
        将SVG中的CSS background样式转换为SVG标准元素
        
        很多SVG转换工具（如CairoSVG、Inkscape）不支持CSS的background属性，
        需要将其转换为SVG的<rect>元素和<linearGradient>元素。
        
        Args:
            svg_code: 原始SVG代码
            
        Returns:
            转换后的SVG代码
        """
        try:
            # 查找SVG标签中的style属性，包含background
            pattern = r'<svg([^>]*?)(style\s*=\s*["\']([^"\']*?)["\'])([^>]*?)>'
            match = re.search(pattern, svg_code, re.IGNORECASE | re.DOTALL)
            
            if not match:
                return svg_code  # 没有找到style属性，直接返回
            
            svg_attrs_before = match.group(1)
            style_attr = match.group(2)
            style_content = match.group(3)
            svg_attrs_after = match.group(4)
            
            # 检查style中是否包含background
            if 'background' not in style_content.lower():
                return svg_code  # 没有background，直接返回
            
            # 提取SVG的width和height属性
            width_match = re.search(r'width\s*=\s*["\'](\d+)["\']', svg_code, re.IGNORECASE)
            height_match = re.search(r'height\s*=\s*["\'](\d+)["\']', svg_code, re.IGNORECASE)
            
            width = width_match.group(1) if width_match else '900'
            height = height_match.group(1) if height_match else '650'
            
            # 解析background样式
            # 支持格式: background: linear-gradient(135deg, #2d1b69 0%, #11998e 100%);
            bg_match = re.search(r'background\s*:\s*linear-gradient\s*\(([^)]+)\)', style_content, re.IGNORECASE)
            
            if not bg_match:
                # 如果不是linear-gradient，尝试提取纯色背景
                color_match = re.search(r'background\s*:\s*([#\w]+)', style_content, re.IGNORECASE)
                if color_match:
                    bg_color = color_match.group(1)
                    # 创建简单的纯色背景rect
                    bg_rect = f'<rect x="0" y="0" width="{width}" height="{height}" fill="{bg_color}"/>'
                    # 移除style中的background
                    new_style = re.sub(r'background\s*:[^;]+;?\s*', '', style_content, flags=re.IGNORECASE).strip()
                    if new_style:
                        new_svg_tag = f'<svg{svg_attrs_before}style="{new_style}"{svg_attrs_after}>'
                    else:
                        new_svg_tag = f'<svg{svg_attrs_before}{svg_attrs_after}>'
                    # 在<svg>标签后插入背景rect
                    new_svg_code = svg_code.replace(match.group(0), new_svg_tag + bg_rect)
                    print_debug("🎨 Converted CSS background to SVG rect element")
                    return new_svg_code
                return svg_code
            
            # 解析linear-gradient参数
            grad_params = bg_match.group(1)
            
            # 提取角度（如果有）
            angle_match = re.search(r'(\d+)deg', grad_params, re.IGNORECASE)
            angle = int(angle_match.group(1)) if angle_match else 0
            
            # 提取颜色停止点
            # 格式: #2d1b69 0%, #11998e 100%
            stops = re.findall(r'([#\w]+)\s+(\d+)%', grad_params)
            
            if not stops or len(stops) < 2:
                return svg_code  # 无法解析，返回原始代码
            
            # 生成唯一的渐变ID
            import random
            grad_id = f'bgGrad_{random.randint(1000, 9999)}'
            
            # 计算渐变方向（根据角度）
            # SVG linearGradient使用x1, y1, x2, y2定义方向
            import math
            rad = math.radians(angle)
            x1 = 0.5 - 0.5 * math.cos(rad)
            y1 = 0.5 - 0.5 * math.sin(rad)
            x2 = 0.5 + 0.5 * math.cos(rad)
            y2 = 0.5 + 0.5 * math.sin(rad)
            
            # 创建linearGradient定义
            gradient_def = f'<defs><linearGradient id="{grad_id}" x1="{x1:.3f}" y1="{y1:.3f}" x2="{x2:.3f}" y2="{y2:.3f}">'
            for color, offset in stops:
                gradient_def += f'<stop offset="{offset}%" style="stop-color:{color};stop-opacity:1" />'
            gradient_def += '</linearGradient></defs>'
            
            # 创建背景rect
            bg_rect = f'<rect x="0" y="0" width="{width}" height="{height}" fill="url(#{grad_id})"/>'
            
            # 移除style中的background
            new_style = re.sub(r'background\s*:[^;]+;?\s*', '', style_content, flags=re.IGNORECASE).strip()
            
            # 构建新的SVG标签
            if new_style:
                new_svg_tag = f'<svg{svg_attrs_before}style="{new_style}"{svg_attrs_after}>'
            else:
                # 如果style为空，完全移除style属性
                new_svg_tag = f'<svg{svg_attrs_before}{svg_attrs_after}>'
            
            # 替换SVG标签，并在<svg>后插入渐变定义和背景rect
            # 需要找到<defs>标签的位置，如果没有则插入在<svg>后
            if '<defs>' in svg_code:
                # 如果有defs，在defs内插入gradient
                defs_pattern = r'(<defs[^>]*>)'
                defs_match = re.search(defs_pattern, svg_code, re.IGNORECASE)
                if defs_match:
                    # 在defs标签后插入gradient
                    new_svg_code = svg_code.replace(match.group(0), new_svg_tag)
                    new_svg_code = new_svg_code.replace(defs_match.group(0), defs_match.group(0) + f'<linearGradient id="{grad_id}" x1="{x1:.3f}" y1="{y1:.3f}" x2="{x2:.3f}" y2="{y2:.3f}">' + ''.join([f'<stop offset="{offset}%" style="stop-color:{color};stop-opacity:1" />' for color, offset in stops]) + '</linearGradient>', 1)
                    # 在第一个非defs元素前插入背景rect
                    first_element_pattern = r'(</defs>\s*)(<[^/])'
                    first_element_match = re.search(first_element_pattern, new_svg_code, re.IGNORECASE)
                    if first_element_match:
                        new_svg_code = new_svg_code.replace(first_element_match.group(0), first_element_match.group(1) + bg_rect + '\n' + first_element_match.group(2), 1)
                    else:
                        # 如果没有其他元素，在defs后插入
                        new_svg_code = new_svg_code.replace('</defs>', '</defs>' + bg_rect, 1)
                else:
                    new_svg_code = svg_code.replace(match.group(0), new_svg_tag + gradient_def + bg_rect)
            else:
                # 没有defs，直接插入
                new_svg_code = svg_code.replace(match.group(0), new_svg_tag + gradient_def + bg_rect)
            
            print_debug("🎨 Converted CSS linear-gradient background to SVG gradient and rect element")
            return new_svg_code
            
        except Exception as e:
            print_debug(f"⚠️ Error converting CSS background to SVG: {e}")
            return svg_code  # 如果转换失败，返回原始代码
    
    def generate_svg_file(self, svg_code: str, output_dir: Path, svg_id: str) -> Optional[Path]:
        """
        Generate an SVG file from SVG code
        
        Args:
            svg_code: SVG source code
            output_dir: Output directory for the SVG file
            svg_id: Unique identifier for the SVG
            
        Returns:
            Path to the generated SVG file, or None if failed
        """
        try:
            # Ensure output directory exists
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # 自动修复SVG中的XML实体问题
            fixed_svg_code = self._fix_svg_xml_entities(svg_code)
            
            # 将CSS背景转换为SVG标准元素（解决转换工具不支持CSS background的问题）
            fixed_svg_code = self._convert_css_background_to_svg(fixed_svg_code)
            
            # Generate SVG filename
            svg_filename = f"svg_{svg_id}.svg"
            svg_path = output_dir / svg_filename
            
            # Write SVG content to file (使用修正后的代码)
            with open(svg_path, 'w', encoding='utf-8') as f:
                f.write(fixed_svg_code)
            
            print_debug(f"📄 Generated SVG file: {svg_path}")
            return svg_path
            
        except Exception as e:
            print_debug(f"❌ Failed to generate SVG file for {svg_id}: {e}")
            return None
    
    def convert_svg_to_png(self, svg_path: Path, png_path: Path) -> bool:
        """
        Convert SVG file to PNG using available conversion tools
        
        Args:
            svg_path: Path to the source SVG file
            png_path: Path for the output PNG file
            
        Returns:
            True if conversion successful, False otherwise
        """
        import platform
        import tempfile
        import os
        
        # 预处理SVG：将CSS背景转换为SVG标准元素
        temp_svg_path = None
        original_svg_path = svg_path
        try:
            with open(svg_path, 'r', encoding='utf-8') as f:
                svg_content = f.read()
            processed_svg = self._convert_css_background_to_svg(svg_content)
            if processed_svg != svg_content:
                # 如果内容被修改，创建临时文件
                with tempfile.NamedTemporaryFile(mode='w', suffix='.svg', delete=False, encoding='utf-8') as tmp_file:
                    tmp_file.write(processed_svg)
                    temp_svg_path = Path(tmp_file.name)
                svg_path = temp_svg_path
        except Exception as e:
            print_debug(f"⚠️ SVG background preprocessing failed: {e}, using original file")
        
        try:
            # On Windows, try enhanced SVG converter first if available
            if platform.system().lower() == "windows":
                try:
                    from .svg_to_png import EnhancedSVGToPNGConverter
                    converter = EnhancedSVGToPNGConverter()
                    success, message = converter.convert(svg_path, png_path)
                    if success:
                        print_debug(f"✅ Enhanced SVG converter successful: {message}")
                        return True
                    else:
                        print_debug(f"⚠️ Enhanced SVG converter failed: {message}")
                except Exception as e:
                    print_debug(f"⚠️ Enhanced SVG converter not available: {e}")
            
            # Try Inkscape first (best quality)
            if self.inkscape_available:
                if self._convert_with_inkscape(svg_path, png_path):
                    return True
            
            # Try rsvg-convert
            if self.rsvg_convert_available:
                if self._convert_with_rsvg(svg_path, png_path):
                    return True
            
            # Try CairoSVG (Python package)
            if self.cairosvg_available:
                if self._convert_with_cairosvg(svg_path, png_path):
                    return True
            
            # Last resort: try Playwright-based conversion
            try:
                from .svg_to_png import EnhancedSVGToPNGConverter
                converter = EnhancedSVGToPNGConverter()
                success, message = converter.convert(svg_path, png_path)
                if success:
                    print_debug(f"✅ Playwright fallback successful: {message}")
                    return True
                else:
                    print_debug(f"❌ Playwright fallback failed: {message}")
            except Exception as e:
                print_debug(f"❌ Playwright fallback error: {e}")
            
            print_debug("❌ All SVG conversion methods failed")
            return False
        finally:
            # 清理临时文件
            if temp_svg_path and temp_svg_path.exists():
                try:
                    os.unlink(temp_svg_path)
                except Exception as e:
                    print_debug(f"⚠️ Failed to clean up temp SVG file: {e}")
    
    def _convert_with_inkscape(self, svg_path: Path, png_path: Path) -> bool:
        """Convert SVG to PNG using Inkscape"""
        import platform
        try:
            # Determine Inkscape command based on platform
            inkscape_cmd = 'inkscape'
            if platform.system().lower() == "windows":
                # On Windows, try different possible commands
                possible_commands = ['inkscape', 'inkscape.exe']
                for cmd in possible_commands:
                    try:
                        test_result = subprocess.run([cmd, '--version'],
                                                   capture_output=True, text=True, encoding='utf-8', errors='ignore', timeout=5)
                        if test_result.returncode == 0:
                            inkscape_cmd = cmd
                            break
                    except:
                        continue
            
            # Try new format first (Inkscape 1.0+)
            cmd_new = [
                inkscape_cmd,
                '--export-type=png',
                '--export-dpi=300',
                f'--export-filename={png_path}',
                str(svg_path)
            ]
            
            result = subprocess.run(cmd_new, capture_output=True, text=True, encoding='utf-8', errors='ignore', timeout=30)
            
            # If new format fails, try old format (Inkscape 0.92)
            if result.returncode != 0:
                print_debug(f"🔄 Trying old Inkscape format...")
                cmd_old = [
                    inkscape_cmd,
                    f'--export-png={png_path}',
                    '--export-dpi=300',
                    str(svg_path)
                ]
                
                result = subprocess.run(cmd_old, capture_output=True, text=True, encoding='utf-8', errors='ignore', timeout=30)
            
            if result.returncode == 0 and png_path.exists():
                print_debug(f"✅ Converted SVG to PNG using Inkscape: {png_path}")
                return True
            else:
                print_debug(f"❌ Inkscape conversion failed: {result.stderr}")
                return False
                
        except Exception as e:
            print_debug(f"❌ Inkscape conversion error: {e}")
            return False
    
    def _convert_with_rsvg(self, svg_path: Path, png_path: Path) -> bool:
        """Convert SVG to PNG using rsvg-convert"""
        try:
            cmd = [
                'rsvg-convert',
                '-f', 'png',
                '-d', '300',  # DPI
                '-p', '300',  # DPI
                '-o', str(png_path),
                str(svg_path)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='ignore', timeout=30)
            
            if result.returncode == 0 and png_path.exists():
                print_debug(f"✅ Converted SVG to PNG using rsvg-convert: {png_path}")
                return True
            else:
                print_debug(f"❌ rsvg-convert conversion failed: {result.stderr}")
                return False
                
        except Exception as e:
            print_debug(f"❌ rsvg-convert conversion error: {e}")
            return False
    
    def _convert_with_cairosvg(self, svg_path: Path, png_path: Path) -> bool:
        """Convert SVG to PNG using CairoSVG Python package"""
        import platform
        try:
            import cairosvg
            
            # On Windows, try to handle common CairoSVG issues
            if platform.system().lower() == "windows":
                try:
                    # Read SVG content and preprocess if needed
                    with open(svg_path, 'r', encoding='utf-8') as f:
                        svg_content = f.read()
                    
                    # Convert with high DPI for better quality
                    cairosvg.svg2png(
                        bytestring=svg_content.encode('utf-8'),
                        write_to=str(png_path),
                        dpi=300
                    )
                except Exception as e:
                    # Fallback to file-based conversion
                    print_debug(f"⚠️ CairoSVG bytestring conversion failed, trying file-based: {e}")
                    cairosvg.svg2png(
                        url=str(svg_path),
                        write_to=str(png_path),
                        dpi=300
                    )
            else:
                # Unix-like systems - use standard approach
                cairosvg.svg2png(
                    url=str(svg_path),
                    write_to=str(png_path),
                    dpi=300
                )
            
            if png_path.exists():
                print_debug(f"✅ Converted SVG to PNG using CairoSVG: {png_path}")
                return True
            else:
                print_debug(f"❌ CairoSVG conversion failed: PNG file not created")
                return False
                
        except Exception as e:
            print_debug(f"❌ CairoSVG conversion error: {e}")
            return False
    
    def process_svg_blocks(self, svg_blocks: List[Dict[str, Any]], markdown_dir: Path) -> List[Dict[str, Any]]:
        """
        Process a list of SVG blocks and generate PNG images
        
        Args:
            svg_blocks: List of SVG block dictionaries
            markdown_dir: Directory containing the markdown file
            
        Returns:
            List of processing results for each SVG block
        """
        results = []
        
        # Create images directory
        images_dir = markdown_dir / self.svg_output_dir
        images_dir.mkdir(parents=True, exist_ok=True)
        
        for block in svg_blocks:
            svg_id = block['id']
            svg_code = block['svg_code']
            
            print_debug(f"🎨 Processing SVG block: {svg_id}")
            
            # Generate SVG file
            svg_path = self.generate_svg_file(svg_code, images_dir, svg_id)
            if not svg_path:
                results.append({
                    'id': svg_id,
                    'status': 'failed',
                    'error': 'Failed to generate SVG file',
                    'block': block
                })
                continue
            
            # Generate PNG file
            png_filename = f"svg_{svg_id}.png"
            png_path = images_dir / png_filename
            
            conversion_success = self.convert_svg_to_png(svg_path, png_path)
            
            if conversion_success:
                # Calculate relative path for markdown
                relative_png_path = f"{self.svg_output_dir}/{png_filename}"
                
                results.append({
                    'id': svg_id,
                    'status': 'success',
                    'svg_file': str(svg_path.relative_to(markdown_dir)),
                    'png_file': relative_png_path,
                    'png_size': png_path.stat().st_size,
                    'block': block
                })
                
                print_debug(f"✅ Successfully processed SVG block {svg_id}")
            else:
                results.append({
                    'id': svg_id,
                    'status': 'failed',
                    'error': 'Failed to convert SVG to PNG',
                    'svg_file': str(svg_path.relative_to(markdown_dir)) if svg_path else None,
                    'block': block
                })
        
        return results
    
    def update_markdown_content(self, content: str, processing_results: List[Dict[str, Any]]) -> str:
        """
        Update markdown content by replacing SVG code blocks with image links or fallback comments
        
        Args:
            content: Original markdown content
            processing_results: Results from processing SVG blocks
            
        Returns:
            Updated markdown content
        """
        updated_content = content
        base_content_for_search = content
        
        # Check if any blocks were corrected and use the corrected content as base
        corrected_blocks = [r for r in processing_results if r['block'].get('is_corrected', False)]
        if corrected_blocks:
            # Use the corrected content from the first corrected block as our base
            updated_content = corrected_blocks[0]['block']['corrected_content']
            base_content_for_search = corrected_blocks[0]['block']['corrected_content']
            print_debug("📝 Using error-corrected content as base for updates")
        
        # Sort ALL results by start position in reverse order to avoid position shifts
        all_results = processing_results
        all_results.sort(key=lambda x: x['block']['start_pos'], reverse=True)
        
        for result in all_results:
            block = result['block']
            svg_id = result['id']
            full_block = block['full_block']
            caption = block.get('caption')
            
            # Build replacement pattern that includes following comment if caption exists
            # This ensures we replace both the SVG block and the comment in one operation
            replacement_pattern = full_block
            if caption:
                # Check if there's a comment immediately following the SVG block
                end_pos = block['end_pos']
                # Use the base content for search (corrected or original)
                following_text = base_content_for_search[end_pos:end_pos+300]  # Check next 300 chars
                # Match comment with the caption, allowing for whitespace
                comment_match = re.search(
                    r'\s*\n\s*<!--\s*' + re.escape(caption) + r'\s*-->\s*\n',
                    following_text,
                    re.IGNORECASE | re.MULTILINE
                )
                if comment_match:
                    # Include the comment in the pattern to be replaced
                    replacement_pattern = full_block + comment_match.group(0)
                    print_debug(f"📝 Found caption comment for SVG block {svg_id}, will be removed")
            
            # Determine replacement content
            if result['status'] == 'success':
                # Successful SVG processing - use SVG image
                svg_file = result.get('svg_file', result['png_file'].replace('.png', '.svg'))
                # Use caption if available, otherwise use default
                alt_text = caption if caption else f"SVG图表 {svg_id}"
                replacement = f"![{alt_text}]({svg_file})\n\n"
                print_debug(f"🔄 Replaced SVG block {svg_id} with SVG image link (caption: {alt_text})")
                
            elif result.get('svg_file'):
                # PNG conversion failed but SVG file exists - use SVG image directly
                svg_file = result['svg_file']
                # Use caption if available, otherwise use default
                alt_text = caption if caption else f"SVG图表 {svg_id}"
                replacement = f"![{alt_text}]({svg_file})\n\n"
                print_debug(f"🔄 Replaced SVG block {svg_id} with SVG image link (PNG conversion failed, caption: {alt_text})")
                
            else:
                # Complete failure - use error comment
                error = result.get('error', 'Unknown error')
                replacement = f"<!-- SVG processing failed: {error} -->"
                print_debug(f"🔄 Replaced SVG block {svg_id} with error comment")
            
            # Replace the SVG code block (and comment if included)
            updated_content = updated_content.replace(replacement_pattern, replacement, 1)
        
        # Final cleanup: Remove orphaned ``` markers left after SVG replacement
        # This single pattern handles: ``` after images, at end of file, or standalone
        before_cleanup = updated_content
        updated_content = re.sub(
            r'(?:^|\n)\s*```\s*(?:\n|$)',  # Match ``` on its own line
            '\n',
            updated_content,
            flags=re.MULTILINE
        )
        
        if updated_content != before_cleanup:
            print_debug("🧹 Cleaned up orphaned ``` markers after SVG replacement")
        
        return updated_content
    
    def process_markdown_file(self, markdown_file: str) -> Dict[str, Any]:
        """
        Process a markdown file and convert all SVG code blocks to PNG images
        
        Args:
            markdown_file: Path to the markdown file
            
        Returns:
            Dictionary containing processing results
        """
        try:
            markdown_path = Path(markdown_file)
            
            if not markdown_path.exists():
                return {
                    'status': 'failed',
                    'file': markdown_file,
                    'error': 'Markdown file not found'
                }
            
            # Read markdown content
            with open(markdown_path, 'r', encoding='utf-8') as f:
                original_content = f.read()
            
            # Extract SVG blocks
            svg_blocks = self.extract_svg_blocks(original_content)
            
            if not svg_blocks:
                return {
                    'status': 'success',
                    'file': markdown_file,
                    'message': 'No SVG code blocks found',
                    'svg_blocks_found': 0
                }
            
            # Process SVG blocks
            processing_results = self.process_svg_blocks(svg_blocks, markdown_path.parent)
            
            # Update markdown content
            updated_content = self.update_markdown_content(original_content, processing_results)
            
            # Check if any changes were made
            if updated_content != original_content:
                # Write updated content back to file
                with open(markdown_path, 'w', encoding='utf-8') as f:
                    f.write(updated_content)
                
                print_debug(f"📝 Updated markdown file: {markdown_file}")
            
            # Prepare summary
            successful_conversions = sum(1 for r in processing_results if r['status'] == 'success')
            failed_conversions = len(processing_results) - successful_conversions
            
            return {
                'status': 'success',
                'file': markdown_file,
                'svg_blocks_found': len(svg_blocks),
                'successful_conversions': successful_conversions,
                'failed_conversions': failed_conversions,
                'processing_results': processing_results,
                'message': f'Processed {successful_conversions}/{len(svg_blocks)} SVG blocks successfully'
            }
            
        except Exception as e:
            print_debug(f"❌ Error processing markdown file {markdown_file}: {e}")
            return {
                'status': 'failed',
                'file': markdown_file,
                'error': str(e)
            }
    
    def cleanup_generated_files(self, processing_results: List[Dict[str, Any]], markdown_dir: Path):
        """
        Clean up generated SVG and PNG files (useful for testing)
        
        Args:
            processing_results: Results from processing SVG blocks
            markdown_dir: Directory containing the markdown file
        """
        for result in processing_results:
            if result['status'] == 'success':
                # Remove SVG file
                if 'svg_file' in result:
                    svg_path = markdown_dir / result['svg_file']
                    if svg_path.exists():
                        svg_path.unlink()
                        print_debug(f"🗑️ Removed SVG file: {svg_path}")
                
                # Remove PNG file
                if 'png_file' in result:
                    png_path = markdown_dir / result['png_file']
                    if png_path.exists():
                        png_path.unlink()
                        print_debug(f"🗑️ Removed PNG file: {png_path}")


# Create a global instance for easy access
svg_processor = SVGProcessor()
