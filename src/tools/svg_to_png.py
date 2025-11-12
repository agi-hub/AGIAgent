#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced SVG to PNG solution
Specifically solves Chinese font display and rendering consistency issues
"""

import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, Tuple
import sys

# 添加src目录到路径中，以便导入裁剪工具
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.png_cropper import PNGCropper

def _check_playwright_available():
    """Check if playwright is available for browser automation"""
    try:
        import playwright
        from playwright.sync_api import sync_playwright
        return True
    except ImportError:
        return False
    except Exception:
        return False

PLAYWRIGHT_AVAILABLE = _check_playwright_available()

class EnhancedSVGToPNGConverter:
    def __init__(self):
        self.chinese_fonts = [
            "Noto Sans CJK SC",
            "Noto Serif CJK SC", 
            "AR PL UMing CN",
            "AR PL UKai CN",
            "WenQuanYi Micro Hei",
            "SimHei",
            "Microsoft YaHei",
            "Source Han Sans CN"
        ]
        self.cropper = PNGCropper()
        
    def get_available_chinese_font(self) -> Optional[str]:
        """Get available Chinese fonts in the system"""
        try:
            # 在Windows系统中，优先使用系统自带的中文字体
            if os.name == 'nt':  # Windows系统
                windows_fonts = [
                    "Microsoft YaHei",
                    "SimHei", 
                    "SimSun",
                    "KaiTi",
                    "FangSong"
                ]
                # 直接返回第一个Windows字体，因为这些字体在Windows中通常都可用
                return windows_fonts[0]
            
            # Linux/Unix系统使用fc-list命令
            result = subprocess.run(['fc-list', ':lang=zh'], 
                                  capture_output=True, text=True, encoding='utf-8', errors='ignore')
            available_fonts = result.stdout
            
            if not available_fonts:
                # 如果fc-list没有返回结果，尝试不带参数的fc-list
                result = subprocess.run(['fc-list'], 
                                      capture_output=True, text=True, encoding='utf-8', errors='ignore')
                available_fonts = result.stdout
            
            for font in self.chinese_fonts:
                if font in available_fonts:
                    return font
            
            # If preset font not found
            lines = available_fonts.strip().split('\n') if available_fonts else []
            if lines and lines[0]:
                # Format is usually: /path/to/font.ttf: Font Name:style=Style
                parts = lines[0].split(':')
                if len(parts) > 1:
                    font_info = parts[1].strip()
                    return font_info
                
        except Exception as e:
            print(f"Check Chinese font failed: {e}")
            # 如果所有方法都失败，返回一个通用的fallback字体
            if os.name == 'nt':  # Windows
                return "Microsoft YaHei"
            else:  # Linux/Unix
                return "DejaVu Sans"
        
        return None
    
    def enhance_svg_for_chinese(self, svg_content: str, chinese_font: str) -> str:
        """Enhance SVG to better support Chinese fonts"""
        
        # 1. Replace font settings
        font_family_pattern = r'font-family:"[^"]*"'
        enhanced_font_family = f'font-family:"{chinese_font}","Noto Sans CJK SC","SimHei",sans-serif'
        svg_content = re.sub(font_family_pattern, enhanced_font_family, svg_content)
        
        # 2. Add Chinese fonts in CSS styles
        style_pattern = r'(#my-svg\{[^}]*font-family:)([^;]*)(;[^}]*\})'
        def replace_font_in_css(match):
            return f'{match.group(1)}"{chinese_font}","Noto Sans CJK SC","SimHei",sans-serif{match.group(3)}'
        
        svg_content = re.sub(style_pattern, replace_font_in_css, svg_content)
        
        # 3. Ensure SVG declares correct character encoding
        if not svg_content.startswith('<?xml'):
            svg_content = '<?xml version="1.0" encoding="UTF-8"?>\\n' + svg_content
        elif 'encoding=' not in svg_content[:100]:
            svg_content = svg_content.replace('<?xml version="1.0"', 
                                            '<?xml version="1.0" encoding="UTF-8"')
        
        # 4. Add font rendering optimization
        svg_content = svg_content.replace('<svg', 
            '<svg text-rendering="optimizeLegibility" shape-rendering="geometricPrecision"')
        
        return svg_content
    
    def convert_with_playwright(self, svg_path: Path, png_path: Path) -> bool:
        """Use Playwright for conversion, ensure font rendering is correct"""
        # Check if Playwright is available before proceeding
        if not PLAYWRIGHT_AVAILABLE:
            print("❌ Playwright not installed")
            return False
        
        try:
            from playwright.sync_api import sync_playwright
            
            # Read SVG content
            with open(svg_path, 'r', encoding='utf-8') as f:
                svg_content = f.read()
            
            # Create HTML page
            html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <style>
        body {{ 
            margin: 0; 
            padding: 20px; 
            background: white;
            font-family: "Noto Sans CJK SC", "SimHei", sans-serif;
        }}
        svg {{ 
            background: white; 
            font-family: "Noto Sans CJK SC", "SimHei", sans-serif;
        }}
    </style>
</head>
<body>
    {svg_content}
</body>
</html>
"""
            
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True)
                # 创建高分辨率页面上下文
                context = browser.new_context(
                    device_scale_factor=2.0,  # 2倍像素密度
                    viewport={"width": 1800, "height": 800}
                )
                page = context.new_page()
                
                # Load HTML content
                page.set_content(html_content)
                
                # Wait for font loading
                page.wait_for_timeout(2000)
                
                # Screenshot with high quality settings
                svg_element = page.locator("svg")
                if svg_element.count() > 0:
                    svg_element.screenshot(
                        path=str(png_path), 
                        type='png',
                        omit_background=False  # 保留背景以获得更好的渲染
                    )
                else:
                    # Fallback: screenshot the entire page if SVG element not found
                    page.screenshot(
                        path=str(png_path),
                        type='png',
                        full_page=True
                    )
                
                browser.close()
            
            if png_path.exists() and png_path.stat().st_size > 0:
                print(f"✅ Playwright conversion successful")
                
                # 自动裁剪PNG图片，去除空白区域
                try:
                    self.cropper.crop_png(png_path, padding=15, verbose=False)
                except Exception:
                    pass  # 静默处理裁剪错误
                
                return True
            else:
                print(f"❌ Playwright conversion failed: File not generated")
                return False
                
        except ImportError:
            print("❌ Playwright not installed")
            return False
        except Exception as e:
            print(f"❌ Playwright conversion error: {e}")
            return False
    
    def convert(self, svg_path: Path, png_path: Path, 
                enhance_chinese: bool = True, dpi: int = 300) -> Tuple[bool, str]:
        """
        Main conversion method
        
        Args:
            svg_path: SVG file path
            png_path: Target PNG file path
            enhance_chinese: Whether to enhance Chinese font support
            dpi: Output resolution
            
        Returns:
            (Success flag, Detailed information)
        """
        
        if not svg_path.exists():
            return False, f"SVG file does not exist: {svg_path}"
        

        
        # Read original SVG content
        try:
            with open(svg_path, 'r', encoding='utf-8') as f:
                original_svg_content = f.read()
        except Exception as e:
            return False, f"Failed to read SVG file: {e}"
        
        # Enhance SVG to support Chinese
        if enhance_chinese:
            chinese_font = self.get_available_chinese_font()
            if chinese_font:
                #print(f"📝 Using Chinese font: {chinese_font}")
                enhanced_svg_content = self.enhance_svg_for_chinese(original_svg_content, chinese_font)
                
                # Create temporary enhanced SVG file
                with tempfile.NamedTemporaryFile(mode='w', suffix='.svg', 
                                               delete=False, encoding='utf-8') as temp_file:
                    temp_file.write(enhanced_svg_content)
                    enhanced_svg_path = Path(temp_file.name)
            else:
                print("⚠️ Chinese font not found")
                enhanced_svg_path = svg_path
        else:
            enhanced_svg_path = svg_path
        
        # Try different conversion methods
        conversion_methods = []
        if PLAYWRIGHT_AVAILABLE:
            conversion_methods.append(("Playwright", lambda: self.convert_with_playwright(enhanced_svg_path, png_path)))
        
        success = False
        error_details = []
        result_message = ""
        
        for method_name, convert_func in conversion_methods:
            try:
                if convert_func():
                    success = True
                    result_message = f"{method_name}Conversion successful"
                    break
                else:
                    error_details.append(f"{method_name}: Conversion failed")
            except Exception as e:
                error_details.append(f"{method_name}: {e}")
        
        # Clean up temporary files
        if enhance_chinese and enhanced_svg_path != svg_path:
            try:
                enhanced_svg_path.unlink()
            except:
                pass
        
        if success:
            return True, result_message
        else:
            return False, "; ".join(error_details)
