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

import re
import colorsys
from typing import Tuple, Optional
import math


def hsl_to_rgb(h: float, s: float, l: float) -> Tuple[int, int, int]:
    """
    将HSL颜色转换为RGB颜色
    
    Args:
        h: 色相 (0-360度)
        s: 饱和度 (0-100%)
        l: 亮度 (0-100%)
    
    Returns:
        RGB颜色元组 (r, g, b)，每个值在0-255范围内
    """
    # 将HSL值转换为0-1范围
    h_norm = h / 360.0
    s_norm = s / 100.0
    l_norm = l / 100.0
    
    # 使用colorsys库进行转换
    r, g, b = colorsys.hls_to_rgb(h_norm, l_norm, s_norm)
    
    # 转换为0-255范围并四舍五入
    r = int(round(r * 255))
    g = int(round(g * 255))
    b = int(round(b * 255))
    
    return (r, g, b)


def rgb_to_hex(r: int, g: int, b: int) -> str:
    """
    将RGB颜色转换为十六进制颜色代码
    
    Args:
        r, g, b: RGB颜色值 (0-255)
    
    Returns:
        十六进制颜色代码，如 "#FF0000"
    """
    return f"#{r:02X}{g:02X}{b:02X}"


def parse_hsl_color(hsl_string: str) -> Optional[Tuple[float, float, float]]:
    """
    解析HSL颜色字符串
    
    Args:
        hsl_string: HSL颜色字符串，如 "hsl(240, 100%, 76.2745098039%)"
    
    Returns:
        HSL值元组 (h, s, l) 或 None（如果解析失败）
    """
    # 匹配HSL颜色格式
    pattern = r'hsl\(\s*([0-9.]+)\s*,\s*([0-9.]+)%\s*,\s*([0-9.]+)%\s*\)'
    match = re.match(pattern, hsl_string.strip())
    
    if match:
        h = float(match.group(1))
        s = float(match.group(2))
        l = float(match.group(3))
        return (h, s, l)
    
    return None


def hsl_to_hex(hsl_string: str) -> Optional[str]:
    """
    将HSL颜色字符串直接转换为十六进制颜色代码
    
    Args:
        hsl_string: HSL颜色字符串，如 "hsl(240, 100%, 76.2745098039%)"
    
    Returns:
        十六进制颜色代码，如 "#B9B9FF"，或 None（如果转换失败）
    """
    hsl_values = parse_hsl_color(hsl_string)
    if hsl_values is None:
        return None
    
    h, s, l = hsl_values
    r, g, b = hsl_to_rgb(h, s, l)
    return rgb_to_hex(r, g, b)


def convert_svg_hsl_colors(svg_content: str) -> str:
    """
    自动转换SVG内容中的所有HSL颜色为十六进制颜色
    
    Args:
        svg_content: SVG文件内容字符串
    
    Returns:
        转换后的SVG内容字符串
    """
    # 匹配属性中的HSL颜色
    hsl_attr_pattern = r'(fill|stroke)=["\']hsl\([^"\']+\)["\']'
    
    def replace_hsl_attr(match):
        full_match = match.group(0)
        attribute = match.group(1)  # fill 或 stroke
        
        # 提取HSL颜色字符串
        hsl_match = re.search(r'hsl\([^)]+\)', full_match)
        if hsl_match:
            hsl_string = hsl_match.group(0)
            hex_color = hsl_to_hex(hsl_string)
            
            if hex_color:
                # 替换为十六进制颜色
                quote_char = '"' if '"' in full_match else "'"
                return f'{attribute}={quote_char}{hex_color}{quote_char}'
        
        # 如果转换失败，返回原始内容
        return full_match
    
    # 匹配CSS样式中的HSL颜色
    hsl_css_pattern = r'(fill|stroke):\s*hsl\([^;)]+\)'
    
    def replace_hsl_css(match):
        full_match = match.group(0)
        attribute = match.group(1)  # fill 或 stroke
        
        # 提取HSL颜色字符串
        hsl_match = re.search(r'hsl\([^)]+\)', full_match)
        if hsl_match:
            hsl_string = hsl_match.group(0)
            hex_color = hsl_to_hex(hsl_string)
            
            if hex_color:
                # 替换为十六进制颜色
                return f'{attribute}:{hex_color}'
        
        # 如果转换失败，返回原始内容
        return full_match
    
    # 匹配任何位置的HSL颜色（通用模式）
    hsl_general_pattern = r'hsl\([^)]+\)'
    
    def replace_hsl_general(match):
        hsl_string = match.group(0)
        hex_color = hsl_to_hex(hsl_string)
        return hex_color if hex_color else hsl_string
    
    # 按顺序执行替换
    converted_content = svg_content
    converted_content = re.sub(hsl_attr_pattern, replace_hsl_attr, converted_content)
    converted_content = re.sub(hsl_css_pattern, replace_hsl_css, converted_content)
    converted_content = re.sub(hsl_general_pattern, replace_hsl_general, converted_content)
    
    return converted_content


def convert_svg_file_hsl_colors(input_file: str, output_file: Optional[str] = None) -> bool:
    """
    转换SVG文件中的HSL颜色为十六进制颜色
    
    Args:
        input_file: 输入SVG文件路径
        output_file: 输出SVG文件路径（如果为None，则覆盖原文件）
    
    Returns:
        转换是否成功
    """
    try:
        # 读取SVG文件
        with open(input_file, 'r', encoding='utf-8') as f:
            svg_content = f.read()
        
        # 转换HSL颜色
        converted_content = convert_svg_hsl_colors(svg_content)
        
        # 写入文件
        output_path = output_file if output_file else input_file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(converted_content)
        
        return True
        
    except Exception as e:
        print(f"❌ HSL颜色转换失败: {e}")
        return False


def get_common_hsl_colors() -> dict:
    """
    获取常见HSL颜色的预定义映射
    
    Returns:
        HSL到十六进制颜色的映射字典
    """
    common_colors = {
        # Mermaid常用颜色
        "hsl(240, 100%, 76.2745098039%)": "#7B68EE",  # 中等蓝紫色
        "hsl(60, 100%, 73.5294117647%)": "#FFD700",   # 金色
        "hsl(80, 100%, 76.2745098039%)": "#90EE90",   # 淡绿色
        "hsl(270, 100%, 76.2745098039%)": "#DA70D6",  # 兰花紫
        "hsl(300, 100%, 76.2745098039%)": "#FF69B4",  # 热粉色
        "hsl(330, 100%, 76.2745098039%)": "#FF1493",  # 深粉色
        "hsl(0, 100%, 76.2745098039%)": "#FF6347",    # 番茄色
        "hsl(30, 100%, 76.2745098039%)": "#FFA500",   # 橙色
        "hsl(90, 100%, 76.2745098039%)": "#ADFF2F",   # 绿黄色
        "hsl(150, 100%, 76.2745098039%)": "#00FF7F",  # 春绿色
        "hsl(180, 100%, 76.2745098039%)": "#00FFFF",  # 青色
        "hsl(210, 100%, 76.2745098039%)": "#87CEEB",  # 天空蓝
        
        # 高亮度颜色
        "hsl(240, 100%, 86.2745098039%)": "#B9B9FF",  # 淡蓝色
        "hsl(60, 100%, 86.2745098039%)": "#FFFF99",   # 淡黄色
        "hsl(80, 100%, 86.2745098039%)": "#D7FF86",   # 淡绿色
        "hsl(270, 100%, 86.2745098039%)": "#E6B3FF",  # 淡紫色
        
        # 中等亮度颜色
        "hsl(80, 100%, 56.2745098039%)": "#B5FF20",   # 亮绿色
        "hsl(60, 100%, 63.5294117647%)": "#FFFF45",   # 亮黄色
    }
    
    return common_colors


def convert_svg_hsl_colors_optimized(svg_content: str) -> str:
    """
    优化版本的HSL颜色转换，使用预定义映射提高性能
    支持XML属性和CSS样式两种形式的HSL颜色
    特别处理思维导图的文本颜色问题

    Args:
        svg_content: SVG文件内容字符串

    Returns:
        转换后的SVG内容字符串
    """
    common_colors = get_common_hsl_colors()

    def get_hex_color(hsl_string: str) -> Optional[str]:
        """获取HSL颜色的十六进制表示"""
        # 首先尝试使用预定义映射
        if hsl_string in common_colors:
            return common_colors[hsl_string]
        else:
            # 如果没有预定义，则进行计算转换
            return hsl_to_hex(hsl_string)

    # 0. 特殊处理：为思维导图节点添加内联文本颜色
    def fix_mindmap_text_colors(svg_content):
        """为思维导图的text元素添加内联fill属性，避免CSS选择器问题"""
        # 思维导图section到颜色的映射
        section_colors = {
            'section--1': '#ffffff',  # 白色
            'section-0': 'black',     # 黑色
            'section-1': 'black',     # 黑色
            'section-2': '#ffffff',   # 白色
            'section-3': 'black',     # 黑色
            'section-4': 'black',     # 黑色
            'section-5': 'black',     # 黑色
            'section-6': 'black',     # 黑色
            'section-7': 'black',     # 黑色
            'section-8': 'black',     # 黑色
            'section-9': 'black',     # 黑色
            'section-10': 'black',    # 黑色
        }

        def add_inline_fill_to_match(match):
            full_match = match.group(0)
            classes_str = match.group(1).strip()
            text_attrs = match.group(2)

            # 解析类名
            classes = classes_str.split()

            # 找到匹配的section类
            text_color = None
            for cls in classes:
                if cls in section_colors:
                    text_color = section_colors[cls]
                    break

            if text_color and 'fill=' not in text_attrs:
                # 在text标签中添加fill属性
                if text_attrs.strip():
                    new_attrs = text_attrs.rstrip() + f' fill="{text_color}"'
                else:
                    new_attrs = f' fill="{text_color}"'

                return full_match.replace(text_attrs, new_attrs)

            return full_match

        # 匹配思维导图节点中的text元素
        mindmap_pattern = r'<g class="mindmap-node ([^"]*?)".*?>.*?<text([^>]*?)>.*?</text>.*?</g>'
        result = re.sub(mindmap_pattern, add_inline_fill_to_match, svg_content, flags=re.DOTALL)

        return result

    # 应用思维导图文本颜色修复
    svg_content = fix_mindmap_text_colors(svg_content)

    # 1. 匹配XML属性中的HSL颜色
    hsl_attr_pattern = r'(fill|stroke)=["\']hsl\([^"\']+\)["\']'

    def replace_hsl_attr(match):
        full_match = match.group(0)
        attribute = match.group(1)  # fill 或 stroke

        # 提取HSL颜色字符串
        hsl_match = re.search(r'hsl\([^)]+\)', full_match)
        if hsl_match:
            hsl_string = hsl_match.group(0)
            hex_color = get_hex_color(hsl_string)

            if hex_color:
                # 替换为十六进制颜色
                quote_char = '"' if '"' in full_match else "'"
                return f'{attribute}={quote_char}{hex_color}{quote_char}'

        # 如果转换失败，返回原始内容
        return full_match

    # 2. 匹配CSS样式中的HSL颜色
    hsl_css_pattern = r'(fill|stroke):\s*hsl\([^;)]+\)'

    def replace_hsl_css(match):
        full_match = match.group(0)
        attribute = match.group(1)  # fill 或 stroke

        # 提取HSL颜色字符串
        hsl_match = re.search(r'hsl\([^)]+\)', full_match)
        if hsl_match:
            hsl_string = hsl_match.group(0)
            hex_color = get_hex_color(hsl_string)

            if hex_color:
                # 替换为十六进制颜色
                return f'{attribute}:{hex_color}'

        # 如果转换失败，返回原始内容
        return full_match

    # 3. 匹配任何位置的HSL颜色（通用模式）
    hsl_general_pattern = r'hsl\([^)]+\)'

    def replace_hsl_general(match):
        hsl_string = match.group(0)
        hex_color = get_hex_color(hsl_string)
        return hex_color if hex_color else hsl_string

    # 按顺序执行替换
    converted_content = svg_content
    converted_content = re.sub(hsl_attr_pattern, replace_hsl_attr, converted_content)
    converted_content = re.sub(hsl_css_pattern, replace_hsl_css, converted_content)
    converted_content = re.sub(hsl_general_pattern, replace_hsl_general, converted_content)

    return converted_content


if __name__ == "__main__":
    # 测试函数
    test_cases = [
        "hsl(240, 100%, 76.2745098039%)",
        "hsl(60, 100%, 73.5294117647%)",
        "hsl(80, 100%, 56.2745098039%)",
        "hsl(270, 100%, 86.2745098039%)"
    ]
    
    print("HSL颜色转换测试:")
    print("-" * 50)
    
    for hsl_color in test_cases:
        hex_color = hsl_to_hex(hsl_color)
        print(f"{hsl_color:30} → {hex_color}")
    
    # 测试SVG转换
    test_svg = '''<path fill="hsl(240, 100%, 76.2745098039%)" stroke="hsl(60, 100%, 73.5294117647%)"/>'''
    converted_svg = convert_svg_hsl_colors(test_svg)
    
    print(f"\nSVG转换测试:")
    print(f"原始: {test_svg}")
    print(f"转换: {converted_svg}")
