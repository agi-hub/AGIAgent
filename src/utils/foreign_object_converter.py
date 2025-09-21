#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ForeignObject 转换工具
将SVG中的foreignObject元素转换为原生SVG text元素，解决PDF转换时的兼容性问题
"""

import re
from pathlib import Path
from typing import Optional, Tuple
import html


def extract_text_from_html(html_content: str) -> str:
    """从HTML内容中提取纯文本"""
    # 解码HTML实体
    text = html.unescape(html_content)
    # 移除HTML标签
    text = re.sub(r'<[^>]+>', '', text)
    # 清理空白字符
    return ' '.join(text.split()).strip()


def extract_text_lines_from_html(html_content: str) -> list[str]:
    """从HTML内容中提取文本行，保持换行结构"""
    # 解码HTML实体
    text = html.unescape(html_content)

    # 将<br>和<br/>转换为换行符
    text = re.sub(r'<br\s*/?>', '\n', text, flags=re.IGNORECASE)

    # 移除其他HTML标签
    text = re.sub(r'<[^>]+>', '', text)

    # 按行分割，并清理每行的空白字符
    lines = [line.strip() for line in text.split('\n') if line.strip()]

    return lines if lines else ['']


def calculate_text_position(x: float, y: float, width: float, height: float,
                          text_anchor: str = "middle") -> Tuple[float, float]:
    """计算文本在SVG中的正确位置"""
    if text_anchor == "middle":
        text_x = x + width / 2
    elif text_anchor == "start":
        text_x = x + 5  # 左边距
    elif text_anchor == "end":
        text_x = x + width - 5  # 右边距
    else:
        text_x = x + width / 2

    # Y坐标：foreignObject的y坐标 + 高度的一半 + 字体大小的一半（用于垂直居中）
    text_y = y + height / 2 + 6  # 6是大约字体大小的一半，用于视觉居中

    return text_x, text_y


def create_multiline_svg_text(text_lines: list[str], x: float, y: float, width: float, height: float,
                             font_size: int = 16) -> str:
    """创建多行SVG文本元素"""
    if not text_lines:
        return ''

    # 计算行高（字体大小 + 行间距）
    line_height = font_size + 2

    # 计算文本块的总高度
    total_text_height = len(text_lines) * line_height

    # 计算第一行的Y坐标，使整个文本块垂直居中
    start_y = y - total_text_height / 2 + line_height / 2

    # 生成tspan元素
    tspan_elements = []
    for i, line in enumerate(text_lines):
        line_y = start_y + i * line_height
        tspan_elements.append(f'<tspan x="{x}" y="{line_y}" text-anchor="middle">{line}</tspan>')

    # 生成完整的text元素
    text_element = f'''<text x="{x}" y="{start_y}" text-anchor="middle" dominant-baseline="central" font-family="Microsoft YaHei, SimHei, SimSun, Arial, sans-serif" font-size="{font_size}" fill="#333">{"".join(tspan_elements)}</text>'''

    return text_element


def get_font_size_from_style(style_attr: str) -> int:
    """从样式属性中提取字体大小"""
    if not style_attr:
        return 16
    
    # 查找font-size属性
    font_size_match = re.search(r'font-size:\s*(\d+)(?:px)?', style_attr, re.IGNORECASE)
    if font_size_match:
        return int(font_size_match.group(1))
    
    return 16  # 默认字体大小


def extract_transform_values(transform_str: str) -> Tuple[float, float]:
    """从transform属性中提取translate的x,y值"""
    if not transform_str:
        return 0.0, 0.0
    
    # 匹配translate(x, y)格式
    translate_match = re.search(r'translate\s*\(\s*([^,\s]+)\s*,\s*([^)]+)\s*\)', transform_str)
    if translate_match:
        try:
            x = float(translate_match.group(1))
            y = float(translate_match.group(2))
            return x, y
        except ValueError:
            return 0.0, 0.0
    
    return 0.0, 0.0


def convert_foreign_object_to_text(match) -> str:
    """将单个foreignObject转换为SVG text元素"""
    full_match = match.group(0)
    
    try:
        # 提取foreignObject的属性
        x_match = re.search(r'x\s*=\s*["\']([^"\']+)["\']', full_match)
        y_match = re.search(r'y\s*=\s*["\']([^"\']+)["\']', full_match)
        width_match = re.search(r'width\s*=\s*["\']([^"\']+)["\']', full_match)
        height_match = re.search(r'height\s*=\s*["\']([^"\']+)["\']', full_match)
        
        x = float(x_match.group(1)) if x_match else 0
        y = float(y_match.group(1)) if y_match else 0
        width = float(width_match.group(1)) if width_match else 100
        height = float(height_match.group(1)) if height_match else 24
        
        # 提取内容
        content_match = re.search(r'<foreignObject[^>]*>(.*?)</foreignObject>', 
                                full_match, re.DOTALL | re.IGNORECASE)
        if not content_match:
            return ''
        
        inner_content = content_match.group(1)
        text_content = extract_text_from_html(inner_content)
        
        if not text_content:
            return ''
        
        # 检查是否有样式信息
        style_match = re.search(r'style\s*=\s*["\']([^"\']+)["\']', inner_content)
        style_attr = style_match.group(1) if style_match else ''
        
        # 获取字体大小
        font_size = get_font_size_from_style(style_attr)
        
        # 计算文本位置（相对于foreignObject的坐标系）
        text_x, text_y = calculate_text_position(x, y, width, height)
        
        # 检查是否是集群标签（通常在顶部）
        is_cluster_label = 'cluster-label' in full_match or y < 30
        
        # 为集群标签调整位置
        if is_cluster_label:
            text_y = y + height - 5  # 集群标签通常在矩形的底部
        
        # 生成SVG text元素
        text_element = f'''<text x="{text_x}" y="{text_y}" text-anchor="middle" dominant-baseline="central" font-family="Microsoft YaHei, SimHei, SimSun, Arial, sans-serif" font-size="{font_size}" fill="#333">{text_content}</text>'''
        
        return text_element
        
    except Exception as e:
        print(f"⚠️ Error converting foreignObject: {e}")
        return ''


def convert_svg_foreign_objects(svg_content: str) -> str:
    """
    将SVG内容中的所有foreignObject转换为原生SVG text元素
    
    Args:
        svg_content: SVG文件内容字符串
    
    Returns:
        转换后的SVG内容字符串
    """
    # 匹配所有foreignObject元素
    foreign_object_pattern = r'<foreignObject[^>]*>.*?</foreignObject>'
    
    # 执行转换
    converted_content = re.sub(foreign_object_pattern, convert_foreign_object_to_text, 
                             svg_content, flags=re.DOTALL | re.IGNORECASE)
    
    return converted_content


def convert_svg_file_foreign_objects(input_file: str, output_file: Optional[str] = None) -> bool:
    """
    转换SVG文件中的foreignObject为原生text元素
    
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
        
        # 转换foreignObject
        converted_content = convert_svg_foreign_objects(svg_content)
        
        # 写入文件
        output_path = output_file if output_file else input_file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(converted_content)
        
        return True
        
    except Exception as e:
        print(f"❌ ForeignObject转换失败: {e}")
        return False


def has_foreign_objects(svg_content: str) -> bool:
    """检查SVG内容是否包含foreignObject元素"""
    return '<foreignObject' in svg_content


def get_foreign_object_count(svg_content: str) -> int:
    """统计SVG中foreignObject元素的数量"""
    return len(re.findall(r'<foreignObject[^>]*>', svg_content, re.IGNORECASE))


# 改进版本的转换函数，专门处理Mermaid生成的SVG
def convert_mermaid_foreign_objects(svg_content: str) -> str:
    """
    专门处理Mermaid生成的SVG中的foreignObject
    正确处理transform坐标变换和foreignObject的嵌套关系，并支持多行文本

    关键理解：
    1. foreignObject通常没有x,y属性（默认0,0）
    2. 它们在transform="translate(x,y)"容器内
    3. 最终坐标 = transform坐标 + foreignObject相对坐标
    4. HTML中的<br>标签需要转换为SVG的多行文本
    """

    # 直接匹配整个transform容器和其内部的foreignObject
    pattern = r'<g[^>]*class="[^"]*cluster-label[^"]*"[^>]*transform="translate\(([^,]+),\s*([^)]+)\)"[^>]*>\s*<foreignObject[^>]*width="([^"]+)"[^>]*height="([^"]+)"[^>]*>(.*?)</foreignObject>\s*</g>'

    def replace_cluster_label(match):
        try:
            # 提取transform坐标
            transform_x = float(match.group(1))
            transform_y = float(match.group(2))

            # 提取foreignObject尺寸
            fo_width = float(match.group(3))
            fo_height = float(match.group(4))

            # 提取文本内容并解析换行
            inner_content = match.group(5)
            text_lines = extract_text_lines_from_html(inner_content)

            if not text_lines or not any(text_lines):
                return match.group(0)  # 返回原始内容

            # 创建多行SVG文本
            text_element = create_multiline_svg_text(text_lines, fo_width/2, fo_height/2, fo_width, fo_height)

            # 生成替换的text元素，保持相同的容器结构
            replacement = f'''<g class="cluster-label" transform="translate({transform_x}, {transform_y})">{text_element}</g>'''

            return replacement

        except Exception as e:
            print(f"⚠️ Error converting cluster label: {e}")
            return match.group(0)  # 返回原始内容

    # 处理集群标签
    converted_content = re.sub(pattern, replace_cluster_label, svg_content, flags=re.DOTALL | re.IGNORECASE)

    # 处理节点标签（在transform容器内的foreignObject）
    node_pattern = r'(<g[^>]*class="[^"]*label[^"]*"[^>]*transform="translate\(([^,]+),\s*([^)]+)\)"[^>]*>.*?)<foreignObject[^>]*width="([^"]+)"[^>]*height="([^"]+)"[^>]*>(.*?)</foreignObject>(.*?</g>)'

    def replace_node_label(match):
        try:
            prefix = match.group(1)
            transform_x = float(match.group(2))
            transform_y = float(match.group(3))
            fo_width = float(match.group(4))
            fo_height = float(match.group(5))
            inner_content = match.group(6)
            suffix = match.group(7)

            # 提取文本内容并解析换行
            text_lines = extract_text_lines_from_html(inner_content)
            if not text_lines or not any(text_lines):
                return match.group(0)

            # 节点标签的text元素使用相对坐标
            text_element = create_multiline_svg_text(text_lines, fo_width/2, fo_height/2, fo_width, fo_height)

            return prefix + text_element + suffix

        except Exception as e:
            print(f"⚠️ Error converting node label: {e}")
            return match.group(0)

    # 处理节点标签
    converted_content = re.sub(node_pattern, replace_node_label, converted_content, flags=re.DOTALL | re.IGNORECASE)

    # 处理剩余的foreignObject（备用）
    remaining_pattern = r'<foreignObject[^>]*>.*?</foreignObject>'
    def replace_remaining(match):
        return ''  # 简单删除剩余的foreignObject

    converted_content = re.sub(remaining_pattern, replace_remaining, converted_content, flags=re.DOTALL | re.IGNORECASE)

    return converted_content


if __name__ == "__main__":
    # 测试代码
    test_svg = '''<foreignObject width="64.015625" height="23">
        <div xmlns="http://www.w3.org/1999/xhtml" style="display: inline-block; white-space: nowrap;">
            <span class="nodeLabel">监控维度</span>
        </div>
    </foreignObject>'''
    
    print("ForeignObject转换测试:")
    print("原始:", test_svg)
    print("转换:", convert_svg_foreign_objects(test_svg))
