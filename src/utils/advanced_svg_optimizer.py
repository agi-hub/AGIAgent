#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
高级SVG图优化器
功能：
1. 智能特殊字符转义和验证
2. 文本元素重叠检测和智能重排
3. 边框和线条智能优化
4. 线条穿越文字检测和自动重路由
5. SVG结构优化和清理
6. 生成优化报告

作者：AI Assistant  
日期：2025年9月18日
"""

import re
import xml.etree.ElementTree as ET
from typing import List, Dict, Tuple, Optional, Set
import math
import json
from dataclasses import dataclass
from enum import Enum


class OptimizationLevel(Enum):
    """优化级别"""
    BASIC = "basic"
    STANDARD = "standard"
    AGGRESSIVE = "aggressive"


@dataclass
class OptimizationReport:
    """优化报告"""
    original_issues: List[str]
    fixed_issues: List[str]
    remaining_issues: List[str]
    optimization_stats: Dict[str, int]


class AdvancedSVGOptimizer:
    """高级SVG优化器"""
    
    def __init__(self, optimization_level: OptimizationLevel = OptimizationLevel.STANDARD):
        self.optimization_level = optimization_level
        
        # 特殊字符映射（顺序很重要）
        self.special_chars = {
            '&': '&amp;',   # 必须首先处理
            '<': '&lt;',
            '>': '&gt;',
            '"': '&quot;',
            "'": '&apos;'
        }
        
        # 反向映射用于检测已转义的字符
        self.escaped_chars = {v: k for k, v in self.special_chars.items()}
        
        # 配置参数
        self.config = {
            'min_font_size': 8,
            'max_font_size': 72,
            'min_text_spacing': 3,
            'min_line_width': 0.5,
            'max_line_width': 10,
            'text_line_clearance': 2,
            'overlap_tolerance': 1,
            'auto_resize_text': True,
            'auto_reroute_lines': True
        }
        
        # 统计信息
        self.stats = {
            'chars_escaped': 0,
            'texts_repositioned': 0,
            'lines_optimized': 0,
            'overlaps_fixed': 0,
            'intersections_fixed': 0
        }
    
    def optimize_svg_file(self, input_file: str, output_file: str) -> OptimizationReport:
        """优化SVG文件"""
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                svg_content = f.read()
            
            optimized_svg, report = self.optimize_svg_with_report(svg_content)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(optimized_svg)
            
            print(f"✓ SVG文件已优化并保存到: {output_file}")
            return report
            
        except Exception as e:
            print(f"✗ 文件处理错误: {e}")
            raise
    
    def optimize_svg_with_report(self, svg_content: str) -> Tuple[str, OptimizationReport]:
        """优化SVG并生成详细报告"""
        print("开始高级SVG优化...")
        
        # 重置统计
        self.stats = {k: 0 for k in self.stats}
        original_issues = []
        fixed_issues = []
        remaining_issues = []
        
        # 1. 预检查和问题识别
        original_issues.extend(self._detect_issues(svg_content))
        print(f"✓ 检测到 {len(original_issues)} 个潜在问题")
        
        # 2. 特殊字符处理
        svg_content, char_issues = self._advanced_char_escape(svg_content)
        fixed_issues.extend(char_issues)
        print(f"✓ 处理了 {self.stats['chars_escaped']} 个特殊字符")
        
        try:
            # 3. 解析SVG
            root = ET.fromstring(svg_content)
            
            # 4. 收集和分析元素
            elements = self._collect_and_analyze_elements(root)
            print(f"✓ 分析了 {len(elements)} 个元素")
            
            # 5. 智能文本优化
            text_fixes = self._intelligent_text_optimization(elements)
            fixed_issues.extend(text_fixes)
            print(f"✓ 优化了 {self.stats['texts_repositioned']} 个文本元素")
            
            # 6. 线条和形状优化
            shape_fixes = self._optimize_shapes_and_lines(elements)
            fixed_issues.extend(shape_fixes)
            print(f"✓ 优化了 {self.stats['lines_optimized']} 个线条/形状")
            
            # 7. 交叉问题修复
            intersection_fixes = self._fix_intersections_advanced(elements)
            fixed_issues.extend(intersection_fixes)
            print(f"✓ 修复了 {self.stats['intersections_fixed']} 个交叉问题")
            
            # 8. 布局重叠问题修复
            overlap_fixes = self._fix_layout_overlaps(elements)
            fixed_issues.extend(overlap_fixes)
            print(f"✓ 修复了 {self.stats.get('layout_overlaps_fixed', 0)} 个布局重叠问题")
            
            # 8. SVG结构优化
            if self.optimization_level in [OptimizationLevel.STANDARD, OptimizationLevel.AGGRESSIVE]:
                self._optimize_svg_structure(root)
                print("✓ 完成SVG结构优化")
            
            # 9. 应用所有优化
            self._apply_all_optimizations(root, elements)
            
            # 10. 生成优化后的SVG
            optimized_svg = self._generate_optimized_svg(root)
            
            # 11. 后处理检查
            remaining_issues = self._post_optimization_check(optimized_svg)
            
            print("✓ 高级SVG优化完成")
            
        except ET.ParseError as e:
            print(f"✗ SVG解析错误: {e}")
            remaining_issues.append(f"SVG解析失败: {e}")
            optimized_svg = svg_content
        
        # 生成报告
        report = OptimizationReport(
            original_issues=original_issues,
            fixed_issues=fixed_issues,
            remaining_issues=remaining_issues,
            optimization_stats=self.stats.copy()
        )
        
        return optimized_svg, report
    
    def _detect_issues(self, svg_content: str) -> List[str]:
        """检测SVG中的问题"""
        issues = []
        
        # 检测未转义的特殊字符
        for char in self.special_chars.keys():
            if char in svg_content and f"&{char}" not in svg_content:
                # 避免误报已转义的字符
                pattern = f'[^&]{re.escape(char)}|^{re.escape(char)}'
                if re.search(pattern, svg_content):
                    issues.append(f"发现未转义的特殊字符: '{char}'")
        
        # 解析所有元素的边界框
        elements = self._parse_elements_for_detection(svg_content)
        
        # 检测元素重叠（包括text与rect, rect与rect等）
        for i, elem1 in enumerate(elements):
            for j, elem2 in enumerate(elements[i+1:], i+1):
                if self._elements_overlap(elem1, elem2):
                    issues.append(f"元素重叠: {elem1['type']}({elem1['x']},{elem1['y']}) 与 {elem2['type']}({elem2['x']},{elem2['y']})")
        
        # 检测文本溢出边界框
        for elem in elements:
            if elem['type'] == 'text' and elem.get('parent_rect'):
                parent = elem['parent_rect']
                if elem['text_width'] > parent['width']:
                    issues.append(f"文本溢出: 文本宽度{elem['text_width']:.0f}px 超出框宽{parent['width']}px")
        
        # 检测过细的线条
        thin_lines = re.findall(r'stroke-width="([^"]*)"', svg_content)
        for width in thin_lines:
            try:
                if float(width) < 0.5:
                    issues.append(f"线条过细: stroke-width='{width}'")
            except ValueError:
                continue
        
        return issues
    
    def _parse_elements_for_detection(self, svg_content: str) -> List[Dict]:
        """解析SVG元素用于问题检测"""
        elements = []
        
        # 解析矩形元素
        rect_pattern = r'<rect[^>]*x="([^"]*)"[^>]*y="([^"]*)"[^>]*width="([^"]*)"[^>]*height="([^"]*)"[^>]*>'
        for match in re.finditer(rect_pattern, svg_content):
            try:
                x, y, width, height = map(float, match.groups())
                elements.append({
                    'type': 'rect',
                    'x': x, 'y': y, 'width': width, 'height': height,
                    'bbox': (x, y, x + width, y + height)
                })
            except ValueError:
                continue
        
        # 解析文本元素
        text_pattern = r'<text[^>]*x="([^"]*)"[^>]*y="([^"]*)"[^>]*font-size="([^"]*)"[^>]*>([^<]*)</text>'
        for match in re.finditer(text_pattern, svg_content):
            try:
                x, y, font_size, content = match.groups()
                x, y, font_size = float(x), float(y), float(font_size)
                
                # 估算文本宽度
                text_width = len(content.strip()) * font_size * 0.6
                text_height = font_size * 1.2
                
                # 检查是否在某个矩形内
                parent_rect = None
                for rect_elem in elements:
                    if rect_elem['type'] == 'rect':
                        if (rect_elem['x'] <= x <= rect_elem['x'] + rect_elem['width'] and 
                            rect_elem['y'] <= y <= rect_elem['y'] + rect_elem['height']):
                            parent_rect = rect_elem
                            break
                
                elements.append({
                    'type': 'text',
                    'x': x, 'y': y, 'font_size': font_size,
                    'content': content.strip(),
                    'text_width': text_width,
                    'text_height': text_height,
                    'bbox': (x - text_width/2, y - text_height, x + text_width/2, y),
                    'parent_rect': parent_rect
                })
            except ValueError:
                continue
        
        # 解析圆形元素
        circle_pattern = r'<circle[^>]*cx="([^"]*)"[^>]*cy="([^"]*)"[^>]*r="([^"]*)"[^>]*>'
        for match in re.finditer(circle_pattern, svg_content):
            try:
                cx, cy, r = map(float, match.groups())
                elements.append({
                    'type': 'circle',
                    'x': cx, 'y': cy, 'r': r,
                    'bbox': (cx - r, cy - r, cx + r, cy + r)
                })
            except ValueError:
                continue
        
        return elements
    
    def _elements_overlap(self, elem1: Dict, elem2: Dict) -> bool:
        """检查两个元素是否重叠"""
        # 使用预计算的边界框，如果没有则计算
        if 'bbox' in elem1:
            bbox1 = elem1['bbox']
        else:
            bbox1 = self._calculate_element_bbox(elem1)
            
        if 'bbox' in elem2:
            bbox2 = elem2['bbox']
        else:
            bbox2 = self._calculate_element_bbox(elem2)
        
        # 检查边界框重叠
        return not (bbox1[2] < bbox2[0] or bbox2[2] < bbox1[0] or 
                   bbox1[3] < bbox2[1] or bbox2[3] < bbox1[1])
    
    def _calculate_element_bbox(self, elem: Dict) -> Tuple[float, float, float, float]:
        """计算元素的边界框"""
        if elem['type'] == 'text':
            # 文本元素边界框计算
            if 'text_width' in elem:
                width = elem['text_width']
                height = elem.get('text_height', elem.get('font_size', 12) * 1.2)
            else:
                # 对于没有预计算宽度的文本，估算
                font_size = elem.get('font_size', 12)
                content = elem.get('content', elem.get('text', ''))
                width = len(str(content)) * font_size * 0.6
                height = font_size * 1.2
            
            # 假设text-anchor="middle"
            x, y = elem['x'], elem['y']
            return (x - width/2, y - height, x + width/2, y)
            
        elif elem['type'] == 'rect':
            x, y = elem['x'], elem['y']
            width, height = elem['width'], elem['height']
            return (x, y, x + width, y + height)
            
        elif elem['type'] == 'circle':
            x, y, r = elem['x'], elem['y'], elem['r']
            return (x - r, y - r, x + r, y + r)
            
        else:
            # 默认边界框
            return (elem.get('x', 0), elem.get('y', 0), 
                   elem.get('x', 0) + elem.get('width', 0), 
                   elem.get('y', 0) + elem.get('height', 0))
    
    def _advanced_char_escape(self, svg_content: str) -> Tuple[str, List[str]]:
        """高级特殊字符转义"""
        fixed_issues = []
        
        # 先检查哪些字符需要转义
        chars_to_escape = set()
        
        # 在文本内容中查找需要转义的字符
        text_pattern = r'<(text|tspan)[^>]*>([^<]*)</(?:text|tspan)>'
        for match in re.finditer(text_pattern, svg_content):
            text_content = match.group(2)
            for char in self.special_chars.keys():
                if char in text_content:
                    # 检查是否已经转义
                    escaped_form = self.special_chars[char]
                    if escaped_form not in text_content or text_content.count(char) > text_content.count(escaped_form):
                        chars_to_escape.add(char)
        
        # 执行转义（按顺序处理）
        for char in ['&', '<', '>', '"', "'"]:  # 固定顺序
            if char in chars_to_escape:
                escaped_form = self.special_chars[char]
                
                def escape_in_text(match):
                    tag_start = match.group(1)
                    text_content = match.group(2)
                    tag_end = match.group(3)
                    
                    # 只转义未转义的字符
                    if char == '&':
                        # 特殊处理&符号，避免重复转义
                        text_content = re.sub(r'&(?!(?:amp|lt|gt|quot|apos);)', '&amp;', text_content)
                    else:
                        text_content = text_content.replace(char, escaped_form)
                    
                    return f"{tag_start}{text_content}{tag_end}"
                
                old_content = svg_content
                svg_content = re.sub(r'(<(?:text|tspan)[^>]*>)([^<]*)(</(?:text|tspan)>)', 
                                   escape_in_text, svg_content)
                
                if svg_content != old_content:
                    count = old_content.count(char) - svg_content.count(char)
                    self.stats['chars_escaped'] += count
                    fixed_issues.append(f"转义了 {count} 个 '{char}' 字符")
        
        return svg_content, fixed_issues
    
    def _collect_and_analyze_elements(self, root: ET.Element) -> List[Dict]:
        """收集和分析SVG元素"""
        elements = []
        
        for elem in root.iter():
            tag_name = elem.tag.split('}')[-1] if '}' in elem.tag else elem.tag
            
            if tag_name in ['text', 'tspan']:
                elements.append(self._analyze_text_element(elem))
            elif tag_name in ['rect', 'circle', 'ellipse', 'polygon', 'path']:
                elements.append(self._analyze_shape_element(elem))
            elif tag_name in ['line', 'polyline']:
                elements.append(self._analyze_line_element(elem))
        
        return elements
    
    def _analyze_text_element(self, elem: ET.Element) -> Dict:
        """分析文本元素"""
        x = self._safe_float(elem.get('x', '0'))
        y = self._safe_float(elem.get('y', '0'))
        font_size = self._safe_float(elem.get('font-size', '12'))
        font_family = elem.get('font-family', 'sans-serif')
        text_content = (elem.text or '').strip()
        
        # 更精确的文本尺寸估算
        char_width_ratio = self._get_font_width_ratio(font_family)
        text_width = len(text_content) * font_size * char_width_ratio
        text_height = font_size * 1.2  # 包含行高
        
        # 计算精确边界框
        bbox = (x, y - font_size, x + text_width, y + font_size * 0.2)
        
        return {
            'type': 'text',
            'element': elem,
            'x': x, 'y': y,
            'width': text_width,
            'height': text_height,
            'font_size': font_size,
            'font_family': font_family,
            'content': text_content,
            'bbox': bbox,
            'original_pos': (x, y),
            'needs_adjustment': False
        }
    
    def _analyze_shape_element(self, elem: ET.Element) -> Dict:
        """分析形状元素"""
        tag_name = elem.tag.split('}')[-1] if '}' in elem.tag else elem.tag
        
        bbox = self._calculate_shape_bbox(elem, tag_name)
        stroke_width = self._safe_float(elem.get('stroke-width', '1'))
        
        return {
            'type': 'shape',
            'element': elem,
            'tag': tag_name,
            'bbox': bbox,
            'stroke_width': stroke_width,
            'needs_optimization': stroke_width < self.config['min_line_width']
        }
    
    def _analyze_line_element(self, elem: ET.Element) -> Dict:
        """分析线条元素"""
        tag_name = elem.tag.split('}')[-1] if '}' in elem.tag else elem.tag
        
        points = self._extract_line_points(elem, tag_name)
        stroke_width = self._safe_float(elem.get('stroke-width', '1'))
        
        return {
            'type': 'line',
            'element': elem,
            'tag': tag_name,
            'points': points,
            'stroke_width': stroke_width,
            'original_points': points.copy(),
            'needs_rerouting': False
        }
    
    def _intelligent_text_optimization(self, elements: List[Dict]) -> List[str]:
        """智能文本优化"""
        fixed_issues = []
        text_elements = [e for e in elements if e['type'] == 'text']
        
        # 检测重叠
        for i, text1 in enumerate(text_elements):
            for j, text2 in enumerate(text_elements[i+1:], i+1):
                if self._texts_overlap(text1, text2):
                    # 智能重新定位
                    self._smart_text_reposition(text1, text2, text_elements)
                    self.stats['overlaps_fixed'] += 1
                    fixed_issues.append(f"修复文本重叠: '{text1['content'][:20]}...' 和 '{text2['content'][:20]}...'")
        
        # 字体大小优化
        for text_elem in text_elements:
            if text_elem['font_size'] < self.config['min_font_size']:
                text_elem['font_size'] = self.config['min_font_size']
                text_elem['needs_adjustment'] = True
                fixed_issues.append(f"调整过小字体: {text_elem['content'][:20]}...")
        
        return fixed_issues
    
    def _optimize_shapes_and_lines(self, elements: List[Dict]) -> List[str]:
        """优化形状和线条"""
        fixed_issues = []
        
        for elem in elements:
            if elem['type'] in ['shape', 'line']:
                if elem.get('stroke_width', 1) < self.config['min_line_width']:
                    elem['stroke_width'] = self.config['min_line_width']
                    elem['element'].set('stroke-width', str(self.config['min_line_width']))
                    self.stats['lines_optimized'] += 1
                    fixed_issues.append(f"调整过细线条宽度: {elem.get('tag', 'unknown')}")
                
                # 确保有描边颜色
                if not elem['element'].get('stroke'):
                    elem['element'].set('stroke', '#000000')
                    fixed_issues.append(f"添加缺失的描边颜色: {elem.get('tag', 'unknown')}")
        
        return fixed_issues
    
    def _fix_intersections_advanced(self, elements: List[Dict]) -> List[str]:
        """高级交叉问题修复"""
        fixed_issues = []
        
        text_elements = [e for e in elements if e['type'] == 'text']
        line_elements = [e for e in elements if e['type'] == 'line']
        
        for line_elem in line_elements:
            for text_elem in text_elements:
                if self._line_intersects_text_advanced(line_elem, text_elem):
                    if self.config['auto_reroute_lines']:
                        self._reroute_line_around_text(line_elem, text_elem)
                        self.stats['intersections_fixed'] += 1
                        fixed_issues.append(f"重新路由线条避开文本: '{text_elem['content'][:20]}...'")
        
        return fixed_issues
    
    def _texts_overlap(self, text1: Dict, text2: Dict) -> bool:
        """检查两个文本是否重叠"""
        bbox1 = text1['bbox']
        bbox2 = text2['bbox']
        
        # 添加容差
        tolerance = self.config['overlap_tolerance']
        
        return not (bbox1[2] + tolerance < bbox2[0] or 
                   bbox2[2] + tolerance < bbox1[0] or 
                   bbox1[3] + tolerance < bbox2[1] or 
                   bbox2[3] + tolerance < bbox1[1])
    
    def _smart_text_reposition(self, text1: Dict, text2: Dict, all_texts: List[Dict]) -> None:
        """智能文本重新定位"""
        # 选择移动哪个文本（通常移动后面的）
        text_to_move = text2
        
        # 计算新位置
        new_y = text1['bbox'][3] + self.config['min_text_spacing'] + text_to_move['height']
        
        # 检查新位置是否与其他文本冲突
        attempts = 0
        while attempts < 5:
            temp_bbox = (text_to_move['x'], new_y - text_to_move['height'], 
                        text_to_move['x'] + text_to_move['width'], new_y)
            
            conflict = False
            for other_text in all_texts:
                if other_text != text_to_move and other_text != text1:
                    if self._boxes_overlap(temp_bbox, other_text['bbox']):
                        conflict = True
                        break
            
            if not conflict:
                break
            
            new_y += text_to_move['height'] + self.config['min_text_spacing']
            attempts += 1
        
        # 应用新位置
        text_to_move['y'] = new_y
        text_to_move['bbox'] = (text_to_move['x'], new_y - text_to_move['height'],
                               text_to_move['x'] + text_to_move['width'], new_y)
        text_to_move['needs_adjustment'] = True
        self.stats['texts_repositioned'] += 1
    
    def _line_intersects_text_advanced(self, line_elem: Dict, text_elem: Dict) -> bool:
        """高级线条文本交叉检测"""
        text_bbox = text_elem['bbox']
        clearance = self.config['text_line_clearance']
        
        # 扩展文本边界框以包含间隙
        expanded_bbox = (text_bbox[0] - clearance, text_bbox[1] - clearance,
                        text_bbox[2] + clearance, text_bbox[3] + clearance)
        
        points = line_elem['points']
        for i in range(len(points) - 1):
            if self._line_segment_intersects_box_advanced(points[i], points[i+1], expanded_bbox):
                return True
        
        return False
    
    def _reroute_line_around_text(self, line_elem: Dict, text_elem: Dict) -> None:
        """重新路由线条绕过文本"""
        # 简化的重路由策略：在文本周围添加控制点
        text_bbox = text_elem['bbox']
        points = line_elem['points']
        
        new_points = []
        for i, point in enumerate(points):
            new_points.append(point)
            
            # 如果下一个点会穿过文本，添加绕行点
            if i < len(points) - 1:
                next_point = points[i + 1]
                if self._line_segment_intersects_box_advanced(point, next_point, text_bbox):
                    # 添加绕行点（简化版本）
                    bypass_y = text_bbox[3] + self.config['text_line_clearance']
                    new_points.append((point[0], bypass_y))
                    new_points.append((next_point[0], bypass_y))
        
        line_elem['points'] = new_points
        line_elem['needs_rerouting'] = True
    
    def _fix_layout_overlaps(self, elements: List[Dict]) -> List[str]:
        """修复布局重叠问题"""
        fixed_issues = []
        self.stats['layout_overlaps_fixed'] = 0
        
        # 检测并修复元素重叠
        for i, elem1 in enumerate(elements):
            for j, elem2 in enumerate(elements[i+1:], i+1):
                if self._elements_overlap(elem1, elem2):
                    # 优先调整文本框位置，避免调整图形元素
                    if elem1['type'] == 'rect' and elem2['type'] == 'text':
                        self._adjust_text_to_avoid_rect(elem2, elem1)
                        fixed_issues.append(f"调整文本位置避免与矩形重叠")
                        self.stats['layout_overlaps_fixed'] += 1
                    elif elem1['type'] == 'text' and elem2['type'] == 'rect':
                        self._adjust_text_to_avoid_rect(elem1, elem2)
                        fixed_issues.append(f"调整文本位置避免与矩形重叠")
                        self.stats['layout_overlaps_fixed'] += 1
                    elif elem1['type'] == 'rect' and elem2['type'] == 'rect':
                        # 对于重叠的矩形，尝试调整较小的那个
                        smaller_rect = elem1 if elem1['width'] * elem1['height'] < elem2['width'] * elem2['height'] else elem2
                        self._adjust_rect_position(smaller_rect, elements)
                        fixed_issues.append(f"调整矩形位置避免重叠")
                        self.stats['layout_overlaps_fixed'] += 1
        
        # 检测并修复文本溢出
        for elem in elements:
            if elem['type'] == 'text' and elem.get('parent_rect'):
                parent = elem['parent_rect']
                text_width = elem.get('text_width', len(str(elem.get('content', elem.get('text', '')))) * elem.get('font_size', 12) * 0.6)
                if text_width > parent['width']:
                    # 缩小字体或扩大矩形
                    if self.optimization_level == OptimizationLevel.AGGRESSIVE:
                        # 扩大矩形宽度
                        new_width = text_width + 20  # 添加一些边距
                        old_width = parent['width']
                        parent['width'] = new_width
                        # 调整矩形x位置保持居中
                        parent['x'] -= (new_width - old_width) / 2
                        fixed_issues.append(f"扩大矩形宽度以容纳文本: {new_width:.0f}px")
                        self.stats['layout_overlaps_fixed'] += 1
                    else:
                        # 缩小字体
                        scale_factor = parent['width'] / text_width * 0.9  # 留10%边距
                        elem['font_size'] = elem.get('font_size', 12) * scale_factor
                        if 'text_width' in elem:
                            elem['text_width'] *= scale_factor
                        fixed_issues.append(f"缩小字体以适应矩形宽度: {elem['font_size']:.1f}px")
                        self.stats['layout_overlaps_fixed'] += 1
        
        return fixed_issues
    
    def _adjust_text_to_avoid_rect(self, text_elem: Dict, rect_elem: Dict) -> None:
        """调整文本位置避免与矩形重叠"""
        # 简单策略：将文本移动到矩形上方或下方
        rect_top = rect_elem['y']
        rect_bottom = rect_elem['y'] + rect_elem['height']
        text_height = text_elem.get('text_height', text_elem.get('font_size', 12) * 1.2)
        
        # 检查文本移动到矩形上方是否更好
        if text_elem['y'] > rect_elem['y']:
            # 移动到矩形下方
            text_elem['y'] = rect_bottom + text_height + 10
        else:
            # 移动到矩形上方
            text_elem['y'] = rect_top - 10
        
        text_elem['needs_adjustment'] = True
    
    def _adjust_rect_position(self, rect_elem: Dict, all_elements: List[Dict]) -> None:
        """调整矩形位置避免重叠"""
        # 简单策略：向右移动矩形
        original_x = rect_elem['x']
        rect_elem['x'] += rect_elem['width'] + 20  # 移动一个宽度加间距
        
        # 检查新位置是否与其他元素冲突
        conflicts = 0
        for other_elem in all_elements:
            if other_elem != rect_elem and self._elements_overlap(rect_elem, other_elem):
                conflicts += 1
        
        # 如果冲突更多，恢复原位置
        if conflicts > 1:
            rect_elem['x'] = original_x
        else:
            rect_elem['needs_adjustment'] = True
    
    def _apply_all_optimizations(self, root: ET.Element, elements: List[Dict]) -> None:
        """应用所有优化"""
        for elem_data in elements:
            element = elem_data['element']
            
            if elem_data['type'] == 'text' and elem_data.get('needs_adjustment'):
                element.set('x', str(elem_data['x']))
                element.set('y', str(elem_data['y']))
                element.set('font-size', str(elem_data['font_size']))
            
            elif elem_data['type'] == 'line' and elem_data.get('needs_rerouting'):
                self._update_line_element(element, elem_data)
    
    def _update_line_element(self, element: ET.Element, line_data: Dict) -> None:
        """更新线条元素"""
        tag_name = line_data['tag']
        points = line_data['points']
        
        if tag_name == 'line' and len(points) >= 2:
            element.set('x1', str(points[0][0]))
            element.set('y1', str(points[0][1]))
            element.set('x2', str(points[-1][0]))
            element.set('y2', str(points[-1][1]))
        elif tag_name == 'polyline':
            points_str = ' '.join([f"{p[0]},{p[1]}" for p in points])
            element.set('points', points_str)
    
    def _generate_optimized_svg(self, root: ET.Element) -> str:
        """生成优化后的SVG"""
        # 添加XML声明和格式化
        ET.register_namespace('', 'http://www.w3.org/2000/svg')
        svg_str = ET.tostring(root, encoding='unicode')
        
        # 添加XML声明如果缺失
        if not svg_str.startswith('<?xml'):
            svg_str = '<?xml version="1.0" encoding="UTF-8"?>\n' + svg_str
        
        return svg_str
    
    def _post_optimization_check(self, svg_content: str) -> List[str]:
        """优化后检查"""
        remaining_issues = []
        
        # 检查是否还有未转义的字符
        for char in ['<', '>', '&']:
            if char in svg_content:
                # 更精确的检查
                if char == '&' and re.search(r'&(?!(?:amp|lt|gt|quot|apos);)', svg_content):
                    remaining_issues.append(f"仍有未转义的字符: '{char}'")
                elif char in ['<', '>']:
                    # 检查是否在标签外
                    if re.search(f'>[^<]*{re.escape(char)}[^>]*<', svg_content):
                        remaining_issues.append(f"仍有未转义的字符: '{char}'")
        
        return remaining_issues
    
    # 辅助方法
    def _safe_float(self, value: str, default: float = 0.0) -> float:
        """安全转换为浮点数"""
        try:
            return float(value)
        except (ValueError, TypeError):
            return default
    
    def _get_font_width_ratio(self, font_family: str) -> float:
        """获取字体宽度比例"""
        ratios = {
            'monospace': 0.6,
            'serif': 0.55,
            'sans-serif': 0.5,
            'arial': 0.5,
            'helvetica': 0.5,
            'times': 0.55,
            'courier': 0.6
        }
        return ratios.get(font_family.lower(), 0.5)
    
    def _calculate_shape_bbox(self, elem: ET.Element, tag_name: str) -> Tuple[float, float, float, float]:
        """计算形状边界框"""
        if tag_name == 'rect':
            x = self._safe_float(elem.get('x', '0'))
            y = self._safe_float(elem.get('y', '0'))
            width = self._safe_float(elem.get('width', '0'))
            height = self._safe_float(elem.get('height', '0'))
            return (x, y, x + width, y + height)
        elif tag_name == 'circle':
            cx = self._safe_float(elem.get('cx', '0'))
            cy = self._safe_float(elem.get('cy', '0'))
            r = self._safe_float(elem.get('r', '0'))
            return (cx - r, cy - r, cx + r, cy + r)
        else:
            return (0, 0, 100, 100)  # 默认边界框
    
    def _extract_line_points(self, elem: ET.Element, tag_name: str) -> List[Tuple[float, float]]:
        """提取线条点"""
        if tag_name == 'line':
            x1 = self._safe_float(elem.get('x1', '0'))
            y1 = self._safe_float(elem.get('y1', '0'))
            x2 = self._safe_float(elem.get('x2', '0'))
            y2 = self._safe_float(elem.get('y2', '0'))
            return [(x1, y1), (x2, y2)]
        elif tag_name == 'polyline':
            points_str = elem.get('points', '')
            return self._parse_points_string(points_str)
        return []
    
    def _parse_points_string(self, points_str: str) -> List[Tuple[float, float]]:
        """解析点字符串"""
        points = []
        # 支持多种格式：x,y x,y 或 x y x y
        coords = re.findall(r'[\d.-]+', points_str)
        for i in range(0, len(coords) - 1, 2):
            x = self._safe_float(coords[i])
            y = self._safe_float(coords[i + 1])
            points.append((x, y))
        return points
    
    def _boxes_overlap(self, bbox1: Tuple[float, float, float, float], 
                       bbox2: Tuple[float, float, float, float]) -> bool:
        """检查边界框重叠"""
        return not (bbox1[2] < bbox2[0] or bbox2[2] < bbox1[0] or 
                   bbox1[3] < bbox2[1] or bbox2[3] < bbox1[1])
    
    def _line_segment_intersects_box_advanced(self, p1: Tuple[float, float], 
                                            p2: Tuple[float, float],
                                            bbox: Tuple[float, float, float, float]) -> bool:
        """高级线段边界框相交检测"""
        x1, y1 = p1
        x2, y2 = p2
        box_x1, box_y1, box_x2, box_y2 = bbox
        
        # 检查线段端点是否在框内
        if (box_x1 <= x1 <= box_x2 and box_y1 <= y1 <= box_y2) or \
           (box_x1 <= x2 <= box_x2 and box_y1 <= y2 <= box_y2):
            return True
        
        # 检查线段是否与框的边相交
        return self._line_intersects_rectangle(p1, p2, bbox)
    
    def _line_intersects_rectangle(self, p1: Tuple[float, float], 
                                  p2: Tuple[float, float],
                                  bbox: Tuple[float, float, float, float]) -> bool:
        """检查线段是否与矩形相交"""
        x1, y1 = p1
        x2, y2 = p2
        box_x1, box_y1, box_x2, box_y2 = bbox
        
        # 使用简化的相交测试
        line_min_x, line_max_x = min(x1, x2), max(x1, x2)
        line_min_y, line_max_y = min(y1, y2), max(y1, y2)
        
        return not (line_max_x < box_x1 or line_min_x > box_x2 or 
                   line_max_y < box_y1 or line_min_y > box_y2)
    
    def _optimize_svg_structure(self, root: ET.Element) -> None:
        """优化SVG结构"""
        # 移除空的文本元素
        for elem in root.iter():
            if elem.tag.endswith('text') and not (elem.text or '').strip():
                parent = root.find(f".//{elem.tag}/..")
                if parent is not None:
                    parent.remove(elem)
    
    def print_optimization_report(self, report: OptimizationReport) -> None:
        """打印优化报告"""
        print("\n" + "="*60)
        print("SVG 优化报告")
        print("="*60)
        
        print(f"\n原始问题 ({len(report.original_issues)}):")
        for issue in report.original_issues:
            print(f"  • {issue}")
        
        print(f"\n已修复问题 ({len(report.fixed_issues)}):")
        for fix in report.fixed_issues:
            print(f"  ✓ {fix}")
        
        if report.remaining_issues:
            print(f"\n剩余问题 ({len(report.remaining_issues)}):")
            for issue in report.remaining_issues:
                print(f"  ⚠ {issue}")
        
        print(f"\n优化统计:")
        for key, value in report.optimization_stats.items():
            print(f"  {key}: {value}")
        
        print("="*60)


def main():
    """主函数 - 使用示例"""
    # 创建测试SVG
    test_svg = '''<?xml version="1.0" encoding="UTF-8"?>
<svg width="500" height="400" xmlns="http://www.w3.org/2000/svg">
    <!-- 测试特殊字符 -->
    <text x="50" y="50" font-size="14">Hello & World < > " '</text>
    
    <!-- 测试文本重叠 -->
    <text x="60" y="80" font-size="12">重叠文本1</text>
    <text x="65" y="85" font-size="12">重叠文本2</text>
    
    <!-- 测试过细线条 -->
    <line x1="0" y1="75" x2="200" y2="75" stroke="red" stroke-width="0.1"/>
    
    <!-- 测试线条穿越文字 -->
    <text x="100" y="120" font-size="16">被线条穿过的文字</text>
    <line x1="50" y1="115" x2="250" y2="125" stroke="blue" stroke-width="2"/>
    
    <!-- 测试形状 -->
    <rect x="300" y="100" width="100" height="50" fill="lightblue" stroke-width="0.2"/>
    <circle cx="200" cy="200" r="30" fill="yellow"/>
    
    <!-- 测试更多特殊字符 -->
    <text x="50" y="300" font-size="12">测试 &amp; 已转义 < 未转义</text>
</svg>'''
    
    # 创建优化器
    optimizer = AdvancedSVGOptimizer(OptimizationLevel.STANDARD)
    
    # 执行优化
    optimized_svg, report = optimizer.optimize_svg_with_report(test_svg)
    
    # 保存结果
    with open('/home/zhenzhi.wu/AGIAgent/test_original.svg', 'w', encoding='utf-8') as f:
        f.write(test_svg)
    
    with open('/home/zhenzhi.wu/AGIAgent/test_optimized.svg', 'w', encoding='utf-8') as f:
        f.write(optimized_svg)
    
    # 打印报告
    optimizer.print_optimization_report(report)
    
    print(f"\n文件已保存:")
    print(f"  原始SVG: /home/zhenzhi.wu/AGIAgent/test_original.svg")
    print(f"  优化后SVG: /home/zhenzhi.wu/AGIAgent/test_optimized.svg")


if __name__ == "__main__":
    main()
