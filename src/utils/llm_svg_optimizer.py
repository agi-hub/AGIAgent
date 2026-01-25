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

文生图SVG优化器
使用图像生成API将SVG源代码转换为图像
"""

import os
import base64
import requests
import re
from typing import Dict, Optional, Tuple
from pathlib import Path


class ImageGenerationSVGOptimizer:
    """文生图驱动的SVG优化器"""
    
    def __init__(self, api_key: str, api_base: str, model: str):
        """
        初始化文生图优化器
        
        Args:
            api_key: API密钥
            api_base: API基础URL
            model: 模型名称
        """
        self.api_key = api_key
        self.api_base = api_base.rstrip('/')
        self.model = model
        
    def generate_image_from_svg(self, svg_content: str, original_file_path: Optional[str] = None) -> Tuple[str, Dict]:
        """
        从SVG源代码生成图像
        
        Args:
            svg_content: SVG源代码
            original_file_path: 原始SVG文件路径（用于生成输出文件名）
            
        Returns:
            Tuple[生成的图像文件路径, 生成报告]
        """
        print("🎨 开始生成图像...")
        
        # 构建提示词：将SVG代码转换为自然语言描述
        prompt = self._svg_to_prompt(svg_content)
        
        print(f"📝 提示词: {prompt[:100]}...")
        print("🤖 调用文生图API...")
        
        # 调用文生图API
        image_data, image_format = self._call_image_generation_api(prompt)
        
        # 生成输出文件路径
        if original_file_path:
            output_path = self._generate_output_path(original_file_path)
        else:
            # 如果没有原始路径，使用临时路径
            output_path = "svg_generated_image.png"
        
        # 保存图像文件
        self._save_image(image_data, output_path, image_format)
        
        print(f"💾 图像已保存到: {output_path}")
        
        report = {
            "method": "ImageGeneration",
            "model": self.model,
            "api_base": self.api_base,
            "output_path": output_path,
            "image_format": image_format
        }
        
        return output_path, report
    
    def _detect_language(self, text: str) -> str:
        """
        检测文本的主要语言
        
        Args:
            text: 文本内容
            
        Returns:
            语言代码: 'zh' (中文), 'en' (英文), 'other'
        """
        if not text or not text.strip():
            return 'en'  # 默认英文
        
        # 统计中文字符
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
        total_chars = len(re.findall(r'[a-zA-Z\u4e00-\u9fff]', text))
        
        if total_chars == 0:
            return 'en'
        
        # 如果中文字符占比超过30%，认为是中文
        if chinese_chars / total_chars > 0.3:
            return 'zh'
        
        return 'en'
    
    def _svg_to_prompt(self, svg_content: str) -> str:
        """
        将SVG代码转换为文生图提示词（英文）
        
        Args:
            svg_content: SVG源代码
            
        Returns:
            提示词字符串（英文）
        """
        # 提取所有文本元素
        text_elements = re.findall(r'<text[^>]*>(.*?)</text>', svg_content, re.DOTALL)
        text_content = ' '.join([t.strip() for t in text_elements if t.strip()])
        
        # 检测语言
        detected_language = self._detect_language(text_content)
        language_name = "Chinese" if detected_language == 'zh' else "English"
        
        # 提取形状和结构信息
        has_circle = '<circle' in svg_content
        has_rect = '<rect' in svg_content
        has_path = '<path' in svg_content
        has_line = '<line' in svg_content
        has_polygon = '<polygon' in svg_content
        has_ellipse = '<ellipse' in svg_content
        
        # 分析SVG结构，提取逻辑关系
        connections = len(re.findall(r'<line[^>]*>', svg_content))
        arrows = len(re.findall(r'marker-end|marker-start', svg_content))
        
        # 构建英文提示词
        prompt_parts = []
        
        # Style requirements (highest priority)
        prompt_parts.append(
            "Create a diagram illustration in a schematic style that is fresh, exquisite, "
            "rich in details, suitable for high-level academic publications, "
            "similar to the drawing style of Nature and Science journals, "
            "professional, scientific, and rigorous."
        )
        
        # Language specification
        if text_content:
            prompt_parts.append(
                f"The text labels and descriptions in the SVG are in {language_name}. "
                f"Please generate the image using the same language ({language_name}). "
                f"The text content includes: {text_content}"
            )
        else:
            prompt_parts.append(
                f"Please generate the image using {language_name} language."
            )
        
        # Describe graphical elements
        shape_descriptions = []
        if has_circle:
            shape_descriptions.append("circles")
        if has_rect:
            shape_descriptions.append("rectangles")
        if has_ellipse:
            shape_descriptions.append("ellipses")
        if has_path:
            shape_descriptions.append("complex paths")
        if has_polygon:
            shape_descriptions.append("polygons")
        if has_line:
            if arrows > 0:
                shape_descriptions.append("arrowed connecting lines")
            else:
                shape_descriptions.append("connecting lines")
        
        if shape_descriptions:
            prompt_parts.append(
                f"The diagram contains the following graphical elements: {', '.join(shape_descriptions)}."
            )
        
        # Describe logical relationships
        if connections > 0:
            prompt_parts.append(
                f"Elements are connected through {connections} connecting line(s) to establish logical relationships."
            )
        
        # If no useful information extracted, analyze SVG structure
        if not text_content and not shape_descriptions:
            svg_preview = svg_content[:800].replace('\n', ' ').replace('\r', ' ')
            prompt_parts.append(
                f"Please draw based on the structure and logical relationships of the following SVG code: {svg_preview}"
            )
        else:
            # Add logical relationship requirements
            prompt_parts.append(
                "Please draw according to the expression logic described in the SVG source code, "
                "maintaining the logical relationships and hierarchical structure between elements."
            )
        
        # Quality requirements
        prompt_parts.append(
            "Requirements: Use high-quality color schemes, ensure text is clear and readable, "
            "maintain balanced and beautiful overall layout, pay attention to detail representation, "
            "and meet the visual standards of scientific publications. "
            "The background must be pure white (#FFFFFF)."
        )
        
        prompt = " ".join(prompt_parts)
        
        return prompt
    
    def _call_image_generation_api(self, prompt: str) -> Tuple[bytes, str]:
        """
        调用文生图API
        
        Args:
            prompt: 提示词
            
        Returns:
            Tuple[图像二进制数据, 图像格式]
        """
        # 构建API端点
        # SVG优化器只支持OpenRouter格式（使用vision API）
        api_base = self.api_base.rstrip('/')
        
        # OpenRouter使用chat completions API
        if '/chat/completions' in api_base or '/v1/chat' in api_base:
            api_url = api_base
        elif '/v1' in api_base:
            api_url = f"{api_base}/chat/completions"
        elif 'openrouter.ai' in api_base:
            api_url = f"{api_base}/v1/chat/completions"
        else:
            api_url = f"{api_base}/v1/chat/completions"
        
        print(f"🔗 API端点: {api_url}")
        print(f"🤖 使用模型: {self.model}")
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/AGIAgent",
            "X-Title": "AGIAgent SVG Image Generation"
        }
        
        # OpenRouter的图像生成需要使用chat completions API，并设置modalities
        data = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "modalities": ["image", "text"],  # 关键：指定生成图像
            "max_tokens": 4096
        }
        
        print(f"📤 请求数据: model={data['model']}, modalities={data['modalities']}")
        
        try:
            response = requests.post(
                api_url,
                headers=headers,
                json=data,
                timeout=120  # 图像生成可能需要更长时间
            )
            
            print(f"📥 响应状态码: {response.status_code}")
            
            if response.status_code != 200:
                error_msg = response.text
                print(f"❌ API错误响应: {error_msg}")
                try:
                    error_json = response.json()
                    error_msg = error_json.get('error', {}).get('message', error_msg) or error_json.get('detail', error_msg)
                except:
                    pass
                
                # 如果是404错误，可能是端点不正确
                if response.status_code == 404:
                    raise Exception(
                        f"API端点不存在 (404)。\n"
                        f"可能的原因：\n"
                        f"1. API base URL配置不正确: {self.api_base}\n"
                        f"2. 请检查config.txt中的svg_optimizer_api_base配置\n"
                        f"3. 建议使用OpenRouter官方API: https://openrouter.ai/api/v1\n"
                        f"原始错误: {error_msg}"
                    )
                
                raise Exception(f"文生图API错误: {response.status_code} - {error_msg}")
            
            result = response.json()
            
            # 调试：打印API响应结构
            print(f"🔍 API响应结构: {list(result.keys())}")
            
            # OpenRouter响应格式
            if 'choices' in result:
                print(f"🔍 choices字段类型: {type(result['choices'])}, 长度: {len(result['choices']) if isinstance(result['choices'], list) else 'N/A'}")
                if isinstance(result['choices'], list) and len(result['choices']) > 0:
                    choice = result['choices'][0]
                    print(f"🔍 choice结构: {list(choice.keys())}")
                    if 'message' in choice:
                        print(f"🔍 message结构: {list(choice['message'].keys())}")
                        if 'content' in choice['message']:
                            content = choice['message']['content']
                            print(f"🔍 content类型: {type(content)}")
                            if isinstance(content, list):
                                print(f"🔍 content[0]结构: {list(content[0].keys()) if isinstance(content[0], dict) else 'not dict'}")
            
            # 提取图像数据 - OpenRouter返回格式: {"choices": [{"message": {"content": [{"type": "image", "image_url": {"url": "data:image/..."}}]}}]}
            image_data = None
            image_format = 'png'
            
            # OpenRouter chat completions格式
            if 'choices' in result and isinstance(result['choices'], list) and len(result['choices']) > 0:
                # 遍历所有choices，查找图像
                for choice_idx, choice in enumerate(result['choices']):
                    print(f"🔍 处理choice[{choice_idx}]")
                    if 'message' in choice:
                        message = choice['message']
                        print(f"🔍 message结构: {list(message.keys())}")
                        
                        # 优先检查images字段（OpenRouter可能在这里返回图像）
                        if 'images' in message:
                            images_value = message['images']
                            print(f"🔍 发现images字段，类型: {type(images_value)}, 值: {str(images_value)[:200] if isinstance(images_value, str) else images_value}")
                            
                            if images_value:
                                images_list = images_value
                                if isinstance(images_list, list) and len(images_list) > 0:
                                    # 取第一个图像
                                    image_item = images_list[0]
                                    print(f"🔍 images[0]类型: {type(image_item)}")
                                    if isinstance(image_item, dict):
                                        print(f"🔍 images[0]结构: {list(image_item.keys())}")
                                        # 打印所有字段的前100个字符
                                        for key, value in image_item.items():
                                            if isinstance(value, str):
                                                print(f"🔍   images[0][{key}]: {value[:100]}...")
                                            elif isinstance(value, dict):
                                                print(f"🔍   images[0][{key}]: dict with keys {list(value.keys())}")
                                                # 如果是字典，打印其内容的前100个字符
                                                for sub_key, sub_value in value.items():
                                                    if isinstance(sub_value, str):
                                                        print(f"🔍     images[0][{key}][{sub_key}]: {sub_value[:100]}...")
                                            else:
                                                print(f"🔍   images[0][{key}]: {type(value)}")
                                        
                                        # 检查image_url字段（OpenRouter格式）
                                        if 'image_url' in image_item:
                                            image_url_obj = image_item['image_url']
                                            if isinstance(image_url_obj, dict):
                                                # image_url是一个字典，包含url字段
                                                image_data_b64 = image_url_obj.get('url', '')
                                                print(f"🔍 从image_url.url提取数据，长度: {len(image_data_b64) if isinstance(image_data_b64, str) else 'N/A'}")
                                            else:
                                                image_data_b64 = str(image_url_obj)
                                        else:
                                            # 可能包含url、data、base64等字段
                                            image_data_b64 = image_item.get('url') or image_item.get('data') or image_item.get('base64') or image_item.get('image') or image_item.get('content')
                                        
                                        if image_data_b64:
                                            if isinstance(image_data_b64, str):
                                                if image_data_b64.startswith('data:image'):
                                                    header, encoded = image_data_b64.split(',', 1)
                                                    image_format = header.split('/')[1].split(';')[0]
                                                    image_data = base64.b64decode(encoded)
                                                    print(f"✅ 从images[0]提取图像成功（data URI）")
                                                    break
                                                elif len(image_data_b64) > 100:
                                                    # 可能是纯base64字符串
                                                    try:
                                                        image_data = base64.b64decode(image_data_b64)
                                                        image_format = 'png'
                                                        print(f"✅ 从images[0]提取图像成功（base64字符串）")
                                                        break
                                                    except Exception as e:
                                                        print(f"⚠️ Base64解码失败: {e}")
                                                        # 尝试作为URL下载
                                                        try:
                                                            image_data = self._download_image_from_url(image_data_b64)
                                                            image_format = 'png'
                                                            print(f"✅ 从images[0] URL下载图像成功")
                                                            break
                                                        except:
                                                            pass
                                elif isinstance(image_item, str):
                                    # 直接是base64字符串或data URI
                                    if image_item.startswith('data:image'):
                                        header, encoded = image_item.split(',', 1)
                                        image_format = header.split('/')[1].split(';')[0]
                                        image_data = base64.b64decode(encoded)
                                        print(f"✅ 从images[0]字符串提取图像成功")
                                        break
                                    elif len(image_item) > 100:
                                        try:
                                            image_data = base64.b64decode(image_item)
                                            image_format = 'png'
                                            print(f"✅ 从images[0]字符串base64提取图像成功")
                                            break
                                        except:
                                            pass
                        
                        if 'content' in message:
                            content = message['content']
                            print(f"🔍 content类型: {type(content)}, 长度: {len(content) if isinstance(content, (list, str)) else 'N/A'}")
                            
                            # content可能是字符串或列表
                            if isinstance(content, list):
                                # 查找图像内容
                                for item_idx, item in enumerate(content):
                                    print(f"🔍 content[{item_idx}]类型: {type(item)}")
                                    if isinstance(item, dict):
                                        print(f"🔍 content[{item_idx}]结构: {list(item.keys())}")
                                        item_type = item.get('type', '')
                                        print(f"🔍 content[{item_idx}] type字段: {item_type}")
                                        
                                        # 检查是否是图像类型
                                        if item_type == 'image':
                                            # 可能有image_url字段
                                            if 'image_url' in item:
                                                image_url_obj = item['image_url']
                                                print(f"🔍 image_url结构: {list(image_url_obj.keys()) if isinstance(image_url_obj, dict) else type(image_url_obj)}")
                                                image_data_b64 = image_url_obj.get('url', '') if isinstance(image_url_obj, dict) else str(image_url_obj)
                                                
                                                if image_data_b64.startswith('data:image'):
                                                    # data URI格式: data:image/png;base64,...
                                                    header, encoded = image_data_b64.split(',', 1)
                                                    image_format = header.split('/')[1].split(';')[0]
                                                    image_data = base64.b64decode(encoded)
                                                    print(f"✅ 从image_url提取图像成功")
                                                    break
                                            # 也可能直接包含base64数据
                                            elif 'data' in item or 'base64' in item:
                                                image_data_b64 = item.get('data') or item.get('base64') or item.get('image')
                                                if image_data_b64:
                                                    if image_data_b64.startswith('data:image'):
                                                        header, encoded = image_data_b64.split(',', 1)
                                                        image_format = header.split('/')[1].split(';')[0]
                                                        image_data = base64.b64decode(encoded)
                                                        print(f"✅ 从item直接提取图像成功")
                                                        break
                                                    else:
                                                        # 纯base64字符串
                                                        try:
                                                            image_data = base64.b64decode(image_data_b64)
                                                            image_format = 'png'
                                                            print(f"✅ 从base64字符串提取图像成功")
                                                            break
                                                        except Exception as e:
                                                            print(f"⚠️ Base64解码失败: {e}")
                                            
                                            # 如果item本身包含base64数据（可能在text字段中）
                                            if image_data is None:
                                                for key in ['text', 'content', 'data', 'image']:
                                                    if key in item:
                                                        value = item[key]
                                                        if isinstance(value, str) and 'data:image' in value:
                                                            import re
                                                            match = re.search(r'data:image/([^;]+);base64,([A-Za-z0-9+/=]+)', value)
                                                            if match:
                                                                image_format = match.group(1)
                                                                encoded = match.group(2)
                                                                image_data = base64.b64decode(encoded)
                                                                print(f"✅ 从item[{key}]字段提取图像成功")
                                                                break
                                
                                if image_data is not None:
                                    break
                                    
                            elif isinstance(content, str):
                                # 如果content是字符串，尝试查找data URI
                                print(f"🔍 content是字符串，长度: {len(content)}")
                                if 'data:image' in content:
                                    import re
                                    match = re.search(r'data:image/([^;]+);base64,([A-Za-z0-9+/=]+)', content)
                                    if match:
                                        image_format = match.group(1)
                                        encoded = match.group(2)
                                        image_data = base64.b64decode(encoded)
                                        print(f"✅ 从字符串content提取图像成功")
                                
                                # 也可能是纯base64字符串（没有data:image前缀）
                                if image_data is None and len(content) > 1000:
                                    # 尝试直接解码（可能是base64）
                                    try:
                                        # 检查是否是base64格式
                                        if re.match(r'^[A-Za-z0-9+/=]+$', content.replace('\n', '').replace(' ', '')):
                                            image_data = base64.b64decode(content)
                                            image_format = 'png'
                                            print(f"✅ 从纯base64字符串提取图像成功")
                                    except:
                                        pass
            
            # 兼容格式1: OpenAI/OpenRouter标准格式 {"data": [{"url": "...", "b64_json": "..."}]}
            if image_data is None and 'data' in result and isinstance(result['data'], list) and len(result['data']) > 0:
                first_item = result['data'][0]
                image_data_b64 = first_item.get('url') or first_item.get('b64_json') or first_item.get('image')
                
                if image_data_b64:
                    if image_data_b64.startswith('data:image'):
                        header, encoded = image_data_b64.split(',', 1)
                        image_format = header.split('/')[1].split(';')[0]
                        image_data = base64.b64decode(encoded)
                    elif len(image_data_b64) > 100:
                        try:
                            image_data = base64.b64decode(image_data_b64)
                            image_format = 'png'
                        except Exception as e:
                            print(f"⚠️ Base64解码失败，尝试作为URL下载: {e}")
                            image_data = self._download_image_from_url(image_data_b64)
                            image_format = 'png'
                    else:
                        image_data = self._download_image_from_url(image_data_b64)
                        image_format = 'png'
            
            # 兼容格式2: 直接返回base64字符串
            if image_data is None and ('image' in result or 'b64_json' in result):
                image_data_b64 = result.get('image') or result.get('b64_json')
                if image_data_b64:
                    try:
                        image_data = base64.b64decode(image_data_b64)
                        image_format = 'png'
                    except:
                        image_data = self._download_image_from_url(image_data_b64)
                        image_format = 'png'
            
            # 兼容格式3: 直接返回URL
            if image_data is None and 'url' in result:
                image_data = self._download_image_from_url(result['url'])
                image_format = 'png'
            
            # 如果仍然没有找到图像数据，打印完整响应用于调试
            if image_data is None:
                print(f"❌ 无法解析API响应")
                print(f"🔍 响应顶层keys: {list(result.keys())}")
                if 'choices' in result:
                    print(f"🔍 choices数量: {len(result['choices'])}")
                    for idx, choice in enumerate(result['choices']):
                        print(f"🔍 choice[{idx}] keys: {list(choice.keys())}")
                        if 'message' in choice:
                            msg = choice['message']
                            print(f"🔍 choice[{idx}].message keys: {list(msg.keys())}")
                            if 'content' in msg:
                                cnt = msg['content']
                                print(f"🔍 choice[{idx}].message.content类型: {type(cnt)}")
                                if isinstance(cnt, list):
                                    for cidx, citem in enumerate(cnt):
                                        print(f"🔍 choice[{idx}].message.content[{cidx}]类型: {type(citem)}")
                                        if isinstance(citem, dict):
                                            print(f"🔍 choice[{idx}].message.content[{cidx}] keys: {list(citem.keys())}")
                                            # 打印前100个字符的内容预览
                                            for k, v in citem.items():
                                                if isinstance(v, str):
                                                    print(f"🔍   {k}: {v[:100]}...")
                                                else:
                                                    print(f"🔍   {k}: {type(v)}")
                
                # 只打印部分响应，避免输出过长
                import json
                try:
                    # 尝试提取关键信息
                    debug_info = {
                        'keys': list(result.keys()),
                        'choices_count': len(result.get('choices', [])),
                    }
                    if 'choices' in result and len(result['choices']) > 0:
                        choice = result['choices'][0]
                        if 'message' in choice:
                            msg = choice['message']
                            if 'content' in msg:
                                cnt = msg['content']
                                if isinstance(cnt, list):
                                    debug_info['content_items'] = [
                                        {'type': item.get('type'), 'keys': list(item.keys()) if isinstance(item, dict) else None}
                                        for item in cnt[:3]  # 只取前3个
                                    ]
                    print(f"🔍 调试信息: {json.dumps(debug_info, indent=2, ensure_ascii=False)}")
                except:
                    pass
                
                raise ValueError(f"API响应格式不正确，无法找到图像数据。响应结构: {list(result.keys())}")
            
            print(f"✅ 成功提取图像数据，格式: {image_format}, 大小: {len(image_data)} 字节")
            return image_data, image_format
            
        except requests.exceptions.Timeout:
            raise Exception("文生图API请求超时，请稍后重试")
        except requests.exceptions.RequestException as e:
            raise Exception(f"文生图API请求失败: {str(e)}")
    
    def _download_image_from_url(self, url: str) -> bytes:
        """
        从URL下载图像
        
        Args:
            url: 图像URL
            
        Returns:
            图像二进制数据
        """
        response = requests.get(url, timeout=60)
        if response.status_code != 200:
            raise Exception(f"下载图像失败: {response.status_code}")
        return response.content
    
    def _generate_output_path(self, original_path: str) -> str:
        """
        生成输出文件路径
        
        Args:
            original_path: 原始SVG文件路径
            
        Returns:
            输出图像文件路径
        """
        path_obj = Path(original_path)
        directory = path_obj.parent
        stem = path_obj.stem
        # 生成新文件名：原文件名_aicreate.png
        output_filename = f"{stem}_aicreate.png"
        output_path = directory / output_filename
        
        return str(output_path)
    
    def _save_image(self, image_data: bytes, output_path: str, image_format: str):
        """
        保存图像文件
        
        Args:
            image_data: 图像二进制数据
            output_path: 输出文件路径
            image_format: 图像格式
        """
        # 确保目录存在
        output_path_obj = Path(output_path)
        output_path_obj.parent.mkdir(parents=True, exist_ok=True)
        
        # 根据格式确定文件扩展名
        if not output_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            if image_format.lower() in ('jpg', 'jpeg'):
                output_path = str(output_path_obj.with_suffix('.jpg'))
            else:
                output_path = str(output_path_obj.with_suffix('.png'))
        
        # 保存文件
        with open(output_path, 'wb') as f:
            f.write(image_data)
        
        print(f"✅ 图像已保存: {output_path}")


def create_image_generation_optimizer_from_config() -> ImageGenerationSVGOptimizer:
    """从config/config.txt配置文件创建文生图优化器（使用svg_optimizer_*配置）"""
    import sys
    import os
    
    # 添加src目录到Python路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    src_dir = os.path.join(current_dir, 'src')
    if src_dir not in sys.path:
        sys.path.append(src_dir)
    
    try:
        from config_loader import (
            get_svg_optimizer_api_key,
            get_svg_optimizer_api_base,
            get_svg_optimizer_model
        )
        
        # 从config.txt读取svg_optimizer_*配置
        api_key = get_svg_optimizer_api_key()
        api_base = get_svg_optimizer_api_base()
        model = get_svg_optimizer_model()
        
        if not api_key or api_key == 'your key':
            raise ValueError(
                "未配置有效的SVG优化器API密钥，请在config/config.txt中设置svg_optimizer_api_key"
            )
        
        if not api_base:
            raise ValueError("未配置SVG优化器API base URL，请在config/config.txt中设置svg_optimizer_api_base")
        
        print(f"🔧 从config.txt读取SVG优化器配置:")
        print(f"  API Base: {api_base}")
        print(f"  Model: {model}")
        
        return ImageGenerationSVGOptimizer(api_key, api_base, model)
        
    except ImportError as e:
        print(f"⚠️ 无法导入config_loader: {e}")
        raise
    except Exception as e:
        print(f"⚠️ 读取配置失败: {e}")
        raise


# 示例用法
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("用法: python llm_svg_optimizer.py <input.svg>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    
    if not os.path.exists(input_file):
        print(f"❌ 文件不存在: {input_file}")
        sys.exit(1)
    
    try:
        optimizer = create_image_generation_optimizer_from_config()
        
        with open(input_file, 'r', encoding='utf-8') as f:
            svg_content = f.read()
        
        output_path, report = optimizer.generate_image_from_svg(svg_content, input_file)
        
        print("\n" + "="*50)
        print("📊 图像生成报告")
        print("="*50)
        print(f"使用模型: {report.get('model', 'unknown')}")
        print(f"输出路径: {report.get('output_path', 'unknown')}")
        print(f"图像格式: {report.get('image_format', 'unknown')}")
        print(f"✅ 图像生成成功！")
                
    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
