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

Image Generation Tools
Handles image generation tasks using AI models
"""

import os
import base64
import requests
from typing import Dict, Any, Optional, Tuple
from pathlib import Path

# Import config_loader
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config_loader import (
    get_image_generation_api_key,
    get_image_generation_api_base,
    get_image_generation_model
)
from .print_system import print_current


class ImageGenerationTools:
    """Image generation tool using AI models"""
    
    def __init__(self, workspace_root: str = "."):
        self.workspace_root = workspace_root
    
    def create_img(self, prompt: str, resolution: str = "1280x1280", file_path: str = None) -> Dict[str, Any]:
        """
        Generate an image from a text prompt and save it to a file.
        
        Args:
            prompt: Text prompt describing the image to generate
            resolution: Image resolution in format "WIDTHxHEIGHT" (e.g., "1280x1280", "1024x768")
                       Default: "1280x1280"
            file_path: Path where the image should be saved (PNG format).
                      If not provided, will generate a filename based on prompt.
                      Supports relative paths (relative to workspace_root) or absolute paths.
        
        Returns:
            Dictionary with status and result information:
            - status: "success" or "failed"
            - prompt: The input prompt
            - resolution: The resolution used
            - file_path: The path where the image was saved
            - file_size: Size of the saved file in bytes (if successful)
            - error: Error message (if failed)
        """
        try:
            # Validate prompt
            if not prompt or not prompt.strip():
                return {
                    "status": "failed",
                    "error": "Prompt cannot be empty",
                    "prompt": prompt,
                    "resolution": resolution
                }
            
            # Validate resolution format
            try:
                width, height = resolution.split('x')
                width = int(width.strip())
                height = int(height.strip())
                if width <= 0 or height <= 0:
                    raise ValueError("Resolution dimensions must be positive")
            except (ValueError, AttributeError) as e:
                return {
                    "status": "failed",
                    "error": f"Invalid resolution format. Expected format: 'WIDTHxHEIGHT' (e.g., '1280x1280'), got: {resolution}",
                    "prompt": prompt,
                    "resolution": resolution
                }
            
            # Determine output file path
            if not file_path:
                # Generate filename from prompt (sanitized)
                safe_prompt = "".join(c for c in prompt[:50] if c.isalnum() or c in (' ', '-', '_')).strip()
                safe_prompt = safe_prompt.replace(' ', '_')
                file_path = f"generated_image_{safe_prompt}.png"
            
            # Convert to absolute path if relative
            if not os.path.isabs(file_path):
                file_path = os.path.join(self.workspace_root, file_path)
            
            # Ensure directory exists
            output_dir = os.path.dirname(file_path)
            if output_dir and not os.path.exists(output_dir):
                try:
                    os.makedirs(output_dir, exist_ok=True)
                except Exception as e:
                    return {
                        "status": "failed",
                        "error": f"Failed to create output directory: {str(e)}",
                        "prompt": prompt,
                        "resolution": resolution,
                        "file_path": file_path
                    }
            
            # Ensure file_path ends with .png
            if not file_path.lower().endswith('.png'):
                file_path = file_path + '.png'
            
            # Load configuration (use default config file path)
            # The config_loader functions default to "config/config.txt" relative to project root
            api_key = get_image_generation_api_key()
            api_base = get_image_generation_api_base()
            model = get_image_generation_model()
            
            if not api_key:
                return {
                    "status": "failed",
                    "error": "Image generation API key not configured. Please set image_generation_api_key in config/config.txt",
                    "prompt": prompt,
                    "resolution": resolution
                }
            
            if not api_base:
                return {
                    "status": "failed",
                    "error": "Image generation API base URL not configured. Please set image_generation_api_base in config/config.txt",
                    "prompt": prompt,
                    "resolution": resolution
                }
            
            if not model:
                return {
                    "status": "failed",
                    "error": "Image generation model not configured. Please set image_generation_model in config/config.txt",
                    "prompt": prompt,
                    "resolution": resolution
                }
            
            print_current(f"🎨 正在生成图像...")
            print_current(f"📝 提示词: {prompt}")
            print_current(f"📐 分辨率: {resolution}")
            print_current(f"💾 保存路径: {file_path}")
            
            # Determine which API to use
            is_zhipu = 'bigmodel.cn' in api_base
            
            try:
                if is_zhipu:
                    image_data, image_format = self._generate_image_zhipu(
                        prompt, api_key, api_base, model, resolution
                    )
                else:
                    image_data, image_format = self._generate_image_openrouter(
                        prompt, api_key, api_base, model, resolution
                    )
                
                # Save image to file
                with open(file_path, 'wb') as f:
                    f.write(image_data)
                
                file_size = len(image_data)
                file_size_mb = file_size / (1024 * 1024)
                
                print_current(f"✅ 图像生成成功！")
                print_current(f"📊 文件大小: {file_size_mb:.2f} MB ({file_size} 字节)")
                
                return {
                    "status": "success",
                    "prompt": prompt,
                    "resolution": resolution,
                    "file_path": file_path,
                    "file_size": file_size,
                    "file_size_mb": round(file_size_mb, 2),
                    "model": model,
                    "api": "智谱AI" if is_zhipu else "OpenRouter"
                }
                
            except Exception as e:
                error_msg = str(e)
                print_current(f"❌ 图像生成失败: {error_msg}")
                return {
                    "status": "failed",
                    "error": error_msg,
                    "prompt": prompt,
                    "resolution": resolution,
                    "file_path": file_path
                }
        
        except Exception as e:
            error_msg = str(e)
            print_current(f"❌ 发生错误: {error_msg}")
            return {
                "status": "failed",
                "error": error_msg,
                "prompt": prompt,
                "resolution": resolution,
                "file_path": file_path if 'file_path' in locals() else None
            }
    
    def _generate_image_zhipu(self, prompt: str, api_key: str, api_base: str, 
                               model: str, resolution: str) -> Tuple[bytes, str]:
        """
        Generate image using Zhipu AI API
        
        Returns:
            (image_data: bytes, image_format: str)
        """
        # Build API endpoint
        api_base = api_base.rstrip('/')
        
        if '/paas/v4/images/generations' in api_base:
            api_url = api_base
        elif '/paas/v4' in api_base:
            api_url = f"{api_base}/images/generations"
        elif '/api' in api_base and '/paas' not in api_base:
            api_url = f"{api_base}/paas/v4/images/generations"
        else:
            api_url = f"{api_base}/api/paas/v4/images/generations"
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": model,
            "prompt": prompt,
            "size": resolution,
            "quality": "hd"
        }
        
        response = requests.post(
            api_url,
            headers=headers,
            json=data,
            timeout=120
        )
        
        if response.status_code != 200:
            error_msg = response.text
            raise Exception(f"API错误: {response.status_code} - {error_msg}")
        
        result = response.json()
        
        # Zhipu AI response format: {"data": [{"url": "..."}]}
        if 'data' in result and isinstance(result['data'], list) and len(result['data']) > 0:
            first_item = result['data'][0]
            image_url = first_item.get('url', '')
            if image_url:
                # Download image
                img_response = requests.get(image_url, timeout=60)
                if img_response.status_code == 200:
                    image_data = img_response.content
                    return image_data, 'png'
                else:
                    raise Exception(f"下载图像失败: {img_response.status_code}")
            else:
                raise ValueError("响应中未找到图像URL")
        else:
            raise ValueError(f"API响应格式不正确: {result}")
    
    def _generate_image_openrouter(self, prompt: str, api_key: str, api_base: str,
                                   model: str, resolution: str) -> Tuple[bytes, str]:
        """
        Generate image using OpenRouter API
        
        Returns:
            (image_data: bytes, image_format: str)
        """
        api_base = api_base.rstrip('/')
        
        if '/chat/completions' in api_base:
            api_url = api_base
        elif '/v1' in api_base:
            api_url = f"{api_base}/chat/completions"
        else:
            api_url = f"{api_base}/v1/chat/completions"
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/AGIAgent",
            "X-Title": "AGIAgent Image Generation"
        }
        
        # Parse resolution for OpenRouter (if needed)
        # Note: OpenRouter may have different parameter format
        data = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "modalities": ["image", "text"],
            "max_tokens": 4096
        }
        
        # Add resolution if supported by the API
        # Some models may support size parameter
        try:
            width, height = resolution.split('x')
            # Some APIs accept size as a parameter
            # This is model-dependent, so we'll try common formats
            data["size"] = resolution
        except:
            pass
        
        response = requests.post(
            api_url,
            headers=headers,
            json=data,
            timeout=120
        )
        
        if response.status_code != 200:
            error_msg = response.text
            raise Exception(f"API错误: {response.status_code} - {error_msg}")
        
        result = response.json()
        
        # OpenRouter response format processing
        if 'choices' in result and isinstance(result['choices'], list) and len(result['choices']) > 0:
            choice = result['choices'][0]
            if 'message' in choice:
                message = choice['message']
                
                # Check images field
                if 'images' in message:
                    images = message['images']
                    if isinstance(images, list) and len(images) > 0:
                        image_item = images[0]
                        if isinstance(image_item, dict):
                            # Try various possible field names
                            for key in ['url', 'b64_json', 'image', 'data', 'base64']:
                                if key in image_item:
                                    value = image_item[key]
                                    if isinstance(value, str):
                                        if value.startswith('data:image'):
                                            # Base64 data URL
                                            header, data_str = value.split(',', 1)
                                            image_data = base64.b64decode(data_str)
                                            return image_data, 'png'
                                        elif value.startswith('http'):
                                            # URL
                                            img_response = requests.get(value, timeout=60)
                                            if img_response.status_code == 200:
                                                return img_response.content, 'png'
                
                # Check content field
                if 'content' in message:
                    content = message['content']
                    if isinstance(content, list):
                        for item in content:
                            if isinstance(item, dict) and item.get('type') == 'image':
                                image_url = item.get('image_url', {})
                                if isinstance(image_url, dict):
                                    url = image_url.get('url', '')
                                    if url.startswith('data:image'):
                                        header, data_str = url.split(',', 1)
                                        image_data = base64.b64decode(data_str)
                                        return image_data, 'png'
                                    elif url.startswith('http'):
                                        img_response = requests.get(url, timeout=60)
                                        if img_response.status_code == 200:
                                            return img_response.content, 'png'
        
        raise ValueError(f"无法从响应中提取图像数据: {result}")
