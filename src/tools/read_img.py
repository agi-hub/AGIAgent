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

import os
import base64
import io
from typing import Dict, Any, Optional, Tuple

from .print_system import print_current, print_debug


class ImageReader:
    """Image reading and analysis using vision models."""
    
    def __init__(self, workspace_root: Optional[str] = None):
        """
        Initialize image reader.
        
        Args:
            workspace_root: Root directory for resolving relative image paths
        """
        self.workspace_root = workspace_root or os.getcwd()
    
    def _compress_image(self, image_path: str, target_size_kb: int = 100) -> Tuple[bytes, str]:
        """
        Compress image to approximately target_size_kb (in KB).
        
        Args:
            image_path: Path to the image file
            target_size_kb: Target size in KB (default: 100KB)
            
        Returns:
            Tuple of (compressed_image_data: bytes, mime_type: str)
        """
        # Determine MIME type from file extension
        file_ext = os.path.splitext(image_path)[1].lower()
        if file_ext in ['.jpg', '.jpeg']:
            mime_type = 'image/jpeg'
            output_format = 'JPEG'
        elif file_ext == '.png':
            mime_type = 'image/png'
            output_format = 'PNG'
        elif file_ext == '.gif':
            mime_type = 'image/gif'
            output_format = 'GIF'
        elif file_ext == '.webp':
            mime_type = 'image/webp'
            output_format = 'WEBP'
        else:
            mime_type = 'image/jpeg'
            output_format = 'JPEG'
        
        target_size_bytes = target_size_kb * 1024
        
        # Read original image
        with open(image_path, 'rb') as f:
            original_data = f.read()
        
        # If already small enough, return as is
        if len(original_data) <= target_size_bytes:
            return original_data, mime_type
        
        # Try to compress using PIL
        try:
            from PIL import Image
            
            # Open image
            img = Image.open(io.BytesIO(original_data))
            
            # Convert RGBA/LA/P to RGB for JPEG
            if output_format == 'JPEG' and img.mode in ('RGBA', 'LA', 'P'):
                # Create white background
                rgb_img = Image.new('RGB', img.size, (255, 255, 255))
                if img.mode == 'P':
                    img = img.convert('RGBA')
                rgb_img.paste(img, mask=img.split()[-1] if img.mode in ('RGBA', 'LA') else None)
                img = rgb_img
            elif img.mode not in ('RGB', 'L'):
                img = img.convert('RGB')
            
            # Start with high quality and reduce if needed
            # Prioritize maintaining image completeness (no truncation) over size constraints
            quality = 90  # Start with higher quality
            scale_factor = 1.0
            
            # Try different compression levels
            for attempt in range(5):  # Max 5 attempts
                buffer = io.BytesIO()
                
                # Resize if needed (after first attempt if still too large)
                if attempt > 0 and scale_factor < 1.0:
                    new_size = (int(img.size[0] * scale_factor), int(img.size[1] * scale_factor))
                    if hasattr(Image, 'Resampling'):
                        resample = Image.Resampling.LANCZOS
                    else:
                        resample = Image.LANCZOS
                    resized_img = img.resize(new_size, resample)
                else:
                    resized_img = img
                
                # Save with compression - ensure full image is saved (no truncation)
                if output_format == 'JPEG':
                    resized_img.save(buffer, format=output_format, quality=quality, optimize=True)
                elif output_format == 'PNG':
                    # PNG compression level (0-9, 9 is highest compression)
                    png_compress_level = min(9, 6 + attempt)
                    resized_img.save(buffer, format=output_format, optimize=True, compress_level=png_compress_level)
                else:
                    resized_img.save(buffer, format=output_format, optimize=True)
                
                compressed_data = buffer.getvalue()
                
                # Verify image is complete (not truncated) by checking if we can read it back
                # This ensures the full image data is present, not just partial data
                try:
                    verify_buffer = io.BytesIO(compressed_data)
                    verify_img = Image.open(verify_buffer)
                    # Try to load the image to verify it's complete
                    verify_img.load()  # Load image data to verify completeness
                    # If we get here, image is complete and not truncated
                    del verify_img  # Clean up
                except Exception as verify_error:
                    # Image verification failed, might be truncated or corrupted
                    print_debug(f"⚠️ Image verification failed at attempt {attempt + 1}: {verify_error}, trying next compression level")
                    if attempt < 4:  # Don't fail on last attempt, use what we have
                        continue  # Try next compression level
                    # On last attempt, use the data anyway (better than failing completely)
                
                # Check if we're close enough to target size
                if len(compressed_data) <= target_size_bytes * 1.1:  # Allow 10% tolerance
                    return compressed_data, mime_type
                
                # Adjust for next attempt - allow aggressive compression but ensure completeness
                if attempt == 0:
                    # First attempt: reduce quality slightly
                    quality = max(75, quality - 10)
                elif attempt == 1:
                    # Second attempt: reduce quality more, keep size
                    quality = max(65, quality - 10)
                elif attempt == 2:
                    # Third attempt: reduce size slightly
                    scale_factor = 0.9
                    quality = max(60, quality - 5)
                elif attempt == 3:
                    # Fourth attempt: reduce size more
                    scale_factor = 0.75
                    quality = max(55, quality - 5)
                else:
                    # Last attempt: more aggressive compression
                    scale_factor = max(0.5, scale_factor * 0.9)
                    quality = max(50, quality - 5)
            
            # Return best compression achieved
            return compressed_data, mime_type
            
        except ImportError:
            # PIL not available, return original
            print_debug("⚠️ PIL not available, using original image (may be large)")
            return original_data, mime_type
        except Exception as e:
            # Compression failed, return original
            print_debug(f"⚠️ Image compression failed: {e}, using original image")
            return original_data, mime_type
    
    def _resolve_image_path(self, image_path: str) -> Optional[str]:
        """
        Resolve image file path from various possible locations.
        
        Args:
            image_path: Image file path (absolute, relative to workspace, or relative to current dir)
            
        Returns:
            Resolved absolute path if found, None otherwise
        """
        # Try absolute path first
        if os.path.isabs(image_path) and os.path.isfile(image_path):
            return image_path
        
        # Try relative to workspace root
        if self.workspace_root:
            workspace_path = os.path.join(self.workspace_root, image_path)
            if os.path.isfile(workspace_path):
                return workspace_path
        
        # Try current directory
        if os.path.isfile(image_path):
            return os.path.abspath(image_path)
        
        return None
    
    def read_img(self, query: str, image_path: str) -> Dict[str, Any]:
        """
        Read and analyze an image using the vision model, returning text description.
        
        This tool uses the configured vision_model to analyze images and return
        text descriptions instead of base64-encoded image data.
        
        Args:
            query: Query or instruction for what to analyze in the image
            image_path: Path to the image file (supports relative and absolute paths)
            
        Returns:
            Dictionary containing:
            - status: 'success' or 'failed'
            - text: Text description of the image (if successful)
            - error: Error message (if failed)
            - image_path: Path to the analyzed image
        """
        try:
            from src.config_loader import (
                get_vision_model, get_vision_api_key, get_vision_api_base,
                get_vision_max_tokens, has_vision_config
            )
            
            # Helper function to check if API base uses Anthropic format
            def is_anthropic_api(api_base: str) -> bool:
                """Check if the API base URL uses Anthropic format"""
                return api_base.lower().endswith('/anthropic') if api_base else False
            
            # Check if vision model is configured
            if not has_vision_config():
                return {
                    'status': 'failed',
                    'error': 'Vision model not configured. Please set vision_model, vision_api_key, and vision_api_base in config.txt',
                    'image_path': image_path
                }
            
            # Load vision model configuration
            vision_model = get_vision_model()
            vision_api_key = get_vision_api_key()
            vision_api_base = get_vision_api_base()
            vision_max_tokens = get_vision_max_tokens()
            
            if not vision_model or not vision_api_key or not vision_api_base:
                return {
                    'status': 'failed',
                    'error': 'Vision model configuration incomplete. Please check vision_model, vision_api_key, and vision_api_base in config.txt',
                    'image_path': image_path
                }
            
            print_debug(f"🖼️ Reading image with vision model: {vision_model}")
            print_debug(f"   Image path: {image_path}")
            print_debug(f"   Query: {query}")
            
            # Resolve image file path
            resolved_path = self._resolve_image_path(image_path)
            
            if not resolved_path or not os.path.isfile(resolved_path):
                return {
                    'status': 'failed',
                    'error': f'Image file not found: {image_path}',
                    'image_path': image_path
                }
            
            print_debug(f"📁 Resolved image path: {resolved_path}")
            
            # Read and compress image to ~100KB
            image_data, mime_type = self._compress_image(resolved_path, target_size_kb=100)
            original_size = os.path.getsize(resolved_path)
            compressed_size = len(image_data)
            print_debug(f"📦 Image compressed: {original_size / 1024:.1f}KB → {compressed_size / 1024:.1f}KB")
            
            base64_data = base64.b64encode(image_data).decode('utf-8')
            
            # Check if using Anthropic API
            is_claude = is_anthropic_api(vision_api_base)
            
            # Create vision client
            if is_claude:
                from anthropic import Anthropic
                vision_client = Anthropic(
                    api_key=vision_api_key,
                    base_url=vision_api_base
                )
            else:
                from openai import OpenAI
                vision_client = OpenAI(
                    api_key=vision_api_key,
                    base_url=vision_api_base
                )
            
            # Prepare vision API call with streaming
            if is_claude:
                # Claude format with streaming
                content_parts = [
                    {
                        "type": "text",
                        "text": query
                    },
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": mime_type,
                            "data": base64_data
                        }
                    }
                ]
                
                print_debug("🔄 Using streaming API for vision analysis...")
                text_result = ""
                
                # Use streaming API
                with vision_client.messages.stream(
                    model=vision_model,
                    max_tokens=vision_max_tokens or 4096,
                    messages=[
                        {
                            "role": "user",
                            "content": content_parts
                        }
                    ]
                ) as stream:
                    for event in stream:
                        event_type = getattr(event, 'type', None)
                        
                        # Handle content block delta events
                        if event_type == "content_block_delta":
                            delta = getattr(event, 'delta', None)
                            if delta:
                                delta_type = getattr(delta, 'type', None)
                                # Extract text from text_delta events
                                if delta_type == "text_delta":
                                    text_chunk = getattr(delta, 'text', '')
                                    if text_chunk:
                                        text_result += text_chunk
                        
                        # Handle content block done events (fallback)
                        elif event_type == "content_block_done":
                            content_block = getattr(event, 'content_block', None)
                            if content_block:
                                block_type = getattr(content_block, 'type', None)
                                if block_type == "text":
                                    text_content = getattr(content_block, 'text', '')
                                    if text_content:
                                        text_result += text_content
                
            else:
                # OpenAI format with streaming
                print_debug("🔄 Using streaming API for vision analysis...")
                text_result = ""
                
                stream = vision_client.chat.completions.create(
                    model=vision_model,
                    max_tokens=vision_max_tokens or 4096,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": query
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:{mime_type};base64,{base64_data}",
                                        "detail": "high"  # Use high detail for better OCR and text extraction accuracy
                                    }
                                }
                            ]
                        }
                    ],
                    stream=True
                )
                
                # Extract text from streaming response
                for chunk in stream:
                    if chunk.choices and len(chunk.choices) > 0:
                        delta = chunk.choices[0].delta
                        if hasattr(delta, 'content') and delta.content:
                            text_result += delta.content
            
            print_debug(f"✅ Image analysis completed: {len(text_result)} characters")
            
            return {
                'status': 'success',
                'text': text_result,
                'image_path': resolved_path,
                'vision_model': vision_model,
                'query': query
            }
            
        except ImportError as e:
            return {
                'status': 'failed',
                'error': f'Failed to import required library: {str(e)}',
                'image_path': image_path
            }
        except Exception as e:
            error_msg = str(e)
            print_debug(f"❌ Error reading image: {error_msg}")
            return {
                'status': 'failed',
                'error': f'Failed to read image: {error_msg}',
                'image_path': image_path
            }

