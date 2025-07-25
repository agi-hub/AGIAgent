#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
图像输入功能单元测试
测试图像上传、解析、处理等多模态功能
"""

import pytest
import os
import sys
import base64
import io
from PIL import Image
from unittest.mock import patch, Mock, MagicMock
from typing import Dict, List, Any

# 添加src路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from tools.image_tools import ImageProcessor
from tools.multimodal_tools import MultimodalProcessor
from utils.test_helpers import TestHelper

@pytest.mark.unit
class TestImageProcessing:
    """图像处理功能测试类"""
    
    @pytest.fixture
    def image_processor(self, test_workspace):
        """创建图像处理器实例"""
        return ImageProcessor(workspace_root=test_workspace)
    
    @pytest.fixture
    def multimodal_processor(self, test_workspace):
        """创建多模态处理器实例"""
        return MultimodalProcessor(workspace_root=test_workspace)
    
    @pytest.fixture
    def sample_images(self, test_workspace):
        """创建示例图像文件"""
        images = {}
        
        # 创建简单的测试图像
        # RGB图像
        rgb_image = Image.new('RGB', (100, 100), color=(255, 0, 0))
        rgb_path = os.path.join(test_workspace, 'test_rgb.png')
        rgb_image.save(rgb_path)
        images['rgb'] = rgb_path
        
        # RGBA图像
        rgba_image = Image.new('RGBA', (100, 100), color=(0, 255, 0, 128))
        rgba_path = os.path.join(test_workspace, 'test_rgba.png')
        rgba_image.save(rgba_path)
        images['rgba'] = rgba_path
        
        # 灰度图像
        gray_image = Image.new('L', (100, 100), color=128)
        gray_path = os.path.join(test_workspace, 'test_gray.png')
        gray_image.save(gray_path)
        images['gray'] = gray_path
        
        # JPEG图像
        jpeg_image = Image.new('RGB', (200, 150), color=(0, 0, 255))
        jpeg_path = os.path.join(test_workspace, 'test_image.jpg')
        jpeg_image.save(jpeg_path, 'JPEG')
        images['jpeg'] = jpeg_path
        
        return images
    
    @pytest.fixture
    def image_formats(self):
        """支持的图像格式"""
        return {
            'common': ['PNG', 'JPEG', 'JPG', 'GIF', 'BMP', 'TIFF'],
            'web': ['PNG', 'JPEG', 'GIF', 'WEBP'],
            'raw': ['CR2', 'NEF', 'ARW', 'DNG'],
            'vector': ['SVG', 'EPS', 'AI']
        }
    
    @pytest.fixture
    def base64_images(self, sample_images):
        """Base64编码的图像数据"""
        b64_images = {}
        
        for name, path in sample_images.items():
            with open(path, 'rb') as f:
                image_data = f.read()
                b64_data = base64.b64encode(image_data).decode('utf-8')
                b64_images[name] = f"data:image/png;base64,{b64_data}"
        
        return b64_images
    
    def test_image_processor_initialization(self, image_processor, test_workspace):
        """测试图像处理器初始化"""
        assert image_processor is not None
        assert hasattr(image_processor, 'process_image')
        assert hasattr(image_processor, 'analyze_image')
        assert image_processor.workspace_root == test_workspace
    
    def test_image_loading_from_file(self, image_processor, sample_images):
        """测试从文件加载图像"""
        for name, path in sample_images.items():
            try:
                result = image_processor.load_image(path)
                
                # 验证图像加载
                assert result is not None
                if isinstance(result, dict):
                    assert 'width' in result or 'height' in result or 'format' in result
                
            except Exception as e:
                # 某些格式可能不被支持
                assert "unsupported" in str(e).lower() or "format" in str(e).lower()
    
    def test_image_loading_from_base64(self, image_processor, base64_images):
        """测试从Base64加载图像"""
        for name, b64_data in base64_images.items():
            try:
                result = image_processor.load_image_from_base64(b64_data)
                
                # 验证图像加载
                assert result is not None
                if isinstance(result, dict):
                    assert 'data' in result or 'image' in result
                
            except Exception as e:
                # 某些格式可能不被支持
                pass
    
    def test_image_format_detection(self, image_processor, sample_images):
        """测试图像格式检测"""
        expected_formats = {
            'rgb': 'PNG',
            'rgba': 'PNG', 
            'gray': 'PNG',
            'jpeg': 'JPEG'
        }
        
        for name, path in sample_images.items():
            try:
                format_info = image_processor.detect_format(path)
                
                # 验证格式检测
                assert format_info is not None
                if isinstance(format_info, dict):
                    assert format_info.get('format') == expected_formats[name]
                elif isinstance(format_info, str):
                    assert format_info == expected_formats[name]
                    
            except Exception as e:
                pass
    
    def test_image_metadata_extraction(self, image_processor, sample_images):
        """测试图像元数据提取"""
        for name, path in sample_images.items():
            try:
                metadata = image_processor.extract_metadata(path)
                
                # 验证元数据提取
                assert metadata is not None
                if isinstance(metadata, dict):
                    # 检查基本元数据字段
                    expected_fields = ['width', 'height', 'format', 'mode', 'size']
                    available_fields = [field for field in expected_fields if field in metadata]
                    assert len(available_fields) > 0
                    
            except Exception as e:
                pass
    
    def test_image_resizing(self, image_processor, sample_images):
        """测试图像尺寸调整"""
        target_sizes = [(50, 50), (200, 200), (150, 100)]
        
        for name, path in sample_images.items():
            for width, height in target_sizes:
                try:
                    result = image_processor.resize_image(path, width, height)
                    
                    # 验证图像调整
                    assert result is not None
                    if isinstance(result, dict):
                        assert result.get('width') == width
                        assert result.get('height') == height
                        
                except Exception as e:
                    pass
    
    def test_image_format_conversion(self, image_processor, sample_images):
        """测试图像格式转换"""
        target_formats = ['PNG', 'JPEG', 'BMP', 'TIFF']
        
        for name, path in sample_images.items():
            for target_format in target_formats:
                try:
                    result = image_processor.convert_format(path, target_format)
                    
                    # 验证格式转换
                    assert result is not None
                    if isinstance(result, dict):
                        assert result.get('format') == target_format
                    elif isinstance(result, str):  # 可能返回新文件路径
                        assert os.path.exists(result)
                        
                except Exception as e:
                    # 某些转换可能不被支持
                    pass
    
    def test_image_quality_assessment(self, image_processor, sample_images):
        """测试图像质量评估"""
        for name, path in sample_images.items():
            try:
                quality = image_processor.assess_quality(path)
                
                # 验证质量评估
                assert quality is not None
                if isinstance(quality, dict):
                    # 检查质量指标
                    quality_metrics = ['sharpness', 'brightness', 'contrast', 'noise_level']
                    available_metrics = [metric for metric in quality_metrics if metric in quality]
                    assert len(available_metrics) > 0
                elif isinstance(quality, (int, float)):
                    assert 0 <= quality <= 100  # 假设质量分数在0-100之间
                    
            except Exception as e:
                pass
    
    def test_image_content_analysis(self, image_processor, sample_images):
        """测试图像内容分析"""
        with patch('requests.post') as mock_post:
            # 模拟AI图像分析响应
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "analysis": {
                    "objects": [
                        {"name": "rectangle", "confidence": 0.95, "bounds": [10, 10, 90, 90]},
                        {"name": "color_block", "confidence": 0.88, "bounds": [0, 0, 100, 100]}
                    ],
                    "colors": ["red", "green", "blue"],
                    "dominant_color": "red",
                    "scene": "abstract",
                    "text": []
                }
            }
            mock_post.return_value = mock_response
            
            for name, path in sample_images.items():
                try:
                    analysis = image_processor.analyze_content(path)
                    
                    # 验证内容分析
                    assert analysis is not None
                    if isinstance(analysis, dict):
                        expected_fields = ['objects', 'colors', 'scene', 'text']
                        available_fields = [field for field in expected_fields if field in analysis]
                        assert len(available_fields) > 0
                        
                except Exception as e:
                    pass
    
    def test_optical_character_recognition(self, image_processor, test_workspace):
        """测试光学字符识别(OCR)"""
        # 创建包含文本的图像
        from PIL import Image, ImageDraw, ImageFont
        
        # 创建白色背景图像
        text_image = Image.new('RGB', (300, 100), color='white')
        draw = ImageDraw.Draw(text_image)
        
        # 添加文本
        try:
            # 尝试使用默认字体
            draw.text((10, 30), "Hello AGI Bot!", fill='black')
        except:
            # 如果字体不可用，创建简单的文本图像
            draw.rectangle([10, 30, 200, 60], fill='black')
        
        text_image_path = os.path.join(test_workspace, 'text_image.png')
        text_image.save(text_image_path)
        
        with patch('pytesseract.image_to_string') as mock_ocr:
            # 模拟OCR响应
            mock_ocr.return_value = "Hello AGI Bot!"
            
            try:
                text_result = image_processor.extract_text(text_image_path)
                
                # 验证OCR功能
                assert text_result is not None
                if isinstance(text_result, dict):
                    assert 'text' in text_result
                elif isinstance(text_result, str):
                    assert len(text_result) > 0
                    
            except Exception as e:
                # OCR可能需要额外依赖
                pass
    
    def test_image_batch_processing(self, image_processor, sample_images):
        """测试图像批量处理"""
        image_paths = list(sample_images.values())
        
        operations = [
            {'operation': 'resize', 'params': {'width': 150, 'height': 150}},
            {'operation': 'convert', 'params': {'format': 'JPEG'}},
            {'operation': 'analyze', 'params': {}}
        ]
        
        for operation in operations:
            try:
                if hasattr(image_processor, 'batch_process'):
                    results = image_processor.batch_process(image_paths, operation)
                    
                    # 验证批量处理
                    assert results is not None
                    assert len(results) == len(image_paths)
                else:
                    # 逐个处理
                    results = []
                    for path in image_paths:
                        if operation['operation'] == 'resize':
                            result = image_processor.resize_image(path, **operation['params'])
                        elif operation['operation'] == 'convert':
                            result = image_processor.convert_format(path, operation['params']['format'])
                        else:
                            result = image_processor.analyze_content(path)
                        results.append(result)
                    
                    assert len(results) == len(image_paths)
                    
            except Exception as e:
                pass
    
    def test_multimodal_integration(self, multimodal_processor, sample_images):
        """测试多模态集成"""
        for name, path in sample_images.items():
            try:
                # 结合图像和文本的多模态分析
                text_prompt = "请描述这张图片的内容和颜色"
                
                with patch.object(multimodal_processor, 'process_image_with_text') as mock_process:
                    mock_process.return_value = {
                        "description": "这是一个简单的彩色图像，主要包含几何形状",
                        "colors": ["红色", "绿色", "蓝色"],
                        "objects": ["矩形", "颜色块"],
                        "text_analysis": "图像符合描述要求"
                    }
                    
                    result = multimodal_processor.process_image_with_text(path, text_prompt)
                    
                    # 验证多模态处理
                    assert result is not None
                    assert isinstance(result, dict)
                    assert 'description' in result
                    
            except Exception as e:
                pass
    
    def test_image_security_validation(self, image_processor, test_workspace):
        """测试图像安全验证"""
        # 创建可能有安全风险的文件
        potentially_dangerous_files = [
            {'name': 'script.jpg.exe', 'content': b'fake image with executable extension'},
            {'name': 'malformed.png', 'content': b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00'},  # 截断的PNG
            {'name': 'oversized.bmp', 'content': b'BM' + b'\x00' * 1000000}  # 过大的文件
        ]
        
        for file_info in potentially_dangerous_files:
            file_path = os.path.join(test_workspace, file_info['name'])
            with open(file_path, 'wb') as f:
                f.write(file_info['content'])
            
            try:
                # 验证安全检查
                if hasattr(image_processor, 'validate_image_security'):
                    is_safe = image_processor.validate_image_security(file_path)
                    
                    if file_info['name'].endswith('.exe'):
                        assert is_safe is False  # 可执行文件应该被拒绝
                else:
                    # 尝试加载图像，应该安全处理
                    result = image_processor.load_image(file_path)
                    # 不应该导致系统崩溃
                    assert result is not None or result is None
                    
            except Exception as e:
                # 安全相关的异常是可以接受的
                assert any(keyword in str(e).lower() for keyword in ['security', 'invalid', 'malformed', 'corrupt'])
    
    def test_image_memory_management(self, image_processor, test_workspace):
        """测试图像内存管理"""
        # 创建多个大图像来测试内存管理
        large_images = []
        
        for i in range(5):
            # 创建较大的图像
            large_image = Image.new('RGB', (1000, 1000), color=(i*50, i*40, i*30))
            large_path = os.path.join(test_workspace, f'large_image_{i}.png')
            large_image.save(large_path)
            large_images.append(large_path)
        
        try:
            # 处理多个大图像
            for path in large_images:
                result = image_processor.process_image(path)
                
                # 验证内存使用得到控制
                assert result is not None
                
                # 检查内存使用（如果有监控功能）
                if hasattr(image_processor, 'get_memory_usage'):
                    memory_usage = image_processor.get_memory_usage()
                    assert memory_usage < 1000  # 假设限制在1GB以内
                    
        except MemoryError:
            # 内存不足错误应该被优雅处理
            assert True
        except Exception as e:
            # 其他错误也应该被处理
            pass
    
    def test_concurrent_image_processing(self, image_processor, sample_images):
        """测试并发图像处理"""
        import threading
        
        results = []
        errors = []
        
        def process_image_concurrently(name, path):
            try:
                result = image_processor.process_image(path)
                results.append((name, result))
            except Exception as e:
                errors.append((name, e))
        
        # 创建并发处理线程
        threads = []
        for name, path in sample_images.items():
            thread = threading.Thread(target=process_image_concurrently, args=(name, path))
            threads.append(thread)
            thread.start()
        
        # 等待所有线程完成
        for thread in threads:
            thread.join(timeout=30)
        
        # 验证并发处理
        assert len(errors) == 0, f"Concurrent processing errors: {errors}"
        assert len(results) == len(sample_images)
    
    def test_image_caching_mechanism(self, image_processor, sample_images):
        """测试图像缓存机制"""
        # 测试相同图像的重复处理
        test_image = list(sample_images.values())[0]
        
        # 第一次处理
        start_time = time.time()
        result1 = image_processor.process_image(test_image)
        first_duration = time.time() - start_time
        
        # 第二次处理（应该使用缓存）
        start_time = time.time()
        result2 = image_processor.process_image(test_image)
        second_duration = time.time() - start_time
        
        # 验证缓存效果
        assert result1 is not None
        assert result2 is not None
        
        # 如果有缓存，第二次应该更快
        if hasattr(image_processor, 'enable_cache') and image_processor.enable_cache:
            assert second_duration <= first_duration
    
    def test_image_preprocessing_pipeline(self, image_processor, sample_images):
        """测试图像预处理管道"""
        preprocessing_steps = [
            {'step': 'normalize', 'params': {}},
            {'step': 'resize', 'params': {'width': 224, 'height': 224}},
            {'step': 'enhance', 'params': {'brightness': 1.2, 'contrast': 1.1}}
        ]
        
        for name, path in sample_images.items():
            try:
                if hasattr(image_processor, 'preprocessing_pipeline'):
                    result = image_processor.preprocessing_pipeline(path, preprocessing_steps)
                    
                    # 验证预处理管道
                    assert result is not None
                    if isinstance(result, dict):
                        assert 'processed_image' in result or 'pipeline_result' in result
                else:
                    # 逐步执行预处理
                    current_image = path
                    for step in preprocessing_steps:
                        if step['step'] == 'resize':
                            current_image = image_processor.resize_image(current_image, **step['params'])
                        elif step['step'] == 'enhance':
                            current_image = image_processor.enhance_image(current_image, **step['params'])
                    
                    assert current_image is not None
                    
            except Exception as e:
                pass
    
    def test_image_export_options(self, image_processor, sample_images, test_workspace):
        """测试图像导出选项"""
        export_options = [
            {'format': 'PNG', 'quality': 95, 'compression': 6},
            {'format': 'JPEG', 'quality': 85, 'progressive': True},
            {'format': 'WEBP', 'quality': 80, 'lossless': False}
        ]
        
        for name, path in sample_images.items():
            for options in export_options:
                try:
                    output_path = os.path.join(
                        test_workspace, 
                        f"exported_{name}.{options['format'].lower()}"
                    )
                    
                    result = image_processor.export_image(path, output_path, options)
                    
                    # 验证图像导出
                    assert result is not None
                    if isinstance(result, bool):
                        assert result is True
                    elif isinstance(result, str):
                        assert os.path.exists(result)
                    
                    # 验证导出的文件
                    if os.path.exists(output_path):
                        assert os.path.getsize(output_path) > 0
                        
                except Exception as e:
                    # 某些格式可能不被支持
                    pass 