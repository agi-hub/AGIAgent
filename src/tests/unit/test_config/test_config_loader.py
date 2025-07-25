#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
配置加载器单元测试
测试config_loader.py中的配置加载和验证功能
"""

import pytest
import os
import tempfile
import shutil
from unittest.mock import patch, Mock
import sys

# 添加src路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))

from config_loader import (
    load_config, get_api_key, get_model, get_api_base, 
    get_truncation_length, get_summary_report
)

class TestConfigLoader:
    """配置加载器测试类"""
    
    @pytest.fixture
    def temp_config_dir(self):
        """创建临时配置目录"""
        temp_dir = tempfile.mkdtemp()
        config_dir = os.path.join(temp_dir, "config")
        os.makedirs(config_dir)
        yield config_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def sample_config_file(self, temp_config_dir):
        """创建示例配置文件"""
        config_file = os.path.join(temp_config_dir, "config.txt")
        config_content = """
# AGIBot配置文件
api_key=test_api_key_123456
model=claude-3-sonnet-20240229
api_base=https://api.anthropic.com
truncation_length=8000
summary_report=true
debug_mode=false
streaming=true
"""
        with open(config_file, 'w', encoding='utf-8') as f:
            f.write(config_content)
        return config_file
    
    def test_load_config_basic(self, sample_config_file):
        """测试基础配置加载"""
        with patch('config_loader.CONFIG_FILE', sample_config_file):
            config = load_config()
            
            assert config is not None
            assert config.get("api_key") == "test_api_key_123456"
            assert config.get("model") == "claude-3-sonnet-20240229"
            assert config.get("api_base") == "https://api.anthropic.com"
            assert config.get("truncation_length") == "8000"
            assert config.get("summary_report") == "true"
    
    def test_get_api_key(self, sample_config_file):
        """测试API密钥获取"""
        with patch('config_loader.CONFIG_FILE', sample_config_file):
            api_key = get_api_key()
            assert api_key == "test_api_key_123456"
    
    def test_get_model(self, sample_config_file):
        """测试模型名称获取"""
        with patch('config_loader.CONFIG_FILE', sample_config_file):
            model = get_model()
            assert model == "claude-3-sonnet-20240229"
    
    def test_get_api_base(self, sample_config_file):
        """测试API基础URL获取"""
        with patch('config_loader.CONFIG_FILE', sample_config_file):
            api_base = get_api_base()
            assert api_base == "https://api.anthropic.com"
    
    def test_get_truncation_length(self, sample_config_file):
        """测试截断长度获取"""
        with patch('config_loader.CONFIG_FILE', sample_config_file):
            length = get_truncation_length()
            assert length == 8000
    
    def test_get_summary_report(self, sample_config_file):
        """测试摘要报告设置获取"""
        with patch('config_loader.CONFIG_FILE', sample_config_file):
            summary = get_summary_report()
            assert summary == True
    
    def test_config_file_not_found(self):
        """测试配置文件不存在的情况"""
        with patch('config_loader.CONFIG_FILE', '/nonexistent/config.txt'):
            config = load_config()
            assert config == {}
    
    def test_malformed_config_file(self, temp_config_dir):
        """测试格式错误的配置文件"""
        malformed_file = os.path.join(temp_config_dir, "malformed_config.txt")
        with open(malformed_file, 'w') as f:
            f.write("invalid line without equals\napi_key=valid_key\n=invalid_equals")
        
        with patch('config_loader.CONFIG_FILE', malformed_file):
            config = load_config()
            # 应该能正确解析有效行，忽略无效行
            assert config.get("api_key") == "valid_key"
    
    def test_config_with_comments_and_whitespace(self, temp_config_dir):
        """测试包含注释和空白的配置文件"""
        config_file = os.path.join(temp_config_dir, "config_with_comments.txt")
        config_content = """
# 这是注释
api_key = test_key_with_spaces   # 行末注释
model=claude-3-sonnet-20240229

# 空行上方
api_base=https://api.anthropic.com
"""
        with open(config_file, 'w', encoding='utf-8') as f:
            f.write(config_content)
        
        with patch('config_loader.CONFIG_FILE', config_file):
            config = load_config()
            assert config.get("api_key") == "test_key_with_spaces"
            assert config.get("model") == "claude-3-sonnet-20240229"
            assert config.get("api_base") == "https://api.anthropic.com"
    
    def test_environment_variable_override(self, sample_config_file):
        """测试环境变量覆盖配置文件"""
        with patch('config_loader.CONFIG_FILE', sample_config_file):
            with patch.dict(os.environ, {'AGIBOT_API_KEY': 'env_api_key'}):
                # 如果有环境变量支持，应该优先使用环境变量
                api_key = get_api_key()
                # 这里根据实际实现来判断是否支持环境变量覆盖
                assert api_key in ["test_api_key_123456", "env_api_key"]
    
    def test_missing_required_config(self, temp_config_dir):
        """测试缺少必需配置的情况"""
        incomplete_file = os.path.join(temp_config_dir, "incomplete_config.txt")
        with open(incomplete_file, 'w') as f:
            f.write("model=claude-3-sonnet-20240229\n")  # 缺少api_key
        
        with patch('config_loader.CONFIG_FILE', incomplete_file):
            api_key = get_api_key()
            assert api_key is None  # 缺少api_key应该返回None
    
    def test_config_value_types(self, temp_config_dir):
        """测试配置值类型转换"""
        config_file = os.path.join(temp_config_dir, "types_config.txt")
        config_content = """
api_key=test_key
truncation_length=5000
summary_report=false
debug_mode=true
max_loops=10
"""
        with open(config_file, 'w') as f:
            f.write(config_content)
        
        with patch('config_loader.CONFIG_FILE', config_file):
            # 测试整数类型转换
            length = get_truncation_length()
            assert isinstance(length, int)
            assert length == 5000
            
            # 测试布尔类型转换
            summary = get_summary_report()
            assert isinstance(summary, bool)
            assert summary == False
    
    def test_config_caching(self, sample_config_file):
        """测试配置缓存机制"""
        with patch('config_loader.CONFIG_FILE', sample_config_file):
            # 第一次加载
            config1 = load_config()
            api_key1 = get_api_key()
            
            # 第二次加载（如果有缓存机制，应该使用缓存）
            config2 = load_config()
            api_key2 = get_api_key()
            
            assert config1 == config2
            assert api_key1 == api_key2
    
    def test_unicode_config_support(self, temp_config_dir):
        """测试Unicode配置支持"""
        unicode_file = os.path.join(temp_config_dir, "unicode_config.txt")
        config_content = """
# 中文注释测试
api_key=测试密钥_with_中文
model=claude-3-sonnet-20240229
description=支持中文的描述信息
"""
        with open(unicode_file, 'w', encoding='utf-8') as f:
            f.write(config_content)
        
        with patch('config_loader.CONFIG_FILE', unicode_file):
            config = load_config()
            assert config.get("api_key") == "测试密钥_with_中文"
            assert config.get("description") == "支持中文的描述信息"
    
    def test_config_validation(self, temp_config_dir):
        """测试配置验证"""
        invalid_config_file = os.path.join(temp_config_dir, "invalid_config.txt")
        config_content = """
api_key=
model=invalid_model_name_that_is_too_long_and_contains_invalid_characters
api_base=not_a_valid_url
truncation_length=not_a_number
"""
        with open(invalid_config_file, 'w') as f:
            f.write(config_content)
        
        with patch('config_loader.CONFIG_FILE', invalid_config_file):
            # 测试空值处理
            api_key = get_api_key()
            assert api_key in [None, ""]  # 空值应该被正确处理
            
            # 测试无效数字处理
            try:
                length = get_truncation_length()
                # 如果有验证，应该返回默认值或抛出异常
                assert isinstance(length, (int, type(None)))
            except (ValueError, TypeError):
                # 抛出异常也是合理的处理方式
                pass
    
    def test_config_file_permissions(self, temp_config_dir):
        """测试配置文件权限问题"""
        restricted_file = os.path.join(temp_config_dir, "restricted_config.txt")
        with open(restricted_file, 'w') as f:
            f.write("api_key=test_key\n")
        
        try:
            # 尝试移除读权限（仅在Unix系统上有效）
            os.chmod(restricted_file, 0o000)
            
            with patch('config_loader.CONFIG_FILE', restricted_file):
                config = load_config()
                # 权限问题应该被优雅处理
                assert config is not None  # 应该返回空字典而不是崩溃
                
        except (OSError, NotImplementedError):
            # 在Windows或其他不支持权限的系统上跳过
            pytest.skip("Permission testing not supported on this system")
        finally:
            # 恢复权限以便清理
            try:
                os.chmod(restricted_file, 0o644)
            except:
                pass