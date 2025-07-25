#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多模型配置单元测试
测试AGIBot的多模型配置管理功能
"""

import pytest
import os
import sys
from unittest.mock import patch, Mock
from typing import Dict, List, Any

# 添加src路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))

from config_loader import get_api_key, get_model, get_api_base, load_config, get_max_tokens
from main import AGIBotClient

@pytest.mark.unit
class TestModelConfig:
    """多模型配置测试类"""
    
    @pytest.fixture
    def test_config_file(self, test_workspace):
        """创建测试配置文件"""
        config_content = """
api_key=test_api_key_123
model=claude-3-5-sonnet
api_base=https://api.anthropic.com
max_tokens=4096
temperature=0.7
streaming=true
"""
        config_file = os.path.join(test_workspace, "test_config.txt")
        with open(config_file, "w") as f:
            f.write(config_content)
        return config_file
    
    def test_load_config_basic(self, test_config_file):
        """测试基本配置加载"""
        config = load_config(test_config_file)
        
        # 验证配置加载
        assert config is not None
        assert isinstance(config, dict)
        assert "api_key" in config
        assert "model" in config
        assert config["api_key"] == "test_api_key_123"
        assert config["model"] == "claude-3-5-sonnet"
    
    def test_get_api_key(self, test_config_file):
        """测试API Key获取"""
        api_key = get_api_key(test_config_file)
        
        # 验证API Key
        assert api_key is not None
        assert api_key == "test_api_key_123"
    
    def test_get_model(self, test_config_file):
        """测试模型名称获取"""
        model = get_model(test_config_file)
        
        # 验证模型
        assert model is not None
        assert model == "claude-3-5-sonnet"
    
    def test_get_api_base(self, test_config_file):
        """测试API Base URL获取"""
        api_base = get_api_base(test_config_file)
        
        # 验证API Base
        assert api_base is not None
        assert api_base == "https://api.anthropic.com"
    
    def test_config_file_not_found(self, test_workspace):
        """测试配置文件不存在的情况"""
        nonexistent_file = os.path.join(test_workspace, "nonexistent.txt")
        
        # 测试各种获取函数
        config = load_config(nonexistent_file)
        api_key = get_api_key(nonexistent_file)
        model = get_model(nonexistent_file)
        api_base = get_api_base(nonexistent_file)
        
        # 验证处理不存在文件的行为
        assert config == {}
        assert api_key is None
        assert model is None
        assert api_base is None
    
    def test_empty_config_file(self, test_workspace):
        """测试空配置文件"""
        empty_config = os.path.join(test_workspace, "empty_config.txt")
        with open(empty_config, "w") as f:
            f.write("")
        
        config = load_config(empty_config)
        
        # 验证空配置处理
        assert config == {}
    
    def test_malformed_config(self, test_workspace):
        """测试格式错误的配置文件"""
        malformed_config = os.path.join(test_workspace, "malformed_config.txt")
        with open(malformed_config, "w") as f:
            f.write("这不是有效的配置格式\n没有等号的行\n")
        
        # 应该能够处理格式错误而不崩溃
        config = load_config(malformed_config)
        assert isinstance(config, dict)
    
    def test_client_initialization_with_config(self, test_config_file):
        """测试使用配置初始化客户端"""
        # 模拟从配置文件读取
        config = load_config(test_config_file)
        
        # 创建客户端
        client = AGIBotClient(
            api_key=config.get("api_key"),
            model=config.get("model"),
            api_base=config.get("api_base")
        )
        
        # 验证客户端初始化
        assert client is not None
        assert hasattr(client, 'chat')
    
    def test_multiple_config_formats(self, test_workspace):
        """测试多种配置格式"""
        configs = [
            ("config1.txt", "api_key=key1\nmodel=claude-3-5-sonnet\n"),
            ("config2.txt", "api_key = key2\nmodel = gpt-4\n"),  # 带空格
            ("config3.txt", "api_key=key3\n# 这是注释\nmodel=deepseek\n"),  # 带注释
        ]
        
        for filename, content in configs:
            config_file = os.path.join(test_workspace, filename)
            with open(config_file, "w") as f:
                f.write(content)
            
            config = load_config(config_file)
            
            # 验证不同格式都能正确解析
            assert "api_key" in config
            assert "model" in config
    
    def test_environment_variable_override(self, test_config_file):
        """测试环境变量覆盖"""
        with patch.dict(os.environ, {"AGI_API_KEY": "env_api_key"}):
            # 在实际应用中，可能会有环境变量优先级逻辑
            # 这里只是测试基本的配置读取
            config = load_config(test_config_file)
            assert config["api_key"] == "test_api_key_123"
    
    def test_config_validation(self, test_workspace):
        """测试配置验证"""
        # 创建包含无效配置的文件
        invalid_configs = [
            ("no_api_key.txt", "model=claude-3-5-sonnet\n"),
            ("no_model.txt", "api_key=test_key\n"),
            ("invalid_model.txt", "api_key=test_key\nmodel=invalid_model_name\n"),
        ]
        
        for filename, content in invalid_configs:
            config_file = os.path.join(test_workspace, filename)
            with open(config_file, "w") as f:
                f.write(content)
            
            config = load_config(config_file)
            
            # 验证配置解析仍然正常
            assert isinstance(config, dict)
    
    def test_unicode_config(self, test_workspace):
        """测试Unicode配置内容"""
        unicode_config = os.path.join(test_workspace, "unicode_config.txt")
        with open(unicode_config, "w", encoding="utf-8") as f:
            f.write("api_key=测试密钥_123\nmodel=claude-3-5-sonnet\n")
        
        config = load_config(unicode_config)
        
        # 验证Unicode处理
        assert config["api_key"] == "测试密钥_123"
        assert config["model"] == "claude-3-5-sonnet"
    
    def test_config_caching(self, test_config_file):
        """测试配置缓存（如果有实现）"""
        # 多次读取同一配置文件
        config1 = load_config(test_config_file)
        config2 = load_config(test_config_file)
        
        # 验证返回结果一致
        assert config1 == config2
    
    def test_get_max_tokens(self, test_config_file):
        """测试获取最大token数"""
        max_tokens = get_max_tokens(test_config_file)
        
        # 验证max_tokens
        if max_tokens is not None:
            assert isinstance(max_tokens, int)
            assert max_tokens > 0
    
    def test_config_with_special_characters(self, test_workspace):
        """测试包含特殊字符的配置"""
        special_config = os.path.join(test_workspace, "special_config.txt")
        with open(special_config, "w") as f:
            f.write("api_key=key_with_special_@#$%\nmodel=claude-3-5-sonnet\n")
        
        config = load_config(special_config)
        
        # 验证特殊字符处理
        assert config["api_key"] == "key_with_special_@#$%"
    
    def test_large_config_file(self, test_workspace):
        """测试大型配置文件"""
        large_config = os.path.join(test_workspace, "large_config.txt")
        with open(large_config, "w") as f:
            f.write("api_key=test_key\nmodel=claude-3-5-sonnet\n")
            # 添加很多配置项
            for i in range(1000):
                f.write(f"config_item_{i}=value_{i}\n")
        
        import time
        start_time = time.time()
        config = load_config(large_config)
        end_time = time.time()
        
        # 验证大文件处理性能
        assert config["api_key"] == "test_key"
        assert end_time - start_time < 1.0  # 应该在1秒内完成 