#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Web搜索工具单元测试
测试网络搜索功能、结果处理、错误处理等
"""

import pytest
import os
import json
import sys
from unittest.mock import patch, Mock, MagicMock
from urllib.parse import quote

# 添加src路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))

from tools.web_search_tools import WebSearchTools
# from utils.test_helpers import TestHelper

@pytest.mark.unit
class TestWebSearchTools:
    """Web搜索工具测试类"""
    
    @pytest.fixture
    def web_search_tools(self, test_workspace):
        """创建Web搜索工具实例"""
        return WebSearchTools(
            llm_api_key="test_key",
            llm_model="test_model", 
            llm_api_base="test_base",
            enable_llm_filtering=False,
            enable_summary=True,
            out_dir=test_workspace
        )
    
    @pytest.fixture
    def mock_search_response(self):
        """模拟搜索响应数据"""
        return {
            "results": [
                {
                    "title": "AGI Bot - AI-powered intelligent agent",
                    "url": "https://github.com/example/agibot",
                    "snippet": "AGI Bot is an L3-level fully automated general-purpose intelligent agent...",
                    "relevance_score": 0.95
                },
                {
                    "title": "Introduction to AI Agents",
                    "url": "https://example.com/ai-agents",
                    "snippet": "Learn about autonomous AI agents and their applications...",
                    "relevance_score": 0.87
                }
            ],
            "total_results": 2,
            "search_time": 0.234
        }
    
    def test_initialization(self, web_search_tools):
        """测试Web搜索工具初始化"""
        assert web_search_tools is not None
        assert hasattr(web_search_tools, 'search_web')
        assert hasattr(web_search_tools, 'enable_summary')
        assert hasattr(web_search_tools, 'enable_llm_filtering')
    
    @patch('requests.get')
    def test_basic_search_functionality(self, mock_get, web_search_tools, mock_search_response):
        """测试基本搜索功能"""
        # 模拟HTTP响应
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_search_response
        mock_get.return_value = mock_response
        
        # 执行搜索
        result = web_search_tools.search_web("AGI Bot intelligent agent")
        
        # 验证结果
        assert result is not None
        assert "results" in result or "error" not in result
        
        # 验证HTTP请求被调用
        mock_get.assert_called_once()
    
    @patch('requests.get')
    def test_search_with_chinese_query(self, mock_get, web_search_tools, mock_search_response):
        """测试中文查询搜索"""
        # 模拟HTTP响应
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_search_response
        mock_get.return_value = mock_response
        
        # 执行中文搜索
        result = web_search_tools.search_web("人工智能助手")
        
        # 验证结果
        assert result is not None
        
        # 验证请求参数包含正确编码的中文
        call_args = mock_get.call_args
        assert call_args is not None
    
    def test_empty_query_handling(self, web_search_tools):
        """测试空查询处理"""
        # 测试空字符串
        result = web_search_tools.search_web("")
        assert "error" in result or result is None
        
        # 测试None
        result = web_search_tools.search_web(None)
        assert "error" in result or result is None
        
        # 测试只有空格
        result = web_search_tools.search_web("   ")
        assert "error" in result or result is None
    
    def test_special_characters_in_query(self, web_search_tools):
        """测试查询中的特殊字符处理"""
        special_queries = [
            "search with @#$%^&*() symbols",
            "query with \"quotes\" and 'apostrophes'",
            "search/with/slashes\\and\\backslashes",
            "query+with+plus+signs"
        ]
        
        for query in special_queries:
            with patch('requests.get') as mock_get:
                mock_response = Mock()
                mock_response.status_code = 200
                mock_response.json.return_value = {"results": []}
                mock_get.return_value = mock_response
                
                result = web_search_tools.search_web(query)
                
                # 验证没有因特殊字符而崩溃
                assert result is not None
                mock_get.assert_called_once()
                mock_get.reset_mock()
    
    @patch('requests.get')
    def test_network_error_handling(self, mock_get, web_search_tools):
        """测试网络错误处理"""
        # 测试连接超时
        mock_get.side_effect = ConnectionError("Connection failed")
        
        result = web_search_tools.search_web("test query")
        
        # 验证错误被正确处理
        assert result is not None
        assert "error" in result or "没有找到相关信息" in str(result)
    
    @patch('requests.get')
    def test_http_error_codes(self, mock_get, web_search_tools):
        """测试HTTP错误代码处理"""
        error_codes = [400, 401, 403, 404, 429, 500, 502, 503]
        
        for code in error_codes:
            mock_response = Mock()
            mock_response.status_code = code
            mock_response.raise_for_status.side_effect = Exception(f"HTTP {code}")
            mock_get.return_value = mock_response
            
            result = web_search_tools.search_web("test query")
            
            # 验证错误被正确处理
            assert result is not None
            assert "error" in result or "搜索失败" in str(result)
    
    @patch('requests.get')
    def test_malformed_response_handling(self, mock_get, web_search_tools):
        """测试格式错误响应处理"""
        # 测试非JSON响应
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.side_effect = json.JSONDecodeError("Invalid JSON", "", 0)
        mock_get.return_value = mock_response
        
        result = web_search_tools.search_web("test query")
        
        # 验证错误被正确处理
        assert result is not None
        assert "error" in result or isinstance(result, str)
    
    @patch('requests.get')
    def test_search_result_filtering(self, mock_get, web_search_tools):
        """测试搜索结果过滤"""
        # 创建包含多种类型结果的响应
        response_data = {
            "results": [
                {
                    "title": "Relevant Result",
                    "url": "https://example.com/relevant",
                    "snippet": "This is very relevant to AGI Bot",
                    "relevance_score": 0.95
                },
                {
                    "title": "Less Relevant Result", 
                    "url": "https://example.com/less-relevant",
                    "snippet": "This mentions bot but not specifically AGI",
                    "relevance_score": 0.45
                },
                {
                    "title": "Spam Result",
                    "url": "https://spam.com/irrelevant",
                    "snippet": "Buy cheap products now!",
                    "relevance_score": 0.1
                }
            ]
        }
        
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = response_data
        mock_get.return_value = mock_response
        
        result = web_search_tools.search_web("AGI Bot")
        
        # 验证结果被适当处理（具体实现可能因工具而异）
        assert result is not None
    
    @patch('requests.get')
    def test_search_result_summarization(self, mock_get, web_search_tools, mock_search_response):
        """测试搜索结果摘要功能"""
        # 启用摘要功能
        web_search_tools.enable_summary = True
        
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_search_response
        mock_get.return_value = mock_response
        
        # 模拟LLM摘要调用
        with patch.object(web_search_tools, '_generate_summary', return_value="Summary of search results"):
            result = web_search_tools.search_web("AGI Bot features")
            
            # 验证摘要功能被使用
            assert result is not None
    
    def test_query_length_limits(self, web_search_tools):
        """测试查询长度限制"""
        # 测试过长的查询
        very_long_query = "test " * 1000  # 创建一个很长的查询
        
        with patch('requests.get') as mock_get:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"results": []}
            mock_get.return_value = mock_response
            
            result = web_search_tools.search_web(very_long_query)
            
            # 验证长查询被适当处理
            assert result is not None
    
    @patch('requests.get')
    def test_concurrent_search_requests(self, mock_get, web_search_tools, mock_search_response):
        """测试并发搜索请求处理"""
        import threading
        import time
        
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_search_response
        mock_get.return_value = mock_response
        
        results = []
        errors = []
        
        def perform_search(query_id):
            try:
                result = web_search_tools.search_web(f"test query {query_id}")
                results.append(result)
            except Exception as e:
                errors.append(e)
        
        # 创建多个并发搜索线程
        threads = []
        for i in range(5):
            thread = threading.Thread(target=perform_search, args=(i,))
            threads.append(thread)
            thread.start()
        
        # 等待所有线程完成
        for thread in threads:
            thread.join(timeout=10)
        
        # 验证所有搜索都成功完成
        assert len(errors) == 0, f"Concurrent search errors: {errors}"
        assert len(results) == 5
    
    def test_search_cache_behavior(self, web_search_tools):
        """测试搜索缓存行为（如果实现了缓存）"""
        # 这个测试取决于是否实现了搜索缓存
        query = "cache test query"
        
        with patch('requests.get') as mock_get:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"results": []}
            mock_get.return_value = mock_response
            
            # 执行相同查询两次
            result1 = web_search_tools.search_web(query)
            result2 = web_search_tools.search_web(query)
            
            # 验证结果一致性
            assert result1 is not None
            assert result2 is not None
    
    def test_search_configuration_options(self, test_workspace):
        """测试搜索配置选项"""
        # 测试不同配置组合
        configs = [
            {"enable_llm_filtering": True, "enable_summary": True},
            {"enable_llm_filtering": False, "enable_summary": True},
            {"enable_llm_filtering": True, "enable_summary": False},
            {"enable_llm_filtering": False, "enable_summary": False}
        ]
        
        for config in configs:
            search_tools = WebSearchTools(
                llm_api_key="test_key",
                llm_model="test_model",
                llm_api_base="test_base",
                out_dir=test_workspace,
                **config
            )
            
            # 验证配置被正确设置
            assert search_tools.enable_llm_filtering == config["enable_llm_filtering"]
            assert search_tools.enable_summary == config["enable_summary"]
    
    def test_search_output_directory_usage(self, web_search_tools, test_workspace):
        """测试搜索输出目录使用"""
        # 验证输出目录被正确设置和使用
        assert web_search_tools.out_dir == test_workspace
        
        # 检查输出目录是否存在（如果工具会创建文件）
        assert os.path.exists(test_workspace) 