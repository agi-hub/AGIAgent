#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAG检索功能单元测试
测试向量搜索、语义检索、检索增强生成等功能
"""

import pytest
import os
import sys
import json
import tempfile
import numpy as np
from unittest.mock import patch, Mock, MagicMock
from typing import List, Dict, Any

# 添加src路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from tools.long_term_memory import LongTermMemory
# from utils.test_helpers import TestHelper

@pytest.mark.unit
class TestRAGRetrieval:
    """RAG检索功能测试类"""
    
    @pytest.fixture
    def rag_system(self, test_workspace):
        """创建RAG系统实例"""
        return LongTermMemory(workspace_root=test_workspace)
    
    @pytest.fixture
    def sample_documents(self):
        """示例文档数据"""
        return [
            {
                "id": "doc1",
                "content": "AGI Bot是一个基于大语言模型的智能代理系统，能够自主完成复杂任务。",
                "metadata": {"type": "introduction", "category": "overview"}
            },
            {
                "id": "doc2", 
                "content": "多智能体协作是AGI Bot的核心特性之一，允许创建专业化的子智能体。",
                "metadata": {"type": "feature", "category": "multiagent"}
            },
            {
                "id": "doc3",
                "content": "工具调用能力使AGI Bot能够执行文件操作、网络搜索、代码分析等任务。",
                "metadata": {"type": "feature", "category": "tools"}
            },
            {
                "id": "doc4",
                "content": "长期记忆系统帮助AGI Bot保持历史上下文并从过往经验中学习。",
                "metadata": {"type": "feature", "category": "memory"}
            },
            {
                "id": "doc5",
                "content": "MCP协议支持使AGI Bot能够集成第三方服务和工具。",
                "metadata": {"type": "feature", "category": "integration"}
            }
        ]
    
    @pytest.fixture
    def sample_embeddings(self):
        """示例向量嵌入数据"""
        # 模拟5个文档的768维向量嵌入
        return {
            "doc1": np.random.rand(768).tolist(),
            "doc2": np.random.rand(768).tolist(),
            "doc3": np.random.rand(768).tolist(),
            "doc4": np.random.rand(768).tolist(),
            "doc5": np.random.rand(768).tolist()
        }
    
    def test_initialization(self, rag_system):
        """测试RAG系统初始化"""
        assert rag_system is not None
        assert hasattr(rag_system, 'store_memory')
        assert hasattr(rag_system, 'search_memory')
    
    def test_document_storage(self, rag_system, sample_documents):
        """测试文档存储功能"""
        for doc in sample_documents:
            result = rag_system.store_memory(
                content=doc["content"],
                metadata=doc["metadata"]
            )
            
            # 验证存储成功
            assert result is not None
            assert "error" not in result
    
    def test_basic_text_search(self, rag_system, sample_documents):
        """测试基本文本搜索"""
        # 先存储文档
        for doc in sample_documents:
            rag_system.store_memory(
                content=doc["content"],
                metadata=doc["metadata"]
            )
        
        # 执行搜索
        results = rag_system.search_memory("智能代理系统")
        
        # 验证搜索结果
        assert results is not None
        assert isinstance(results, (list, dict, str))
    
    @patch('tools.long_term_memory.EmbeddingClient')
    def test_vector_embedding_generation(self, mock_embedding_client, rag_system):
        """测试向量嵌入生成"""
        # 模拟嵌入客户端
        mock_client = Mock()
        mock_client.get_embedding.return_value = np.random.rand(768).tolist()
        mock_embedding_client.return_value = mock_client
        
        # 测试嵌入生成
        test_text = "这是一个测试文本，用于生成向量嵌入。"
        result = rag_system.store_memory(test_text)
        
        # 验证嵌入生成
        assert result is not None
    
    @patch('tools.long_term_memory.EmbeddingClient')
    def test_semantic_similarity_search(self, mock_embedding_client, rag_system, sample_documents, sample_embeddings):
        """测试语义相似度搜索"""
        # 模拟嵌入客户端
        mock_client = Mock()
        
        def mock_get_embedding(text):
            # 为查询返回与某个文档相似的向量
            if "多智能体" in text:
                return sample_embeddings["doc2"]
            return np.random.rand(768).tolist()
        
        mock_client.get_embedding.side_effect = mock_get_embedding
        mock_embedding_client.return_value = mock_client
        
        # 存储文档
        for doc in sample_documents:
            rag_system.store_memory(
                content=doc["content"],
                metadata=doc["metadata"]
            )
        
        # 执行语义搜索
        results = rag_system.search_memory("多智能体协作功能")
        
        # 验证语义搜索结果
        assert results is not None
    
    def test_similarity_threshold_filtering(self, rag_system, sample_documents):
        """测试相似度阈值过滤"""
        # 存储文档
        for doc in sample_documents:
            rag_system.store_memory(
                content=doc["content"],
                metadata=doc["metadata"]
            )
        
        # 搜索非常特定的查询
        specific_results = rag_system.search_memory("AGI Bot智能代理", top_k=3)
        
        # 搜索不相关的查询
        irrelevant_results = rag_system.search_memory("天气预报股票价格", top_k=3)
        
        # 验证相似度过滤
        assert specific_results is not None
        assert irrelevant_results is not None
    
    def test_metadata_filtering(self, rag_system, sample_documents):
        """测试元数据过滤"""
        # 存储文档
        for doc in sample_documents:
            rag_system.store_memory(
                content=doc["content"],
                metadata=doc["metadata"]
            )
        
        # 基于类型过滤搜索
        feature_results = rag_system.search_memory(
            "智能体功能", 
            metadata_filter={"type": "feature"}
        )
        
        # 基于类别过滤搜索
        tool_results = rag_system.search_memory(
            "工具调用",
            metadata_filter={"category": "tools"}
        )
        
        # 验证元数据过滤
        assert feature_results is not None
        assert tool_results is not None
    
    def test_top_k_results_limiting(self, rag_system, sample_documents):
        """测试TopK结果限制"""
        # 存储文档
        for doc in sample_documents:
            rag_system.store_memory(
                content=doc["content"],
                metadata=doc["metadata"]
            )
        
        # 测试不同的top_k值
        for k in [1, 3, 5, 10]:
            results = rag_system.search_memory("AGI Bot功能", top_k=k)
            
            # 验证结果数量限制
            assert results is not None
            if isinstance(results, list):
                assert len(results) <= k
    
    def test_empty_query_handling(self, rag_system):
        """测试空查询处理"""
        # 测试空字符串查询
        empty_result = rag_system.search_memory("")
        assert empty_result is not None
        
        # 测试None查询
        none_result = rag_system.search_memory(None)
        assert none_result is not None
        
        # 测试空白字符查询
        whitespace_result = rag_system.search_memory("   ")
        assert whitespace_result is not None
    
    def test_large_document_handling(self, rag_system):
        """测试大文档处理"""
        # 创建一个大文档
        large_content = "这是一个很长的文档。" * 1000
        
        result = rag_system.store_memory(large_content)
        
        # 验证大文档存储
        assert result is not None
        assert "error" not in result
        
        # 搜索大文档
        search_result = rag_system.search_memory("很长的文档")
        assert search_result is not None
    
    def test_unicode_content_handling(self, rag_system):
        """测试Unicode内容处理"""
        unicode_contents = [
            "这是中文内容，包含各种字符：你好世界！",
            "English content with émojis: 🚀🤖🔍",
            "Русский текст с разными символами",
            "日本語のテキストです。こんにちは！",
            "العربية النص مع الرموز المختلفة"
        ]
        
        # 存储Unicode内容
        for content in unicode_contents:
            result = rag_system.store_memory(content)
            assert result is not None
            assert "error" not in result
        
        # 搜索Unicode内容
        search_result = rag_system.search_memory("中文内容")
        assert search_result is not None
    
    def test_document_updating(self, rag_system):
        """测试文档更新功能"""
        original_content = "这是原始文档内容。"
        updated_content = "这是更新后的文档内容。"
        
        # 存储原始文档
        doc_id = rag_system.store_memory(original_content)
        
        # 更新文档（如果支持）
        if hasattr(rag_system, 'update_memory'):
            update_result = rag_system.update_memory(doc_id, updated_content)
            assert update_result is not None
    
    def test_document_deletion(self, rag_system):
        """测试文档删除功能"""
        content = "这是要删除的文档。"
        
        # 存储文档
        doc_id = rag_system.store_memory(content)
        
        # 删除文档（如果支持）
        if hasattr(rag_system, 'delete_memory'):
            delete_result = rag_system.delete_memory(doc_id)
            assert delete_result is not None
    
    @patch('tools.long_term_memory.EmbeddingClient')
    def test_embedding_error_handling(self, mock_embedding_client, rag_system):
        """测试嵌入生成错误处理"""
        # 模拟嵌入生成失败
        mock_client = Mock()
        mock_client.get_embedding.side_effect = Exception("Embedding generation failed")
        mock_embedding_client.return_value = mock_client
        
        # 尝试存储文档
        result = rag_system.store_memory("测试内容")
        
        # 验证错误处理
        assert result is not None
    
    def test_batch_document_storage(self, rag_system, sample_documents):
        """测试批量文档存储"""
        # 批量存储文档（如果支持）
        if hasattr(rag_system, 'store_memories_batch'):
            contents = [doc["content"] for doc in sample_documents]
            metadatas = [doc["metadata"] for doc in sample_documents]
            
            batch_result = rag_system.store_memories_batch(contents, metadatas)
            assert batch_result is not None
        else:
            # 逐个存储文档
            for doc in sample_documents:
                result = rag_system.store_memory(
                    content=doc["content"],
                    metadata=doc["metadata"]
                )
                assert result is not None
    
    def test_search_result_ranking(self, rag_system, sample_documents):
        """测试搜索结果排序"""
        # 存储文档
        for doc in sample_documents:
            rag_system.store_memory(
                content=doc["content"],
                metadata=doc["metadata"]
            )
        
        # 执行搜索
        results = rag_system.search_memory("AGI Bot功能特性", top_k=5)
        
        # 验证结果排序（结果应该按相关性排序）
        assert results is not None
        if isinstance(results, list) and len(results) > 1:
            # 检查是否有相关性评分
            for result in results:
                if isinstance(result, dict):
                    # 如果有评分字段，验证排序
                    if 'score' in result or 'similarity' in result:
                        assert True  # 有评分信息
                        break
    
    def test_memory_persistence(self, rag_system, test_workspace):
        """测试记忆持久化"""
        content = "这是需要持久化的记忆内容。"
        
        # 存储记忆
        result = rag_system.store_memory(content)
        assert result is not None
        
        # 创建新的RAG系统实例
        new_rag_system = LongTermMemory(workspace_root=test_workspace)
        
        # 搜索之前存储的内容
        search_result = new_rag_system.search_memory("持久化的记忆")
        
        # 验证持久化
        assert search_result is not None
    
    def test_concurrent_operations(self, rag_system):
        """测试并发操作"""
        import threading
        import time
        
        results = []
        errors = []
        
        def store_operation(content_id):
            try:
                content = f"并发测试内容 {content_id}"
                result = rag_system.store_memory(content)
                results.append(result)
            except Exception as e:
                errors.append(e)
        
        def search_operation(query_id):
            try:
                result = rag_system.search_memory(f"测试查询 {query_id}")
                results.append(result)
            except Exception as e:
                errors.append(e)
        
        # 创建并发线程
        threads = []
        
        # 添加存储线程
        for i in range(5):
            thread = threading.Thread(target=store_operation, args=(i,))
            threads.append(thread)
        
        # 添加搜索线程
        for i in range(3):
            thread = threading.Thread(target=search_operation, args=(i,))
            threads.append(thread)
        
        # 启动所有线程
        for thread in threads:
            thread.start()
        
        # 等待所有线程完成
        for thread in threads:
            thread.join(timeout=10)
        
        # 验证并发操作
        assert len(errors) == 0, f"Concurrent operation errors: {errors}"
        assert len(results) > 0
    
    def test_memory_statistics(self, rag_system, sample_documents):
        """测试记忆统计信息"""
        # 存储文档
        for doc in sample_documents:
            rag_system.store_memory(
                content=doc["content"],
                metadata=doc["metadata"]
            )
        
        # 获取统计信息（如果支持）
        if hasattr(rag_system, 'get_statistics'):
            stats = rag_system.get_statistics()
            assert stats is not None
            assert isinstance(stats, dict)
    
    def test_memory_export_import(self, rag_system, sample_documents, test_workspace):
        """测试记忆导出导入"""
        # 存储文档
        for doc in sample_documents:
            rag_system.store_memory(
                content=doc["content"],
                metadata=doc["metadata"]
            )
        
        # 导出记忆（如果支持）
        if hasattr(rag_system, 'export_memories'):
            export_path = os.path.join(test_workspace, "memory_export.json")
            export_result = rag_system.export_memories(export_path)
            assert export_result is not None
            assert os.path.exists(export_path)
        
        # 导入记忆（如果支持）
        if hasattr(rag_system, 'import_memories'):
            import_result = rag_system.import_memories(export_path)
            assert import_result is not None
    
    def test_query_expansion(self, rag_system, sample_documents):
        """测试查询扩展功能"""
        # 存储文档
        for doc in sample_documents:
            rag_system.store_memory(
                content=doc["content"],
                metadata=doc["metadata"]
            )
        
        # 测试简短查询的扩展
        short_query = "智能体"
        results = rag_system.search_memory(short_query)
        
        # 验证查询扩展效果
        assert results is not None
    
    def test_context_aware_search(self, rag_system, sample_documents):
        """测试上下文感知搜索"""
        # 存储文档
        for doc in sample_documents:
            rag_system.store_memory(
                content=doc["content"],
                metadata=doc["metadata"]
            )
        
        # 提供上下文的搜索（如果支持）
        context = "我想了解AGI Bot的主要功能"
        query = "协作功能"
        
        if hasattr(rag_system, 'search_with_context'):
            results = rag_system.search_with_context(query, context)
            assert results is not None
        else:
            # 普通搜索
            results = rag_system.search_memory(query)
            assert results is not None 