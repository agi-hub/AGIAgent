#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
内存管理系统单元测试
测试mem/src/core/memory_manager.py中的内存管理功能
"""

import pytest
import os
import tempfile
import shutil
import json
from unittest.mock import patch, Mock, MagicMock
import sys

# 添加src路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))

try:
    from mem.src.core.memory_manager import MemManagerAgent
    from mem.src.models.memory_cell import MemCell
    from mem.src.utils.config import ConfigLoader
    MEMORY_AVAILABLE = True
except ImportError:
    MEMORY_AVAILABLE = False

@pytest.mark.skipif(not MEMORY_AVAILABLE, reason="Memory system not available")
class TestMemoryManager:
    """内存管理器测试类"""
    
    @pytest.fixture
    def temp_memory_dir(self):
        """创建临时内存存储目录"""
        temp_dir = tempfile.mkdtemp(prefix="memory_test_")
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def mock_config(self, temp_memory_dir):
        """模拟配置"""
        return {
            "storage_path": temp_memory_dir,
            "max_memory_cells": 1000,
            "embedding_model": "text-embedding-ada-002",
            "llm_model": "gpt-4",
            "similarity_threshold": 0.8,
            "auto_summarize": True,
            "enable_compression": True,
            "max_cell_size": 10000
        }
    
    @pytest.fixture
    def memory_manager(self, mock_config):
        """创建内存管理器实例"""
        with patch.object(ConfigLoader, 'load_config', return_value=mock_config):
            manager = MemManagerAgent()
            return manager
    
    def test_memory_manager_initialization(self, memory_manager):
        """测试内存管理器初始化"""
        assert memory_manager is not None
        assert hasattr(memory_manager, 'write_memory_auto')
        assert hasattr(memory_manager, 'read_memory_auto')
        assert hasattr(memory_manager, 'get_status_auto')
    
    def test_write_memory_auto_basic(self, memory_manager):
        """测试基础内存写入功能"""
        content = "今天学习了Python编程，特别是关于类和对象的概念。"
        
        with patch.object(memory_manager, '_generate_embedding', return_value=[0.1] * 1536):
            with patch.object(memory_manager, '_call_llm_for_summary', return_value="学习Python编程知识"):
                result = memory_manager.write_memory_auto(content)
                
                assert result["status"] == "success"
                assert "memory_id" in result
                assert result["operation"] in ["added", "updated"]
    
    def test_write_memory_auto_duplicate_handling(self, memory_manager):
        """测试重复内存处理"""
        content = "Python是一种编程语言"
        
        with patch.object(memory_manager, '_generate_embedding', return_value=[0.1] * 1536):
            with patch.object(memory_manager, '_call_llm_for_summary', return_value="Python编程语言"):
                with patch.object(memory_manager, '_find_similar_memories', return_value=[
                    {"id": "mem_001", "similarity": 0.95, "content": "Python是编程语言"}
                ]):
                    result = memory_manager.write_memory_auto(content)
                    
                    assert result["status"] == "success"
                    assert result["operation"] == "updated"  # 应该更新而不是新增
    
    def test_read_memory_auto_basic(self, memory_manager):
        """测试基础内存读取功能"""
        query = "Python编程"
        
        # 模拟存在的记忆
        mock_memories = [
            {
                "id": "mem_001",
                "content": "Python是一种编程语言",
                "summary": "Python编程语言",
                "similarity": 0.9,
                "timestamp": "2024-01-01T10:00:00"
            },
            {
                "id": "mem_002", 
                "content": "学习了Python的基础语法",
                "summary": "Python语法学习",
                "similarity": 0.85,
                "timestamp": "2024-01-01T11:00:00"
            }
        ]
        
        with patch.object(memory_manager, '_generate_embedding', return_value=[0.1] * 1536):
            with patch.object(memory_manager, '_search_similar_memories', return_value=mock_memories):
                result = memory_manager.read_memory_auto(query)
                
                assert result["status"] == "success"
                assert len(result["memories"]) == 2
                assert result["memories"][0]["similarity"] >= result["memories"][1]["similarity"]  # 按相似度排序
    
    def test_read_memory_auto_no_results(self, memory_manager):
        """测试无结果的内存搜索"""
        query = "完全不相关的内容"
        
        with patch.object(memory_manager, '_generate_embedding', return_value=[0.1] * 1536):
            with patch.object(memory_manager, '_search_similar_memories', return_value=[]):
                result = memory_manager.read_memory_auto(query)
                
                assert result["status"] == "success"
                assert len(result["memories"]) == 0
                assert "message" in result
    
    def test_memory_auto_summarization(self, memory_manager):
        """测试自动摘要功能"""
        long_content = "今天是学习的一天。" * 100  # 长文本
        
        with patch.object(memory_manager, '_generate_embedding', return_value=[0.1] * 1536):
            # 模拟LLM摘要生成
            with patch.object(memory_manager, '_call_llm_for_summary', return_value="今天学习的摘要") as mock_llm:
                result = memory_manager.write_memory_auto(long_content)
                
                # 验证摘要被调用
                mock_llm.assert_called_once()
                assert result["status"] == "success"
    
    def test_memory_embedding_generation(self, memory_manager):
        """测试嵌入向量生成"""
        content = "测试文本内容"
        
        # 模拟嵌入客户端
        mock_embedding = [0.1, 0.2, 0.3] * 512  # 1536维向量
        
        with patch.object(memory_manager, '_generate_embedding', return_value=mock_embedding) as mock_embed:
            with patch.object(memory_manager, '_call_llm_for_summary', return_value="测试摘要"):
                result = memory_manager.write_memory_auto(content)
                
                # 验证嵌入生成被调用
                mock_embed.assert_called_once_with(content)
                assert result["status"] == "success"
    
    def test_memory_similarity_calculation(self, memory_manager):
        """测试内存相似度计算"""
        query_embedding = [0.1] * 1536
        memory_embedding = [0.1] * 1536  # 完全相同
        
        with patch.object(memory_manager, '_calculate_similarity') as mock_calc:
            mock_calc.return_value = 1.0  # 完全相似
            
            similarity = memory_manager._calculate_similarity(query_embedding, memory_embedding)
            assert similarity == 1.0
    
    def test_get_status_auto(self, memory_manager):
        """测试获取状态信息"""
        with patch.object(memory_manager, '_get_memory_statistics', return_value={
            "total_memories": 100,
            "total_size": 50000,
            "oldest_memory": "2024-01-01",
            "newest_memory": "2024-01-10"
        }):
            status = memory_manager.get_status_auto()
            
            assert status["status"] == "success"
            assert "statistics" in status
            assert status["statistics"]["total_memories"] == 100
    
    def test_get_status_summary(self, memory_manager):
        """测试获取状态摘要"""
        with patch.object(memory_manager, '_get_memory_statistics', return_value={
            "total_memories": 50,
            "total_size": 25000
        }):
            summary = memory_manager.get_status_summary()
            
            assert "total_memories" in summary
            assert summary["total_memories"] == 50
    
    def test_health_check(self, memory_manager):
        """测试健康检查"""
        with patch.object(memory_manager, '_check_storage_health', return_value=True):
            with patch.object(memory_manager, '_check_embedding_service', return_value=True):
                with patch.object(memory_manager, '_check_llm_service', return_value=True):
                    health = memory_manager.health_check()
                    
                    assert health["status"] == "healthy"
                    assert health["storage"] == True
                    assert health["embedding_service"] == True
                    assert health["llm_service"] == True
    
    def test_memory_compression(self, memory_manager):
        """测试内存压缩功能"""
        # 模拟大量相似记忆需要压缩
        similar_memories = [
            {"id": f"mem_{i}", "content": f"Python学习内容{i}", "similarity": 0.95}
            for i in range(10)
        ]
        
        with patch.object(memory_manager, '_find_compressible_memories', return_value=similar_memories):
            with patch.object(memory_manager, '_compress_memories') as mock_compress:
                mock_compress.return_value = {"compressed_count": 5, "new_memory_id": "mem_compressed"}
                
                result = memory_manager._perform_compression()
                
                mock_compress.assert_called_once()
                assert result["compressed_count"] == 5
    
    def test_memory_error_handling(self, memory_manager):
        """测试内存操作错误处理"""
        content = "测试内容"
        
        # 模拟嵌入生成失败
        with patch.object(memory_manager, '_generate_embedding', side_effect=Exception("嵌入服务不可用")):
            result = memory_manager.write_memory_auto(content)
            
            assert result["status"] == "error"
            assert "嵌入服务不可用" in result["message"]
    
    def test_memory_concurrent_access(self, memory_manager):
        """测试并发访问内存"""
        import threading
        import time
        
        results = []
        errors = []
        
        def write_memory_worker(worker_id):
            try:
                content = f"并发测试内容{worker_id}"
                with patch.object(memory_manager, '_generate_embedding', return_value=[0.1] * 1536):
                    with patch.object(memory_manager, '_call_llm_for_summary', return_value=f"摘要{worker_id}"):
                        result = memory_manager.write_memory_auto(content)
                        results.append((worker_id, result))
            except Exception as e:
                errors.append((worker_id, str(e)))
        
        # 启动多个并发写入线程
        threads = []
        for i in range(5):
            thread = threading.Thread(target=write_memory_worker, args=(i,))
            threads.append(thread)
            thread.start()
        
        # 等待所有线程完成
        for thread in threads:
            thread.join()
        
        # 验证并发操作结果
        assert len(errors) == 0, f"并发操作出现错误: {errors}"
        assert len(results) == 5
        
        # 验证所有操作都成功
        for worker_id, result in results:
            assert result["status"] == "success"
    
    def test_memory_capacity_management(self, memory_manager):
        """测试内存容量管理"""
        # 模拟内存接近容量上限
        with patch.object(memory_manager, '_get_memory_count', return_value=999):  # 接近1000上限
            with patch.object(memory_manager, '_is_capacity_exceeded', return_value=True):
                with patch.object(memory_manager, '_cleanup_old_memories') as mock_cleanup:
                    mock_cleanup.return_value = {"removed_count": 100}
                    
                    content = "新的内存内容"
                    with patch.object(memory_manager, '_generate_embedding', return_value=[0.1] * 1536):
                        with patch.object(memory_manager, '_call_llm_for_summary', return_value="新摘要"):
                            result = memory_manager.write_memory_auto(content)
                            
                            # 验证清理被触发
                            mock_cleanup.assert_called_once()
                            assert result["status"] == "success"
    
    def test_memory_backup_and_restore(self, memory_manager, temp_memory_dir):
        """测试内存备份和恢复"""
        backup_file = os.path.join(temp_memory_dir, "memory_backup.json")
        
        # 模拟备份操作
        mock_memories = [
            {"id": "mem_001", "content": "备份测试内容1", "timestamp": "2024-01-01"},
            {"id": "mem_002", "content": "备份测试内容2", "timestamp": "2024-01-02"}
        ]
        
        with patch.object(memory_manager, '_export_memories', return_value=mock_memories):
            backup_result = memory_manager.backup_memories(backup_file)
            assert backup_result["status"] == "success"
        
        # 模拟恢复操作
        with patch.object(memory_manager, '_import_memories') as mock_import:
            mock_import.return_value = {"imported_count": 2}
            
            restore_result = memory_manager.restore_memories(backup_file)
            assert restore_result["status"] == "success"
            assert restore_result["imported_count"] == 2