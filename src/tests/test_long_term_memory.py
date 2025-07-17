#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
长期记忆系统测试
"""

import os
import sys
import unittest
import tempfile
import shutil
from unittest.mock import Mock, patch

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

try:
    from tools.long_term_memory import LongTermMemoryManager, LongTermMemoryTools
except ImportError as e:
    print(f"Warning: Could not import long-term memory modules: {e}")
    LongTermMemoryManager = None
    LongTermMemoryTools = None


class TestLongTermMemory(unittest.TestCase):
    """长期记忆系统测试类"""
    
    def setUp(self):
        """测试前设置"""
        if LongTermMemoryManager is None:
            self.skipTest("Long-term memory modules not available")
        
        # 创建临时目录
        self.temp_dir = tempfile.mkdtemp()
        self.workspace_dir = os.path.join(self.temp_dir, "workspace")
        os.makedirs(self.workspace_dir, exist_ok=True)
        
        # 创建测试配置文件
        self.config_file = os.path.join(self.temp_dir, "test_config.txt")
        with open(self.config_file, 'w', encoding='utf-8') as f:
            f.write("""
# Test configuration
api_key=test_key
model=test_model
api_base=http://test.com
enable_long_term_memory=True
memory_similarity_threshold=0.85
""")
    
    def tearDown(self):
        """测试后清理"""
        if hasattr(self, 'temp_dir') and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_memory_manager_initialization(self):
        """测试记忆管理器初始化"""
        memory_manager = LongTermMemoryManager(
            workspace_root=self.workspace_dir,
            memory_config_file=self.config_file
        )
        
        # 检查基本属性
        self.assertEqual(memory_manager.workspace_root, self.workspace_dir)
        self.assertTrue(os.path.exists(memory_manager.memory_dir))
        
        # 清理
        memory_manager.cleanup()
    
    def test_memory_tools_initialization(self):
        """测试记忆工具初始化"""
        memory_tools = LongTermMemoryTools(
            workspace_root=self.workspace_dir,
            memory_config_file=self.config_file
        )
        
        # 检查工具方法存在
        self.assertTrue(hasattr(memory_tools, 'recall_memories'))
        self.assertTrue(hasattr(memory_tools, 'recall_memories_by_time'))
        self.assertTrue(hasattr(memory_tools, 'get_memory_summary'))
        
        # 清理
        memory_tools.cleanup()
    
    @patch('tools.long_term_memory.MemManagerAgent')
    def test_store_task_memory(self, mock_mem_agent):
        """测试任务记忆存储"""
        # 模拟记忆管理器
        mock_manager = Mock()
        mock_manager.write_memory_auto.return_value = {
            "success": True,
            "action": "added",
            "mem_id": "test_mem_001"
        }
        mock_mem_agent.return_value = mock_manager
        
        memory_manager = LongTermMemoryManager(
            workspace_root=self.workspace_dir,
            memory_config_file=self.config_file
        )
        memory_manager.memory_manager = mock_manager
        memory_manager.initialized = True
        
        # 测试存储任务记忆
        result = memory_manager.store_task_memory(
            task_prompt="测试任务",
            task_result="测试结果",
            execution_metadata={"test": True}
        )
        
        # 验证结果
        self.assertTrue(result["success"])
        self.assertEqual(result["action"], "added")
        self.assertEqual(result["memory_id"], "test_mem_001")
        
        # 验证调用
        mock_manager.write_memory_auto.assert_called_once()
    
    @patch('tools.long_term_memory.MemManagerAgent')
    def test_recall_memories(self, mock_mem_agent):
        """测试记忆召回"""
        # 模拟记忆管理器
        mock_manager = Mock()
        mock_manager.read_memory_auto.return_value = {
            "success": True,
            "memories": [
                {
                    "mem_cell": {
                        "summary": "测试记忆1",
                        "create_time": 1640995200.0,  # 2022-01-01
                        "mem_id": "mem_001"
                    },
                    "similarity_score": 0.95
                }
            ],
            "search_method": "auto"
        }
        mock_mem_agent.return_value = mock_manager
        
        memory_tools = LongTermMemoryTools(
            workspace_root=self.workspace_dir,
            memory_config_file=self.config_file
        )
        memory_tools.memory_manager.memory_manager = mock_manager
        memory_tools.memory_manager.initialized = True
        
        # 测试召回记忆
        result = memory_tools.recall_memories(
            query="测试查询",
            top_k=5
        )
        
        # 验证结果
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["memories_count"], 1)
        self.assertIn("测试记忆1", result["message"])
        
        # 验证调用
        mock_manager.read_memory_auto.assert_called_once_with("测试查询", top_k=5)
    
    def test_memory_unavailable_handling(self):
        """测试记忆功能不可用时的处理"""
        memory_manager = LongTermMemoryManager(
            workspace_root=self.workspace_dir,
            memory_config_file=self.config_file
        )
        memory_manager.initialized = False
        memory_manager.memory_manager = None
        
        # 测试存储记忆
        result = memory_manager.store_task_memory("test", "test")
        self.assertFalse(result["success"])
        self.assertIn("不可用", result["error"])
        
        # 测试召回记忆
        result = memory_manager.recall_relevant_memories("test")
        self.assertFalse(result["success"])
        self.assertIn("不可用", result["error"])
    
    def test_memory_text_building(self):
        """测试记忆文本构建"""
        memory_manager = LongTermMemoryManager(
            workspace_root=self.workspace_dir,
            memory_config_file=self.config_file
        )
        
        # 测试记忆文本构建
        memory_text = memory_manager._build_memory_text(
            task_prompt="测试任务提示",
            task_result="测试任务结果",
            metadata={
                "execution_time": 1.5,
                "tool_calls_count": 2,
                "model_used": "test_model"
            }
        )
        
        # 验证文本内容
        self.assertIn("测试任务提示", memory_text)
        self.assertIn("测试任务结果", memory_text)
        self.assertIn("execution_time: 1.5", memory_text)
        self.assertIn("tool_calls_count: 2", memory_text)
        self.assertIn("model_used: test_model", memory_text)
    
    def test_memory_format_for_context(self):
        """测试记忆格式化"""
        memory_manager = LongTermMemoryManager(
            workspace_root=self.workspace_dir,
            memory_config_file=self.config_file
        )
        
        # 测试记忆格式化
        memory_data = {
            "mem_cell": {
                "summary": "测试记忆摘要",
                "create_time": 1640995200.0,  # 2022-01-01
                "mem_id": "test_mem_001"
            },
            "similarity_score": 0.95
        }
        
        formatted_memory = memory_manager._format_memory_for_context(memory_data)
        
        # 验证格式化结果
        self.assertIsNotNone(formatted_memory)
        self.assertEqual(formatted_memory["memory_id"], "test_mem_001")
        self.assertEqual(formatted_memory["summary"], "测试记忆摘要")
        self.assertEqual(formatted_memory["similarity_score"], 0.95)
        self.assertIn("2022-01-01", formatted_memory["create_time"])


class TestMemoryIntegration(unittest.TestCase):
    """记忆系统集成测试"""
    
    def setUp(self):
        """测试前设置"""
        self.temp_dir = tempfile.mkdtemp()
        self.workspace_dir = os.path.join(self.temp_dir, "workspace")
        os.makedirs(self.workspace_dir, exist_ok=True)
    
    def tearDown(self):
        """测试后清理"""
        if hasattr(self, 'temp_dir') and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_memory_tools_error_handling(self):
        """测试记忆工具错误处理"""
        if LongTermMemoryTools is None:
            self.skipTest("Long-term memory modules not available")
        
        # 创建无效配置的记忆工具
        memory_tools = LongTermMemoryTools(
            workspace_root=self.workspace_dir,
            memory_config_file="/nonexistent/config/config.txt"
        )
        
        # 测试召回记忆的错误处理
        result = memory_tools.recall_memories("test query")
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["memories_count"], 0)
        
        # 测试时间召回的错误处理
        result = memory_tools.recall_memories_by_time("yesterday")
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["memories_count"], 0)
        
        # 测试摘要获取的错误处理
        result = memory_tools.get_memory_summary()
        self.assertEqual(result["status"], "error")


if __name__ == '__main__':
    # 运行测试
    unittest.main(verbosity=2) 