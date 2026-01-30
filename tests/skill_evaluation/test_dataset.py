#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试数据集定义
定义测试样本的数据结构和加载函数
"""

import json
import os
from typing import Dict, Any, List, Optional
from pathlib import Path


class TestCase:
    """测试用例数据结构"""
    
    def __init__(self, data: Dict[str, Any]):
        """
        初始化测试用例
        
        Args:
            data: 测试用例数据字典
        """
        self.task_id = data.get("task_id", "")
        self.task_description = data.get("task_description", "")
        self.output_requirements = data.get("output_requirements", {})
        self.evaluation_criteria = data.get("evaluation_criteria", {})
        self.expected_outputs = data.get("expected_outputs", {})
        self.category = data.get("category", "")
        self.difficulty = data.get("difficulty", "medium")
        self.raw_data = data
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return self.raw_data


class TestDataset:
    """测试数据集管理器"""
    
    def __init__(self, test_cases_dir: Optional[str] = None):
        """
        初始化测试数据集
        
        Args:
            test_cases_dir: 测试用例目录路径，如果为None则使用默认路径
        """
        if test_cases_dir is None:
            # 默认使用当前文件所在目录下的test_cases目录
            current_dir = Path(__file__).parent
            test_cases_dir = os.path.join(current_dir, "test_cases")
        
        self.test_cases_dir = test_cases_dir
        self.test_cases: List[TestCase] = []
    
    def load_test_cases(self) -> List[TestCase]:
        """
        加载所有测试用例
        
        Returns:
            测试用例列表
        """
        if not os.path.exists(self.test_cases_dir):
            raise FileNotFoundError(f"Test cases directory not found: {self.test_cases_dir}")
        
        test_cases = []
        for filename in sorted(os.listdir(self.test_cases_dir)):
            if filename.endswith(".json"):
                filepath = os.path.join(self.test_cases_dir, filename)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        test_case = TestCase(data)
                        test_cases.append(test_case)
                except Exception as e:
                    print(f"Warning: Failed to load test case {filename}: {e}")
        
        self.test_cases = test_cases
        return test_cases
    
    def get_test_case(self, task_id: str) -> Optional[TestCase]:
        """
        根据任务ID获取测试用例
        
        Args:
            task_id: 任务ID
            
        Returns:
            测试用例对象，如果不存在则返回None
        """
        for test_case in self.test_cases:
            if test_case.task_id == task_id:
                return test_case
        return None
    
    def get_test_cases_by_category(self, category: str) -> List[TestCase]:
        """
        根据类别获取测试用例
        
        Args:
            category: 任务类别
            
        Returns:
            测试用例列表
        """
        return [tc for tc in self.test_cases if tc.category == category]
    
    def get_all_categories(self) -> List[str]:
        """
        获取所有任务类别
        
        Returns:
            类别列表
        """
        categories = set()
        for test_case in self.test_cases:
            if test_case.category:
                categories.add(test_case.category)
        return sorted(list(categories))


