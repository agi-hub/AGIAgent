#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
评测程序和打分函数
实现任务执行的自动评测和打分功能
"""

import os
import re
import time
import json
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path

import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from tests.skill_evaluation.test_dataset import TestCase
from src.main import AGIAgentMain
from src.config_loader import get_api_key, get_api_base, get_model


class TaskEvaluator:
    """任务评测器"""
    
    def __init__(self, 
                 root_dir: str,
                 config_file: str = "config/config.txt",
                 user_id: Optional[str] = None,
                 enable_long_term_memory: bool = False):
        """
        初始化任务评测器
        
        Args:
            root_dir: 根目录（用于存储任务输出）
            config_file: 配置文件路径
            user_id: 用户ID
            enable_long_term_memory: 是否启用长期记忆（skill功能）
        """
        self.root_dir = root_dir
        self.config_file = config_file
        self.user_id = user_id
        self.enable_long_term_memory = enable_long_term_memory
        
        # 加载配置
        self.api_key = get_api_key(config_file)
        self.api_base = get_api_base(config_file)
        self.model = get_model(config_file)
        
        if not self.api_key or not self.api_base or not self.model:
            raise ValueError("API配置不完整，请检查config/config.txt")
    
    def execute_task(self, test_case: TestCase, output_subdir: str) -> Dict[str, Any]:
        """
        执行单个任务
        
        Args:
            test_case: 测试用例
            output_subdir: 输出子目录名
            
        Returns:
            执行结果字典，包含：
            - success: 是否成功
            - workspace_dir: 工作空间目录
            - execution_time: 执行时间
            - rounds: 执行轮数
            - tool_calls: 工具调用次数
            - skill_used: 是否使用了skill
            - error: 错误信息（如果有）
        """
        # 创建输出目录
        timestamp = int(time.time())
        output_dir = os.path.join(self.root_dir, output_subdir, f"output_{test_case.task_id}_{timestamp}")
        os.makedirs(output_dir, exist_ok=True)
        
        workspace_dir = os.path.join(output_dir, "workspace")
        os.makedirs(workspace_dir, exist_ok=True)
        
        start_time = time.time()
        
        try:
            # 创建AGIAgent实例
            agia = AGIAgentMain(
                out_dir=output_dir,
                api_key=self.api_key,
                model=self.model,
                api_base=self.api_base,
                debug_mode=False,
                detailed_summary=True,
                single_task_mode=True,
                interactive_mode=False,
                continue_mode=False,
                streaming=False,
                user_id=self.user_id,
                enable_thinking=None
            )
            
            # 如果启用长期记忆，需要在AGIAgent中设置
            # 注意：这里需要通过config.txt或环境变量来设置enable_long_term_memory
            
            # 执行任务
            success = agia.execute_single_task(test_case.task_description, loops=100)
            
            execution_time = time.time() - start_time
            
            # 统计执行信息（从日志中读取）
            rounds, tool_calls = self._extract_execution_stats(output_dir)
            skill_used = self._check_skill_usage(output_dir)
            
            return {
                "success": success,
                "workspace_dir": workspace_dir,
                "output_dir": output_dir,
                "execution_time": execution_time,
                "rounds": rounds,
                "tool_calls": tool_calls,
                "skill_used": skill_used,
                "error": None
            }
            
        except Exception as e:
            execution_time = time.time() - start_time
            return {
                "success": False,
                "workspace_dir": workspace_dir,
                "output_dir": output_dir,
                "execution_time": execution_time,
                "rounds": 0,
                "tool_calls": 0,
                "skill_used": False,
                "error": str(e)
            }
    
    def _extract_execution_stats(self, output_dir: str) -> Tuple[int, int]:
        """
        从日志中提取执行统计信息
        
        Args:
            output_dir: 输出目录
            
        Returns:
            (轮数, 工具调用次数)
        """
        logs_dir = os.path.join(output_dir, "logs")
        manager_log = os.path.join(logs_dir, "manager.out")
        
        rounds = 0
        tool_calls = 0
        
        if os.path.exists(manager_log):
            try:
                with open(manager_log, 'r', encoding='utf-8') as f:
                    content = f.read()
                    # 统计轮数（通过查找"Round"或"轮次"关键词）
                    rounds = len(re.findall(r'(?i)(round|轮次|执行轮)', content))
                    # 统计工具调用（通过查找<invoke>标签）
                    tool_calls = len(re.findall(r'<invoke>', content))
            except Exception:
                pass
        
        return rounds, tool_calls
    
    def _check_skill_usage(self, output_dir: str) -> bool:
        """
        检查是否使用了skill
        
        Args:
            output_dir: 输出目录
            
        Returns:
            是否使用了skill
        """
        logs_dir = os.path.join(output_dir, "logs")
        manager_log = os.path.join(logs_dir, "manager.out")
        
        if os.path.exists(manager_log):
            try:
                with open(manager_log, 'r', encoding='utf-8') as f:
                    content = f.read()
                    # 查找skill相关关键词
                    skill_keywords = ['query_skill', 'fetch_skill', 'skill', '经验']
                    return any(keyword.lower() in content.lower() for keyword in skill_keywords)
            except Exception:
                pass
        
        return False
    
    def evaluate_output(self, test_case: TestCase, execution_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        评估任务输出
        
        Args:
            test_case: 测试用例
            execution_result: 执行结果
            
        Returns:
            评估结果字典
        """
        workspace_dir = execution_result.get("workspace_dir", "")
        output_dir = execution_result.get("output_dir", "")
        
        evaluation = {
            "completion_score": 0.0,
            "quality_score": 0.0,
            "efficiency_score": 0.0,
            "innovation_score": 0.0,
            "total_score": 0.0,
            "details": {}
        }
        
        # 1. 完成度得分 (0.4)
        completion_score, completion_details = self._evaluate_completion(
            test_case, workspace_dir, output_dir
        )
        evaluation["completion_score"] = completion_score
        evaluation["details"]["completion"] = completion_details
        
        # 2. 质量得分 (0.3)
        quality_score, quality_details = self._evaluate_quality(
            test_case, workspace_dir
        )
        evaluation["quality_score"] = quality_score
        evaluation["details"]["quality"] = quality_details
        
        # 3. 效率得分 (0.2)
        efficiency_score, efficiency_details = self._evaluate_efficiency(
            test_case, execution_result
        )
        evaluation["efficiency_score"] = efficiency_score
        evaluation["details"]["efficiency"] = efficiency_details
        
        # 4. 创新度得分 (0.1)
        innovation_score, innovation_details = self._evaluate_innovation(
            execution_result
        )
        evaluation["innovation_score"] = innovation_score
        evaluation["details"]["innovation"] = innovation_details
        
        # 计算总分
        evaluation["total_score"] = (
            completion_score * 0.4 +
            quality_score * 0.3 +
            efficiency_score * 0.2 +
            innovation_score * 0.1
        )
        
        return evaluation
    
    def _evaluate_completion(self, test_case: TestCase, workspace_dir: str, output_dir: str) -> Tuple[float, Dict[str, Any]]:
        """
        评估完成度
        
        Returns:
            (得分, 详细信息)
        """
        score = 0.0
        details = {}
        
        requirements = test_case.output_requirements
        criteria = test_case.evaluation_criteria.get("completion", {})
        
        # 检查必需文件是否存在
        required_files = requirements.get("files", [])
        files_found = []
        files_missing = []
        
        for file_name in required_files:
            file_path = os.path.join(workspace_dir, file_name)
            if os.path.exists(file_path):
                files_found.append(file_name)
            else:
                files_missing.append(file_name)
        
        details["files_found"] = files_found
        details["files_missing"] = files_missing
        
        if criteria.get("file_exists", False):
            if len(files_found) == len(required_files):
                score += 0.2
            elif len(files_found) > 0:
                score += 0.1 * (len(files_found) / len(required_files))
        
        # 检查内容有效性
        content_check = requirements.get("content_check", {})
        content_valid_count = 0
        content_total = len(content_check)
        
        for file_name, check_rules in content_check.items():
            file_path = os.path.join(workspace_dir, file_name)
            if os.path.exists(file_path):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # 检查必须包含的关键词
                    must_contain = check_rules.get("must_contain", [])
                    if must_contain:
                        found_keywords = [kw for kw in must_contain if kw in content]
                        if len(found_keywords) == len(must_contain):
                            content_valid_count += 1
                        elif len(found_keywords) > 0:
                            content_valid_count += 0.5
                    
                    # 检查格式
                    if "format" in check_rules:
                        # 简单格式检查
                        if check_rules["format"] == "Markdown文件" and file_name.endswith(".md"):
                            content_valid_count += 0.5
                        elif check_rules["format"] == "文本文件" and file_name.endswith(".txt"):
                            content_valid_count += 0.5
                        elif check_rules["format"] == "Python代码文件" and file_name.endswith(".py"):
                            content_valid_count += 0.5
                
                except Exception as e:
                    details[f"{file_name}_error"] = str(e)
        
        if content_total > 0:
            score += 0.2 * (content_valid_count / content_total)
        
        details["content_valid_count"] = content_valid_count
        details["content_total"] = content_total
        
        return min(score, 1.0), details
    
    def _evaluate_quality(self, test_case: TestCase, workspace_dir: str) -> Tuple[float, Dict[str, Any]]:
        """
        评估代码质量
        
        Returns:
            (得分, 详细信息)
        """
        score = 0.0
        details = {}
        
        criteria = test_case.evaluation_criteria.get("quality", {})
        
        # 查找代码文件
        code_files = []
        for root, dirs, files in os.walk(workspace_dir):
            for file in files:
                if file.endswith(".py"):
                    code_files.append(os.path.join(root, file))
        
        details["code_files_found"] = len(code_files)
        
        if len(code_files) == 0:
            return 0.0, details
        
        # 检查代码风格和错误处理
        has_error_handling = False
        has_docstrings = False
        has_good_structure = False
        
        for code_file in code_files:
            try:
                with open(code_file, 'r', encoding='utf-8') as f:
                    code_content = f.read()
                
                # 检查错误处理
                if any(keyword in code_content for keyword in ['try:', 'except', 'if', 'assert']):
                    has_error_handling = True
                
                # 检查文档字符串
                if '"""' in code_content or "'''" in code_content:
                    has_docstrings = True
                
                # 检查代码结构（类、函数定义）
                if 'def ' in code_content or 'class ' in code_content:
                    has_good_structure = True
            
            except Exception:
                pass
        
        if criteria.get("error_handling") == "有异常处理" and has_error_handling:
            score += 0.1
        if criteria.get("docstrings") == "有文档字符串" and has_docstrings:
            score += 0.1
        if criteria.get("code_style") == "良好" and has_good_structure:
            score += 0.1
        
        details["has_error_handling"] = has_error_handling
        details["has_docstrings"] = has_docstrings
        details["has_good_structure"] = has_good_structure
        
        return min(score, 1.0), details
    
    def _evaluate_efficiency(self, test_case: TestCase, execution_result: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        """
        评估执行效率
        
        Returns:
            (得分, 详细信息)
        """
        score = 1.0
        details = {}
        
        criteria = test_case.evaluation_criteria.get("efficiency", {})
        max_rounds = criteria.get("max_rounds", 10)
        max_tool_calls = criteria.get("max_tool_calls", 20)
        
        rounds = execution_result.get("rounds", 0)
        tool_calls = execution_result.get("tool_calls", 0)
        
        details["rounds"] = rounds
        details["tool_calls"] = tool_calls
        details["max_rounds"] = max_rounds
        details["max_tool_calls"] = max_tool_calls
        
        # 如果超过最大轮数或工具调用次数，扣分
        if rounds > max_rounds:
            score -= 0.5 * ((rounds - max_rounds) / max_rounds)
        if tool_calls > max_tool_calls:
            score -= 0.5 * ((tool_calls - max_tool_calls) / max_tool_calls)
        
        # 如果执行时间过长，也扣分
        execution_time = execution_result.get("execution_time", 0)
        if execution_time > 300:  # 5分钟
            score -= 0.2
        
        details["execution_time"] = execution_time
        
        return max(score, 0.0), details
    
    def _evaluate_innovation(self, execution_result: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        """
        评估创新度（是否使用了skill等）
        
        Returns:
            (得分, 详细信息)
        """
        score = 0.0
        details = {}
        
        skill_used = execution_result.get("skill_used", False)
        
        if skill_used:
            score = 1.0
            details["skill_used"] = True
        else:
            details["skill_used"] = False
        
        return score, details
    
    def calculate_score(self, test_case: TestCase, execution_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        计算任务得分（综合评估）
        
        Args:
            test_case: 测试用例
            execution_result: 执行结果
            
        Returns:
            完整的评估结果
        """
        evaluation = self.evaluate_output(test_case, execution_result)
        
        return {
            "task_id": test_case.task_id,
            "success": execution_result.get("success", False),
            "total_score": evaluation["total_score"],
            "completion_score": evaluation["completion_score"],
            "quality_score": evaluation["quality_score"],
            "efficiency_score": evaluation["efficiency_score"],
            "innovation_score": evaluation["innovation_score"],
            "execution_time": execution_result.get("execution_time", 0),
            "rounds": execution_result.get("rounds", 0),
            "tool_calls": execution_result.get("tool_calls", 0),
            "skill_used": execution_result.get("skill_used", False),
            "details": evaluation["details"],
            "output_dir": execution_result.get("output_dir", "")
        }

