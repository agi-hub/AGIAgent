#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
评测壳程序
实现完整的评测流程：基线测试 -> 任务整理 -> Skill整理 -> Skill测试 -> 对比分析
"""

import os
import sys
import json
import argparse
import time
from datetime import datetime
from typing import Dict, Any, List, Optional
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from tests.skill_evaluation.test_dataset import TestDataset, TestCase
from tests.skill_evaluation.evaluator import TaskEvaluator
from src.skill_evolve.task_reflection import TaskReflection
from src.skill_evolve.skill_manager import SkillManager
from src.config_loader import get_gui_default_data_directory


class BenchmarkRunner:
    """评测运行器"""
    
    def __init__(self, 
                 root_dir: Optional[str] = None,
                 config_file: str = "config/config.txt",
                 user_id: Optional[str] = None):
        """
        初始化评测运行器
        
        Args:
            root_dir: 根目录（如果为None，则从config读取）
            config_file: 配置文件路径
            user_id: 用户ID
        """
        self.config_file = config_file
        
        # 确定根目录
        if root_dir:
            self.root_dir = os.path.abspath(root_dir)
        else:
            data_dir = get_gui_default_data_directory(config_file)
            if data_dir:
                self.root_dir = data_dir
            else:
                project_root = self._find_project_root()
                self.root_dir = os.path.join(project_root, "data") if project_root else "data"
        
        self.user_id = user_id
        self.test_dataset = TestDataset()
        self.test_dataset.load_test_cases()
        
        # 创建评测结果目录
        self.results_dir = os.path.join(self.root_dir, "benchmark_results")
        os.makedirs(self.results_dir, exist_ok=True)
        
        print(f"📁 根目录: {self.root_dir}")
        print(f"📁 结果目录: {self.results_dir}")
        print(f"📋 测试用例数量: {len(self.test_dataset.test_cases)}")
    
    def _find_project_root(self) -> Optional[str]:
        """查找项目根目录"""
        current = Path(__file__).parent.resolve()
        for _ in range(10):
            config_dir = current / "config"
            if config_dir.exists() and config_dir.is_dir():
                return str(current)
            if current == current.parent:
                break
            current = current.parent
        return None
    
    def run_baseline(self) -> Dict[str, Any]:
        """
        运行基线测试（无skill）
        
        Returns:
            基线测试结果
        """
        print("\n" + "="*60)
        print("第一阶段：基线测试（无skill）")
        print("="*60)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        baseline_dir = os.path.join(self.results_dir, f"baseline_{timestamp}")
        os.makedirs(baseline_dir, exist_ok=True)
        
        results = {
            "timestamp": timestamp,
            "test_type": "baseline",
            "test_cases": [],
            "summary": {}
        }
        
        evaluator = TaskEvaluator(
            root_dir=baseline_dir,
            config_file=self.config_file,
            user_id=self.user_id,
            enable_long_term_memory=False
        )
        
        total_score = 0.0
        success_count = 0
        
        for i, test_case in enumerate(self.test_dataset.test_cases, 1):
            print(f"\n[{i}/{len(self.test_dataset.test_cases)}] 执行测试用例: {test_case.task_id}")
            print(f"任务描述: {test_case.task_description[:100]}...")
            
            # 执行任务
            execution_result = evaluator.execute_task(test_case, "baseline_outputs")
            
            # 评估结果
            evaluation = evaluator.calculate_score(test_case, execution_result)
            
            results["test_cases"].append(evaluation)
            
            total_score += evaluation["total_score"]
            if evaluation["success"]:
                success_count += 1
            
            print(f"  得分: {evaluation['total_score']:.2f} | "
                  f"完成度: {evaluation['completion_score']:.2f} | "
                  f"质量: {evaluation['quality_score']:.2f} | "
                  f"效率: {evaluation['efficiency_score']:.2f} | "
                  f"创新: {evaluation['innovation_score']:.2f}")
            print(f"  成功: {'是' if evaluation['success'] else '否'} | "
                  f"轮数: {evaluation['rounds']} | "
                  f"工具调用: {evaluation['tool_calls']} | "
                  f"使用skill: {'是' if evaluation['skill_used'] else '否'}")
        
        # 计算汇总统计
        avg_score = total_score / len(self.test_dataset.test_cases) if self.test_dataset.test_cases else 0.0
        success_rate = success_count / len(self.test_dataset.test_cases) if self.test_dataset.test_cases else 0.0
        
        results["summary"] = {
            "total_cases": len(self.test_dataset.test_cases),
            "success_count": success_count,
            "success_rate": success_rate,
            "average_score": avg_score,
            "total_score": total_score
        }
        
        # 保存结果
        results_file = os.path.join(baseline_dir, "results.json")
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"\n基线测试完成！")
        print(f"平均得分: {avg_score:.2f}")
        print(f"成功率: {success_rate:.2%}")
        print(f"结果已保存到: {results_file}")
        
        return results
    
    def run_task_reflection(self) -> bool:
        """
        运行任务整理，生成skill
        
        Returns:
            是否成功
        """
        print("\n" + "="*60)
        print("第二阶段：任务整理（生成skill）")
        print("="*60)
        
        try:
            task_reflection = TaskReflection(
                root_dir=self.root_dir,
                config_file=self.config_file
            )
            
            print("开始处理任务日志，生成skill...")
            task_reflection.run()
            
            print("任务整理完成！")
            return True
            
        except Exception as e:
            print(f"❌ 任务整理失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def run_skill_manager(self) -> bool:
        """
        运行skill整理，整合skill
        
        Returns:
            是否成功
        """
        print("\n" + "="*60)
        print("第三阶段：Skill整理（整合skill）")
        print("="*60)
        
        try:
            skill_manager = SkillManager(
                root_dir=self.root_dir,
                config_file=self.config_file
            )
            
            print("开始整理skill，进行合并和整合...")
            skill_manager.run()
            
            print("Skill整理完成！")
            return True
            
        except Exception as e:
            print(f"❌ Skill整理失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def run_with_skills(self) -> Dict[str, Any]:
        """
        运行skill测试（有skill）
        
        Returns:
            Skill测试结果
        """
        print("\n" + "="*60)
        print("第四阶段：Skill测试（有skill）")
        print("="*60)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        skill_dir = os.path.join(self.results_dir, f"skill_{timestamp}")
        os.makedirs(skill_dir, exist_ok=True)
        
        results = {
            "timestamp": timestamp,
            "test_type": "skill",
            "test_cases": [],
            "summary": {}
        }
        
        # 注意：skill工具会在long_term_memory启用时自动注册
        # 需要在config.txt中设置enable_long_term_memory=True
        # 或者确保环境变量AGIBOT_LONG_TERM_MEMORY不为'false'/'0'/'no'/'off'
        # 注意：TaskEvaluator中的enable_long_term_memory参数目前仅用于标记，实际启用需要通过config.txt
        evaluator = TaskEvaluator(
            root_dir=skill_dir,
            config_file=self.config_file,
            user_id=self.user_id,
            enable_long_term_memory=True
        )
        
        total_score = 0.0
        success_count = 0
        
        for i, test_case in enumerate(self.test_dataset.test_cases, 1):
            print(f"\n[{i}/{len(self.test_dataset.test_cases)}] 执行测试用例: {test_case.task_id}")
            print(f"任务描述: {test_case.task_description[:100]}...")
            
            # 执行任务
            execution_result = evaluator.execute_task(test_case, "skill_outputs")
            
            # 评估结果
            evaluation = evaluator.calculate_score(test_case, execution_result)
            
            results["test_cases"].append(evaluation)
            
            total_score += evaluation["total_score"]
            if evaluation["success"]:
                success_count += 1
            
            print(f"  得分: {evaluation['total_score']:.2f} | "
                  f"完成度: {evaluation['completion_score']:.2f} | "
                  f"质量: {evaluation['quality_score']:.2f} | "
                  f"效率: {evaluation['efficiency_score']:.2f} | "
                  f"创新: {evaluation['innovation_score']:.2f}")
            print(f"  成功: {'是' if evaluation['success'] else '否'} | "
                  f"轮数: {evaluation['rounds']} | "
                  f"工具调用: {evaluation['tool_calls']} | "
                  f"使用skill: {'是' if evaluation['skill_used'] else '否'}")
        
        # 计算汇总统计
        avg_score = total_score / len(self.test_dataset.test_cases) if self.test_dataset.test_cases else 0.0
        success_rate = success_count / len(self.test_dataset.test_cases) if self.test_dataset.test_cases else 0.0
        
        results["summary"] = {
            "total_cases": len(self.test_dataset.test_cases),
            "success_count": success_count,
            "success_rate": success_rate,
            "average_score": avg_score,
            "total_score": total_score
        }
        
        # 保存结果
        results_file = os.path.join(skill_dir, "results.json")
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"\nSkill测试完成！")
        print(f"平均得分: {avg_score:.2f}")
        print(f"成功率: {success_rate:.2%}")
        print(f"结果已保存到: {results_file}")
        
        return results
    
    def compare_results(self, baseline_results: Dict[str, Any], skill_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        对比两次测试结果
        
        Args:
            baseline_results: 基线测试结果
            skill_results: Skill测试结果
            
        Returns:
            对比结果
        """
        print("\n" + "="*60)
        print("第五阶段：结果对比分析")
        print("="*60)
        
        comparison = {
            "baseline": baseline_results["summary"],
            "skill": skill_results["summary"],
            "improvements": {},
            "detailed_comparison": []
        }
        
        # 计算改进指标
        baseline_avg = baseline_results["summary"]["average_score"]
        skill_avg = skill_results["summary"]["average_score"]
        score_improvement = skill_avg - baseline_avg
        score_improvement_pct = (score_improvement / baseline_avg * 100) if baseline_avg > 0 else 0
        
        baseline_success_rate = baseline_results["summary"]["success_rate"]
        skill_success_rate = skill_results["summary"]["success_rate"]
        success_rate_improvement = skill_success_rate - baseline_success_rate
        success_rate_improvement_pct = (success_rate_improvement / baseline_success_rate * 100) if baseline_success_rate > 0 else 0
        
        comparison["improvements"] = {
            "score_improvement": score_improvement,
            "score_improvement_pct": score_improvement_pct,
            "success_rate_improvement": success_rate_improvement,
            "success_rate_improvement_pct": success_rate_improvement_pct
        }
        
        # 详细对比每个测试用例
        for i, (baseline_case, skill_case) in enumerate(zip(
            baseline_results["test_cases"],
            skill_results["test_cases"]
        )):
            case_comparison = {
                "task_id": baseline_case["task_id"],
                "baseline_score": baseline_case["total_score"],
                "skill_score": skill_case["total_score"],
                "score_improvement": skill_case["total_score"] - baseline_case["total_score"],
                "baseline_success": baseline_case["success"],
                "skill_success": skill_case["success"],
                "baseline_rounds": baseline_case["rounds"],
                "skill_rounds": skill_case["rounds"],
                "rounds_improvement": baseline_case["rounds"] - skill_case["rounds"],
                "baseline_tool_calls": baseline_case["tool_calls"],
                "skill_tool_calls": skill_case["tool_calls"],
                "tool_calls_improvement": baseline_case["tool_calls"] - skill_case["tool_calls"],
                "skill_used": skill_case["skill_used"]
            }
            comparison["detailed_comparison"].append(case_comparison)
        
        # 打印对比结果
        print(f"\n📊 总体对比:")
        print(f"  平均得分: {baseline_avg:.2f} -> {skill_avg:.2f} "
              f"({'+' if score_improvement >= 0 else ''}{score_improvement:.2f}, "
              f"{'+' if score_improvement_pct >= 0 else ''}{score_improvement_pct:.1f}%)")
        print(f"  成功率: {baseline_success_rate:.2%} -> {skill_success_rate:.2%} "
              f"({'+' if success_rate_improvement >= 0 else ''}{success_rate_improvement:.2%}, "
              f"{'+' if success_rate_improvement_pct >= 0 else ''}{success_rate_improvement_pct:.1f}%)")
        
        print(f"\n📋 详细对比:")
        for case_comp in comparison["detailed_comparison"]:
            print(f"\n  测试用例: {case_comp['task_id']}")
            print(f"    得分: {case_comp['baseline_score']:.2f} -> {case_comp['skill_score']:.2f} "
                  f"({'+' if case_comp['score_improvement'] >= 0 else ''}{case_comp['score_improvement']:.2f})")
            print(f"    成功: {case_comp['baseline_success']} -> {case_comp['skill_success']}")
            print(f"    轮数: {case_comp['baseline_rounds']} -> {case_comp['skill_rounds']} "
                  f"({'+' if case_comp['rounds_improvement'] >= 0 else ''}{case_comp['rounds_improvement']})")
            print(f"    工具调用: {case_comp['baseline_tool_calls']} -> {case_comp['skill_tool_calls']} "
                  f"({'+' if case_comp['tool_calls_improvement'] >= 0 else ''}{case_comp['tool_calls_improvement']})")
            print(f"    使用skill: {case_comp['skill_used']}")
        
        return comparison
    
    def generate_report(self, comparison: Dict[str, Any], output_file: Optional[str] = None) -> str:
        """
        生成评测报告
        
        Args:
            comparison: 对比结果
            output_file: 输出文件路径（如果为None，则自动生成）
            
        Returns:
            报告文件路径
        """
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = os.path.join(self.results_dir, f"report_{timestamp}.json")
        
        # 保存对比结果
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(comparison, f, ensure_ascii=False, indent=2)
        
        print(f"\n📄 评测报告已保存到: {output_file}")
        
        return output_file
    
    def run_full_benchmark(self) -> Dict[str, Any]:
        """
        运行完整评测流程
        
        Returns:
            完整的评测结果
        """
        print("\n" + "="*60)
        print("开始完整评测流程")
        print("="*60)
        
        # 第一阶段：基线测试
        baseline_results = self.run_baseline()
        
        # 第二阶段：任务整理
        if not self.run_task_reflection():
            print("⚠️ 警告：任务整理失败，但继续执行后续步骤")
        
        # 第三阶段：Skill整理
        if not self.run_skill_manager():
            print("⚠️ 警告：Skill整理失败，但继续执行后续步骤")
        
        # 等待一下，确保skill已经保存
        time.sleep(2)
        
        # 第四阶段：Skill测试
        skill_results = self.run_with_skills()
        
        # 第五阶段：结果对比
        comparison = self.compare_results(baseline_results, skill_results)
        
        # 生成报告
        report_file = self.generate_report(comparison)
        
        print("\n" + "="*60)
        print("完整评测流程结束")
        print("="*60)
        
        return {
            "baseline": baseline_results,
            "skill": skill_results,
            "comparison": comparison,
            "report_file": report_file
        }


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="Skill系统评测程序")
    parser.add_argument("--root-dir", type=str, help="根目录路径")
    parser.add_argument("--config", type=str, default="config/config.txt", help="配置文件路径")
    parser.add_argument("--user-id", type=str, help="用户ID")
    parser.add_argument("--baseline-only", action="store_true", help="只运行基线测试")
    parser.add_argument("--skill-only", action="store_true", help="只运行skill测试")
    parser.add_argument("--reflection-only", action="store_true", help="只运行任务整理")
    parser.add_argument("--manager-only", action="store_true", help="只运行skill整理")
    
    args = parser.parse_args()
    
    runner = BenchmarkRunner(
        root_dir=args.root_dir,
        config_file=args.config,
        user_id=args.user_id
    )
    
    if args.baseline_only:
        runner.run_baseline()
    elif args.skill_only:
        runner.run_with_skills()
    elif args.reflection_only:
        runner.run_task_reflection()
    elif args.manager_only:
        runner.run_skill_manager()
    else:
        runner.run_full_benchmark()


if __name__ == "__main__":
    main()

