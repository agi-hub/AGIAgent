#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AGIBot测试运行器
提供灵活的测试执行选项和报告生成
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
import json

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

class TestRunner:
    """测试运行器"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent.parent
        self.test_dir = Path(__file__).parent
        self.reports_dir = self.test_dir / "test_reports"
        self.reports_dir.mkdir(exist_ok=True)
    
    def run_unit_tests(self, verbose=True):
        """运行单元测试"""
        print("🔧 运行单元测试...")
        cmd = [
            "python", "-m", "pytest", 
            str(self.test_dir / "unit"),
            "-m", "unit",
            "--tb=short"
        ]
        if verbose:
            cmd.append("-v")
        
        return self._execute_command(cmd)
    
    def run_integration_tests(self, verbose=True):
        """运行集成测试"""
        print("🔗 运行集成测试...")
        cmd = [
            "python", "-m", "pytest",
            str(self.test_dir / "integration"),
            "-m", "integration",
            "--tb=short"
        ]
        if verbose:
            cmd.append("-v")
        
        return self._execute_command(cmd)
    
    def run_e2e_tests(self, verbose=True):
        """运行端到端测试"""
        print("🚀 运行端到端测试...")
        cmd = [
            "python", "-m", "pytest",
            str(self.test_dir / "e2e"),
            "-m", "e2e",
            "--tb=short"
        ]
        if verbose:
            cmd.append("-v")
        
        return self._execute_command(cmd)
    
    def run_performance_tests(self, verbose=True):
        """运行性能测试"""
        print("⚡ 运行性能测试...")
        cmd = [
            "python", "-m", "pytest",
            str(self.test_dir / "performance"),
            "-m", "performance",
            "--tb=short"
        ]
        if verbose:
            cmd.append("-v")
        
        return self._execute_command(cmd)
    
    def run_security_tests(self, verbose=True):
        """运行安全测试"""
        print("🔒 运行安全测试...")
        cmd = [
            "python", "-m", "pytest",
            str(self.test_dir / "security"),
            "-m", "security",
            "--tb=short"
        ]
        if verbose:
            cmd.append("-v")
        
        return self._execute_command(cmd)
    
    def run_all_tests(self, skip_slow=False, verbose=True):
        """运行所有测试"""
        print("🎯 运行完整测试套件...")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        cmd = [
            "python", "-m", "pytest",
            str(self.test_dir),
            "--tb=short",
            f"--junit-xml={self.reports_dir}/junit_{timestamp}.xml",
            f"--html={self.reports_dir}/report_{timestamp}.html",
            "--self-contained-html",
            "--cov=src",
            f"--cov-report=html:{self.reports_dir}/coverage_{timestamp}",
            f"--cov-report=xml:{self.reports_dir}/coverage_{timestamp}.xml",
            "--cov-report=term-missing"
        ]
        
        if skip_slow:
            cmd.extend(["-m", "not slow"])
        
        if verbose:
            cmd.append("-v")
        
        return self._execute_command(cmd)
    
    def run_specific_test(self, test_path, verbose=True):
        """运行特定测试"""
        print(f"🎪 运行特定测试: {test_path}")
        cmd = [
            "python", "-m", "pytest",
            test_path,
            "--tb=short"
        ]
        if verbose:
            cmd.append("-v")
        
        return self._execute_command(cmd)
    
    def run_parallel_tests(self, num_workers="auto", verbose=True):
        """并行运行测试"""
        print(f"🚄 并行运行测试 (workers: {num_workers})...")
        cmd = [
            "python", "-m", "pytest",
            str(self.test_dir),
            "-n", str(num_workers),
            "--tb=short"
        ]
        if verbose:
            cmd.append("-v")
        
        return self._execute_command(cmd)
    
    def run_with_coverage(self, min_coverage=80, verbose=True):
        """运行测试并生成覆盖率报告"""
        print(f"📊 运行测试并生成覆盖率报告 (最低要求: {min_coverage}%)...")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        cmd = [
            "python", "-m", "pytest",
            str(self.test_dir),
            "--cov=src",
            f"--cov-fail-under={min_coverage}",
            f"--cov-report=html:{self.reports_dir}/coverage_{timestamp}",
            f"--cov-report=xml:{self.reports_dir}/coverage_{timestamp}.xml",
            "--cov-report=term-missing",
            "--tb=short"
        ]
        
        if verbose:
            cmd.append("-v")
        
        return self._execute_command(cmd)
    
    def run_smoke_tests(self, verbose=True):
        """运行冒烟测试（快速验证）"""
        print("💨 运行冒烟测试...")
        cmd = [
            "python", "-m", "pytest",
            str(self.test_dir),
            "-m", "not slow and not performance",
            "--maxfail=5",  # 失败5个测试后停止
            "--tb=short"
        ]
        if verbose:
            cmd.append("-v")
        
        return self._execute_command(cmd)
    
    def lint_and_format_check(self):
        """代码风格检查"""
        print("🎨 检查代码风格...")
        
        results = {}
        
        # 运行flake8
        try:
            result = subprocess.run(
                ["flake8", str(self.project_root / "src")],
                capture_output=True,
                text=True
            )
            results["flake8"] = {
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr
            }
        except FileNotFoundError:
            print("⚠️  flake8 未安装，跳过代码风格检查")
            results["flake8"] = {"returncode": -1, "message": "flake8 not found"}
        
        # 运行black检查
        try:
            result = subprocess.run(
                ["black", "--check", str(self.project_root / "src")],
                capture_output=True,
                text=True
            )
            results["black"] = {
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr
            }
        except FileNotFoundError:
            print("⚠️  black 未安装，跳过格式检查")
            results["black"] = {"returncode": -1, "message": "black not found"}
        
        return results
    
    def security_scan(self):
        """安全扫描"""
        print("🔍 运行安全扫描...")
        
        results = {}
        
        # 运行bandit
        try:
            result = subprocess.run(
                ["bandit", "-r", str(self.project_root / "src"), "-f", "json"],
                capture_output=True,
                text=True
            )
            results["bandit"] = {
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr
            }
        except FileNotFoundError:
            print("⚠️  bandit 未安装，跳过安全扫描")
            results["bandit"] = {"returncode": -1, "message": "bandit not found"}
        
        # 运行safety
        try:
            result = subprocess.run(
                ["safety", "check", "--json"],
                capture_output=True,
                text=True
            )
            results["safety"] = {
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr
            }
        except FileNotFoundError:
            print("⚠️  safety 未安装，跳过依赖安全检查")
            results["safety"] = {"returncode": -1, "message": "safety not found"}
        
        return results
    
    def _execute_command(self, cmd):
        """执行命令"""
        try:
            print(f"执行命令: {' '.join(cmd)}")
            result = subprocess.run(cmd, cwd=self.project_root, capture_output=True, text=True)
            
            if result.stdout:
                print(result.stdout)
            if result.stderr:
                print(result.stderr)
            
            return {
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "success": result.returncode == 0
            }
        except Exception as e:
            print(f"命令执行失败: {e}")
            return {
                "returncode": -1,
                "stdout": "",
                "stderr": str(e),
                "success": False
            }
    
    def generate_test_report(self):
        """生成测试执行报告"""
        print("📄 生成测试报告...")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = self.reports_dir / f"test_execution_report_{timestamp}.json"
        
        # 收集所有测试结果
        report_data = {
            "timestamp": timestamp,
            "execution_time": datetime.now().isoformat(),
            "project_root": str(self.project_root),
            "test_results": {
                "unit": self.run_unit_tests(verbose=False),
                "integration": self.run_integration_tests(verbose=False),
                "security": self.run_security_tests(verbose=False)
            },
            "code_quality": self.lint_and_format_check(),
            "security_scan": self.security_scan()
        }
        
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)
        
        print(f"📊 测试报告已生成: {report_file}")
        return str(report_file)

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="AGIBot测试运行器")
    
    parser.add_argument("--unit", action="store_true", help="运行单元测试")
    parser.add_argument("--integration", action="store_true", help="运行集成测试")
    parser.add_argument("--e2e", action="store_true", help="运行端到端测试")
    parser.add_argument("--performance", action="store_true", help="运行性能测试")
    parser.add_argument("--security", action="store_true", help="运行安全测试")
    parser.add_argument("--all", action="store_true", help="运行所有测试")
    parser.add_argument("--smoke", action="store_true", help="运行冒烟测试")
    parser.add_argument("--parallel", type=str, default=None, help="并行运行测试 (指定worker数量)")
    parser.add_argument("--coverage", type=int, default=None, help="运行覆盖率测试 (指定最低覆盖率)")
    parser.add_argument("--skip-slow", action="store_true", help="跳过慢速测试")
    parser.add_argument("--test", type=str, help="运行特定测试文件或路径")
    parser.add_argument("--report", action="store_true", help="生成测试执行报告")
    parser.add_argument("--lint", action="store_true", help="运行代码风格检查")
    parser.add_argument("--scan", action="store_true", help="运行安全扫描")
    parser.add_argument("--quiet", action="store_true", help="静默模式")
    
    args = parser.parse_args()
    
    runner = TestRunner()
    verbose = not args.quiet
    
    if args.unit:
        result = runner.run_unit_tests(verbose=verbose)
        sys.exit(0 if result["success"] else 1)
    
    elif args.integration:
        result = runner.run_integration_tests(verbose=verbose)
        sys.exit(0 if result["success"] else 1)
    
    elif args.e2e:
        result = runner.run_e2e_tests(verbose=verbose)
        sys.exit(0 if result["success"] else 1)
    
    elif args.performance:
        result = runner.run_performance_tests(verbose=verbose)
        sys.exit(0 if result["success"] else 1)
    
    elif args.security:
        result = runner.run_security_tests(verbose=verbose)
        sys.exit(0 if result["success"] else 1)
    
    elif args.smoke:
        result = runner.run_smoke_tests(verbose=verbose)
        sys.exit(0 if result["success"] else 1)
    
    elif args.parallel:
        result = runner.run_parallel_tests(num_workers=args.parallel, verbose=verbose)
        sys.exit(0 if result["success"] else 1)
    
    elif args.coverage is not None:
        result = runner.run_with_coverage(min_coverage=args.coverage, verbose=verbose)
        sys.exit(0 if result["success"] else 1)
    
    elif args.test:
        result = runner.run_specific_test(args.test, verbose=verbose)
        sys.exit(0 if result["success"] else 1)
    
    elif args.all:
        result = runner.run_all_tests(skip_slow=args.skip_slow, verbose=verbose)
        sys.exit(0 if result["success"] else 1)
    
    elif args.report:
        runner.generate_test_report()
        sys.exit(0)
    
    elif args.lint:
        results = runner.lint_and_format_check()
        all_passed = all(r.get("returncode", 0) == 0 for r in results.values() if isinstance(r, dict))
        sys.exit(0 if all_passed else 1)
    
    elif args.scan:
        results = runner.security_scan()
        all_passed = all(r.get("returncode", 0) == 0 for r in results.values() if isinstance(r, dict))
        sys.exit(0 if all_passed else 1)
    
    else:
        # 默认运行冒烟测试
        print("🎯 没有指定测试类型，运行冒烟测试...")
        result = runner.run_smoke_tests(verbose=verbose)
        sys.exit(0 if result["success"] else 1)

if __name__ == "__main__":
    main() 