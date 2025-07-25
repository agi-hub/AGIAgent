#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试报告生成器
生成详细的测试执行报告，包括HTML、JSON、XML格式
"""

import json
import os
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from pathlib import Path
import statistics
import xml.etree.ElementTree as ET

class TestMetricsCollector:
    """测试指标收集器"""
    
    def __init__(self):
        self.test_results = []
        self.start_time = None
        self.end_time = None
        self.environment_info = {}
    
    def start_collection(self):
        """开始收集指标"""
        self.start_time = datetime.now()
        self.collect_environment_info()
    
    def end_collection(self):
        """结束收集指标"""
        self.end_time = datetime.now()
    
    def collect_environment_info(self):
        """收集环境信息"""
        import platform
        import sys
        import psutil
        
        self.environment_info = {
            "python_version": sys.version,
            "platform": platform.platform(),
            "cpu_count": psutil.cpu_count(),
            "memory_total_gb": psutil.virtual_memory().total / 1024**3,
            "disk_total_gb": psutil.disk_usage('/').total / 1024**3,
            "hostname": platform.node(),
            "architecture": platform.architecture()[0],
            "timestamp": datetime.now().isoformat()
        }
    
    def add_test_result(self, test_result: Dict[str, Any]):
        """添加测试结果"""
        if "timestamp" not in test_result:
            test_result["timestamp"] = datetime.now().isoformat()
        self.test_results.append(test_result)
    
    def get_summary_statistics(self) -> Dict[str, Any]:
        """获取汇总统计"""
        if not self.test_results:
            return {}
        
        total_tests = len(self.test_results)
        passed_tests = len([r for r in self.test_results if r.get("outcome") == "passed"])
        failed_tests = len([r for r in self.test_results if r.get("outcome") == "failed"])
        skipped_tests = len([r for r in self.test_results if r.get("outcome") == "skipped"])
        error_tests = len([r for r in self.test_results if r.get("outcome") == "error"])
        
        durations = [r.get("duration", 0) for r in self.test_results if "duration" in r]
        total_duration = sum(durations)
        
        execution_time = None
        if self.start_time and self.end_time:
            execution_time = (self.end_time - self.start_time).total_seconds()
        
        return {
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "failed_tests": failed_tests,
            "skipped_tests": skipped_tests,
            "error_tests": error_tests,
            "success_rate": passed_tests / total_tests if total_tests > 0 else 0,
            "total_duration": total_duration,
            "average_test_duration": statistics.mean(durations) if durations else 0,
            "max_test_duration": max(durations) if durations else 0,
            "min_test_duration": min(durations) if durations else 0,
            "execution_time": execution_time,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None
        }
    
    def get_test_categories(self) -> Dict[str, List[Dict]]:
        """按类别分组测试结果"""
        categories = {
            "unit": [],
            "integration": [],
            "e2e": [],
            "performance": [],
            "security": []
        }
        
        for result in self.test_results:
            test_name = result.get("name", "").lower()
            if "unit" in test_name or "/unit/" in result.get("file", ""):
                categories["unit"].append(result)
            elif "integration" in test_name or "/integration/" in result.get("file", ""):
                categories["integration"].append(result)
            elif "e2e" in test_name or "/e2e/" in result.get("file", ""):
                categories["e2e"].append(result)
            elif "performance" in test_name or "/performance/" in result.get("file", ""):
                categories["performance"].append(result)
            elif "security" in test_name or "/security/" in result.get("file", ""):
                categories["security"].append(result)
        
        return categories

class TestReportGenerator:
    """测试报告生成器"""
    
    def __init__(self, output_dir: str = "test_reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.metrics_collector = TestMetricsCollector()
    
    def generate_html_report(self, metrics: TestMetricsCollector, filename: str = None) -> str:
        """生成HTML格式报告"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"test_report_{timestamp}.html"
        
        report_path = self.output_dir / filename
        
        summary = metrics.get_summary_statistics()
        categories = metrics.get_test_categories()
        
        html_content = self._generate_html_template(summary, categories, metrics)
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return str(report_path)
    
    def _generate_html_template(self, summary: Dict, categories: Dict, metrics: TestMetricsCollector) -> str:
        """生成HTML模板"""
        passed_rate = summary.get("success_rate", 0) * 100
        
        html = f"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AGIBot 测试报告</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; }}
        .header {{ text-align: center; margin-bottom: 30px; }}
        .summary {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin-bottom: 30px; }}
        .metric-card {{ background: #f8f9fa; padding: 20px; border-radius: 8px; text-align: center; }}
        .metric-value {{ font-size: 2em; font-weight: bold; margin-bottom: 5px; }}
        .metric-label {{ color: #666; }}
        .passed {{ color: #28a745; }}
        .failed {{ color: #dc3545; }}
        .skipped {{ color: #ffc107; }}
        .progress-bar {{ width: 100%; height: 20px; background: #e9ecef; border-radius: 10px; overflow: hidden; }}
        .progress-fill {{ height: 100%; background: #28a745; transition: width 0.3s; }}
        .test-categories {{ margin-bottom: 30px; }}
        .category {{ margin-bottom: 20px; }}
        .category-header {{ background: #e9ecef; padding: 10px; border-radius: 5px; font-weight: bold; }}
        .test-list {{ margin-top: 10px; }}
        .test-item {{ padding: 8px; border-left: 4px solid #ddd; margin-bottom: 5px; background: #f8f9fa; }}
        .test-item.passed {{ border-left-color: #28a745; }}
        .test-item.failed {{ border-left-color: #dc3545; }}
        .test-item.skipped {{ border-left-color: #ffc107; }}
        .environment {{ background: #f8f9fa; padding: 15px; border-radius: 8px; margin-top: 30px; }}
        .environment h3 {{ margin-top: 0; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🤖 AGIBot 测试报告</h1>
            <p>生成时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
        </div>
        
        <div class="summary">
            <div class="metric-card">
                <div class="metric-value passed">{summary.get('passed_tests', 0)}</div>
                <div class="metric-label">通过测试</div>
            </div>
            <div class="metric-card">
                <div class="metric-value failed">{summary.get('failed_tests', 0)}</div>
                <div class="metric-label">失败测试</div>
            </div>
            <div class="metric-card">
                <div class="metric-value skipped">{summary.get('skipped_tests', 0)}</div>
                <div class="metric-label">跳过测试</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{summary.get('total_tests', 0)}</div>
                <div class="metric-label">总测试数</div>
            </div>
        </div>
        
        <div class="metric-card">
            <h3>测试通过率</h3>
            <div class="progress-bar">
                <div class="progress-fill" style="width: {passed_rate}%"></div>
            </div>
            <p>{passed_rate:.1f}%</p>
        </div>
        
        <div class="test-categories">
            <h2>测试分类详情</h2>
            {self._generate_category_html(categories)}
        </div>
        
        <div class="environment">
            <h3>测试环境信息</h3>
            {self._generate_environment_html(metrics.environment_info)}
        </div>
    </div>
</body>
</html>
"""
        return html
    
    def _generate_category_html(self, categories: Dict) -> str:
        """生成分类HTML"""
        html_parts = []
        
        for category_name, tests in categories.items():
            if not tests:
                continue
                
            passed = len([t for t in tests if t.get("outcome") == "passed"])
            failed = len([t for t in tests if t.get("outcome") == "failed"])
            total = len(tests)
            
            html_parts.append(f"""
            <div class="category">
                <div class="category-header">
                    {category_name.upper()} 测试 ({passed}/{total} 通过)
                </div>
                <div class="test-list">
                    {self._generate_test_items_html(tests)}
                </div>
            </div>
            """)
        
        return "".join(html_parts)
    
    def _generate_test_items_html(self, tests: List[Dict]) -> str:
        """生成测试项HTML"""
        html_parts = []
        
        for test in tests:
            outcome = test.get("outcome", "unknown")
            name = test.get("name", "Unknown Test")
            duration = test.get("duration", 0)
            error_msg = test.get("error_message", "")
            
            html_parts.append(f"""
            <div class="test-item {outcome}">
                <strong>{name}</strong>
                <span style="float: right;">{duration:.3f}s</span>
                {f'<br><small style="color: #dc3545;">{error_msg}</small>' if error_msg else ''}
            </div>
            """)
        
        return "".join(html_parts)
    
    def _generate_environment_html(self, env_info: Dict) -> str:
        """生成环境信息HTML"""
        html_parts = []
        
        for key, value in env_info.items():
            if key != "timestamp":
                display_key = key.replace("_", " ").title()
                html_parts.append(f"<p><strong>{display_key}:</strong> {value}</p>")
        
        return "".join(html_parts)
    
    def generate_json_report(self, metrics: TestMetricsCollector, filename: str = None) -> str:
        """生成JSON格式报告"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"test_report_{timestamp}.json"
        
        report_path = self.output_dir / filename
        
        report_data = {
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "generator": "AGIBot Test Report Generator",
                "version": "1.0.0"
            },
            "summary": metrics.get_summary_statistics(),
            "environment": metrics.environment_info,
            "test_categories": metrics.get_test_categories(),
            "detailed_results": metrics.test_results
        }
        
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)
        
        return str(report_path)
    
    def generate_junit_xml(self, metrics: TestMetricsCollector, filename: str = None) -> str:
        """生成JUnit XML格式报告"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"junit_report_{timestamp}.xml"
        
        report_path = self.output_dir / filename
        summary = metrics.get_summary_statistics()
        
        # 创建根元素
        root = ET.Element("testsuites")
        root.set("name", "AGIBot Tests")
        root.set("tests", str(summary.get("total_tests", 0)))
        root.set("failures", str(summary.get("failed_tests", 0)))
        root.set("errors", str(summary.get("error_tests", 0)))
        root.set("skipped", str(summary.get("skipped_tests", 0)))
        root.set("time", str(summary.get("total_duration", 0)))
        
        # 按类别创建测试套件
        categories = metrics.get_test_categories()
        
        for category_name, tests in categories.items():
            if not tests:
                continue
            
            testsuite = ET.SubElement(root, "testsuite")
            testsuite.set("name", f"AGIBot.{category_name}")
            testsuite.set("tests", str(len(tests)))
            
            category_failures = len([t for t in tests if t.get("outcome") == "failed"])
            category_errors = len([t for t in tests if t.get("outcome") == "error"])
            category_skipped = len([t for t in tests if t.get("outcome") == "skipped"])
            category_time = sum(t.get("duration", 0) for t in tests)
            
            testsuite.set("failures", str(category_failures))
            testsuite.set("errors", str(category_errors))
            testsuite.set("skipped", str(category_skipped))
            testsuite.set("time", str(category_time))
            
            for test in tests:
                testcase = ET.SubElement(testsuite, "testcase")
                testcase.set("name", test.get("name", "Unknown"))
                testcase.set("classname", f"AGIBot.{category_name}")
                testcase.set("time", str(test.get("duration", 0)))
                
                outcome = test.get("outcome", "unknown")
                if outcome == "failed":
                    failure = ET.SubElement(testcase, "failure")
                    failure.set("message", test.get("error_message", "Test failed"))
                    failure.text = test.get("error_traceback", "")
                elif outcome == "error":
                    error = ET.SubElement(testcase, "error")
                    error.set("message", test.get("error_message", "Test error"))
                    error.text = test.get("error_traceback", "")
                elif outcome == "skipped":
                    skipped = ET.SubElement(testcase, "skipped")
                    skipped.set("message", test.get("skip_reason", "Test skipped"))
        
        # 写入XML文件
        tree = ET.ElementTree(root)
        tree.write(report_path, encoding='utf-8', xml_declaration=True)
        
        return str(report_path)
    
    def generate_coverage_report(self, coverage_data: Dict, filename: str = None) -> str:
        """生成覆盖率报告"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"coverage_report_{timestamp}.html"
        
        report_path = self.output_dir / filename
        
        html_content = self._generate_coverage_html(coverage_data)
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return str(report_path)
    
    def _generate_coverage_html(self, coverage_data: Dict) -> str:
        """生成覆盖率HTML报告"""
        total_lines = coverage_data.get("total_lines", 0)
        covered_lines = coverage_data.get("covered_lines", 0)
        coverage_percent = (covered_lines / total_lines * 100) if total_lines > 0 else 0
        
        html = f"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <title>代码覆盖率报告</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .coverage-summary {{ background: #f8f9fa; padding: 20px; border-radius: 8px; }}
        .coverage-bar {{ width: 100%; height: 30px; background: #e9ecef; border-radius: 15px; }}
        .coverage-fill {{ height: 100%; background: #28a745; border-radius: 15px; }}
        .file-coverage {{ margin: 20px 0; }}
        .file-name {{ font-weight: bold; margin-bottom: 10px; }}
    </style>
</head>
<body>
    <h1>代码覆盖率报告</h1>
    
    <div class="coverage-summary">
        <h2>总体覆盖率: {coverage_percent:.1f}%</h2>
        <div class="coverage-bar">
            <div class="coverage-fill" style="width: {coverage_percent}%"></div>
        </div>
        <p>覆盖行数: {covered_lines} / {total_lines}</p>
    </div>
    
    <div class="file-coverage">
        <h3>文件覆盖率详情</h3>
        {self._generate_file_coverage_html(coverage_data.get("files", {}))}
    </div>
</body>
</html>
"""
        return html
    
    def _generate_file_coverage_html(self, files_data: Dict) -> str:
        """生成文件覆盖率HTML"""
        html_parts = []
        
        for file_path, file_coverage in files_data.items():
            lines_total = file_coverage.get("lines_total", 0)
            lines_covered = file_coverage.get("lines_covered", 0)
            percent = (lines_covered / lines_total * 100) if lines_total > 0 else 0
            
            html_parts.append(f"""
            <div class="file-coverage">
                <div class="file-name">{file_path}</div>
                <div class="coverage-bar">
                    <div class="coverage-fill" style="width: {percent}%"></div>
                </div>
                <small>{lines_covered}/{lines_total} lines ({percent:.1f}%)</small>
            </div>
            """)
        
        return "".join(html_parts)
    
    def generate_performance_report(self, performance_data: Dict, filename: str = None) -> str:
        """生成性能测试报告"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"performance_report_{timestamp}.html"
        
        report_path = self.output_dir / filename
        
        html_content = self._generate_performance_html(performance_data)
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return str(report_path)
    
    def _generate_performance_html(self, performance_data: Dict) -> str:
        """生成性能测试HTML报告"""
        html = f"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <title>性能测试报告</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .performance-metric {{ background: #f8f9fa; padding: 15px; margin: 10px 0; border-radius: 8px; }}
        .metric-value {{ font-size: 1.5em; font-weight: bold; color: #007bff; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
    </style>
</head>
<body>
    <h1>性能测试报告</h1>
    
    <div class="performance-metric">
        <h3>平均响应时间</h3>
        <div class="metric-value">{performance_data.get('avg_response_time', 0):.3f}s</div>
    </div>
    
    <div class="performance-metric">
        <h3>吞吐量</h3>
        <div class="metric-value">{performance_data.get('throughput', 0):.2f} req/s</div>
    </div>
    
    <div class="performance-metric">
        <h3>内存峰值使用</h3>
        <div class="metric-value">{performance_data.get('peak_memory_mb', 0):.1f} MB</div>
    </div>
    
    <h2>详细性能指标</h2>
    <table>
        <tr>
            <th>指标</th>
            <th>值</th>
            <th>单位</th>
        </tr>
        {self._generate_performance_table_rows(performance_data)}
    </table>
</body>
</html>
"""
        return html
    
    def _generate_performance_table_rows(self, performance_data: Dict) -> str:
        """生成性能指标表格行"""
        rows = []
        
        metrics = [
            ("最小响应时间", "min_response_time", "秒"),
            ("最大响应时间", "max_response_time", "秒"),
            ("95百分位响应时间", "p95_response_time", "秒"),
            ("99百分位响应时间", "p99_response_time", "秒"),
            ("CPU使用率峰值", "peak_cpu_percent", "%"),
            ("平均CPU使用率", "avg_cpu_percent", "%"),
            ("磁盘I/O读取", "total_disk_read_mb", "MB"),
            ("磁盘I/O写入", "total_disk_write_mb", "MB")
        ]
        
        for display_name, key, unit in metrics:
            value = performance_data.get(key, 0)
            if isinstance(value, float):
                value_str = f"{value:.3f}"
            else:
                value_str = str(value)
            
            rows.append(f"""
            <tr>
                <td>{display_name}</td>
                <td>{value_str}</td>
                <td>{unit}</td>
            </tr>
            """)
        
        return "".join(rows)
    
    def generate_comprehensive_report(self, metrics: TestMetricsCollector, 
                                    coverage_data: Dict = None, 
                                    performance_data: Dict = None) -> Dict[str, str]:
        """生成综合报告"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        reports = {}
        
        # 生成各种格式的报告
        reports["html"] = self.generate_html_report(metrics, f"comprehensive_report_{timestamp}.html")
        reports["json"] = self.generate_json_report(metrics, f"comprehensive_report_{timestamp}.json")
        reports["junit"] = self.generate_junit_xml(metrics, f"junit_report_{timestamp}.xml")
        
        if coverage_data:
            reports["coverage"] = self.generate_coverage_report(coverage_data, f"coverage_report_{timestamp}.html")
        
        if performance_data:
            reports["performance"] = self.generate_performance_report(performance_data, f"performance_report_{timestamp}.html")
        
        return reports 