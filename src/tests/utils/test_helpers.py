#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试辅助工具
为精简测试框架提供实用的测试工具和辅助函数
"""

import os
import json
import tempfile
import time
import random
import string
from typing import Dict, Any, List, Optional
from unittest.mock import Mock, MagicMock

class TestHelper:
    """测试辅助工具类"""
    
    @staticmethod
    def generate_test_requirement(complexity: str = "simple") -> str:
        """生成测试需求"""
        requirements = {
            "simple": [
                "创建一个Hello World程序",
                "编写一个计算器函数",
                "生成一个配置文件",
                "创建一个简单的Python脚本"
            ],
            "medium": [
                "开发一个简单的Web应用",
                "创建一个数据处理脚本",
                "实现一个命令行工具",
                "设计一个简单的API接口"
            ],
            "complex": [
                "构建一个完整的项目管理系统",
                "开发一个多用户博客平台",
                "创建一个分布式任务调度器",
                "实现一个微服务架构应用"
            ]
        }
        return random.choice(requirements.get(complexity, requirements["simple"]))
    
    @staticmethod
    def create_mock_llm_response(content: str, tool_calls: Optional[List] = None, finish_reason: str = "stop") -> Dict:
        """创建模拟LLM响应"""
        return {
            "choices": [{
                "message": {
                    "content": content,
                    "tool_calls": tool_calls or [],
                    "role": "assistant"
                },
                "finish_reason": finish_reason
            }],
            "usage": {
                "prompt_tokens": random.randint(50, 200),
                "completion_tokens": random.randint(20, 100),
                "total_tokens": random.randint(70, 300)
            }
        }
    
    @staticmethod
    def create_mock_tool_call(tool_name: str, arguments: Dict[str, Any], call_id: Optional[str] = None) -> Dict:
        """创建模拟工具调用"""
        return {
            "id": call_id or f"call_{int(time.time())}_{random.randint(1000, 9999)}",
            "type": "function",
            "function": {
                "name": tool_name,
                "arguments": json.dumps(arguments)
            }
        }
    
    @staticmethod
    def create_mock_tool_result(status: str = "success", message: str = "", content: str = "", output: str = "") -> Dict:
        """创建模拟工具执行结果"""
        result = {"status": status}
        if message:
            result["message"] = message
        if content:
            result["content"] = content
        if output:
            result["output"] = output
        return result
    
    @staticmethod
    def generate_test_file_content(file_type: str = "python") -> str:
        """生成测试文件内容"""
        templates = {
            "python": '''#!/usr/bin/env python3
def main():
    print("Hello, World!")
    return 0

if __name__ == "__main__":
    main()
''',
            "javascript": '''console.log("Hello, World!");

function greet(name) {
    return `Hello, ${name}!`;
}

module.exports = { greet };
''',
            "html": '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Test Page</title>
</head>
<body>
    <h1>Hello, World!</h1>
    <p>This is a test page.</p>
</body>
</html>
''',
            "json": '''{
    "name": "test-project",
    "version": "1.0.0",
    "description": "A test project",
    "main": "index.js",
    "dependencies": {}
}
''',
            "config": '''# Configuration file
debug=true
port=8080
host=localhost
timeout=30
'''
        }
        return templates.get(file_type, templates["python"])
    
    @staticmethod
    def create_temp_config(config_data: Dict[str, Any]) -> str:
        """创建临时配置文件"""
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        json.dump(config_data, temp_file, indent=2)
        temp_file.close()
        return temp_file.name
    
    @staticmethod
    def cleanup_temp_files(file_paths: List[str]):
        """清理临时文件"""
        for file_path in file_paths:
            try:
                if os.path.exists(file_path):
                    os.unlink(file_path)
            except Exception:
                pass  # 忽略清理错误
    
    @staticmethod
    def measure_execution_time(func, *args, **kwargs) -> tuple:
        """测量函数执行时间"""
        start_time = time.time()
        result = func(*args, **kwargs)
        execution_time = time.time() - start_time
        return result, execution_time
    
    @staticmethod
    def generate_random_string(length: int = 10) -> str:
        """生成随机字符串"""
        return ''.join(random.choices(string.ascii_letters + string.digits, k=length))
    
    @staticmethod
    def create_simple_project_structure(base_dir: str) -> Dict[str, str]:
        """创建简单的项目结构"""
        structure = {
            "main.py": TestHelper.generate_test_file_content("python"),
            "config.json": TestHelper.generate_test_file_content("json"),
            "README.md": "# Test Project\n\nThis is a test project.",
            "requirements.txt": "pytest==7.4.0\nrequests==2.31.0\n"
        }
        
        created_files = {}
        for filename, content in structure.items():
            file_path = os.path.join(base_dir, filename)
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            created_files[filename] = file_path
        
        return created_files

class MockLLMClient:
    """模拟LLM客户端"""
    
    def __init__(self):
        self.response_patterns = {}
        self.call_count = 0
        self.call_history = []
    
    def add_response_pattern(self, pattern: str, response: Dict):
        """添加响应模式"""
        self.response_patterns[pattern] = response
    
    def get_response(self, prompt: str) -> Dict:
        """根据提示获取响应"""
        self.call_count += 1
        self.call_history.append(prompt)
        
        # 查找匹配的模式
        for pattern, response in self.response_patterns.items():
            if pattern.lower() in prompt.lower():
                return response
        
        # 默认响应
        return TestHelper.create_mock_llm_response("任务已完成。")
    
    def get_call_count(self) -> int:
        """获取调用次数"""
        return self.call_count
    
    def get_call_history(self) -> List[str]:
        """获取调用历史"""
        return self.call_history.copy()
    
    def reset(self):
        """重置状态"""
        self.call_count = 0
        self.call_history = []

class TestValidator:
    """测试验证工具类"""
    
    @staticmethod
    def validate_agibot_result(result: Dict) -> bool:
        """验证AGIBot结果格式"""
        required_fields = ["success", "message", "output_dir", "execution_time", "details"]
        return all(field in result for field in required_fields)
    
    @staticmethod
    def validate_file_exists(file_path: str) -> bool:
        """验证文件是否存在"""
        return os.path.exists(file_path) and os.path.isfile(file_path)
    
    @staticmethod
    def validate_directory_exists(dir_path: str) -> bool:
        """验证目录是否存在"""
        return os.path.exists(dir_path) and os.path.isdir(dir_path)
    
    @staticmethod
    def validate_python_syntax(file_path: str) -> tuple:
        """验证Python文件语法"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                code = f.read()
            compile(code, file_path, 'exec')
            return True, None
        except SyntaxError as e:
            return False, str(e)
        except Exception as e:
            return False, str(e)
    
    @staticmethod
    def validate_json_format(file_path: str) -> tuple:
        """验证JSON文件格式"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                json.load(f)
            return True, None
        except json.JSONDecodeError as e:
            return False, str(e)
        except Exception as e:
            return False, str(e)
    
    @staticmethod
    def validate_performance_metrics(metrics: Dict, thresholds: Dict) -> Dict:
        """验证性能指标"""
        results = {}
        for metric, value in metrics.items():
            threshold = thresholds.get(metric)
            if threshold is not None:
                results[metric] = {
                    "value": value,
                    "threshold": threshold,
                    "passed": value <= threshold
                }
        return results

class PerformanceMonitor:
    """性能监控工具"""
    
    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.peak_memory = 0
        self.initial_memory = 0
        
    def __enter__(self):
        import psutil
        self.process = psutil.Process()
        self.start_time = time.time()
        self.initial_memory = self.process.memory_info().rss
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        
    def get_execution_time(self) -> float:
        """获取执行时间"""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return 0
    
    def get_memory_usage(self) -> Dict:
        """获取内存使用情况"""
        if hasattr(self, 'process'):
            current_memory = self.process.memory_info().rss
            return {
                "initial_mb": self.initial_memory / 1024 / 1024,
                "current_mb": current_memory / 1024 / 1024,
                "increase_mb": (current_memory - self.initial_memory) / 1024 / 1024
            }
        return {}
    
    def get_metrics(self) -> Dict:
        """获取所有性能指标"""
        return {
            "execution_time": self.get_execution_time(),
            "memory_usage": self.get_memory_usage()
        }
    
    def get_max_memory_usage(self) -> int:
        """获取最大内存使用量"""
        if hasattr(self, 'process'):
            return self.process.memory_info().rss
        return 0
    
    def get_memory_growth(self) -> int:
        """获取内存增长量"""
        if hasattr(self, 'process'):
            current_memory = self.process.memory_info().rss
            return current_memory - self.initial_memory
        return 0

class ResourceTracker:
    """资源跟踪器"""
    
    def __init__(self):
        self.samples = []
        self.monitoring = False
        
    def __enter__(self):
        self.start_monitoring()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop_monitoring()
        
    def start_monitoring(self):
        """开始监控"""
        import threading
        import psutil
        
        self.monitoring = True
        self.process = psutil.Process()
        
        def monitor():
            while self.monitoring:
                try:
                    sample = {
                        "timestamp": time.time(),
                        "memory_mb": self.process.memory_info().rss / 1024 / 1024,
                        "cpu_percent": self.process.cpu_percent()
                    }
                    self.samples.append(sample)
                    time.sleep(0.1)
                except:
                    break
        
        self.monitor_thread = threading.Thread(target=monitor)
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        """停止监控"""
        self.monitoring = False
        if hasattr(self, 'monitor_thread'):
            self.monitor_thread.join(timeout=1)
    
    def get_report(self) -> Dict:
        """获取监控报告"""
        if not self.samples:
            return {}
        
        memory_values = [s["memory_mb"] for s in self.samples]
        cpu_values = [s["cpu_percent"] for s in self.samples if s["cpu_percent"] > 0]
        
        return {
            "peak_memory_mb": max(memory_values),
            "avg_memory_mb": sum(memory_values) / len(memory_values),
            "peak_cpu_percent": max(cpu_values) if cpu_values else 0,
            "avg_cpu_percent": sum(cpu_values) / len(cpu_values) if cpu_values else 0,
            "sample_count": len(self.samples)
        }
    
    def get_statistics(self) -> Dict:
        """获取统计信息"""
        return self.get_report()

# 常用的测试数据和配置
TEST_CONFIGS = {
    "minimal": {
        "api_key": "test_key_minimal",
        "model": "test_model",
        "debug_mode": False,
        "loops": 3
    },
    "standard": {
        "api_key": "test_key_standard", 
        "model": "test_model",
        "debug_mode": True,
        "loops": 5
    },
    "performance": {
        "api_key": "test_key_perf",
        "model": "test_model",
        "debug_mode": False,
        "loops": 10
    }
}

# 性能基准线
PERFORMANCE_BASELINES = {
    "client_init_time": 1.0,           # 客户端初始化时间（秒）
    "simple_task_time": 5.0,           # 简单任务执行时间（秒）
    "memory_increase_mb": 50,          # 内存增长限制（MB）
    "max_cpu_percent": 80,             # 最大CPU使用率（%）
    "concurrent_tasks_time": 15.0,     # 并发任务总时间（秒）
} 