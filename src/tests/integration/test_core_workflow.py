#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AGIBot核心工作流程集成测试
测试完整的任务执行流程，包括自主执行、多轮迭代、工具调用等
"""

import pytest
import os
import json
import time
from unittest.mock import patch, Mock, MagicMock
import sys

# 添加src路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from main import AGIBotClient, AGIBotMain
from utils.test_helpers import TestHelper, MockLLMClient, TestValidator

@pytest.mark.integration
class TestCoreWorkflow:
    """核心工作流程集成测试"""
    
    @pytest.fixture
    def mock_llm_with_tool_calling(self):
        """创建支持工具调用的模拟LLM客户端"""
        client = MockLLMClient()
        
        # 添加各种响应模式
        client.add_response_pattern(r"创建.*计算器", {
            "choices": [{
                "message": {
                    "content": "我需要创建一个计算器应用。让我先创建主文件。",
                    "role": "assistant",
                    "tool_calls": [{
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "edit_file",
                            "arguments": json.dumps({
                                "target_file": "calculator.py",
                                "instructions": "Create a calculator application",
                                "code_edit": '''
def add(a, b):
    return a + b

def subtract(a, b):
    return a - b

def multiply(a, b):
    return a * b

def divide(a, b):
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b

if __name__ == "__main__":
    print("Calculator ready!")
    print(f"2 + 3 = {add(2, 3)}")
'''
                            })
                        }
                    }]
                },
                "finish_reason": "tool_calls"
            }],
            "usage": {"prompt_tokens": 100, "completion_tokens": 200, "total_tokens": 300}
        })
        
        client.add_response_pattern(r"测试.*文件", {
            "choices": [{
                "message": {
                    "content": "现在让我创建测试文件来验证计算器功能。",
                    "role": "assistant",
                    "tool_calls": [{
                        "id": "call_2",
                        "type": "function",
                        "function": {
                            "name": "edit_file",
                            "arguments": json.dumps({
                                "target_file": "test_calculator.py",
                                "instructions": "Create test file for calculator",
                                "code_edit": '''
import unittest
from calculator import add, subtract, multiply, divide

class TestCalculator(unittest.TestCase):
    def test_add(self):
        self.assertEqual(add(2, 3), 5)
    
    def test_divide_by_zero(self):
        with self.assertRaises(ValueError):
            divide(5, 0)

if __name__ == "__main__":
    unittest.main()
'''
                            })
                        }
                    }]
                },
                "finish_reason": "tool_calls"
            }],
            "usage": {"prompt_tokens": 150, "completion_tokens": 180, "total_tokens": 330}
        })
        
        client.add_response_pattern(r"运行.*测试", {
            "choices": [{
                "message": {
                    "content": "让我运行测试来验证计算器功能是否正常工作。",
                    "role": "assistant",
                    "tool_calls": [{
                        "id": "call_3",
                        "type": "function",
                        "function": {
                            "name": "run_terminal_cmd",
                            "arguments": json.dumps({
                                "command": "python test_calculator.py",
                                "is_background": False
                            })
                        }
                    }]
                },
                "finish_reason": "tool_calls"
            }],
            "usage": {"prompt_tokens": 200, "completion_tokens": 120, "total_tokens": 320}
        })
        
        # 默认完成响应
        client.add_response_pattern(r".*", {
            "choices": [{
                "message": {
                    "content": "任务已经完成！我已经创建了一个功能完整的计算器应用，包括主文件和测试文件。",
                    "role": "assistant",
                    "tool_calls": []
                },
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}
        })
        
        return client
    
    def test_basic_autonomous_execution(self, agibot_client, test_workspace, mock_llm_with_tool_calling):
        """测试基础自主执行能力"""
        with patch.object(agibot_client, '_llm_client', mock_llm_with_tool_calling):
            # 模拟工具执行结果
            with patch('tools.edit_file') as mock_edit, \
                 patch('tools.run_terminal_cmd') as mock_run:
                
                mock_edit.return_value = {"status": "success", "message": "File created successfully"}
                mock_run.return_value = {"status": "success", "output": "All tests passed"}
                
                # 执行任务
                result = agibot_client.chat(
                    messages=[{"role": "user", "content": "创建一个简单的计算器应用"}],
                    dir=test_workspace,
                    loops=5
                )
                
                # 验证结果
                assert result["success"] == True
                assert "workspace_dir" in result
                assert result["execution_time"] > 0
                
                # 验证LLM被调用
                assert mock_llm_with_tool_calling.get_call_count() > 0

    def test_multi_round_iteration(self, agibot_main, test_workspace):
        """测试多轮迭代执行"""
        # 创建一个复杂任务，需要多轮迭代
        requirement = """
        创建一个Web应用，包括：
        1. 一个Flask应用文件
        2. HTML模板
        3. CSS样式文件
        4. 一个简单的API端点
        """
        
        # 模拟LLM响应序列
        llm_responses = [
            # 第1轮：创建Flask应用
            {
                "choices": [{
                    "message": {
                        "content": "我将创建一个完整的Web应用。首先创建Flask主文件。",
                        "tool_calls": [{
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "edit_file",
                                "arguments": json.dumps({
                                    "target_file": "app.py",
                                    "code_edit": "from flask import Flask\napp = Flask(__name__)\n@app.route('/')\ndef home():\n    return 'Hello'"
                                })
                            }
                        }]
                    },
                    "finish_reason": "tool_calls"
                }]
            },
            # 第2轮：创建HTML模板
            {
                "choices": [{
                    "message": {
                        "content": "现在创建HTML模板文件。",
                        "tool_calls": [{
                            "id": "call_2",
                            "type": "function",
                            "function": {
                                "name": "edit_file",
                                "arguments": json.dumps({
                                    "target_file": "templates/index.html",
                                    "code_edit": "<!DOCTYPE html><html><head><title>Test App</title></head><body><h1>Welcome</h1></body></html>"
                                })
                            }
                        }]
                    },
                    "finish_reason": "tool_calls"
                }]
            },
            # 第3轮：完成任务
            {
                "choices": [{
                    "message": {
                        "content": "Web应用已经创建完成！",
                        "tool_calls": []
                    },
                    "finish_reason": "stop"
                }]
            }
        ]
        
        call_count = 0
        def mock_llm_call(*args, **kwargs):
            nonlocal call_count
            if call_count < len(llm_responses):
                response = llm_responses[call_count]
                call_count += 1
                return response
            return llm_responses[-1]  # 返回最后一个响应
        
        with patch('tool_executor.ToolExecutor._call_llm_api', side_effect=mock_llm_call):
            # 模拟工具执行
            with patch('tools.edit_file') as mock_edit:
                mock_edit.return_value = {"status": "success", "message": "File created"}
                
                # 执行任务
                success = agibot_main.execute_single_task(requirement, loops=5)
                
                # 验证多轮调用
                assert call_count >= 2, f"Expected multiple rounds, but only had {call_count} calls"
                assert success == True

    def test_tool_calling_integration(self, agibot_client, test_workspace):
        """测试工具调用集成"""
        # 创建一个需要多种工具的任务
        task = "分析当前目录结构，创建一个Python脚本列出所有Python文件"
        
        tool_call_sequence = [
            # 第1轮：列出目录
            {
                "choices": [{
                    "message": {
                        "content": "我需要先查看当前目录结构。",
                        "tool_calls": [{
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "list_dir",
                                "arguments": json.dumps({"relative_workspace_path": "."})
                            }
                        }]
                    },
                    "finish_reason": "tool_calls"
                }]
            },
            # 第2轮：创建分析脚本
            {
                "choices": [{
                    "message": {
                        "content": "基于目录结构，我将创建一个分析脚本。",
                        "tool_calls": [{
                            "id": "call_2",
                            "type": "function",
                            "function": {
                                "name": "edit_file",
                                "arguments": json.dumps({
                                    "target_file": "analyze_files.py",
                                    "code_edit": '''
import os
import glob

def find_python_files():
    """Find all Python files in current directory"""
    python_files = glob.glob("*.py")
    return python_files

if __name__ == "__main__":
    files = find_python_files()
    print(f"Found {len(files)} Python files:")
    for file in files:
        print(f"  - {file}")
'''
                                })
                            }
                        }]
                    },
                    "finish_reason": "tool_calls"
                }]
            },
            # 第3轮：运行脚本
            {
                "choices": [{
                    "message": {
                        "content": "现在运行脚本来验证功能。",
                        "tool_calls": [{
                            "id": "call_3",
                            "type": "function",
                            "function": {
                                "name": "run_terminal_cmd",
                                "arguments": json.dumps({
                                    "command": "python analyze_files.py",
                                    "is_background": False
                                })
                            }
                        }]
                    },
                    "finish_reason": "tool_calls"
                }]
            }
        ]
        
        call_count = 0
        def mock_sequential_responses(*args, **kwargs):
            nonlocal call_count
            if call_count < len(tool_call_sequence):
                response = tool_call_sequence[call_count]
                call_count += 1
                return response
            return {"choices": [{"message": {"content": "完成", "tool_calls": []}, "finish_reason": "stop"}]}
        
        # 模拟各种工具的执行结果
        mock_tools_responses = {
            'list_dir': "Contents: test_file.py, README.md, data.txt",
            'edit_file': {"status": "success", "message": "File created successfully"},
            'run_terminal_cmd': {"status": "success", "output": "Found 2 Python files:\n  - test_file.py\n  - analyze_files.py"}
        }
        
        with patch('tool_executor.ToolExecutor._call_llm_api', side_effect=mock_sequential_responses):
            with patch.object(agibot_client, '_execute_tool_call') as mock_tool_exec:
                # 模拟工具执行结果
                def tool_exec_side_effect(tool_call):
                    tool_name = tool_call['function']['name']
                    return mock_tools_responses.get(tool_name, {"status": "success"})
                
                mock_tool_exec.side_effect = tool_exec_side_effect
                
                # 执行任务
                result = agibot_client.chat(
                    messages=[{"role": "user", "content": task}],
                    dir=test_workspace,
                    loops=5
                )
                
                # 验证工具调用序列
                assert mock_tool_exec.call_count >= 3, "Expected multiple tool calls"
                assert result["success"] == True

    def test_error_handling_and_recovery(self, agibot_client, test_workspace):
        """测试错误处理和恢复能力"""
        # 模拟一个会出错的任务执行序列
        error_then_success_responses = [
            # 第1轮：尝试创建文件但失败
            {
                "choices": [{
                    "message": {
                        "content": "我将创建一个配置文件。",
                        "tool_calls": [{
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "edit_file",
                                "arguments": json.dumps({
                                    "target_file": "/invalid/path/config.json",  # 无效路径
                                    "code_edit": "{\"key\": \"value\"}"
                                })
                            }
                        }]
                    },
                    "finish_reason": "tool_calls"
                }]
            },
            # 第2轮：处理错误，尝试正确的路径
            {
                "choices": [{
                    "message": {
                        "content": "看起来路径有问题，让我使用正确的路径。",
                        "tool_calls": [{
                            "id": "call_2",
                            "type": "function",
                            "function": {
                                "name": "edit_file",
                                "arguments": json.dumps({
                                    "target_file": "config.json",  # 正确路径
                                    "code_edit": "{\"key\": \"value\", \"status\": \"success\"}"
                                })
                            }
                        }]
                    },
                    "finish_reason": "tool_calls"
                }]
            },
            # 第3轮：验证文件创建成功
            {
                "choices": [{
                    "message": {
                        "content": "配置文件已成功创建！",
                        "tool_calls": []
                    },
                    "finish_reason": "stop"
                }]
            }
        ]
        
        call_count = 0
        def mock_error_recovery(*args, **kwargs):
            nonlocal call_count
            if call_count < len(error_then_success_responses):
                response = error_then_success_responses[call_count]
                call_count += 1
                return response
            return error_then_success_responses[-1]
        
        tool_call_results = [
            {"status": "error", "message": "Permission denied: /invalid/path/"},  # 第一次失败
            {"status": "success", "message": "File created successfully"},  # 第二次成功
        ]
        
        tool_call_count = 0
        def mock_tool_with_error(tool_call):
            nonlocal tool_call_count
            if tool_call_count < len(tool_call_results):
                result = tool_call_results[tool_call_count]
                tool_call_count += 1
                return result
            return {"status": "success"}
        
        with patch('tool_executor.ToolExecutor._call_llm_api', side_effect=mock_error_recovery):
            with patch.object(agibot_client, '_execute_tool_call', side_effect=mock_tool_with_error):
                
                result = agibot_client.chat(
                    messages=[{"role": "user", "content": "创建一个配置文件"}],
                    dir=test_workspace,
                    loops=5
                )
                
                # 验证错误恢复
                assert call_count >= 2, "Expected error recovery with multiple attempts"
                assert result["success"] == True

    def test_context_management(self, agibot_main, test_workspace):
        """测试上下文管理和历史总结"""
        # 模拟一个需要长上下文的复杂任务
        long_requirement = """
        创建一个完整的博客系统，包括以下功能：
        1. 用户注册和登录系统
        2. 文章发布和编辑功能
        3. 评论系统
        4. 用户权限管理
        5. 数据库设计
        6. API接口设计
        7. 前端界面
        8. 单元测试
        9. 部署脚本
        10. 文档编写
        """
        
        # 模拟多轮对话，每轮都有工具调用
        conversation_rounds = []
        for i in range(10):  # 模拟10轮对话
            conversation_rounds.append({
                "choices": [{
                    "message": {
                        "content": f"正在处理功能 {i+1}：创建相关文件和代码。",
                        "tool_calls": [{
                            "id": f"call_{i+1}",
                            "type": "function",
                            "function": {
                                "name": "edit_file",
                                "arguments": json.dumps({
                                    "target_file": f"feature_{i+1}.py",
                                    "code_edit": f"# Feature {i+1} implementation\nprint('Feature {i+1} ready')"
                                })
                            }
                        }]
                    },
                    "finish_reason": "tool_calls" if i < 9 else "stop"
                }]
            })
        
        call_count = 0
        def mock_long_conversation(*args, **kwargs):
            nonlocal call_count
            if call_count < len(conversation_rounds):
                response = conversation_rounds[call_count]
                call_count += 1
                return response
            return conversation_rounds[-1]
        
        with patch('tool_executor.ToolExecutor._call_llm_api', side_effect=mock_long_conversation):
            with patch('tools.edit_file') as mock_edit:
                mock_edit.return_value = {"status": "success", "message": "File created"}
                
                # 模拟历史总结触发
                with patch('multi_round_executor.MultiRoundTaskExecutor._should_summarize_history', return_value=True):
                    with patch('multi_round_executor.MultiRoundTaskExecutor._summarize_history') as mock_summarize:
                        mock_summarize.return_value = "历史对话已总结：完成了博客系统的多个核心功能开发。"
                        
                        success = agibot_main.execute_single_task(long_requirement, loops=15)
                        
                        # 验证历史总结被调用
                        assert mock_summarize.call_count > 0, "History summarization should be triggered"
                        assert call_count >= 5, "Should have multiple conversation rounds"

    def test_workspace_isolation(self, test_workspace):
        """测试工作空间隔离"""
        # 创建两个独立的AGIBot实例
        client1 = AGIBotClient(debug_mode=True)
        client2 = AGIBotClient(debug_mode=True)
        
        workspace1 = os.path.join(test_workspace, "workspace1")
        workspace2 = os.path.join(test_workspace, "workspace2")
        
        os.makedirs(workspace1, exist_ok=True)
        os.makedirs(workspace2, exist_ok=True)
        
        # 模拟LLM响应
        def mock_llm_response(*args, **kwargs):
            return {
                "choices": [{
                    "message": {
                        "content": "我将创建一个测试文件。",
                        "tool_calls": [{
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "edit_file",
                                "arguments": json.dumps({
                                    "target_file": "unique_file.txt",
                                    "code_edit": "This is a unique file"
                                })
                            }
                        }]
                    },
                    "finish_reason": "tool_calls"
                }]
            }
        
        with patch('tool_executor.ToolExecutor._call_llm_api', side_effect=mock_llm_response):
            with patch('tools.edit_file') as mock_edit:
                mock_edit.return_value = {"status": "success", "message": "File created"}
                
                # 在两个工作空间中执行任务
                result1 = client1.chat(
                    messages=[{"role": "user", "content": "创建一个文件"}],
                    dir=workspace1
                )
                
                result2 = client2.chat(
                    messages=[{"role": "user", "content": "创建一个文件"}],
                    dir=workspace2
                )
                
                # 验证工作空间隔离
                assert result1["success"] == True
                assert result2["success"] == True
                assert result1["workspace_dir"] != result2["workspace_dir"]

    def test_task_completion_detection(self, agibot_client, test_workspace):
        """测试任务完成检测"""
        # 模拟一个明确完成的任务序列
        completion_sequence = [
            # 工作轮次
            {
                "choices": [{
                    "message": {
                        "content": "我正在创建文件。",
                        "tool_calls": [{
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "edit_file",
                                "arguments": json.dumps({
                                    "target_file": "completed_task.py",
                                    "code_edit": "print('Task completed successfully')"
                                })
                            }
                        }]
                    },
                    "finish_reason": "tool_calls"
                }]
            },
            # 完成轮次
            {
                "choices": [{
                    "message": {
                        "content": "任务已经完成！我已经成功创建了所需的文件。",
                        "tool_calls": []
                    },
                    "finish_reason": "stop"
                }]
            }
        ]
        
        call_count = 0
        def mock_completion_detection(*args, **kwargs):
            nonlocal call_count
            if call_count < len(completion_sequence):
                response = completion_sequence[call_count]
                call_count += 1
                return response
            return completion_sequence[-1]
        
        with patch('tool_executor.ToolExecutor._call_llm_api', side_effect=mock_completion_detection):
            with patch('tools.edit_file') as mock_edit:
                mock_edit.return_value = {"status": "success", "message": "File created"}
                
                start_time = time.time()
                result = agibot_client.chat(
                    messages=[{"role": "user", "content": "创建一个简单的Python文件"}],
                    dir=test_workspace,
                    loops=10  # 给足够多的轮数
                )
                end_time = time.time()
                
                # 验证任务提前完成（没有用完所有轮数）
                assert result["success"] == True
                assert call_count == 2, "Task should complete after 2 rounds"
                assert end_time - start_time < 10, "Task should complete quickly"

    @pytest.mark.slow
    def test_performance_under_load(self, test_workspace):
        """测试负载下的性能表现"""
        import threading
        import queue
        
        results_queue = queue.Queue()
        error_queue = queue.Queue()
        
        def worker_task(worker_id):
            try:
                client = AGIBotClient(debug_mode=False)  # 关闭调试模式以提高性能
                
                # 模拟快速LLM响应
                def quick_response(*args, **kwargs):
                    return {
                        "choices": [{
                            "message": {
                                "content": f"Worker {worker_id} task completed",
                                "tool_calls": []
                            },
                            "finish_reason": "stop"
                        }]
                    }
                
                with patch('tool_executor.ToolExecutor._call_llm_api', side_effect=quick_response):
                    result = client.chat(
                        messages=[{"role": "user", "content": f"Simple task for worker {worker_id}"}],
                        dir=os.path.join(test_workspace, f"worker_{worker_id}"),
                        loops=3
                    )
                    results_queue.put((worker_id, result))
                    
            except Exception as e:
                error_queue.put((worker_id, str(e)))
        
        # 启动多个并发任务
        num_workers = 5
        threads = []
        
        start_time = time.time()
        for i in range(num_workers):
            thread = threading.Thread(target=worker_task, args=(i,))
            threads.append(thread)
            thread.start()
        
        # 等待所有任务完成
        for thread in threads:
            thread.join(timeout=30)  # 30秒超时
        
        end_time = time.time()
        
        # 收集结果
        results = []
        errors = []
        
        while not results_queue.empty():
            results.append(results_queue.get())
        
        while not error_queue.empty():
            errors.append(error_queue.get())
        
        # 验证性能
        assert len(errors) == 0, f"Concurrent execution had errors: {errors}"
        assert len(results) == num_workers, f"Expected {num_workers} results, got {len(results)}"
        assert end_time - start_time < 60, f"Concurrent execution took too long: {end_time - start_time} seconds"
        
        # 验证所有任务都成功
        for worker_id, result in results:
            assert result["success"] == True, f"Worker {worker_id} failed: {result}" 