#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
单任务完整执行流程集成测试
测试从用户需求到任务完成的完整流程
"""

import pytest
import os
import tempfile
import shutil
import json
from unittest.mock import patch, Mock
import sys

# 添加src路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from main import AGIBotClient, AGIBotMain

@pytest.mark.integration
class TestSingleTaskWorkflow:
    """单任务工作流程集成测试"""
    
    @pytest.fixture
    def temp_workspace(self):
        """创建临时工作空间"""
        temp_dir = tempfile.mkdtemp(prefix="workflow_test_")
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def agibot_client(self):
        """创建测试客户端"""
        return AGIBotClient(
            api_key="test_key",
            model="test_model",
            debug_mode=True,
            single_task_mode=True
        )
    
    def test_simple_code_generation_workflow(self, agibot_client, temp_workspace):
        """测试简单代码生成工作流程"""
        user_requirement = "创建一个简单的Python计算器函数"
        
        # 模拟LLM响应序列
        llm_responses = [
            # 第1轮：分析需求并创建文件
            {
                "choices": [{
                    "message": {
                        "content": "我将创建一个简单的Python计算器函数。",
                        "tool_calls": [{
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "edit_file",
                                "arguments": json.dumps({
                                    "target_file": "calculator.py",
                                    "instructions": "Create a simple calculator function",
                                    "code_edit": '''def calculator(operation, a, b):
    """Simple calculator function"""
    if operation == "add":
        return a + b
    elif operation == "subtract":
        return a - b
    elif operation == "multiply":
        return a * b
    elif operation == "divide":
        if b != 0:
            return a / b
        else:
            raise ValueError("Cannot divide by zero")
    else:
        raise ValueError("Invalid operation")

if __name__ == "__main__":
    print("Calculator ready!")
    print(f"2 + 3 = {calculator('add', 2, 3)}")'''
                                })
                            }
                        }],
                        "role": "assistant"
                    },
                    "finish_reason": "tool_calls"
                }]
            },
            
            # 第2轮：完成任务
            {
                "choices": [{
                    "message": {
                        "content": "计算器函数已创建完成！包含了基本的四则运算功能和错误处理。",
                        "tool_calls": [],
                        "role": "assistant"
                    },
                    "finish_reason": "stop"
                }]
            }
        ]
        
        # 模拟工具调用结果
        tool_results = {
            "edit_file": {"status": "success", "message": "File created successfully"}
        }
        
        call_count = 0
        def mock_llm_sequence(*args, **kwargs):
            nonlocal call_count
            if call_count < len(llm_responses):
                response = llm_responses[call_count]
                call_count += 1
                return response
            return llm_responses[-1]
        
        def mock_tool_execution(tool_call):
            tool_name = tool_call['function']['name']
            return tool_results.get(tool_name, {"status": "success"})
        
        # 执行工作流程
        with patch('tool_executor.ToolExecutor._call_llm_api', side_effect=mock_llm_sequence):
            with patch.object(agibot_client, '_execute_tool_call', side_effect=mock_tool_execution):
                result = agibot_client.chat(
                    messages=[{"role": "user", "content": user_requirement}],
                    dir=temp_workspace,
                    loops=5
                )
        
        # 验证工作流程结果
        assert result["success"] == True
        assert result["execution_time"] > 0
        assert os.path.exists(result["output_dir"])
        assert call_count >= 2, "Expected multiple LLM interactions"
    
    def test_file_creation_and_modification_workflow(self, agibot_client, temp_workspace):
        """测试文件创建和修改工作流程"""
        user_requirement = "创建一个配置文件并添加一些设置"
        
        llm_responses = [
            # 创建初始配置文件
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
                                    "target_file": "config.json",
                                    "code_edit": '{"version": "1.0", "debug": false}'
                                })
                            }
                        }],
                        "role": "assistant"
                    },
                    "finish_reason": "tool_calls"
                }]
            },
            
            # 添加更多设置
            {
                "choices": [{
                    "message": {
                        "content": "现在添加更多配置选项。",
                        "tool_calls": [{
                            "id": "call_2",
                            "type": "function",
                            "function": {
                                "name": "edit_file",
                                "arguments": json.dumps({
                                    "target_file": "config.json",
                                    "code_edit": '{"version": "1.0", "debug": false, "timeout": 30, "retry_count": 3}'
                                })
                            }
                        }],
                        "role": "assistant"
                    },
                    "finish_reason": "tool_calls"
                }]
            },
            
            # 完成
            {
                "choices": [{
                    "message": {
                        "content": "配置文件已创建并配置完成！",
                        "tool_calls": [],
                        "role": "assistant"
                    },
                    "finish_reason": "stop"
                }]
            }
        ]
        
        call_count = 0
        def mock_llm_sequence(*args, **kwargs):
            nonlocal call_count
            if call_count < len(llm_responses):
                response = llm_responses[call_count]
                call_count += 1
                return response
            return llm_responses[-1]
        
        with patch('tool_executor.ToolExecutor._call_llm_api', side_effect=mock_llm_sequence):
            with patch.object(agibot_client, '_execute_tool_call') as mock_tool:
                mock_tool.return_value = {"status": "success", "message": "File operation successful"}
                
                result = agibot_client.chat(
                    messages=[{"role": "user", "content": user_requirement}],
                    dir=temp_workspace,
                    loops=5
                )
        
        assert result["success"] == True
        assert mock_tool.call_count >= 2, "Expected multiple file operations"
    
    def test_error_recovery_workflow(self, agibot_client, temp_workspace):
        """测试错误恢复工作流程"""
        user_requirement = "创建一个文件，但可能会遇到错误"
        
        llm_responses = [
            # 第一次尝试（失败）
            {
                "choices": [{
                    "message": {
                        "content": "我将创建文件。",
                        "tool_calls": [{
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "edit_file",
                                "arguments": json.dumps({
                                    "target_file": "/invalid/path/file.txt",
                                    "code_edit": "test content"
                                })
                            }
                        }],
                        "role": "assistant"
                    },
                    "finish_reason": "tool_calls"
                }]
            },
            
            # 错误恢复
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
                                    "target_file": "file.txt",
                                    "code_edit": "test content"
                                })
                            }
                        }],
                        "role": "assistant"
                    },
                    "finish_reason": "tool_calls"
                }]
            },
            
            # 成功完成
            {
                "choices": [{
                    "message": {
                        "content": "文件创建成功！",
                        "tool_calls": [],
                        "role": "assistant"
                    },
                    "finish_reason": "stop"
                }]
            }
        ]
        
        # 模拟工具调用结果（第一次失败，第二次成功）
        tool_call_results = [
            {"status": "error", "message": "Invalid path"},
            {"status": "success", "message": "File created"}
        ]
        
        call_count = 0
        tool_call_count = 0
        
        def mock_llm_sequence(*args, **kwargs):
            nonlocal call_count
            if call_count < len(llm_responses):
                response = llm_responses[call_count]
                call_count += 1
                return response
            return llm_responses[-1]
        
        def mock_tool_with_error(tool_call):
            nonlocal tool_call_count
            if tool_call_count < len(tool_call_results):
                result = tool_call_results[tool_call_count]
                tool_call_count += 1
                return result
            return {"status": "success"}
        
        with patch('tool_executor.ToolExecutor._call_llm_api', side_effect=mock_llm_sequence):
            with patch.object(agibot_client, '_execute_tool_call', side_effect=mock_tool_with_error):
                result = agibot_client.chat(
                    messages=[{"role": "user", "content": user_requirement}],
                    dir=temp_workspace,
                    loops=5
                )
        
        # 验证错误恢复成功
        assert result["success"] == True
        assert call_count >= 3, "Expected error recovery attempts"
        assert tool_call_count >= 2, "Expected multiple tool calls with error recovery"
    
    def test_multi_tool_workflow(self, agibot_client, temp_workspace):
        """测试多工具协作工作流程"""
        user_requirement = "创建一个Python脚本并运行它"
        
        llm_responses = [
            # 创建脚本
            {
                "choices": [{
                    "message": {
                        "content": "我将创建一个Python脚本。",
                        "tool_calls": [{
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "edit_file",
                                "arguments": json.dumps({
                                    "target_file": "hello.py",
                                    "code_edit": 'print("Hello, World!")\nprint("Script is working!")'
                                })
                            }
                        }],
                        "role": "assistant"
                    },
                    "finish_reason": "tool_calls"
                }]
            },
            
            # 运行脚本
            {
                "choices": [{
                    "message": {
                        "content": "现在运行脚本来验证它是否工作。",
                        "tool_calls": [{
                            "id": "call_2",
                            "type": "function",
                            "function": {
                                "name": "run_terminal_cmd",
                                "arguments": json.dumps({
                                    "command": "python hello.py",
                                    "is_background": False
                                })
                            }
                        }],
                        "role": "assistant"
                    },
                    "finish_reason": "tool_calls"
                }]
            },
            
            # 完成
            {
                "choices": [{
                    "message": {
                        "content": "Python脚本已创建并成功运行！",
                        "tool_calls": [],
                        "role": "assistant"
                    },
                    "finish_reason": "stop"
                }]
            }
        ]
        
        # 模拟不同工具的结果
        def mock_tool_execution(tool_call):
            tool_name = tool_call['function']['name']
            if tool_name == "edit_file":
                return {"status": "success", "message": "File created"}
            elif tool_name == "run_terminal_cmd":
                return {"status": "success", "output": "Hello, World!\nScript is working!"}
            else:
                return {"status": "success"}
        
        call_count = 0
        def mock_llm_sequence(*args, **kwargs):
            nonlocal call_count
            if call_count < len(llm_responses):
                response = llm_responses[call_count]
                call_count += 1
                return response
            return llm_responses[-1]
        
        with patch('tool_executor.ToolExecutor._call_llm_api', side_effect=mock_llm_sequence):
            with patch.object(agibot_client, '_execute_tool_call', side_effect=mock_tool_execution):
                result = agibot_client.chat(
                    messages=[{"role": "user", "content": user_requirement}],
                    dir=temp_workspace,
                    loops=5
                )
        
        assert result["success"] == True
        assert call_count >= 3, "Expected multiple tool interactions"
    
    def test_workflow_timeout_handling(self, agibot_client, temp_workspace):
        """测试工作流程超时处理"""
        user_requirement = "执行一个可能需要很长时间的任务"
        
        # 模拟达到最大轮数的情况
        long_responses = []
        for i in range(10):  # 生成很多轮响应
            long_responses.append({
                "choices": [{
                    "message": {
                        "content": f"正在处理步骤 {i+1}...",
                        "tool_calls": [{
                            "id": f"call_{i+1}",
                            "type": "function",
                            "function": {
                                "name": "edit_file",
                                "arguments": json.dumps({
                                    "target_file": f"step_{i+1}.txt",
                                    "code_edit": f"Step {i+1} content"
                                })
                            }
                        }],
                        "role": "assistant"
                    },
                    "finish_reason": "tool_calls"
                }]
            })
        
        call_count = 0
        def mock_long_sequence(*args, **kwargs):
            nonlocal call_count
            if call_count < len(long_responses):
                response = long_responses[call_count]
                call_count += 1
                return response
            return long_responses[-1]
        
        with patch('tool_executor.ToolExecutor._call_llm_api', side_effect=mock_long_sequence):
            with patch.object(agibot_client, '_execute_tool_call') as mock_tool:
                mock_tool.return_value = {"status": "success", "message": "Step completed"}
                
                result = agibot_client.chat(
                    messages=[{"role": "user", "content": user_requirement}],
                    dir=temp_workspace,
                    loops=3  # 设置较小的最大轮数
                )
        
        # 验证超时处理
        assert result["success"] == False  # 应该因为达到最大轮数而失败
        assert "execution rounds" in result["message"]
    
    def test_workflow_with_different_configurations(self, temp_workspace):
        """测试不同配置下的工作流程"""
        # 测试调试模式
        debug_client = AGIBotClient(
            api_key="test_key",
            model="test_model",
            debug_mode=True,
            detailed_summary=True
        )
        
        # 测试非调试模式
        production_client = AGIBotClient(
            api_key="test_key",
            model="test_model",
            debug_mode=False,
            detailed_summary=False
        )
        
        user_requirement = "创建一个简单的文件"
        
        mock_response = {
            "choices": [{
                "message": {
                    "content": "任务完成！",
                    "tool_calls": [],
                    "role": "assistant"
                },
                "finish_reason": "stop"
            }]
        }
        
        with patch('tool_executor.ToolExecutor._call_llm_api', return_value=mock_response):
            with patch.object(debug_client, '_execute_tool_call') as mock_debug_tool:
                with patch.object(production_client, '_execute_tool_call') as mock_prod_tool:
                    mock_debug_tool.return_value = {"status": "success"}
                    mock_prod_tool.return_value = {"status": "success"}
                    
                    # 执行调试模式
                    debug_result = debug_client.chat(
                        messages=[{"role": "user", "content": user_requirement}],
                        dir=os.path.join(temp_workspace, "debug")
                    )
                    
                    # 执行生产模式  
                    prod_result = production_client.chat(
                        messages=[{"role": "user", "content": user_requirement}],
                        dir=os.path.join(temp_workspace, "production")
                    )
        
        # 验证两种模式都成功
        assert debug_result["success"] == True
        assert prod_result["success"] == True
        assert debug_result["output_dir"] != prod_result["output_dir"] 