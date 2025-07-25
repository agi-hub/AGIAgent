#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文件系统工具单元测试
测试read_file, edit_file, list_dir等文件操作工具
"""

import pytest
import os
import tempfile
import shutil
from unittest.mock import patch, Mock
import sys

# 添加src路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))

from tools.file_system_tools import FileSystemTools
from utils.test_helpers import TestHelper, TestValidator

class TestFileSystemTools:
    """文件系统工具测试类"""
    
    @pytest.fixture
    def file_tools(self, test_workspace):
        """创建文件系统工具实例"""
        tools = FileSystemTools()
        tools.workspace_root = test_workspace
        return tools
    
    @pytest.fixture
    def sample_files(self, test_workspace):
        """创建示例测试文件"""
        files = {}
        
        # 简单文本文件
        simple_file = os.path.join(test_workspace, "simple.txt")
        with open(simple_file, 'w', encoding='utf-8') as f:
            f.write("Hello, World!\nThis is a test file.\nLine 3")
        files['simple'] = simple_file
        
        # Python文件
        python_file = os.path.join(test_workspace, "example.py")
        with open(python_file, 'w', encoding='utf-8') as f:
            f.write('''
def hello_world():
    """Print hello world"""
    print("Hello, World!")

def add_numbers(a, b):
    """Add two numbers"""
    return a + b

if __name__ == "__main__":
    hello_world()
    result = add_numbers(2, 3)
    print(f"2 + 3 = {result}")
''')
        files['python'] = python_file
        
        # 大文件
        large_file = os.path.join(test_workspace, "large.txt")
        with open(large_file, 'w', encoding='utf-8') as f:
            for i in range(100):
                f.write(f"This is line {i+1}\n")
        files['large'] = large_file
        
        # 创建子目录和文件
        subdir = os.path.join(test_workspace, "subdir")
        os.makedirs(subdir, exist_ok=True)
        subfile = os.path.join(subdir, "subfile.txt")
        with open(subfile, 'w', encoding='utf-8') as f:
            f.write("This is a file in subdirectory")
        files['subfile'] = subfile
        
        return files

    # read_file 测试
    def test_read_file_basic(self, file_tools, sample_files):
        """测试基础文件读取功能"""
        result = file_tools.read_file(relative_workspace_path="simple.txt")
        
        assert result is not None
        assert "Hello, World!" in result
        assert "This is a test file." in result

    def test_read_file_with_line_range(self, file_tools, sample_files):
        """测试按行范围读取文件"""
        result = file_tools.read_file(
            relative_workspace_path="large.txt",
            start_line_one_indexed=10,
            end_line_one_indexed_inclusive=15
        )
        
        assert result is not None
        lines = result.strip().split('\n')
        assert len(lines) == 6  # 10-15 inclusive
        assert "This is line 10" in result
        assert "This is line 15" in result

    def test_read_file_nonexistent(self, file_tools):
        """测试读取不存在的文件"""
        result = file_tools.read_file(relative_workspace_path="nonexistent.txt")
        
        # 应该返回错误信息或None
        assert result is None or "not found" in str(result).lower()

    def test_read_file_entire_file(self, file_tools, sample_files):
        """测试读取整个文件"""
        result = file_tools.read_file(
            relative_workspace_path="example.py",
            should_read_entire_file=True
        )
        
        assert result is not None
        assert "def hello_world():" in result
        assert "def add_numbers(a, b):" in result
        assert "if __name__ == \"__main__\":" in result

    def test_read_file_path_traversal_protection(self, file_tools):
        """测试路径遍历攻击防护"""
        # 尝试访问上级目录
        result = file_tools.read_file(relative_workspace_path="../../../etc/passwd")
        
        # 应该被阻止或返回错误
        assert result is None or "not allowed" in str(result).lower()

    # edit_file 测试
    def test_edit_file_create_new(self, file_tools, test_workspace):
        """测试创建新文件"""
        new_file_path = "new_file.py"
        content = """
def test_function():
    return "Hello from new file"
"""
        
        result = file_tools.edit_file(
            target_file=new_file_path,
            instructions="Create a new Python file with a test function",
            code_edit=content
        )
        
        # 验证文件被创建
        full_path = os.path.join(test_workspace, new_file_path)
        assert os.path.exists(full_path)
        
        # 验证内容
        with open(full_path, 'r', encoding='utf-8') as f:
            file_content = f.read()
        assert "def test_function():" in file_content

    def test_edit_file_modify_existing(self, file_tools, sample_files):
        """测试修改现有文件"""
        # 修改Python文件，添加新函数
        new_content = '''
def hello_world():
    """Print hello world"""
    print("Hello, World!")

def add_numbers(a, b):
    """Add two numbers"""
    return a + b

def multiply_numbers(a, b):
    """Multiply two numbers"""
    return a * b

if __name__ == "__main__":
    hello_world()
    result = add_numbers(2, 3)
    print(f"2 + 3 = {result}")
    product = multiply_numbers(4, 5)
    print(f"4 * 5 = {product}")
'''
        
        result = file_tools.edit_file(
            target_file="example.py",
            instructions="Add a multiply function",
            code_edit=new_content
        )
        
        # 验证文件被修改
        with open(sample_files['python'], 'r', encoding='utf-8') as f:
            content = f.read()
        assert "def multiply_numbers(a, b):" in content

    def test_edit_file_invalid_syntax(self, file_tools, test_workspace):
        """测试编辑文件时的语法验证"""
        invalid_python = """
def broken_function(
    # 缺少参数列表的闭合括号
    return "This is broken"
"""
        
        result = file_tools.edit_file(
            target_file="broken.py",
            instructions="Create a file with syntax error",
            code_edit=invalid_python
        )
        
        # 文件可能被创建，但应该有语法错误警告
        file_path = os.path.join(test_workspace, "broken.py")
        if os.path.exists(file_path):
            is_valid, error = TestValidator.validate_python_syntax(file_path)
            assert not is_valid

    # list_dir 测试
    def test_list_dir_root(self, file_tools, sample_files):
        """测试列出根目录"""
        result = file_tools.list_dir(relative_workspace_path=".")
        
        assert result is not None
        
        # 验证包含创建的文件
        expected_items = ["simple.txt", "example.py", "large.txt", "subdir"]
        for item in expected_items:
            assert item in str(result)

    def test_list_dir_subdirectory(self, file_tools, sample_files):
        """测试列出子目录"""
        result = file_tools.list_dir(relative_workspace_path="subdir")
        
        assert result is not None
        assert "subfile.txt" in str(result)

    def test_list_dir_nonexistent(self, file_tools):
        """测试列出不存在的目录"""
        result = file_tools.list_dir(relative_workspace_path="nonexistent_dir")
        
        # 应该返回错误或空结果
        assert result is None or "not found" in str(result).lower()

    def test_list_dir_empty_directory(self, file_tools, test_workspace):
        """测试列出空目录"""
        empty_dir = os.path.join(test_workspace, "empty_dir")
        os.makedirs(empty_dir, exist_ok=True)
        
        result = file_tools.list_dir(relative_workspace_path="empty_dir")
        
        assert result is not None
        # 空目录应该返回空列表或表示为空的信息

    # 边界条件测试
    def test_large_file_handling(self, file_tools, test_workspace):
        """测试处理大文件"""
        # 创建一个相对较大的文件（1MB）
        large_file = os.path.join(test_workspace, "large_file.txt")
        with open(large_file, 'w', encoding='utf-8') as f:
            for i in range(10000):
                f.write(f"This is line {i+1} with some additional content to make it longer.\n")
        
        # 测试读取大文件的一部分
        result = file_tools.read_file(
            relative_workspace_path="large_file.txt",
            start_line_one_indexed=1000,
            end_line_one_indexed_inclusive=1100
        )
        
        assert result is not None
        lines = result.strip().split('\n')
        assert len(lines) == 101  # 1000-1100 inclusive

    def test_unicode_file_handling(self, file_tools, test_workspace):
        """测试处理Unicode文件"""
        unicode_content = """
# 中文测试文件
def 问候():
    print("你好，世界！")

# Emoji 测试
def emoji_function():
    return "🚀🤖💻 AGI Bot 测试"

# 特殊字符
special_chars = "àáâãäåæçèéêëìíîïðñòóôõö÷øùúûüýþÿ"
"""
        
        unicode_file = os.path.join(test_workspace, "unicode.py")
        with open(unicode_file, 'w', encoding='utf-8') as f:
            f.write(unicode_content)
        
        # 测试读取Unicode文件
        result = file_tools.read_file(relative_workspace_path="unicode.py")
        
        assert result is not None
        assert "你好，世界！" in result
        assert "🚀🤖💻" in result

    def test_binary_file_handling(self, file_tools, test_workspace):
        """测试处理二进制文件"""
        # 创建一个简单的二进制文件
        binary_file = os.path.join(test_workspace, "binary.bin")
        with open(binary_file, 'wb') as f:
            f.write(b'\x00\x01\x02\x03\xFF\xFE\xFD')
        
        # 尝试读取二进制文件
        result = file_tools.read_file(relative_workspace_path="binary.bin")
        
        # 应该优雅处理或给出适当错误
        assert result is None or "binary" in str(result).lower()

    # 权限和安全测试
    def test_workspace_boundary_enforcement(self, file_tools, test_workspace):
        """测试工作空间边界强制执行"""
        # 尝试访问工作空间外的文件
        outside_paths = [
            "../outside.txt",
            "../../etc/passwd",
            "/etc/hosts",
            "/tmp/test.txt"
        ]
        
        for path in outside_paths:
            result = file_tools.read_file(relative_workspace_path=path)
            # 应该被阻止或返回安全错误
            assert result is None or "not allowed" in str(result).lower() or "access denied" in str(result).lower()

    def test_special_filename_handling(self, file_tools, test_workspace):
        """测试特殊文件名处理"""
        special_names = [
            "file with spaces.txt",
            "file-with-dashes.txt", 
            "file_with_underscores.txt",
            "file.with.dots.txt",
            "文件中文名.txt"
        ]
        
        for name in special_names:
            # 创建文件
            file_path = os.path.join(test_workspace, name)
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(f"Content of {name}")
            
            # 测试读取
            result = file_tools.read_file(relative_workspace_path=name)
            assert result is not None
            assert f"Content of {name}" in result

    # 性能测试
    def test_concurrent_file_operations(self, file_tools, test_workspace):
        """测试并发文件操作"""
        import threading
        import time
        
        results = []
        errors = []
        
        def read_file_worker(file_name):
            try:
                result = file_tools.read_file(relative_workspace_path=file_name)
                results.append(result)
            except Exception as e:
                errors.append(e)
        
        # 创建测试文件
        for i in range(10):
            file_path = os.path.join(test_workspace, f"concurrent_{i}.txt")
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(f"Content of file {i}")
        
        # 启动并发读取
        threads = []
        for i in range(10):
            thread = threading.Thread(target=read_file_worker, args=(f"concurrent_{i}.txt",))
            threads.append(thread)
            thread.start()
        
        # 等待所有线程完成
        for thread in threads:
            thread.join()
        
        # 验证结果
        assert len(errors) == 0, f"Concurrent operations had errors: {errors}"
        assert len(results) == 10, f"Expected 10 results, got {len(results)}"

    # 错误恢复测试
    def test_file_operation_error_recovery(self, file_tools, test_workspace):
        """测试文件操作错误恢复"""
        # 测试读取权限被拒绝的文件（在支持的系统上）
        restricted_file = os.path.join(test_workspace, "restricted.txt")
        with open(restricted_file, 'w') as f:
            f.write("This file will have restricted permissions")
        
        try:
            # 尝试移除读权限（仅在Unix系统上有效）
            os.chmod(restricted_file, 0o000)
            
            result = file_tools.read_file(relative_workspace_path="restricted.txt")
            
            # 应该优雅处理权限错误
            assert result is None or "permission" in str(result).lower()
            
        except (OSError, NotImplementedError):
            # 在Windows上或其他不支持权限的系统上跳过
            pytest.skip("Permission testing not supported on this system")
        finally:
            # 恢复权限以便清理
            try:
                os.chmod(restricted_file, 0o644)
            except:
                pass

    def test_memory_usage_large_operations(self, file_tools, test_workspace):
        """测试大操作的内存使用"""
        import psutil
        import os as os_module
        
        # 获取当前进程
        process = psutil.Process(os_module.getpid())
        
        # 记录初始内存使用
        initial_memory = process.memory_info().rss
        
        # 执行大文件操作
        large_content = "x" * (1024 * 1024)  # 1MB content
        
        result = file_tools.edit_file(
            target_file="memory_test.txt",
            instructions="Create large file for memory testing",
            code_edit=large_content
        )
        
        # 读取大文件
        read_result = file_tools.read_file(relative_workspace_path="memory_test.txt")
        
        # 检查内存使用
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        
        # 内存增长应该在合理范围内（比如不超过10MB）
        assert memory_increase < 10 * 1024 * 1024, f"Memory usage increased by {memory_increase} bytes" 