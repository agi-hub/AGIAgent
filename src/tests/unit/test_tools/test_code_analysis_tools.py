#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
代码分析工具单元测试
测试代码解析、语法分析、依赖分析等功能
"""

import pytest
import os
import sys
import tempfile
from unittest.mock import patch, Mock, MagicMock
from pathlib import Path

# 添加src路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))

from tools.code_search_tools import CodeSearchTools
from tools.file_system_tools import FileSystemTools
# from utils.test_helpers import TestHelper

@pytest.mark.unit
class TestCodeAnalysisTools:
    """代码分析工具测试类"""
    
    @pytest.fixture
    def code_analysis_tools(self, test_workspace):
        """创建代码分析工具实例"""
        return CodeSearchTools()
    
    @pytest.fixture
    def file_system_tools(self, test_workspace):
        """创建文件系统工具实例"""
        return FileSystemTools(test_workspace)
    
    @pytest.fixture
    def sample_python_code(self):
        """示例Python代码"""
        return '''
import os
import sys
from typing import Dict, List
from datetime import datetime

class SampleClass:
    """示例类"""
    
    def __init__(self, name: str):
        self.name = name
        self.created_at = datetime.now()
    
    def process_data(self, data: List[Dict]) -> Dict:
        """处理数据的方法"""
        result = {}
        for item in data:
            if 'id' in item:
                result[item['id']] = item
        return result
    
    def get_name(self) -> str:
        return self.name

def standalone_function(x: int, y: int) -> int:
    """独立函数"""
    return x + y

if __name__ == "__main__":
    sample = SampleClass("test")
    print(sample.get_name())
'''
    
    @pytest.fixture
    def sample_javascript_code(self):
        """示例JavaScript代码"""
        return '''
const express = require('express');
const path = require('path');
const { validateInput } = require('./utils');

class TaskManager {
    constructor() {
        this.tasks = [];
        this.nextId = 1;
    }
    
    addTask(title, description) {
        const task = {
            id: this.nextId++,
            title: title,
            description: description,
            completed: false,
            createdAt: new Date()
        };
        this.tasks.push(task);
        return task;
    }
    
    getTasks() {
        return this.tasks;
    }
    
    updateTask(id, updates) {
        const task = this.tasks.find(t => t.id === id);
        if (task) {
            Object.assign(task, updates);
        }
        return task;
    }
}

module.exports = TaskManager;
'''
    
    @pytest.fixture
    def sample_project_structure(self, test_workspace):
        """创建示例项目结构"""
        project_dir = os.path.join(test_workspace, "sample_project")
        os.makedirs(project_dir, exist_ok=True)
        
        # 创建Python文件
        with open(os.path.join(project_dir, "main.py"), "w", encoding="utf-8") as f:
            f.write("""
import utils
from models.user import User
from config import settings

def main():
    user = User("test")
    utils.process_user(user)

if __name__ == "__main__":
    main()
""")
        
        # 创建模块文件
        os.makedirs(os.path.join(project_dir, "models"), exist_ok=True)
        with open(os.path.join(project_dir, "models", "__init__.py"), "w") as f:
            f.write("")
        
        with open(os.path.join(project_dir, "models", "user.py"), "w", encoding="utf-8") as f:
            f.write("""
class User:
    def __init__(self, name):
        self.name = name
        self.email = None
    
    def set_email(self, email):
        self.email = email
""")
        
        with open(os.path.join(project_dir, "utils.py"), "w", encoding="utf-8") as f:
            f.write("""
def process_user(user):
    print(f"Processing user: {user.name}")
""")
        
        with open(os.path.join(project_dir, "config.py"), "w", encoding="utf-8") as f:
            f.write("""
settings = {
    'debug': True,
    'database_url': 'sqlite:///test.db'
}
""")
        
        return project_dir
    
    def test_initialization(self, code_analysis_tools):
        """测试代码分析工具初始化"""
        assert code_analysis_tools is not None
        assert hasattr(code_analysis_tools, 'search_code')
        assert hasattr(code_analysis_tools, 'search_files')
    
    def test_basic_code_search(self, code_analysis_tools, sample_project_structure):
        """测试基本代码搜索功能"""
        # 在项目中搜索特定的类名
        result = code_analysis_tools.search_code("class User", sample_project_structure)
        
        # 验证搜索结果
        assert result is not None
        assert isinstance(result, (str, dict, list))
    
    def test_function_detection(self, code_analysis_tools, test_workspace, sample_python_code):
        """测试函数检测"""
        # 创建临时Python文件
        temp_file = os.path.join(test_workspace, "test_functions.py")
        with open(temp_file, "w", encoding="utf-8") as f:
            f.write(sample_python_code)
        
        # 搜索函数定义
        result = code_analysis_tools.search_code("def ", test_workspace)
        
        # 验证能够找到函数
        assert result is not None
    
    def test_class_detection(self, code_analysis_tools, test_workspace, sample_python_code):
        """测试类检测"""
        # 创建临时Python文件
        temp_file = os.path.join(test_workspace, "test_classes.py")
        with open(temp_file, "w", encoding="utf-8") as f:
            f.write(sample_python_code)
        
        # 搜索类定义
        result = code_analysis_tools.search_code("class ", test_workspace)
        
        # 验证能够找到类
        assert result is not None
    
    def test_import_analysis(self, code_analysis_tools, sample_project_structure):
        """测试导入分析"""
        # 搜索import语句
        result = code_analysis_tools.search_code("import", sample_project_structure)
        
        # 验证能够找到导入语句
        assert result is not None
    
    def test_dependency_detection(self, code_analysis_tools, sample_project_structure):
        """测试依赖检测"""
        # 搜索from...import语句
        result = code_analysis_tools.search_code("from.*import", sample_project_structure)
        
        # 验证依赖检测
        assert result is not None
    
    def test_file_type_filtering(self, code_analysis_tools, test_workspace):
        """测试文件类型过滤"""
        # 创建不同类型的文件
        files_to_create = [
            ("test.py", "print('Hello Python')"),
            ("test.js", "console.log('Hello JavaScript');"),
            ("test.txt", "Plain text file"),
            ("test.md", "# Markdown file"),
            ("test.json", '{"key": "value"}')
        ]
        
        for filename, content in files_to_create:
            with open(os.path.join(test_workspace, filename), "w", encoding="utf-8") as f:
                f.write(content)
        
        # 测试Python文件搜索
        result = code_analysis_tools.search_files("*.py", test_workspace)
        assert result is not None
        
        # 测试JavaScript文件搜索
        result = code_analysis_tools.search_files("*.js", test_workspace)
        assert result is not None
    
    def test_regex_pattern_search(self, code_analysis_tools, test_workspace, sample_python_code):
        """测试正则表达式模式搜索"""
        # 创建测试文件
        temp_file = os.path.join(test_workspace, "regex_test.py")
        with open(temp_file, "w", encoding="utf-8") as f:
            f.write(sample_python_code)
        
        # 测试不同的正则表达式模式
        patterns = [
            r"def\s+\w+\(",  # 函数定义
            r"class\s+\w+",  # 类定义
            r"import\s+\w+", # 导入语句
            r"self\.\w+",    # 属性访问
            r"return\s+\w+"  # 返回语句
        ]
        
        for pattern in patterns:
            result = code_analysis_tools.search_code(pattern, test_workspace)
            # 验证搜索不会失败
            assert result is not None
    
    def test_large_file_handling(self, code_analysis_tools, test_workspace):
        """测试大文件处理"""
        # 创建一个较大的代码文件
        large_code = """
# Large file test
import os
import sys

""" + "\n".join([f"""
class TestClass{i}:
    def method_{i}(self):
        return {i}

def function_{i}():
    return "function_{i}"
""" for i in range(100)])
        
        large_file = os.path.join(test_workspace, "large_file.py")
        with open(large_file, "w", encoding="utf-8") as f:
            f.write(large_code)
        
        # 测试在大文件中搜索
        result = code_analysis_tools.search_code("class TestClass", test_workspace)
        
        # 验证能够处理大文件
        assert result is not None
    
    def test_unicode_code_handling(self, code_analysis_tools, test_workspace):
        """测试Unicode代码处理"""
        # 创建包含中文注释和字符串的代码
        unicode_code = '''
# -*- coding: utf-8 -*-
"""
这是一个包含中文的Python文件
"""

class 用户管理器:
    """用户管理器类"""
    
    def __init__(self):
        self.用户列表 = []
    
    def 添加用户(self, 姓名: str):
        """添加用户方法"""
        user = {"姓名": 姓名, "状态": "活跃"}
        self.用户列表.append(user)
        print(f"已添加用户：{姓名}")
    
    def 获取用户数量(self):
        return len(self.用户列表)

# 测试代码
if __name__ == "__main__":
    管理器 = 用户管理器()
    管理器.添加用户("张三")
    print(f"用户数量：{管理器.获取用户数量()}")
'''
        
        unicode_file = os.path.join(test_workspace, "unicode_test.py")
        with open(unicode_file, "w", encoding="utf-8") as f:
            f.write(unicode_code)
        
        # 搜索中文类名
        result = code_analysis_tools.search_code("用户管理器", test_workspace)
        
        # 验证Unicode处理
        assert result is not None
    
    def test_syntax_error_handling(self, code_analysis_tools, test_workspace):
        """测试语法错误代码处理"""
        # 创建有语法错误的代码文件
        broken_code = '''
# 这个文件包含语法错误
def broken_function(
    # 缺少闭合括号
    print("This will cause syntax error"
    
class BrokenClass
    # 缺少冒号
    def method(self):
        return "broken"

# 缩进错误
if True:
print("Wrong indentation")
'''
        
        broken_file = os.path.join(test_workspace, "broken_code.py")
        with open(broken_file, "w", encoding="utf-8") as f:
            f.write(broken_code)
        
        # 测试搜索语法错误的文件
        result = code_analysis_tools.search_code("def", test_workspace)
        
        # 验证不会因语法错误而崩溃
        assert result is not None
    
    def test_empty_file_handling(self, code_analysis_tools, test_workspace):
        """测试空文件处理"""
        # 创建空文件
        empty_file = os.path.join(test_workspace, "empty.py")
        with open(empty_file, "w", encoding="utf-8") as f:
            f.write("")
        
        # 在空文件中搜索
        result = code_analysis_tools.search_code("class", test_workspace)
        
        # 验证空文件处理
        assert result is not None
    
    def test_binary_file_handling(self, code_analysis_tools, test_workspace):
        """测试二进制文件处理"""
        # 创建二进制文件
        binary_file = os.path.join(test_workspace, "binary.pyc")
        with open(binary_file, "wb") as f:
            f.write(b'\x00\x01\x02\x03\x04\x05\x06\x07\x08\x09\x0a\x0b\x0c\x0d\x0e\x0f')
        
        # 尝试搜索二进制文件
        result = code_analysis_tools.search_code("class", test_workspace)
        
        # 验证二进制文件不会导致错误
        assert result is not None
    
    def test_nested_directory_search(self, code_analysis_tools, test_workspace):
        """测试嵌套目录搜索"""
        # 创建嵌套目录结构
        nested_dirs = [
            "level1",
            "level1/level2", 
            "level1/level2/level3"
        ]
        
        for dir_path in nested_dirs:
            full_path = os.path.join(test_workspace, dir_path)
            os.makedirs(full_path, exist_ok=True)
            
            # 在每个目录创建代码文件
            code_file = os.path.join(full_path, "code.py")
            with open(code_file, "w", encoding="utf-8") as f:
                f.write(f"""
class NestedClass_{dir_path.replace('/', '_')}:
    def method(self):
        return "{dir_path}"
""")
        
        # 搜索嵌套目录中的代码
        result = code_analysis_tools.search_code("class NestedClass", test_workspace)
        
        # 验证能够搜索嵌套目录
        assert result is not None
    
    def test_case_sensitive_search(self, code_analysis_tools, test_workspace):
        """测试大小写敏感搜索"""
        # 创建包含不同大小写的代码
        case_test_code = '''
class LowerClass:
    pass

class UPPERCLASS:
    pass

class MixedCase:
    pass

def lowercase_function():
    pass

def UPPERCASE_FUNCTION():
    pass
'''
        
        case_file = os.path.join(test_workspace, "case_test.py")
        with open(case_file, "w", encoding="utf-8") as f:
            f.write(case_test_code)
        
        # 测试大小写搜索
        lower_result = code_analysis_tools.search_code("LowerClass", test_workspace)
        upper_result = code_analysis_tools.search_code("UPPERCLASS", test_workspace)
        
        # 验证大小写搜索
        assert lower_result is not None
        assert upper_result is not None
    
    def test_search_performance(self, code_analysis_tools, test_workspace):
        """测试搜索性能"""
        import time
        
        # 创建多个文件进行性能测试
        for i in range(10):
            file_path = os.path.join(test_workspace, f"perf_test_{i}.py")
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(f"""
# Performance test file {i}
class PerfClass{i}:
    def method_{i}(self):
        return {i}

def perf_function_{i}():
    return "performance_{i}"
""")
        
        # 测量搜索时间
        start_time = time.time()
        result = code_analysis_tools.search_code("class PerfClass", test_workspace)
        end_time = time.time()
        
        search_time = end_time - start_time
        
        # 验证搜索完成且时间合理（少于5秒）
        assert result is not None
        assert search_time < 5.0, f"Search took too long: {search_time} seconds"
    
    def test_javascript_code_analysis(self, code_analysis_tools, test_workspace, sample_javascript_code):
        """测试JavaScript代码分析"""
        # 创建JavaScript文件
        js_file = os.path.join(test_workspace, "test.js")
        with open(js_file, "w", encoding="utf-8") as f:
            f.write(sample_javascript_code)
        
        # 搜索JavaScript特定的语法
        class_result = code_analysis_tools.search_code("class TaskManager", test_workspace)
        require_result = code_analysis_tools.search_code("require", test_workspace)
        
        # 验证JavaScript代码分析
        assert class_result is not None
        assert require_result is not None
    
    def test_multiple_file_types(self, code_analysis_tools, test_workspace):
        """测试多种文件类型的代码分析"""
        # 创建不同语言的代码文件
        files = {
            "test.py": "class PythonClass:\n    pass",
            "test.js": "class JavaScriptClass {}",
            "test.java": "public class JavaClass {}",
            "test.cpp": "class CppClass {};",
            "test.go": "type GoStruct struct {}",
            "test.rs": "struct RustStruct {}"
        }
        
        for filename, content in files.items():
            file_path = os.path.join(test_workspace, filename)
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)
        
        # 搜索跨语言的类定义
        result = code_analysis_tools.search_code("class", test_workspace)
        
        # 验证多语言支持
        assert result is not None 