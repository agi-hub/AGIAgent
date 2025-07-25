#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AGIBot测试框架全局配置
提供测试夹具和全局配置
"""

import pytest
import tempfile
import shutil
import os
import json
import time
from typing import Dict, Any, Optional
from unittest.mock import Mock, patch

# 添加src目录到Python路径
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from main import AGIBotClient, AGIBotMain
from tools import Tools
from config_loader import load_config

# 测试配置常量
TEST_CONFIG = {
    "api_key": "test_api_key_123456789",
    "model": "test-model-v1",
    "api_base": "https://test-api.example.com/v1",
    "debug_mode": True,
    "max_loops": 5,  # 测试时使用较少的轮数
    "timeout": 30    # 测试超时时间
}

@pytest.fixture(scope="session")
def test_workspace():
    """创建临时测试工作空间"""
    temp_dir = tempfile.mkdtemp(prefix="agibot_test_")
    print(f"Created test workspace: {temp_dir}")
    yield temp_dir
    shutil.rmtree(temp_dir)
    print(f"Cleaned up test workspace: {temp_dir}")

@pytest.fixture(scope="session")
def test_config_dir(test_workspace):
    """创建测试配置目录"""
    config_dir = os.path.join(test_workspace, "config")
    os.makedirs(config_dir, exist_ok=True)
    
    # 创建测试配置文件
    config_file = os.path.join(config_dir, "config.txt")
    with open(config_file, 'w') as f:
        for key, value in TEST_CONFIG.items():
            f.write(f"{key}={value}\n")
    
    # 创建MCP配置文件
    mcp_config = {
        "mcpServers": {
            "test_server": {
                "command": "echo",
                "args": ["test"],
                "env": {}
            }
        }
    }
    mcp_file = os.path.join(config_dir, "mcp_servers.json")
    with open(mcp_file, 'w') as f:
        json.dump(mcp_config, f, indent=2)
    
    return config_dir

@pytest.fixture
def agibot_client(test_workspace):
    """AGIBot客户端测试实例"""
    return AGIBotClient(
        api_key=TEST_CONFIG["api_key"],
        model=TEST_CONFIG["model"],
        api_base=TEST_CONFIG["api_base"],
        debug_mode=True
    )

@pytest.fixture
def agibot_main(test_workspace):
    """AGIBot主程序测试实例"""
    output_dir = os.path.join(test_workspace, "test_output")
    return AGIBotMain(
        out_dir=output_dir,
        api_key=TEST_CONFIG["api_key"],
        model=TEST_CONFIG["model"],
        api_base=TEST_CONFIG["api_base"],
        debug_mode=True,
        single_task_mode=True
    )

@pytest.fixture
def tools_instance(test_workspace):
    """工具实例"""
    return Tools(
        workspace_root=test_workspace,
        llm_api_key=TEST_CONFIG["api_key"],
        llm_model=TEST_CONFIG["model"],
        llm_api_base=TEST_CONFIG["api_base"]
    )

@pytest.fixture
def mock_llm_response():
    """模拟LLM响应的工厂函数"""
    def _create_response(content: str, tool_calls: Optional[list] = None, finish_reason: str = "stop"):
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
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "total_tokens": 150
            }
        }
    return _create_response

@pytest.fixture
def mock_successful_tool_call():
    """模拟成功的工具调用"""
    def _create_tool_call(tool_name: str, arguments: Dict[str, Any]):
        return {
            "id": f"call_{int(time.time())}",
            "type": "function",
            "function": {
                "name": tool_name,
                "arguments": json.dumps(arguments)
            }
        }
    return _create_tool_call

@pytest.fixture
def sample_project_structure():
    """示例项目结构"""
    return {
        "web_app": {
            "app.py": '''
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/data', methods=['GET'])
def get_data():
    return jsonify({"message": "Hello from API"})

if __name__ == '__main__':
    app.run(debug=True)
''',
            "templates/index.html": '''
<!DOCTYPE html>
<html>
<head>
    <title>Test Web App</title>
    <link rel="stylesheet" href="{{ url_for('static', filename='style.css') }}">
</head>
<body>
    <h1>Welcome to Test App</h1>
    <div id="content"></div>
    <script src="{{ url_for('static', filename='script.js') }}"></script>
</body>
</html>
''',
            "static/style.css": '''
body {
    font-family: Arial, sans-serif;
    margin: 0;
    padding: 20px;
    background-color: #f5f5f5;
}

h1 {
    color: #333;
    text-align: center;
}

#content {
    max-width: 800px;
    margin: 0 auto;
    background: white;
    padding: 20px;
    border-radius: 8px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}
''',
            "static/script.js": '''
document.addEventListener('DOMContentLoaded', function() {
    fetch('/api/data')
        .then(response => response.json())
        .then(data => {
            document.getElementById('content').innerHTML = '<p>' + data.message + '</p>';
        })
        .catch(error => {
            console.error('Error:', error);
        });
});
''',
            "requirements.txt": "flask==2.3.3\ngunicorn==21.2.0",
            "README.md": '''# Test Web Application

A simple Flask web application for testing purposes.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
python app.py
```

Visit http://localhost:5000 to see the application.
'''
        },
        
        "data_analysis": {
            "analysis.py": '''
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def load_data(file_path):
    """Load data from CSV file"""
    return pd.read_csv(file_path)

def analyze_sales_trends(df):
    """Analyze sales trends"""
    df['date'] = pd.to_datetime(df['date'])
    monthly_sales = df.groupby(df['date'].dt.to_period('M'))['sales'].sum()
    
    plt.figure(figsize=(12, 6))
    monthly_sales.plot(kind='line')
    plt.title('Monthly Sales Trends')
    plt.xlabel('Month')
    plt.ylabel('Sales')
    plt.grid(True)
    plt.savefig('sales_trends.png')
    plt.close()
    
    return monthly_sales

def generate_report(df):
    """Generate analysis report"""
    total_sales = df['sales'].sum()
    avg_sales = df['sales'].mean()
    max_sales = df['sales'].max()
    min_sales = df['sales'].min()
    
    report = f"""
# Sales Analysis Report

## Summary Statistics
- Total Sales: ${total_sales:,.2f}
- Average Sales: ${avg_sales:,.2f}
- Maximum Sales: ${max_sales:,.2f}
- Minimum Sales: ${min_sales:,.2f}

## Data Points
- Total Records: {len(df)}
- Date Range: {df['date'].min()} to {df['date'].max()}
"""
    
    with open('sales_report.md', 'w') as f:
        f.write(report)
    
    return report

if __name__ == '__main__':
    # Sample usage
    data = {
        'date': pd.date_range('2023-01-01', periods=100, freq='D'),
        'sales': np.random.normal(1000, 200, 100)
    }
    df = pd.DataFrame(data)
    df.to_csv('sample_sales.csv', index=False)
    
    # Run analysis
    trends = analyze_sales_trends(df)
    report = generate_report(df)
    print("Analysis complete!")
''',
            "requirements.txt": "pandas==2.0.3\nmatplotlib==3.7.2\nnumpy==1.24.3",
            "sample_data.csv": '''date,sales,product,region
2023-01-01,1200.50,Product A,North
2023-01-02,980.25,Product B,South
2023-01-03,1150.00,Product A,East
2023-01-04,890.75,Product C,West
2023-01-05,1300.25,Product B,North
''',
            "README.md": '''# Sales Data Analysis

Data analysis project for sales trends and reporting.

## Features
- Sales trend analysis
- Statistical summaries
- Automated report generation
- Data visualization

## Usage
```bash
python analysis.py
```
'''
        }
    }

@pytest.fixture
def create_test_project():
    """创建测试项目的工厂函数"""
    def _create_project(workspace: str, project_type: str, project_data: Dict):
        project_dir = os.path.join(workspace, f"test_{project_type}")
        os.makedirs(project_dir, exist_ok=True)
        
        for file_path, content in project_data.items():
            full_path = os.path.join(project_dir, file_path)
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            
            with open(full_path, 'w', encoding='utf-8') as f:
                f.write(content)
        
        return project_dir
    
    return _create_project

# pytest配置
def pytest_configure(config):
    """pytest配置钩子"""
    # 添加自定义标记
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "e2e: marks tests as end-to-end tests"
    )
    config.addinivalue_line(
        "markers", "performance: marks tests as performance tests"
    )
    config.addinivalue_line(
        "markers", "security: marks tests as security tests"
    )

def pytest_collection_modifyitems(config, items):
    """修改测试收集行为"""
    # 为慢速测试添加标记
    slow_tests = []
    for item in items:
        if "performance" in item.nodeid or "e2e" in item.nodeid:
            item.add_marker(pytest.mark.slow)
            slow_tests.append(item.nodeid)
    
    if slow_tests:
        print(f"Marked {len(slow_tests)} tests as slow")

@pytest.fixture(autouse=True)
def test_environment_setup():
    """自动设置测试环境"""
    # 设置环境变量
    os.environ["AGIBOT_TEST_MODE"] = "true"
    os.environ["AGIBOT_DEBUG"] = "true"
    
    yield
    
    # 清理环境变量
    os.environ.pop("AGIBOT_TEST_MODE", None)
    os.environ.pop("AGIBOT_DEBUG", None)

# 错误处理和日志
import logging

@pytest.fixture(scope="session", autouse=True)
def setup_test_logging():
    """设置测试日志"""
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('test.log'),
            logging.StreamHandler()
        ]
    )
    
    # 创建测试专用logger
    test_logger = logging.getLogger('agibot_test')
    test_logger.info("AGIBot test session started")
    
    yield test_logger
    
    test_logger.info("AGIBot test session ended") 