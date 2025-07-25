#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
端到端完整场景测试
测试真实的用户使用场景，从需求输入到最终交付
"""

import pytest
import os
import time
import json
from unittest.mock import patch, Mock
import subprocess
import sys

# 添加src路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from main import AGIBotClient, AGIBotMain
from utils.test_helpers import TestHelper, TestValidator

@pytest.mark.e2e
@pytest.mark.slow
class TestCompleteScenarios:
    """完整场景端到端测试"""
    
    def test_web_application_development_e2e(self, test_workspace, sample_project_structure, create_test_project):
        """端到端测试：完整Web应用开发"""
        
        # 用户需求
        requirement = """
        开发一个任务管理Web应用，包括：
        1. 用户可以添加、编辑、删除任务
        2. 任务有标题、描述、优先级、状态
        3. 支持任务列表查看和筛选
        4. 简洁的Web界面
        5. 使用Flask作为后端
        6. 包含基本的单元测试
        """
        
        # 创建AGIBot实例
        client = AGIBotClient(
            api_key="test_key",
            model="test_model",
            debug_mode=True
        )
        
        # 模拟完整的开发流程响应
        development_phases = [
            # 阶段1：项目规划和结构创建
            {
                "content": "我将开发一个完整的任务管理Web应用。首先创建项目结构和主应用文件。",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "edit_file",
                            "arguments": json.dumps({
                                "target_file": "app.py",
                                "instructions": "Create Flask main application",
                                "code_edit": '''
from flask import Flask, render_template, request, jsonify, redirect, url_for
import sqlite3
import os
from datetime import datetime

app = Flask(__name__)

# Database initialization
def init_db():
    conn = sqlite3.connect('tasks.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS tasks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            description TEXT,
            priority TEXT DEFAULT 'medium',
            status TEXT DEFAULT 'pending',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/tasks', methods=['GET'])
def get_tasks():
    conn = sqlite3.connect('tasks.db')
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM tasks ORDER BY created_at DESC')
    tasks = cursor.fetchall()
    conn.close()
    
    task_list = []
    for task in tasks:
        task_list.append({
            'id': task[0],
            'title': task[1],
            'description': task[2],
            'priority': task[3],
            'status': task[4],
            'created_at': task[5]
        })
    
    return jsonify(task_list)

@app.route('/api/tasks', methods=['POST'])
def create_task():
    data = request.get_json()
    conn = sqlite3.connect('tasks.db')
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO tasks (title, description, priority, status)
        VALUES (?, ?, ?, ?)
    ''', (data['title'], data.get('description', ''), 
          data.get('priority', 'medium'), data.get('status', 'pending')))
    conn.commit()
    task_id = cursor.lastrowid
    conn.close()
    
    return jsonify({'id': task_id, 'message': 'Task created successfully'})

if __name__ == '__main__':
    init_db()
    app.run(debug=True)
'''
                            })
                        }
                    }
                ]
            },
            
            # 阶段2：创建前端模板
            {
                "content": "现在创建HTML模板和CSS样式文件。",
                "tool_calls": [
                    {
                        "id": "call_2",
                        "type": "function",
                        "function": {
                            "name": "edit_file",
                            "arguments": json.dumps({
                                "target_file": "templates/index.html",
                                "instructions": "Create main HTML template",
                                "code_edit": '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Task Manager</title>
    <link rel="stylesheet" href="{{ url_for('static', filename='css/style.css') }}">
</head>
<body>
    <div class="container">
        <header>
            <h1>Task Manager</h1>
        </header>
        
        <main>
            <section class="task-form">
                <h2>Add New Task</h2>
                <form id="taskForm">
                    <input type="text" id="taskTitle" placeholder="Task Title" required>
                    <textarea id="taskDescription" placeholder="Task Description"></textarea>
                    <select id="taskPriority">
                        <option value="low">Low Priority</option>
                        <option value="medium" selected>Medium Priority</option>
                        <option value="high">High Priority</option>
                    </select>
                    <button type="submit">Add Task</button>
                </form>
            </section>
            
            <section class="task-list">
                <h2>Tasks</h2>
                <div class="filters">
                    <button class="filter-btn active" data-status="all">All</button>
                    <button class="filter-btn" data-status="pending">Pending</button>
                    <button class="filter-btn" data-status="completed">Completed</button>
                </div>
                <div id="tasksContainer">
                    <!-- Tasks will be loaded here -->
                </div>
            </section>
        </main>
    </div>
    
    <script src="{{ url_for('static', filename='js/app.js') }}"></script>
</body>
</html>
'''
                            })
                        }
                    }
                ]
            },
            
            # 阶段3：创建CSS样式
            {
                "content": "添加CSS样式使界面美观易用。",
                "tool_calls": [
                    {
                        "id": "call_3",
                        "type": "function",
                        "function": {
                            "name": "edit_file",
                            "arguments": json.dumps({
                                "target_file": "static/css/style.css",
                                "instructions": "Create CSS styles",
                                "code_edit": '''
* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

body {
    font-family: 'Arial', sans-serif;
    background-color: #f5f7fa;
    color: #333;
    line-height: 1.6;
}

.container {
    max-width: 800px;
    margin: 0 auto;
    padding: 20px;
}

header {
    text-align: center;
    margin-bottom: 30px;
}

header h1 {
    color: #2c3e50;
    font-size: 2.5rem;
    margin-bottom: 10px;
}

.task-form {
    background: white;
    padding: 25px;
    border-radius: 10px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    margin-bottom: 30px;
}

.task-form h2 {
    margin-bottom: 20px;
    color: #34495e;
}

.task-form input,
.task-form textarea,
.task-form select {
    width: 100%;
    padding: 12px;
    margin-bottom: 15px;
    border: 1px solid #ddd;
    border-radius: 5px;
    font-size: 16px;
}

.task-form button {
    background: #3498db;
    color: white;
    padding: 12px 30px;
    border: none;
    border-radius: 5px;
    cursor: pointer;
    font-size: 16px;
    transition: background 0.3s;
}

.task-form button:hover {
    background: #2980b9;
}

.task-list {
    background: white;
    padding: 25px;
    border-radius: 10px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
}

.filters {
    margin-bottom: 20px;
}

.filter-btn {
    background: #ecf0f1;
    border: none;
    padding: 8px 16px;
    margin-right: 10px;
    border-radius: 5px;
    cursor: pointer;
    transition: background 0.3s;
}

.filter-btn.active,
.filter-btn:hover {
    background: #3498db;
    color: white;
}

.task-item {
    border: 1px solid #eee;
    border-radius: 5px;
    padding: 15px;
    margin-bottom: 10px;
    transition: transform 0.2s;
}

.task-item:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 8px rgba(0,0,0,0.1);
}

.task-title {
    font-weight: bold;
    font-size: 1.1rem;
    margin-bottom: 5px;
}

.task-meta {
    font-size: 0.9rem;
    color: #7f8c8d;
    margin-top: 10px;
}

.priority-high { border-left: 4px solid #e74c3c; }
.priority-medium { border-left: 4px solid #f39c12; }
.priority-low { border-left: 4px solid #27ae60; }
'''
                            })
                        }
                    }
                ]
            },
            
            # 阶段4：创建JavaScript功能
            {
                "content": "添加JavaScript实现动态交互功能。",
                "tool_calls": [
                    {
                        "id": "call_4",
                        "type": "function",
                        "function": {
                            "name": "edit_file",
                            "arguments": json.dumps({
                                "target_file": "static/js/app.js",
                                "instructions": "Create JavaScript functionality",
                                "code_edit": '''
document.addEventListener('DOMContentLoaded', function() {
    const taskForm = document.getElementById('taskForm');
    const tasksContainer = document.getElementById('tasksContainer');
    const filterButtons = document.querySelectorAll('.filter-btn');
    
    let tasks = [];
    let currentFilter = 'all';
    
    // Load tasks on page load
    loadTasks();
    
    // Task form submission
    taskForm.addEventListener('submit', function(e) {
        e.preventDefault();
        
        const title = document.getElementById('taskTitle').value;
        const description = document.getElementById('taskDescription').value;
        const priority = document.getElementById('taskPriority').value;
        
        if (!title.trim()) {
            alert('Please enter a task title');
            return;
        }
        
        createTask({
            title: title,
            description: description,
            priority: priority,
            status: 'pending'
        });
        
        // Clear form
        taskForm.reset();
    });
    
    // Filter buttons
    filterButtons.forEach(button => {
        button.addEventListener('click', function() {
            filterButtons.forEach(btn => btn.classList.remove('active'));
            this.classList.add('active');
            currentFilter = this.dataset.status;
            renderTasks();
        });
    });
    
    // Load tasks from API
    async function loadTasks() {
        try {
            const response = await fetch('/api/tasks');
            tasks = await response.json();
            renderTasks();
        } catch (error) {
            console.error('Error loading tasks:', error);
        }
    }
    
    // Create new task
    async function createTask(taskData) {
        try {
            const response = await fetch('/api/tasks', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(taskData)
            });
            
            if (response.ok) {
                loadTasks(); // Reload tasks
            } else {
                alert('Error creating task');
            }
        } catch (error) {
            console.error('Error creating task:', error);
            alert('Error creating task');
        }
    }
    
    // Render tasks based on current filter
    function renderTasks() {
        const filteredTasks = currentFilter === 'all' 
            ? tasks 
            : tasks.filter(task => task.status === currentFilter);
        
        if (filteredTasks.length === 0) {
            tasksContainer.innerHTML = '<p>No tasks found.</p>';
            return;
        }
        
        tasksContainer.innerHTML = filteredTasks.map(task => `
            <div class="task-item priority-${task.priority}">
                <div class="task-title">${task.title}</div>
                <div class="task-description">${task.description || 'No description'}</div>
                <div class="task-meta">
                    Priority: ${task.priority} | Status: ${task.status} | Created: ${new Date(task.created_at).toLocaleDateString()}
                </div>
            </div>
        `).join('');
    }
});
'''
                            })
                        }
                    }
                ]
            },
            
            # 阶段5：创建测试文件
            {
                "content": "创建单元测试以确保应用质量。",
                "tool_calls": [
                    {
                        "id": "call_5",
                        "type": "function",
                        "function": {
                            "name": "edit_file",
                            "arguments": json.dumps({
                                "target_file": "test_app.py",
                                "instructions": "Create unit tests",
                                "code_edit": '''
import unittest
import json
import tempfile
import os
from app import app, init_db

class TaskManagerTestCase(unittest.TestCase):
    
    def setUp(self):
        """Set up test client and database"""
        self.db_fd, app.config['DATABASE'] = tempfile.mkstemp()
        app.config['TESTING'] = True
        self.client = app.test_client()
        
        with app.app_context():
            init_db()
    
    def tearDown(self):
        """Clean up after tests"""
        os.close(self.db_fd)
        os.unlink(app.config['DATABASE'])
    
    def test_home_page(self):
        """Test home page loads"""
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
    
    def test_get_tasks_empty(self):
        """Test getting tasks when database is empty"""
        response = self.client.get('/api/tasks')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(len(data), 0)
    
    def test_create_task(self):
        """Test creating a new task"""
        task_data = {
            'title': 'Test Task',
            'description': 'This is a test task',
            'priority': 'high',
            'status': 'pending'
        }
        
        response = self.client.post('/api/tasks',
                                   data=json.dumps(task_data),
                                   content_type='application/json')
        
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('id', data)
        self.assertEqual(data['message'], 'Task created successfully')
    
    def test_create_task_minimal(self):
        """Test creating task with minimal data"""
        task_data = {'title': 'Minimal Task'}
        
        response = self.client.post('/api/tasks',
                                   data=json.dumps(task_data),
                                   content_type='application/json')
        
        self.assertEqual(response.status_code, 200)
    
    def test_get_tasks_after_creation(self):
        """Test getting tasks after creating one"""
        # Create a task
        task_data = {'title': 'Test Task', 'priority': 'medium'}
        self.client.post('/api/tasks',
                        data=json.dumps(task_data),
                        content_type='application/json')
        
        # Get tasks
        response = self.client.get('/api/tasks')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(len(data), 1)
        self.assertEqual(data[0]['title'], 'Test Task')

if __name__ == '__main__':
    unittest.main()
'''
                            })
                        }
                    }
                ]
            },
            
            # 阶段6：创建requirements和README
            {
                "content": "创建项目依赖文件和说明文档。",
                "tool_calls": [
                    {
                        "id": "call_6",
                        "type": "function",
                        "function": {
                            "name": "edit_file",
                            "arguments": json.dumps({
                                "target_file": "requirements.txt",
                                "instructions": "Create requirements file",
                                "code_edit": "flask==2.3.3\npytest==7.4.0\n"
                            })
                        }
                    }
                ]
            },
            
            # 阶段7：完成
            {
                "content": "任务管理Web应用开发完成！应用包含了所有要求的功能：任务的增删改查、优先级管理、状态筛选、现代化的Web界面以及完整的单元测试。",
                "tool_calls": []
            }
        ]
        
        # 模拟LLM调用序列
        call_count = 0
        def mock_llm_sequence(*args, **kwargs):
            nonlocal call_count
            if call_count < len(development_phases):
                phase = development_phases[call_count]
                call_count += 1
                return {
                    "choices": [{
                        "message": {
                            "content": phase["content"],
                            "tool_calls": phase["tool_calls"],
                            "role": "assistant"
                        },
                        "finish_reason": "tool_calls" if phase["tool_calls"] else "stop"
                    }],
                    "usage": {"prompt_tokens": 200, "completion_tokens": 300, "total_tokens": 500}
                }
            return development_phases[-1]
        
        # 执行端到端测试
        with patch('tool_executor.ToolExecutor._call_llm_api', side_effect=mock_llm_sequence):
            with patch.object(client, '_execute_tool_call') as mock_tool:
                mock_tool.return_value = {"status": "success", "message": "File created successfully"}
                
                start_time = time.time()
                result = client.chat(
                    messages=[{"role": "user", "content": requirement}],
                    dir=test_workspace,
                    loops=10
                )
                end_time = time.time()
                
                # 验证整体结果
                assert result["success"] == True
                assert result["execution_time"] > 0
                assert "workspace_dir" in result
                
                # 验证开发阶段执行
                assert call_count >= 6, f"Expected at least 6 development phases, got {call_count}"
                assert mock_tool.call_count >= 6, f"Expected multiple tool calls, got {mock_tool.call_count}"
                
                # 验证执行时间合理
                execution_time = end_time - start_time
                assert execution_time < 60, f"E2E test took too long: {execution_time} seconds"

    def test_data_analysis_project_e2e(self, test_workspace):
        """端到端测试：数据分析项目"""
        
        requirement = """
        创建一个数据分析项目：
        1. 读取CSV销售数据
        2. 进行数据清洗和预处理
        3. 生成销售趋势分析
        4. 创建可视化图表
        5. 输出分析报告
        """
        
        # 首先创建测试数据
        test_data = TestHelper.generate_test_data("csv", 50)
        test_data_file = os.path.join(test_workspace, "sales_data.csv")
        with open(test_data_file, 'w') as f:
            f.write("date,sales,product,region\n")
            f.write("2023-01-01,1000,Product A,North\n")
            f.write("2023-01-02,1200,Product B,South\n")
            f.write("2023-01-03,800,Product A,East\n")
            for i in range(47):  # 添加更多数据
                f.write(f"2023-01-{i+4:02d},{1000+i*10},Product {chr(65+i%3)},{['North','South','East','West'][i%4]}\n")
        
        client = AGIBotClient(debug_mode=True)
        
        # 模拟数据分析开发流程
        analysis_phases = [
            # 阶段1：数据探索
            {
                "content": "首先让我查看数据文件结构。",
                "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "arguments": json.dumps({
                            "relative_workspace_path": "sales_data.csv",
                            "start_line_one_indexed": 1,
                            "end_line_one_indexed_inclusive": 10
                        })
                    }
                }]
            },
            
            # 阶段2：创建数据分析脚本
            {
                "content": "基于数据结构，我将创建分析脚本。",
                "tool_calls": [{
                    "id": "call_2",
                    "type": "function",
                    "function": {
                        "name": "edit_file",
                        "arguments": json.dumps({
                            "target_file": "data_analysis.py",
                            "code_edit": '''
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import numpy as np

def load_and_clean_data(file_path):
    """Load and clean the sales data"""
    df = pd.read_csv(file_path)
    
    # Convert date column to datetime
    df['date'] = pd.to_datetime(df['date'])
    
    # Clean any missing values
    df = df.dropna()
    
    # Ensure sales is numeric
    df['sales'] = pd.to_numeric(df['sales'], errors='coerce')
    
    return df

def analyze_sales_trends(df):
    """Analyze sales trends over time"""
    # Group by date and sum sales
    daily_sales = df.groupby('date')['sales'].sum().reset_index()
    
    # Calculate basic statistics
    total_sales = df['sales'].sum()
    avg_daily_sales = daily_sales['sales'].mean()
    max_daily_sales = daily_sales['sales'].max()
    min_daily_sales = daily_sales['sales'].min()
    
    return {
        'total_sales': total_sales,
        'avg_daily_sales': avg_daily_sales,
        'max_daily_sales': max_daily_sales,
        'min_daily_sales': min_daily_sales,
        'daily_sales': daily_sales
    }

def analyze_by_product(df):
    """Analyze sales by product"""
    product_sales = df.groupby('product')['sales'].agg(['sum', 'mean', 'count']).reset_index()
    product_sales.columns = ['product', 'total_sales', 'avg_sales', 'transaction_count']
    return product_sales

def analyze_by_region(df):
    """Analyze sales by region"""
    region_sales = df.groupby('region')['sales'].agg(['sum', 'mean', 'count']).reset_index()
    region_sales.columns = ['region', 'total_sales', 'avg_sales', 'transaction_count']
    return region_sales

def create_visualizations(df, trends, product_analysis, region_analysis):
    """Create visualization charts"""
    plt.style.use('seaborn-v0_8')
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Sales trend over time
    axes[0, 0].plot(trends['daily_sales']['date'], trends['daily_sales']['sales'])
    axes[0, 0].set_title('Daily Sales Trend')
    axes[0, 0].set_xlabel('Date')
    axes[0, 0].set_ylabel('Sales')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # 2. Sales by product
    axes[0, 1].bar(product_analysis['product'], product_analysis['total_sales'])
    axes[0, 1].set_title('Total Sales by Product')
    axes[0, 1].set_xlabel('Product')
    axes[0, 1].set_ylabel('Total Sales')
    
    # 3. Sales by region
    axes[1, 0].pie(region_analysis['total_sales'], labels=region_analysis['region'], autopct='%1.1f%%')
    axes[1, 0].set_title('Sales Distribution by Region')
    
    # 4. Sales distribution histogram
    axes[1, 1].hist(df['sales'], bins=20, alpha=0.7)
    axes[1, 1].set_title('Sales Amount Distribution')
    axes[1, 1].set_xlabel('Sales Amount')
    axes[1, 1].set_ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig('sales_analysis_charts.png', dpi=300, bbox_inches='tight')
    plt.close()

def generate_report(trends, product_analysis, region_analysis):
    """Generate analysis report"""
    report = f"""
# Sales Data Analysis Report

## Executive Summary
- **Total Sales**: ${trends['total_sales']:,.2f}
- **Average Daily Sales**: ${trends['avg_daily_sales']:,.2f}
- **Highest Daily Sales**: ${trends['max_daily_sales']:,.2f}
- **Lowest Daily Sales**: ${trends['min_daily_sales']:,.2f}

## Product Performance
{product_analysis.to_string(index=False)}

## Regional Performance
{region_analysis.to_string(index=False)}

## Key Insights
1. **Top Performing Product**: {product_analysis.loc[product_analysis['total_sales'].idxmax(), 'product']}
2. **Top Performing Region**: {region_analysis.loc[region_analysis['total_sales'].idxmax(), 'region']}
3. **Analysis Period**: {trends['daily_sales']['date'].min().strftime('%Y-%m-%d')} to {trends['daily_sales']['date'].max().strftime('%Y-%m-%d')}

## Recommendations
- Focus marketing efforts on top-performing products and regions
- Investigate opportunities in underperforming areas
- Monitor daily sales trends for seasonal patterns

Report generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
    
    with open('sales_analysis_report.md', 'w') as f:
        f.write(report)
    
    return report

def main():
    """Main analysis function"""
    print("Loading and cleaning data...")
    df = load_and_clean_data('sales_data.csv')
    
    print("Analyzing sales trends...")
    trends = analyze_sales_trends(df)
    
    print("Analyzing by product...")
    product_analysis = analyze_by_product(df)
    
    print("Analyzing by region...")
    region_analysis = analyze_by_region(df)
    
    print("Creating visualizations...")
    create_visualizations(df, trends, product_analysis, region_analysis)
    
    print("Generating report...")
    report = generate_report(trends, product_analysis, region_analysis)
    
    print("Analysis complete!")
    print(f"Total sales: ${trends['total_sales']:,.2f}")
    print("Charts saved as: sales_analysis_charts.png")
    print("Report saved as: sales_analysis_report.md")

if __name__ == "__main__":
    main()
'''
                        })
                    }
                }]
            },
            
            # 阶段3：运行分析
            {
                "content": "运行数据分析脚本。",
                "tool_calls": [{
                    "id": "call_3",
                    "type": "function",
                    "function": {
                        "name": "run_terminal_cmd",
                        "arguments": json.dumps({
                            "command": "python data_analysis.py",
                            "is_background": False
                        })
                    }
                }]
            },
            
            # 阶段4：完成
            {
                "content": "数据分析项目已完成！生成了完整的销售数据分析，包括趋势分析、产品表现、区域分析和可视化图表。",
                "tool_calls": []
            }
        ]
        
        call_count = 0
        def mock_analysis_sequence(*args, **kwargs):
            nonlocal call_count
            if call_count < len(analysis_phases):
                phase = analysis_phases[call_count]
                call_count += 1
                return {
                    "choices": [{
                        "message": {
                            "content": phase["content"],
                            "tool_calls": phase["tool_calls"],
                            "role": "assistant"
                        },
                        "finish_reason": "tool_calls" if phase["tool_calls"] else "stop"
                    }]
                }
            return analysis_phases[-1]
        
        # 模拟工具执行结果
        tool_results = {
            "read_file": "date,sales,product,region\n2023-01-01,1000,Product A,North\n...",
            "edit_file": {"status": "success", "message": "File created"},
            "run_terminal_cmd": {"status": "success", "output": "Analysis complete!\nTotal sales: $52,350.00\nCharts saved as: sales_analysis_charts.png"}
        }
        
        with patch('tool_executor.ToolExecutor._call_llm_api', side_effect=mock_analysis_sequence):
            with patch.object(client, '_execute_tool_call') as mock_tool:
                def tool_side_effect(tool_call):
                    tool_name = tool_call['function']['name']
                    return tool_results.get(tool_name, {"status": "success"})
                
                mock_tool.side_effect = tool_side_effect
                
                result = client.chat(
                    messages=[{"role": "user", "content": requirement}],
                    dir=test_workspace,
                    loops=8
                )
                
                # 验证分析项目完成
                assert result["success"] == True
                assert call_count >= 3, "Expected data analysis phases"
                assert mock_tool.call_count >= 3, "Expected multiple tool executions"

    def test_api_development_e2e(self, test_workspace):
        """端到端测试：API开发项目"""
        
        requirement = """
        开发一个RESTful API服务：
        1. 用户管理API（注册、登录、获取用户信息）
        2. JWT认证机制
        3. 数据验证和错误处理
        4. API文档生成
        5. 单元测试覆盖
        """
        
        client = AGIBotClient(debug_mode=True)
        
        # 执行API开发（简化版，主要测试流程）
        with patch('tool_executor.ToolExecutor._call_llm_api') as mock_llm:
            mock_llm.return_value = {
                "choices": [{
                    "message": {
                        "content": "API服务开发完成！",
                        "tool_calls": [],
                        "role": "assistant"
                    },
                    "finish_reason": "stop"
                }]
            }
            
            with patch.object(client, '_execute_tool_call') as mock_tool:
                mock_tool.return_value = {"status": "success", "message": "Operation completed"}
                
                result = client.chat(
                    messages=[{"role": "user", "content": requirement}],
                    dir=test_workspace,
                    loops=5
                )
                
                assert result["success"] == True
                assert "workspace_dir" in result

    def test_integration_with_external_tools(self, test_workspace):
        """测试与外部工具的集成"""
        
        # 模拟一个需要使用外部工具的项目
        requirement = "创建一个项目并使用git进行版本控制"
        
        client = AGIBotClient(debug_mode=True)
        
        # 模拟使用git等外部工具
        external_tool_sequence = [
            {
                "content": "我将创建项目并初始化git仓库。",
                "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "run_terminal_cmd",
                        "arguments": json.dumps({
                            "command": "git init",
                            "is_background": False
                        })
                    }
                }]
            },
            {
                "content": "创建项目文件。",
                "tool_calls": [{
                    "id": "call_2",
                    "type": "function",
                    "function": {
                        "name": "edit_file",
                        "arguments": json.dumps({
                            "target_file": "README.md",
                            "code_edit": "# My Project\n\nThis is a test project."
                        })
                    }
                }]
            },
            {
                "content": "提交到git。",
                "tool_calls": [{
                    "id": "call_3",
                    "type": "function",
                    "function": {
                        "name": "run_terminal_cmd",
                        "arguments": json.dumps({
                            "command": "git add . && git commit -m 'Initial commit'",
                            "is_background": False
                        })
                    }
                }]
            }
        ]
        
        call_count = 0
        def mock_external_tools(*args, **kwargs):
            nonlocal call_count
            if call_count < len(external_tool_sequence):
                phase = external_tool_sequence[call_count]
                call_count += 1
                return {
                    "choices": [{
                        "message": {
                            "content": phase["content"],
                            "tool_calls": phase["tool_calls"],
                            "role": "assistant"
                        },
                        "finish_reason": "tool_calls" if phase["tool_calls"] else "stop"
                    }]
                }
            return {"choices": [{"message": {"content": "完成", "tool_calls": []}, "finish_reason": "stop"}]}
        
        with patch('tool_executor.ToolExecutor._call_llm_api', side_effect=mock_external_tools):
            with patch.object(client, '_execute_tool_call') as mock_tool:
                mock_tool.return_value = {"status": "success", "output": "Command executed successfully"}
                
                result = client.chat(
                    messages=[{"role": "user", "content": requirement}],
                    dir=test_workspace,
                    loops=5
                )
                
                # 验证外部工具集成
                assert result["success"] == True
                assert mock_tool.call_count >= 3, "Expected multiple external tool calls"

    def test_error_recovery_in_complex_scenario(self, test_workspace):
        """测试复杂场景中的错误恢复"""
        
        requirement = "创建一个会遇到各种错误的复杂项目"
        
        client = AGIBotClient(debug_mode=True)
        
        # 模拟错误和恢复序列
        error_recovery_sequence = [
            # 第1次尝试：失败
            {
                "content": "尝试创建文件。",
                "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "edit_file",
                        "arguments": json.dumps({
                            "target_file": "/invalid/path/file.py",
                            "code_edit": "print('hello')"
                        })
                    }
                }]
            },
            # 第2次尝试：修正路径
            {
                "content": "路径有误，使用正确的路径。",
                "tool_calls": [{
                    "id": "call_2",
                    "type": "function",
                    "function": {
                        "name": "edit_file",
                        "arguments": json.dumps({
                            "target_file": "file.py",
                            "code_edit": "print('hello world')"
                        })
                    }
                }]
            },
            # 第3次：命令执行失败
            {
                "content": "运行文件。",
                "tool_calls": [{
                    "id": "call_3",
                    "type": "function",
                    "function": {
                        "name": "run_terminal_cmd",
                        "arguments": json.dumps({
                            "command": "python nonexistent.py",
                            "is_background": False
                        })
                    }
                }]
            },
            # 第4次：修正命令
            {
                "content": "修正文件名。",
                "tool_calls": [{
                    "id": "call_4",
                    "type": "function",
                    "function": {
                        "name": "run_terminal_cmd",
                        "arguments": json.dumps({
                            "command": "python file.py",
                            "is_background": False
                        })
                    }
                }]
            },
            # 完成
            {"content": "项目创建完成，所有错误已修复！", "tool_calls": []}
        ]
        
        # 对应的工具执行结果
        tool_execution_results = [
            {"status": "error", "message": "Permission denied: invalid path"},
            {"status": "success", "message": "File created successfully"},
            {"status": "error", "message": "File not found: nonexistent.py"},
            {"status": "success", "output": "hello world"},
        ]
        
        call_count = 0
        tool_call_count = 0
        
        def mock_error_sequence(*args, **kwargs):
            nonlocal call_count
            if call_count < len(error_recovery_sequence):
                phase = error_recovery_sequence[call_count]
                call_count += 1
                return {
                    "choices": [{
                        "message": {
                            "content": phase["content"],
                            "tool_calls": phase["tool_calls"],
                            "role": "assistant"
                        },
                        "finish_reason": "tool_calls" if phase["tool_calls"] else "stop"
                    }]
                }
            return error_recovery_sequence[-1]
        
        def mock_tool_results(tool_call):
            nonlocal tool_call_count
            if tool_call_count < len(tool_execution_results):
                result = tool_execution_results[tool_call_count]
                tool_call_count += 1
                return result
            return {"status": "success"}
        
        with patch('tool_executor.ToolExecutor._call_llm_api', side_effect=mock_error_sequence):
            with patch.object(client, '_execute_tool_call', side_effect=mock_tool_results):
                
                result = client.chat(
                    messages=[{"role": "user", "content": requirement}],
                    dir=test_workspace,
                    loops=8
                )
                
                # 验证错误恢复
                assert result["success"] == True
                assert call_count >= 4, "Expected multiple recovery attempts"
                assert tool_call_count >= 4, "Expected multiple tool executions with errors and recoveries"

    def test_performance_large_project(self, test_workspace):
        """测试大型项目的性能表现"""
        
        requirement = """
        创建一个大型项目，包含多个模块：
        1. 创建20个Python文件
        2. 每个文件包含多个函数
        3. 创建相应的测试文件
        4. 生成项目文档
        """
        
        client = AGIBotClient(debug_mode=False)  # 关闭调试模式以提高性能
        
        # 模拟大量文件创建
        def mock_large_project(*args, **kwargs):
            return {
                "choices": [{
                    "message": {
                        "content": "正在创建大型项目的文件...",
                        "tool_calls": [{
                            "id": f"call_{hash(str(args)) % 1000}",
                            "type": "function",
                            "function": {
                                "name": "edit_file",
                                "arguments": json.dumps({
                                    "target_file": f"module_{hash(str(args)) % 20}.py",
                                    "code_edit": "# Module code here\npass"
                                })
                            }
                        }],
                        "role": "assistant"
                    },
                    "finish_reason": "tool_calls"
                }]
            }
        
        start_time = time.time()
        
        with patch('tool_executor.ToolExecutor._call_llm_api', side_effect=mock_large_project):
            with patch.object(client, '_execute_tool_call') as mock_tool:
                mock_tool.return_value = {"status": "success", "message": "File created"}
                
                result = client.chat(
                    messages=[{"role": "user", "content": requirement}],
                    dir=test_workspace,
                    loops=25  # 足够的轮数处理大项目
                )
                
                end_time = time.time()
                execution_time = end_time - start_time
                
                # 验证性能
                assert result["success"] == True
                assert execution_time < 120, f"Large project took too long: {execution_time} seconds"
                assert mock_tool.call_count >= 20, "Expected many file operations for large project" 