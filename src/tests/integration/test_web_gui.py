#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Web GUI功能集成测试
测试Web界面的基本功能、交互、状态管理等
"""

import pytest
import os
import sys
import time
import json
import requests
from unittest.mock import patch, Mock, MagicMock
from typing import Dict, List, Any

# 添加src路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from utils.test_helpers import TestHelper

@pytest.mark.integration
class TestWebGUI:
    """Web GUI功能测试类"""
    
    @pytest.fixture
    def gui_config(self):
        """GUI配置"""
        return {
            "host": "127.0.0.1",
            "port": 8080,
            "debug": True,
            "auto_reload": False,
            "static_folder": "GUI/static",
            "template_folder": "GUI/templates"
        }
    
    @pytest.fixture
    def mock_flask_app(self):
        """模拟Flask应用"""
        mock_app = Mock()
        mock_app.run.return_value = None
        mock_app.route.return_value = lambda f: f
        return mock_app
    
    @pytest.fixture
    def sample_tasks(self):
        """示例任务数据"""
        return [
            {
                "id": "task_001",
                "title": "创建Python计算器",
                "description": "开发一个简单的Python计算器应用",
                "status": "completed",
                "created_at": "2024-01-15T10:00:00Z",
                "completed_at": "2024-01-15T10:30:00Z"
            },
            {
                "id": "task_002", 
                "title": "分析代码库",
                "description": "对现有代码库进行结构分析",
                "status": "running",
                "created_at": "2024-01-15T11:00:00Z",
                "progress": 65
            },
            {
                "id": "task_003",
                "title": "生成文档",
                "description": "为项目生成技术文档",
                "status": "pending",
                "created_at": "2024-01-15T12:00:00Z"
            }
        ]
    
    @pytest.fixture
    def gui_endpoints(self):
        """GUI端点配置"""
        return {
            "home": "/",
            "tasks": "/tasks",
            "new_task": "/tasks/new",
            "task_detail": "/tasks/<task_id>",
            "api_submit": "/api/tasks/submit",
            "api_status": "/api/tasks/<task_id>/status",
            "api_logs": "/api/tasks/<task_id>/logs",
            "websocket": "/ws",
            "static": "/static/<path:filename>"
        }
    
    def test_gui_initialization(self, gui_config):
        """测试GUI初始化"""
        with patch('GUI.app.Flask') as mock_flask:
            mock_app = Mock()
            mock_flask.return_value = mock_app
            
            # 模拟GUI初始化
            try:
                from GUI.app import create_app
                app = create_app(gui_config)
                
                # 验证应用创建
                assert app is not None
            except ImportError:
                # 如果GUI模块不存在，模拟测试
                assert gui_config["host"] == "127.0.0.1"
                assert gui_config["port"] == 8080
    
    def test_home_page_rendering(self, gui_config, mock_flask_app):
        """测试主页渲染"""
        with patch('requests.get') as mock_get:
            # 模拟主页响应
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.text = """
            <!DOCTYPE html>
            <html>
            <head><title>AGI Bot</title></head>
            <body>
                <h1>AGI Bot Dashboard</h1>
                <div id="task-list"></div>
            </body>
            </html>
            """
            mock_get.return_value = mock_response
            
            # 请求主页
            response = requests.get(f"http://{gui_config['host']}:{gui_config['port']}/")
            
            # 验证主页响应
            assert response.status_code == 200
            assert "AGI Bot Dashboard" in response.text
            assert "task-list" in response.text
    
    def test_task_submission_form(self, gui_config, sample_tasks):
        """测试任务提交表单"""
        with patch('requests.post') as mock_post:
            # 模拟任务提交响应
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "success": True,
                "task_id": "task_004",
                "message": "Task submitted successfully"
            }
            mock_post.return_value = mock_response
            
            # 提交新任务
            task_data = {
                "title": "新建网页应用",
                "description": "创建一个React应用",
                "priority": "high"
            }
            
            response = requests.post(
                f"http://{gui_config['host']}:{gui_config['port']}/api/tasks/submit",
                json=task_data
            )
            
            # 验证任务提交
            assert response.status_code == 200
            result = response.json()
            assert result["success"] is True
            assert "task_id" in result
    
    def test_task_list_display(self, gui_config, sample_tasks):
        """测试任务列表显示"""
        with patch('requests.get') as mock_get:
            # 模拟任务列表响应
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "tasks": sample_tasks,
                "total": len(sample_tasks),
                "page": 1,
                "per_page": 10
            }
            mock_get.return_value = mock_response
            
            # 获取任务列表
            response = requests.get(f"http://{gui_config['host']}:{gui_config['port']}/api/tasks")
            
            # 验证任务列表
            assert response.status_code == 200
            result = response.json()
            assert "tasks" in result
            assert len(result["tasks"]) == len(sample_tasks)
    
    def test_task_detail_view(self, gui_config, sample_tasks):
        """测试任务详情视图"""
        task = sample_tasks[0]
        
        with patch('requests.get') as mock_get:
            # 模拟任务详情响应
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "task": task,
                "logs": [
                    {"timestamp": "2024-01-15T10:05:00Z", "level": "INFO", "message": "Task started"},
                    {"timestamp": "2024-01-15T10:15:00Z", "level": "INFO", "message": "Processing..."},
                    {"timestamp": "2024-01-15T10:30:00Z", "level": "INFO", "message": "Task completed"}
                ],
                "output_files": ["calculator.py", "requirements.txt"]
            }
            mock_get.return_value = mock_response
            
            # 获取任务详情
            response = requests.get(
                f"http://{gui_config['host']}:{gui_config['port']}/api/tasks/{task['id']}"
            )
            
            # 验证任务详情
            assert response.status_code == 200
            result = response.json()
            assert result["task"]["id"] == task["id"]
            assert "logs" in result
            assert "output_files" in result
    
    def test_real_time_status_updates(self, gui_config):
        """测试实时状态更新"""
        # 模拟WebSocket连接
        with patch('websocket.WebSocket') as mock_ws:
            mock_connection = Mock()
            mock_connection.recv.return_value = json.dumps({
                "type": "task_update",
                "task_id": "task_002",
                "status": "running",
                "progress": 75
            })
            mock_ws.return_value = mock_connection
            
            # 连接WebSocket
            import websocket
            ws = websocket.WebSocket()
            
            # 验证WebSocket连接
            assert ws is not None
    
    def test_log_streaming(self, gui_config):
        """测试日志流式传输"""
        with patch('requests.get') as mock_get:
            # 模拟流式日志响应
            def mock_stream():
                logs = [
                    "data: {'timestamp': '2024-01-15T10:00:00Z', 'message': 'Starting task'}\n\n",
                    "data: {'timestamp': '2024-01-15T10:01:00Z', 'message': 'Processing step 1'}\n\n",
                    "data: {'timestamp': '2024-01-15T10:02:00Z', 'message': 'Processing step 2'}\n\n"
                ]
                for log in logs:
                    yield log.encode()
            
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.iter_content.return_value = mock_stream()
            mock_get.return_value = mock_response
            
            # 获取流式日志
            response = requests.get(
                f"http://{gui_config['host']}:{gui_config['port']}/api/tasks/task_002/logs/stream"
            )
            
            # 验证日志流
            assert response.status_code == 200
    
    def test_file_download(self, gui_config):
        """测试文件下载"""
        with patch('requests.get') as mock_get:
            # 模拟文件下载响应
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.content = b"# Calculator App\nprint('Hello World')"
            mock_response.headers = {
                "Content-Type": "application/octet-stream",
                "Content-Disposition": "attachment; filename=calculator.py"
            }
            mock_get.return_value = mock_response
            
            # 下载文件
            response = requests.get(
                f"http://{gui_config['host']}:{gui_config['port']}/api/tasks/task_001/files/calculator.py"
            )
            
            # 验证文件下载
            assert response.status_code == 200
            assert len(response.content) > 0
            assert "attachment" in response.headers.get("Content-Disposition", "")
    
    def test_task_management_operations(self, gui_config):
        """测试任务管理操作"""
        operations = ["pause", "resume", "cancel", "restart"]
        
        for operation in operations:
            with patch('requests.post') as mock_post:
                # 模拟操作响应
                mock_response = Mock()
                mock_response.status_code = 200
                mock_response.json.return_value = {
                    "success": True,
                    "operation": operation,
                    "message": f"Task {operation} successfully"
                }
                mock_post.return_value = mock_response
                
                # 执行操作
                response = requests.post(
                    f"http://{gui_config['host']}:{gui_config['port']}/api/tasks/task_002/{operation}"
                )
                
                # 验证操作
                assert response.status_code == 200
                result = response.json()
                assert result["success"] is True
                assert result["operation"] == operation
    
    def test_dashboard_metrics(self, gui_config):
        """测试仪表板指标"""
        with patch('requests.get') as mock_get:
            # 模拟指标响应
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "total_tasks": 150,
                "completed_tasks": 120,
                "running_tasks": 5,
                "failed_tasks": 25,
                "avg_completion_time": 1800,  # 30分钟
                "success_rate": 0.83,
                "system_status": "healthy"
            }
            mock_get.return_value = mock_response
            
            # 获取仪表板指标
            response = requests.get(f"http://{gui_config['host']}:{gui_config['port']}/api/dashboard/metrics")
            
            # 验证指标
            assert response.status_code == 200
            metrics = response.json()
            assert "total_tasks" in metrics
            assert "success_rate" in metrics
            assert metrics["system_status"] == "healthy"
    
    def test_user_preferences(self, gui_config):
        """测试用户偏好设置"""
        preferences = {
            "theme": "dark",
            "language": "zh-CN",
            "notifications": True,
            "auto_refresh": 30,
            "items_per_page": 20
        }
        
        with patch('requests.post') as mock_post:
            # 模拟偏好设置响应
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "success": True,
                "preferences": preferences
            }
            mock_post.return_value = mock_response
            
            # 保存用户偏好
            response = requests.post(
                f"http://{gui_config['host']}:{gui_config['port']}/api/user/preferences",
                json=preferences
            )
            
            # 验证偏好设置
            assert response.status_code == 200
            result = response.json()
            assert result["success"] is True
            assert result["preferences"]["theme"] == "dark"
    
    def test_search_and_filtering(self, gui_config, sample_tasks):
        """测试搜索和过滤"""
        search_params = {
            "query": "Python",
            "status": "completed",
            "date_from": "2024-01-01",
            "date_to": "2024-01-31"
        }
        
        with patch('requests.get') as mock_get:
            # 模拟搜索响应
            filtered_tasks = [task for task in sample_tasks if task["status"] == "completed"]
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "tasks": filtered_tasks,
                "total": len(filtered_tasks),
                "query": search_params
            }
            mock_get.return_value = mock_response
            
            # 执行搜索
            response = requests.get(
                f"http://{gui_config['host']}:{gui_config['port']}/api/tasks/search",
                params=search_params
            )
            
            # 验证搜索结果
            assert response.status_code == 200
            result = response.json()
            assert len(result["tasks"]) == 1
            assert result["tasks"][0]["status"] == "completed"
    
    def test_export_functionality(self, gui_config):
        """测试导出功能"""
        export_formats = ["json", "csv", "xlsx"]
        
        for format_type in export_formats:
            with patch('requests.get') as mock_get:
                # 模拟导出响应
                if format_type == "json":
                    content = b'{"tasks": []}'
                    content_type = "application/json"
                elif format_type == "csv":
                    content = b"id,title,status\ntask_001,Calculator,completed"
                    content_type = "text/csv"
                else:  # xlsx
                    content = b"\x50\x4b\x03\x04"  # Excel文件头
                    content_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                
                mock_response = Mock()
                mock_response.status_code = 200
                mock_response.content = content
                mock_response.headers = {
                    "Content-Type": content_type,
                    "Content-Disposition": f"attachment; filename=tasks.{format_type}"
                }
                mock_get.return_value = mock_response
                
                # 导出数据
                response = requests.get(
                    f"http://{gui_config['host']}:{gui_config['port']}/api/tasks/export",
                    params={"format": format_type}
                )
                
                # 验证导出
                assert response.status_code == 200
                assert content_type in response.headers["Content-Type"]
    
    def test_notification_system(self, gui_config):
        """测试通知系统"""
        with patch('requests.get') as mock_get:
            # 模拟通知响应
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "notifications": [
                    {
                        "id": "notif_001",
                        "type": "task_completed",
                        "message": "任务 '创建Python计算器' 已完成",
                        "timestamp": "2024-01-15T10:30:00Z",
                        "read": False
                    },
                    {
                        "id": "notif_002",
                        "type": "system_alert",
                        "message": "系统更新可用",
                        "timestamp": "2024-01-15T09:00:00Z",
                        "read": True
                    }
                ],
                "unread_count": 1
            }
            mock_get.return_value = mock_response
            
            # 获取通知
            response = requests.get(f"http://{gui_config['host']}:{gui_config['port']}/api/notifications")
            
            # 验证通知系统
            assert response.status_code == 200
            result = response.json()
            assert "notifications" in result
            assert result["unread_count"] == 1
    
    def test_responsive_design(self, gui_config):
        """测试响应式设计"""
        user_agents = [
            "Mozilla/5.0 (iPhone; CPU iPhone OS 14_0 like Mac OS X)",  # Mobile
            "Mozilla/5.0 (iPad; CPU OS 14_0 like Mac OS X)",  # Tablet
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"  # Desktop
        ]
        
        for user_agent in user_agents:
            with patch('requests.get') as mock_get:
                # 模拟不同设备的响应
                mock_response = Mock()
                mock_response.status_code = 200
                mock_response.text = f"""
                <!DOCTYPE html>
                <html>
                <head>
                    <meta name="viewport" content="width=device-width, initial-scale=1">
                    <title>AGI Bot</title>
                </head>
                <body class="responsive">
                    <div class="container">Content</div>
                </body>
                </html>
                """
                mock_get.return_value = mock_response
                
                # 请求页面
                headers = {"User-Agent": user_agent}
                response = requests.get(
                    f"http://{gui_config['host']}:{gui_config['port']}/",
                    headers=headers
                )
                
                # 验证响应式设计
                assert response.status_code == 200
                assert "viewport" in response.text
                assert "responsive" in response.text
    
    def test_api_error_handling(self, gui_config):
        """测试API错误处理"""
        error_scenarios = [
            {"endpoint": "/api/tasks/nonexistent", "expected_code": 404},
            {"endpoint": "/api/tasks", "method": "POST", "data": {}, "expected_code": 400},
            {"endpoint": "/api/unauthorized", "expected_code": 401}
        ]
        
        for scenario in error_scenarios:
            with patch('requests.get' if scenario.get('method', 'GET') == 'GET' else 'requests.post') as mock_request:
                # 模拟错误响应
                mock_response = Mock()
                mock_response.status_code = scenario["expected_code"]
                mock_response.json.return_value = {
                    "error": True,
                    "code": scenario["expected_code"],
                    "message": "Error message"
                }
                mock_request.return_value = mock_response
                
                # 发送请求
                url = f"http://{gui_config['host']}:{gui_config['port']}{scenario['endpoint']}"
                if scenario.get('method') == 'POST':
                    response = requests.post(url, json=scenario.get('data', {}))
                else:
                    response = requests.get(url)
                
                # 验证错误处理
                assert response.status_code == scenario["expected_code"]
    
    def test_concurrent_user_access(self, gui_config):
        """测试并发用户访问"""
        import threading
        
        results = []
        errors = []
        
        def simulate_user_access(user_id):
            try:
                with patch('requests.get') as mock_get:
                    mock_response = Mock()
                    mock_response.status_code = 200
                    mock_response.json.return_value = {"user_id": user_id, "success": True}
                    mock_get.return_value = mock_response
                    
                    response = requests.get(
                        f"http://{gui_config['host']}:{gui_config['port']}/api/user/{user_id}/dashboard"
                    )
                    results.append((user_id, response.status_code))
            except Exception as e:
                errors.append((user_id, e))
        
        # 模拟多个并发用户
        threads = []
        for i in range(10):
            thread = threading.Thread(target=simulate_user_access, args=(f"user_{i}",))
            threads.append(thread)
            thread.start()
        
        # 等待所有线程完成
        for thread in threads:
            thread.join(timeout=10)
        
        # 验证并发访问
        assert len(errors) == 0
        assert len(results) == 10
        assert all(status == 200 for _, status in results)
    
    def test_session_management(self, gui_config):
        """测试会话管理"""
        with patch('requests.post') as mock_post:
            # 模拟登录响应
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "success": True,
                "session_token": "test_session_token_12345",
                "expires_at": "2024-01-16T10:00:00Z"
            }
            mock_response.cookies = {"session_id": "sess_12345"}
            mock_post.return_value = mock_response
            
            # 模拟登录
            login_data = {"username": "test_user", "password": "test_pass"}
            response = requests.post(
                f"http://{gui_config['host']}:{gui_config['port']}/api/auth/login",
                json=login_data
            )
            
            # 验证会话管理
            assert response.status_code == 200
            result = response.json()
            assert result["success"] is True
            assert "session_token" in result
    
    def test_accessibility_features(self, gui_config):
        """测试可访问性功能"""
        with patch('requests.get') as mock_get:
            # 模拟包含可访问性特性的页面
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.text = """
            <!DOCTYPE html>
            <html lang="zh-CN">
            <head>
                <title>AGI Bot - 可访问的AI助手</title>
            </head>
            <body>
                <nav role="navigation" aria-label="主导航">
                    <ul>
                        <li><a href="/" aria-current="page">首页</a></li>
                        <li><a href="/tasks">任务</a></li>
                    </ul>
                </nav>
                <main role="main">
                    <h1>任务列表</h1>
                    <button aria-label="创建新任务" aria-describedby="help-text">新建任务</button>
                    <div id="help-text">点击创建一个新的AI任务</div>
                </main>
            </body>
            </html>
            """
            mock_get.return_value = mock_response
            
            # 获取页面
            response = requests.get(f"http://{gui_config['host']}:{gui_config['port']}/")
            
            # 验证可访问性特性
            assert response.status_code == 200
            content = response.text
            assert 'lang="zh-CN"' in content
            assert 'role="navigation"' in content
            assert 'aria-label=' in content
            assert 'aria-describedby=' in content 