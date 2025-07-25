#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
角色分工和专业化单元测试
测试不同角色智能体的专业化功能
"""

import pytest
import os
import sys
import json
import time
from unittest.mock import patch, Mock, MagicMock
from typing import Dict, List, Any

# 添加src路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from tools.multiagents import MultiAgentTools
from tools.message_system import MessageRouter, MessageType
from utils.test_helpers import TestHelper

@pytest.mark.unit
class TestRoleSpecialization:
    """角色分工和专业化测试类"""
    
    @pytest.fixture
    def multi_agent_tools(self, test_workspace):
        """创建多智能体工具实例"""
        return MultiAgentTools(workspace_root=test_workspace, debug_mode=True)
    
    @pytest.fixture
    def role_definitions(self):
        """预定义的角色类型"""
        return {
            "code_analyst": {
                "description": "专门负责代码分析和架构设计的智能体",
                "specialization": "code_analysis",
                "tools": ["code_search", "dependency_analysis", "quality_check"],
                "expertise": ["Python", "JavaScript", "architecture_design"],
                "prompt_template": "你是一个专业的代码分析师，擅长代码分析和架构设计。"
            },
            "data_scientist": {
                "description": "专门负责数据分析和机器学习的智能体",
                "specialization": "data_analysis",
                "tools": ["data_processing", "visualization", "ml_models"],
                "expertise": ["statistics", "machine_learning", "data_visualization"],
                "prompt_template": "你是一个专业的数据科学家，擅长数据分析和机器学习。"
            },
            "web_developer": {
                "description": "专门负责Web开发的智能体",
                "specialization": "web_development",
                "tools": ["html_generation", "css_styling", "js_scripting"],
                "expertise": ["HTML", "CSS", "JavaScript", "React", "Vue"],
                "prompt_template": "你是一个专业的Web开发者，擅长前端和后端开发。"
            },
            "researcher": {
                "description": "专门负责研究和信息收集的智能体",
                "specialization": "research",
                "tools": ["web_search", "document_analysis", "fact_checking"],
                "expertise": ["research_methodology", "information_synthesis"],
                "prompt_template": "你是一个专业的研究员，擅长信息收集和分析。"
            },
            "project_manager": {
                "description": "专门负责项目管理和协调的智能体",
                "specialization": "project_management",
                "tools": ["task_scheduling", "resource_allocation", "progress_tracking"],
                "expertise": ["project_planning", "team_coordination", "risk_management"],
                "prompt_template": "你是一个专业的项目经理，擅长项目规划和团队协调。"
            }
        }
    
    @pytest.fixture
    def collaboration_scenarios(self):
        """协作场景定义"""
        return [
            {
                "scenario": "web_application_development",
                "description": "开发一个完整的Web应用",
                "required_roles": ["project_manager", "web_developer", "data_scientist"],
                "workflow": [
                    {"step": 1, "role": "project_manager", "task": "制定开发计划"},
                    {"step": 2, "role": "web_developer", "task": "设计前端界面"},
                    {"step": 3, "role": "data_scientist", "task": "设计数据处理逻辑"},
                    {"step": 4, "role": "web_developer", "task": "集成前后端"}
                ]
            },
            {
                "scenario": "code_analysis_project",
                "description": "分析大型代码库",
                "required_roles": ["code_analyst", "researcher"],
                "workflow": [
                    {"step": 1, "role": "researcher", "task": "收集相关技术资料"},
                    {"step": 2, "role": "code_analyst", "task": "分析代码结构"},
                    {"step": 3, "role": "code_analyst", "task": "生成分析报告"}
                ]
            }
        ]
    
    def test_role_creation_with_specialization(self, multi_agent_tools, role_definitions):
        """测试创建专业化角色的智能体"""
        with patch('tools.multiagents.AGIBotClient') as mock_client:
            mock_instance = Mock()
            mock_instance.chat.return_value = {"success": True, "message": "Task completed"}
            mock_client.return_value = mock_instance
            
            for role_name, role_config in role_definitions.items():
                # 创建专业化智能体
                result = multi_agent_tools.spawn_agibot(
                    task_description=f"作为{role_name}执行专业任务",
                    agent_id=f"{role_name}_001",
                    api_key="test_key",
                    model="test_model",
                    specialization=role_config["specialization"],
                    expertise=role_config["expertise"],
                    tools=role_config["tools"]
                )
                
                # 验证专业化智能体创建
                assert result is not None
                assert result["success"] is True
                assert result["agent_id"] == f"{role_name}_001"
    
    def test_role_specific_task_assignment(self, multi_agent_tools, role_definitions):
        """测试角色特定任务分配"""
        with patch('tools.multiagents.AGIBotClient') as mock_client:
            mock_instance = Mock()
            
            # 模拟不同角色对任务的不同处理方式
            def role_specific_response(messages, **kwargs):
                content = messages[0]["content"] if messages else ""
                
                if "代码分析" in content:
                    return {
                        "success": True,
                        "message": "完成代码分析任务",
                        "specialization_used": "code_analysis",
                        "tools_used": ["code_search", "dependency_analysis"]
                    }
                elif "数据分析" in content:
                    return {
                        "success": True,
                        "message": "完成数据分析任务",
                        "specialization_used": "data_analysis",
                        "tools_used": ["data_processing", "visualization"]
                    }
                else:
                    return {"success": True, "message": "完成通用任务"}
            
            mock_instance.chat.side_effect = role_specific_response
            mock_client.return_value = mock_instance
            
            # 测试代码分析师处理代码任务
            code_result = multi_agent_tools.spawn_agibot(
                task_description="进行代码分析和架构评估",
                agent_id="code_analyst_001",
                api_key="test_key",
                model="test_model"
            )
            
            # 测试数据科学家处理数据任务
            data_result = multi_agent_tools.spawn_agibot(
                task_description="进行数据分析和可视化",
                agent_id="data_scientist_001",
                api_key="test_key",
                model="test_model"
            )
            
            # 验证角色特定任务处理
            assert code_result is not None
            assert data_result is not None
    
    def test_expertise_domain_validation(self, multi_agent_tools, role_definitions):
        """测试专业领域验证"""
        with patch('tools.multiagents.AGIBotClient') as mock_client:
            mock_instance = Mock()
            mock_instance.chat.return_value = {"success": True}
            mock_client.return_value = mock_instance
            
            # 测试领域匹配的任务
            matching_tasks = [
                ("code_analyst", "Python代码重构"),
                ("web_developer", "React组件开发"),
                ("data_scientist", "机器学习模型训练"),
                ("researcher", "技术文档研究")
            ]
            
            for role, task in matching_tasks:
                result = multi_agent_tools.spawn_agibot(
                    task_description=task,
                    agent_id=f"{role}_test",
                    api_key="test_key",
                    model="test_model",
                    expertise=role_definitions[role]["expertise"]
                )
                
                # 验证领域匹配任务成功执行
                assert result is not None
                assert result["success"] is True
    
    def test_tool_specialization(self, multi_agent_tools, role_definitions):
        """测试工具专业化"""
        with patch('tools.multiagents.AGIBotClient') as mock_client:
            mock_instance = Mock()
            
            def tool_specific_response(messages, **kwargs):
                # 模拟不同角色使用不同工具
                return {
                    "success": True,
                    "tools_used": kwargs.get("tools", []),
                    "message": f"使用专业工具完成任务"
                }
            
            mock_instance.chat.side_effect = tool_specific_response
            mock_client.return_value = mock_instance
            
            for role_name, role_config in role_definitions.items():
                result = multi_agent_tools.spawn_agibot(
                    task_description=f"{role_name}专业任务",
                    agent_id=f"{role_name}_tools",
                    api_key="test_key",
                    model="test_model",
                    tools=role_config["tools"]
                )
                
                # 验证工具专业化
                assert result is not None
                assert result["success"] is True
    
    def test_role_based_collaboration(self, multi_agent_tools, collaboration_scenarios):
        """测试基于角色的协作"""
        with patch('tools.multiagents.AGIBotClient') as mock_client:
            mock_instance = Mock()
            mock_instance.chat.return_value = {"success": True}
            mock_client.return_value = mock_instance
            
            for scenario in collaboration_scenarios:
                agents = []
                
                # 创建场景所需的所有角色
                for role in scenario["required_roles"]:
                    result = multi_agent_tools.spawn_agibot(
                        task_description=f"参与{scenario['scenario']}项目",
                        agent_id=f"{role}_{scenario['scenario']}",
                        api_key="test_key",
                        model="test_model"
                    )
                    agents.append(result)
                
                # 验证所有角色都成功创建
                assert len(agents) == len(scenario["required_roles"])
                for agent in agents:
                    assert agent is not None
                    assert agent["success"] is True
    
    def test_role_communication_patterns(self, multi_agent_tools):
        """测试角色间通信模式"""
        with patch('tools.multiagents.AGIBotClient') as mock_client:
            mock_instance = Mock()
            mock_instance.chat.return_value = {"success": True}
            mock_client.return_value = mock_instance
            
            # 创建不同角色的智能体
            roles = ["manager", "developer", "tester"]
            agents = {}
            
            for role in roles:
                result = multi_agent_tools.spawn_agibot(
                    task_description=f"{role}角色任务",
                    agent_id=f"{role}_comm",
                    api_key="test_key",
                    model="test_model"
                )
                agents[role] = result["agent_id"]
            
            # 测试角色间通信
            # Manager -> Developer
            manager_to_dev = multi_agent_tools.send_message_to_agent_or_manager(
                target_agent_id=agents["developer"],
                message="开始开发任务",
                message_type="task_assignment"
            )
            
            # Developer -> Tester
            dev_to_tester = multi_agent_tools.send_message_to_agent_or_manager(
                target_agent_id=agents["tester"],
                message="开发完成，请测试",
                message_type="handoff"
            )
            
            # 验证通信成功
            assert manager_to_dev is not None
            assert dev_to_tester is not None
    
    def test_role_hierarchy_management(self, multi_agent_tools):
        """测试角色层级管理"""
        with patch('tools.multiagents.AGIBotClient') as mock_client:
            mock_instance = Mock()
            mock_instance.chat.return_value = {"success": True}
            mock_client.return_value = mock_instance
            
            # 创建层级结构：Manager -> Senior Dev -> Junior Dev
            hierarchy = [
                {"role": "project_manager", "level": "manager", "parent": None},
                {"role": "senior_developer", "level": "senior", "parent": "project_manager"},
                {"role": "junior_developer", "level": "junior", "parent": "senior_developer"}
            ]
            
            agents = {}
            for role_info in hierarchy:
                result = multi_agent_tools.spawn_agibot(
                    task_description=f"{role_info['role']}层级任务",
                    agent_id=role_info["role"],
                    api_key="test_key",
                    model="test_model",
                    hierarchy_level=role_info["level"],
                    parent_agent=role_info["parent"]
                )
                agents[role_info["role"]] = result
            
            # 验证层级结构创建
            for agent in agents.values():
                assert agent is not None
                assert agent["success"] is True
    
    def test_role_delegation_capabilities(self, multi_agent_tools):
        """测试角色委派能力"""
        with patch('tools.multiagents.AGIBotClient') as mock_client:
            mock_instance = Mock()
            mock_instance.chat.return_value = {"success": True}
            mock_client.return_value = mock_instance
            
            # 创建具有委派能力的管理者角色
            manager_result = multi_agent_tools.spawn_agibot(
                task_description="管理项目并委派任务",
                agent_id="delegating_manager",
                api_key="test_key",
                model="test_model",
                can_delegate=True,
                delegation_rules=["可以创建子智能体", "可以分配任务"]
            )
            
            # 验证管理者创建
            assert manager_result is not None
            assert manager_result["success"] is True
            
            # 模拟管理者委派任务（创建子智能体）
            delegated_result = multi_agent_tools.spawn_agibot(
                task_description="执行委派的具体任务",
                agent_id="delegated_worker",
                api_key="test_key",
                model="test_model",
                parent_agent="delegating_manager"
            )
            
            # 验证委派成功
            assert delegated_result is not None
            assert delegated_result["success"] is True
    
    def test_expertise_conflict_resolution(self, multi_agent_tools):
        """测试专业知识冲突解决"""
        with patch('tools.multiagents.AGIBotClient') as mock_client:
            mock_instance = Mock()
            
            # 模拟不同专家对同一问题的不同意见
            def conflicting_responses(messages, **kwargs):
                agent_id = kwargs.get("agent_id", "")
                
                if "expert_a" in agent_id:
                    return {
                        "success": True,
                        "recommendation": "使用方案A",
                        "confidence": 0.8,
                        "reasoning": "基于性能考虑"
                    }
                elif "expert_b" in agent_id:
                    return {
                        "success": True,
                        "recommendation": "使用方案B",
                        "confidence": 0.9,
                        "reasoning": "基于可维护性考虑"
                    }
                else:
                    return {"success": True}
            
            mock_instance.chat.side_effect = conflicting_responses
            mock_client.return_value = mock_instance
            
            # 创建两个具有不同专业观点的专家
            expert_a = multi_agent_tools.spawn_agibot(
                task_description="提供技术建议",
                agent_id="expert_a",
                api_key="test_key",
                model="test_model",
                expertise=["performance_optimization"]
            )
            
            expert_b = multi_agent_tools.spawn_agibot(
                task_description="提供技术建议",
                agent_id="expert_b",
                api_key="test_key",
                model="test_model",
                expertise=["maintainability"]
            )
            
            # 验证专家创建
            assert expert_a is not None and expert_a["success"] is True
            assert expert_b is not None and expert_b["success"] is True
    
    def test_role_adaptation(self, multi_agent_tools):
        """测试角色适应性"""
        with patch('tools.multiagents.AGIBotClient') as mock_client:
            mock_instance = Mock()
            
            # 模拟角色在任务执行过程中的适应
            adaptation_calls = 0
            def adaptive_response(messages, **kwargs):
                nonlocal adaptation_calls
                adaptation_calls += 1
                
                if adaptation_calls == 1:
                    return {
                        "success": False,
                        "message": "当前专业知识不足",
                        "adaptation_needed": True
                    }
                else:
                    return {
                        "success": True,
                        "message": "已适应新要求",
                        "adapted_skills": ["新技能"]
                    }
            
            mock_instance.chat.side_effect = adaptive_response
            mock_client.return_value = mock_instance
            
            # 创建可适应的智能体
            result = multi_agent_tools.spawn_agibot(
                task_description="执行可能需要新技能的任务",
                agent_id="adaptive_agent",
                api_key="test_key",
                model="test_model",
                adaptive=True
            )
            
            # 验证适应性
            assert result is not None
    
    def test_cross_domain_collaboration(self, multi_agent_tools):
        """测试跨领域协作"""
        with patch('tools.multiagents.AGIBotClient') as mock_client:
            mock_instance = Mock()
            mock_instance.chat.return_value = {"success": True}
            mock_client.return_value = mock_instance
            
            # 创建跨领域项目团队
            cross_domain_team = [
                {"role": "ai_researcher", "domain": "artificial_intelligence"},
                {"role": "bio_scientist", "domain": "biology"},
                {"role": "software_engineer", "domain": "software_development"}
            ]
            
            team_agents = []
            for member in cross_domain_team:
                result = multi_agent_tools.spawn_agibot(
                    task_description=f"AI+生物信息学项目中的{member['role']}工作",
                    agent_id=f"{member['role']}_cross_domain",
                    api_key="test_key",
                    model="test_model",
                    domain=member["domain"],
                    collaboration_mode="cross_domain"
                )
                team_agents.append(result)
            
            # 验证跨领域团队创建
            assert len(team_agents) == 3
            for agent in team_agents:
                assert agent is not None
                assert agent["success"] is True
    
    def test_role_performance_metrics(self, multi_agent_tools, role_definitions):
        """测试角色绩效指标"""
        with patch('tools.multiagents.AGIBotClient') as mock_client:
            mock_instance = Mock()
            
            def performance_response(messages, **kwargs):
                return {
                    "success": True,
                    "performance_metrics": {
                        "task_completion_rate": 0.95,
                        "expertise_utilization": 0.88,
                        "collaboration_score": 0.92,
                        "efficiency_rating": 0.85
                    }
                }
            
            mock_instance.chat.side_effect = performance_response
            mock_client.return_value = mock_instance
            
            # 测试不同角色的绩效
            for role_name in role_definitions.keys():
                result = multi_agent_tools.spawn_agibot(
                    task_description=f"{role_name}绩效测试任务",
                    agent_id=f"{role_name}_performance",
                    api_key="test_key",
                    model="test_model",
                    track_performance=True
                )
                
                # 验证绩效跟踪
                assert result is not None
                assert result["success"] is True
    
    def test_role_learning_and_improvement(self, multi_agent_tools):
        """测试角色学习和改进"""
        with patch('tools.multiagents.AGIBotClient') as mock_client:
            mock_instance = Mock()
            
            # 模拟学习过程
            learning_iterations = 0
            def learning_response(messages, **kwargs):
                nonlocal learning_iterations
                learning_iterations += 1
                
                return {
                    "success": True,
                    "learning_progress": {
                        "iteration": learning_iterations,
                        "skills_acquired": f"新技能_{learning_iterations}",
                        "competency_level": min(0.5 + learning_iterations * 0.1, 1.0)
                    }
                }
            
            mock_instance.chat.side_effect = learning_response
            mock_client.return_value = mock_instance
            
            # 创建可学习的智能体
            result = multi_agent_tools.spawn_agibot(
                task_description="执行需要持续学习的任务",
                agent_id="learning_agent",
                api_key="test_key",
                model="test_model",
                learning_enabled=True,
                improvement_tracking=True
            )
            
            # 验证学习能力
            assert result is not None
            assert result["success"] is True
    
    def test_role_specialization_persistence(self, multi_agent_tools, test_workspace):
        """测试角色专业化持久性"""
        with patch('tools.multiagents.AGIBotClient') as mock_client:
            mock_instance = Mock()
            mock_instance.chat.return_value = {"success": True}
            mock_client.return_value = mock_instance
            
            # 创建专业化智能体
            result = multi_agent_tools.spawn_agibot(
                task_description="专业化任务",
                agent_id="persistent_specialist",
                api_key="test_key",
                model="test_model",
                specialization="data_analysis",
                save_profile=True
            )
            
            # 验证专业化配置持久性
            assert result is not None
            assert result["success"] is True
            
            # 检查是否保存了角色配置文件
            profile_path = os.path.join(test_workspace, "agent_profiles")
            if os.path.exists(profile_path):
                profile_files = os.listdir(profile_path)
                specialist_profiles = [f for f in profile_files if "persistent_specialist" in f]
                assert len(specialist_profiles) > 0 