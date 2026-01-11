#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright (c) 2025 AGI Agent Research Group.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import os
import json
import shutil
from typing import Optional, Dict, List
from pathlib import Path


class AppManager:
    """应用管理器，负责加载应用配置和处理用户shared目录"""
    
    def __init__(self, app_name: Optional[str] = None, base_dir: Optional[str] = None):
        """
        初始化应用管理器
        
        Args:
            app_name: 应用名称（如 'patent'），如果为None则使用默认配置
            base_dir: 项目根目录，如果为None则自动检测
        """
        if base_dir is None:
            # 自动检测项目根目录（从GUI目录向上两级）
            current_file = os.path.abspath(__file__)
            gui_dir = os.path.dirname(current_file)
            base_dir = os.path.dirname(gui_dir)
        
        self.base_dir = base_dir
        self.app_name = app_name
        self.app_config = None
        self.app_dir = None
        
        # 如果指定了应用名称，加载应用配置
        if app_name:
            self._load_app_config()
    
    def _load_app_config(self) -> bool:
        """加载应用配置"""
        if not self.app_name:
            return False
        
        app_json_path = os.path.join(self.base_dir, 'apps', self.app_name, 'app.json')
        
        if not os.path.exists(app_json_path):
            print(f"⚠️ Warning: App config not found: {app_json_path}")
            return False
        
        try:
            with open(app_json_path, 'r', encoding='utf-8') as f:
                self.app_config = json.load(f)
            
            self.app_dir = os.path.join(self.base_dir, 'apps', self.app_name)
            return True
        except Exception as e:
            print(f"⚠️ Error loading app config: {e}")
            return False
    
    def get_app_name(self) -> str:
        """获取应用显示名称"""
        if self.app_config:
            # 如果 app_title 存在（即使是空字符串），优先使用它
            if 'app_title' in self.app_config:
                return self.app_config['app_title']
            # 如果 app_title 不存在，使用 app_name
            if 'app_name' in self.app_config:
                return self.app_config['app_name']
        return "AGI Agent"  # 默认名称
    
    def get_prompts_folder(self, user_dir: Optional[str] = None) -> Optional[str]:
        """
        获取提示词目录路径
        
        Args:
            user_dir: 用户目录路径，用于检查shared目录
        
        Returns:
            提示词目录的绝对路径，如果不存在则返回None
        """
        if not self.app_config:
            # 使用默认路径
            default_path = os.path.join(self.base_dir, 'prompts')
            return default_path if os.path.exists(default_path) else None
        
        prompts_path = self.app_config.get('prompts_path', 'prompts')
        app_prompts_dir = os.path.join(self.app_dir, prompts_path)
        
        # 检查用户shared目录
        if user_dir:
            shared_prompts_dir = os.path.join(user_dir, 'shared', prompts_path)
            if os.path.exists(shared_prompts_dir):
                return os.path.abspath(shared_prompts_dir)
        
        # 使用应用目录
        if os.path.exists(app_prompts_dir):
            return os.path.abspath(app_prompts_dir)
        
        return None
    
    def get_routine_path(self, user_dir: Optional[str] = None) -> Optional[str]:
        """
        获取routine目录路径
        
        Args:
            user_dir: 用户目录路径，用于检查shared目录
        
        Returns:
            routine目录的绝对路径，如果不存在则返回None
        """
        if not self.app_config:
            # 使用默认路径
            default_path = os.path.join(self.base_dir, 'routine')
            return default_path if os.path.exists(default_path) else None
        
        routine_path = self.app_config.get('routine_path', 'routine')
        app_routine_dir = os.path.join(self.app_dir, routine_path)
        
        # 检查用户shared目录
        if user_dir:
            shared_routine_dir = os.path.join(user_dir, 'shared', routine_path)
            if os.path.exists(shared_routine_dir):
                return os.path.abspath(shared_routine_dir)
        
        # 使用应用目录
        if os.path.exists(app_routine_dir):
            return os.path.abspath(app_routine_dir)
        
        return None
    
    def get_logo_path(self, user_dir: Optional[str] = None) -> Optional[str]:
        """
        获取logo文件路径
        
        Args:
            user_dir: 用户目录路径，用于检查shared目录
        
        Returns:
            logo文件的绝对路径，如果不存在则返回None
        """
        if not self.app_config:
            return None
        
        logo_path = self.app_config.get('logo_path', 'logo.png')
        
        # 检查用户shared目录
        if user_dir:
            shared_logo = os.path.join(user_dir, 'shared', logo_path)
            if os.path.exists(shared_logo):
                return os.path.abspath(shared_logo)
        
        # 使用应用目录
        app_logo = os.path.join(self.app_dir, logo_path)
        if os.path.exists(app_logo):
            return os.path.abspath(app_logo)
        
        return None
    
    def get_config_path(self, user_dir: Optional[str] = None) -> Optional[str]:
        """
        获取配置文件路径
        
        Args:
            user_dir: 用户目录路径，用于检查shared目录
        
        Returns:
            配置文件的绝对路径，如果不存在则返回None
        """
        if not self.app_config:
            # 使用默认路径
            default_path = os.path.join(self.base_dir, 'config', 'config.txt')
            return default_path if os.path.exists(default_path) else None
        
        config_path = self.app_config.get('config_path', 'config.txt')
        
        # 检查用户shared目录
        if user_dir:
            shared_config = os.path.join(user_dir, 'shared', config_path)
            if os.path.exists(shared_config):
                return os.path.abspath(shared_config)
        
        # 使用应用目录
        app_config = os.path.join(self.app_dir, config_path)
        if os.path.exists(app_config):
            return os.path.abspath(app_config)
        
        return None
    
    def copy_app_to_shared(self, user_dir: str) -> bool:
        """
        将应用配置拷贝到用户的shared目录
        
        Args:
            user_dir: 用户目录路径
        
        Returns:
            是否成功
        """
        if not self.app_config or not self.app_dir:
            return False
        
        shared_dir = os.path.join(user_dir, 'shared')
        os.makedirs(shared_dir, exist_ok=True)
        
        try:
            # 拷贝prompts目录
            prompts_path = self.app_config.get('prompts_path', 'prompts')
            app_prompts = os.path.join(self.app_dir, prompts_path)
            if os.path.exists(app_prompts):
                shared_prompts = os.path.join(shared_dir, prompts_path)
                if os.path.exists(shared_prompts):
                    shutil.rmtree(shared_prompts)
                shutil.copytree(app_prompts, shared_prompts)
            
            # 拷贝routine目录
            routine_path = self.app_config.get('routine_path', 'routine')
            app_routine = os.path.join(self.app_dir, routine_path)
            if os.path.exists(app_routine):
                shared_routine = os.path.join(shared_dir, routine_path)
                if os.path.exists(shared_routine):
                    shutil.rmtree(shared_routine)
                shutil.copytree(app_routine, shared_routine)
            
            # 拷贝logo文件
            logo_path = self.app_config.get('logo_path', 'logo.png')
            app_logo = os.path.join(self.app_dir, logo_path)
            if os.path.exists(app_logo):
                shared_logo = os.path.join(shared_dir, logo_path)
                shutil.copy2(app_logo, shared_logo)
            
            # 拷贝config文件
            config_path = self.app_config.get('config_path', 'config.txt')
            app_config = os.path.join(self.app_dir, config_path)
            if os.path.exists(app_config):
                shared_config = os.path.join(shared_dir, config_path)
                shutil.copy2(app_config, shared_config)
            
            return True
        except Exception as e:
            print(f"⚠️ Error copying app to shared directory: {e}")
            return False
    
    def list_available_apps(self) -> List[Dict[str, str]]:
        """
        列出所有可用的应用
        
        Returns:
            应用列表，每个应用包含 name 和 display_name
        """
        apps = []
        apps_dir = os.path.join(self.base_dir, 'apps')
        
        if not os.path.exists(apps_dir):
            return apps
        
        try:
            for item in os.listdir(apps_dir):
                app_path = os.path.join(apps_dir, item)
                if os.path.isdir(app_path):
                    app_json = os.path.join(app_path, 'app.json')
                    if os.path.exists(app_json):
                        try:
                            with open(app_json, 'r', encoding='utf-8') as f:
                                config = json.load(f)
                            apps.append({
                                'name': item,
                                'display_name': config.get('app_name', item)
                            })
                        except Exception:
                            # 如果JSON解析失败，仍然列出应用
                            apps.append({
                                'name': item,
                                'display_name': item
                            })
        except Exception as e:
            print(f"⚠️ Error listing apps: {e}")
        
        return apps
    
    def is_app_mode(self) -> bool:
        """检查是否处于应用模式"""
        return self.app_config is not None

