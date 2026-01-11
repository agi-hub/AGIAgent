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

"""
Skill管理与经验总结工具集
提供skill查询、评价、编辑、删除和文件备份功能
"""

import os
import re
import yaml
import shutil
import time
from datetime import datetime
from typing import Dict, Any, List, Optional
from pathlib import Path

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

from src.tools.print_system import print_current, print_error, print_debug
from src.config_loader import get_gui_default_data_directory


class SkillTools:
    """
    Skill管理与经验总结工具类
    """
    
    def __init__(self, workspace_root: Optional[str] = None, user_id: Optional[str] = None):
        """
        初始化Skill工具
        
        Args:
            workspace_root: 工作空间根目录
            user_id: 用户ID（可选，用于确定用户目录）
        """
        self.workspace_root = workspace_root or os.getcwd()
        self.user_id = user_id
        
        # 查找experience目录
        self.experience_dir = self._find_experience_directory()
        if not self.experience_dir:
            print_debug("⚠️ Experience directory not found. Skill tools may not work properly.")
        
        # 确保目录存在
        if self.experience_dir:
            os.makedirs(self.experience_dir, exist_ok=True)
            os.makedirs(os.path.join(self.experience_dir, "legacy"), exist_ok=True)
            os.makedirs(os.path.join(self.experience_dir, "codes"), exist_ok=True)
            os.makedirs(os.path.join(self.experience_dir, "logs"), exist_ok=True)
    
    def _find_experience_directory(self) -> Optional[str]:
        """
        查找experience目录
        
        查找顺序：
        1. 从workspace_root向上查找，寻找包含user目录的结构
        2. 使用gui_default_data_directory配置
        3. 使用默认的data目录
        
        Returns:
            experience目录路径，如果找不到则返回None
        """
        # 方法1: 从workspace_root向上查找
        current = Path(self.workspace_root).resolve()
        for _ in range(5):  # 最多向上5层
            # 查找包含output_XXX的目录结构
            parent = current.parent
            for item in parent.iterdir():
                if item.is_dir() and item.name.startswith('output_'):
                    # 找到output目录，查找对应的user目录
                    user_dir = self._find_user_dir_from_output(parent, item.name)
                    if user_dir:
                        exp_dir = os.path.join(user_dir, "general", "experience")
                        if os.path.exists(os.path.dirname(exp_dir)):  # 至少general目录存在
                            return exp_dir
            
            # 检查当前目录是否有user目录结构
            for item in current.iterdir():
                if item.is_dir() and not item.name.startswith('.'):
                    user_dir = os.path.join(str(current), item.name)
                    exp_dir = os.path.join(user_dir, "general", "experience")
                    if os.path.exists(os.path.dirname(exp_dir)):
                        return exp_dir
            
            current = parent
            if current == current.parent:  # 到达根目录
                break
        
        # 方法2: 使用gui_default_data_directory
        data_dir = get_gui_default_data_directory()
        if data_dir:
            # 尝试查找user目录
            for item in os.listdir(data_dir):
                item_path = os.path.join(data_dir, item)
                if os.path.isdir(item_path) and not item.startswith('.'):
                    exp_dir = os.path.join(item_path, "general", "experience")
                    if os.path.exists(os.path.dirname(exp_dir)):
                        return exp_dir
        
        # 方法3: 使用默认data目录（相对于项目根）
        project_root = self._find_project_root()
        if project_root:
            data_dir = os.path.join(project_root, "data")
            if os.path.exists(data_dir):
                for item in os.listdir(data_dir):
                    item_path = os.path.join(data_dir, item)
                    if os.path.isdir(item_path) and not item.startswith('.'):
                        exp_dir = os.path.join(item_path, "general", "experience")
                        if os.path.exists(os.path.dirname(exp_dir)):
                            return exp_dir
        
        # 方法4: 如果都找不到，在data目录下创建默认的experience目录
        project_root = self._find_project_root()
        if project_root:
            data_dir = os.path.join(project_root, "data")
            if os.path.exists(data_dir):
                # 创建默认的experience目录结构
                default_exp_dir = os.path.join(data_dir, "default", "general", "experience")
                os.makedirs(default_exp_dir, exist_ok=True)
                return default_exp_dir
        
        return None
    
    def _find_project_root(self) -> Optional[str]:
        """查找项目根目录（包含config目录的目录）"""
        current = Path(self.workspace_root).resolve()
        for _ in range(10):
            config_dir = current / "config"
            if config_dir.exists() and config_dir.is_dir():
                return str(current)
            if current == current.parent:
                break
            current = current.parent
        return None
    
    def _find_user_dir_from_output(self, base_dir: str, output_name: str) -> Optional[str]:
        """从output目录名推断user目录"""
        # output_20260104_155505 -> 查找可能的user目录
        # 通常user目录在base_dir下
        if not os.path.exists(base_dir):
            return None
        
        for item in os.listdir(base_dir):
            item_path = os.path.join(base_dir, item)
            if os.path.isdir(item_path) and not item.startswith('.'):
                # 检查是否有general目录
                general_dir = os.path.join(item_path, "general")
                if os.path.exists(general_dir):
                    return item_path
        
        return None
    
    def _load_skill_file(self, skill_file_path: str) -> Optional[Dict[str, Any]]:
        """
        加载skill文件
        
        Args:
            skill_file_path: skill文件路径
            
        Returns:
            包含front matter和content的字典，如果失败返回None
        """
        try:
            with open(skill_file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 解析front matter
            if not content.startswith('---'):
                return None
            
            # 找到front matter结束位置
            end_pos = content.find('---', 3)
            if end_pos == -1:
                return None
            
            front_matter_str = content[3:end_pos].strip()
            body_content = content[end_pos + 3:].strip()
            
            # 解析YAML
            try:
                front_matter = yaml.safe_load(front_matter_str)
                if not isinstance(front_matter, dict):
                    return None
            except yaml.YAMLError:
                return None
            
            return {
                'front_matter': front_matter,
                'content': body_content,
                'full_content': content
            }
        except Exception as e:
            print_error(f"Error loading skill file {skill_file_path}: {e}")
            return None
    
    def _save_skill_file(self, skill_file_path: str, front_matter: Dict[str, Any], content: str):
        """保存skill文件"""
        try:
            os.makedirs(os.path.dirname(skill_file_path), exist_ok=True)
            
            # 构建文件内容
            yaml_str = yaml.dump(front_matter, allow_unicode=True, default_flow_style=False, sort_keys=False)
            file_content = f"---\n{yaml_str}---\n\n{content}"
            
            with open(skill_file_path, 'w', encoding='utf-8') as f:
                f.write(file_content)
        except Exception as e:
            print_error(f"Error saving skill file {skill_file_path}: {e}")
            raise
    
    def _get_skill_file_path(self, skill_id: str) -> Optional[str]:
        """根据skill_id查找skill文件"""
        if not self.experience_dir:
            return None
        
        # 确保skill_id是字符串类型（统一类型以便比较）
        skill_id_str = str(skill_id)
        
        # 遍历所有skill文件，查找匹配的skill_id
        for filename in os.listdir(self.experience_dir):
            if filename.startswith('skill_') and filename.endswith('.md'):
                file_path = os.path.join(self.experience_dir, filename)
                skill_data = self._load_skill_file(file_path)
                if skill_data:
                    # 获取文件中的skill_id，确保转换为字符串进行比较
                    file_skill_id = skill_data['front_matter'].get('skill_id')
                    if file_skill_id is not None:
                        # 统一转换为字符串进行比较，避免类型不匹配
                        if str(file_skill_id) == skill_id_str:
                            return file_path
        
        return None
    
    def _sanitize_filename(self, title: str) -> str:
        """清理文件名，使用标题前20个字符"""
        # 去除特殊字符
        safe_title = re.sub(r'[<>:"/\\|?*]', '', title)
        # 取前20个字符
        safe_title = safe_title[:20]
        # 去除首尾空格和点
        safe_title = safe_title.strip(' .')
        return safe_title if safe_title else "skill"
    
    def query_skill(self, query: str) -> Dict[str, Any]:
        """
        查询相关skill
        
        Args:
            query: 查询字符串
            
        Returns:
            包含status, message, skills_count, skills的字典
        """
        if not SKLEARN_AVAILABLE:
            return {
                "status": "error",
                "message": "scikit-learn not available. Please install it with: pip install scikit-learn",
                "skills_count": 0,
                "skills": []
            }
        
        if not self.experience_dir:
            return {
                "status": "error",
                "message": "Experience directory not found",
                "skills_count": 0,
                "skills": []
            }
        
        try:
            # 加载所有skill文件
            skill_files = []
            for filename in os.listdir(self.experience_dir):
                if filename.startswith('skill_') and filename.endswith('.md'):
                    file_path = os.path.join(self.experience_dir, filename)
                    skill_data = self._load_skill_file(file_path)
                    if skill_data:
                        skill_files.append((file_path, skill_data))
            
            if not skill_files:
                return {
                    "status": "success",
                    "message": "No skills found",
                    "skills_count": 0,
                    "skills": []
                }
            
            # 准备文本用于TF-IDF
            texts = []
            skill_infos = []
            
            for file_path, skill_data in skill_files:
                front_matter = skill_data['front_matter']
                content = skill_data['content']
                
                # 组合标题和内容
                title = front_matter.get('title', '')
                usage_conditions = front_matter.get('usage_conditions', '')
                combined_text = f"{title} {usage_conditions} {content}"
                
                texts.append(combined_text)
                skill_infos.append({
                    'file_path': file_path,
                    'skill_id': str(front_matter.get('skill_id', '')),
                    'front_matter': front_matter,
                    'content': content
                })
            
            # TF-IDF向量化
            vectorizer = TfidfVectorizer(max_features=1000, stop_words=None)
            try:
                tfidf_matrix = vectorizer.fit_transform(texts)
                query_vector = vectorizer.transform([query])
                
                # 计算相似度
                similarities = cosine_similarity(query_vector, tfidf_matrix)[0]
                
                # 获取TOP3
                top_indices = similarities.argsort()[-3:][::-1]
                
                # 构建结果
                results = []
                formatted_output = []
                formatted_output.append(f"Found {min(len(top_indices), len(skill_files))} relevant skills:")
                formatted_output.append("")
                
                for i, idx in enumerate(top_indices, 1):
                    similarity_score = similarities[idx]
                    # 允许相似度为0的情况（至少返回一个结果）
                    if i == 1 or similarity_score > 0:
                        skill_info = skill_infos[idx]
                        front_matter = skill_info['front_matter']
                    
                        skill_result = {
                            "skill_id": skill_info['skill_id'],
                            "title": front_matter.get('title', ''),
                            "usage_conditions": front_matter.get('usage_conditions', ''),
                            "quality_index": front_matter.get('quality_index', 0.5),
                            "fetch_count": front_matter.get('fetch_count', 0),
                            "similarity_score": float(similarity_score),
                            "content": skill_info['content'],
                            "related_code": front_matter.get('related_code', ''),
                            "task_directories": front_matter.get('task_directories', []),
                            "user_preferences": front_matter.get('user_preferences', ''),
                            "created_at": front_matter.get('created_at', ''),
                            "updated_at": front_matter.get('updated_at', ''),
                            "last_used_at": front_matter.get('last_used_at')
                        }
                        results.append(skill_result)
                        
                        # 更新fetch_count
                        front_matter['fetch_count'] = front_matter.get('fetch_count', 0) + 1
                        front_matter['last_used_at'] = datetime.now().isoformat()
                        self._save_skill_file(skill_info['file_path'], front_matter, skill_info['content'])
                        
                        # 格式化输出
                        formatted_output.append(f"Skill {i}:")
                        formatted_output.append(f"  ID: {skill_result['skill_id']}")
                        formatted_output.append(f"  Title: {skill_result['title']}")
                        formatted_output.append(f"  Similarity: {skill_result['similarity_score']:.3f}")
                        formatted_output.append(f"  Usage Conditions: {skill_result['usage_conditions']}")
                        formatted_output.append(f"  Quality Index: {skill_result['quality_index']:.2f}")
                        formatted_output.append("")
                
                message = "\n".join(formatted_output)
                
                return {
                    "status": "success",
                    "message": message,
                    "skills_count": len(results),
                    "skills": results
                }
            except Exception as e:
                print_error(f"Error in TF-IDF calculation: {e}")
                return {
                    "status": "error",
                    "message": f"Error calculating similarity: {str(e)}",
                    "skills_count": 0,
                    "skills": []
                }
        except Exception as e:
            print_error(f"Error querying skills: {e}")
            return {
                "status": "error",
                "message": f"Error querying skills: {str(e)}",
                "skills_count": 0,
                "skills": []
            }
    
    def rate_skill(self, skill_id: str, rating: float) -> Dict[str, Any]:
        """
        评价skill质量
        
        Args:
            skill_id: skill ID
            rating: 评分（0-1范围）
            
        Returns:
            操作结果字典
        """
        if not self.experience_dir:
            return {
                "status": "error",
                "message": "Experience directory not found"
            }
        
        # 确保rating是数字类型（处理字符串或整数输入）
        try:
            if isinstance(rating, str):
                rating = float(rating)
            else:
                rating = float(rating)
        except (ValueError, TypeError):
            return {
                "status": "error",
                "message": f"Invalid rating value: {rating}. Rating must be a number between 0.0 and 1.0"
            }
        
        # 验证rating范围
        rating = max(0.0, min(1.0, rating))
        
        skill_file_path = self._get_skill_file_path(skill_id)
        if not skill_file_path:
            # 提供更详细的错误信息，包括可能的调试信息
            debug_info = []
            if self.experience_dir:
                debug_info.append(f"Experience directory: {self.experience_dir}")
                try:
                    skill_files = [f for f in os.listdir(self.experience_dir) 
                                 if f.startswith('skill_') and f.endswith('.md')]
                    debug_info.append(f"Found {len(skill_files)} skill files in directory")
                    if skill_files:
                        # 尝试列出前几个技能的ID以便调试
                        sample_ids = []
                        for filename in skill_files[:3]:
                            file_path = os.path.join(self.experience_dir, filename)
                            skill_data = self._load_skill_file(file_path)
                            if skill_data:
                                sid = skill_data['front_matter'].get('skill_id')
                                if sid is not None:
                                    sample_ids.append(str(sid))
                        if sample_ids:
                            debug_info.append(f"Sample skill IDs found: {', '.join(sample_ids)}")
                except Exception as e:
                    debug_info.append(f"Error listing skills: {str(e)}")
            
            error_msg = f"Skill with ID '{skill_id}' not found"
            if debug_info:
                error_msg += f". Debug info: {'; '.join(debug_info)}"
            
            return {
                "status": "error",
                "message": error_msg
            }
        
        try:
            skill_data = self._load_skill_file(skill_file_path)
            if not skill_data:
                return {
                    "status": "error",
                    "message": f"Failed to load skill file for ID {skill_id}"
                }
            
            front_matter = skill_data['front_matter']
            old_quality = front_matter.get('quality_index', 0.5)
            
            # 加权平均更新质量指数
            new_quality = 0.7 * old_quality + 0.3 * rating
            front_matter['quality_index'] = round(new_quality, 3)
            front_matter['updated_at'] = datetime.now().isoformat()
            
            self._save_skill_file(skill_file_path, front_matter, skill_data['content'])
            
            return {
                "status": "success",
                "message": f"Skill {skill_id} quality index updated from {old_quality:.3f} to {new_quality:.3f}",
                "skill_id": skill_id,
                "old_quality": old_quality,
                "new_quality": new_quality
            }
        except Exception as e:
            print_error(f"Error rating skill {skill_id}: {e}")
            return {
                "status": "error",
                "message": f"Error rating skill: {str(e)}"
            }
    
    def edit_skill(self, skill_id: str, edit_mode: str, code_edit: str, old_code: Optional[str] = None) -> Dict[str, Any]:
        """
        编辑skill文件
        
        Args:
            skill_id: skill ID
            edit_mode: 编辑模式 - "lines_replace", "append", "full_replace"
            code_edit: 要编辑的内容
            old_code: 对于lines_replace模式，需要替换的旧代码
            
        Returns:
            操作结果字典
        """
        if not self.experience_dir:
            return {
                "status": "error",
                "message": "Experience directory not found"
            }
        
        skill_file_path = self._get_skill_file_path(skill_id)
        if not skill_file_path:
            return {
                "status": "error",
                "message": f"Skill with ID {skill_id} not found"
            }
        
        try:
            skill_data = self._load_skill_file(skill_file_path)
            if not skill_data:
                return {
                    "status": "error",
                    "message": f"Failed to load skill file for ID {skill_id}"
                }
            
            front_matter = skill_data['front_matter']
            content = skill_data['content']
            
            # 根据edit_mode处理内容
            if edit_mode == "full_replace":
                new_content = code_edit
            elif edit_mode == "append":
                new_content = content + "\n\n" + code_edit
            elif edit_mode == "lines_replace":
                if old_code:
                    new_content = content.replace(old_code, code_edit)
                else:
                    new_content = code_edit
            else:
                return {
                    "status": "error",
                    "message": f"Invalid edit_mode: {edit_mode}. Must be 'full_replace', 'append', or 'lines_replace'"
                }
            
            # 更新updated_at
            front_matter['updated_at'] = datetime.now().isoformat()
            
            # 保存文件
            self._save_skill_file(skill_file_path, front_matter, new_content)
            
            return {
                "status": "success",
                "message": f"Skill {skill_id} updated successfully",
                "skill_id": skill_id,
                "edit_mode": edit_mode
            }
        except Exception as e:
            print_error(f"Error editing skill {skill_id}: {e}")
            return {
                "status": "error",
                "message": f"Error editing skill: {str(e)}"
            }
    
    def delete_skill(self, skill_id: str) -> Dict[str, Any]:
        """
        删除skill文件（移动到legacy目录）
        
        Args:
            skill_id: skill ID
            
        Returns:
            操作结果字典
        """
        if not self.experience_dir:
            return {
                "status": "error",
                "message": "Experience directory not found"
            }
        
        skill_file_path = self._get_skill_file_path(skill_id)
        if not skill_file_path:
            return {
                "status": "error",
                "message": f"Skill with ID {skill_id} not found"
            }
        
        try:
            legacy_dir = os.path.join(self.experience_dir, "legacy")
            os.makedirs(legacy_dir, exist_ok=True)
            
            filename = os.path.basename(skill_file_path)
            legacy_path = os.path.join(legacy_dir, filename)
            
            # 如果legacy目录中已存在同名文件，添加时间戳
            if os.path.exists(legacy_path):
                name, ext = os.path.splitext(filename)
                timestamp = int(time.time())
                legacy_path = os.path.join(legacy_dir, f"{name}_{timestamp}{ext}")
            
            shutil.move(skill_file_path, legacy_path)
            
            return {
                "status": "success",
                "message": f"Skill {skill_id} moved to legacy directory",
                "skill_id": skill_id,
                "legacy_path": legacy_path
            }
        except Exception as e:
            print_error(f"Error deleting skill {skill_id}: {e}")
            return {
                "status": "error",
                "message": f"Error deleting skill: {str(e)}"
            }
    
    def copy_skill_files(self, skill_id: str, file_paths: List[str]) -> Dict[str, Any]:
        """
        复制文件到skill的代码备份目录
        
        Args:
            skill_id: skill ID
            file_paths: 要复制的文件路径列表
            
        Returns:
            操作结果字典
        """
        if not self.experience_dir:
            return {
                "status": "error",
                "message": "Experience directory not found"
            }
        
        skill_file_path = self._get_skill_file_path(skill_id)
        if not skill_file_path:
            return {
                "status": "error",
                "message": f"Skill with ID {skill_id} not found"
            }
        
        try:
            skill_data = self._load_skill_file(skill_file_path)
            if not skill_data:
                return {
                    "status": "error",
                    "message": f"Failed to load skill file for ID {skill_id}"
                }
            
            front_matter = skill_data['front_matter']
            
            # 确定备份目录（使用task_directories中的第一个，或创建新的）
            task_dirs = front_matter.get('task_directories', [])
            if task_dirs:
                task_name = task_dirs[0].replace('output_', 'task_')
            else:
                task_name = f"task_{skill_id}"
            
            backup_dir = os.path.join(self.experience_dir, "codes", task_name)
            os.makedirs(backup_dir, exist_ok=True)
            
            copied_files = []
            failed_files = []
            
            for file_path in file_paths:
                try:
                    # 解析文件路径（可能是相对路径或绝对路径）
                    if os.path.isabs(file_path):
                        src_path = file_path
                    else:
                        # 相对于workspace_root
                        src_path = os.path.join(self.workspace_root, file_path)
                    
                    if not os.path.exists(src_path):
                        failed_files.append(file_path)
                        continue
                    
                    # 保持目录结构
                    rel_path = os.path.relpath(src_path, self.workspace_root)
                    dst_path = os.path.join(backup_dir, rel_path)
                    
                    # 创建目标目录
                    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
                    
                    # 复制文件
                    shutil.copy2(src_path, dst_path)
                    copied_files.append(rel_path)
                    
                except Exception as e:
                    print_error(f"Error copying file {file_path}: {e}")
                    failed_files.append(file_path)
            
            # 更新related_code字段
            if copied_files:
                existing_code = front_matter.get('related_code', '')
                if existing_code:
                    code_paths = existing_code.split(',') if ',' in existing_code else [existing_code]
                else:
                    code_paths = []
                
                for rel_path in copied_files:
                    code_path = os.path.join("codes", task_name, rel_path)
                    if code_path not in code_paths:
                        code_paths.append(code_path)
                
                front_matter['related_code'] = ', '.join(code_paths)
                front_matter['updated_at'] = datetime.now().isoformat()
                self._save_skill_file(skill_file_path, front_matter, skill_data['content'])
            
            return {
                "status": "success",
                "message": f"Copied {len(copied_files)} files to skill backup directory",
                "skill_id": skill_id,
                "copied_files": copied_files,
                "failed_files": failed_files,
                "backup_dir": backup_dir
            }
        except Exception as e:
            print_error(f"Error copying files for skill {skill_id}: {e}")
            return {
                "status": "error",
                "message": f"Error copying files: {str(e)}"
            }


