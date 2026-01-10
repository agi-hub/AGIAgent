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
Skill整理脚本
合并相似skill，清理无用skill，跨skill整合
"""

import os
import re
import argparse
import logging
import yaml
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.cluster import DBSCAN
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

from src.config_loader import (
    load_config, get_api_key, get_api_base, get_model,
    get_gui_default_data_directory
)
from src.tools.print_system import print_current, print_error, print_system
from .skill_tools import SkillTools


class SkillManager:
    """Skill整理管理器"""
    
    def __init__(self, root_dir: Optional[str] = None, config_file: str = "config/config.txt"):
        """
        初始化Skill管理器
        
        Args:
            root_dir: 根目录（如果指定，覆盖config中的设置）
            config_file: 配置文件路径
        """
        self.config_file = config_file
        self.config = load_config(config_file)
        
        # 确定根目录
        if root_dir:
            self.root_dir = os.path.abspath(root_dir)
        else:
            data_dir = get_gui_default_data_directory(config_file)
            if data_dir:
                self.root_dir = data_dir
            else:
                project_root = self._find_project_root()
                self.root_dir = os.path.join(project_root, "data") if project_root else "data"
        
        # 初始化LLM客户端
        self.api_key = get_api_key(config_file)
        self.api_base = get_api_base(config_file)
        self.model = get_model(config_file)
        
        self.llm_client = None
        self.is_claude = False
        
        if self.api_key and self.model:
            # 检查是否是真正的Anthropic API还是兼容端点
            # 真正的Anthropic API base通常是 https://api.anthropic.com
            api_base_str = str(self.api_base).lower() if self.api_base else ""
            is_real_anthropic = ('api.anthropic.com' in api_base_str)
            is_anthropic_compatible = ('anthropic' in api_base_str) and not is_real_anthropic
            
            # 对于真正的Anthropic API，使用Anthropic SDK
            if is_real_anthropic and ('claude' in self.model.lower()) and ANTHROPIC_AVAILABLE:
                try:
                    self.llm_client = anthropic.Anthropic(api_key=self.api_key)
                    self.is_claude = True
                except Exception as e:
                    self.logger.warning(f"Failed to initialize Anthropic client: {e}, falling back to OpenAI-compatible client")
                    self.llm_client = None
                    self.is_claude = False
            
            # 对于兼容端点或其他情况，使用OpenAI兼容客户端
            # 兼容端点通常需要Bearer token认证，而不是x-api-key
            if not self.llm_client and OPENAI_AVAILABLE:
                try:
                    self.llm_client = OpenAI(api_key=self.api_key, base_url=self.api_base)
                    self.is_claude = False
                    # 对于兼容Anthropic的端点，虽然使用OpenAI客户端，但调用时可能需要特殊处理
                    # 这里先使用标准OpenAI格式，如果失败会在调用时处理
                except Exception as e:
                    self.logger.warning(f"Failed to initialize OpenAI-compatible client: {e}")
        
        # 初始化skill工具
        self.skill_tools = SkillTools(workspace_root=self.root_dir)
        
        # 设置日志
        self.logger = self._setup_logger()
        
        # 相似度阈值
        self.similarity_threshold = 0.7
    
    def _find_project_root(self) -> Optional[str]:
        """查找项目根目录"""
        current = Path(__file__).resolve()
        for _ in range(10):
            config_dir = current / "config"
            if config_dir.exists() and config_dir.is_dir():
                return str(current)
            if current == current.parent:
                break
            current = current.parent
        return None
    
    def _setup_logger(self) -> logging.Logger:
        """设置日志记录器"""
        logger = logging.getLogger('skill_manager')
        logger.setLevel(logging.INFO)
        
        if self.skill_tools.experience_dir:
            log_dir = os.path.join(self.skill_tools.experience_dir, "logs")
            os.makedirs(log_dir, exist_ok=True)
            
            log_file = os.path.join(log_dir, f"skill_manager_{datetime.now().strftime('%Y%m%d')}.log")
            
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setLevel(logging.INFO)
            
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(formatter)
            console_handler.setFormatter(formatter)
            
            logger.addHandler(file_handler)
            logger.addHandler(console_handler)
        
        return logger
    
    def _load_all_skills(self) -> List[Tuple[str, Dict[str, Any]]]:
        """
        加载所有skill文件
        
        Returns:
            [(文件路径, skill数据), ...] 列表
        """
        if not self.skill_tools.experience_dir:
            return []
        
        skills = []
        for filename in os.listdir(self.skill_tools.experience_dir):
            if filename.startswith('skill_') and filename.endswith('.md'):
                file_path = os.path.join(self.skill_tools.experience_dir, filename)
                try:
                    skill_data = self.skill_tools._load_skill_file(file_path)
                    if skill_data:
                        skills.append((file_path, skill_data))
                except Exception as e:
                    self.logger.error(f"Error loading skill file {file_path}: {e}")
                    print_error(f"Error loading skill file {file_path}: {e}, skipping...")
        
        return skills
    
    def _calculate_similarity_matrix(self, skills: List[Tuple[str, Dict[str, Any]]]) -> Tuple[List[List[float]], List[str]]:
        """
        计算skill之间的相似度矩阵
        
        Args:
            skills: skill列表
            
        Returns:
            (相似度矩阵, skill_id列表)
        """
        if not SKLEARN_AVAILABLE:
            return [], []
        
        texts = []
        skill_ids = []
        
        for file_path, skill_data in skills:
            front_matter = skill_data['front_matter']
            content = skill_data['content']
            
            title = front_matter.get('title', '')
            usage_conditions = front_matter.get('usage_conditions', '')
            combined_text = f"{title} {usage_conditions} {content}"
            
            texts.append(combined_text)
            skill_ids.append(str(front_matter.get('skill_id', '')))
        
        if not texts:
            return [], []
        
        try:
            vectorizer = TfidfVectorizer(max_features=1000, stop_words=None)
            tfidf_matrix = vectorizer.fit_transform(texts)
            similarity_matrix = cosine_similarity(tfidf_matrix)
            
            return similarity_matrix.tolist(), skill_ids
        except Exception as e:
            self.logger.error(f"Error calculating similarity matrix: {e}")
            return [], []
    
    def _merge_similar_skills(self, skills: List[Tuple[str, Dict[str, Any]]]) -> int:
        """
        合并相似度高的skill
        
        Args:
            skills: skill列表
            
        Returns:
            合并的skill数量
        """
        if not SKLEARN_AVAILABLE:
            self.logger.warning("scikit-learn not available, skipping similarity merge")
            return 0
        
        if len(skills) < 2:
            return 0
        
        similarity_matrix, skill_ids = self._calculate_similarity_matrix(skills)
        if not similarity_matrix:
            return 0
        
        merged_count = 0
        processed = set()
        
        # 创建skill_id到索引的映射
        skill_id_to_idx = {sid: idx for idx, sid in enumerate(skill_ids)}
        idx_to_skill = {idx: skill for idx, skill in enumerate(skills)}
        
        for i in range(len(skills)):
            if i in processed:
                continue
            
            # 查找相似度高的skill
            similar_indices = []
            for j in range(i + 1, len(skills)):
                if j in processed:
                    continue
                
                similarity = similarity_matrix[i][j]
                if similarity > self.similarity_threshold:
                    similar_indices.append(j)
            
            if not similar_indices:
                continue
            
            # 合并skill
            main_skill = skills[i]
            main_front_matter = main_skill[1]['front_matter']
            main_content = main_skill[1]['content']
            main_quality = main_front_matter.get('quality_index', 0.5)
            
            # 找到质量指数最高的作为主skill
            for idx in similar_indices:
                other_skill = skills[idx]
                other_front_matter = other_skill[1]['front_matter']
                other_quality = other_front_matter.get('quality_index', 0.5)
                
                if other_quality > main_quality:
                    main_skill = other_skill
                    main_front_matter = other_front_matter
                    main_content = other_skill[1]['content']
                    main_quality = other_quality
            
            # 合并内容
            merged_content = main_content
            merged_task_dirs = list(main_front_matter.get('task_directories', []))
            merged_fetch_count = main_front_matter.get('fetch_count', 0)
            
            for idx in similar_indices:
                other_skill = skills[idx]
                other_front_matter = other_skill[1]['front_matter']
                other_content = other_skill[1]['content']
                
                # 合并内容
                if other_content not in merged_content:
                    merged_content += f"\n\n---\n\n{other_content}"
                
                # 合并task_directories
                other_dirs = other_front_matter.get('task_directories', [])
                for dir_name in other_dirs:
                    if dir_name not in merged_task_dirs:
                        merged_task_dirs.append(dir_name)
                
                # 合并fetch_count
                merged_fetch_count += other_front_matter.get('fetch_count', 0)
                
                # 删除其他skill（移动到legacy）
                other_skill_id = str(other_front_matter.get('skill_id', ''))
                result = self.skill_tools.delete_skill(other_skill_id)
                if result.get('status') == 'success':
                    merged_count += 1
                    processed.add(idx)
            
            # 更新主skill
            main_front_matter['task_directories'] = merged_task_dirs
            main_front_matter['fetch_count'] = merged_fetch_count
            main_front_matter['updated_at'] = datetime.now().isoformat()
            
            # 重新计算质量指数（加权平均）
            if similar_indices:
                qualities = [main_quality]
                for idx in similar_indices:
                    other_quality = skills[idx][1]['front_matter'].get('quality_index', 0.5)
                    qualities.append(other_quality)
                avg_quality = sum(qualities) / len(qualities)
                main_front_matter['quality_index'] = round(avg_quality, 3)
            
            self.skill_tools._save_skill_file(main_skill[0], main_front_matter, merged_content)
            processed.add(i)
        
        return merged_count
    
    def _cluster_skills_with_dbscan(self, skills: List[Tuple[str, Dict[str, Any]]]) -> Dict[int, List[int]]:
        """
        使用DBSCAN对skill进行聚类
        
        Args:
            skills: skill列表
            
        Returns:
            {聚类ID: [skill索引列表], ...} 字典
        """
        if not SKLEARN_AVAILABLE:
            return {}
        
        if len(skills) < 2:
            return {}
        
        similarity_matrix, skill_ids = self._calculate_similarity_matrix(skills)
        if not similarity_matrix:
            return {}
        
        try:
            # 转换为距离矩阵（1 - 相似度）
            import numpy as np
            similarity_array = np.array(similarity_matrix)
            
            # 确保相似度值在[0, 1]范围内
            similarity_array = np.clip(similarity_array, 0.0, 1.0)
            
            # 转换为距离矩阵（1 - 相似度），确保距离值非负
            distance_matrix = 1.0 - similarity_array
            distance_matrix = np.clip(distance_matrix, 0.0, 1.0)
            
            # DBSCAN聚类
            dbscan = DBSCAN(eps=0.5, min_samples=2, metric='precomputed')
            labels = dbscan.fit_predict(distance_matrix)
            
            # 组织聚类结果
            clusters = {}
            for idx, label in enumerate(labels):
                if label != -1:  # -1表示噪声点
                    if label not in clusters:
                        clusters[label] = []
                    clusters[label].append(idx)
            
            return clusters
        except Exception as e:
            self.logger.error(f"Error in DBSCAN clustering: {e}")
            return {}
    
    def _call_llm_for_merge_decision(self, skill_group: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        调用LLM决定是否合并skill组
        
        Args:
            skill_group: skill组列表
            
        Returns:
            LLM决策结果，包含是否合并和合并后的内容
        """
        if not self.llm_client:
            return {
                'should_merge': False,
                'reason': 'LLM client not available'
            }
        
        if len(skill_group) < 2:
            return {
                'should_merge': False,
                'reason': 'Not enough skills to merge'
            }
        
        # 构建提示
        system_prompt = """你是一个skill整合专家。请分析以下一组相关的skill，决定是否应该将它们合并成一个更高级的综合skill。

如果这些skill可以整合成一个更有价值的综合skill，请：
1. 决定是否合并（输出"MERGE: yes"或"MERGE: no"）
2. 如果合并，提供合并后的skill标题、使用条件和详细内容
3. 说明合并的理由

输出格式：
MERGE: yes/no
REASON: 合并理由
TITLE: 新skill标题
USAGE_CONDITIONS: 新skill使用条件
CONTENT: 合并后的详细内容"""

        skill_descriptions = []
        for i, skill in enumerate(skill_group, 1):
            front_matter = skill['front_matter']
            skill_descriptions.append(f"""
Skill {i}:
- ID: {front_matter.get('skill_id')}
- Title: {front_matter.get('title')}
- Usage Conditions: {front_matter.get('usage_conditions')}
- Quality Index: {front_matter.get('quality_index')}
- Content: {skill['content'][:500]}...
""")
        
        user_prompt = f"""请分析以下{len(skill_group)}个相关的skill：

{''.join(skill_descriptions)}

请决定是否应该将它们合并成一个综合skill。"""
        
        try:
            if self.is_claude:
                response = self.llm_client.messages.create(
                    model=self.model,
                    max_tokens=3000,
                    system=system_prompt,
                    messages=[{"role": "user", "content": user_prompt}],
                    temperature=0.7
                )
                decision_text = response.content[0].text if response.content else ""
            else:
                response = self.llm_client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    max_tokens=3000,
                    temperature=0.7
                )
                decision_text = response.choices[0].message.content if response.choices else ""
            
            # 解析决策
            should_merge = 'MERGE: yes' in decision_text.upper()
            
            # 提取信息
            reason = ""
            if 'REASON:' in decision_text:
                reason = decision_text.split('REASON:')[1].split('\n')[0].strip()
            
            title = ""
            if 'TITLE:' in decision_text:
                title = decision_text.split('TITLE:')[1].split('\n')[0].strip()
            
            usage_conditions = ""
            if 'USAGE_CONDITIONS:' in decision_text:
                usage_conditions = decision_text.split('USAGE_CONDITIONS:')[1].split('CONTENT:')[0].strip()
            
            content = ""
            if 'CONTENT:' in decision_text:
                content = decision_text.split('CONTENT:')[1].strip()
            
            return {
                'should_merge': should_merge,
                'reason': reason,
                'title': title,
                'usage_conditions': usage_conditions,
                'content': content
            }
        
        except Exception as e:
            error_str = str(e)
            # 检查是否是认证错误
            if '401' in error_str or 'authentication' in error_str.lower() or 'invalid' in error_str.lower() and 'key' in error_str.lower():
                self.logger.warning(f"LLM API authentication error: {e}. Please check your API key in config file.")
                return {
                    'should_merge': False,
                    'reason': 'LLM API authentication failed. Please check your API key configuration.'
                }
            else:
                self.logger.error(f"Error calling LLM for merge decision: {e}")
                return {
                    'should_merge': False,
                    'reason': f'Error in LLM call: {str(e)}'
                }
    
    def _clean_unused_skills(self, skills: List[Tuple[str, Dict[str, Any]]]) -> int:
        """
        清理长期不使用的skill
        
        Args:
            skills: skill列表
            
        Returns:
            清理的skill数量
        """
        cleaned_count = 0
        cutoff_date = datetime.now() - timedelta(days=30)
        
        for file_path, skill_data in skills:
            front_matter = skill_data['front_matter']
            fetch_count = front_matter.get('fetch_count', 0)
            created_at_str = front_matter.get('created_at', '')
            
            if fetch_count == 0 and created_at_str:
                try:
                    created_at = datetime.fromisoformat(created_at_str.replace('Z', '+00:00'))
                    if created_at < cutoff_date:
                        skill_id = str(front_matter.get('skill_id', ''))
                        result = self.skill_tools.delete_skill(skill_id)
                        if result.get('status') == 'success':
                            cleaned_count += 1
                            self.logger.info(f"Cleaned unused skill: {skill_id}")
                except Exception:
                    pass
        
        return cleaned_count
    
    def run(self):
        """运行skill整理流程"""
        self.logger.info("Starting skill management process")
        print_system("Starting skill management process")
        
        # 加载所有skill
        skills = self._load_all_skills()
        
        if not skills:
            self.logger.info("No skills found")
            print_current("No skills found")
            return
        
        self.logger.info(f"Loaded {len(skills)} skills")
        print_current(f"Loaded {len(skills)} skills")
        
        # 1. 基础合并（相似度 > 0.7）
        print_current("Step 1: Merging similar skills...")
        merged_count = self._merge_similar_skills(skills)
        self.logger.info(f"Merged {merged_count} similar skills")
        print_current(f"✅ Merged {merged_count} similar skills")
        
        # 重新加载skill（因为可能有变化）
        skills = self._load_all_skills()
        
        # 2. DBSCAN聚类和跨skill整合
        if SKLEARN_AVAILABLE and len(skills) >= 2:
            print_current("Step 2: Cross-skill integration...")
            
            # 检查LLM是否可用
            if not self.llm_client:
                print_current("⚠️  LLM client not available, skipping cross-skill integration. Please configure valid API key in config file.")
                self.logger.warning("LLM client not initialized, skipping cross-skill integration step")
                integrated_count = 0
            else:
                clusters = self._cluster_skills_with_dbscan(skills)
                
                integrated_count = 0
                for cluster_id, indices in clusters.items():
                    if len(indices) < 2:
                        continue
                    
                    # 获取聚类中的skill
                    cluster_skills = []
                    for idx in indices:
                        file_path, skill_data = skills[idx]
                        cluster_skills.append({
                            'file_path': file_path,
                            'front_matter': skill_data['front_matter'],
                            'content': skill_data['content']
                        })
                    
                    # LLM决策
                    decision = self._call_llm_for_merge_decision(cluster_skills)
                    
                    if decision.get('should_merge'):
                        # 创建新的综合skill
                        skill_id = str(int(time.time()))
                        title = decision.get('title', f"Integrated Skill {cluster_id}")
                        usage_conditions = decision.get('usage_conditions', '')
                        content = decision.get('content', '')
                        
                        # 合并task_directories和fetch_count
                        merged_task_dirs = []
                        merged_fetch_count = 0
                        qualities = []
                        
                        for skill in cluster_skills:
                            front_matter = skill['front_matter']
                            merged_task_dirs.extend(front_matter.get('task_directories', []))
                            merged_fetch_count += front_matter.get('fetch_count', 0)
                            qualities.append(front_matter.get('quality_index', 0.5))
                        
                        # 创建新skill
                        front_matter = {
                            'skill_id': skill_id,
                            'title': title,
                            'usage_conditions': usage_conditions,
                            'quality_index': round(sum(qualities) / len(qualities), 3),
                            'fetch_count': merged_fetch_count,
                            'related_code': '',
                            'task_directories': list(set(merged_task_dirs)),
                            'created_at': datetime.now().isoformat(),
                            'updated_at': datetime.now().isoformat(),
                            'last_used_at': None,
                            'user_preferences': ''
                        }
                        
                        safe_title = self.skill_tools._sanitize_filename(title)
                        skill_filename = f"skill_{safe_title}.md"
                        skill_file_path = os.path.join(self.skill_tools.experience_dir, skill_filename)
                        
                        if os.path.exists(skill_file_path):
                            name, ext = os.path.splitext(skill_filename)
                            skill_filename = f"{name}_{skill_id}{ext}"
                            skill_file_path = os.path.join(self.skill_tools.experience_dir, skill_filename)
                        
                        self.skill_tools._save_skill_file(skill_file_path, front_matter, content)
                        
                        # 删除旧skill
                        for skill in cluster_skills:
                            old_skill_id = str(skill['front_matter'].get('skill_id', ''))
                            self.skill_tools.delete_skill(old_skill_id)
                        
                        integrated_count += 1
            
            self.logger.info(f"Integrated {integrated_count} skill clusters")
            print_current(f"✅ Integrated {integrated_count} skill clusters")
        
        # 3. 清理长期不使用的skill
        print_current("Step 3: Cleaning unused skills...")
        skills = self._load_all_skills()
        cleaned_count = self._clean_unused_skills(skills)
        self.logger.info(f"Cleaned {cleaned_count} unused skills")
        print_current(f"✅ Cleaned {cleaned_count} unused skills")
        
        self.logger.info("Skill management process completed")
        print_system("✅ Skill management process completed")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Skill management script')
    parser.add_argument('--root-dir', type=str, help='Root directory for data (overrides config)')
    parser.add_argument('--config', type=str, default='config/config.txt', help='Config file path')
    
    args = parser.parse_args()
    
    manager = SkillManager(root_dir=args.root_dir, config_file=args.config)
    manager.run()


if __name__ == '__main__':
    main()

