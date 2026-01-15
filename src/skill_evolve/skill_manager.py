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
        
        # 初始化skill工具（logger需要用到）
        self.skill_tools = SkillTools(workspace_root=self.root_dir)
        
        # 设置日志（需要在LLM客户端初始化之前，因为异常处理会用到logger）
        self.logger = self._setup_logger()
        
        # 初始化LLM客户端
        self.api_key = get_api_key(config_file)
        self.api_base = get_api_base(config_file)
        self.model = get_model(config_file)
        
        self.llm_client = None
        self.is_claude = False
        
        if self.api_key and self.model:
            # 参考task_reflection.py的逻辑：如果模型名包含claude或api_base包含anthropic，使用Anthropic SDK
            if 'claude' in self.model.lower() or 'anthropic' in str(self.api_base).lower():
                if ANTHROPIC_AVAILABLE:
                    try:
                        # 对于minimax和GLM等使用Anthropic兼容API的服务，需要传入base_url
                        if 'bigmodel.cn' in str(self.api_base).lower() or 'minimaxi.com' in str(self.api_base).lower():
                            self.llm_client = anthropic.Anthropic(api_key=self.api_key, base_url=self.api_base)
                        else:
                            self.llm_client = anthropic.Anthropic(api_key=self.api_key)
                        self.is_claude = True
                    except Exception as e:
                        self.logger.warning(f"Failed to initialize Anthropic client: {e}")
                        self.llm_client = None
                        self.is_claude = False
                else:
                    self.logger.warning("Anthropic SDK not available, cannot initialize LLM client")
            else:
                # 对于非Anthropic模型，使用OpenAI兼容客户端
                if OPENAI_AVAILABLE:
                    try:
                        self.llm_client = OpenAI(api_key=self.api_key, base_url=self.api_base)
                        self.is_claude = False
                    except Exception as e:
                        self.logger.warning(f"Failed to initialize OpenAI-compatible client: {e}")
                else:
                    self.logger.warning("OpenAI SDK not available, cannot initialize LLM client")
        else:
            missing = []
            if not self.api_key:
                missing.append("api_key")
            if not self.model:
                missing.append("model")
            self.logger.warning(f"Missing required configuration: {', '.join(missing)}, cannot initialize LLM client")
        
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
    
    def _cluster_by_usage_conditions(self, skills: List[Tuple[str, Dict[str, Any]]]) -> Dict[int, List[int]]:
        """
        基于usage_conditions对skill进行聚类
        
        Args:
            skills: skill列表
            
        Returns:
            {聚类ID: [skill索引列表], ...} 字典
        """
        clusters = {}
        cluster_id = 0
        usage_to_indices = {}
        
        # 按usage_conditions分组
        for idx, (file_path, skill_data) in enumerate(skills):
            front_matter = skill_data['front_matter']
            usage = front_matter.get('usage_conditions', '').strip()
            
            # 提取关键词（去除常见词）
            if usage:
                # 简单提取关键工具或操作
                key_terms = []
                if 'custom_command' in usage.lower():
                    key_terms.append('custom_command')
                if 'game' in usage.lower():
                    key_terms.append('game')
                if 'type=' in usage.lower():
                    # 提取type的值
                    import re
                    type_match = re.search(r"type=['\"]?(\w+)['\"]?", usage, re.IGNORECASE)
                    if type_match:
                        key_terms.append(f"type_{type_match.group(1)}")
                
                # 创建聚类键
                cluster_key = '_'.join(sorted(key_terms)) if key_terms else usage[:50]
            else:
                cluster_key = 'unknown'
            
            if cluster_key not in usage_to_indices:
                usage_to_indices[cluster_key] = []
            usage_to_indices[cluster_key].append(idx)
        
        # 只保留包含至少2个skill的聚类
        for cluster_key, indices in usage_to_indices.items():
            if len(indices) >= 2:
                clusters[cluster_id] = indices
                cluster_id += 1
        
        return clusters
    
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
            
            # DBSCAN聚类 - 尝试多个eps值
            # 首先尝试标准参数
            dbscan = DBSCAN(eps=0.5, min_samples=2, metric='precomputed')
            labels = dbscan.fit_predict(distance_matrix)
            
            # 统计聚类结果
            unique_labels = set(labels)
            noise_count = list(labels).count(-1)
            cluster_count = len(unique_labels) - (1 if -1 in unique_labels else 0)
            
            # 如果没有找到聚类，尝试更宽松的参数
            if cluster_count == 0 and len(skills) >= 2:
                dbscan = DBSCAN(eps=0.6, min_samples=2, metric='precomputed')
                labels = dbscan.fit_predict(distance_matrix)
                unique_labels = set(labels)
                noise_count = list(labels).count(-1)
                cluster_count = len(unique_labels) - (1 if -1 in unique_labels else 0)
            
            # 如果还是没有聚类，尝试基于相似度的简单聚类（逐步降低阈值）
            if cluster_count == 0 and len(skills) >= 2:
                # 尝试多个阈值，从高到低
                thresholds = [0.3, 0.2, 0.15, 0.1]
                for threshold in thresholds:
                    clusters = {}
                    cluster_id = 0
                    assigned = set()
                    
                    for i in range(len(skills)):
                        if i in assigned:
                            continue
                        
                        # 找到与当前skill相似的skill
                        similar_indices = [i]
                        for j in range(i + 1, len(skills)):
                            if j in assigned:
                                continue
                            if similarity_array[i][j] > threshold:
                                similar_indices.append(j)
                        
                        if len(similar_indices) >= 2:
                            clusters[cluster_id] = similar_indices
                            assigned.update(similar_indices)
                            cluster_id += 1
                    
                    if clusters:
                        print_current(f"✅ 找到 {len(clusters)} 个skill聚类 (相似度阈值: {threshold})")
                        return clusters
                
                # 如果所有阈值都失败，尝试基于usage_conditions的聚类
                clusters = self._cluster_by_usage_conditions(skills)
                if clusters:
                    print_current(f"✅ 基于使用条件找到 {len(clusters)} 个skill聚类")
                    return clusters
            
            # 组织DBSCAN聚类结果
            clusters = {}
            for idx, label in enumerate(labels):
                if label != -1:  # -1表示噪声点
                    if label not in clusters:
                        clusters[label] = []
                    clusters[label].append(idx)
            
            return clusters
        except Exception as e:
            self.logger.error(f"Error in DBSCAN clustering: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
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
        
        # 构建提示 - 根据需求文档，强调跨任务经验总结
        system_prompt = """你是一个skill整合专家，专门负责将多个任务的skill整合成更高级的综合skill。

**重要说明：**
这些skill已经被相似度算法识别为相关任务，它们来自相似的任务类型或执行场景。你的任务是分析它们，并决定是否应该整合成一个更高级的综合skill。

**核心目标：**
跨任务skill整合的目的是形成更高级的新skill（多任务综合出来的skill），特别是要总结出：
1. **成功经验**：哪些策略和方法导致了任务成功完成
2. **失败教训**：哪些做法导致了任务失败或未完成
3. **成功路径**：如何从失败走向成功，最终成功的核心方法和关键步骤
4. **任务规律**：总结任务类型、执行规律和关键成功因素

**重要要求：**
- 必须使用中文输出
- **如果这些skill来自相似的任务类型，应该倾向于合并**，除非它们确实无法整合
- 必须对比成功和失败的案例，深入分析成功与失败的根本原因
- 对于多次尝试的任务，必须总结出最短成功路径和关键成功因素
- 整合后的skill应该比单个skill更有价值，能够指导未来类似任务的执行
- 重点关注任务执行的失败/成功总结，提炼出可复用的经验和教训
- **必须明确说明合并的理由，即使决定不合并也要给出详细原因**

**输出格式（严格按照此格式输出）：**
MERGE: yes
REASON: 合并理由（详细说明为什么这些skill可以整合，整合后的价值。如果不合并，说明为什么不合并）
TITLE: 新skill标题（简洁明确，体现综合经验。如果不合并，可以留空）
USAGE_CONDITIONS: 新skill使用条件（具体明确，说明何时使用。如果不合并，可以留空）
CONTENT: 合并后的详细内容（必须包含：1. 任务类型和执行规律总结；2. 成功经验；3. 失败教训；4. 成功策略和关键方法；5. 最短成功路径；6. 用户偏好。如果不合并，可以留空）"""

        skill_descriptions = []
        for i, skill in enumerate(skill_group, 1):
            front_matter = skill['front_matter']
            # 传递完整的skill内容，而不是只传递前500个字符
            full_content = skill['content']
            
            # 检查skill是否涉及成功或失败
            title = front_matter.get('title', '')
            is_success = '成功' in title or '完成' in title or '获胜' in title or '战胜' in title
            is_failure = '失败' in title or '未完成' in title or '输' in title or '惜败' in title
            
            skill_descriptions.append(f"""
Skill {i}:
- ID: {front_matter.get('skill_id')}
- Title: {front_matter.get('title')}
- Usage Conditions: {front_matter.get('usage_conditions')}
- Quality Index: {front_matter.get('quality_index')}
- 任务结果: {'成功' if is_success else '失败' if is_failure else '未知'}
- 完整内容:
{full_content}
""")
        
        user_prompt = f"""请分析以下{len(skill_group)}个相关的skill，它们来自不同的任务执行。

**重要提示：**这些skill已经被相似度算法识别为相关任务，它们很可能来自相似的任务类型或执行场景。请仔细分析它们，**如果它们确实相关，应该倾向于合并**。

{''.join(skill_descriptions)}

**分析要求：**
1. **首先判断**：这些skill是否来自相似的任务类型或执行场景？如果相似，应该合并。
2. **重点对比成功和失败的案例**：
   - 分析成功案例中的关键策略、方法和操作
   - 分析失败案例中的错误做法、失败原因和关键问题点
   - 总结出任务成功完成的核心方法和必要条件
   - 总结出导致任务失败的主要原因和需要避免的陷阱
3. 如果涉及多次迭代的任务，必须总结：
   - 失败的主要原因（关键点）
   - 最终成功的策略（核心方法）
   - 最短成功路径和关键步骤
4. 整合后的skill应该能够指导未来执行类似任务时：
   - 如何避免失败
   - 如何成功完成任务
   - 采用哪些关键策略和方法

**请严格按照输出格式回答：**
- 如果这些skill来自相似任务，输出 MERGE: yes，并提供完整的整合内容
- 如果确实无法整合，输出 MERGE: no，并在REASON中详细说明为什么不合并"""
        
        try:
            print_current(f"🔄 调用LLM进行skill整合决策...")
            if self.is_claude:
                response = self.llm_client.messages.create(
                    model=self.model,
                    max_tokens=6000,  # 增加token数量以支持完整内容整合
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
                    max_tokens=6000,  # 增加token数量以支持完整内容整合
                    temperature=0.7
                )
                
                # 检查API响应是否有错误
                if hasattr(response, 'success') and response.success is False:
                    error_msg = getattr(response, 'msg', 'Unknown error')
                    error_code = getattr(response, 'code', 'Unknown')
                    self.logger.error(f"API调用失败: code={error_code}, msg={error_msg}")
                    print_error(f"❌ API调用失败: {error_msg} (code: {error_code})")
                    print_error(f"   请检查API配置、模型名称和端点是否正确")
                    return {
                        'should_merge': False,
                        'reason': f'API调用失败: {error_msg} (code: {error_code})'
                    }
                
                decision_text = response.choices[0].message.content if response.choices else ""
            
            # 检查响应结构
            if not decision_text:
                if self.is_claude:
                    self.logger.error(f"Claude API response structure: {response}")
                    print_error(f"❌ Claude API响应结构异常: {response}")
                else:
                    # 检查是否有choices
                    if not hasattr(response, 'choices') or not response.choices:
                        self.logger.error(f"OpenAI API response has no choices: {response}")
                        print_error(f"❌ OpenAI API响应中没有choices字段")
                        print_error(f"   响应对象: {response}")
                        # 尝试获取错误信息
                        if hasattr(response, 'error'):
                            print_error(f"   错误信息: {response.error}")
                        return {
                            'should_merge': False,
                            'reason': 'API响应中没有choices字段，可能是API调用失败'
                        }
                    self.logger.error(f"OpenAI API response structure: {response}")
                    print_error(f"❌ OpenAI API响应结构异常: {response}")
            
            # 检查响应是否为空
            if not decision_text or not decision_text.strip():
                self.logger.error("LLM returned empty response!")
                print_error("❌ LLM返回空响应，请检查API配置和网络连接")
                return {
                    'should_merge': False,
                    'reason': 'LLM返回空响应，可能是API调用失败或配置问题'
                }
            
            # 记录LLM完整响应用于调试
            print_current(f"📝 LLM响应 (前500字符): {decision_text[:500]}")
            if len(decision_text) > 500:
                print_current(f"📝 LLM响应 (继续): ...{decision_text[500:1000]}")
            self.logger.info(f"LLM完整响应: {decision_text}")
            
            # 解析决策 - 支持多种格式
            decision_upper = decision_text.upper()
            # 检查是否明确说不合并
            should_not_merge = 'MERGE: no' in decision_upper or 'MERGE:NO' in decision_upper or 'MERGE: NO' in decision_upper
            # 检查是否明确说合并
            should_merge = ('MERGE: yes' in decision_upper or 'MERGE:YES' in decision_upper or 
                          'MERGE: YES' in decision_upper)
            
            # 如果没有明确的yes/no，尝试从上下文推断
            if not should_merge and not should_not_merge:
                # 如果提到了"合并"、"整合"等关键词，且没有明确拒绝，倾向于合并
                if any(keyword in decision_text for keyword in ['合并', '整合', '综合', '整合后', '合并后']):
                    if not any(keyword in decision_text for keyword in ['不合并', '不整合', '无法合并', '不能合并']):
                        should_merge = True
                        self.logger.info("Inferred merge decision from context keywords")
            
            # 提取信息 - 改进解析逻辑，支持多行内容和中文冒号
            reason = ""
            # 尝试多种格式：REASON:、REASON：、原因：等
            reason_markers = ['REASON:', 'REASON：', '原因:', '原因：', '理由:', '理由：']
            for marker in reason_markers:
                if marker in decision_text:
                    reason_part = decision_text.split(marker, 1)[1]
                    # 提取到下一个标记或段落结束
                    next_markers = ['TITLE:', 'TITLE：', '标题:', '标题：', 'USAGE_CONDITIONS:', 'USAGE_CONDITIONS：', 
                                   '使用条件:', '使用条件：', 'CONTENT:', 'CONTENT：', '内容:', '内容：', '\n\n']
                    for next_marker in next_markers:
                        if next_marker in reason_part:
                            reason = reason_part.split(next_marker)[0].strip()
                            break
                    if not reason:
                        # 如果没有找到下一个标记，取第一行或前200字符
                        reason = reason_part.split('\n')[0].strip()[:200]
                    break
            
            if not reason and not should_merge:
                # 如果没有找到reason标记，尝试从决策文本中提取
                reason = "LLM未提供明确的合并理由"
            
            title = ""
            if 'TITLE:' in decision_text:
                title_part = decision_text.split('TITLE:')[1]
                # 提取到下一个标记或段落结束
                if 'USAGE_CONDITIONS:' in title_part:
                    title = title_part.split('USAGE_CONDITIONS:')[0].strip()
                elif 'CONTENT:' in title_part:
                    title = title_part.split('CONTENT:')[0].strip()
                else:
                    title = title_part.split('\n\n')[0].strip()
            
            usage_conditions = ""
            if 'USAGE_CONDITIONS:' in decision_text:
                usage_part = decision_text.split('USAGE_CONDITIONS:')[1]
                # 提取到CONTENT标记或段落结束
                if 'CONTENT:' in usage_part:
                    usage_conditions = usage_part.split('CONTENT:')[0].strip()
                else:
                    usage_conditions = usage_part.split('\n\n')[0].strip()
            
            content = ""
            if 'CONTENT:' in decision_text:
                content = decision_text.split('CONTENT:')[1].strip()
            elif should_merge and not content:
                # 如果没有明确的CONTENT标记，但决定合并，尝试提取整个决策文本作为内容
                # 跳过前面的标记部分
                content_start = max(
                    decision_text.find('CONTENT:'),
                    decision_text.find('TITLE:'),
                    decision_text.find('USAGE_CONDITIONS:'),
                    decision_text.find('REASON:')
                )
                if content_start > 0:
                    # 找到最后一个标记后的内容
                    last_marker = max(
                        decision_text.rfind('TITLE:'),
                        decision_text.rfind('USAGE_CONDITIONS:'),
                        decision_text.rfind('REASON:')
                    )
                    if last_marker > 0:
                        content = decision_text[last_marker:].split(':', 1)[1].strip() if ':' in decision_text[last_marker:] else decision_text[last_marker:].strip()
            
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
                print_current("⚠️  LLM客户端不可用，跳过跨skill整合。请在配置文件中配置有效的API密钥。")
                self.logger.warning("LLM client not initialized, skipping cross-skill integration step")
                integrated_count = 0
            else:
                # 显示API配置信息用于调试
                print_current(f"🔧 API配置: model={self.model}, api_base={self.api_base}, is_claude={self.is_claude}")
                self.logger.info(f"API config: model={self.model}, api_base={self.api_base}, is_claude={self.is_claude}")
                # 初始化integrated_count
                integrated_count = 0
                
                clusters = self._cluster_skills_with_dbscan(skills)
                
                if not clusters:
                    print_current("ℹ️  未找到skill聚类，尝试让LLM评估所有skill...")
                    self.logger.info("No skill clusters found by DBSCAN, trying LLM-based integration for all skills")
                    
                    # 备选方案：让LLM判断所有skill是否可以整合
                    # 但只尝试较小的skill组（2-4个），避免token过多
                    if len(skills) >= 2 and len(skills) <= 6:
                        # 尝试将所有skill作为一个组让LLM判断
                        cluster_skills = []
                        skill_titles = []
                        for idx, (file_path, skill_data) in enumerate(skills):
                            front_matter = skill_data['front_matter']
                            skill_titles.append(front_matter.get('title', 'Unknown'))
                            cluster_skills.append({
                                'file_path': file_path,
                                'front_matter': front_matter,
                                'content': skill_data['content']
                            })
                        
                        print_current(f"🤔 请求LLM评估 {len(cluster_skills)} 个skill是否应该整合...")
                        
                        decision = self._call_llm_for_merge_decision(cluster_skills)
                        
                        print_current(f"🤖 LLM决策: {'✅ 合并' if decision.get('should_merge') else '❌ 不合并'}")
                        if decision.get('reason'):
                            print_current(f"   理由: {decision.get('reason')[:200]}")
                        
                        if decision.get('should_merge'):
                            # 创建整合后的skill
                            skill_id = str(int(time.time()))
                            title = decision.get('title', f"Integrated Skill from {len(cluster_skills)} tasks")
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
                            # 使用skill_adv_前缀标记为高级整合skill
                            skill_filename = f"skill_adv_{safe_title}.md"
                            skill_file_path = os.path.join(self.skill_tools.experience_dir, skill_filename)
                            
                            if os.path.exists(skill_file_path):
                                name, ext = os.path.splitext(skill_filename)
                                skill_filename = f"{name}_{skill_id}{ext}"
                                skill_file_path = os.path.join(self.skill_tools.experience_dir, skill_filename)
                            
                            # 记录来源skill的ID
                            source_skill_ids = [str(skill['front_matter'].get('skill_id', '')) for skill in cluster_skills]
                            front_matter['source_skill_ids'] = source_skill_ids
                            
                            self.skill_tools._save_skill_file(skill_file_path, front_matter, content)
                            print_current(f"✅ 已创建整合skill: {skill_filename}")
                            print_current(f"   标题: {title}")
                            print_current(f"   来源: {len(cluster_skills)} 个原始skill")
                            
                            integrated_count = 1
                        else:
                            reason = decision.get('reason', '未提供理由')
                            print_current(f"⏭️  LLM决定不合并: {reason[:150]}")
                            integrated_count = 0
                    else:
                        integrated_count = 0
                
                # 处理找到的聚类
                if clusters:
                    for cluster_id, indices in clusters.items():
                        if len(indices) < 2:
                            continue
                        
                        # 获取聚类中的skill
                        cluster_skills = []
                        skill_titles = []
                        for idx in indices:
                            file_path, skill_data = skills[idx]
                            front_matter = skill_data['front_matter']
                            skill_titles.append(front_matter.get('title', 'Unknown'))
                            cluster_skills.append({
                                'file_path': file_path,
                                'front_matter': front_matter,
                                'content': skill_data['content']
                            })
                        
                        print_current(f"📦 处理聚类 {cluster_id} ({len(cluster_skills)} 个skill): {', '.join([t[:30] + '...' if len(t) > 30 else t for t in skill_titles])}")
                        
                        # LLM决策
                        decision = self._call_llm_for_merge_decision(cluster_skills)
                        
                        print_current(f"🤖 LLM决策: {'✅ 合并' if decision.get('should_merge') else '❌ 不合并'}")
                        if decision.get('reason'):
                            print_current(f"   理由: {decision.get('reason')[:200]}")
                        
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
                            
                            # 记录来源skill的ID
                            source_skill_ids = [str(skill['front_matter'].get('skill_id', '')) for skill in cluster_skills]
                            
                            # 创建新skill
                            front_matter = {
                                'skill_id': skill_id,
                                'title': title,
                                'usage_conditions': usage_conditions,
                                'quality_index': round(sum(qualities) / len(qualities), 3),
                                'fetch_count': merged_fetch_count,
                                'related_code': '',
                                'task_directories': list(set(merged_task_dirs)),
                                'source_skill_ids': source_skill_ids,  # 记录来源skill
                                'created_at': datetime.now().isoformat(),
                                'updated_at': datetime.now().isoformat(),
                                'last_used_at': None,
                                'user_preferences': ''
                            }
                            
                            safe_title = self.skill_tools._sanitize_filename(title)
                            # 使用skill_adv_前缀标记为高级整合skill
                            skill_filename = f"skill_adv_{safe_title}.md"
                            skill_file_path = os.path.join(self.skill_tools.experience_dir, skill_filename)
                            
                            if os.path.exists(skill_file_path):
                                name, ext = os.path.splitext(skill_filename)
                                skill_filename = f"{name}_{skill_id}{ext}"
                                skill_file_path = os.path.join(self.skill_tools.experience_dir, skill_filename)
                            
                            self.skill_tools._save_skill_file(skill_file_path, front_matter, content)
                            print_current(f"✅ 已创建整合skill: {skill_filename}")
                            print_current(f"   标题: {title}")
                            print_current(f"   来源: {len(cluster_skills)} 个原始skill")
                            
                            integrated_count += 1
                        else:
                            reason = decision.get('reason', '未提供理由')
                            print_current(f"⏭️  聚类 {cluster_id} 未合并: {reason[:150]}")
            
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

