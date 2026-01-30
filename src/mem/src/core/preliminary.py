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

Primary Memory Management Module
"""

import os
import time
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from functools import lru_cache
import pickle
import re
import warnings

# Suppress pkg_resources deprecation warning from jieba
warnings.filterwarnings('ignore', category=UserWarning, message='.*pkg_resources.*')

# Set sklearn warning handling: redirect warnings to logs instead of terminal
with warnings.catch_warnings():
    warnings.simplefilter("ignore", UserWarning)
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

# Import jieba after setting up warning filters
import jieba

from ..models.memory_cell import MemCell
from ..clients.llm_client import LLMClient
from ..clients.embedding_client import EmbeddingClient
from ..utils.config import ConfigLoader
from ..utils.logger import get_logger
from ..utils.exceptions import MemorySystemError
from ..models.mem import Mem
from ..utils.embedding_cache import get_global_cache_manager

logger = get_logger(__name__)


class PreliminaryMemoryManager:
    """
    Basic memory management module, responsible for summary+text, embedding, TF-IDF and other basic memory operations.
    """

    def __init__(self, storage_path: str = "memory/preliminary_memory", config_file: str = "config.txt", max_tokens: int = 4096):
        self.storage_path = storage_path
        os.makedirs(self.storage_path, exist_ok=True)
        self.max_tokens = max_tokens
        self.config_loader = ConfigLoader(config_file)
        self.similarity_threshold = self.config_loader.get_similarity_threshold()
        # Memory storage
        self.mem = Mem(
            storage_dir=self.storage_path,
            memory_name="preliminary"
        )
        self.llm_client = LLMClient(config_file=config_file)
        self.embedding_client = EmbeddingClient(config_file=config_file)
        # Embedding cache bound to current memory library directory
        from ..utils.embedding_cache import EmbeddingCacheManager
        self.embedding_cache = EmbeddingCacheManager(
            cache_path=os.path.join(self.storage_path, "embedding_cache")
        )

        # TF-IDF related initialization
        self.tfidf_cache_path = os.path.join(
            self.storage_path, "summary_tfidf_cache")
        os.makedirs(self.tfidf_cache_path, exist_ok=True)
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        self.memory_texts = []
        self.memory_ids = []
        self._init_tfidf()

    @staticmethod
    def _tokenize_chinese(text: str) -> List[str]:
        """Chinese word segmentation function"""
        return list(jieba.cut(text))

    def write_memory_auto(self, text: str) -> Dict[str, Any]:
        """Intelligently write memory, automatically determine whether to add or update."""
        try:
            if not text or not str(text).strip():
                return {
                    'success': False,
                    'error': 'Text is empty',
                    'action': 'error'
                }

            text_str = text if isinstance(text, str) else str(text)

            # Generate summary
            summary = self._generate_summary(text_str)

            # Find similar memories
            similar_memories = self._find_similar_memories(text_str)

            # Add debug information
            logger.info(f"Similarity threshold: {self.similarity_threshold}")
            if similar_memories:
                logger.info(f"Found {len(similar_memories)} similar memories")
                for i, mem in enumerate(similar_memories[:3]):  # Only show first 3
                    logger.info(
                        f"  Similar memory {i+1}: ID={mem['id']}, similarity={mem['similarity']:.3f}")
            else:
                logger.info("No similar memories found")

            if similar_memories and similar_memories[0]['similarity'] >= self.similarity_threshold:
                # Found similar memory and similarity reaches threshold, perform update
                best_match = similar_memories[0]
                mem_id = best_match['id']  # Use 'id' instead of 'mem_cell'
                logger.info(f"Similarity reaches threshold, updating memory: {mem_id}")

                # Use intelligent update method, consistent with legacy version
                updated_result = self._update_existing_memory_intelligently(
                    mem_id=mem_id,
                    new_text=text_str,
                    new_summary=summary
                )

                # Update TF-IDF model
                self._update_tfidf_model()

                return {
                    'success': True,
                    'action': 'updated',
                    'mem_id': mem_id,
                    'similarity_score': best_match['similarity'],
                    'mem_cell': updated_result.get('mem_cell'),
                    'versioned_text': updated_result.get('versioned_text', False),
                    'text_length': updated_result.get('text_length', 0),
                    'estimated_tokens': updated_result.get('estimated_tokens', 0),
                    'was_truncated': updated_result.get('was_truncated', False)
                }
            else:
                # No similar memory or similarity not enough, create new memory
                logger.info("Similarity not reaching threshold, creating new memory")
                new_mem = self.mem.add_memory(
                    text=text_str,
                    summary=summary
                )

                # Update TF-IDF model
                self._update_tfidf_model()

                return {
                    'success': True,
                    'action': 'added',
                    'mem_id': new_mem.mem_id,
                    'mem_cell': new_mem
                }

        except Exception as e:
            logger.error(f"Failed to write memory: {e}")
            return {
                'success': False,
                'error': str(e),
                'action': 'error'
            }

    def search_memories_by_query(self, query: str, top_k: int = 5, weights: Tuple[float, float] = (0.5, 0.5)) -> List[Dict[str, Any]]:
        """Search memories by content, supporting TF-IDF and embedding hybrid search."""
        try:
            if not query or not str(query).strip():
                logger.warning("Query is empty, returning empty results")
                return []

            mem_cells = self.mem.list_all()
            if not mem_cells:
                return []

            query_str = query if isinstance(query, str) else str(query)

            # Calculate embedding similarity
            query_embedding = self._create_embedding(query_str)
            embedding_similarities = {}

            for mem_cell in mem_cells:
                summary = mem_cell.summary
                if not isinstance(summary, str):
                    summary = str(summary)
                emb = self._create_embedding(summary)
                similarity = float(np.dot(query_embedding, emb))
                embedding_similarities[mem_cell.mem_id] = similarity

            # Calculate TF-IDF similarity
            tfidf_similarities = {}
            tfidf_results = self._calculate_tfidf_similarity(query_str)
            for mem_id, similarity in tfidf_results:
                tfidf_similarities[mem_id] = similarity

            # Combine similarity scores
            candidates = []
            embedding_weight, tfidf_weight = weights

            for mem_cell in mem_cells:
                mem_id = mem_cell.mem_id
                embedding_sim = embedding_similarities.get(mem_id, 0.0)
                tfidf_sim = tfidf_similarities.get(mem_id, 0.0)

                # Weighted combination
                combined_score = (embedding_weight * embedding_sim +
                                  tfidf_weight * tfidf_sim)

                candidates.append({
                    'mem_cell': mem_cell,
                    'similarity_score': combined_score,
                    'embedding_similarity': embedding_sim,
                    'tfidf_similarity': tfidf_sim
                })

            # Sort by combined score
            candidates.sort(key=lambda x: x['similarity_score'], reverse=True)

            # Take top_k results and increase recall count
            results = []
            for candidate in candidates[:top_k]:
                mem_cell = candidate['mem_cell']
                # Increase recall count using the memory manager's method
                self.mem.increment_recall(mem_cell.mem_id)
                # Update recall count in results
                candidate['recall_count'] = mem_cell.recall_cnt + 1
                results.append(candidate)

            return results

        except Exception as e:
            logger.error(f"Search memories failed: {e}")
            return []

    def search_memories_by_time(self, target_date: str, top_k: int = 5, sort_by: Optional[str] = None) -> List[Dict[str, Any]]:
        """Search memories by time."""
        try:
            time_info = self._parse_chinese_date(target_date)
            if time_info is None:
                logger.error(f"Could not parse date format: {target_date}")
                return []

            mem_cells = self.mem.list_all()
            if not mem_cells:
                return []

            target_memories = []
            for mem_cell in mem_cells:
                if self._is_in_time_range(mem_cell.create_time, time_info):
                    result_dict = {
                        "mem_cell": mem_cell,
                        "match_type": "create_time",
                        "time_query_type": time_info["type"],
                        "recall_count": mem_cell.recall_cnt,
                        "update_time": mem_cell.update_time,
                        "create_time": mem_cell.create_time
                    }
                    target_memories.append(result_dict)
                elif (mem_cell.update_time != mem_cell.create_time and
                      self._is_in_time_range(mem_cell.update_time, time_info)):
                    result_dict = {
                        "mem_cell": mem_cell,
                        "match_type": "update_time",
                        "time_query_type": time_info["type"],
                        "recall_count": mem_cell.recall_cnt,
                        "update_time": mem_cell.update_time,
                        "create_time": mem_cell.create_time
                    }
                    target_memories.append(result_dict)

            # Sort
            if sort_by == "recall_count":
                target_memories.sort(
                    key=lambda x: x["recall_count"], reverse=True)
            elif sort_by == "update_time":
                target_memories.sort(
                    key=lambda x: x["update_time"], reverse=True)
            elif sort_by == "create_time":
                target_memories.sort(
                    key=lambda x: x["create_time"], reverse=True)
            else:
                # Default sort by recall count
                target_memories.sort(
                    key=lambda x: x["recall_count"], reverse=True)

            query_type_desc = {
                "year_only": f"{time_info['year']}",
                "year_month": f"{time_info['year']}-{time_info.get('month', '')}",
                "full_date": f"{time_info['year']}-{time_info.get('month', '')}-{time_info.get('day', '')}"
            }.get(time_info["type"], target_date)

            logger.info(f"Found {len(target_memories)} memories for {query_type_desc}")

            # Take top_k results and increase recall count
            final_results = target_memories[:top_k]
            for result in final_results:
                mem_cell = result['mem_cell']
                # Increase recall count
                self.mem.increment_recall(mem_cell.mem_id)
                # Update recall count in results
                result['recall_count'] = mem_cell.recall_cnt + 1

            return final_results

        except Exception as e:
            logger.error(f"Search memories by time failed: {e}")
            return []

    def _parse_chinese_date(self, date_str: str) -> Optional[Dict[str, Any]]:
        """
        Parse Chinese date string.
        Args:
            date_str (str): Date string.
        Returns:
            Optional[Dict[str, Any]]: Parsed result.
        """
        try:
            date_str = date_str.strip().replace(' ', '').replace('\n', '')

            # Match "2024Year" format
            year_match = re.match(r'^(\d{4})Year$', date_str)
            if year_match:
                return {
                    "type": "year_only",
                    "year": int(year_match.group(1))
                }

            # Match "2024Year3Month" format
            year_month_match = re.match(r'^(\d{4})Year(\d{1,2})Month$', date_str)
            if year_month_match:
                return {
                    "type": "year_month",
                    "year": int(year_month_match.group(1)),
                    "month": int(year_month_match.group(2))
                }

            # Match "2024Year3Month15日" format
            full_date_match = re.match(
                r'^(\d{4})Year(\d{1,2})Month(\d{1,2})Day$', date_str)
            if full_date_match:
                return {
                    "type": "full_date",
                    "year": int(full_date_match.group(1)),
                    "month": int(full_date_match.group(2)),
                    "day": int(full_date_match.group(3))
                }

            # Handle fuzzy time expressions as fallback
            now = time.localtime()
            
            if "Today" in date_str:
                return {
                    "type": "full_date",
                    "year": now.tm_year,
                    "month": now.tm_mon,
                    "day": now.tm_mday
                }
            elif "Yesterday" in date_str:
                yesterday = time.localtime(time.time() - 24*60*60)
                return {
                    "type": "full_date",
                    "year": yesterday.tm_year,
                    "month": yesterday.tm_mon,
                    "day": yesterday.tm_mday
                }
            elif "Tomorrow" in date_str:
                tomorrow = time.localtime(time.time() + 24*60*60)
                return {
                    "type": "full_date",
                    "year": tomorrow.tm_year,
                    "month": tomorrow.tm_mon,
                    "day": tomorrow.tm_mday
                }
            elif "This Month" in date_str or "This Month" in date_str:
                return {
                    "type": "year_month",
                    "year": now.tm_year,
                    "month": now.tm_mon
                }
            elif "Last Month" in date_str:
                if now.tm_mon == 1:
                    return {
                        "type": "year_month",
                        "year": now.tm_year - 1,
                        "month": 12
                    }
                else:
                    return {
                        "type": "year_month",
                        "year": now.tm_year,
                        "month": now.tm_mon - 1
                    }
            elif "This Year" in date_str:
                return {
                    "type": "year_only",
                    "year": now.tm_year
                }
            elif "Last Year" in date_str:
                return {
                    "type": "year_only",
                    "year": now.tm_year - 1
                }

            return None
        except Exception as e:
            logger.error(f"Failed to parse Chinese date: {e}")
            return None

    def _is_in_time_range(self, timestamp: float, time_info: Dict[str, Any]) -> bool:
        """
        Check if timestamp is within the specified time range.
        Args:
            timestamp (float): Timestamp.
            time_info (Dict[str, Any]): Time range information.
        Returns:
            bool: Whether it is within the range.
        """
        try:
            target_time = time.localtime(timestamp)
            if time_info["type"] == "year_only":
                return target_time.tm_year == time_info["year"]
            elif time_info["type"] == "year_month":
                return (target_time.tm_year == time_info["year"] and
                        target_time.tm_mon == time_info["month"])
            elif time_info["type"] == "full_date":
                return (target_time.tm_year == time_info["year"] and
                        target_time.tm_mon == time_info["month"] and
                        target_time.tm_mday == time_info["day"])
            return False
        except Exception as e:
            logger.error(f"Failed to check time range: {e}")
            return False

    def add_memory_with_timestamp(self, text: str, create_time: float, summary: str = None) -> Dict[str, Any]:
        """Add memory and specify creation time."""
        try:
            if summary is None:
                summary = self._generate_summary(text)
            mem_id = f"mem_{int(create_time * 1000)}"

            # Create text file path
            text_file_path = os.path.join(self.mem.text_dir, f"{mem_id}.md")

            mem_cell = MemCell(
                text_file_path=text_file_path,
                summary=summary,
                create_time=create_time,
                update_time=create_time,
                mem_id=mem_id
            )
            # Set text content (will be saved to file automatically)
            mem_cell.text = [text]

            self.mem.mem_cells.append(mem_cell)
            self.mem.mem_id_index[mem_cell.mem_id] = len(
                self.mem.mem_cells) - 1

            # Save index file
            self.mem._save()

            return {
                'action': 'create',
                'mem_id': mem_id,
                'summary': summary,
                'operation': 'created'
            }
        except Exception as e:
            logger.error(f"Failed to add memory with timestamp: {e}")
            return {
                'action': 'error',
                'error': str(e),
                'operation': 'error'
            }

    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory statistics."""
        try:
            count = len(self.mem.mem_cells)
            return {
                'memory_count': count,
                'storage_size_mb': 0.0  # TODO: Calculate storage size
            }
        except Exception as e:
            logger.error(f"Failed to get memory statistics: {e}")
            return {
                'memory_count': 0,
                'storage_size_mb': 0.0
            }

    # --- TF-IDF related methods ---
    def _init_tfidf(self):
        """Initialize TF-IDF model"""
        try:
            tfidf_model_path = os.path.join(
                self.tfidf_cache_path, "tfidf_model.pkl")
            if os.path.exists(tfidf_model_path):
                with open(tfidf_model_path, 'rb') as f:
                    data = pickle.load(f)
                    self.tfidf_vectorizer = data['vectorizer']
                    self.tfidf_matrix = data['matrix']
                    self.memory_texts = data['texts']
                    self.memory_ids = data['ids']
                logger.info(f"Loaded TF-IDF model, containing {len(self.memory_ids)} documents")
            else:
                self._build_tfidf_model()
        except Exception as e:
            logger.error(f"Failed to initialize TF-IDF: {e}")
            self._build_tfidf_model()

    def _build_tfidf_model(self):
        """Build TF-IDF model"""
        try:
            mem_cells = self.mem.list_all()
            if not mem_cells:
                self.tfidf_vectorizer = None
                self.tfidf_matrix = None
                self.memory_texts = []
                self.memory_ids = []
                return

            # Prepare text data
            texts = []
            ids = []
            for mem_cell in mem_cells:
                # Use combined summary and original text
                combined_text = f"{mem_cell.summary} {' '.join(mem_cell.text)}"
                texts.append(combined_text)
                ids.append(mem_cell.mem_id)

            # Create TF-IDF vectorizer, using class-level word segmentation function
            self.tfidf_vectorizer = TfidfVectorizer(
                tokenizer=self._tokenize_chinese,
                token_pattern=None,  # Explicitly set to None to avoid warnings
                max_features=6000,  # Reduced from 10000 to 6000 for better performance
                stop_words=None,
                ngram_range=(1, 2)
            )

            # Build TF-IDF matrix
            self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
            self.memory_texts = texts
            self.memory_ids = ids

            # Save model
            self._save_tfidf_model()
            logger.info(f"TF-IDF model built, containing {len(ids)} documents")
        except Exception as e:
            logger.error(f"Failed to build TF-IDF model: {e}")

    def _save_tfidf_model(self):
        """Save TF-IDF model"""
        try:
            if self.tfidf_vectorizer is not None:
                data = {
                    'vectorizer': self.tfidf_vectorizer,
                    'matrix': self.tfidf_matrix,
                    'texts': self.memory_texts,
                    'ids': self.memory_ids
                }
                tfidf_model_path = os.path.join(
                    self.tfidf_cache_path, "tfidf_model.pkl")
                with open(tfidf_model_path, 'wb') as f:
                    pickle.dump(data, f)
                logger.info("TF-IDF model saved successfully")
        except Exception as e:
            logger.error(f"Failed to save TF-IDF model: {e}")

    def _update_tfidf_model(self):
        """Update TF-IDF model"""
        self._build_tfidf_model()

    @lru_cache(maxsize=64)
    def _calculate_tfidf_similarity(self, query: str) -> List[Tuple[str, float]]:
        """Calculate TF-IDF similarity"""
        try:
            if self.tfidf_vectorizer is None or self.tfidf_matrix is None:
                logger.warning("TF-IDF model not initialized")
                return []

            # Vectorize query text
            query_vector = self.tfidf_vectorizer.transform([query])

            # Calculate similarity
            similarities = cosine_similarity(
                query_vector, self.tfidf_matrix).flatten()

            # Return results
            results = []
            for i, similarity in enumerate(similarities):
                if similarity > 0:
                    results.append((self.memory_ids[i], float(similarity)))

            # Sort by similarity
            results.sort(key=lambda x: x[1], reverse=True)
            return results
        except Exception as e:
            logger.error(f"Failed to calculate TF-IDF similarity: {e}")
            return []

    def update_tfidf_model(self) -> Dict[str, Any]:
        """Manually update TF-IDF model"""
        try:
            self._update_tfidf_model()
            return {
                'success': True,
                'message': 'TF-IDF model updated successfully',
                'document_count': len(self.memory_ids)
            }
        except Exception as e:
            logger.error(f"Failed to update TF-IDF model: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    def get_tfidf_stats(self) -> Dict[str, Any]:
        """Get TF-IDF model statistics"""
        try:
            if self.tfidf_vectorizer is None:
                return {
                    'model_initialized': False,
                    'document_count': 0,
                    'feature_count': 0
                }

            return {
                'model_initialized': True,
                'document_count': len(self.memory_ids),
                'feature_count': self.tfidf_vectorizer.get_feature_names_out().shape[0],
                'matrix_shape': self.tfidf_matrix.shape if self.tfidf_matrix is not None else None
            }
        except Exception as e:
            logger.error(f"Failed to get TF-IDF statistics: {e}")
            return {
                'model_initialized': False,
                'error': str(e)
            }

    # --- Internal methods ---
    def _generate_summary(self, text: str) -> str:
        """Generate summary (call LLM)"""
        try:
            response = self.llm_client.generate_response(
                prompt=text,
                system_prompt="Please generate a concise summary for the following content.",
                max_tokens=128,
                stream=False  # Force non-streaming response
            )

            # Ensure return is string
            if isinstance(response, dict) and response.get("stream"):
                # If streaming response, extract content
                return self.llm_client.get_stream_content(response)
            elif isinstance(response, str):
                return response
            else:
                # Other cases, convert to string
                return str(response)

        except Exception as e:
            logger.warning(f"Failed to generate summary: {e}")
            return text[:50]

    @lru_cache(maxsize=64)
    def _create_embedding(self, text: str) -> np.ndarray:
        """Create text embedding, prioritize cache, ensure text is str type"""
        if not isinstance(text, str):
            text = str(text)
        # Prioritize cache
        cached = self.embedding_cache.get_cached_embedding(text)
        if cached is not None:
            return cached
        emb = self.embedding_client.create_embedding(text)
        self.embedding_cache.cache_embedding(text, emb)
        return emb

    def _find_similar_memories(self, text: str) -> List[Dict[str, Any]]:
        """Find similar memories (based on embedding)"""
        if not isinstance(text, str):
            text = str(text)
        query_emb = self._create_embedding(text)
        mem_cells = self.mem.list_all()
        results = []
        for mem_cell in mem_cells:
            summary = mem_cell.summary
            if not isinstance(summary, str):
                summary = str(summary)
            emb = self._create_embedding(summary)
            similarity = float(np.dot(query_emb, emb))
            results.append({'id': mem_cell.mem_id, 'similarity': similarity})
        results.sort(key=lambda x: x['similarity'], reverse=True)
        return results

    def _update_existing_memory_intelligently(self, mem_id: str, new_text: str, new_summary: str) -> Dict[str, Any]:
        """Intelligent update of existing memory, consistent with legacy version"""
        try:
            existing_mem = self.mem.get(mem_id)
            if not existing_mem:
                return {
                    "action": "error",
                    "error": f"Memory not found: {mem_id}",
                    "mem_id": mem_id,
                    "operation": "update_failed"
                }

            # Build versioned text
            versioned_text = self._build_versioned_text(existing_mem, new_text)

            # Check token limit
            current_tokens = self._estimate_tokens(versioned_text)
            if current_tokens > self.max_tokens:
                logger.info(
                    f"Text exceeds token limit ({current_tokens} > {self.max_tokens}), truncating")
                versioned_text = self._truncate_text_to_tokens(
                    versioned_text, self.max_tokens)
                truncated_tokens = self._estimate_tokens(versioned_text)
                logger.info(f"Token count after truncation: {truncated_tokens}")

            # Generate new summary for versioned text
            updated_summary = self._generate_summary(versioned_text)

            # Update memory
            updated_mem = self.mem.update_memory(
                mem_id=mem_id,
                new_text=versioned_text,
                new_summary=updated_summary
            )

            logger.info(f"Memory updated successfully: {mem_id}")
            logger.info(f"Versioned text length: {len(versioned_text)} characters")
            logger.info(f"Estimated token count: {self._estimate_tokens(versioned_text)}")

            return {
                "action": "update",
                "mem_id": mem_id,
                "summary": updated_summary,
                "operation": "memory_updated",
                "original_summary": existing_mem.summary,
                "new_content_added": new_text,
                "text_replaced": True,
                "summary_merged": True,
                "versioned_text": True,
                "text_length": len(versioned_text),
                "estimated_tokens": self._estimate_tokens(versioned_text),
                "was_truncated": current_tokens > self.max_tokens,
                "mem_cell": updated_mem
            }

        except Exception as e:
            logger.error(f"Failed to update memory intelligently: {e}")
            return {
                "action": "error",
                "error": str(e),
                "mem_id": mem_id,
                "operation": "update_failed"
            }

    def _build_versioned_text(self, existing_mem: MemCell, new_text: str) -> str:
        """
        Build versioned text content, consistent with legacy version.
        Args:
            existing_mem (MemCell): Existing memory cell.
            new_text (str): New text.
        Returns:
            str: Versioned text.
        """
        try:
            existing_text = " ".join(
                existing_mem.text) if existing_mem.text else ""
            current_version = existing_mem.update_cnt + 1

            if existing_text:
                if existing_text.startswith("Version"):
                    version_match = re.match(r'Version(\d+)：', existing_text)
                    if version_match:
                        existing_version = int(version_match.group(1))
                        content_start = existing_text.find("：", 2) + 1
                        existing_content = existing_text[content_start:].strip(
                        )
                        versioned_text = f"Version{current_version}：{new_text}\nVersion{existing_version}：{existing_content}"
                        logger.info(
                            f"Updated from version {existing_version} to version {current_version}")
                    else:
                        versioned_text = f"Version{current_version}：{new_text}\nVersion0：{existing_text}"
                        logger.info(f"Could not parse version number, treated as version 0, new version is {current_version}")
                else:
                    versioned_text = f"Version{current_version}：{new_text}\nVersion0：{existing_text}"
                    logger.info(f"Non-versioned text, treated as version 0, new version is {current_version}")
            else:
                versioned_text = f"Version{current_version}：{new_text}"
                logger.info(f"No existing text, directly as version {current_version}")

            logger.info(f"Building versioned text, version number: {current_version}")
            logger.info(f"Existing memory update count: {existing_mem.update_cnt}")
            return versioned_text

        except Exception as e:
            logger.error(f"Failed to build versioned text: {e}")
            existing_text = " ".join(
                existing_mem.text) if existing_mem.text else ""
            return f"{new_text}\n{existing_text}".strip()

    def _create_new_memory(self, text: str, summary: str) -> Dict[str, Any]:
        """Create new memory"""
        create_time = time.time()
        mem_id = f"mem_{int(create_time * 1000)}"
        mem_cell = MemCell(
            text_file_path=os.path.join(self.mem.text_dir, f"{mem_id}.md"),
            summary=summary,
            create_time=create_time,
            update_time=create_time,
            mem_id=mem_id
        )
        mem_cell.text = [text]
        self.mem.mem_cells.append(mem_cell)
        self.mem.mem_id_index[mem_cell.mem_id] = len(self.mem.mem_cells) - 1
        return {
            'action': 'create',
            'mem_id': mem_id,
            'summary': summary,
            'operation': 'created'
        }

    def _estimate_tokens(self, text: str) -> int:
        """
        Roughly estimate the number of tokens in the text, consistent with legacy version.
        Args:
            text (str): Input text.
        Returns:
            int: Estimated token count.
        """
        if not text:
            return 0
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
        english_words = len(re.findall(r'[a-zA-Z]+', text))
        other_chars = len(text) - chinese_chars - english_words
        estimated_tokens = chinese_chars // 2 + english_words + other_chars // 4
        return max(1, estimated_tokens)

    def _truncate_text_to_tokens(self, text: str, max_tokens: int) -> str:
        """
        Truncate text to the maximum number of tokens, consistent with legacy version.
        Args:
            text (str): Input text.
            max_tokens (int): Maximum number of tokens.
        Returns:
            str: Truncated text.
        """
        if not text:
            return text
        current_tokens = self._estimate_tokens(text)
        if current_tokens <= max_tokens:
            return text
        left, right = 0, len(text)
        best_length = 0
        while left <= right:
            mid = (left + right) // 2
            truncated_text = text[:mid]
            tokens = self._estimate_tokens(truncated_text)
            if tokens <= max_tokens:
                best_length = mid
                left = mid + 1
            else:
                right = mid - 1
        return text[:best_length]
