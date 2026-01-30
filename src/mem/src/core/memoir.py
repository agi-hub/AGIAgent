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

Advanced Memory Management Module
"""

import os
import time
import json
import datetime
import numpy as np
import pickle
import functools
from typing import List, Dict, Any, Optional, Tuple, Callable
from ..utils.logger import get_logger
from ..clients.llm_client import LLMClient
from ..clients.embedding_client import EmbeddingClient
from ..utils.config import ConfigLoader
from ..utils.exceptions import MemorySystemError
from ..utils.embedding_cache import get_global_cache_manager
from ..utils.monitor import monitor_operation

logger = get_logger(__name__)


def handle_exceptions(func: Callable) -> Callable:
    """Exception handling decorator"""
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        try:
            return func(self, *args, **kwargs)
        except Exception as e:
            logger.error(f"{func.__name__} encountered an exception: {e}")
            return {
                "success": False,
                "error": str(e),
                "timestamp": time.time()
            }
    return wrapper


def cache_result(ttl_seconds: int = 300) -> Callable:
    """Cache result decorator"""
    def decorator(func: Callable) -> Callable:
        cache: Dict[str, Tuple[Any, float]] = {}

        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            cache_key = f"{func.__name__}:{hash(str(args) + str(sorted(kwargs.items())))}"
            current_time = time.time()

            if cache_key in cache:
                result, timestamp = cache[cache_key]
                if current_time - timestamp < ttl_seconds:
                    return result
                else:
                    del cache[cache_key]

            result = func(self, *args, **kwargs)
            cache[cache_key] = (result, current_time)

            # Clean up expired cache
            expired_keys = [k for k, (_, ts) in cache.items(
            ) if current_time - ts >= ttl_seconds]
            for k in expired_keys:
                del cache[k]

            return result
        return wrapper
    return decorator


class MemoirCell:
    """Memoir memory unit, compatible with preliminary_memory format"""

    def __init__(self, mem_id: str, content: str, summary: str,
                 create_time: float, update_time: float, recall_count: int = 0):
        self.mem_id = mem_id
        self.content = content
        self.summary = summary
        self.create_time = create_time
        self.update_time = update_time
        self.recall_cnt = recall_count


class MemoirManager:
    """
    Advanced Memory Manager with Intelligent Integration

    Responsible for automatically generating daily, monthly, and yearly summaries using intelligent integration strategy.
    The system preserves existing summaries and integrates them with new memory content, providing intelligent search 
    based on embedding and TF-IDF, supporting version control and incremental updates.

    Key Features:
    - Intelligent Integration: Preserves existing summaries and integrates with new content
    - Incremental Processing: Only processes unprocessed memories for efficiency
    - Hierarchical Summarization: Day → Month → Year progressive summarization
    - Version Control: Tracks processing status and update timestamps
    - Hybrid Search: Combines embedding and TF-IDF for optimal search results

    Storage structure:
    - Store by year: memoir_2025.json
    - Hierarchical structure: Year -> Month -> Day
    - Version control: Record processing status and version information
    """

    def __init__(self, storage_path: str = "memory/memoir", preliminary_memory=None, config_file: str = "config.txt"):
        self.storage_path = storage_path
        self.preliminary_memory = preliminary_memory
        self.config_loader = ConfigLoader(config_file)

        # Initialize clients
        try:
            self.llm_client = LLMClient(config_file=config_file)
            logger.info("LLM client initialized successfully")
        except Exception as e:
            logger.warning(f"LLM client initialization failed: {e}")
            self.llm_client = None

        try:
            self.embedding_client = EmbeddingClient(config_file=config_file)
            logger.info("Embedding client initialized successfully")
        except Exception as e:
            logger.warning(f"Embedding client initialization failed: {e}")
            self.embedding_client = None

        # Create storage directory
        os.makedirs(self.storage_path, exist_ok=True)

        # Initialize configuration
        self.similarity_threshold = self.config_loader.get(
            'similarity_threshold', 0.7)
        self.max_tokens = self.config_loader.get('max_tokens', 4096)

        # Cache manager
        self.cache_manager = get_global_cache_manager(self.storage_path)

        # Initialize TF-IDF related
        self.tfidf_cache_path = os.path.join(
            self.storage_path, "memoir_tfidf_cache")
        os.makedirs(self.tfidf_cache_path, exist_ok=True)
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        self.memoir_texts = []
        self.memoir_ids = []
        self._init_tfidf()

        logger.info(f"MemoirManager initialization completed, storage path: {storage_path}")

    def _init_tfidf(self):
        """Initialize TF-IDF model"""
        try:
            tfidf_model_path = os.path.join(
                self.tfidf_cache_path, "memoir_tfidf_model.pkl")
            if os.path.exists(tfidf_model_path):
                with open(tfidf_model_path, 'rb') as f:
                    data = pickle.load(f)
                    self.tfidf_vectorizer = data['vectorizer']
                    self.tfidf_matrix = data['matrix']
                    self.memoir_texts = data['texts']
                    self.memoir_ids = data['ids']
                logger.info(f"Loaded cached TF-IDF model, containing {len(self.memoir_ids)} documents")
            else:
                logger.info("No cached TF-IDF model found, will build on first use")
        except Exception as e:
            logger.error(f"TF-IDF initialization failed: {e}")
            self.tfidf_vectorizer = None
            self.tfidf_matrix = None
            self.memoir_texts = []
            self.memoir_ids = []

    @cache_result(ttl_seconds=60)
    def _get_memoir_file_path(self, year: int) -> str:
        """Get memoir file path for specified year"""
        return os.path.join(self.storage_path, f"memoir_{year}.json")

    @cache_result(ttl_seconds=60)
    def _load_memoir(self, year: int) -> Dict[str, Any]:
        """Load memoir data for specified year"""
        file_path = self._get_memoir_file_path(year)
        try:
            if os.path.exists(file_path):
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    logger.info(f"Loaded memoir file: {file_path}")
                    # Clean months, remove non-dict items
                    if "months" in data:
                        data["months"] = {
                            k: v for k, v in data["months"].items() if isinstance(v, dict)
                        }
                    return data
            else:
                # Return empty memoir structure
                return {
                    "year": year,
                    "year_summary": "",
                    "months": {},
                    "version_control": {
                        "last_update_time": 0,
                        "processed_memories": {},
                        "summary_versions": {
                            "year": 0,
                            "months": {},
                            "days": {}
                        }
                    }
                }
        except Exception as e:
            logger.error(f"Failed to load memoir file {file_path}: {e}")
            return {
                "year": year,
                "year_summary": "",
                "months": {},
                "version_control": {
                    "last_update_time": 0,
                    "processed_memories": {},
                    "summary_versions": {
                        "year": 0,
                        "months": {},
                        "days": {}
                    }
                }
            }

    def _save_memoir(self, year: int, memoir_data: Dict[str, Any]) -> bool:
        """Save memoir data for specified year"""
        file_path = self._get_memoir_file_path(year)
        try:
            # Update last_update_time before saving
            self._update_last_update_time(memoir_data)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(memoir_data, f, ensure_ascii=False, indent=2)
            logger.info(f"Saved memoir file: {file_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to save memoir file {file_path}: {e}")
            return False

    def _get_day_memories(self, target_date: str) -> List[Dict[str, Any]]:
        """Get memory list for specified date"""
        try:
            if self.preliminary_memory is None:
                logger.warning("_get_day_memories method requires basic memory manager support")
                return []
            # Get All Memories for the Day
            # Remove Cache
            return self.preliminary_memory.search_memories_by_time(target_date, top_k=10000)
        except Exception as e:
            logger.error(f"Failed to get date memories: {e}")
            return []

    def _is_memory_processed(self, memoir_data: Dict[str, Any], mem_cell: Any, date_str: str) -> bool:
        """Check if memory has been processed"""
        try:
            processed_memories = memoir_data.get(
                "version_control", {}).get("processed_memories", {})
            date_processed = processed_memories.get(date_str, [])
            return mem_cell.mem_id in date_processed
        except Exception:
            return False

    def _mark_memory_processed(self, memoir_data: Dict[str, Any], mem_cell: Any, date_str: str):
        """Mark memory as processed"""
        try:
            if "version_control" not in memoir_data:
                memoir_data["version_control"] = {}
            if "processed_memories" not in memoir_data["version_control"]:
                memoir_data["version_control"]["processed_memories"] = {}
            if date_str not in memoir_data["version_control"]["processed_memories"]:
                memoir_data["version_control"]["processed_memories"][date_str] = []

            if mem_cell.mem_id not in memoir_data["version_control"]["processed_memories"][date_str]:
                memoir_data["version_control"]["processed_memories"][date_str].append(
                    mem_cell.mem_id)
                
                # Update last_update_time when new memory is marked as processed
                self._update_last_update_time(memoir_data)
        except Exception as e:
            logger.error(f"Failed to mark memory processing status: {e}")

    def _update_summary_version(self, memoir_data: Dict[str, Any], level: str, date_str: str):
        """Update summary version"""
        try:
            if "version_control" not in memoir_data:
                memoir_data["version_control"] = {}
            if "summary_versions" not in memoir_data["version_control"]:
                memoir_data["version_control"]["summary_versions"] = {}
            if level not in memoir_data["version_control"]["summary_versions"]:
                memoir_data["version_control"]["summary_versions"][level] = {}

            current_version = memoir_data["version_control"]["summary_versions"][level].get(
                date_str, 0)
            memoir_data["version_control"]["summary_versions"][level][date_str] = current_version + 1
            
            # Update last_update_time when summary version is updated
            self._update_last_update_time(memoir_data)
        except Exception as e:
            logger.error(f"Failed to update summary version: {e}")

    def _update_last_update_time(self, memoir_data: Dict[str, Any]):
        """Update last_update_time in version_control"""
        try:
            if "version_control" not in memoir_data:
                memoir_data["version_control"] = {}
            memoir_data["version_control"]["last_update_time"] = time.time()
        except Exception as e:
            logger.error(f"Failed to update last_update_time: {e}")

    @handle_exceptions
    @monitor_operation("update_memoir_all")
    def update_memoir_all(self, force_update: bool = False, max_days_back: int = 30) -> Dict[str, Any]:
        """
        Batch update all updatable memoirs

        Args:
            force_update: Whether to force update
            max_days_back: Maximum number of days to backtrack

        Returns:
            Update result
        """
        try:
            if not self.preliminary_memory:
                logger.warning("preliminary_memory not set, cannot update memoir")
                return {
                    "success": False,
                    "error": "preliminary_memory not set",
                    "updated_dates": [],
                    "new_memories_processed": 0
                }

            # Get current time
            current_time = time.localtime()
            current_year = current_time.tm_year
            current_month = current_time.tm_mon
            current_day = current_time.tm_mday

            # Collect dates to update
            dates_to_update = self._collect_dates_to_update(
                current_year, current_month, current_day, max_days_back)

            logger.info(f"Found {len(dates_to_update)} dates to check")

            # Execute batch update
            update_results = self._execute_batch_update(
                dates_to_update, force_update)

            return update_results

        except Exception as e:
            logger.error(f"Failed to update memoir: {e}")
            return {
                "success": False,
                "error": str(e),
                "updated_dates": [],
                "new_memories_processed": 0
            }

    def _collect_dates_to_update(self, current_year: int, current_month: int, current_day: int, max_days_back: Optional[int]) -> List[Dict[str, Any]]:
        """Collect dates to update"""
        dates_to_update = []

        if max_days_back is None:
            max_days_back = 30

        # Generate dates for the last max_days_back days
        current_date = datetime.datetime(
            current_year, current_month, current_day)
        for i in range(max_days_back):
            target_date = current_date - datetime.timedelta(days=i)
            # Use Chinese format to match preliminary memory search format
            date_str = f"{target_date.year}Year{target_date.month}Month{target_date.day}Day"

            dates_to_update.append({
                "year": target_date.year,
                "month": target_date.month,
                "day": target_date.day,
                "date_str": date_str
            })

        return dates_to_update

    def _execute_batch_update(self, dates_to_update: List[Dict[str, Any]], force_update: bool) -> Dict[str, Any]:
        """Execute batch update"""
        updated_dates = []
        new_memories_processed = 0
        updated_months = set()  # Record months that need monthly summary update
        updated_years = set()   # Record years that need yearly summary update

        for date_info in dates_to_update:
            year = date_info["year"]
            month = date_info["month"]
            day = date_info["day"]
            date_str = date_info["date_str"]

            try:
                # Load memoir data for that year
                memoir_data = self._load_memoir(year)

                # Get memories for that date
                day_memories = self._get_day_memories(date_str)

                # Only process dates with memories, or process when forced
                if not day_memories and not force_update:
                    continue

                # Check if update is needed
                if self._needs_day_update(memoir_data, date_str, day_memories, force_update):
                    # Generate or update daily summary
                    existing_day_memoir = self._get_existing_day_memoir(
                        memoir_data, month, day)
                    new_day_memoir = self._generate_day_memoir(
                        date_str, existing_day_memoir, day_memories, memoir_data)

                    # Only update memoir data structure if there is actual content
                    if new_day_memoir or force_update:
                        self._update_memoir_structure(
                            memoir_data, year, month, day, new_day_memoir)

                        # Mark for monthly and yearly summary update
                        updated_months.add((year, month))
                        updated_years.add(year)

                        # Save updated data
                        if self._save_memoir(year, memoir_data):
                            updated_dates.append(date_str)
                            new_memories_processed += len(day_memories)
                            logger.info(
                                f"Successfully updated memoir for {date_str}, containing {len(day_memories)} memories")

            except Exception as e:
                logger.error(f"Failed to update memoir for {date_str}: {e}")
                continue

        # Update monthly summary
        for year, month in updated_months:
            try:
                memoir_data = self._load_memoir(year)
                # Check if the Month Has Actual Daily Summary
                months = memoir_data.get("months", {})
                month_data = months.get(str(month), {})
                days = month_data.get("days", {})
                
                # Only Generate Monthly Summary When the Month Has Actual Daily Summary
                has_daily_summaries = any(day_data.get("summary") for day_data in days.values())
                
                if has_daily_summaries and self._needs_month_update(memoir_data, year, month, force_update):
                    new_month_memoir = self._generate_month_memoir(
                        year, month, memoir_data)
                    if new_month_memoir:  # Only update if there is content
                        self._update_month_memoir(
                            memoir_data, year, month, new_month_memoir)
                        self._save_memoir(year, memoir_data)
                        logger.info(f"Successfully updated monthly summary for {year}-{month}")
            except Exception as e:
                logger.error(f"Failed to update monthly summary for {year}-{month}: {e}")

        # Update yearly summary
        for year in updated_years:
            try:
                memoir_data = self._load_memoir(year)
                # Check if the Year Has Actual Monthly Summary
                months = memoir_data.get("months", {})
                
                # Only Generate Yearly Summary When the Year Has Actual Monthly Summary
                has_monthly_summaries = any(month_data.get("month_summary") for month_data in months.values())
                
                if has_monthly_summaries and self._needs_year_update(memoir_data, year, force_update):
                    new_year_memoir = self._generate_year_memoir(
                        year, memoir_data)
                    if new_year_memoir:  # Only update if there is content
                        self._update_year_memoir(
                            memoir_data, year, new_year_memoir)
                        self._save_memoir(year, memoir_data)
                        logger.info(f"Successfully updated yearly summary for {year}")
            except Exception as e:
                logger.error(f"Failed to update yearly summary for {year}: {e}")

        # Clean Up Empty Month Nodes (Months Without Actual Daily Summary)
        for year in set([date_info["year"] for date_info in dates_to_update]):
            try:
                memoir_data = self._load_memoir(year)
                months = memoir_data.get("months", {})
                empty_months = []
                
                for month, month_data in months.items():
                    days = month_data.get("days", {})
                    # Check if the Month Has Actual Daily Summary
                    has_daily_summaries = any(day_data.get("summary") for day_data in days.values())
                    
                    # If No Daily Summary
                    if not has_daily_summaries:
                        empty_months.append(month)
                
                # Delete Empty Month Nodes
                for month in empty_months:
                    del months[month]
                    logger.info(f"Removed empty month {year}-{month}")
                
                # Save Cleaned Data
                if empty_months:
                    self._save_memoir(year, memoir_data)
                    
            except Exception as e:
                logger.error(f"Failed to clean empty months for {year}: {e}")

        return {
            "success": True,
            "updated_dates": updated_dates,
            "new_memories_processed": new_memories_processed
        }

    def _needs_day_update(self, memoir_data: Dict[str, Any], date_str: str, day_memories: List[Dict[str, Any]], force_update: bool = False) -> bool:
        """Check if daily summary needs update"""
        if force_update:
            return True

        # Check for new unprocessed memories
        for mem in day_memories:
            # Handle different memory formats
            if isinstance(mem, dict) and 'mem_cell' in mem:
                mem_cell = mem['mem_cell']
            else:
                mem_cell = mem

            if not self._is_memory_processed(memoir_data, mem_cell, date_str):
                return True

        return False

    def _get_existing_day_memoir(self, memoir_data: Dict[str, Any], month: int, day: int) -> str:
        """Get existing daily summary"""
        try:
            months = memoir_data.get("months", {})
            month_data = months.get(str(month), {})
            days = month_data.get("days", {})
            return days.get(str(day), {}).get("summary", "")
        except Exception:
            return ""

    def _generate_day_memoir(self, date_str: str, existing_memoir: str, day_memories: List[Dict[str, Any]], memoir_data: Dict[str, Any]) -> str:
        """Generate daily memoir with intelligent integration"""
        try:
            if not day_memories:
                return existing_memoir

            # Filter out unprocessed memories
            unprocessed_memories = []
            for mem in day_memories:
                # Handle different memory formats
                if isinstance(mem, dict) and 'mem_cell' in mem:
                    mem_cell = mem['mem_cell']
                else:
                    mem_cell = mem
                
                if not self._is_memory_processed(memoir_data, mem_cell, date_str):
                    unprocessed_memories.append(mem)

            # If no new memories, keep existing summary
            if not unprocessed_memories:
                if existing_memoir and existing_memoir.strip():
                    return existing_memoir
                else:
                    return f"{date_str}: Recorded {len(day_memories)} memories today."

            # Extract summaries from new memories
            new_summaries = []
            for mem in unprocessed_memories:
                # Handle different memory formats
                if isinstance(mem, dict) and 'mem_cell' in mem:
                    mem_cell = mem['mem_cell']
                else:
                    mem_cell = mem
                
                if hasattr(mem_cell, 'summary') and mem_cell.summary:
                    new_summaries.append(mem_cell.summary)
                elif hasattr(mem_cell, 'text') and mem_cell.text:
                    text = mem_cell.text[0] if isinstance(mem_cell.text, list) else str(mem_cell.text)
                    new_summaries.append(text[:100] + "..." if len(text) > 100 else text)

            if not new_summaries:
                # Even without summaries, generate basic date summary
                if existing_memoir and existing_memoir.strip():
                    return existing_memoir
                else:
                    return f"{date_str}: Recorded {len(day_memories)} memories today."

            # Build LLM prompt with intelligent integration
            new_summaries_text = "\n".join([f"- {summary}" for summary in new_summaries])

            system_prompt = """You are an intelligent memory summarization assistant. Your task is to integrate existing summaries with new memory content into concise, meaningful diary entries.

Requirements:
1. Summary should be concise and clear, highlighting important information
2. Maintain objective tone
3. Integrate existing summary with new content, avoid repetition
4. Control summary length between 100-200 characters
5. Output in English
6. Preserve important information from existing summary while incorporating new content

Please generate an integrated summary based on the provided existing summary and new memory content."""

            user_prompt = f"""Please generate a daily summary for {date_str} based on the following information:

Existing daily summary:
{existing_memoir if existing_memoir else "None"}

New memory summaries:
{new_summaries_text}

Please generate a new daily summary with the following requirements:
1. Concise and clear, within 100-200 characters
2. Highlight important information and themes
3. Maintain objectivity
4. Show chronological order and logical relationships
5. Integrate existing summary with new content, avoid repetition
6. Preserve important information from existing summary

Please return the summary content directly without any additional formatting:"""

            # Call LLM to generate integrated summary
            if self.llm_client:
                response = self.llm_client.generate_response(
                    prompt=user_prompt,
                    system_prompt=system_prompt,
                    max_tokens=300,
                    temperature=0.3
                )

                # Handle LLM response format
                if isinstance(response, dict) and response.get("stream"):
                    summary = self.llm_client.get_stream_content(response)
                elif isinstance(response, str):
                    summary = response
                elif isinstance(response, dict) and response.get('content'):
                    summary = response['content']
                else:
                    summary = str(response)

                summary = summary.strip()

                if summary:
                    # Mark all unprocessed memories as processed
                    for mem in unprocessed_memories:
                        # Handle different memory formats
                        if isinstance(mem, dict) and 'mem_cell' in mem:
                            self._mark_memory_processed(memoir_data, mem['mem_cell'], date_str)
                        else:
                            self._mark_memory_processed(memoir_data, mem, date_str)

                    # Update version
                    self._update_summary_version(memoir_data, "days", date_str)

                    logger.info(f"Generated integrated memoir for {date_str}: {summary[:50]}... (processed {len(unprocessed_memories)} new memories)")
                    return summary
                else:
                    logger.warning(f"LLM returned empty summary")

            # Fallback to simple summary if LLM fails
            memoir = f"{date_str}: Recorded {len(day_memories)} memories today."
            if new_summaries:
                memoir += f" Main content includes: {new_summaries[0][:50]}..."
            if len(new_summaries) > 1:
                memoir += f" and {len(new_summaries)} other topics."

            # Mark all unprocessed memories as processed
            for mem in unprocessed_memories:
                if isinstance(mem, dict) and 'mem_cell' in mem:
                    self._mark_memory_processed(memoir_data, mem['mem_cell'], date_str)
                else:
                    self._mark_memory_processed(memoir_data, mem, date_str)

            # Update version
            self._update_summary_version(memoir_data, "days", date_str)

            return memoir

        except Exception as e:
            logger.error(f"Failed to generate daily memoir: {e}")
            return existing_memoir if existing_memoir else f"{date_str}: Error occurred while generating summary."

    def _update_memoir_structure(self, memoir_data: Dict[str, Any], year: int, month: int, day: int, day_memoir: str):
        """Update memoir data structure"""
        try:
            # Only create entry if there is actual content
            if not day_memoir:
                return

            if "months" not in memoir_data:
                memoir_data["months"] = {}
            if str(month) not in memoir_data["months"]:
                memoir_data["months"][str(month)] = {
                    "month_summary": "", "days": {}}
            if "days" not in memoir_data["months"][str(month)]:
                memoir_data["months"][str(month)]["days"] = {}
            if str(day) not in memoir_data["months"][str(month)]["days"]:
                memoir_data["months"][str(month)]["days"][str(day)] = {}

            memoir_data["months"][str(month)]["days"][str(
                day)]["summary"] = day_memoir
            memoir_data["months"][str(month)]["days"][str(
                day)]["update_time"] = time.time()

        except Exception as e:
            logger.error(f"Failed to update memoir structure: {e}")

    def search_memoir_by_query(self, query: str, top_k: int = 5, weights: Tuple[float, float] = (0.5, 0.5)) -> List[Dict[str, Any]]:
        """Search advanced memories by query, using embedding and TF-IDF hybrid search."""
        try:
            if not query or not str(query).strip():
                logger.warning("Query is empty, returning empty results")
                return []

            # 1. Get all memoir summaries and their unique IDs
            summaries, ids, meta = self._get_all_summaries_and_ids()
            if not summaries:
                return []

            # 2. Batch calculate embedding similarities (optimized version)
            embedding_similarities = self._calculate_embedding_similarities_batch(query, summaries, ids)

            # 3. Calculate TF-IDF similarities
            tfidf_similarities = {}
            tfidf_results = self._calculate_tfidf_similarity(query, summaries, ids)
            for mem_id, similarity in tfidf_results:
                tfidf_similarities[mem_id] = similarity

            # 4. Combine scores
            embedding_weight, tfidf_weight = weights
            candidates = []
            for idx, summary in enumerate(summaries):
                mem_id = ids[idx]
                embedding_sim = embedding_similarities.get(mem_id, 0.0)
                tfidf_sim = tfidf_similarities.get(mem_id, 0.0)
                combined_score = (embedding_weight * embedding_sim + tfidf_weight * tfidf_sim)
                candidates.append({
                    'mem_cell': meta[mem_id]['mem_cell'],
                    'similarity_score': combined_score,
                    'embedding_similarity': embedding_sim,
                    'tfidf_similarity': tfidf_sim,
                    'level': meta[mem_id]['level'],
                    'year': meta[mem_id].get('year'),
                    'month': meta[mem_id].get('month'),
                    'day': meta[mem_id].get('day')
                })

            # 5. Sort and return top_k
            candidates.sort(key=lambda x: x['similarity_score'], reverse=True)
            return candidates[:top_k]

        except Exception as e:
            logger.error(f"Failed to search memoir: {e}")
            return []

    def _calculate_embedding_similarities_batch(self, query: str, summaries: list, ids: list) -> Dict[str, float]:
        """Batch calculate embedding similarities, optimized for cache usage"""
        try:
            # Get query embedding
            query_embedding = self._create_embedding(query)
            
            # Batch get embeddings for all summaries (prioritize cache)
            summary_embeddings = []
            for summary in summaries:
                emb = self._create_embedding(summary)
                summary_embeddings.append(emb)
            
            # Batch calculate similarities
            embedding_similarities = {}
            for idx, emb in enumerate(summary_embeddings):
                similarity = float(np.dot(query_embedding, emb))
                embedding_similarities[ids[idx]] = similarity
            
            return embedding_similarities
            
        except Exception as e:
            logger.error(f"Failed to calculate embedding similarities: {e}")
            return {}

    def _get_all_summaries_and_ids(self) -> Tuple[list, list, dict]:
        """Get all memoir summaries and their unique IDs and metadata"""
        summaries = []
        ids = []
        meta = {}
        memoir_files = self._get_all_memoir_files()
        for year in memoir_files:
            memoir_data = self._load_memoir(year)
            if not memoir_data:
                continue
            # Year summary
            year_summary = memoir_data.get("year_summary", "")
            if year_summary:
                mem_id = f"{year}_year"
                cell = MemoirCell(mem_id, year_summary, year_summary, time.time(), time.time(), 0)
                summaries.append(year_summary)
                ids.append(mem_id)
                meta[mem_id] = {'mem_cell': cell, 'level': 'year', 'year': year}
            # Month summary
            months = memoir_data.get("months", {})
            for month, month_data in months.items():
                month_summary = month_data.get("month_summary", "")
                if month_summary:
                    mem_id = f"{year}_{month}_month"
                    cell = MemoirCell(mem_id, month_summary, month_summary, time.time(), time.time(), 0)
                    summaries.append(month_summary)
                    ids.append(mem_id)
                    meta[mem_id] = {'mem_cell': cell, 'level': 'month', 'year': year, 'month': int(month)}
                # Day summary
                days = month_data.get("days", {})
                for day, day_data in days.items():
                    day_summary = day_data.get("summary", "")
                    if day_summary:
                        mem_id = f"{year}_{month}_{day}_day"
                        cell = MemoirCell(mem_id, day_summary, day_summary, time.time(), time.time(), 0)
                        summaries.append(day_summary)
                        ids.append(mem_id)
                        meta[mem_id] = {'mem_cell': cell, 'level': 'day', 'year': year, 'month': int(month), 'day': int(day)}
        return summaries, ids, meta

    def _create_embedding(self, text: str) -> np.ndarray:
        """Create text embedding, prioritize cache"""
        if not isinstance(text, str):
            text = str(text)
        cached = self.cache_manager.get_cached_embedding(text)
        if cached is not None:
            return cached
        emb = self.embedding_client.create_embedding(text)
        self.cache_manager.cache_embedding(text, emb)
        return emb

    def _calculate_tfidf_similarity(self, query: str, summaries: list, ids: list) -> list:
        """Calculate TF-IDF similarity, return [(id, similarity)]"""
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        
        # Check if TF-IDF model needs update
        if self._needs_tfidf_update(summaries, ids):
            self._build_tfidf_model(summaries, ids)
        
        if self.tfidf_vectorizer is None or self.tfidf_matrix is None:
            logger.warning("TF-IDF model not available")
            return []
        
        # Use cached TF-IDF model to calculate similarity
        query_vec = self.tfidf_vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        
        results = []
        for i, sim in enumerate(similarities):
            if sim > 0:
                results.append((ids[i], float(sim)))
        results.sort(key=lambda x: x[1], reverse=True)
        return results

    def _needs_tfidf_update(self, summaries: list, ids: list) -> bool:
        """Check if TF-IDF model needs update"""
        # Check if text count and IDs have changed
        if (len(summaries) != len(self.memoir_texts) or 
            len(ids) != len(self.memoir_ids) or
            self.tfidf_vectorizer is None or 
            self.tfidf_matrix is None):
            return True
        
        # Check if content has changed
        for i, (summary, mem_id) in enumerate(zip(summaries, ids)):
            if (i >= len(self.memoir_texts) or 
                i >= len(self.memoir_ids) or
                summary != self.memoir_texts[i] or 
                mem_id != self.memoir_ids[i]):
                return True
        
        return False

    def _build_tfidf_model(self, summaries: list, ids: list):
        """Build and cache TF-IDF model"""
        try:
            if not summaries:
                self.tfidf_vectorizer = None
                self.tfidf_matrix = None
                self.memoir_texts = []
                self.memoir_ids = []
                return

            # Create TF-IDF vectorizer
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=6000,  # Reduced from 10000 to 6000 for better performance
                stop_words=None
            )

            # Build TF-IDF matrix
            self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(summaries)
            self.memoir_texts = summaries.copy()
            self.memoir_ids = ids.copy()

            # Save model to cache
            self._save_tfidf_model()
            logger.info(f"TF-IDF model built and cached, containing {len(ids)} documents")
            
        except Exception as e:
            logger.error(f"Failed to build TF-IDF model: {e}")

    def _save_tfidf_model(self):
        """Save TF-IDF model to cache"""
        try:
            if self.tfidf_vectorizer is not None:
                data = {
                    'vectorizer': self.tfidf_vectorizer,
                    'matrix': self.tfidf_matrix,
                    'texts': self.memoir_texts,
                    'ids': self.memoir_ids
                }
                tfidf_model_path = os.path.join(
                    self.tfidf_cache_path, "memoir_tfidf_model.pkl")
                with open(tfidf_model_path, 'wb') as f:
                    pickle.dump(data, f)
                logger.info("TF-IDF model saved to cache")
        except Exception as e:
            logger.error(f"Failed to save TF-IDF model: {e}")

    @handle_exceptions
    @monitor_operation("search_memoir_by_time")
    def search_memoir_by_time(self, target_date: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search advanced memories by time"""
        try:
            # Parse date
            time_info = self._parse_chinese_date(target_date)
            if not time_info:
                return []

            # Get all memoir files for all years
            memoir_files = self._get_all_memoir_files()
            if not memoir_files:
                return []

            target_memories = []

            # Traverse memoirs for all years
            for year in memoir_files:
                memoir_data = self._load_memoir(year)
                if not memoir_data:
                    continue

                # Search according to time query type
                if time_info["type"] == "year_only":
                    # Search the whole year
                    if year == time_info["year"]:
                        year_memories = self._extract_year_memories(
                            memoir_data, time_info)
                        target_memories.extend(year_memories)

                elif time_info["type"] == "year_month":
                    # Search for the specified year and month
                    if year == time_info["year"]:
                        month_memories = self._extract_month_memories(
                            memoir_data, time_info)
                        target_memories.extend(month_memories)

                elif time_info["type"] == "full_date":
                    # Search for the specified date
                    if year == time_info["year"]:
                        day_memories = self._extract_day_memories(
                            memoir_data, time_info)
                        target_memories.extend(day_memories)

            return target_memories[:top_k]

        except Exception as e:
            logger.error(f"Failed to search memoir by time: {e}")
            return []

    def _extract_year_memories(self, memoir_data: Dict[str, Any], time_info: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract yearly memories"""
        results = []
        try:
            year_summary = memoir_data.get("year_summary", "")
            if year_summary:
                results.append({
                    'mem_cell': MemoirCell(
                        mem_id=f"{time_info['year']}_year",
                        content=year_summary,
                        summary=year_summary,
                        create_time=time.time(),
                        update_time=time.time(),
                        recall_count=0
                    ),
                    'similarity_score': 1.0,
                    'level': 'year',
                    'year': time_info['year']
                })
        except Exception as e:
            logger.error(f"Failed to extract yearly memories: {e}")
        return results

    def _extract_month_memories(self, memoir_data: Dict[str, Any], time_info: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract monthly memories"""
        results = []
        try:
            months = memoir_data.get("months", {})
            month = str(time_info.get("month", ""))
            if month in months:
                month_summary = months[month].get("month_summary", "")
                if month_summary:
                    results.append({
                        'mem_cell': MemoirCell(
                            mem_id=f"{time_info['year']}_{month}_month",
                            content=month_summary,
                            summary=month_summary,
                            create_time=time.time(),
                            update_time=time.time(),
                            recall_count=0
                        ),
                        'similarity_score': 1.0,
                        'level': 'month',
                        'year': time_info['year'],
                        'month': time_info['month']
                    })
        except Exception as e:
            logger.error(f"Failed to extract monthly memories: {e}")
        return results

    def _extract_day_memories(self, memoir_data: Dict[str, Any], time_info: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract daily memories"""
        results = []
        try:
            months = memoir_data.get("months", {})
            month = str(time_info.get("month", ""))
            day = str(time_info.get("day", ""))

            if month in months:
                days = months[month].get("days", {})
                if day in days:
                    day_summary = days[day].get("summary", "")
                    if day_summary:
                        results.append({
                            'mem_cell': MemoirCell(
                                mem_id=f"{time_info['year']}_{month}_{day}_day",
                                content=day_summary,
                                summary=day_summary,
                                create_time=time.time(),
                                update_time=time.time(),
                                recall_count=0
                            ),
                            'similarity_score': 1.0,
                            'level': 'day',
                            'year': time_info['year'],
                            'month': time_info['month'],
                            'day': time_info['day']
                        })
        except Exception as e:
            logger.error(f"Failed to extract daily memories: {e}")
        return results

    @cache_result(ttl_seconds=1800)
    def _parse_chinese_date(self, date_str: str) -> Optional[Dict[str, Any]]:
        """Parse date string (supports both Chinese and English formats)"""
        try:
            # Support multiple date formats
            date_formats = [
                ("%YYear%mMonth%dDay", "full_date"),
                ("%YYear%mMonth", "year_month"),
                ("%YYear", "year_only"),
                ("%Y-%m-%d", "full_date"),
                ("%Y/%m/%d", "full_date"),
                ("%B %d, %Y", "full_date"),
                ("%b %d, %Y", "full_date"),
                ("%Y-%m", "year_month"),
                ("%Y", "year_only")
            ]

            for fmt, date_type in date_formats:
                try:
                    parsed_date = datetime.datetime.strptime(date_str, fmt)
                    result = {
                        "type": date_type,
                        "year": parsed_date.year,
                        "month": parsed_date.month if date_type != "year_only" else None,
                        "day": parsed_date.day if date_type == "full_date" else None
                    }
                    return result
                except ValueError:
                    continue

            # Handle fuzzy time expressions
            now = datetime.datetime.now()

            # Chinese expressions
            if "Today" in date_str or "today" in date_str.lower():
                return {
                    "type": "full_date",
                    "year": now.year,
                    "month": now.month,
                    "day": now.day
                }
            elif "Yesterday" in date_str or "yesterday" in date_str.lower():
                yesterday = now - datetime.timedelta(days=1)
                return {
                    "type": "full_date",
                    "year": yesterday.year,
                    "month": yesterday.month,
                    "day": yesterday.day
                }
            elif "Tomorrow" in date_str or "tomorrow" in date_str.lower():
                tomorrow = now + datetime.timedelta(days=1)
                return {
                    "type": "full_date",
                    "year": tomorrow.year,
                    "month": tomorrow.month,
                    "day": tomorrow.day
                }
            elif "This Month" in date_str or "This Month" in date_str or "this month" in date_str.lower():
                return {
                    "type": "year_month",
                    "year": now.year,
                    "month": now.month,
                    "day": None
                }
            elif "Last Month" in date_str or "last month" in date_str.lower():
                if now.month == 1:
                    return {
                        "type": "year_month",
                        "year": now.year - 1,
                        "month": 12,
                        "day": None
                    }
                else:
                    return {
                        "type": "year_month",
                        "year": now.year,
                        "month": now.month - 1,
                        "day": None
                    }
            elif "This Year" in date_str or "this year" in date_str.lower():
                return {
                    "type": "year_only",
                    "year": now.year,
                    "month": None,
                    "day": None
                }
            elif "Last Year" in date_str or "last year" in date_str.lower():
                return {
                    "type": "year_only",
                    "year": now.year - 1,
                    "month": None,
                    "day": None
                }
            elif "This Week" in date_str or "This Week" in date_str or "this week" in date_str.lower():
                return {
                    "type": "year_month",
                    "year": now.year,
                    "month": now.month,
                    "day": None
                }
            elif "Last Week" in date_str or "last week" in date_str.lower():
                last_week = now - datetime.timedelta(weeks=1)
                return {
                    "type": "year_month",
                    "year": last_week.year,
                    "month": last_week.month,
                    "day": None
                }

            return None

        except Exception as e:
            logger.error(f"Failed to parse date: {e}")
            return None

    @cache_result(ttl_seconds=3600)
    def _get_all_memoir_files(self) -> List[int]:
        """Get all years corresponding to memoir files"""
        try:
            years = []
            for filename in os.listdir(self.storage_path):
                if filename.startswith("memoir_") and filename.endswith(".json"):
                    try:
                        year = int(filename[7:-5])  # Extract year
                        years.append(year)
                    except ValueError:
                        continue
            return sorted(years)
        except Exception as e:
            logger.error(f"Failed to get memoir file list: {e}")
            return []

    def _calculate_simple_similarity(self, query: str, text: str) -> float:
        """Calculate simple text similarity"""
        try:
            # Improved Chinese word segmentation processing
            import re
            
            # Convert query and text to lowercase
            query_lower = query.lower()
            text_lower = text.lower()
            
            # Use regex for simple Chinese word segmentation
            def tokenize_chinese(text):
                # Separate Chinese characters and English words
                chinese_chars = re.findall(r'[\u4e00-\u9fff]', text)
                english_words = re.findall(r'[a-zA-Z]+', text)
                return set(chinese_chars + english_words)
            
            query_words = tokenize_chinese(query_lower)
            text_words = tokenize_chinese(text_lower)

            if not query_words or not text_words:
                return 0.0

            # Calculate intersection and union
            intersection = query_words.intersection(text_words)
            union = query_words.union(text_words)

            # Calculate Jaccard similarity
            jaccard_similarity = len(intersection) / len(union) if union else 0.0
            
            # If Jaccard similarity is 0, try partial matching
            if jaccard_similarity == 0.0:
                # Check if each character in query appears in text
                partial_matches = 0
                for char in query_words:
                    if any(char in word for word in text_words):
                        partial_matches += 1
                
                if partial_matches > 0:
                    partial_similarity = partial_matches / len(query_words)
                    return partial_similarity * 0.5  # Reduce weight for partial matches
            
            return jaccard_similarity

        except Exception as e:
            logger.error(f"Error calculating similarity: {e}")
            return 0.0

    @handle_exceptions
    @monitor_operation("get_memory_stats")
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get advanced memory statistics"""
        try:
            memoir_files = self._get_all_memoir_files()

            total_size = 0
            total_entries = 0

            for year in memoir_files:
                memoir_data = self._load_memoir(year)
                if memoir_data:
                    # Count yearly summaries
                    if memoir_data.get("year_summary"):
                        total_entries += 1

                    # Count monthly summaries
                    months = memoir_data.get("months", {})
                    for month_data in months.values():
                        if month_data.get("month_summary"):
                            total_entries += 1

                        # Count daily summaries
                        days = month_data.get("days", {})
                        for day_data in days.values():
                            if day_data.get("summary"):
                                total_entries += 1

            # Calculate storage size
            for filename in os.listdir(self.storage_path):
                if filename.endswith('.json'):
                    file_path = os.path.join(self.storage_path, filename)
                    if os.path.exists(file_path):
                        total_size += os.path.getsize(file_path)

            return {
                "total_entries": total_entries,
                "storage_size_mb": round(total_size / (1024 * 1024), 2)
            }

        except Exception as e:
            logger.error(f"Failed to get memoir statistics: {e}")
            return {
                "total_entries": 0,
                "storage_size_mb": 0.0
            }

    def _needs_month_update(self, memoir_data: Dict[str, Any], year: int, month: int, force_update: bool = False) -> bool:
        """Check if monthly summary needs to be updated"""
        if force_update:
            return True

        # Check if monthly summary is empty or version needs to be updated
        months = memoir_data.get("months", {})
        month_data = months.get(str(month), {})
        month_summary = month_data.get("month_summary", "")

        # If monthly summary is empty, needs update
        if not month_summary:
            return True

        # Check if there are new daily summaries
        days = month_data.get("days", {})
        for day_data in days.values():
            if day_data.get("summary") and not day_data.get("month_processed", False):
                return True

        return False

    def _needs_year_update(self, memoir_data: Dict[str, Any], year: int, force_update: bool = False) -> bool:
        """Check if yearly summary needs to be updated"""
        if force_update:
            return True

        # Check if yearly summary is empty
        year_summary = memoir_data.get("year_summary", "")
        if not year_summary:
            return True

        # Check if there are new monthly summaries
        months = memoir_data.get("months", {})
        for month_data in months.values():
            if month_data.get("month_summary") and not month_data.get("year_processed", False):
                return True

        return False

    def _generate_month_memoir(self, year: int, month: int, memoir_data: Dict[str, Any]) -> str:
        """Generate monthly memoir with intelligent integration"""
        try:
            months = memoir_data.get("months", {})
            month_data = months.get(str(month), {})
            days = month_data.get("days", {})

            # Get existing monthly summary
            existing_month_memoir = month_data.get("month_summary", "")

            # Collect all daily summaries for the month
            day_summaries = []
            for day, day_data in days.items():
                day_summary = day_data.get("summary", "")
                if day_summary:
                    day_summaries.append(f"Day {day}: {day_summary}")

            if not day_summaries:
                return existing_month_memoir if existing_month_memoir else f"{year}Year{month}Month: No records for this month."

            # Build LLM prompt with intelligent integration
            summaries_content = "\n".join(day_summaries)

            system_prompt = """You are an intelligent memory summarization assistant. Your task is to integrate existing monthly summaries with daily summaries into concise, meaningful monthly summaries.

Requirements:
1. Summary should be concise and clear, highlighting monthly important events and trends
2. Maintain objective tone
3. Organize content by theme or time
4. Control summary length between 150-250 characters
5. Output in English
6. Integrate existing monthly summary with daily summaries, avoid repetition
7. Preserve important information from existing summary while incorporating new daily content

Please generate an integrated monthly summary based on the provided existing monthly summary and daily summaries."""

            user_prompt = f"""Please generate a monthly summary for {year}Year{month}Month based on the following information:

Existing monthly summary:
{existing_month_memoir if existing_month_memoir else "None"}

Daily summaries for this month:
{summaries_content}

Please generate a new monthly summary with the following requirements:
1. Concise and clear, within 150-250 characters
2. Highlight monthly themes and important events
3. Show time progression and trends
4. Maintain objectivity
5. Integrate existing monthly summary with new daily content, avoid repetition
6. Preserve important information from existing summary

Please return the summary content directly without any additional formatting:"""

            # Call LLM to generate integrated summary
            if self.llm_client:
                response = self.llm_client.generate_response(
                    prompt=user_prompt,
                    system_prompt=system_prompt,
                    max_tokens=400,
                    temperature=0.3
                )

                # Handle LLM response format
                if isinstance(response, dict) and response.get("stream"):
                    summary = self.llm_client.get_stream_content(response)
                elif isinstance(response, str):
                    summary = response
                elif isinstance(response, dict) and response.get('content'):
                    summary = response['content']
                else:
                    summary = str(response)

                summary = summary.strip()

                if summary:
                    # Mark all daily summaries as processed
                    for day_data in days.values():
                        if day_data.get("summary"):
                            day_data["month_processed"] = True

                    # Update version
                    self._update_summary_version(
                        memoir_data, "months", f"{year}Year{month}Month")

                    logger.info(f"Generated integrated monthly memoir for {year}Year{month}Month: {summary[:50]}...")
                    return summary
                else:
                    logger.warning(f"LLM returned empty monthly summary")

            # Fallback to simple summary if LLM fails
            memoir = f"{year}Year{month}Month: Recorded content for {len(day_summaries)} days this month."
            if day_summaries:
                memoir += f" Main activities include: {day_summaries[0][:50]}..."
            if len(day_summaries) > 1:
                memoir += f" and {len(day_summaries)} other topics."

            # Mark all daily summaries as processed
            for day_data in days.values():
                if day_data.get("summary"):
                    day_data["month_processed"] = True

            # Update version
            self._update_summary_version(memoir_data, "months", f"{year}Year{month}Month")

            return memoir

        except Exception as e:
            logger.error(f"Failed to generate monthly memoir: {e}")
            return existing_month_memoir if existing_month_memoir else f"{year}Year{month}Month: Error occurred while generating summary."

    def _generate_year_memoir(self, year: int, memoir_data: Dict[str, Any]) -> str:
        """Generate yearly memoir with intelligent integration"""
        try:
            months = memoir_data.get("months", {})

            # Get existing yearly summary
            existing_year_memoir = memoir_data.get("year_summary", "")

            # Collect all monthly summaries for the year
            month_summaries = []
            for month, month_data in months.items():
                month_summary = month_data.get("month_summary", "")
                if month_summary:
                    month_summaries.append(f"Month {month}: {month_summary}")

            if not month_summaries:
                return existing_year_memoir if existing_year_memoir else f"{year}: No records for this year."

            # Build LLM prompt with intelligent integration
            summaries_content = "\n".join(month_summaries)

            system_prompt = """You are an intelligent memory summarization assistant. Your task is to integrate existing annual summaries with monthly summaries into concise, meaningful annual summaries.

Requirements:
1. Summary should be concise and clear, highlighting annual important events, achievements, and trends
2. Maintain objective tone
3. Organize content by theme or time
4. Control summary length between 200-300 characters
5. Output in English
6. Integrate existing annual summary with monthly summaries, avoid repetition
7. Preserve important information from existing summary while incorporating new monthly content

Please generate an integrated annual summary based on the provided existing annual summary and monthly summaries."""

            user_prompt = f"""Please generate an annual summary for {year} based on the following information:

Existing annual summary:
{existing_year_memoir if existing_year_memoir else "None"}

Monthly summaries for this year:
{summaries_content}

Please generate a new annual summary with the following requirements:
1. Concise and clear, within 200-300 characters
2. Highlight annual important events, achievements, and trends
3. Show time progression and changes
4. Maintain objectivity
5. Integrate existing annual summary with new monthly content, avoid repetition
6. Preserve important information from existing summary

Please return the summary content directly without any additional formatting:"""

            # Call LLM to generate integrated summary
            if self.llm_client:
                response = self.llm_client.generate_response(
                    prompt=user_prompt,
                    system_prompt=system_prompt,
                    max_tokens=500,
                    temperature=0.3
                )

                # Handle LLM response format
                if isinstance(response, dict) and response.get("stream"):
                    summary = self.llm_client.get_stream_content(response)
                elif isinstance(response, str):
                    summary = response
                elif isinstance(response, dict) and response.get('content'):
                    summary = response['content']
                else:
                    summary = str(response)

                summary = summary.strip()

                if summary:
                    # Mark all monthly summaries as processed
                    for month_data in months.values():
                        if month_data.get("month_summary"):
                            month_data["year_processed"] = True

                    # Update version
                    self._update_summary_version(
                        memoir_data, "year", str(year))

                    logger.info(f"Generated integrated yearly memoir for {year}: {summary[:50]}...")
                    return summary
                else:
                    logger.warning(f"LLM returned empty annual summary")

            # Fallback to simple summary if LLM fails
            memoir = f"{year}: Recorded content for {len(month_summaries)} months this year."
            if month_summaries:
                memoir += f" Main activities include: {month_summaries[0][:50]}..."
            if len(month_summaries) > 1:
                memoir += f" and {len(month_summaries)} other topics."

            # Mark all monthly summaries as processed
            for month_data in months.values():
                if month_data.get("month_summary"):
                    month_data["year_processed"] = True

            # Update version
            self._update_summary_version(memoir_data, "year", str(year))

            return memoir

        except Exception as e:
            logger.error(f"Failed to generate annual memoir: {e}")
            return existing_year_memoir if existing_year_memoir else f"{year}: Error occurred while generating summary."

    def _update_month_memoir(self, memoir_data: Dict[str, Any], year: int, month: int, month_memoir: str):
        """Update monthly memoir"""
        try:
            # Only create entry if there is actual content
            if not month_memoir:
                return

            if "months" not in memoir_data:
                memoir_data["months"] = {}
            if str(month) not in memoir_data["months"]:
                memoir_data["months"][str(month)] = {
                    "month_summary": "", "days": {}}

            memoir_data["months"][str(month)]["month_summary"] = month_memoir
            memoir_data["months"][str(month)]["update_time"] = time.time()

        except Exception as e:
            logger.error(f"Failed to update monthly memoir: {e}")

    def _update_year_memoir(self, memoir_data: Dict[str, Any], year: int, year_memoir: str):
        """Update annual memoir"""
        try:
            # Only create entry if there is actual content
            if not year_memoir:
                return

            memoir_data["year_summary"] = year_memoir
            memoir_data["year_update_time"] = time.time()

        except Exception as e:
            logger.error(f"Failed to update annual memoir: {e}")

    def clean_empty_entries(self, year: int = None) -> Dict[str, Any]:
        """Clean up empty memory entries"""
        try:
            if year is None:
                # Clean all years
                years = self._get_all_memoir_files()
            else:
                years = [year]

            cleaned_count = 0

            for target_year in years:
                memoir_data = self._load_memoir(target_year)
                if not memoir_data:
                    continue

                # Clean empty monthly summaries
                months = memoir_data.get("months", {})
                empty_months = []
                for month, month_data in months.items():
                    month_summary = month_data.get("month_summary", "")
                    days = month_data.get("days", {})

                    # Remove empty daily summaries
                    empty_days = []
                    for day, day_data in days.items():
                        if not day_data.get("summary"):
                            empty_days.append(day)

                    for day in empty_days:
                        del days[day]
                        cleaned_count += 1

                    # If monthly summary is empty and no daily summaries, mark for deletion
                    if not month_summary and not days:
                        empty_months.append(month)

                # Remove empty months
                for month in empty_months:
                    del months[month]
                    cleaned_count += 1

                # Save cleaned data
                if self._save_memoir(target_year, memoir_data):
                    logger.info(f"Cleaned {target_year} of {cleaned_count} empty entries")

            return {
                "success": True,
                "cleaned_count": cleaned_count,
                "years_processed": len(years)
            }

        except Exception as e:
            logger.error(f"Failed to clean empty entries: {e}")
            return {
                "success": False,
                "error": str(e),
                "cleaned_count": 0
            }
