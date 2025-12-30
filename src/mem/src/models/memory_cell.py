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

Memory Cell Data Model
"""

import os
import json
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field


@dataclass
class MemCell:
    """
    Memory cell class, representing an independent memory fragment
    """

    # Recall count overflow protection threshold (int32 maximum value)
    RECALL_COUNT_MAX = 2147483647

    text_file_path: str = ""
    summary: str = ""
    summary_embedding_path: str = ""
    recall_cnt: int = 0
    create_time: Optional[float] = None
    update_time: Optional[float] = None
    update_cnt: int = 0
    mem_id: Optional[str] = None
    _text: List[str] = field(default_factory=list, init=False)

    def __post_init__(self):
        """Post-initialization processing"""
        if self.create_time is None:
            self.create_time = time.time()
        if self.update_time is None:
            self.update_time = self.create_time
        if self.mem_id is None:
            self.mem_id = self._generate_mem_id()

    def _generate_mem_id(self) -> str:
        """Generate memory ID"""
        timestamp = int(self.create_time * 1000)
        random_suffix = int(time.time() * 1000000) % 10000
        return f"mem_{timestamp}_{random_suffix}"

    def _save_text_to_file(self, text: List[str]) -> None:
        """Save text to file"""
        if not text or not self.text_file_path:
            return

        # Ensure directory exists
        os.makedirs(os.path.dirname(self.text_file_path), exist_ok=True)

        # Save text
        with open(self.text_file_path, 'w', encoding='utf-8') as f:
            for line in text:
                f.write(line + '\n')

    def _load_text_from_file(self) -> List[str]:
        """Load text from file"""
        if not self.text_file_path or not os.path.exists(self.text_file_path):
            return []

        try:
            with open(self.text_file_path, 'r', encoding='utf-8') as f:
                return [line.strip() for line in f.readlines() if line.strip()]
        except Exception:
            return []

    @property
    def text(self) -> List[str]:
        """Get text content (read from file)"""
        if not self._text and self.text_file_path:
            self._text = self._load_text_from_file()
        return self._text

    @text.setter
    def text(self, value: List[str]):
        """Set text content (save to file)"""
        self._text = value
        if value and self.text_file_path:
            self._save_text_to_file(value)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "text_file_path": self.text_file_path,
            "summary": self.summary,
            "summary_embedding_path": self.summary_embedding_path,
            "recall_cnt": self.recall_cnt,
            "create_time": self.create_time,
            "update_time": self.update_time,
            "update_cnt": self.update_cnt,
            "mem_id": self.mem_id
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MemCell':
        """Create instance from dictionary"""
        return cls(
            text_file_path=data.get("text_file_path", ""),
            summary=data.get("summary", ""),
            summary_embedding_path=data.get("summary_embedding_path", ""),
            recall_cnt=data.get("recall_cnt", 0),
            create_time=data.get("create_time"),
            update_time=data.get("update_time"),
            update_cnt=data.get("update_cnt", 0),
            mem_id=data.get("mem_id")
        )

    def update(self, text: List[str] = None, summary: str = None) -> None:
        """Update memory cell"""
        if text is not None:
            self.text = text
        if summary is not None:
            self.summary = summary

        self.update_time = time.time()
        self.update_cnt += 1

    def increment_recall(self) -> None:
        """Increment recall count"""
        if self.recall_cnt < self.RECALL_COUNT_MAX:
            self.recall_cnt += 1

    def get_create_time_str(self) -> str:
        """Get creation time string"""
        return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.create_time))

    def get_update_time_str(self) -> str:
        """Get update time string"""
        return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.update_time))

    def __str__(self) -> str:
        return f"MemCell(id={self.mem_id}, summary='{self.summary[:50]}...')"

    def __repr__(self) -> str:
        return self.__str__()


@dataclass
class MemoirEntry:
    """
    Memoir entry class, representing an entry in advanced memory organization
    """

    date: str
    content: str
    summary: str = ""
    embedding_path: str = ""
    create_time: Optional[float] = None
    update_time: Optional[float] = None
    version: int = 1

    def __post_init__(self):
        """Post-initialization processing"""
        if self.create_time is None:
            self.create_time = time.time()
        if self.update_time is None:
            self.update_time = self.create_time

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "date": self.date,
            "content": self.content,
            "summary": self.summary,
            "embedding_path": self.embedding_path,
            "create_time": self.create_time,
            "update_time": self.update_time,
            "version": self.version
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MemoirEntry':
        """Create instance from dictionary"""
        return cls(
            date=data.get("date", ""),
            content=data.get("content", ""),
            summary=data.get("summary", ""),
            embedding_path=data.get("embedding_path", ""),
            create_time=data.get("create_time"),
            update_time=data.get("update_time"),
            version=data.get("version", 1)
        )

    def update(self, content: str = None, summary: str = None) -> None:
        """Update entry"""
        if content is not None:
            self.content = content
        if summary is not None:
            self.summary = summary

        self.update_time = time.time()
        self.version += 1

    def get_create_time_str(self) -> str:
        """Get creation time string"""
        return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.create_time))

    def get_update_time_str(self) -> str:
        """Get update time string"""
        return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.update_time))

    def __str__(self) -> str:
        return f"MemoirEntry(date={self.date}, summary='{self.summary[:50]}...')"

    def __repr__(self) -> str:
        return self.__str__()
