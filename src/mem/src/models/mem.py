"""
Low-level Memory Storage and Management Module
"""

import os
import json
import time
import numpy as np
from typing import List, Dict, Any, Optional
from .memory_cell import MemCell


class Mem:
    """
    Memory storage management class, responsible for low-level memory cell operations.
    """

    def __init__(self, storage_dir: str = "memory", memory_name: str = "default"):
        self.storage_dir = storage_dir
        self.memory_name = memory_name
        self.mem_cells: List[MemCell] = []
        self.mem_id_index: Dict[str, int] = {}
        self.text_dir = os.path.join(self.storage_dir, "texts")
        os.makedirs(self.text_dir, exist_ok=True)
        self._load()

    def add(self, text: List[str], summary: str = "", embedding: Optional[Any] = None) -> str:
        create_time = time.time()
        mem_id = f"mem_{int(create_time * 1000)}"

        # Create text file path
        text_file_path = os.path.join(self.text_dir, f"{mem_id}.md")

        mem_cell = MemCell(
            text=text,
            text_file_path=text_file_path,
            summary=summary,
            create_time=create_time,
            update_time=create_time,
            mem_id=mem_id
        )
        self.mem_cells.append(mem_cell)
        self.mem_id_index[mem_id] = len(self.mem_cells) - 1
        self._save()
        return mem_id

    def get(self, mem_id: str) -> Optional[MemCell]:
        idx = self.mem_id_index.get(mem_id)
        if idx is not None:
            return self.mem_cells[idx]
        return None

    def update(self, mem_id: str, text: List[str] = None, summary: str = None) -> bool:
        idx = self.mem_id_index.get(mem_id)
        if idx is not None:
            mem_cell = self.mem_cells[idx]
            # Use MemCell's update method to correctly update update_cnt
            mem_cell.update(text=text, summary=summary)
            self._save()
            return True
        return False

    def delete(self, mem_id: str) -> bool:
        idx = self.mem_id_index.get(mem_id)
        if idx is not None:
            mem_cell = self.mem_cells[idx]
            # Delete text file
            if os.path.exists(mem_cell.text_file_path):
                os.remove(mem_cell.text_file_path)

            self.mem_cells.pop(idx)
            self.mem_id_index = {cell.mem_id: i for i,
                                 cell in enumerate(self.mem_cells)}
            self._save()
            return True
        return False

    def list_all(self) -> List[MemCell]:
        return self.mem_cells

    def add_memory(self, text: str, summary: str = "") -> MemCell:
        """Convenient method to add memory"""
        create_time = time.time()
        mem_id = f"mem_{int(create_time * 1000)}"

        # Create text file path
        text_file_path = os.path.join(self.text_dir, f"{mem_id}.md")

        mem_cell = MemCell(
            text_file_path=text_file_path,
            summary=summary,
            create_time=create_time,
            update_time=create_time,
            mem_id=mem_id
        )
        # Set text content (will be saved to file automatically)
        mem_cell.text = [text]

        self.mem_cells.append(mem_cell)
        self.mem_id_index[mem_id] = len(self.mem_cells) - 1
        self._save()
        return mem_cell

    def update_memory(self, mem_id: str, new_text: str, new_summary: str) -> MemCell:
        """Convenient method to update memory"""
        idx = self.mem_id_index.get(mem_id)
        if idx is not None:
            mem_cell = self.mem_cells[idx]
            # Use MemCell's update method to correctly update update_cnt
            mem_cell.update(text=[new_text], summary=new_summary)
            self._save()
            return mem_cell
        else:
            raise ValueError(f"Memory ID {mem_id} does not exist")

    def increment_recall(self, mem_id: str) -> bool:
        """
        Increment the recall count of a memory cell

        Args:
            mem_id: Memory cell ID

        Returns:
            bool: Whether the operation was successful
        """
        mem_cell = self.get(mem_id)
        if mem_cell is None:
            return False

        mem_cell.increment_recall()
        self._save()
        return True

    def _save(self):
        data = [cell.to_dict() for cell in self.mem_cells]
        save_path = os.path.join(
            self.storage_dir, f"{self.memory_name}_mem.json")
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def _load(self):
        # Load from {memory_name}_mem.json
        save_path = os.path.join(
            self.storage_dir, f"{self.memory_name}_mem.json")
        if os.path.exists(save_path):
            try:
                with open(save_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.mem_cells = [MemCell.from_dict(item) for item in data]
                    self.mem_id_index = {cell.mem_id: i for i,
                                         cell in enumerate(self.mem_cells)}
            except Exception as e:
                print(f"Failed to load from {self.memory_name}_mem.json: {e}")
