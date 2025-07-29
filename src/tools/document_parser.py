#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Document Parser Tool
Handles parsing of various document formats
"""

import os
import json
from typing import Dict, Any, List, Optional


class DocumentParser:
    """Document parser for various file formats"""
    
    def __init__(self, workspace_root: str = "."):
        self.workspace_root = workspace_root
        self.supported_formats = ['.txt', '.md', '.json', '.yaml', '.yml', '.csv']
    
    def parse_document(self, file_path: str) -> Dict[str, Any]:
        """Parse a document and return structured data"""
        if not os.path.exists(file_path):
            return {"success": False, "error": "File not found"}
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            return {
                "success": True,
                "content": content,
                "format": os.path.splitext(file_path)[1],
                "size": len(content),
                "lines": len(content.splitlines())
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def parse_multiple(self, file_paths: List[str]) -> List[Dict[str, Any]]:
        """Parse multiple documents"""
        results = []
        for file_path in file_paths:
            results.append(self.parse_document(file_path))
        return results
    
    def get_document_info(self, file_path: str) -> Dict[str, Any]:
        """Get basic document information"""
        if not os.path.exists(file_path):
            return {"exists": False}
        
        stat = os.stat(file_path)
        return {
            "exists": True,  
            "size": stat.st_size,
            "modified": stat.st_mtime,
            "format": os.path.splitext(file_path)[1],
            "supported": os.path.splitext(file_path)[1] in self.supported_formats
        }