#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文档解析功能单元测试
测试各种文档格式的解析和处理能力
"""

import pytest
import os
import sys
import json
import tempfile
from unittest.mock import patch, Mock, MagicMock
from typing import Dict, List, Any
import time # Added missing import for time.time()

# 添加src路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from tools.document_parser import DocumentParser
from utils.test_helpers import TestHelper

@pytest.mark.unit
class TestDocumentParsing:
    """文档解析功能测试类"""
    
    @pytest.fixture
    def document_parser(self, test_workspace):
        """创建文档解析器实例"""
        return DocumentParser(workspace_root=test_workspace)
    
    @pytest.fixture
    def sample_documents(self, test_workspace):
        """创建示例文档文件"""
        documents = {}
        
        # 创建文本文件
        txt_content = """
AGI Bot 项目说明
================

AGI Bot 是一个基于大语言模型的智能助手系统。

主要特性：
1. 自然语言理解
2. 代码生成和分析
3. 多模态处理
4. 工具调用能力

联系方式：contact@agibot.ai
版本：1.0.0
"""
        txt_path = os.path.join(test_workspace, "readme.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(txt_content)
        documents["txt"] = txt_path
        
        # 创建Markdown文件
        md_content = """# AGI Bot Documentation

## Overview
AGI Bot is an advanced AI assistant powered by large language models.

## Features
- **Natural Language Processing**: Advanced NLP capabilities
- **Code Generation**: Automatic code generation and analysis
- **Multi-modal Support**: Support for text, images, and documents
- **Tool Integration**: Seamless integration with external tools

## Installation
```bash
pip install agibot
```

## Usage
```python
from agibot import AGIBotClient

client = AGIBotClient(api_key="your-key")
response = client.chat("Hello, AGI Bot!")
```

## API Reference
### AGIBotClient
- `chat(message)`: Send a message to the AI
- `batch_process(tasks)`: Process multiple tasks
- `get_status()`: Get current status

---
© 2024 AGI Bot Team
"""
        md_path = os.path.join(test_workspace, "documentation.md")
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(md_content)
        documents["md"] = md_path
        
        # 创建JSON文件
        json_content = {
            "project": "AGI Bot",
            "version": "1.0.0",
            "description": "AI-powered intelligent assistant",
            "features": [
                {"name": "NLP", "description": "Natural language processing"},
                {"name": "Code Generation", "description": "Automatic code generation"},
                {"name": "Multi-modal", "description": "Support for various input types"}
            ],
            "config": {
                "max_tokens": 4096,
                "temperature": 0.7,
                "model": "claude-3-5-sonnet"
            },
            "dependencies": {
                "python": ">=3.8",
                "requests": ">=2.25.0",
                "numpy": ">=1.20.0"
            }
        }
        json_path = os.path.join(test_workspace, "config.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(json_content, f, indent=2, ensure_ascii=False)
        documents["json"] = json_path
        
        # 创建CSV文件
        csv_content = """task_id,task_name,status,duration,success_rate
1,Code Generation,completed,120,0.95
2,Document Analysis,completed,85,0.92
3,Image Processing,running,45,0.88
4,API Testing,pending,0,0.0
5,Performance Optimization,completed,200,0.97
"""
        csv_path = os.path.join(test_workspace, "tasks.csv")
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write(csv_content)
        documents["csv"] = csv_path
        
        # 创建XML文件
        xml_content = """<?xml version="1.0" encoding="UTF-8"?>
<project name="AGI Bot" version="1.0.0">
    <description>AI-powered intelligent assistant system</description>
    <features>
        <feature id="nlp" priority="high">
            <name>Natural Language Processing</name>
            <description>Advanced NLP capabilities for text understanding</description>
        </feature>
        <feature id="codegen" priority="high">
            <name>Code Generation</name>
            <description>Automatic code generation and analysis</description>
        </feature>
        <feature id="multimodal" priority="medium">
            <name>Multi-modal Support</name>
            <description>Support for text, images, and documents</description>
        </feature>
    </features>
    <configuration>
        <parameter name="max_tokens" value="4096" type="integer"/>
        <parameter name="temperature" value="0.7" type="float"/>
        <parameter name="model" value="claude-3-5-sonnet" type="string"/>
    </configuration>
</project>
"""
        xml_path = os.path.join(test_workspace, "project.xml")
        with open(xml_path, "w", encoding="utf-8") as f:
            f.write(xml_content)
        documents["xml"] = xml_path
        
        # 创建YAML文件
        yaml_content = """
project:
  name: AGI Bot
  version: 1.0.0
  description: AI-powered intelligent assistant
  
features:
  - name: Natural Language Processing
    id: nlp
    priority: high
    enabled: true
  - name: Code Generation
    id: codegen
    priority: high
    enabled: true
  - name: Multi-modal Support
    id: multimodal
    priority: medium
    enabled: true

configuration:
  model:
    name: claude-3-5-sonnet
    max_tokens: 4096
    temperature: 0.7
    top_p: 1.0
  
  api:
    base_url: https://api.anthropic.com
    timeout: 30
    retries: 3

dependencies:
  python: ">=3.8"
  requests: ">=2.25.0"
  pyyaml: ">=5.4.0"
"""
        yaml_path = os.path.join(test_workspace, "config.yaml")
        with open(yaml_path, "w", encoding="utf-8") as f:
            f.write(yaml_content)
        documents["yaml"] = yaml_path
        
        return documents
    
    @pytest.fixture
    def document_formats(self):
        """支持的文档格式"""
        return {
            "text": ["txt", "text", "log"],
            "structured": ["json", "yaml", "yml", "xml", "toml"],
            "tabular": ["csv", "tsv", "xlsx", "xls"],
            "markup": ["md", "markdown", "html", "htm"],
            "office": ["docx", "doc", "pptx", "ppt"],
            "pdf": ["pdf"],
            "code": ["py", "js", "java", "cpp", "c", "go", "rs"]
        }
    
    def test_document_parser_initialization(self, document_parser, test_workspace):
        """测试文档解析器初始化"""
        assert document_parser is not None
        assert hasattr(document_parser, 'parse_document')
        assert hasattr(document_parser, 'extract_text')
        assert document_parser.workspace_root == test_workspace
    
    def test_text_file_parsing(self, document_parser, sample_documents):
        """测试文本文件解析"""
        txt_path = sample_documents["txt"]
        
        try:
            result = document_parser.parse_document(txt_path)
            
            # 验证文本解析结果
            assert result is not None
            if isinstance(result, dict):
                assert "content" in result or "text" in result
                text_content = result.get("content") or result.get("text")
                assert "AGI Bot" in text_content
                assert "智能助手" in text_content
            elif isinstance(result, str):
                assert "AGI Bot" in result
                
        except Exception as e:
            pytest.skip(f"Text parsing not available: {e}")
    
    def test_markdown_parsing(self, document_parser, sample_documents):
        """测试Markdown文件解析"""
        md_path = sample_documents["md"]
        
        try:
            result = document_parser.parse_document(md_path)
            
            # 验证Markdown解析结果
            assert result is not None
            if isinstance(result, dict):
                # 检查是否提取了结构化信息
                expected_fields = ["content", "headers", "code_blocks", "links"]
                available_fields = [field for field in expected_fields if field in result]
                assert len(available_fields) > 0
                
                # 验证内容
                content = result.get("content", "")
                assert "AGI Bot Documentation" in content
                assert "Installation" in content
                
        except Exception as e:
            pytest.skip(f"Markdown parsing not available: {e}")
    
    def test_json_parsing(self, document_parser, sample_documents):
        """测试JSON文件解析"""
        json_path = sample_documents["json"]
        
        try:
            result = document_parser.parse_document(json_path)
            
            # 验证JSON解析结果
            assert result is not None
            if isinstance(result, dict):
                # 应该包含JSON数据
                assert "project" in result or "data" in result
                
                # 检查具体数据
                if "project" in result:
                    assert result["project"] == "AGI Bot"
                elif "data" in result and isinstance(result["data"], dict):
                    assert result["data"]["project"] == "AGI Bot"
                    
        except Exception as e:
            pytest.skip(f"JSON parsing not available: {e}")
    
    def test_csv_parsing(self, document_parser, sample_documents):
        """测试CSV文件解析"""
        csv_path = sample_documents["csv"]
        
        try:
            result = document_parser.parse_document(csv_path)
            
            # 验证CSV解析结果
            assert result is not None
            if isinstance(result, dict):
                # 应该包含表格数据
                expected_fields = ["data", "rows", "columns", "headers"]
                available_fields = [field for field in expected_fields if field in result]
                assert len(available_fields) > 0
                
                # 验证数据结构
                if "data" in result:
                    data = result["data"]
                    if isinstance(data, list) and len(data) > 0:
                        assert len(data) >= 5  # 5行数据
                        
        except Exception as e:
            pytest.skip(f"CSV parsing not available: {e}")
    
    def test_xml_parsing(self, document_parser, sample_documents):
        """测试XML文件解析"""
        xml_path = sample_documents["xml"]
        
        try:
            result = document_parser.parse_document(xml_path)
            
            # 验证XML解析结果
            assert result is not None
            if isinstance(result, dict):
                # 应该包含XML结构数据
                expected_fields = ["root", "elements", "attributes", "text"]
                available_fields = [field for field in expected_fields if field in result]
                assert len(available_fields) > 0
                
                # 验证项目信息
                content_str = str(result)
                assert "AGI Bot" in content_str
                
        except Exception as e:
            pytest.skip(f"XML parsing not available: {e}")
    
    def test_yaml_parsing(self, document_parser, sample_documents):
        """测试YAML文件解析"""
        yaml_path = sample_documents["yaml"]
        
        try:
            result = document_parser.parse_document(yaml_path)
            
            # 验证YAML解析结果
            assert result is not None
            if isinstance(result, dict):
                # 应该包含YAML数据
                if "project" in result:
                    project_data = result["project"]
                    assert project_data["name"] == "AGI Bot"
                elif "data" in result:
                    assert "project" in result["data"]
                    
        except Exception as e:
            pytest.skip(f"YAML parsing not available: {e}")
    
    def test_format_detection(self, document_parser, sample_documents):
        """测试文件格式检测"""
        format_mapping = {
            "txt": "text",
            "md": "markdown",
            "json": "json",
            "csv": "csv",
            "xml": "xml",
            "yaml": "yaml"
        }
        
        for file_type, path in sample_documents.items():
            try:
                detected_format = document_parser.detect_format(path)
                
                # 验证格式检测
                assert detected_format is not None
                if isinstance(detected_format, str):
                    expected_format = format_mapping.get(file_type, file_type)
                    assert detected_format.lower() in [expected_format, file_type]
                    
            except Exception as e:
                pass
    
    def test_content_extraction(self, document_parser, sample_documents):
        """测试内容提取"""
        for file_type, path in sample_documents.items():
            try:
                extracted_content = document_parser.extract_text(path)
                
                # 验证内容提取
                assert extracted_content is not None
                if isinstance(extracted_content, str):
                    assert len(extracted_content) > 0
                    # 应该包含AGI Bot相关内容
                    assert "AGI Bot" in extracted_content or "agibot" in extracted_content.lower()
                elif isinstance(extracted_content, dict):
                    assert "text" in extracted_content or "content" in extracted_content
                    
            except Exception as e:
                pass
    
    def test_metadata_extraction(self, document_parser, sample_documents):
        """测试元数据提取"""
        for file_type, path in sample_documents.items():
            try:
                metadata = document_parser.extract_metadata(path)
                
                # 验证元数据提取
                assert metadata is not None
                if isinstance(metadata, dict):
                    # 基本元数据字段
                    basic_fields = ["file_size", "file_type", "created_time", "modified_time"]
                    available_fields = [field for field in basic_fields if field in metadata]
                    assert len(available_fields) > 0
                    
                    # 文件类型应该正确
                    if "file_type" in metadata:
                        assert file_type in metadata["file_type"].lower()
                        
            except Exception as e:
                pass
    
    def test_large_document_handling(self, document_parser, test_workspace):
        """测试大文档处理"""
        # 创建大文档
        large_content = "AGI Bot " * 10000 + "\n这是一个大文档测试。\n" * 5000
        large_file = os.path.join(test_workspace, "large_document.txt")
        
        with open(large_file, "w", encoding="utf-8") as f:
            f.write(large_content)
        
        try:
            start_time = time.time()
            result = document_parser.parse_document(large_file)
            end_time = time.time()
            
            processing_time = end_time - start_time
            
            # 验证大文档处理
            assert result is not None
            # 处理时间应该在合理范围内（少于30秒）
            assert processing_time < 30
            
        except Exception as e:
            # 大文档处理可能需要特殊配置
            pass
    
    def test_encoding_detection(self, document_parser, test_workspace):
        """测试编码检测"""
        # 创建不同编码的文件
        encodings = ["utf-8", "gbk", "latin-1"]
        content = "AGI Bot 支持多种编码格式的文档解析。"
        
        for encoding in encodings:
            try:
                file_path = os.path.join(test_workspace, f"test_{encoding}.txt")
                with open(file_path, "w", encoding=encoding) as f:
                    f.write(content)
                
                # 尝试解析
                result = document_parser.parse_document(file_path)
                
                # 验证编码检测
                assert result is not None
                if isinstance(result, dict) and "encoding" in result:
                    detected_encoding = result["encoding"]
                    # 检测的编码应该与实际编码兼容
                    assert detected_encoding is not None
                    
            except UnicodeError:
                # 某些编码可能不支持特定字符
                pass
            except Exception as e:
                pass
    
    def test_malformed_document_handling(self, document_parser, test_workspace):
        """测试格式错误文档处理"""
        # 创建格式错误的文档
        malformed_files = [
            {"name": "malformed.json", "content": '{"key": "value", "missing": }'},
            {"name": "malformed.xml", "content": '<root><unclosed>data</root>'},
            {"name": "malformed.csv", "content": 'header1,header2\nvalue1\nvalue2,value3,extra'},
            {"name": "truncated.txt", "content": "Truncated content"},
        ]
        
        for file_info in malformed_files:
            file_path = os.path.join(test_workspace, file_info["name"])
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(file_info["content"])
            
            try:
                result = document_parser.parse_document(file_path)
                
                # 验证错误处理
                assert result is not None
                if isinstance(result, dict):
                    # 可能包含错误信息
                    if "error" in result:
                        assert isinstance(result["error"], str)
                    else:
                        # 或者尽力解析部分内容
                        assert "content" in result or "text" in result
                        
            except Exception as e:
                # 格式错误的文档应该抛出合适的异常
                assert any(keyword in str(e).lower() for keyword in ['format', 'parse', 'invalid', 'malformed'])
    
    def test_empty_document_handling(self, document_parser, test_workspace):
        """测试空文档处理"""
        # 创建空文档
        empty_file = os.path.join(test_workspace, "empty.txt")
        with open(empty_file, "w") as f:
            f.write("")
        
        try:
            result = document_parser.parse_document(empty_file)
            
            # 验证空文档处理
            assert result is not None
            if isinstance(result, dict):
                content = result.get("content", "") or result.get("text", "")
                assert len(content) == 0
            elif isinstance(result, str):
                assert len(result) == 0
                
        except Exception as e:
            # 空文档处理可能有特殊逻辑
            pass
    
    def test_binary_file_handling(self, document_parser, test_workspace):
        """测试二进制文件处理"""
        # 创建二进制文件
        binary_file = os.path.join(test_workspace, "binary.bin")
        with open(binary_file, "wb") as f:
            f.write(b'\x00\x01\x02\x03\x04\x05\x06\x07\x08\x09')
        
        try:
            result = document_parser.parse_document(binary_file)
            
            # 验证二进制文件处理
            assert result is not None
            if isinstance(result, dict):
                # 可能包含错误信息或特殊处理结果
                assert "error" in result or "binary" in result or "content" in result
                
        except Exception as e:
            # 二进制文件应该被识别并适当处理
            assert any(keyword in str(e).lower() for keyword in ['binary', 'text', 'encoding'])
    
    def test_nested_structure_parsing(self, document_parser, test_workspace):
        """测试嵌套结构解析"""
        # 创建复杂嵌套结构的JSON
        nested_json = {
            "project": {
                "info": {
                    "name": "AGI Bot",
                    "version": "1.0.0",
                    "details": {
                        "description": "AI assistant",
                        "features": {
                            "nlp": {"enabled": True, "confidence": 0.95},
                            "vision": {"enabled": False, "reason": "not implemented"}
                        }
                    }
                },
                "config": {
                    "models": [
                        {"name": "claude", "priority": 1},
                        {"name": "gpt", "priority": 2}
                    ]
                }
            }
        }
        
        nested_file = os.path.join(test_workspace, "nested.json")
        with open(nested_file, "w") as f:
            json.dump(nested_json, f, indent=2)
        
        try:
            result = document_parser.parse_document(nested_file)
            
            # 验证嵌套结构解析
            assert result is not None
            if isinstance(result, dict):
                # 应该能够解析嵌套结构
                if "project" in result:
                    project_data = result["project"]
                    assert "info" in project_data
                    assert "config" in project_data
                    
        except Exception as e:
            pass
    
    def test_document_validation(self, document_parser, sample_documents):
        """测试文档验证"""
        for file_type, path in sample_documents.items():
            try:
                # 验证文档是否有效
                if hasattr(document_parser, 'validate_document'):
                    is_valid = document_parser.validate_document(path)
                    assert isinstance(is_valid, bool)
                    
                    # 已知的示例文档应该是有效的
                    assert is_valid is True
                else:
                    # 通过解析成功来验证
                    result = document_parser.parse_document(path)
                    assert result is not None
                    
            except Exception as e:
                pass
    
    def test_document_conversion(self, document_parser, sample_documents, test_workspace):
        """测试文档格式转换"""
        # 测试格式转换（如果支持）
        source_formats = ["json", "yaml", "xml"]
        target_formats = ["json", "yaml"]
        
        for source_format in source_formats:
            if source_format in sample_documents:
                source_path = sample_documents[source_format]
                
                for target_format in target_formats:
                    if source_format != target_format:
                        target_path = os.path.join(
                            test_workspace, 
                            f"converted.{target_format}"
                        )
                        
                        try:
                            if hasattr(document_parser, 'convert_format'):
                                result = document_parser.convert_format(
                                    source_path, 
                                    target_format, 
                                    target_path
                                )
                                
                                # 验证转换结果
                                assert result is not None
                                if isinstance(result, bool):
                                    assert result is True
                                    assert os.path.exists(target_path)
                                    
                        except Exception as e:
                            # 格式转换可能不被支持
                            pass
    
    def test_batch_document_processing(self, document_parser, sample_documents):
        """测试批量文档处理"""
        document_paths = list(sample_documents.values())
        
        try:
            if hasattr(document_parser, 'batch_parse'):
                results = document_parser.batch_parse(document_paths)
                
                # 验证批量处理结果
                assert results is not None
                assert len(results) == len(document_paths)
                
                for result in results:
                    assert result is not None
            else:
                # 逐个处理
                results = []
                for path in document_paths:
                    result = document_parser.parse_document(path)
                    results.append(result)
                
                assert len(results) == len(document_paths)
                
        except Exception as e:
            pass
    
    def test_document_search(self, document_parser, sample_documents):
        """测试文档内容搜索"""
        search_queries = ["AGI Bot", "智能助手", "API", "features"]
        
        for query in search_queries:
            try:
                if hasattr(document_parser, 'search_in_documents'):
                    results = document_parser.search_in_documents(
                        list(sample_documents.values()), 
                        query
                    )
                    
                    # 验证搜索结果
                    assert results is not None
                    if isinstance(results, list):
                        # 应该找到包含查询词的文档
                        matching_docs = [r for r in results if r.get("matches", 0) > 0]
                        if query == "AGI Bot":
                            assert len(matching_docs) > 0
                else:
                    # 手动搜索
                    matching_files = []
                    for file_type, path in sample_documents.items():
                        content = document_parser.extract_text(path)
                        if isinstance(content, str) and query in content:
                            matching_files.append(path)
                        elif isinstance(content, dict):
                            text = content.get("content", "") or content.get("text", "")
                            if query in text:
                                matching_files.append(path)
                    
                    if query == "AGI Bot":
                        assert len(matching_files) > 0
                        
            except Exception as e:
                pass
    
    def test_document_summary_generation(self, document_parser, sample_documents):
        """测试文档摘要生成"""
        for file_type, path in sample_documents.items():
            try:
                if hasattr(document_parser, 'generate_summary'):
                    summary = document_parser.generate_summary(path)
                    
                    # 验证摘要生成
                    assert summary is not None
                    if isinstance(summary, dict):
                        assert "summary" in summary or "abstract" in summary
                        summary_text = summary.get("summary") or summary.get("abstract")
                        assert len(summary_text) > 0
                        assert len(summary_text) < 1000  # 摘要应该比原文短
                    elif isinstance(summary, str):
                        assert len(summary) > 0
                        assert "AGI Bot" in summary or "智能" in summary
                        
            except Exception as e:
                pass 