#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Fallback conversion strategies for handling markdown to PDF conversion failures
"""

import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Tuple, Optional, List


def apply_fallback_strategies(input_file: str, output_file: str) -> Tuple[bool, str, Dict]:
    """
    Apply multiple fallback strategies when standard PDF conversion fails
    
    Args:
        input_file: Input markdown file path
        output_file: Desired PDF output file path
        
    Returns:
        Tuple[bool, str, Dict]: (success, message, result_info)
    """
    fallback_results = {
        'strategies_attempted': [],
        'final_strategy': None,
        'output_files': []
    }
    
    strategies = [
        ('image_stripping', 'Remove images and convert'),
        ('simplified_markdown', 'Simplify markdown and convert'),  
        ('html_intermediate', 'Convert via HTML intermediate'),
        ('plain_text', 'Convert to plain text PDF'),
        ('word_fallback', 'Generate Word document instead')
    ]
    
    for strategy_name, strategy_desc in strategies:
        print(f"🔄 Trying fallback strategy: {strategy_desc}")
        fallback_results['strategies_attempted'].append(strategy_name)
        
        try:
            if strategy_name == 'image_stripping':
                success, message, files = _try_image_stripping_strategy(input_file, output_file)
            elif strategy_name == 'simplified_markdown':
                success, message, files = _try_simplified_markdown_strategy(input_file, output_file)
            elif strategy_name == 'html_intermediate':
                success, message, files = _try_html_intermediate_strategy(input_file, output_file)
            elif strategy_name == 'plain_text':
                success, message, files = _try_plain_text_strategy(input_file, output_file)
            elif strategy_name == 'word_fallback':
                success, message, files = _try_word_fallback_strategy(input_file, output_file)
            else:
                continue
                
            if success:
                fallback_results['final_strategy'] = strategy_name
                fallback_results['output_files'] = files
                return True, f"Success with {strategy_desc}: {message}", fallback_results
                
        except Exception as e:
            print(f"❌ Strategy {strategy_name} failed: {e}")
            continue
    
    # All strategies failed
    return False, "All fallback strategies failed", fallback_results


def _try_image_stripping_strategy(input_file: str, output_file: str) -> Tuple[bool, str, List[str]]:
    """Strip images and try PDF conversion"""
    try:
        # Read markdown and remove image references
        with open(input_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Remove image references but keep alt text as regular text
        image_pattern = r'!\[([^\]]*)\]\([^)]+\)'
        
        def replace_image(match):
            alt_text = match.group(1)
            return f"[Image: {alt_text}]" if alt_text else "[Image]"
        
        stripped_content = re.sub(image_pattern, replace_image, content)
        
        # Create temporary stripped markdown file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False, encoding='utf-8') as f:
            f.write(stripped_content)
            temp_md_file = f.name
        
        try:
            # Try pandoc conversion without images
            cmd = [
                'pandoc',
                temp_md_file,
                '-o', output_file,
                '--pdf-engine=xelatex',
                '--quiet'
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0 and os.path.exists(output_file):
                return True, "PDF generated without images", [output_file]
            else:
                return False, f"Pandoc failed: {result.stderr}", []
                
        finally:
            # Clean up temp file
            if os.path.exists(temp_md_file):
                os.remove(temp_md_file)
                
    except Exception as e:
        return False, str(e), []


def _try_simplified_markdown_strategy(input_file: str, output_file: str) -> Tuple[bool, str, List[str]]:
    """Simplify markdown content and try conversion"""
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Simplify markdown - remove complex elements
        simplified_content = content
        
        # Remove complex tables (keep simple ones)
        simplified_content = re.sub(r'\|[^|\n]*\|[^|\n]*\|[^\n]*\n\|[-\s:|]*\|[^\n]*\n(\|[^\n]*\n)*', 
                                   '[Complex Table Removed]\n', simplified_content)
        
        # Remove HTML tags
        simplified_content = re.sub(r'<[^>]+>', '', simplified_content)
        
        # Remove multiple consecutive blank lines
        simplified_content = re.sub(r'\n{3,}', '\n\n', simplified_content)
        
        # Remove emoji and special characters that might cause issues
        simplified_content = re.sub(r'[^\x00-\x7F\u4e00-\u9fff\u3400-\u4dbf\n\r\t ]', '', simplified_content)
        
        # Create temporary simplified markdown file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False, encoding='utf-8') as f:
            f.write(simplified_content)
            temp_md_file = f.name
        
        try:
            # Try basic pandoc conversion
            cmd = [
                'pandoc',
                temp_md_file,
                '-o', output_file,
                '--pdf-engine=xelatex',
                '-V', 'fontsize=12pt',
                '--quiet'
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0 and os.path.exists(output_file):
                return True, "PDF generated with simplified markdown", [output_file]
            else:
                return False, f"Simplified conversion failed: {result.stderr}", []
                
        finally:
            if os.path.exists(temp_md_file):
                os.remove(temp_md_file)
                
    except Exception as e:
        return False, str(e), []


def _try_html_intermediate_strategy(input_file: str, output_file: str) -> Tuple[bool, str, List[str]]:
    """Convert markdown to HTML first, then HTML to PDF"""
    try:
        html_file = output_file.replace('.pdf', '_temp.html')
        
        # Step 1: Convert markdown to HTML
        cmd1 = [
            'pandoc',
            input_file,
            '-o', html_file,
            '--from', 'markdown',
            '--to', 'html',
            '--standalone'
        ]
        
        result1 = subprocess.run(cmd1, capture_output=True, text=True)
        
        if result1.returncode != 0 or not os.path.exists(html_file):
            return False, f"Markdown to HTML failed: {result1.stderr}", []
        
        try:
            # Step 2: Convert HTML to PDF using wkhtmltopdf if available
            cmd2 = [
                'wkhtmltopdf',
                '--quiet',
                '--page-size', 'A4',
                '--margin-top', '0.75in',
                '--margin-right', '0.75in', 
                '--margin-bottom', '0.75in',
                '--margin-left', '0.75in',
                html_file,
                output_file
            ]
            
            result2 = subprocess.run(cmd2, capture_output=True, text=True)
            
            if result2.returncode == 0 and os.path.exists(output_file):
                return True, "PDF generated via HTML intermediate", [output_file]
            else:
                # Try with pandoc HTML to PDF
                cmd3 = [
                    'pandoc',
                    html_file,
                    '-o', output_file,
                    '--pdf-engine=wkhtmltopdf',
                    '--quiet'
                ]
                
                result3 = subprocess.run(cmd3, capture_output=True, text=True)
                
                if result3.returncode == 0 and os.path.exists(output_file):
                    return True, "PDF generated via pandoc HTML", [output_file]
                else:
                    return False, f"HTML to PDF failed: {result3.stderr}", []
                    
        finally:
            # Clean up HTML file
            if os.path.exists(html_file):
                os.remove(html_file)
                
    except Exception as e:
        return False, str(e), []


def _try_plain_text_strategy(input_file: str, output_file: str) -> Tuple[bool, str, List[str]]:
    """Convert to plain text and then to PDF"""
    try:
        # Extract plain text from markdown
        with open(input_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Remove markdown formatting to get plain text
        plain_text = content
        
        # Remove markdown syntax
        plain_text = re.sub(r'^#{1,6}\s+', '', plain_text, flags=re.MULTILINE)  # Headers
        plain_text = re.sub(r'\*\*(.*?)\*\*', r'\1', plain_text)  # Bold
        plain_text = re.sub(r'\*(.*?)\*', r'\1', plain_text)  # Italic
        plain_text = re.sub(r'`([^`]+)`', r'\1', plain_text)  # Inline code
        plain_text = re.sub(r'```[^`]*```', '[Code Block]', plain_text, flags=re.DOTALL)  # Code blocks
        plain_text = re.sub(r'!\[[^\]]*\]\([^)]+\)', '[Image]', plain_text)  # Images
        plain_text = re.sub(r'\[[^\]]*\]\([^)]+\)', '', plain_text)  # Links (keep link text)
        plain_text = re.sub(r'^\s*[-*+]\s+', '• ', plain_text, flags=re.MULTILINE)  # Lists
        plain_text = re.sub(r'^\s*\d+\.\s+', '', plain_text, flags=re.MULTILINE)  # Numbered lists
        
        # Create temporary text file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
            f.write(plain_text)
            temp_txt_file = f.name
        
        try:
            # Convert plain text to PDF
            cmd = [
                'pandoc',
                temp_txt_file,
                '-o', output_file,
                '--pdf-engine=xelatex',
                '-V', 'fontsize=12pt',
                '-V', 'geometry:margin=1in',
                '--quiet'
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0 and os.path.exists(output_file):
                return True, "PDF generated from plain text", [output_file]
            else:
                return False, f"Plain text to PDF failed: {result.stderr}", []
                
        finally:
            if os.path.exists(temp_txt_file):
                os.remove(temp_txt_file)
                
    except Exception as e:
        return False, str(e), []


def _try_word_fallback_strategy(input_file: str, output_file: str) -> Tuple[bool, str, List[str]]:
    """Generate Word document as final fallback"""
    try:
        # Generate Word document instead of PDF
        word_output = output_file.replace('.pdf', '_fallback.docx')
        
        cmd = [
            'pandoc',
            input_file,
            '-o', word_output,
            '--from', 'markdown',
            '--to', 'docx',
            '--toc'
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0 and os.path.exists(word_output):
            return True, f"Word document generated: {word_output}", [word_output]
        else:
            return False, f"Word fallback failed: {result.stderr}", []
            
    except Exception as e:
        return False, str(e), []
