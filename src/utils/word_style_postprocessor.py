#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Word文档样式后处理器
用于修改生成的Word文档中的标题颜色，将蓝色改为黑色
"""

import zipfile
import tempfile
import os
import shutil
import re


def fix_word_title_colors(docx_path):
    """
    修改Word文档中的标题颜色，将蓝色改为黑色
    
    Args:
        docx_path: Word文档的路径
    
    Returns:
        bool: 修改是否成功
    """
    try:
        # 创建临时目录
        with tempfile.TemporaryDirectory() as temp_dir:
            # 解压docx文件
            with zipfile.ZipFile(docx_path, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)
            
            # 读取styles.xml文件
            styles_path = os.path.join(temp_dir, 'word', 'styles.xml')
            if not os.path.exists(styles_path):
                print(f"Warning: styles.xml not found in {docx_path}")
                return False
            
            with open(styles_path, 'r', encoding='utf-8') as f:
                styles_content = f.read()
            
            # 备份原始内容
            original_content = styles_content
            
            # 修改标题样式的颜色定义
            # 将蓝色相关的颜色值改为黑色
            modifications = [
                # Heading1 - 将蓝色改为黑色
                (r'<w:color w:val="345A8A"[^>]*?/>', '<w:color w:val="000000" />'),
                (r'<w:color w:val="365F91"[^>]*?/>', '<w:color w:val="000000" />'),
                
                # 其他可能的蓝色变体
                (r'<w:color w:val="[0-9A-Fa-f]{6}" w:themeColor="accent1"[^>]*?/>', '<w:color w:val="000000" />'),
                
                # 移除主题颜色引用，直接使用黑色
                (r'w:themeColor="accent1"[^>]*?', ''),
                (r'w:themeShade="[^"]*"[^>]*?', ''),
                (r'w:themeTint="[^"]*"[^>]*?', ''),
            ]
            
            # 应用修改
            for pattern, replacement in modifications:
                styles_content = re.sub(pattern, replacement, styles_content)
            
            # 确保所有标题样式都有黑色定义
            heading_patterns = [
                (r'(<w:style[^>]*w:styleId="Heading[1-6]"[^>]*>.*?<w:rPr>)(.*?)(</w:rPr>)', 
                 r'\1\2<w:color w:val="000000" />\3'),
            ]
            
            for pattern, replacement in heading_patterns:
                # 只有当没有颜色定义时才添加
                if not re.search(r'<w:color[^>]*?/>', styles_content):
                    styles_content = re.sub(pattern, replacement, styles_content, flags=re.DOTALL)
            
            # 如果内容有变化，写回文件
            if styles_content != original_content:
                with open(styles_path, 'w', encoding='utf-8') as f:
                    f.write(styles_content)
                
                # 重新打包docx文件
                with zipfile.ZipFile(docx_path, 'w', zipfile.ZIP_DEFLATED) as zip_ref:
                    for root, dirs, files in os.walk(temp_dir):
                        for file in files:
                            file_path = os.path.join(root, file)
                            arcname = os.path.relpath(file_path, temp_dir)
                            zip_ref.write(file_path, arcname)
                
                print(f"Successfully modified title colors in {docx_path}")
                return True
            else:
                print(f"No modifications needed for {docx_path}")
                return True
                
    except Exception as e:
        print(f"Error processing {docx_path}: {str(e)}")
        return False


def main():
    """测试函数"""
    import sys
    if len(sys.argv) != 2:
        print("Usage: python word_style_postprocessor.py <docx_file>")
        sys.exit(1)
    
    docx_file = sys.argv[1]
    if not os.path.exists(docx_file):
        print(f"Error: File {docx_file} not found")
        sys.exit(1)
    
    success = fix_word_title_colors(docx_file)
    if success:
        print("Title colors fixed successfully")
    else:
        print("Failed to fix title colors")


if __name__ == "__main__":
    main()
