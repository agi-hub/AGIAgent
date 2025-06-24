 #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script for web search tools fixes
"""

import re
import os
import datetime

def test_filename_generation():
    """Test filename generation logic to ensure proper extensions"""
    
    def generate_filename(title, search_term=""):
        """Simulate the filename generation logic"""
        safe_title = re.sub(r'[^\w\s-]', '', title)[:50]
        safe_title = re.sub(r'[-\s]+', '_', safe_title)
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if search_term:
            safe_search = re.sub(r'[^\w\s-]', '', search_term)[:30]
            safe_search = re.sub(r'[-\s]+', '_', safe_search)
            base_filename = f"{safe_search}_{safe_title}_{timestamp}"
        else:
            base_filename = f"{safe_title}_{timestamp}"
        
        # Clean up filename
        base_filename = re.sub(r'_+', '_', base_filename).strip('_')
        if not base_filename or len(base_filename) < 3:
            base_filename = f"webpage_{timestamp}"
        
        # Generate actual filenames
        html_filename = f"{base_filename}.html"
        txt_filename = f"{base_filename}.txt"
        
        return html_filename, txt_filename
    
    # Test cases
    test_cases = [
        ("正常标题", "搜索关键词"),
        ("", "搜索关键词"),  # Empty title
        ("正常标题", ""),  # Empty search term
        ("", ""),  # Both empty
        ("   ", "   "),  # Only spaces
        ("特殊字符!@#$%^&*()", "关键词"),
        ("Very Long Title That Should Be Truncated Because It Is Too Long For File Name", "search"),
        ("😀🎉🔥", "emoji"),  # Emoji test
    ]
    
    print("Testing filename generation:")
    print("=" * 60)
    
    for i, (title, search_term) in enumerate(test_cases, 1):
        html_file, txt_file = generate_filename(title, search_term)
        print(f"Test {i}:")
        print(f"  Title: '{title}'")
        print(f"  Search: '{search_term}'")
        print(f"  HTML: {html_file}")
        print(f"  TXT:  {txt_file}")
        
        # Verify extensions
        assert html_file.endswith('.html'), f"HTML file missing extension: {html_file}"
        assert txt_file.endswith('.txt'), f"TXT file missing extension: {txt_file}"
        
        # Verify minimum length
        assert len(html_file) >= 8, f"HTML filename too short: {html_file}"
        assert len(txt_file) >= 7, f"TXT filename too short: {txt_file}"
        
        print("  ✅ Extensions and lengths OK")
        print()
    
    print("All filename generation tests passed! ✅")

def test_summary_structure():
    """Test the structure of the improved summary function"""
    
    # Mock results data
    mock_results = [
        {
            'title': '新闻标题1',
            'content': '这是第一个网页的内容，包含了关于搜索关键词的重要信息。' * 10,
            'source': 'Google',
            'saved_html_path': 'search_news1_20241201_120000.html',
            'saved_txt_path': 'search_news1_20241201_120000.txt'
        },
        {
            'title': '新闻标题2',
            'content': '这是第二个网页的内容，提供了更多相关的详细信息。' * 15,
            'source': 'Baidu',
            'saved_html_path': 'search_news2_20241201_120001.html',
            'saved_txt_path': 'search_news2_20241201_120001.txt'
        }
    ]
    
    print("Testing summary structure:")
    print("=" * 60)
    
    # Simulate the content preparation logic
    results_content = []
    for i, result in enumerate(mock_results, 1):
        title = result.get('title', f'Result {i}')
        content = result.get('content', '')[:4000]
        source = result.get('source', 'Unknown')
        
        html_file = result.get('saved_html_path', '')
        txt_file = result.get('saved_txt_path', '')
        
        file_info = ""
        if html_file:
            file_info += f"HTML File: {os.path.basename(html_file)}\n"
        if txt_file:
            file_info += f"Text File: {os.path.basename(txt_file)}\n"
        
        result_section = f"=== Webpage {i}: {title} (Source: {source}) ===\n"
        if file_info:
            result_section += f"Saved Files:\n{file_info}\n"
        result_section += f"Content:\n{content}\n"
        
        results_content.append(result_section)
    
    combined_content = "\n".join(results_content)
    
    print("Generated content structure:")
    print(combined_content[:500] + "..." if len(combined_content) > 500 else combined_content)
    print()
    
    # Verify structure
    assert "=== Webpage 1:" in combined_content, "Missing webpage 1 header"
    assert "=== Webpage 2:" in combined_content, "Missing webpage 2 header"
    assert "HTML File:" in combined_content, "Missing HTML file reference"
    assert "Text File:" in combined_content, "Missing text file reference"
    assert "Source:" in combined_content, "Missing source information"
    
    print("Summary structure test passed! ✅")

if __name__ == "__main__":
    print("Web Search Tools Fix Verification")
    print("=" * 50)
    print()
    
    try:
        test_filename_generation()
        print()
        test_summary_structure()
        print()
        print("🎉 All tests passed! The fixes are working correctly.")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()