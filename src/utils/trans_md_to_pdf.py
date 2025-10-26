#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import subprocess
import argparse
import tempfile
import re
import platform
from pathlib import Path

# 设置控制台编码为UTF-8（Windows兼容性）
try:
    if sys.platform == 'win32':
        import codecs
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'replace')
        sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'replace')
except:
    pass


def remove_emoji_from_text(text):
    """
    从文本中删除emoji字符
    保留普通的中文、英文、数字和标点符号
    """
    if not text:
        return text
    
    # 使用正则表达式删除emoji
    # 匹配各种emoji Unicode范围
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # 表情符号
        "\U0001F300-\U0001F5FF"  # 杂项符号和象形文字
        "\U0001F680-\U0001F6FF"  # 交通和地图符号
        "\U0001F700-\U0001F77F"  # 炼金术符号
        "\U0001F780-\U0001F7FF"  # 几何形状扩展
        "\U0001F800-\U0001F8FF"  # 补充箭头-C
        "\U0001F900-\U0001F9FF"  # 补充符号和象形文字
        "\U0001FA00-\U0001FA6F"  # 棋牌符号
        "\U0001FA70-\U0001FAFF"  # 符号和象形文字扩展-A
        "\U00002600-\U000026FF"  # 杂项符号
        "\U00002700-\U000027BF"  # 装饰符号
        "\U0001F1E6-\U0001F1FF"  # 地区指示符号（国旗）
        "\U00002B50-\U00002B55"  # 星星等
        "\U0000FE00-\U0000FE0F"  # 变体选择器
        "]+", 
        flags=re.UNICODE
    )
    
    # 删除emoji
    text_without_emoji = emoji_pattern.sub('', text)
    
    # 清理多余的空格，但保留换行符
    # 将多个连续的空格合并为一个，但保留换行符
    text_without_emoji = re.sub(r'[ \t]+', ' ', text_without_emoji)  # 只合并空格和tab
    text_without_emoji = re.sub(r' *\n *', '\n', text_without_emoji)  # 清理换行符前后的空格
    text_without_emoji = re.sub(r'\n{3,}', '\n\n', text_without_emoji)  # 限制连续换行符数量
    
    return text_without_emoji.strip()


def create_emoji_free_markdown(input_file):
    """
    创建一个删除了emoji的临时markdown文件
    
    Args:
        input_file: 输入的markdown文件路径
    
    Returns:
        str: 临时文件路径，如果失败返回None
    """
    try:
        # 读取原始markdown文件
        with open(input_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 删除emoji
        cleaned_content = remove_emoji_from_text(content)
        
        # 如果内容没有变化，就不需要创建临时文件
        if cleaned_content == content:
            print("📝 No emoji found in markdown, using original file")
            return None
        
        # 创建临时文件
        temp_fd, temp_path = tempfile.mkstemp(suffix='.md', prefix='emoji_free_')
        
        try:
            # 写入清理后的内容
            with os.fdopen(temp_fd, 'w', encoding='utf-8') as temp_file:
                temp_file.write(cleaned_content)
            
            print(f"📝 Created emoji-free temporary markdown: {temp_path}")
            return temp_path
            
        except Exception as e:
            # 如果写入失败，关闭并删除临时文件
            os.close(temp_fd)
            if os.path.exists(temp_path):
                os.remove(temp_path)
            raise e
            
    except Exception as e:
        print(f"❌ Error creating emoji-free markdown: {e}")
        return None


def check_file_exists(file_path, description):
    """Check if file exists"""
    if not os.path.isfile(file_path):
        print(f"Error: {description} '{file_path}' Does not exist")
        return False
    return True


def get_script_dir():
    """Get script directory"""
    return Path(__file__).parent.absolute()


def get_cjk_fonts():
    """
    Get CJK fonts based on the operating system
    
    Returns:
        tuple: (main_font, sans_font, mono_font)
    """
    system = platform.system().lower()
    
    if system == 'darwin':  # macOS
        # macOS 系统自带的中文字体
        return (
            'Hiragino Sans GB',
            'Hiragino Sans GB', 
            'Hiragino Sans GB'
        )
    elif system == 'linux':
        # Linux 系统使用 Noto 字体（需要安装 fonts-noto-cjk）
        return (
            'Noto Serif CJK SC',
            'Noto Sans CJK SC',
            'Noto Sans Mono CJK SC'
        )
    elif system == 'windows':
        # Windows 系统使用微软雅黑等字体
        return (
            'Microsoft YaHei',
            'Microsoft YaHei',
            'Microsoft YaHei'
        )
    else:
        # 默认使用 Noto 字体
        return (
            'Noto Serif CJK SC',
            'Noto Sans CJK SC',
            'Noto Sans Mono CJK SC'
        )


def check_pdf_engine_availability():
    """Check which PDF engines are available and return the best one"""
    engines = [
        ('xelatex', '--pdf-engine=xelatex'),
        ('lualatex', '--pdf-engine=lualatex'),
        ('pdflatex', '--pdf-engine=pdflatex'),
        ('wkhtmltopdf', '--pdf-engine=wkhtmltopdf'),
        ('weasyprint', '--pdf-engine=weasyprint')
    ]
    
    available_engines = []
    
    for engine_name, engine_option in engines:
        try:
            if engine_name in ['xelatex', 'lualatex', 'pdflatex']:
                # Check if LaTeX engine is available
                result = subprocess.run([engine_name, '--version'], 
                                     capture_output=True, text=True, encoding='utf-8', errors='ignore', timeout=5)
                if result.returncode == 0:
                    available_engines.append((engine_name, engine_option))
            elif engine_name == 'wkhtmltopdf':
                # Check if wkhtmltopdf is available
                result = subprocess.run(['wkhtmltopdf', '--version'], 
                                     capture_output=True, text=True, encoding='utf-8', errors='ignore', timeout=5)
                if result.returncode == 0:
                    available_engines.append((engine_name, engine_option))
        except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.CalledProcessError):
            continue
    
    if not available_engines:
        return None, None
    
    # Return the best available engine (prioritize xelatex > lualatex > pdflatex > others)
    priority_order = ['xelatex', 'lualatex', 'pdflatex', 'wkhtmltopdf', 'weasyprint']
    
    for preferred in priority_order:
        for engine_name, engine_option in available_engines:
            if engine_name == preferred:
                return engine_name, engine_option
    
    # Fallback to first available
    selected_engine = available_engines[0]
    return selected_engine[0], selected_engine[1]


def get_engine_specific_options(engine_name):
    """Get engine-specific options based on the selected PDF engine"""
    if engine_name in ['xelatex', 'lualatex']:
        # XeLaTeX and LuaLaTeX support CJK fonts
        # Get fonts based on operating system
        main_font, sans_font, mono_font = get_cjk_fonts()
        return [
            '-V', f'CJKmainfont={main_font}',
            '-V', f'CJKsansfont={sans_font}',
            '-V', f'CJKmonofont={mono_font}',
        ]
    elif engine_name == 'pdflatex':
        # pdfLaTeX doesn't support CJK fonts natively, use basic fonts
        return []
    else:
        # wkhtmltopdf and weasyprint don't use LaTeX, return minimal options
        return []


def run_pandoc_latex_conversion(input_file, output_file, filter_path=None, template_path=None):
    """Execute pandoc conversion to LaTeX format"""
    import tempfile
    
    # Preprocess images and create emoji-free markdown
    temp_files = []
    actual_input_file = input_file
    
    try:
        # Step 1: Preprocess images for LaTeX compatibility
        import sys
        from pathlib import Path
        
        # Add the project root to path for absolute imports
        script_dir = Path(__file__).parent.parent.parent
        if str(script_dir) not in sys.path:
            sys.path.insert(0, str(script_dir))
        
        from src.utils.image_preprocessor import create_preprocessed_markdown
        
        print("🖼️ Preprocessing images for LaTeX compatibility...")
        preprocessed_file, image_temp_files = create_preprocessed_markdown(Path(input_file))
        
        if preprocessed_file and preprocessed_file != Path(input_file):
            actual_input_file = str(preprocessed_file)
            temp_files.extend(image_temp_files)
            print(f"✅ Image preprocessing completed: {len(image_temp_files)} files processed")
        
    except Exception as e:
        print(f"⚠️ Warning: Image preprocessing failed: {e}")
        print("📝 Continuing with original file...")
    
    # Step 2: Create emoji-free version if needed
    try:
        temp_md_file = create_emoji_free_markdown(actual_input_file)
        if temp_md_file:
            actual_input_file = temp_md_file
            temp_files.append(temp_md_file)
    except Exception as e:
        print(f"⚠️ Warning: Failed to create emoji-free markdown: {e}")
    
    # Create temporary LaTeX header file with OS-specific fonts
    main_font, sans_font, mono_font = get_cjk_fonts()
    latex_header = f"""
\\usepackage{{float}}
\\floatplacement{{figure}}{{H}}
\\usepackage{{xeCJK}}
\\setCJKmainfont{{{main_font}}}
\\setCJKsansfont{{{sans_font}}}
\\setCJKmonofont{{{mono_font}}}
"""
    header_file = None
    
    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.tex', delete=False) as f:
            f.write(latex_header)
            header_file = f.name
    except Exception as e:
        print(f"Warning: Cannot create LaTeX header file: {e}")
    
    # Build pandoc command for LaTeX output
    cmd = [
        'pandoc',
        actual_input_file,
        '-o', output_file,
        '--to', 'latex',  # 指定输出为LaTeX格式
    ]
    
    # Add common options
    cmd.extend([
        '-V', 'fontsize=12pt',
        '-V', 'geometry:margin=2.5cm',
        '-V', 'geometry:a4paper',
        '-V', 'linestretch=1.15',
        '--highlight-style=tango',
        '-V', 'colorlinks=true',
        '-V', 'linkcolor=blue',
        '-V', 'urlcolor=blue',
        '--toc',
        '--wrap=preserve',
        '--quiet'  # Reduce warning output
    ])
    
    # Add LaTeX header file
    if header_file:
        cmd.extend(['-H', header_file])
    
    # Add filter options
    if filter_path and os.path.isfile(filter_path):
        # Filter file has shebang and execute permission, use directly
        cmd.extend(['--filter', filter_path])
    
    # Add template options
    if template_path and os.path.isfile(template_path):
        cmd.extend(['--template', template_path])
    
    # Display system and font information
    system_name = platform.system()
    main_font, sans_font, mono_font = get_cjk_fonts()
    print(f"🖥️  Operating System: {system_name}")
    print(f"🔤 CJK Fonts: {main_font}")
    
    # Execute conversion
    print(f"Converting to LaTeX: {input_file} -> {output_file}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='ignore')
        
        # Check if LaTeX file was generated
        if os.path.isfile(output_file):
            if result.returncode != 0:
                print(f"⚠️ Warnings during conversion")
                if result.stderr:
                    print(f"Warning information: {result.stderr}")
            return True, result.stdout
        else:
            print(f"❌ LaTeX conversion failed: {result.stderr}")
            return False, result.stderr
                
    except Exception as e:
        return False, str(e)
    finally:
        # Clean up temporary LaTeX header file
        if header_file and os.path.exists(header_file):
            try:
                os.remove(header_file)
            except Exception as e:
                pass
        
        # Clean up all temporary files
        try:
            import sys
            from pathlib import Path
            
            # Add the project root to path for absolute imports
            script_dir = Path(__file__).parent.parent.parent
            if str(script_dir) not in sys.path:
                sys.path.insert(0, str(script_dir))
                
            from src.utils.image_preprocessor import cleanup_temp_files
            cleanup_temp_files(temp_files)
            if temp_files:
                print(f"🗑️ Cleaned up {len(temp_files)} temporary files")
        except Exception as e:
            print(f"⚠️ Warning: Failed to clean up temporary files: {e}")
            # Fallback manual cleanup
            for temp_file in temp_files:
                if temp_file and os.path.exists(temp_file):
                    try:
                        os.remove(temp_file)
                    except Exception:
                        pass


def run_pandoc_conversion(input_file, output_file, filter_path=None, template_path=None):
    """Execute pandoc conversion with fallback PDF engines"""
    import tempfile
    
    # Check available PDF engines
    engine_name, engine_option = check_pdf_engine_availability()
    if not engine_name:
        # No PDF engines available - fail the conversion
        error_msg = "No PDF engines available. "
        return False, error_msg
    
    # Preprocess images and create emoji-free markdown
    temp_files = []
    actual_input_file = input_file
    
    try:
        # Step 1: Preprocess images for PDF compatibility
        import sys
        from pathlib import Path
        
        # Add the project root to path for absolute imports
        script_dir = Path(__file__).parent.parent.parent
        if str(script_dir) not in sys.path:
            sys.path.insert(0, str(script_dir))
        
        from src.utils.image_preprocessor import create_preprocessed_markdown
        
        print("🖼️ Preprocessing images for PDF compatibility...")
        preprocessed_file, image_temp_files = create_preprocessed_markdown(Path(input_file))
        
        if preprocessed_file and preprocessed_file != Path(input_file):
            actual_input_file = str(preprocessed_file)
            temp_files.extend(image_temp_files)
            print(f"✅ Image preprocessing completed: {len(image_temp_files)} files processed")
        
    except Exception as e:
        print(f"⚠️ Warning: Image preprocessing failed: {e}")
        print("📝 Continuing with original file...")
    
    # Step 2: Create emoji-free version if needed
    try:
        temp_md_file = create_emoji_free_markdown(actual_input_file)
        if temp_md_file:
            actual_input_file = temp_md_file
            temp_files.append(temp_md_file)
    except Exception as e:
        print(f"⚠️ Warning: Failed to create emoji-free markdown: {e}")
    
    # Create temporary LaTeX header file to fix image position (only for LaTeX engines)
    latex_header = None
    header_file = None
    
    if engine_name in ['xelatex', 'lualatex', 'pdflatex']:
        # Get OS-specific fonts
        main_font, sans_font, mono_font = get_cjk_fonts()
        latex_header = f"""
\\usepackage{{float}}
\\floatplacement{{figure}}{{H}}
\\usepackage{{xeCJK}}
\\setCJKmainfont{{{main_font}}}
\\setCJKsansfont{{{sans_font}}}
\\setCJKmonofont{{{mono_font}}}
"""
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.tex', delete=False) as f:
                f.write(latex_header)
                header_file = f.name
        except Exception as e:
            print(f"Warning: Cannot create LaTeX header file: {e}")
    
    # Build pandoc command
    cmd = [
        'pandoc',
        actual_input_file,  # Use the emoji-free file if available
        '-o', output_file,
        engine_option,  # Use the selected engine
    ]
    
    # Add engine-specific options
    engine_options = get_engine_specific_options(engine_name)
    cmd.extend(engine_options)
    
    # Add common options
    cmd.extend([
        '-V', 'fontsize=12pt',
        '-V', 'geometry:margin=2.5cm',
        '-V', 'geometry:a4paper',
        '-V', 'linestretch=1.15',
        '--highlight-style=tango',
        '-V', 'colorlinks=true',
        '-V', 'linkcolor=blue',
        '-V', 'urlcolor=blue',
        '--toc',
        '--wrap=preserve',
        '--quiet'  # Reduce warning output
    ])
    
    # Add LaTeX-specific options only for LaTeX engines
    if engine_name in ['xelatex', 'lualatex', 'pdflatex']:
        cmd.extend([
            '-V', 'graphics=true',
        ])
        
        # Add LaTeX header file for fixing image position
        if header_file:
            cmd.extend(['-H', header_file])
    
    # Add filter options (only for LaTeX engines)
    if engine_name in ['xelatex', 'lualatex', 'pdflatex'] and filter_path and os.path.isfile(filter_path):
        # Filter file has shebang and execute permission, use directly
        cmd.extend(['--filter', filter_path])
    
    # Add template options (only for LaTeX engines)
    if engine_name in ['xelatex', 'lualatex', 'pdflatex'] and template_path and os.path.isfile(template_path):
        cmd.extend(['--template', template_path])
    
    # Display system and font information
    system_name = platform.system()
    main_font, sans_font, mono_font = get_cjk_fonts()
    print(f"🖥️  Operating System: {system_name}")
    print(f"🔤 CJK Fonts: {main_font}")
    
    # Execute conversion
    print(f"Converting: {input_file} -> {output_file}")
    print(f"Using PDF engine: {engine_name}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='ignore')
        
        # Check if PDF file was generated
        if os.path.isfile(output_file):
            # Even with warnings or errors
            if result.returncode != 0:
                print(f"⚠️ Warnings during conversion")
                if result.stderr:
                    print(f"Warning information: {result.stderr}")
            return True, result.stdout
        else:
            # No PDF file generated
            print(f"❌ PDF conversion failed: {result.stderr}")
            return False, result.stderr
                
    except Exception as e:
        return False, str(e)
    finally:
        # Clean up temporary LaTeX header file
        if header_file and os.path.exists(header_file):
            try:
                os.remove(header_file)
            except Exception as e:
                pass
        
        # Clean up all temporary files
        try:
            import sys
            from pathlib import Path
            
            # Add the project root to path for absolute imports
            script_dir = Path(__file__).parent.parent.parent
            if str(script_dir) not in sys.path:
                sys.path.insert(0, str(script_dir))
                
            from src.utils.image_preprocessor import cleanup_temp_files
            cleanup_temp_files(temp_files)
            if temp_files:
                print(f"🗑️ Cleaned up {len(temp_files)} temporary files")
        except Exception as e:
            print(f"⚠️ Warning: Failed to clean up temporary files: {e}")
            # Fallback manual cleanup
            for temp_file in temp_files:
                if temp_file and os.path.exists(temp_file):
                    try:
                        os.remove(temp_file)
                    except Exception:
                        pass





def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description='Convert Markdown files to PDF or LaTeX',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  %(prog)s document.md output.pdf
  %(prog)s -i document.md -o output.pdf
  %(prog)s -i document.md -o output.tex --latex
  %(prog)s document.md output.tex --latex
        """
    )
    
    parser.add_argument('input_file', nargs='?', help='Input Markdown file')
    parser.add_argument('output_file', nargs='?', help='Output PDF or LaTeX file')
    parser.add_argument('-i', '--input', help='Input Markdown file')
    parser.add_argument('-o', '--output', help='Output PDF or LaTeX file')
    parser.add_argument('--latex', action='store_true', help='Generate LaTeX source file instead of PDF')
    parser.add_argument('--no-template', action='store_true', help='Do not use LaTeX template (useful when template dependencies are missing)')
    
    args = parser.parse_args()
    
    # Determine input and output files
    input_file = args.input_file or args.input
    output_file = args.output_file or args.output
    
    # Check parameters
    if not input_file or not output_file:
        print("Usage: python trans_md_to_pdf.py <input.md> <output.pdf|output.tex> [--latex]")
        print("Example: python trans_md_to_pdf.py document.md output.pdf")
        print("Example: python trans_md_to_pdf.py document.md output.tex --latex")
        sys.exit(1)
    
    # Check if input file exists
    if not check_file_exists(input_file, "Input file"):
        sys.exit(1)
    
    # Get script directory
    script_dir = get_script_dir()
    
    # Set filter path
    filter_path = script_dir / "svg_chinese_filter.py"
    
    # Set template path (only if not disabled)
    template_path = None
    if not args.no_template:
        template_path = script_dir / "template.latex"
        if not template_path.exists():
            template_path = None
    
    # Display template status
    if template_path:
        print(f"📄 Using template: {template_path.name}")
    else:
        print("📄 Template: Disabled")
    
    # Execute conversion based on output format
    if args.latex or output_file.lower().endswith('.tex'):
        # Generate LaTeX source file
        success, output = run_pandoc_latex_conversion(
            input_file, 
            output_file, 
            str(filter_path) if filter_path.exists() else None,
            str(template_path) if template_path else None
        )
    else:
        # Generate PDF file
        success, output = run_pandoc_conversion(
            input_file, 
            output_file, 
            str(filter_path) if filter_path.exists() else None,
            str(template_path) if template_path else None
        )
    
    # Check conversion result
    if success and os.path.isfile(output_file):
        output_type = "LaTeX" if (args.latex or output_file.lower().endswith('.tex')) else "PDF"
        print(f"✓ {output_type} conversion successful: {output_file}")
        # Display file size
        file_size = os.path.getsize(output_file)
        print(f"File size: {file_size / 1024:.1f} KB")
    else:
        output_type = "LaTeX" if (args.latex or output_file.lower().endswith('.tex')) else "PDF"
        print(f"✗ {output_type} conversion failed")
        if output:
            print(f"Error information: {output}")
        sys.exit(1)


if __name__ == "__main__":
    main() 