#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import subprocess
import argparse
from pathlib import Path


def check_file_exists(file_path, description):
    """Check if file exists"""
    if not os.path.isfile(file_path):
        print(f"Error: {description} '{file_path}' Does not exist")
        return False
    return True


def get_script_dir():
    """Get script directory"""
    return Path(__file__).parent.absolute()


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
                                     capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    available_engines.append((engine_name, engine_option))
            elif engine_name == 'wkhtmltopdf':
                # Check if wkhtmltopdf is available
                result = subprocess.run(['wkhtmltopdf', '--version'], 
                                     capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    available_engines.append((engine_name, engine_option))
            elif engine_name == 'weasyprint':
                # Check if weasyprint is available
                result = subprocess.run(['weasyprint', '--version'], 
                                     capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    available_engines.append((engine_name, engine_option))
        except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.CalledProcessError):
            continue
    
    if not available_engines:
        print("❌ No PDF engines available. Please install at least one of: xelatex, lualatex, pdflatex, wkhtmltopdf, or weasyprint")
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
        return [
            '-V', 'CJKmainfont=Noto Serif CJK SC',
            '-V', 'CJKsansfont=Noto Sans CJK SC',
            '-V', 'CJKmonofont=Noto Sans Mono CJK SC',
            '-V', 'mainfont=DejaVu Serif',
            '-V', 'sansfont=DejaVu Sans',
            '-V', 'monofont=DejaVu Sans Mono',
        ]
    elif engine_name == 'pdflatex':
        # pdfLaTeX doesn't support CJK fonts natively, use basic fonts
        return [
            '-V', 'mainfont=DejaVu Serif',
            '-V', 'sansfont=DejaVu Sans',
            '-V', 'monofont=DejaVu Sans Mono',
        ]
    else:
        # wkhtmltopdf and weasyprint don't use LaTeX, return minimal options
        return []


def run_pandoc_conversion(input_file, output_file, filter_path=None, template_path=None):
    """Execute pandoc conversion with fallback PDF engines"""
    import tempfile
    
    # Check available PDF engines
    engine_name, engine_option = check_pdf_engine_availability()
    if not engine_name:
        return False, "No PDF engines available"
    
    # Create temporary LaTeX header file to fix image position (only for LaTeX engines)
    latex_header = None
    header_file = None
    
    if engine_name in ['xelatex', 'lualatex', 'pdflatex']:
        latex_header = """
\\usepackage{float}
\\floatplacement{figure}{H}
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
        input_file,
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
        '-V', 'linestretch=2.0',
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
        cmd.extend(['--filter', filter_path])
    
    # Add template options (only for LaTeX engines)
    if engine_name in ['xelatex', 'lualatex', 'pdflatex'] and template_path and os.path.isfile(template_path):
        cmd.extend(['--template', template_path])
    
    # Execute conversion
    print(f"Converting: {input_file} -> {output_file}")
    print(f"Using PDF engine: {engine_name}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        
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
            return False, result.stderr if result.stderr else result.stdout
    except Exception as e:
        return False, str(e)
    finally:
        # Clean up temporary LaTeX header file
        if header_file and os.path.exists(header_file):
            try:
                os.remove(header_file)
            except Exception as e:
                pass


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description='Convert Markdown files to PDF',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  %(prog)s document.md output.pdf
  %(prog)s -i document.md -o output.pdf
        """
    )
    
    parser.add_argument('input_file', nargs='?', help='Input Markdown file')
    parser.add_argument('output_file', nargs='?', help='Output PDF file')
    parser.add_argument('-i', '--input', help='Input Markdown file')
    parser.add_argument('-o', '--output', help='Output PDF file')
    
    args = parser.parse_args()
    
    # Determine input and output files
    input_file = args.input_file or args.input
    output_file = args.output_file or args.output
    
    # Check parameters
    if not input_file or not output_file:
        print("Usage: python trans_md_to_pdf.py <input.md> <output.pdf>")
        print("Example: python trans_md_to_pdf.py document.md output.pdf")
        sys.exit(1)
    
    # Check if input file exists
    if not check_file_exists(input_file, "Input file"):
        sys.exit(1)
    
    # Get script directory
    script_dir = get_script_dir()
    
    # Set filter path
    filter_path = script_dir / "svg_chinese_filter.py"
    
    # Set template path
    template_path = script_dir / "template.latex"
    
    # Execute conversion
    success, output = run_pandoc_conversion(
        input_file, 
        output_file, 
        str(filter_path) if filter_path.exists() else None,
        str(template_path) if template_path.exists() else None
    )
    
    # Check conversion result
    if success and os.path.isfile(output_file):
        print(f"✓ Conversion successful: {output_file}")
        # Display file size
        file_size = os.path.getsize(output_file)
        print(f"File size: {file_size / 1024:.1f} KB")
    else:
        print("✗ Conversion failed")
        if output:
            print(f"Error information: {output}")
        sys.exit(1)


if __name__ == "__main__":
    main() 