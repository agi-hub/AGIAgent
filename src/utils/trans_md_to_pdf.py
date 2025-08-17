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


def run_pandoc_conversion(input_file, output_file, filter_path=None, template_path=None):
    """Execute pandoc conversion"""
    import tempfile
    
    # Create temporary LaTeX header file to fix image position
    latex_header = """
\\usepackage{float}
\\floatplacement{figure}{H}
"""
    
    header_file = None
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
        '--pdf-engine=xelatex',
        '-V', 'CJKmainfont=Noto Serif CJK SC',
        '-V', 'CJKsansfont=Noto Sans CJK SC',
        '-V', 'CJKmonofont=Noto Sans Mono CJK SC',
        '-V', 'mainfont=DejaVu Serif',
        '-V', 'sansfont=DejaVu Sans',
        '-V', 'monofont=DejaVu Sans Mono',
        '-V', 'fontsize=12pt',
        '-V', 'geometry:margin=2.5cm',
        '-V', 'geometry:a4paper',
        '-V', 'linestretch=2.0',
        '--highlight-style=tango',
        '-V', 'colorlinks=true',
        '-V', 'linkcolor=blue',
        '-V', 'urlcolor=blue',
        '-V', 'graphics=true',
        '--toc',
        '--wrap=preserve',
        '--quiet'  # Reduce warning output
    ]
    
    # Add LaTeX header file for fixing image position
    if header_file:
        cmd.extend(['-H', header_file])
        print(f"Use LaTeX header file: {header_file}")
    
    # Add filter options
    if filter_path and os.path.isfile(filter_path):
        cmd.extend(['--filter', filter_path])
        print(f"Use SVG Chinese filter: {filter_path}")
    else:
        print("Warning: SVG Chinese filter does not exist")
    
    # Add template options
    if template_path and os.path.isfile(template_path):
        cmd.extend(['--template', template_path])
        print(f"Use custom template: {template_path}")
    else:
        print("Warning: Custom template does not exist")
    
    # Execute conversion
    print(f"Converting: {input_file} -> {output_file}")
    print(f"Execute command: {' '.join(cmd)}")
    
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
                print(f"Clean up temporary files: {header_file}")
            except Exception as e:
                print(f"Warning: Cannot delete temporary file {header_file}: {e}")


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