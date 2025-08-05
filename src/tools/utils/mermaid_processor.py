#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mermaid Chart Processor Utility

This module provides functionality to process markdown files containing Mermaid charts,
convert them to images using multiple methods (CLI, Playwright, Python library, or online API),
and replace the chart code with image references.
"""

import re
import os
import subprocess
import tempfile
import requests
import base64
from pathlib import Path
from typing import List, Dict, Any, Optional
from ..print_system import print_current, print_system_info, print_debug

# Check for multiple mermaid rendering methods
def _check_mermaid_cli():
    """Check if mermaid-cli (mmdc) is available"""
    try:
        result = subprocess.run(['mmdc', '--version'], 
                              capture_output=True, text=True, timeout=10)
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False

def _check_playwright():
    """Check if playwright is available for rendering"""
    try:
        import playwright
        return True
    except ImportError:
        return False

def _check_mermaid_py():
    """Check if mmdc CLI is available (fallback method)"""
    # This is now just a fallback that uses the same CLI as the primary method
    return MERMAID_CLI_AVAILABLE

# Check available methods
MERMAID_CLI_AVAILABLE = _check_mermaid_cli()
PLAYWRIGHT_AVAILABLE = _check_playwright()
MERMAID_PY_AVAILABLE = _check_mermaid_py()
ONLINE_FALLBACK = True  # Always available as fallback

# Determine best available method
if MERMAID_CLI_AVAILABLE:
    PREFERRED_METHOD = "cli"
    print_debug("✅ Mermaid CLI (mmdc) available - using CLI method")
elif PLAYWRIGHT_AVAILABLE:
    PREFERRED_METHOD = "playwright"
    print_debug("✅ Playwright available - using browser rendering method")
elif MERMAID_PY_AVAILABLE:
    PREFERRED_METHOD = "python"
    print_debug("✅ mermaid-py available - using Python method")
else:
    PREFERRED_METHOD = "online"
    print_debug("⚠️ No local Mermaid tools available - using online fallback method")


class MermaidProcessor:
    """
    Processor for converting Mermaid charts in markdown files to images using multiple methods.
    
    Supports the following rendering methods (in order of preference):
    1. Mermaid CLI (mmdc) with default theme - requires npm install -g @mermaid-js/mermaid-cli
    2. Playwright - requires pip install playwright && playwright install chromium
    3. Mermaid CLI (mmdc) with neutral theme - fallback using same CLI tool
    4. Online API fallback - uses mermaid.ink (requires internet connection)
    """
    
    def __init__(self):
        """Initialize the Mermaid processor."""
        self.preferred_method = PREFERRED_METHOD
        self.mermaid_available = (MERMAID_CLI_AVAILABLE or PLAYWRIGHT_AVAILABLE or 
                                MERMAID_PY_AVAILABLE or ONLINE_FALLBACK)
        
        print_debug(f"🎨 Mermaid processor initialized with method: {self.preferred_method}")
        
        if not self.mermaid_available:
            print_debug("⚠️ No Mermaid rendering methods available")
    
    def process_markdown_file(self, md_file_path: str, output_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Process markdown file to convert Mermaid charts to images.
        
        Args:
            md_file_path: Path to the markdown file
            output_dir: Output directory for images (optional)
            
        Returns:
            Dictionary with processing results
        """
        try:
            #print_current(f"🎨 Processing Mermaid charts in: {md_file_path}")
            
            md_path = Path(md_file_path).absolute()
            md_dir = md_path.parent
            
            # Use markdown file directory if no output dir specified
            if not output_dir:
                output_dir = md_dir
            else:
                output_dir = Path(output_dir)
            
            # Create images directory
            img_dir = output_dir / "images"
            img_dir.mkdir(exist_ok=True)
            
            # Read markdown file
            with open(md_file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Extract all Mermaid code blocks
            pattern = re.compile(r'```mermaid\n(.*?)\n```', re.DOTALL)
            matches = list(pattern.finditer(content))
            
            if not matches:
                print_current("📝 No Mermaid charts found")
                return {
                    'status': 'success',
                    'file': md_file_path,
                    'charts_found': 0,
                    'charts_processed': 0,
                    'message': 'No Mermaid charts found'
                }
            
            print_current(f"📊 Found {len(matches)} Mermaid chart(s)")
            
            processed_count = 0
            
            # Process from end to beginning to avoid position shifts
            for i, match in enumerate(reversed(matches)):
                try:
                    code = match.group(1).strip()
                    # Generate image filename (use SVG for vector graphics)
                    img_name = f"mermaid_{len(matches)-i}.svg"
                    img_path = img_dir / img_name
                    rel_img_path = f"images/{img_name}"
                    
                    # Generate corresponding mermaid code filename
                    mermaid_code_name = f"mermaid_{len(matches)-i}.mmd"
                    mermaid_code_path = img_dir / mermaid_code_name
                    rel_mermaid_path = f"images/{mermaid_code_name}"
                    
                    #print_current(f"🔧 Processing Mermaid chart {len(matches)-i}...")
                    
                    # Save original Mermaid code to separate file
                    try:
                        with open(mermaid_code_path, 'w', encoding='utf-8') as f:
                            f.write(code)
                        print_current(f"📝 Saved Mermaid code: {mermaid_code_name}")
                    except Exception as e:
                        print_current(f"❌ Failed to save Mermaid code: {e}")
                    
                    # Try to generate image using the best available method
                    success = self._generate_mermaid_image(code, img_path)
                    
                    # If successful, replace content
                    if success:
                        # Determine appropriate size based on chart type
                        chart_type = code.strip().split('\n')[0] if '\n' in code else code.strip()
                        if 'mindmap' in chart_type.lower():
                            # Mind maps tend to be wide, limit height more
                            size_style = "max-width: 80%; height: auto; max-height: 400px; width: 80%;"
                        elif 'sequenceDiagram' in chart_type or 'sequence' in chart_type.lower():
                            # Sequence diagrams can be tall, allow more height
                            size_style = "max-width: 80%; height: auto; max-height: 640px; width: 80%;"
                        elif 'flowchart' in chart_type.lower() or 'graph' in chart_type.lower():
                            # Flowcharts are usually moderate size
                            size_style = "max-width: 80%; height: auto; max-height: 480px; width: 80%;"
                        elif 'gantt' in chart_type.lower():
                            # Gantt charts can be wide but not too tall
                            size_style = "max-width: 80%; height: auto; max-height: 320px; width: 80%;"
                        else:
                            # Default size for other chart types
                            size_style = "max-width: 80%; height: auto; max-height: 480px; width: 80%;"
                        
                        replacement = f"\n<div align=\"center\">\n\n<img src=\"{rel_img_path}\" alt=\"Mermaid Chart {len(matches)-i}\" style=\"{size_style}\" />\n\n</div>\n\n<!-- 源代码文件: {rel_mermaid_path} -->\n"
                        
                        # Get complete Mermaid code block positions
                        start_pos = match.start()
                        end_pos = match.end()
                        
                        # Replace original content
                        content = content[:start_pos] + replacement + content[end_pos:]
                        
                        print_current(f"✅ Successfully generated: {img_name}")
                        processed_count += 1
                    else:
                        print_current(f"❌ Failed to generate image, keeping original Mermaid code")
                        
                except Exception as e:
                    print_current(f"❌ Error processing Mermaid chart: {e}")
            
            # Write updated file
            with open(md_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            print_current(f"✅ Processing complete. Modified file saved to: {md_path}")
            print_current(f"📁 Generated images saved in: {img_dir}")
            
            return {
                'status': 'success',
                'file': md_file_path,
                'charts_found': len(matches),
                'charts_processed': processed_count,
                'images_dir': str(img_dir),
                'message': f'Successfully processed {processed_count}/{len(matches)} Mermaid charts'
            }
            
        except Exception as e:
            print_current(f"❌ Error processing markdown file: {e}")
            return {
                'status': 'failed',
                'file': md_file_path,
                'error': str(e),
                'message': f'Failed to process markdown file: {e}'
            }
    
    def _generate_mermaid_image(self, mermaid_code: str, output_path: Path) -> bool:
        """
        Generate Mermaid image using the best available method.
        
        Args:
            mermaid_code: Mermaid chart code
            output_path: Output image path
            
        Returns:
            True if successful, False otherwise
        """
        # Try methods in order of preference
        methods = [
            ("cli", self._generate_mermaid_image_cli),
            ("playwright", self._generate_mermaid_image_playwright),
            ("python", self._generate_mermaid_image_python),
            ("online", self._generate_mermaid_image_online)
        ]
        
        # Try preferred method first
        for method_name, method_func in methods:
            if method_name == self.preferred_method:
                try:
                    if method_func(mermaid_code, output_path):
                        print_debug(f"✅ Successfully generated using {method_name} method")
                        return True
                except Exception as e:
                    print_debug(f"❌ {method_name} method failed: {e}")
        
        # Try other methods as fallbacks
        for method_name, method_func in methods:
            if method_name != self.preferred_method:
                try:
                    if method_func(mermaid_code, output_path):
                        print_debug(f"✅ Successfully generated using fallback {method_name} method")
                        return True
                except Exception as e:
                    print_debug(f"❌ Fallback {method_name} method failed: {e}")
        
        print_debug("❌ All methods failed to generate image")
        return False
    
    def _generate_mermaid_image_cli(self, mermaid_code: str, output_path: Path) -> bool:
        """
        Generate Mermaid image using mermaid CLI (mmdc).
        
        Args:
            mermaid_code: Mermaid chart code
            output_path: Output image path
            
        Returns:
            True if successful, False otherwise
        """
        if not MERMAID_CLI_AVAILABLE:
            return False
            
        try:
            print_debug(f"🔧 Using Mermaid CLI to generate image...")
            
            # Create temporary mermaid file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.mmd', delete=False, encoding='utf-8') as temp_file:
                temp_file.write(mermaid_code)
                temp_mermaid_path = temp_file.name
            
            try:
                # Use mermaid CLI to generate high-quality SVG image
                cmd = [
                    'mmdc',
                    '-i', temp_mermaid_path,
                    '-o', str(output_path),
                    '-b', 'transparent',  # transparent background
                    '--quiet'  # suppress output
                ]
                
                # Only add theme if the mermaid code doesn't have custom theme configuration
                if not ('%%{init:' in mermaid_code and 'theme' in mermaid_code):
                    cmd.extend(['-t', 'default'])  # use default theme only if no custom theme
                
                # Add SVG-specific parameters if output is SVG
                if str(output_path).endswith('.svg'):
                    cmd.extend(['-e', 'svg'])  # explicitly set SVG format
                
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
                
                # Check if command was successful and file was created
                if result.returncode == 0 and output_path.exists() and output_path.stat().st_size > 0:
                    print_debug(f"✅ Mermaid CLI generation successful")
                    return True
                else:
                    print_debug(f"❌ Mermaid CLI generation failed: {result.stderr}")
                    return False
                    
            finally:
                # Clean up temporary file
                try:
                    os.unlink(temp_mermaid_path)
                except OSError:
                    pass
                
        except subprocess.TimeoutExpired:
            print_debug(f"❌ Mermaid CLI generation timed out")
            return False
        except Exception as e:
            print_debug(f"❌ Mermaid CLI generation failed: {e}")
            return False
    
    def _generate_mermaid_image_playwright(self, mermaid_code: str, output_path: Path) -> bool:
        """
        Generate Mermaid image using Playwright browser automation.
        
        Args:
            mermaid_code: Mermaid chart code
            output_path: Output image path
            
        Returns:
            True if successful, False otherwise
        """
        if not PLAYWRIGHT_AVAILABLE:
            return False
            
        try:
            from playwright.sync_api import sync_playwright
            
            print_debug(f"🌐 Using Playwright to generate image...")
            
            # Create HTML with Mermaid chart
            html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
    <style>
        body {{ margin: 0; padding: 20px; background: transparent; }}
        .mermaid {{ background: transparent; }}
    </style>
</head>
<body>
    <div class="mermaid">{mermaid_code}</div>
    <script>
        mermaid.initialize({{ startOnLoad: true, theme: 'default' }});
    </script>
</body>
</html>"""
            
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True)
                page = browser.new_page()
                
                # Set viewport
                page.set_viewport_size({"width": 1200, "height": 800})
                
                # Load HTML content
                page.set_content(html_content)
                
                # Wait for Mermaid to render
                page.wait_for_selector(".mermaid svg", timeout=10000)
                
                # Take screenshot of the mermaid element
                mermaid_element = page.locator(".mermaid")
                mermaid_element.screenshot(path=str(output_path))
                
                browser.close()
            
            # Check if file was created successfully
            if output_path.exists() and output_path.stat().st_size > 0:
                print_debug(f"✅ Playwright generation successful")
                return True
            else:
                print_debug(f"❌ Playwright generation failed: output file not created or empty")
                return False
                
        except Exception as e:
            print_debug(f"❌ Playwright generation failed: {e}")
            return False
    
    def _generate_mermaid_image_python(self, mermaid_code: str, output_path: Path) -> bool:
        """
        Generate Mermaid image using mmdc CLI (fallback method).
        
        Args:
            mermaid_code: Mermaid chart code
            output_path: Output image path
            
        Returns:
            True if successful, False otherwise
        """
        if not MERMAID_PY_AVAILABLE:
            return False
            
        try:
            import tempfile
            import subprocess
            import os
            
            print_debug(f"🐍 Using mmdc CLI (fallback method) to generate image...")
            
            # Create temporary mermaid file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.mmd', delete=False, encoding='utf-8') as temp_file:
                temp_file.write(mermaid_code)
                temp_mermaid_path = temp_file.name
            
            try:
                # Use mermaid CLI with neutral theme for fallback
                cmd = [
                    'mmdc',
                    '-i', temp_mermaid_path,
                    '-o', str(output_path),
                    '-b', 'transparent',  # transparent background
                    '--quiet'  # suppress output
                ]
                
                # Only add theme if the mermaid code doesn't have custom theme configuration
                if not ('%%{init:' in mermaid_code and 'theme' in mermaid_code):
                    cmd.extend(['-t', 'neutral'])  # use neutral theme for fallback only if no custom theme
                
                # Add SVG-specific parameters if output is SVG
                if str(output_path).endswith('.svg'):
                    cmd.extend(['-e', 'svg'])  # explicitly set SVG format
                
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
                
                # Check if command was successful and file was created
                if result.returncode == 0 and output_path.exists() and output_path.stat().st_size > 0:
                    print_debug(f"✅ CLI fallback generation successful")
                    return True
                else:
                    print_debug(f"❌ CLI fallback generation failed: {result.stderr}")
                    return False
                    
            finally:
                # Clean up temporary file
                try:
                    os.unlink(temp_mermaid_path)
                except OSError:
                    pass
                
        except Exception as e:
            print_debug(f"❌ CLI fallback generation failed: {e}")
            return False
    
    def _generate_mermaid_image_online(self, mermaid_code: str, output_path: Path) -> bool:
        """
        Generate Mermaid image using online mermaid.ink API.
        
        Args:
            mermaid_code: Mermaid chart code
            output_path: Output image path
            
        Returns:
            True if successful, False otherwise
        """
        try:
            print_debug(f"🌐 Using online mermaid.ink API to generate image...")
            
            # Encode Mermaid code to base64
            encoded_code = base64.b64encode(mermaid_code.encode('utf-8')).decode('utf-8')
            # Use SVG endpoint for vector graphics
            if str(output_path).endswith('.svg'):
                api_url = f"https://mermaid.ink/svg/{encoded_code}"
            else:
                api_url = f"https://mermaid.ink/img/{encoded_code}"
            
            # Download image
            response = requests.get(api_url, timeout=30)
            if response.status_code == 200:
                with open(output_path, 'wb') as f:
                    f.write(response.content)
                print_debug(f"✅ Online API generation successful")
                return True
            else:
                print_debug(f"❌ Online API failed, status code: {response.status_code}")
                return False
                
        except requests.exceptions.RequestException as e:
            print_debug(f"❌ Network request failed: {e}")
            return False
        except Exception as e:
            print_debug(f"❌ Online API generation failed: {e}")
            return False
    
    def has_mermaid_charts(self, md_file_path: str) -> bool:
        """
        Check if a markdown file contains Mermaid charts.
        
        Args:
            md_file_path: Path to the markdown file
            
        Returns:
            True if Mermaid charts are found, False otherwise
        """
        try:
            with open(md_file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            pattern = re.compile(r'```mermaid\n(.*?)\n```', re.DOTALL)
            matches = pattern.findall(content)
            
            return len(matches) > 0
            
        except Exception as e:
            print_debug(f"❌ Error checking for Mermaid charts: {e}")
            return False
    
    def scan_and_process_directory(self, directory_path: str) -> Dict[str, Any]:
        """
        Scan directory for markdown files and process Mermaid charts.
        
        Args:
            directory_path: Directory to scan
            
        Returns:
            Dictionary with processing results
        """
        try:
            print_current(f"📂 Scanning directory: {directory_path}")
            
            # Find markdown files in root directory only (not recursive)
            markdown_files = []
            try:
                for file in os.listdir(directory_path):
                    file_path = os.path.join(directory_path, file)
                    if os.path.isfile(file_path) and file.endswith('.md'):
                        markdown_files.append(file_path)
            except Exception as e:
                print_current(f"❌ Failed to scan directory: {e}")
                return {
                    'status': 'failed',
                    'error': str(e),
                    'message': f'Failed to scan directory: {e}'
                }
            
            if not markdown_files:
                print_current("❌ No markdown files found")
                return {
                    'status': 'success',
                    'files_found': 0,
                    'files_processed': 0,
                    'message': 'No markdown files found'
                }
            
            print_current(f"📄 Found {len(markdown_files)} markdown file(s):")
            for file in markdown_files:
                print_current(f"   - {file}")
            
            # Process each markdown file
            processed_count = 0
            total_charts = 0
            total_processed_charts = 0
            
            for markdown_file in markdown_files:
                print_current(f"\n🔧 Processing file: {markdown_file}")
                try:
                    result = self.process_markdown_file(markdown_file)
                    if result['status'] == 'success':
                        processed_count += 1
                        total_charts += result['charts_found']
                        total_processed_charts += result['charts_processed']
                except Exception as e:
                    print_current(f"❌ Failed to process file: {markdown_file}, error: {e}")
            
            print_current(f"\n✅ Mermaid processing complete! Successfully processed {processed_count}/{len(markdown_files)} files")
            print_current(f"📊 Total charts processed: {total_processed_charts}/{total_charts}")
            print_current(f"📁 Images saved in images directories alongside markdown files")
            
            return {
                'status': 'success',
                'files_found': len(markdown_files),
                'files_processed': processed_count,
                'total_charts_found': total_charts,
                'total_charts_processed': total_processed_charts,
                'message': f'Successfully processed {processed_count}/{len(markdown_files)} files'
            }
            
        except Exception as e:
            print_current(f"❌ Error during directory processing: {e}")
            return {
                'status': 'failed',
                'error': str(e),
                'message': f'Failed to process directory: {e}'
            }


# Create a global instance for easy access
mermaid_processor = MermaidProcessor()