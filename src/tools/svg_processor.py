#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SVG Processor for Markdown Files

This module processes SVG code blocks in markdown files and converts them to PNG images.
It detects ```svg code blocks, generates separate SVG files, converts them to PNG,
and updates the markdown with image links.

Copyright (c) 2025 AGI Agent Research Group.
Licensed under the Apache License, Version 2.0
"""

import os
import re
import hashlib
import tempfile
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from .print_system import print_system, print_current, print_debug


class SVGProcessor:
    """SVG processor for markdown files"""
    
    def __init__(self, workspace_root: Optional[str] = None):
        """Initialize the SVG processor"""
        self.workspace_root = workspace_root or os.getcwd()
        self.svg_output_dir = "images"  # Directory for generated images
        self._check_dependencies()
    
    def set_workspace_root(self, workspace_root: str):
        """Set the workspace root directory"""
        self.workspace_root = workspace_root
    
    def _check_dependencies(self):
        """Check if required dependencies are available"""
        self.inkscape_available = self._check_command_available('inkscape')
        self.rsvg_convert_available = self._check_command_available('rsvg-convert')
        self.cairosvg_available = self._check_python_package('cairosvg')
        
        if self.inkscape_available:
            print_debug("🎨 Inkscape detected for SVG to PNG conversion")
        elif self.rsvg_convert_available:
            print_debug("🎨 rsvg-convert detected for SVG to PNG conversion")
        elif self.cairosvg_available:
            print_debug("🎨 CairoSVG detected for SVG to PNG conversion")
        else:
            print_debug("⚠️ No SVG conversion tools available. Please install inkscape, rsvg-convert, or cairosvg")
    
    def _check_command_available(self, command: str) -> bool:
        """Check if a command is available in the system"""
        try:
            result = subprocess.run([command, '--version'], 
                                  capture_output=True, text=True, timeout=5)
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.CalledProcessError):
            return False
    
    def _check_python_package(self, package: str) -> bool:
        """Check if a Python package is available"""
        try:
            __import__(package)
            return True
        except (ImportError, OSError, Exception):
            # Catch all exceptions including OSError from missing libraries
            return False
    
    def has_svg_blocks(self, markdown_file: str) -> bool:
        """
        Check if a markdown file contains SVG code blocks (including malformed ones)
        
        Args:
            markdown_file: Path to the markdown file
            
        Returns:
            True if SVG blocks are found, False otherwise
        """
        try:
            with open(markdown_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Pattern to match standard ```svg code blocks
            standard_pattern = r'```svg\s*\n(.*?)\n```'
            standard_matches = re.findall(standard_pattern, content, re.DOTALL | re.IGNORECASE)
            
            # Pattern to match malformed ```svg code blocks (missing closing ```)
            malformed_pattern = r'```svg\s*\n(.*?</svg>)(?!\s*\n```)'
            malformed_matches = re.findall(malformed_pattern, content, re.DOTALL | re.IGNORECASE)
            
            total_matches = len(standard_matches) + len(malformed_matches)
            
            if total_matches > 0:
                print_debug(f"📊 Found {len(standard_matches)} standard + {len(malformed_matches)} malformed SVG blocks")
            
            return total_matches > 0
            
        except Exception as e:
            print_debug(f"❌ Error checking SVG blocks in {markdown_file}: {e}")
            return False
    
    def extract_svg_blocks(self, content: str) -> List[Dict[str, Any]]:
        """
        Extract SVG code blocks from markdown content with error tolerance
        
        Args:
            content: Markdown content
            
        Returns:
            List of dictionaries containing SVG block information
        """
        svg_blocks = []
        
        # First try standard pattern
        standard_pattern = r'```svg\s*\n(.*?)\n```'
        
        for match in re.finditer(standard_pattern, content, re.DOTALL | re.IGNORECASE):
            svg_code = match.group(1).strip()
            start_pos = match.start()
            end_pos = match.end()
            full_block = match.group(0)
            
            # Generate a unique ID for this SVG block based on content hash
            svg_hash = hashlib.md5(svg_code.encode('utf-8')).hexdigest()[:8]
            
            svg_blocks.append({
                'id': svg_hash,
                'svg_code': svg_code,
                'full_block': full_block,
                'start_pos': start_pos,
                'end_pos': end_pos,
                'is_corrected': False
            })
        
        # Apply error tolerance for malformed SVG blocks
        corrected_content = self._apply_svg_error_tolerance(content)
        
        # If content was corrected, re-extract blocks
        if corrected_content != content:
            print_debug("🔧 Applied SVG error tolerance corrections")
            
            # Clear previous blocks and re-extract from corrected content
            svg_blocks = []
            
            for match in re.finditer(standard_pattern, corrected_content, re.DOTALL | re.IGNORECASE):
                svg_code = match.group(1).strip()
                start_pos = match.start()
                end_pos = match.end()
                full_block = match.group(0)
                
                # Generate a unique ID for this SVG block based on content hash
                svg_hash = hashlib.md5(svg_code.encode('utf-8')).hexdigest()[:8]
                
                svg_blocks.append({
                    'id': svg_hash,
                    'svg_code': svg_code,
                    'full_block': full_block,
                    'start_pos': start_pos,
                    'end_pos': end_pos,
                    'is_corrected': True,
                    'original_content': content,
                    'corrected_content': corrected_content
                })
        
        print_debug(f"📊 Found {len(svg_blocks)} SVG code blocks")
        return svg_blocks
    
    def _apply_svg_error_tolerance(self, content: str) -> str:
        """
        Apply error tolerance to fix malformed SVG code blocks
        
        This method looks for SVG blocks that start with ```svg but end with </svg>
        without a proper ``` ending, and fixes them by adding the missing closing marker.
        
        Args:
            content: Original markdown content
            
        Returns:
            Corrected markdown content
        """
        corrected_content = content
        corrections_made = 0
        
        # Pattern to find ```svg blocks that might be malformed
        # Look for ```svg followed by content ending with </svg> but no closing ```
        malformed_pattern = r'```svg\s*\n(.*?</svg>)(?!\s*\n```)'
        
        def fix_malformed_block(match):
            nonlocal corrections_made
            svg_content = match.group(1)
            
            # Check if the next line after </svg> is NOT ```
            full_match = match.group(0)
            
            # Add the missing closing ```
            corrected_block = f"```svg\n{svg_content}\n```"
            corrections_made += 1
            
            print_debug(f"🔧 Fixed malformed SVG block #{corrections_made}: added missing closing ```")
            return corrected_block
        
        # Apply the fix
        corrected_content = re.sub(malformed_pattern, fix_malformed_block, corrected_content, flags=re.DOTALL | re.IGNORECASE)
        
        if corrections_made > 0:
            print_debug(f"✅ Applied {corrections_made} SVG error tolerance corrections")
        
        return corrected_content
    
    def generate_svg_file(self, svg_code: str, output_dir: Path, svg_id: str) -> Optional[Path]:
        """
        Generate an SVG file from SVG code
        
        Args:
            svg_code: SVG source code
            output_dir: Output directory for the SVG file
            svg_id: Unique identifier for the SVG
            
        Returns:
            Path to the generated SVG file, or None if failed
        """
        try:
            # Ensure output directory exists
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate SVG filename
            svg_filename = f"svg_{svg_id}.svg"
            svg_path = output_dir / svg_filename
            
            # Write SVG content to file
            with open(svg_path, 'w', encoding='utf-8') as f:
                f.write(svg_code)
            
            print_debug(f"📄 Generated SVG file: {svg_path}")
            return svg_path
            
        except Exception as e:
            print_debug(f"❌ Failed to generate SVG file for {svg_id}: {e}")
            return None
    
    def convert_svg_to_png(self, svg_path: Path, png_path: Path) -> bool:
        """
        Convert SVG file to PNG using available conversion tools
        
        Args:
            svg_path: Path to the source SVG file
            png_path: Path for the output PNG file
            
        Returns:
            True if conversion successful, False otherwise
        """
        # Try Inkscape first (best quality)
        if self.inkscape_available:
            return self._convert_with_inkscape(svg_path, png_path)
        
        # Try rsvg-convert
        elif self.rsvg_convert_available:
            return self._convert_with_rsvg(svg_path, png_path)
        
        # Try CairoSVG (Python package)
        elif self.cairosvg_available:
            return self._convert_with_cairosvg(svg_path, png_path)
        
        else:
            print_debug("❌ No SVG conversion tools available")
            return False
    
    def _convert_with_inkscape(self, svg_path: Path, png_path: Path) -> bool:
        """Convert SVG to PNG using Inkscape"""
        try:
            cmd = [
                'inkscape',
                '--export-type=png',
                '--export-dpi=300',  # High DPI for better quality
                f'--export-filename={png_path}',
                str(svg_path)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0 and png_path.exists():
                print_debug(f"✅ Converted SVG to PNG using Inkscape: {png_path}")
                return True
            else:
                print_debug(f"❌ Inkscape conversion failed: {result.stderr}")
                return False
                
        except Exception as e:
            print_debug(f"❌ Inkscape conversion error: {e}")
            return False
    
    def _convert_with_rsvg(self, svg_path: Path, png_path: Path) -> bool:
        """Convert SVG to PNG using rsvg-convert"""
        try:
            cmd = [
                'rsvg-convert',
                '-f', 'png',
                '-d', '300',  # DPI
                '-p', '300',  # DPI
                '-o', str(png_path),
                str(svg_path)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0 and png_path.exists():
                print_debug(f"✅ Converted SVG to PNG using rsvg-convert: {png_path}")
                return True
            else:
                print_debug(f"❌ rsvg-convert conversion failed: {result.stderr}")
                return False
                
        except Exception as e:
            print_debug(f"❌ rsvg-convert conversion error: {e}")
            return False
    
    def _convert_with_cairosvg(self, svg_path: Path, png_path: Path) -> bool:
        """Convert SVG to PNG using CairoSVG Python package"""
        try:
            import cairosvg
            
            # Convert with high DPI for better quality
            cairosvg.svg2png(
                url=str(svg_path),
                write_to=str(png_path),
                dpi=300
            )
            
            if png_path.exists():
                print_debug(f"✅ Converted SVG to PNG using CairoSVG: {png_path}")
                return True
            else:
                print_debug(f"❌ CairoSVG conversion failed: PNG file not created")
                return False
                
        except Exception as e:
            print_debug(f"❌ CairoSVG conversion error: {e}")
            return False
    
    def process_svg_blocks(self, svg_blocks: List[Dict[str, Any]], markdown_dir: Path) -> List[Dict[str, Any]]:
        """
        Process a list of SVG blocks and generate PNG images
        
        Args:
            svg_blocks: List of SVG block dictionaries
            markdown_dir: Directory containing the markdown file
            
        Returns:
            List of processing results for each SVG block
        """
        results = []
        
        # Create images directory
        images_dir = markdown_dir / self.svg_output_dir
        images_dir.mkdir(parents=True, exist_ok=True)
        
        for block in svg_blocks:
            svg_id = block['id']
            svg_code = block['svg_code']
            
            print_debug(f"🎨 Processing SVG block: {svg_id}")
            
            # Generate SVG file
            svg_path = self.generate_svg_file(svg_code, images_dir, svg_id)
            if not svg_path:
                results.append({
                    'id': svg_id,
                    'status': 'failed',
                    'error': 'Failed to generate SVG file',
                    'block': block
                })
                continue
            
            # Generate PNG file
            png_filename = f"svg_{svg_id}.png"
            png_path = images_dir / png_filename
            
            conversion_success = self.convert_svg_to_png(svg_path, png_path)
            
            if conversion_success:
                # Calculate relative path for markdown
                relative_png_path = f"{self.svg_output_dir}/{png_filename}"
                
                results.append({
                    'id': svg_id,
                    'status': 'success',
                    'svg_file': str(svg_path.relative_to(markdown_dir)),
                    'png_file': relative_png_path,
                    'png_size': png_path.stat().st_size,
                    'block': block
                })
                
                print_debug(f"✅ Successfully processed SVG block {svg_id}")
            else:
                results.append({
                    'id': svg_id,
                    'status': 'failed',
                    'error': 'Failed to convert SVG to PNG',
                    'svg_file': str(svg_path.relative_to(markdown_dir)) if svg_path else None,
                    'block': block
                })
        
        return results
    
    def update_markdown_content(self, content: str, processing_results: List[Dict[str, Any]]) -> str:
        """
        Update markdown content by replacing SVG code blocks with image links
        
        Args:
            content: Original markdown content
            processing_results: Results from processing SVG blocks
            
        Returns:
            Updated markdown content
        """
        updated_content = content
        
        # Check if any blocks were corrected and use the corrected content as base
        corrected_blocks = [r for r in processing_results if r['status'] == 'success' and r['block'].get('is_corrected', False)]
        if corrected_blocks:
            # Use the corrected content from the first corrected block as our base
            updated_content = corrected_blocks[0]['block']['corrected_content']
            print_debug("📝 Using error-corrected content as base for updates")
        
        # Sort results by start position in reverse order to avoid position shifts
        successful_results = [r for r in processing_results if r['status'] == 'success']
        successful_results.sort(key=lambda x: x['block']['start_pos'], reverse=True)
        
        for result in successful_results:
            block = result['block']
            png_file = result['png_file']
            svg_id = result['id']
            
            # Create image markdown with alt text
            alt_text = f"SVG图表 {svg_id}"
            image_markdown = f"![{alt_text}]({png_file})"
            
            # Use image markdown directly without comment
            replacement = image_markdown
            
            # Replace the original SVG code block
            full_block = block['full_block']
            updated_content = updated_content.replace(full_block, replacement, 1)
            
            print_debug(f"🔄 Replaced SVG block {svg_id} with image link")
        
        return updated_content
    
    def process_markdown_file(self, markdown_file: str) -> Dict[str, Any]:
        """
        Process a markdown file and convert all SVG code blocks to PNG images
        
        Args:
            markdown_file: Path to the markdown file
            
        Returns:
            Dictionary containing processing results
        """
        try:
            markdown_path = Path(markdown_file)
            
            if not markdown_path.exists():
                return {
                    'status': 'failed',
                    'file': markdown_file,
                    'error': 'Markdown file not found'
                }
            
            # Read markdown content
            with open(markdown_path, 'r', encoding='utf-8') as f:
                original_content = f.read()
            
            # Extract SVG blocks
            svg_blocks = self.extract_svg_blocks(original_content)
            
            if not svg_blocks:
                return {
                    'status': 'success',
                    'file': markdown_file,
                    'message': 'No SVG code blocks found',
                    'svg_blocks_found': 0
                }
            
            # Process SVG blocks
            processing_results = self.process_svg_blocks(svg_blocks, markdown_path.parent)
            
            # Update markdown content
            updated_content = self.update_markdown_content(original_content, processing_results)
            
            # Check if any changes were made
            if updated_content != original_content:
                # Write updated content back to file
                with open(markdown_path, 'w', encoding='utf-8') as f:
                    f.write(updated_content)
                
                print_debug(f"📝 Updated markdown file: {markdown_file}")
            
            # Prepare summary
            successful_conversions = sum(1 for r in processing_results if r['status'] == 'success')
            failed_conversions = len(processing_results) - successful_conversions
            
            return {
                'status': 'success',
                'file': markdown_file,
                'svg_blocks_found': len(svg_blocks),
                'successful_conversions': successful_conversions,
                'failed_conversions': failed_conversions,
                'processing_results': processing_results,
                'message': f'Processed {successful_conversions}/{len(svg_blocks)} SVG blocks successfully'
            }
            
        except Exception as e:
            print_debug(f"❌ Error processing markdown file {markdown_file}: {e}")
            return {
                'status': 'failed',
                'file': markdown_file,
                'error': str(e)
            }
    
    def cleanup_generated_files(self, processing_results: List[Dict[str, Any]], markdown_dir: Path):
        """
        Clean up generated SVG and PNG files (useful for testing)
        
        Args:
            processing_results: Results from processing SVG blocks
            markdown_dir: Directory containing the markdown file
        """
        for result in processing_results:
            if result['status'] == 'success':
                # Remove SVG file
                if 'svg_file' in result:
                    svg_path = markdown_dir / result['svg_file']
                    if svg_path.exists():
                        svg_path.unlink()
                        print_debug(f"🗑️ Removed SVG file: {svg_path}")
                
                # Remove PNG file
                if 'png_file' in result:
                    png_path = markdown_dir / result['png_file']
                    if png_path.exists():
                        png_path.unlink()
                        print_debug(f"🗑️ Removed PNG file: {png_path}")


# Create a global instance for easy access
svg_processor = SVGProcessor()
