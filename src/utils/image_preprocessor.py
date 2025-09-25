#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import sys
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional


def preprocess_images_for_pdf(markdown_content: str, markdown_dir: Path) -> Tuple[str, List[str]]:
    """
    预处理markdown中的图像，将不兼容格式转换为PDF兼容格式
    
    Args:
        markdown_content: markdown内容
        markdown_dir: markdown文件所在目录
        
    Returns:
        Tuple[str, List[str]]: (处理后的markdown内容, 临时文件列表)
    """
    try:
        from PIL import Image
        PIL_AVAILABLE = True
    except ImportError:
        print("⚠️ PIL/Pillow not available, skipping image preprocessing")
        return markdown_content, []
    
    # 查找markdown中的图像引用
    image_pattern = r'!\[([^\]]*)\]\(([^)]+)\)'
    temp_files = []
    processed_content = markdown_content
    
    # 查找所有图像引用
    matches = re.finditer(image_pattern, markdown_content)
    
    for match in matches:
        alt_text = match.group(1)
        image_path = match.group(2)
        
        # 跳过网络图像
        if image_path.startswith(('http://', 'https://', 'ftp://')):
            continue
            
        # 构建完整的图像路径
        if os.path.isabs(image_path):
            full_image_path = Path(image_path)
        else:
            full_image_path = markdown_dir / image_path
        
        # 检查文件是否存在
        if not full_image_path.exists():
            print(f"⚠️ Image file not found: {full_image_path}")
            continue
            
        # 检查是否需要转换
        if needs_conversion(full_image_path):
            try:
                # 特殊处理SVG文件：先进行HSL颜色转换
                processed_image_path = full_image_path
                if full_image_path.suffix.lower() == '.svg':
                    try:
                        # 导入HSL颜色转换器
                        from src.utils.hsl_color_converter import convert_svg_file_hsl_colors

                        # 创建临时文件来保存HSL转换后的SVG
                        import tempfile
                        with tempfile.NamedTemporaryFile(suffix='.svg', delete=False) as temp_svg:
                            temp_svg_path = temp_svg.name

                        # 转换HSL颜色
                        success = convert_svg_file_hsl_colors(str(full_image_path), temp_svg_path)
                        if success:
                            processed_image_path = Path(temp_svg_path)
                            temp_files.append(temp_svg_path)
                            print(f"🎨 Converted HSL colors in SVG: {full_image_path.name}")
                        else:
                            print(f"⚠️ HSL color conversion failed for: {full_image_path.name}")

                    except Exception as hsl_error:
                        print(f"⚠️ HSL color conversion failed: {hsl_error}")
                        # 继续使用原始文件

                # 进行图像格式转换
                converted_path = convert_image_for_pdf(processed_image_path, markdown_dir)
                if converted_path:
                    temp_files.append(str(converted_path))

                    # 计算相对路径
                    try:
                        rel_path = converted_path.relative_to(markdown_dir)
                    except ValueError:
                        # 如果无法计算相对路径，使用绝对路径
                        rel_path = converted_path

                    # 替换markdown中的图像路径
                    old_ref = f'![{alt_text}]({image_path})'
                    new_ref = f'![{alt_text}]({rel_path})'
                    processed_content = processed_content.replace(old_ref, new_ref)

                    print(f"✅ Converted image: {image_path} -> {rel_path}")

            except Exception as e:
                print(f"❌ Failed to convert image {image_path}: {e}")
                continue
    
    return processed_content, temp_files


def needs_conversion(image_path: Path) -> bool:
    """
    检查图像是否需要转换为PDF兼容格式
    
    Args:
        image_path: 图像文件路径
        
    Returns:
        bool: 是否需要转换
    """
    # PDF兼容的图像格式
    pdf_compatible_formats = {'.jpg', '.jpeg', '.png', '.pdf', '.eps'}
    
    # 不兼容的格式需要转换
    incompatible_formats = {'.webp', '.bmp', '.tiff', '.tif', '.gif', '.svg'}
    
    file_ext = image_path.suffix.lower()
    
    # 明确需要转换的格式
    if file_ext in incompatible_formats:
        return True
        
    # 已经兼容的格式
    if file_ext in pdf_compatible_formats:
        return False
        
    # 未知格式，尝试检查文件头
    try:
        from PIL import Image
        with Image.open(image_path) as img:
            # 如果PIL可以打开但格式不在兼容列表中，转换它
            return img.format.lower() not in ['jpeg', 'png', 'pdf']
    except Exception:
        # 如果PIL无法打开，可能需要转换
        return True


def convert_image_for_pdf(image_path: Path, output_dir: Path) -> Optional[Path]:
    """
    将图像转换为PDF兼容格式

    Args:
        image_path: 源图像路径
        output_dir: 输出目录

    Returns:
        Optional[Path]: 转换后的图像路径，失败返回None
    """
    try:
        # 特殊处理SVG文件
        if image_path.suffix.lower() == '.svg':
            return convert_svg_for_pdf(image_path, output_dir)

        from PIL import Image

        # 生成输出文件名
        base_name = image_path.stem
        output_path = output_dir / f"{base_name}_converted.png"

        # 避免文件名冲突
        counter = 1
        while output_path.exists():
            output_path = output_dir / f"{base_name}_converted_{counter}.png"
            counter += 1

        # 打开并转换图像
        with Image.open(image_path) as img:
            # 转换为RGB模式（处理透明度）
            if img.mode in ('RGBA', 'LA', 'P'):
                # 创建白色背景
                background = Image.new('RGB', img.size, (255, 255, 255))
                if img.mode == 'P':
                    img = img.convert('RGBA')
                if img.mode in ('RGBA', 'LA'):
                    background.paste(img, mask=img.split()[-1])  # 使用alpha通道作为mask
                else:
                    background.paste(img)
                img = background
            elif img.mode != 'RGB':
                img = img.convert('RGB')

            # 确保图像大小合理（避免过大的图像）
            max_size = (2048, 2048)
            if img.size[0] > max_size[0] or img.size[1] > max_size[1]:
                img.thumbnail(max_size, Image.Resampling.LANCZOS)
                print(f"📏 Resized image to {img.size}")

            # 保存为PNG格式
            img.save(output_path, 'PNG', optimize=True)

        return output_path

    except Exception as e:
        print(f"❌ Image conversion failed for {image_path}: {e}")
        return None


def convert_svg_for_pdf(svg_path: Path, output_dir: Path) -> Optional[Path]:
    """
    专门处理SVG到PDF的转换

    Args:
        svg_path: SVG文件路径
        output_dir: 输出目录

    Returns:
        Optional[Path]: 转换后的PDF文件路径，失败返回None
    """
    try:
        import subprocess
        import xml.etree.ElementTree as ET

        # 生成输出文件名
        base_name = svg_path.stem
        output_path = output_dir / f"{base_name}_converted.pdf"

        # 避免文件名冲突
        counter = 1
        while output_path.exists():
            output_path = output_dir / f"{base_name}_converted_{counter}.pdf"
            counter += 1

        # 检查并修复SVG文件（添加viewBox如果缺失）
        fixed_svg_path = svg_path
        try:
            tree = ET.parse(svg_path)
            root = tree.getroot()

            # 检查是否已有viewBox
            if 'viewBox' not in root.attrib:
                # 如果没有viewBox，尝试从width和height添加
                width = root.get('width')
                height = root.get('height')

                if width and height:
                    # 提取数值部分（去除单位）
                    width_val = ''.join(filter(str.isdigit, width))
                    height_val = ''.join(filter(str.isdigit, height))

                    if width_val and height_val:
                        # 添加viewBox属性
                        root.set('viewBox', f'0 0 {width_val} {height_val}')

                        # 保存修复后的SVG
                        fixed_svg_path = output_dir / f"{base_name}_fixed.svg"
                        tree.write(fixed_svg_path, encoding='utf-8', xml_declaration=True)
                        print(f"📝 Fixed SVG viewBox: {fixed_svg_path}")

        except Exception as e:
            print(f"⚠️ Warning: Could not fix SVG viewBox: {e}")

        # 使用SVG中文过滤器处理SVG并转换为PDF
        try:
            # 导入SVG中文过滤器
            sys.path.append(str(svg_path.parent.parent.parent / "src" / "utils"))
            from svg_chinese_filter import process_svg_file

            # 使用SVG中文过滤器处理SVG
            # process_svg_file会生成PDF文件并返回PDF路径
            pdf_result = process_svg_file(str(fixed_svg_path))
            if pdf_result:
                # 移动生成的PDF到期望的位置
                import shutil
                shutil.move(pdf_result, output_path)
                print(f"✅ SVG converted to PDF with Chinese support: {output_path}")
                return output_path
            else:
                print("❌ SVG Chinese filter conversion failed")
                return None

        except ImportError:
            print("⚠️ SVG Chinese filter not available, falling back to direct cairosvg conversion")
            # 回退到直接使用cairosvg（不处理中文）
            result = subprocess.run([
                'cairosvg', str(fixed_svg_path), '-o', str(output_path)
            ], capture_output=True, text=True, timeout=30)

            if result.returncode == 0 and output_path.exists():
                print(f"✅ SVG converted to PDF (fallback): {output_path}")
                return output_path
            else:
                print(f"❌ cairosvg conversion failed: {result.stderr}")
                return None

        except (subprocess.TimeoutExpired, FileNotFoundError):
            print("⚠️ cairosvg not available, falling back to PIL conversion")

            # 回退到PIL转换（转换为PNG）
            try:
                from PIL import Image
                import io

                # 读取SVG内容
                with open(svg_path, 'r', encoding='utf-8') as f:
                    svg_content = f.read()

                # 简单检查SVG是否有效（这里可以改进）
                if '<svg' in svg_content and '</svg>' in svg_content:
                    # 由于PIL不支持SVG，这里返回None表示转换失败
                    # 或者可以考虑其他转换方法
                    print("⚠️ PIL cannot handle SVG files directly")
                    return None

            except Exception as e:
                print(f"❌ Fallback SVG conversion failed: {e}")
                return None

    except Exception as e:
        print(f"❌ SVG conversion failed for {svg_path}: {e}")
        return None


def create_preprocessed_markdown(input_file: Path, output_dir: Optional[Path] = None) -> Tuple[Optional[Path], List[str]]:
    """
    创建预处理后的markdown文件
    
    Args:
        input_file: 输入的markdown文件
        output_dir: 输出目录，如果为None则使用临时目录
        
    Returns:
        Tuple[Optional[Path], List[str]]: (预处理后的markdown文件路径, 临时文件列表)
    """
    if output_dir is None:
        output_dir = Path(tempfile.mkdtemp(prefix='markdown_preprocessed_'))
    
    temp_files = []
    
    try:
        # 读取markdown内容
        with open(input_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 预处理图像
        processed_content, image_temp_files = preprocess_images_for_pdf(content, input_file.parent)
        temp_files.extend(image_temp_files)
        
        # 检查是否有变化
        if processed_content == content:
            print("📝 No image preprocessing needed")
            return input_file, temp_files
        
        # 创建预处理后的markdown文件
        output_file = output_dir / f"{input_file.stem}_preprocessed.md"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(processed_content)
        
        temp_files.append(str(output_file))
        print(f"📝 Created preprocessed markdown: {output_file}")
        
        return output_file, temp_files
        
    except Exception as e:
        print(f"❌ Markdown preprocessing failed: {e}")
        # 清理已创建的临时文件
        cleanup_temp_files(temp_files)
        return None, []


def cleanup_temp_files(temp_files: List[str]):
    """
    清理临时文件
    
    Args:
        temp_files: 临时文件路径列表
    """
    for temp_file in temp_files:
        try:
            if os.path.exists(temp_file):
                if os.path.isfile(temp_file):
                    os.remove(temp_file)
                elif os.path.isdir(temp_file):
                    shutil.rmtree(temp_file)
        except Exception as e:
            print(f"⚠️ Failed to cleanup temp file {temp_file}: {e}")
