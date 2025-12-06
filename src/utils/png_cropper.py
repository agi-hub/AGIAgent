#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PNG图片裁剪工具
用于去除PNG图片中的空白区域，特别适用于从SVG转换而来的PNG图片
"""

import os
from pathlib import Path
from typing import Tuple, Optional
from PIL import Image, ImageOps

# ========================================
# 🚀 延迟导入优化：numpy 延迟加载
# ========================================
# numpy 是重量级库，只在实际使用图片裁剪功能时才加载
# 避免启动时加载，节省约 0.7秒

np = None

def _ensure_numpy():
    """确保 numpy 已加载（延迟加载）"""
    global np
    if np is None:
        import numpy as _np
        np = _np

class PNGCropper:
    """PNG图片裁剪工具，用于自动去除空白区域"""
    
    def __init__(self):
        self.background_color = (255, 255, 255, 0)  # 透明白色背景
        self.tolerance = 10  # 颜色容差
        
    def detect_content_bounds(self, image: Image.Image, padding: int = 10) -> Tuple[int, int, int, int]:
        """
        检测图片中内容的边界
        
        Args:
            image: PIL Image对象
            padding: 保留的边距像素数
            
        Returns:
            (left, top, right, bottom) 内容边界坐标
        """
        # 🚀 延迟加载：只在实际使用时才加载 numpy
        _ensure_numpy()
        
        # 转换为RGBA模式以处理透明度
        if image.mode != 'RGBA':
            image = image.convert('RGBA')
        
        # 转换为numpy数组进行处理
        img_array = np.array(image)
        
        # 检测非透明且非白色的像素
        # 白色像素：RGB接近(255,255,255)
        # 透明像素：alpha < 128
        height, width = img_array.shape[:2]
        
        # 创建掩码：非空白内容的像素
        if img_array.shape[2] == 4:  # RGBA
            # 非透明像素
            alpha_mask = img_array[:, :, 3] > 32  # alpha > 32
            # 非白色像素（考虑容差）
            rgb_sum = img_array[:, :, :3].sum(axis=2)
            white_mask = rgb_sum < (255 * 3 - self.tolerance * 3)
            # 内容掩码：非透明且非白色
            content_mask = alpha_mask & white_mask
        else:  # RGB
            # 非白色像素
            rgb_sum = img_array.sum(axis=2)
            content_mask = rgb_sum < (255 * 3 - self.tolerance * 3)
        
        # 如果没有找到内容，返回原图边界
        if not content_mask.any():
            return 0, 0, width, height
        
        # 找到内容的边界
        content_rows = np.any(content_mask, axis=1)
        content_cols = np.any(content_mask, axis=0)
        
        top = np.argmax(content_rows)
        bottom = height - np.argmax(content_rows[::-1])
        left = np.argmax(content_cols)
        right = width - np.argmax(content_cols[::-1])
        
        # 添加边距
        left = max(0, left - padding)
        top = max(0, top - padding)
        right = min(width, right + padding)
        bottom = min(height, bottom + padding)
        
        return left, top, right, bottom
    
    def crop_png(self, input_path: Path, output_path: Optional[Path] = None, 
                 padding: int = 10, min_size: Tuple[int, int] = (100, 100), verbose: bool = True) -> bool:
        """
        裁剪PNG图片，去除空白区域
        
        Args:
            input_path: 输入PNG文件路径
            output_path: 输出PNG文件路径，如果为None则覆盖原文件
            padding: 保留的边距像素数
            min_size: 最小输出尺寸(width, height)
            verbose: 是否输出详细信息
            
        Returns:
            是否成功裁剪
        """
        try:
            if not input_path.exists():
                if verbose:
                    print(f"❌ 输入文件不存在: {input_path}")
                return False
            
            # 读取图片
            with Image.open(input_path) as image:
                original_size = image.size
                if verbose:
                    print(f"📏 原始图片尺寸: {original_size[0]}x{original_size[1]}")
                
                # 检测内容边界
                left, top, right, bottom = self.detect_content_bounds(image, padding)
                
                # 计算裁剪后的尺寸
                crop_width = right - left
                crop_height = bottom - top
                
                if verbose:
                    print(f"🔍 检测到内容区域: ({left}, {top}) -> ({right}, {bottom})")
                    print(f"📐 裁剪后尺寸: {crop_width}x{crop_height}")
                
                # 检查是否需要裁剪
                if (left <= 5 and top <= 5 and 
                    right >= original_size[0] - 5 and bottom >= original_size[1] - 5):
                    if verbose:
                        print("ℹ️ 图片已经没有明显的空白区域，无需裁剪")
                    if output_path and output_path != input_path:
                        image.save(output_path, 'PNG', optimize=True)
                    return True
                
                # 确保最小尺寸
                if crop_width < min_size[0] or crop_height < min_size[1]:
                    # 计算需要扩展的区域
                    expand_width = max(0, min_size[0] - crop_width) // 2
                    expand_height = max(0, min_size[1] - crop_height) // 2
                    
                    left = max(0, left - expand_width)
                    right = min(original_size[0], right + expand_width)
                    top = max(0, top - expand_height)
                    bottom = min(original_size[1], bottom + expand_height)
                    
                    crop_width = right - left
                    crop_height = bottom - top
                    
                    if verbose:
                        print(f"🔧 调整到最小尺寸: {crop_width}x{crop_height}")
                
                # 执行裁剪
                cropped_image = image.crop((left, top, right, bottom))
                
                # 保存裁剪后的图片
                if output_path is None:
                    output_path = input_path
                
                cropped_image.save(output_path, 'PNG', optimize=True)
                
                # 计算压缩比例
                original_pixels = original_size[0] * original_size[1]
                cropped_pixels = crop_width * crop_height
                compression_ratio = (1 - cropped_pixels / original_pixels) * 100
                
                if verbose:
                    print(f"✅ 裁剪完成: {output_path}")
                    print(f"📊 空白区域减少: {compression_ratio:.1f}%")
                
                return True
                
        except Exception as e:
            if verbose:
                print(f"❌ 裁剪失败: {e}")
            return False
    
    def batch_crop(self, directory: Path, pattern: str = "*.png", 
                   padding: int = 10, backup: bool = True) -> int:
        """
        批量裁剪目录中的PNG文件
        
        Args:
            directory: 目录路径
            pattern: 文件匹配模式
            padding: 保留的边距像素数
            backup: 是否备份原文件
            
        Returns:
            成功处理的文件数量
        """
        if not directory.exists():
            print(f"❌ 目录不存在: {directory}")
            return 0
        
        png_files = list(directory.glob(pattern))
        if not png_files:
            print(f"📁 在 {directory} 中未找到匹配的PNG文件")
            return 0
        
        success_count = 0
        print(f"🔄 开始批量处理 {len(png_files)} 个PNG文件...")
        
        for png_file in png_files:
            print(f"\n📝 处理: {png_file.name}")
            
            # 创建备份
            if backup:
                backup_path = png_file.with_suffix('.png.backup')
                if not backup_path.exists():
                    try:
                        import shutil
                        shutil.copy2(png_file, backup_path)
                        print(f"💾 已备份到: {backup_path.name}")
                    except Exception as e:
                        print(f"⚠️ 备份失败: {e}")
            
            # 执行裁剪
            if self.crop_png(png_file, padding=padding):
                success_count += 1
        
        print(f"\n🎉 批量处理完成: {success_count}/{len(png_files)} 个文件成功处理")
        return success_count


def main():
    """命令行工具入口"""
    import argparse
    
    parser = argparse.ArgumentParser(description="PNG图片裁剪工具 - 自动去除空白区域")
    parser.add_argument("input", help="输入PNG文件或目录路径")
    parser.add_argument("-o", "--output", help="输出文件路径（仅用于单文件处理）")
    parser.add_argument("-p", "--padding", type=int, default=10, help="保留的边距像素数 (默认: 10)")
    parser.add_argument("-b", "--no-backup", action="store_true", help="不创建备份文件")
    parser.add_argument("--batch", action="store_true", help="批量处理目录中的所有PNG文件")
    
    args = parser.parse_args()
    
    cropper = PNGCropper()
    input_path = Path(args.input)
    
    if args.batch or input_path.is_dir():
        # 批量处理
        success_count = cropper.batch_crop(
            input_path, 
            padding=args.padding,
            backup=not args.no_backup
        )
        print(f"\n✨ 总计处理了 {success_count} 个文件")
    else:
        # 单文件处理
        output_path = Path(args.output) if args.output else None
        
        # 创建备份
        if not args.no_backup and output_path is None:
            backup_path = input_path.with_suffix('.png.backup')
            if not backup_path.exists():
                try:
                    import shutil
                    shutil.copy2(input_path, backup_path)
                    print(f"💾 已备份到: {backup_path}")
                except Exception as e:
                    print(f"⚠️ 备份失败: {e}")
        
        success = cropper.crop_png(input_path, output_path, args.padding)
        if success:
            print("🎉 处理完成!")
        else:
            print("❌ 处理失败!")


if __name__ == "__main__":
    main()
