#!/bin/bash

# 检查参数
if [ $# -lt 2 ]; then
    echo "用法: $0 <输入.md> <输出.pdf>"
    echo "示例: $0 document.md output.pdf"
    exit 1
fi

INPUT_FILE="$1"
OUTPUT_FILE="$2"

# 检查输入文件是否存在
if [ ! -f "$INPUT_FILE" ]; then
    echo "错误: 输入文件 '$INPUT_FILE' 不存在"
    exit 1
fi

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 设置过滤器路径
FILTER_PATH="$SCRIPT_DIR/svg_chinese_filter.py"

# 设置模板路径
TEMPLATE_PATH="$SCRIPT_DIR/template.latex"

# 检查过滤器是否存在
if [ ! -f "$FILTER_PATH" ]; then
    echo "警告: SVG中文过滤器 '$FILTER_PATH' 不存在，使用默认转换"
    FILTER_OPTION=""
else
    FILTER_OPTION="--filter=$FILTER_PATH"
    echo "使用SVG中文过滤器: $FILTER_PATH"
fi

# 检查模板是否存在
if [ ! -f "$TEMPLATE_PATH" ]; then
    echo "警告: 自定义模板 '$TEMPLATE_PATH' 不存在，使用默认模板"
    TEMPLATE_OPTION=""
else
    TEMPLATE_OPTION="--template=$TEMPLATE_PATH"
    echo "使用自定义模板: $TEMPLATE_PATH"
fi

echo "正在转换: $INPUT_FILE -> $OUTPUT_FILE"

# 执行pandoc转换
pandoc "$INPUT_FILE" \
  -o "$OUTPUT_FILE" \
  --pdf-engine=xelatex \
  $FILTER_OPTION \
  $TEMPLATE_OPTION \
  -V CJKmainfont='Noto Serif CJK SC' \
  -V CJKsansfont='Noto Sans CJK SC' \
  -V CJKmonofont='Noto Sans Mono CJK SC' \
  -V mainfont='DejaVu Serif' \
  -V sansfont='DejaVu Sans' \
  -V monofont='DejaVu Sans Mono' \
  -V fontsize=12pt \
  -V geometry:margin=2.5cm \
  -V geometry:a4paper \
  -V linestretch=1.5 \
  --highlight-style=tango \
  -V colorlinks=true \
  -V linkcolor=blue \
  -V urlcolor=blue \
  --toc \
  --wrap=preserve

# 检查转换结果
if [ $? -eq 0 ] && [ -f "$OUTPUT_FILE" ]; then
    echo "✓ 转换成功: $OUTPUT_FILE"
    ls -lh "$OUTPUT_FILE"
else
    echo "✗ 转换失败"
    exit 1
fi