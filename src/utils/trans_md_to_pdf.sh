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

# 检测可用的PDF引擎
check_pdf_engine() {
    local engine_name="$1"
    local engine_option="$2"
    
    if command -v "$engine_name" >/dev/null 2>&1; then
        return 0
    else
        return 1
    fi
}

# 按优先级检查PDF引擎
select_pdf_engine() {
    local engines=("xelatex" "lualatex" "pdflatex" "wkhtmltopdf" "weasyprint")
    local engine_options=("--pdf-engine=xelatex" "--pdf-engine=lualatex" "--pdf-engine=pdflatex" "--pdf-engine=wkhtmltopdf" "--pdf-engine=weasyprint")
    
    for i in "${!engines[@]}"; do
        if check_pdf_engine "${engines[$i]}" "${engine_options[$i]}"; then
            SELECTED_ENGINE="${engines[$i]}"
            SELECTED_OPTION="${engine_options[$i]}"
            return 0
        fi
    done
    
    echo "❌ 没有可用的PDF引擎。请安装以下之一: xelatex, lualatex, pdflatex, wkhtmltopdf, 或 weasyprint"
    return 1
}

# 获取引擎特定的选项
get_engine_options() {
    local engine_name="$1"
    
    case "$engine_name" in
        "xelatex"|"lualatex")
            # XeLaTeX 和 LuaLaTeX 支持CJK字体
            echo "-V CJKmainfont='Noto Serif CJK SC' -V CJKsansfont='Noto Sans CJK SC' -V CJKmonofont='Noto Sans Mono CJK SC' -V mainfont='DejaVu Serif' -V sansfont='DejaVu Sans' -V monofont='DejaVu Sans Mono'"
            ;;
        "pdflatex")
            # pdfLaTeX 不支持CJK字体，使用基本字体
            echo "-V mainfont='DejaVu Serif' -V sansfont='DejaVu Sans' -V monofont='DejaVu Sans Mono'"
            ;;
        *)
            # wkhtmltopdf 和 weasyprint 不使用LaTeX，返回最小选项
            echo ""
            ;;
    esac
}

# 选择PDF引擎
if ! select_pdf_engine; then
    exit 1
fi

# 获取引擎特定选项
ENGINE_OPTIONS=$(get_engine_options "$SELECTED_ENGINE")

echo "正在转换: $INPUT_FILE -> $OUTPUT_FILE"
echo "使用PDF引擎: $SELECTED_ENGINE"

# 构建pandoc命令
PANDOC_CMD="pandoc \"$INPUT_FILE\" -o \"$OUTPUT_FILE\" $SELECTED_OPTION"

# 添加引擎特定选项
if [ -n "$ENGINE_OPTIONS" ]; then
    PANDOC_CMD="$PANDOC_CMD $ENGINE_OPTIONS"
fi

# 添加过滤器选项（仅对LaTeX引擎）
if [[ "$SELECTED_ENGINE" =~ ^(xelatex|lualatex|pdflatex)$ ]] && [ -n "$FILTER_OPTION" ]; then
    PANDOC_CMD="$PANDOC_CMD $FILTER_OPTION"
fi

# 添加模板选项（仅对LaTeX引擎）
if [[ "$SELECTED_ENGINE" =~ ^(xelatex|lualatex|pdflatex)$ ]] && [ -n "$TEMPLATE_OPTION" ]; then
    PANDOC_CMD="$PANDOC_CMD $TEMPLATE_OPTION"
fi

# 添加通用选项
PANDOC_CMD="$PANDOC_CMD -V fontsize=12pt -V geometry:margin=2.5cm -V geometry:a4paper -V linestretch=1.5 --highlight-style=tango -V colorlinks=true -V linkcolor=blue -V urlcolor=blue --toc --wrap=preserve"

# 添加LaTeX特定选项（仅对LaTeX引擎）
if [[ "$SELECTED_ENGINE" =~ ^(xelatex|lualatex|pdflatex)$ ]]; then
    PANDOC_CMD="$PANDOC_CMD -V graphics=true"
fi

echo "执行命令: $PANDOC_CMD"

# 执行pandoc转换
eval $PANDOC_CMD

# 检查转换结果
if [ $? -eq 0 ] && [ -f "$OUTPUT_FILE" ]; then
    echo "✓ 转换成功: $OUTPUT_FILE"
    ls -lh "$OUTPUT_FILE"
else
    echo "✗ 转换失败"
    exit 1
fi