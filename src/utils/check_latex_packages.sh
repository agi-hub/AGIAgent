#!/bin/bash
# LaTeX 包检查脚本

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  检查 LaTeX 包安装状态"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# 必需的基础包（中文支持）
echo "📦 基础包（中文支持）："
base_packages=("xeCJK.sty" "ctex.sty" "fontspec.sty")
base_ok=true

for pkg in "${base_packages[@]}"; do
    if kpsewhich "$pkg" > /dev/null 2>&1; then
        echo "  ✅ ${pkg%.sty}"
    else
        echo "  ❌ ${pkg%.sty} - 未安装"
        base_ok=false
    fi
done

echo ""

# 模板所需的额外包
echo "📄 模板所需的额外包："
template_packages=("datetime2.sty" "fvextra.sty" "adjustbox.sty" "lastpage.sty" "fancyhdr.sty" "framed.sty" "seqsplit.sty" "xurl.sty")
template_ok=true

for pkg in "${template_packages[@]}"; do
    if kpsewhich "$pkg" > /dev/null 2>&1; then
        echo "  ✅ ${pkg%.sty}"
    else
        echo "  ❌ ${pkg%.sty} - 未安装"
        template_ok=false
    fi
done

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# 总结
if $base_ok && $template_ok; then
    echo "🎉 所有包已安装！可以使用完整的模板功能"
    echo ""
    echo "使用方法："
    echo "  python src/utils/trans_md_to_pdf.py input.md output.pdf"
    exit 0
elif $base_ok; then
    echo "⚠️  基础包已安装，但缺少模板包"
    echo ""
    echo "选项 1 - 安装模板包（享受完整功能）："
    echo "  sudo tlmgr install datetime2 tracklang fvextra adjustbox lastpage framed seqsplit xurl"
    echo ""
    echo "选项 2 - 不使用模板（推荐，依赖更少）："
    echo "  python src/utils/trans_md_to_pdf.py input.md output.pdf --no-template"
    exit 1
else
    echo "❌ 缺少必需的基础包"
    echo ""
    echo "请先安装基础包："
    echo "  sudo tlmgr install xecjk ctex fontspec"
    echo ""
    echo "然后选择："
    echo "  1. 安装模板包: sudo tlmgr install datetime2 tracklang fvextra adjustbox lastpage framed seqsplit xurl"
    echo "  2. 或使用 --no-template 选项"
    exit 1
fi

