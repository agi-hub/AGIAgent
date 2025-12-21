# LaTeX 包完整安装指南

## 概述

AGIAgent 的 PDF 转换功能需要多个 LaTeX 包。本文档提供一次性安装所有必需包的完整方案。

## 快速安装（推荐）

### 方法 1：使用自动安装脚本

```bash
cd /Users/zhenzhiwu/AGIAgent
bash md/install_all_latex_packages.sh
```

脚本会自动：
1. 检查 tlmgr 是否可用
2. 更新 tlmgr
3. 安装所有必需的包
4. 验证安装状态

### 方法 2：手动安装

如果脚本无法运行，可以手动执行：

```bash
export PATH="/Library/TeX/texbin:$PATH"

# 更新 tlmgr
sudo tlmgr update --self

# 安装所有包（一次性）
sudo tlmgr install \
    xeCJK \
    geometry \
    fancyhdr \
    datetime2 \
    lastpage \
    fancyvrb \
    fvextra \
    framed \
    listings \
    seqsplit \
    longtable \
    booktabs \
    xurl \
    adjustbox \
    float

# 刷新数据库
sudo mktexlsr
```

## 模板使用的所有 LaTeX 包

根据 `src/utils/template.latex`，以下是所有使用的包：

### 核心包（必需）

| 包名 | 用途 | 是否必需 |
|------|------|---------|
| `xeCJK` | 中文支持 | ✅ 必需 |
| `geometry` | 页面布局 | ✅ 必需 |
| `fancyhdr` | 页眉页脚 | ✅ 必需 |
| `datetime2` | 日期时间格式化 | ✅ 必需 |
| `lastpage` | 获取总页数 | ✅ 必需 |
| `fvextra` | fancyvrb 扩展 | ✅ 必需 |

### 代码高亮相关

| 包名 | 用途 |
|------|------|
| `fancyvrb` | 代码块显示 |
| `framed` | 代码框架 |
| `listings` | 代码列表 |

### 表格和布局

| 包名 | 用途 |
|------|------|
| `longtable` | 长表格 |
| `booktabs` | 表格线 |
| `seqsplit` | 长行分割 |
| `adjustbox` | 图片调整 |
| `float` | 浮动体 |

### 其他工具

| 包名 | 用途 |
|------|------|
| `xurl` | URL 换行 |
| `graphicx` | 图片（通常已包含） |
| `hyperref` | 超链接（通常已包含） |
| `xcolor` | 颜色（通常已包含） |
| `array` | 数组（通常已包含） |
| `calc` | 计算（通常已包含） |

## 验证安装

安装完成后，验证关键包：

```bash
export PATH="/Library/TeX/texbin:$PATH"

# 检查关键包
kpsewhich datetime2.sty
kpsewhich lastpage.sty
kpsewhich fvextra.sty
kpsewhich xeCJK.sty
kpsewhich fancyhdr.sty
kpsewhich fancyvrb.sty
```

如果所有命令都返回文件路径，说明安装成功。

## 常见问题

### 问题 1：tlmgr 命令不存在

**症状**：`command not found: tlmgr`

**解决方案**：
1. 检查 TeX 是否安装：
   ```bash
   ls -la /Library/TeX/texbin/tlmgr
   ```

2. 如果不存在，安装 MacTeX：
   - 下载：https://www.tug.org/mactex/
   - 安装后，tlmgr 会自动可用

3. 添加路径到环境变量：
   ```bash
   echo 'export PATH="/Library/TeX/texbin:$PATH"' >> ~/.zshrc
   source ~/.zshrc
   ```

### 问题 2：权限不足

**症状**：`You don't have permission to change the installation`

**解决方案**：使用 `sudo` 执行安装命令

### 问题 3：某些包安装失败

**可能原因**：
- 网络连接问题
- 包名错误
- TeX Live 版本不兼容

**解决方案**：
1. 检查网络连接
2. 更新 tlmgr：`sudo tlmgr update --self`
3. 手动安装失败的包：`sudo tlmgr install <包名>`

### 问题 4：安装后仍找不到包

**解决方案**：
```bash
# 刷新 LaTeX 数据库
sudo mktexlsr

# 清除缓存（如果需要）
sudo tlmgr path add
```

## 最小安装方案

如果完整安装遇到问题，可以只安装最关键的包：

```bash
export PATH="/Library/TeX/texbin:$PATH"

sudo tlmgr install \
    datetime2 \
    lastpage \
    fvextra \
    xeCJK \
    fancyhdr \
    fancyvrb
```

## 安装后测试

创建一个简单的测试文件 `test.tex`：

```latex
\documentclass{article}
\usepackage{xeCJK}
\usepackage{datetime2}
\usepackage{lastpage}
\usepackage{fvextra}
\begin{document}
测试文档 - \DTMnow
\end{document}
```

编译测试：
```bash
xelatex test.tex
```

如果编译成功，说明安装正确。

## 注意事项

1. **需要管理员权限**：所有安装命令都需要 `sudo`
2. **网络连接**：tlmgr 需要从网络下载包
3. **安装时间**：完整安装可能需要 5-10 分钟
4. **磁盘空间**：确保有足够的磁盘空间（约 500MB-1GB）

## 相关文件

- 模板文件：`src/utils/template.latex`
- 安装脚本：`md/install_all_latex_packages.sh`
- PDF 转换脚本：`src/utils/trans_md_to_pdf.py`

## 支持

如果遇到问题，请检查：
1. TeX Live 版本：`tlmgr --version`
2. 已安装的包：`tlmgr list --only-installed`
3. 日志文件：查看转换日志中的错误信息

