# task_reflection.py 详细文档

## 1. 概述

`task_reflection.py` 是 AGIAgent 技能进化系统的核心组件之一，负责**任务反思与技能生成**。该脚本通过分析任务执行日志，利用大语言模型（LLM）进行深度反思，自动生成可复用的 skill 文件，实现 Agent 的经验积累和能力进化。

### 1.1 核心功能
- 自动扫描和发现任务输出目录
- 解析任务执行日志（manager.out、agent_*.out）
- 调用 LLM 进行深度反思分析
- 自动生成结构化的 skill 文件
- 备份相关代码文件到技能库

### 1.2 设计理念
该脚本实现了 AGIAgent 的"学习-反思-积累"循环：
```
任务执行 → 日志记录 → 日志分析 → LLM反思 → Skill生成 → 经验复用
```

---

## 2. 类结构

### 2.1 TaskReflection 类

主类，负责整个任务反思流程的协调和执行。

```python
class TaskReflection:
    """任务反思处理器"""
```

#### 2.1.1 初始化参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| root_dir | Optional[str] | None | 根目录，覆盖配置文件设置 |
| config_file | str | "config/config.txt" | 配置文件路径 |

#### 2.1.2 主要属性

| 属性 | 类型 | 说明 |
|------|------|------|
| config | Dict | 加载的配置信息 |
| root_dir | str | 数据根目录 |
| skill_tools | SkillTools | 技能工具实例 |
| logger | Logger | 日志记录器 |
| api_key | str | LLM API 密钥 |
| api_base | str | LLM API 基础 URL |
| model | str | 使用的模型名称 |
| llm_client | OpenAI/Anthropic | LLM 客户端实例 |
| is_claude | bool | 是否使用 Claude 模型 |

---

## 3. 核心方法详解

### 3.1 `__init__` - 初始化方法

```python
def __init__(self, root_dir: Optional[str] = None, config_file: str = "config/config.txt")
```

**功能**：初始化任务反思处理器

**执行流程**：
1. 加载配置文件
2. 确定根目录（优先级：参数 > 配置文件 > 默认 data 目录）
3. 初始化 SkillTools 工具
4. 设置日志记录器
5. 初始化 LLM 客户端（支持 OpenAI 和 Anthropic）

**LLM 客户端初始化逻辑**：
```python
# 判断模型类型
if 'claude' in model.lower() or 'anthropic' in api_base.lower():
    # 使用 Anthropic 客户端
    # 支持标准 Anthropic API 和兼容 API（如 minimax、GLM）
else:
    # 使用 OpenAI 兼容客户端
```

### 3.2 `_find_project_root` - 查找项目根目录

```python
def _find_project_root(self) -> Optional[str]
```

**功能**：从当前文件位置向上查找包含 `config` 目录的项目根目录

**返回**：项目根目录路径或 None

### 3.3 `_setup_logger` - 设置日志记录器

```python
def _setup_logger(self) -> logging.Logger
```

**功能**：配置日志系统

**日志配置**：
- 日志级别：INFO
- 日志文件：`{experience_dir}/logs/task_reflection_{YYYYMMDD}.log`
- 同时输出到控制台和文件
- 格式：`%(asctime)s - %(levelname)s - %(message)s`

### 3.4 `_find_all_output_dirs` - 查找所有输出目录

```python
def _find_all_output_dirs(self) -> List[Tuple[str, float]]
```

**功能**：扫描并返回所有 `output_XXX` 格式的任务输出目录

**查找范围（按优先级）**：
1. `data/output_XXX/` - 直接在 data 目录下（由 agia.py 生成）
2. `data/{user_dir}/output_XXX/` - 标准用户目录结构
3. `data/benchmark_results/*/baseline_outputs/output_XXX/` - 评测基线结构
4. `data/benchmark_results/*/skill_outputs/output_XXX/` - 评测技能结构

**返回**：`[(目录路径, 修改时间), ...]` 列表，按时间倒序排列

**验证条件**：目录必须包含 `workspace` 或 `logs` 子目录

### 3.5 `_parse_log_file` - 解析日志文件

```python
def _parse_log_file(self, log_file_path: str) -> Dict[str, Any]
```

**功能**：解析任务执行日志，提取关键信息

**返回结构**：
```python
{
    'user_requirements': [],      # 用户需求列表
    'tool_calls': [],             # 工具调用记录
    'errors': [],                 # 错误信息
    'task_completed': False,      # 任务是否完成
    'user_interruptions': [],     # 用户中断点
    'agent_messages': [],         # Agent 消息
    'log_content': ''             # 完整日志内容
}
```

**日志截取策略**（当日志超过 50000 字符时）：
1. 保留前 2000 行
2. 提取游戏信息相关行（前后各 50 行）
3. 提取失败点（前后各 150 行）
4. 提取成功点（前后各 200 行）
5. 按行号排序保持时间顺序

**提取的信息类型**：
| 信息类型 | 正则模式 | 说明 |
|----------|----------|------|
| 用户需求 | `Received user requirement[:\s]+(.+?)` | 提取用户输入的需求 |
| 工具调用 | `<invoke[^>]*>(.*?)</invoke>` | XML 格式的工具调用 |
| 错误反馈 | `ERROR FEEDBACK[:\s]+(.+?)` | 错误信息 |
| 任务完成 | `TASK_COMPLETED` | 任务完成标志 |

### 3.6 `_call_llm_reflection` - 调用 LLM 进行反思

```python
def _call_llm_reflection(self, task_info: Dict[str, Any]) -> Dict[str, Any]
```

**功能**：调用大语言模型对任务执行历史进行深度反思分析

**系统提示词要求**：
1. 必须使用中文输出
2. 输出简洁凝练，避免重复
3. 不包含思考过程或过渡语句
4. 直接按格式输出

**分析角度**：
1. 任务完成情况
2. 用户中断分析
3. 核心经验总结（游戏规则、失败原因、成功策略）
4. 最短成功路径
5. 用户偏好
6. Skill 使用条件
7. 需要备份的文件

**返回结构**：
```python
{
    'reflection': str,           # 反思内容
    'files_to_backup': List[str], # 需要备份的文件列表
    'usage_conditions': str       # Skill 使用条件
}
```

**LLM 调用参数**：
- max_tokens: 4000
- temperature: 0.7

### 3.7 `_backup_files` - 备份文件

```python
def _backup_files(self, output_dir: str, files_to_backup: List[str], skill_id: str) -> List[str]
```

**功能**：将相关代码文件备份到技能代码目录

**支持的文件类型**：
| 类型 | 扩展名 |
|------|--------|
| 代码文件 | .py, .js, .ts, .java, .cpp, .c, .h, .hpp, .go, .rs, .rb, .php |
| 文档文件 | .md, .txt, .rst |

**排除的文件类型**：配置文件（.json, .yaml）、图片文件

### 3.8 `_generate_skill` - 生成 Skill 文件

```python
def _generate_skill(self, task_info: Dict[str, Any], reflection_result: Dict[str, Any]) -> Optional[str]
```

**功能**：根据反思结果生成结构化的 skill 文件

**Skill 文件结构**：
```yaml
---
skill_id: "1234567890"           # 时间戳 ID
title: "任务标题"                 # 从反思内容提取
usage_conditions: "使用条件"      # LLM 生成或从需求提取
quality_index: 0.5               # 初始质量指数
fetch_count: 0                   # 获取次数
related_code: ""                 # 相关代码文件
task_directories: ["output_xxx"] # 关联的任务目录
created_at: "2025-01-15T..."     # 创建时间
updated_at: "2025-01-15T..."     # 更新时间
last_used_at: null               # 最后使用时间
user_preferences: ""             # 用户偏好
---

[反思内容]
```

**文件命名规则**：
- 格式：`skill_{safe_title}.md`
- 如果文件已存在，添加时间戳：`skill_{safe_title}_{timestamp}.md`

### 3.9 `process_task` - 处理单个任务

```python
def process_task(self, output_dir: str) -> bool
```

**功能**：处理单个任务目录，完成从日志解析到 skill 生成的完整流程

**处理流程**：
```
1. 检查 logs 目录是否存在
       ↓
2. 解析 manager.out 日志
       ↓
3. 解析所有 agent_*.out 日志
       ↓
4. 生成日志摘要
       ↓
5. 调用 LLM 进行反思
       ↓
6. 生成 skill 文件
       ↓
7. 返回处理结果
```

### 3.10 `run` - 运行主流程

```python
def run(self)
```

**功能**：运行完整的任务反思流程

**执行步骤**：
1. 查找所有 output 目录
2. 按时间顺序处理每个任务
3. 统计并输出处理结果

---

## 4. 命令行接口

### 4.1 使用方法

```bash
python task_reflection.py [--root-dir ROOT_DIR] [--config CONFIG]
```

### 4.2 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| --root-dir | str | None | 数据根目录（覆盖配置） |
| --config | str | config/config.txt | 配置文件路径 |

### 4.3 使用示例

```bash
# 使用默认配置
python task_reflection.py

# 指定数据目录
python task_reflection.py --root-dir /path/to/data

# 指定配置文件
python task_reflection.py --config /path/to/config.txt

# 同时指定
python task_reflection.py --root-dir /path/to/data --config /path/to/config.txt
```

---

## 5. 依赖关系

### 5.1 内部依赖

| 模块 | 用途 |
|------|------|
| src.config_loader | 配置加载（load_config, get_api_key 等） |
| src.tools.print_system | 输出系统（print_current, print_error 等） |
| .skill_tools | 技能工具（SkillTools 类） |

### 5.2 外部依赖

| 包 | 用途 | 可选 |
|----|------|------|
| openai | OpenAI API 客户端 | 是 |
| anthropic | Anthropic API 客户端 | 是 |
| yaml | YAML 文件处理 | 否 |
| requests | HTTP 请求 | 否 |

---

## 6. 数据流图

```
┌─────────────────────────────────────────────────────────────────┐
│                        输入数据                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  output_XXX/                                              │  │
│  │  ├── logs/                                                │  │
│  │  │   ├── manager.out      ← 主日志文件                    │  │
│  │  │   └── agent_*.out      ← Agent 日志文件                │  │
│  │  └── workspace/           ← 工作空间（代码文件）           │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      处理流程                                    │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │ 日志解析      │ →  │ LLM 反思     │ →  │ Skill 生成   │      │
│  │ _parse_log   │    │ _call_llm    │    │ _generate    │      │
│  └──────────────┘    └──────────────┘    └──────────────┘      │
│         │                   │                   │               │
│         ▼                   ▼                   ▼               │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │ 提取信息：    │    │ 分析内容：    │    │ 生成内容：    │      │
│  │ - 用户需求   │    │ - 任务完成度  │    │ - Front Matter│     │
│  │ - 工具调用   │    │ - 失败原因   │    │ - 反思内容    │      │
│  │ - 错误信息   │    │ - 成功策略   │    │ - 文件备份    │      │
│  │ - 完成状态   │    │ - 使用条件   │    │              │      │
│  └──────────────┘    └──────────────┘    └──────────────┘      │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                        输出数据                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  experience/                                              │  │
│  │  ├── skill_xxx.md         ← 生成的 Skill 文件             │  │
│  │  ├── codes/               ← 备份的代码文件                 │  │
│  │  └── logs/                ← 反思日志                      │  │
│  │      └── task_reflection_YYYYMMDD.log                    │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 7. 配置要求

### 7.1 必需配置项

在 `config/config.txt` 中需要配置：

```ini
# API 配置（必需）
api_key=your_api_key
api_base=https://api.openai.com/v1
model=gpt-4

# 数据目录（可选，有默认值）
gui_default_data_directory=/path/to/data
```

### 7.2 支持的模型

| 模型类型 | 识别方式 | 客户端 |
|----------|----------|--------|
| Claude | 模型名包含 "claude" 或 API base 包含 "anthropic" | Anthropic |
| GLM | API base 包含 "bigmodel.cn" | Anthropic（兼容模式） |
| Minimax | API base 包含 "minimaxi.com" | Anthropic（兼容模式） |
| 其他 | 默认 | OpenAI |

---

## 8. 错误处理

### 8.1 常见错误场景

| 场景 | 处理方式 |
|------|----------|
| LLM 客户端初始化失败 | 记录警告，继续运行（反思功能不可用） |
| 日志目录不存在 | 跳过该任务，记录警告 |
| 日志解析失败 | 返回空结果，记录错误 |
| LLM 调用失败 | 返回错误信息，继续处理 |
| Skill 文件生成失败 | 记录错误，返回 None |

### 8.2 日志级别

| 级别 | 使用场景 |
|------|----------|
| INFO | 正常处理流程 |
| WARNING | 非致命问题（如配置缺失） |
| ERROR | 处理失败（带堆栈跟踪） |

---

## 9. 最佳实践

### 9.1 使用建议

1. **定期运行**：建议在完成一批任务后运行反思脚本
2. **配置 LLM**：确保配置了有效的 LLM API 以获得高质量反思
3. **检查输出**：定期检查生成的 skill 文件质量
4. **清理旧数据**：定期清理过时的 output 目录

### 9.2 性能优化

1. **日志截取**：大日志文件会自动截取关键部分，避免 token 浪费
2. **批量处理**：支持一次处理多个任务目录
3. **增量处理**：可以通过指定 root_dir 只处理特定目录

---

## 10. 与其他模块的关系

```
┌─────────────────────────────────────────────────────────────────┐
│                     Skill Evolve 系统                           │
│                                                                 │
│  ┌─────────────────┐                                           │
│  │ task_reflection │ ← 本模块：任务反思，生成 Skill             │
│  └────────┬────────┘                                           │
│           │                                                     │
│           ▼                                                     │
│  ┌─────────────────┐                                           │
│  │  skill_tools    │ ← 技能工具：查询、评价、编辑、备份          │
│  └────────┬────────┘                                           │
│           │                                                     │
│           ▼                                                     │
│  ┌─────────────────┐                                           │
│  │ skill_manager   │ ← 技能管理：合并、清理、整合               │
│  └─────────────────┘                                           │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     AGIAgent 主系统                             │
│                                                                 │
│  任务执行时通过 query_skill 查询相关经验                         │
│  任务完成后通过 rate_skill 更新质量指数                          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 11. 版本历史

| 版本 | 日期 | 变更内容 |
|------|------|----------|
| 1.0 | 2025 | 初始版本，支持基本的任务反思和 Skill 生成 |

---

## 12. 总结

`task_reflection.py` 是 AGIAgent 实现"经验学习"能力的关键组件。通过自动化的日志分析和 LLM 反思，它能够：

1. **自动化经验提取**：从任务执行历史中提取有价值的经验
2. **结构化知识存储**：将经验转化为可查询、可复用的 Skill 文件
3. **持续能力进化**：通过不断积累 Skill，提升 Agent 的任务执行能力

这种设计使得 AGIAgent 能够从每次任务执行中学习，逐步积累领域知识和最佳实践，实现真正的"越用越聪明"。
