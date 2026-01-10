---
name: Skill管理与经验总结系统实现计划
overview: ""
todos:
  - id: create_task_reflection
    content: 创建任务整理脚本task_reflection.py，实现日志解析、LLM反思和skill生成功能
    status: pending
  - id: create_skill_manager
    content: 创建skill整理脚本skill_manager.py，实现相似度计算、skill合并和清理功能
    status: pending
  - id: create_skill_tools
    content: 创建skill_tools.py，实现query_skill、rate_skill、edit_skill、delete_skill和copy_skill_files工具
    status: pending
  - id: update_memory_tools
    content: 在memory_tools.json中添加query_skill、rate_skill、edit_skill、delete_skill和copy_skill_files工具描述
    status: pending
    dependencies:
      - create_skill_tools
  - id: integrate_tools
    content: 在tool_executor.py中注册新的skill工具
    status: pending
    dependencies:
      - create_skill_tools
  - id: add_skill_query_integration
    content: 在任务执行流程中集成skill查询功能（如果启用记忆开关）
    status: pending
    dependencies:
      - integrate_tools
---

# Skill管理与经验总结系统实现计划

## 概述

构建一个完整的技能管理和经验总结系统，包括任务反思脚本、skill整理脚本，以及query_skill和rate_skill工具。

## 文件结构

### 1. 任务整理脚本

**文件**: `AGIAgent/src/skill_evolve/task_reflection.py`主要功能：

- 读取config.txt获取`gui_default_data_directory`（默认`data`目录）
- 遍历所有用户目录下的工作空间（`output_XXX`目录）
- 分析`logs/manager.out`和`logs/agent_XXX.out`日志文件
- 使用LLM进行深度反思，提取经验
- 生成skill文件并保存到`user_dir/general/experience/`目录

关键实现点：

- 日志解析：识别工具调用、TASK_COMPLETED信号、用户中断、错误反馈等
- LLM反思：调用配置的模型进行任务分析
- Skill生成：每个任务生成一个skill文件（Markdown格式）
- 多智能体支持：分析agent间消息传递和协作

### 2. Skill整理脚本

**文件**: `AGIAgent/src/skill_evolve/skill_manager.py`主要功能：

- 读取`user_dir/general/experience/`目录下的所有skill文件
- 使用TF-IDF计算skill之间的相似度
- 合并相似度高的skill（阈值可配置）
- 清理长期不使用的skill（基于使用次数和时间）
- 跨skill整合：生成更高级的综合skill

关键实现点：

- TF-IDF相似度计算
- Skill合并策略：保留质量指数高的skill，合并内容
- 使用统计：跟踪skill的调用次数和最后使用时间
- 质量指数更新：合并后重新计算质量指数
- 支持命令行参数：`--root-dir`指定根目录
- 显示进度信息：处理每个skill时显示进度
- 错误处理：skill文件格式错误时，报错并跳过，继续处理其他文件

### 3. Skill查询和评价工具

**文件**: `AGIAgent/src/skill_evolve/skill_tools.py`新增工具：

- `query_skill(query: str)`: 使用TF-IDF检索最相似的TOP3 skills，返回完整skill内容（包含skill_id）和相似度分数，返回格式参考`recall_memories`工具格式
- `rate_skill(skill_id: str, rating: float)`: 评价skill质量（rating范围0-1），使用加权平均更新质量指数
- `edit_skill(skill_id: str, edit_mode: str, code_edit: str, old_code: str = None)`: 编辑skill文件（参数与edit_file完全一致，不需要验证文件格式）
- `delete_skill(skill_id: str)`: 删除skill文件（移动到`experience/legacy/`目录，不真正删除）
- `copy_skill_files(skill_id: str, file_paths: List[str])`: 复制文件到skill的代码备份目录（大模型根据manager.out决定备份哪些文件）

集成点：

- 在`tool_executor.py`中导入并注册新工具（从`skill_evolve`模块）
- 在`memory_tools.json`中添加工具描述（与长期记忆工具放在一起）
- 工具仅在`enable_long_term_memory=True`时可用

### 4. Memory Tools更新

**文件**: `AGIAgent/prompts/memory_tools.json`添加五个新工具：

- `query_skill`: 查询相关skill（返回完整内容，包含skill_id）
- `rate_skill`: 评价skill质量
- `edit_skill`: 编辑skill文件
- `delete_skill`: 删除skill文件（移动到legacy）
- `copy_skill_files`: 复制文件到skill的代码备份目录
- `copy_skill_files`: 复制文件到skill的代码备份目录

## Skill数据结构

### Skill文件命名规则

- 文件名：使用skill标题的前20个字符（去除特殊字符，确保文件系统安全）
- 格式：`skill_{title_prefix}.md`
- 示例：标题为"Python排序算法实现经验总结"，文件名为`skill_Python排序算法实现经验.md`

### Skill文件内容（Markdown格式）

使用Markdown front matter（YAML header）存储元数据，格式如下：

```markdown
---
skill_id: 1704067200
title: Python排序算法实现经验总结
usage_conditions: 当需要实现排序算法时使用
quality_index: 0.5
fetch_count: 0
related_code: codes/task_20260104_155505/workspace/sorting_algorithms.py
task_directories:
    - output_20260104_155505
created_at: 2026-01-04T15:55:05
updated_at: 2026-01-04T15:55:05
last_used_at: null
user_preferences: 用户偏好简洁的代码实现
---

# 详细内容

经验文字、成功代码等...
```

字段说明：

- **skill_id**: 时间戳（Unix timestamp）
- **title**: skill标题和概述
- **usage_conditions**: 使用条件
- **content**: 详细内容（经验文字、成功代码等，在front matter之后）
- **user_preferences**: 用户偏好情况
- **quality_index**: 质量指数（0-1，初始值0.5）
- **fetch_count**: 被查询的总次数（初始为0）
- **related_code**: 相关代码脚本路径（备份到`user_dir/general/experience/codes/`）
- **task_directories**: 对应的任务工作目录列表
- **created_at**: 创建时间（ISO格式）
- **updated_at**: 更新时间（ISO格式）
- **last_used_at**: 最后使用时间（ISO格式，初始为null）

## 实现细节

### 任务反思流程

1. **扫描阶段**：读取config.txt获取`gui_default_data_directory`（默认`data`目录），或使用命令行参数指定根目录，遍历所有用户目录下的工作空间（`output_XXX`目录）

- 支持命令行参数：`--root-dir`指定根目录
- 显示进度：处理每个任务时显示进度信息（如：`[1/10] Processing output_20260104_155505...`）

2. **排序阶段**：按时间倒序处理最近更新的任务
3. **日志解析阶段**：解析`logs/manager.out`和`logs/agent_XXX.out`日志文件，提取关键信息：

                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                - 用户需求（从"Received user requirement"提取）
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                - 工具调用序列（XML格式的`<invoke>`标签）
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                - 错误和反馈（ERROR FEEDBACK标记）
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                - TASK_COMPLETED信号
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                - 用户中断点：检测到新的user requirement，且在该requirement之前200个字符内没有TASK_COMPLETED（第一行除外）
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                - 多智能体消息传递（从agent_XXX.out解析）

4. **LLM反思阶段**：调用主模型（使用现有API接口或单独实现）进行深度反思分析，输出自然语言格式
5. **代码备份阶段**：大模型根据manager.out日志内容，使用`copy_skill_files`工具决定备份哪些文件到`user_dir/general/experience/codes/task_XXX/`目录

- 大模型会分析日志，识别哪些是成功产出的有效文件
- 备份文件类型：代码文件（.py, .js, .ts, .java等）和文档文件（.md, .txt等）
- 不包括：配置文件（.json, .yaml, .toml等）、图片文件
- 必须是有效输出物（用户实际需要的文件）

6. **Skill生成阶段**：生成skill文件（Markdown格式，包含front matter），文件名使用标题前20个字符，skill_id使用时间戳
7. **日志记录阶段**：将处理日志和错误日志统一输出到`user_dir/general/experience/logs/`目录（日志级别：INFO、ERROR）

### Skill相似度计算

- 使用TF-IDF向量化skill标题和内容
- 计算余弦相似度
- 相似度阈值：0.7（固定值）

### Skill合并策略

- **基础合并**：相似度 > 0.7时合并
- **合并操作**：保留质量指数更高的skill作为主skill，将另一个skill文件移动到`experience/legacy/`目录（不需要保留元数据）
- **内容整合**：合并内容、更新统计信息、更新质量指数（加权平均：`new_quality = 0.7 * old_quality + 0.3 * rating`）
- **legacy目录**：legacy目录下的skill不参与召回查询

### 跨Skill整合策略

1. **聚类阶段**：使用TF-IDF对所有skill进行向量化，使用DBSCAN算法自动确定聚类数量，找到聚类中心

- DBSCAN参数：`eps=0.5`（邻域半径），`min_samples=2`（最小样本数）

2. **候选识别**：遍历每个聚类，找到属于同一聚类的所有skill（一般2-3个skill合并成1个，距离不关键）
3. **LLM决策**：将同一聚类中的skill（2-3个）作为上下文，让LLM决定是否进行合并（输出自然语言）
4. **工具调用**：LLM可以调用`edit_skill`工具进行写入（参数与edit_file一致，不需要验证文件格式），调用`delete_skill`工具移除旧skill（移动到legacy目录，不需要保留元数据）
5. **执行合并**：根据LLM的决策执行合并操作

## 配置和依赖

- 使用现有的`config_loader.py`读取配置
- 使用现有的LLM API调用接口
- 需要安装`scikit-learn`用于TF-IDF计算
- 使用现有的日志解析逻辑

## 集成点

1. **任务开始前（plan之前）**：如果启用了`enable_long_term_memory`开关，在`system_prompt.txt`中添加提示词（英文），提醒大模型可以调用`query_skill`查找相关skill（特别是复杂任务）

- 提示词示例：
     ```javascript
          ## Skill Query Feature
          For complex tasks, you can use the `query_skill` tool to search for relevant historical experiences and skills that might help you complete the task more efficiently. This is especially useful when you encounter similar problems or need to follow established patterns.
          When you use skills from `query_skill`, make sure to:
                                        1. Keep the skill_id in your conversation history for reference
                                        2. Explicitly document which skills you referenced in plan.md
                                        3. After task completion, use `rate_skill` to update the quality index of skills you used
     ```


2. **任务执行中**：大模型可以自主调用`query_skill`查询相关skill，skill_id会保留在聊天历史中
3. **Plan.md撰写**：大模型在撰写plan.md时，需要明确标注参考了哪些skill（包含skill_id）
4. **任务完成后**：如果使用了skill，大模型调用`rate_skill`更新质量指数
5. **手动运行**：task_reflection.py和skill_manager.py都是手动脚本，需要用户主动运行
6. **工具可用性**：skill相关工具仅在`enable_long_term_memory=True`时可用

## 目录结构

### 代码目录

```javascript
AGIAgent/
├── src/
│   └── skill_evolve/
│       ├── __init__.py
│       ├── task_reflection.py
│       ├── skill_manager.py
│       └── skill_tools.py
└── prompts/
    └── tool_prompt.json
```



### 数据目录

```javascript
user_dir/
├── general/
│   └── experience/
│       ├── codes/              # 成功代码备份目录
│       │   └── task_XXX/       # 按任务目录组织
│       ├── logs/               # 处理日志和错误日志
│       │   ├── task_reflection_YYYYMMDD.log  # INFO和ERROR级别
│       │   └── skill_manager_YYYYMMDD.log    # INFO和ERROR级别
│       ├── legacy/             # 已合并的旧skill（不参与召回）
│       │   └── skill_XXX.md
│       └── skill_XXX.md         # 活跃的skill文件
└── output_XXX/
    ├── workspace/
    └── logs/
        ├── manager.out
        └── agent_XXX.out
```



## 注意事项

- Skill文件使用Markdown格式，便于阅读和编辑
- 需要处理编码问题（UTF-8）
- 日志解析需要考虑多种格式（XML工具调用、文本输出等）
- 需要处理多智能体场景下的消息传递分析
- **Skill ID生成**：使用时间戳（Unix timestamp）
- **Skill文件格式**：Markdown front matter（YAML header）+ 内容部分
- **质量指数**：初始值0.5，使用加权平均更新（`new_quality = 0.7 * old_quality + 0.3 * rating`），rating范围0-1
- **长期不使用的定义**：`fetch_count=0`且创建时间超过30天
- **用户中断检测**：user requirement之前200个字符内没有TASK_COMPLETED（第一行除外）
- **代码备份**：备份产出的独立文件（代码文件如.py、.js、.ts等和文档文件如.md、.txt等），不包括配置文件（.json、.yaml等）和图片文件，必须是有效输出物
- **聚类算法**：使用DBSCAN算法（eps=0.5, min_samples=2），一般2-3个skill合并成1个，距离不关键
- **LLM反思输出**：自然语言格式
- **query_skill返回格式**：参考`recall_memories`工具格式，包含status、message、skills_count、skills等字段，返回完整skill内容（包含skill_id）和相似度分数
- **skill_id使用**：query_skill返回的skill包含skill_id，大模型会在聊天历史中保留，并在plan.md中明确标注参考的skill
- **代码备份方式**：大模型根据manager.out日志内容，使用`copy_skill_files`工具决定备份哪些文件
- **任务反思脚本**：支持命令行参数`--root-dir`指定根目录，需要显示进度信息
- **skill整理脚本**：支持命令行参数`--root-dir`指定根目录，需要显示进度信息，skill文件格式错误时报错并跳过
- **DBSCAN参数**：固定使用eps=0.5, min_samples=2
- **edit_skill工具**：参数与edit_file完全一致，不需要验证文件格式
- **delete_skill工具**：将文件移动到`experience/legacy/`目录，不真正删除，不需要保留元数据
- **所有skill工具写入`memory_tools.json`**：与长期记忆工具放在一起
- **工具可用性**：仅在`enable_long_term_memory=True`时可用