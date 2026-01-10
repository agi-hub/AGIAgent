# Skill System Evaluation Test Suite

This test suite is used to evaluate whether AGIAgent's skill system can improve task execution accuracy through experience accumulation.

## Table of Contents

- [File Structure](#file-structure)
- [Test Case Description](#test-case-description)
- [Evaluation Metrics and Scoring Standards](#evaluation-metrics-and-scoring-standards)
- [Evaluation Process](#evaluation-process)
- [Quick Start](#quick-start)
- [Detailed Running Guide](#detailed-running-guide)
- [Evaluation Report Format and Examples](#evaluation-report-format-and-examples)
- [Viewing Evaluation Results](#viewing-evaluation-results)
- [Common Issues](#common-issues)
- [Interpreting Evaluation Results](#interpreting-evaluation-results)
- [Extending Test Cases](#extending-test-cases)
- [Notes](#notes)

## File Structure

```
tests/skill_evaluation/
├── __init__.py
├── test_dataset.py          # Test dataset definition
├── evaluator.py             # Evaluation program and scoring functions
├── benchmark_runner.py      # Evaluation runner program
├── test_cases/              # Test cases directory
│   ├── case_001_file_ops.json
│   ├── case_002_data_processing.json
│   ├── case_003_code_generation.json
│   └── case_004_text_processing.json
└── README.md                # This file
```

## Test Case Description

### case_001_file_ops.json
**File Operation Task**
- Task: Read file, count word occurrences, write results
- Difficulty: Easy
- Test Skills: Basic file operations, text processing, error handling

### case_002_data_processing.json
**Data Processing Task**
- Task: CSV data processing, calculate statistics, generate reports
- Difficulty: Medium
- Test Skills: CSV processing, data analysis, statistical calculations

### case_003_code_generation.json
**Code Generation Task**
- Task: Implement todo manager class, create example code
- Difficulty: Medium
- Test Skills: Object-oriented programming, class design, documentation writing

### case_004_text_processing.json
**Text Processing Task**
- Task: Text analysis, statistics, Markdown formatting
- Difficulty: Medium
- Test Skills: Text analysis, Markdown formatting, statistics

### Dataset Characteristics

1. **Task Diversity**: Covers different domains including file operations, data processing, code generation, and text processing
2. **Progressive Difficulty**: From easy to medium difficulty, gradually testing system capabilities
3. **Reusability**: Tasks have similarities (e.g., all involve file operations), allowing testing of cross-task skill integration effects
4. **Verifiability**: Each task has clear output requirements and verification standards

## Evaluation Metrics and Scoring Standards

### Scoring System

The evaluation uses a **multi-dimensional weighted scoring mechanism** with a total score range of 0-1.0:

```
Total Score = Completion Score × 0.4 + Quality Score × 0.3 + Efficiency Score × 0.2 + Innovation Score × 0.1
```

### Scoring Standards for Each Dimension

#### Completion Score (40% weight)

**Scoring Items**:
- **File Existence** (20 points): Check if required files are created
  - All required files exist: 20 points
  - Some files exist: Score proportionally
- **Content Validity** (20 points): Check if output content meets requirements
  - Whether required keywords exist
  - Whether file format is correct

**Example Scores**:
- Perfect completion: 40 points (all files exist and content fully meets requirements)
- Partial completion: 20-30 points (files exist but content partially meets requirements)
- Incomplete: 0-10 points (files missing or content doesn't meet requirements)

#### Quality Score (30% weight)

**Scoring Items**:
- **Error Handling** (10 points): Whether code contains exception handling
- **Documentation Strings** (10 points): Whether code contains docstrings
- **Code Structure** (10 points): Whether code has good structure (classes, function definitions)

**Example Scores**:
- High quality code: 30 points (includes error handling, docstrings, and good structure)
- Medium quality: 15-20 points (includes some quality features)
- Low quality: 0-10 points (lacks quality features)

#### Efficiency Score (20% weight)

**Scoring Items**:
- **Execution Rounds**: Compare with maximum rounds set in test case
- **Tool Call Count**: Compare with maximum tool calls set in test case
- **Execution Time**: Deduct points if exceeds 5 minutes

**Example Scores** (assuming max rounds = 5, max tool calls = 12):
- Efficient execution: 20 points (completed in 3 rounds, 8 tool calls)
- Medium efficiency: 10-15 points (completed in 5-6 rounds, 12-15 tool calls)
- Inefficient execution: 0-10 points (exceeds max rounds or tool call count)

#### Innovation Score (10% weight)

**Scoring Items**:
- **Skill Usage**: Whether skills were queried and used during task execution
  - Skills used: 10 points
  - Skills not used: 0 points

## Evaluation Process

### Complete Evaluation Process

```
Stage 1: Baseline Testing (without skills)
    ↓
   Execute all test cases, record scores
    ↓
Stage 2: Task Reflection (generate skills)
    ↓
   Analyze task logs, extract experience, generate skill files
    ↓
Stage 3: Skill Management (integrate skills)
    ↓
   Merge similar skills, perform cross-task integration
    ↓
Stage 4: Skill Testing (with skills)
    ↓
   Enable skill memory, re-execute all test cases, record scores
    ↓
Stage 5: Result Comparison and Analysis
    ↓
   Compare two test results, calculate improvement metrics
```

### Stage Descriptions

1. **Stage 1: Baseline Testing**
   - Execute all test cases without skills
   - Record scores, success rate, execution rounds, and other metrics

2. **Stage 2: Task Reflection**
   - Call `task_reflection.py` to analyze task logs
   - Generate skill files

3. **Stage 3: Skill Management**
   - Call `skill_manager.py` to organize skills
   - Merge similar skills, perform cross-task integration

4. **Stage 4: Skill Testing**
   - Enable skill memory, re-execute all test cases
   - Record scores, success rate, execution rounds, and other metrics

5. **Stage 5: Result Comparison**
   - Compare baseline and skill test results
   - Calculate score improvement, success rate improvement, and other metrics
   - Generate evaluation report

## Quick Start

### 1. Complete Evaluation Process (Recommended)

Run the complete evaluation process, including baseline testing, task reflection, skill management, skill testing, and result comparison:

```bash
cd /home/agibot/AGIAgent
python tests/skill_evaluation/benchmark_runner.py --root-dir data --config config/config.txt
```

**Estimated Time**: 20-30 minutes (depending on API response speed and task complexity)

### 2. Stage-by-Stage Execution

If the complete evaluation takes too long, you can run it stage by stage:

#### Stage 1: Baseline Testing (without skills)
```bash
python tests/skill_evaluation/benchmark_runner.py --baseline-only --root-dir data --config config/config.txt
```
**Estimated Time**: 5-10 minutes

#### Stage 2: Task Reflection (generate skills)
```bash
python tests/skill_evaluation/benchmark_runner.py --reflection-only --root-dir data --config config/config.txt
```
**Estimated Time**: 2-5 minutes

#### Stage 3: Skill Management (integrate skills)
```bash
python tests/skill_evaluation/benchmark_runner.py --manager-only --root-dir data --config config/config.txt
```
**Estimated Time**: 1-3 minutes

#### Stage 4: Skill Testing (with skills)
```bash
python tests/skill_evaluation/benchmark_runner.py --skill-only --root-dir data --config config/config.txt
```
**Estimated Time**: 5-10 minutes

## Detailed Running Guide

### Parameter Description

#### Required Parameters

- `--root-dir`: Root directory path for data
  - Default: `data`
  - Example: `--root-dir /path/to/data`

- `--config`: Configuration file path
  - Default: `config/config.txt`
  - Example: `--config config/config.txt`

#### Optional Parameters

- `--user-id`: User ID (optional)
  - Example: `--user-id user123`

#### Execution Modes

- `--baseline-only`: Run only baseline testing
- `--skill-only`: Run only skill testing
- `--reflection-only`: Run only task reflection
- `--manager-only`: Run only skill management
- No mode specified: Run complete evaluation process

### Pre-Run Preparation

#### 1. Check Configuration File

Ensure `config/config.txt` is configured correctly:

```bash
# Check API configuration
grep -E "^api_key=|^api_base=|^model=" config/config.txt | tail -3

# Check if long-term memory is enabled
grep "enable_long_term_memory" config/config.txt
```

**Important**: Ensure `enable_long_term_memory=True` to enable skill functionality

#### 2. Check Test Cases

```bash
# View test cases
ls -la tests/skill_evaluation/test_cases/
```

#### 3. Ensure Sufficient Disk Space

The evaluation generates many output files. Ensure sufficient disk space (recommended at least 500MB)

### Running Examples

#### Example 1: Complete Evaluation (Most Common)

```bash
cd /home/agibot/AGIAgent
python tests/skill_evaluation/benchmark_runner.py \
    --root-dir data \
    --config config/config.txt
```

#### Example 2: Baseline Testing Only

```bash
python tests/skill_evaluation/benchmark_runner.py \
    --baseline-only \
    --root-dir data \
    --config config/config.txt
```

#### Example 3: Specify User ID

```bash
python tests/skill_evaluation/benchmark_runner.py \
    --root-dir data \
    --config config/config.txt \
    --user-id test_user
```

#### Example 4: Background Execution with Logging

```bash
nohup python tests/skill_evaluation/benchmark_runner.py \
    --root-dir data \
    --config config/config.txt \
    > benchmark_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

### Enabling Skill Functionality

To enable skill functionality, set in `config/config.txt`:

```
enable_long_term_memory=True
```

Or ensure the environment variable `AGIBOT_LONG_TERM_MEMORY` is not `false`/`0`/`no`/`off`.

## Evaluation Report Format and Examples

### Evaluation Report Format

The actual generated evaluation report is in JSON format and contains the following:

```json
{
  "baseline": {
    "summary": {
      "total_cases": 4,
      "success_count": 4,
      "success_rate": 1.0,
      "average_score": 0.22,
      "total_score": 0.88
    },
    "test_cases": [
      {
        "task_id": "case_001",
        "success": true,
        "total_score": 0.22,
        "completion_score": 0.40,
        "quality_score": 0.20,
        "efficiency_score": 0.15,
        "innovation_score": 0.00,
        "execution_time": 45.2,
        "rounds": 5,
        "tool_calls": 12,
        "skill_used": false
      }
    ]
  },
  "skill": {
    "summary": {
      "total_cases": 4,
      "success_count": 4,
      "success_rate": 1.0,
      "average_score": 0.28,
      "total_score": 1.12
    }
  },
  "improvements": {
    "score_improvement": 0.06,
    "score_improvement_pct": 27.3,
    "success_rate_improvement": 0.0,
    "success_rate_improvement_pct": 0.0
  },
  "detailed_comparison": [
    {
      "task_id": "case_001",
      "baseline_score": 0.22,
      "skill_score": 0.28,
      "score_improvement": 0.06,
      "baseline_rounds": 5,
      "skill_rounds": 4,
      "rounds_improvement": 1,
      "skill_used": true
    }
  ]
}
```

### Expected Evaluation Result Examples

#### Baseline Test Results (without skills)

| Test Case | Completion | Quality | Efficiency | Innovation | Total | Success |
|-----------|------------|---------|------------|------------|-------|---------|
| case_001 | 0.40 | 0.20 | 0.15 | 0.00 | 0.22 | Yes |
| case_002 | 0.35 | 0.15 | 0.10 | 0.00 | 0.19 | Yes |
| case_003 | 0.30 | 0.25 | 0.12 | 0.00 | 0.22 | Yes |
| case_004 | 0.35 | 0.20 | 0.15 | 0.00 | 0.24 | Yes |

**Baseline Test Summary**:
- Average Score: 0.22
- Success Rate: 100% (4/4)
- Average Execution Rounds: 5.2 rounds
- Average Tool Calls: 11.5 times

#### Skill Test Results (with skills)

| Test Case | Completion | Quality | Efficiency | Innovation | Total | Success |
|-----------|------------|---------|------------|------------|-------|---------|
| case_001 | 0.40 | 0.25 | 0.18 | 0.10 | 0.28 | Yes |
| case_002 | 0.40 | 0.20 | 0.15 | 0.10 | 0.26 | Yes |
| case_003 | 0.40 | 0.30 | 0.16 | 0.10 | 0.30 | Yes |
| case_004 | 0.40 | 0.25 | 0.18 | 0.10 | 0.29 | Yes |

**Skill Test Summary**:
- Average Score: 0.28
- Success Rate: 100% (4/4)
- Average Execution Rounds: 4.1 rounds
- Average Tool Calls: 9.2 times

#### Comparison Analysis Results

**Improvement Metrics**:
- **Score Improvement**: 0.06 (from 0.22 to 0.28)
- **Score Improvement Percentage**: 27.3%
- **Success Rate Improvement**: 0% (maintained at 100%)
- **Rounds Improvement**: 1.1 rounds (reduced from 5.2 to 4.1 rounds)
- **Tool Calls Improvement**: 2.3 times (reduced from 11.5 to 9.2 times)

**Detailed Comparison**:

| Test Case | Baseline Score | Skill Score | Score Improvement | Rounds Improvement | Tool Calls Improvement | Skill Used |
|-----------|----------------|-------------|-------------------|---------------------|------------------------|------------|
| case_001 | 0.22 | 0.28 | +0.06 | -1.2 | -2.5 | Yes |
| case_002 | 0.19 | 0.26 | +0.07 | -1.0 | -2.0 | Yes |
| case_003 | 0.22 | 0.30 | +0.08 | -1.3 | -2.8 | Yes |
| case_004 | 0.24 | 0.29 | +0.05 | -1.0 | -2.0 | Yes |

### Skill System Effectiveness Validation

Based on expected evaluation results, the Skill system demonstrates effectiveness in the following aspects:

1. **Task Quality Improvement**:
   - Average score improvement of 27.3%
   - All test cases show improvements in completion and quality scores

2. **Execution Efficiency Improvement**:
   - Average execution rounds reduced by 21.2% (from 5.2 to 4.1 rounds)
   - Average tool calls reduced by 20.0% (from 11.5 to 9.2 times)

3. **Skill Usage**:
   - All test cases successfully used skills
   - Skills were correctly queried and applied

### Cross-Task Skill Integration Effects

Due to similarities between test cases (e.g., all involve file operations), the Skill system can:

1. **Cross-Task Reuse**: Skills from one task can help complete another task
2. **Experience Accumulation**: After multiple executions, the system can better complete tasks
3. **Efficiency Improvement**: Through experience reuse, execution rounds and tool calls are reduced

## Viewing Evaluation Results

### 1. Evaluation Results Directory

Evaluation results are saved in the `{root_dir}/benchmark_results/` directory:

```bash
ls -la data/benchmark_results/
```

### 2. View Latest Report

```bash
# View the latest evaluation report (JSON format)
ls -lt data/benchmark_results/report_*.json | head -1 | awk '{print $NF}' | xargs cat | python3 -m json.tool

# Or view directly
cat data/benchmark_results/report_*.json | python3 -m json.tool | less
```

### 3. View Baseline Test Results

```bash
# Find the latest baseline test directory
BASELINE_DIR=$(ls -td data/benchmark_results/baseline_* | head -1)

# View results
cat $BASELINE_DIR/results.json | python3 -m json.tool
```

### 4. View Skill Test Results

```bash
# Find the latest skill test directory
SKILL_DIR=$(ls -td data/benchmark_results/skill_* | head -1)

# View results
cat $SKILL_DIR/results.json | python3 -m json.tool
```

### 5. View Task Outputs

```bash
# View output files for a test case
BASELINE_DIR=$(ls -td data/benchmark_results/baseline_* | head -1)
ls -la $BASELINE_DIR/baseline_outputs/output_case_001_*/workspace/
```

### Evaluation Output Description

#### Console Output

During evaluation execution, the following will be displayed:

```
📁 Root Directory: /home/agibot/AGIAgent/data
📁 Results Directory: /home/agibot/AGIAgent/data/benchmark_results
📋 Test Cases Count: 4

============================================================
Stage 1: Baseline Testing (without skills)
============================================================

[1/4] Executing test case: case_001
Task Description: ...
  Score: 0.XX | Completion: 0.XX | Quality: 0.XX | Efficiency: 0.XX | Innovation: 0.XX
  Success: Yes/No | Rounds: X | Tool Calls: X | Skill Used: Yes/No

...

Baseline testing completed!
Average Score: 0.XX
Success Rate: XX.XX%
Results saved to: ...
```

#### Result File Structure

```
data/benchmark_results/
├── baseline_YYYYMMDD_HHMMSS/          # Baseline test results
│   ├── results.json                    # Test results JSON
│   └── baseline_outputs/               # Task output directory
│       └── output_case_XXX_*/          # Outputs for each test case
│           └── workspace/              # Workspace files
├── skill_YYYYMMDD_HHMMSS/              # Skill test results
│   ├── results.json                    # Test results JSON
│   └── skill_outputs/                  # Task output directory
└── report_YYYYMMDD_HHMMSS.json         # Comparison analysis report
```

## Common Issues

### Q1: API Call Error (401 Error)

**Problem**: `Error code: 401 - {'error': {'message': 'User not found.', 'code': 401}}`

**Solution**:
1. Check if the API key in `config/config.txt` is correct
2. Verify API endpoint configuration is correct
3. Verify API service is available

### Q2: Evaluation Takes Too Long

**Problem**: Evaluation runs for more than 30 minutes without completion

**Solution**:
1. Check if network connection is normal
2. Check API response speed
3. Run stage by stage, starting with baseline testing

### Q3: Test Cases Not Found

**Problem**: `FileNotFoundError: Test cases directory not found`

**Solution**:
1. Verify test case files exist:
   ```bash
   ls tests/skill_evaluation/test_cases/*.json
   ```
2. Ensure running command from project root directory

### Q4: Skill Tool Not Enabled

**Problem**: Evaluation shows `⚠️ Experience directory not found`

**Solution**:
1. Ensure `enable_long_term_memory=True` in `config/config.txt`
2. Check if user directory structure exists

### Q5: How to View Evaluation Progress

**Solution**:
```bash
# View evaluation process
ps aux | grep benchmark_runner

# View latest logs (if running in background)
tail -f benchmark_*.log
```

## Interpreting Evaluation Results

### Key Metrics

1. **Average Score**: Average score of all test cases (0-1.0)
2. **Success Rate**: Proportion of test cases that successfully completed tasks
3. **Score Improvement**: Skill test score - Baseline test score
4. **Score Improvement Percentage**: (Score Improvement / Baseline Test Score) × 100%

### Evaluation Standards

- **Significant Improvement**: Score improvement > 10%, success rate improvement > 20%
- **Moderate Improvement**: Score improvement 5-10%, success rate improvement 10-20%
- **Slight Improvement**: Score improvement < 5%, success rate improvement < 10%
- **No Improvement or Decline**: Need to check skill system configuration and usage

### Quick Reference

#### Most Common Commands

```bash
# Complete evaluation
python tests/skill_evaluation/benchmark_runner.py --root-dir data --config config/config.txt

# View latest report
cat data/benchmark_results/report_*.json | python3 -m json.tool | tail -100

# View evaluation progress
ps aux | grep benchmark_runner
```

## Extending Test Cases

To add new test cases, simply create a new JSON file in the `test_cases/` directory, following the format of existing test cases.

Test case JSON files should include:
- `task_id`: Task ID
- `task_description`: Task description
- `output_requirements`: Output requirements
- `evaluation_criteria`: Evaluation criteria
- `category`: Task category
- `difficulty`: Difficulty level

## Notes

1. **Time Requirement**: Complete evaluation takes 20-30 minutes, ensure sufficient time
2. **API Quota**: Evaluation makes many API calls, be aware of API quota limits
3. **Disk Space**: Ensure sufficient disk space for output files (recommended at least 500MB)
4. **Network Connection**: Ensure stable network connection and normal API calls
5. **Configuration Check**: Before running, check API configuration and long-term memory settings
6. **Ensure correct API keys and model information are configured in `config/config.txt`**
7. **Ensure `enable_long_term_memory=True` to enable skill functionality**
8. **If a stage fails, you can run that stage separately for debugging**
