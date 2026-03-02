# tale_jericho

Jericho app for AGIAgent (single-action tool mode).

## Run One Session

```bash
cd /Users/apple/Desktop/ZiSuo/proj_03/AGIAgent
python agia.py --app tale_jericho
```

## Run All Tasks (isolated sessions)

Each task runs in an independent AGIAgent session and creates its own `output_*` directory.

```bash
cd /Users/apple/Desktop/ZiSuo/proj_03
conda run -n tale-suite python AGIAgent/apps/tale_jericho/run_all_tasks.py
cd /Users/apple/Desktop/ZiSuo/proj_03
conda run -n tale-suite python AGIAgent/apps/tale_jericho/run_all_tasks.py --task-list "[1,2,3]"

[21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35] 
[36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
[47, 48, 49, 50, 51, 52, 53, 54]



```

## Tool exposed to agent

- `tale_jericho_action`
