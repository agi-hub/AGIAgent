# tale_scienceworld

ScienceWorld app for AGIAgent (single-action tool mode).

## Run One Session

```bash
cd /Users/apple/Desktop/ZiSuo/proj_03/AGIAgent
python agia.py --app tale_scienceworld
```

## Run All Tasks (isolated sessions)

Each task runs in an independent AGIAgent session and creates its own `output_*` directory.

```bash
cd /Users/apple/Desktop/ZiSuo/proj_03
conda run -n tale-suite python AGIAgent/apps/tale_scienceworld/run_all_tasks.py
```

## Tool exposed to agent

- `tale_scienceworld_action`
