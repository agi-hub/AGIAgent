Run full Dataset:
Re-write tests/test_cases/*.json to fit the dataset samples
Then:
python tests/skill_evaluation/benchmark_runner.py --baseline-only --root-dir data --config config/config.txt
python tests/skill_evaluation/benchmark_runner.py --reflection-only --root-dir data --config config/config.txt
python tests/skill_evaluation/benchmark_runner.py --manager-only --root-dir data --config config/config.txt
python tests/skill_evaluation/benchmark_runner.py --skill-only --root-dir data --config config/config.txt



Run single example:
python agia.py "run custom tool to play hanoi" -d data/benchmark_results/baseline_hanoi

Then:
python tests/skill_evaluation/benchmark_runner.py --reflection-only --root-dir data --config config/config.txt

python tests/skill_evaluation/benchmark_runner.py --manager-only --root-dir data --config config/config.txt

python agia.py "run custom tool to play hanoi" -d data/benchmark_results/evolved_hanoi
