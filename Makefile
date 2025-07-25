# AGIBot Makefile
# 简化开发和测试流程

.PHONY: help install test test-unit test-integration test-e2e test-performance test-security test-all
.PHONY: coverage lint format security-scan clean docker-test setup-dev
.PHONY: docs serve-docs build release

# 默认目标
help:
	@echo "AGIBot 开发工具命令："
	@echo ""
	@echo "安装和设置:"
	@echo "  install       安装项目依赖"
	@echo "  setup-dev     设置开发环境"
	@echo ""
	@echo "测试命令:"
	@echo "  test          运行快速测试套件"
	@echo "  test-unit     运行单元测试"
	@echo "  test-integration  运行集成测试"
	@echo "  test-e2e      运行端到端测试"
	@echo "  test-performance  运行性能测试"
	@echo "  test-security 运行安全测试"
	@echo "  test-all      运行完整测试套件"
	@echo "  coverage      生成覆盖率报告"
	@echo ""
	@echo "代码质量:"
	@echo "  lint          代码风格检查"
	@echo "  format        代码格式化"
	@echo "  security-scan 安全扫描"
	@echo ""
	@echo "Docker:"
	@echo "  docker-test   在Docker中运行测试"
	@echo ""
	@echo "其他:"
	@echo "  clean         清理临时文件"
	@echo "  docs          生成文档"
	@echo "  serve-docs    启动文档服务器"

# 安装依赖
install:
	@echo "📦 安装项目依赖..."
	pip install --upgrade pip
	pip install -r requirements.txt

# 设置开发环境
setup-dev: install
	@echo "🔧 设置开发环境..."
	pip install pytest pytest-cov pytest-html pytest-xdist
	pip install bandit safety flake8 black isort
	pip install sphinx sphinx-rtd-theme
	@echo "✅ 开发环境设置完成"

# 快速测试（跳过慢速测试）
test:
	@echo "🚀 运行快速测试套件..."
	python -m pytest src/tests -m "not slow and not performance" -v --tb=short

# 单元测试
test-unit:
	@echo "🔧 运行单元测试..."
	python -m pytest src/tests/unit -v --tb=short

# 集成测试
test-integration:
	@echo "🔗 运行集成测试..."
	python -m pytest src/tests/integration -v --tb=short

# 端到端测试
test-e2e:
	@echo "🎯 运行端到端测试..."
	python -m pytest src/tests/e2e -v --tb=short

# 性能测试
test-performance:
	@echo "⚡ 运行性能测试..."
	python -m pytest src/tests/performance -v --tb=short -m performance

# 安全测试
test-security:
	@echo "🔒 运行安全测试..."
	python -m pytest src/tests/security -v --tb=short

# 完整测试套件
test-all:
	@echo "🎪 运行完整测试套件..."
	python -m pytest src/tests -v --tb=short \
		--junit-xml=test_reports/junit.xml \
		--html=test_reports/report.html \
		--self-contained-html

# 覆盖率测试
coverage:
	@echo "📊 生成覆盖率报告..."
	python -m pytest src/tests \
		--cov=src \
		--cov-report=html:test_reports/coverage_html \
		--cov-report=xml:test_reports/coverage.xml \
		--cov-report=term-missing \
		--cov-fail-under=80

# 代码风格检查
lint:
	@echo "🎨 检查代码风格..."
	flake8 src --count --select=E9,F63,F7,F82 --show-source --statistics
	flake8 src --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics
	isort --check-only src

# 代码格式化
format:
	@echo "✨ 格式化代码..."
	black src
	isort src
	@echo "✅ 代码格式化完成"

# 安全扫描
security-scan:
	@echo "🔍 运行安全扫描..."
	bandit -r src -f json -o test_reports/bandit-report.json || true
	safety check --json --output test_reports/safety-report.json || true
	@echo "📋 安全扫描报告已生成"

# Docker测试
docker-test:
	@echo "🐳 在Docker中运行测试..."
	docker build -t agibot:test .
	docker run --rm -v $(PWD):/workspace agibot:test \
		python -m pytest /workspace/src/tests/unit -v --tb=short

# 清理临时文件
clean:
	@echo "🧹 清理临时文件..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	rm -rf build/
	rm -rf dist/
	rm -rf test_reports/*.xml
	rm -rf test_reports/*.json
	rm -rf test_reports/coverage_html/
	@echo "✅ 清理完成"

# 生成文档
docs:
	@echo "📚 生成文档..."
	@if [ ! -d "docs" ]; then \
		mkdir docs; \
		cd docs && sphinx-quickstart -q -p "AGIBot" -a "AGIBot Team" -v "1.0" --ext-autodoc --ext-viewcode --makefile .; \
	fi
	cd docs && make html
	@echo "📖 文档已生成到 docs/_build/html/"

# 启动文档服务器
serve-docs: docs
	@echo "🌐 启动文档服务器..."
	cd docs/_build/html && python -m http.server 8000

# 并行测试
test-parallel:
	@echo "🚄 并行运行测试..."
	python -m pytest src/tests -n auto -v --tb=short

# 持续测试（文件变化时自动运行）
test-watch:
	@echo "👀 启动持续测试模式..."
	@command -v pytest-watch >/dev/null 2>&1 || { echo "需要安装 pytest-watch: pip install pytest-watch"; exit 1; }
	ptw src/tests -- -v --tb=short

# 基准测试
benchmark:
	@echo "📈 运行基准测试..."
	python -m pytest src/tests/performance -v --tb=short \
		--benchmark-only \
		--benchmark-json=test_reports/benchmark.json

# 内存分析
profile-memory:
	@echo "🧠 运行内存分析..."
	python -m pytest src/tests -v --tb=short \
		--profile --profile-svg

# 依赖检查
check-deps:
	@echo "🔍 检查依赖安全性..."
	safety check
	pip-audit || echo "pip-audit not installed, skipping"

# 类型检查
type-check:
	@echo "🔍 运行类型检查..."
	@command -v mypy >/dev/null 2>&1 || { echo "需要安装 mypy: pip install mypy"; exit 1; }
	mypy src --ignore-missing-imports

# 完整质量检查
quality: lint type-check security-scan test-all
	@echo "✅ 质量检查完成"

# 发布前检查
pre-release: clean quality coverage
	@echo "🚀 发布前检查完成"

# 创建测试报告目录
test_reports:
	@mkdir -p test_reports

# 所有测试命令都依赖报告目录
test test-unit test-integration test-e2e test-performance test-security test-all coverage: test_reports 