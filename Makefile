# The checks. `make check` is the gate; everything else is a piece of it.
#
# There is no CI. The gate is a local git hook, installed by `make hooks`, and
# `git commit --no-verify` walks straight past it. That is deliberate -- it is a
# gate you choose to keep. It also means a green branch you did not watch go
# green is not evidence of anything.

PY := .venv/bin/python

.PHONY: check fix lint format test mutation secrets hooks clean-cache help

help:
	@echo "make check     lint + format check + the full test suite (the gate)"
	@echo "make fix       autofix lint and rewrite formatting -- run before reading errors"
	@echo "make test      the test suite alone"
	@echo "make mutation  mutation testing: do the tests FAIL when the code breaks? (slow)"
	@echo "make secrets   scan the whole git history for leaked credentials"
	@echo "make hooks     install the git hooks (once per clone)"

check: lint format test

lint:
	@$(PY) -m ruff check .

format:
	@$(PY) -m ruff format --check .

fix:
	@$(PY) -m ruff check . --fix
	@$(PY) -m ruff format .

test:
	@$(PY) -m pytest -q

# Python caches compiled bytecode by (size, mtime). A mutant and its original are
# often the same length, so a verification run can silently execute the stale
# copy and report nonsense. This has already happened once in this repo. Clear
# it before any run whose entire purpose is that the code changed.
clean-cache:
	@find . -path ./.venv -prune -o -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
	@rm -rf .pytest_cache

mutation: clean-cache
	@echo "Mutation testing nlt/engine, nlt/costs, nlt/spec: ~2400 mutants, 12-15 min."
	@echo "A surviving mutant means the code was broken and no test noticed."
	@$(PY) -m mutmut run
	@$(PY) -m mutmut results

# PATH is set explicitly because git hooks do not inherit the mise shims, and
# this target is called from one. `mise install` rather than `mise use -g`:
# the version is already pinned in ~/.config/mise/config.toml, which survives a
# container rebuild even though the installed binary does not.
secrets:
	@PATH="/opt/mise/shims:$$HOME/.local/bin:$$PATH"; \
	command -v gitleaks >/dev/null 2>&1 || { \
		echo "gitleaks is not installed, so the history cannot be scanned."; \
		echo "Run: mise install"; exit 1; }; \
	gitleaks git --no-banner --redact . && echo "No secrets found in history."

hooks:
	@git config core.hooksPath .githooks
	@chmod +x .githooks/*
	@echo "Hooks installed: pre-commit runs 'make check', pre-push runs 'make secrets'."
	@echo "Both are bypassable with --no-verify."
