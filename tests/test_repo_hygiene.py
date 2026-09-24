"""Guards against source files silently missing from the repository.

This exists because of a real incident. The `.gitignore` line `data/` matches a
directory of that name at ANY depth, so it quietly excluded `nlt/data/` -- the
entire market-data layer. Every push succeeded, `git status` reported a clean
tree, and the files were plainly there on disk. The repository on GitHub was
broken for days and nothing said so.

The only moment that failure surfaces is when someone clones, which is exactly
the moment the backup is being relied upon. So it is checked here instead: every
source file must be tracked by git, and no source path may be ignored.

A pattern-level fix would have addressed one instance. This addresses the class.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]

# Directories holding code that must reach the remote. `data/` (the bar cache)
# and `.venv/` are legitimately ignored and are deliberately not listed.
SOURCE_DIRS = ("nlt", "app", "tests", "scripts")

# Files that are genuinely generated and should not be tracked.
IGNORABLE_SUFFIXES = (".pyc", ".pyo", ".parquet", ".sqlite", ".log")


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO, capture_output=True, text=True, check=True
    ).stdout


def _is_repo() -> bool:
    return (REPO / ".git").exists()


pytestmark = pytest.mark.skipif(not _is_repo(), reason="not a git checkout")


def _source_files() -> list[Path]:
    found = []
    for directory in SOURCE_DIRS:
        root = REPO / directory
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file() or "__pycache__" in path.parts:
                continue
            if path.suffix in IGNORABLE_SUFFIXES:
                continue
            found.append(path)
    return found


def test_every_source_file_is_tracked_by_git():
    """A source file git does not know about does not exist for anyone else."""
    tracked = {REPO / line for line in _git("ls-files").splitlines()}
    untracked = sorted(str(p.relative_to(REPO)) for p in _source_files() if p not in tracked)

    assert not untracked, (
        "these source files are not tracked by git and would be missing from a "
        "fresh clone:\n  " + "\n  ".join(untracked)
    )


def test_no_source_directory_is_ignored():
    """Catches the exact bug: an unanchored pattern swallowing a package.

    `git check-ignore` is asked directly, so this reflects what git will really
    do rather than our reading of the patterns.
    """
    candidates = [f"{d}/" for d in SOURCE_DIRS]
    candidates += [
        # Packages whose names collide with common ignore patterns. These are
        # the landmines: a bare `data/`, `lib/` or `build/` in .gitignore takes
        # them out silently.
        "nlt/data/",
        "nlt/data/source.py",
        "nlt/report/",
        "app/main.py",
    ]

    result = subprocess.run(
        ["git", "check-ignore", "-v", *candidates],
        cwd=REPO, capture_output=True, text=True,
    )
    # Exit code 1 means nothing matched, which is what we want.
    assert result.returncode == 1, (
        "these source paths are excluded by .gitignore and would never be "
        f"pushed:\n{result.stdout}"
    )


def test_gitignore_patterns_that_could_swallow_a_package_are_anchored():
    """Directory patterns must be rooted, or they match at every depth.

    `data/` matches `nlt/data/`; `/data/` does not. Anything in this list is a
    word likely to appear as a package name somewhere in a Python project.
    """
    risky = {"data", "lib", "build", "dist", "src", "docs", "bin", "test", "tests", "app"}

    offenders = []
    for raw in (REPO / ".gitignore").read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or line.startswith("/") or line.startswith("!"):
            continue
        if line.rstrip("/").lower() in risky and "/" not in line.rstrip("/"):
            offenders.append(line)

    assert not offenders, (
        "these .gitignore patterns are unanchored and match a directory of that "
        f"name at any depth: {offenders}. Prefix each with '/' to root it at the "
        "repository top level."
    )


def test_a_fresh_clone_would_contain_the_whole_package():
    """Every importable module under nlt/ and app/ is in git's index."""
    tracked = set(_git("ls-files").splitlines())
    modules = [
        str(p.relative_to(REPO))
        for p in (REPO / "nlt").rglob("*.py")
        if "__pycache__" not in p.parts
    ]
    missing = sorted(m for m in modules if m not in tracked)

    assert not missing, f"importable modules absent from the repository: {missing}"
