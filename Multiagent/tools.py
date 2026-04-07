from __future__ import annotations

import os
import subprocess
from pathlib import Path


def _safe_resolve(repo_path: str, relative_path: str) -> Path:
    root = Path(repo_path).resolve()
    target = (root / relative_path).resolve()

    if root != target and root not in target.parents:
        raise ValueError(f"Path escapes repo root: {relative_path}")

    return target


def list_files(repo_path: str, max_files: int = 200) -> str:
    root = Path(repo_path).resolve()
    if not root.exists():
        return f"ERROR: repo path does not exist: {root}"

    files: list[str] = []
    ignore_dirs = {
        ".git", ".venv", "venv", "__pycache__", ".mypy_cache", ".pytest_cache",
        "node_modules", "dist", "build"
    }

    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in ignore_dirs]
        for name in filenames:
            full = Path(dirpath) / name
            rel = full.relative_to(root).as_posix()
            files.append(rel)
            if len(files) >= max_files:
                return "\n".join(files) + "\n...TRUNCATED..."

    return "\n".join(sorted(files))


def read_file(repo_path: str, relative_path: str, max_chars: int = 20000) -> str:
    try:
        target = _safe_resolve(repo_path, relative_path)
    except ValueError as e:
        return f"ERROR: {e}"

    if not target.exists():
        return f"ERROR: file does not exist: {relative_path}"

    if not target.is_file():
        return f"ERROR: not a file: {relative_path}"

    text = target.read_text(encoding="utf-8", errors="replace")
    if len(text) > max_chars:
        return text[:max_chars] + "\n...TRUNCATED..."
    return text


def _strip_fences(content: str) -> str:
    """Normalize model-generated file content: strip markdown fences and unescape newlines."""
    import re

    # Decode literal \n escape sequences if present (model serialized newlines as \n)
    real_newlines = content.count("\n")
    escaped_newlines = content.count("\\n")
    if escaped_newlines > real_newlines and escaped_newlines > 2:
        content = content.replace("\\n", "\n").replace("\\t", "\t")

    # Always unescape \" → " (JSON-serialized quotes in Python source code)
    if '\\"' in content:
        content = content.replace('\\"', '"')

    # Strip markdown code fences: ```lang\n...\n```
    stripped = content.strip()
    match = re.match(r"^```(?:\w+)?\n([\s\S]*?)```\s*$", stripped)
    return match.group(1) if match else stripped


def write_file(repo_path: str, relative_path: str, content: str) -> str:
    try:
        target = _safe_resolve(repo_path, relative_path)
    except ValueError as e:
        return f"ERROR: {e}"

    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(_strip_fences(content), encoding="utf-8")
    return f"OK: wrote {relative_path}"


def run_tests(repo_path: str) -> str:
    root = Path(repo_path).resolve()
    if not root.exists():
        return f"ERROR: repo path does not exist: {root}"

    # Use the venv Python if available, otherwise fall back to system python
    venv_python = root / ".venv" / "Scripts" / "python.exe"
    if not venv_python.exists():
        venv_python = root / ".venv" / "bin" / "python"

    if venv_python.exists():
        cmd = [str(venv_python), "-m", "pytest", "-q", "--tb=short"]
    else:
        cmd = ["python", "-m", "pytest", "-q", "--tb=short"]

    try:
        completed = subprocess.run(
            cmd,
            cwd=str(root),
            capture_output=True,
            text=True,
            timeout=180,
        )
    except subprocess.TimeoutExpired:
        return "ERROR: test command timed out after 180 seconds"
    except Exception as e:
        return f"ERROR: failed to run tests: {e}"

    stdout = completed.stdout.strip()
    stderr = completed.stderr.strip()

    result = [
        f"exit_code: {completed.returncode}",
        "--- STDOUT ---",
        stdout or "(empty)",
        "--- STDERR ---",
        stderr or "(empty)",
    ]
    return "\n".join(result)