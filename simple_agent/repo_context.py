from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


DEFAULT_IGNORE_DIRS = {
    ".git",
    ".hg",
    ".idea",
    ".mypy_cache",
    ".pytest_cache",
    ".pycache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "build",
    "dist",
    "node_modules",
}


@dataclass(frozen=True)
class FileContext:
    path: str
    lines: int
    bytes: int
    summary: str


@dataclass(frozen=True)
class RepoContext:
    root: Path
    tree: str
    files: list[FileContext]

    def compressed(self, max_files: int = 80) -> str:
        file_lines = []
        for item in self.files[:max_files]:
            file_lines.append(f"- {item.path}: {item.lines} lines, {item.summary}")
        omitted = "" if len(self.files) <= max_files else f"\n... omitted {len(self.files) - max_files} files"
        return f"Repo root: {self.root}\n\nTree:\n{self.tree}\n\nFile context:\n" + "\n".join(file_lines) + omitted


class ContextManager:
    """Incremental context manager with file caching and token budgeting."""

    def __init__(self, root: str | Path, max_tokens: int = 100000) -> None:
        self.root = Path(root).resolve()
        self.max_tokens = max_tokens
        self._file_cache: dict[str, tuple[float, str]] = {}  # rel_path -> (mtime, content)
        self._repo_context_cache: tuple[float, RepoContext] | None = None
        self._accessed_files: set[str] = set()

    def get_repo_context(self, max_depth: int = 3, max_files: int = 120) -> RepoContext:
        current_mtime = self._get_max_mtime()
        if self._repo_context_cache is not None and current_mtime <= self._repo_context_cache[0]:
            return self._repo_context_cache[1]

        ctx = build_repo_context(self.root, max_depth=max_depth, max_files=max_files)
        self._repo_context_cache = (current_mtime, ctx)
        return ctx

    def read_file(self, path: str) -> str:
        """Read a file with caching. Returns content or raises FileNotFoundError."""
        safe_path = self._safe_path(path)
        if safe_path is None:
            raise FileNotFoundError(f"Path outside repo: {path}")
        if not safe_path.exists() or not safe_path.is_file():
            raise FileNotFoundError(f"File not found: {path}")

        rel = self._rel(safe_path)
        mtime = safe_path.stat().st_mtime

        if rel in self._file_cache:
            cached_mtime, cached_content = self._file_cache[rel]
            if cached_mtime >= mtime:
                self._accessed_files.add(rel)
                return cached_content

        content = safe_path.read_text(encoding="utf-8")
        self._file_cache[rel] = (mtime, content)
        self._accessed_files.add(rel)
        return content

    def file_exists(self, path: str) -> bool:
        safe_path = self._safe_path(path)
        return safe_path is not None and safe_path.exists() and safe_path.is_file()

    def get_accessed_files(self) -> list[str]:
        return sorted(self._accessed_files)

    def get_repo_summary(self, max_depth: int = 3, max_files: int = 120) -> str:
        """Get compressed repo context, respecting token budget."""
        ctx = self.get_repo_context(max_depth, max_files)
        full = ctx.compressed(max_files)

        if self.estimate_tokens(full) <= self.max_tokens:
            return full

        # Compress by reducing file details
        tree_only = f"Repo root: {ctx.root}\n\nTree:\n{ctx.tree}"
        if self.estimate_tokens(tree_only) <= self.max_tokens:
            return tree_only

        # Still too long, truncate tree
        truncated_tree = ctx.tree[:8000] + "\n... (truncated)"
        return f"Repo root: {ctx.root}\n\nTree:\n{truncated_tree}"

    @staticmethod
    def estimate_tokens(text: str) -> int:
        """Rough token estimation. Falls back to character count if tiktoken unavailable."""
        try:
            import tiktoken
            enc = tiktoken.get_encoding("cl100k_base")
            return len(enc.encode(text))
        except Exception:
            # Rough fallback: ~3-4 chars per token on average
            return len(text) // 3

    def _get_max_mtime(self) -> float:
        """Get the most recent mtime in the repo to detect changes."""
        max_mtime = 0.0
        for current, dirs, files in os.walk(self.root):
            dirs[:] = [name for name in dirs if name not in DEFAULT_IGNORE_DIRS]
            for name in files:
                path = Path(current) / name
                if _ignored(path):
                    continue
                try:
                    mtime = path.stat().st_mtime
                    if mtime > max_mtime:
                        max_mtime = mtime
                except OSError:
                    pass
        return max_mtime

    def _safe_path(self, raw_path: str) -> Path | None:
        if not raw_path:
            return None
        path = (self.root / raw_path).resolve()
        try:
            path.relative_to(self.root)
        except ValueError:
            return None
        return path

    def _rel(self, path: Path) -> str:
        return path.relative_to(self.root).as_posix()


def build_repo_context(root: str | Path, max_depth: int = 3, max_files: int = 120) -> RepoContext:
    root_path = Path(root).resolve()
    files = _collect_files(root_path, max_depth=max_depth, max_files=max_files)
    return RepoContext(
        root=root_path,
        tree=render_tree(root_path, max_depth=max_depth, max_entries=max_files),
        files=[summarize_file(path, root_path) for path in files],
    )


def render_tree(root: str | Path, max_depth: int = 3, max_entries: int = 120) -> str:
    root_path = Path(root).resolve()
    lines = [root_path.name + "/"]
    count = 0

    def walk(path: Path, depth: int, prefix: str) -> None:
        nonlocal count
        if depth > max_depth or count >= max_entries:
            return
        entries = sorted(
            [entry for entry in path.iterdir() if not _ignored(entry)],
            key=lambda entry: (entry.is_file(), entry.name.lower()),
        )
        for index, entry in enumerate(entries):
            if count >= max_entries:
                lines.append(prefix + "... more")
                return
            count += 1
            connector = "`-- " if index == len(entries) - 1 else "|-- "
            lines.append(prefix + connector + entry.name + ("/" if entry.is_dir() else ""))
            if entry.is_dir():
                extension = "    " if index == len(entries) - 1 else "|   "
                walk(entry, depth + 1, prefix + extension)

    walk(root_path, 1, "")
    return "\n".join(lines)


def summarize_file(path: Path, root: Path) -> FileContext:
    relative = _relative(path, root)
    try:
        data = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return FileContext(relative, 0, path.stat().st_size, "binary file")

    stripped = data.strip()
    lines = data.count("\n") + (1 if data else 0)
    summary = "empty file"
    if stripped:
        first_line = next((line.strip() for line in data.splitlines() if line.strip()), "")
        defs = [
            line.strip()
            for line in data.splitlines()
            if line.lstrip().startswith(("def ", "class ", "function ", "const ", "export "))
        ][:3]
        summary = "; ".join(defs) if defs else first_line[:100]
    return FileContext(relative, lines, len(data.encode("utf-8")), summary)


def _collect_files(root: Path, max_depth: int, max_files: int) -> list[Path]:
    selected: list[Path] = []
    for current, dirs, files in os.walk(root):
        current_path = Path(current)
        rel_depth = len(current_path.relative_to(root).parts)
        dirs[:] = sorted([name for name in dirs if name not in DEFAULT_IGNORE_DIRS])
        if rel_depth >= max_depth:
            dirs[:] = []
        for name in sorted(files):
            path = current_path / name
            if _ignored(path):
                continue
            selected.append(path)
            if len(selected) >= max_files:
                return selected
    return selected


def _ignored(path: Path) -> bool:
    return path.name in DEFAULT_IGNORE_DIRS or path.name.endswith((".pyc", ".pyo"))


def _relative(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()
