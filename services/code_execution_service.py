from __future__ import annotations

import asyncio
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path

from core.config import (
    CODE_ALLOWED_ROLE_IDS,
    CODE_ALLOWED_USER_IDS,
    CODE_EXECUTION_TIMEOUT_SECONDS,
    CODE_MAX_OUTPUT_CHARS,
    CODE_SANDBOX_MODE,
    CODE_WORKSPACE_ROOT,
)
from core.logging_config import get_logger
from database.execution_repository import add_code_run, get_code_run

logger = get_logger(__name__)

DANGEROUS_PATTERNS = tuple(p.lower() for p in (
    "os.system(",
    "subprocess.",
    "shutil.rmtree(",
    "Path('..')",
    'Path("..")',
    "../",
    "..\\",
    "import ctypes",
    "import socket",
    "import winreg",
    "import requests",
    "import pickle",
    "pickle.loads(",
    "pickle.load(",
    "open('.env'",
    'open(".env"',
    "eval(",
    "exec(",
    "__import__",
    "compile(",
    "globals()",
    "open(os.",
    "open(path",
))


class CodeExecutionService:
    def __init__(self, performance_tracker=None) -> None:
        self.performance_tracker = performance_tracker
        self.workspace_root = Path(CODE_WORKSPACE_ROOT)
        self.sandbox_mode = CODE_SANDBOX_MODE
        self.execution_timeout_seconds = CODE_EXECUTION_TIMEOUT_SECONDS
        self.max_output_chars = CODE_MAX_OUTPUT_CHARS
        self.allowed_user_ids = set(CODE_ALLOWED_USER_IDS)
        self.allowed_role_ids = set(CODE_ALLOWED_ROLE_IDS)
        self.workspace_root.mkdir(parents=True, exist_ok=True)

    def initialize_workspace(self) -> None:
        self.workspace_root.mkdir(parents=True, exist_ok=True)

    def user_is_allowed(self, member) -> bool:
        user_id = getattr(member, "id", None)
        if user_id in self.allowed_user_ids:
            return True

        guild_permissions = getattr(member, "guild_permissions", None)
        if guild_permissions and guild_permissions.administrator:
            return True

        roles = getattr(member, "roles", [])
        for role in roles:
            if getattr(role, "id", None) in self.allowed_role_ids:
                return True

        return False

    def resolve_workspace_path(self, filename: str) -> Path:
        cleaned = filename.strip().replace("\\", "/")
        if not cleaned or cleaned.endswith("/"):
            raise ValueError("Provide a valid filename.")

        candidate = (self.workspace_root / cleaned).resolve()
        workspace_root = self.workspace_root.resolve()
        if workspace_root != candidate and workspace_root not in candidate.parents:
            raise ValueError("Path traversal outside the workspace is not allowed.")

        return candidate

    def list_files(self) -> list[str]:
        self.initialize_workspace()
        files = []
        for path in sorted(self.workspace_root.rglob("*")):
            if path.is_file():
                files.append(str(path.relative_to(self.workspace_root)).replace("\\", "/"))
        return files

    def read_file(self, filename: str) -> str:
        path = self.resolve_workspace_path(filename)
        if not path.exists():
            raise FileNotFoundError(f"`{filename}` does not exist in the workspace.")
        return path.read_text(encoding="utf-8")

    def create_file(self, filename: str, content: str) -> str:
        path = self.resolve_workspace_path(filename)
        if path.exists():
            raise FileExistsError(f"`{filename}` already exists. Use edit instead.")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return str(path.relative_to(self.workspace_root)).replace("\\", "/")

    def edit_file(self, filename: str, content: str) -> str:
        path = self.resolve_workspace_path(filename)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return str(path.relative_to(self.workspace_root)).replace("\\", "/")

    def delete_file(self, filename: str) -> str:
        path = self.resolve_workspace_path(filename)
        if not path.exists():
            raise FileNotFoundError(f"`{filename}` does not exist in the workspace.")
        path.unlink()
        return str(path.relative_to(self.workspace_root)).replace("\\", "/")

    def requires_dangerous_confirmation(self, content: str) -> bool:
        lowered = content.lower()
        return any(pattern in lowered for pattern in DANGEROUS_PATTERNS)

    async def run_file(self, filename: str, *, user_id: str, channel_id: str, allow_dangerous: bool = False):
        started_at = time.perf_counter()
        path = self.resolve_workspace_path(filename)
        if not path.exists():
            raise FileNotFoundError(f"`{filename}` does not exist in the workspace.")

        if path.suffix.lower() != ".py":
            raise RuntimeError("Only Python execution is supported right now.")

        source_text = path.read_text(encoding="utf-8")
        if self.requires_dangerous_confirmation(source_text) and not allow_dangerous:
            raise RuntimeError(
                "This file includes blocked or risky patterns. Re-run with `--allow-dangerous` if you really want to execute it."
            )

        command = [sys.executable, str(path)]
        env = {
            "PYTHONIOENCODING": "utf-8",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PATH": os.environ.get("PATH", ""),
            "SYSTEMROOT": os.environ.get("SYSTEMROOT", ""),
            "WINDIR": os.environ.get("WINDIR", ""),
        }

        try:
            completed = await asyncio.to_thread(
                subprocess.run,
                command,
                cwd=str(path.parent),
                capture_output=True,
                text=True,
                timeout=self.execution_timeout_seconds,
                env=env,
                shell=False,
            )
            stdout_text = completed.stdout or ""
            stderr_text = completed.stderr or ""
            exit_code = completed.returncode
        except subprocess.TimeoutExpired as exc:
            stdout_text = exc.stdout or ""
            stderr_text = (exc.stderr or "") + f"\nExecution timed out after {self.execution_timeout_seconds} seconds."
            exit_code = -1
        except Exception as exc:
            stdout_text = ""
            stderr_text = f"Execution failed: {exc}"
            exit_code = -1

        duration_ms = (time.perf_counter() - started_at) * 1000
        stdout_text = self._trim_output(stdout_text)
        stderr_text = self._trim_output(stderr_text)
        run_id = uuid.uuid4().hex[:12]

        await add_code_run(
            run_id=run_id,
            user_id=user_id,
            channel_id=channel_id,
            filename=str(path.relative_to(self.workspace_root)).replace("\\", "/"),
            command=" ".join(command),
            exit_code=exit_code,
            duration_ms=duration_ms,
            stdout_text=stdout_text,
            stderr_text=stderr_text,
        )

        logger.info(
            "[code_run] run_id=%s user_id=%s file=%s exit_code=%s duration_ms=%.2f sandbox=%s",
            run_id,
            user_id,
            filename,
            exit_code,
            duration_ms,
            self.sandbox_mode,
        )

        if self.performance_tracker is not None:
            self.performance_tracker.record_service_call("code_execution.run_file", duration_ms)

        return {
            "run_id": run_id,
            "filename": str(path.relative_to(self.workspace_root)).replace("\\", "/"),
            "exit_code": exit_code,
            "duration_ms": duration_ms,
            "stdout_text": stdout_text,
            "stderr_text": stderr_text,
            "sandbox_mode": self.sandbox_mode,
        }

    async def get_run_output(self, run_id: str):
        return await get_code_run(run_id)

    def _trim_output(self, value: str) -> str:
        value = value or ""
        if len(value) <= self.max_output_chars:
            return value
        return value[: self.max_output_chars] + "\n...[truncated]"
