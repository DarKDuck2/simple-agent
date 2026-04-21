from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from simple_agent.planner import PlanStep


DANGEROUS_COMMANDS = {
    "rm", "sudo", "chmod", "chown", "mkfs", "dd", "fdisk", "mount", "umount",
    "curl", "wget", "nc", "netcat", "telnet", "ssh", "scp", "sftp",
    "iptables", "ufw", "firewall-cmd",
    "kill", "killall", "pkill",
    "pip", "pip3", "npm", "yarn", "pnpm",  # package managers can be risky
}

SENSITIVE_FILE_PATTERNS = [
    r"\.env",
    r"\.env\.",
    r".*\.key$",
    r".*\.pem$",
    r"id_rsa",
    r"id_ed25519",
    r"\.ssh",
    r"\.aws",
    r"credentials",
    r"secret",
    r"token",
    r"password",
]


@dataclass
class PermissionManager:
    """Manage permissions for dangerous operations."""

    auto_approve: bool = False
    approved_commands: set[str] = field(default_factory=set)

    def needs_confirmation(self, step: PlanStep) -> bool:
        if self.auto_approve:
            return False

        if step.tool == "run_shell":
            command = str(step.params.get("command", ""))
            return self._is_dangerous_command(command)

        if step.tool in ("edit_file", "file_create", "file_delete"):
            path = str(step.params.get("path", ""))
            return self._is_sensitive_file(path)

        if step.tool == "git_commit":
            return True

        return False

    def _is_dangerous_command(self, command: str) -> bool:
        # Extract the main command from shell command
        # Handle cases like: rm -rf, sudo apt-get, python script.py
        parts = command.split()
        if not parts:
            return False

        # Get the base command name
        main_cmd = parts[0]
        # Handle paths like /usr/bin/rm
        main_cmd = main_cmd.split("/")[-1]

        if main_cmd in DANGEROUS_COMMANDS:
            return True

        # Check for rm-like patterns
        if re.search(r'\brm\b', command):
            return True

        # Check for destructive redirects
        if "> /" in command or ">/dev" in command:
            return True

        return False

    def _is_sensitive_file(self, path: str) -> bool:
        path_lower = path.lower()
        for pattern in SENSITIVE_FILE_PATTERNS:
            if re.search(pattern, path_lower):
                return True
        return False

    def confirm(self, action_desc: str) -> bool:
        """Interactive confirmation. Returns True if user approves."""
        print(f"\n[权限确认] {action_desc}")
        print("  允许执行？(y/n/a=全部允许)")
        try:
            response = input("> ").strip().lower()
            if response == "a":
                self.auto_approve = True
                return True
            return response in ("y", "yes", "")
        except (EOFError, KeyboardInterrupt):
            print("  已取消")
            return False
