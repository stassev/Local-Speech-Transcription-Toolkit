#!/usr/bin/env python3
# Copyright (C) 2026 Svetlin Tassev. SPDX-License-Identifier: GPL-3.0.
"""Run a command while monitoring its Linux /proc process-tree memory footprint.

The reported total is PSS (proportional set size) across descendants when the
kernel exposes /proc/PID/smaps_rollup; RSS is used per-process as a fallback.
This is a polling watchdog, not a kernel-enforced cgroup limit.
"""
from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import NamedTuple


class MemorySample(NamedTuple):
    bytes_used: int
    method: str
    process_count: int


def process_table(proc_root: Path = Path("/proc")) -> dict[int, int]:
    """Return PID -> PPID, handling spaces/parentheses in /proc/PID/stat comm."""
    table: dict[int, int] = {}
    try:
        entries = list(proc_root.iterdir())
    except OSError:
        return table
    for entry in entries:
        if not entry.name.isdigit():
            continue
        try:
            raw = (entry / "stat").read_text(encoding="ascii")
            # comm is parenthesized and can itself contain parentheses; split
            # after its final ')' so fields start at state (field 3).
            tail = raw[raw.rfind(")") + 1 :].split()
            if len(tail) >= 2:
                table[int(entry.name)] = int(tail[1])  # ppid is field 4
        except (OSError, ValueError, IndexError):
            continue
    return table


def descendants(root_pid: int, table: dict[int, int]) -> list[int]:
    children: dict[int, list[int]] = {}
    for pid, ppid in table.items():
        children.setdefault(ppid, []).append(pid)
    result: list[int] = []
    pending = [root_pid]
    seen: set[int] = set()
    while pending:
        pid = pending.pop()
        if pid in seen:
            continue
        seen.add(pid)
        if pid in table or pid == root_pid:
            result.append(pid)
            pending.extend(children.get(pid, ()))
    return result


def _pid_memory(pid: int) -> tuple[int, str] | None:
    """Read PSS (preferred) or RSS for a live PID; return bytes and method."""
    try:
        with (Path("/proc") / str(pid) / "smaps_rollup").open(
            "r", encoding="ascii", errors="replace"
        ) as stream:
            for line in stream:
                if line.startswith("Pss:"):
                    return int(line.split()[1]) * 1024, "PSS"
    except (OSError, ValueError, IndexError):
        pass
    try:
        for line in (Path("/proc") / str(pid) / "status").read_text(
            encoding="ascii", errors="replace"
        ).splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024, "RSS-fallback"
    except (OSError, ValueError, IndexError):
        pass
    return None


def sample_tree(root_pid: int) -> MemorySample:
    table = process_table()
    pids = descendants(root_pid, table)
    total = 0
    pss_count = 0
    rss_count = 0
    for pid in pids:
        measured = _pid_memory(pid)
        if measured is None:
            continue
        size, method = measured
        total += size
        if method == "PSS":
            pss_count += 1
        else:
            rss_count += 1
    method = "PSS" if rss_count == 0 else ("PSS+RSS-fallback" if pss_count else "RSS-fallback")
    return MemorySample(total, method, len(pids))


def gb(value: int) -> str:
    return f"{value / 1_000_000_000:.2f} GB"


def _process_group_exists(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def stop_process_group(proc: subprocess.Popen, grace_seconds: float = 3.0) -> None:
    """Terminate the whole child group, force-killing survivors after a grace period."""
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    deadline = time.monotonic() + grace_seconds
    while _process_group_exists(proc.pid) and time.monotonic() < deadline:
        time.sleep(0.1)
    # The launcher can exit before a model worker, so check the group rather
    # than only proc.poll(); otherwise a surviving descendant could be orphaned.
    if _process_group_exists(proc.pid):
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    try:
        proc.wait(timeout=2)
    except subprocess.TimeoutExpired:
        pass


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit-gb", type=float, required=True,
                        help="Maximum process-tree memory in decimal GB (1 GB = 1,000,000,000 bytes)")
    parser.add_argument("--interval", type=float, default=0.5,
                        help="Memory polling interval in seconds (default: 0.5)")
    parser.add_argument("command", nargs=argparse.REMAINDER,
                        help="Command to execute after --")
    args = parser.parse_args(argv)
    command = list(args.command)
    if command and command[0] == "--":
        command.pop(0)
    if not command:
        parser.error("a command is required after --")
    if not (args.limit_gb > 0):
        parser.error("--limit-gb must be > 0")
    if not (args.interval >= 0.05):
        parser.error("--interval must be >= 0.05 seconds")

    limit_bytes = int(args.limit_gb * 1_000_000_000)
    print(f"Memory guard: limit {args.limit_gb:g} GB; poll {args.interval:g}s; PSS preferred", flush=True)
    print("Command: " + " ".join(command), flush=True)
    proc = subprocess.Popen(command, start_new_session=True)
    peak = 0
    peak_method = "PSS"
    try:
        while proc.poll() is None:
            sample = sample_tree(proc.pid)
            if sample.bytes_used > peak:
                peak = sample.bytes_used
                peak_method = sample.method
            if sample.bytes_used >= limit_bytes:
                print(
                    f"RAM_LIMIT_EXCEEDED: process tree reached {gb(sample.bytes_used)} "
                    f"({sample.method}, {sample.process_count} processes), limit {args.limit_gb:g} GB; "
                    "sending SIGTERM then SIGKILL if needed.",
                    file=sys.stderr, flush=True,
                )
                stop_process_group(proc)
                # 137 is reserved here to identify watchdog termination distinctly.
                print(f"Memory guard: terminated process tree; peak observed {gb(peak)} ({peak_method})", flush=True)
                return 137
            time.sleep(args.interval)
        status = proc.wait()
        print(f"Memory guard: process exited {status}; peak observed {gb(peak)} ({peak_method})", flush=True)
        return status if status >= 0 else min(255, 128 + -status)
    except KeyboardInterrupt:
        stop_process_group(proc)
        return 130
    finally:
        if proc.poll() is None:
            stop_process_group(proc)


if __name__ == "__main__":
    raise SystemExit(main())
