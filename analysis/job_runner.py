"""
Tiny background-job runner backing the dashboard's two action buttons ("Refresh market
data" and "Rebuild features & rescore"). Both are long-running, memory-heavy processes
(the scraper drives real Chrome instances; the feature rebuild processes the full price
panel), so they run as detached subprocesses rather than blocking a web request, and
only ONE of them may run at a time via a single shared lock file -- running the scraper
alongside the feature pipeline is exactly what caused an OOM kill earlier in this
project on this same 11GB VPS.

The lock self-heals: if the recorded PID is no longer alive, the lock is treated as
stale and cleared on the next check, rather than needing an explicit "job finished"
handshake from the subprocess.
"""
import json
import os
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
JOBS_DIR = DATA_DIR / "jobs"
JOBS_DIR.mkdir(exist_ok=True, parents=True)
LOCK_PATH = JOBS_DIR / "current_job.lock"


def _read_lock():
    if not LOCK_PATH.exists():
        return None
    try:
        return json.loads(LOCK_PATH.read_text())
    except Exception:
        return None


def current_job():
    """The currently-running job's info dict ({job_name, pid, started_at}), or None.

    Reaps the child with a non-blocking waitpid first: the launching process (this
    Streamlit server) is the subprocess's real parent, so an exited-but-unreaped child
    is a zombie that os.kill(pid, 0) would still report as "alive" -- waitpid is what
    actually collects its exit status and frees the PID. Falls back to a plain
    liveness probe for the rare case this lock was written by a process that's since
    restarted (pid is then someone else's, not our child)."""
    info = _read_lock()
    if info is None:
        return None
    pid = info["pid"]
    try:
        reaped_pid, _ = os.waitpid(pid, os.WNOHANG)
        if reaped_pid == pid:
            LOCK_PATH.unlink(missing_ok=True)
            return None
        return info
    except ChildProcessError:
        try:
            os.kill(pid, 0)
            return info
        except OSError:
            LOCK_PATH.unlink(missing_ok=True)
            return None


def start_job(job_name, command):
    """command: list of str, e.g. ["bash", "-c", "..."], run with cwd=repo root."""
    running = current_job()
    if running is not None:
        raise RuntimeError(f"Job '{running['job_name']}' is already running -- wait for it to "
                            f"finish before starting '{job_name}'.")
    log_path = JOBS_DIR / f"{job_name}.log"
    log_f = open(log_path, "w")
    proc = subprocess.Popen(command, stdout=log_f, stderr=subprocess.STDOUT, cwd=str(ROOT))
    LOCK_PATH.write_text(json.dumps({"job_name": job_name, "pid": proc.pid, "started_at": time.time()}))
    return proc.pid


def job_log_tail(job_name, n_lines=300):
    log_path = JOBS_DIR / f"{job_name}.log"
    if not log_path.exists():
        return ""
    lines = log_path.read_text(errors="replace").splitlines()
    return "\n".join(lines[-n_lines:])


def job_last_started(job_name):
    """Best-effort: the started_at time recorded the last time this specific job_name
    was launched, even if it's since finished (lock is shared across job types, so this
    reads the job's own log file's mtime as a proxy once the lock has moved on/cleared)."""
    log_path = JOBS_DIR / f"{job_name}.log"
    if not log_path.exists():
        return None
    return log_path.stat().st_mtime
