"""
Lean REPL server interface.

Provides fast Lean 4 code execution using persistent REPL process.
"""

import json
import os
import queue
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# =============================================================================
# VERSIONS - Must match scripts/setup_mathlib.sh
# =============================================================================
MATHLIB_VERSION = "v4.25.2"
REPL_VERSION = "v4.25.0"
LEAN_TOOLCHAIN = "leanprover/lean4:v4.25.2"
# =============================================================================

DEFAULT_IMPORTS = "import Mathlib\nimport Aesop\nopen BigOperators Real Nat Topology"


def _get_lean_env() -> dict:
    """Get environment with elan PATH included."""
    env = os.environ.copy()
    elan_bin = Path.home() / ".elan" / "bin"
    if elan_bin.exists():
        env["PATH"] = f"{elan_bin}:{env.get('PATH', '')}"
    return env


@dataclass
class LeanResult:
    """Result of Lean code execution."""
    success: bool
    complete: bool  # Proof is complete (no goals, no sorry)
    output: str
    errors: list[str] = field(default_factory=list)
    goals: list[str] = field(default_factory=list)
    execution_time: float = 0.0

    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "complete": self.complete,
            "output": self.output,
            "errors": self.errors,
            "goals": self.goals,
            "execution_time": self.execution_time,
        }


class LeanServer:
    """
    Lean REPL server for fast proof checking.

    First check loads imports (~60s), subsequent checks are fast (~0.1s).
    """

    def __init__(
        self,
        project_path: Optional[str] = None,
        timeout: int = 30,
    ):
        _default = str(Path.home() / ".lean-bench")
        _resolved = project_path or os.environ.get("MATHLIB_PROJECT_PATH", _default)
        self.project_path = Path(_resolved)
        self.timeout = timeout
        self._process: Optional[subprocess.Popen] = None
        self._temp_dir: Optional[tempfile.TemporaryDirectory] = None
        self._lock = threading.Lock()
        self._imports_loaded = False
        self._env_id = 0
        self._line_queue: queue.Queue = queue.Queue()
        self._reader_thread: Optional[threading.Thread] = None

    def start(self) -> None:
        """Start the REPL process and preload imports."""
        self._ensure_project()
        self._start_process()
        self._load_imports()

    def stop(self) -> None:
        """Stop the REPL process."""
        if self._process:
            self._process.terminate()
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.kill()
            self._process = None
        if self._temp_dir:
            self._temp_dir.cleanup()
            self._temp_dir = None

    def _ensure_project(self) -> None:
        """Ensure we have a valid Lean project with repl."""
        if self.project_path and (self.project_path / "lakefile.lean").exists():
            return

        self._temp_dir = tempfile.TemporaryDirectory(prefix="lean_bench_")
        self.project_path = Path(self._temp_dir.name)

        # Create lakefile with repl dependency
        lakefile = self.project_path / "lakefile.lean"
        lakefile.write_text(f"""import Lake
open Lake DSL

package «lean_bench»

require mathlib from git
  "https://github.com/leanprover-community/mathlib4" @ "{MATHLIB_VERSION}"

require «repl» from git
  "https://github.com/leanprover-community/repl" @ "{REPL_VERSION}"

@[default_target]
lean_lib «LeanBench»
""")

        toolchain = self.project_path / "lean-toolchain"
        toolchain.write_text(f"{LEAN_TOOLCHAIN}\n")

        (self.project_path / "LeanBench").mkdir(exist_ok=True)
        (self.project_path / "LeanBench" / "Basic.lean").write_text("-- Lean Bench\n")

        print(f"[lean] Created project at {self.project_path}")

        # Download cache
        print("[lean] Downloading Mathlib cache (this may take a while)...")
        try:
            subprocess.run(
                ["lake", "exe", "cache", "get"],
                cwd=self.project_path,
                capture_output=True,
                timeout=600,
                env=_get_lean_env(),
            )
            print("[lean] Cache downloaded")
        except Exception as e:
            print(f"[lean] Cache warning: {e}")

    def _start_process(self) -> None:
        """Start the REPL subprocess and a background line-reader thread."""
        print(f"[lean] Starting REPL in {self.project_path}...")
        self._process = subprocess.Popen(
            ["lake", "exe", "repl"],
            cwd=self.project_path,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=_get_lean_env(),
        )
        self._imports_loaded = False

        # Drain and restart the queue
        self._line_queue = queue.Queue()
        self._reader_thread = threading.Thread(
            target=self._read_stdout_loop, daemon=True
        )
        self._reader_thread.start()

        time.sleep(0.5)
        if self._process.poll() is not None:
            stderr = self._process.stderr.read() if self._process.stderr else ""
            print(f"[lean] REPL failed to start: {stderr[:500]}")
        else:
            print("[lean] REPL process started")

    def _read_stdout_loop(self) -> None:
        """Background thread: read stdout lines into the queue."""
        for line in self._process.stdout:
            self._line_queue.put(line)
        self._line_queue.put(None)  # sentinel: process ended

    def _send_command(self, cmd: dict, timeout: Optional[int] = None) -> dict:
        """Send JSON command to REPL and get response."""
        if not self._process or self._process.poll() is not None:
            self._start_process()

        timeout = timeout or self.timeout

        with self._lock:
            try:
                cmd_json = json.dumps(cmd)
                self._process.stdin.write(cmd_json + "\n\n")
                self._process.stdin.flush()

                deadline = time.time() + timeout
                buffer = ""
                while True:
                    remaining = deadline - time.time()
                    if remaining <= 0:
                        return {"error": f"Timeout after {timeout}s"}

                    try:
                        line = self._line_queue.get(timeout=remaining)
                    except queue.Empty:
                        return {"error": f"Timeout after {timeout}s"}

                    if line is None:
                        return {"error": "REPL process ended"}

                    # Skip non-JSON preamble lines (lake warnings etc.)
                    if not buffer and not line.lstrip().startswith("{"):
                        continue

                    buffer += line
                    try:
                        return json.loads(buffer)
                    except json.JSONDecodeError:
                        continue

            except Exception as e:
                return {"error": str(e)}

    def _load_imports(self) -> bool:
        """Load Mathlib imports. Returns True on success."""
        if self._imports_loaded:
            return True

        print("[lean] Loading Mathlib imports (first time, ~60s, up to a few minutes on slow/NFS disks)...")
        resp = self._send_command({"cmd": DEFAULT_IMPORTS}, timeout=600)

        if "error" in resp:
            print(f"[lean] Failed to load imports: {resp['error']}")
            return False

        self._imports_loaded = True
        self._env_id = resp.get("env", 0)
        print("[lean] Imports loaded")
        return True

    def check_proof(self, statement: str, proof: str) -> LeanResult:
        """
        Check if a proof is valid for a given statement.

        Args:
            statement: Lean theorem statement (with 'sorry' placeholder)
            proof: Proof tactics to verify

        Returns:
            LeanResult with success/complete status and any errors/goals
        """
        start_time = time.time()

        if not self._load_imports():
            return LeanResult(
                success=False,
                complete=False,
                output="",
                errors=["Failed to load Mathlib imports"],
                execution_time=time.time() - start_time,
            )

        # Build code - replace sorry with proof
        proof = proof.strip()
        if proof.startswith("by "):
            proof = proof[3:].strip()

        indented = "\n".join("  " + line for line in proof.splitlines())
        if ":= by" in statement:
            code = statement.replace("sorry", proof)
        elif ":= sorry" in statement:
            code = statement.replace(":= sorry", f":= by\n{indented}")
        else:
            code = f"{statement} := by\n{indented}"

        # Send to REPL
        cmd = {"cmd": code, "env": self._env_id}
        resp = self._send_command(cmd)

        execution_time = time.time() - start_time
        # import pdb; pdb.set_trace()
        return self._parse_response(resp, execution_time)

    def _parse_response(self, resp: dict, execution_time: float) -> LeanResult:
        """Parse REPL response into LeanResult."""
        if "error" in resp:
            return LeanResult(
                success=False,
                complete=False,
                output=str(resp),
                errors=[resp["error"]],
                execution_time=execution_time,
            )

        messages = resp.get("messages", [])
        errors = [m["data"] for m in messages if m.get("severity") == "error"]

        # Check for remaining goals
        sorries = resp.get("sorries", [])
        goals = [s.get("goal", "") for s in sorries if s.get("goal")]

        success = len(errors) == 0
        complete = success and len(goals) == 0 and len(sorries) == 0

        return LeanResult(
            success=success,
            complete=complete,
            output=json.dumps(resp),
            errors=errors,
            goals=goals,
            execution_time=execution_time,
        )

    def run_tactic(self, statement: str, tactic: str) -> LeanResult:
        """
        Run a single tactic on a theorem statement.

        Returns remaining goals or errors.
        """
        return self.check_proof(statement, tactic)

    def __enter__(self) -> "LeanServer":
        self.start()
        return self

    def __exit__(self, *_) -> None:
        self.stop()


class LeanServerPool:
    """Pool of N parallel LeanServer instances for concurrent proof checking."""

    def __init__(self, n_workers: int = 4, **server_kwargs):
        self._servers = [LeanServer(**server_kwargs) for _ in range(n_workers)]
        self._queue: queue.Queue = queue.Queue()

    def start(self) -> None:
        print(f"[lean] Starting pool of {len(self._servers)} servers...")
        threads = [threading.Thread(target=s.start) for s in self._servers]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        for s in self._servers:
            self._queue.put(s)
        print(f"[lean] Pool ready")

    def stop(self) -> None:
        for s in self._servers:
            s.stop()

    def check_proof(self, statement: str, proof: str) -> LeanResult:
        server = self._queue.get()
        try:
            return server.check_proof(statement, proof)
        finally:
            self._queue.put(server)

    def __enter__(self) -> "LeanServerPool":
        self.start()
        return self

    def __exit__(self, *_) -> None:
        self.stop()
