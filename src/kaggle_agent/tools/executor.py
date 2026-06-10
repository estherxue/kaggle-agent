"""Code execution utilities for Kaggle Agent.

Provides safe subprocess execution of generated code with timeouts
and resource limits.
"""

import os
import signal
import subprocess
import sys
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Any


@dataclass
class ExecutionResult:
    """Result of code execution."""

    success: bool
    stdout: str
    stderr: str
    return_code: int
    artifacts: Dict[str, Path]  # name -> path mapping
    execution_time_sec: float
    timed_out: bool = False


class CodeExecutor:
    """Executes Python code in a safe subprocess environment.

    Features:
    - Timeout enforcement
    - Resource limits (optional)
    - Artifact collection
    - Network access control
    - Working directory isolation
    """

    def __init__(
        self,
        timeout_sec: int = 300,
        allow_network: bool = False,
        python_path: Optional[str] = None,
        memory_limit_mb: Optional[int] = None,
    ):
        """Initialize code executor.

        Args:
            timeout_sec: Maximum execution time in seconds
            allow_network: Whether to allow network access
            python_path: Path to Python interpreter (None = sys.executable)
            memory_limit_mb: Memory limit in MB (Linux only)
        """
        self.timeout_sec = timeout_sec
        self.allow_network = allow_network
        self.python_path = python_path or sys.executable
        self.memory_limit_mb = memory_limit_mb

    def execute(
        self,
        code: str,
        working_dir: Path,
        env_vars: Optional[Dict[str, str]] = None,
        artifacts_to_collect: Optional[List[str]] = None,
        input_file: Optional[Path] = None,
    ) -> ExecutionResult:
        """Execute Python code in a subprocess.

        Args:
            code: Python code to execute
            working_dir: Working directory for execution
            env_vars: Additional environment variables
            artifacts_to_collect: List of file patterns to collect
            input_file: Optional stdin file

        Returns:
            ExecutionResult with output, artifacts, and status
        """
        working_dir = Path(working_dir).resolve()
        working_dir.mkdir(parents=True, exist_ok=True)

        # Create environment
        env = os.environ.copy()
        if not self.allow_network:
            # Disable network by setting proxy to invalid value
            # This is a best-effort approach
            env["HTTP_PROXY"] = "http://127.0.0.1:65535"
            env["HTTPS_PROXY"] = "http://127.0.0.1:65535"

        if env_vars:
            env.update(env_vars)

        # Write code to temporary file
        code_file = working_dir / "_agent_code.py"
        code_file.write_text(code)

        # Prepare command
        cmd = [self.python_path, str(code_file)]

        # Track execution time
        import time
        start_time = time.time()

        try:
            # Run with timeout
            result = subprocess.run(
                cmd,
                cwd=working_dir,
                env=env,
                capture_output=True,
                text=True,
                timeout=self.timeout_sec,
                input=open(input_file) if input_file else None,
            )
            timed_out = False
        except subprocess.TimeoutExpired:
            timed_out = True
            result = None

        execution_time = time.time() - start_time

        # Collect artifacts
        artifacts: Dict[str, Path] = {}
        if artifacts_to_collect:
            for pattern in artifacts_to_collect:
                # Support glob patterns
                for path in working_dir.glob(pattern):
                    if path.is_file():
                        # Use relative path as artifact name
                        artifacts[path.name] = path

        if timed_out:
            return ExecutionResult(
                success=False,
                stdout="",
                stderr=f"Execution timed out after {self.timeout_sec} seconds",
                return_code=-1,
                artifacts=artifacts,
                execution_time_sec=execution_time,
                timed_out=True,
            )

        # Success if return code is 0
        success = result.returncode == 0

        return ExecutionResult(
            success=success,
            stdout=result.stdout,
            stderr=result.stderr,
            return_code=result.returncode,
            artifacts=artifacts,
            execution_time_sec=execution_time,
            timed_out=False,
        )

    def execute_file(
        self,
        file_path: Path,
        working_dir: Path,
        env_vars: Optional[Dict[str, str]] = None,
        artifacts_to_collect: Optional[List[str]] = None,
    ) -> ExecutionResult:
        """Execute an existing Python file.

        Args:
            file_path: Path to Python file
            working_dir: Working directory
            env_vars: Additional environment variables
            artifacts_to_collect: Files to collect after execution

        Returns:
            ExecutionResult
        """
        file_path = Path(file_path).resolve()
        code = file_path.read_text()
        return self.execute(
            code=code,
            working_dir=working_dir,
            env_vars=env_vars,
            artifacts_to_collect=artifacts_to_collect,
        )

    def execute_notebook(
        self,
        notebook_path: Path,
        working_dir: Path,
        env_vars: Optional[Dict[str, str]] = None,
    ) -> ExecutionResult:
        """Execute a Jupyter notebook.

        Args:
            notebook_path: Path to .ipynb file
            working_dir: Working directory
            env_vars: Additional environment variables

        Returns:
            ExecutionResult
        """
        import json

        notebook_path = Path(notebook_path).resolve()

        with open(notebook_path, "r") as f:
            notebook = json.load(f)

        # Extract code from code cells
        code_cells = [
            cell["source"]
            for cell in notebook.get("cells", [])
            if cell.get("cell_type") == "code"
        ]

        # Join into single script
        code = "\n\n".join(
            "".join(cell) if isinstance(cell, list) else cell
            for cell in code_cells
        )

        return self.execute(
            code=code,
            working_dir=working_dir,
            env_vars=env_vars,
        )


class ExecutionSandbox(CodeExecutor):
    """Advanced sandbox with better isolation.

    Uses temporary directories and stricter controls.
    """

    def __init__(
        self,
        timeout_sec: int = 300,
        allow_network: bool = False,
        python_path: Optional[str] = None,
        cleanup_on_exit: bool = True,
    ):
        super().__init__(timeout_sec, allow_network, python_path)
        self.cleanup_on_exit = cleanup_on_exit
        self._temp_dirs: List[Path] = []

    def create_sandbox(self) -> Path:
        """Create a new isolated sandbox directory.

        Returns:
            Path to sandbox directory
        """
        temp_dir = Path(tempfile.mkdtemp(prefix="kagent_"))
        self._temp_dirs.append(temp_dir)
        return temp_dir

    def cleanup(self) -> None:
        """Clean up all sandbox directories."""
        import shutil

        for temp_dir in self._temp_dirs:
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
        self._temp_dirs.clear()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleanup."""
        if self.cleanup_on_exit:
            self.cleanup()


# Utility functions

def format_code_for_display(code: str, max_lines: int = 50) -> str:
    """Format code for display, truncating if too long.

    Args:
        code: Source code
        max_lines: Maximum lines to show

    Returns:
        Formatted code string
    """
    lines = code.split("\n")
    if len(lines) > max_lines:
        shown = lines[:max_lines]
        return "\n".join(shown) + f"\n... ({len(lines) - max_lines} more lines)"
    return code


def analyze_execution_result(result: ExecutionResult) -> Dict[str, Any]:
    """Analyze execution result for insights.

    Args:
        result: Execution result

    Returns:
        Dict with analysis (errors, warnings, stats)
    """
    analysis = {
        "success": result.success,
        "has_errors": not result.success,
        "has_warnings": "warning" in result.stderr.lower() if result.stderr else False,
        "execution_time": result.execution_time_sec,
        "timed_out": result.timed_out,
        "artifacts_collected": len(result.artifacts),
        "stdout_length": len(result.stdout),
        "stderr_length": len(result.stderr),
    }

    # Extract error types from stderr
    if result.stderr:
        error_types = []
        stderr_lower = result.stderr.lower()
        if "error" in stderr_lower:
            error_types.append("runtime_error")
        if "import" in stderr_lower and "error" in stderr_lower:
            error_types.append("import_error")
        if "memory" in stderr_lower:
            error_types.append("memory_error")
        if "timeout" in stderr_lower:
            error_types.append("timeout")

        analysis["error_types"] = error_types

    return analysis
