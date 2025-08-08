"""
Ensure the repository root does not contain stray .log files.

This guards against scripts writing logs into the project root instead of logs/.
"""

from pathlib import Path


def test_no_log_files_in_repo_root():
    repo_root = Path(__file__).resolve().parent.parent
    root_log_files = sorted(p.name for p in repo_root.glob("*.log"))
    assert (
        len(root_log_files) == 0
    ), f"Unexpected .log files in repo root: {root_log_files} (logs should go under 'logs/')"

