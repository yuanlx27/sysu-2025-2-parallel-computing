#!/usr/bin/env python3

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parent
BUILD_DIR = ROOT / "build"
ASSETS_DIR = ROOT / "report" / "assets"

HELLO_OUTPUT = ASSETS_DIR / "hello_world_output.txt"
MATRIX_METRICS = ASSETS_DIR / "matrix_transpose_metrics.csv"

HELLO_ARGS = ("2", "4", "4")
MATRIX_ARGS = ("--benchmark", "--repeat", "20")


def command_text(command: Iterable[os.PathLike[str] | str]) -> str:
    return " ".join(str(part) for part in command)


def run_checked(command: list[os.PathLike[str] | str]) -> None:
    print(f"$ {command_text(command)}", flush=True)
    try:
        subprocess.run([str(part) for part in command], cwd=ROOT, check=True)
    except FileNotFoundError as error:
        raise SystemExit(f"Error: command not found: {command[0]}") from error
    except subprocess.CalledProcessError as error:
        raise SystemExit(
            f"Error: command failed with exit code {error.returncode}: {command_text(command)}"
        ) from error


def executable_path(name: str) -> Path:
    executable_name = name + (".exe" if os.name == "nt" else "")
    candidates = (
        BUILD_DIR / executable_name,
        BUILD_DIR / "Release" / executable_name,
        BUILD_DIR / "Debug" / executable_name,
        BUILD_DIR / "RelWithDebInfo" / executable_name,
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def run_to_file(
    command: list[os.PathLike[str] | str],
    output_path: Path,
    *,
    merge_stderr: bool,
    replace_on_success: bool,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    target = output_path.with_suffix(output_path.suffix + ".tmp") if replace_on_success else output_path

    print(f"$ {command_text(command)} > {output_path}", flush=True)
    try:
        with target.open("w", encoding="utf-8") as output:
            subprocess.run(
                [str(part) for part in command],
                cwd=ROOT,
                stdout=output,
                stderr=subprocess.STDOUT if merge_stderr else None,
                check=True,
            )
    except FileNotFoundError as error:
        if replace_on_success and target.exists():
            target.unlink()
        raise SystemExit(f"Error: command not found: {command[0]}") from error
    except subprocess.CalledProcessError as error:
        if replace_on_success and target.exists():
            target.unlink()
        raise SystemExit(
            f"Error: command failed with exit code {error.returncode}: {command_text(command)}"
        ) from error

    if replace_on_success:
        target.replace(output_path)


def main() -> int:
    BUILD_DIR.mkdir(exist_ok=True)
    run_checked(["cmake", "-S", ROOT, "-B", BUILD_DIR, "-DCMAKE_BUILD_TYPE=Release"])
    run_checked(["cmake", "--build", BUILD_DIR, "--config", "Release"])

    ASSETS_DIR.mkdir(parents=True, exist_ok=True)

    hello_command = [executable_path("cuda_hello_world"), *HELLO_ARGS]
    run_to_file(hello_command, HELLO_OUTPUT, merge_stderr=True, replace_on_success=False)

    matrix_command = [executable_path("cuda_matrix_transpose"), *MATRIX_ARGS]
    run_to_file(matrix_command, MATRIX_METRICS, merge_stderr=False, replace_on_success=True)

    print(f"Generated {HELLO_OUTPUT.relative_to(ROOT)}")
    print(f"Generated {MATRIX_METRICS.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
