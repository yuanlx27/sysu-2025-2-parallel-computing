#!/usr/bin/env python3

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parent


def run(command: list[str]) -> None:
    print("+", " ".join(os.fspath(part) for part in command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def detect_cuda_architecture() -> str | None:
    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        return None

    try:
        result = subprocess.run(
            [
                nvidia_smi,
                "--query-gpu=compute_cap",
                "--format=csv,noheader",
            ],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError:
        return None

    capabilities = {
        line.strip().replace(".", "")
        for line in result.stdout.splitlines()
        if line.strip()
    }
    return ";".join(sorted(capabilities)) or None


def compile_report() -> None:
    source = ROOT / "report" / "report.typ"
    run(["typst", "compile", source.relative_to(ROOT)])


def build_and_run(build_dir: Path, jobs: int | None, smoke: bool) -> None:
    build_dir.mkdir(parents=True, exist_ok=True)
    metrics = ROOT / "report" / "assets" / "metrics.csv"
    metrics.parent.mkdir(parents=True, exist_ok=True)

    configure = [
        "cmake",
        "-S",
        ".",
        "-B",
        os.fspath(build_dir),
        "-DCMAKE_BUILD_TYPE=Release",
    ]
    architecture = detect_cuda_architecture()
    if architecture:
        configure.append(f"-DCMAKE_CUDA_ARCHITECTURES={architecture}")

    run(configure)

    build = ["cmake", "--build", os.fspath(build_dir), "--config", "Release"]
    if jobs:
        build.extend(["--parallel", str(jobs)])
    else:
        build.append("--parallel")
    run(build)

    executable = build_dir / "lab10_conv"
    command = [
        os.fspath(executable),
        "--output",
        os.fspath(metrics),
    ]
    if smoke:
        command.append("--smoke")
    run(command)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build and run CUDA convolution experiments."
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="compile report/report.typ without building or running experiments",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="run a small benchmark matrix for quick environment validation",
    )
    parser.add_argument(
        "--build-dir",
        type=Path,
        default=Path("build"),
        help="CMake build directory (default: build)",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        help="maximum number of parallel build jobs",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if args.report:
            compile_report()
        else:
            build_and_run(args.build_dir, args.jobs, args.smoke)
    except (FileNotFoundError, subprocess.CalledProcessError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
