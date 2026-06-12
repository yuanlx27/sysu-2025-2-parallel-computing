#!/usr/bin/env python3

import argparse
import csv
import os
from pathlib import Path
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parent
BUILD_DIR = ROOT / "build"
EXECUTABLE = BUILD_DIR / "matmul"
REPORT_SOURCE = ROOT / "report" / "report.typ"
REPORT_OUTPUT = ROOT / "report" / "report.pdf"
METRICS_FILE = ROOT / "report" / "assets" / "metrics.csv"

SIZES = (128, 256, 512, 1024, 2048)
BLOCK_SIZES = (8, 16, 32)
KERNELS = ("naive", "tiled", "coarsened")
WARMUP = 2
REPEAT = 10

METRIC_FIELDS = (
    "m",
    "n",
    "k",
    "kernel",
    "block",
    "warmup",
    "repeat",
    "kernel_ms",
    "total_ms",
    "gflops",
    "max_sample_abs_error",
    "max_sample_rel_error",
    "verify",
)


def run_command(command: list[str], *, capture: bool = False) -> subprocess.CompletedProcess[str]:
    print("+", " ".join(command), flush=True)
    return subprocess.run(
        command,
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=capture,
    )


def build() -> None:
    run_command(
        [
            "cmake",
            "-S",
            str(ROOT),
            "-B",
            str(BUILD_DIR),
            "-DCMAKE_BUILD_TYPE=Release",
        ]
    )
    run_command(["cmake", "--build", str(BUILD_DIR), "--parallel"])


def parse_metric(output: str) -> dict[str, str]:
    reader = csv.DictReader(output.splitlines())
    rows = list(reader)
    if reader.fieldnames != list(METRIC_FIELDS):
        raise RuntimeError(
            "matmul returned unexpected CSV columns: "
            f"{reader.fieldnames!r}"
        )
    if len(rows) != 1:
        raise RuntimeError(f"matmul returned {len(rows)} data rows; expected exactly one")
    if rows[0]["verify"] != "PASS":
        raise RuntimeError(f"CUDA result verification failed: {rows[0]}")
    return rows[0]


def benchmark() -> None:
    METRICS_FILE.parent.mkdir(parents=True, exist_ok=True)
    combinations = [
        (size, block_size, kernel)
        for size in SIZES
        for block_size in BLOCK_SIZES
        for kernel in KERNELS
    ]
    rows: list[dict[str, str]] = []
    gpu_info_printed = False

    for index, (size, block_size, kernel) in enumerate(combinations, start=1):
        print(
            f"[{index:02d}/{len(combinations)}] "
            f"{size}x{size}x{size}, kernel={kernel}, "
            f"block={block_size}x{block_size}",
            flush=True,
        )
        command = [
            str(EXECUTABLE),
            str(size),
            str(size),
            str(size),
            "--kernel",
            kernel,
            "--block",
            str(block_size),
            "--warmup",
            str(WARMUP),
            "--repeat",
            str(REPEAT),
            "--csv",
        ]
        result = run_command(command, capture=True)
        if not gpu_info_printed and result.stderr.strip():
            print(result.stderr.strip())
            gpu_info_printed = True
        rows.append(parse_metric(result.stdout))

    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            dir=METRICS_FILE.parent,
            prefix=f".{METRICS_FILE.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            writer = csv.DictWriter(temporary, fieldnames=METRIC_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary_path, METRICS_FILE)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()

    print(f"Wrote {len(rows)} measurements to {METRICS_FILE}")


def compile_report() -> None:
    if not REPORT_SOURCE.is_file():
        raise FileNotFoundError(
            f"report source does not exist: {REPORT_SOURCE}\n"
            "Create report/report.typ before using --report."
        )
    REPORT_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    run_command(
        [
            "typst",
            "compile",
            str(REPORT_SOURCE),
            str(REPORT_OUTPUT),
        ]
    )
    print(f"Wrote report to {REPORT_OUTPUT}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build and benchmark the CUDA matrix multiplication experiment."
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="compile report/report.typ to report/report.pdf instead of benchmarking",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if args.report:
            compile_report()
        else:
            build()
            benchmark()
    except (FileNotFoundError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
