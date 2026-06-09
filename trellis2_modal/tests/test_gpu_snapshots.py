#!/usr/bin/env python3
"""
GPU Memory Snapshots Cold Start Test.

Result: GPU snapshots don't help (~143s vs ~146s) due to flex_gemm/Triton reinit.
Kept for future retesting if Modal improves support for Triton-based models.

Usage:
    pip install -r trellis2_modal/requirements-deploy.txt
    python -m trellis2_modal.tests.test_gpu_snapshots full
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


RESULTS_FILE = Path.home() / ".trellis2_gpu_snapshot_results.json"


def run_command(cmd: list[str], env: dict | None = None) -> tuple[int, str, str]:
    """Run a command and return exit code, stdout, stderr."""
    full_env = os.environ.copy()
    if env:
        full_env.update(env)

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        env=full_env,
    )
    return result.returncode, result.stdout, result.stderr


def deploy_service(with_snapshot: bool = False) -> bool:
    """Deploy the TRELLIS.2 service with or without GPU snapshots."""
    env = {"GPU_MEMORY_SNAPSHOT": "true" if with_snapshot else "false"}
    snapshot_str = "enabled" if with_snapshot else "disabled"

    print(f"\n{'='*60}")
    print(f"Deploying TRELLIS.2 with GPU Memory Snapshots {snapshot_str}")
    print(f"{'='*60}\n")

    # Force rebuild by adding a timestamp to ensure fresh snapshot
    code, stdout, stderr = run_command(
        ["modal", "deploy", "-m", "trellis2_modal.service.service"],
        env=env,
    )

    if code != 0:
        print(f"Deploy failed (exit {code}):")
        print(stderr)
        return False

    print("Deploy successful")
    return True


def stop_containers() -> bool:
    """Stop all running containers to force cold start."""
    print("\nStopping all containers...")

    code, stdout, stderr = run_command(
        ["modal", "app", "stop", "trellis2-3d", "--force"],
    )

    # It's okay if there are no containers to stop
    if code != 0 and "not found" not in stderr.lower():
        print(f"Warning: stop command returned {code}")

    # Wait a bit for containers to fully stop
    time.sleep(5)
    print("Containers stopped")
    return True


def measure_cold_start() -> tuple[float, dict]:
    """
    Measure cold start time by calling health_check on the GPU service.

    Returns:
        Tuple of (cold_start_seconds, health_check_result)
    """
    print("\nMeasuring cold start time...")
    print("(This may take 2-3 minutes for model loading)")

    start = time.perf_counter()

    # Use modal run to call the health_check method
    code, stdout, stderr = run_command(
        ["modal", "run", "-m", "trellis2_modal.service.service"],
        env=None,
    )

    elapsed = time.perf_counter() - start

    if code != 0:
        print(f"Cold start test failed (exit {code}):")
        print(stderr)
        return elapsed, {"error": stderr}

    # Parse the output to get health check result
    try:
        # The test_service entrypoint prints JSON result
        lines = stdout.strip().split("\n")
        for line in lines:
            if "Result:" in line:
                # Next line should be JSON
                idx = lines.index(line)
                if idx + 1 < len(lines):
                    result = json.loads(lines[idx + 1].strip())
                    return elapsed, result

        # Fallback: try to extract timing from output
        result = {"output": stdout}
        return elapsed, result
    except Exception as e:
        return elapsed, {"error": str(e), "output": stdout}


def run_baseline_test() -> dict:
    """Run baseline test without GPU snapshots."""
    print("\n" + "="*60)
    print("BASELINE TEST (No GPU Snapshots)")
    print("="*60)

    result = {
        "test": "baseline",
        "gpu_snapshot": False,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    # Deploy without snapshots
    if not deploy_service(with_snapshot=False):
        result["error"] = "Deploy failed"
        return result

    # Stop containers to force cold start
    stop_containers()

    # Measure cold start
    cold_start_time, health_result = measure_cold_start()

    result["cold_start_seconds"] = cold_start_time
    result["health_result"] = health_result

    print(f"\n{'='*60}")
    print(f"BASELINE RESULT: {cold_start_time:.1f}s cold start")
    print(f"{'='*60}")

    # Save result
    save_result("baseline", result)

    return result


def run_snapshot_test() -> dict:
    """Run test with GPU Memory Snapshots enabled."""
    print("\n" + "="*60)
    print("GPU SNAPSHOT TEST (Enabled)")
    print("="*60)

    result = {
        "test": "snapshot",
        "gpu_snapshot": True,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    # Deploy with snapshots enabled
    if not deploy_service(with_snapshot=True):
        result["error"] = "Deploy failed - check if GPU snapshots cause issues"
        return result

    print("\nFirst run creates the snapshot (may take longer)...")
    print("Waiting 60s for snapshot to be created...")
    time.sleep(60)

    # Stop containers to force cold start from snapshot
    stop_containers()

    # Measure cold start from snapshot
    cold_start_time, health_result = measure_cold_start()

    result["cold_start_seconds"] = cold_start_time
    result["health_result"] = health_result

    print(f"\n{'='*60}")
    print(f"SNAPSHOT RESULT: {cold_start_time:.1f}s cold start")
    print(f"{'='*60}")

    # Save result
    save_result("snapshot", result)

    return result


def save_result(test_name: str, result: dict) -> None:
    """Save test result to file."""
    results = {}
    if RESULTS_FILE.exists():
        try:
            results = json.loads(RESULTS_FILE.read_text())
        except json.JSONDecodeError:
            pass

    results[test_name] = result
    RESULTS_FILE.write_text(json.dumps(results, indent=2))
    print(f"Result saved to {RESULTS_FILE}")


def compare_results() -> None:
    """Compare baseline and snapshot test results."""
    print("\n" + "="*60)
    print("COMPARISON")
    print("="*60)

    if not RESULTS_FILE.exists():
        print("No results found. Run baseline and snapshot tests first.")
        return

    try:
        results = json.loads(RESULTS_FILE.read_text())
    except json.JSONDecodeError:
        print("Invalid results file.")
        return

    baseline = results.get("baseline", {})
    snapshot = results.get("snapshot", {})

    print("\nResults:")
    print("-" * 40)

    if baseline:
        baseline_time = baseline.get("cold_start_seconds", "N/A")
        print(f"Baseline (no snapshot): {baseline_time:.1f}s" if isinstance(baseline_time, (int, float)) else f"Baseline: {baseline_time}")
    else:
        print("Baseline: Not run yet")

    if snapshot:
        snapshot_time = snapshot.get("cold_start_seconds", "N/A")
        print(f"With GPU Snapshot:      {snapshot_time:.1f}s" if isinstance(snapshot_time, (int, float)) else f"Snapshot: {snapshot_time}")
    else:
        print("Snapshot: Not run yet")

    # Calculate improvement
    if baseline and snapshot:
        baseline_time = baseline.get("cold_start_seconds")
        snapshot_time = snapshot.get("cold_start_seconds")

        if isinstance(baseline_time, (int, float)) and isinstance(snapshot_time, (int, float)):
            improvement = baseline_time - snapshot_time
            percent = (improvement / baseline_time) * 100 if baseline_time > 0 else 0

            print("-" * 40)
            print(f"Improvement: {improvement:.1f}s ({percent:.1f}%)")

            if percent > 30:
                print("\n✅ GPU Memory Snapshots provide significant benefit!")
            elif percent > 10:
                print("\n⚡ GPU Memory Snapshots provide moderate benefit.")
            elif percent > 0:
                print("\n➡️ GPU Memory Snapshots provide minimal benefit.")
            else:
                print("\n❌ GPU Memory Snapshots don't help (or failed).")

    print()


def main():
    parser = argparse.ArgumentParser(
        description="Test GPU Memory Snapshots for TRELLIS.2 Modal service"
    )
    parser.add_argument(
        "command",
        choices=["baseline", "snapshot", "compare", "full"],
        help="Test command: baseline (no snapshots), snapshot (with), compare, full (all)",
    )

    args = parser.parse_args()

    if args.command == "baseline":
        run_baseline_test()
    elif args.command == "snapshot":
        run_snapshot_test()
    elif args.command == "compare":
        compare_results()
    elif args.command == "full":
        run_baseline_test()
        run_snapshot_test()
        compare_results()


if __name__ == "__main__":
    main()
