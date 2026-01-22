#!/usr/bin/env python3
"""
MediSync Demo Runner
Run: python3 demo/run_all.py [demo_name]
"""

import sys
import os
import subprocess

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

DEMOS = {
    "demo": ("conversation.py", "Full Interactive Demo (All Features)"),
    "test": ("test_suite.py", "Full Test Suite (16 tests)"),
    "verify": ("quick_verify.py", "Quick Verification (30 sec)"),
}


def print_menu():
    print("\n" + "=" * 60)
    print("   MEDISYNC DEMO SUITE")
    print("   Qdrant Convolve 4.0 Pan-IIT Hackathon")
    print("=" * 60)
    print("\nAvailable demos:\n")

    for key, (filename, description) in DEMOS.items():
        print(f"  {key:12} - {description}")

    print("\nUsage:")
    print("  python3 demo/run_all.py demo      # Main demo (recommended)")
    print("  python3 demo/run_all.py verify    # Quick health check")
    print("  python3 demo/run_all.py test      # Full test suite")
    print()


def run_demo(demo_key: str):
    if demo_key not in DEMOS:
        print(f"[ERROR] Unknown demo: {demo_key}")
        print_menu()
        return False

    filename, description = DEMOS[demo_key]
    demo_path = os.path.join(os.path.dirname(__file__), filename)

    if not os.path.exists(demo_path):
        print(f"[ERROR] Demo file not found: {demo_path}")
        return False

    print(f"\n{'=' * 60}")
    print(f"  Running: {description}")
    print(f"  File: {filename}")
    print("=" * 60 + "\n")

    result = subprocess.run([sys.executable, demo_path], cwd=os.path.dirname(demo_path))
    return result.returncode == 0


def main():
    if len(sys.argv) < 2:
        print_menu()
        return

    arg = sys.argv[1].lower()

    if arg in DEMOS:
        run_demo(arg)
    elif arg in ["help", "-h", "--help"]:
        print_menu()
    else:
        print(f"[ERROR] Unknown option: {arg}")
        print_menu()


if __name__ == "__main__":
    main()
