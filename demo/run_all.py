#!/usr/bin/env python3
"""
MediSync Demo Runner
Run all demos in sequence or select specific ones.
Run: python3 demo/run_all.py [demo_name]
"""

import sys
import os
import subprocess

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

DEMOS = {
    "test": ("test_suite.py", "Full Test Suite (16 tests)"),
    "conversation": ("conversation.py", "Interactive Clinical Conversation"),
    "evidence": ("evidence_graph.py", "Evidence Graph Visualization"),
    "hybrid": ("hybrid_search_demo.py", "Hybrid Search Demo"),
    "discovery": ("discovery_demo.py", "Discovery API Demo"),
    "insights": ("global_insights_demo.py", "Global Insights & K-Anonymity"),
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
    print("  python3 demo/run_all.py <demo_name>")
    print("  python3 demo/run_all.py all          # Run all demos")
    print("  python3 demo/run_all.py              # Show this menu")
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


def run_all():
    """Run all demos in order"""
    order = ["verify", "test", "hybrid", "discovery", "insights", "evidence"]

    print("\n" + "=" * 60)
    print("   RUNNING ALL DEMOS")
    print("=" * 60)

    results = {}
    for demo in order:
        print(f"\n>>> Starting {demo}...")
        try:
            success = run_demo(demo)
            results[demo] = "PASS" if success else "FAIL"
        except Exception as e:
            results[demo] = f"ERROR: {e}"

        input("\nPress Enter to continue to next demo...")

    # Summary
    print("\n" + "=" * 60)
    print("   DEMO SUMMARY")
    print("=" * 60)
    for demo, status in results.items():
        icon = "[PASS]" if status == "PASS" else "[FAIL]"
        print(f"  {icon} {demo}: {status}")


def main():
    if len(sys.argv) < 2:
        print_menu()
        return

    arg = sys.argv[1].lower()

    if arg == "all":
        run_all()
    elif arg in DEMOS:
        run_demo(arg)
    elif arg in ["help", "-h", "--help"]:
        print_menu()
    else:
        print(f"[ERROR] Unknown option: {arg}")
        print_menu()


if __name__ == "__main__":
    main()
