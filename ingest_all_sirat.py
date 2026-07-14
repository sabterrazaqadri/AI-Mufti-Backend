#!/usr/bin/env python3
"""
Sequential ingestion of all Sirat-ul-Jinan Jilds 1-10.
Each jild is completed fully before moving to the next.
Uses 6-key rotation: GEMINI_API_KEY + INGESTION_KEY_1 through 5.
"""
import subprocess
import sys
from pathlib import Path

def ingest_jild(jild_num):
    """Ingest a single jild and wait for completion."""
    print(f"\n{'='*80}")
    print(f"Starting Jild {jild_num} ingestion...")
    print(f"{'='*80}\n")

    cmd = [
        sys.executable, "-u", "ingest_book.py",
        f"../data/sirat_ul_jinan/jild_{jild_num}",
        "--book", "Sirat-ul-Jinan",
        "--jild", str(jild_num)
    ]

    result = subprocess.run(cmd, cwd=Path(__file__).parent)
    return result.returncode == 0

def main():
    print("\n" + "="*80)
    print("SIRAT-UL-JINAN SEQUENTIAL INGESTION (Jild 1→2→3...→10)")
    print("="*80)
    print("\nKey Rotation: GEMINI_API_KEY + INGESTION_KEY_1 through 5 (6 keys total)")
    print("Mode: Sequential (complete each jild before moving to next)\n")

    completed = []
    failed = []

    for jild in range(1, 11):
        success = ingest_jild(jild)
        if success:
            completed.append(jild)
            print(f"\n✅ Jild {jild} COMPLETED\n")
        else:
            failed.append(jild)
            print(f"\n⚠️  Jild {jild} stopped (quota likely exhausted)\n")

    # Summary
    print("\n" + "="*80)
    print("INGESTION SUMMARY")
    print("="*80)
    print(f"Completed Jilds: {completed}")
    print(f"Incomplete Jilds: {failed}")
    print(f"\nProgress: {len(completed)}/10 jilds")

    if failed:
        print(f"\n⏳ To resume from Jild {failed[0]}, run: python ingest_all_sirat.py")
        print("   (Script will skip completed jilds and continue from the first incomplete one)\n")

if __name__ == "__main__":
    main()
