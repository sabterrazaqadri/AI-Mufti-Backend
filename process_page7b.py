"""One-shot OCR -> clean -> ingest for Ahkam-e-Shariat (deferred page-7 book).
242 pages. Resume-able (skips existing/blocked pages, checkpointed ingest).
Re-run to resume after any Neon drop / quota stall."""
import subprocess
import sys

BOOKS = [
    ("../data/ahkam_e_shariat", "Ahkam-e-Shariat"),
]

def run(args):
    r = subprocess.run([sys.executable, "-u", *args], cwd=".")
    if r.returncode != 0:
        print(f"STEP FAILED ({args[0]}) rc={r.returncode}; stopping. Re-run to resume.", flush=True)
        sys.exit(1)

for d, _ in BOOKS:
    print(f"\n=== OCR {d} ===", flush=True)
    run(["ocr_pdf_book.py", f"{d}/book.pdf", d])

print("\n=== CLEAN ===", flush=True)
run(["clean_alahazrat_ocr.py", *[d for d, _ in BOOKS]])

for d, name in BOOKS:
    print(f"\n=== INGEST {name} ===", flush=True)
    run(["ingest_book.py", d, "--book", name, "--jild", "1"])

print("\n=== AHKAM-E-SHARIAT DONE ===", flush=True)
