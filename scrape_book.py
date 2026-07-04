"""
Scrape a book's Unicode text from the Dawat-e-Islami online reader
(https://www.dawateislami.net/bookslibrary/...) into local page files.

The reader serves real Urdu Unicode text (the PDFs use non-Unicode fonts and
extract as garbage), one topic/bab per web page, inside <div id="book__content__main">.

Usage:
    python scrape_book.py bahar-e-shariat-jild-1 ../data/bahar_e_shariat/jild_1

Output: one .txt per web page (first line = topic heading, rest = body text)
plus meta.json. Already-downloaded pages are skipped, so it is resume-able.
"""
import json
import re
import sys
import time

# Windows console defaults to cp1252, which cannot print Urdu.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from html import unescape
from pathlib import Path

import requests

BASE = "https://www.dawateislami.net/bookslibrary/ur"
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; AI-Mufti-ingest; personal scholarly use)"}
DELAY_SECONDS = 1.5

_DIV_RE = re.compile(r"<(/?)div\b")
_TAG_RE = re.compile(r"<[^>]+>")
_HEADING_RE = re.compile(r"<h[1-4][^>]*>(.*?)</h[1-4]>", re.S)


def _extract_content(raw: str) -> tuple[str, str]:
    """Return (heading, text) from the book__content__main div (balanced-div parse)."""
    start = raw.index("book__content__main")
    start = raw.index(">", start) + 1
    depth, i = 1, start
    while depth > 0:
        m = _DIV_RE.search(raw, i)
        if not m:
            break
        depth += -1 if m.group(1) else 1
        i = m.end()
    body = raw[start:i]

    hm = _HEADING_RE.search(body)
    heading = ""
    if hm:
        heading = unescape(_TAG_RE.sub(" ", hm.group(1)))
        heading = re.sub(r"\s+", " ", heading).strip()

    # Keep paragraph breaks: block-level closers become newlines before stripping tags.
    body = re.sub(r"</(p|div|h[1-6]|li|tr)>", "\n", body)
    body = re.sub(r"<br\s*/?>", "\n", body)
    text = unescape(_TAG_RE.sub(" ", body))
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" ?\n ?", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return heading, text


def _last_page(raw: str, slug: str) -> int:
    # Pagination is rendered as JS calls: gotToReadWithFilters("ur","<slug>",...,"page-71",...)
    nums = [int(n) for n in re.findall(r"page-(\d+)", raw)]
    return max(nums) if nums else 1


def main(slug: str, out_dir: str):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers.update(HEADERS)

    first = session.get(f"{BASE}/{slug}/page-1", timeout=60)
    first.raise_for_status()
    total = _last_page(first.text, slug)
    print(f"{slug}: {total} web pages")

    meta = {"slug": slug, "total_pages": total, "pages": {}}
    meta_path = out / "meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))

    for page in range(1, total + 1):
        txt_path = out / f"page_{page:03d}.txt"
        if txt_path.exists() and txt_path.stat().st_size > 0:
            continue
        if page == 1:
            raw = first.text
        else:
            time.sleep(DELAY_SECONDS)
            resp = session.get(f"{BASE}/{slug}/page-{page}", timeout=60)
            resp.raise_for_status()
            raw = resp.text
        try:
            heading, text = _extract_content(raw)
        except ValueError:
            print(f"  ! page {page}: content div not found, skipped")
            continue
        txt_path.write_text(heading + "\n" + text, encoding="utf-8")
        meta["pages"][str(page)] = {"heading": heading, "chars": len(text)}
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"  [{page}/{total}] {heading[:60]} ({len(text)} chars)")

    print("Done.")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit("usage: python scrape_book.py <book-slug> <output-dir>")
    main(sys.argv[1], sys.argv[2])
