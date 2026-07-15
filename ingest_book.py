"""
Ingest a scraped book (see scrape_book.py) into the RAG store with exact,
machine-generated references — the model must never invent jild/masla numbers.

Chunking: each web page is one bab/topic. Text splits at numbered masla
boundaries (مسئلہ ۱: / مسئلہ (۱):) and consecutive masail are packed into
~1500-char chunks; a chunk's reference records the bab and the masla range it
actually contains, e.g.:

    Bahar-e-Shariat, Jild 1, Hissa 2, پانی کا بیان, مسئلہ ۵-۸

Embeddings are batched (one API call per ~25 chunks) and progress is
checkpointed per page, so the script is resume-able after rate limits.

Usage:
    python ingest_book.py ../data/bahar_e_shariat/jild_1 --book "Bahar-e-Shariat" --jild 1
    python ingest_book.py ../data/bahar_e_shariat/jild_1 --jild 1 --dry-run
"""
import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Load .env before reading environment variables
from dotenv import load_dotenv
load_dotenv()

# ---- Key rotation for multi-key ingestion (ONLY INGESTION_KEY_1 through 4) ----
# STRICT RULE: main GEMINI_API_KEY is NEVER used
INGESTION_KEYS = []

# Add INGESTION_KEY_1 through INGESTION_KEY_4 only (skip 5, skip main key)
for i in range(1, 5):
    key = os.getenv(f"INGESTION_KEY_{i}", "").strip()
    if key:
        INGESTION_KEYS.append((f"INGESTION_KEY_{i}", key))

_current_key_index = 0

def get_current_key():
    global _current_key_index
    if _current_key_index >= len(INGESTION_KEYS):
        return None
    return INGESTION_KEYS[_current_key_index][1]

def next_key():
    global _current_key_index
    _current_key_index += 1
    if _current_key_index >= len(INGESTION_KEYS):
        return None

    key_name, key_value = INGESTION_KEYS[_current_key_index]
    import google.generativeai as genai
    genai.configure(api_key=key_value)
    print(f"[KEY ROTATION] Switched to {key_name} ({_current_key_index}/{len(INGESTION_KEYS)} keys)")
    return key_value

TARGET_CHUNK = 1500   # chars; masail are packed up to this size
MAX_CHUNK = 3000      # a single masla longer than this is split at sentence ends
OVERLAP = 200         # overlap when force-splitting oversized text
# Free tier allows ~30k embedding tokens/minute and Urdu tokenizes heavily
# (~2 chars/token), so cap each API call by characters and pace the calls —
# a 70k-char batch 429s permanently no matter how long we retry.
MAX_BATCH_CHARS = int(os.getenv("INGEST_BATCH_CHARS", "20000"))
BATCH_SLEEP = float(os.getenv("INGEST_BATCH_SLEEP", "25"))  # seconds between calls (stay under TPM)

URDU_DIGITS = str.maketrans("۰۱۲۳۴۵۶۷۸۹", "0123456789")
MASLA_RE = re.compile(r"مسئلہ\s*[\(（]?\s*([۰-۹]+)\s*[\)）]?\s*[:：]?")
# NOTE: hissa numbers are deliberately NOT in references — the online text has no
# reliable hissa boundary markers (every "حصہ N" match is a cross-reference or
# footnote), and a wrong hissa is worse than none. Jild + bab + masla is exact.

# One web page can span several babs (e.g. نماز مریض + سجدۂ تلاوت + نمازِ مسافر);
# bab titles appear as short standalone lines, so detect them and switch the
# heading mid-page — otherwise later babs get cited under the page's first bab.
BAB_LINE_RE = re.compile(r"(کا بیان|کے مسائل|کا طریقہ)\s*$")


def split_babs(page_heading: str, body: str):
    """Yield (bab_heading, section_text) for each bab section within a page."""
    sections, cur_head, cur_lines = [], page_heading, []
    for line in body.split("\n"):
        s = line.strip()
        if (
            6 < len(s) < 50
            and BAB_LINE_RE.search(s)
            and ":" not in s and "۔" not in s and "مسئلہ" not in s
        ):
            if cur_lines:
                sections.append((cur_head, "\n".join(cur_lines)))
                cur_lines = []
            cur_head = s
            continue
        cur_lines.append(line)
    if cur_lines:
        sections.append((cur_head, "\n".join(cur_lines)))
    return sections


def split_page(text: str):
    """Yield (masla_numbers, segment_text) — intro text has an empty numbers list."""
    starts = [m.start() for m in MASLA_RE.finditer(text)]
    if not starts:
        yield [], text
        return
    if starts[0] > 0:
        yield [], text[: starts[0]]
    for i, s in enumerate(starts):
        e = starts[i + 1] if i + 1 < len(starts) else len(text)
        seg = text[s:e]
        num = MASLA_RE.match(seg)
        yield [int(num.group(1).translate(URDU_DIGITS))] if num else [], seg


def _force_split(text: str):
    """Split oversized text at sentence ends (۔) near MAX_CHUNK, with overlap."""
    out, pos = [], 0
    while pos < len(text):
        end = min(pos + MAX_CHUNK, len(text))
        if end < len(text):
            cut = text.rfind("۔", pos + MAX_CHUNK // 2, end)
            if cut != -1:
                end = cut + 1
        out.append(text[pos:end])
        if end >= len(text):
            break
        pos = max(end - OVERLAP, pos + 1)
    return out


def build_chunks(pages_dir: Path, book: str, jild: int, book_tag: str):
    """Return list of dicts: {title, reference, content, tags}."""
    chunks = []
    for path in sorted(pages_dir.glob("page_*.txt")):
        raw = path.read_text(encoding="utf-8")
        page_heading, _, body = raw.partition("\n")
        # May be empty (many pages have no standalone bab title on the first
        # line). Do NOT fall back to a book name here — that leaks one book's
        # title into another book's citations (e.g. Qanoon rows once read
        # "…بہارِ شریعت"). Empty heading → the ref just uses the safha.
        page_heading = page_heading.strip()

        for heading, section in split_babs(page_heading, body.strip()):

            page_match = re.search(r"(\d+)", path.stem)
            page_num = int(page_match.group(1)) if page_match else None

            def ref(nums):
                parts = [book, f"Jild {jild}"]
                # The page heading defaults to "بہارِ شریعت" when the page has no
                # real bab line (e.g. Sirat-ul-Jinan tafseer pages) — that is a
                # placeholder, not a locator, so drop it and rely on the safha.
                if heading and heading != "بہارِ شریعت":
                    parts.append(heading)
                if page_num is not None:
                    parts.append(f"صفحہ {page_num}")
                if nums:
                    lo, hi = min(nums), max(nums)
                    parts.append(f"مسئلہ {lo}" if lo == hi else f"مسئلہ {lo}-{hi}")
                return ", ".join(parts)

            def emit(nums, text):
                text = text.strip()
                if len(text) < 40:  # skip stray fragments
                    return
                chunks.append({
                    "title": heading,
                    "reference": ref(nums),
                    "content": f"[{book}، جلد {jild} — {heading}]\n{text}",
                    "tags": [book_tag, f"jild-{jild}", str(path.name)],
                })

            buf_nums, buf = [], ""
            for nums, seg in split_page(section):
                if len(buf) + len(seg) > TARGET_CHUNK and buf:
                    emit(buf_nums, buf)
                    buf_nums, buf = [], ""
                if len(seg) > MAX_CHUNK:
                    if buf:
                        emit(buf_nums, buf)
                        buf_nums, buf = [], ""
                    for piece in _force_split(seg):
                        emit(nums, piece)
                    continue
                buf += ("\n" if buf else "") + seg
                buf_nums += nums
            if buf:
                emit(buf_nums, buf)
    return chunks


def main():
    global _current_key_index

    ap = argparse.ArgumentParser()
    ap.add_argument("pages_dir")
    ap.add_argument("--book", default="Bahar-e-Shariat")
    ap.add_argument("--jild", type=int, required=True)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    pages_dir = Path(args.pages_dir)
    book_tag = re.sub(r"[^a-z0-9]+", "-", args.book.lower()).strip("-")
    chunks = build_chunks(pages_dir, args.book, args.jild, book_tag)
    total_chars = sum(len(c["content"]) for c in chunks)
    print(f"{len(chunks)} chunks, {total_chars} chars, ~{total_chars // 4} tokens")

    if args.dry_run:
        for c in chunks[:5] + chunks[len(chunks) // 2 : len(chunks) // 2 + 5]:
            print("-", c["reference"], f"({len(c['content'])} chars)")
        return

    import database as db
    import rag
    import google.generativeai as genai

    db.init_db()
    rag.init_rag()

    # Initialize with first key (main GEMINI_API_KEY or first ingestion key)
    if INGESTION_KEYS:
        key_name, key_value = INGESTION_KEYS[0]
        genai.configure(api_key=key_value)
        print(f"[KEY ROTATION] Starting with {key_name} (1/{len(INGESTION_KEYS)} keys)")

    ckpt_path = pages_dir / "ingest_checkpoint.json"
    done = json.loads(ckpt_path.read_text(encoding="utf-8")) if ckpt_path.exists() else {}

    # Group chunks by source page (3rd tag) for per-page checkpointing.
    by_page = {}
    for c in chunks:
        by_page.setdefault(c["tags"][2], []).append(c)

    for page_name, page_chunks in by_page.items():
        if done.get(page_name):
            continue
        # Re-running a page after a crash: clear its rows first (no duplicates).
        # The jild tag is essential: page file names repeat across jilds, and
        # without it ingesting Jild 2's page_001 deletes Jild 1's page_001 rows.
        with db.get_cursor(commit=True) as cur:
            cur.execute(
                "DELETE FROM sources WHERE %s = ANY(tags) AND %s = ANY(tags) AND %s = ANY(tags);",
                (book_tag, f"jild-{args.jild}", page_name),
            )
        batches, cur_batch, cur_chars = [], [], 0
        for c in page_chunks:
            if cur_batch and cur_chars + len(c["content"]) > MAX_BATCH_CHARS:
                batches.append(cur_batch)
                cur_batch, cur_chars = [], 0
            cur_batch.append(c)
            cur_chars += len(c["content"])
        if cur_batch:
            batches.append(cur_batch)

        for batch in batches:
            for attempt in range(5):
                try:
                    vecs = rag.embed_batch([c["content"] for c in batch])
                    break
                except Exception as exc:
                    exc_str = str(exc)
                    # Check if 429 quota error — try next key instead of waiting
                    if "429" in exc_str or "You exceeded your current quota" in exc_str:
                        new_key = next_key()
                        if not new_key:
                            print(f"Stopped at {page_name} (all keys exhausted). Re-run later to resume.")
                            return
                        print(f"  {exc_str[:80]}... Trying next key.")
                        time.sleep(2)  # brief pause before retry with new key
                        continue
                    # Non-quota error: exponential backoff
                    wait = 30 * (attempt + 1)
                    print(f"  embed failed ({exc}); retry in {wait}s")
                    time.sleep(wait)
            else:
                print(f"Stopped at {page_name} (quota?). Re-run later to resume.")
                return
            for c, v in zip(batch, vecs):
                # The dead pooled handle is discarded on the next borrow, so a
                # retry after a dropped Neon connection gets a fresh one.
                for db_attempt in range(3):
                    try:
                        rag.add_source_with_vector(
                            title=c["title"], content=c["content"], vec=v,
                            reference=c["reference"], lang="ur", tags=c["tags"],
                        )
                        break
                    except Exception as exc:
                        print(f"  db insert failed ({exc}); retry in 10s")
                        time.sleep(10)
                else:
                    print(f"Stopped at {page_name} (db). Re-run later to resume.")
                    return
            time.sleep(BATCH_SLEEP)
        done[page_name] = len(page_chunks)
        ckpt_path.write_text(json.dumps(done, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"[{len(done)}/{len(by_page)}] {page_name}: {len(page_chunks)} chunks ingested")

    print("Done.")


if __name__ == "__main__":
    main()
