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
        page_heading = page_heading.strip() or "بہارِ شریعت"

        for heading, section in split_babs(page_heading, body.strip()):

            def ref(nums):
                parts = [book, f"Jild {jild}", heading]
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

    db.init_db()
    rag.init_rag()

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
        with db.get_cursor(commit=True) as cur:
            cur.execute(
                "DELETE FROM sources WHERE %s = ANY(tags) AND %s = ANY(tags);",
                (book_tag, page_name),
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
                    wait = 30 * (attempt + 1)
                    print(f"  embed failed ({exc}); retry in {wait}s")
                    time.sleep(wait)
            else:
                print(f"Stopped at {page_name} (quota?). Re-run later to resume.")
                return
            for c, v in zip(batch, vecs):
                rag.add_source_with_vector(
                    title=c["title"], content=c["content"], vec=v,
                    reference=c["reference"], lang="ur", tags=c["tags"],
                )
            time.sleep(BATCH_SLEEP)
        done[page_name] = len(page_chunks)
        ckpt_path.write_text(json.dumps(done, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"[{len(done)}/{len(by_page)}] {page_name}: {len(page_chunks)} chunks ingested")

    print("Done.")


if __name__ == "__main__":
    main()
