"""
Ingest source excerpts into the RAG store.

Usage:
    python ingest.py                    # ingests sources_seed.jsonl
    python ingest.py my_sources.jsonl   # ingests a custom JSONL file

Each line is one JSON object:
    {"title": "...", "reference": "...", "content": "...", "lang": "en", "tags": ["..."]}

Only `title` and `content` are required. Add ONLY verified, authentic excerpts —
never fabricate page numbers or attributions.
"""
import json
import sys

import database as db
import rag


def main(path: str = "sources_seed.jsonl"):
    db.init_db()
    rag.init_rag()

    added, skipped = 0, 0
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                rag.add_source(
                    title=item["title"],
                    content=item["content"],
                    reference=item.get("reference"),
                    lang=item.get("lang", "en"),
                    tags=item.get("tags"),
                )
                added += 1
                print(f"[{added}] added: {item['title']}")
            except Exception as exc:
                skipped += 1
                print(f"  ! line {line_no} skipped: {exc}")

    print(f"\nDone. Added {added}, skipped {skipped}.")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "sources_seed.jsonl")
