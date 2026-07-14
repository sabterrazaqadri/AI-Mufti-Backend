#!/usr/bin/env python3
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import database as db

db.init_db()

with db.get_cursor() as cur:
    cur.execute("SELECT COUNT(*) as cnt FROM sources;")
    result = cur.fetchone()
    total = result['cnt'] if result else 0
    print(f"Total chunks in DB: {total}")

    # Per-jild breakdown
    for jild in [1, 2, 3]:
        cur.execute(f"SELECT COUNT(*) as cnt FROM sources WHERE tags && ARRAY['jild-{jild}'];")
        result = cur.fetchone()
        jild_count = result['cnt'] if result else 0
        print(f"  Jild {jild}: {jild_count}")

    # Table size
    cur.execute("SELECT pg_size_pretty(pg_total_relation_size('sources')) as size;")
    result = cur.fetchone()
    size = result['size'] if result else 'unknown'
    print(f"Table size: {size}")
