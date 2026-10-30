"""Recoverable catalog deletion. Media stays available to immutable references."""
from __future__ import annotations

import time


def initialize(db):
    db.execute("""CREATE TABLE IF NOT EXISTS catalog_trash(
        kind TEXT NOT NULL, item_id TEXT NOT NULL, label TEXT NOT NULL,
        deleted_at REAL NOT NULL, PRIMARY KEY(kind,item_id))""")


def hidden(db, kind):
    return {row[0] for row in db.execute("SELECT item_id FROM catalog_trash WHERE kind=?", (kind,))}


def mark(db, kind, item_id, label, *, restore=False):
    if type(restore) is not bool:
        raise ValueError("恢复选项必须是布尔值")
    if restore:
        db.execute("DELETE FROM catalog_trash WHERE kind=? AND item_id=?", (kind, item_id))
    else:
        db.execute("INSERT OR REPLACE INTO catalog_trash VALUES(?,?,?,?)", (kind, item_id, label, time.time()))


def entries(db):
    return [dict(kind=row[0], id=row[1], label=row[2], deleted_at=row[3])
            for row in db.execute("SELECT kind,item_id,label,deleted_at FROM catalog_trash ORDER BY deleted_at DESC,item_id DESC")]
