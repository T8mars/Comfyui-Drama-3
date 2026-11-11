"""Recoverable catalog deletion. Media stays available to immutable references."""
from __future__ import annotations

import time


def initialize(db):
    db.execute("""CREATE TABLE IF NOT EXISTS catalog_trash(
        kind TEXT NOT NULL, item_id TEXT NOT NULL, label TEXT NOT NULL,
        deleted_at REAL NOT NULL, PRIMARY KEY(kind,item_id))""")
    db.execute("""CREATE TABLE IF NOT EXISTS catalog_trash_cleared(
        kind TEXT NOT NULL, item_id TEXT NOT NULL, cleared_at REAL NOT NULL,
        PRIMARY KEY(kind,item_id))""")


def hidden(db, kind):
    return {row[0] for row in db.execute("SELECT item_id FROM catalog_trash WHERE kind=?", (kind,))}


def mark(db, kind, item_id, label, *, restore=False):
    if type(restore) is not bool:
        raise ValueError("恢复选项必须是布尔值")
    if restore:
        if db.execute("SELECT 1 FROM catalog_trash_cleared WHERE kind=? AND item_id=?", (kind, item_id)).fetchone():
            raise ValueError("此回收站记录已清除，无法恢复")
        db.execute("DELETE FROM catalog_trash WHERE kind=? AND item_id=?", (kind, item_id))
    else:
        db.execute("INSERT OR REPLACE INTO catalog_trash VALUES(?,?,?,?)", (kind, item_id, label, time.time()))


def clear(db, kind, item_id):
    """Dismiss a trash entry without reviving it or breaking pinned references."""
    if not db.execute("SELECT 1 FROM catalog_trash WHERE kind=? AND item_id=?", (kind, item_id)).fetchone():
        raise ValueError("此记录已不在回收站，请刷新")
    db.execute("INSERT OR IGNORE INTO catalog_trash_cleared VALUES(?,?,?)", (kind, item_id, time.time()))


def entries(db):
    return [dict(kind=row[0], id=row[1], label=row[2], deleted_at=row[3])
            for row in db.execute("SELECT kind,item_id,label,deleted_at FROM catalog_trash t "
                "WHERE NOT EXISTS(SELECT 1 FROM catalog_trash_cleared c WHERE c.kind=t.kind AND c.item_id=t.item_id) "
                "ORDER BY deleted_at DESC,item_id DESC")]
