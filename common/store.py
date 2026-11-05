"""SQLite event store (backup of every alert). The dashboard reads it and exports CSV / PDF.

Detectors write, the dashboard reads at the same time: WAL mode makes that safe.
An old outputs/events.csv is imported once the first time the database is created.
"""
import csv
import sqlite3
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    ts        TEXT NOT NULL,          -- ISO time, e.g. 2026-10-22T14:03:11
    camera    TEXT NOT NULL DEFAULT '',
    detector  TEXT NOT NULL,          -- gloves_detection
    event     TEXT NOT NULL,          -- no_gloves
    count     INTEGER,
    snapshot  TEXT,                   -- file name inside outputs/snapshots
    status    TEXT NOT NULL DEFAULT 'new'   -- reserved: new | confirmed | false_alarm
);
CREATE INDEX IF NOT EXISTS idx_events_ts ON events(ts);
CREATE INDEX IF NOT EXISTS idx_events_det ON events(detector);
"""


def connect(path):
    con = sqlite3.connect(str(path), timeout=10)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA busy_timeout=10000")
    return con


class EventStore:
    def __init__(self, path):
        self.path = Path(path)
        first = not self.path.exists()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with connect(self.path) as con:
            con.executescript(SCHEMA)
        if first:
            self._import_csv(self.path.with_name("events.csv"))

    def _import_csv(self, csv_path):
        if not csv_path.exists():
            return
        try:
            with open(csv_path, newline="", encoding="utf-8") as f, connect(self.path) as con:
                for r in csv.DictReader(f):
                    con.execute("INSERT INTO events (ts, detector, event, count, snapshot) VALUES (?,?,?,?,?)",
                                (r.get("timestamp"), r.get("detector"), r.get("event"),
                                 int(r.get("count") or 0), r.get("snapshot")))
        except Exception as e:           # a bad old CSV must never stop the detectors
            print("[store] could not import events.csv:", e)

    def add(self, ts, camera, detector, event, count, snapshot):
        with connect(self.path) as con:
            con.execute("INSERT INTO events (ts, camera, detector, event, count, snapshot) VALUES (?,?,?,?,?,?)",
                        (ts, camera, detector, event, int(count), snapshot))
