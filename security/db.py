# dp.py

import os
import uuid
from typing import List, Dict

import psycopg
from psycopg.rows import dict_row

# Expected env: DATABASE_URL=postgresql://user:pass@host:port/dbname
DATABASE_URL = os.getenv("DATABASE_URL")

def get_conn():
    if not DATABASE_URL:
        raise RuntimeError("Missing DATABASE_URL environment variable.")
    return psycopg.connect(DATABASE_URL, autocommit=True)

def init_db():
    """
    Create tables if not exists:
      - person: id, username (unique), user_salt (bytea), created_at
      - template: id, person_id, tokens_80 (bytea, 80 bytes), version, created_at
    """
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute("""
        CREATE TABLE IF NOT EXISTS person (
          id UUID PRIMARY KEY,
          username TEXT UNIQUE NOT NULL,
          user_salt BYTEA NOT NULL,
          created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );
        """)
        cur.execute("""
        CREATE TABLE IF NOT EXISTS template (
          id UUID PRIMARY KEY,
          person_id UUID NOT NULL REFERENCES person(id) ON DELETE CASCADE,
          tokens_80 BYTEA NOT NULL,    -- 16 * 5 bytes = 80 bytes
          version TEXT NOT NULL,       -- e.g., "NH6->40:v1"
          created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );
        """)
    print("Database initialized (or already present).")

def upsert_person(username: str, user_salt: bytes) -> uuid.UUID:
    """
    Insert a person if not exists, else return existing id.
    psycopg v3 returns UUID columns as uuid.UUID already.
    """
    with get_conn() as conn, conn.cursor(row_factory=dict_row) as cur:
        cur.execute("SELECT id FROM person WHERE username=%s", (username,))
        row = cur.fetchone()
        if row:
            pid = row["id"]
            # If it's already a uuid.UUID (psycopg v3 default), return it.
            # Otherwise, coerce from string.
            return pid if isinstance(pid, uuid.UUID) else uuid.UUID(str(pid))
        pid = uuid.uuid4()
        cur.execute(
            "INSERT INTO person (id, username, user_salt) VALUES (%s, %s, %s)",
            (str(pid), username, psycopg.Binary(user_salt)),
        )
        return pid

def insert_template(person_id: uuid.UUID, tokens_80: bytes, version: str) -> uuid.UUID:
    if len(tokens_80) != 80:
        raise ValueError("tokens_80 must be exactly 80 bytes")
    tid = uuid.uuid4()
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            "INSERT INTO template (id, person_id, tokens_80, version) VALUES (%s, %s, %s, %s)",
            (str(tid), str(person_id), psycopg.Binary(tokens_80), version),
        )
    return tid

def fetch_all_templates() -> List[Dict]:
    """
    Returns a list of rows: [{ template_id, person_id, username, tokens_80 (bytes), version }]
    """
    with get_conn() as conn, conn.cursor(row_factory=dict_row) as cur:
        cur.execute("""
        SELECT t.id AS template_id, p.id AS person_id, p.username, t.tokens_80, t.version
        FROM template t
        JOIN person p ON p.id = t.person_id
        """)
        rows = cur.fetchall()
    # Convert memoryview to bytes for tokens_80
    for r in rows:
        if isinstance(r["tokens_80"], memoryview):
            r["tokens_80"] = bytes(r["tokens_80"])
    return rows