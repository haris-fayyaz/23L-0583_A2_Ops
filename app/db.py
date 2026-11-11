import os
from datetime import datetime, timezone

from sqlalchemy import (Column, DateTime, Float, Integer, MetaData, String,
                        Table, create_engine, insert, select)

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///instance/visionguard.db")

engine = create_engine(DATABASE_URL, pool_pre_ping=True)
metadata = MetaData()

request_log = Table(
    "request_log", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("filename", String(255), nullable=False),
    Column("label", String(32), nullable=False),
    Column("confidence", Float, nullable=False),
    Column("threshold", Float, nullable=False),
    Column("model", String(64), nullable=False),
)


def init_db():
    """Create the table if missing. Makes the folder for a SQLite file."""
    if engine.url.get_backend_name() == "sqlite":
        path = engine.url.database
        if path and path != ":memory:":
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    metadata.create_all(engine)


def log_request(filename, label, confidence, threshold, model):
    with engine.begin() as conn:
        conn.execute(insert(request_log).values(
            created_at=datetime.now(timezone.utc), filename=filename,
            label=label, confidence=confidence, threshold=threshold,
            model=model))


def recent(limit=100):
    q = select(request_log).order_by(request_log.c.id.desc()).limit(limit)
    with engine.connect() as conn:
        return [dict(r._mapping) for r in conn.execute(q)]
