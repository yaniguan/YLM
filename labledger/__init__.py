"""labledger: a hierarchical, append-only record of lab experiments.

Levels: project > campaign > experiment > step > observation > artifact.
Every change is a new event in a hash-chained JSONL log. Raw files live in a
content-addressed blob store (sha256) and are never changed.
"""

from .ledger import Ledger, LedgerError
from .schema import SCHEMA, SchemaError, validate_event

__all__ = ["SCHEMA", "Ledger", "LedgerError", "SchemaError", "validate_event"]
