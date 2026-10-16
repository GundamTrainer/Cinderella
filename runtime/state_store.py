from __future__ import annotations

import json
import sqlite3
import threading
from datetime import UTC, datetime
from pathlib import Path

from .companion_state import (
    IdentityChangeProposal,
    IdentityState,
    MemoryEntry,
    MemoryType,
    RelationshipState,
)


class SQLiteStateStore:
    """Explicit local persistence. No state is saved unless a caller invokes a save method."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._connection: sqlite3.Connection | None = None
        self._lock = threading.RLock()

    def initialize(self) -> None:
        if self._connection is not None:
            return
        if self.path.parent != Path("."):
            self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.path, check_same_thread=False)
        try:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS state_records ("
                "kind TEXT NOT NULL, record_id TEXT NOT NULL, data_json TEXT NOT NULL, "
                "PRIMARY KEY (kind, record_id))"
            )
            connection.execute(
                "CREATE TABLE IF NOT EXISTS memory_records ("
                "memory_id TEXT PRIMARY KEY, memory_type TEXT NOT NULL, data_json TEXT NOT NULL)"
            )
            connection.execute(
                "CREATE TABLE IF NOT EXISTS identity_change_history ("
                "proposal_id TEXT PRIMARY KEY, data_json TEXT NOT NULL)"
            )
            connection.execute(
                "CREATE TABLE IF NOT EXISTS store_settings ("
                "setting TEXT PRIMARY KEY, value TEXT NOT NULL)"
            )
            connection.execute(
                "INSERT OR IGNORE INTO store_settings(setting, value) VALUES ('memory_enabled', 'true')"
            )
            connection.commit()
        except sqlite3.Error:
            connection.close()
            raise
        self._connection = connection

    def save_identity(self, state: IdentityState) -> None:
        self._save_state("identity", state.system_name, state.to_dict())

    def load_identity(self, system_name: str = "Cinderella") -> IdentityState | None:
        data = self._load_state("identity", system_name)
        return None if data is None else IdentityState.from_dict(data)

    def save_relationship(self, state: RelationshipState) -> None:
        self._save_state("relationship", state.participant_id, state.to_dict())

    def load_relationship(self, participant_id: str) -> RelationshipState | None:
        data = self._load_state("relationship", participant_id)
        return None if data is None else RelationshipState.from_dict(data)

    def save_identity_proposal(self, proposal: IdentityChangeProposal) -> None:
        with self._lock:
            connection = self._require_connection()
            with connection:
                connection.execute(
                    "INSERT INTO identity_change_history(proposal_id, data_json) VALUES (?, ?) "
                    "ON CONFLICT(proposal_id) DO UPDATE SET data_json=excluded.data_json",
                    (
                        proposal.proposal_id,
                        json.dumps(proposal.to_dict(), ensure_ascii=False, sort_keys=True),
                    ),
                )

    def load_identity_proposals(self) -> tuple[IdentityChangeProposal, ...]:
        with self._lock:
            connection = self._require_connection()
            rows = connection.execute(
                "SELECT data_json FROM identity_change_history ORDER BY rowid"
            ).fetchall()
        return tuple(IdentityChangeProposal.from_dict(json.loads(row[0])) for row in rows)

    def apply_identity_approval(
        self,
        state: IdentityState,
        proposal: IdentityChangeProposal,
    ) -> None:
        with self._lock:
            connection = self._require_connection()
            with connection:
                connection.execute(
                    "INSERT INTO state_records(kind, record_id, data_json) VALUES ('identity', ?, ?) "
                    "ON CONFLICT(kind, record_id) DO UPDATE SET data_json=excluded.data_json",
                    (
                        state.system_name,
                        json.dumps(state.to_dict(), ensure_ascii=False, sort_keys=True),
                    ),
                )
                connection.execute(
                    "INSERT INTO identity_change_history(proposal_id, data_json) VALUES (?, ?) "
                    "ON CONFLICT(proposal_id) DO UPDATE SET data_json=excluded.data_json",
                    (
                        proposal.proposal_id,
                        json.dumps(proposal.to_dict(), ensure_ascii=False, sort_keys=True),
                    ),
                )

    def save_memory(self, entry: MemoryEntry) -> None:
        if entry.memory_type == MemoryType.EXTERNAL_FACT_CACHE:
            raise ValueError("External facts must not be stored as personal memory entries.")
        if not entry.user_confirmed or entry.consent_status.value != "granted":
            raise PermissionError("Persistent memory requires explicit user confirmation and consent.")
        if entry.sensitive and entry.consent_status.value != "granted":
            raise PermissionError("Sensitive memory requires explicit consent.")
        with self._lock:
            connection = self._require_connection()
            if not self._memory_enabled(connection):
                raise PermissionError("Persistent memory is disabled.")
            with connection:
                connection.execute(
                    "INSERT INTO memory_records(memory_id, memory_type, data_json) VALUES (?, ?, ?) "
                    "ON CONFLICT(memory_id) DO UPDATE SET "
                    "memory_type=excluded.memory_type, data_json=excluded.data_json",
                    (
                        entry.memory_id,
                        entry.memory_type.value,
                        json.dumps(entry.to_dict(), ensure_ascii=False, sort_keys=True),
                    ),
                )

    def find_memories(self, query: str, *, limit: int = 20) -> tuple[MemoryEntry, ...]:
        if not query.strip():
            raise ValueError("Memory query must not be empty.")
        if not 1 <= limit <= 100:
            raise ValueError("Memory result limit must be between 1 and 100.")
        with self._lock:
            connection = self._require_connection()
            if not self._memory_enabled(connection):
                return ()
            rows = connection.execute(
                "SELECT data_json FROM memory_records ORDER BY memory_id LIMIT ?",
                (min(limit * 10, 1000),),
            ).fetchall()
        words = query.casefold().split()
        result = []
        for (encoded,) in rows:
            entry = MemoryEntry.from_dict(json.loads(encoded))
            if entry.expiration is not None:
                expiration = datetime.fromisoformat(entry.expiration)
                if expiration <= datetime.now(UTC):
                    continue
            searchable = f"{entry.subject} {entry.content_summary}".casefold()
            if all(word in searchable for word in words):
                result.append(entry)
                if len(result) >= limit:
                    break
        return tuple(result)

    def delete_memory(self, memory_id: str) -> bool:
        with self._lock:
            connection = self._require_connection()
            with connection:
                cursor = connection.execute(
                    "DELETE FROM memory_records WHERE memory_id = ?", (memory_id,)
                )
            return cursor.rowcount > 0

    def list_memories(self, *, limit: int = 100) -> tuple[MemoryEntry, ...]:
        if not 1 <= limit <= 1000:
            raise ValueError("Memory listing limit must be between 1 and 1000.")
        with self._lock:
            connection = self._require_connection()
            if not self._memory_enabled(connection):
                return ()
            rows = connection.execute(
                "SELECT data_json FROM memory_records ORDER BY memory_id LIMIT ?",
                (limit,),
            ).fetchall()
        now = datetime.now(UTC)
        entries = tuple(MemoryEntry.from_dict(json.loads(row[0])) for row in rows)
        return tuple(
            entry
            for entry in entries
            if entry.expiration is None
            or datetime.fromisoformat(entry.expiration) > now
        )

    def set_memory_retention(
        self,
        memory_id: str,
        *,
        retention_policy: str,
        expiration: str | None,
    ) -> MemoryEntry:
        if not retention_policy.strip():
            raise ValueError("retention_policy must not be empty.")
        with self._lock:
            connection = self._require_connection()
            row = connection.execute(
                "SELECT data_json FROM memory_records WHERE memory_id = ?", (memory_id,)
            ).fetchone()
            if row is None:
                raise KeyError(f"Unknown memory {memory_id!r}.")
            entry = MemoryEntry.from_dict(json.loads(row[0]))
            updated = MemoryEntry.from_dict(
                {
                    **entry.to_dict(),
                    "retention_policy": retention_policy,
                    "expiration": expiration,
                }
            )
            with connection:
                connection.execute(
                    "UPDATE memory_records SET data_json = ? WHERE memory_id = ?",
                    (json.dumps(updated.to_dict(), ensure_ascii=False, sort_keys=True), memory_id),
                )
            return updated

    def delete_memories_by_type(self, memory_type: MemoryType) -> int:
        with self._lock:
            connection = self._require_connection()
            with connection:
                cursor = connection.execute(
                    "DELETE FROM memory_records WHERE memory_type = ?",
                    (MemoryType(memory_type).value,),
                )
            return cursor.rowcount

    def correct_memory(self, memory_id: str, corrected_summary: str) -> MemoryEntry:
        if not corrected_summary.strip():
            raise ValueError("Corrected memory summary must not be empty.")
        with self._lock:
            connection = self._require_connection()
            row = connection.execute(
                "SELECT data_json FROM memory_records WHERE memory_id = ?", (memory_id,)
            ).fetchone()
            if row is None:
                raise KeyError(f"Unknown memory {memory_id!r}.")
            entry = MemoryEntry.from_dict(json.loads(row[0]))
            corrected = MemoryEntry.from_dict(
                {
                    **entry.to_dict(),
                    "content_summary": corrected_summary,
                    "correction_history": (*entry.correction_history, entry.content_summary),
                }
            )
            with connection:
                connection.execute(
                    "UPDATE memory_records SET data_json = ? WHERE memory_id = ?",
                    (json.dumps(corrected.to_dict(), ensure_ascii=False, sort_keys=True), memory_id),
                )
            return corrected

    def set_memory_enabled(self, enabled: bool) -> None:
        if not isinstance(enabled, bool):
            raise TypeError("enabled must be a boolean.")
        with self._lock:
            connection = self._require_connection()
            with connection:
                connection.execute(
                    "UPDATE store_settings SET value = ? WHERE setting = 'memory_enabled'",
                    ("true" if enabled else "false",),
                )

    def close(self) -> None:
        with self._lock:
            if self._connection is not None:
                self._connection.close()
                self._connection = None

    def _save_state(self, kind: str, record_id: str, data: dict[str, object]) -> None:
        with self._lock:
            connection = self._require_connection()
            with connection:
                connection.execute(
                    "INSERT INTO state_records(kind, record_id, data_json) VALUES (?, ?, ?) "
                    "ON CONFLICT(kind, record_id) DO UPDATE SET data_json=excluded.data_json",
                    (kind, record_id, json.dumps(data, ensure_ascii=False, sort_keys=True)),
                )

    def _load_state(self, kind: str, record_id: str) -> dict[str, object] | None:
        with self._lock:
            connection = self._require_connection()
            row = connection.execute(
                "SELECT data_json FROM state_records WHERE kind = ? AND record_id = ?",
                (kind, record_id),
            ).fetchone()
        return None if row is None else json.loads(row[0])

    @staticmethod
    def _memory_enabled(connection: sqlite3.Connection) -> bool:
        row = connection.execute(
            "SELECT value FROM store_settings WHERE setting = 'memory_enabled'"
        ).fetchone()
        return row is not None and row[0] == "true"

    def _require_connection(self) -> sqlite3.Connection:
        if self._connection is None:
            raise RuntimeError("State store is not initialized.")
        return self._connection
