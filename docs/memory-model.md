# Memory model

`MemoryEntry` includes type, subject, summary, source, timestamps, confidence, importance, retention, expiry, consent, sensitivity, correction history, and supersession.

`SQLiteStateStore` requires an explicit path and explicit method calls. Persistent memory requires both `user_confirmed=True` and granted consent. External facts are rejected from the personal-memory table. The store supports listing/search, correction, retention/expiration updates, per-entry deletion, category deletion, and disablement.

Memory search is a bounded local substring match, not semantic retrieval. There is no automatic memory extraction from chat, no expiration worker, no dedicated external-fact cache persistence, and no UI. SQLite contents are plaintext; use a protected user directory and do not commit them. Database extensions are ignored by Git.
