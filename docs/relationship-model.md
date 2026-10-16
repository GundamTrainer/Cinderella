# Relationship model

`RelationshipState` stores interaction timestamps/counts, familiarity, cooperation/trust/comfort values, unresolved topics, shared experiences, confirmed preferences, boundaries, and evidence IDs.

`RelationshipManager` requires a unique evidence event before recording an interaction, ignores duplicate event IDs, and bounds numeric fields to `[0, 1]`. Current behavior only makes a very small familiarity update; it does not infer trust, affection, or a human relationship.

Relationship state cannot change factual confidence or hardware safety decisions. It is not automatically created from face recognition or conversation. Persistence requires an explicit call to `SQLiteStateStore.save_relationship`; no user-facing inspection/deletion UI exists yet.
