# Identity model

`IdentityState` is a neutral, non-character state model. It describes honesty, uncertainty, user autonomy, privacy minimization, non-manipulation, communication preferences, goals, and relationship boundaries. It does not assert biological emotion, human identity, or proven consciousness.

`IdentityManager` stages a proposed change, requires evidence, records stability and safety review results, waits for the evaluation period, and requires explicit user approval before applying it. Core identity fields cannot be changed through proposals. Provider results have no mutation path into this manager. When constructed with an initialized `SQLiteStateStore`, proposal stages and approved state are persisted; approval writes state and history in one database transaction.

State can be explicitly persisted through `SQLiteStateStore.save_identity`. The live conversation does not yet load, update, or review this state automatically. Values are stored as plaintext in the user-selected local database; the application does not encrypt the database.
