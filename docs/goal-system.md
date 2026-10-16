# Goal system

`GoalManager` models the requested goal types and origins, prioritizes pending goals, expires them, and supports cancellation and completion. Goals with external data transfer, personal persistence, hardware movement, messages, or high risk remain `awaiting_approval` until explicit approval with a recorded approval reference. Payment, software installation, API key creation, and model training/deployment goals are rejected.

The manager does not execute tools, send messages, install software, persist data, or move hardware. Goals are currently in-memory and are not automatically created from user speech or observations. A persistence/approval UI and execution safety controller are future work.
