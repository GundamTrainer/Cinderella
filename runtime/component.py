from __future__ import annotations

from abc import ABC, abstractmethod
from enum import StrEnum
from typing import Any


class ComponentState(StrEnum):
    NEW = "new"
    INITIALIZING = "initializing"
    INITIALIZED = "initialized"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    CLOSING = "closing"
    CLOSED = "closed"
    FAILED = "failed"


class LifecycleError(RuntimeError):
    """Raised when a component lifecycle operation fails."""


class InvalidStateTransitionError(LifecycleError):
    """Raised when a component transitions to an illegal state."""


class Component(ABC):
    """Base interface for long-running components."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.state = ComponentState.NEW

    def initialize(self) -> None:
        self._transition(ComponentState.INITIALIZING)
        try:
            self._do_initialize()
        except Exception:
            self.state = ComponentState.FAILED
            raise
        self.state = ComponentState.INITIALIZED

    def start(self) -> None:
        if self.state == ComponentState.STOPPED:
            self._transition(ComponentState.STARTING)
        else:
            self._transition(ComponentState.STARTING)
        try:
            self._do_start()
        except Exception:
            self.state = ComponentState.FAILED
            raise
        self.state = ComponentState.RUNNING

    def stop(self) -> None:
        if self.state not in {ComponentState.RUNNING, ComponentState.STARTING}:
            raise InvalidStateTransitionError(
                f"{self.name} cannot stop while in state {self.state.value}."
            )
        self._transition(ComponentState.STOPPING)
        try:
            self._do_stop()
        except Exception:
            self.state = ComponentState.FAILED
            raise
        self.state = ComponentState.STOPPED

    def close(self) -> None:
        if self.state == ComponentState.CLOSED:
            return
        self._transition(ComponentState.CLOSING)
        try:
            self._do_close()
        except Exception:
            self.state = ComponentState.FAILED
            raise
        self.state = ComponentState.CLOSED

    def health(self) -> dict[str, Any]:
        return {"name": self.name, "state": self.state.value, "healthy": self.state in {ComponentState.RUNNING, ComponentState.INITIALIZED, ComponentState.STOPPED}}

    def _transition(self, target: ComponentState) -> None:
        allowed = {
            ComponentState.NEW: {ComponentState.INITIALIZING},
            ComponentState.INITIALIZING: {ComponentState.INITIALIZED, ComponentState.FAILED},
            ComponentState.INITIALIZED: {ComponentState.STARTING, ComponentState.CLOSING, ComponentState.CLOSED},
            ComponentState.STARTING: {ComponentState.RUNNING, ComponentState.FAILED},
            ComponentState.RUNNING: {ComponentState.STOPPING, ComponentState.CLOSING},
            ComponentState.STOPPING: {ComponentState.STOPPED, ComponentState.FAILED},
            ComponentState.STOPPED: {ComponentState.STARTING, ComponentState.CLOSING, ComponentState.CLOSED},
            ComponentState.CLOSING: {ComponentState.CLOSED, ComponentState.FAILED},
            ComponentState.CLOSED: set(),
            ComponentState.FAILED: {ComponentState.CLOSING},
        }

        if self.state not in allowed or target not in allowed.get(self.state, set()):
            raise InvalidStateTransitionError(
                f"{self.name} cannot transition from {self.state.value} to {target.value}."
            )
        self.state = target

    @abstractmethod
    def _do_initialize(self) -> None:
        """Set up resources."""

    @abstractmethod
    def _do_start(self) -> None:
        """Begin work."""

    @abstractmethod
    def _do_stop(self) -> None:
        """Stop work."""

    @abstractmethod
    def _do_close(self) -> None:
        """Release resources."""
