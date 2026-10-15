from __future__ import annotations

import logging
import threading
import time
from collections.abc import Sequence

from .component import Component, ComponentState
from .config import RuntimeSettings
from .event_bus import Event, EventBus, EventType

logger = logging.getLogger(__name__)


class RuntimeOrchestrator:
    """Coordinates runtime components and consumes lifecycle events."""

    def __init__(
        self,
        settings: RuntimeSettings,
        event_bus: EventBus | None = None,
        bus: EventBus | None = None,
        components: Sequence[Component] | None = None,
    ) -> None:
        self.settings = settings
        selected_bus = event_bus if event_bus is not None else bus
        self.event_bus = selected_bus if selected_bus is not None else EventBus(settings.runtime_queue_capacity)
        self.components = list(components or [])
        self._shutdown_requested = threading.Event()

    def initialize(self) -> RuntimeOrchestrator:
        initialized: list[Component] = []
        try:
            for component in self.components:
                component.initialize()
                initialized.append(component)
        except Exception:
            logger.exception("Runtime initialization failed; cleaning up already initialized components")
            for component in reversed(initialized):
                try:
                    component.close()
                except Exception:
                    logger.exception("Failed to close component %s during partial initialization cleanup", component.name)
            raise
        return self

    def start(self) -> RuntimeOrchestrator:
        for component in self.components:
            component.start()
        return self

    def _handle_event(self, event: Event) -> None:
        if event.event_type == EventType.SYSTEM_STOPPING:
            self._shutdown_requested.set()

    def run_until_shutdown(self, *, timeout: float | None = None) -> list[Event]:
        limit = timeout if timeout is not None else self.settings.shutdown_timeout
        deadline = time.monotonic() + limit
        consumed: list[Event] = []
        while not self._shutdown_requested.is_set() and time.monotonic() < deadline:
            event = self.event_bus.poll(block=True, timeout=min(0.05, max(0.01, deadline - time.monotonic())))
            if event is None:
                continue
            consumed.append(event)
            self._handle_event(event)
        if not self._shutdown_requested.is_set():
            raise TimeoutError("Runtime shutdown was not requested within the configured timeout.")
        return consumed

    def shutdown(self) -> RuntimeOrchestrator:
        self._shutdown_requested.set()
        for component in reversed(self.components):
            if component.state in {ComponentState.RUNNING, ComponentState.STARTING, ComponentState.STOPPED}:
                try:
                    component.stop()
                except Exception:
                    logger.exception("Failed to stop component %s during shutdown", component.name)
        for component in reversed(self.components):
            if component.state != ComponentState.CLOSED:
                try:
                    component.close()
                except Exception:
                    logger.exception("Failed to close component %s during shutdown", component.name)
        self.event_bus.close()
        return self
