"""Adapt persisted agent IDs and detect platform delivery failures."""
from __future__ import annotations

from contextlib import contextmanager
import datetime as dt
import hashlib
import logging
import threading
import uuid


class PlatformDeliveryError(RuntimeError):
    """The SDK did not complete a clean delivery."""


def platform_id(source_id: str, timestamp: str, kind: str) -> str:
    """Map an internal ID to a stable RFC UUIDv7 with recorded time."""
    try:
        original = uuid.UUID(source_id)
        if original.version == 7 and original.variant == uuid.RFC_4122:
            return str(original)
    except ValueError:
        pass
    recorded = dt.datetime.fromisoformat(timestamp)
    if recorded.tzinfo is None:
        recorded = recorded.replace(tzinfo=dt.timezone.utc)
    milliseconds = int(recorded.timestamp() * 1000)
    if not 0 <= milliseconds < 1 << 48:
        raise ValueError("platform UUID timestamp is outside the supported range")
    data = bytearray(hashlib.sha256(f"vbe-platform:{kind}:{source_id}".encode()).digest()[:16])
    data[:6] = milliseconds.to_bytes(6, "big")
    data[6] = 0x70 | (data[6] & 0x0F)
    data[8] = 0x80 | (data[8] & 0x3F)
    return str(uuid.UUID(bytes=bytes(data)))


class _DeliveryErrors(logging.Handler):
    def __init__(self):
        super().__init__(level=logging.WARNING)
        self.count = 0
        self.counter_lock = threading.Lock()

    def emit(self, record):
        # The SDK logs server rejection and then removes the rejected message.
        if record.levelno >= logging.ERROR or "failed" in record.getMessage().lower() or "discard" in record.getMessage().lower():
            with self.counter_lock:
                self.count += 1


@contextmanager
def checked_delivery():
    """Observe SDK background failures through its documented logging channel."""
    logger = logging.getLogger("opik.message_processing")
    observer = _DeliveryErrors()
    logger.addHandler(observer)
    try:
        yield
        if observer.count:
            raise PlatformDeliveryError(f"platform SDK reported {observer.count} background delivery error(s); inspect the SDK log")
    finally:
        logger.removeHandler(observer)


def checked_flush(client) -> None:
    if client.flush() is False:
        raise PlatformDeliveryError("platform SDK flush did not complete before its timeout")
