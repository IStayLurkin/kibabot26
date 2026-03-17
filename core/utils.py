from __future__ import annotations

_CHUNK_SIZE = 1990


async def send_chunked(destination, text: str, **kwargs) -> None:
    """Send text to a Discord destination, splitting into <=1990-char chunks if needed."""
    if not text:
        return
    if len(text) <= _CHUNK_SIZE:
        await destination.send(text, **kwargs)
        return
    for i in range(0, len(text), _CHUNK_SIZE):
        await destination.send(text[i : i + _CHUNK_SIZE], **kwargs)
