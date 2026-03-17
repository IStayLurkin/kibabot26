import pytest

def test_streaming_chunk_boundaries():
    content = "x" * 500
    chunk_size = 250
    chunks = [content[:i+chunk_size] for i in range(0, len(content), chunk_size)]
    assert chunks[0] == "x" * 250
    assert chunks[1] == "x" * 500
