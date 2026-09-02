from __future__ import annotations

import array
import asyncio
import sqlite3
import sys

import pytest

from vllm.entrypoints.openai.responses.store.crypto import (
    DataDecryptionError,
    FramedAESGCMCipher,
)
from vllm.entrypoints.openai.responses.store.disk import SQLiteSessionStore
from vllm.entrypoints.openai.responses.store.key_provider import StaticKeyProvider

_KEY = b"k" * 32


async def _wait_until_complete(store: SQLiteSessionStore, session_id: str) -> None:
    for _ in range(200):
        if await store.is_complete(session_id):
            return
        await asyncio.sleep(0.001)
    pytest.fail(f"disk write did not complete for {session_id}")


def _encode_token_ids(token_ids: list[int]) -> bytes:
    values = array.array("I", token_ids)
    if sys.byteorder != "little":
        values.byteswap()
    return values.tobytes()


def test_framed_cipher_round_trip_and_authentication() -> None:
    cipher = FramedAESGCMCipher(StaticKeyProvider(_KEY))
    context = b"session-1"

    first = cipher.encrypt(b"first", context)
    second = cipher.encrypt(b"second", context)

    assert first != cipher.encrypt(b"first", context)
    assert cipher.decrypt(first + second, context) == b"firstsecond"

    tampered = bytearray(first)
    tampered[-1] ^= 1
    with pytest.raises(DataDecryptionError, match="authentication failed"):
        cipher.decrypt(bytes(tampered), context)


@pytest.mark.asyncio
async def test_sqlite_store_encrypts_incremental_token_data(tmp_path) -> None:
    db_path = tmp_path / "responses.db"
    provider = StaticKeyProvider(_KEY)
    store = SQLiteSessionStore(
        str(db_path),
        write_interval_seconds=0.001,
        key_provider=provider,
    )
    token_ids = [1, 2, 3, 4, 1000, 2000, 3000, 4000]

    try:
        await store.save("session-1", "response-1", token_ids[:4])
        await _wait_until_complete(store, "session-1")
        assert await store.get("session-1") == token_ids[:4]

        await store.save("session-1", "response-2", token_ids[4:])
        await _wait_until_complete(store, "session-1")
        assert await store.get("session-1") == token_ids

        with sqlite3.connect(db_path) as connection:
            row = connection.execute(
                "SELECT token_ids FROM session_state WHERE session_id = ?",
                ("session-1",),
            ).fetchone()

        assert row is not None
        encrypted_blob = bytes(row[0])
        plaintext_blob = _encode_token_ids(token_ids)
        assert encrypted_blob != plaintext_blob
        assert plaintext_blob not in encrypted_blob

        cipher = FramedAESGCMCipher(provider)
        assert cipher.decrypt(encrypted_blob, b"session-1") == plaintext_blob
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_get_quarantines_corrupted_ciphertext(tmp_path) -> None:
    db_path = tmp_path / "responses.db"
    store = SQLiteSessionStore(
        str(db_path),
        write_interval_seconds=0.001,
        key_provider=StaticKeyProvider(_KEY),
    )

    try:
        await store.save("corrupt", "response-1", [1, 2, 3, 4])
        await _wait_until_complete(store, "corrupt")
        assert await store.get("corrupt") == [1, 2, 3, 4]

        with sqlite3.connect(db_path) as connection:
            connection.execute(
                """
                UPDATE session_state
                SET token_ids = zeroblob(length(token_ids))
                WHERE session_id = ?
                """,
                ("corrupt",),
            )

        assert await store.get("corrupt") is None
        assert await store.needs_snapshot("corrupt")
        assert not await store.is_complete("corrupt")

        with sqlite3.connect(db_path) as connection:
            row_count = connection.execute(
                "SELECT COUNT(*) FROM session_state WHERE session_id = ?",
                ("corrupt",),
            ).fetchone()
        assert row_count == (0,)
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_list_skips_corrupted_ciphertext(tmp_path) -> None:
    db_path = tmp_path / "responses.db"
    store = SQLiteSessionStore(
        str(db_path),
        write_interval_seconds=0.001,
        key_provider=StaticKeyProvider(_KEY),
    )

    try:
        await store.save("corrupt", "response-1", [1, 2])
        await store.save("healthy", "response-2", [3, 4])
        await _wait_until_complete(store, "corrupt")
        await _wait_until_complete(store, "healthy")
        assert await store.get("corrupt") == [1, 2]
        assert await store.get("healthy") == [3, 4]

        with sqlite3.connect(db_path) as connection:
            connection.execute(
                """
                UPDATE session_state
                SET token_ids = zeroblob(length(token_ids))
                WHERE session_id = ?
                """,
                ("corrupt",),
            )

        states = await store.list()
        assert [(state.session_id, state.token_ids) for state in states] == [
            ("healthy", [3, 4])
        ]
        assert [state.session_id for state in await store.list()] == ["healthy"]
        assert await store.needs_snapshot("corrupt")
    finally:
        await store.close()
