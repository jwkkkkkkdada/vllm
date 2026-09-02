from __future__ import annotations

import array
import asyncio
import sqlite3
import sys
import time
from dataclasses import dataclass
from enum import Enum, unique
from pathlib import Path

from vllm.logger import init_logger

from .base import (
    DiskEvictionResult,
    DiskEvictionStatus,
    SessionState,
    SessionStore,
)
from .crypto import DataDecryptionError, FramedAESGCMCipher
from .key_provider import EphemeralKeyProvider, KeyProvider
from .metrics import ResponseStoreMetrics

logger = init_logger(__name__)


@unique
class DiskCopyStatus(str, Enum):
    """Disk copy state observed by memory eviction."""

    READY = "ready"
    NOT_FOUND = "not_found"
    RESPONSE_CHANGED = "response_changed"
    VERSION_CHANGED = "version_changed"
    EXPIRED = "expired"
    PENDING = "pending"
    FAILED = "failed"
    INVALID = "invalid"


@dataclass(frozen=True, slots=True)
class DiskCopyResult:
    status: DiskCopyStatus


@dataclass(slots=True)
class _PendingWrite:
    session_id: str
    response_id: str
    delta_token_ids: list[int]
    created_at: int
    updated_at: int
    disk_idle_expires_at: int | None
    version: int
    replace_existing: bool = False


@dataclass(frozen=True, slots=True)
class _CorruptDiskCopy:
    session_id: str
    write_version: int


@dataclass(frozen=True, slots=True)
class _ReadStateResult:
    state: SessionState | None = None
    corrupt_copy: _CorruptDiskCopy | None = None


class SQLiteSessionStore(SessionStore):
    """
    SQLite 二级存储。

    - 构造时清空旧表：不做重启恢复。
    - save() 只写进程内 pending，不做 SQLite I/O。
    - 后台单 Writer 按时间窗口合并同一 Session 的多次 save。
    - Token ID 以独立 AES-GCM 帧加密，增量写入不解密历史数据。
    - SQLite transaction 保证批次原子提交。
    - latest_version / committed_version 保证只恢复“最新且完整”的状态。
    - Memory miss 时若最新版本仍在写，短暂等待 commit，超时才返回 None。
    """

    _TOKEN_TYPECODE = "I"  # uint32

    def __init__(
        self,
        db_path: str,
        disk_idle_ttl_seconds: int | None = None,
        write_interval_seconds: float = 0.05,
        recovery_wait_timeout_seconds: float = 0.2,
        key_provider: KeyProvider | None = None,
        metrics: ResponseStoreMetrics | None = None,
    ) -> None:
        if not db_path:
            raise ValueError("db_path must not be empty")
        if disk_idle_ttl_seconds is not None and disk_idle_ttl_seconds <= 0:
            raise ValueError("disk_idle_ttl_seconds must be greater than 0")
        if write_interval_seconds <= 0:
            raise ValueError("write_interval_seconds must be greater than 0")
        if recovery_wait_timeout_seconds <= 0:
            raise ValueError("recovery_wait_timeout_seconds must be greater than 0")
        if array.array(self._TOKEN_TYPECODE).itemsize != 4:
            raise RuntimeError("32-bit unsigned int array is required")

        self._db_path = db_path
        self._disk_idle_ttl_seconds = disk_idle_ttl_seconds
        self._write_interval_seconds = write_interval_seconds
        self._recovery_wait_timeout_seconds = recovery_wait_timeout_seconds
        self._cipher = FramedAESGCMCipher(
            key_provider if key_provider is not None else EphemeralKeyProvider()
        )
        self._metrics = metrics if metrics is not None else ResponseStoreMetrics()

        # 创建 SQLite 数据库文件所在的目录
        if db_path != ":memory:":
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)

        # 同一 Session 在 pending 中最多一条，连续 save 会合并 delta。
        self._pending: dict[str, _PendingWrite] = {}

        # Store 级单调版本避免为每个历史 Session 永久保留计数器。
        self._version_counter = 0
        self._latest_versions: dict[str, int] = {}
        self._committed_versions: dict[str, int] = {}
        self._latest_response_ids: dict[str, str] = {}
        self._failed_versions: dict[str, int] = {}

        # delete 后，<= 此版本的已捕获 Writer 批次全部作废，避免删除复活。
        self._discard_through_versions: dict[str, int] = {}
        self._inflight_versions: dict[str, int] = {}

        self._meta_lock = asyncio.Lock()

        # 等待恢复
        # get_state() 在发现 latest_version 尚未 commit 时，通过 Condition
        # 等待 Writer 完成，而不是立即返回 None 触发全量 prompt 重算。
        # Condition 与 _meta_lock 共用同一把锁，版本状态变化时统一通知。
        self._commit_condition = asyncio.Condition(self._meta_lock)

        self._io_lock = asyncio.Lock()
        # Writer 通知机制
        self._writer_event = asyncio.Event()
        self._writer_task: asyncio.Task[None] | None = None
        self._closing = False
        self._closed = False

        # sqlite3 是阻塞 API。connection 可跨 worker thread 使用，但所有
        # SQLite 操作仍通过 _io_lock 串行，并放入 asyncio.to_thread()。
        # 对调用方异步执行，但多个SQLite操作串行执行
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._initialize_database()

    async def save(
        self,
        session_id: str,
        response_id: str,
        delta_token_ids: list[int],
    ) -> None:
        """只进入 pending buffer；真正 SQLite I/O 由后台 Writer 完成。"""
        self._ensure_open()
        self._ensure_writer_started()
        now = int(time.time())

        async with self._meta_lock:
            version = self._allocate_version_locked()
            self._latest_versions[session_id] = version
            self._latest_response_ids[session_id] = response_id

            pending = self._pending.get(session_id)
            if pending is None:
                self._pending[session_id] = _PendingWrite(
                    session_id=session_id,
                    response_id=response_id,
                    delta_token_ids=list(delta_token_ids),
                    created_at=now,
                    updated_at=now,
                    disk_idle_expires_at=self._build_expire_time(now),
                    version=version,
                )
            else:
                pending.delta_token_ids.extend(delta_token_ids)
                pending.response_id = response_id
                pending.updated_at = now
                pending.disk_idle_expires_at = self._build_expire_time(now)
                pending.version = version

            # latest_version 已变化，唤醒可能正在等待旧版本完成的恢复请求。
            self._commit_condition.notify_all()
            self._writer_event.set()

    async def save_snapshot(self, state: SessionState) -> None:
        """使用 Memory 完整快照重建 Disk 副本。"""
        self._ensure_open()
        self._ensure_writer_started()
        now = int(time.time())

        async with self._meta_lock:
            version = self._allocate_version_locked()
            self._latest_versions[state.session_id] = version
            self._latest_response_ids[state.session_id] = state.response_id

            pending = self._pending.get(state.session_id)
            if pending is None:
                self._pending[state.session_id] = _PendingWrite(
                    session_id=state.session_id,
                    response_id=state.response_id,
                    delta_token_ids=list(state.token_ids),
                    created_at=state.created_at,
                    updated_at=state.updated_at,
                    disk_idle_expires_at=self._build_expire_time(now),
                    version=version,
                    replace_existing=True,
                )
            else:
                pending.response_id = state.response_id
                pending.delta_token_ids = list(state.token_ids)
                pending.created_at = state.created_at
                pending.updated_at = state.updated_at
                pending.disk_idle_expires_at = self._build_expire_time(now)
                pending.version = version
                pending.replace_existing = True

            self._commit_condition.notify_all()
            self._writer_event.set()

    async def needs_snapshot(self, session_id: str) -> bool:
        async with self._meta_lock:
            return session_id not in self._latest_versions

    async def used_bytes(self) -> int:
        """返回当前 SQLite Session 行记录的逻辑总字节数。"""
        async with self._io_lock:
            return await asyncio.to_thread(self._used_bytes_sync)

    @property
    def metrics(self) -> ResponseStoreMetrics:
        return self._metrics

    async def get(self, session_id: str) -> list[int] | None:
        state = await self.get_state(session_id)
        return None if state is None else state.token_ids.copy()

    async def get_state(self, session_id: str) -> SessionState | None:
        """
        获取当前 Session 的完整 Disk 状态。

        如果最新版本还在 pending / writing：
            不立即返回 None；
            先等待最新版本 commit，避免因为几十毫秒的异步落盘窗口
            触发 Server 重新获取完整 prompt 并重新 tokenizer。

        只有以下情况才返回 None：
            1. Disk 中不存在该 Session；
            2. Session 已被 delete / Disk 淘汰；
            3. 等待最新版本 commit 超时；
            4. 最新完整版本在读取期间再次发生变化，且最终仍未能在
               timeout 内得到稳定完整版本。
        """
        self._ensure_open()

        loop = asyncio.get_running_loop()
        deadline = loop.time() + self._recovery_wait_timeout_seconds

        while True:
            remaining = deadline - loop.time()
            if remaining <= 0:
                return None

            # 如果 latest != committed，这里等待 Writer 通知。
            expected = await self._wait_for_complete_version(
                session_id,
                timeout_seconds=remaining,
            )
            if expected is None:
                return None

            async with self._io_lock:
                # 等待 _io_lock 的过程中可能又发生 save/delete，
                # 所以真正读取前再确认一次完整版本。
                current = await self._get_complete_version(session_id)
                if current != expected:
                    continue

                read_result = await asyncio.to_thread(
                    self._read_state_sync,
                    session_id,
                    expected,
                    int(time.time()),
                )

                if read_result.corrupt_copy is not None:
                    async with self._meta_lock:
                        await asyncio.to_thread(
                            self._delete_corrupt_copies_sync,
                            [read_result.corrupt_copy],
                        )
                        self._invalidate_corrupt_copy_locked(read_result.corrupt_copy)
                    logger.warning(
                        "Removed corrupted ResponseStore SQLite copy for session %s",
                        session_id,
                    )
                    return None

            if read_result.state is None:
                return None

            # SQLite 读取期间 save() 不需要 _io_lock。返回前最后检查一次；
            # 如果版本又前进了，则在剩余 timeout 内继续等待最新版本。
            if await self._get_complete_version(session_id) == expected:
                return read_result.state

    async def exists(self, session_id: str) -> bool:
        """只有最新版本完整落盘时才认为 Disk 中存在可恢复 Session。"""
        self._ensure_open()
        expected = await self._get_complete_version(session_id)
        if expected is None:
            return False

        async with self._io_lock:
            expected = await self._get_complete_version(session_id)
            if expected is None:
                return False
            exists = await asyncio.to_thread(self._exists_sync, session_id, expected)

        return exists and await self._get_complete_version(session_id) == expected

    async def delete(self, session_id: str) -> bool:
        """删除已落盘数据，同时取消 pending，并防止已捕获批次重新写回。"""
        self._ensure_open()

        async with self._meta_lock:
            current = self._latest_versions.get(session_id, 0)
            if current:
                self._discard_through_versions[session_id] = max(
                    current, self._discard_through_versions.get(session_id, 0)
                )
            had_pending = self._pending.pop(session_id, None) is not None
            had_latest = self._latest_versions.pop(session_id, None) is not None
            self._committed_versions.pop(session_id, None)
            self._latest_response_ids.pop(session_id, None)
            self._failed_versions.pop(session_id, None)
            self._prune_discard_tombstone_locked(session_id)

            # 正在等待该 Session commit 的恢复请求应立即结束等待。
            self._commit_condition.notify_all()

        async with self._io_lock:
            deleted = await asyncio.to_thread(self._delete_sync, session_id)

        return had_pending or had_latest or deleted

    async def list(self) -> list[SessionState]:
        """低频管理接口：只返回当前完整可恢复的 Disk 状态。"""
        self._ensure_open()

        # list 期间短暂阻塞新的 Disk enqueue，以获得稳定版本关系。
        async with self._io_lock, self._meta_lock:
            rows, corrupt_copies = await asyncio.to_thread(self._list_sync)
            if corrupt_copies:
                await asyncio.to_thread(
                    self._delete_corrupt_copies_sync, corrupt_copies
                )
                for corrupt_copy in corrupt_copies:
                    self._invalidate_corrupt_copy_locked(corrupt_copy)
                logger.warning(
                    "Removed %d corrupted ResponseStore SQLite copies",
                    len(corrupt_copies),
                )

            result: list[SessionState] = []
            for state, version in rows:
                if (
                    self._latest_versions.get(state.session_id) == version
                    and self._committed_versions.get(state.session_id) == version
                ):
                    result.append(state)
            return result

    async def is_complete(self, session_id: str) -> bool:
        """
        给 Memory 淘汰模块使用的轻量接口。

        推荐只淘汰 is_complete(session_id) == True 的 Memory Session，
        这样不会出现 Memory 刚淘汰、Disk 最新版本还未提交的短暂空窗。
        """
        return await self._get_complete_version(session_id) is not None

    async def ensure_eviction_copy(
        self,
        session_id: str,
        expected_response_id: str,
        now: int,
    ) -> DiskCopyResult:
        """等待并验证淘汰所需的同版本 Disk 副本，不刷新 Disk TTL。"""
        self._ensure_open()
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self._recovery_wait_timeout_seconds

        async with self._commit_condition:
            while True:
                latest = self._latest_versions.get(session_id)
                if latest is None:
                    return DiskCopyResult(DiskCopyStatus.NOT_FOUND)

                if self._latest_response_ids.get(session_id) != expected_response_id:
                    return DiskCopyResult(DiskCopyStatus.RESPONSE_CHANGED)

                if self._failed_versions.get(session_id) == latest:
                    return DiskCopyResult(DiskCopyStatus.FAILED)

                if self._committed_versions.get(session_id) == latest:
                    expected_version = latest
                    break

                self._writer_event.set()
                remaining = deadline - loop.time()
                if remaining <= 0:
                    return DiskCopyResult(DiskCopyStatus.PENDING)

                try:
                    await asyncio.wait_for(
                        self._commit_condition.wait(),
                        timeout=remaining,
                    )
                except asyncio.TimeoutError:
                    return DiskCopyResult(DiskCopyStatus.PENDING)

        try:
            async with self._io_lock:
                status = await asyncio.to_thread(
                    self._inspect_eviction_copy_sync,
                    session_id,
                    expected_response_id,
                    expected_version,
                    now,
                )
        except sqlite3.Error:
            logger.exception("Failed to inspect ResponseStore SQLite copy")
            return DiskCopyResult(DiskCopyStatus.FAILED)

        async with self._meta_lock:
            if (
                self._latest_versions.get(session_id) != expected_version
                or self._committed_versions.get(session_id) != expected_version
                or self._latest_response_ids.get(session_id) != expected_response_id
            ):
                return DiskCopyResult(DiskCopyStatus.VERSION_CHANGED)

        return DiskCopyResult(status)

    async def evict_if_unchanged(
        self,
        session_id: str,
        expected_response_id: str,
    ) -> DiskEvictionResult:
        """response_id 未变化时删除 Disk 副本并返回实际大小。"""
        self._ensure_open()

        async with self._io_lock, self._meta_lock:
            latest = self._latest_versions.get(session_id)
            if latest is None:
                return DiskEvictionResult(
                    session_id,
                    expected_response_id,
                    DiskEvictionStatus.NOT_FOUND,
                )
            if self._latest_response_ids.get(session_id) != expected_response_id:
                return DiskEvictionResult(
                    session_id,
                    expected_response_id,
                    DiskEvictionStatus.RESPONSE_CHANGED,
                )
            if self._committed_versions.get(session_id) != latest:
                return DiskEvictionResult(
                    session_id,
                    expected_response_id,
                    DiskEvictionStatus.VERSION_CHANGED,
                )

            status, freed_bytes = await asyncio.to_thread(
                self._delete_if_unchanged_sync,
                session_id,
                expected_response_id,
                latest,
            )
            if status is not DiskEvictionStatus.EVICTED:
                return DiskEvictionResult(
                    session_id,
                    expected_response_id,
                    status,
                )

            self._discard_through_versions[session_id] = max(
                latest,
                self._discard_through_versions.get(session_id, 0),
            )
            self._pending.pop(session_id, None)
            self._latest_versions.pop(session_id, None)
            self._committed_versions.pop(session_id, None)
            self._latest_response_ids.pop(session_id, None)
            self._failed_versions.pop(session_id, None)
            self._prune_discard_tombstone_locked(session_id)
            self._commit_condition.notify_all()

            return DiskEvictionResult(
                session_id,
                expected_response_id,
                DiskEvictionStatus.EVICTED,
                freed_bytes=freed_bytes,
            )

    async def close(self) -> None:
        if self._closed:
            return

        # 不直接 cancel 正在 to_thread() 中执行的 SQLite 操作；取消 await
        # 并不能停止已经运行的工作线程。让 Writer 自然收尾后再 close。
        async with self._commit_condition:
            self._closing = True
            # 让所有等待 commit 的恢复请求及时退出。
            self._commit_condition.notify_all()

        self._writer_event.set()

        task = self._writer_task
        self._writer_task = None
        if task is not None:
            await task

        async with self._io_lock:
            await asyncio.to_thread(self._conn.close)
        self._closed = True

    def _initialize_database(self) -> None:
        cursor = self._conn.cursor()
        try:
            cursor.execute("PRAGMA journal_mode=WAL")
            # Disk 是当前进程生命周期内的可丢弃二级缓存，不要求 crash durability。
            cursor.execute("PRAGMA synchronous=NORMAL")

            # 明确不做重启恢复：启动时直接清空旧数据。
            cursor.execute("DROP TABLE IF EXISTS session_state")
            cursor.execute(
                """
                CREATE TABLE session_state (
                    session_id TEXT PRIMARY KEY,
                    response_id TEXT NOT NULL,
                    token_ids BLOB NOT NULL,
                    created_at INTEGER NOT NULL,
                    updated_at INTEGER NOT NULL,
                    disk_idle_expires_at INTEGER,
                    disk_size_bytes INTEGER NOT NULL,
                    write_version INTEGER NOT NULL
                )
                """
            )
            self._conn.commit()
        finally:
            cursor.close()

    def _ensure_open(self) -> None:
        if self._closed or self._closing:
            raise RuntimeError("SQLiteSessionStore is closing or already closed")

    def _ensure_writer_started(self) -> None:
        if self._writer_task is None or self._writer_task.done():
            # 创建后台Writer Task
            self._writer_task = asyncio.create_task(
                self._writer_loop(), name="responses-sqlite-session-writer"
            )

    def _allocate_version_locked(self) -> int:
        self._version_counter += 1
        return self._version_counter

    def _prune_discard_tombstone_locked(self, session_id: str) -> None:
        discarded_through = self._discard_through_versions.get(session_id)
        if discarded_through is None:
            return

        inflight_version = self._inflight_versions.get(session_id)
        if inflight_version is not None and inflight_version <= discarded_through:
            return

        self._discard_through_versions.pop(session_id, None)

    def _invalidate_corrupt_copy_locked(self, corrupt_copy: _CorruptDiskCopy) -> None:
        session_id = corrupt_copy.session_id
        if session_id not in self._committed_versions:
            return

        latest = self._latest_versions.get(session_id, corrupt_copy.write_version)
        self._discard_through_versions[session_id] = max(
            latest,
            self._discard_through_versions.get(session_id, 0),
        )
        self._pending.pop(session_id, None)
        self._latest_versions.pop(session_id, None)
        self._committed_versions.pop(session_id, None)
        self._latest_response_ids.pop(session_id, None)
        self._failed_versions.pop(session_id, None)
        self._prune_discard_tombstone_locked(session_id)
        self._commit_condition.notify_all()

    async def _writer_loop(self) -> None:
        while True:
            await self._writer_event.wait()
            if not self._closing:
                # 短合并窗口：多个 save 尽量进入一个 SQLite transaction。
                await asyncio.sleep(self._write_interval_seconds)

            async with self._meta_lock:
                batch = self._pending
                self._pending = {}
                self._writer_event.clear()
                self._inflight_versions.update(
                    (session_id, pending.version)
                    for session_id, pending in batch.items()
                )

            if not batch:
                if self._closing:
                    return
                continue

            try:
                try:
                    async with self._io_lock:
                        # 只过滤 delete 已明确作废的批次。不能因为出现更新版本就
                        # 跳过旧批次，否则下一批 delta 会失去落盘基础。
                        async with self._meta_lock:
                            valid = {
                                sid: p
                                for sid, p in batch.items()
                                if p.version
                                > self._discard_through_versions.get(sid, 0)
                            }

                        if valid:
                            await asyncio.to_thread(self._write_batch_sync, valid)

                        async with self._meta_lock:
                            for sid, pending in valid.items():
                                if pending.version > (
                                    self._discard_through_versions.get(sid, 0)
                                ):
                                    self._committed_versions[sid] = pending.version
                                    if (
                                        self._failed_versions.get(sid)
                                        == pending.version
                                    ):
                                        self._failed_versions.pop(sid, None)

                            if valid:
                                # 唤醒等待这些 Session 最新版本 commit 的
                                # get_state()。
                                self._commit_condition.notify_all()

                except Exception:
                    logger.exception("Failed to flush ResponseStore SQLite batch")
                    if not self._closing:
                        await self._mark_failed_batch(batch)
                        await self._requeue_failed_batch(batch)
            finally:
                async with self._meta_lock:
                    for session_id, pending in batch.items():
                        if self._inflight_versions.get(session_id) == pending.version:
                            self._inflight_versions.pop(session_id, None)
                        self._prune_discard_tombstone_locked(session_id)

            if self._closing:
                async with self._meta_lock:
                    if not self._pending:
                        return
                    self._writer_event.set()

    async def _requeue_failed_batch(self, batch: dict[str, _PendingWrite]) -> None:
        """事务失败后把旧 delta 放回 pending 前部，避免历史缺口。"""
        async with self._meta_lock:
            for sid, failed in batch.items():
                if failed.version <= self._discard_through_versions.get(sid, 0):
                    continue

                current = self._pending.get(sid)
                if current is None:
                    self._pending[sid] = failed
                elif not current.replace_existing:
                    current.delta_token_ids = (
                        failed.delta_token_ids + current.delta_token_ids
                    )
                    current.created_at = min(failed.created_at, current.created_at)
                    current.replace_existing = failed.replace_existing

            if self._pending:
                self._writer_event.set()

    async def _mark_failed_batch(self, batch: dict[str, _PendingWrite]) -> None:
        async with self._commit_condition:
            for session_id, pending in batch.items():
                if pending.version > self._discard_through_versions.get(session_id, 0):
                    self._failed_versions[session_id] = pending.version
            self._commit_condition.notify_all()

    async def _wait_for_complete_version(
        self,
        session_id: str,
        timeout_seconds: float,
    ) -> int | None:
        """
        等待指定 Session 的“当前最新版本”完成 commit。

        返回：
            int:
                当前 latest_version 已经完整 commit。

            None:
                Session 不存在、已删除/作废、Store 正在关闭，
                或等待超过 timeout_seconds。

        这里等待的是“当前最新版本”，而不是固定某个旧版本。
        如果等待期间又发生 save，latest_version 前进，会继续等待新的
        latest_version，避免刚等到 v5 又把过期 v5 恢复进 Memory。
        """
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout_seconds

        async with self._commit_condition:
            while True:
                if self._closing or self._closed:
                    return None

                latest = self._latest_versions.get(session_id)
                if latest is None:
                    return None

                if latest <= self._discard_through_versions.get(session_id, 0):
                    return None

                if self._committed_versions.get(session_id) == latest:
                    return latest

                remaining = deadline - loop.time()
                if remaining <= 0:
                    return None

                try:
                    await asyncio.wait_for(
                        self._commit_condition.wait(),
                        timeout=remaining,
                    )
                except asyncio.TimeoutError:
                    return None

    async def _get_complete_version(self, session_id: str) -> int | None:
        async with self._meta_lock:
            latest = self._latest_versions.get(session_id)
            if latest is None or self._committed_versions.get(session_id) != latest:
                return None
            if latest <= self._discard_through_versions.get(session_id, 0):
                return None
            return latest

    def _write_batch_sync(self, batch: dict[str, _PendingWrite]) -> None:
        cursor = self._conn.cursor()
        try:
            cursor.execute("BEGIN IMMEDIATE")
            for pending in batch.values():
                row = cursor.execute(
                    """
                    SELECT token_ids, created_at FROM session_state
                    WHERE session_id = ?
                    """,
                    (pending.session_id,),
                ).fetchone()

                plaintext_blob = self._encode_token_ids(pending.delta_token_ids)
                encryption_started_ns = time.perf_counter_ns()
                delta_blob = self._cipher.encrypt(
                    plaintext_blob,
                    self._encryption_context(pending.session_id),
                )
                self._metrics.record_encryption(
                    plaintext_bytes=len(plaintext_blob),
                    duration_ns=time.perf_counter_ns() - encryption_started_ns,
                )
                if row is None or pending.replace_existing:
                    token_blob = delta_blob
                    created_at = pending.created_at
                else:
                    # 不解密历史 token；直接追加独立密文帧。
                    token_blob = bytes(row[0]) + delta_blob
                    created_at = int(row[1])

                disk_size = self._estimate_disk_size_bytes(
                    pending.session_id, pending.response_id, token_blob
                )

                cursor.execute(
                    """
                    INSERT INTO session_state (
                        session_id, response_id, token_ids, created_at, updated_at,
                        disk_idle_expires_at, disk_size_bytes, write_version
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(session_id) DO UPDATE SET
                        response_id = excluded.response_id,
                        token_ids = excluded.token_ids,
                        updated_at = excluded.updated_at,
                        disk_idle_expires_at = excluded.disk_idle_expires_at,
                        disk_size_bytes = excluded.disk_size_bytes,
                        write_version = excluded.write_version
                    """,
                    (
                        pending.session_id,
                        pending.response_id,
                        token_blob,
                        created_at,
                        pending.updated_at,
                        pending.disk_idle_expires_at,
                        disk_size,
                        pending.version,
                    ),
                )
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise
        finally:
            cursor.close()

    def _read_state_sync(
        self, session_id: str, expected_version: int, now: int
    ) -> _ReadStateResult:
        cursor = self._conn.cursor()
        try:
            row = cursor.execute(
                """
                SELECT response_id, token_ids, created_at,
                       disk_size_bytes, write_version
                FROM session_state WHERE session_id = ?
                """,
                (session_id,),
            ).fetchone()
            if row is None or int(row[4]) != expected_version:
                return _ReadStateResult()

            try:
                token_ids = self._decrypt_token_ids(session_id, row[1])
            except DataDecryptionError:
                return _ReadStateResult(
                    corrupt_copy=_CorruptDiskCopy(session_id, expected_version)
                )

            disk_expires_at = self._build_expire_time(now)
            cursor.execute(
                """
                UPDATE session_state
                SET updated_at = ?, disk_idle_expires_at = ?
                WHERE session_id = ? AND write_version = ?
                """,
                (now, disk_expires_at, session_id, expected_version),
            )
            self._conn.commit()

            return _ReadStateResult(
                state=SessionState(
                    session_id=session_id,
                    response_id=str(row[0]),
                    token_ids=token_ids,
                    created_at=int(row[2]),
                    updated_at=now,
                    disk_idle_expires_at=disk_expires_at,
                    disk_size_bytes=int(row[3]),
                )
            )
        except Exception:
            self._conn.rollback()
            raise
        finally:
            cursor.close()

    def _exists_sync(self, session_id: str, expected_version: int) -> bool:
        row = self._conn.execute(
            """
            SELECT 1 FROM session_state
            WHERE session_id = ? AND write_version = ? LIMIT 1
            """,
            (session_id, expected_version),
        ).fetchone()
        return row is not None

    def _inspect_eviction_copy_sync(
        self,
        session_id: str,
        expected_response_id: str,
        expected_version: int,
        now: int,
    ) -> DiskCopyStatus:
        row = self._conn.execute(
            """
            SELECT response_id, token_ids, disk_idle_expires_at, write_version
            FROM session_state WHERE session_id = ?
            """,
            (session_id,),
        ).fetchone()
        if row is None:
            return DiskCopyStatus.NOT_FOUND
        if int(row[3]) != expected_version:
            return DiskCopyStatus.VERSION_CHANGED
        if str(row[0]) != expected_response_id:
            return DiskCopyStatus.RESPONSE_CHANGED

        expires_at = None if row[2] is None else int(row[2])
        if expires_at is not None and expires_at <= now:
            return DiskCopyStatus.EXPIRED

        try:
            self._decrypt_token_ids(session_id, row[1])
        except DataDecryptionError:
            return DiskCopyStatus.INVALID
        return DiskCopyStatus.READY

    def _delete_if_unchanged_sync(
        self,
        session_id: str,
        expected_response_id: str,
        expected_version: int,
    ) -> tuple[DiskEvictionStatus, int]:
        cursor = self._conn.cursor()
        try:
            row = cursor.execute(
                """
                SELECT response_id, disk_size_bytes, write_version
                FROM session_state WHERE session_id = ?
                """,
                (session_id,),
            ).fetchone()
            if row is None:
                return DiskEvictionStatus.NOT_FOUND, 0
            if str(row[0]) != expected_response_id:
                return DiskEvictionStatus.RESPONSE_CHANGED, 0
            if int(row[2]) != expected_version:
                return DiskEvictionStatus.VERSION_CHANGED, 0

            cursor.execute(
                """
                DELETE FROM session_state
                WHERE session_id = ? AND response_id = ? AND write_version = ?
                """,
                (session_id, expected_response_id, expected_version),
            )
            if cursor.rowcount == 0:
                return DiskEvictionStatus.VERSION_CHANGED, 0

            self._conn.commit()
            return DiskEvictionStatus.EVICTED, int(row[1])
        except Exception:
            self._conn.rollback()
            raise
        finally:
            cursor.close()

    def _delete_sync(self, session_id: str) -> bool:
        cursor = self._conn.cursor()
        try:
            cursor.execute(
                "DELETE FROM session_state WHERE session_id = ?", (session_id,)
            )
            deleted = cursor.rowcount > 0
            self._conn.commit()
            return deleted
        except Exception:
            self._conn.rollback()
            raise
        finally:
            cursor.close()

    def _delete_corrupt_copies_sync(
        self, corrupt_copies: list[_CorruptDiskCopy]
    ) -> None:
        cursor = self._conn.cursor()
        try:
            cursor.execute("BEGIN IMMEDIATE")
            cursor.executemany(
                """
                DELETE FROM session_state
                WHERE session_id = ? AND write_version = ?
                """,
                [(copy.session_id, copy.write_version) for copy in corrupt_copies],
            )
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise
        finally:
            cursor.close()

    def _used_bytes_sync(self) -> int:
        row = self._conn.execute(
            "SELECT COALESCE(SUM(disk_size_bytes), 0) FROM session_state"
        ).fetchone()
        return int(row[0])

    def _list_sync(
        self,
    ) -> tuple[list[tuple[SessionState, int]], list[_CorruptDiskCopy]]:
        rows = self._conn.execute(
            """
            SELECT session_id, response_id, token_ids, created_at, updated_at,
                   disk_idle_expires_at, disk_size_bytes, write_version
            FROM session_state
            """
        ).fetchall()

        result: list[tuple[SessionState, int]] = []
        corrupt_copies: list[_CorruptDiskCopy] = []
        for row in rows:
            session_id = str(row[0])
            write_version = int(row[7])
            try:
                token_ids = self._decrypt_token_ids(session_id, row[2])
            except DataDecryptionError:
                corrupt_copies.append(_CorruptDiskCopy(session_id, write_version))
                continue

            state = SessionState(
                session_id=session_id,
                response_id=str(row[1]),
                token_ids=token_ids,
                created_at=int(row[3]),
                updated_at=int(row[4]),
                disk_idle_expires_at=None if row[5] is None else int(row[5]),
                disk_size_bytes=int(row[6]),
            )
            result.append((state, write_version))
        return result, corrupt_copies

    def _build_expire_time(self, now: int) -> int | None:
        if self._disk_idle_ttl_seconds is None:
            return None
        return now + self._disk_idle_ttl_seconds

    @classmethod
    def _encode_token_ids(cls, token_ids: list[int]) -> bytes:
        values = array.array(cls._TOKEN_TYPECODE, token_ids)
        if sys.byteorder != "little":
            values.byteswap()
        return values.tobytes()

    @classmethod
    def _decode_token_ids(cls, data: bytes) -> list[int]:
        if len(data) % 4 != 0:
            raise ValueError("invalid token_ids blob length")
        values = array.array(cls._TOKEN_TYPECODE)
        values.frombytes(data)
        if sys.byteorder != "little":
            values.byteswap()
        return values.tolist()

    def _decrypt_token_ids(self, session_id: str, data: bytes) -> list[int]:
        try:
            plaintext = self._cipher.decrypt(
                bytes(data), self._encryption_context(session_id)
            )
            return self._decode_token_ids(plaintext)
        except DataDecryptionError:
            raise
        except (TypeError, ValueError) as exc:
            raise DataDecryptionError("invalid encrypted token data") from exc

    @staticmethod
    def _encryption_context(session_id: str) -> bytes:
        return session_id.encode("utf-8")

    @staticmethod
    def _estimate_disk_size_bytes(
        session_id: str, response_id: str, token_blob: bytes
    ) -> int:
        """逻辑大小估算，不追求精确等于 SQLite page 占用。"""
        fixed_integer_bytes = 5 * 8
        return (
            len(session_id.encode("utf-8"))
            + len(response_id.encode("utf-8"))
            + len(token_blob)
            + fixed_integer_bytes
        )
