# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import os
import tempfile
from argparse import Namespace
from dataclasses import dataclass
from pathlib import Path

from vllm.logger import init_logger

from .cleanup import (
    PeriodicCleanupConfig,
    PeriodicSessionStoreCleanup,
    TierCleanupConfig,
)
from .disk import SQLiteSessionStore
from .eviction import CapacityWaterMarks, EvictionSelectionBudget
from .memory import MemorySessionStore
from .metrics import ResponseStoreMetrics
from .tiered import TieredSessionStore

logger = init_logger(__name__)

_MIB = 1024 * 1024


@dataclass(frozen=True, slots=True)
class ResponsesStoreConfig:
    disk_path: str
    memory_capacity_bytes: int
    disk_capacity_bytes: int
    memory_low_watermark_bytes: int
    memory_high_watermark_bytes: int
    disk_low_watermark_bytes: int
    disk_high_watermark_bytes: int
    memory_ttl_seconds: int | None
    disk_ttl_seconds: int | None
    cleanup_interval_seconds: float
    cleanup_max_candidates: int
    cleanup_max_bytes: int

    @classmethod
    def from_args(cls, args: Namespace) -> ResponsesStoreConfig:
        memory_capacity_bytes = args.responses_store_memory_capacity_mb * _MIB
        disk_capacity_bytes = args.responses_store_disk_capacity_mb * _MIB
        return cls(
            disk_path=(
                args.responses_store_disk_path
                or str(
                    Path(tempfile.gettempdir())
                    / f"vllm-responses-store-{os.getpid()}.sqlite3"
                )
            ),
            memory_capacity_bytes=memory_capacity_bytes,
            disk_capacity_bytes=disk_capacity_bytes,
            memory_low_watermark_bytes=int(
                memory_capacity_bytes * args.responses_store_memory_low_watermark
            ),
            memory_high_watermark_bytes=int(
                memory_capacity_bytes * args.responses_store_memory_high_watermark
            ),
            disk_low_watermark_bytes=int(
                disk_capacity_bytes * args.responses_store_disk_low_watermark
            ),
            disk_high_watermark_bytes=int(
                disk_capacity_bytes * args.responses_store_disk_high_watermark
            ),
            memory_ttl_seconds=(args.responses_store_memory_ttl_seconds or None),
            disk_ttl_seconds=args.responses_store_disk_ttl_seconds or None,
            cleanup_interval_seconds=(args.responses_store_cleanup_interval_seconds),
            cleanup_max_candidates=(args.responses_store_cleanup_max_candidates),
            cleanup_max_bytes=(args.responses_store_cleanup_max_bytes_mb * _MIB),
        )

    def build_cleanup_config(self) -> PeriodicCleanupConfig:
        budget = EvictionSelectionBudget(
            max_candidates=self.cleanup_max_candidates,
            max_bytes=self.cleanup_max_bytes,
        )
        return PeriodicCleanupConfig(
            interval_seconds=self.cleanup_interval_seconds,
            memory=TierCleanupConfig(
                watermarks=CapacityWaterMarks(
                    max_bytes=self.memory_capacity_bytes,
                    low_watermark_bytes=self.memory_low_watermark_bytes,
                    high_watermark_bytes=self.memory_high_watermark_bytes,
                ),
                budget=budget,
            ),
            disk=TierCleanupConfig(
                watermarks=CapacityWaterMarks(
                    max_bytes=self.disk_capacity_bytes,
                    low_watermark_bytes=self.disk_low_watermark_bytes,
                    high_watermark_bytes=self.disk_high_watermark_bytes,
                ),
                budget=budget,
            ),
        )


def create_responses_store(
    args: Namespace,
) -> tuple[TieredSessionStore, PeriodicSessionStoreCleanup]:
    """Create the Responses token store and its periodic cleanup service."""
    config = ResponsesStoreConfig.from_args(args)
    metrics = ResponseStoreMetrics()
    memory_store = MemorySessionStore(
        max_capacity_bytes=config.memory_capacity_bytes,
        mem_idle_ttl_seconds=config.memory_ttl_seconds,
    )
    disk_store = SQLiteSessionStore(
        db_path=config.disk_path,
        disk_idle_ttl_seconds=config.disk_ttl_seconds,
        metrics=metrics,
    )
    store = TieredSessionStore(memory_store, disk_store)
    cleanup = PeriodicSessionStoreCleanup(
        store=store,
        config=config.build_cleanup_config(),
    )
    logger.info(
        "Responses token store enabled: memory_capacity_bytes=%d "
        "disk_capacity_bytes=%d cleanup_interval_seconds=%.3f",
        config.memory_capacity_bytes,
        config.disk_capacity_bytes,
        config.cleanup_interval_seconds,
    )
    return store, cleanup
