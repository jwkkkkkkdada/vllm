# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""CLI configuration helpers for the tiered Responses token store.

This module only defines and validates configuration. It deliberately does not
register with vLLM's global parser, create stores, or start cleanup tasks.
"""

from __future__ import annotations

import argparse
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .cleanup import PeriodicCleanupConfig

_MIB = 1024 * 1024


@dataclass(frozen=True, slots=True)
class ResponsesStoreConfig:
    """Validated configuration for the tiered Responses token store."""

    enabled: bool
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
    num_shards: int
    disk_write_interval_seconds: float
    disk_enabled: bool = True
    key_file: str | None = None

    @classmethod
    def from_cli_args(cls, args: argparse.Namespace) -> ResponsesStoreConfig:
        """Convert CLI-friendly units into the store's internal units."""
        if (
            args.responses_store_memory_capacity_mb <= 0
            or args.responses_store_disk_capacity_mb <= 0
        ):
            raise ValueError("responses store capacities must be greater than 0")
        if not (
            0
            <= args.responses_store_memory_low_watermark
            < args.responses_store_memory_high_watermark
            <= 1
        ) or not (
            0
            <= args.responses_store_disk_low_watermark
            < args.responses_store_disk_high_watermark
            <= 1
        ):
            raise ValueError(
                "responses store watermarks must satisfy 0 <= low < high <= 1"
            )
        if (
            args.responses_store_memory_ttl_seconds < 0
            or args.responses_store_disk_ttl_seconds < 0
        ):
            raise ValueError("responses store TTL must be non-negative")
        if args.responses_store_cleanup_interval_seconds <= 0:
            raise ValueError("cleanup interval must be greater than 0")
        if args.responses_store_cleanup_max_candidates <= 0:
            raise ValueError("cleanup max candidates must be greater than 0")
        if args.responses_store_cleanup_max_bytes_mb <= 0:
            raise ValueError("cleanup max bytes must be greater than 0")
        if args.responses_store_num_shards <= 0:
            raise ValueError("responses store num shards must be greater than 0")
        if args.responses_store_disk_write_interval_seconds <= 0:
            raise ValueError("disk write interval must be greater than 0")

        key_file = getattr(args, "responses_store_key_file", None)
        if key_file is not None:
            if not getattr(args, "responses_store_disk_enabled", True):
                raise ValueError(
                    "responses store key file requires the disk tier to be enabled"
                )
            if not args.responses_store_disk_path:
                raise ValueError(
                    "responses store key file requires an explicit disk path"
                )
            if args.responses_store_disk_path == ":memory:":
                raise ValueError(
                    "responses store key management requires a persistent disk path"
                )

        memory_capacity_bytes = args.responses_store_memory_capacity_mb * _MIB
        disk_capacity_bytes = args.responses_store_disk_capacity_mb * _MIB

        return cls(
            enabled=args.enable_responses_store,
            disk_path=args.responses_store_disk_path or _default_disk_path(),
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
            memory_ttl_seconds=args.responses_store_memory_ttl_seconds or None,
            disk_ttl_seconds=args.responses_store_disk_ttl_seconds or None,
            cleanup_interval_seconds=args.responses_store_cleanup_interval_seconds,
            cleanup_max_candidates=args.responses_store_cleanup_max_candidates,
            cleanup_max_bytes=args.responses_store_cleanup_max_bytes_mb * _MIB,
            num_shards=args.responses_store_num_shards,
            disk_write_interval_seconds=(
                args.responses_store_disk_write_interval_seconds
            ),
            disk_enabled=getattr(args, "responses_store_disk_enabled", True),
            key_file=key_file,
        )

    def build_cleanup_config(self) -> PeriodicCleanupConfig:
        """Build the configuration consumed by periodic cleanup."""
        from .cleanup import PeriodicCleanupConfig, TierCleanupConfig
        from .eviction import CapacityWaterMarks, EvictionSelectionBudget

        budget = EvictionSelectionBudget(
            max_candidates=self.cleanup_max_candidates,
            max_bytes=self.cleanup_max_bytes,
        )
        return PeriodicCleanupConfig(
            interval_seconds=self.cleanup_interval_seconds,
            memory=TierCleanupConfig(
                watermarks=CapacityWaterMarks(
                    max_bytes=self.memory_capacity_bytes,
                    high_watermark_bytes=self.memory_high_watermark_bytes,
                    low_watermark_bytes=self.memory_low_watermark_bytes,
                ),
                budget=budget,
            ),
            disk=TierCleanupConfig(
                watermarks=CapacityWaterMarks(
                    max_bytes=self.disk_capacity_bytes,
                    high_watermark_bytes=self.disk_high_watermark_bytes,
                    low_watermark_bytes=self.disk_low_watermark_bytes,
                ),
                budget=budget,
            ),
        )


def add_responses_store_cli_args(
    parser: argparse.ArgumentParser,
) -> argparse.ArgumentParser:
    """Register Responses token-store arguments without starting the store."""
    group = parser.add_argument_group("Responses token store")
    group.add_argument(
        "--enable-responses-store",
        action="store_true",
        help="Enable the Responses token store when integrated by the server.",
    )
    group.add_argument(
        "--responses-store-disk-enabled",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable SQLite as the Responses store's secondary storage tier.",
    )
    group.add_argument(
        "--responses-store-disk-path",
        default=None,
        help="SQLite path. Defaults to a process-local file in the temp directory.",
    )
    group.add_argument(
        "--responses-store-key-file",
        default=None,
        help=(
            "File containing a Base64-encoded 32-byte AES-256 key. Providing "
            "this option preserves key metadata across restarts and enables "
            "atomic database-wide key rotation every 90 days. Session data is "
            "cleared on restart. The file and its parent directory must be writable."
        ),
    )
    group.add_argument(
        "--responses-store-memory-capacity-mb",
        type=int,
        default=512,
        help=(
            "Memory capacity target for periodic cleanup, in MiB; "
            "writes are not rejected."
        ),
    )
    group.add_argument(
        "--responses-store-disk-capacity-mb",
        type=int,
        default=4096,
        help=(
            "Disk capacity target for periodic cleanup, in MiB; "
            "writes are not rejected."
        ),
    )
    group.add_argument(
        "--responses-store-memory-low-watermark",
        type=float,
        default=0.6,
        help="Memory cleanup target as a capacity ratio.",
    )
    group.add_argument(
        "--responses-store-memory-high-watermark",
        type=float,
        default=0.8,
        help="Memory pressure trigger as a capacity ratio.",
    )
    group.add_argument(
        "--responses-store-disk-low-watermark",
        type=float,
        default=0.7,
        help="Disk cleanup target as a capacity ratio.",
    )
    group.add_argument(
        "--responses-store-disk-high-watermark",
        type=float,
        default=0.9,
        help="Disk pressure trigger as a capacity ratio.",
    )
    group.add_argument(
        "--responses-store-memory-ttl-seconds",
        type=int,
        default=300,
        help="Memory idle TTL in seconds; 0 disables it.",
    )
    group.add_argument(
        "--responses-store-disk-ttl-seconds",
        type=int,
        default=3600,
        help="Disk idle TTL in seconds; 0 disables it.",
    )
    group.add_argument(
        "--responses-store-cleanup-interval-seconds",
        type=float,
        default=30.0,
        help="Interval between periodic cleanup runs.",
    )
    group.add_argument(
        "--responses-store-cleanup-max-candidates",
        type=int,
        default=128,
        help="Maximum candidates processed per tier and cleanup run.",
    )
    group.add_argument(
        "--responses-store-cleanup-max-bytes-mb",
        type=int,
        default=512,
        help="Maximum planned bytes reclaimed per tier and run, in MiB.",
    )
    group.add_argument(
        "--responses-store-num-shards",
        type=int,
        default=64,
        help="Number of per-session lock shards.",
    )
    group.add_argument(
        "--responses-store-disk-write-interval-seconds",
        type=float,
        default=0.05,
        help="Disk writer batching interval in seconds.",
    )
    return parser


def _default_disk_path() -> str:
    return str(
        Path(tempfile.gettempdir()) / f"vllm-responses-store-{os.getpid()}.sqlite3"
    )


__all__ = [
    "ResponsesStoreConfig",
    "add_responses_store_cli_args",
]
