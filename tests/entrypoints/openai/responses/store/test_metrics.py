from __future__ import annotations

from unittest.mock import patch

from vllm.entrypoints.openai.responses.store.metrics import (
    CleanupMetricsSnapshot,
    ResponseStoreMetrics,
    StoragePressureMetrics,
)


def _cleanup_snapshot() -> CleanupMetricsSnapshot:
    return CleanupMetricsSnapshot(
        duration_seconds=0.025,
        memory_pressure=StoragePressureMetrics(
            used_bytes=600,
            capacity_bytes=1_000,
        ),
        disk_pressure=StoragePressureMetrics(
            used_bytes=1_500,
            capacity_bytes=4_000,
        ),
        memory_freed_bytes=512,
        disk_freed_bytes=256,
    )


def test_log_cleanup_reports_pressure_encryption_speed_and_eviction() -> None:
    metrics = ResponseStoreMetrics()
    metrics.record_encryption(
        plaintext_bytes=1024**2,
        duration_ns=1_000_000_000,
    )

    with patch(
        "vllm.entrypoints.openai.responses.store.metrics.logger.info"
    ) as log_info:
        metrics.log_cleanup(_cleanup_snapshot())

    log_format, *log_args = log_info.call_args.args
    message = log_format % tuple(log_args)
    assert "event=response_store_metrics" in message
    assert "memory_usage_ratio=0.600000" in message
    assert "disk_usage_ratio=0.375000" in message
    assert "encryption_throughput_mib_per_second=1.000" in message
    assert "memory_freed_bytes=512" in message
    assert "disk_freed_bytes=256" in message
    assert "reason=" not in message
    assert "processed=" not in message
    assert "skipped=" not in message


def test_log_cleanup_resets_encryption_interval() -> None:
    metrics = ResponseStoreMetrics()
    metrics.record_encryption(plaintext_bytes=10, duration_ns=1)

    with patch(
        "vllm.entrypoints.openai.responses.store.metrics.logger.info"
    ) as log_info:
        metrics.log_cleanup(_cleanup_snapshot())
        metrics.log_cleanup(_cleanup_snapshot())

    log_format, *log_args = log_info.call_args_list[1].args
    message = log_format % tuple(log_args)
    assert "encryption_operations=0" in message
    assert "encryption_plaintext_bytes=0" in message
