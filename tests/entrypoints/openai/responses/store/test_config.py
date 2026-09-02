# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse

import pytest

from vllm.entrypoints.openai.responses.store import (
    ResponsesStoreConfig,
    add_responses_store_cli_args,
)


def _parse_args(*args: str) -> argparse.Namespace:
    parser = add_responses_store_cli_args(argparse.ArgumentParser())
    return parser.parse_args(args)


def test_default_cli_config() -> None:
    config = ResponsesStoreConfig.from_cli_args(_parse_args())

    assert not config.enabled
    assert config.memory_capacity_bytes == 512 * 1024 * 1024
    assert config.disk_capacity_bytes == 4096 * 1024 * 1024
    assert config.memory_ttl_seconds == 300
    assert config.disk_ttl_seconds == 3600


def test_cli_config_converts_units_and_builds_cleanup_config(tmp_path) -> None:
    config = ResponsesStoreConfig.from_cli_args(
        _parse_args(
            "--enable-responses-store",
            "--responses-store-disk-path",
            str(tmp_path / "responses.sqlite3"),
            "--responses-store-memory-capacity-mb",
            "10",
            "--responses-store-disk-capacity-mb",
            "100",
            "--responses-store-memory-low-watermark",
            "0.5",
            "--responses-store-memory-high-watermark",
            "0.8",
            "--responses-store-memory-ttl-seconds",
            "0",
            "--responses-store-cleanup-max-bytes-mb",
            "8",
        )
    )

    assert config.enabled
    assert config.memory_capacity_bytes == 10 * 1024 * 1024
    assert config.memory_low_watermark_bytes == 5 * 1024 * 1024
    assert config.memory_high_watermark_bytes == 8 * 1024 * 1024
    assert config.memory_ttl_seconds is None

    cleanup_config = config.build_cleanup_config()
    assert cleanup_config.disk.watermarks.max_bytes == 100 * 1024 * 1024
    assert cleanup_config.memory.budget.max_bytes == 8 * 1024 * 1024


def test_cli_config_rejects_invalid_watermarks() -> None:
    args = _parse_args(
        "--responses-store-disk-low-watermark",
        "0.9",
        "--responses-store-disk-high-watermark",
        "0.8",
    )

    with pytest.raises(ValueError, match="responses store watermarks"):
        ResponsesStoreConfig.from_cli_args(args)
