# SPDX-License-Identifier: Apache-2.0

"""Install the Responses extension without editing existing vLLM modules."""

from __future__ import annotations

import hashlib
import importlib.abc
import importlib.util
import sys
from pathlib import Path

import vllm

BASE_COMMIT = "752a3a504485790a2e8491cacbb35c137339ad34"

_PATCH_ROOT = Path(__file__).resolve().parent
_VLLM_ROOT = Path(vllm.__file__).resolve().parent

# Each overlay is the complete module from the feature commit.  Loading it
# under the original module name preserves imports, annotations, and FastAPI /
# Pydantic registration semantics that cannot be reproduced safely by setting
# attributes after the route modules have already been imported.
_OVERLAYS: dict[str, tuple[str, str, str]] = {
    "vllm.entrypoints.cli.serve": (
        "entrypoints/cli/serve.py",
        "overrides/entrypoints/cli/serve.py",
        "d5e9b2bc874a946cbc422ae3a8e899845f98eb20",
    ),
    "vllm.entrypoints.openai.api_server": (
        "entrypoints/openai/api_server.py",
        "overrides/entrypoints/openai/api_server.py",
        "71dc508a34a77bd8957d0523b4fbacfd1828ed2c",
    ),
    "vllm.entrypoints.openai.cli_args": (
        "entrypoints/openai/cli_args.py",
        "overrides/entrypoints/openai/cli_args.py",
        "8dbb69943903895278213eb61eb03f85b73ee55c",
    ),
    "vllm.entrypoints.openai.responses.api_router": (
        "entrypoints/openai/responses/api_router.py",
        "overrides/entrypoints/openai/responses/api_router.py",
        "7f83a44e67e8aec32c2e3e7499af04359352b6a5",
    ),
    "vllm.entrypoints.openai.responses.protocol": (
        "entrypoints/openai/responses/protocol.py",
        "overrides/entrypoints/openai/responses/protocol.py",
        "ba8bc5a40f1bcffe8073cfdb4f0a8995da5e02e4",
    ),
    "vllm.entrypoints.openai.responses.serving": (
        "entrypoints/openai/responses/serving.py",
        "overrides/entrypoints/openai/responses/serving.py",
        "40a52012792096f2466cc04247606e0f80c236cd",
    ),
    "vllm.entrypoints.openai.responses.streaming_events": (
        "entrypoints/openai/responses/streaming_events.py",
        "overrides/entrypoints/openai/responses/streaming_events.py",
        "531a35c5722679d14dcfaf64def52afa37e44d48",
    ),
    "vllm.entrypoints.openai.responses.utils": (
        "entrypoints/openai/responses/utils.py",
        "overrides/entrypoints/openai/responses/utils.py",
        "07a9704f9b54ba63d963e721dfbbb6aeb63b762f",
    ),
    "vllm.entrypoints.serve.utils.server_utils": (
        "entrypoints/serve/utils/server_utils.py",
        "overrides/entrypoints/serve/utils/server_utils.py",
        "4479c576adaff04075ab81accdc0e498c887f53b",
    ),
    "vllm.tokenizers.deepseek_v4": (
        "tokenizers/deepseek_v4.py",
        "overrides/tokenizers/deepseek_v4.py",
        "2a6aaaf7397500d967ae78ed1c06474625666e4d",
    ),
    "vllm.tool_parsers.utils": (
        "tool_parsers/utils.py",
        "overrides/tool_parsers/utils.py",
        "95769bafd7f3a99efb8b53fd36a30a440fffd75a",
    ),
}

_STORE_MODULE = "vllm.entrypoints.openai.responses.store"


class PatchCompatibilityError(RuntimeError):
    """Raised when the patch is applied too late or to an unknown vLLM tree."""


def _git_blob_id(path: Path) -> str:
    data = path.read_bytes().replace(b"\r\n", b"\n")
    header = f"blob {len(data)}\0".encode()
    return hashlib.sha1(header + data).hexdigest()  # noqa: S324


def verify_compatibility() -> None:
    """Fail closed unless existing files match the supported base commit."""
    mismatches: list[str] = []
    for module, (source, _, expected_blob) in _OVERLAYS.items():
        path = _VLLM_ROOT / source
        actual = _git_blob_id(path) if path.is_file() else "missing"
        if actual != expected_blob:
            mismatches.append(f"{module}: expected {expected_blob}, got {actual}")

    if mismatches:
        details = "\n  - ".join(mismatches)
        raise PatchCompatibilityError(
            "Responses runtime patch only supports vLLM base commit "
            f"{BASE_COMMIT}. Mismatched modules:\n  - {details}"
        )


class _OverlayFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == _STORE_MODULE:
            store_dir = _PATCH_ROOT / "store"
            return importlib.util.spec_from_file_location(
                fullname,
                store_dir / "__init__.py",
                submodule_search_locations=[str(store_dir)],
            )

        overlay = _OVERLAYS.get(fullname)
        if overlay is None:
            return None
        return importlib.util.spec_from_file_location(
            fullname,
            _PATCH_ROOT / overlay[1],
        )


_FINDER = _OverlayFinder()


def is_applied() -> bool:
    """Return whether this process has installed the overlay finder."""
    return _FINDER in sys.meta_path


def apply(*, check_compatibility: bool = True) -> None:
    """Install the import-time monkey patch.

    This must run before any patched module is imported.  Import-time
    replacement is required because FastAPI inspects request annotations while
    registering routes and Pydantic finalizes model fields when defining the
    class.

    Args:
        check_compatibility: Validate the untouched vLLM modules against the
            supported base commit before installing the patch.

    Raises:
        PatchCompatibilityError: If a target module is already imported or the
            installed vLLM source is not the supported base revision.
    """
    if is_applied():
        return

    loaded = sorted(
        module
        for module in (*_OVERLAYS, _STORE_MODULE)
        if module in sys.modules
    )
    if loaded:
        raise PatchCompatibilityError(
            "Responses runtime patch must be applied before these modules are "
            f"imported: {', '.join(loaded)}"
        )

    if check_compatibility:
        verify_compatibility()
    sys.meta_path.insert(0, _FINDER)
