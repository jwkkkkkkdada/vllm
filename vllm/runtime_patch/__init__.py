# SPDX-License-Identifier: Apache-2.0

"""Runtime monkey patch for the Responses store extension.

Call :func:`apply` before importing vLLM CLI, OpenAI Responses, tokenizer, or
tool-parser modules.  The supported launcher does this automatically::

    python -m vllm.runtime_patch serve <model> [options]
"""

from .bootstrap import (
    BASE_COMMIT,
    PatchCompatibilityError,
    apply,
    is_applied,
    verify_compatibility,
)

__all__ = [
    "BASE_COMMIT",
    "PatchCompatibilityError",
    "apply",
    "is_applied",
    "verify_compatibility",
]
