# vLLM Responses runtime patch

This package reproduces the behavior of feature commit `343005a5792fa9147e54599f564fd1693d9dc0ad`
on top of untouched vLLM commit `752a3a504485790a2e8491cacbb35c137339ad34`.
It is a Python runtime monkey patch, not a Git patch.

## What it patches

- Responses token persistence, incremental-token reuse, session get/delete,
  lifecycle management, CLI configuration, and response/session headers.
- Responses custom-tool conversion and streaming events.
- DeepSeek V4 incremental chat-template tokenization.

The existing vLLM source files are checked against the base commit and are not
edited. At process startup an import hook loads the feature versions from
`overrides/` under their original module names. The ResponseStore implementation
lives under `store/` and is exposed in memory as
`vllm.entrypoints.openai.responses.store`.

## Install dependency

Follow the repository policy and use the vLLM virtual environment:

```bash
uv pip install -r vllm/runtime_patch/requirements.txt
```

## Start vLLM

Replace the normal `vllm serve ...` invocation with:

```bash
.venv/bin/python -m vllm.runtime_patch serve <model> [vllm options]
```

All existing options still work. ResponseStore options from the feature commit
are available, for example:

```bash
.venv/bin/python -m vllm.runtime_patch serve <model> \
  --responses-store-config '{"enabled": true}' \
  --responses-store-disk-path /var/lib/vllm/responses.sqlite3
```

For an embedded server, install the patch before importing any vLLM CLI,
Responses, tokenizer, or tool-parser module:

```python
from vllm.runtime_patch import apply

apply()

from vllm.entrypoints.openai.api_server import build_app
```

Calling `apply()` after a target module is imported fails deliberately because
late replacement would leave stale Pydantic models and FastAPI route schemas.
The compatibility check also fails if any original target file differs from the
supported base commit.

## Test

The feature tests are colocated with the patch and install it from their root
`conftest.py` before test modules are collected:

```bash
.venv/bin/python -m pytest vllm/runtime_patch/tests -v
```

The larger store acceptance suite can be run directly with pytest:

```bash
.venv/bin/python -c "from vllm.runtime_patch import apply; apply(); import pytest; raise SystemExit(pytest.main(['vllm/runtime_patch/store/integration/store_acceptance_test.py', '-v']))"
```
