# SPDX-License-Identifier: Apache-2.0

"""Install the runtime patch before test modules are collected."""

from vllm.runtime_patch import apply

apply()
