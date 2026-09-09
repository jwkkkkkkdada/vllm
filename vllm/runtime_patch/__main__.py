# SPDX-License-Identifier: Apache-2.0

"""Launch the vLLM CLI with the runtime patch installed."""

from . import apply


def main() -> None:
    apply()

    from vllm.entrypoints.cli.main import main as vllm_main

    vllm_main()


if __name__ == "__main__":
    main()
