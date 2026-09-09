"""ResponseStore 独立冒烟测试（不依赖 api_server 集成）。

在服务器上、正式启动 vllm 之前运行，用与 vllm serve 相同的 Python 环境：

    python store_smoke_test.py

预期输出：
    [1] save/get roundtrip OK, token count = 8
    [2] disk restore OK, token count = 8
    [3] delete OK
    ALL SMOKE TESTS PASSED

任一步骤抛异常，说明模块本身或依赖有问题，先解决再接入 api_server.py。
"""

import asyncio
import os
import tempfile

from vllm.entrypoints.openai.responses.store.disk import SQLiteSessionStore
from vllm.entrypoints.openai.responses.store.memory import MemorySessionStore
from vllm.entrypoints.openai.responses.store.tiered import TieredSessionStore


async def main() -> None:
    tmpdir = tempfile.mkdtemp(prefix="vllm-responses-smoke-")
    db_path = os.path.join(tmpdir, "smoke.sqlite3")

    memory = MemorySessionStore(max_capacity_bytes=64 * 1024 * 1024)
    disk = SQLiteSessionStore(
        db_path=db_path,
        disk_idle_ttl_seconds=3600,
        write_interval_seconds=0.05,
    )
    store = TieredSessionStore(memory_store=memory, disk_store=disk)

    # 1. 基本写入/读取（覆盖加密写入 + 后台 Writer 落盘）
    tokens = list(range(8))
    await store.save("sess-smoke", "resp-1", tokens)
    got = await store.get("sess-smoke")
    assert got == tokens, f"roundtrip mismatch: {got}"
    print(f"[1] save/get roundtrip OK, token count = {len(got)}")

    # 2. 等 Writer 落盘，然后淘汰 Memory 副本，验证 Disk 层可恢复
    for _ in range(100):
        if await disk.is_complete("sess-smoke"):
            break
        await asyncio.sleep(0.05)
    assert await disk.is_complete("sess-smoke"), "disk snapshot not committed"
    assert await memory.delete("sess-smoke")
    restored = await store.get("sess-smoke")
    assert restored == tokens, f"disk restore mismatch: {restored}"
    print(f"[2] disk restore OK, token count = {len(restored)}")

    # 3. 删除
    assert await store.delete("sess-smoke")
    assert not await store.exists("sess-smoke")
    print("[3] delete OK")

    await store.close()
    print("ALL SMOKE TESTS PASSED")


if __name__ == "__main__":
    asyncio.run(main())
