from __future__ import annotations

import subprocess
import sys


def test_qwen_worker_modules_do_not_import_main_asr_runtime() -> None:
    script = """
import sys

class BlockMainRuntime:
    def find_spec(self, fullname, path=None, target=None):
        blocked = ("onnxruntime", "sounddevice", "silero_vad")
        if fullname == blocked or fullname.startswith(tuple(name + "." for name in blocked)):
            raise RuntimeError(f"unexpected main-runtime import: {fullname}")
        return None

sys.meta_path.insert(0, BlockMainRuntime())
import soca.asr.qwen_backend
import soca.asr.qwen_ipc_protocol
import soca.asr.qwen_service_server
print("worker-import-ok")
"""

    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "worker-import-ok"
