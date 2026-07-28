"""Exercise a real local summary worker, compaction, and private checkpoint.

This is an explicit benchmark/smoke command. It never selects a production
summary default and requires an already provisioned candidate.
"""

from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path

from soca.memory.compaction_coordinator import WorkingMemoryCompactionCoordinator
from soca.memory.session_store import SessionCheckpointStore
from soca.memory.summary import SUMMARY_MODEL_REGISTRY, LocalSummaryWorkerProcess
from soca.memory.working import WorkingMemory


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=sorted(SUMMARY_MODEL_REGISTRY), required=True)
    parser.add_argument("--model-root", type=Path, default=Path("models/summary"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    memory = WorkingMemory(thread_id="real-e2e-summary")
    for user, assistant in (
        ("Tôi muốn câu trả lời bằng tiếng Việt và ngắn gọn.", "Đã ghi nhận yêu cầu."),
        ("Dự án chọn TTS local để giữ riêng tư.", "Đã ghi nhận quyết định."),
        ("Đính chính: ta chọn TTS B, không phải TTS A.", "Đã cập nhật TTS B."),
        ("Việc còn lại là đo độ trễ voice trước khi bật mặc định.", "Đã ghi nhận việc còn lại."),
        ("Tôi đang ghi chú về ONNX Runtime cho RAG.", "Đã ghi nhận bối cảnh RAG."),
        ("Nhớ giúp tôi các điểm đó.", "Tôi sẽ giữ bối cảnh làm việc."),
    ):
        turn = memory.begin_turn(user)
        memory.finish_turn(turn.sequence, assistant)
    worker = LocalSummaryWorkerProcess(SUMMARY_MODEL_REGISTRY[args.model], model_root=args.model_root)
    coordinator = WorkingMemoryCompactionCoordinator(memory, worker)
    accepted = coordinator.request(manual=True)
    result = accepted
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        result = coordinator.poll()
        if result.status != "running":
            break
        time.sleep(0.1)
    with tempfile.TemporaryDirectory(prefix="soca-real-e2e-") as directory:
        checkpoint = SessionCheckpointStore(directory)
        path = checkpoint.save(memory)
        restored = checkpoint.load(memory.thread_id)
        rendered = memory.render()
        payload = {
            "accepted": accepted.status,
            "final": result.status,
            "worker_state_after_job": worker.status.state,
            "summary_published": memory.snapshot.summary is not None,
            "checkpoint_mode": oct(path.stat().st_mode & 0o777),
            "checkpoint_round_trip": restored is not None and restored.snapshot.summary == memory.snapshot.summary,
            "rendered_state_present": "Earlier conversation state:" in rendered,
            "render_round_trip": restored is not None and restored.render() == rendered,
        }
    encoded = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
        args.output.chmod(0o600)
    else:
        print(encoded, end="")
    passed = (
        payload["final"] == "published"
        and payload["checkpoint_round_trip"]
        and payload["rendered_state_present"]
        and payload["render_round_trip"]
        and payload["worker_state_after_job"] == "idle"
    )
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
