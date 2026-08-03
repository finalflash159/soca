"""Exercise the production SessionMemory summary and private checkpoint."""

from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path

from soca.knowledge.vault import default_vault_root
from soca.memory.session import SessionMemory
from soca.memory.session_store import SessionCheckpointStore
from soca.memory.summary import (
    PRODUCTION_SUMMARY_MODEL_KEY,
    default_summary_model_root,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-root", type=Path, default=default_summary_model_root())
    parser.add_argument("--output", type=Path)
    parser.add_argument("--answer-smoke", action="store_true")
    args = parser.parse_args()
    memory = SessionMemory(
        thread_id="real-e2e-summary",
        max_chars=4000,
        summary_model_root=args.model_root,
    )
    for user, assistant in (
        ("Tôi muốn câu trả lời bằng tiếng Việt và ngắn gọn.", "Đã ghi nhận yêu cầu."),
        ("Dự án chọn TTS local để giữ riêng tư.", "Đã ghi nhận quyết định."),
        ("Đính chính: ta chọn TTS B, không phải TTS A.", "Đã cập nhật TTS B."),
        ("Việc còn lại là đo độ trễ voice trước khi bật mặc định.", "Đã ghi nhận việc còn lại."),
        ("Tôi đang ghi chú về ONNX Runtime cho RAG.", "Đã ghi nhận bối cảnh RAG."),
        ("Nhớ giúp tôi các điểm đó.", "Tôi sẽ giữ bối cảnh làm việc."),
    ):
        memory.append("user", user)
        memory.append("assistant", assistant)
    filler_index = 0
    while not memory.working.snapshot.pending_compaction:
        filler_index += 1
        memory.append(
            "user",
            (
                f"Trao đổi tạm thời số {filler_index}: "
                + "đây là chi tiết triển khai không tạo quyết định mới. " * 9
            ),
        )
        memory.append(
            "assistant",
            (
                f"Đã hiểu trao đổi tạm thời số {filler_index}. "
                + "Chưa có thay đổi nào đối với quyết định đã chốt. " * 9
            ),
        )
        if filler_index > 200:
            raise RuntimeError("automatic compaction did not start at the configured threshold")
    trigger_token_count = memory.working.snapshot.token_count
    accepted = memory.compaction_status()
    result = accepted
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        result = memory.compaction_status()
        if result.status != "running":
            break
        time.sleep(0.1)
    with tempfile.TemporaryDirectory(prefix="soca-real-e2e-") as directory:
        checkpoint = SessionCheckpointStore(directory)
        path = checkpoint.save(memory.working)
        restored = checkpoint.load(memory.working.thread_id)
        rendered = memory.render()
        payload = {
            "production_model": memory.summary_model_key,
            "trigger_token_count": trigger_token_count,
            "accepted": accepted.status,
            "final": result.status,
            "worker_state_after_job": memory.summary_worker_state,
            "worker_telemetry": memory.summary_telemetry,
            "summary_published": memory.working.snapshot.summary is not None,
            "checkpoint_mode": oct(path.stat().st_mode & 0o777),
            "checkpoint_round_trip": (
                restored is not None
                and restored.snapshot.summary == memory.working.snapshot.summary
            ),
            "rendered_state_present": "Earlier conversation state:" in rendered,
            "render_round_trip": (
                restored is not None and restored.render() == memory.working.render()
            ),
        }
    if args.answer_smoke:
        from soca.app.text_runtime import TextRuntimeConfig, build_text_runtime
        from soca.config import load_settings

        settings = load_settings()
        bundle = build_text_runtime(
            TextRuntimeConfig(
                vault=default_vault_root(),
                max_tokens=256,
                tool_router_mode="deterministic",
                semantic_router_enabled=False,
            ),
            session_memory=memory,
            llm_settings=settings,
        )
        answer = bundle.runtime.run_text_turn(
            "Hãy trả lời ngắn: trong hội thoại hiện tại chúng ta đã chọn TTS nào?",
            source="production_summary_smoke",
        )
        prompt = str(getattr(answer.llm_result, "prompt", ""))
        payload.update(
            {
                "answer_backend": settings.backend,
                "answer_provider": settings.provider_key if settings.backend == "remote" else "local",
                "answer_model": settings.model_id,
                "answer_used_llm": bool(answer.trace and answer.trace.used_llm),
                "answer_context_received": (
                    "Earlier conversation state:" in prompt and "TTS B" in prompt
                ),
                "answer_nonempty": bool(answer.response_text.strip()),
            }
        )
    memory.close()
    encoded = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
        args.output.chmod(0o600)
    else:
        print(encoded, end="")
    passed = (
        payload["production_model"] == PRODUCTION_SUMMARY_MODEL_KEY
        and payload["trigger_token_count"] >= 15_000
        and payload["accepted"] in {"running", "published"}
        and payload["final"] == "published"
        and payload["checkpoint_round_trip"]
        and payload["rendered_state_present"]
        and payload["render_round_trip"]
        and payload["worker_state_after_job"] == "idle"
    )
    if args.answer_smoke:
        passed = (
            passed
            and payload["answer_used_llm"]
            and payload["answer_context_received"]
            and payload["answer_nonempty"]
        )
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
