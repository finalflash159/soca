from __future__ import annotations

from soca.llm.llamacpp_runner import LocalLlamaCppLLM


def test_chunk_text_reads_non_streaming_chat_message_content() -> None:
    response = {"choices": [{"message": {"content": '{"summary":"ok"}'}}]}
    assert LocalLlamaCppLLM._chunk_text(response) == '{"summary":"ok"}'
