from __future__ import annotations

from random import Random

import numpy as np

from soca.core import NullAudioPlayer, VoicePipeline
from soca.core.repair import (
    RepairAction,
    RepairCatalog,
    RepairKind,
    RepairState,
    RepairTimings,
    default_repair_catalog,
    kind_for_reason,
    plan_no_reply,
    plan_repair,
)
from tests.test_pipeline_integration import FakeASR, SpyLLM, SpyTTS


def test_default_catalog_loads_and_validates() -> None:
    catalog = default_repair_catalog()
    assert catalog.validate() == []
    # The playful "anyone there?" slot must exist with several variants to randomize.
    assert catalog.has(RepairKind.NO_INPUT, "attempt_1")
    assert catalog.has(RepairKind.NO_INPUT, "handover")


def test_no_input_attempt_1_includes_playful_greetings() -> None:
    catalog = default_repair_catalog()
    variants = catalog._slots["no_input.attempt_1"].variants
    blob = " ".join(variants).lower()
    # Foreign trend greetings the user asked for (moshi moshi / annyeong-family / alo).
    assert "moshi moshi" in blob
    assert "alo" in blob


def test_select_avoids_recently_used_variant() -> None:
    catalog = default_repair_catalog()
    rng = Random(0)
    first = catalog.select(RepairKind.NO_INPUT, "attempt_1", rng=rng)
    # With the just-used id excluded, the next pick must differ (slot has >1 variant).
    second = catalog.select(
        RepairKind.NO_INPUT,
        "attempt_1",
        rng=rng,
        recent_ids=(first.prompt_id,),
    )
    assert second.prompt_id != first.prompt_id


def test_select_falls_back_when_all_variants_recent() -> None:
    catalog = default_repair_catalog()
    ids = tuple(
        f"no_input.attempt_1#{i}"
        for i in range(len(catalog._slots["no_input.attempt_1"].variants))
    )
    choice = catalog.select(RepairKind.NO_INPUT, "attempt_1", rng=Random(1), recent_ids=ids)
    # Exhausted history must still return a valid variant, not crash.
    assert choice.text
    assert choice.kind == RepairKind.NO_INPUT


def test_kind_for_reason_maps_uncertain_vs_no_input() -> None:
    assert kind_for_reason("no_speech") == RepairKind.NO_INPUT
    assert kind_for_reason("") == RepairKind.NO_INPUT
    assert kind_for_reason("low_confidence:-0.90") == RepairKind.UNCERTAIN_INPUT
    assert kind_for_reason("compression_model_mismatch") == RepairKind.UNCERTAIN_INPUT


def test_plan_repair_escalates_no_input_then_handover() -> None:
    catalog = default_repair_catalog()
    state = RepairState()
    rng = Random(7)

    first = plan_repair(catalog, rejection_reason="no_speech", state=state, rng=rng)
    second = plan_repair(catalog, rejection_reason="no_speech", state=state, rng=rng)
    third = plan_repair(catalog, rejection_reason="no_speech", state=state, rng=rng)

    assert first.action == RepairAction.REPROMPT  # attempt_1
    assert second.action == RepairAction.CONTEXTUAL_REPROMPT  # attempt_2
    assert third.action == RepairAction.HANDOVER_TO_CHAT  # handover
    assert state.no_input_attempts == 3


def test_repair_state_reset_clears_ladder() -> None:
    state = RepairState()
    plan_repair(default_repair_catalog(), rejection_reason="no_speech", state=state, rng=Random(0))
    assert state.no_input_attempts == 1
    state.reset()
    assert state.no_input_attempts == 0


def test_pipeline_with_catalog_uses_varied_repair_text() -> None:
    catalog = default_repair_catalog()
    pipeline = VoicePipeline(
        asr=FakeASR("", rejection_reason="no_speech"),
        llm=SpyLLM(),
        tts=SpyTTS(),
        repair_catalog=catalog,
    )

    result = pipeline.turn(np.zeros(16000, dtype=np.float32))

    assert result.rejected is True
    # The spoken text now comes from the catalog, not the fixed reject_response.
    variants = catalog._slots["no_input.attempt_1"].variants
    assert result.response_text in variants
    assert result.response_text != pipeline.reject_response


def test_pipeline_without_catalog_keeps_fixed_reject_response() -> None:
    pipeline = VoicePipeline(asr=FakeASR("", rejection_reason="no_speech"), llm=SpyLLM(), tts=SpyTTS())
    result = pipeline.turn(np.zeros(16000, dtype=np.float32))
    assert result.response_text == pipeline.reject_response


def test_pipeline_streaming_repair_speaks_catalog_text() -> None:
    catalog = default_repair_catalog()
    pipeline = VoicePipeline(
        asr=FakeASR("", rejection_reason="no_speech"),
        llm=SpyLLM(),
        tts=SpyTTS(),
        repair_catalog=catalog,
    )

    events = list(
        pipeline.turn_streaming(np.zeros(16000, dtype=np.float32), audio_sink=NullAudioPlayer())
    )
    done = [event for event in events if event.type == "done"][0]

    assert done.metadata["rejected"] is True
    assert done.text in catalog._slots["no_input.attempt_1"].variants


def test_pipeline_turn_exposes_repair_fields() -> None:
    pipeline = VoicePipeline(
        asr=FakeASR("", rejection_reason="no_speech"),
        llm=SpyLLM(),
        tts=SpyTTS(),
        repair_catalog=default_repair_catalog(),
    )
    result = pipeline.turn(np.zeros(16000, dtype=np.float32))

    assert result.repair_kind == "no_input"
    assert result.repair_action == "reprompt"
    assert result.repair_attempt == 1
    assert result.handover_target is None


def test_pipeline_streaming_emits_repair_event_with_metadata() -> None:
    pipeline = VoicePipeline(
        asr=FakeASR("", rejection_reason="no_speech"),
        llm=SpyLLM(),
        tts=SpyTTS(),
        repair_catalog=default_repair_catalog(),
    )
    from soca.core import NullAudioPlayer

    events = list(
        pipeline.turn_streaming(np.zeros(16000, dtype=np.float32), audio_sink=NullAudioPlayer())
    )
    repair = next(event for event in events if event.type == "repair")
    assert repair.metadata["repair_kind"] == "no_input"
    assert repair.metadata["technical_reason"] == "no_speech"
    assert repair.text


def test_handover_on_third_no_input_sets_handover_target() -> None:
    catalog = default_repair_catalog()
    state = RepairState()
    rng = Random(0)
    for _ in range(2):
        plan_repair(catalog, rejection_reason="no_speech", state=state, rng=rng)
    third = plan_repair(catalog, rejection_reason="no_speech", state=state, rng=rng)
    assert third.action == RepairAction.HANDOVER_TO_CHAT


def test_plan_no_reply_ladder_when_waiting() -> None:
    timings = RepairTimings()
    # Quiet below the first threshold: stay silent.
    assert plan_no_reply(10_000, expects_response=True, attempts_fired=0, timings=timings) is None
    # Cross 45s -> gentle follow-up (once).
    assert plan_no_reply(50_000, expects_response=True, attempts_fired=0, timings=timings) == "no_reply_1"
    assert plan_no_reply(50_000, expects_response=True, attempts_fired=1, timings=timings) is None
    # Cross 120s -> guidance.
    assert plan_no_reply(130_000, expects_response=True, attempts_fired=1, timings=timings) == "no_reply_2"
    # Cross 300s -> sleep.
    assert plan_no_reply(310_000, expects_response=True, attempts_fired=2, timings=timings) == "sleep"


def test_plan_no_reply_passive_silence_never_speaks() -> None:
    timings = RepairTimings()
    # Not waiting on the user: no follow-up until the long passive-sleep mark.
    assert plan_no_reply(60_000, expects_response=False, attempts_fired=0, timings=timings) is None
    assert plan_no_reply(310_000, expects_response=False, attempts_fired=0, timings=timings) == "sleep"


def test_catalog_from_toml_can_be_built_standalone(tmp_path) -> None:
    toml = tmp_path / "repair.toml"
    toml.write_text(
        '[no_input.attempt_1]\naction = "reprompt"\nvariants = ["Alo?", "Có ai không?"]\n',
        encoding="utf-8",
    )
    catalog = RepairCatalog.from_toml(toml)
    assert catalog.validate() == []
    assert catalog.has(RepairKind.NO_INPUT, "attempt_1")
