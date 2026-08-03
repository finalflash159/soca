from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
ASSETS_DIR = REPO_ROOT / "docs" / "assets" / "benchmarks"
DATA_PATH = ASSETS_DIR / "figure_data.json"

# One accent is reserved for "this is what production runs" so a reader can find the
# shipped configuration in any figure without reading the caption first.
INK = "#22262b"
MUTED = "#7d868f"
GRID = "#dfe3e7"
PRODUCTION = "#c8912f"
CANDIDATE = "#5b7fa6"
ALTERNATE = "#a8b4c0"
WARN = "#b4483f"
GOOD = "#4f7d5c"


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": GRID,
            "axes.labelcolor": INK,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.labelsize": 9.5,
            "text.color": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 8.5,
            "legend.frameon": False,
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans"],
            "figure.dpi": 160,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.25,
        }
    )


def _despine(axis: Axes, *, left: bool = False) -> None:
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    if left:
        axis.spines["left"].set_visible(False)
        axis.tick_params(axis="y", length=0)


def _caption(figure: Figure, text: str) -> None:
    figure.text(0.0, -0.035, text, ha="left", va="top", fontsize=7.6, color=MUTED, wrap=True)


# --------------------------------------------------------------------------- ASR


def asr_production_disposition(data: dict) -> Figure:
    block = data["asr_production"]
    disposition = block["disposition"]
    comparison = block["wer_comparison"]

    figure, (left, right) = plt.subplots(1, 2, figsize=(10.2, 4.2), gridspec_kw={"width_ratios": [1, 1.05]})

    groups = disposition["groups"]
    positions = range(len(groups))
    correct = [item["correct"] for item in groups]
    wrong = [item["wrong"] for item in groups]
    left.bar(list(positions), correct, width=0.5, color=GOOD, label="correct")
    left.bar(list(positions), wrong, width=0.5, bottom=correct, color=WARN, label="incorrect")
    for index, item in enumerate(groups):
        total = item["correct"] + item["wrong"]
        left.text(
            index,
            item["correct"] / 2,
            f"{item['correct']} {item['correct_label']}",
            ha="center",
            va="center",
            fontsize=8.5,
            color="white",
            fontweight="bold",
        )
        note = f"{item['wrong']} {item['wrong_label']}"
        left.text(index, total + 1.2, note, ha="center", fontsize=8.5, color=WARN if item["wrong"] else GOOD)
    left.set_xticks(list(positions))
    left.set_xticklabels([item["label"] for item in groups])
    left.set_ylabel("clips")
    left.set_ylim(0, max(c + w for c, w in zip(correct, wrong, strict=True)) * 1.28)
    left.set_title("What the shipped ASR lets through")
    left.legend(loc="upper right")
    _despine(left)

    candidates = comparison["candidates"]
    cand_positions = range(len(candidates))
    values = [item["wer"] * 100 for item in candidates]
    colors = [PRODUCTION if item["shipped"] else ALTERNATE for item in candidates]
    right.bar(list(cand_positions), values, width=0.5, color=colors)
    for index, value in zip(cand_positions, values, strict=True):
        right.text(index, value + 0.35, f"{value:.2f}%", ha="center", fontsize=9.5, fontweight="bold")
    right.set_xticks(list(cand_positions))
    # The slice each bar was measured on rides along with its tick label, because the
    # slices differ and a bare bar chart would read as a controlled comparison.
    right.set_xticklabels(
        [f"{item['label']}\n{item['slice_short']}" for item in candidates], fontsize=7.6
    )
    right.set_ylabel("Vietnamese WER (%) ↓")
    right.set_ylim(0, max(values) * 1.22)
    right.set_title("Vietnamese WER across the available ASR options")
    _despine(right)

    _caption(
        figure,
        "Left: the production paired run. Every one of the five leaked non-speech rows was speech-like; pure-noise\n"
        "hallucination was 0/45, and no real utterance was falsely rejected. Right: the two lower-WER options are exactly\n"
        f"the ones whose release is blocked. {comparison['warning']}\n"
        "Source: BENCHMARKS.md, speech recognition.",
    )
    return figure


def asr_hallucination_ablation(data: dict) -> Figure:
    block = data["asr_hallucination_ablation"]
    configs = block["configs"]
    labels = [item["label"] for item in configs]
    halluc = [item["hallucination_rate"] * 100 for item in configs]
    wer = [item["wer"] * 100 for item in configs]
    positions = range(len(configs))

    figure, (top, bottom) = plt.subplots(
        2, 1, figsize=(7.6, 5.2), sharex=True, gridspec_kw={"height_ratios": [2, 1.15]}
    )

    colors = [WARN if value > 0.5 else GOOD for value in halluc]
    top.bar(positions, halluc, width=0.62, color=colors)
    for index, value in zip(positions, halluc, strict=True):
        top.text(index, value + 2.5, f"{value:.0f}%", ha="center", fontsize=8.5, fontweight="bold")
    top.set_ylabel("Hallucination rate on non-speech ↓")
    top.set_ylim(0, 115)
    top.set_yticks([0, 25, 50, 75, 100])
    top.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
    top.set_title("Anti-hallucination stages on PhoWhisper-tiny (200 speech / 50 non-speech)")
    _despine(top)

    # WER shares the x axis but gets its own panel: plotting a ~25% quantity and a
    # 0-100% quantity on one axis would make the WER differences invisible.
    bottom.bar(positions, wer, width=0.62, color=ALTERNATE)
    for index, value in zip(positions, wer, strict=True):
        bottom.text(index, value + 0.6, f"{value:.2f}%", ha="center", fontsize=8, color=INK)
    bottom.set_ylabel("WER on real speech ↓")
    bottom.set_ylim(23.5, 26.5)
    bottom.set_yticks([24, 25, 26])
    bottom.set_yticklabels(["24%", "25%", "26%"])
    bottom.set_xticks(list(positions))
    bottom.set_xticklabels(labels)
    _despine(bottom)

    _caption(
        figure,
        "Hallucination rate counts any non-empty output on non-speech, which credits VAD (it emits nothing) and\n"
        "under-credits BoH (it deletes matched phrases but leaves residue). Measured separately, BoH alone empties\n"
        "44% of non-speech outputs with 0/200 false positives on real speech. Source: BENCHMARKS.md appendix A.1.",
    )
    return figure


def asr_qwen_release_matrix(data: dict) -> Figure:
    block = data["qwen_release_matrix"]
    quality = block["quality"]
    latency = block["latency"]
    footprint = block["footprint"]

    # Latency and footprint get separate axes on purpose: milliseconds and megabytes
    # cannot share a scale, and the partial-latency gate applies to exactly one bar
    # group, so a full-width gate line would falsely imply startup is also failing it.
    figure, (left, right, far_right) = plt.subplots(
        1, 3, figsize=(11.6, 4.0), gridspec_kw={"width_ratios": [1.05, 1.05, 0.75]}
    )
    width = 0.36

    def _paired_bars(axis: Axes, section: dict, *, ylabel: str, title: str, scale: float = 1.0) -> None:
        positions = range(len(section["metrics"]))
        for offset, key in ((-width / 2, "qwen3_asr_0_6b"), (width / 2, "qwen3_asr_1_7b")):
            color = PRODUCTION if key.endswith("0_6b") else CANDIDATE
            values = [v * scale for v in section[key]]
            axis.bar(
                [p + offset for p in positions],
                values,
                width=width,
                color=color,
                label=("0.6B (release candidate)" if key.endswith("0_6b") else "1.7B (reference)")
                if title.startswith("Quality")
                else None,
            )
            for index, value in zip(positions, values, strict=True):
                text = f"{value:.1f}" if scale != 1.0 else f"{value:,.0f}"
                axis.text(index + offset, value * 1.02, text, ha="center", fontsize=7.8)
        axis.set_xticks(list(positions))
        axis.set_xticklabels(section["metrics"])
        axis.set_ylabel(ylabel)
        axis.set_title(title)
        _despine(axis)

    quality_view = dict(quality)
    quality_view["metrics"] = ["FLEURS-vi\nWER ↓", "code-switch\nWER ↓", "term\nrecall ↑"]
    _paired_bars(left, quality_view, ylabel="percent", title="Quality (pinned manifests)", scale=100.0)
    left.set_ylim(0, 68)
    left.legend(loc="upper left")

    _paired_bars(right, latency, ylabel="milliseconds ↓", title="Latency (20 start/stop cycles)")
    gate_index = latency["gate_applies_to_index"]
    gate = latency["gate_partial_latency_ms"]
    right.hlines(gate, gate_index - 0.5, gate_index + 0.5, color=WARN, linestyle="--", linewidth=1.2)
    right.text(
        gate_index - 0.5,
        gate + 90,
        f"gate {gate:,.0f} ms (partial only)",
        ha="left",
        va="bottom",
        fontsize=7.6,
        color=WARN,
    )
    right.set_ylim(0, 5600)

    _paired_bars(far_right, footprint, ylabel="megabytes ↓", title="Footprint")
    far_right.set_ylim(0, 5200)

    _caption(
        figure,
        "Both artifacts come from one pinned run on Apple M4 Pro. The 1.7B model is better on every quality metric and\n"
        "is still the reference profile, but its partial latency is 1.7x the 0.6B candidate and the release remains blocked\n"
        "on repeated full-stack memory evidence. Source: docs/evidence/qwen-asr-mps-20260803.json.",
    )
    return figure


# ------------------------------------------------------------------ conversation


def conversation_barge_in(data: dict) -> Figure:
    runs = data["turn_taking"]["barge_in"]["runs"]
    figure, axis = plt.subplots(figsize=(7.6, 4.2))

    positions = range(len(runs))
    width = 0.34
    detection = [item["detection"] * 100 for item in runs]
    false_interrupt = [item["false_interrupt"] * 100 for item in runs]

    axis.bar(
        [p - width / 2 for p in positions],
        detection,
        width=width,
        color=GOOD,
        label="detection of a real barge-in ↑",
    )
    axis.bar(
        [p + width / 2 for p in positions],
        false_interrupt,
        width=width,
        color=WARN,
        label="false interrupt (takeover rate) ↓",
    )
    for index in positions:
        axis.text(index - width / 2, detection[index] + 1.8, f"{detection[index]:.1f}%", ha="center", fontsize=9, fontweight="bold")
        axis.text(index + width / 2, false_interrupt[index] + 1.8, f"{false_interrupt[index]:.1f}%", ha="center", fontsize=9)

    axis.set_xticks(list(positions))
    axis.set_xticklabels([item["label"] for item in runs])
    axis.set_ylabel("percent")
    axis.set_ylim(0, 112)
    axis.set_title("Barge-in under real and synthetic echo")
    axis.legend(loc="upper right")
    _despine(axis)

    _caption(
        figure,
        "The two sets share no audio. Synthetic Vietnamese speech convolved with real measured room impulse responses\n"
        "reproduces the real-echo result within 0.2 pp on false interrupt and 2.2 pp on detection, which is what makes the\n"
        "synthesis usable as a stand-in. Source: BENCHMARKS.md, conversational robustness.",
    )
    return figure


def conversation_turn_taking(data: dict) -> Figure:
    block = data["turn_taking"]
    policies = block["policies"]
    names = list(policies)

    figure, (left, right) = plt.subplots(1, 2, figsize=(9.4, 4.0), gridspec_kw={"width_ratios": [1.35, 1]})

    metrics = [("cut_in_rate", "cut-in rate ↓"), ("premature_close_rate", "premature close ↓")]
    width = 0.34
    positions = range(len(metrics))
    for offset, name, color in ((-width / 2, names[0], ALTERNATE), (width / 2, names[1], PRODUCTION)):
        values = [policies[name][key] * 100 for key, _ in metrics]
        left.bar([p + offset for p in positions], values, width=width, label=name, color=color)
        for index, value in zip(positions, values, strict=True):
            left.text(index + offset, value + 1.6, f"{value:.1f}%", ha="center", fontsize=8.5)
    left.set_xticks(list(positions))
    left.set_xticklabels([label for _, label in metrics])
    left.set_ylabel("percent of 120 scenarios")
    left.set_ylim(0, 118)
    left.set_title("Turn-taking errors (800 ms within-turn pause)")
    left.legend(loc="upper right")
    _despine(left)

    # Short axis labels only: the full policy names already carry their colour in the
    # left panel's legend, and spelling them out again overflows this narrow axis.
    waits = [policies[name]["median_over_wait_ms"] for name in names]
    wait_positions = range(len(waits))
    right.bar(list(wait_positions), waits, width=0.45, color=[ALTERNATE, PRODUCTION])
    for index, value in zip(wait_positions, waits, strict=True):
        right.text(index, value + 45, f"{value:,} ms", ha="center", fontsize=9, fontweight="bold")
    right.set_xticks(list(wait_positions))
    right.set_xticklabels(["fixed", "p_based"])
    right.set_ylim(0, max(waits) * 1.25)
    right.set_ylabel("median over-wait (ms) ↓")
    right.set_title("Cost of that accuracy")
    _despine(right)

    _caption(
        figure,
        "Replacing a fixed 700 ms silence timer with Smart Turn v3.2 probability-based endpointing cuts interruptions of the\n"
        "user from 100% to 3.3% and premature closes from 61.7% to 18.3%, paid for with ~608 ms more patience. The residual\n"
        "18.3% is expected: Smart Turn is English-trained. Source: BENCHMARKS.md, conversational robustness.",
    )
    return figure


# --------------------------------------------------------------------------- TTS


def tts_release_and_first_clause(data: dict) -> Figure:
    block = data["tts"]
    figure, (left, right) = plt.subplots(1, 2, figsize=(9.8, 4.0))

    gates = block["release_gates"]
    labels = [item["metric"] for item in gates]
    # Gates are on four different scales, so each bar is drawn as a fraction of its
    # own gate. A bar below 1.0 passes; the raw value is column-aligned past the gate
    # rule so no label can ever be crossed by it.
    ratios = [item["measured"] / item["gate"] for item in gates]
    positions = range(len(gates))
    left.barh(list(positions), ratios, height=0.5, color=PRODUCTION)
    left.axvline(1.0, color=WARN, linestyle="--", linewidth=1.1)
    for index, item in enumerate(gates):
        unit = f" {item['unit']}" if item["unit"] else ""
        left.text(
            1.1,
            index,
            f"{item['measured']:g}{unit}   (gate {item['gate']:g}{unit})",
            va="center",
            ha="left",
            fontsize=8,
        )
    left.set_yticks(list(positions))
    left.set_yticklabels(labels)
    left.set_xlim(0, 2.45)
    left.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
    left.set_xlabel("measured ÷ gate  (lower is better)")
    left.set_title("Valtec ONNX release gates")
    # A clear row above the first bar is reserved for the gate annotation.
    left.set_ylim(len(gates) - 0.45, -1.05)
    left.text(1.0, -0.78, "release gate", fontsize=7.8, color=WARN, ha="center", va="center")
    _despine(left, left=True)

    ab = block["first_clause_ab"]
    deltas = ab["deltas"]
    positions = range(len(deltas))
    p50 = [item["p50_ms"] for item in deltas]
    lower = [item["p50_ms"] - item["min_ms"] for item in deltas]
    upper = [item["max_ms"] - item["p50_ms"] for item in deltas]
    right.bar(list(positions), p50, width=0.44, color=CANDIDATE, zorder=2)
    right.errorbar(
        list(positions), p50, yerr=[lower, upper], fmt="none", ecolor=INK, elinewidth=1.0, capsize=5, zorder=3
    )
    for index, item in enumerate(deltas):
        right.text(
            index,
            item["max_ms"] + 45,
            f"+{item['p50_ms']} ms",
            ha="center",
            fontsize=9,
            fontweight="bold",
        )
    right.axhline(0, color=GRID, linewidth=1.0)
    right.set_xticks(list(positions))
    right.set_xticklabels([item["label"] for item in deltas])
    right.set_ylabel("milliseconds saved ↑")
    right.set_ylim(-90, 1120)
    right.set_title(f"First-clause A/B ({ab['prompts_helped']}/{ab['prompts']} prompts helped)")
    _despine(right)

    _caption(
        figure,
        "Left: the shipped fp32 Valtec release against its four gates. Right: a controlled A/B that replays one recorded LLM\n"
        "token stream (same tokens, same inter-token delays) with first-clause flushing on and off, so the delta isolates the\n"
        "flush point rather than model speed. Bars are p50, whiskers the observed range. Source: BENCHMARKS.md, text to speech.",
    )
    return figure


# --------------------------------------------------------------------- retrieval


def retrieval_pareto(data: dict) -> Figure:
    block = data["retrieval_pareto"]
    figure, axis = plt.subplots(figsize=(8.0, 4.8))

    family_color = {"sparse": ALTERNATE, "dense": CANDIDATE, "fusion": "#8aa4bd", "production": PRODUCTION}
    seen: set[str] = set()
    for item in block["candidates"]:
        family = item["family"]
        is_production = family == "production"
        axis.scatter(
            item["p95_ms"],
            item["recall_at_5"] * 100,
            s=170 if is_production else 78,
            color=family_color[family],
            edgecolor="white" if is_production else "none",
            linewidth=1.4,
            zorder=3,
            label=family if family not in seen else None,
        )
        seen.add(family)
        offset = (item.get("dx", 0), item.get("dy", 10))
        # A leader line is drawn only when the label sits far enough away that the eye
        # could pair it with the wrong marker in the 70 ms cluster.
        needs_leader = abs(offset[0]) >= 16 or abs(offset[1]) >= 18
        axis.annotate(
            item["label"],
            (item["p95_ms"], item["recall_at_5"] * 100),
            textcoords="offset points",
            xytext=offset,
            ha=item.get("ha", "center"),
            va="center" if item.get("ha") in ("left", "right") and abs(offset[1]) < 18 else "bottom",
            fontsize=7.6,
            color=INK if is_production else MUTED,
            fontweight="bold" if is_production else "normal",
            arrowprops=(
                {"arrowstyle": "-", "color": GRID, "linewidth": 0.8, "shrinkA": 2, "shrinkB": 5}
                if needs_leader
                else None
            ),
        )

    axis.set_xscale("log")
    axis.set_xlabel("query latency p95 (ms, log scale) ↓")
    axis.set_ylabel("Recall@5 (%) ↑")
    axis.set_xlim(0.22, 500)
    axis.set_ylim(66, 96)
    axis.set_title("Vietnamese retrieval: quality against query cost (TVPL, 1,000 queries)")
    axis.grid(axis="y", color=GRID, linewidth=0.8)
    axis.set_axisbelow(True)
    axis.legend(loc="lower right")
    _despine(axis)

    _caption(
        figure,
        "Production is BM25 + AITeamVN/Vietnamese_Embedding_v2 with min-max linear fusion at dense weight 0.75.\n"
        "It costs ~71 ms p95 against BM25's 0.4 ms and buys +21.6 pp Recall@5; the release is deliberately accuracy-oriented.\n"
        "Reciprocal rank fusion lost to linear fusion at every dense backbone. Source: BENCHMARKS.md, knowledge retrieval.",
    )
    return figure


def retrieval_reranker(data: dict) -> Figure:
    block = data["retrieval_reranker"]
    figure, axis = plt.subplots(figsize=(8.4, 4.8))

    # Absolute recall is the wrong axis here: one domain saturates at 1.00, so three
    # configurations would land on the same point. What the decision actually turns on
    # is how much recall each reranker buys and what it charges in latency, so both are
    # plotted relative to the no-reranker base of the same domain.
    markers = ["o", "s"]
    colors = [CANDIDATE, "#8aa4bd"]
    for domain, marker, color in zip(block["domains"], markers, colors, strict=True):
        points = domain["points"]
        base = points[0]
        for index, item in enumerate(points[1:]):
            gain = (item["recall_at_5"] - base["recall_at_5"]) * 100
            cost = item["p95_ms"] - base["p95_ms"]
            axis.scatter(cost, gain, s=95, marker=marker, color=color, zorder=3)
            # Labels alternate above and below: on the saturated domain two
            # configurations land on the same gain and would otherwise overprint.
            axis.annotate(
                item["label"].replace("+", "").replace(" reranker", ""),
                (cost, gain),
                textcoords="offset points",
                xytext=(0, 11 if index % 2 == 0 else -18),
                ha="center",
                fontsize=7.6,
                color=MUTED,
            )
        axis.scatter([], [], marker=marker, color=color, s=95, label=domain["domain"])

    axis.scatter(0, 0, s=190, color=PRODUCTION, edgecolor="white", linewidth=1.5, zorder=4)
    axis.annotate(
        "no reranker\n(production)",
        (0, 0),
        textcoords="offset points",
        xytext=(14, -4),
        ha="left",
        va="center",
        fontsize=8,
        color=INK,
        fontweight="bold",
    )
    axis.axhline(0, color=GRID, linewidth=1.0)
    axis.set_xlabel("added query latency p95 (ms) ↓")
    axis.set_ylabel("Recall@5 gained over the same domain's base (pp) ↑")
    axis.set_xlim(-450, 5200)
    axis.set_ylim(-2.2, 13)
    axis.set_title("What reranking buys, and what it charges")
    axis.grid(axis="y", color=GRID, linewidth=0.8)
    axis.set_axisbelow(True)
    axis.legend(loc="upper left")
    _despine(axis)

    _caption(
        figure,
        "Every configuration charges between 1.5 and 4.6 seconds of p95 latency against a ~70 ms query encoder. The best\n"
        "one buys +10 pp on the harder domain but only +4 pp on the easier one, where the base is already at 0.96 — so the\n"
        "gain is real, unevenly distributed, and expensive. Production ships no reranker; the components stay available for\n"
        "evaluation. Source: BENCHMARKS.md, knowledge retrieval.",
    )
    return figure


def retrieval_evidence_floor(data: dict) -> Figure:
    block = data["evidence_floor"]
    points = block["operating_points"]
    answerable_total = block["answerable_total"]
    unanswerable_total = block["unanswerable_total"]

    figure, axis = plt.subplots(figsize=(7.4, 4.2))
    positions = range(len(points))
    width = 0.34

    recovered = [item["answerable_recovered"] for item in points]
    false_evidence = [item["false_evidence"] for item in points]

    axis.bar(
        [p - width / 2 for p in positions],
        recovered,
        width=width,
        color=PRODUCTION,
        label=f"answerable questions with accepted evidence (of {answerable_total}) ↑",
    )
    axis.bar(
        [p + width / 2 for p in positions],
        false_evidence,
        width=width,
        color=WARN,
        label=f"unanswerable questions given false evidence (of {unanswerable_total}) ↓",
    )
    for index in positions:
        axis.text(index - width / 2, recovered[index] + 0.25, str(recovered[index]), ha="center", fontsize=9, fontweight="bold")
        axis.text(index + width / 2, false_evidence[index] + 0.25, str(false_evidence[index]), ha="center", fontsize=9)

    axis.set_xticks(list(positions))
    axis.set_xticklabels([item["label"] for item in points])
    axis.set_ylabel("question count")
    axis.set_ylim(0, answerable_total + 1.8)
    axis.set_title("An evidence floor does not survive an embedding change")
    axis.legend(loc="upper left")
    _despine(axis)

    _caption(
        figure,
        "The 0.85 floor was calibrated for FastEmbed. Carried unchanged onto AITeamVN embeddings it admitted 1 of 12\n"
        f"answerable questions, even though raw retrieval found all 12. With only {unanswerable_total} negatives the Wilson 95% upper bound on\n"
        f"false evidence is still {block['wilson_95_upper_bound_false_evidence']:.1%}, so this is a calibration result, not a solved no-answer problem.\n"
        "On the real vault that same contract fails outright — see the failing gate. Source: BENCHMARKS.md, knowledge retrieval.",
    )
    return figure


# ------------------------------------------------------------- routing / summary


def routing_margin_sweep(data: dict) -> Figure:
    block = data["routing_margin_sweep"]
    points = block["points"]
    figure, axis = plt.subplots(figsize=(7.0, 4.0))

    margins = [item["margin"] for item in points]
    accuracy = [item["disposition_accuracy"] * 100 for item in points]
    colors = [PRODUCTION if item["selected"] else ALTERNATE for item in points]

    axis.plot(margins, accuracy, color=GRID, linewidth=1.4, zorder=1)
    axis.scatter(margins, accuracy, s=[150 if item["selected"] else 78 for item in points], color=colors, zorder=3)
    for margin, value, item in zip(margins, accuracy, points, strict=True):
        axis.annotate(
            f"{value:.2f}%" + ("\nselected" if item["selected"] else ""),
            (margin, value),
            textcoords="offset points",
            xytext=(0, 14),
            ha="center",
            fontsize=8.5,
            fontweight="bold" if item["selected"] else "normal",
            color=INK if item["selected"] else MUTED,
        )
    axis.set_xlabel("ambiguity margin")
    axis.set_ylabel("held-out disposition accuracy (%) ↑")
    axis.set_xticks(margins)
    axis.set_ylim(30, 95)
    axis.set_title("Semantic router margin sweep (threshold 0.58, held-out split)")
    axis.grid(axis="y", color=GRID, linewidth=0.8)
    axis.set_axisbelow(True)
    _despine(axis)

    _caption(
        figure,
        "Requiring a confidence margin between the top two dispositions was expected to trade recall for precision. It did\n"
        "not: a 0.02 margin nearly halves held-out accuracy because most correct decisions are themselves close calls. The\n"
        "margin is therefore pinned at 0.00 and the abstention job is left to the cascade. Source: BENCHMARKS.md, capability routing.",
    )
    return figure


def summary_bakeoff(data: dict) -> Figure:
    block = data["summary_bakeoff"]
    candidates = block["candidates"]
    # Both panels run horizontally and share the candidate axis: five quantized model
    # names do not fit as vertical tick labels without colliding.
    figure, (left, right) = plt.subplots(
        1, 2, figsize=(10.8, 4.4), sharey=True, gridspec_kw={"width_ratios": [0.8, 1.2]}
    )

    positions = range(len(candidates))
    schema = [item["schema"] * 100 for item in candidates]
    colors = [PRODUCTION if item["winner"] else ALTERNATE for item in candidates]
    left.barh(list(positions), schema, height=0.55, color=colors)
    left.axvline(100, color=WARN, linestyle="--", linewidth=1.1)
    for index, value in zip(positions, schema, strict=True):
        left.text(value - 3, index, f"{value:.1f}", ha="right", va="center", fontsize=8.5, color="white", fontweight="bold")
    left.set_yticks(list(positions))
    left.set_yticklabels([item["label"].replace("\n", " ") for item in candidates], fontsize=7.6)
    left.set_xlabel("schema-valid rate (%) ↑")
    left.set_xlim(0, 128)
    left.set_xticks([0, 50, 100])
    left.set_ylim(len(candidates) - 0.4, -0.9)
    left.text(100, -0.62, "hard gate 100%", ha="center", fontsize=7.6, color=WARN)
    left.set_title("A hard gate removes two candidates")
    _despine(left)

    series = [
        ("token_f1", "Token F1", CANDIDATE),
        ("rouge_l", "ROUGE-L F1", "#8aa4bd"),
        ("cosine", "Model2Vec cosine", GOOD),
    ]
    height = 0.24
    for offset, (key, label, color) in zip((height, 0.0, -height), series, strict=True):
        values = [item[key] for item in candidates]
        right.barh([p + offset for p in positions], values, height=height, label=label, color=color)
        for index, value in zip(positions, values, strict=True):
            right.text(value + 0.008, index + offset, f"{value:.3f}", va="center", fontsize=7)
    right.set_xlabel("score ↑")
    right.set_xlim(0, 0.92)
    right.set_title("No single similarity metric picks the winner")
    # Legend sits under the axis: every row already carries a long bar, so any in-axes
    # placement would cover a value label.
    right.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=3)
    _despine(right, left=True)
    # Mark the selected model on the shared axis.
    winner_index = next(index for index, item in enumerate(candidates) if item["winner"])
    for axis in (left, right):
        axis.axhspan(winner_index - 0.42, winner_index + 0.42, color=PRODUCTION, alpha=0.07, zorder=0)

    _caption(
        figure,
        "The two smallest candidates fail the schema gate outright, so their similarity scores never mattered. Among the\n"
        "three survivors Qwen2.5-3B leads on Token F1 and ROUGE-L while Qwen3-4B-Instruct-2507 leads on embedding cosine;\n"
        "the Instruct-2507 model was selected and its 8% recall on mixed Vietnamese/code/path content is recorded as debt\n"
        "rather than averaged away. Source: BENCHMARKS.md, working-memory summarization.",
    )
    return figure


FIGURES: dict[str, Callable[[dict], Figure]] = {
    "asr-production-disposition": asr_production_disposition,
    "asr-hallucination-ablation": asr_hallucination_ablation,
    "asr-qwen-release-matrix": asr_qwen_release_matrix,
    "conversation-barge-in": conversation_barge_in,
    "conversation-turn-taking": conversation_turn_taking,
    "tts-release-and-first-clause": tts_release_and_first_clause,
    "retrieval-pareto": retrieval_pareto,
    "retrieval-reranker": retrieval_reranker,
    "retrieval-evidence-floor": retrieval_evidence_floor,
    "routing-margin-sweep": routing_margin_sweep,
    "summary-bakeoff": summary_bakeoff,
}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Render the BENCHMARKS.md figures from the recorded values in figure_data.json."
    )
    parser.add_argument("--data", type=Path, default=DATA_PATH)
    parser.add_argument("--output-dir", type=Path, default=ASSETS_DIR)
    parser.add_argument("--only", action="append", default=[], choices=sorted(FIGURES))
    args = parser.parse_args()

    data = json.loads(args.data.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _style()

    for name in args.only or sorted(FIGURES):
        figure = FIGURES[name](data)
        destination = args.output_dir / f"{name}.png"
        figure.savefig(destination)
        plt.close(figure)
        print(f"wrote {destination.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
