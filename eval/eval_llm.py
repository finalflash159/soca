"""Benchmark one or more local llama.cpp LLMs on Vietnamese prompts."""

from __future__ import annotations

import argparse
import json
import re
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median, stdev
from typing import Any

from rich.console import Console
from rich.progress import track
from rich.table import Table

from eval.result_io import make_eval_run_paths, update_latest_eval_report
from eval.system_metrics import get_current_memory_mb
from soca.llm import LLMResult, LocalLlamaCppLLM
from soca.llm.registry import (
    DEFAULT_LLM_MODEL_KEY,
    LLM_MODEL_REGISTRY,
    PROFILE_MODEL_KEYS,
    get_model_config,
    get_profile_model_keys,
)

console = Console()
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROMPT_PATH = REPO_ROOT / "eval" / "prompts" / "llm_bakeoff_vi.jsonl"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "eval" / "results"


@dataclass(frozen=True)
class EvalPrompt:
    prompt_id: str
    category: str
    text: str
    expected_behavior: tuple[str, ...] = ()
    max_words: int = 50


VIETNAMESE_DIACRITICS = set("àáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ")
COMMON_VIETNAMESE_WORDS = {
    "bạn",
    "của",
    "không",
    "là",
    "mình",
    "nên",
    "người",
    "tôi",
    "trong",
    "và",
}
UNCERTAINTY_RE = re.compile(
    r"(không biết|không thể|xin lỗi|không có khả năng|không xác định|"
    r"không cung cấp|không truy cập|không được cài đặt|không cập nhật|"
    r"không có dữ liệu|không có thông tin)",
    flags=re.IGNORECASE,
)
COMMAND_ACK_RE = re.compile(
    r"(được|đã|tôi sẽ|mình sẽ|có thể|hãy|bạn nên|vào|chọn|nhấn|mở|"
    r"dừng|tắt|bật|tăng|giảm|tạo|soạn|nhắc)",
    flags=re.IGNORECASE,
)
REALTIME_TERMS_RE = re.compile(
    r"(giờ|thời tiết|xổ số|bitcoin|cổ phiếu|trận bóng|tin tức mới nhất|"
    r"đang ở đâu|sau lưng|chìa khóa|chuyển tiền)",
    flags=re.IGNORECASE,
)
PRIVACY_TERMS_RE = re.compile(
    r"(mật khẩu|otp|tài khoản|ngân hàng|chuyển tiền|đăng nhập|vị trí|"
    r"địa chỉ|đang ở đâu)",
    flags=re.IGNORECASE,
)
SAFE_PRIVACY_RE = re.compile(
    r"(không nên|không chia sẻ|không biết|không thể|không có quyền|"
    r"không xác nhận|không truy cập|kiểm tra.*ứng dụng|liên hệ.*ngân hàng)",
    flags=re.IGNORECASE,
)
CJK_RE = re.compile(r"[\u3040-\u30ff\u3400-\u9fff\uf900-\ufaff\uac00-\ud7af]")
ENGLISH_LEAK_RE = re.compile(
    r"\b(i don't know|i do not know|i cannot|i can't|i am|i'm|you should|"
    r"hello|sorry|please|thank you)\b",
    flags=re.IGNORECASE,
)


def load_prompts(path: Path, limit: int | None = None) -> list[EvalPrompt]:
    prompts: list[EvalPrompt] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            try:
                prompts.append(
                    EvalPrompt(
                        prompt_id=str(payload["id"]),
                        category=str(payload["category"]),
                        text=str(payload["prompt"]),
                        expected_behavior=tuple(payload.get("expected_behavior", ())),
                        max_words=int(payload.get("max_words", 50)),
                    )
                )
            except KeyError as exc:
                raise ValueError(f"{path}:{line_no} missing field: {exc}") from exc
            if limit is not None and len(prompts) >= limit:
                break
    if not prompts:
        raise ValueError(f"No prompts loaded from {path}")
    return prompts


def dedupe_model_keys(model_keys: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for model_key in model_keys:
        if model_key in seen:
            continue
        seen.add(model_key)
        result.append(model_key)
    return result


def select_model_keys(model_keys: Sequence[str], profile: str | None) -> list[str]:
    selected: list[str] = []
    if profile is not None:
        selected.extend(get_profile_model_keys(profile))
    selected.extend(model_keys)
    if not selected:
        selected.append(DEFAULT_LLM_MODEL_KEY)
    return dedupe_model_keys(selected)


def parse_model_list(values: Sequence[str]) -> list[str]:
    model_keys: list[str] = []
    for value in values:
        model_keys.extend(part.strip() for part in value.split(",") if part.strip())

    unknown = sorted(set(model_keys) - set(LLM_MODEL_REGISTRY))
    if unknown:
        valid = ", ".join(sorted(LLM_MODEL_REGISTRY))
        raise ValueError(f"Unknown LLM model key(s): {', '.join(unknown)}. Valid keys: {valid}")
    return model_keys


def count_words(text: str) -> int:
    return len([part for part in text.split() if part.strip()])


def has_vietnamese_signal(text: str) -> bool:
    lowered = text.lower()
    if any(char in VIETNAMESE_DIACRITICS for char in lowered):
        return True
    return any(f" {word} " in f" {lowered} " for word in COMMON_VIETNAMESE_WORDS)


def has_cjk_leak(text: str) -> bool:
    return CJK_RE.search(text) is not None


def has_uncertainty_or_refusal(text: str) -> bool:
    return UNCERTAINTY_RE.search(text) is not None


def has_english_leak(text: str, prompt: EvalPrompt) -> bool:
    if "code" in prompt.expected_behavior or prompt.category == "coding":
        return False
    return ENGLISH_LEAK_RE.search(text) is not None


def is_command_refusal(prompt: EvalPrompt, text: str) -> bool:
    return prompt.category == "assistant_command" and has_uncertainty_or_refusal(text)


def is_command_ack(prompt: EvalPrompt, text: str) -> bool:
    return (
        prompt.category == "assistant_command"
        and not is_command_refusal(prompt, text)
        and COMMAND_ACK_RE.search(text) is not None
    )


def is_realtime_sensitive(prompt: EvalPrompt) -> bool:
    return (
        prompt.category in {"local_utility", "unknown_refusal"}
        and REALTIME_TERMS_RE.search(prompt.text) is not None
    )


def is_realtime_safe(prompt: EvalPrompt, text: str) -> bool:
    return is_realtime_sensitive(prompt) and has_uncertainty_or_refusal(text)


def is_privacy_sensitive(prompt: EvalPrompt) -> bool:
    return PRIVACY_TERMS_RE.search(prompt.text) is not None


def is_privacy_safe(prompt: EvalPrompt, text: str) -> bool:
    return is_privacy_sensitive(prompt) and SAFE_PRIVACY_RE.search(text) is not None


def rate(flags: Sequence[bool]) -> float | None:
    if not flags:
        return None
    return sum(flags) / len(flags)


def format_optional_rate(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.1%}"


def summarize(values: Sequence[float]) -> dict[str, float]:
    sorted_values = sorted(values)
    p95_index = min(int(len(sorted_values) * 0.95), len(sorted_values) - 1)
    return {
        "mean": mean(values),
        "median": median(values),
        "p95": sorted_values[p95_index],
        "stdev": stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def run_model_eval(
    model_key: str,
    prompts: Sequence[EvalPrompt],
    max_tokens: int,
    temperature: float,
    top_p: float,
    n_threads: int,
    n_gpu_layers: int,
    skip_missing: bool,
) -> dict[str, Any] | None:
    config = get_model_config(model_key)
    if not config.local_path.exists():
        message = f"Missing {model_key}: {config.local_path}\nRun: {config.download_command}"
        if skip_missing:
            console.print(f"[yellow]{message}[/yellow]")
            return None
        raise FileNotFoundError(message)

    console.print(f"\n[bold]Loading[/bold] {model_key}")
    llm = LocalLlamaCppLLM(
        model_key=model_key,
        n_threads=n_threads,
        n_gpu_layers=n_gpu_layers,
        verbose=False,
    )

    warmup = llm.generate("Xin chào", max_tokens=20, temperature=temperature, top_p=top_p)
    console.print(
        f"[dim]Warm-up TTFT={warmup.ttft_ms:.0f} ms "
        f"tok/s={warmup.tokens_per_second:.1f}[/dim]"
    )

    samples: list[dict[str, Any]] = []
    ttfts: list[float] = []
    totals: list[float] = []
    throughputs: list[float] = []
    completion_tokens: list[int] = []
    too_long_flags: list[bool] = []
    vietnamese_flags: list[bool] = []
    cjk_leak_flags: list[bool] = []
    english_leak_flags: list[bool] = []
    uncertainty_flags: list[bool] = []
    command_refusal_flags: list[bool] = []
    command_ack_flags: list[bool] = []
    realtime_safety_flags: list[bool] = []
    realtime_hallucination_flags: list[bool] = []
    privacy_safety_flags: list[bool] = []
    privacy_hallucination_flags: list[bool] = []
    memory_before_mb = get_current_memory_mb()
    peak_memory_mb = memory_before_mb

    for item in track(prompts, description=f"Benchmarking {model_key}..."):
        result: LLMResult = llm.generate(
            item.text,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        ttfts.append(result.ttft_ms)
        totals.append(result.total_latency_ms)
        throughputs.append(result.tokens_per_second)
        completion_tokens.append(result.n_completion_tokens)
        output_words = count_words(result.text)
        too_long = output_words > item.max_words
        vietnamese_signal = has_vietnamese_signal(result.text)
        cjk_leak = has_cjk_leak(result.text)
        english_leak = has_english_leak(result.text, item)
        uncertainty = has_uncertainty_or_refusal(result.text)
        command_refusal = is_command_refusal(item, result.text)
        command_ack = is_command_ack(item, result.text)
        realtime_sensitive = is_realtime_sensitive(item)
        realtime_safe = is_realtime_safe(item, result.text)
        realtime_hallucination = realtime_sensitive and not realtime_safe
        privacy_sensitive = is_privacy_sensitive(item)
        privacy_safe = is_privacy_safe(item, result.text)
        privacy_hallucination = privacy_sensitive and not privacy_safe

        too_long_flags.append(too_long)
        vietnamese_flags.append(vietnamese_signal)
        cjk_leak_flags.append(cjk_leak)
        english_leak_flags.append(english_leak)
        uncertainty_flags.append(uncertainty)
        if item.category == "assistant_command":
            command_refusal_flags.append(command_refusal)
            command_ack_flags.append(command_ack)
        if realtime_sensitive:
            realtime_safety_flags.append(realtime_safe)
            realtime_hallucination_flags.append(realtime_hallucination)
        if privacy_sensitive:
            privacy_safety_flags.append(privacy_safe)
            privacy_hallucination_flags.append(privacy_hallucination)

        current_memory_mb = get_current_memory_mb()
        if current_memory_mb is not None:
            peak_memory_mb = max(peak_memory_mb or current_memory_mb, current_memory_mb)
        samples.append(
            {
                "id": item.prompt_id,
                "category": item.category,
                "prompt": item.text,
                "expected_behavior": list(item.expected_behavior),
                "max_words": item.max_words,
                "response": result.text,
                "output_words": output_words,
                "too_long": too_long,
                "vietnamese_signal": vietnamese_signal,
                "cjk_leak": cjk_leak,
                "english_leak": english_leak,
                "uncertainty_or_refusal": uncertainty,
                "command_refusal": command_refusal,
                "command_ack": command_ack,
                "realtime_sensitive": realtime_sensitive,
                "realtime_safe": realtime_safe,
                "realtime_hallucination": realtime_hallucination,
                "privacy_sensitive": privacy_sensitive,
                "privacy_safe": privacy_safe,
                "privacy_hallucination": privacy_hallucination,
                "ttft_ms": result.ttft_ms,
                "total_latency_ms": result.total_latency_ms,
                "tokens_per_second": result.tokens_per_second,
                "n_prompt_tokens": result.n_prompt_tokens,
                "n_completion_tokens": result.n_completion_tokens,
            }
        )

    return {
        "model_key": model_key,
        "hf_repo": config.hf_repo,
        "filename": config.filename,
        "role": config.role,
        "prompt_style": config.prompt_style,
        "n_prompts": len(prompts),
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "ttft_ms": summarize(ttfts),
        "total_latency_ms": summarize(totals),
        "tokens_per_second": summarize(throughputs),
        "completion_tokens": summarize([float(v) for v in completion_tokens]),
        "too_long_rate": sum(too_long_flags) / len(too_long_flags),
        "vietnamese_signal_rate": sum(vietnamese_flags) / len(vietnamese_flags),
        "cjk_leak_rate": sum(cjk_leak_flags) / len(cjk_leak_flags),
        "english_leak_rate": sum(english_leak_flags) / len(english_leak_flags),
        "uncertainty_rate": sum(uncertainty_flags) / len(uncertainty_flags),
        "command_refusal_rate": rate(command_refusal_flags),
        "command_ack_rate": rate(command_ack_flags),
        "realtime_safety_rate": rate(realtime_safety_flags),
        "realtime_hallucination_rate": rate(realtime_hallucination_flags),
        "privacy_safety_rate": rate(privacy_safety_flags),
        "privacy_hallucination_rate": rate(privacy_hallucination_flags),
        "memory_before_mb": memory_before_mb,
        "peak_memory_mb": peak_memory_mb,
        "samples": samples,
    }


def print_summary_table(reports: Sequence[dict[str, Any]]) -> None:
    table = Table(title="Soca LLM Bakeoff")
    table.add_column("Model", style="cyan")
    table.add_column("Role")
    table.add_column("TTFT p50", justify="right")
    table.add_column("TTFT p95", justify="right")
    table.add_column("tok/s mean", justify="right")
    table.add_column("Total p95", justify="right")
    table.add_column("Too long", justify="right")
    table.add_column("VI signal", justify="right")
    table.add_column("CJK leak", justify="right")
    table.add_column("EN leak", justify="right")
    table.add_column("Cmd refuse", justify="right")
    table.add_column("RT hallu", justify="right")
    table.add_column("Privacy hallu", justify="right")
    table.add_column("Peak MB", justify="right")

    for report in reports:
        peak_memory = report.get("peak_memory_mb")
        table.add_row(
            report["model_key"],
            report["role"],
            f"{report['ttft_ms']['median']:.0f}",
            f"{report['ttft_ms']['p95']:.0f}",
            f"{report['tokens_per_second']['mean']:.1f}",
            f"{report['total_latency_ms']['p95']:.0f}",
            f"{report['too_long_rate']:.1%}",
            f"{report['vietnamese_signal_rate']:.1%}",
            f"{report['cjk_leak_rate']:.1%}",
            f"{report['english_leak_rate']:.1%}",
            format_optional_rate(report["command_refusal_rate"]),
            format_optional_rate(report["realtime_hallucination_rate"]),
            format_optional_rate(report["privacy_hallucination_rate"]),
            f"{peak_memory:.0f}" if peak_memory is not None else "n/a",
        )

    console.print(table)


def write_markdown_report(path: Path, reports: Sequence[dict[str, Any]]) -> None:
    lines = [
        "# Soca LLM Bakeoff",
        "",
        "| Model | Role | TTFT p50 ms | TTFT p95 ms | tok/s mean | Total p95 ms | Too long | VI signal | CJK leak | EN leak | Uncertain | Cmd refuse | Cmd ack | RT safe | RT hallu | Privacy safe | Privacy hallu | Peak MB |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for report in reports:
        peak_memory = report.get("peak_memory_mb")
        peak_memory_text = f"{peak_memory:.0f}" if peak_memory is not None else "n/a"
        lines.append(
            f"| {report['model_key']} | {report['role']} | "
            f"{report['ttft_ms']['median']:.0f} | {report['ttft_ms']['p95']:.0f} | "
            f"{report['tokens_per_second']['mean']:.1f} | "
            f"{report['total_latency_ms']['p95']:.0f} | "
            f"{report['too_long_rate']:.1%} | {report['vietnamese_signal_rate']:.1%} | "
            f"{report['cjk_leak_rate']:.1%} | "
            f"{report['english_leak_rate']:.1%} | "
            f"{report['uncertainty_rate']:.1%} | "
            f"{format_optional_rate(report['command_refusal_rate'])} | "
            f"{format_optional_rate(report['command_ack_rate'])} | "
            f"{format_optional_rate(report['realtime_safety_rate'])} | "
            f"{format_optional_rate(report['realtime_hallucination_rate'])} | "
            f"{format_optional_rate(report['privacy_safety_rate'])} | "
            f"{format_optional_rate(report['privacy_hallucination_rate'])} | "
            f"{peak_memory_text} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark registered Soca LLM candidates.")
    parser.add_argument(
        "--model",
        action="append",
        default=[],
        choices=sorted(LLM_MODEL_REGISTRY),
        help="Model key to benchmark. May be passed multiple times.",
    )
    parser.add_argument(
        "--models",
        action="append",
        default=[],
        help="Comma-separated model keys. May be passed multiple times.",
    )
    parser.add_argument(
        "--profile",
        choices=sorted(PROFILE_MODEL_KEYS),
        help="Benchmark a profile: minimal, bakeoff, or full.",
    )
    parser.add_argument("--prompts", type=Path, default=DEFAULT_PROMPT_PATH)
    parser.add_argument("--limit", type=int, help="Limit number of prompts for a quick run.")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--n-threads", type=int, default=8)
    parser.add_argument("--n-gpu-layers", type=int, default=-1)
    parser.add_argument("--skip-missing", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    prompts = load_prompts(args.prompts, limit=args.limit)
    try:
        explicit_models = [*args.model, *parse_model_list(args.models)]
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    model_keys = select_model_keys(explicit_models, args.profile)

    reports: list[dict[str, Any]] = []
    for model_key in model_keys:
        report = run_model_eval(
            model_key=model_key,
            prompts=prompts,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            n_threads=args.n_threads,
            n_gpu_layers=args.n_gpu_layers,
            skip_missing=args.skip_missing,
        )
        if report is not None:
            reports.append(report)

    if not reports:
        console.print("[yellow]No LLM reports were produced; all selected models were skipped.[/yellow]")
        return 0

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_paths = make_eval_run_paths(args.output_dir, "llm_bakeoff", timestamp)
    run_paths.run_dir.mkdir(parents=True, exist_ok=True)
    json_path = run_paths.json_path
    md_path = run_paths.md_path
    json_path.write_text(json.dumps(reports, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown_report(md_path, reports)
    update_latest_eval_report(run_paths)

    print_summary_table(reports)
    console.print(f"\n[green]Saved[/green] {json_path}")
    console.print(f"[green]Saved[/green] {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
