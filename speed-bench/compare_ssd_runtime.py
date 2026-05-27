#!/usr/bin/env python3
"""Compare native ds4 SSD Flash-MoE against the reference llama-cli runtime."""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


DS4_SPEED_RE = re.compile(
    r"ds4: prefill:\s*([0-9.]+)\s*t/s,\s*generation:\s*([0-9.]+)\s*t/s"
)
DS4_PREFILL_TOKENS_RE = re.compile(
    r"using chunked GPU prefill \([^)]+ for ([0-9]+) prompt tokens\)"
)
REF_SPEED_RE = re.compile(
    r"\[\s*Prompt:\s*([0-9.]+)\s*t/s\s*\|\s*Generation:\s*([0-9.]+)\s*t/s\s*\]"
)
REF_TOKENS_RE = re.compile(r"\[\s*tokens\s*-\s*prefill:\s*([0-9]+),\s*decode:\s*([0-9]+)\s*\]")


DEFAULT_MODEL = Path("/Users/eouya/llm/models/DeepSeek-V4-Flash-FP4-FP8-SSD")
DEFAULT_REF_BIN = Path(
    "/Users/eouya/llm/PROJECT/deepseek-v4-SSD/"
    "anemll-flash-llama.cpp/build/bin/llama-cli"
)


@dataclass(frozen=True)
class Case:
    name: str
    prompt: str


@dataclass(frozen=True)
class Speed:
    prefill: float
    generation: float


@dataclass(frozen=True)
class Ds4Result:
    speed: Speed
    prefill_tokens: int | None


@dataclass(frozen=True)
class ReferenceResult:
    speed: Speed
    prefill_tokens: int
    decode_tokens: int


CASES = [
    Case("short", "are you deepseek?"),
    Case("code", "stack and queue python code"),
    Case("ko", "딥시크에 대해 한글로 짧게 설명해줘."),
    Case(
        "long",
        "Write a compact technical explanation of how DeepSeek V4 Flash can "
        "store dense weights in FP8 and routed MoE experts in MXFP4 sidecar "
        "files while keeping inference correct. Include routing, prefill, "
        "decode, cache reuse, and why a slot-bank can help on Apple Metal. "
        "Then give two practical risks and two mitigations. Keep the answer "
        "structured.",
    ),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ds4-bin", default="./ds4", help="native ds4 binary")
    p.add_argument(
        "--model",
        default=os.environ.get("DS4_SSD_MODEL", str(DEFAULT_MODEL)),
        help="SSD package directory containing dense/ and sidecar/",
    )
    p.add_argument(
        "--reference-bin",
        default=os.environ.get("DS4_SSD_LLAMA_CLI", str(DEFAULT_REF_BIN)),
        help="reference anemll-flash llama-cli binary",
    )
    p.add_argument("--ctx", type=int, default=512)
    p.add_argument("--n-predict", type=int, default=64)
    p.add_argument("--ds4-slot-bank", type=int, default=16)
    p.add_argument(
        "--ds4-system",
        default="",
        help="system prompt passed to native ds4; empty matches reference simple prompt",
    )
    p.add_argument("--reference-slot-bank", type=int, default=6)
    p.add_argument(
        "--min-prefill-ratio",
        type=float,
        default=0.0,
        help="optional prefill-only gate; 0 disables this gate",
    )
    p.add_argument("--min-generation-ratio", type=float, default=1.0)
    p.add_argument("--min-total-ratio", type=float, default=1.0)
    p.add_argument(
        "--warmup-runs",
        type=int,
        default=0,
        help="discarded warmup runs per engine/case before measurement",
    )
    p.add_argument(
        "--measure-order",
        choices=("reference-first", "ds4-first"),
        default="reference-first",
        help="measurement order after warmups",
    )
    p.add_argument(
        "--cases",
        default=",".join(c.name for c in CASES),
        help="comma-separated case names, or 'all'",
    )
    return p.parse_args()


def selected_cases(spec: str) -> list[Case]:
    if spec == "all":
        names = {c.name for c in CASES}
    else:
        names = {item.strip() for item in spec.split(",") if item.strip()}
    by_name = {c.name: c for c in CASES}
    missing = sorted(names - set(by_name))
    if missing:
        raise SystemExit(f"unknown benchmark case(s): {', '.join(missing)}")
    return [c for c in CASES if c.name in names]


def run_command(cmd: list[str], *, env: dict[str, str]) -> str:
    proc = subprocess.run(
        cmd,
        env=env,
        text=True,
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if proc.returncode != 0:
        tail = "\n".join(proc.stdout.splitlines()[-80:])
        raise RuntimeError(
            f"command failed ({proc.returncode}): {' '.join(cmd)}\n{tail}"
        )
    return proc.stdout


def parse_speed(regex: re.Pattern[str], output: str, label: str) -> Speed:
    match = regex.search(output)
    if not match:
        tail = "\n".join(output.splitlines()[-80:])
        raise RuntimeError(f"could not parse {label} speed from output:\n{tail}")
    return Speed(prefill=float(match.group(1)), generation=float(match.group(2)))


def run_ds4(args: argparse.Namespace, case: Case, repo: Path) -> Ds4Result:
    env = os.environ.copy()
    env["DS4_LOCK_FILE"] = f"/tmp/ds4_ssd_compare_{case.name}_native.lock"
    ds4_bin = Path(args.ds4_bin)
    if not ds4_bin.is_absolute():
        ds4_bin = repo / ds4_bin
    cmd = [
        str(ds4_bin),
        "-m",
        args.model,
        "--moe-slot-bank",
        str(args.ds4_slot_bank),
        "--ctx",
        str(args.ctx),
        "--nothink",
        "-sys",
        args.ds4_system,
        "--temp",
        "0",
        "-n",
        str(args.n_predict),
        "-p",
        case.prompt,
    ]
    out = run_command(cmd, env=env)
    speed = parse_speed(DS4_SPEED_RE, out, "native ds4")
    token_match = DS4_PREFILL_TOKENS_RE.search(out)
    return Ds4Result(
        speed=speed,
        prefill_tokens=int(token_match.group(1)) if token_match else None,
    )


def run_reference(args: argparse.Namespace, case: Case) -> ReferenceResult:
    model = Path(args.model)
    dense = model / "dense" / "model-dense.gguf"
    sidecar = model / "sidecar"
    cmd = [
        args.reference_bin,
        "-m",
        str(dense),
        "--moe-sidecar",
        str(sidecar),
        "--moe-mode",
        "slot-bank",
        "--moe-slot-bank",
        str(args.reference_slot_bank),
        "--ctx-size",
        str(args.ctx),
        "--n-predict",
        str(args.n_predict),
        "--temp",
        "0",
        "--reasoning",
        "off",
        "--no-warmup",
        "--simple-io",
        "--no-display-prompt",
        "--single-turn",
        "-p",
        case.prompt,
    ]
    out = run_command(cmd, env=os.environ.copy())
    speed = parse_speed(REF_SPEED_RE, out, "reference llama-cli")
    token_match = REF_TOKENS_RE.search(out)
    if not token_match:
        tail = "\n".join(out.splitlines()[-80:])
        raise RuntimeError(f"could not parse reference token counts from output:\n{tail}")
    return ReferenceResult(
        speed=speed,
        prefill_tokens=int(token_match.group(1)),
        decode_tokens=int(token_match.group(2)),
    )


def total_tokens_per_second(speed: Speed, prefill_tokens: int, decode_tokens: int) -> float:
    if speed.prefill <= 0.0 or speed.generation <= 0.0:
        return 0.0
    seconds = prefill_tokens / speed.prefill + decode_tokens / speed.generation
    tokens = prefill_tokens + decode_tokens
    return tokens / seconds if seconds > 0.0 else 0.0


def main() -> int:
    args = parse_args()
    repo = Path(__file__).resolve().parents[1]
    failures: list[str] = []

    print(
        "case,ds4_prompt_tokens,ref_prompt_tokens,ds4_prefill,ref_prefill,prefill_ratio,"
        "ds4_generation,ref_generation,generation_ratio,"
        "ds4_total,ref_total,total_ratio,status"
    )
    for case in selected_cases(args.cases):
        print(f"# running {case.name}", flush=True)
        for _ in range(max(args.warmup_runs, 0)):
            run_reference(args, case)
            run_ds4(args, case, repo)
        if args.measure_order == "reference-first":
            ref = run_reference(args, case)
            ds4_result = run_ds4(args, case, repo)
        else:
            ds4_result = run_ds4(args, case, repo)
            ref = run_reference(args, case)
        ds4 = ds4_result.speed
        ref_speed = ref.speed
        prefill_ratio = ds4.prefill / ref_speed.prefill if ref_speed.prefill > 0 else 0.0
        generation_ratio = ds4.generation / ref_speed.generation if ref_speed.generation > 0 else 0.0
        ds4_prefill_tokens = ds4_result.prefill_tokens or ref.prefill_tokens
        ds4_total = total_tokens_per_second(ds4, ds4_prefill_tokens, ref.decode_tokens)
        ref_total = total_tokens_per_second(ref_speed, ref.prefill_tokens, ref.decode_tokens)
        total_ratio = ds4_total / ref_total if ref_total > 0.0 else 0.0
        ok = (
            prefill_ratio >= args.min_prefill_ratio
            and generation_ratio >= args.min_generation_ratio
            and total_ratio >= args.min_total_ratio
        )
        status = "pass" if ok else "fail"
        print(
            f"{case.name},{ds4_prefill_tokens},{ref.prefill_tokens},"
            f"{ds4.prefill:.2f},{ref_speed.prefill:.2f},{prefill_ratio:.3f},"
            f"{ds4.generation:.2f},{ref_speed.generation:.2f},{generation_ratio:.3f},"
            f"{ds4_total:.2f},{ref_total:.2f},{total_ratio:.3f},{status}"
        )
        if not ok:
            failures.append(case.name)

    if failures:
        print(f"failed cases: {', '.join(failures)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
