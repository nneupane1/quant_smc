from __future__ import annotations

import argparse
import time
from typing import List

from train_BTCUSD_tcn_model import run_target
from quant_system.utils.logger import console_kv, console_rule, console_stage, fmt_seconds


DEFAULT_TARGETS = ["liq_flow", "bos_cont", "flow_1h", "momo", "eop", "edp"]


def _parse_targets(raw: str) -> List[str]:
    tokens = [t.strip() for t in str(raw).split(",") if t.strip()]
    if not tokens:
        return list(DEFAULT_TARGETS)
    bad = [t for t in tokens if t not in DEFAULT_TARGETS]
    if bad:
        raise SystemExit(f"Unsupported targets: {', '.join(bad)}")
    seen = []
    for tok in tokens:
        if tok not in seen:
            seen.append(tok)
    return seen


def main() -> None:
    parser = argparse.ArgumentParser(description="Train all BTCUSD TCN specialist models sequentially.")
    parser.add_argument(
        "--targets",
        default=",".join(DEFAULT_TARGETS),
        help=f"Comma-separated subset of targets. Allowed: {', '.join(DEFAULT_TARGETS)}",
    )
    parser.add_argument("--trials", type=int, default=None, help="Override HPO trial count for each target.")
    parser.add_argument("--cv-splits", type=int, default=None, help="Override CV split count for each target.")
    parser.add_argument("--no-resume", action="store_true", help="Disable cache resume for training frame.")
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue training remaining targets if one target fails.",
    )
    args = parser.parse_args()

    targets = _parse_targets(args.targets)
    t0 = time.perf_counter()
    results = []

    console_rule("Train All TCN Specialists | BTCUSD", style="bright_magenta")
    console_kv(
        "TCN Batch Plan",
        {
            "targets": ", ".join(targets),
            "count": len(targets),
            "trials_override": args.trials if args.trials is not None else "-",
            "cv_splits_override": args.cv_splits if args.cv_splits is not None else "-",
            "resume": not args.no_resume,
            "continue_on_error": bool(args.continue_on_error),
        },
        style="bright_magenta",
    )

    for idx, target in enumerate(targets, start=1):
        elapsed_before = time.perf_counter() - t0
        done_before = idx - 1
        rate_before = done_before / max(elapsed_before, 1e-6) if done_before > 0 else 0.0
        eta_before = (len(targets) - done_before) / max(rate_before, 1e-6) if rate_before > 0 else None
        console_rule(f"TCN Batch {idx}/{len(targets)} | {target}", style="bright_magenta")
        console_stage(
            "TCN batch progress",
            (
                f"completed={done_before}/{len(targets)} remaining={len(targets) - done_before} "
                f"elapsed={fmt_seconds(elapsed_before)} "
                f"eta={fmt_seconds(eta_before) if eta_before is not None else '-'}"
            ),
            status="info",
        )
        target_started = time.perf_counter()
        try:
            manifest = run_target(
                target,
                trials=args.trials,
                cv_splits=args.cv_splits,
                resume=not args.no_resume,
            )
            model_key = f"{target}_tcn"
            metrics = (manifest.get("metrics", {}) or {}).get("by_model", {}).get(model_key, {})
            results.append(
                {
                    "target": target,
                    "status": "ok",
                    "version": manifest.get("version"),
                    "cv_score": metrics.get("cv_score"),
                    "accept_score": ((metrics.get("acceptance", {}) or {}).get("metrics") or {}).get("score"),
                    "elapsed": fmt_seconds(time.perf_counter() - target_started),
                }
            )
        except Exception as exc:
            results.append(
                {
                    "target": target,
                    "status": "failed",
                    "version": "-",
                    "cv_score": None,
                    "accept_score": None,
                    "elapsed": fmt_seconds(time.perf_counter() - target_started),
                    "error": str(exc),
                }
            )
            console_stage(
                "TCN batch target failed",
                f"target={target} error={exc}",
                status="warn",
            )
            if not args.continue_on_error:
                break

        done_now = len(results)
        elapsed_now = time.perf_counter() - t0
        rate_now = done_now / max(elapsed_now, 1e-6)
        eta_now = (len(targets) - done_now) / max(rate_now, 1e-6)
        console_stage(
            "TCN batch heartbeat",
            (
                f"completed={done_now}/{len(targets)} remaining={max(len(targets)-done_now, 0)} "
                f"elapsed={fmt_seconds(elapsed_now)} eta={fmt_seconds(eta_now)}"
            ),
            status="info",
        )

    done = sum(1 for row in results if row["status"] == "ok")
    failed = sum(1 for row in results if row["status"] != "ok")
    console_kv(
        "TCN Batch Summary",
        {
            "completed": done,
            "failed": failed,
            "elapsed": fmt_seconds(time.perf_counter() - t0),
        },
        style="green" if failed == 0 else "yellow",
    )
    for row in results:
        console_stage(
            f"TCN {row['target']}",
            (
                f"status={row['status']} version={row.get('version')} "
                f"cv_score={row.get('cv_score')} accept_score={row.get('accept_score')} "
                f"elapsed={row.get('elapsed')}"
            ),
            status="ok" if row["status"] == "ok" else "warn",
        )

    if failed > 0 and not args.continue_on_error:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
