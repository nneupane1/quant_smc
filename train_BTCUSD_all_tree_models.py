from __future__ import annotations

import argparse
import time
from typing import List

from launcher_common import inspect_training_artifacts, train_model
from quant_system.utils.logger import console_kv, console_rule, console_stage, fmt_progress, fmt_seconds


DEFAULT_TARGETS = ["liq_flow", "bos_cont", "flow_1h", "momo", "eop", "edp"]
EXTENDED_TARGETS = ["meta_model", "confluence_model", "hazard", "quantile"]
ALLOWED_TARGETS = DEFAULT_TARGETS + EXTENDED_TARGETS


def _parse_targets(raw: str) -> List[str]:
    tokens = [t.strip() for t in str(raw).split(",") if t.strip()]
    if not tokens:
        return list(DEFAULT_TARGETS)
    bad = [t for t in tokens if t not in ALLOWED_TARGETS]
    if bad:
        raise SystemExit(f"Unsupported targets: {', '.join(bad)}")
    seen: List[str] = []
    for tok in tokens:
        if tok not in seen:
            seen.append(tok)
    return seen


def _fmt_score(value) -> str:
    if isinstance(value, (int, float)):
        return f"{float(value):.4f}"
    return "-"


def _remaining_targets(targets: List[str], idx: int) -> str:
    remaining = targets[idx:]
    return ", ".join(remaining) if remaining else "-"


def main() -> None:
    parser = argparse.ArgumentParser(description="Train BTCUSD tree-based models sequentially.")
    parser.add_argument(
        "--targets",
        default=",".join(DEFAULT_TARGETS),
        help=(
            "Comma-separated subset to train. "
            f"Allowed: {', '.join(ALLOWED_TARGETS)}. "
            "Default trains core specialists only."
        ),
    )
    parser.add_argument(
        "--include-extended",
        action="store_true",
        help="Append meta_model, confluence_model, hazard, quantile after selected targets.",
    )
    parser.add_argument("--no-resume", action="store_true", help="Disable cache resume for training frame.")
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue training remaining targets if one target fails.",
    )
    args = parser.parse_args()

    targets = _parse_targets(args.targets)
    if args.include_extended:
        for model_name in EXTENDED_TARGETS:
            if model_name not in targets:
                targets.append(model_name)

    t0 = time.perf_counter()
    results = []

    console_rule("Train All Tree Models | BTCUSD", style="cyan")
    console_kv(
        "Tree Batch Plan",
        {
            "targets": ", ".join(targets),
            "count": len(targets),
            "resume": not args.no_resume,
            "continue_on_error": bool(args.continue_on_error),
        },
        style="cyan",
    )
    for idx, target in enumerate(targets, start=1):
        status = inspect_training_artifacts(target)
        state = "ready" if status["manifest_exists"] else "pending"
        console_stage(
            f"Queue {idx}/{len(targets)}",
            (
                f"target={target} state={state} "
                f"version={status['version']} cv_score={_fmt_score(status.get('cv_score'))} "
                f"artifact_root={status['artifact_root']}"
            ),
            status="ok" if state == "ready" else "info",
        )

    for idx, target in enumerate(targets, start=1):
        pre_status = inspect_training_artifacts(target)
        elapsed_before = time.perf_counter() - t0
        done_before = idx - 1
        rate_before = done_before / max(elapsed_before, 1e-6) if done_before > 0 else 0.0
        eta_before = (len(targets) - done_before) / max(rate_before, 1e-6) if rate_before > 0 else None
        console_rule(f"Tree Batch {idx}/{len(targets)} | {target}", style="cyan")
        console_stage(
            "Tree batch progress",
            (
                f"progress={fmt_progress(done_before, len(targets))} "
                f"completed={done_before}/{len(targets)} remaining={len(targets) - done_before} "
                f"elapsed={fmt_seconds(elapsed_before)} "
                f"eta={fmt_seconds(eta_before) if eta_before is not None else '-'}"
            ),
            status="info",
        )
        console_kv(
            "Target Plan",
            {
                "target": target,
                "position": f"{idx}/{len(targets)}",
                "previous_status": "ready" if pre_status["manifest_exists"] else "pending",
                "previous_version": pre_status["version"],
                "previous_cv_score": _fmt_score(pre_status.get("cv_score")),
                "artifact_root": pre_status["artifact_root"],
                "manifest": pre_status["manifest_path"],
                "remaining_queue": _remaining_targets(targets, idx),
            },
            style="cyan",
        )

        started = time.perf_counter()
        try:
            manifest = train_model(target, resume=not args.no_resume)
            post_status = inspect_training_artifacts(target)
            outcome = "trained"
            if (
                not args.no_resume
                and pre_status.get("manifest_exists")
                and post_status.get("manifest_exists")
                and pre_status.get("manifest_mtime") == post_status.get("manifest_mtime")
                and pre_status.get("version") == post_status.get("version")
            ):
                outcome = "resume_hit"
            model_metrics = (manifest.get("metrics", {}) or {}).get("by_model", {}).get(target, {})
            results.append(
                {
                    "target": target,
                    "status": "ok",
                    "outcome": outcome,
                    "version": post_status.get("version") or manifest.get("version"),
                    "cv_score": post_status.get("cv_score", model_metrics.get("cv_score")),
                    "artifact_root": post_status.get("artifact_root"),
                    "manifest": post_status.get("manifest_path"),
                    "elapsed": fmt_seconds(time.perf_counter() - started),
                }
            )
        except Exception as exc:
            results.append(
                {
                    "target": target,
                    "status": "failed",
                    "outcome": "failed",
                    "version": "-",
                    "cv_score": None,
                    "artifact_root": pre_status.get("artifact_root"),
                    "manifest": pre_status.get("manifest_path"),
                    "elapsed": fmt_seconds(time.perf_counter() - started),
                    "error": str(exc),
                }
            )
            console_stage("Tree batch target failed", f"target={target} error={exc}", status="warn")
            if not args.continue_on_error:
                break

        done_now = len(results)
        elapsed_now = time.perf_counter() - t0
        rate_now = done_now / max(elapsed_now, 1e-6)
        eta_now = (len(targets) - done_now) / max(rate_now, 1e-6)
        latest = results[-1]
        console_kv(
            "Target Result",
            {
                "target": latest["target"],
                "outcome": latest.get("outcome"),
                "version": latest.get("version"),
                "cv_score": _fmt_score(latest.get("cv_score")),
                "target_elapsed": latest.get("elapsed"),
                "progress": fmt_progress(done_now, len(targets)),
                "batch_completed": f"{done_now}/{len(targets)}",
                "batch_remaining": max(len(targets) - done_now, 0),
                "batch_elapsed": fmt_seconds(elapsed_now),
                "batch_eta": fmt_seconds(eta_now),
                "next_target": targets[done_now] if done_now < len(targets) else "-",
            },
            style="green" if latest["status"] == "ok" else "yellow",
        )
        console_stage(
            "Tree batch heartbeat",
            (
                f"progress={fmt_progress(done_now, len(targets))} "
                f"completed={done_now}/{len(targets)} remaining={max(len(targets)-done_now, 0)} "
                f"elapsed={fmt_seconds(elapsed_now)} eta={fmt_seconds(eta_now)} "
                f"remaining_targets={_remaining_targets(targets, done_now)}"
            ),
            status="info",
        )

    done = sum(1 for row in results if row["status"] == "ok")
    trained = sum(1 for row in results if row.get("outcome") == "trained")
    resumed = sum(1 for row in results if row.get("outcome") == "resume_hit")
    failed = sum(1 for row in results if row["status"] != "ok")
    console_kv(
        "Tree Batch Summary",
        {
            "completed": done,
            "trained": trained,
            "resume_hits": resumed,
            "failed": failed,
            "elapsed": fmt_seconds(time.perf_counter() - t0),
        },
        style="green" if failed == 0 else "yellow",
    )

    for row in results:
        console_stage(
            f"Tree {row['target']}",
            (
                f"status={row['status']} outcome={row.get('outcome')} "
                f"version={row.get('version')} cv_score={_fmt_score(row.get('cv_score'))} "
                f"elapsed={row.get('elapsed')} manifest={row.get('manifest')}"
            ),
            status="ok" if row["status"] == "ok" else "warn",
        )

    if failed > 0 and not args.continue_on_error:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
