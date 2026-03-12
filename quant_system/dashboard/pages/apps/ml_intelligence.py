from __future__ import annotations

import json
from collections import deque
from pathlib import Path
from typing import Any, Dict, Iterable, List

import altair as alt
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from quant_system.dashboard.data_access import DashboardContext
from quant_system.dashboard.ui import inject_page_notice, metric_grid, page_header, section_title


def _safe_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return dict(json.loads(path.read_text(encoding="utf-8")) or {})
    except Exception:
        return {}


def _target_and_family(model_name: str) -> Dict[str, str]:
    name = str(model_name)
    if "_" in name:
        prefix, rest = name.split("_", 1)
        if prefix.isupper():
            name = rest
    family = "tcn" if name.endswith("_tcn") else "tree"
    target = name[:-4] if family == "tcn" else name
    return {"target": target, "family": family}


def _extract_counts(class_counts: Any) -> Dict[str, float]:
    if not isinstance(class_counts, dict):
        return {"neg": 0.0, "pos": 0.0, "pos_rate": 0.0}
    neg = float(class_counts.get("0", class_counts.get(0, 0.0)) or 0.0)
    pos = float(class_counts.get("1", class_counts.get(1, 0.0)) or 0.0)
    total = max(neg + pos, 1.0)
    return {"neg": neg, "pos": pos, "pos_rate": pos / total}


def _scan_model_registry(model_root: Path) -> pd.DataFrame:
    if not model_root.exists() or not model_root.is_dir():
        return pd.DataFrame()
    rows: List[Dict[str, Any]] = []
    for model_dir in sorted(p for p in model_root.iterdir() if p.is_dir()):
        versions = sorted(p for p in model_dir.iterdir() if p.is_dir())
        if not versions:
            continue
        latest = versions[-1]
        metrics = _safe_json(latest / "metrics.json")
        cfg = _safe_json(latest / "config.json")
        mapping = _target_and_family(model_dir.name)
        feature_cols = metrics.get("selected_feature_cols") or cfg.get("features") or []
        counts = _extract_counts(metrics.get("class_counts"))
        cv_score = metrics.get("cv_score")
        cv_auc = metrics.get("cv_auc", metrics.get("auc"))
        cv_ap = metrics.get("cv_ap", metrics.get("pr_auc"))
        rows.append(
            {
                "model": model_dir.name,
                "target": mapping["target"],
                "family": mapping["family"],
                "version": latest.name,
                "updated_at": pd.to_datetime(latest.stat().st_mtime, unit="s", utc=True),
                "feature_count": int(len(feature_cols)) if isinstance(feature_cols, list) else None,
                "transform_dim": metrics.get("feature_transform_dim"),
                "cv_score": float(cv_score) if cv_score is not None else None,
                "cv_auc": float(cv_auc) if cv_auc is not None else None,
                "cv_ap": float(cv_ap) if cv_ap is not None else None,
                "hpo_trials": metrics.get("hpo_trials"),
                "hpo_trials_completed": metrics.get("hpo_trials_completed"),
                "pos_rate": counts["pos_rate"],
                "positive_count": counts["pos"],
                "negative_count": counts["neg"],
                "delta_vs_tree_cv_score": metrics.get("delta_vs_tree_cv_score"),
                "path": str(latest),
            }
        )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["target", "family", "updated_at"], ascending=[True, True, False]).reset_index(drop=True)


def _read_last_ndjson(path: Path, max_rows: int = 5000) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    tail: deque[str] = deque(maxlen=max_rows)
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    tail.append(line)
    except Exception:
        return []
    out: List[Dict[str, Any]] = []
    for line in tail:
        try:
            payload = json.loads(line)
            if isinstance(payload, dict):
                out.append(payload)
        except Exception:
            continue
    return out


def _scan_hpo_progress(train_root: Path) -> pd.DataFrame:
    if not train_root.exists():
        return pd.DataFrame()
    rows: List[Dict[str, Any]] = []
    for snapshot_path in sorted(train_root.glob("**/hpo_progress.json")):
        payload = _safe_json(snapshot_path)
        if not payload:
            continue
        payload["snapshot_path"] = str(snapshot_path)
        payload["events_path"] = str(snapshot_path.with_name("hpo_progress.ndjson"))
        rows.append(payload)

    if not rows:
        for events_path in sorted(train_root.glob("**/hpo_progress.ndjson")):
            events = _read_last_ndjson(events_path, max_rows=1)
            if not events:
                continue
            payload = dict(events[-1])
            payload["snapshot_path"] = str(events_path.with_name("hpo_progress.json"))
            payload["events_path"] = str(events_path)
            rows.append(payload)

    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    if "updated_at_utc" in df.columns:
        df["updated_at_utc"] = pd.to_datetime(df["updated_at_utc"], errors="coerce", utc=True)
    for col in (
        "requested_trials",
        "completed_trials",
        "remaining_trials",
        "best_value",
        "elapsed_sec",
        "eta_sec",
        "eta_by_avg_sec",
    ):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.sort_values("updated_at_utc", ascending=False, na_position="last").reset_index(drop=True)


def _scan_hpo_events(train_root: Path) -> pd.DataFrame:
    if not train_root.exists():
        return pd.DataFrame()
    rows: List[Dict[str, Any]] = []
    for events_path in sorted(train_root.glob("**/hpo_progress.ndjson")):
        rows.extend(_read_last_ndjson(events_path, max_rows=800))
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    if "updated_at_utc" in df.columns:
        df["updated_at_utc"] = pd.to_datetime(df["updated_at_utc"], errors="coerce", utc=True)
    for col in ("best_value", "completed_trials", "requested_trials", "remaining_trials"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    cols = [c for c in ("target", "status", "phase", "best_value", "completed_trials", "requested_trials", "remaining_trials", "updated_at_utc") if c in df.columns]
    if cols:
        df = df[cols]
    return df.dropna(subset=["updated_at_utc"]).sort_values("updated_at_utc").reset_index(drop=True)


def _scan_train_manifests(train_root: Path) -> pd.DataFrame:
    if not train_root.exists():
        return pd.DataFrame()
    rows: List[Dict[str, Any]] = []
    for path in sorted(train_root.glob("**/train_manifest.json")):
        payload = _safe_json(path)
        if not payload:
            continue
        requested = list(payload.get("requested_models", []) or [])
        target_name = str(requested[0]) if requested else path.parent.name
        family = "tcn" if str(target_name).endswith("_tcn") else ("tcn" if path.parent.name.endswith("_tcn") else "tree")
        by_model = payload.get("metrics", {}).get("by_model", {})
        metric_obj = by_model.get(target_name, {})
        rows.append(
            {
                "asset": payload.get("asset"),
                "target": target_name,
                "family": family,
                "version": payload.get("version"),
                "rows": payload.get("rows"),
                "cv_score": metric_obj.get("cv_score"),
                "hpo_trials": metric_obj.get("hpo_trials"),
                "feature_count": len(metric_obj.get("selected_feature_cols", []) or []),
                "manifest_path": str(path),
                "updated_at": pd.to_datetime(path.stat().st_mtime, unit="s", utc=True),
            }
        )
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    for col in ("rows", "cv_score", "hpo_trials", "feature_count"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.sort_values("updated_at", ascending=False).reset_index(drop=True)


def _render_route_table(state: Dict[str, Any], model_root: Path) -> None:
    routes = state.get("specialist_model_source", {}) if isinstance(state.get("specialist_model_source"), dict) else {}
    if not routes:
        inject_page_notice("No specialist route map found yet. It will appear after models load in forward/live runtime.")
        return
    rows = []
    for specialist, resolved in sorted(routes.items()):
        meta = _target_and_family(str(resolved))
        exists = (model_root / str(resolved)).exists()
        rows.append(
            {
                "specialist": specialist,
                "resolved_model": resolved,
                "family": meta["family"],
                "model_dir_exists": bool(exists),
            }
        )
    route_df = pd.DataFrame(rows)
    st.dataframe(route_df, use_container_width=True, hide_index=True)


def _render_family_comparison(models_df: pd.DataFrame) -> None:
    if models_df.empty or "cv_score" not in models_df.columns:
        st.info("No model metrics were found to compare families.")
        return
    specialists = {"liq_flow", "bos_cont", "flow_1h", "momo", "eop", "edp", "hazard"}
    scoped = models_df[models_df["target"].isin(specialists)].copy()
    if scoped.empty:
        st.info("No specialist models in registry yet.")
        return

    comp = (
        scoped.pivot_table(index="target", columns="family", values="cv_score", aggfunc="max")
        .reset_index()
        .rename_axis(None, axis=1)
    )
    if "tcn" not in comp.columns:
        comp["tcn"] = pd.NA
    if "tree" not in comp.columns:
        comp["tree"] = pd.NA
    comp["delta_tcn_minus_tree"] = pd.to_numeric(comp["tcn"], errors="coerce") - pd.to_numeric(comp["tree"], errors="coerce")
    st.dataframe(comp.sort_values("target"), use_container_width=True, hide_index=True)

    long_df = scoped.dropna(subset=["cv_score"]).copy()
    if long_df.empty:
        return
    chart = (
        alt.Chart(long_df)
        .mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
        .encode(
            x=alt.X("target:N", title="Target"),
            y=alt.Y("cv_score:Q", title="CV score"),
            color=alt.Color("family:N", scale=alt.Scale(domain=["tcn", "tree"], range=["#4ac7ff", "#ffb000"])),
            tooltip=["target", "family", "cv_score", "version", "feature_count", "hpo_trials"],
        )
        .properties(height=300)
    )
    st.altair_chart(chart, use_container_width=True)


def _render_hpo_progress(train_root: Path) -> None:
    snap_df = _scan_hpo_progress(train_root)
    if snap_df.empty:
        st.info("No HPO progress snapshot found yet.")
        return

    keep_cols = [
        c
        for c in (
            "asset",
            "target",
            "status",
            "phase",
            "best_value",
            "completed_trials",
            "requested_trials",
            "remaining_trials",
            "elapsed_sec",
            "eta_sec",
            "updated_at_utc",
        )
        if c in snap_df.columns
    ]
    st.dataframe(snap_df[keep_cols], use_container_width=True, hide_index=True)

    events_df = _scan_hpo_events(train_root)
    if events_df.empty:
        return
    plot_df = events_df.dropna(subset=["best_value"]).copy()
    if plot_df.empty:
        return
    plot_df["target"] = plot_df["target"].fillna("unknown")
    line = (
        alt.Chart(plot_df)
        .mark_line(point=True)
        .encode(
            x=alt.X("updated_at_utc:T", title="Time (UTC)"),
            y=alt.Y("best_value:Q", title="Best CV score"),
            color=alt.Color("target:N"),
            tooltip=["target", "best_value", "completed_trials", "requested_trials", "remaining_trials", "status", "phase", "updated_at_utc"],
        )
        .properties(height=290)
    )
    st.altair_chart(line, use_container_width=True)


def _render_feature_footprint(models_df: pd.DataFrame) -> None:
    scoped = models_df.dropna(subset=["feature_count", "cv_score"]).copy() if not models_df.empty else pd.DataFrame()
    if scoped.empty:
        st.info("No feature-footprint metrics available yet.")
        return
    scoped["feature_count"] = pd.to_numeric(scoped["feature_count"], errors="coerce")
    scoped["pos_rate"] = pd.to_numeric(scoped["pos_rate"], errors="coerce")
    scatter = (
        alt.Chart(scoped)
        .mark_circle(size=120, opacity=0.85)
        .encode(
            x=alt.X("feature_count:Q", title="Feature count"),
            y=alt.Y("cv_score:Q", title="CV score"),
            color=alt.Color("family:N", scale=alt.Scale(domain=["tcn", "tree"], range=["#4ac7ff", "#ffb000"])),
            tooltip=["model", "target", "family", "version", "feature_count", "transform_dim", "cv_score", "cv_auc", "cv_ap", "pos_rate"],
        )
        .properties(height=300)
    )
    st.altair_chart(scatter, use_container_width=True)


def _render_auto_refresh_controls() -> None:
    with st.expander("Live Refresh", expanded=False):
        auto_refresh = st.checkbox("Auto refresh this page", value=True, key="mli_auto_refresh")
        refresh_s = st.slider("Refresh every (seconds)", min_value=5, max_value=180, value=20, step=5, key="mli_refresh_sec")
    if auto_refresh:
        components.html(
            f"""
            <script>
              window.setTimeout(function() {{
                window.parent.location.reload();
              }}, {int(refresh_s) * 1000});
            </script>
            """,
            height=0,
        )


def render_ml_intelligence(theme_choice: str, model_version: str, *, context: DashboardContext) -> None:
    page_header(
        "ML Intelligence",
        "Dedicated ML control desk: model-family routing, live HPO progress, training lifecycle, and quality surfaces in one page.",
        kicker="Machine Learning Desk",
    )
    _render_auto_refresh_controls()

    state = dict(context.forward.get("state", {}) or {})
    model_root = Path(context.model_dir)
    train_root = Path.cwd() / "artifacts" / "train"
    models_df = _scan_model_registry(model_root)
    manifests_df = _scan_train_manifests(train_root)
    running_df = _scan_hpo_progress(train_root)

    tcn_count = int((models_df["family"] == "tcn").sum()) if not models_df.empty else 0
    tree_count = int((models_df["family"] == "tree").sum()) if not models_df.empty else 0
    best_score = float(models_df["cv_score"].max()) if (not models_df.empty and models_df["cv_score"].notna().any()) else float("nan")
    source_mode = str(state.get("inference_source_mode") or ("tcn_first" if state.get("prefer_tcn_specialists") else "tree_first"))
    routes = state.get("specialist_model_source", {}) if isinstance(state.get("specialist_model_source"), dict) else {}

    metric_grid(
        [
            {"label": "Model Version", "value": str(model_version)},
            {"label": "Registry Models", "value": f"{len(models_df)}"},
            {"label": "TCN / Tree", "value": f"{tcn_count} / {tree_count}"},
            {"label": "Best CV Score", "value": f"{best_score:.4f}" if pd.notna(best_score) else "--"},
            {"label": "Inference Route", "value": source_mode},
            {"label": "Resolved Specialists", "value": f"{len(routes)}"},
        ]
    )

    section_title("Inference Routing", "Live route map used by forward/live prediction layer")
    _render_route_table(state, model_root)

    left, right = st.columns([1.3, 1.0])
    with left:
        section_title("Family Comparison (TCN vs Tree)", "Per-target cross-family score comparison")
        _render_family_comparison(models_df)
    with right:
        section_title("Training Manifest Registry", "Latest completed runs from artifacts/train")
        if manifests_df.empty:
            st.info("No training manifests found yet.")
        else:
            show_cols = [c for c in ("asset", "target", "family", "version", "rows", "cv_score", "hpo_trials", "feature_count", "updated_at") if c in manifests_df.columns]
            st.dataframe(manifests_df[show_cols], use_container_width=True, hide_index=True)

    section_title("HPO Live Monitor", "Current/last run snapshots + best-score timeline")
    _render_hpo_progress(train_root)

    section_title("Feature Footprint", "Feature width, imbalance profile, and achieved score")
    _render_feature_footprint(models_df)

    if models_df.empty and manifests_df.empty and running_df.empty:
        inject_page_notice(
            "No ML artifacts detected yet. Run training first, then this page will populate automatically."
        )
