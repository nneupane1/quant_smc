"""
Train HDBSCAN liquidity/microstructure clusters on 15m/1h features.

Usage (PowerShell example):
    python -m quant_system.cli.train_hdbscan_clusters `
      --input data/kraken_xbtusd_direct/XBTUSD_15m.csv `
      --asset XBTUSD `
      --timeframe 15m `
      --out-dir models/hdbscan_XBTUSD_15m

Notes:
- Auto-selects numeric feature columns unless you pass --feature-cols.
- Saves: hdbscan_model.joblib, meta.json, train_summary.json,
        {asset}_{timeframe}_hdbscan_clusters.csv
"""

import argparse
import json
from pathlib import Path
from typing import List, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler

try:
    import hdbscan
    _HAS_HDBSCAN = True
except ImportError:
    hdbscan = None  # type: ignore
    _HAS_HDBSCAN = False


def _parse_features_arg(arg: Optional[str]) -> Optional[List[str]]:
    if not arg:
        return None
    return [c.strip() for c in arg.split(",") if c.strip()]


def train_hdbscan(
    df: pd.DataFrame,
    feature_cols: List[str],
    min_cluster_size: int,
    min_samples: Optional[int],
    metric: str,
    pca_components: Optional[int],
) -> dict:
    df_feat = df[feature_cols].copy()
    df_feat = df_feat.replace([np.inf, -np.inf], np.nan).dropna()

    scaler = RobustScaler()
    X = scaler.fit_transform(df_feat.values)

    pca = None
    if pca_components and pca_components > 0:
        pca = PCA(n_components=pca_components, random_state=42)
        X = pca.fit_transform(X)

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric=metric,
        core_dist_n_jobs=-1,
    )
    cluster_labels = clusterer.fit_predict(X)

    meta = {
        "params": {
            "min_cluster_size": min_cluster_size,
            "min_samples": min_samples,
            "metric": metric,
            "pca_components": pca_components,
        },
        "n_clusters": int(len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)),
        "cluster_counts": pd.Series(cluster_labels).value_counts(dropna=False).to_dict(),
        "feature_cols": feature_cols,
    }

    return {
        "clusterer": clusterer,
        "scaler": scaler,
        "pca": pca,
        "labels": cluster_labels,
        "probabilities": clusterer.probabilities_,
        "outlier_scores": clusterer.outlier_scores_,
        "meta": meta,
    }


def main():
    ap = argparse.ArgumentParser(description="Train HDBSCAN clusters on liquidity/vol features.")
    ap.add_argument("--input", required=True, help="Input CSV with dt, timestamp, and feature columns.")
    ap.add_argument("--asset", required=True, help="Asset symbol (e.g., XBTUSD).")
    ap.add_argument("--timeframe", required=True, help="TF label (e.g., 15m, 1h).")
    ap.add_argument("--out-dir", required=True, help="Output directory for model + artifacts.")
    ap.add_argument("--feature-cols", default=None, help="Comma-separated feature columns to use.")
    ap.add_argument("--min-cluster-size", type=int, default=50, help="HDBSCAN min_cluster_size.")
    ap.add_argument("--min-samples", type=int, default=None, help="HDBSCAN min_samples (optional).")
    ap.add_argument("--metric", default="euclidean", help="Distance metric (default: euclidean).")
    ap.add_argument("--pca-components", type=int, default=None, help="Optional PCA components before clustering.")
    args = ap.parse_args()
    if not _HAS_HDBSCAN:
        raise SystemExit("hdbscan is required. Please install with `pip install hdbscan`.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input, parse_dates=["dt"])
    df = df.sort_values("dt").reset_index(drop=True)

    feature_cols = _parse_features_arg(args.feature_cols)
    if feature_cols is None:
        # Auto-pick numeric columns excluding known time/id columns
        drop_cols = {"dt", "timestamp"}
        feature_cols = [
            c for c in df.columns
            if c not in drop_cols and pd.api.types.is_numeric_dtype(df[c])
        ]

    result = train_hdbscan(
        df=df,
        feature_cols=feature_cols,
        min_cluster_size=args.min_cluster_size,
        min_samples=args.min_samples,
        metric=args.metric,
        pca_components=args.pca_components,
    )

    # Save assignments
    assign = pd.DataFrame({
        "dt": df.loc[df.index[: len(result["labels"])], "dt"].values,
        "timestamp": df.loc[df.index[: len(result["labels"])], "timestamp"].values if "timestamp" in df else np.nan,
        "cluster": result["labels"],
        "probability": result["probabilities"],
        "outlier_score": result["outlier_scores"],
    })
    assign_path = out_dir / f"{args.asset}_{args.timeframe}_hdbscan_clusters.csv"
    assign.to_csv(assign_path, index=False)

    # Save model bundle
    bundle = {
        "clusterer": result["clusterer"],
        "scaler": result["scaler"],
        "pca": result["pca"],
        "feature_cols": feature_cols,
    }
    joblib.dump(bundle, out_dir / "hdbscan_model.joblib")

    # Save meta
    with open(out_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(result["meta"], f, indent=2)

    # Train summary
    summary = {
        "asset": args.asset,
        "timeframe": args.timeframe,
        "rows_used": int(len(result["labels"])),
        "n_clusters": result["meta"]["n_clusters"],
        "cluster_counts": result["meta"]["cluster_counts"],
        "feature_cols": feature_cols,
    }
    with open(out_dir / "train_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"[ok] HDBSCAN trained. Clusters saved to {assign_path}")


if __name__ == "__main__":
    main()
