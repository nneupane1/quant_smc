"""
Train an HDBSCAN liquidity/chop clustering model on 15m/1h bars.

Inputs:
  - CSV with at least: dt, open, high, low, close, volume
    (Same format as your other TF CSVs, UTC dt column)

Outputs (out-dir):
  - hdbscan_model.joblib
  - scaler.joblib
  - meta.json (config + basic stats)
  - labels.csv (dt, label, probability) when --save-labels is set

Example (PowerShell):
  python -m quant_system.cli.train_hdbscan_liquidity `
    --input data/kraken_xbtusd_direct/XBTUSD_15m.csv `
    --out-dir models/liquidity_hdbscan_15m `
    --min-cluster-size 80 `
    --min-samples 10 `
    --save-labels
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

try:
    import hdbscan
    _HAS_HDBSCAN = True
except ImportError:
    hdbscan = None  # type: ignore
    _HAS_HDBSCAN = False


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Basic microstructure feature block for clustering.
    Assumes df has dt, open, high, low, close, volume.
    """
    frame = df.copy()
    frame = frame.sort_values("dt").reset_index(drop=True)

    # core prices
    close = frame["close"].astype(float)
    high = frame["high"].astype(float)
    low = frame["low"].astype(float)
    volume = frame["volume"].astype(float)

    # features
    feat = pd.DataFrame(index=frame.index)
    feat["log_ret"] = np.log(close / close.shift(1)).fillna(0.0)
    feat["range_pct"] = (high - low) / close.replace(0, np.nan)
    feat["log_vol"] = np.log(volume.replace(0, np.nan)).fillna(0.0)
    feat["dollar_vol"] = (close * volume).replace(0, np.nan)
    feat["dollar_vol_log"] = np.log(feat["dollar_vol"].replace(0, np.nan)).fillna(0.0)
    feat["range_z"] = (
        (high - low)
        .rolling(48, min_periods=10)
        .apply(lambda x: (x.iloc[-1] - x.mean()) / (x.std() + 1e-9), raw=False)
    )
    feat["vol_z"] = (
        volume.rolling(48, min_periods=10)
        .apply(lambda x: (x.iloc[-1] - x.mean()) / (x.std() + 1e-9), raw=False)
    )

    feat = feat.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    feat["dt"] = frame["dt"].values
    return feat


def train_hdbscan(
    X: pd.DataFrame,
    min_cluster_size: int,
    min_samples: int,
    epsilon: float,
    metric: str,
    random_state: int,
):
    model = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_epsilon=epsilon,
        metric=metric,
        cluster_selection_method="leaf",
        prediction_data=True,
        core_dist_n_jobs=-1,
        gen_min_span_tree=False,
    )
    model.fit(X)
    return model


def summarize_labels(labels: np.ndarray) -> Dict:
    uniq, counts = np.unique(labels, return_counts=True)
    label_counts = {int(k): int(v) for k, v in zip(uniq, counts)}
    noise = label_counts.get(-1, 0)
    total = len(labels)
    n_clusters = len([k for k in uniq if k != -1])
    return {
        "n_clusters": n_clusters,
        "noise_fraction": float(noise / total if total else 0.0),
        "label_counts": label_counts,
    }


def main():
    ap = argparse.ArgumentParser(description="Train HDBSCAN liquidity/chop clustering.")
    ap.add_argument("--input", required=True, help="Input CSV (15m or 1h bars with dt, ohlcv)")
    ap.add_argument("--out-dir", required=True, help="Output directory for artifacts")
    ap.add_argument("--min-cluster-size", type=int, default=80, help="HDBSCAN min_cluster_size")
    ap.add_argument("--min-samples", type=int, default=10, help="HDBSCAN min_samples")
    ap.add_argument("--epsilon", type=float, default=0.0, help="cluster_selection_epsilon")
    ap.add_argument("--metric", default="euclidean", help="distance metric (default euclidean)")
    ap.add_argument("--random-state", type=int, default=42, help="random state for reproducibility")
    ap.add_argument("--save-labels", action="store_true", help="Save dt,label,probability CSV")
    args = ap.parse_args()
    if not _HAS_HDBSCAN:
        raise SystemExit("hdbscan is required for this trainer. Install with: pip install hdbscan")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Loading data: {args.input}")
    df = pd.read_csv(args.input, parse_dates=["dt"])

    feat_df = build_features(df)
    dt_col = feat_df.pop("dt")
    feature_cols: List[str] = feat_df.columns.tolist()

    print(f"[INFO] Scaling features ({len(feature_cols)} cols)")
    scaler = StandardScaler()
    Xs = scaler.fit_transform(feat_df.values)

    print(
        f"[INFO] Training HDBSCAN | min_cluster_size={args.min_cluster_size} "
        f"min_samples={args.min_samples} eps={args.epsilon}"
    )
    model = train_hdbscan(
        Xs,
        min_cluster_size=args.min_cluster_size,
        min_samples=args.min_samples,
        epsilon=args.epsilon,
        metric=args.metric,
        random_state=args.random_state,
    )

    labels = model.labels_
    probs = model.probabilities_
    stats = summarize_labels(labels)

    meta = {
        "input": args.input,
        "features": feature_cols,
        "model": "HDBSCAN",
        "params": {
            "min_cluster_size": args.min_cluster_size,
            "min_samples": args.min_samples,
            "epsilon": args.epsilon,
            "metric": args.metric,
        },
        "stats": stats,
    }

    print(f"[INFO] Clusters: {stats['n_clusters']} | noise frac: {stats['noise_fraction']:.2%}")

    # persist
    joblib.dump(model, out_dir / "hdbscan_model.joblib")
    joblib.dump(scaler, out_dir / "scaler.joblib")
    with open(out_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    if args.save_labels:
        labels_df = pd.DataFrame({"dt": dt_col, "label": labels, "probability": probs})
        labels_df.to_csv(out_dir / "labels.csv", index=False)
        print(f"[INFO] Saved labels -> {out_dir / 'labels.csv'}")

    print(f"[OK] Saved model to {out_dir}")


if __name__ == "__main__":
    main()
