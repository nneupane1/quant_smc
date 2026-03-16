# Quant SMC

Quant SMC is a research-to-execution quantitative trading platform for crypto markets. It is built around a deterministic multi-timeframe feature graph, supervised specialist models, structured execution logic, and operator-facing telemetry. The system is designed so that research, backtest, forward test, live runtime, and dashboards all reason over the same `15m` decision spine with joined `1h`, `6h`, and `12h` context, rather than each runtime inventing its own version of market state.

This repository contains the repaired baseline of that platform: historical ingestion, feature engineering, label generation, model training, backtesting, forward simulation, live orchestration, a Streamlit research shell, and a `Next.js + FastAPI + WebSocket` operator terminal. The current tagged baseline is `v0.1.0`.

## Executive Summary

Quant SMC is not a single model and it is not a single dashboard. It is a layered decision system. Raw exchange data is normalized into canonical bars; canonical bars are transformed into engineered structural, statistical, and regime-aware features; those features are converted into supervised targets; specialist models and stackers are trained over those targets; runtime engines score new bars against the same feature contract; execution logic then combines gates, confluence, EVR, tiering, sizing, and trailing into a trade decision that can be inspected by an operator. The repo is therefore best understood as a full pipeline for building, validating, and operating a structured trading process.

The central design promise of the project is research-to-execution parity. The same market state representation should feed backtests, forward simulation, live operation, and operator telemetry. That is the reason for the multi-timeframe contract, the repaired feature builder, the governed label tuning path, the versioned model registry, and the telemetry layer that exposes decision reasoning rather than only final alerts.

## At A Glance

| Dimension | Current Baseline |
|---|---|
| Market focus | Crypto spot / derivatives-style directional trading logic |
| Primary asset launcher | `BTCUSD` |
| Exchange integration | Kraken historical fetch and streaming paths |
| Decision spine | `15m` |
| Higher-timeframe context | `1h`, `6h`, `12h` |
| Regime layer | `12h` HMM + HDBSCAN |
| Main supervised layer | tabular specialists + stackers |
| Forecast layer | quantile forecaster active, `NARX` deferred |
| Runtime modes | backtest, forward, live |
| UI surfaces | Streamlit research shell + Next terminal |
| Telemetry transport | FastAPI + WebSocket |
| Capital baseline | `20,000 USD` ticket policy with compounding and cooling |

## Table Of Contents

- [Executive Summary](#executive-summary)
- [At A Glance](#at-a-glance)
- [1. Project Purpose](#1-project-purpose)
- [2. What This System Actually Does](#2-what-this-system-actually-does)
- [3. Core Design Principles](#3-core-design-principles)
- [4. Timeframe Contract](#4-timeframe-contract)
- [5. End-To-End Architecture](#5-end-to-end-architecture)
- [6. Data Pipeline](#6-data-pipeline)
- [7. Feature Engineering](#7-feature-engineering)
- [8. Labeling And Supervised Targets](#8-labeling-and-supervised-targets)
- [9. Model Stack](#9-model-stack)
- [9A. Temporal Convolutional Network (TCN) Specialist Benchmark Stack](#9a-temporal-convolutional-network-tcn-specialist-benchmark-stack)
- [9B. Temporal Convolutional Network (TCN) Design Rationale And Learning Mechanics](#9b-temporal-convolutional-network-tcn-design-rationale-and-learning-mechanics)
- [10. Execution, Risk, And Capital Logic](#10-execution-risk-and-capital-logic)
- [10A. Operator Guide: Reading Confluence, EVR, And Hazard Together](#10a-operator-guide-reading-confluence-evr-and-hazard-together)
- [10B. Post-Backtest Trade-Policy Overlay](#10b-post-backtest-trade-policy-overlay)
- [11. Runtime Modes](#11-runtime-modes)
- [12. Dashboards, Terminal, And Telemetry](#12-dashboards-terminal-and-telemetry)
- [13. Installation](#13-installation)
- [14. Environment And Secrets](#14-environment-and-secrets)
- [15. Main Entry Points](#15-main-entry-points)
- [16. BTCUSD Sequential Runbook](#16-btcusd-sequential-runbook)
- [16A. BTCUSD Temporal Convolutional Network (TCN)-Only Sequential Runbook](#16a-btcusd-temporal-convolutional-network-tcn-only-sequential-runbook)
- [17. Label Horizon Governance](#17-label-horizon-governance)
- [18. Expected Artifacts By Stage](#18-expected-artifacts-by-stage)
- [19. Project Layout](#19-project-layout)
- [20. Release Status And Current Caveats](#20-release-status-and-current-caveats)
- [21. Troubleshooting And Operator Notes](#21-troubleshooting-and-operator-notes)

## 1. Project Purpose

The goal of Quant SMC is not merely to generate trading alerts. The goal is to maintain a coherent, inspectable decision system in which structural market context, engineered statistical state, supervised probabilities, execution gates, risk controls, and operator visibility all agree on the same underlying market state. In practice that means the repo is designed around deterministic transformations: raw market data becomes canonical bars, canonical bars become engineered features, engineered features become labels and model inputs, and model outputs become auditable trade decisions.

The repo is oriented toward discretionary-quant and semi-automated operation rather than blind black-box execution. Decisions should be inspectable, reproducible, and decomposable. That is why the system includes confluence breakdowns, hazard scores, EVR logic, reasoning payloads, and terminal/dashboards that expose the internal decision state rather than just a buy/sell light.

## 2. What This System Actually Does

At a practical level, the system does the following:

| Layer | Function | Output |
|---|---|---|
| Data | Fetches Kraken market history or stream updates, normalizes bars, rebuilds timeframes | `1m`, `15m`, `1h`, `6h`, `12h` canonical bars |
| Features | Computes SMC structure, liquidity state, EMA state, volatility state, session state, regime context | Engineered `15m` decision frame |
| Labels | Turns forward market behavior into supervised targets | Specialist labels + hazard survival targets |
| ML | Trains specialist models, stackers, hazard models, and quantile forecasters | Versioned model registry artifacts |
| Execution | Applies gates, confluence, EVR, tiering, sizing, profit ladder, and hazard exits | Entries, runners, exits, risk actions |
| Runtime | Replays this logic in backtest, forward, and live modes | Consistent decision flow across environments |
| UI / Telemetry | Publishes decision state to Streamlit and Next terminal surfaces | Real-time operator visibility and auditability |

## 3. Core Design Principles

### Deterministic Research Parity

The same feature graph should feed research, backtest, forward test, and live runtime. This is the central anti-drift principle of the repository. The `15m` row that is scored by the model stack should carry the same higher-timeframe context and engineered features regardless of whether it came from historical CSVs, forward simulation, or live aggregation.

### Multi-Timeframe Role Separation

The repo is not intended to throw one undifferentiated model at a mixed bag of bars. Each timeframe has a specific job, and the system is more coherent because those responsibilities are separated rather than blurred.

### Structured Decision Making

The system does not directly map raw features to a single opaque trade score. It separates the logic into gates, specialist probabilities, confluence, EVR, tiering, sizing, and trailing. This makes the decision path more understandable and easier to audit.

### Governance Over Constant Mutation

Defaults are allowed to exist as stable production baselines. Tuning and challenger logic can improve those defaults, but the repo is moving toward governed promotion rather than uncontrolled runtime mutation.

## 4. Timeframe Contract

This repo is wired around a fixed timeframe contract:

| Timeframe | Role | Current Use |
|---|---|---|
| `12h` | Regime | HMM + HDBSCAN regime features and macro context |
| `6h` | Structure | Structural bias, premium/discount, compression, zone quality |
| `1h` | Flow | Flow readiness and impulse/follow-through context; dedicated `flow_1h` model |
| `15m` | Execution | Main engineered decision frame and supervised execution layer |
| `1m` | Plumbing | Raw ingestion and live aggregation source |

This separation matters because the models and labels are not meant to answer the same question at every timeframe. `12h` is asking whether the environment is supportive. `6h` is asking whether price is sitting in a structurally meaningful location. `1h` is asking whether flow is fresh enough to matter. `15m` is asking whether to act now.

## 5. End-To-End Architecture

### Architectural Flow

```mermaid
flowchart LR
    A[Kraken Historical Fetch / Live Stream] --> B[Canonical 1m Bars]
    B --> C[15m / 1h / 6h / 12h Rebuild]
    C --> D[Multi-Timeframe Feature Graph]
    D --> E[Label Builder]
    D --> F[Inference Frame]
    E --> G[Model Training]
    G --> H[Registry + Versioned Artifacts]
    H --> F
    F --> I[Gates + Confluence + EVR + Tiering]
    I --> J[Capital + Risk + Profit Ladder + Hazard]
    J --> K[Backtest / Forward / Live]
    K --> L[FastAPI + WebSocket Telemetry]
    L --> M[Next Terminal / Streamlit Research Shell]
```

The high-level system loop is:

1. Fetch or receive market data.
2. Normalize and store canonical bars.
3. Rebuild `15m`, `1h`, `6h`, and `12h`.
4. Engineer a `15m` decision frame with higher-timeframe context attached.
5. Generate forward-looking labels from the engineered frame.
6. Train specialist models and downstream stackers.
7. Score incoming decision rows.
8. Apply gates, confluence, EVR, and tiering.
9. Size positions using current capital/risk policy.
10. Manage exits using the profit ladder and hazard logic.
11. Publish everything to telemetry and dashboards.

## 6. Data Pipeline

### Historical Acquisition

For BTCUSD, the canonical top-level fetch script is:

```bash
python fetch_BTCUSD_resample.py
```

This script is intentionally named for what it does. It is the correct manual starting point for a fresh BTCUSD pipeline run. It uses a Berlin-local default window:

- start: `2017-01-01 00:00:00 Europe/Berlin`
- end: yesterday `23:59:59 Europe/Berlin`

For shallow recent windows, the repo can use the standard Kraken OHLC path. For deep history, the fetch script automatically switches to the trades-backed bootstrap path because Kraken’s public OHLC endpoint does not provide years of reliable `1m` history.

### Checkpointing And Resume

The data layer is resumable. The current BTCUSD fetch path can persist and resume from canonical checkpoints rather than restarting from zero each time. For deep-history jobs, the fetch script also trims stale out-of-window data and rewinds checkpoints when necessary so the stored data remains aligned to the requested Berlin-local window.

### Timeframe Outputs

The fetch stage writes:

- `data/raw_1m/BTCUSD_1m.csv`
- `data/tf/BTCUSD_15m.csv`
- `data/tf/BTCUSD_1h.csv`
- `data/tf/BTCUSD_6h.csv`
- `data/tf/BTCUSD_12h.csv`

You do not need a separate manual resample step when using the top-level BTCUSD fetch script. The timeframe rebuild is part of that script.

## 7. Feature Engineering

### Why Features Matter

Training in this project is not meant to happen on raw exchange data. The model stack is designed to learn over engineered state, not over plain OHLC candles. That is why the feature engineering stage is a first-class step with its own launcher.

### Feature Launcher

```bash
python build_BTCUSD_features.py
```

This stage consumes the timeframe CSVs and produces the engineered `15m` feature frame used everywhere else.

### What The Feature Graph Contains

The engineered feature space includes:

| Family | Examples |
|---|---|
| SMC | swings, BOS, CHOCH, FVG, sweeps, zones |
| Structure context | structural bias, premium/discount, compression, zone score |
| EMA state | EMA distance, EMA relations, band regime |
| Volatility | ATR, realized vol, vol z-score, range percent |
| Liquidity | wick pressure, liquidity proximity, sweep strength, absorption proxies |
| Regime | `12h` regime state and regime probabilities |
| Session | session flags, session weights, session bucket state, minutes-from-open/to-overlap-close, session-relative volume/range/wick percentiles, Asia/London distance features |
| Joined HTF state | `1h`, `6h`, and `12h` context projected onto the `15m` spine |

### Main Feature Artifact

The canonical BTCUSD feature output is:

- `artifacts/features/BTCUSD/BTCUSD_features.csv`

This file is the substrate for labels and downstream training.

The session layer is now first-class. Beyond one-hot session flags, the feature graph carries explicit execution-time context such as bucketized market quality (`dead_zone`, `pre_expansion`, `expansion`, `overlap`), time-to/from key opens, overlap breakout markers, and session-relative diagnostics. This allows the runtime to distinguish structurally similar setups that occur in very different liquidity conditions.

## 8. Labeling And Supervised Targets

### Label Launcher

```bash
python build_BTCUSD_labels.py
```

This does not operate on raw bars. It operates on the engineered feature frame.

### What Labels Are Built

The canonical label builder creates these targets:

| Label | Purpose | High-Level Meaning |
|---|---|---|
| `label_liq_flow` | liquidity-flow specialist | did a sweep-led setup continue roughly `+1R` before invalidation? |
| `label_bos_cont` | BOS continuation specialist | did a BOS setup reach roughly `+3R` before invalidation? |
| `label_momo` | momentum specialist | did short-horizon forward return beat a noise-adjusted threshold? |
| `label_flow_1h` | `1h` flow specialist | did the current `1h` impulse context support follow-through? |
| `label_eop` | opportunity model | did a future high-opportunity setup appear within the horizon? |
| `label_edp` | danger model | did the market reach a significant drawdown condition? |
| `hazard_event`, `hazard_time` | hazard model | when does the first failure event happen, if at all? |

### Why These Labels Exist

They were chosen to match the actual questions the runtime asks:

- is this liquidity event worth trading?
- will this BOS continue?
- is `1h` flow sufficiently fresh?
- is momentum present?
- is upside opportunity forming?
- is downside danger increasing?
- how soon is failure likely?

These are not generic ML classes. They are trade-behavior labels expressed in the same language the execution engine uses: continuation, invalidation, `R`-multiples, drawdown, and event timing.

### Defaults vs Tuned Label Horizons

The current label horizons in `labels.yaml` are not random, but they are also not mathematically proven optimal. They are stable domain priors expressed in `15m` bars. The repo now supports empirical horizon tuning and governed promotion of better challenger profiles, described later in this README.

## 9. Model Stack

### Modeling Baseline

The current modeling contract is:

| Layer | Current Modeling Choice |
|---|---|
| `12h` regime | `HMM + HDBSCAN` |
| `6h` structure | deterministic context layer |
| `1h` flow | supervised tabular specialist |
| `15m` execution | supervised tabular specialists + stackers |
| auxiliary forecast | quantile forecaster now; `NARX` reserved for later soft use only |

### Training Contract

The trained models in this repository are not interchangeable generic classifiers. Each one exists to answer a specific trading question over a specific slice of the engineered state space. The correct way to reason about the stack is therefore model-by-model rather than as a single monolith.

Before model fitting, the trainer now applies a feature-selection hygiene pass: constant/near-constant pruning, high-missingness pruning, exact-duplicate removal, and Pearson-correlation clustering for numeric columns. Optional mutual-information filtering is then applied per classification target. This keeps specialist and stacker inputs cleaner and reduces redundancy propagation into confluence layers.

| Model | Target | Feature Scope | Why It Exists | Main Runtime Use |
|---|---|---|---|---|
| `liq_flow` | `label_liq_flow` | `15m` execution row with joined HTF context | evaluate sweep/liquidity reaction quality | contributes to entry-quality assessment |
| `bos_cont` | `label_bos_cont` | `15m` execution row with joined HTF context | evaluate continuation after structural break | continuation confidence |
| `flow_1h` | `label_flow_1h` | `1h` flow features plus allowed `6h/12h` context | determine whether higher-timeframe impulse is fresh enough | flow readiness gating and ranking |
| `momo` | `label_momo` | `15m` execution row | capture short-horizon directional energy | execution timing and near-term momentum |
| `eop` | `label_eop` | `15m` execution row | estimate upside opportunity state | opportunity ranking and positive pressure |
| `edp` | `label_edp` | `15m` execution row | estimate downside danger state | danger awareness and de-risking context |
| `meta_model` | stack over specialists | specialist outputs | produce an integrated second-layer probability | downstream fusion input |
| `confluence_model` | stack over specialists and meta | specialist outputs + meta outputs | represent the final learned decision strength | primary ranked confluence score |
| `hazard` | `hazard_event`, `hazard_time` | `15m` execution row | estimate failure timing / exit risk | trailing, tightening, exit pressure |
| `quantile` | direct future return distribution | `15m` execution row | estimate tail behavior and distribution shape | EV context, runner logic, risk shaping |

### Specialist Models

The repo currently trains these specialist families:

- `liq_flow`
- `bos_cont`
- `flow_1h`
- `momo`
- `eop`
- `edp`

These are trained on the engineered `15m` decision frame, with the proper higher-timeframe feature scope already joined in.

### Downstream Models

On top of the specialist layer, the repo trains:

- `meta_model`
- `confluence_model`
- `hazard`
- `quantile`

`meta_model` and `confluence_model` are stackers. `hazard` models failure timing. `quantile` provides return-distribution shape for risk and runner logic.

### NARX Status

`NARX` remains outside the main runtime path. The agreed repo policy is:

- allowed later as a soft ranking / runner / moonshot enhancer
- not used as a hard guardrail
- not used for cooling triggers
- not used to override the main `12h/6h/1h/15m` decision stack

### Inference Contract

At inference time, the predictor is expected to return a structured output rather than a single scalar. The downstream execution layer expects specialist probabilities, meta probability, confluence probability, hazard shape, and quantile outputs in a schema-stable form. That is why model registry artifacts persist their feature lists and why the predictor reads those persisted feature contracts instead of guessing from column names at runtime.

### Current Persisted Selected Feature Contracts

The following lists are the current BTCUSD specialist feature contracts persisted by the trained tree winners. These are post-hygiene, post-filtering, post-selection input lists, not the raw pre-filter feature universe. When TCN training is configured to align to the tree manifest, it inherits these same selected columns for fair comparison.

#### `liq_flow`

```python
[
    "swing_equal_high",
    "swing_equal_low",
    "bos_up",
    "bos_down",
    "choch_up",
    "choch_down",
    "bos_flag",
    "choch_flag",
    "bias",
    "fvg_up",
    "fvg_down",
    "fvg_touch_flag",
    "fvg_ctx_dir",
    "fvg_ctx_age",
    "fvg_ctx_weight",
    "fvg_ctx_fresh",
    "fvg_ctx_stale",
    "fvg_ctx_dist_top",
    "fvg_ctx_dist_bot",
    "sweep_high",
    "sweep_low",
    "sweep_flag",
    "sweep_dir",
    "demand_quality",
    "demand_touched",
    "supply_quality",
    "supply_age",
    "supply_touched",
    "zone_id",
    "zone_recency",
    "zone_displacement",
    "zone_mitigation",
    "displacement_15m",
    "fresh_retest_15m",
    "compression_6h",
    "bos_flag_1h",
    "choch_flag_1h",
    "bias_1h",
    "sweep_flag_1h",
    "demand_quality_1h",
    "supply_quality_1h",
    "demand_age_1h",
    "supply_age_1h",
    "zone_recency_1h",
    "zone_displacement_1h",
    "zone_mitigation_1h",
    "bos_flag_6h",
    "choch_flag_6h",
    "sweep_flag_6h",
    "demand_quality_6h",
    "supply_quality_6h",
    "supply_age_6h",
    "zone_recency_6h",
    "zone_displacement_6h",
    "zone_mitigation_6h",
    "bos_flag_12h",
    "choch_flag_12h",
    "sweep_flag_12h",
    "demand_quality_12h",
    "supply_quality_12h",
    "demand_age_12h",
    "supply_age_12h",
    "zone_recency_12h",
    "zone_displacement_12h",
    "ema_slope_21_15m",
    "band_regime_21_15m",
    "ema_slope_55_15m",
    "band_regime_55_15m",
    "dist_to_ema_15m",
    "ema_alignment_15m",
    "band_regime_50_1h",
    "band_regime_200_1h",
    "ema_alignment_1h",
    "band_regime_100_6h",
    "band_regime_200_12h",
    "liq_eq_high_density",
    "liq_eq_low_density",
    "liq_sweep_strength",
    "liq_displacement_ratio",
    "liq_wick_pressure",
    "liq_volume_pressure",
    "liq_near_pool_dist",
    "liq_sweep_lag1",
    "liq_sweep_lag2",
    "liq_sweep_lag3",
    "liq_sweep_strength_lag1",
    "liq_sweep_strength_lag2",
    "liq_wick_pressure_lag3",
    "liq_near_pool_dist_lag1",
    "liq_near_pool_dist_lag2",
    "liq_near_pool_dist_lag3",
    "atr",
    "realized_vol",
    "atr_slope",
    "vol_zscore",
    "dollar_volume",
    "absorption_score",
    "absorption_near_entry",
    "absorption_near_stop",
    "vol_pct",
    "trend_persist",
    "session_london",
    "session_ny",
    "session_overlap",
    "session_offhours",
    "session_name",
    "session_bucket",
    "session_bucket_id",
    "session_pre_expansion",
    "session_expansion",
    "wick_asymmetry_15m",
    "session_volume_pct",
    "session_range_pct",
    "session_wick_asym_pct",
    "session_atr_pct",
    "bos_flag_lag1",
    "choch_flag_lag1",
    "fvg_touch_flag_lag1",
    "atr_rstd4",
    "atr_rstd16",
]
```

#### `bos_cont`

```python
[
    "swing_equal_high",
    "swing_equal_low",
    "bos_up",
    "bos_down",
    "choch_up",
    "choch_down",
    "bos_flag",
    "choch_flag",
    "bias",
    "fvg_up",
    "fvg_down",
    "fvg_touch_flag",
    "fvg_ctx_dir",
    "fvg_ctx_age",
    "fvg_ctx_weight",
    "fvg_ctx_fresh",
    "fvg_ctx_dist_top",
    "fvg_ctx_dist_bot",
    "sweep_high",
    "sweep_low",
    "sweep_flag",
    "sweep_dir",
    "demand_top",
    "demand_quality",
    "demand_touched",
    "supply_quality",
    "supply_age",
    "supply_touched",
    "zone_recency",
    "zone_displacement",
    "zone_mitigation",
    "displacement_15m",
    "fresh_retest_15m",
    "structural_bias_6h",
    "zone_score_6h",
    "bos_flag_1h",
    "choch_flag_1h",
    "bias_1h",
    "sweep_flag_1h",
    "demand_quality_1h",
    "supply_quality_1h",
    "demand_age_1h",
    "supply_age_1h",
    "zone_recency_1h",
    "zone_displacement_1h",
    "zone_mitigation_1h",
    "displacement_body_pct_1h",
    "ret_1h_1h",
    "range_pct_1h",
    "range_z_1h",
    "bos_flag_6h",
    "choch_flag_6h",
    "bias_6h_y",
    "sweep_flag_6h",
    "demand_quality_6h",
    "supply_quality_6h",
    "demand_age_6h",
    "supply_age_6h",
    "zone_recency_6h",
    "zone_displacement_6h",
    "bos_flag_12h",
    "choch_flag_12h",
    "bias_12h",
    "sweep_flag_12h",
    "demand_quality_12h",
    "supply_quality_12h",
    "demand_age_12h",
    "supply_age_12h",
    "zone_recency_12h",
    "zone_displacement_12h",
    "ema_slope_21_15m",
    "band_regime_21_15m",
    "ema_slope_55_15m",
    "band_regime_55_15m",
    "dist_to_ema_15m",
    "ema_alignment_15m",
    "ema_slope_50_1h",
    "band_regime_50_1h",
    "ema_slope_200_1h",
    "band_regime_200_1h",
    "dist_to_ema_1h",
    "ema_alignment_1h",
    "ema_slope_100_6h",
    "band_regime_100_6h",
    "ema_slope_200_12h",
    "band_regime_200_12h",
    "dist_to_ema_12h",
    "atr",
    "realized_vol",
    "atr_slope",
    "vol_zscore",
    "vol_pct",
    "trend_persist",
    "compression_12h",
    "toxicity_12h",
    "compression_12h_lag1",
    "compression_12h_lag2",
    "p_regime_trend",
    "p_regime_expansion",
    "p_regime_collapse",
    "p_regime_range",
    "session_london",
    "session_ny",
    "session_overlap",
    "session_offhours",
    "session_name",
    "session_bucket",
    "session_bucket_id",
    "session_pre_expansion",
    "session_expansion",
    "session_volume_pct",
    "session_range_pct",
    "session_wick_asym_pct",
    "session_atr_pct",
    "bos_flag_lag1",
    "choch_flag_lag1",
    "sweep_flag_lag1",
    "fvg_touch_flag_lag1",
    "atr_rstd4",
    "atr_rstd8",
]
```

#### `flow_1h`

```python
[
    "structural_bias_6h",
    "zone_score_6h",
    "bos_flag_1h",
    "choch_flag_1h",
    "bias_1h",
    "sweep_flag_1h",
    "zone_id_1h",
    "zone_hi_1h",
    "demand_quality_1h",
    "supply_quality_1h",
    "demand_age_1h",
    "supply_age_1h",
    "zone_recency_1h",
    "zone_displacement_1h",
    "zone_mitigation_1h",
    "displacement_body_pct_1h",
    "body_dir_1h",
    "ret_1h_1h",
    "close_loc_1h",
    "range_pct_1h",
    "range_z_1h",
    "volume_z_1h",
    "flow_signal_1h",
    "flow_age_bars_1h",
    "flow_ok_1h",
    "flow_strength_1h",
    "ema_slope_50_1h",
    "dist_ema_50_1h",
    "z_dist_ema_50_1h",
    "band_regime_50_1h",
    "ema_slope_200_1h",
    "dist_ema_200_1h",
    "z_dist_ema_200_1h",
    "band_regime_200_1h",
    "ema_alignment_1h",
    "atr",
    "realized_vol",
    "vol_zscore",
    "vol_pct",
    "trend_persist",
    "p_regime_trend",
    "p_regime_expansion",
    "p_regime_collapse",
    "p_regime_range",
    "regime_state",
    "session_weight",
    "session",
]
```

#### `momo`

```python
[
    "bos_up",
    "bos_down",
    "choch_up",
    "bos_flag",
    "choch_flag",
    "fvg_down",
    "fvg_touch_flag",
    "fvg_ctx_dir",
    "fvg_ctx_age",
    "fvg_ctx_fresh",
    "fvg_ctx_dist_top",
    "fvg_ctx_dist_bot",
    "sweep_flag",
    "sweep_dir",
    "demand_top",
    "demand_quality",
    "demand_touched",
    "supply_quality",
    "supply_touched",
    "zone_displacement",
    "displacement_15m",
    "fresh_retest_15m",
    "structural_bias_6h",
    "pd_context_6h",
    "zone_score_6h",
    "bos_flag_1h",
    "choch_flag_1h",
    "bias_1h",
    "sweep_flag_1h",
    "demand_quality_1h",
    "supply_quality_1h",
    "demand_age_1h",
    "zone_displacement_1h",
    "displacement_body_pct_1h",
    "ret_1h_1h",
    "range_pct_1h",
    "range_z_1h",
    "bos_flag_6h",
    "choch_flag_6h",
    "bias_6h_y",
    "sweep_flag_6h",
    "demand_quality_6h",
    "supply_quality_6h",
    "demand_age_6h",
    "supply_age_6h",
    "zone_recency_6h",
    "zone_displacement_6h",
    "zone_mitigation_6h",
    "bos_flag_12h",
    "choch_flag_12h",
    "bias_12h",
    "sweep_flag_12h",
    "demand_quality_12h",
    "supply_quality_12h",
    "demand_age_12h",
    "supply_age_12h",
    "zone_recency_12h",
    "zone_displacement_12h",
    "ema_slope_21_15m",
    "band_regime_21_15m",
    "ema_slope_55_15m",
    "band_regime_55_15m",
    "dist_to_ema_15m",
    "ema_alignment_15m",
    "ema_slope_50_1h",
    "band_regime_50_1h",
    "ema_slope_200_1h",
    "band_regime_200_1h",
    "dist_to_ema_1h",
    "ema_alignment_1h",
    "ema_slope_100_6h",
    "band_regime_100_6h",
    "ema_slope_200_12h",
    "band_regime_200_12h",
    "dist_to_ema_12h",
    "liq_eq_high_density",
    "liq_eq_low_density",
    "liq_wick_pressure",
    "liq_near_pool_dist",
    "liq_sweep_lag1",
    "liq_sweep_lag2",
    "liq_sweep_lag3",
    "liq_wick_pressure_lag1",
    "liq_wick_pressure_lag2",
    "liq_near_pool_dist_lag1",
    "liq_near_pool_dist_lag3",
    "realized_vol",
    "atr_slope",
    "vol_zscore",
    "dollar_volume",
    "absorption_near_entry",
    "absorption_near_stop",
    "vol_pct",
    "trend_persist",
    "compression_12h",
    "toxicity_12h",
    "compression_12h_lag1",
    "compression_12h_lag2",
    "p_regime_trend",
    "p_regime_expansion",
    "p_regime_collapse",
    "p_regime_range",
    "session_london",
    "session_ny",
    "session_overlap",
    "session_offhours",
    "session_name",
    "session_bucket",
    "session_bucket_id",
    "session_pre_expansion",
    "session_expansion",
    "session_volume_pct",
    "session_range_pct",
    "session_wick_asym_pct",
    "session_atr_pct",
    "bos_flag_lag1",
    "choch_flag_lag1",
    "fvg_touch_flag_lag1",
    "atr_rstd8",
    "atr_rstd16",
]
```

#### `eop`

```python
[
    "bos_up",
    "bos_down",
    "choch_up",
    "choch_down",
    "bos_flag",
    "choch_flag",
    "bias",
    "fvg_up",
    "fvg_down",
    "fvg_touch_flag",
    "fvg_ctx_dir",
    "fvg_ctx_age",
    "fvg_ctx_weight",
    "fvg_ctx_fresh",
    "fvg_ctx_stale",
    "fvg_ctx_dist_top",
    "fvg_ctx_dist_bot",
    "sweep_flag",
    "sweep_dir",
    "demand_top",
    "demand_quality",
    "demand_age",
    "demand_touched",
    "supply_quality",
    "supply_touched",
    "zone_id",
    "zone_recency",
    "zone_displacement",
    "displacement_15m",
    "fresh_retest_15m",
    "structural_bias_6h",
    "pd_context_6h",
    "zone_score_6h",
    "bos_flag_1h",
    "choch_flag_1h",
    "bias_1h",
    "sweep_flag_1h",
    "zone_id_1h",
    "demand_quality_1h",
    "supply_quality_1h",
    "demand_age_1h",
    "supply_age_1h",
    "zone_recency_1h",
    "zone_displacement_1h",
    "zone_mitigation_1h",
    "displacement_body_pct_1h",
    "ret_1h_1h",
    "range_pct_1h",
    "range_z_1h",
    "bos_flag_6h",
    "choch_flag_6h",
    "bias_6h_y",
    "zone_id_6h",
    "demand_quality_6h",
    "supply_quality_6h",
    "demand_age_6h",
    "supply_age_6h",
    "zone_recency_6h",
    "zone_displacement_6h",
    "zone_mitigation_6h",
    "bos_flag_12h",
    "choch_flag_12h",
    "bias_12h",
    "sweep_flag_12h",
    "zone_id_12h",
    "demand_quality_12h",
    "supply_quality_12h",
    "demand_age_12h",
    "supply_age_12h",
    "zone_recency_12h",
    "zone_displacement_12h",
    "band_regime_21_15m",
    "ema_slope_55_15m",
    "band_regime_55_15m",
    "dist_to_ema_15m",
    "ema_alignment_15m",
    "ema_slope_50_1h",
    "band_regime_50_1h",
    "ema_slope_200_1h",
    "band_regime_200_1h",
    "dist_to_ema_1h",
    "ema_alignment_1h",
    "ema_slope_100_6h",
    "band_regime_100_6h",
    "ema_slope_200_12h",
    "band_regime_200_12h",
    "dist_to_ema_12h",
    "atr",
    "realized_vol",
    "atr_slope",
    "vol_zscore",
    "vol_pct",
    "trend_persist",
    "compression_12h",
    "toxicity_12h",
    "compression_12h_lag1",
    "compression_12h_lag2",
    "p_regime_trend",
    "p_regime_expansion",
    "p_regime_collapse",
    "p_regime_range",
    "session_london",
    "session_ny",
    "session_offhours",
    "session_name",
    "session_bucket",
    "session_bucket_id",
    "session_pre_expansion",
    "session_expansion",
    "session_volume_pct",
    "session_range_pct",
    "session_wick_asym_pct",
    "session_atr_pct",
    "bos_flag_lag1",
    "choch_flag_lag1",
    "sweep_flag_lag1",
    "fvg_touch_flag_lag1",
    "atr_rstd4",
    "atr_rstd8",
    "atr_rstd16",
]
```

#### `edp`

```python
[
    "swing_equal_high",
    "swing_equal_low",
    "bos_up",
    "bos_down",
    "choch_up",
    "choch_down",
    "bos_flag",
    "choch_flag",
    "bias",
    "fvg_up",
    "fvg_down",
    "fvg_touch_flag",
    "fvg_ctx_dir",
    "fvg_ctx_age",
    "fvg_ctx_weight",
    "fvg_ctx_fresh",
    "fvg_ctx_stale",
    "fvg_ctx_dist_top",
    "fvg_ctx_dist_bot",
    "sweep_high",
    "sweep_low",
    "sweep_flag",
    "sweep_dir",
    "demand_top",
    "demand_quality",
    "demand_age",
    "demand_touched",
    "supply_quality",
    "supply_age",
    "supply_touched",
    "zone_id",
    "zone_recency",
    "zone_displacement",
    "zone_mitigation",
    "displacement_15m",
    "fresh_retest_15m",
    "structural_bias_6h",
    "pd_context_6h",
    "compression_6h",
    "zone_score_6h",
    "bos_flag_1h",
    "choch_flag_1h",
    "bias_1h",
    "sweep_flag_1h",
    "zone_id_1h",
    "demand_quality_1h",
    "supply_quality_1h",
    "demand_age_1h",
    "supply_age_1h",
    "zone_recency_1h",
    "zone_displacement_1h",
    "zone_mitigation_1h",
    "displacement_body_pct_1h",
    "ret_1h_1h",
    "range_pct_1h",
    "range_z_1h",
    "bos_flag_6h",
    "choch_flag_6h",
    "bias_6h_y",
    "sweep_flag_6h",
    "zone_id_6h",
    "demand_quality_6h",
    "supply_quality_6h",
    "demand_age_6h",
    "supply_age_6h",
    "zone_recency_6h",
    "zone_displacement_6h",
    "zone_mitigation_6h",
    "bos_flag_12h",
    "choch_flag_12h",
    "bias_12h",
    "sweep_flag_12h",
    "zone_id_12h",
    "demand_quality_12h",
    "supply_quality_12h",
    "demand_age_12h",
    "supply_age_12h",
    "zone_recency_12h",
    "zone_displacement_12h",
    "zone_mitigation_12h",
    "atr",
    "realized_vol",
    "atr_slope",
    "vol_zscore",
    "vol_pct",
    "trend_persist",
    "compression_12h",
    "toxicity_12h",
    "compression_12h_lag1",
    "compression_12h_lag2",
    "p_regime_trend",
    "p_regime_expansion",
    "p_regime_collapse",
    "p_regime_range",
    "regime_state",
    "session_london",
    "session_ny",
    "session_overlap",
    "session_offhours",
    "session_name",
    "session_bucket",
    "session_bucket_id",
    "session_pre_expansion",
    "session_expansion",
    "session_volume_pct",
    "session_range_pct",
    "session_wick_asym_pct",
    "session_atr_pct",
    "bos_flag_lag1",
    "choch_flag_lag1",
    "sweep_flag_lag1",
    "fvg_touch_flag_lag1",
    "atr_rstd4",
    "atr_rstd8",
    "atr_rstd16",
]
```

## 9A. Temporal Convolutional Network (TCN) Specialist Benchmark Stack

The repository now includes a full deep-learning specialist benchmark path based on a Temporal Convolutional Network (TCN). It is not a side notebook path. It is integrated into the same training artifact, registry, and telemetry contract used by the tabular stack.

### Why TCN Is Added

TCN is used as a sequence-aware challenger to tabular specialists on the same engineered feature frame and labels. The intent is not to replace the stack blindly, but to evaluate whether a causal sequence model can extract extra edge from the time-ordered feature state that tree models might compress away.

### Architecture

The TCN trainer is implemented in [`quant_system/ml/training/tcn_trainer.py`](./quant_system/ml/training/tcn_trainer.py). Core model structure:

- causal dilated 1D convolutions
- residual temporal blocks
- configurable depth (`levels`)
- configurable channel width (`channels`)
- configurable kernel (`kernel_size`)
- dropout regularization
- sigmoid binary head

Each sample is a rolling lookback window ending at decision bar `t`. The model predicts probability for the label at that endpoint, preserving causality.

### Data, Features, And Labels

TCN runs consume the same merged training frame as tree models:

- features: `artifacts/features/BTCUSD/BTCUSD_features.csv`
- labels: `artifacts/labels/BTCUSD/BTCUSD_labels.csv`

Target mapping:

| TCN Target | Label Column |
|---|---|
| `liq_flow` | `label_liq_flow` |
| `bos_cont` | `label_bos_cont` |
| `momo` | `label_momo` |
| `flow_1h` | `label_flow_1h` |
| `eop` | `label_eop` |
| `edp` | `label_edp` |

Feature scope defaults to `tree_manifest` alignment, meaning TCN can inherit selected feature columns from the corresponding tree manifest for fair comparison. If unavailable, it falls back to a safe auto feature set.

### Preprocessing Contract

Inside each fold, preprocessing is fit on training indices only (leak-safe):

- numeric imputation (`median` default)
- categorical imputation (`most_frequent` default)
- optional quantile clipping (winsorization)
- optional scaling (`standard`, `robust`, or none)
- one-hot encoding for categorical features

This preprocessing is implemented with a `ColumnTransformer` pipeline and serialized into the persisted inference bundle.

### CV, Embargo, And Objective

TCN HPO uses time-series CV (`TimeSeriesSplit`) with embargo bars to reduce leakage around split boundaries. For each trial:

1. sample params from Optuna search space
2. run purged/embargoed fold evaluations
3. report partial fold objective to Optuna
4. allow trial pruning when underperforming

Primary objective is classification quality over out-of-fold probabilities (AP/AUC-derived score, task-dependent safety fallbacks for degenerate folds).

### HPO Search Space And Runtime Controls

The default TCN search space is in `quant_system/config/models/models.yaml` under `tcn_training.default.hpo_space`, including:

- `lookback`
- `levels`
- `channels`
- `kernel_size`
- `dropout`
- `lr`
- `weight_decay`
- `batch_size`
- `max_epochs`
- `patience`

Current default TCN policy:

- `hpo_trials: 20`
- `cv_splits: 4`
- `hpo_adaptive_stop: true`
- `hpo_adaptive_min_completed_trials: 10`
- `hpo_adaptive_no_improve_trials: 6`
- `hpo_adaptive_min_delta: 0.001`

Runtime specialist inference source preference is configured in `quant_system/config/models/models.yaml`:

- `inference_preference.prefer_tcn_specialists: true` (default)

With this flag on, forward/live/backtest predictor resolution is:

1. try `<specialist>_tcn`
2. fallback to `<specialist>` tree model

Downstream keys remain canonical (`prob_liq_flow`, `p_liq_flow`, etc.), so existing gating/confluence/EVR logic is preserved.

### Resume, Checkpointing, And Progress Files

Each target writes to its own isolated artifact root:

- `artifacts/train/BTCUSD/<target>_tcn/`

Per-target HPO persistence:

- SQLite Optuna storage: `optuna_<target>.db`
- latest progress snapshot: `hpo_progress.json`
- append-only event stream: `hpo_progress.ndjson`

This allows interruption-safe resume without recomputing completed trials.

### Adaptive Plateau Stop (Automatic Early Stop)

If adaptive stop is enabled, HPO stops when both conditions hold:

1. completed trials >= `hpo_adaptive_min_completed_trials`
2. no best-score improvement of at least `hpo_adaptive_min_delta` for `hpo_adaptive_no_improve_trials`

This is budget-aware stopping, not score-target stopping. It is designed to avoid burning compute in flat search regions while still allowing exploration.

### Post-HPO Training Gates

After best-trial selection, the trainer runs additional safeguards:

- stability check across multiple seeds
- acceptance holdout gate with score-drop and precision constraints
- probability calibration (`auto`/Platt/isotonic/empirical)
- threshold tuning for deployment decision cutoff (F1/precision/recall constrained)

### Saved Model Artifacts

For each completed TCN specialist run:

- registry model saved as `BTCUSD_<target>_tcn` and alias `<target>_tcn`
- versioned metrics saved in registry
- `model_state.json` and `train_manifest.json` in target artifact folder

This means trained TCN outputs are inference-ready and version-tracked, not just trial logs.

## 9B. Temporal Convolutional Network (TCN) Design Rationale And Learning Mechanics

This section explains, at model-mechanics level, how TCN is used in this repo and why it was selected as the deep-learning challenger.

### Why TCN Instead Of Defaulting To LSTM/GRU

TCN was chosen because it gives a better operational tradeoff for this stack:

1. Causal convolutions make leakage control explicit.
2. Dilations capture long context without recurrent state propagation.
3. Training is generally more stable and parallelizable than recurrent unrolling.
4. It works naturally on the already-engineered `15m` decision frame.
5. It scales well with high-dimensional tabular-plus-context feature matrices.

TCN is not assumed to always beat trees or LSTM. It is the preferred deep candidate for this repository's data contract and runtime constraints.

### What Happens To Data Before Training

For each target, training operates on the merged feature+label frame. The flow is:

1. Resolve feature columns (prefer tree-manifest-selected features for fair A/B comparison).
2. Apply leak-safe preprocessing per fold:
   - imputation
   - optional clipping
   - optional scaling
   - one-hot for categoricals
3. Convert transformed rows into rolling sequences of shape `[B, T, F]`.
4. Use endpoint label at index `t` for each sequence ending at `t`.

So the model learns from feature trajectories through time, not independent static rows.

### How Multi-Timeframe Information Enters TCN

Multi-timeframe handling occurs in two layers:

1. Feature engineering aligns `1h`, `6h`, and `12h` context onto each `15m` row.
2. TCN learns temporal interactions across those aligned rows.

This means cross-timeframe effects are available both as instantaneous context (`15m` row fields) and as temporal evolution across the lookback window.

### Long-Term Dependency Handling

TCN handles long-range dependencies via:

- increasing dilation by block depth
- stacked temporal blocks
- residual skip paths

Compared with plain CNN stacks, dilation expands receptive field quickly. Compared with recurrent models, it avoids long-chain hidden-state transport and associated gradient instability.

### How "Neurons" Behave In This TCN

At implementation level, each 1D conv filter acts as a learned time-pattern detector over feature channels. Typical detector behavior:

- short-term filters: local impulse/liquidity/session transitions
- deeper dilated filters: slower structural/regime flow interactions

After each conv:

- `ReLU` keeps nonlinear capacity
- dropout regularizes
- residual adds optimization stability

The final head reads the last temporal state and outputs a logit, then sigmoid probability.

### How Training Actually Proceeds

Per target, the pipeline runs:

1. time-series CV with embargo
2. Optuna HPO over architecture + optimizer + training knobs
3. pruner-based early culling of weak trials
4. best-config re-evaluation + seed-stability checks
5. final fit on development split
6. probability calibration
7. threshold tuning with precision/recall constraints
8. acceptance gate checks
9. model + metrics + manifests persisted to registry/artifacts

### High-Dimensional Feature Behavior: TCN vs Trees

For this repo, high-dimensional engineered features include numeric structure metrics, session/regime context, and categorical expansions.

Why TCN can work well:

- temporal filters can learn cross-feature interactions over time directly
- shared filters across timesteps provide parameter efficiency
- sequence modeling can capture order-sensitive motifs that row-wise trees only see via handcrafted lags

Why trees still remain strong:

- excellent tabular bias
- fast iteration
- robust with smaller compute budgets

Practical policy here: treat tree stack as baseline, TCN as challenger, and promote by measured out-of-sample/stability/acceptance evidence rather than architecture preference.

### TCN vs Tree vs LSTM (Project-Specific View)

| Axis | Trees (`LGBM/XGB`) | TCN | LSTM/GRU |
|---|---|---|---|
| Tabular baseline strength | high | medium-high | medium |
| Native sequence modeling | low (needs lag engineering) | high | high |
| Long-context efficiency | medium | high (dilation) | medium |
| Parallel training efficiency | high | high | lower |
| Runtime/ops simplicity | high | medium | medium-low |
| Fit for this repo's current contract | baseline production | primary deep challenger | optional deep alternative |

## 10. Execution, Risk, And Capital Logic

### Decision Flow

The execution stack is layered:

1. higher-timeframe gates
2. confluence
3. EVR
4. tiering
5. sizing
6. entry
7. profit ladder
8. hazard trailing
9. cooling / reset logic

### Capital Policy

The current baseline is a ticket-based capital policy:

| Parameter | Current Default |
|---|---|
| starting equity | `20,000 USD` |
| base ticket | `20,000 USD` |
| compounding | enabled |
| cooling / vault reset | enabled |
| MPC | available but disabled by default |

This means the system compounds during healthy cycles, but it is designed to protect profits and cool down when danger signals intensify.

### Session-Aware Monetization Policy

Execution policy now includes a configurable session layer that modulates:

- confluence score multiplier and threshold offsets by session bucket
- pre-trade gate strictness by session bucket (flow/hazard/session-weight requirements)
- position size multipliers by session bucket
- ticket capital multipliers by session bucket

This keeps the engine running 24/7 for telemetry while explicitly treating overlap and expansion windows as higher-quality monetization periods than dead-hours flow.

### Profit Ladder And Hazard

The repaired runtime now supports a more explicit trade-management structure:

- core/runner split
- profit ladder logic
- hazard-based trailing
- longer hold bias for stronger continuation conditions

## 10A. Operator Guide: Reading Confluence, EVR, And Hazard Together

This is the operator-facing interpretation layer for the three most important runtime decision variables:

- `confluence`
- `EVR`
- `hazard`

They do not mean the same thing and they should not be read independently.

### What Each Number Actually Means

| Variable | What It Measures | Plain-English Meaning | Primary Use |
|---|---|---|---|
| `confluence` | agreement and setup quality across rule logic and model context | “Does the setup look coherent enough to trust?” | entry quality / ranking |
| `EVR` | expectancy-style reward profile | “If this works, is the payoff profile worth the risk?” | opportunity quality / trade worthiness |
| `hazard` | near-term adverse-event risk | “How likely is this to go wrong soon?” | fragility, trailing, exits, cooling |

The shortest useful mental model is:

- `confluence` = quality
- `EVR` = payoff
- `hazard` = danger

If you only remember one line, remember that one.

### How To Read Them Together

The system is not asking for one magic number. It is asking for a structured answer:

1. Is the setup coherent?
2. Is the payoff worth taking?
3. Is the environment too fragile to trust?

That is why the runtime keeps these surfaces separate instead of collapsing everything into one opaque score.

### Practical Interpretation Matrix

| Confluence | EVR | Hazard | Operator Read |
|---|---|---|---|
| high | high | low | best case; strong setup, worthwhile payoff, limited fragility |
| high | high | high | attractive but dangerous; size/hold logic must stay defensive |
| high | low | low | coherent but not worth much; a clean-looking mediocre trade is still mediocre |
| low | high | low | tempting but weakly aligned; often the trade that looks clever after the fact and annoying in real time |
| low | low | low | no edge, no urgency, no reason |
| high | low | high | worst kind of false friend; looks organized, pays badly, fails easily |

### Confluence: What It Should Mean To An Operator

Confluence is not “up probability.” It is setup coherence.

It tells you whether the decision stack is seeing:

- agreement between specialists
- reasonable structural context
- acceptable rule-based execution state
- enough supporting context to take the setup seriously

Operationally:

- high confluence means the setup is internally consistent
- mediocre confluence means signals are mixed, incomplete, or thin
- low confluence means the market is not presenting a convincing execution-grade picture

Confluence should make you ask:

- “Is this worth paying attention to?”

It should not make you say:

- “This must win.”

That confusion is expensive.

### EVR: Why A Good Setup Still May Not Be Worth Taking

EVR is the trade-worthiness layer.

It asks whether the projected reward profile is attractive enough relative to the risk and distribution shape. In practice, EVR helps separate:

- clean but small setups
- noisy but asymmetric setups
- genuinely high-quality payoff situations

Operationally:

- high EVR means the upside profile is meaningful
- mediocre EVR means the trade may work but does not pay enough
- low EVR means the opportunity is not worth much even if the chart looks organized

This is the variable that stops the operator from taking every “nice-looking” setup.

### Hazard: The System’s Fragility Meter

Hazard is the danger model.

It does not answer:

- “Will price go up?”

It answers:

- “How likely is an adverse event soon?”

In this repository, hazard is built from future stop-like failure and opposite-structure-event timing. At runtime, the predictor produces a hazard curve and then compresses that curve into a single `hazard_score` with higher weight on earlier bars. That means the score cares more about danger soon than danger eventually.

Operationally:

- low hazard means the trade has room to breathe
- rising hazard means the setup is becoming fragile
- high hazard means the system should protect capital, tighten, partial, or exit

Hazard influences:

- trade tiering
- entry strictness
- position risk mode
- hedge ratio
- cooling / profit locking
- trailing and forced exit behavior

If confluence is the “why enter?” voice, hazard is the “do not get stubborn” voice.

### The Correct Operator Workflow

The correct reading order is:

1. Check `confluence`
2. Check `EVR`
3. Check `hazard`
4. Only then decide whether the setup is worth acting on

This avoids the two classic operator mistakes:

- taking a coherent but low-payoff trade
- holding a high-payoff-looking trade after hazard has already turned against it

### Recommended Operator Rules

Use the variables like this:

- High `confluence` and high `EVR` with low `hazard`: this is the cleanest actionable state.
- High `confluence` and high `EVR` with medium/high `hazard`: allow the setup to exist, but expect tighter risk handling and lower tolerance for drift.
- High `confluence` with low `EVR`: do not confuse orderliness with profitability.
- Low `confluence` with high `EVR`: treat this as a speculative outlier, not a core process trade.
- Rising `hazard` after entry: stop asking whether the original thesis was beautiful. Start asking whether capital now needs defending.

### What Not To Do

Do not use these values incorrectly:

- do not read `hazard` as a directional short signal; it is a fragility signal
- do not read `confluence` as guaranteed win probability
- do not read `EVR` as permission to ignore risk
- do not compare raw scores across different model targets as if they are on the same task scale

### Example Reads

#### Example 1: Strong Candidate

- `confluence` high
- `EVR` high
- `hazard` low

Read:

- the setup is coherent
- the payoff looks worthwhile
- the environment is not yet screaming “protect immediately”

This is the type of setup the system is designed to prioritize.

#### Example 2: Good Looking, Fragile

- `confluence` high
- `EVR` high
- `hazard` high

Read:

- the setup may still be attractive
- but the market is also telling you the failure clock is active

This should lead to stricter sizing, faster trailing, or quicker partial behavior, not blind confidence.

#### Example 3: Pretty But Cheap

- `confluence` high
- `EVR` low
- `hazard` low

Read:

- the setup is orderly
- but the upside distribution is weak

This is the kind of trade that fills journals and does not move equity much.

#### Example 4: Tempting But Undisciplined

- `confluence` low
- `EVR` high
- `hazard` low

Read:

- payoff might be interesting
- but setup agreement is weak

This is where discretionary operators often start narrating. The system is correct to stay stricter than the operator’s imagination.

### How This Maps To The Dashboard

On the operator surfaces:

- `confluence` should be read as setup coherence
- `EVR` should be read as trade economics
- `hazard` should be read as fragility and exit pressure

If the dashboard shows all three in one place, the operator should be able to answer:

- “Is this good enough to act on?”
- “Is it worth enough to act on?”
- “Is it stable enough to hold?”

That is the real job of the stack.

### Final Operator Summary

When the three are aligned:

- high `confluence`
- high `EVR`
- low `hazard`

the system has both permission and economic reason to care.

When they disagree, believe the disagreement. The disagreement is information, not clutter.

That is the point of building a layered trading system instead of a single glowing button.

## 10B. Post-Backtest Trade-Policy Overlay

Once a backtest has been run, the repo can now build a second-stage dataset from `ledger.csv` and optional `reasoning.json` and train a trade-policy overlay on those executed-trade rows. This is not a replacement for the primary model stack. It is a post-backtest layer intended to answer:

- should this kind of trade be trusted more or less at entry?
- what realized `R` distribution should be expected for this setup family?
- should a future deployment use this information for veto, ranking, or size adjustment?

### Why The Default Model Family Is Tabular, Not Sequence-Based

The backtest policy dataset is not raw candle history. Each row is already a compressed entry-state observation containing:

- `confluence`, `EVR`, `hazard_entry`
- tier, regime, session, leg, side
- gate checks and gate reasons
- sizing and stop context
- optional reasoning-derived probabilities and decomposition fields

That is a tabular decision-state problem. For this data shape, the right default is:

- linear baseline for calibration and interpretability
- tree challengers for nonlinear interaction capture
- quantile regression for realized-`R` distribution

This is why the trade-policy trainer uses:

- `logistic` as the baseline quality model
- `lightgbm` and `xgboost` as challengers
- LightGBM quantile regression for realized-`R`

Sequence architectures such as `TCN`, `LSTM`, or `NARX` are not the first choice here because the temporal structure has already been summarized upstream into a single entry row. Those models are appropriate earlier in the stack when learning directly from ordered feature windows. They are not the best first instrument for a ledger-driven policy overlay.

### What Gets Built

After a backtest, the CLI now writes:

- `trade_policy_dataset.csv`
- `trade_policy_dataset_meta.json`

The dataset builder keeps only entry-safe features as model inputs and leaves realized outcomes as targets:

- `label_trade_positive`
- `label_trade_nonnegative`
- `label_trade_ge_1r`
- `target_realized_r`
- `target_realized_pnl`
- `target_duration_min`

### Training Path

The top-level launcher is:

```bash
python train_BTCUSD_trade_policy_models.py --backtest-dir artifacts/backtest/latest
```

This produces a separate training manifest under:

- `artifacts/train/BTCUSD/trade_policy/train_manifest.json`

and saves versioned models to the normal registry:

- `trade_policy_quality`
- `trade_policy_return`

### What The Two Models Are For

`trade_policy_quality`
- binary classifier over trade quality
- calibrated and threshold-tuned
- intended for veto, acceptance tightening, or rank ordering

`trade_policy_return`
- realized-`R` quantile forecaster
- intended for sizing and payoff-shape estimation
- best used as a distributional overlay, not as a single magic target

### Validation Standard

This overlay must be held to a higher standard than a normal backtest summary. It should be trained only on past trade rows and validated on later trade rows by `entry_ts`. The correct deployment order is:

1. base models generate trades
2. backtest artifacts create the trade-policy dataset
3. trade-policy models are trained on historical trade rows
4. trade-policy overlay is shadowed in forward or paper runtime
5. only then should it influence production sizing or veto logic

That order matters because post-backtest overlays are otherwise an easy place to create elegant but non-transferable overfit.

## 11. Runtime Modes

### Backtest

Historical execution and analytics are under `quant_system/backtest/`. The core backtester is the authoritative historical runtime path.

Primary use:
- research validation
- trade distribution analysis
- equity curve and drawdown analysis
- replay exports and post-trade audit

### Forward Test

The forward runtime in `quant_system/forward_test/` mirrors live-style decisioning on new bars without placing real exchange orders.

Primary use:
- pre-production runtime validation
- model and execution parity checks
- dashboard and reasoning inspection under fresh incoming data

### Live

The live runtime in `quant_system/live/` mirrors the repaired forward logic and publishes structured telemetry. The transport architecture is moving toward a stricter shared event plane rather than independent console/UI logic.

Primary use:
- real-time alerting
- execution intent generation
- operator monitoring
- telemetry publishing to the terminal and research surfaces

## 12. Dashboards, Terminal, And Telemetry

### Streamlit Research Shell

The Streamlit layer remains the research, audit, and reporting surface:

```bash
streamlit run quant_system/dashboard/app.py
```

### Next Operator Terminal

The live operator UI is now a real frontend:

- `Next.js`
- `React`
- `Tailwind`
- `FastAPI + WebSocket` backend transport

Start backend:

```bash
python -m quant_system.cli.forward_cli --terminal-server
```

Start frontend:

```bash
cd frontend
npm run dev
```

### Terminal Domains

The terminal is organized into six domains:

1. `Mission Control`
2. `Insights`
3. `Regime Briefings`
4. `Signal Intelligence`
5. `Risk Radar`
6. `Research & Audit`

The performance domain now includes TradeZella-style richer analytics payloads from telemetry:

- attribution by model/session/regime/hour/hold bucket
- expectancy metrics (avg win/loss, payoff, streaks, max drawdown)
- top winners/losers
- equity, daily, and monthly timelines
- expanded trade table with session/regime/MAE/MFE context

### Reasoning Payloads

Forward/live alerts carry structured reasoning built in [`forward_reasoning_attach.py`](./quant_system/forward_test/forward_reasoning_attach.py), including:

- `ml`
- `smc`
- `regime`
- `flow`
- `ema`
- `confluence_breakdown`
- `evr`
- `hazard`
- `final_decision`
- `session`

These can be inspected in both the Streamlit shell and the Next terminal.

## 13. Installation

### Prerequisites

The baseline environment assumes:

| Component | Recommended |
|---|---|
| Python | `3.11` |
| Node.js | `>= 20` |
| Package manager | `pip` and `npm` |
| Optional shell | `conda` recommended for isolation |
| OS | macOS or Linux-like environment |

### Python

```bash
conda create -n quant_smc python=3.11 -y
conda activate quant_smc
pip install -r requirements.txt
```

### Frontend

```bash
cd frontend
npm install
```

### Backend + Frontend Quick Start

Once models and artifacts exist, the normal operator startup flow is:

```bash
python -m quant_system.cli.forward_cli --terminal-server
cd frontend
npm run dev
```

Optional research shell:

```bash
streamlit run quant_system/dashboard/app.py
```

### Note About Local Python

The top-level BTCUSD launcher scripts now include a small bootstrap helper so they can recover if launched with the wrong Python binary. That helps when `pandas` or other repo dependencies are available in your conda Python but not in `/usr/bin/python3`.

## 14. Environment And Secrets

Create `.env` from [`.env.example`](./.env.example).

Important variables:

- `KRAKEN_API_KEY`
- `KRAKEN_API_SECRET`
- `KRAKEN_OTP` (optional, only if Kraken account requires OTP for private endpoints)
- `QS_ENV=DEV`

Frontend env template:

- `QUANT_TERMINAL_API_URL=http://127.0.0.1:8100/snapshot`
- `NEXT_PUBLIC_TERMINAL_WS_URL=ws://127.0.0.1:8100/ws/terminal`

Do not commit exchange credentials. The config-local `secrets.env` is a sensitive file and should not be treated as safe for source control.

Live order routing is API-driven. The system does not require UI scraping or Kraken web-interface layout knowledge for automated execution.

### Live Leverage Guardrails

The live orchestrator can automatically step from `1x` to `2x` leverage only when a setup is marked high-confidence. This is controlled in `quant_system/config/execution/execution.yaml` under `live_trading.leverage`:

- `enabled`: global leverage switch
- `default`: baseline leverage (normally `1`)
- `high_conf_leverage`: target leverage when confidence policy passes (set to `2`)
- `high_conf_tiers`: tier allowlist (default `["A+"]`)
- `high_conf_min_conf`, `high_conf_min_evr`, `high_conf_max_hazard`, `high_conf_min_bos_cont`: quality thresholds

If thresholds fail, the trade stays on default leverage.

### Manual Alert-Only Mode

If you want to trade manually after receiving alerts, set:

- `execution.manual_alert_only: true`
- keep `live_trading.enabled: false` unless you intentionally want exchange order routing

With `manual_alert_only=true`, forward/live engines still compute confluence, EVR, risk, and full reasoning, but they emit `alert` events and do not open/close positions automatically.

### Configuration Model

The configuration system merges a base config, module configs, overrides, and environment substitutions into one unified runtime config. In practical terms, that means:

1. `base.yaml` establishes the global baseline
2. module YAMLs under `quant_system/config/` fill in data, features, labels, models, execution, and observability
3. environment values can override placeholders
4. runtime components read the merged contract rather than one-off local files

High-impact config files now also use Pydantic validation at load time:

- `quant_system/config/models/models.yaml`
- `quant_system/config/labels/labels.yaml`

This catches schema mistakes early (invalid horizon bounds, unsupported scaler/calibrator values, malformed quantile bounds, missing core sections) before long tuning/training runs start.

This matters because the repo is trying to avoid per-script drift. Data, training, execution, and telemetry should all read the same resolved configuration model.

## 15. Main Entry Points

### Core CLIs

```bash
python -m quant_system.cli.train_cli
python -m quant_system.cli.backtest_cli
python -m quant_system.cli.forward_cli
python -m quant_system.cli.live_cli
python -m quant_system.cli.terminal_api_cli
```

### Terminal API Console Script

```bash
quant-terminal-api
```

### BTCUSD Manual Launchers

These are the human-friendly top-level scripts that make the pipeline explicit:

- `fetch_BTCUSD_resample.py`
- `build_BTCUSD_features.py`
- `build_BTCUSD_labels.py`
- `train_BTCUSD_liq_flow_model.py`
- `train_BTCUSD_bos_cont_model.py`
- `train_BTCUSD_flow_1h_model.py`
- `train_BTCUSD_momo_model.py`
- `train_BTCUSD_eop_model.py`
- `train_BTCUSD_edp_model.py`
- `train_BTCUSD_meta_model.py`
- `train_BTCUSD_confluence_model.py`
- `train_BTCUSD_hazard_model.py`
- `train_BTCUSD_quantile_model.py`
- `train_BTCUSD_tcn_model.py` (Temporal Convolutional Network benchmark runner)
- `train_BTCUSD_all_tcn_models.py` (sequential Temporal Convolutional Network specialist batch runner)
- `train_BTCUSD_all_tree_models.py` (sequential tree-based specialist batch runner)
- `train_BTCUSD_liq_flow_tcn_model.py`
- `train_BTCUSD_bos_cont_tcn_model.py`
- `train_BTCUSD_flow_1h_tcn_model.py`
- `train_BTCUSD_momo_tcn_model.py`
- `train_BTCUSD_eop_tcn_model.py`
- `train_BTCUSD_edp_tcn_model.py`
- `train_BTCUSD_tcn_stack_model.py`
- `train_BTCUSD_meta_tcn_model.py`
- `train_BTCUSD_confluence_tcn_model.py`
- `configure_BTCUSD_hybrid_routes.py`
- `train_BTCUSD_hybrid_stack_model.py`
- `train_BTCUSD_meta_hybrid_model.py`
- `train_BTCUSD_confluence_hybrid_model.py`
- `tune_BTCUSD_label_horizons.py`

Temporal Convolutional Network (TCN) launchers require PyTorch in the active interpreter:

```bash
pip install torch
```

Batch Temporal Convolutional Network (TCN) run across all specialist targets:

```bash
python train_BTCUSD_all_tcn_models.py
```

Single-target Temporal Convolutional Network (TCN) default run (now defaults to `flow_1h`):

```bash
python train_BTCUSD_tcn_model.py
```

Batch tree-model run across core specialists:

```bash
python train_BTCUSD_all_tree_models.py
```

Optional tree batch extension (adds `meta_model`, `confluence_model`, `hazard`, `quantile`):

```bash
python train_BTCUSD_all_tree_models.py --include-extended
```

Temporal Convolutional Network (TCN) HPO now supports SQLite-backed resume by default under each target artifact folder, so interrupted runs can continue without restarting from trial 1.

During long Temporal Convolutional Network (TCN) runs, live progress is persisted to:

- `artifacts/train/BTCUSD/<target>_tcn/hpo_progress.json` (latest snapshot)
- `artifacts/train/BTCUSD/<target>_tcn/hpo_progress.ndjson` (event history)

Example monitor command:

```bash
tail -f artifacts/train/BTCUSD/flow_1h_tcn/hpo_progress.ndjson
```

### One-Command Live-Style Rooms

If you want simple launch commands (no long CLI flags), use:

```bash
python run_BTCUSD_backtest_live_room.py
python run_BTCUSD_forward_live_room.py
python run_BTCUSD_paper_live_room.py
python run_BTCUSD_stress_matrix.py
```

These wrappers automatically:

- start the React operator terminal (`Next.js`) by default
- fallback to Streamlit automatically if React frontend cannot boot
- start the FastAPI + WebSocket telemetry backend
- run the selected engine mode with BTCUSD defaults
- route telemetry through `http://127.0.0.1:8100` (`/snapshot` + `/ws/terminal`)

UI selection can be controlled with:

- `QUANT_UI_SURFACE=next` (default): React terminal, Streamlit fallback
- `QUANT_UI_SURFACE=streamlit`: Streamlit only
- `QUANT_UI_SURFACE=both`: React + Streamlit together

### Deterministic Stress Matrix (No Monte Carlo)

After running backtest, you can run a deterministic scenario matrix:

```bash
python run_BTCUSD_stress_matrix.py
```

This runner:

- reads backtest ledger artifacts
- applies fixed deterministic shock scenarios (cost, latency, fill quality, adverse tails)
- computes scenario metrics (ending equity, max drawdown, VaR/CVaR, ruin proxy)
- writes reports to `backtest_outputs/stress_matrix/`

Important: this is an offline validation gate. It does not tighten live signal guardrails unless you explicitly enforce it in your deployment promotion policy.

## 16. BTCUSD Sequential Runbook

### Overview

The cleanest manual workflow is:

1. fetch data
2. engineer features
3. build labels
4. train specialists
5. train stackers
6. train hazard and quantile layers

### Ordered Runbook

| Step | Run | What it does | Main outputs |
|---|---|---|---|
| 1 | `python fetch_BTCUSD_resample.py` | Fetches BTCUSD history from Kraken using the Berlin-local window from `2017-01-01 00:00` through yesterday `23:59:59`. Deep history automatically uses trades bootstrap, then rebuilds canonical timeframe bars. | `data/raw_1m/BTCUSD_1m.csv`, `data/tf/BTCUSD_15m.csv`, `data/tf/BTCUSD_1h.csv`, `data/tf/BTCUSD_6h.csv`, `data/tf/BTCUSD_12h.csv` |
| 2 | `python build_BTCUSD_features.py` | Builds the engineered `15m` decision frame from the timeframe bars. | `artifacts/features/BTCUSD/BTCUSD_features.csv` |
| 3 | `python build_BTCUSD_labels.py` | Builds canonical labels from the engineered feature frame. | `artifacts/labels/BTCUSD/BTCUSD_labels.csv` |
| 4 | `python train_BTCUSD_liq_flow_model.py` | Trains the liquidity-flow specialist. | `artifacts/train/BTCUSD/liq_flow/` |
| 5 | `python train_BTCUSD_bos_cont_model.py` | Trains the BOS continuation specialist. | `artifacts/train/BTCUSD/bos_cont/` |
| 6 | `python train_BTCUSD_flow_1h_model.py` | Trains the `1h` flow specialist. | `artifacts/train/BTCUSD/flow_1h/` |
| 7 | `python train_BTCUSD_momo_model.py` | Trains the momentum specialist. | `artifacts/train/BTCUSD/momo/` |
| 8 | `python train_BTCUSD_eop_model.py` | Trains the opportunity specialist. | `artifacts/train/BTCUSD/eop/` |
| 9 | `python train_BTCUSD_edp_model.py` | Trains the downside-danger specialist. | `artifacts/train/BTCUSD/edp/` |
| 10 | `python train_BTCUSD_meta_model.py` | Trains the meta stacker over specialists. | `artifacts/train/BTCUSD/meta_model/` |
| 11 | `python train_BTCUSD_confluence_model.py` | Trains the confluence stacker. | `artifacts/train/BTCUSD/confluence_model/` |
| 12 | `python train_BTCUSD_hazard_model.py` | Trains the hazard timing layer. | `artifacts/train/BTCUSD/hazard/` |
| 13 | `python train_BTCUSD_quantile_model.py` | Trains the quantile forecaster. | `artifacts/train/BTCUSD/quantile/` |

## 16A. BTCUSD Temporal Convolutional Network (TCN)-Only Sequential Runbook

If you are running Temporal Convolutional Network (TCN) specialists only (no tree retraining in the same cycle), use this sequence:

| Step | Run | Output Folder |
|---|---|---|
| 1 | `python train_BTCUSD_tcn_model.py` | `artifacts/train/BTCUSD/flow_1h_tcn/` |
| 2 | `python train_BTCUSD_liq_flow_tcn_model.py` | `artifacts/train/BTCUSD/liq_flow_tcn/` |
| 3 | `python train_BTCUSD_bos_cont_tcn_model.py` | `artifacts/train/BTCUSD/bos_cont_tcn/` |
| 4 | `python train_BTCUSD_momo_tcn_model.py` | `artifacts/train/BTCUSD/momo_tcn/` |
| 5 | `python train_BTCUSD_eop_tcn_model.py` | `artifacts/train/BTCUSD/eop_tcn/` |
| 6 | `python train_BTCUSD_edp_tcn_model.py` | `artifacts/train/BTCUSD/edp_tcn/` |
| 7 | `python run_BTCUSD_stress_matrix.py` | `backtest_outputs/stress_matrix/` |

If you want one command for all Temporal Convolutional Network (TCN) specialists:

```bash
python train_BTCUSD_all_tcn_models.py
```

If you want one command for all core tree specialists:

```bash
python train_BTCUSD_all_tree_models.py
```

During any long run:

```bash
tail -f artifacts/train/BTCUSD/<target>_tcn/hpo_progress.ndjson
```

and inspect latest compact snapshot:

```bash
cat artifacts/train/BTCUSD/<target>_tcn/hpo_progress.json
```

## 16B. BTCUSD Hybrid-Explicit Stack Runbook

Hybrid-explicit means:

- specialists can be mixed per target, e.g. `flow_1h_tcn` with the rest still on tree
- `meta_model_hybrid` and `confluence_model_hybrid` are then trained on that exact mixed specialist route set
- backtest / forward / live use the mixed specialist slot only when `routing_mode: hybrid_explicit` is enabled

This is the correct way to mix families without silently feeding a tree-trained stack with a different specialist-output distribution.

### Hybrid Prerequisites

Before training hybrid stacks, you should already have:

- the tree specialist family trained
- any TCN specialists you want to compete already trained
- a deliberate winner chosen per specialist target

### Step 1: Pin Hybrid Specialist Winners Into A Slot

Use the route configurator to define the exact mixed specialist set. Example:

```bash
python configure_BTCUSD_hybrid_routes.py --slot hybrid_candidate \
  --route liq_flow=liq_flow \
  --route bos_cont=bos_cont \
  --route flow_1h=flow_1h_tcn \
  --route momo=momo \
  --route eop=eop \
  --route edp=edp
```

Notes:

- left side is the requested specialist used by the runtime stack
- right side is the saved model family to use for that specialist
- omit `@version` to use the registry-selected best saved version
- add `@v000X` if you want to pin an exact version

This writes routes into `models/active_models.json` under the chosen slot and also saves a readable copy in:

- `artifacts/train/BTCUSD/hybrid_routes/<slot>.json`

### Step 2: Train The Hybrid Stackers

Default wrappers use slot `hybrid_candidate`:

```bash
python train_BTCUSD_meta_hybrid_model.py
python train_BTCUSD_confluence_hybrid_model.py
```

If you want a different slot or strict/fallback control:

```bash
python train_BTCUSD_hybrid_stack_model.py --target meta_model --slot hybrid_candidate
python train_BTCUSD_hybrid_stack_model.py --target confluence_model --slot hybrid_candidate
```

The trainer is strict by default: every specialist used by the hybrid stack must be explicitly routed in the chosen slot.

Hybrid stack outputs are written to:

- `artifacts/train/BTCUSD/meta_model_hybrid/`
- `artifacts/train/BTCUSD/confluence_model_hybrid/`

Registry aliases created by the trainer:

- `meta_model_hybrid`
- `confluence_model_hybrid`
- `BTCUSD_meta_model_hybrid`
- `BTCUSD_confluence_model_hybrid`

### Step 3: Enable Hybrid Inference Later

When you are ready to backtest or run the hybrid route, set this in `quant_system/config/models/models.yaml`:

```yaml
inference_preference:
  routing_mode: hybrid_explicit
  challenger_mode: tcn
  allow_hybrid_explicit: true
  active_slot: hybrid_candidate
```

Then the predictor will:

- resolve specialists from the chosen active slot
- require `meta_model_hybrid` and `confluence_model_hybrid`
- use the mixed-family route in backtest / forward / live

Until you flip those settings, the default runtime route remains the regular tree family.

### Recommended Interpretation Of The Sequence

This runbook is intentionally explicit. A professional workflow should make the dependency chain visible rather than hide it behind a single opaque command. The order matters because every downstream stage assumes a completed upstream contract:

- training assumes engineered features and labels already exist
- labels assume engineered features already exist
- engineered features assume canonical timeframe bars already exist
- runtime inference assumes versioned model artifacts already exist
- dashboards become meaningfully informative only after runtime and model artifacts exist

If a later stage fails, the right response is usually to inspect the expected outputs of the immediately previous stage rather than rerunning unrelated parts of the stack.

### Strict Sequence Warnings

| Step | Must already be completed | Warning |
|---|---|---|
| `build_BTCUSD_features.py` | fetch must be finished | do not run this before the timeframe CSVs exist |
| `build_BTCUSD_labels.py` | features must be finished | labels are derived from engineered states, not raw bars |
| specialist model launchers | fetch + features + labels must be finished | do not train from raw data only |
| `train_BTCUSD_meta_model.py` | specialist models should already exist | the meta model is not a first-stage model |
| `train_BTCUSD_confluence_model.py` | specialists and ideally `meta_model` should exist | confluence is the scored aggregation layer |
| `train_BTCUSD_hazard_model.py` | fetch + features + labels must be finished | hazard is a risk/exit layer, not a substitute for specialists |
| `train_BTCUSD_quantile_model.py` | fetch + features + labels must be finished | quantile is a distribution layer, not the main execution model |

### Minimum Safe Sequence

1. `python fetch_BTCUSD_resample.py`
2. `python build_BTCUSD_features.py`
3. `python build_BTCUSD_labels.py`
4. train the specialist models you need
5. train `meta_model`
6. train `confluence_model`
7. train `hazard`
8. train `quantile`

## 17. Label Horizon Governance

### Why Governance Exists

The label horizons in `labels.yaml` are currently the production baseline, but they are not treated as sacred or permanently optimal. They are sensible starting values. The repo now includes an empirical challenger path so horizon choices can be tested, scored, and promoted without mutating production behavior on every run.

### Tuning Launcher

```bash
python tune_BTCUSD_label_horizons.py
```

The tuner now requires `xgboost`, `lightgbm`, and `optuna` in the active Python environment.

### What It Does

The tuner:

1. loads the engineered BTCUSD feature frame
2. rebuilds candidate label sets for multiple horizon choices
3. includes the current default profile as the baseline
4. scores challengers with purged + embargoed time-series CV using production learners (`lightgbm`, `xgboost`) plus logistic baseline
5. applies leak-safe preprocessing (median impute, quantile clipping, conditional scaling) and optional Bayesian HPO (Optuna) per model
6. computes a consensus objective and confidence bounds per candidate
7. compares challenger objective score vs baseline objective score
8. promotes only if improvement threshold, model-consensus gate, and CI gate all pass

### Checkpointing And Resume Semantics

The tuner supports resume and writes progress checkpoints so interrupted runs can continue without recomputing every candidate from zero. Current progress is tracked in:

- `artifacts/label_tuning/BTCUSD/progress.json`
- `artifacts/label_tuning/BTCUSD/progress.ndjson`

Per-task candidate summaries are stored as CSV files in the same directory (`*_tuning.csv`). When rerun with resume enabled, previously scored candidates are loaded from checkpoint and skipped.

### Promotion Behavior

The active promoted label profile is stored under:

- `artifacts/label_profiles/active_label_profile.json`

Historical promoted snapshots are stored under:

- `artifacts/label_profiles/history/`

When [LabelBuilder](/Users/mac/Documents/quant_smc/quant_system/label_generation/label_builder.py) runs, it now automatically checks for an active promoted profile and applies it on top of the defaults. That means:

- defaults remain the production baseline
- tuning can recommend challengers
- challengers can be promoted automatically
- future label builds can then use the promoted profile without hand-editing `labels.yaml`

### Important Constraint

This is governed promotion, not uncontrolled mutation. The intent is:

- tune periodically
- validate challengers
- promote only if better
- keep the promoted profile stable until the next research cycle

## 18. Expected Artifacts By Stage

| Stage | Expected artifacts |
|---|---|
| Fetch | `data/raw_1m/BTCUSD_1m.csv`, `data/tf/BTCUSD_15m.csv`, `data/tf/BTCUSD_1h.csv`, `data/tf/BTCUSD_6h.csv`, `data/tf/BTCUSD_12h.csv` |
| Features | `artifacts/features/BTCUSD/BTCUSD_features.csv` |
| Labels | `artifacts/labels/BTCUSD/BTCUSD_labels.csv` |
| Training | per-model `artifacts/train/BTCUSD/<model_name>/` manifests and registry artifacts |
| TCN training | per-target `artifacts/train/BTCUSD/<target>_tcn/` with `optuna_<target>.db`, `hpo_progress.json`, `hpo_progress.ndjson`, manifests, and model state |
| Label tuning | `artifacts/label_tuning/BTCUSD/` reports |
| Promoted labels | `artifacts/label_profiles/active_label_profile.json` |
| Deterministic stress | `backtest_outputs/stress_matrix/` reports |

If those expected files are missing, the correct move is usually to go back one stage rather than trying to force the next stage to run.

## 19. Project Layout

```text
quant_system/
  backtest/          historical execution, replay, reports
  cli/               train/backtest/forward/live/terminal entrypoints
  config/            YAML config system
  dashboard/         Streamlit research and audit dashboards
  data/              ingestion, prep, storage
  execution/         gating, confluence, EVR, risk, adapters
  features/          SMC, liquidity, EMA, volatility, regime features
  forward_test/      paper-forward runtime and dashboard bridge
  label_generation/  canonical labels, tuning, and profile governance
  live/              live orchestration and execution loop
  live_data/         streaming market data and replay
  ml/                training, prediction, registry, regime tooling
  model_ensemble/    optional ensemble governance
  model_switcher/    optional model selection layer
  models/            legacy wrappers + NARX module
  replay_export/     standalone replay export
  strategy/          pyramiding / trade management policy
  telemetry/         FastAPI + WebSocket terminal backend
frontend/            Next.js + Tailwind operator terminal
```

## 20. Release Status And Current Caveats

### Release Status

Current tagged baseline: `v0.1.0`

- release notes: [`CHANGELOG.md`](./CHANGELOG.md)
- repair summary: [`quant_smc_pipeline_report.md`](./quant_smc_pipeline_report.md)

### Current Caveats

- This repo still contains overlapping generations of code in some areas, even though the main runtime paths have been repaired.
- The fetch layer is correct for deep history, but full-history Kraken trades bootstrap can take a long time.
- TCN HPO can run for many hours depending on trial budget, fold count, and sampled epoch settings; use adaptive stop + progress files to control runtime.
- Artifact fallback still exists in some surfaces, though the preferred live transport is now `FastAPI + WebSocket`.
- Default label horizons are baseline values, not mathematically guaranteed optima.
- `NARX` remains intentionally outside the main runtime path for now.
- `flow_1h` tree and TCN trainers now emit feature-dominance audit diagnostics (metrics + warnings) to flag over-dominant predictors early; this is diagnostic and does not auto-drop features.

### Summary

This project is now best understood as a structured quant trading platform with:

- deterministic multi-timeframe feature parity
- specialist and stacked supervised models
- explicit execution and risk logic
- live operator telemetry
- governed tuning paths for key research assumptions

It is not just a signal script. It is an evolving decision system whose design intent is to make research, execution, and operator judgment coherent with each other rather than separate worlds.

## 21. Troubleshooting And Operator Notes

### The Fetch Script Shows A 2016 UTC Start When I Asked For 2017 Berlin

That is expected. `2017-01-01 00:00 Europe/Berlin` converts to `2016-12-31 23:00 UTC`. The fetch script logs both representations because Kraken requests are executed in UTC while the requested business boundary is Berlin-local.

### `pandas` Or Another Dependency Is Missing When Running A Top-Level Script

The BTCUSD launchers include a Python bootstrap helper and will try to re-exec into a compatible interpreter when possible. The preferred environment is still a dedicated project environment with `requirements.txt` installed.

The bootstrap now validates by import (not only module presence) and applies a pandas compatibility shim before importing LightGBM. This prevents false "missing dependency" failures in environments where LightGBM import can fail due to pandas API differences.

### Why Is `data/raw_1m/` Filling Before `data/tf/`?

That is normal. For a long Kraken history job, the raw download and normalization step finishes first. The `15m`, `1h`, `6h`, and `12h` outputs are rebuilt after the raw stage completes.

### Why Are Labels Using Default Horizons Unless Tuning Has Been Run?

Because default label horizons are the stable production baseline. Tuning is a governed challenger path. Once a challenger profile is promoted, future label builds can automatically pick it up from the active label profile file.

### Why Is `NARX` Not In The Main Runtime Yet?

Because the current repo policy keeps `NARX` as a later soft-enhancement layer for ranking, runner handling, and moonshot logic only. It is intentionally not allowed to become a hard guardrail or a replacement for the main multi-timeframe supervised stack.
