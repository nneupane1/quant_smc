# Quant SMC Pipeline Report

This report summarizes the current implementation state of the Quant SMC platform after the major repair and alignment passes. It is not a marketing description. It is an implementation-level account of how the current repository is intended to operate and what assumptions now define the authoritative pipeline.

## Report Purpose

The repo previously contained overlapping generations of runtime code, stale entrypoints, and partially disconnected subsystems. The purpose of this report is to document what now counts as authoritative, which contracts have been standardized, and where important caveats still remain.

## System Overview

Quant SMC now operates as a multi-stage decision pipeline:

1. acquire and normalize market data
2. rebuild canonical timeframe bars
3. engineer a `15m` decision frame with higher-timeframe context attached
4. generate supervised labels over that feature frame
5. train specialist, stacker, hazard, and quantile models
6. run execution logic over new decision rows
7. publish the resulting state to telemetry and operator interfaces

The central invariant of the repaired system is that the `15m` decision frame is the primary execution surface across research, backtest, forward test, and live runtime.

## Canonical Timeframe Contract

| Timeframe | Role | Current Interpretation |
|---|---|---|
| `1m` | ingestion plumbing | raw historical/live bar source |
| `15m` | execution spine | main feature, label, and inference frame |
| `1h` | flow context | higher-timeframe impulse/follow-through layer |
| `6h` | structural context | structural bias, compression, zone quality |
| `12h` | regime | HMM/HDBSCAN regime characterization |

This separation is now treated as a structural design rule rather than an informal convention.

## Runtime Shape

### Config Layer

Configuration is merged through `quant_system/config/config_loader.py` and exposed through the config manager. The repaired config system now behaves as a unified control plane rather than a set of disconnected YAML fragments.

### Main Runtime Entry Surfaces

The documented module entrypoints now resolve and represent the current runtime surfaces:

- `python -m quant_system.cli.train_cli`
- `python -m quant_system.cli.backtest_cli`
- `python -m quant_system.cli.forward_cli`
- `python -m quant_system.cli.live_cli`

In addition to these, the repo now includes top-level BTCUSD launchers for explicit staged operation.

## Data Flow

### Historical And Live Acquisition

The data layer now supports both historical Kraken acquisition and live streaming aggregation. Historical data is normalized into canonical bar files; live data is aggregated from closed `1m` bars into higher-timeframe bars.

### Canonical Data Stages

1. acquire raw `1m` or raw trades
2. normalize to canonical `1m`
3. rebuild `15m`, `1h`, `6h`, `12h`
4. persist outputs under the canonical storage layout

### Current BTCUSD Historical Path

The top-level BTCUSD fetch launcher now:

- defaults to `2017-01-01 00:00 Europe/Berlin` through yesterday `23:59:59 Europe/Berlin`
- uses deep-history trades bootstrap for wide windows
- supports checkpointing and resume
- rebuilds timeframe bars automatically

## Feature Flow

Feature engineering is now a first-class pipeline stage rather than an implicit side effect of training.

### Canonical Feature Families

- SMC structure:
  - swings
  - BOS / CHOCH
  - FVG
  - sweeps
  - zones
- higher-timeframe structure context:
  - structural bias
  - premium / discount
  - compression
  - zone score
- EMA context
- volatility state
- liquidity state
- session state
- `12h` regime features

### Join Strategy

The repaired system projects higher-timeframe context onto the `15m` execution spine by timestamp alignment. That means the `15m` row used for training or inference already contains the relevant `1h`, `6h`, and `12h` context rather than relying on separate runtime-only lookups.

## Label Flow

Labels are generated from the engineered `15m` feature spine using forward windows and `R`-style behavioral definitions.

### Canonical Targets

- `label_liq_flow`
- `label_bos_cont`
- `label_momo`
- `label_flow_1h`
- `label_eop`
- `label_edp`
- `hazard_event`
- `hazard_time`

### Current Label Philosophy

The label layer is intentionally aligned to trading questions rather than generic classification tasks. Each label exists because the runtime needs an answer to a specific question such as continuation, opportunity, danger, or time-to-failure.

### Governance Update

The repo now includes governed label-horizon tuning. Stable defaults remain the production baseline, but challenger horizon profiles can be evaluated and promoted when they beat the baseline by a configured threshold. Promoted profiles are persisted and picked up automatically by the label builder.

## Model Flow

### Specialist Layer

The current supervised stack trains specialists for:

- liquidity flow
- BOS continuation
- `1h` flow
- momentum
- opportunity
- downside danger

### Downstream Layer

On top of the specialists, the system trains:

- `meta_model`
- `confluence_model`
- `hazard`
- `quantile`

### Model Intent

The repaired model stack is now treated as question-specific:

- specialists answer local behavioral questions
- stackers integrate specialist outputs
- hazard models failure timing
- quantiles shape distribution-aware risk logic

### Regime Modeling

The `12h` regime layer uses HMM and HDBSCAN features. It is part of the feature/inference contract, but it is not treated as a generic execution trigger on its own.

### NARX Position

`NARX` remains present in the codebase but intentionally outside the main runtime contract. The current repo policy keeps it reserved for future soft use in ranking, runner extension, and moonshot logic only.

## Execution Flow

The repaired execution path now applies a layered decision contract:

1. higher-timeframe gates
2. confluence scoring
3. EVR reasoning
4. tiering
5. position sizing
6. entry
7. profit ladder
8. hazard trailing
9. cooling / reset logic

This is important because the runtime is no longer treated as “model output in, order out.” The execution layer is a structured policy surface.

## Capital Policy

The default capital policy is now ticket-based instead of MPC-led.

| Setting | Current Baseline |
|---|---|
| starting equity | `20,000 USD` |
| base ticket | `20,000 USD` |
| compound ticket | `true` |
| cooling / reset | enabled |
| MPC | available but `false` by default |

The practical interpretation is that the engines deploy a `20k` ticket that compounds with deployable equity during healthy periods, while preserving a reset-and-cooldown logic when danger conditions intensify.

## Runtime Modes

### Backtest

The `backtest` package now has a clear authoritative core runtime. Replay and reporting modules have been aligned to the repaired trade and artifact contracts.

### Forward Test

The `forward_test` package now mirrors live-style decisioning while preserving deterministic reasoning payloads and dashboard compatibility.

### Live

The `live` package now mirrors the repaired forward logic and publishes structured telemetry instead of behaving as a disconnected runtime.

## Telemetry And Operator Surfaces

The platform now includes a shared telemetry plane:

- FastAPI control / snapshot backend
- WebSocket push transport
- Streamlit research shell
- Next.js terminal for operator-facing live views

The key improvement is that console/runtime state and operator-facing state are now intended to flow from a shared telemetry path rather than separate, drifting surfaces.

## What Is Now Authoritative

The following should be treated as authoritative for current operation:

- canonical Kraken BTCUSD fetch launcher
- canonical feature builder
- canonical label builder with profile governance
- canonical training path in `quant_system/ml`
- canonical execution logic in backtest / forward / live
- telemetry-backed operator views

## Remaining Caveats

The repo is substantially more coherent than before, but some caveats remain:

- overlapping generations of helper code still exist outside the main authoritative paths
- full-history Kraken trades bootstrap is correct but time-consuming
- artifact fallback remains present in some places for resilience
- default label horizons are still baselines, not mathematically final truths
- `NARX` is intentionally deferred from the main execution path

## Bottom Line

The repository now behaves like a platform with a declared architecture rather than a collection of partially related scripts. The main runtime contracts are aligned, the decision spine is explicit, the operator surfaces are much more coherent, and the major loops now close from data through telemetry. The remaining work is therefore incremental platform maturation, not basic architectural rescue.
