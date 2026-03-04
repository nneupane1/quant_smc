# Changelog

This changelog tracks meaningful project-level milestones rather than every minor local edit. The intent is to document the state of the platform as it evolves from a repaired baseline into a governed research-to-execution system.

## v0.1.0 - Telemetry Terminal Baseline

Date: 2026-03-04

### Release Summary

`v0.1.0` establishes the first coherent baseline of the repaired Quant SMC platform. This release consolidates the repo into a working research-to-execution stack with deterministic feature parity, repaired runtime paths, governed training surfaces, and a modern operator-facing telemetry plane.

This is the release in which the repository becomes understandable as a system rather than a loose collection of scripts. Historical ingestion, feature engineering, label generation, training, backtesting, forward simulation, live orchestration, research dashboards, and the live operator terminal now fit into one declared architecture.

### Highlights

| Area | Outcome |
|---|---|
| Runtime parity | backtest, forward, and live now follow the same repaired execution contract |
| Telemetry | FastAPI + WebSocket backend introduced for shared runtime state delivery |
| Operator UI | Next.js + React + Tailwind terminal added for live monitoring and reasoning inspection |
| Research shell | Streamlit dashboards repaired and aligned to the shared backend contract |
| Data pipeline | Kraken-first historical ingestion and timeframe rebuild paths repaired |
| Modeling | specialist, meta, confluence, hazard, and quantile flows aligned to canonical features/labels |
| Capital policy | baseline moved to ticket-based compounding/cooling with optional MPC kept off by default |
| Documentation | repository docs expanded to describe architecture, contracts, and operating sequence |

### Added

- `Next.js + React + Tailwind` operator terminal in [`frontend/`](./frontend)
- `FastAPI + WebSocket` telemetry backend in [`quant_system/telemetry/`](./quant_system/telemetry)
- terminal reasoning-tree support for live and forward signals
- backend-served dashboard context endpoint for Streamlit research views
- profit ladder and compound-cooling policy for runtime trade management
- top-level BTCUSD launcher scripts for fetch, features, labels, and per-model training
- governed label-profile tuning and promotion flow

### Changed

- repaired end-to-end loops across:
  - `backtest`
  - `cli`
  - `config`
  - `dashboard`
  - `data`
  - `execution`
  - `features`
  - `forward_test`
  - `label_generation`
  - `live`
  - `live_data`
  - `ml`
  - `model_ensemble`
  - `model_switcher`
  - `models`
  - `replay_export`
  - `strategy`
  - `utils`
- standardized the multi-timeframe operating contract:
  - `12h = regime`
  - `6h = structure`
  - `1h = flow`
  - `15m = execution`
- aligned dashboards, runtime adapters, and terminal UI around the same shared telemetry/state plane
- expanded repo documentation and runbooks to reflect the repaired platform architecture

### Modeling Baseline

| Layer | Baseline |
|---|---|
| `12h` regime | `HMM + HDBSCAN` |
| `6h` structure | deterministic structural context layer |
| `1h` flow | supervised tabular specialist |
| `15m` execution | supervised specialists + stackers |
| auxiliary forecast | quantile forecaster active |
| `NARX` | retained only as a future soft layer for ranking / runner / moonshot logic |

### Capital And Risk Baseline

| Setting | Baseline |
|---|---|
| starting equity | `20,000 USD` |
| base ticket | `20,000 USD` |
| compounding | enabled |
| cooling / vault reset | enabled |
| MPC | available, disabled by default |

### Operational Notes

- artifact fallback remains in place where needed, but the preferred live transport is now the telemetry backend
- label horizons still begin from stable defaults, but governed challenger tuning now exists
- the repository still contains some overlapping generations of code outside the main runtime path, even though the authoritative execution and training paths have been repaired

### Upgrade Impact

For users of the current baseline, the most important practical changes are:

1. use the explicit BTCUSD launcher flow rather than older ad hoc scripts
2. treat the Next terminal and telemetry backend as the preferred live operator path
3. treat label tuning as governed promotion rather than manual config mutation
4. treat the repaired feature/label/model stack as the canonical training path
