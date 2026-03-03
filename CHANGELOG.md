# Changelog

## v0.1.0 - Telemetry Terminal Baseline

Date: 2026-03-04

This release establishes the current repo baseline after the full repair pass across runtime, modeling, dashboards, and packaging.

### Added

- `Next.js + Tailwind + React` operator terminal in [`frontend/`](./frontend)
- `FastAPI + WebSocket` telemetry backend in [`quant_system/telemetry/`](./quant_system/telemetry)
- shared terminal API CLI via `python -m quant_system.cli.terminal_api_cli`
- signal-level reasoning tree in the Next terminal
- backend-fed Streamlit dashboard context through `/dashboard/context`
- profit ladder, compound-cooling, and capital allocator modules

### Changed

- repaired end-to-end loops across `backtest`, `cli`, `config`, `dashboard`, `data`, `execution`, `features`, `forward_test`, `label_generation`, `live`, `live_data`, `ml`, `model_ensemble`, `model_switcher`, `models`, `replay_export`, `strategy`, and `utils`
- standardized the multi-timeframe contract:
  - `12h = regime`
  - `6h = structure`
  - `1h = flow`
  - `15m = execution`
- aligned dashboards, runtime adapters, and terminal UI around the same telemetry/state plane
- refreshed repo docs and install flow for the new frontend/backend stack

### Modeling Baseline

- `12h`: `HMM + HDBSCAN`
- `1h` and `15m`: schema-driven `XGBoost / LightGBM`
- `6h`: deterministic context layer
- `NARX`: retained only as a later soft module for ranking / runner / moonshot logic

### Capital / Risk Baseline

- base ticket: `20,000 USD`
- compounding: enabled
- cooling / vault reset: enabled
- `MPC`: optional, disabled by default

### Notes

- artifact fallback remains in place where needed, but the preferred live transport is now `FastAPI + WebSocket`
- this release is the first tagged baseline after the forensic repair pass
