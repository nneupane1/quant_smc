# Quant SMC

Research-to-execution crypto trading stack with deterministic feature parity across backtest, forward test, live runtime, Streamlit research dashboards, and a new `Next.js + FastAPI + WebSocket` operator terminal.

## Release Status

Current tagged baseline: `v0.1.0`

- release notes: [`CHANGELOG.md`](./CHANGELOG.md)
- repair summary: [`quant_smc_pipeline_report.md`](./quant_smc_pipeline_report.md)
- main branch includes the repaired telemetry-backed terminal stack

## What This Repo Now Contains

- `12h` regime modeling with `HMM + HDBSCAN`
- `6h` structure context
- `1h` flow ML
- `15m` execution ML with specialist models, meta/confluence, hazard, and quantile forecasting
- `20k` base-ticket compounding with cooling / vault-reset logic
- profit ladder plus hazard-based trailing for core and runner legs
- repaired backtest, forward, live, replay, dashboard, and CLI paths
- real terminal frontend in [`frontend/`](./frontend) backed by `FastAPI + WebSockets`

## Modeling Contract

This repo is wired around a fixed timeframe contract:

- `12h`: regime assessment
- `6h`: structure / context
- `1h`: flow quality
- `15m`: execution decision layer

Main model usage:

- `12h`: `HMM + HDBSCAN`
- `1h` and `15m`: schema-driven tabular ML, primarily `XGBoost / LightGBM`
- `6h`: deterministic structure context now, with room for later GBM context-quality modeling
- `NARX`: not part of the main runtime path; reserved for later soft ranking / runner / moonshot support only

## Runtime Architecture

### Trading loop

- data ingestion -> resampling -> feature graph
- label generation -> model training / registry
- gate -> confluence -> EVR -> tier -> sizing
- entry -> profit ladder -> hazard trailing -> exit
- compounding -> danger detection -> cooling / vault reset

### UI and telemetry

- `Streamlit` remains the research / audit shell in [`quant_system/dashboard/`](./quant_system/dashboard)
- `Next.js` terminal lives in [`frontend/`](./frontend)
- `FastAPI + WebSocket` backend lives in [`quant_system/telemetry/`](./quant_system/telemetry)
- console/runtime and UI now share the same telemetry plane through [`forward_dashboard_adapter.py`](./quant_system/forward_test/forward_dashboard_adapter.py)

## Project Layout

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
  label_generation/  canonical label builders
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

## Install

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

## Environment

Create `.env` from [`.env.example`](./.env.example).

Important vars:

- `KRAKEN_API_KEY`
- `KRAKEN_API_SECRET`
- `QS_ENV=DEV`

Frontend env template is in [`frontend/.env.example`](./frontend/.env.example):

- `QUANT_TERMINAL_API_URL=http://127.0.0.1:8100/snapshot`
- `NEXT_PUBLIC_TERMINAL_WS_URL=ws://127.0.0.1:8100/ws/terminal`

## Main Commands

### Train

```bash
python -m quant_system.cli.train_cli
```

### Backtest

```bash
python -m quant_system.cli.backtest_cli
```

### Forward test

```bash
python -m quant_system.cli.forward_cli
```

### Live

```bash
python -m quant_system.cli.live_cli
```

### Terminal API

```bash
python -m quant_system.cli.terminal_api_cli
```

or via the console script:

```bash
quant-terminal-api
```

## BTCUSD Runbook

The cleanest manual workflow for this repo is to run the BTCUSD launchers in order. These scripts keep the pipeline explicit:

- fetch and rebuild raw/timeframe data first
- build engineered features next
- build labels from those engineered features
- train each model family with its own launcher

| Step | Run | What it does | Main outputs |
|---|---|---|---|
| 1 | `python fetch_1m_BTCUSD_from_Kraken.py` | Fetches BTCUSD history from Kraken, using the Berlin-local window `2017-01-01 00:00` through yesterday `23:59:59`. For deep history it automatically switches from shallow OHLC paging to the trades-backed bootstrap path, then rebuilds canonical `1m`, `15m`, `1h`, `6h`, and `12h` files. | `data/raw_1m/BTCUSD_1m.csv`, `data/tf/BTCUSD_15m.csv`, `data/tf/BTCUSD_1h.csv`, `data/tf/BTCUSD_6h.csv`, `data/tf/BTCUSD_12h.csv` |
| 2 | `python build_BTCUSD_features.py` | Runs the canonical feature builder over the timeframe CSVs. This is the feature-engineering stage that matters for training: SMC structure, regime context, EMA state, volatility, liquidity, session state, and joined higher-timeframe context are all materialized here. | `artifacts/features/BTCUSD/BTCUSD_features.csv` |
| 3 | `python build_BTCUSD_labels.py` | Applies the canonical label builder on the engineered feature frame. This creates the supervised targets used by the specialist models, hazard models, and ranking stack. | `artifacts/labels/BTCUSD/BTCUSD_labels.csv` |
| 4 | `python train_BTCUSD_liq_flow_model.py` | Trains the liquidity-flow specialist on the engineered BTCUSD feature set. Use this after features and labels are already built. | `artifacts/train/BTCUSD/liq_flow/`, model registry entries for `liq_flow` |
| 5 | `python train_BTCUSD_bos_cont_model.py` | Trains the BOS continuation specialist. This model focuses on whether structure continuation setups are likely to follow through. | `artifacts/train/BTCUSD/bos_cont/`, model registry entries for `bos_cont` |
| 6 | `python train_BTCUSD_flow_1h_model.py` | Trains the dedicated `1h` flow model, using the scoped higher-timeframe feature contract for flow quality and follow-through. | `artifacts/train/BTCUSD/flow_1h/`, model registry entries for `flow_1h` |
| 7 | `python train_BTCUSD_momo_model.py` | Trains the short-horizon momentum specialist used in the `15m` execution stack. | `artifacts/train/BTCUSD/momo/`, model registry entries for `momo` |
| 8 | `python train_BTCUSD_eop_model.py` | Trains the upside opportunity specialist (`EOP`), used to evaluate how favorable the forward opportunity distribution looks. | `artifacts/train/BTCUSD/eop/`, model registry entries for `eop` |
| 9 | `python train_BTCUSD_edp_model.py` | Trains the downside danger specialist (`EDP`), used by confluence and risk logic to model adverse conditions. | `artifacts/train/BTCUSD/edp/`, model registry entries for `edp` |
| 10 | `python train_BTCUSD_meta_model.py` | Trains the meta stacker on top of the specialist outputs. This is the first aggregation layer above the single-purpose specialist models. | `artifacts/train/BTCUSD/meta_model/`, model registry entries for `meta_model` |
| 11 | `python train_BTCUSD_confluence_model.py` | Trains the confluence model that turns specialist outputs into the final scored trade-quality layer used by downstream ranking and gating. | `artifacts/train/BTCUSD/confluence_model/`, model registry entries for `confluence_model` |
| 12 | `python train_BTCUSD_hazard_model.py` | Trains the discrete-time hazard stack used by trailing, danger detection, and exit timing logic. | `artifacts/train/BTCUSD/hazard/`, model registry entries for `hazard` |
| 13 | `python train_BTCUSD_quantile_model.py` | Trains the quantile forecaster used for return-distribution and tail-shape estimates in risk and runner management. | `artifacts/train/BTCUSD/quantile/`, model registry entries for `quantile` |

### Why This Order Matters

- The training launchers are designed around engineered features, not raw exchange data.
- `build_BTCUSD_features.py` is the step that turns raw Kraken data into the model-ready state space the rest of the repo expects.
- `build_BTCUSD_labels.py` should be run after features, because labels are computed from those engineered states.
- The specialist models should be trained before `meta_model` and `confluence_model`, because the stackers sit on top of specialist outputs.
- `hazard` and `quantile` are downstream risk/exit layers and should be trained after the feature and label substrate is stable.

## How To Run In Sequence

Use this section as the strict operator checklist. If a step says a previous step is required, do not skip that requirement.

| Step | Run now | Must already be completed | Warning before you run it |
|---|---|---|---|
| 1 | `python fetch_1m_BTCUSD_from_Kraken.py` | Nothing. This is the starting point. | This is the only correct first step for a fresh BTCUSD pipeline run. It fetches deep history from `2017-01-01 00:00 Europe/Berlin` through yesterday `23:59:59 Europe/Berlin`, then rebuilds `1m`, `15m`, `1h`, `6h`, and `12h`. |
| 2 | `python build_BTCUSD_features.py` | Step 1 must be finished. `data/tf/BTCUSD_15m.csv`, `BTCUSD_1h.csv`, `BTCUSD_6h.csv`, and `BTCUSD_12h.csv` must exist. | Do not run this before the timeframe files exist. This is the feature-engineering stage that converts raw/resampled market data into the model-ready state space used everywhere else. |
| 3 | `python build_BTCUSD_labels.py` | Step 2 must be finished. `artifacts/features/BTCUSD/BTCUSD_features.csv` must exist. | Do not run this before features exist. Labels are computed from engineered states, not directly from exchange candles. |
| 4 | `python train_BTCUSD_liq_flow_model.py` | Steps 1, 2, and 3 must be finished. | Do not start model training from raw data alone. The model expects the canonical engineered feature frame and the canonical label frame. |
| 5 | `python train_BTCUSD_bos_cont_model.py` | Steps 1, 2, and 3 must be finished. | Same warning as above. This specialist should only be trained after the shared BTCUSD features and labels are built. |
| 6 | `python train_BTCUSD_flow_1h_model.py` | Steps 1, 2, and 3 must be finished. | This model relies on the correct `1h/6h/12h` feature scope. Do not run it before the canonical feature build has succeeded. |
| 7 | `python train_BTCUSD_momo_model.py` | Steps 1, 2, and 3 must be finished. | Same requirement: engineered features first, labels second, training third. |
| 8 | `python train_BTCUSD_eop_model.py` | Steps 1, 2, and 3 must be finished. | Same requirement. |
| 9 | `python train_BTCUSD_edp_model.py` | Steps 1, 2, and 3 must be finished. | Same requirement. |
| 10 | `python train_BTCUSD_meta_model.py` | Steps 1 through 9 should already be finished. | The meta model sits above specialist outputs. It is not the first thing to train. Train the specialist family first so the stacker has the intended upstream substrate. |
| 11 | `python train_BTCUSD_confluence_model.py` | Steps 1 through 10 should already be finished. | The confluence model is the final scored aggregation layer. Do not treat it as a standalone first-stage model. |
| 12 | `python train_BTCUSD_hazard_model.py` | Steps 1, 2, and 3 must be finished. Specialist training is strongly recommended before this stage. | Hazard controls exits and danger logic. It should be trained after the base research substrate is stable. |
| 13 | `python train_BTCUSD_quantile_model.py` | Steps 1, 2, and 3 must be finished. Specialist training is strongly recommended before this stage. | Quantile forecasting is a risk/runner layer, not a substitute for the main execution stack. |

### Minimum Safe Sequence

If you want the shortest safe path, use this exact order:

1. `python fetch_1m_BTCUSD_from_Kraken.py`
2. `python build_BTCUSD_features.py`
3. `python build_BTCUSD_labels.py`
4. train the specialist models you need
5. train `meta_model`
6. train `confluence_model`
7. train `hazard`
8. train `quantile`

### Required Files Before Training

Before training any BTCUSD model, these files should exist:

- `data/tf/BTCUSD_15m.csv`
- `data/tf/BTCUSD_1h.csv`
- `data/tf/BTCUSD_6h.csv`
- `data/tf/BTCUSD_12h.csv`
- `artifacts/features/BTCUSD/BTCUSD_features.csv`
- `artifacts/labels/BTCUSD/BTCUSD_labels.csv`

If those are missing, go back and run the earlier step instead of trying to force training forward.

## Dashboards

### Streamlit research shell

```bash
streamlit run quant_system/dashboard/app.py
```

When the telemetry backend is running, the Streamlit dashboards now prefer backend-fed context over local artifact reads.

### Next.js operator terminal

Start the backend:

```bash
python -m quant_system.cli.forward_cli --terminal-server
```

Then start the frontend:

```bash
cd frontend
npm run dev
```

Open `http://127.0.0.1:3100`.

## Terminal Domains

The terminal is organized into six operator domains:

1. `Mission Control`
2. `Insights`
3. `Regime Briefings`
4. `Signal Intelligence`
5. `Risk Radar`
6. `Research & Audit`

The Next terminal now includes:

- hover-reactive navigation
- Bloomberg/TradingView-inspired styling
- WebSocket-fed live state
- full alert reasoning tree for selected signals

## Reasoning Payloads

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

These are visible in:

- Streamlit `Insights`
- Streamlit `Forward Test`
- Next terminal `Insights -> Alert Reasoning Tree`
- Next terminal `Signal Intelligence -> Selected Signal Vector`

## Capital and Risk Defaults

Current default posture is aligned with the repaired execution config:

- base ticket: `20,000 USD`
- compounding: enabled
- cooling / vault reset: enabled through compound-cooling policy
- `MPC`: kept available but off by default
- `NARX`: not wired as a guardrail

## Notes

- This repo contains a lot of repaired legacy surfaces. The authoritative runtime paths are the repaired `backtest`, `forward_test`, `live`, `execution`, `features`, `label_generation`, and `ml` layers.
- The live event plane is now `FastAPI + WebSocket` first, with artifact fallback still preserved where necessary.
- Generated details are summarized in [`quant_smc_pipeline_report.md`](./quant_smc_pipeline_report.md).

## Short Release Notes

`v0.1.0` establishes the first repaired baseline for this repo:

- end-to-end runtime loops repaired across research, forward, and live paths
- backend telemetry plane added for shared console/UI state
- operator terminal added in `frontend/`
- signal-level reasoning tree exposed in the terminal
- docs, install flow, and ignore rules aligned with the new architecture
