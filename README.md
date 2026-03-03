# Quant SMC

Research-to-execution crypto trading stack with deterministic feature parity across backtest, forward test, live runtime, Streamlit research dashboards, and a new `Next.js + FastAPI + WebSocket` operator terminal.

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
