# QuantFund AI - Institutional-Grade Quant Trading Platform
Riding the institutional liquidity wave

QuantFund AI (founded by Nischal) is a full-stack quantitative research and trading platform that unifies data engineering, feature/label generation, ML model lifecycle, risk-aware execution, and rich monitoring dashboards. This repository houses the research-to-production pipeline for systematic digital asset trading.

## Why It Matters
- **Research to production, seamlessly:** Same code paths for backtests, forward tests, and live trading to minimize sim/live drift.
- **Feature-rich microstructure intelligence:** Smart Money Concepts, liquidity signals, volatility regimes, confluence metrics, and hazard detection.
- **Risk-first execution:** Position sizing, exposure controls, EVR/MPCRisk, and execution adapters designed for real venues (e.g., Kraken).
- **Operational visibility:** Replayable timelines, dashboards for live/forward/backtest, equity and orderflow panels, and alerting.

## Architecture Overview
- **Data Layer (`data/`):** Ingestion (Kraken client + retry), resampling, session handling, builders, and persistence.
- **Features (`features/`):** Microstructure features (swings, BOS/CHOCH, FVG, sweeps, zones), liquidity/volatility/regime features, preprocessing, and feature store.
- **Labels (`label_generation/`):** Event-driven labels (BOS continuation, liquidity flow, micro-momentum, hazard, EOP/EDP).
- **ML (`ml/`):** Feature/label loaders, trainers, predictors, optimizers, empirical calibration, registry, and versioning.
- **Execution (`execution/`):** Confluence, EVR, tiering, hazard trailing, MPC risk, position sizing, exposure tracking, and order adapters.
- **Backtest/Forward/Live (`backtest/`, `forward_test/`, `live/`):** Replay controllers, simulators, trade logs, metrics, and live orchestrators.
- **Dashboard (`dashboard/`):** Streamlit/JS hybrid UI, TradingView widgets, replay tools, equity/portfolio visualizations, smart alerts, and overlays.
- **CLI (`cli/`):** Entry points for training, backtesting, forward testing, and live trading orchestration.

## Project Layout (high level)
```
quant_system/
  cli/                 # CLI entrypoints
  config/              # YAML configs + secrets template
  data/                # Ingestion, retry, resampling, sessions, builders
  features/            # Feature engineering + feature store
  label_generation/    # Labeling logic
  ml/                  # Model lifecycle (train/predict/optimize)
  execution/           # Risk, sizing, adapters
  backtest/            # Sim/replay/metrics/reporting
  forward_test/        # Paper trading / forward mode
  live/                # Live orchestrator + venue client
  dashboard/           # UI components (Python + JS/CSS)
  utils/               # Logging, time, rolling helpers, decorators
  train_orchestrator.py
  setup.py
```

## Quickstart
1) **Environment**
```bash
conda create -n quantfund_env python=3.11 -y
conda activate quantfund_env
pip install -r requirements.txt
```

2) **Secrets**
- Copy `.env.example` to `.env` and set credentials:
  - `KRAKEN_API_KEY`
  - `KRAKEN_PRIVATE_KEY`

3) **Configs**
- Review `quant_system/config/base.yaml` and related YAMLs for data paths, instruments, timeframes, regimes, risk, and storage.
- Venue/API routing and execution parameters live in `config/execution.yaml` and `config/routing.yaml`.

4) **Run Pipelines (examples)**
```bash
# Train models with current configs
python -m quant_system.cli.train_cli

# Backtest with configured strategy + execution simulator
python -m quant_system.cli.backtest_cli

# Forward/paper test
python -m quant_system.cli.forward_cli

# Go live (ensure keys/funding are set)
python -m quant_system.cli.live_cli
```

5) **Dashboard**
- Launch Streamlit UI (example):
```bash
streamlit run quant_system/dashboard/app.py
```

## Configuration Model
- **YAML-first:** Strategy, features, labels, risk, regimes, sessions, and overrides are YAML-driven under `quant_system/config/`.
- **Overrides:** Use `regime_overrides*.yaml` and `session_overrides*.yaml` to specialize behavior by regime or session.
- **Secrets:** Never commit `.env`; use `.env.example` as the template.

## Development Notes
- **Style/Quality:** Prefer type hints, unit tests, and small, composable modules.
- **Testing:** Add/extend tests per module (e.g., `tests/` or alongside modules) before modifying live/forward paths.
- **Data Safety:** Guard live trading paths with dry-run flags and venue-specific rate limits; validate configs before firing orders.
- **Logging/Observability:** Use the provided logger utils and dashboard alerting for runtime visibility.

## Roadmap (suggested)
- Strengthen sim/live parity with deterministic fixtures.
- Expand feature/label coverage with richer microstructure signals.
- Add automated hyperparameter sweeps and ensemble governance.
- Enhance dashboards (real-time PnL attribution, execution quality KPIs).

## Support
Questions 
or issues? Open a ticket, ping the team, or reach out to the QuantFund AI (Nischal) ops channel. Please include environment details, configs used, and relevant logs when reporting problems.

## Detailed Pipeline Explanation

This section summarises the full QuantFund AI Smart Money Concept pipeline end-to-end, from data ingestion to execution. For an expanded explanation, please refer to the file `quant_smc_pipeline_report.md`.

### Data ingestion and resampling
- Historical 1‑minute OHLCV data are downloaded from Kraken via a resilient client with exponential backoff.
- A timeframe builder resamples 1‑min data into 15‑min, 1‑h, 6‑h and 12‑h bars with deterministic bar alignment.
- Typed dataclasses ensure consistent field names; resampled bars and raw bars are persisted to disk for reproducibility.

### Feature engineering
- Multi‑timeframe feature builders compute technical indicators (EMAs, ATR, volatility metrics) and liquidity/order‑flow features such as equal‑high/low density, sweeps, wick ratios and volume pressure.
- SMC detectors extract structural context: swing highs/lows, break of structure (BOS) and change of character (CHOCH), fair‑value gaps (FVGs), liquidity sweeps and order blocks. These detectors provide anchors, zones and structure context used for confluence calculations.
- Additional regime features (e.g., Gaussian HMM regimes) classify broad market conditions, while a feature store and resampler align multi‑TF features to 15‑min anchors.

### Label generation
- Specialist labels are derived from SMC events: BOS continuation, liquidity flow and micro‑momentum labels measure whether price reaches +3R or +1R targets before invalidation within defined horizons.
- Risk labels capture expected opportunity (EOP) and expected drawdown probability (EDP), while a hazard model defines survival labels tracking time‑to‑failure when stop loss or CHOCH occurs.

### Model training
- The training pipeline uses time‑series cross‑validation and Optuna‑based hyperparameter tuning for LightGBM/XGBoost classifiers. Each specialist label, meta model and hazard bin has its own calibrated classifier.
- A confluence engine combines specialist probabilities, meta probabilities, SMC strength and regime scores; an empirical calibrator ensures well‑behaved probability outputs.
- The model registry versioning system persists trained models, calibrators and metrics.

### Execution and risk management
- Gating and tiering logic apply multi‑step checks (Ocean/Waves/Flow) based on 12h, 6h and 1h contexts, plus hazard and EVR scores, to decide whether to trade.
- A multi‑parameter confluence score informs the TieringEngine, which assigns trades to tiers (A+, A, B or skip) and drives order execution and sizing.
- The MPC risk manager computes risk mode, lock fractions and hedge ratios using hazard/EDP/EOP probabilities and quantile forecasts; a position sizer scales position size based on risk mode and ATR.
- Hazard trailing, exposure tracking and cooling engines manage open positions, apply trailing stops and enforce cooldowns after large wins.

### Backtesting, forward testing and live trading
- A deterministic intrabar execution simulator and backtester evaluate strategies on historical data, accounting for slippage, fees and borrow/funding costs.
- Forward and live engines aggregate real‑time feeds to multi‑timeframe bars, evaluate SMC and ML signals, manage risk and execute orders via real or simulated adapters.
- Dashboard components provide real‑time analytics and trade monitoring, while a CLI orchestrates data ingestion, feature generation, labeling, training and backtesting tasks.
