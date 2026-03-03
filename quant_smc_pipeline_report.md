# Quant SMC Pipeline Report

This report summarizes the current repo implementation after the consistency fixes.

## Runtime Shape

- Config is merged through `quant_system/config/config_loader.py`.
- The documented module entrypoints now resolve:
  - `python -m quant_system.cli.train_cli`
  - `python -m quant_system.cli.backtest_cli`
  - `python -m quant_system.cli.forward_cli`
  - `python -m quant_system.cli.live_cli`

## Data Flow

1. Historical or live 1m candles are acquired.
2. Bars are normalized into `15m`, `1h`, `6h`, and `12h`.
3. The 15m frame is used as the execution spine.
4. Higher-timeframe SMC context is projected onto 15m rows.
5. Features, labels, and model outputs are attached to the same 15m decision frame.

## Feature Flow

- 15m SMC detectors attach swings, BOS/CHOCH, FVGs, sweeps, and zones.
- 1h/6h/12h SMC context is joined back to 15m via backward-asof alignment.
- EMA, volatility, liquidity, absorption, and regime blocks are then appended.

## Label Flow

- `label_liq_flow`
- `label_bos_cont`
- `label_momo`
- `label_eop`
- `label_edp`
- `hazard_event`
- `hazard_time`

These are generated from the 15m feature spine using forward windows.

## Model Flow

- Specialist classifiers are trained for the five label families.
- A meta model and a confluence model are trained over specialist outputs.
- Hazard is modeled as a per-bin discrete-time classifier.
- Quantile forecasters provide return distribution estimates.

## Execution Flow

- Gate checks: 12h -> 6h -> 1h context.
- Confluence scoring.
- EVR target/stop reasoning.
- Tiering: `A+`, `A`, `B`, or `skip`.
- Hazard trailing and cooldown logic manage open positions.

## Capital Policy

The default capital policy is now ticket-based instead of MPC-led:

- `execution.starting_equity: 20000`
- `execution.capital.base_ticket_usd: 20000`
- `execution.capital.compound_ticket: true`
- `execution.capital.use_mpc: false`

That means the default engines deploy a 20k ticket that compounds with deployable equity over time, while keeping the existing MPC module available as an optional plug-in.

## Current Caveat

The repo still contains overlapping generations of code. The main runtime surfaces now initialize cleanly, but there are still areas that should be consolidated further, especially around duplicated feature/label helpers and dashboard/reporting modules.
