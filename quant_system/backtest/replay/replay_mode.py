"""
Programmatic replay runner.

Provides a small non-UI interface around ReplayController so replay logic can
be exercised without the dashboard layer.
"""

from typing import Any, Dict, List, Optional

from quant_system.backtest.replay.replay_controller import ReplayController


def run(
    controller: Optional[ReplayController] = None,
    *,
    candles_15m=None,
    smc_features=None,
    execution_log=None,
    model_bundle: Optional[Any] = None,
    config: Optional[Dict[str, Any]] = None,
    steps: Optional[int] = None,
) -> List[Dict[str, Any]]:
    if controller is None:
        if candles_15m is None or config is None:
            raise ValueError("Either provide an initialized ReplayController or pass candles_15m and config.")
        controller = ReplayController(
            candles_15m=candles_15m,
            smc_features=smc_features,
            execution_log=execution_log,
            model_bundle=model_bundle,
            config=config,
        )

    payloads = [controller.render_payload()]
    max_steps = controller.n - 1 if steps is None else max(int(steps), 0)
    for _ in range(max_steps):
        payloads.append(controller.step_forward())
    return payloads
