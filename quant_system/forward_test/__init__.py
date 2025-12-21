"""
Forward Test Engine Package

This package handles real-time paper trading simulation using
the complete live execution logic (SMC, ML, Confluence, EVR,
Hazard, MPC, Position Sizing, Exposure Tracking, and Adaptive Cooling).

Components:
 - forward_state.py   : Tracks virtual equity, positions, hedges, locked capital.
 - forward_executor.py: Executes paper trades (entry/exit/hedge/trailing).
 - forward_engine.py  : Real-time orchestrator pulling data + recomputing signals.
 - forward_dashboard_adapter.py : Sends live state to Streamlit dashboards.

Used when you want to simulate the bot in real time (forward testing)
without placing money on Kraken.
"""
# inside ForwardExecutor.__init__
self.exec_state = {
    "leverage": 1,
    "size": 20000.0,
    "tp": None,
    "sl": None,
    "risk_mode": "normal",
    "hedge_enabled": False
}
self.execution_bridge = None
