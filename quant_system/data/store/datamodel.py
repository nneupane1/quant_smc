"""
Alias shim for the data model definitions.
Imports Candle, TFCandleBatch, and Tick from data_model.py to satisfy legacy imports.
"""

from quant_system.data.store.data_model import Candle, TFCandleBatch, Tick  # noqa: F401

__all__ = ["Candle", "TFCandleBatch", "Tick"]
