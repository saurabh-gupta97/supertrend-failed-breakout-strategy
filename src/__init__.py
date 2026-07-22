"""
Supertrend Failed Breakout Trading Strategy.

This package provides utilities for:

    - Loading and validating OHLCV data.
    - Calculating Supertrend indicators.
    - Generating trading signals.
    - Backtesting the trading strategy.
    - Calculating performance metrics.
    - Visualising price and strategy performance.
"""

from . import data_pipeline
from . import supertrend_utils
from . import strategy_engine
from . import plotting_utils

__all__ = [
    "data_pipeline",
    "supertrend_utils",
    "strategy_engine",
    "plotting_utils",
]