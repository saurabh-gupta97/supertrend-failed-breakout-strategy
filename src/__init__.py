"""
Supertrend Failed Breakout Trading Strategy.

This package provides utilities for:

    - Loading and validating OHLCV data.
    - Calculating Supertrend indicators.
    - Generating trading signals.
    - Backtesting the trading strategy.
    - Walk-forward cross-validation ot choose the best parameter combo for each test window.
    - Calculating performance metrics.
    - Visualising price and strategy performance.
"""

from . import data_pipeline
from . import supertrend_utils
from . import strategy_engine
from . import cross_validation
from . import plotting_utils

__all__ = [
    "data_pipeline",
    "supertrend_utils",
    "strategy_engine",
    "cross_validation",
    "plotting_utils",
]