"""
Plotting utilities for OHLCV and Supertrend data.

This module contains visualisation functions only. It does not calculate
technical indicators or modify the input DataFrames.

Expected OHLCV columns:

    open
    high
    low
    close

Optional volume column:

    tick_volume

Expected Supertrend columns:

    supertrend
    trend
    final_upper_band
    final_lower_band
"""

import numpy as np
import pandas as pd

from typing import Optional, Mapping

import matplotlib.pyplot as plt
import mplfinance as mpf


def _prepare_ohlcv_for_mplfinance(
    ohlcv_data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Prepare OHLCV data for plotting with mplfinance.

    The function creates a copy of the input DataFrame and renames the
    standard lower-case OHLCV columns to the column names expected by
    mplfinance.

    Parameters
    ----------
    ohlcv_data : pd.DataFrame
        OHLCV data indexed by a DatetimeIndex.

        Required columns:

            - "open"
            - "high"
            - "low"
            - "close"

        Optional column:

            - "tick_volume"

    Returns
    -------
    pd.DataFrame
        Copy of the input data with columns renamed to:

            - "Open"
            - "High"
            - "Low"
            - "Close"
            - "Volume"

    Raises
    ------
    KeyError
        If any required OHLC columns are missing.

    TypeError
        If the DataFrame index is not a DatetimeIndex.
    """

    if not isinstance(
        ohlcv_data.index,
        pd.DatetimeIndex,
    ):
        raise TypeError(
            "ohlcv_data must have a "
            "pandas DatetimeIndex."
        )

    required_columns = {
        "open",
        "high",
        "low",
        "close",
    }

    missing_columns = (
        required_columns
        - set(ohlcv_data.columns)
    )

    if missing_columns:
        raise KeyError(
            "Missing required OHLC columns: "
            f"{sorted(missing_columns)}"
        )

    mplfinance_data = (
        ohlcv_data[
            [
                "open",
                "high",
                "low",
                "close",
            ]
        ]
        .rename(
            columns={
                "open": "Open",
                "high": "High",
                "low": "Low",
                "close": "Close",
            }
        )
        .copy()
    )

    if "tick_volume" in ohlcv_data.columns:

        mplfinance_data["Volume"] = (
            ohlcv_data["tick_volume"]
        )

    return mplfinance_data


def plot_ohlcv(
    ohlcv_data: pd.DataFrame,
    title: str = "OHLCV Data",
    volume: bool = True,
    figsize: tuple[int, int] = (16, 8),
) -> None:
    """
    Plot OHLCV data as a candlestick chart.

    The function displays the price data using:

        - Open
        - High
        - Low
        - Close

    If the input contains a "tick_volume" column and `volume=True`,
    the volume panel is displayed below the price chart.

    Parameters
    ----------
    ohlcv_data : pd.DataFrame
        OHLCV data indexed by a DatetimeIndex.

        Required columns:

            - "open"
            - "high"
            - "low"
            - "close"

        Optional column:

            - "tick_volume"

    title : str, default="OHLCV Data"
        Title displayed above the chart.

    volume : bool, default=True
        Whether to display the volume panel.

    figsize : tuple[int, int], default=(16, 8)
        Figure dimensions in inches.

    Returns
    -------
    None
        The function displays the chart and does not return a value.

    Notes
    -----
    The function does not modify `ohlcv_data`.
    """

    mplfinance_data = (
        _prepare_ohlcv_for_mplfinance(
            ohlcv_data
        )
    )

    has_volume = (
        "Volume"
        in mplfinance_data.columns
    )

    display_volume = (
        volume
        and has_volume
    )

    mpf.plot(
        mplfinance_data,
        type="candle",
        style="charles",
        volume=display_volume,
        figsize=figsize,
        title=title,
    )


def plot_ohlcv_with_supertrend_bands(
    supertrend_data: pd.DataFrame,
    title: str = (
        "OHLCV Data with Final "
        "Supertrend Bands"
    ),
    volume: bool = True,
    figsize: tuple[int, int] = (18, 10),
) -> None:
    """
    Plot OHLCV candlesticks together with the final Supertrend bands.

    The chart displays:

        - OHLC candlesticks
        - Final upper Supertrend band
        - Final lower Supertrend band

    The final bands are plotted regardless of the current trend state.

    Parameters
    ----------
    supertrend_data : pd.DataFrame
        OHLCV data with calculated Supertrend components.

        Required columns:

            - "open"
            - "high"
            - "low"
            - "close"
            - "final_upper_band"
            - "final_lower_band"

        Optional column:

            - "tick_volume"

    title : str, default="OHLCV Data with Final Supertrend Bands"
        Title displayed above the chart.

    volume : bool, default=True
        Whether to display the volume panel if "tick_volume" exists.

    figsize : tuple[int, int], default=(18, 10)
        Figure dimensions in inches.

    Returns
    -------
    None
        The function displays the chart and does not return a value.

    Notes
    -----
    The input DataFrame is not modified.

    The final upper and lower bands are plotted as continuous series.
    """

    required_columns = {
        "final_upper_band",
        "final_lower_band",
    }

    missing_columns = (
        required_columns
        - set(supertrend_data.columns)
    )

    if missing_columns:
        raise KeyError(
            "Missing required Supertrend columns: "
            f"{sorted(missing_columns)}"
        )

    mplfinance_data = (
        _prepare_ohlcv_for_mplfinance(
            supertrend_data
        )
    )

    upper_band_plot = (
        mpf.make_addplot(
            supertrend_data[
                "final_upper_band"
            ],
            panel=0,
            width=1.0,
            color="red",
        )
    )

    lower_band_plot = (
        mpf.make_addplot(
            supertrend_data[
                "final_lower_band"
            ],
            panel=0,
            width=1.0,
            color="green",
        )
    )

    addplots = [
        upper_band_plot,
        lower_band_plot,
    ]

    has_volume = (
        "Volume"
        in mplfinance_data.columns
    )

    display_volume = (
        volume
        and has_volume
    )

    mpf.plot(
        mplfinance_data,
        type="candle",
        style="charles",
        addplot=addplots,
        volume=display_volume,
        figsize=figsize,
        title=title,
    )



def plot_ohlcv_with_supertrend(
    supertrend_data: pd.DataFrame,
    title: str = (
        "OHLCV Data with Supertrend"
    ),
    volume: bool = True,
    figsize: tuple[int, int] = (18, 10),
) -> None:
    """
    Plot OHLCV candlesticks with the active Supertrend line.

    The Supertrend line is separated into two visual components:

        Uptrend:
            trend == +1

        Downtrend:
            trend == -1

    Only the active Supertrend line is shown at each observation.

    Parameters
    ----------
    supertrend_data : pd.DataFrame
        OHLCV data containing calculated Supertrend values.

        Required columns:

            - "open"
            - "high"
            - "low"
            - "close"
            - "supertrend"
            - "trend"

        Optional column:

            - "tick_volume"

    title : str, default="OHLCV Data with Supertrend"
        Title displayed above the chart.

    volume : bool, default=True
        Whether to display the volume panel if "tick_volume" exists.

    figsize : tuple[int, int], default=(18, 10)
        Figure dimensions in inches.

    Returns
    -------
    None
        The function displays the chart and does not return a value.

    Notes
    -----
    The input DataFrame is not modified.

    The first valid Supertrend observations may be absent because
    the ATR requires a warm-up period.
    """

    required_columns = {
        "supertrend",
        "trend",
    }

    missing_columns = (
        required_columns
        - set(supertrend_data.columns)
    )

    if missing_columns:
        raise KeyError(
            "Missing required Supertrend columns: "
            f"{sorted(missing_columns)}"
        )

    mplfinance_data = (
        _prepare_ohlcv_for_mplfinance(
            supertrend_data
        )
    )

    supertrend_up = (
        supertrend_data[
            "supertrend"
        ]
        .where(
            supertrend_data["trend"]
            == 1
        )
    )

    supertrend_down = (
        supertrend_data[
            "supertrend"
        ]
        .where(
            supertrend_data["trend"]
            == -1
        )
    )

    supertrend_up_plot = (
        mpf.make_addplot(
            supertrend_up,
            panel=0,
            width=1.2,
            color="green",
        )
    )

    supertrend_down_plot = (
        mpf.make_addplot(
            supertrend_down,
            panel=0,
            width=1.2,
            color="red",
        )
    )

    addplots = [
        supertrend_up_plot,
        supertrend_down_plot,
    ]

    has_volume = (
        "Volume"
        in mplfinance_data.columns
    )

    display_volume = (
        volume
        and has_volume
    )

    mpf.plot(
        mplfinance_data,
        type="candle",
        style="charles",
        addplot=addplots,
        volume=display_volume,
        figsize=figsize,
        title=title,
    )



def plot_strategy_trades(
    signal_data: pd.DataFrame,
    trades_data: pd.DataFrame,
    title: str = "Supertrend Failed Breakout Strategy",
    figsize: tuple[float, float] = (18.0, 10.0),
    volume: bool = True,
) -> None:
    """
    Plot OHLCV data with the active Supertrend line and strategy
    entry and exit markers.

    The plot displays:

        1. Candlestick OHLC data.
        2. The active Supertrend line.
        3. Long entry markers.
        4. Short entry markers.
        5. Take-profit exit markers.
        6. Stop-loss exit markers.

    Long entries are plotted at the corresponding trade entry price.

    Short entries are plotted at the corresponding trade entry price.

    Exit markers are plotted at the corresponding trade exit price and
    are classified according to the `exit_reason` column.

    Parameters
    ----------
    signal_data : pd.DataFrame
        Chronologically sorted OHLCV data containing:

            - "open"
            - "high"
            - "low"
            - "close"
            - "supertrend"
            - "trend"

        If `volume=True`, the DataFrame should also contain:

            - "tick_volume"

    trades_data : pd.DataFrame
        Completed trades returned by `backtest_strategy`.

        Required columns:

            - "direction"
            - "entry_time"
            - "entry_price"
            - "exit_time"
            - "exit_price"
            - "exit_reason"

    title : str, default="Supertrend Failed Breakout Strategy"
        Title displayed above the chart.

    figsize : tuple[float, float], default=(18.0, 10.0)
        Figure dimensions in inches.

    volume : bool, default=True
        Whether to display the volume panel.

    Returns
    -------
    None
        The function displays the chart and does not return a value.

    Notes
    -----
    The timestamps in `trades_data` must correspond to timestamps present in
    the index of `signal_data`.

    For large datasets, it is recommended to pass a restricted window
    containing only the period of interest. Plotting millions of
    minute-level observations in a single candlestick chart is neither
    computationally efficient nor visually informative.
    """

    required_strategy_columns = {
        "open",
        "high",
        "low",
        "close",
        "supertrend",
        "trend",
    }

    missing_strategy_columns = (
        required_strategy_columns
        -
        set(signal_data.columns)
    )

    if missing_strategy_columns:

        raise KeyError(
            "Missing required strategy columns: "
            f"{sorted(missing_strategy_columns)}"
        )

    required_trade_columns = {
        "direction",
        "entry_time",
        "entry_price",
        "exit_time",
        "exit_price",
        "exit_reason",
    }

    missing_trade_columns = (
        required_trade_columns
        -
        set(trades_data.columns)
    )

    if missing_trade_columns:

        raise KeyError(
            "Missing required trade columns: "
            f"{sorted(missing_trade_columns)}"
        )

    if signal_data.empty:

        raise ValueError(
            "signal_data must not be empty."
        )

    # --------------------------------------------------
    # Prepare OHLCV data
    # --------------------------------------------------

    ohlcv_plot_data = (
        signal_data[
            [
                "open",
                "high",
                "low",
                "close",
            ]
            +
            (
                ["tick_volume"]
                if volume
                and
                "tick_volume"
                in signal_data.columns
                else []
            )
        ]
        .rename(
            columns={
                "open": "Open",
                "high": "High",
                "low": "Low",
                "close": "Close",
                "tick_volume": "Volume",
            }
        )
    )

    # --------------------------------------------------
    # Supertrend line
    # --------------------------------------------------

    supertrend_plot_data = (
        signal_data["supertrend"]
    )

    supertrend_addplot = (
        mpf.make_addplot(
            supertrend_plot_data,
            panel=0,
            width=1.2,
        )
    )

    # --------------------------------------------------
    # Entry markers
    # --------------------------------------------------

    long_entry_prices = (
        pd.Series(
            np.nan,
            index=signal_data.index,
        )
    )

    short_entry_prices = (
        pd.Series(
            np.nan,
            index=signal_data.index,
        )
    )

    # --------------------------------------------------
    # Exit markers
    # --------------------------------------------------

    take_profit_exit_prices = (
        pd.Series(
            np.nan,
            index=signal_data.index,
        )
    )

    stop_loss_exit_prices = (
        pd.Series(
            np.nan,
            index=signal_data.index,
        )
    )

    for _, trade in trades_data.iterrows():

        entry_time = (
            pd.Timestamp(
                trade["entry_time"]
            )
        )

        exit_time = (
            pd.Timestamp(
                trade["exit_time"]
            )
        )

        if entry_time in signal_data.index:

            if (
                trade["direction"]
                == "long"
            ):

                long_entry_prices.loc[
                    entry_time
                ] = trade["entry_price"]

            elif (
                trade["direction"]
                == "short"
            ):

                short_entry_prices.loc[
                    entry_time
                ] = trade["entry_price"]

        if exit_time in signal_data.index:

            if (
                trade["exit_reason"]
                == "take_profit"
            ):

                take_profit_exit_prices.loc[
                    exit_time
                ] = trade["exit_price"]

            elif (
                trade["exit_reason"]
                == "stop_loss"
            ):

                stop_loss_exit_prices.loc[
                    exit_time
                ] = trade["exit_price"]

    # --------------------------------------------------
    # Add markers
    # --------------------------------------------------

    long_entry_addplot = (
        mpf.make_addplot(
            long_entry_prices,
            type="scatter",
            marker="^",
            markersize=100,
            panel=0,
        )
    )

    short_entry_addplot = (
        mpf.make_addplot(
            short_entry_prices,
            type="scatter",
            marker="v",
            markersize=100,
            panel=0,
        )
    )

    take_profit_exit_addplot = (
        mpf.make_addplot(
            take_profit_exit_prices,
            type="scatter",
            marker="o",
            markersize=60,
            panel=0,
        )
    )

    stop_loss_exit_addplot = (
        mpf.make_addplot(
            stop_loss_exit_prices,
            type="scatter",
            marker="x",
            markersize=80,
            panel=0,
        )
    )

    addplots = [
        supertrend_addplot,
        long_entry_addplot,
        short_entry_addplot,
        take_profit_exit_addplot,
        stop_loss_exit_addplot,
    ]

    # --------------------------------------------------
    # Plot
    # --------------------------------------------------

    mpf.plot(
        ohlcv_plot_data,
        type="candle",
        style="charles",
        addplot=addplots,
        volume=(
            volume
            and
            "Volume"
            in ohlcv_plot_data.columns
        ),
        figsize=figsize,
        title=title,
        ylabel="Price",
    )



def plot_cumulative_pnl(
    trades_data: pd.DataFrame,
    title: str = "Cumulative Strategy PnL",
    figsize: tuple[float, float] = (15.0, 6.0),
) -> None:
    """
    Plot the cumulative realized PnL of completed trades.

    For each completed trade:

        cumulative_PnL_n
        =
        sum(PnL_1, ..., PnL_n)

    The cumulative PnL is plotted at the corresponding trade exit time,
    since the profit or loss becomes realized when the position is closed.

    Parameters
    ----------
    trades_data : pd.DataFrame
        Completed trades returned by `backtest_strategy`.

        Required columns:

            - "exit_time"
            - "pnl"

    title : str, default="Cumulative Strategy PnL"
        Title displayed above the chart.

    figsize : tuple[float, float], default=(15.0, 6.0)
        Figure dimensions in inches.

    Returns
    -------
    None
        The function displays the chart and does not return a value.

    Raises
    ------
    KeyError
        If required columns are missing.

    ValueError
        If the trades_data DataFrame is empty.
    """

    required_columns = {
        "exit_time",
        "pnl",
    }

    missing_columns = (
        required_columns
        -
        set(trades_data.columns)
    )

    if missing_columns:

        raise KeyError(
            "Missing required PnL columns: "
            f"{sorted(missing_columns)}"
        )

    if trades_data.empty:

        raise ValueError(
            "Cannot plot PnL for an empty trades_data DataFrame."
        )

    pnl_plot_data = (
        trades_data[
            [
                "exit_time",
                "pnl",
            ]
        ]
        .copy()
    )

    pnl_plot_data["exit_time"] = (
        pd.to_datetime(
            pnl_plot_data["exit_time"]
        )
    )

    pnl_plot_data = (
        pnl_plot_data
        .sort_values("exit_time")
    )

    pnl_plot_data["cumulative_pnl"] = (
        pnl_plot_data["pnl"]
        .cumsum()
    )

    plt.figure(
        figsize=figsize
    )

    plt.plot(
        pnl_plot_data["exit_time"],
        pnl_plot_data["cumulative_pnl"],
        linewidth=1.5,
    )

    plt.axhline(
        y=0.0,
        linestyle="--",
        linewidth=1.0,
    )

    plt.title(
        title
    )

    plt.xlabel(
        "Exit time"
    )

    plt.ylabel(
        "Cumulative PnL"
    )

    plt.grid(
        True,
        alpha=0.3,
    )

    plt.tight_layout()

    plt.show()


def plot_performance_ratios(
    performance_metrics: Mapping[str, float],
    title: str = "Strategy Performance Ratios",
    figsize: tuple[float, float] = (10.0, 6.0),
) -> None:
    """
    Plot the final Sharpe and Calmar ratios as a bar chart.

    Parameters
    ----------
    performance_metrics : Mapping[str, float]
        Dictionary-like object containing:

            - "trade_sharpe_ratio"
            - "calmar_ratio"

    title : str, default="Strategy Performance Ratios"
        Title displayed above the chart.

    figsize : tuple[float, float], default=(10.0, 6.0)
        Figure dimensions in inches.

    Returns
    -------
    None
        The function displays the chart and does not return a value.

    Raises
    ------
    KeyError
        If either required metric is missing.

    Notes
    -----
    Sharpe and Calmar ratios are dimensionless performance statistics.
    They are plotted as separate bars because their numerical values are
    directly comparable as summary statistics, but they represent
    different concepts:

        Sharpe:
            Return relative to volatility.

        Calmar:
            Annualized return relative to maximum drawdown.
    """

    required_metrics = {
        "trade_sharpe_ratio",
        "calmar_ratio",
    }

    missing_metrics = (
        required_metrics
        -
        set(performance_metrics)
    )

    if missing_metrics:

        raise KeyError(
            "Missing required performance metrics: "
            f"{sorted(missing_metrics)}"
        )

    ratio_names = [
        "Sharpe Ratio",
        "Calmar Ratio",
    ]

    ratio_values = [
        performance_metrics[
            "trade_sharpe_ratio"
        ],
        performance_metrics[
            "calmar_ratio"
        ],
    ]

    plt.figure(
        figsize=figsize
    )

    plt.bar(
        ratio_names,
        ratio_values,
    )

    plt.axhline(
        y=0.0,
        linestyle="--",
        linewidth=1.0,
    )

    plt.title(
        title
    )

    plt.ylabel(
        "Ratio"
    )

    plt.grid(
        axis="y",
        alpha=0.3,
    )

    plt.tight_layout()

    plt.show()


def plot_strategy_performance(
    trades_data: pd.DataFrame,
    performance_metrics: Mapping[str, float],
    title: str = "Strategy Performance",
    figsize: tuple[float, float] = (15.0, 10.0),
) -> None:
    """
    Plot the cumulative PnL together with the final Sharpe and Calmar
    ratios in a single figure.

    The figure contains two vertically stacked panels:

        Panel 1
        -------
        Cumulative realized PnL as a function of trade exit time.

        Panel 2
        -------
        Final trade-level Sharpe ratio and Calmar ratio.

    Parameters
    ----------
    trades_data : pd.DataFrame
        Completed trades returned by `backtest_strategy`.

        Required columns:

            - "exit_time"
            - "pnl"

    performance_metrics : Mapping[str, float]
        Dictionary-like object containing:

            - "trade_sharpe_ratio"
            - "calmar_ratio"

    title : str, default="Strategy Performance"
        Overall figure title.

    figsize : tuple[float, float], default=(15.0, 10.0)
        Figure dimensions in inches.

    Returns
    -------
    None
        The function displays the chart and does not return a value.

    Notes
    -----
    The Sharpe and Calmar ratios are final summary statistics, so they
    are displayed in the lower panel as scalar values rather than as
    time series.
    """

    required_trade_columns = {
        "exit_time",
        "pnl",
    }

    missing_trade_columns = (
        required_trade_columns
        -
        set(trades_data.columns)
    )

    if missing_trade_columns:

        raise KeyError(
            "Missing required trade columns: "
            f"{sorted(missing_trade_columns)}"
        )

    required_metrics = {
        "trade_sharpe_ratio",
        "calmar_ratio",
    }

    missing_metrics = (
        required_metrics
        -
        set(performance_metrics)
    )

    if missing_metrics:

        raise KeyError(
            "Missing required performance metrics: "
            f"{sorted(missing_metrics)}"
        )

    if trades_data.empty:

        raise ValueError(
            "Cannot plot performance for an empty trades_data DataFrame."
        )

    performance_plot_data = (
        trades_data[
            [
                "exit_time",
                "pnl",
            ]
        ]
        .copy()
    )

    performance_plot_data["exit_time"] = (
        pd.to_datetime(
            performance_plot_data["exit_time"]
        )
    )

    performance_plot_data = (
        performance_plot_data
        .sort_values("exit_time")
    )

    performance_plot_data["cumulative_pnl"] = (
        performance_plot_data["pnl"]
        .cumsum()
    )

    figure, axes = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=figsize,
        gridspec_kw={
            "height_ratios": [
                3,
                1,
            ]
        },
    )

    # ==================================================
    # Panel 1: Cumulative PnL
    # ==================================================

    axes[0].plot(
        performance_plot_data[
            "exit_time"
        ],
        performance_plot_data[
            "cumulative_pnl"
        ],
        linewidth=1.5,
    )

    axes[0].axhline(
        y=0.0,
        linestyle="--",
        linewidth=1.0,
    )

    axes[0].set_title(
        "Cumulative PnL"
    )

    axes[0].set_ylabel(
        "PnL"
    )

    axes[0].grid(
        True,
        alpha=0.3,
    )

    # ==================================================
    # Panel 2: Sharpe and Calmar
    # ==================================================

    ratio_names = [
        "Sharpe Ratio",
        "Calmar Ratio",
    ]

    ratio_values = [
        performance_metrics[
            "trade_sharpe_ratio"
        ],
        performance_metrics[
            "calmar_ratio"
        ],
    ]

    axes[1].bar(
        ratio_names,
        ratio_values,
    )

    axes[1].axhline(
        y=0.0,
        linestyle="--",
        linewidth=1.0,
    )

    axes[1].set_ylabel(
        "Ratio"
    )

    axes[1].grid(
        axis="y",
        alpha=0.3,
    )

    figure.suptitle(
        title
    )

    figure.tight_layout()

    plt.show()
