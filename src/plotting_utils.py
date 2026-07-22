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


from typing import Dict
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def print_performance_summary(performance_metrics: Dict[str, float]) -> None:
    """
    Aesthetically prints the performance metrics to the terminal.

    This function groups the raw scalar KPIs into logical categories 
    (Trade Statistics, Absolute Returns, and Risk-Adjusted Metrics) 
    and formats them for readability (e.g., percentages, 2 decimal places).

    Parameters
    ----------
    performance_metrics : Dict[str, float]
        Dictionary containing scalar performance KPIs generated by 
        `calculate_performance`.

    Returns
    -------
    None
        Outputs directly to standard out.
    """
    
    if not performance_metrics:
        print("No performance metrics available (0 trades executed).")
        return

    print("=" * 50)
    print(" " * 10 + "STRATEGY PERFORMANCE TEAR SHEET")
    print("=" * 50)
    
    # --------------------------------------------------
    # Trade Statistics
    # --------------------------------------------------
    print("\n[ TRADE STATISTICS ]")
    print("-" * 50)
    print(f"Total Trades:           {int(performance_metrics.get('total_trades', 0))}")
    print(f"Win Rate:               {performance_metrics.get('win_rate', 0.0) * 100:.2f}%")
    print(f"Profit Factor:          {performance_metrics.get('profit_factor', 0.0):.2f}")
    
    average_r = performance_metrics.get('average_r_multiple', np.nan)
    print(f"Average R-Multiple:     {average_r:.2f}R" if not np.isnan(average_r) else "Average R-Multiple:     N/A")

    # --------------------------------------------------
    # Absolute Returns
    # --------------------------------------------------
    print("\n[ ABSOLUTE RETURNS ]")
    print("-" * 50)
    print(f"Total Net PnL:          ${performance_metrics.get('total_pnl', 0.0):,.2f}")
    print(f"Total Return:           {performance_metrics.get('total_return_percent', 0.0) * 100:.2f}%")
    print(f"Annualized Return:      {performance_metrics.get('annualized_return_percent', 0.0) * 100:.2f}%")

    # --------------------------------------------------
    # Risk & Risk-Adjusted Metrics
    # --------------------------------------------------
    print("\n[ RISK & RISK-ADJUSTED METRICS ]")
    print("-" * 50)
    print(f"Maximum Drawdown ($):   ${performance_metrics.get('maximum_drawdown_absolute', 0.0):,.2f}")
    print(f"Maximum Drawdown (%):   {performance_metrics.get('maximum_drawdown_percent', 0.0) * 100:.2f}%")
    print(f"Annualized Sharpe:      {performance_metrics.get('annualized_sharpe_ratio', 0.0):.2f}")
    
    sortino = performance_metrics.get('annualized_sortino_ratio', 0.0)
    print(f"Annualized Sortino:     {sortino:.2f}" if sortino != np.inf else "Annualized Sortino:     INF")
    
    calmar = performance_metrics.get('calmar_ratio', 0.0)
    print(f"Calmar Ratio:           {calmar:.2f}" if calmar != np.inf else "Calmar Ratio:           INF")
    print("=" * 50 + "\n")


'''def plot_performance_ratios(
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

    plt.show()'''


'''def plot_strategy_performance(
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

    plt.show()'''

def plot_strategy_performance(
    performance_plot_data: pd.DataFrame,
    performance_metrics: Dict[str, float],
    rolling_window_days: int = 126,
    trading_days_per_year: int = 252,
    figsize: tuple[int, int] = (14, 12),
) -> None:
    """
    Plots the equity curve, rolling risk-adjusted metrics, and final KPIs.

    This function utilizes the daily time-series equity curve to dynamically 
    calculate rolling (trailing) Sharpe and Sortino ratios over time. This 
    helps identify periods of strategy decay or regime shifting.

    Parameters
    ----------
    performance_plot_data : pd.DataFrame
        Time-series DataFrame containing "exit_time", "equity", 
        "cumulative_pnl", and "drawdown_percent".
    performance_metrics : Dict[str, float]
        Scalar KPIs outputted from `calculate_performance`.
    rolling_window_days : int, default=126
        The lookback window for rolling metric calculations (126 days ~ 6 months).
    trading_days_per_year : int, default=252
        Number of trading days in a year for annualization.
    figsize : tuple[int, int], default=(14, 12)
        Dimensions of the matplotlib figure.

    Returns
    -------
    None
        Displays a 3-panel matplotlib chart.
    """

    if performance_plot_data.empty:
        print("No plotting data available.")
        return

    # ==================================================
    # 1. Calculate Rolling Time-Series Metrics
    # ==================================================
    # Reconstruct daily returns from the equity curve
    daily_returns = performance_plot_data["equity"].pct_change().fillna(0.0)
    
    # Rolling Annualized Sharpe
    rolling_mean = daily_returns.rolling(window=rolling_window_days).mean()
    rolling_std = daily_returns.rolling(window=rolling_window_days).std()
    
    # Avoid division by zero
    rolling_std = rolling_std.replace(0, np.nan) 
    rolling_sharpe = (rolling_mean / rolling_std) * np.sqrt(trading_days_per_year)

    # Rolling Annualized Sortino
    negative_returns = daily_returns.copy()
    negative_returns[negative_returns > 0] = 0.0
    rolling_downside_std = negative_returns.rolling(window=rolling_window_days).std()
    
    rolling_downside_std = rolling_downside_std.replace(0, np.nan)
    rolling_sortino = (rolling_mean / rolling_downside_std) * np.sqrt(trading_days_per_year)

    # ==================================================
    # 2. Setup Figure and Axes
    # ==================================================
    fig, axes = plt.subplots(
        nrows=3,
        ncols=1,
        figsize=figsize,
        gridspec_kw={"height_ratios": [3, 2, 1]},
        sharex=False
    )
    fig.tight_layout(pad=4.0)

    time_index = performance_plot_data["exit_time"]

    # ==================================================
    # Panel 1: Cumulative PnL and Underwater Drawdown
    # ==================================================
    ax1 = axes[0]
    ax1.plot(time_index, performance_plot_data["cumulative_pnl"], color="blue", linewidth=1.5, label="Cumulative PnL")
    ax1.set_title("Strategy Cumulative PnL & Drawdowns", fontsize=12, fontweight="bold")
    ax1.set_ylabel("PnL ($)", fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper left")

    # Overlay Drawdown on a secondary Y-axis
    ax1_tw = ax1.twinx()
    ax1_tw.fill_between(
        time_index, 
        performance_plot_data["drawdown_percent"] * 100, 
        0, 
        color="red", 
        alpha=0.2, 
        label="Underwater Drawdown (%)"
    )
    ax1_tw.set_ylabel("Drawdown (%)", color="red", fontsize=10)
    ax1_tw.tick_params(axis="y", labelcolor="red")
    ax1_tw.set_ylim(bottom=(performance_plot_data["drawdown_percent"].min() * 100) * 1.2, top=0)

    # ==================================================
    # Panel 2: Rolling Risk-Adjusted Metrics (Time Series)
    # ==================================================
    ax2 = axes[1]
    ax2.plot(time_index, rolling_sharpe, color="purple", linewidth=1.2, label=f"{rolling_window_days}-Day Rolling Sharpe")
    ax2.plot(time_index, rolling_sortino, color="orange", linewidth=1.2, alpha=0.8, label=f"{rolling_window_days}-Day Rolling Sortino")
    
    ax2.axhline(y=0.0, color="black", linestyle="--", linewidth=1.0)
    # Add a benchmark line for a "Good" Sharpe Ratio (e.g., 1.0)
    ax2.axhline(y=1.0, color="green", linestyle=":", linewidth=1.0, alpha=0.5, label="Benchmark (1.0)")
    
    ax2.set_title("Rolling Risk-Adjusted Returns (Time-Series Edge Stability)", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Ratio", fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="upper left")

    # ==================================================
    # Panel 3: Overall Final Static Ratios
    # ==================================================
    ax3 = axes[2]
    
    ratio_names = ["Annualized Sharpe", "Annualized Sortino", "Calmar Ratio"]
    ratio_values = [
        performance_metrics.get("annualized_sharpe_ratio", 0.0),
        performance_metrics.get("annualized_sortino_ratio", 0.0),
        performance_metrics.get("calmar_ratio", 0.0)
    ]
    
    # Cap infinite values for plotting purposes
    ratio_values = [val if val != np.inf else 10.0 for val in ratio_values]

    colors = ['purple', 'orange', 'teal']
    bars = ax3.bar(ratio_names, ratio_values, color=colors, width=0.4)
    ax3.axhline(y=0.0, color="black", linestyle="--", linewidth=1.0)
    
    ax3.set_title("Overall Strategy KPIs", fontsize=12, fontweight="bold")
    ax3.grid(axis='y', alpha=0.3)

    # Add numeric labels to the top of the bars
    for bar in bars:
        yval = bar.get_height()
        ax3.text(
            bar.get_x() + bar.get_width()/2, 
            yval + (0.05 if yval > 0 else -0.15), 
            round(yval, 2), 
            ha='center', 
            va='bottom' if yval > 0 else 'top', 
            fontsize=10
        )

    plt.show()
