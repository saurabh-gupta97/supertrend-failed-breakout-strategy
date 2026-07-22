"""
Trading strategy and backtesting engine for the Supertrend Failed
Breakout strategy.

Strategy logic
--------------

Long entry:

    1. Current Supertrend regime is bullish.
    2. The candle low touches or penetrates the Supertrend line.
    3. The candle closes strictly above the Supertrend line.

Short entry:

    1. Current Supertrend regime is bearish.
    2. The candle high touches or penetrates the Supertrend line.
    3. The candle closes strictly below the Supertrend line.

Trade construction
------------------

Long:

    entry_price = close_t

    stop_loss = low_t

    risk = entry_price - stop_loss

    take_profit = entry_price + reward_risk * risk


Short:

    entry_price = close_t

    stop_loss = high_t

    risk = stop_loss - entry_price

    take_profit = entry_price - reward_risk * risk

The entry candle is not used to monitor the stop-loss or take-profit.
Trade monitoring begins from the following candle.

This module assumes that the input data is chronologically sorted.
"""

from typing import Literal, Any

import numpy as np
import pandas as pd


def _validate_signal_columns(
    supertrend_data: pd.DataFrame,
) -> None:
    """
    Validate that the input contains all columns required for signal
    generation.

    Parameters
    ----------
    supertrend_data : pd.DataFrame
        OHLCV data containing calculated Supertrend values.

    Raises
    ------
    KeyError
        If one or more required columns are missing.
    """

    required_columns = {
        "low",
        "high",
        "close",
        "supertrend",
        "trend",
    }

    missing_columns = (
        required_columns
        - set(supertrend_data.columns)
    )

    if missing_columns:
        raise KeyError(
            "Missing required columns for signal generation: "
            f"{sorted(missing_columns)}"
        )


def _validate_backtest_columns(
    signal_data: pd.DataFrame,
) -> None:
    """
    Validate that the input contains all columns required for
    backtesting.

    Parameters
    ----------
    signal_data : pd.DataFrame
        OHLCV data containing generated long and short signals.

    Raises
    ------
    KeyError
        If required columns are missing.
    """

    required_columns = {
        "low",
        "high",
        "close",
        "long_signal",
        "short_signal",
    }

    missing_columns = (
        required_columns
        - set(signal_data.columns)
    )

    if missing_columns:
        raise KeyError(
            "Missing required columns for backtesting: "
            f"{sorted(missing_columns)}"
        )


def generate_signals(
    supertrend_data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Generate entry signals for the Supertrend Failed Breakout strategy.

    Long entry condition
    --------------------

    A long signal is generated at observation t when:

        trend_t = +1

        low_t <= supertrend_t

        close_t > supertrend_t

    Therefore, the candle must touch or penetrate the Supertrend line
    intrabar, but ultimately close above it.

    Short entry condition
    ---------------------

    A short signal is generated at observation t when:

        trend_t = -1

        high_t >= supertrend_t

        close_t < supertrend_t

    Therefore, the candle must touch or penetrate the Supertrend line
    intrabar, but ultimately close below it.

    The signals are generated using only information available from the
    current candle. The strategy therefore assumes that execution occurs
    at the close of the signal candle.

    Parameters
    ----------
    supertrend_data : pd.DataFrame
        OHLCV data containing:

            - "low"
            - "high"
            - "close"
            - "supertrend"
            - "trend"

    Returns
    -------
    pd.DataFrame
        Copy of the input DataFrame with two additional Boolean columns:

            - "long_signal"
            - "short_signal"

    Notes
    -----
    The input DataFrame is not modified.

    Signals are not generated when the Supertrend value is NaN, which
    normally occurs during the ATR warm-up period.
    """

    _validate_signal_columns(
        supertrend_data
    )

    signal_data = (
        supertrend_data.copy()
    )

    valid_supertrend = (
        signal_data["supertrend"]
        .notna()
    )

    signal_data["long_signal"] = (
        valid_supertrend
        &
        (
            signal_data["trend"]
            == 1
        )
        &
        (
            signal_data["low"]
            <=
            signal_data["supertrend"]
        )
        &
        (
            signal_data["close"]
            >
            signal_data["supertrend"]
        )
    )

    signal_data["short_signal"] = (
        valid_supertrend
        &
        (
            signal_data["trend"]
            == -1
        )
        &
        (
            signal_data["high"]
            >=
            signal_data["supertrend"]
        )
        &
        (
            signal_data["close"]
            <
            signal_data["supertrend"]
        )
    )

    return signal_data


def backtest_strategy(
    signal_data: pd.DataFrame,
    reward_risk: float = 2.0,
    same_bar_priority: Literal[
        "stop",
        "target",
    ] = "stop",
) -> pd.DataFrame:
    """
    Backtest the Supertrend Failed Breakout strategy.

    Entry
    -----

    Long:

        entry_price = close_t

    Short:

        entry_price = close_t

    The entry occurs at the close of the signal candle.

    Long trade construction
    -----------------------

        stop_loss = low_t

        risk = entry_price - stop_loss

        take_profit =
            entry_price + reward_risk * risk

    Short trade construction
    ------------------------

        stop_loss = high_t

        risk = stop_loss - entry_price

        take_profit =
            entry_price - reward_risk * risk

    Trade management
    ----------------

    Only one position may be open at a time.

    The entry candle is not used to monitor the stop-loss or take-profit.
    Monitoring begins from the next observation.

    If both stop-loss and take-profit are touched during the same candle,
    the result is determined by `same_bar_priority`.

    Parameters
    ----------
    signal_data : pd.DataFrame
        Chronologically sorted OHLCV data containing:

            - "low"
            - "high"
            - "close"
            - "long_signal"
            - "short_signal"

    reward_risk : float, default=2.0
        Reward-to-risk ratio.

        For a long trade:

            TP = entry + reward_risk * risk

        For a short trade:

            TP = entry - reward_risk * risk

    same_bar_priority : {"stop", "target"}, default="stop"
        Determines the outcome when both the stop-loss and take-profit
        are reached within the same OHLC candle.

        "stop":
            Conservative assumption. Stop-loss is assumed to occur first.

        "target":
            Optimistic assumption. Take-profit is assumed to occur first.

    Returns
    -------
    pd.DataFrame
        One row per completed trade.

        Columns include:

            direction
            entry_time
            entry_index
            entry_price
            stop_loss
            take_profit
            risk
            exit_time
            exit_index
            exit_price
            exit_reason
            pnl
            r_multiple

    Notes
    -----
    Trades that remain open at the end of the dataset are not included
    in the returned DataFrame.

    This is a deliberate choice because their final exit price is not
    known unless an explicit end-of-data liquidation rule is introduced.
    """

    _validate_backtest_columns(
        signal_data
    )

    if reward_risk <= 0:
        raise ValueError(
            "reward_risk must be greater than zero."
        )

    if same_bar_priority not in {
        "stop",
        "target",
    }:

        raise ValueError(
            "same_bar_priority must be "
            "'stop' or 'target'."
        )

    trades: list[dict] = []

    open_position: dict | None = None

    number_of_observations = (
        len(signal_data)
    )

    for observation_index in range(
        number_of_observations
    ):

        observation = (
            signal_data.iloc[
                observation_index
            ]
        )

        timestamp = (
            signal_data.index[
                observation_index
            ]
        )

        # ==================================================
        # NO OPEN POSITION
        # ==================================================

        if open_position is None:

            # ----------------------------------------------
            # Long entry
            # ----------------------------------------------

            if observation["long_signal"]:

                entry_price = (
                    observation["close"]
                )

                stop_loss = (
                    observation["low"]
                )

                risk = (
                    entry_price
                    -
                    stop_loss
                )

                if risk <= 0:
                    continue

                take_profit = (
                    entry_price
                    +
                    reward_risk
                    * risk
                )

                open_position = {
                    "direction": "long",
                    "entry_time": timestamp,
                    "entry_index": (
                        observation_index
                    ),
                    "entry_price": (
                        entry_price
                    ),
                    "stop_loss": stop_loss,
                    "take_profit": (
                        take_profit
                    ),
                    "risk": risk,
                }

                continue

            # ----------------------------------------------
            # Short entry
            # ----------------------------------------------

            if observation["short_signal"]:

                entry_price = (
                    observation["close"]
                )

                stop_loss = (
                    observation["high"]
                )

                risk = (
                    stop_loss
                    -
                    entry_price
                )

                if risk <= 0:
                    continue

                take_profit = (
                    entry_price
                    -
                    reward_risk
                    * risk
                )

                open_position = {
                    "direction": "short",
                    "entry_time": timestamp,
                    "entry_index": (
                        observation_index
                    ),
                    "entry_price": (
                        entry_price
                    ),
                    "stop_loss": stop_loss,
                    "take_profit": (
                        take_profit
                    ),
                    "risk": risk,
                }

                continue

        # ==================================================
        # MANAGE OPEN POSITION
        # ==================================================

        else:

            # --------------------------------------------------
            # Long position
            # --------------------------------------------------

            if (
                open_position["direction"]
                == "long"
            ):

                stop_hit = (
                    observation["low"]
                    <=
                    open_position[
                        "stop_loss"
                    ]
                )

                target_hit = (
                    observation["high"]
                    >=
                    open_position[
                        "take_profit"
                    ]
                )

                exit_reason = None
                exit_price = None

                if stop_hit and target_hit:

                    if (
                        same_bar_priority
                        == "stop"
                    ):

                        exit_reason = (
                            "stop_loss"
                        )

                        exit_price = (
                            open_position[
                                "stop_loss"
                            ]
                        )

                    else:

                        exit_reason = (
                            "take_profit"
                        )

                        exit_price = (
                            open_position[
                                "take_profit"
                            ]
                        )

                elif stop_hit:

                    exit_reason = (
                        "stop_loss"
                    )

                    exit_price = (
                        open_position[
                            "stop_loss"
                        ]
                    )

                elif target_hit:

                    exit_reason = (
                        "take_profit"
                    )

                    exit_price = (
                        open_position[
                            "take_profit"
                        ]
                    )

                else:

                    continue

                pnl = (
                    exit_price
                    -
                    open_position[
                        "entry_price"
                    ]
                )

            # --------------------------------------------------
            # Short position
            # --------------------------------------------------

            else:

                stop_hit = (
                    observation["high"]
                    >=
                    open_position[
                        "stop_loss"
                    ]
                )

                target_hit = (
                    observation["low"]
                    <=
                    open_position[
                        "take_profit"
                    ]
                )

                exit_reason = None
                exit_price = None

                if stop_hit and target_hit:

                    if (
                        same_bar_priority
                        == "stop"
                    ):

                        exit_reason = (
                            "stop_loss"
                        )

                        exit_price = (
                            open_position[
                                "stop_loss"
                            ]
                        )

                    else:

                        exit_reason = (
                            "take_profit"
                        )

                        exit_price = (
                            open_position[
                                "take_profit"
                            ]
                        )

                elif stop_hit:

                    exit_reason = (
                        "stop_loss"
                    )

                    exit_price = (
                        open_position[
                            "stop_loss"
                        ]
                    )

                elif target_hit:

                    exit_reason = (
                        "take_profit"
                    )

                    exit_price = (
                        open_position[
                            "take_profit"
                        ]
                    )

                else:

                    continue

                pnl = (
                    open_position[
                        "entry_price"
                    ]
                    -
                    exit_price
                )

            r_multiple = (
                pnl
                /
                open_position["risk"]
            )

            completed_trade = {
                **open_position,
                "exit_time": timestamp,
                "exit_index": (
                    observation_index
                ),
                "exit_price": exit_price,
                "exit_reason": exit_reason,
                "pnl": pnl,
                "r_multiple": r_multiple,
            }

            trades.append(
                completed_trade
            )

            open_position = None

    return pd.DataFrame(trades)



def calculate_performance(
    trades: pd.DataFrame,
    initial_capital: float = 100_000.0,
    annualization_factor: int = 252,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Calculate strategy performance from completed trades.

    The strategy is assumed to trade one unit of the asset. Therefore:

        PnL_i = realized price PnL of trade i

        equity_t = initial_capital + cumulative realized PnL

        return_t = equity_t / equity_{t-1} - 1

    Sharpe ratio is calculated from daily account returns:

        Sharpe_t =
            mean(daily_returns[:t])
            /
            std(daily_returns[:t])
            * sqrt(annualization_factor)

    Calmar ratio is calculated as:

        Calmar_t =
            CAGR_t / abs(maximum_drawdown_t)

    Parameters
    ----------
    trades : pd.DataFrame
        Completed trades. Must contain:
        - exit_time
        - pnl
        - r_multiple

    initial_capital : float, default=100_000.0
        Initial account capital.

    annualization_factor : int, default=252
        Number of trading days per year.

    Returns
    -------
    tuple[pd.DataFrame, dict[str, Any]]
        performance_over_time:
            Daily equity, returns, drawdown, Sharpe ratio, and Calmar ratio.

        performance_metrics:
            Summary performance statistics.
    """

    required_columns = {
        "exit_time",
        "pnl",
        "r_multiple",
    }

    missing_columns = required_columns.difference(trades.columns)

    if missing_columns:
        raise ValueError(
            f"trades is missing required columns: {missing_columns}"
        )

    if initial_capital <= 0:
        raise ValueError(
            "initial_capital must be positive."
        )

    if len(trades) == 0:

        return (
            pd.DataFrame(),
            {}
        )

    completed_trades = trades.copy()

    completed_trades["exit_time"] = pd.to_datetime(
        completed_trades["exit_time"]
    )

    completed_trades = completed_trades.sort_values(
        "exit_time"
    )

    # ==================================================
    # Trade statistics
    # ==================================================

    total_trades = len(completed_trades)

    winning_trades = (
        completed_trades["pnl"] > 0
    ).sum()

    losing_trades = (
        completed_trades["pnl"] < 0
    ).sum()

    breakeven_trades = (
        completed_trades["pnl"] == 0
    ).sum()

    win_rate = (
        winning_trades
        /
        total_trades
    )

    average_r = (
        completed_trades["r_multiple"].mean()
    )

    total_r = (
        completed_trades["r_multiple"].sum()
    )

    gross_profit = (
        completed_trades.loc[
            completed_trades["pnl"] > 0,
            "pnl"
        ].sum()
    )

    gross_loss = (
        completed_trades.loc[
            completed_trades["pnl"] < 0,
            "pnl"
        ].sum()
    )

    if gross_loss != 0:

        profit_factor = (
            gross_profit
            /
            abs(gross_loss)
        )

    else:

        profit_factor = np.inf

    total_pnl = (
        completed_trades["pnl"].sum()
    )

    # ==================================================
    # Daily PnL
    # ==================================================

    daily_pnl = (
        completed_trades
        .set_index("exit_time")["pnl"]
        .resample("1D")
        .sum()
    )

    # Include days without closed trades
    daily_pnl = daily_pnl.fillna(0.0)

    # ==================================================
    # Equity curve
    # ==================================================

    equity = (
        initial_capital
        +
        daily_pnl.cumsum()
    )

    # ==================================================
    # Daily returns
    # ==================================================

    daily_returns = (
        equity
        .pct_change()
        .fillna(0.0)
    )

    # ==================================================
    # Running maximum and drawdown
    # ==================================================

    running_maximum = (
        equity.cummax()
    )

    drawdown_absolute = (
        equity
        -
        running_maximum
    )

    drawdown_percent = (
        equity
        /
        running_maximum
        -
        1.0
    )

    # ==================================================
    # Expanding Sharpe ratio
    # ==================================================

    expanding_mean_return = (
        daily_returns.expanding()
        .mean()
    )

    expanding_std_return = (
        daily_returns.expanding()
        .std()
    )

    sharpe_ratio = (
        expanding_mean_return
        /
        expanding_std_return
        *
        np.sqrt(annualization_factor)
    )

    sharpe_ratio = (
        sharpe_ratio.replace(
            [np.inf, -np.inf],
            np.nan
        )
    )

    # ==================================================
    # Expanding CAGR
    # ==================================================

    elapsed_years = (
        (equity.index - equity.index[0])
        .total_seconds()
        /
        (365.25 * 24 * 60 * 60)
    )

    elapsed_years = pd.Series(
        elapsed_years,
        index=equity.index
    )

    cagr = pd.Series(
        np.nan,
        index=equity.index
    )

    valid_cagr = (
        (elapsed_years > 0)
        &
        (equity > 0)
    )

    cagr.loc[valid_cagr] = (
        (
            equity.loc[valid_cagr]
            /
            initial_capital
        )
        **
        (
            1.0
            /
            elapsed_years.loc[valid_cagr]
        )
        -
        1.0
    )

    # ==================================================
    # Expanding Calmar ratio
    # ==================================================

    maximum_drawdown_percent = (
        drawdown_percent.expanding()
        .min()
    )

    calmar_ratio = (
        cagr
        /
        maximum_drawdown_percent.abs()
    )

    calmar_ratio = (
        calmar_ratio.replace(
            [np.inf, -np.inf],
            np.nan
        )
    )

    # ==================================================
    # Performance DataFrame
    # ==================================================

    performance_over_time = pd.DataFrame(
        {
            "daily_pnl": daily_pnl,
            "equity": equity,
            "daily_return": daily_returns,
            "running_maximum": running_maximum,
            "drawdown_absolute": drawdown_absolute,
            "drawdown_percent": drawdown_percent,
            "sharpe_ratio": sharpe_ratio,
            "cagr": cagr,
            "calmar_ratio": calmar_ratio,
        }
    )

    # ==================================================
    # Final performance metrics
    # ==================================================

    final_equity = (
        equity.iloc[-1]
    )

    total_return = (
        final_equity
        /
        initial_capital
        -
        1.0
    )

    maximum_drawdown_absolute = (
        drawdown_absolute.min()
    )

    maximum_drawdown_percent = (
        drawdown_percent.min()
    )

    final_sharpe_ratio = (
        sharpe_ratio.iloc[-1]
    )

    final_calmar_ratio = (
        calmar_ratio.iloc[-1]
    )

    performance_metrics = {

        "total_trades": total_trades,

        "winning_trades": winning_trades,

        "losing_trades": losing_trades,

        "breakeven_trades": breakeven_trades,

        "win_rate": win_rate,

        "average_r": average_r,

        "total_r": total_r,

        "profit_factor": profit_factor,

        "total_pnl": total_pnl,

        "initial_capital": initial_capital,

        "final_equity": final_equity,

        "total_return": total_return,

        "maximum_drawdown_absolute":
            maximum_drawdown_absolute,

        "maximum_drawdown_percent":
            maximum_drawdown_percent,

        "sharpe_ratio":
            final_sharpe_ratio,

        "cagr":
            cagr.iloc[-1],

        "calmar_ratio":
            final_calmar_ratio,
    }

    return (
        performance_over_time,
        performance_metrics
    )

