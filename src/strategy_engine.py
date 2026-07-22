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

from typing import Literal, Dict, Tuple, Any

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
        "open",
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


from typing import Literal, Any
import numpy as np
import pandas as pd

def backtest_strategy(
    signal_data: pd.DataFrame,
    reward_risk: float = 2.0,
    same_bar_priority: Literal["stop", "target"] = "stop",
    slippage_ticks: float = 0.0,
    tick_size: float = 0.01,
) -> pd.DataFrame:
    """
    Backtest the Supertrend Failed Breakout strategy using fast NumPy iteration.

    Entry (Corrected for Look-Ahead Bias)
    -------------------------------------
    A signal is generated based on the closing price of observation t.
    Because the close must be confirmed, the trade is executed at the OPEN of 
    the following candle (t + 1), incorporating transaction slippage.

    Long entry:
        entry_price = open_{t+1} + (slippage_ticks * tick_size)

    Short entry:
        entry_price = open_{t+1} - (slippage_ticks * tick_size)

    Trade construction
    ------------------
    The stop loss is anchored to the extremes of the signal candle (t).

    Long:
        stop_loss = low_t
        risk = entry_price - stop_loss
        take_profit = entry_price + reward_risk * risk

    Short:
        stop_loss = high_t
        risk = stop_loss - entry_price
        take_profit = entry_price - reward_risk * risk

    Trade management
    ----------------
    Only one position may be open at a time. Monitoring begins immediately on 
    the execution candle (t+1). If both stop-loss and take-profit are touched 
    during the same candle, the result is determined by `same_bar_priority`.
    Slippage is applied upon exiting the position.

    Parameters
    ----------
    signal_data : pd.DataFrame
        Chronologically sorted OHLCV data containing:
            - "open", "low", "high", "close", "long_signal", "short_signal"
    reward_risk : float, default=2.0
        Reward-to-risk ratio.
    same_bar_priority : {"stop", "target"}, default="stop"
        Outcome when both stop-loss and take-profit are reached within the same bar.
    slippage_ticks : float, default=0.0
        Execution penalty defined in ticks (e.g., bid-ask spread friction).
    tick_size : float, default=0.01
        The minimum price movement of the asset.

    Returns
    -------
    pd.DataFrame
        One row per completed trade.
    """

    _validate_backtest_columns(signal_data)

    if reward_risk <= 0:
        raise ValueError("reward_risk must be greater than zero.")

    if same_bar_priority not in {"stop", "target"}:
        raise ValueError("same_bar_priority must be 'stop' or 'target'.")

    trades: list[dict] = []
    open_position: dict | None = None
    number_of_observations = len(signal_data)

    # Extract pandas columns to underlying NumPy arrays for extreme speed O(1) lookups
    opens = signal_data["open"].to_numpy(dtype=np.float64)
    highs = signal_data["high"].to_numpy(dtype=np.float64)
    lows = signal_data["low"].to_numpy(dtype=np.float64)
    long_signals = signal_data["long_signal"].to_numpy(dtype=bool)
    short_signals = signal_data["short_signal"].to_numpy(dtype=bool)
    timestamps = signal_data.index.to_numpy()

    for observation_index in range(number_of_observations):

        # ==================================================
        # NO OPEN POSITION
        # ==================================================
        if open_position is None:

            # Cannot execute a new trade if we are on the final dataset observation
            if observation_index + 1 >= number_of_observations:
                continue

            # ----------------------------------------------
            # Long entry
            # ----------------------------------------------
            if long_signals[observation_index]:
                entry_index = observation_index + 1
                entry_time = timestamps[entry_index]
                
                # Execute on the open of the next bar, penalize with slippage
                entry_price = opens[entry_index] + (slippage_ticks * tick_size)
                
                # Stop loss anchored to the signal bar's low
                stop_loss = lows[observation_index]
                risk = entry_price - stop_loss
                
                if risk <= 0:
                    continue  # Gap opened past our intended stop loss
                    
                take_profit = entry_price + (reward_risk * risk)
                
                open_position = {
                    "direction": "long",
                    "entry_time": entry_time,
                    "entry_index": entry_index,
                    "entry_price": entry_price,
                    "stop_loss": stop_loss,
                    "take_profit": take_profit,
                    "risk": risk,
                }
                continue

            # ----------------------------------------------
            # Short entry
            # ----------------------------------------------
            if short_signals[observation_index]:
                entry_index = observation_index + 1
                entry_time = timestamps[entry_index]
                
                # Execute on the open of the next bar, penalize with slippage
                entry_price = opens[entry_index] - (slippage_ticks * tick_size)
                
                # Stop loss anchored to the signal bar's high
                stop_loss = highs[observation_index]
                risk = stop_loss - entry_price
                
                if risk <= 0:
                    continue  # Gap opened past our intended stop loss
                    
                take_profit = entry_price - (reward_risk * risk)
                
                open_position = {
                    "direction": "short",
                    "entry_time": entry_time,
                    "entry_index": entry_index,
                    "entry_price": entry_price,
                    "stop_loss": stop_loss,
                    "take_profit": take_profit,
                    "risk": risk,
                }
                continue

        # ==================================================
        # MANAGE OPEN POSITION
        # ==================================================
        else:
            current_high = highs[observation_index]
            current_low = lows[observation_index]
            current_timestamp = timestamps[observation_index]

            exit_reason = None
            exit_price = None

            # --------------------------------------------------
            # Long position
            # --------------------------------------------------
            if open_position["direction"] == "long":
                stop_hit = current_low <= open_position["stop_loss"]
                target_hit = current_high >= open_position["take_profit"]

                if stop_hit and target_hit:
                    if same_bar_priority == "stop":
                        exit_reason = "stop_loss"
                        exit_price = open_position["stop_loss"]
                    else:
                        exit_reason = "take_profit"
                        exit_price = open_position["take_profit"]
                elif stop_hit:
                    exit_reason = "stop_loss"
                    exit_price = open_position["stop_loss"]
                elif target_hit:
                    exit_reason = "take_profit"
                    exit_price = open_position["take_profit"]
                else:
                    continue

                # Apply friction to the exit
                exit_price -= (slippage_ticks * tick_size)
                pnl = exit_price - open_position["entry_price"]

            # --------------------------------------------------
            # Short position
            # --------------------------------------------------
            else:
                stop_hit = current_high >= open_position["stop_loss"]
                target_hit = current_low <= open_position["take_profit"]

                if stop_hit and target_hit:
                    if same_bar_priority == "stop":
                        exit_reason = "stop_loss"
                        exit_price = open_position["stop_loss"]
                    else:
                        exit_reason = "take_profit"
                        exit_price = open_position["take_profit"]
                elif stop_hit:
                    exit_reason = "stop_loss"
                    exit_price = open_position["stop_loss"]
                elif target_hit:
                    exit_reason = "take_profit"
                    exit_price = open_position["take_profit"]
                else:
                    continue
                
                # Apply friction to the exit
                exit_price += (slippage_ticks * tick_size)
                pnl = open_position["entry_price"] - exit_price

            r_multiple = pnl / open_position["risk"]

            completed_trade = {
                **open_position,
                "exit_time": current_timestamp,
                "exit_index": observation_index,
                "exit_price": exit_price,
                "exit_reason": exit_reason,
                "pnl": pnl,
                "r_multiple": r_multiple,
            }

            trades.append(completed_trade)
            open_position = None

    return pd.DataFrame(trades)

'''def backtest_strategy(
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

    return pd.DataFrame(trades)'''



'''def calculate_performance(
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
    )'''


def calculate_performance(
    trades_data: pd.DataFrame,
    initial_capital: float = 100000.0,
    risk_free_rate: float = 0.0,
    trading_days_per_year: int = 252,
) -> Tuple[Dict[str, float], pd.DataFrame]:
    """
    Calculate institutional-grade performance metrics from trade executions.

    This function constructs a daily equity curve from the discrete trade log 
    to calculate mathematically valid time-series risk metrics (Sharpe, 
    Sortino, Max Drawdown). 

    Parameters
    ----------
    trades_data : pd.DataFrame
        A DataFrame containing completed trades. Must include at least:
            - "exit_time": Datetime of trade exit.
            - "pnl": The realized profit or loss of the trade.
            - "r_multiple": The risk-adjusted return of the trade.
    initial_capital : float, default=100000.0
        The starting capital for the backtest.
    risk_free_rate : float, default=0.0
        The annualized risk-free rate used for Sharpe and Sortino calculations.
    trading_days_per_year : int, default=252
        Number of trading days in a year, used for annualization.

    Returns
    -------
    Tuple[Dict[str, float], pd.DataFrame]
        - A dictionary containing scalar performance metrics (KPIs).
        - A DataFrame containing the daily time-series equity curve and 
          drawdowns, intended to be passed directly to `plotting_utils.py`.
    """

    if trades_data.empty:
        return {}, pd.DataFrame()

    # ==================================================
    # 1. Trade-level statistics
    # ==================================================
    total_trades = len(trades_data)
    winning_trades = trades_data[trades_data["pnl"] > 0]
    losing_trades = trades_data[trades_data["pnl"] < 0]

    win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0.0
    
    gross_profit = winning_trades["pnl"].sum()
    gross_loss = abs(losing_trades["pnl"].sum())
    
    # Profit factor: Gross Profit / Gross Loss
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf

    average_r_multiple = trades_data["r_multiple"].mean() if "r_multiple" in trades_data.columns else np.nan

    # ==================================================
    # 2. Daily Equity Curve Construction
    # ==================================================
    # Group PnL by the day the trade exited to build a continuous time series
    trades_data["exit_date"] = pd.to_datetime(trades_data["exit_time"]).dt.normalize()
    daily_pnl = trades_data.groupby("exit_date")["pnl"].sum()
    
    # Fill in days where no trades occurred with 0.0 PnL to maintain time-series integrity
    full_date_range = pd.date_range(start=daily_pnl.index.min(), end=daily_pnl.index.max(), freq="B")
    daily_pnl = daily_pnl.reindex(full_date_range, fill_value=0.0)

    cumulative_pnl = daily_pnl.cumsum()
    equity_curve = initial_capital + cumulative_pnl
    
    # Calculate daily percentage returns
    daily_returns = equity_curve.pct_change().fillna(0.0)

    # ==================================================
    # 3. Drawdown Calculations
    # ==================================================
    rolling_peak = equity_curve.cummax()
    drawdown_absolute = equity_curve - rolling_peak
    drawdown_percent = drawdown_absolute / rolling_peak
    
    maximum_drawdown_percent = abs(drawdown_percent.min())
    maximum_drawdown_absolute = abs(drawdown_absolute.min())

    # ==================================================
    # 4. Risk-Adjusted Returns (Annualized)
    # ==================================================
    daily_rf_rate = risk_free_rate / trading_days_per_year
    excess_returns = daily_returns - daily_rf_rate

    # Sharpe Ratio
    return_standard_deviation = daily_returns.std()
    if return_standard_deviation > 0:
        annualized_sharpe_ratio = (excess_returns.mean() / return_standard_deviation) * np.sqrt(trading_days_per_year)
    else:
        annualized_sharpe_ratio = 0.0

    # Sortino Ratio (Penalizes only downside volatility)
    negative_returns = excess_returns[excess_returns < 0]
    downside_standard_deviation = negative_returns.std()
    if downside_standard_deviation > 0:
        annualized_sortino_ratio = (excess_returns.mean() / downside_standard_deviation) * np.sqrt(trading_days_per_year)
    else:
        annualized_sortino_ratio = np.inf if excess_returns.mean() > 0 else 0.0

    # Calmar Ratio (Annualized Return / Max Drawdown)
    total_return_percent = (equity_curve.iloc[-1] / initial_capital) - 1.0
    years_in_market = len(daily_returns) / trading_days_per_year
    
    if years_in_market > 0:
        annualized_return = ((1 + total_return_percent) ** (1 / years_in_market)) - 1.0
    else:
        annualized_return = 0.0

    calmar_ratio = annualized_return / maximum_drawdown_percent if maximum_drawdown_percent > 0 else np.inf

    # ==================================================
    # 5. Compile Results
    # ==================================================
    performance_metrics = {
        "total_trades": total_trades,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "average_r_multiple": average_r_multiple,
        "total_pnl": trades_data["pnl"].sum(),
        "total_return_percent": total_return_percent,
        "annualized_return_percent": annualized_return,
        "maximum_drawdown_absolute": maximum_drawdown_absolute,
        "maximum_drawdown_percent": maximum_drawdown_percent,
        "annualized_sharpe_ratio": annualized_sharpe_ratio,
        "annualized_sortino_ratio": annualized_sortino_ratio,
        "calmar_ratio": calmar_ratio,
    }

    # Drop the temporary date column from the original trades dataframe
    trades_data.drop(columns=["exit_date"], inplace=True)

    # Prepare plotting data structure
    performance_plot_data = pd.DataFrame({
        "exit_time": equity_curve.index, # Align index for plotting
        "equity": equity_curve.values,
        "cumulative_pnl": cumulative_pnl.values,
        "drawdown_absolute": drawdown_absolute.values,
        "drawdown_percent": drawdown_percent.values,
    })

    return performance_metrics, performance_plot_data