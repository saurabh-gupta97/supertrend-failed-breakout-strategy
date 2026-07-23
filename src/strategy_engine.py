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
    penetration_model: Literal["ticks", "atr_frac"] = "ticks",
    penetration_ticks: int = 1,
    tick_size: float = 0.01,
    min_penetration_atr_fraction: float = 0.2,
    sensitivity_scalar: float = 1.0,
    squashing_function: Literal["tanh", "relu"] = "tanh",
    volume_multiplier: float = 1.5,
    volume_ma_period: int = 20,
) -> pd.DataFrame:
    """
    Generate entry signals for the Supertrend Failed Breakout strategy
    incorporating volume confirmation and ATR-normalized penetration depth filters.

    Long entry condition
    --------------------
    A long signal is generated at observation t when:
        - trend_t = +1
        - low_t <= supertrend_t - (penetration_ticks * tick_size)
        - (supertrend_t - low_t) >= min_penetration_atr_fraction * atr_t
        - close_t > supertrend_t
        - volume_t >= volume_multiplier * MovingAverage(Volume, volume_ma_period)

    Short entry condition
    ---------------------
    A short signal is generated at observation t when:
        - trend_t = -1
        - high_t >= supertrend_t + (penetration_ticks * tick_size)
        - (high_t - supertrend_t) >= min_penetration_atr_fraction * atr_t
        - close_t < supertrend_t
        - volume_t >= volume_multiplier * MovingAverage(Volume, volume_ma_period)

    The signals are generated using only information available from the
    current candle. The strategy therefore assumes that execution occurs
    at the open of the next signal candle.

    Parameters
    ----------
    supertrend_data : pd.DataFrame
        OHLCV data containing "low", "high", "close", "supertrend", "trend", "atr", 
        and optionally a volume column ("tick_volume" or "volume").
    penetration_model: ticks or atr_frac
        Choose whether to use fixed number of ticks or a fraction of the atr for penetration condition.
    penetration_ticks: int, default=1
        Minimum absolute tick buffer for Supertrend violation.
    tick_size: float, default=0.01
        The minimum price movement of the asset.
    min_penetration_atr_fraction: float, default=0.2
        The minimum required penetration depth expressed as a fraction of the current ATR.
    sensitivity_scalar: float, default=1.0
        Multiplier for the normalized gap before squashing.
    squashing_function: {"tanh", "relu"}, default="tanh"
        The function used to squash signal strength to a value between [0, 1].
    volume_multiplier: float, default=1.5
        The minimum multiple of baseline volume required to confirm a liquidity sweep.
    volume_ma_period: int, default=20
        Lookback window for the moving average of volume.
        
    Returns
    -------
    pd.DataFrame
        Copy of the input DataFrame with updated signal columns and breakout metrics.
    """

    signal_data = supertrend_data.copy()

    # 1. Identify and extract volume column safely
    vol_col = None
    for candidate in ["tick_volume", "volume"]:
        if candidate in signal_data.columns:
            vol_col = candidate
            break

    if vol_col is not None:
        baseline_vol = signal_data[vol_col].rolling(window=volume_ma_period).mean()
        signal_data["volume_ma"] = baseline_vol
        vol_arr = signal_data[vol_col].to_numpy(dtype=np.float64)
        vol_ma_arr = baseline_vol.to_numpy(dtype=np.float64)
        has_volume = True
    else:
        has_volume = False
        vol_arr = np.ones(len(signal_data), dtype=np.float64)
        vol_ma_arr = np.zeros(len(signal_data), dtype=np.float64)

    # 2. Extract ALL arrays strictly to NumPy for C-level speed
    valid_supertrend = signal_data["supertrend"].notna().to_numpy()
    trend_arr = signal_data["trend"].to_numpy()
    supertrend_arr = signal_data["supertrend"].to_numpy()
    close_arr = signal_data["close"].to_numpy(dtype=np.float64)
    low_arr = signal_data["low"].to_numpy(dtype=np.float64)
    high_arr = signal_data["high"].to_numpy(dtype=np.float64)
    atr_arr = signal_data["atr"].to_numpy(dtype=np.float64)

    # 3. Define Filter Conditions
    if has_volume:
        volume_condition = (vol_arr >= volume_multiplier * vol_ma_arr) & ~np.isnan(vol_ma_arr)
    else:
        volume_condition = np.ones(len(signal_data), dtype=bool)

    # ATR-normalized depth conditions
    long_depth_condition = (supertrend_arr - low_arr) >= (min_penetration_atr_fraction * atr_arr)
    short_depth_condition = (high_arr - supertrend_arr) >= (min_penetration_atr_fraction * atr_arr)

    # 4. Generate directional signals incorporating institutional filters
    long_signal = (
        valid_supertrend &
        (trend_arr == 1) &
        (low_arr <= supertrend_arr - (penetration_ticks * tick_size)) &
        (long_depth_condition if penetration_model == "atr_frac" else 1) &
        (close_arr > supertrend_arr) &
        volume_condition
    )

    short_signal = (
        valid_supertrend &
        (trend_arr == -1) &
        (high_arr >= supertrend_arr + (penetration_ticks * tick_size)) &
        (short_depth_condition if penetration_model == "atr_frac" else 1) &
        (close_arr < supertrend_arr) &
        volume_condition
    )

    signal_data["long_signal"] = long_signal
    signal_data["short_signal"] = short_signal

    # 5. Dynamic Breakout Price Assignment (The Trap Extreme Wick)
    breakout_arr = np.full_like(close_arr, np.nan, dtype=np.float64)
    breakout_arr = np.where(long_signal, low_arr, breakout_arr)
    breakout_arr = np.where(short_signal, high_arr, breakout_arr)
    
    signal_data["breakout_price"] = breakout_arr

    # 6. Calculate Signal Strength
    active_signals = long_signal | short_signal
    
    raw_gap = np.abs(close_arr - breakout_arr)
    epsilon = 1e-8
    normalized_gap = raw_gap / (atr_arr + epsilon)
    
    if squashing_function == "tanh":
        strength = np.tanh(sensitivity_scalar * normalized_gap)
    elif squashing_function == "relu":
        strength = np.clip(sensitivity_scalar * normalized_gap, 0.0, 1.0)
    else:
        raise ValueError("squashing_function must be 'tanh' or 'relu'")
    
    valid_mask = active_signals & ~np.isnan(breakout_arr)
    signal_data["signal_strength"] = np.where(valid_mask, strength, 0.0)

    return signal_data


    
'''def generate_signals(
    supertrend_data: pd.DataFrame,
    penetration_ticks: int = 1,
    tick_size: float = 0.01,
    squashing_function: Literal["tanh", "relu"] = "tanh",
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
    penetration_ticks: int
        Number of ticks between the supertrend and the high/low for penetration to be significant/accepted.

    tick_size: float
        The minimum price movement of the asset.

    squashing_function: tanh or relu
        The squashing function used to squash signal strength to a value between 0 and 1.
        
    Returns
    -------
    pd.DataFrame
        Copy of the input DataFrame with two additional Boolean columns:

            - "long_signal"
            - "short_signal"
            - "breakout_price"
            - "signal_strength"

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
    trend_arr = signal_data["trend"].to_numpy()
    supertrend_arr = signal_data["supertrend"].to_numpy()
    close_arr = signal_data["close"].to_numpy()
    low_arr = signal_data["low"].to_numpy()
    high_arr = signal_data["high"].to_numpy()
    atr_arr = signal_data["atr"].to_numpy()
    

    signal_data["long_signal"] = (
        valid_supertrend
        &
        (
            trend_arr
            == 1
        )
        &
        (
            low_arr
            <=
            supertrend_arr - penetration_ticks * tick_size
        )
        &
        (
            close_arr
            >
            supertrend_arr
        )
    )

    signal_data["short_signal"] = (
        valid_supertrend
        &
        (
            trend_arr
            == -1
        )
        &
        (
            high_arr
            >=
            supertrend_arr + penetration_ticks * tick_size
        )
        &
        (
            close_arr
            <
            supertrend_arr
        )
    )

    # 1. Create the boolean mask for active signals (using bitwise OR)
    active_signals = signal_data["long_signal"].to_numpy() | signal_data["short_signal"].to_numpy()
    
    # 2. Calculate gaps (vectorized math is faster than masking the array first)
    raw_gap = np.abs(close_arr - breakout_arr)
    epsilon = 1e-8
    normalized_gap = raw_gap / (atr_arr + epsilon)
    
    # 4. Apply the squashing function
    strength = np.tanh(sensitivity_scalar * normalized_gap)
    
    # 5. Apply the mask: keep strength ONLY if it's an active signal AND breakout isn't NaN
    valid_mask = active_signals & ~np.isnan(breakout_arr)
    signal_data["signal_strength"] = np.where(valid_mask, strength, 0.0)

    return signal_data'''




'''def backtest_strategy(
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

    trades_data: list[dict] = []
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

            trades_data.append(completed_trade)
            open_position = None

    return pd.DataFrame(trades_data)'''




def backtest_strategy(
    signal_data: pd.DataFrame,
    initial_capital: float = 1000000.0,
    reward_risk: float = 2.0,
    stop_loss_ratio: float = 1.0,
    position_sizing: Literal["fixed", "weighted"] = "fixed",
    max_position_per_trade_fraction: float = 0.01,
    max_risk_per_trade_fraction : float = 0.001,
    min_holdings_fraction: float = 0.80,
    transaction_costs_model: Literal["flat", "percentage"] = "percentage",
    transaction_costs_flat: float = 2.50,
    transaction_costs_fraction: float = 0.0005,
    same_bar_priority: Literal["stop", "target"] = "stop",
    slippage_ticks: int = 0,
    tick_size: float = 0.01,
) -> pd.DataFrame:
    """
    Backtest the strategy with portfolio limits, costs, and dynamic sizing.

    Entry
    -----
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
        stop_loss = entry_price - stop_loss_ratio * (close_t - low_t)
        risk = entry_price - stop_loss
        take_profit = entry_price + reward_risk * risk

    Short:
        stop_loss = entry_price + stop_loss_ratio * (high_t - close_t)
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
        Requires: "open", "high", "low", "close", "breakout_price", 
        "long_signal", "short_signal", and optionally "signal_strength".
    initial_capital : float
        Starting account balance.
    reward_risk : float
        Target multiple of the initial risk.
    stop_loss_ratio : float
        Multiplier for the gap between close and breakout price to define the stop.
    position_sizing : {"fixed", "weighted"}
        Whether to trade a fixed size or scale by signal_strength.
    max_position_per_trade_fraction : float
        Maximum fraction of current capital to invest per trade.
    max_risk_per_trade_fraction : float
        Maximum fraction of current capital to risk/expose per trade (this is determined by the stop loss distance).
    min_holdings_fraction : float
        If current capital drops below (initial_capital * min_holdings_fraction), 
        trading halts permanently (Portfolio Liquidation).
    transaction_costs_model: flat or percentage
        Choose whther broker applies flat fees per transaction or percentage of the order value. 
    transaction_costs_flat: float
        Commission fee applied on entry AND exit as a flat fee. 
    transaction_costs_fraction : float
        Commission fee applied on entry AND exit as a fraction of the position size.
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

    trades_data: list[dict] = []
    open_position: dict | None = None
    number_of_observations = len(signal_data)

    current_capital = initial_capital
    liquidation_threshold = initial_capital * min_holdings_fraction

    # Extract to fast NumPy arrays
    opens = signal_data["open"].to_numpy(dtype=np.float64)
    closes = signal_data["close"].to_numpy(dtype=np.float64)
    highs = signal_data["high"].to_numpy(dtype=np.float64)
    lows = signal_data["low"].to_numpy(dtype=np.float64)
    breakout_prices = signal_data["breakout_price"].to_numpy(dtype=np.float64)
    long_signals = signal_data["long_signal"].to_numpy(dtype=bool)
    short_signals = signal_data["short_signal"].to_numpy(dtype=bool)
    timestamps = signal_data.index.to_numpy()
    
    # Handle Signal Strength array if weighted
    if position_sizing == "weighted" and "signal_strength" in signal_data.columns:
        signal_strengths = signal_data["signal_strength"].to_numpy(dtype=np.float64)
    else:
        signal_strengths = np.ones(number_of_observations, dtype=np.float64)

    for i in range(number_of_observations):
        
        # ==================================================
        # CHECK FOR PORTFOLIO RUIN (MAX DRAWDOWN LIMIT)
        # ==================================================
        if current_capital <= liquidation_threshold:
            break

        # ==================================================
        # ENTRY LOGIC (NO OPEN POSITION)
        # ==================================================
        if open_position is None:
            if i + 1 >= number_of_observations:
                continue

            is_long = long_signals[i]
            is_short = short_signals[i]

            if is_long or is_short:
                entry_index = i + 1
                
                # 1. Dynamic Position Sizing (Dollar Target Capital)
                target_capital = max_position_per_trade_fraction * current_capital
                if position_sizing == "weighted":
                    target_capital = max_position_per_trade_fraction * current_capital * signal_strengths[i]

                # 2. Strict Integer Share Sizing (Floored to avoid capital/risk limit breaches)
                raw_shares = target_capital / opens[entry_index]
                integer_shares = np.floor(raw_shares)

                # Guardrail: If stock price exceeds target capital allocation, skip trade
                if integer_shares < 1:
                    continue

                # 3. Entry Price with Slippage
                slippage_penalty = slippage_ticks * tick_size
                entry_price = opens[entry_index] + slippage_penalty if is_long else opens[entry_index] - slippage_penalty

                # 4. Dynamic Stop Loss based on Close/Breakout gap
                gap = abs(closes[i] - breakout_prices[i])
                stop_distance = stop_loss_ratio * gap
                
                # Fallback safeguard against zero/negative distances
                stop_distance = max(stop_distance, tick_size)

                stop_loss = entry_price - stop_distance if is_long else entry_price + stop_distance
                take_profit = entry_price + (reward_risk * stop_distance) if is_long else entry_price - (reward_risk * stop_distance)

                # 5. Calculate Entry Commission (Based on Entry Notional Value)
                entry_notional = entry_price * integer_shares
                entry_commission = transaction_costs_fraction * entry_notional if (transaction_costs_model == "percentage") else transaction_costs_flat

                open_position = {
                    "direction": "long" if is_long else "short",
                    "entry_time": timestamps[entry_index],
                    "entry_price": entry_price,
                    "stop_loss": stop_loss,
                    "take_profit": take_profit,
                    "size": int(integer_shares),       # Strictly stored as integer shares
                    "risk_per_share": stop_distance,
                    "entry_commission": entry_commission, # Tracked for round-trip accounting
                }
                
                # Deduct entry commission from running capital immediately
                current_capital -= entry_commission

        # ==================================================
        # EXIT LOGIC (MANAGE OPEN POSITION)
        # ==================================================
        else:
            current_high = highs[i]
            current_low = lows[i]

            exit_reason = None
            exit_price = None

            if open_position["direction"] == "long":
                stop_hit = current_low <= open_position["stop_loss"]
                target_hit = current_high >= open_position["take_profit"]

                if stop_hit and target_hit:
                    exit_reason = "stop_loss" if same_bar_priority == "stop" else "take_profit"
                    exit_price = open_position["stop_loss"] if same_bar_priority == "stop" else open_position["take_profit"]
                elif stop_hit:
                    exit_reason = "stop_loss"
                    exit_price = open_position["stop_loss"]
                elif target_hit:
                    exit_reason = "take_profit"
                    exit_price = open_position["take_profit"]
                else:
                    continue

                # Apply exit slippage and calculate gross PnL
                exit_price -= (slippage_ticks * tick_size)
                gross_pnl = (exit_price - open_position["entry_price"]) * open_position["size"]

                # Calculate Exit Commission based on the selected model
                if transaction_costs_model == "percentage":
                    exit_notional = exit_price * open_position["size"]
                    exit_commission = transaction_costs_fraction * exit_notional
                elif transaction_costs_model == "flat":
                    exit_commission = transaction_costs_flat
                else:
                    raise ValueError("transaction_costs_model must be 'percentage' or 'flat'")

                # True Round-Trip Net PnL
                total_commission = open_position["entry_commission"] + exit_commission
                net_pnl = gross_pnl - total_commission 

            else: # Short Direction
                stop_hit = current_high >= open_position["stop_loss"]
                target_hit = current_low <= open_position["take_profit"]

                if stop_hit and target_hit:
                    exit_reason = "stop_loss" if same_bar_priority == "stop" else "take_profit"
                    exit_price = open_position["stop_loss"] if same_bar_priority == "stop" else open_position["take_profit"]
                elif stop_hit:
                    exit_reason = "stop_loss"
                    exit_price = open_position["stop_loss"]
                elif target_hit:
                    exit_reason = "take_profit"
                    exit_price = open_position["take_profit"]
                else:
                    continue
                
                # Apply exit slippage and calculate gross PnL
                exit_price += (slippage_ticks * tick_size)
                gross_pnl = (open_position["entry_price"] - exit_price) * open_position["size"]

                # Calculate Exit Commission based on the selected model
                if transaction_costs_model == "percentage":
                    exit_notional = exit_price * open_position["size"]
                    exit_commission = transaction_costs_fraction * exit_notional
                elif transaction_costs_model == "flat":
                    exit_commission = transaction_costs_flat
                else:
                    raise ValueError("transaction_costs_model must be 'percentage' or 'flat'")

                # True Round-Trip Net PnL
                total_commission = open_position["entry_commission"] + exit_commission
                net_pnl = gross_pnl - total_commission 


            # Since entry commission was already deducted at entry, we add gross PnL minus exit commission
            capital_change = gross_pnl - exit_commission
            current_capital += capital_change

            trades_data.append({
                **open_position,
                "exit_time": timestamps[i],
                "exit_price": exit_price,
                "exit_reason": exit_reason,
                "pnl": net_pnl, # Clean, fully accounted round-trip net PnL
                "r_multiple": net_pnl / (open_position["risk_per_share"] * open_position["size"]) if (open_position["risk_per_share"] * open_position["size"]) > 0 else 0.0,
                "running_capital": current_capital
            })
            
            open_position = None

    return pd.DataFrame(trades_data)


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

    # Drop the temporary date column from the original trades_data dataframe
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