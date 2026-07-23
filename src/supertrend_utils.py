"""
Utilities for calculating the Supertrend indicator.

The implementation follows the standard Supertrend construction:

    1. True Range
    2. Wilder's Average True Range
    3. Basic upper and lower bands
    4. Final upper and lower bands
    5. Trend state
    6. Supertrend line

All calculations are causal: the value at timestamp t depends only on
OHLC data available at or before timestamp t.
"""
from typing import Tuple, Literal
import numpy as np
import pandas as pd
from numba import njit



def calculate_average_true_range(
    ohlcv_data: pd.DataFrame,
    atr_period: int,
    smoothing_type: Literal["RMA", "SMA", "EMA"] = "RMA",
    ema_span: float = None,
) -> Tuple[pd.Series, pd.Series]:
    """
    Calculate True Range and Wilder's Average True Range.

    The True Range at time t is defined as:

        TR_t = max(
            high_t - low_t,
            |high_t - close_{t-1}|,
            |low_t - close_{t-1}|
        )

    Wilder's ATR is initialized using the arithmetic mean of the first
    `atr_period` True Range observations and subsequently updated as:

        ATR_t =
            ((atr_period - 1) * ATR_{t-1} + TR_t)
            / atr_period

    Parameters
    ----------
    ohlcv_data : pd.DataFrame
        OHLCV data containing the columns:

            - "high"
            - "low"
            - "close"

        The DataFrame should be chronologically sorted.

    atr_period : int
        Number of observations used for ATR calculation.

    smoothing_type : {"RMA", "SMA", "EMA"}, default="RMA"
        The moving average type used to smooth the True Range.

    Returns
    -------
    Tuple[pd.Series, pd.Series]
        A tuple containing:

            true_range :
                True Range for each observation.

            average_true_range :
                Wilder's Average True Range.

    Raises
    ------
    ValueError
        If `atr_period` is not a positive integer.

    KeyError
        If the required OHLC columns are missing.
    """

    if not isinstance(atr_period, int):
        raise TypeError(
            "atr_period must be an integer."
        )

    if atr_period <= 0:
        raise ValueError(
            "atr_period must be greater than zero."
        )

    required_columns = {
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
            "Missing required columns: "
            f"{sorted(missing_columns)}"
        )

    previous_close = (
        ohlcv_data["close"].shift(1)
    )

    true_range = pd.concat(
        [
            (
                ohlcv_data["high"]
                - ohlcv_data["low"]
            ),

            (
                ohlcv_data["high"]
                - previous_close
            ).abs(),

            (
                ohlcv_data["low"]
                - previous_close
            ).abs(),
        ],
        axis=1,
    ).max(axis=1)

    if smoothing_type == "RMA":
        # Wilder's Smoothing (Exponential with alpha = 1/period)
        average_true_range = true_range.ewm(alpha=1/atr_period, adjust=False).mean()
    elif smoothing_type == "SMA":
        # Simple Moving Average
        average_true_range = true_range.rolling(window=atr_period).mean()
    elif smoothing_type == "EMA":
        # Standard Exponential Moving Average
        if ema_span == None:
            average_true_range = true_range.ewm(span=atr_period, adjust=False).mean()
        else:
            average_true_range = true_range.ewm(span=ema_span, adjust=False).mean()
    else:
        raise ValueError("smoothing_type must be 'RMA', 'SMA', or 'EMA'")

    return (
        true_range,
        average_true_range,
    )




@njit(cache=True)
def _calculate_supertrend_bands_core(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    average_true_range_array: np.ndarray,
    multiplier: float,
    number_of_observations: int,
    first_valid: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Core Numba-compiled recursive calculation for Supertrend bands.
    Runs at C-speed outside of the global interpreter lock (GIL).
    """
    midpoint = (high + low) / 2.0
    basic_upper_band = midpoint + multiplier * average_true_range_array
    basic_lower_band = midpoint - multiplier * average_true_range_array

    final_upper_band = np.full(number_of_observations, np.nan, dtype=np.float64)
    final_lower_band = np.full(number_of_observations, np.nan, dtype=np.float64)
    supertrend = np.full(number_of_observations, np.nan, dtype=np.float64)
    trend = np.full(number_of_observations, np.nan, dtype=np.float64)

    # Initialisation
    final_upper_band[first_valid] = basic_upper_band[first_valid]
    final_lower_band[first_valid] = basic_lower_band[first_valid]

    if close[first_valid] <= final_upper_band[first_valid]:
        trend[first_valid] = -1.0
        supertrend[first_valid] = final_upper_band[first_valid]
    else:
        trend[first_valid] = 1.0
        supertrend[first_valid] = final_lower_band[first_valid]

    # Recursive calculation
    for observation_index in range(first_valid + 1, number_of_observations):
        previous_index = observation_index - 1

        # Final upper band
        if (basic_upper_band[observation_index] < final_upper_band[previous_index]) or \
           (close[previous_index] > final_upper_band[previous_index]):
            final_upper_band[observation_index] = basic_upper_band[observation_index]
        else:
            final_upper_band[observation_index] = final_upper_band[previous_index]

        # Final lower band
        if (basic_lower_band[observation_index] > final_lower_band[previous_index]) or \
           (close[previous_index] < final_lower_band[previous_index]):
            final_lower_band[observation_index] = basic_lower_band[observation_index]
        else:
            final_lower_band[observation_index] = final_lower_band[previous_index]

        # Trend transition
        if trend[previous_index] == -1.0:
            if close[observation_index] > final_upper_band[observation_index]:
                trend[observation_index] = 1.0
                supertrend[observation_index] = final_lower_band[observation_index]
            else:
                trend[observation_index] = -1.0
                supertrend[observation_index] = final_upper_band[observation_index]
        else:
            if close[observation_index] < final_lower_band[observation_index]:
                trend[observation_index] = -1.0
                supertrend[observation_index] = final_upper_band[observation_index]
            else:
                trend[observation_index] = 1.0
                supertrend[observation_index] = final_lower_band[observation_index]

    return (
        basic_upper_band,
        basic_lower_band,
        final_upper_band,
        final_lower_band,
        supertrend,
        trend,
    )


def calculate_supertrend_bands(
    ohlcv_data: pd.DataFrame,
    average_true_range: pd.Series,
    multiplier: float,
) -> Tuple[pd.Series, pd.Series, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate the basic and final Supertrend bands.

    The midpoint is:
        HL2_t = (high_t + low_t) / 2

    The basic bands are:
        BUB_t = HL2_t + multiplier * ATR_t
        BLB_t = HL2_t - multiplier * ATR_t

    The final upper band is recursive:
        FUB_t = BUB_t if BUB_t < FUB_{t-1} or close_{t-1} > FUB_{t-1} else FUB_{t-1}

    The final lower band is recursive:
        FLB_t = BLB_t if BLB_t > FLB_{t-1} or close_{t-1} < FLB_{t-1} else FLB_{t-1}

    Parameters
    ----------
    ohlcv_data : pd.DataFrame
        OHLCV data containing "high", "low", and "close".

    average_true_range : pd.Series
        Wilder's Average True Range, aligned with `ohlcv_data`.

    multiplier : float
        ATR multiplier used to determine the distance of the bands
        from the midpoint.

    Returns
    -------
    Tuple[pd.Series, pd.Series, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        Returns raw upper band, raw lower band, final upper band, final lower band,
        supertrend line, and trend state (-1.0 or +1.0).

    Raises
    ------
    ValueError
        If no valid ATR observations are available, or multiplier is invalid.
    """

    if multiplier <= 0:
        raise ValueError("multiplier must be greater than zero.")

    high_arr = ohlcv_data["high"].to_numpy(dtype=np.float64)
    low_arr = ohlcv_data["low"].to_numpy(dtype=np.float64)
    close_arr = ohlcv_data["close"].to_numpy(dtype=np.float64)
    average_true_range_arr = average_true_range.to_numpy(dtype=np.float64)
    number_of_observations = len(close_arr)

    valid_indices = np.flatnonzero(~np.isnan(average_true_range_arr))
    if len(valid_indices) == 0:
        raise ValueError("No valid ATR observations. Check the input data and atr_period.")
    
    first_valid = valid_indices[0]

    # Pass entirely to the Numba JIT compiler
    return _calculate_supertrend_bands_core(
        high=high_arr,
        low=low_arr,
        close=close_arr,
        average_true_range_array=average_true_range_arr,
        multiplier=multiplier,
        number_of_observations=number_of_observations,
        first_valid=first_valid,
    )



def generate_supertrend(
    ohlcv_data: pd.DataFrame,
    atr_period: int = 10,
    multiplier: float = 3.0,
    smoothing_type: Literal["RMA", "SMA", "EMA"] = "RMA"
) -> pd.DataFrame:
    """
    Master pipeline function to calculate the Supertrend indicator.

    This function sequentially calculates the True Range, applies the 
    specified smoothing to generate the Average True Range (ATR), computes 
    the recursive bands via the Numba JIT-compiled core, and appends all 
    resulting columns to the DataFrame.

    Parameters
    ----------
    ohlcv_data : pd.DataFrame
        OHLCV data containing "high", "low", and "close".
    atr_period : int, default=10
        Lookback window for the Average True Range.
    multiplier : float, default=3.0
        ATR multiplier used to determine the distance of the bands.
    smoothing_type : {"RMA", "SMA", "EMA"}, default="RMA"
        The moving average type used to smooth the True Range.

    Returns
    -------
    pd.DataFrame
        A copy of the input DataFrame appended with:
            - "true_range", "atr"
            - "basic_upper_band", "basic_lower_band"
            - "final_upper_band", "final_lower_band"
            - "supertrend", "trend"
    """
    
    # 1. Calculate True Range and ATR
    true_range, average_true_range = calculate_average_true_range(
        ohlcv_data=ohlcv_data,
        atr_period=atr_period,
        smoothing_type=smoothing_type
    )
    
    # 2. Calculate the recursive Supertrend Bands and Regime
    (
        basic_upper_band,
        basic_lower_band,
        final_upper_band,
        final_lower_band,
        supertrend,
        trend,
    ) = calculate_supertrend_bands(
        ohlcv_data=ohlcv_data,
        average_true_range=average_true_range,
        multiplier=multiplier,
    )
    
    # 3. Create a copy to prevent SettingWithCopy warnings
    supertrend_data = ohlcv_data.copy()
    
    # 4. Append calculated quantities
    supertrend_data["true_range"] = true_range
    supertrend_data["atr"] = average_true_range
    supertrend_data["basic_upper_band"] = basic_upper_band
    supertrend_data["basic_lower_band"] = basic_lower_band
    supertrend_data["final_upper_band"] = final_upper_band
    supertrend_data["final_lower_band"] = final_lower_band
    supertrend_data["supertrend"] = supertrend
    supertrend_data["trend"] = trend
    
    return supertrend_data