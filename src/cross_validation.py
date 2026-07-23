import inspect
import itertools
from typing import Dict, List, Tuple, Any
import pandas as pd
import numpy as np

# Import core functionalities from your other modules
from src.supertrend_utils import generate_supertrend
from src.strategy_engine import (
    generate_signals,
    backtest_strategy,
    calculate_performance
)

# =====================================================================
# 1. Window Generator
# =====================================================================
def create_walk_forward_windows(
    ohlcv_data: pd.DataFrame,
    start_index: int,
    validation_size: int,
    test_size: int
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Slices the OHLCV dataframe into discrete validation and test windows.
    
    Parameters
    ----------
    ohlcv_data : pd.DataFrame
        The full dataset.
    start_index : int
        The row index where the validation window begins.
    validation_size : int
        The number of rows in the validation window.
    test_size : int
        The number of rows in the out-of-sample test window.
        
    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        The validation slice and the test slice.
    """
    val_end = start_index + validation_size
    test_end = val_end + test_size
    
    val_data = ohlcv_data.iloc[start_index:val_end].copy()
    test_data = ohlcv_data.iloc[val_end:test_end].copy()
    
    return val_data, test_data


# =====================================================================
# 2. Parameter Optimizer & Test Evaluator
# =====================================================================
def optimize_and_test_window(
    val_data: pd.DataFrame,
    test_data: pd.DataFrame,
    param_grid: Dict[str, list],
    target_metric: str = "total_pnl",
    test_initial_capital: float = 100000.0,
    val_initial_capital: float = 100000.0
) -> Tuple[Dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, float]]:
    """
    Runs a grid search over the validation window to find optimal parameters,
    then evaluates those parameters on the test window.
    
    To prevent indicators (like Supertrend ATR and Volume MA) from resetting 
    and returning NaNs at the start of the test window, the final test 
    calculations are evaluated on a combined dataset before slicing out the 
    out-of-sample trades.
    """
    
    # 1. Dynamically parse required arguments for each function
    st_keys = set(inspect.signature(generate_supertrend).parameters.keys())
    sig_keys = set(inspect.signature(generate_signals).parameters.keys())
    bt_keys = set(inspect.signature(backtest_strategy).parameters.keys())
    
    keys, values = zip(*param_grid.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    best_metric = -np.inf
    best_params = {}
    
    # 2. Grid Search on Validation Data
    for params in combinations:
        # Route parameters to their respective functions
        st_params = {k: v for k, v in params.items() if k in st_keys}
        sig_params = {k: v for k, v in params.items() if k in sig_keys}
        bt_params = {k: v for k, v in params.items() if k in bt_keys}
        
        # Validation Execution
        val_st = generate_supertrend(val_data, **st_params)
        val_signals = generate_signals(val_st, **sig_params)
        val_trades = backtest_strategy(val_signals, initial_capital=val_initial_capital, **bt_params)
        val_metrics, _ = calculate_performance(val_trades, initial_capital=val_initial_capital)
        
        # Handle cases where no trades were generated
        current_score = val_metrics.get(target_metric, -np.inf)
        
        if current_score > best_metric:
            best_metric = current_score
            best_params = params
            
    # Fallback if no combinations produced trades
    if not best_params:
        best_params = combinations[0] 
        
    # 3. Out-of-Sample Test Execution
    # Route the BEST parameters
    best_st_params = {k: v for k, v in best_params.items() if k in st_keys}
    best_sig_params = {k: v for k, v in best_params.items() if k in sig_keys}
    best_bt_params = {k: v for k, v in best_params.items() if k in bt_keys}
    
    # Concatenate val and test to warm up moving averages and trailing bands
    combined_data = pd.concat([val_data, test_data])
    
    # Generate continuous indicators and signals across the boundary
    combined_st = generate_supertrend(combined_data, **best_st_params)
    combined_signals = generate_signals(combined_st, **best_sig_params)
    
    # Slice out strictly the out-of-sample data for backtesting
    test_signals = combined_signals.loc[test_data.index]
    test_st = combined_st.loc[test_data.index]
    
    # Backtest isolated test set, inheriting the running capital from the WFO engine
    test_trades = backtest_strategy(test_signals, initial_capital=test_initial_capital, **best_bt_params)
    test_metrics, _ = calculate_performance(test_trades, initial_capital=test_initial_capital)
    
    return best_params, test_st, test_signals, test_trades, test_metrics


# =====================================================================
# 3. Walk-Forward Orchestrator
# =====================================================================
def run_walk_forward_cross_validation(
    ohlcv_data: pd.DataFrame,
    validation_size: int,
    test_size: int,
    param_grid: Dict[str, list],
    target_metric: str = "total_pnl",
    initial_capital: float = 100000.0,
    start_index: int = 0
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, float], List[Dict[str, Any]]]:
    """
    Executes walk-forward optimization across the entire dataset.
    
    Parameters
    ----------
    ohlcv_data : pd.DataFrame
        The full dataset.
    validation_size : int
        Size (in rows) of the rolling validation window.
    test_size : int
        Size (in rows) of the out-of-sample test window.
    param_grid : Dict[str, list]
        Dictionary of parameters to optimize.
    target_metric : str, default="total_pnl"
        The KPI metric to maximize during validation.
    initial_capital : float, default=100000.0
        Starting capital.
    start_index : int, default=0
        Where to begin the first validation window.
        
    Returns
    -------
    Tuple
        - global_test_supertrend: Concatenated Out-of-Sample Supertrend data
        - global_test_signals: Concatenated Out-of-Sample Signal data
        - global_test_trades: Concatenated Out-of-Sample Trades log
        - global_test_metrics: Recalculated WFO Performance metrics
        - optimal_parameters_log: List tracking the best params for each fold
    """
    
    all_test_st = []
    all_test_signals = []
    all_test_trades = []
    optimal_parameters_log = []
    
    current_index = start_index
    total_rows = len(ohlcv_data)
    
    # State tracking to stitch equity curve
    running_capital = initial_capital
    fold_number = 1
    
    while (current_index + validation_size + test_size) <= total_rows:
        print(f"Executing WFO Fold {fold_number}...")
        
        # 1. Create windows
        val_data, test_data = create_walk_forward_windows(
            ohlcv_data=ohlcv_data,
            start_index=current_index,
            validation_size=validation_size,
            test_size=test_size
        )
        
        # 2. Optimize and extract out-of-sample results
        (
            best_params, 
            test_st, 
            test_signals, 
            test_trades, 
            test_metrics
        ) = optimize_and_test_window(
            val_data=val_data,
            test_data=test_data,
            param_grid=param_grid,
            target_metric=target_metric,
            test_initial_capital=running_capital,
            val_initial_capital=initial_capital # Val always resets to base to ensure fair comparison
        )
        
        # Track data
        optimal_parameters_log.append({
            "fold": fold_number,
            "start_time": test_data.index.min(),
            "end_time": test_data.index.max(),
            "best_params": best_params
        })
        
        all_test_st.append(test_st)
        all_test_signals.append(test_signals)
        all_test_trades.append(test_trades)
        
        # Update running capital for the next test fold
        if not test_trades.empty:
            running_capital = test_trades["running_capital"].iloc[-1]
            
        # 3. Shift windows right by the step size (test_size)
        current_index += test_size
        fold_number += 1
        
    # Combine all out-of-sample data globally
    global_test_supertrend = pd.concat(all_test_st)
    global_test_signals = pd.concat(all_test_signals)
    global_test_trades = pd.concat(all_test_trades, ignore_index=True)
    
    # Recalculate ultimate global WFO performance metrics
    global_test_metrics, _ = calculate_performance(
        trades_data=global_test_trades, 
        initial_capital=initial_capital
    )
    
    print("\nWalk-Forward Cross-Validation Complete.")
    return (
        global_test_supertrend, 
        global_test_signals, 
        global_test_trades, 
        global_test_metrics, 
        optimal_parameters_log
    )