import pandas as pd
import numpy as np

from pathlib import Path
from typing import List, Sequence

# ============================================================
# Project paths
# ============================================================

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]

DATA_DIR: Path = PROJECT_ROOT / "data"


# ============================================================
# Constants
# ============================================================

REQUIRED_COLUMNS: set[str] = {
    "time",
    "open",
    "high",
    "low",
    "close",
}


# ============================================================
# Functions
# ============================================================


def list_ohlcv_files(data_dir: Path = DATA_DIR) -> List[Path]:
    """
    Discover all CSV files containing OHLCV data.

    Parameters
    ----------
    data_dir : Path, default=DATA_DIR
        Directory containing the OHLCV CSV files.

    Returns
    -------
    List[Path]
        Sorted list of paths to CSV files.

    Raises
    ------
    FileNotFoundError
        If `data_dir` does not exist.

    ValueError
        If no CSV files are found.
    """

    if not data_dir.exists():
        raise FileNotFoundError(
            f"Data directory does not exist: {data_dir}"
        )

    if not data_dir.is_dir():
        raise NotADirectoryError(
            f"Expected a directory, got: {data_dir}"
        )

    csv_files = sorted(
        data_dir.glob("*.csv")
    )

    if not csv_files:
        raise ValueError(
            f"No CSV files found in: {data_dir}"
        )

    return csv_files


def load_ohlcv_file(filepath: Path) -> pd.DataFrame:
    """
    Load a single OHLCV CSV file and standardise its column names.

    Column names are:
        1. stripped of leading/trailing whitespace;
        2. converted to lowercase.

    Parameters
    ----------
    filepath : Path
        Path to the CSV file.

    Returns
    -------
    pd.DataFrame
        Loaded OHLCV data with standardised column names.

    Raises
    ------
    FileNotFoundError
        If `filepath` does not exist.

    ValueError
        If the file is empty.
    """

    if not filepath.exists():
        raise FileNotFoundError(
            f"File does not exist: {filepath}"
        )

    ohlcv_df = pd.read_csv(filepath)

    if ohlcv_df.empty:
        raise ValueError(
            f"CSV file is empty: {filepath}"
        )

    ohlcv_df.columns = (
        ohlcv_df.columns
        .str.strip()
        .str.lower()
    )

    return ohlcv_df

def validate_ohlcv_schema(data: pd.DataFrame) -> None:
    """
    Validate that the DataFrame contains all required OHLCV columns.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame whose schema should be validated.

    Raises
    ------
    ValueError
        If one or more required columns are missing.
    """

    missing_columns = (
        REQUIRED_COLUMNS
        - set(data.columns)
    )

    if missing_columns:
        raise ValueError(
            "Missing required columns: "
            f"{sorted(missing_columns)}"
        )

def load_ohlcv_data(filepaths: Sequence[Path]) -> pd.DataFrame:
    """
    Load, concatenate, validate schema, and chronologically sort
    multiple OHLCV CSV files.

    Processing steps
    ----------------
    1. Load each CSV file (column standardization happens in load_ohlcv_file).
    2. Validate the required schema.
    3. Add the source filename.
    4. Concatenate all files.
    5. Parse timestamps and validate.
    6. Sort chronologically and set timestamp index.

    Parameters
    ----------
    filepaths : Sequence[Path]
        Paths to the OHLCV CSV files.

    Returns
    -------
    pd.DataFrame
        Chronologically sorted OHLCV data indexed by timestamp.

    Raises
    ------
    ValueError
        If no files are provided or invalid timestamps are encountered.
    """

    if not filepaths:
        raise ValueError("No filepaths were provided.")

    ohlcv_dfs: list[pd.DataFrame] = []

    for filepath in filepaths:
        # load_ohlcv_file handles reading, empty checks, and column standardization
        ohlcv_df = load_ohlcv_file(filepath)

        validate_ohlcv_schema(ohlcv_df)

        ohlcv_df["source_file"] = filepath.name

        ohlcv_dfs.append(ohlcv_df)

    ohlcv_data = pd.concat(ohlcv_dfs, ignore_index=True)

    # Parse timestamps
    ohlcv_data["time"] = pd.to_datetime(ohlcv_data["time"], errors="coerce")

    invalid_timestamps = ohlcv_data["time"].isna()
    if invalid_timestamps.any():
        n_invalid = invalid_timestamps.sum()
        raise ValueError(f"Found {n_invalid} invalid timestamps.")

    # Sort chronologically
    ohlcv_data = ohlcv_data.sort_values("time").set_index("time")

    return ohlcv_data



def validate_numeric_columns(
    data: pd.DataFrame,
) -> None:
    """
    Validate that OHLC columns contain finite numerical values.

    Parameters
    ----------
    data : pd.DataFrame
        OHLCV DataFrame.

    Raises
    ------
    ValueError
        If OHLC values are missing, non-finite, or non-positive.
    """

    ohlc_columns = [
        "open",
        "high",
        "low",
        "close",
    ]

    for column in ohlc_columns:

        if not pd.api.types.is_numeric_dtype(
            data[column]
        ):
            raise TypeError(
                f"Column '{column}' "
                "must be numeric."
            )

        if not np.isfinite(
            data[column]
        ).all():

            raise ValueError(
                f"Column '{column}' "
                "contains NaN or infinite values."
            )

        if (
            data[column] <= 0
        ).any():

            raise ValueError(
                f"Column '{column}' "
                "contains non-positive values."
            )


def validate_ohlc_relationships(
    data: pd.DataFrame,
) -> None:
    """
    Validate internal consistency of OHLC prices.

    The following conditions must hold:

        high >= max(open, close)

        low <= min(open, close)

        high >= low

    Parameters
    ----------
    data : pd.DataFrame
        OHLCV DataFrame.

    Raises
    ------
    ValueError
        If any OHLC relationship is violated.
    """

    invalid_high = (
        data["high"]
        <
        data[["open", "close"]]
        .max(axis=1)
    )

    invalid_low = (
        data["low"]
        >
        data[["open", "close"]]
        .min(axis=1)
    )

    invalid_range = (
        data["high"]
        <
        data["low"]
    )

    if invalid_high.any():
        raise ValueError(
            "Found rows where high < "
            "max(open, close)."
        )

    if invalid_low.any():
        raise ValueError(
            "Found rows where low > "
            "min(open, close)."
        )

    if invalid_range.any():
        raise ValueError(
            "Found rows where high < low."
        )


def analyse_timestamp_structure(
    data: pd.DataFrame,
) -> None:
    """
    Analyse the temporal structure of the OHLCV data.

    Reports:
        - starting timestamp;
        - ending timestamp;
        - duplicate timestamps;
        - distribution of time differences;
        - observations per year.

    Parameters
    ----------
    data : pd.DataFrame
        OHLCV DataFrame indexed by timestamp.
    """

    print(
        "Starting timestamp:",
        data.index.min(),
    )

    print(
        "Ending timestamp:",
        data.index.max(),
    )

    print(
        "Duplicate timestamps:",
        data.index.duplicated().sum(),
    )

    print(
        "\nMost common time differences:"
    )

    print(
        data.index
        .to_series()
        .diff()
        .value_counts()
        .head(20)
    )

    print(
        "\nObservations per year:"
    )

    print(
        data.groupby(
            data.index.year
        ).size()
    )



def analyse_ohlcv_data(ohlcv_data: pd.DataFrame) -> None:
    """
    Print a data-quality report for OHLCV data.

    The report includes:

        - DataFrame structure and memory usage;
        - descriptive statistics;
        - missing-value counts;

    Parameters
    ----------
    ohlcv_data : pd.DataFrame
        OHLCV data indexed by timestamp.
    """

    print("DATAFRAME INFORMATION")
    print("=" * 60)

    ohlcv_data.info()

    print("\nDESCRIPTIVE STATISTICS")
    print("=" * 60)

    print(
        ohlcv_data[
            [
                "open",
                "high",
                "low",
                "close",
            ]
        ].describe()
    )

    print("\nMISSING VALUES")
    print("=" * 60)

    print(
        ohlcv_data.isna().sum()
    )
    


def run_data_pipeline(
    data_dir: Path = DATA_DIR,
    print_report: bool = True
) -> pd.DataFrame:
    """
    Master pipeline function to discover, load, strictly validate, 
    and analyze all OHLCV data within a target directory.

    Parameters
    ----------
    data_dir : Path, default=DATA_DIR
        Directory containing the OHLCV CSV files.
    print_report : bool, default=True
        Whether to print the standalone data quality reports.

    Returns
    -------
    pd.DataFrame
        A strictly validated, chronologically sorted OHLCV DataFrame.
    """
    
    # 1. Discover and Load
    csv_filepaths = list_ohlcv_files(data_dir=data_dir)
    ohlcv_data = load_ohlcv_data(filepaths=csv_filepaths)
    
    # 2. Strict Validation (Will raise errors and halt if data is corrupt)
    validate_numeric_columns(data=ohlcv_data)
    validate_ohlc_relationships(data=ohlcv_data)
    
    # 3. Standalone Analysis
    if print_report:
        analyse_ohlcv_data(ohlcv_data=ohlcv_data)
        print("\n")
        analyse_timestamp_structure(data=ohlcv_data)
        
    return ohlcv_data
    


def select_time_window(
    ohlcv_data: pd.DataFrame,
    start: str,
    end: str,
) -> pd.DataFrame:
    """
    Select a time interval from an OHLCV DataFrame.

    Parameters
    ----------
    ohlcv_data : pd.DataFrame
        OHLCV data indexed by a DatetimeIndex.

    start : str
        Start timestamp accepted by pandas.

    end : str
        End timestamp accepted by pandas.

    Returns
    -------
    pd.DataFrame
        Copy of the data between `start` and `end`, inclusive.

    Raises
    ------
    TypeError
        If the input does not have a DatetimeIndex.
    """

    if not isinstance(
        ohlcv_data.index,
        pd.DatetimeIndex,
    ):
        raise TypeError(
            "ohlcv_data must have a "
            "pandas DatetimeIndex."
        )

    return (
        ohlcv_data
        .loc[start:end]
        .copy()
    )