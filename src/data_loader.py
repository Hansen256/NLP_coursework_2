from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd


COMMON_TEXT_COLUMN_NAMES = [
    "text",
    "tweet",
    "tweet_text",
    "content",
    "body",
    "message",
]


def _score_text_column(series: pd.Series) -> float:
    non_null = series.dropna().astype(str).str.strip()
    non_empty = non_null[non_null != ""]
    if non_empty.empty:
        return 0.0

    avg_length = float(non_empty.str.len().mean())
    fill_ratio = float(len(non_empty) / len(series))
    return (avg_length * 0.8) + (fill_ratio * 20.0)


def _detect_text_column(df: pd.DataFrame) -> str:
    lower_map = {col.lower(): col for col in df.columns}
    for candidate in COMMON_TEXT_COLUMN_NAMES:
        if candidate in lower_map:
            return lower_map[candidate]

    candidate_cols: list[str] = []
    for col in df.columns:
        if pd.api.types.is_string_dtype(df[col]) or pd.api.types.is_object_dtype(df[col]):
            candidate_cols.append(col)

    if not candidate_cols:
        raise ValueError("No text-like column found in dataset.")

    best_col = max(candidate_cols, key=lambda col: _score_text_column(df[col]))
    return best_col


def _download_kaggle_dataset_csvs(raw_dir: Path, dataset_handle: str) -> list[Path]:
    """Download a Kaggle dataset via kagglehub and copy CSVs into raw_dir."""
    try:
        import kagglehub
    except ImportError as exc:
        raise ImportError(
            "kagglehub is required for auto-download. Install it with `pip install kagglehub`."
        ) from exc

    download_root = Path(kagglehub.dataset_download(dataset_handle))
    csv_files = sorted(download_root.rglob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files found in downloaded dataset path: {download_root}"
        )

    raw_dir.mkdir(parents=True, exist_ok=True)
    copied_files: list[Path] = []
    for csv_path in csv_files:
        target = raw_dir / csv_path.name
        shutil.copy2(csv_path, target)
        copied_files.append(target)

    return copied_files


def load_tweets_dataframe(
    raw_dir: Path,
    minimum_rows: int = 1000,
    sample_size: int = 10000,
    random_state: int = 42,
    auto_download: bool = False,
    dataset_handle: str = "gpreda/covid19_tweets",
) -> tuple[pd.DataFrame, str, Path]:
    """Load a tweet-like CSV dataset, detect text column, and optionally sample rows."""
    csv_files = sorted(raw_dir.rglob("*.csv"))
    if not csv_files and auto_download:
        _download_kaggle_dataset_csvs(raw_dir=raw_dir, dataset_handle=dataset_handle)
        csv_files = sorted(raw_dir.rglob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in raw data directory: {raw_dir}")

    selected_path = csv_files[0]
    df = pd.read_csv(selected_path)
    text_col = _detect_text_column(df)

    # Keep rows with non-empty text in the detected text column.
    text_series = df[text_col].astype(str).str.strip()
    df = df[text_series != ""].copy()

    if len(df) < minimum_rows:
        raise ValueError(
            f"Dataset has {len(df)} rows after cleaning, below minimum required {minimum_rows}."
        )

    if sample_size > 0 and len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=random_state).reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)

    return df, text_col, selected_path
