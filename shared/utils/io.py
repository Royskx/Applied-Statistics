from pathlib import Path
from typing import Union
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def data_path(name: str) -> Path:
    """Return the absolute path to a dataset in shared/data."""
    return DATA_DIR / name


def load_csv(name: str) -> pd.DataFrame:
    """Load a CSV from shared/data by file name."""
    path = data_path(name)
    return pd.read_csv(path)
