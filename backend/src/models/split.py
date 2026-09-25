from pathlib import Path
import pandas as pd

DATA_PATH = Path("data/flights_clean.parquet")
TARGET = "is_delayed"
DELAY_THRESHOLD = 15  # minutes -> the standard DOT definition of a delayed departure

#chronological split: train on the past, score on the future (no random shuffling)
TRAIN_MONTHS = range(1, 9) #Jan - Aug
VAL_MONTHS = range(9, 11) #Sep - Oct
TEST_MONTHS = range(11, 13) #Nov - Dec

def load_flights(path: Path = DATA_PATH) -> pd.DataFrame:
    """Load cleaned flights and add columns every model needs"""
    df = pd.read_parquet(path)

    return df