import pandas as pd
import numpy as np
from category_encoders import TargetEncoder

#target encoder uses smoothed encoding (smoothed = 10) on training data
#to prevent leakage  -> encoders are fitted once and serialized alongside the model


def parse_hhmm(series: pd.Series) -> tuple[pd.Series, pd.Series]:
    """convert HHMM int (e.g. 1435) to hour and minute."""
    hour = (series // 100).clip(0, 23).astype("int8")
    minute = (series % 100).clip(0, 59).astype("int8")
    return hour, minute

def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """CRSDepTime is used as data for predicted departure time, which will be the
    given information that the machine learns with"""
    df = df.copy()
    df["dep_hour"], df["dep_minutes"] = parse_hhmm(df["CRSDepTime"])
    df["FlightDate"] = pd.to_datetime(df["FlightDate"])
    df["day_of_week"] = df["FlightDate"].dt.dayofweek.astype("int8")
    df["month"] = df["FlightDate"].dt.month.astype("int8")
    df["is_weekend"] = (df["day_of_week"] >= 5).astype("int8")
    df["is_peak_evening"] = df["dep_hour"].between(19, 23).astype("int8")
    df["is_early_morning"] = df["dep_hour"].between(5, 7).astype("int8")
    season_map = {12:0, 1:0, 2:0, 3:1, 4:1, 5:1, 6:2, 7:2, 8:2, 9:3, 10:3, 11:3}
    df["seasons"] = df["month"].astype(int).map(season_map).astype("int8")

    return df

def add_route_feature(df: pd.DataFrame) -> pd.DataFrame:
    """adding route information (origin - destination)"""
    df = df.copy()
    df["route"] = df["Origin"] + "-" + df["Dest"]
    return df

def add_distance_bin(df: pd.DataFrame, bins=10) -> pd.DataFrame:
    df = df.copy()
    df["distance_bin"] = pd.cut(df["Distance"], bins = bins, labels = False).astype("int8")
    return df 

def fit_target_encoders(
        X_train: pd.DataFrame, y_train: pd.Series
) -> dict[str, TargetEncoder]: 
    """fit one encoder per categorical column and call it once on training data"""
    encoders = {} #empty dic
    for col in ["Reporting_Airline", "route", "dep_hour"]: #loop through high-cardinality columns
        enc = TargetEncoder(cols=[col], smoothing=10) # create new encoder instance for current column
        enc.fit(X_train[[col]], y_train) #encoder looks at the column in X_train and matches it against outcomes in y_train to calculate and memorize avg
        #double brackets [[col]] forces DataFrame instead of series
        encoders[col] = enc # fully trained encoder saved into dictionary locker, labeled by column name
    return encoders

def apply_target_encoders(
        df: pd.DataFrame, encoders: dict[str, TargetEncoder]
) -> pd.DataFrame:
    df = df.copy()
    for col, enc in encoders.items(): #items()=dict method that returns (key, value) list
        encoded = enc.transform(df[[col]]) #checks internal memory from .fit() and then outputs the number and if can't be found fills with overal gloabl average 
        df[f"{col}_mean_delay"] = encoded[col]
    return df

FEATURE_COLS = [
    "dep_hour", "dep_minute", "day_of_week", "month", "season", 
    "is_weekend", "is_peak_evening", "is_early_morning",
    "distance_bin",
    "Reporting_Airline_mean_delay", "route_mean_delay", "dep_hour_mean_delay",
]

def build_features(
        df: pd.DataFrame,
        encoders: dict | None = None,
        y: pd.Series | None = None,
) -> tuple[pd.DataFrame, dict]:
    """
    Full pipeline of passing y and encoders = None on the training data 
    then, passsing fitted encoders and y=None on inference data.
    """
    df = add_temporal_features(df)
    df = add_route_feature(df)
    df = add_distance_bin(df)
    if encoders is None:
        if y is None:
            raise ValueError("y must be provided when fitting encoders (training mode)")
        encoders = fit_target_encoders(df, y)
    df = apply_target_encoders(df, encoders)
    return df[FEATURE_COLS], encoders