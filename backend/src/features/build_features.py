from pathlib import Path

import numpy as np
import pandas as pd
from category_encoders import TargetEncoder

WEATHER_PATH = Path("data/weather_cache.parquet")

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

def add_weather_features(df: pd.DataFrame) -> pd.DataFrame:
    """Join hourly weather from the pre-fetched cache on (Origin, FlightDate, dep_hour).
    Falls back to zeros if the cache doesn't exist so training still works without it."""
    if not WEATHER_PATH.exists():
        df = df.copy()
        for col in ["wind_speed", "precipitation", "snow_depth", "temp", "weather_severity"]:
            df[col] = 0
        return df

    weather = pd.read_parquet(WEATHER_PATH)
    weather = weather.rename(columns={
        "wspd": "wind_speed",
        "prcp": "precipitation",
        "snow": "snow_depth",
    })
    weather["FlightDate"] = weather["FlightDate"].astype(str)
    weather["dep_hour"] = weather["dep_hour"].astype(int)

    df = df.copy()
    df["_fd_str"] = pd.to_datetime(df["FlightDate"]).dt.strftime("%Y-%m-%d")
    df["_dh"] = df["dep_hour"].astype(int)

    df = df.merge(
        weather,
        left_on=["Origin", "_fd_str", "_dh"],
        right_on=["Origin", "FlightDate", "dep_hour"],
        how="left",
        suffixes=("", "_w"),
    )
    df = df.drop(columns=["_fd_str", "_dh", "FlightDate_w", "dep_hour_w"], errors="ignore")

    for col in ["wind_speed", "precipitation", "snow_depth", "temp", "weather_severity"]:
        df[col] = df[col].fillna(0)

    df["weather_severity"] = df["weather_severity"].astype("int8")
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
    "dep_hour", "dep_minutes", "day_of_week", "month", "seasons",
    "is_weekend", "is_peak_evening", "is_early_morning",
    "distance_bin",
    "Reporting_Airline_mean_delay", "route_mean_delay", "dep_hour_mean_delay",
    "wind_speed", "precipitation", "snow_depth", "temp", "weather_severity",
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
    df = add_weather_features(df)
    if encoders is None:
        if y is None:
            raise ValueError("y must be provided when fitting encoders (training mode)")
        encoders = fit_target_encoders(df, y)
    df = apply_target_encoders(df, encoders)
    return df[FEATURE_COLS], encoders