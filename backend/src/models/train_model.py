import joblib
import numpy as np
import pandas as pd
import shap
import lightgbm as lgb
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from loguru import logger

from backend.src.features.build_features import build_features, FEATURE_COLS


DATA_PATH = Path("data/flights_clean.parquet")
MODEL_PATH = Path("backend/models/lgbm_delay.pkl")
ENCODERS_PATH = Path("backend/models/encoders.pkl")
SHAP_PATH = Path("backend/models/shap_values.npy")

def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    return pd.read_parquet(path)

def split_data(df: pd.DataFrame, test_size=0.2, random_state=42):
    return train_test_split(
        df, test_size=test_size, random_state=random_state,
        stratify=df["month"]
    )

def train(df_train: pd.DataFrame) -> tuple:
    y_train = df_train["DepDelay"]
    X_train, encoders = build_features(df_train, encoders=None, y=y_train)
    logger.info(f"Training on {X_train.shape[1]} features: {X_train.columns.tolist()}")
    model = lgb.LGBMRegressor(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=63,
        min_child_samples=50,
        n_jobs=-1,
        random_state=42,
    )
    model.fit(X_train, y_train)
    return model, encoders

def evaluate(model, encoders, df_test: pd.DataFrame) -> dict:
    y_test = df_test["DepDelay"]
    X_test, _ = build_features(df_test, encoders=encoders)
    y_pred = model.predict(X_test)

    metrics = {
        "mae": mean_absolute_error(y_test, y_pred),
        "rmse": mean_squared_error(y_test, y_pred) ** 0.5,
        "r2": r2_score(y_test, y_pred),
    }
    logger.info(f"MAE={metrics['mae']:.2f} RMSE={metrics['rmse']:.2f} R^2={metrics['r2']:.3f}")
    return metrics

def compute_shap(model, X_sample: pd.DataFrame) -> np.ndarray:
    explainer = shap.TreeExplainer(model)
    return explainer.shap_values(X_sample)

def save_artifacts(model, encoders, shap_values):
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(encoders, ENCODERS_PATH)
    np.save(SHAP_PATH, shap_values)
    logger.info(f"Saved model to {MODEL_PATH}")


def main():
    logger.info("Loading data....")
    df = load_data()

    logger.info("Splitting....")
    df_train, df_test = split_data(df)

    logger.info("Training....")
    model, encoders = train(df_train)

    logger.info("Evaluating....")
    metrics = evaluate(model, encoders, df_test)

    logger.info("Computing SHAP on 5k sample ....")
    X_sample, _ = build_features(df_test.sample(5000, random_state=42), encoders=encoders)
    shap_values = compute_shap(model, X_sample)

    save_artifacts(model, encoders, shap_values)
    return metrics

if __name__ == "__main__":
    main()
