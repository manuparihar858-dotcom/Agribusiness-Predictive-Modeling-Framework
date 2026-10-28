"""Train and compare baseline regression models for agricultural yield forecasting.

The included CSV is synthetic demonstration data, not real agricultural observations.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_pipeline(model, numeric_features, categorical_features, scale_numeric=False):
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))
    numeric_pipe = Pipeline(numeric_steps)

    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipe, numeric_features),
        ("categorical", categorical_pipe, categorical_features),
    ], remainder="drop")

    return Pipeline([("preprocess", preprocessor), ("model", model)])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/sample_agricultural_yields.csv")
    parser.add_argument("--target", default="yield_tonnes_per_hectare")
    parser.add_argument("--test-years", type=int, default=2,
                        help="Number of latest seasons held out for testing.")
    args = parser.parse_args()

    path = Path(args.data)
    df = pd.read_csv(path)
    if args.target not in df.columns:
        raise ValueError(f"Target '{args.target}' not found. Available columns: {list(df.columns)}")

    df = df.dropna(subset=[args.target]).copy()
    if len(df) < 10:
        raise ValueError("Dataset is too small for a meaningful demonstration.")

    # Keep the latest seasons for testing when season_year is available.
    if "season_year" in df.columns:
        years = sorted(df["season_year"].dropna().unique())
        if len(years) < 3:
            raise ValueError("At least three distinct season_year values are needed for a chronological split.")
        n_test = min(args.test_years, max(1, len(years) - 2))
        test_years = set(years[-n_test:])
        train_df = df[~df["season_year"].isin(test_years)].copy()
        test_df = df[df["season_year"].isin(test_years)].copy()
        # Season year is retained as a feature only if it would be known at forecast time.
    else:
        # Fallback only for datasets without a time column; sort by row order and do not shuffle.
        split = int(len(df) * 0.8)
        train_df, test_df = df.iloc[:split].copy(), df.iloc[split:].copy()

    if train_df.empty or test_df.empty:
        raise ValueError("Train/test split produced an empty partition.")

    excluded = {args.target}
    # IDs can cause memorization and are not treated as predictive features.
    excluded.update(c for c in ["record_id", "farm_id"] if c in df.columns)
    features = [c for c in df.columns if c not in excluded]
    X_train, y_train = train_df[features], train_df[args.target].astype(float)
    X_test, y_test = test_df[features], test_df[args.target].astype(float)

    categorical_features = X_train.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    numeric_features = [c for c in features if c not in categorical_features]

    models = {
        "Mean baseline": (DummyRegressor(strategy="mean"), False),
        "Linear regression": (LinearRegression(), True),
        "Random forest": (RandomForestRegressor(
            n_estimators=250, min_samples_leaf=2, random_state=42, n_jobs=-1
        ), False),
        "Histogram gradient boosting": (HistGradientBoostingRegressor(
            max_iter=150, learning_rate=0.08, l2_regularization=1.0, random_state=42
        ), False),
    }

    rows = []
    for name, (estimator, scale_numeric) in models.items():
        pipeline = build_pipeline(estimator, numeric_features, categorical_features, scale_numeric)
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)
        rows.append({
            "model": name,
            "MAE": mean_absolute_error(y_test, predictions),
            "RMSE": np.sqrt(mean_squared_error(y_test, predictions)),
            "R2": r2_score(y_test, predictions),
        })

    results = pd.DataFrame(rows).sort_values("MAE")
    print(f"Data file: {path}")
    print(f"Training rows: {len(train_df)} | Test rows: {len(test_df)}")
    if "season_year" in df.columns:
        print(f"Test seasons: {sorted(test_years)}")
    print("\nTest-set metrics (synthetic sample results are illustrative only):")
    print(results.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
