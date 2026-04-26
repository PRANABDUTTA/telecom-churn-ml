"""
Train Logistic Regression + preprocessor (same settings as the notebook) and save
artifacts under ./artifacts for Streamlit / Streamlit Cloud.

Run from repo root:
  python train_export.py

If your notebook champion is not Logistic Regression, export from the notebook
(section 8) instead and commit those files.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from inference_utils import clean_training_dataframe, dataframe_to_model_X, form_defaults_from_X


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("Telco_Customer_Churn.csv"),
        help="Path to Telco_Customer_Churn.csv",
    )
    args = parser.parse_args()
    if not args.csv.is_file():
        raise SystemExit(f"CSV not found: {args.csv.resolve()}")

    df = pd.read_csv(args.csv)
    df = clean_training_dataframe(df)
    X = dataframe_to_model_X(df)
    y = df["Churn_binary"]

    numeric_features = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]
    categorical_features = [c for c in X.columns if c not in numeric_features]

    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", ohe),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    X_train_p = preprocessor.fit_transform(X_train)
    model = LogisticRegression(max_iter=2000, random_state=42)
    model.fit(X_train_p, y_train)

    out = Path("artifacts")
    out.mkdir(exist_ok=True)
    joblib.dump(preprocessor, out / "preprocessor.joblib")
    joblib.dump(model, out / "model.joblib")
    (out / "model_name.txt").write_text("Logistic Regression", encoding="utf-8")
    joblib.dump(form_defaults_from_X(X_train), out / "form_defaults.joblib")

    print("Saved:", out / "preprocessor.joblib")
    print("Saved:", out / "model.joblib")
    print("Saved:", out / "model_name.txt")
    print("Saved:", out / "form_defaults.joblib")


if __name__ == "__main__":
    main()
