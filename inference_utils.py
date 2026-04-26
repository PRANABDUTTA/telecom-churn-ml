"""Feature table for model input — matches the notebook preprocessing (section 5)."""

from __future__ import annotations

import pandas as pd


def clean_training_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Add Churn_binary and TotalCharges_clean like the notebook."""
    out = df.copy()
    if "Churn" in out.columns and "Churn_binary" not in out.columns:
        out["Churn_binary"] = out["Churn"].map({"Yes": 1, "No": 0})
    out["TotalCharges_clean"] = pd.to_numeric(out["TotalCharges"], errors="coerce")
    mask_na = out["TotalCharges_clean"].isna()
    out.loc[mask_na, "TotalCharges_clean"] = out.loc[mask_na, "tenure"] * out.loc[mask_na, "MonthlyCharges"]
    still_na = out["TotalCharges_clean"].isna()
    if still_na.any():
        out.loc[still_na, "TotalCharges_clean"] = out["TotalCharges_clean"].median()
    return out


def dataframe_to_model_X(df: pd.DataFrame) -> pd.DataFrame:
    """One or more rows, same columns as sklearn `X` before ColumnTransformer."""
    d = df.copy()
    if "TotalCharges_clean" not in d.columns:
        d = clean_training_dataframe(d)
    X = d.drop(columns=["customerID", "Churn", "Churn_binary", "TotalCharges"], errors="ignore")
    X = X.copy()
    X["TotalCharges"] = d["TotalCharges_clean"].values
    return X.reset_index(drop=True)


def form_defaults_from_X(X: pd.DataFrame) -> dict:
    """Median for numeric columns, mode for categorical — for Streamlit defaults."""
    defaults: dict = {}
    for col in X.columns:
        s = X[col]
        if pd.api.types.is_numeric_dtype(s):
            defaults[col] = float(s.median())
        else:
            mode = s.mode(dropna=True)
            defaults[col] = str(mode.iloc[0]) if len(mode) else ""
    return defaults
