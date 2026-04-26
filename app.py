"""
Streamlit app: Telco customer churn prediction.
Loads preprocessor + classifier from ./artifacts (see notebook section 8 or train_export.py).
"""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from inference_utils import dataframe_to_model_X

ROOT = Path(__file__).resolve().parent
ART = ROOT / "artifacts"


@st.cache_resource
def load_bundle():
    pre = joblib.load(ART / "preprocessor.joblib")
    model = joblib.load(ART / "model.joblib")
    name = (ART / "model_name.txt").read_text(encoding="utf-8").strip()
    defaults = joblib.load(ART / "form_defaults.joblib")
    return pre, model, name, defaults


def predict_batch(pre, model, X: pd.DataFrame):
    Xv = pre.transform(X)
    proba = model.predict_proba(Xv)[:, 1]
    pred = model.predict(Xv)
    return pred, proba


st.set_page_config(page_title="Telco Churn", layout="wide")
st.title("Telco customer churn prediction")

try:
    preprocessor, model, model_name, form_defaults = load_bundle()
except Exception as e:
    st.error(
        "Could not load `artifacts/`. Run **section 8** in the notebook or `python train_export.py`, "
        "then commit the `artifacts/` folder to GitHub."
    )
    st.caption(str(e))
    st.stop()

st.caption(f"Deployed model: **{model_name}**")

tab_csv, tab_form = st.tabs(["Upload CSV", "Single customer (form)"])

with tab_csv:
    st.markdown(
        "Upload a CSV with the same feature columns as `Telco_Customer_Churn.csv` "
        "(`customerID` / `Churn` optional). `TotalCharges` may contain blanks like the raw Kaggle file."
    )
    up = st.file_uploader("CSV file", type=["csv"])
    if up is not None:
        raw = pd.read_csv(up)
        try:
            X = dataframe_to_model_X(raw)
        except Exception as ex:
            st.error(f"Could not build feature table: {ex}")
            st.stop()
        pred, proba = predict_batch(preprocessor, model, X)
        out = raw.copy()
        out["churn_probability"] = proba
        out["predicted_churn"] = ["Yes" if p == 1 else "No" for p in pred]
        st.dataframe(out, use_container_width=True)
        st.download_button(
            "Download predictions CSV",
            data=out.to_csv(index=False).encode("utf-8"),
            file_name="churn_predictions.csv",
            mime="text/csv",
        )

with tab_form:
    st.markdown("Enter one customer (defaults are training-set medians / modes).")
    row = {}
    cols = list(form_defaults.keys())
    n = len(cols)
    for i, col in enumerate(cols):
        default = form_defaults[col]
        if col == "SeniorCitizen":
            row[col] = st.selectbox(col, options=[0, 1], index=int(default) if default in (0, 1) else 0, key=f"f_{col}")
        elif col in ("tenure", "MonthlyCharges", "TotalCharges"):
            row[col] = st.number_input(col, min_value=0.0, value=float(default), step=1.0 if col == "tenure" else 0.5, key=f"f_{col}")
        else:
            row[col] = st.text_input(col, value=str(default), key=f"f_{col}")

    if st.button("Predict churn"):
        one = pd.DataFrame([row])
        for c in ("SeniorCitizen", "tenure"):
            if c in one.columns:
                one[c] = one[c].astype(int)
        for c in ("MonthlyCharges", "TotalCharges"):
            if c in one.columns:
                one[c] = one[c].astype(float)
        X1 = dataframe_to_model_X(one)
        pred, proba = predict_batch(preprocessor, model, X1)
        st.success(f"**Churn probability:** {proba[0]:.3f}  →  **Predicted class:** {'Churn (Yes)' if pred[0] == 1 else 'No churn'}")
