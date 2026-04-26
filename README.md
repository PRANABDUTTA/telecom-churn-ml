# Telco customer churn prediction (Streamlit)

**Repo folder (copy for GitHub):**  
`CAPSTONE PROJECT 1\telco-churn-streamlit`  
Full path: `c:\Share\personal\ai_ml_projects\capstone_projects\CAPSTONE PROJECT 1\telco-churn-streamlit`

Interactive app to score **customer churn** using a trained scikit-learn model and the same preprocessing as the capstone notebook. `git init` is already run; stage with `git add .`, then `git commit` and push (see **DEPLOY.md**).

## Run locally

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
streamlit run app.py
```

## Deploy

See **[DEPLOY.md](DEPLOY.md)** for GitHub + [Streamlit Community Cloud](https://share.streamlit.io) (main file: **`app.py`**).

## Repo contents

| Item | Purpose |
|------|--------|
| `app.py` | Streamlit UI (CSV upload + single-customer form) |
| `artifacts/` | Saved `preprocessor` + model (`joblib`) |
| `train_export.py` | Regenerate `artifacts/` from `Telco_Customer_Churn.csv` |
| `Telco_Customer_Churn_Prediction.ipynb` | Full ML workflow (EDA, models, export) |
