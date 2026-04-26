# Deploy Telco Churn app (GitHub + Streamlit Community Cloud)

This folder is a **standalone Git repo** (see `README.md`). From here:

```bash
git status
git commit -m "Initial commit: Telco churn Streamlit app"   # if nothing committed yet
git branch -M main
git remote add origin https://github.com/<YOUR_USER>/<YOUR_REPO>.git
git push -u origin main
```

If `git commit` fails with an error about **`trailer`**, your Git client is very old—install a current [Git for Windows](https://git-scm.com/download/win) or commit with **GitHub Desktop**.

## 1. Create artifacts locally

From this folder (with `Telco_Customer_Churn.csv` present):

```bash
python train_export.py
```

Or run **section 8** in `Telco_Customer_Churn_Prediction.ipynb` after training (exports the **Test F1** champion + `form_defaults`).

You should have:

- `artifacts/preprocessor.joblib`
- `artifacts/model.joblib`
- `artifacts/model_name.txt`
- `artifacts/form_defaults.joblib`

Commit these files so Streamlit Cloud can load them without retraining.

## 2. Push to GitHub

1. Create a new **empty** repository on GitHub (no README/license if you already have files here).
2. In **this** directory (`telco-churn-streamlit`), run:

```bash
git add .
git commit -m "Add Telco churn Streamlit app and ML artifacts"
git branch -M main
git remote add origin https://github.com/<YOUR_USER>/<YOUR_REPO>.git
git push -u origin main
```

Replace `<YOUR_USER>` / `<YOUR_REPO>` with your GitHub username and repository name. (`git init` is already done in this copy.)

**Note:** If you do not want the dataset in the repo, remove `Telco_Customer_Churn.csv` from `git add` and keep only `artifacts/` for inference. The app does not need the CSV at runtime.

## 3. Deploy on Streamlit Community Cloud

1. Sign in at [https://share.streamlit.io](https://share.streamlit.io) with GitHub.
2. **New app** → pick the repository and branch (**`main`** or **`master`**, whichever you use on GitHub).
3. **Main file path:** `app.py`
4. Click **Advanced settings** → set **Python version to 3.12** (or **3.11**).  
   **Do not use Python 3.14** for this app: logs like `Python 3.14.x` often mean `pip`/`uv` has little or no binary wheel support yet for **numpy / pandas / scikit-learn**, so the install can sit for a long time or fail. See [Upgrade your app’s Python version](https://docs.streamlit.io/deploy/streamlit-community-cloud/manage-your-app/upgrade-python) — if you already deployed on 3.14, you must **Delete** the app and **deploy again** with **Advanced settings → Python 3.12**.
5. **App URL:** choose a subdomain → **Deploy**.

Streamlit installs dependencies from `requirements.txt` and runs `streamlit run app.py`.

## 4. After you change the model

1. Re-run the notebook export cell (or `python train_export.py`).
2. Commit the updated files under `artifacts/`.
3. Push to GitHub; Streamlit Cloud will redeploy automatically (or use **Manage app → Reboot**).

## Troubleshooting

- **“In the oven” forever / slow `Processing dependencies`:** Check logs for **`Python 3.14`**. Switch to **Python 3.12** (delete app → redeploy → Advanced settings). See section **3** above.
- **Missing artifacts:** Cloud build succeeds but the app shows an error → ensure `artifacts/*.joblib` and `model_name.txt` are committed and paths are correct.
- **Import / sklearn errors:** Use `scikit-learn>=1.5.0,<1.7.0` in `requirements.txt` to match Colab-pickled `ColumnTransformer` artifacts.
