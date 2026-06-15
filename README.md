# Bladder Cancer Distant Metastasis Calculator

Online risk calculator (GBDT model) for synchronous distant metastasis in invasive bladder cancer.

## Files
- `app.py` — Streamlit web app
- `gbdt_model.joblib` — trained model (preprocessing + GBDT)
- `requirements.txt` — dependencies

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy free on Streamlit Community Cloud
1. Create a **public GitHub repo** and upload these 3 files (`app.py`, `gbdt_model.joblib`, `requirements.txt`).
2. Go to https://share.streamlit.io → sign in with GitHub → **New app**.
3. Pick your repo, branch `main`, main file `app.py` → **Deploy**.
4. You get a public URL like `https://<name>.streamlit.app` → screenshot = Figure 11.
