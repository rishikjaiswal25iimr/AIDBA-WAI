# AI-Driven Product Quality Intelligence — Streamlit App

IIM Ranchi · WAI project · Prof. Sasadhar Bera

## Repository layout (all three files in the ROOT)
```
app.py
requirements.txt
Womens Clothing E-Commerce Reviews.csv   <- your dataset (exact name)
```
README.md is optional documentation.

## Run locally
```
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud
1. Push the files above to the root of a GitHub repository (branch `main`).
2. In Streamlit Community Cloud choose **New app**, select the repository, branch `main`, main file `app.py`.
3. Under **Advanced settings** choose **Python 3.11** (TensorFlow does not support 3.13).
4. Deploy. The first start takes a minute (sentiment and complaint tagging are cached afterwards).

## How heavy work is handled
- Startup: CSV load, cleaning, VADER sentiment, complaint tagging (cached once).
- Cached on demand: n-gram/TF-IDF matrices, TF-IDF + Logistic Regression / Linear SVM.
- Explicit buttons: LDA topic modelling, **Train Deep Learning Models** (ANN, LSTM, GRU).
- If TensorFlow is unavailable the app still works; deep-learning panels explain this and show no invented metrics.

## Troubleshooting
- **Dependency conflict / build fails:** select Python 3.11; if TensorFlow still fails, delete the `tensorflow-cpu` line (everything except ANN/LSTM/GRU keeps working).
- **Memory limit / app restarts during training:** use fewer reviews (≤ 8,000), 1 model at a time, fewer epochs, shorter sequences.
- **"Dataset not found":** the CSV must sit next to `app.py` and be named exactly `Womens Clothing E-Commerce Reviews.csv`.
- **Slow first load:** one-off caching of sentiment/complaint scoring; later interactions are fast.

## Notes on rigour
All figures are computed from the CSV at run time. DPMO is a review-based customer-dissatisfaction proxy, not a factory defect rate. Scenario outputs are hypothetical. Complaint themes are keyword + negative-evidence heuristics; LDA topics are exploratory.
