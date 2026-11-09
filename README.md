# AI-Driven Product Quality Intelligence — Streamlit App (Python 3.14)

IIM Ranchi · WAI project · Prof. Sasadhar Bera

## Repository layout (all files in the ROOT, no folders)
```
app.py            <- Streamlit interface (9 pages)
core.py           <- data processing, text analytics, modelling (no Streamlit code)
requirements.txt
Womens Clothing E-Commerce Reviews.csv   <- exact file name
```
`app.py` imports `core.py`, so upload both. README.md is optional.

## Deploy on Streamlit Community Cloud (Python 3.14)
1. Replace the files in your GitHub repo root and push to `main`.
2. Reboot/redeploy the app (main file `app.py`). Keep Python 3.14.
3. Pinned versions in `requirements.txt` were resolved with `--only-binary :all:` on Python 3.14 (Linux), i.e. prebuilt wheels only, no source builds.

## Why the earlier deployment failed
- The old pins (pandas 2.2.3, numpy 1.26.4, scipy 1.13.1, scikit-learn 1.5.2, wordcloud 1.9.3, plotly 5.24.1) predate Python 3.14, so the installer compiled them from source (~13 min).
- TensorFlow has **no Python 3.14 wheels**. Deep learning now uses **Keras 3 on the JAX CPU backend** (`keras`, `jax`), which does.
- Plotly Express needs pandas (and, in current Plotly, narwhals). If any dependency is missing, the app now prints the exact failing module and a package table instead of a bare `ModuleNotFoundError`.

## Start-up architecture
1. **Start-up:** read CSV, validate columns, KPIs, Executive Dashboard (no NLP, no models).
2. **Cached analytics (on demand):** text cleaning (pages 4, 7), VADER sentiment + complaint tags (button on page 1, automatic on first visit to pages 5, 6, 8, 9), n-gram/TF-IDF matrices. Computed once per server process, then shared.
3. **Explicit actions:** LDA topic model (button), TF-IDF classical models (cached per setting), ANN/LSTM/GRU (**Train Deep Learning Models** button).

## Troubleshooting
- **Dataset not found:** CSV must sit next to `app.py` with the exact name above.
- **Memory restarts during deep-learning training:** use ≤ 8,000 reviews, one model at a time, fewer epochs, shorter sequences.
- **Import error screen:** read the module table; add the missing package to `requirements.txt`. Page 2 → Data audit → *Environment check* lists installed versions.
- **Slow first NLP step:** VADER + complaint tagging takes ~15–30 s once, then is cached.

## Rigour notes
All figures are computed from the CSV at run time. DPMO is a review-based customer-dissatisfaction proxy, not a factory defect rate. Scenario outputs are hypothetical. Deep-learning results appear only after training in your session (or valid imported results).
