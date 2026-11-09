"""
core.py — data processing, text analytics and modelling for the Fashion Quality Intelligence app.
No Streamlit code here. Heavy libraries (scikit-learn, VADER, wordcloud, Keras/JAX) are imported lazily inside functions
so that the dashboard starts fast.
"""
from __future__ import annotations

import importlib
import importlib.util
import os
import re
import time
from functools import lru_cache
from pathlib import Path
from statistics import NormalDist

os.environ.setdefault("KERAS_BACKEND", "jax")  # must be set before keras is imported (TensorFlow has no Python 3.14 wheels)

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent

DATA_FILE = BASE_DIR / "Womens Clothing E-Commerce Reviews.csv"

REQUIRED_COLS = ["Clothing ID", "Age", "Title", "Review Text", "Rating", "Recommended IND",
                 "Positive Feedback Count", "Division Name", "Department Name", "Class Name"]

SEED = 42

NEGATIONS = {"no", "not", "nor", "never", "none", "nothing", "neither", "cannot", "without", "hardly", "barely", "nobody"}

KEEP_WORDS = NEGATIONS | {"too", "very", "enough", "against"}

THEMES = {
    "Fit & shape": r"\b(fit|fits|fitted|fitting|tight|loose|baggy|boxy|snug|tent|clingy|unflattering|bust|waist|hips?|shoulders?|armholes?|sleeves?|length|short|long|tapered|cut)\b",
    "Sizing": r"\b(size|sizes|sizing|small|large|runs|petite|xs|xxs|xl|xxl|tts|smaller|larger|bigger|oversized|tiny|huge)\b",
    "Fabric & material": r"\b(fabric|material|cotton|polyester|linen|silk|wool|rayon|synthetic|thin|thick|sheer|see[- ]through|lining|texture|scratchy|itchy|stretch|stretchy|wrinkle|wrinkles|wrinkly)\b",
    "Comfort": r"\b(comfortable|comfy|uncomfortable|itchy|scratchy|stiff|heavy|hot|cozy|soft|rough|restrictive|digs?|irritat\w*)\b",
    "Design & style": r"\b(style|design|pattern|print|neckline|ruffles?|silhouette|frumpy|dated|ugly|unflattering|look|looks|looked|shape|shapeless)\b",
    "Quality & construction": r"\b(quality|cheap|cheaply|poorly|stitching|stitch|seams?|hem|buttons?|zipper|fell apart|ripped|tore|torn|pilling|pilled|shrunk|shrink|shrinks|faded|defect\w*|flimsy|holes?|unravel\w*|snag\w*)\b",
    "Value & price": r"\b(price|priced|pricey|expensive|overpriced|cost|worth|money|value|pay|paid)\b",
    "Colour & look vs. photo": r"\b(color|colour|colors|shade|bright|dull|photo|photos|picture|pictures|online|website|different|washed out)\b",
}

THEME_RX = {k: re.compile(v, re.I) for k, v in THEMES.items()}

CUE_RX = re.compile(r"\b(not|no|never|nothing|too|poor|poorly|cheap|cheaply|bad|terrible|awful|disappoint\w*|unfortunate\w*|"
                    r"return\w*|sent back|wrong|uncomfortable|itchy|scratchy|unflattering|ripped|tore|torn|pill\w*|shrunk|"
                    r"shrink\w*|sheer|flimsy|overpriced|expensive|hate|ugly|baggy|boxy|tight|loose|huge|tiny|stiff|faded|"
                    r"different|hardly|barely|frumpy|thin|waste|junk|defect\w*|fell apart|see[- ]through)\b|n't", re.I)

SENT_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")

RECS = {
    "Fit & shape": ("Review pattern grading and fit specifications for the flagged styles; add model measurements and fit notes to product pages.",
                    "Technical Design / Product Development", "Fit-complaint reviews per 100 reviews"),
    "Sizing": ("Audit size charts against measured garments; publish 'runs small/large' guidance and collect fit feedback at review time.",
               "Merchandising / Size-chart owners", "Sizing-complaint reviews per 100 reviews"),
    "Fabric & material": ("Re-check fabric specifications and supplier lots for flagged classes; add fabric weight/transparency descriptors.",
                          "Sourcing / Supplier Quality", "Fabric-complaint reviews per 100 reviews"),
    "Comfort": ("Run wear-test panels for flagged styles; review lining, finishing and seam placement.",
                "Product Development / Quality Assurance", "Comfort-complaint reviews per 100 reviews"),
    "Design & style": ("Compare flagged designs with best-rated peers in the same class; review silhouette and styling guidance.",
                       "Design / Merchandising", "Design-complaint reviews per 100 reviews"),
    "Quality & construction": ("Introduce inline/final inspection checkpoints for stitching, seams and trims on flagged products; escalate to vendors.",
                               "Quality Assurance / Vendor Management", "Construction-complaint reviews per 100 reviews"),
    "Value & price": ("Review price-to-quality positioning on flagged products; test value messaging.",
                      "Pricing / Merchandising", "Value-complaint reviews per 100 reviews"),
    "Colour & look vs. photo": ("Audit photography/colour calibration versus physical product; add true-to-life images.",
                                "Content & Photography / Merchandising", "Appearance-mismatch reviews per 100 reviews"),
}

DATA_DICT = pd.DataFrame([
    ("Clothing ID", "Integer", "Anonymised product identifier (many reviews per product)."),
    ("Age", "Integer", "Reviewer age in years."),
    ("Title", "Text", "Optional review title."),
    ("Review Text", "Text", "Free-text review; some rows are blank."),
    ("Rating", "Ordinal 1–5", "Customer rating, 5 = best."),
    ("Recommended IND", "Binary", "1 = customer recommends the product, 0 = does not. Main modelling target."),
    ("Positive Feedback Count", "Integer", "Number of other customers who found the review helpful."),
    ("Division Name", "Categorical", "High-level product division."),
    ("Department Name", "Categorical", "Product department."),
    ("Class Name", "Categorical", "Product class (finest category level)."),
], columns=["Column", "Type", "Description"])

def locate_csv() -> Path:
    if DATA_FILE.exists():
        return DATA_FILE
    candidates = [p for p in BASE_DIR.glob("*.csv") if "review" in p.name.lower() and "cloth" in p.name.lower()]
    if candidates:
        return candidates[0]
    found = sorted(p.name for p in BASE_DIR.glob("*.csv"))
    raise FileNotFoundError(f"Expected '{DATA_FILE.name}' next to app.py. CSV files found in the app directory: {found or 'none'}.")

def clean_text(s: str) -> str:
    """Lower-case, expand contractions, keep negations, strip punctuation/digits and stop-words."""
    s = str(s).lower().replace("’", "'")
    s = s.replace("won't", "will not").replace("can't", "can not").replace("n't", " not")
    s = re.sub(r"'(s|d|ll|ve|re|m)\b", " ", s)
    s = re.sub(r"[^a-z\s]", " ", s)
    return " ".join(t for t in s.split() if len(t) > 1 and t not in stop_words())

def prepare(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()
    df = df.drop(columns=[c for c in df.columns if str(c).startswith("Unnamed")], errors="ignore")
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"The CSV is missing required columns: {missing}. Found: {list(df.columns)}")
    for c in ["Clothing ID", "Age", "Rating", "Recommended IND", "Positive Feedback Count"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["Rating", "Recommended IND"]).reset_index(drop=True)
    df["Rating"] = df["Rating"].astype(int)
    df["Recommended IND"] = df["Recommended IND"].astype(int)
    for c in ["Division Name", "Department Name", "Class Name"]:
        df[c] = df[c].fillna("Unknown").astype(str).str.strip()
    df["Title"] = df["Title"].fillna("").astype(str).str.strip()
    df["Review Text"] = df["Review Text"].fillna("").astype(str).str.strip()
    df["has_text"] = df["Review Text"].str.len() > 0
    df["full_text"] = np.where(df["has_text"], (df["Title"] + ". " + df["Review Text"]).str.strip(". "), "")
    df["n_words"] = df["Review Text"].str.split().str.len().fillna(0).astype(int)
    df["low_rating"] = df["Rating"] <= 2
    df["non_rec"] = 1 - df["Recommended IND"]
    df["age_group"] = pd.cut(df["Age"], [0, 24, 34, 44, 54, 64, 200],
                             labels=["<25", "25–34", "35–44", "45–54", "55–64", "65+"])
    df["row_id"] = np.arange(len(df))
    return df

def tag_complaints(text: str, analyzer) -> set:
    """Sentence-level, negation-aware complaint tagging: theme keyword AND negative evidence in the same sentence."""
    found = set()
    for sent in SENT_SPLIT.split(text):
        sl = sent.lower().replace("’", "'")
        if len(sl) < 3:
            continue
        hits = [t for t, rx in THEME_RX.items() if rx.search(sl)]
        if not hits:
            continue
        comp = analyzer.polarity_scores(sl)["compound"]
        if comp <= -0.25 or (CUE_RX.search(sl) and comp < 0.5):
            found.update(hits)
    return found

def data_signature(df: pd.DataFrame) -> str:
    return f"{len(df)}-{int(pd.util.hash_pandas_object(df['full_text']).sum() % 10**9)}"

def wilson(k, n, z=1.96):
    k, n = np.asarray(k, float), np.asarray(n, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        p = k / n
        d = 1 + z * z / n
        c = p + z * z / (2 * n)
        a = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
        return (c - a) / d, (c + a) / d

def seg_table(df: pd.DataFrame, by, min_n: int = 1) -> pd.DataFrame:
    g = df.groupby(by, observed=True).agg(n=("Rating", "size"), avg_rating=("Rating", "mean"),
                                          rec_rate=("Recommended IND", "mean"), nonrec_n=("non_rec", "sum"),
                                          low_rate=("low_rating", "mean")).reset_index()
    g["nonrec_rate"] = g["nonrec_n"] / g["n"]
    g["ci_low"], g["ci_high"] = wilson(g["nonrec_n"], g["n"])
    g["dpmo"] = g["nonrec_rate"] * 1e6
    return g[g["n"] >= min_n].sort_values("nonrec_rate", ascending=False).reset_index(drop=True)

def theme_table(df: pd.DataFrame, min_n: int = 30) -> pd.DataFrame:
    d = df[df["has_text"]]
    cols = ["Theme", "n_flagged", "share_reviews", "nonrec_rate_flagged", "nonrec_rate_other", "lift",
            "ci_low", "excess_nonrec", "share_of_nonrec", "reliable"]
    if d.empty:
        return pd.DataFrame(columns=cols)
    rows, tot_nr = [], max(int(d["non_rec"].sum()), 1)
    for th in THEMES:
        f = d[f"cmp_{th}"]
        nf = int(f.sum())
        nr_f = int(d.loc[f, "non_rec"].sum())
        n_u = len(d) - nf
        rate_f = nr_f / nf if nf else np.nan
        rate_u = d.loc[~f, "non_rec"].sum() / n_u if n_u else np.nan
        lo, _ = wilson(nr_f, nf) if nf else (np.nan, np.nan)
        rows.append(dict(Theme=th, n_flagged=nf, share_reviews=nf / len(d), nonrec_rate_flagged=rate_f,
                         nonrec_rate_other=rate_u, lift=rate_f / rate_u if rate_u else np.nan, ci_low=float(lo),
                         excess_nonrec=nf * (rate_f - rate_u) if nf else 0.0, share_of_nonrec=nr_f / tot_nr,
                         reliable=bool(nf >= min_n and lo > rate_u)))
    return pd.DataFrame(rows).sort_values("excess_nonrec", ascending=False).reset_index(drop=True)

def core_kpis(df: pd.DataFrame) -> dict:
    t = df["has_text"]
    return dict(n=len(df), products=int(df["Clothing ID"].nunique()), avg=df["Rating"].mean(),
                rec=df["Recommended IND"].mean(), nonrec=df["non_rec"].mean(), text_n=int(t.sum()),
                neg=((df.loc[t, "sent_class"] == "Negative").mean() if ("sent_class" in df.columns and t.any()) else np.nan),
                dpmo=df["non_rec"].mean() * 1e6, low=df["low_rating"].mean())

def sigma_level(dpmo: float) -> float:
    p = min(max(dpmo / 1e6, 1e-9), 1 - 1e-9)
    return float(NormalDist().inv_cdf(1 - p) + 1.5)

def modeling_frame(df: pd.DataFrame) -> pd.DataFrame:
    d = df[df["has_text"] & (df["clean"].str.len() > 0)].copy()
    d["key"] = d["full_text"].str.lower().str.strip()
    d = d.drop_duplicates("key")
    return d[["row_id", "clean", "full_text", "Recommended IND"]].rename(columns={"Recommended IND": "y"}).reset_index(drop=True)

def get_split(df, sample_n, seed=SEED):
    from sklearn.model_selection import train_test_split
    d = modeling_frame(df)
    if sample_n and sample_n < len(d):
        d, _ = train_test_split(d, train_size=int(sample_n), stratify=d["y"], random_state=seed)
    return train_test_split(d, test_size=0.2, stratify=d["y"], random_state=seed)

def thin(a, n=300):
    a = np.asarray(a)
    return a[np.linspace(0, len(a) - 1, min(len(a), n)).astype(int)].tolist()

def eval_scores(y, score, pred) -> dict:
    from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix, f1_score, precision_recall_curve,
                                 precision_recall_fscore_support, roc_auc_score, roc_curve)
    y, score, pred = np.asarray(y), np.asarray(score), np.asarray(pred)
    p, r, f, _ = precision_recall_fscore_support(y, pred, labels=[0, 1], zero_division=0)
    fpr, tpr, _ = roc_curve(y, score)
    pr, rc, _ = precision_recall_curve(y, score)
    return dict(
        metrics=dict(accuracy=accuracy_score(y, pred), precision_rec=p[1], recall_rec=r[1], f1_rec=f[1], precision_non=p[0],
                     recall_non=r[0], f1_non=f[0], macro_f1=f1_score(y, pred, average="macro"), roc_auc=roc_auc_score(y, score),
                     pr_auc_rec=average_precision_score(y, score), pr_auc_non=average_precision_score(1 - y, -score)),
        cm=confusion_matrix(y, pred, labels=[0, 1]).tolist(),
        roc=(thin(fpr), thin(tpr)), pr=(thin(rc), thin(pr)))

def build_vocab(texts, size):
    from collections import Counter
    c = Counter(w for t in texts for w in t.split())
    return {w: i + 2 for i, (w, _) in enumerate(c.most_common(size))}

def encode(texts, vocab, maxlen):
    arr = np.zeros((len(texts), maxlen), dtype="int32")
    for i, t in enumerate(texts):
        ids = [vocab.get(w, 1) for w in t.split()][-maxlen:]
        if ids:
            arr[i, maxlen - len(ids):] = ids  # pre-padding: real tokens sit next to the recurrent output
    return arr

# ----------------------------------------------------------------------------------------------
# Lazy stop-words, dependency report, CSV
# ----------------------------------------------------------------------------------------------
@lru_cache(maxsize=1)
def stop_words() -> frozenset:
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
    s = set(ENGLISH_STOP_WORDS) - KEEP_WORDS
    s |= {"just", "really", "would", "also", "got", "get", "one", "im", "ive", "dont", "didnt"}
    return frozenset(s)


def lda_stop_words() -> list:
    return sorted(set(stop_words()) | NEGATIONS | {"too", "very", "enough"})


REQUIRED_MODULES = {"numpy": "numpy", "pandas": "pandas", "plotly": "plotly", "plotly.express": "plotly", "scipy": "scipy",
                    "sklearn": "scikit-learn", "vaderSentiment": "vaderSentiment", "wordcloud": "wordcloud", "networkx": "networkx"}


def dependency_report(optional: bool = False) -> list:
    """Import every module by name and report the exact one that fails (with the real error text)."""
    mods = {"keras": "keras", "jax": "jax"} if optional else REQUIRED_MODULES
    out = []
    for mod, pkg in mods.items():
        try:
            m = importlib.import_module(mod)
            out.append((mod, pkg, True, getattr(m, "__version__", "")))
        except Exception as err:  # report any import-time failure, including transitive ones
            out.append((mod, pkg, False, f"{type(err).__name__}: {err}"))
    return out


def read_csv_file() -> pd.DataFrame:
    path = locate_csv()
    last = None
    for enc in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError as err:
            last = err
    raise last


def add_clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["clean"] = df["full_text"].map(clean_text)
    return df


@lru_cache(maxsize=1)
def get_analyzer():
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    return SentimentIntensityAnalyzer()


def add_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    an = get_analyzer()
    df["compound"] = [an.polarity_scores(t)["compound"] if t else np.nan for t in df["full_text"].tolist()]
    df["sent_class"] = np.where(~df["has_text"], "No text", np.where(df["compound"] >= 0.05, "Positive",
                                np.where(df["compound"] <= -0.05, "Negative", "Neutral")))
    return df


def add_complaints(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    an = get_analyzer()
    tags = [tag_complaints(t, an) if t else set() for t in df["full_text"].tolist()]
    for th in THEMES:
        df[f"cmp_{th}"] = [th in s for s in tags]
    df["any_complaint"] = df[[f"cmp_{t}" for t in THEMES]].any(axis=1)
    return df


# ----------------------------------------------------------------------------------------------
# Text features
# ----------------------------------------------------------------------------------------------
def build_text_matrices(df: pd.DataFrame) -> dict:
    from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
    docs = df["clean"].tolist()
    cv1 = CountVectorizer(ngram_range=(1, 1), min_df=3, max_features=20000)
    cv2 = CountVectorizer(ngram_range=(2, 2), min_df=3, max_features=30000)
    tf = TfidfVectorizer(ngram_range=(1, 2), min_df=5, max_features=20000, sublinear_tf=True)
    return dict(M1=cv1.fit_transform(docs).tocsr(), M2=cv2.fit_transform(docs).tocsr(), T=tf.fit_transform(docs).tocsr(),
                t1=cv1.get_feature_names_out(), t2=cv2.get_feature_names_out(), tt=tf.get_feature_names_out())


def make_wordcloud(freq_items: tuple, colormap: str):
    from wordcloud import WordCloud
    wc = WordCloud(width=800, height=380, background_color="white", colormap=colormap, max_words=100,
                   prefer_horizontal=0.95, random_state=SEED).generate_from_frequencies(dict(freq_items))
    return wc.to_array()


def fit_lda(df: pd.DataFrame, n_topics: int, max_iter: int, seed: int = SEED) -> dict:
    from sklearn.decomposition import LatentDirichletAllocation
    from sklearn.feature_extraction.text import CountVectorizer
    sub = df[df["has_text"] & (df["clean"].str.len() > 0)]
    cv = CountVectorizer(min_df=10, max_df=0.5, max_features=3000, stop_words=lda_stop_words())
    X = cv.fit_transform(sub["clean"])
    lda = LatentDirichletAllocation(n_components=n_topics, learning_method="online", max_iter=max_iter, batch_size=512,
                                    random_state=seed, n_jobs=1)
    theta = lda.fit_transform(X)
    return dict(row_ids=sub["row_id"].to_numpy(), theta=theta, comps=lda.components_, terms=cv.get_feature_names_out(),
                perplexity=float(lda.perplexity(X)), n_docs=X.shape[0], n_terms=X.shape[1])


# ----------------------------------------------------------------------------------------------
# Models
# ----------------------------------------------------------------------------------------------
def train_classical(df: pd.DataFrame, sample_n, C: float, seed: int = SEED) -> dict:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import LinearSVC
    tr, te = get_split(df, sample_n, seed)
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=30000, sublinear_tf=True)
    Xtr, Xte = vec.fit_transform(tr["clean"]), vec.transform(te["clean"])
    lr = LogisticRegression(C=C, max_iter=1000, class_weight="balanced").fit(Xtr, tr["y"])
    svm = LinearSVC(C=max(C / 10, 0.01), class_weight="balanced", max_iter=5000).fit(Xtr, tr["y"])
    out = {}
    p = lr.predict_proba(Xte)[:, 1]
    out["TF-IDF + Logistic Regression"] = dict(**eval_scores(te["y"], p, (p >= 0.5).astype(int)), n_train=len(tr), n_test=len(te),
                                              input="TF-IDF (1–2-grams, ≤30k features, sublinear tf)",
                                              params=f"C={C}, class_weight=balanced, max_iter=1000, threshold 0.5")
    s = svm.decision_function(Xte)
    out["TF-IDF + Linear SVM"] = dict(**eval_scores(te["y"], s, (s >= 0).astype(int)), n_train=len(tr), n_test=len(te),
                                      input="TF-IDF (1–2-grams, ≤30k features, sublinear tf)",
                                      params=f"C={max(C / 10, 0.01)}, class_weight=balanced, margin 0")
    names = vec.get_feature_names_out()
    co = lr.coef_[0]
    return dict(results=out, vec=vec, lr=lr, svm=svm, top_pos=names[np.argsort(-co)[:15]].tolist(), top_neg=names[np.argsort(co)[:15]].tolist(),
                coef_pos=np.sort(co)[::-1][:15].tolist(), coef_neg=np.sort(co)[:15].tolist())


def keras_available() -> bool:
    return importlib.util.find_spec("keras") is not None and importlib.util.find_spec("jax") is not None


def train_deep(df: pd.DataFrame, models, sample_n, epochs, batch, maxlen, vocab_size, progress_cb=None) -> dict:
    """Train ANN / LSTM / GRU with Keras 3 (JAX CPU backend). Returns {model name: result dict}."""
    import keras
    from keras import layers
    from sklearn.feature_extraction.text import TfidfVectorizer
    keras.utils.set_random_seed(SEED)
    tr, te = get_split(df, sample_n)
    ytr, yte = tr["y"].to_numpy(), te["y"].to_numpy()
    cw = {0: len(ytr) / (2 * max((ytr == 0).sum(), 1)), 1: len(ytr) / (2 * max((ytr == 1).sum(), 1))}
    vocab = build_vocab(tr["clean"], vocab_size)
    Xtr_s, Xte_s = encode(tr["clean"], vocab, maxlen), encode(te["clean"], vocab, maxlen)
    vec = TfidfVectorizer(max_features=2000, ngram_range=(1, 2), min_df=3, sublinear_tf=True)
    Xtr_t = vec.fit_transform(tr["clean"]).astype("float32").toarray()
    Xte_t = vec.transform(te["clean"]).astype("float32").toarray()
    sig = data_signature(df)
    results = {}
    for mi, name in enumerate(models):
        keras.backend.clear_session()
        if name == "Multilayer ANN":
            model = keras.Sequential([layers.Input((Xtr_t.shape[1],)), layers.Dense(128, activation="relu"), layers.Dropout(0.4),
                                      layers.Dense(64, activation="relu"), layers.Dropout(0.3), layers.Dense(1, activation="sigmoid")])
            Xa, Xb, inp = Xtr_t, Xte_t, f"TF-IDF dense vector ({Xtr_t.shape[1]} features)"
        else:
            rnn = layers.LSTM(64, dropout=0.2) if name == "LSTM" else layers.GRU(64, dropout=0.2)
            model = keras.Sequential([layers.Input((maxlen,), dtype="int32"), layers.Embedding(len(vocab) + 2, 64), rnn, layers.Dropout(0.3),
                                      layers.Dense(32, activation="relu"), layers.Dense(1, activation="sigmoid")])
            Xa, Xb, inp = Xtr_s, Xte_s, f"Token-id sequences (pre-padded, maxlen {maxlen}, vocab {len(vocab):,})"
        model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="binary_crossentropy", metrics=["accuracy"])

        class Prog(keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                if progress_cb:
                    progress_cb(mi, len(models), name, epoch + 1, epochs, logs or {})

        t0 = time.time()
        hist = model.fit(Xa, ytr, validation_split=0.15, epochs=epochs, batch_size=batch, class_weight=cw, verbose=0,
                         callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True), Prog()])
        p = np.asarray(model.predict(Xb, batch_size=256, verbose=0)).ravel()
        res = eval_scores(yte, p, (p >= 0.5).astype(int))
        res.update(n_train=len(tr), n_test=len(te), input=inp, sig=sig, seconds=round(time.time() - t0, 1), epochs_run=len(hist.history["loss"]),
                   params=f"epochs≤{epochs} (early stopping, patience 2), batch {batch}, Adam 1e-3, dropout, class-weighted, threshold 0.5, Keras {keras.__version__} / {keras.backend.backend()} backend",
                   history={k: [float(x) for x in v] for k, v in hist.history.items()}, kind=name,
                   obj=dict(model=model, vec=vec, vocab=vocab, maxlen=maxlen))
        results[name] = res
    return results
