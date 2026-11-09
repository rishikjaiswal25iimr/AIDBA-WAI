"""
AI-Driven Product Quality Intelligence and Customer Dissatisfaction Prediction
in Fashion E-Commerce — Text Analytics, Deep Learning and Six Sigma-Based Quality Improvement

Single-file Streamlit interface. Required files in the repository root:
    app.py, core.py, requirements.txt, Womens Clothing E-Commerce Reviews.csv
app.py = user interface only; core.py = data processing, text analytics and modelling.
Every number shown is calculated at run time from the CSV.
"""
from __future__ import annotations

import html
import importlib.util
import json
import re

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Fashion Quality Intelligence", page_icon="👗",
                   layout="wide", initial_sidebar_state="expanded")

# ---- Dependency validation: report the EXACT module that fails (including transitive ones) instead of a bare ModuleNotFoundError
try:
    import plotly.express as px
    import plotly.graph_objects as go
    import core
    from core import (DATA_DICT, RECS, SEED, THEMES, add_complaints, add_sentiment, clean_text, core_kpis, data_signature,
                      encode, modeling_frame, seg_table, sigma_level, tag_complaints, theme_table)
except Exception as _err:  # noqa: BLE001 - deliberately broad: any import-time failure must be shown with its real cause
    st.error(f"**Start-up import failed:** `{type(_err).__name__}: {_err}`")
    try:
        import core as _core
        rows = [{"module": m, "pip package": p, "status": "OK " + v if ok else "FAILED", "detail": "" if ok else v}
                for m, p, ok, v in _core.dependency_report()]
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    except Exception as _err2:  # noqa: BLE001
        st.error(f"core.py could not be imported either: `{type(_err2).__name__}: {_err2}`. Make sure core.py is in the repository root next to app.py.")
    st.info("Check requirements.txt: every package listed above must install on this Python version.")
    st.stop()

# ----------------------------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------------------------
NAVY, TEAL, SKY, AMBER, RED, OFFWHITE = "#14213D", "#00A896", "#3A86FF", "#F4A261", "#E63946", "#F8FAFC"
SLATE, INK, LINE = "#5B6B7F", "#33415C", "#E3E8F0"
COLORWAY = [SKY, TEAL, AMBER, RED, NAVY, "#8E7DBE", "#6C757D", "#2A9D8F"]
RATING_COLORS = {1: RED, 2: "#F08A5D", 3: AMBER, 4: "#5FC9B8", 5: TEAL}
PAGES = ["Executive Dashboard", "Objectives & Data Audit", "Satisfaction & EDA", "Text Analytics", "Sentiment Intelligence",
         "Topics & Complaint Explorer", "Machine & Deep Learning", "Quality & Six Sigma", "Management Insights"]
PAGE_ICONS = ["📊", "🧭", "🛍️", "🔤", "💬", "🧩", "🤖", "🎯", "📌"]
FILTER_PAGES = {PAGES[2], PAGES[3], PAGES[4], PAGES[5], PAGES[7]}
_KEY = [0]
_PAGE = [0]  # index of the page being rendered (used by the header badge)

# ----------------------------------------------------------------------------------------------
# Styling helpers
# ----------------------------------------------------------------------------------------------
CSS = """
<style>
.block-container {max-width: 1500px; padding-top: 3rem; padding-bottom: 3rem;}
/* ---------- header ---------- */
.hero {background: linear-gradient(120deg,__NAVY__ 0%,#1d3557 58%,__TEAL__ 160%); color:#fff; border-radius:18px; padding:22px 28px;
       margin:0 0 18px 0; display:flex; justify-content:space-between; align-items:flex-start; gap:24px;
       box-shadow:0 8px 24px rgba(20,33,61,.18);}
.hero-main {min-width:0;}
.hero-kicker {font-size:.7rem; letter-spacing:.16em; text-transform:uppercase; color:#8EE3D8; font-weight:700; margin-bottom:6px;}
.hero-title {font-size:1.6rem; font-weight:750; line-height:1.25; color:#fff;}
.hero-sub {color:#D3E0F2; font-size:.95rem; margin-top:6px; max-width:920px;}
.hero-chips {display:flex; flex-wrap:wrap; gap:8px; margin-top:14px;}
.hero-chip {background:rgba(255,255,255,.12); border:1px solid rgba(255,255,255,.22); border-radius:999px; padding:4px 13px;
            font-size:.8rem; color:#fff; white-space:nowrap;}
.hero-chip b {color:#8EE3D8; font-weight:700;}
.hero-badge {flex:none; background:rgba(255,255,255,.12); border:1px solid rgba(255,255,255,.2); border-radius:14px; padding:8px 16px;
             text-align:center; min-width:70px;}
.hero-badge .n {font-size:1.5rem; font-weight:750; line-height:1.1; color:#fff;}
.hero-badge .n span {font-size:.8rem; font-weight:500; color:#BFD0E8;}
.hero-badge .l {font-size:.65rem; letter-spacing:.14em; text-transform:uppercase; color:#8EE3D8;}
/* ---------- section headers ---------- */
.sec {display:flex; align-items:baseline; gap:12px; flex-wrap:wrap; margin:26px 0 12px 0; padding-bottom:7px; border-bottom:2px solid __LINE__;}
.sec h3 {margin:0; padding:0; font-size:1.15rem; font-weight:700; color:inherit; border-left:5px solid __TEAL__; padding-left:10px; line-height:1.3;}
.sec span {font-size:.85rem; opacity:.7;}
.ctrl-label {font-size:.72rem; letter-spacing:.12em; text-transform:uppercase; font-weight:700; color:__SLATE__; margin:0 0 4px 0;}
/* ---------- KPI cards ---------- */
.kpi {background:#fff; border:1px solid __LINE__; border-left:5px solid var(--accent,__TEAL__); border-radius:14px; padding:13px 16px 11px 16px;
      min-height:116px; box-shadow:0 1px 4px rgba(20,33,61,.07); display:flex; flex-direction:column; justify-content:space-between;
      margin-bottom:10px;}
.kpi-l {font-size:.7rem; text-transform:uppercase; letter-spacing:.07em; color:__SLATE__; font-weight:600; line-height:1.3;}
.kpi-i {margin-left:6px; opacity:.55; font-size:.8rem; cursor:help; text-transform:none;}
.kpi-v {font-size:1.7rem; font-weight:750; color:__NAVY__; line-height:1.2; word-break:break-word; margin:4px 0 2px 0;}
.kpi-s {font-size:.74rem; color:__SLATE__; line-height:1.3;}
/* ---------- panels, callouts, insights ---------- */
.callout {background:#EAF6F4; border-left:4px solid __TEAL__; padding:.7rem 1rem; border-radius:10px; color:__NAVY__; font-size:.9rem; margin:.5rem 0 .9rem 0;
          line-height:1.5;}
.callout.caution {background:#FFF3E8; border-left-color:__AMBER__;}
.callout code, .panel code {background:rgba(20,33,61,.08); color:__NAVY__; padding:1px 5px; border-radius:4px;}
.panel {background:#F6F8FB; border:1px solid __LINE__; border-radius:14px; padding:14px 18px; color:__INK__; margin-bottom:10px;}
.panel h4 {margin:0 0 8px 0; padding:0; font-size:1rem; color:__NAVY__;}
.panel ul {margin:0; padding-left:1.1rem;}
.panel li {margin:0 0 7px 0; font-size:.9rem; line-height:1.5; color:__INK__;}
.panel li::marker {color:__TEAL__;}
.panel p {margin:0; font-size:.9rem; color:__INK__;}
.explain {display:grid; grid-template-columns:repeat(3,1fr); gap:12px; margin:.6rem 0 1rem 0;}
.explain > div {background:#F6F8FB; border:1px solid __LINE__; border-top:3px solid __SKY__; border-radius:12px; padding:10px 14px;}
.explain > div:nth-child(2) {border-top-color:__TEAL__;}
.explain > div:nth-child(3) {border-top-color:__AMBER__;}
.explain h6 {margin:0 0 4px 0; padding:0; font-size:.7rem; letter-spacing:.1em; text-transform:uppercase; color:__SLATE__;}
.explain p {margin:0; font-size:.84rem; color:__INK__; line-height:1.45;}
.rec {background:#fff; border:1px solid __LINE__; border-top:4px solid __TEAL__; border-radius:14px; padding:14px 16px; height:100%; margin-bottom:10px;
      box-shadow:0 1px 4px rgba(20,33,61,.06);}
.rec .rk {display:inline-block; background:__NAVY__; color:#fff; border-radius:999px; font-size:.7rem; font-weight:700; padding:2px 10px; margin-bottom:8px;}
.rec .th {font-size:1.02rem; font-weight:700; color:__NAVY__; margin-bottom:6px;}
.rec .ac {font-size:.86rem; color:__INK__; line-height:1.45; margin-bottom:10px;}
.rec .mt {font-size:.76rem; color:__SLATE__; line-height:1.45;}
.rec .mt b {color:__NAVY__;}
.flow {display:flex; flex-wrap:wrap; align-items:center; gap:6px; margin:.3rem 0 .8rem 0;}
.flow .st {background:__NAVY__; color:#fff; padding:.4rem .85rem; border-radius:999px; font-size:.82rem; display:inline-flex; gap:7px; align-items:center;}
.flow .st i {font-style:normal; background:__TEAL__; color:#fff; border-radius:50%; width:18px; height:18px; display:inline-flex; align-items:center;
             justify-content:center; font-size:.68rem; font-weight:700;}
.flow .ar {color:__TEAL__; font-weight:700;}
.pill {display:inline-block; padding:2px 10px; border-radius:999px; font-size:.74rem; font-weight:600; background:#E3F4EC; color:#1E7A52;}
.pill.warn {background:#FFF3D6; color:#8A6200;} .pill.bad {background:#FBE4E4; color:#9C2B2B;}
.foot {font-size:.76rem; color:__SLATE__; border-top:1px solid __LINE__; margin-top:26px; padding-top:10px; line-height:1.5;}
/* ---------- Streamlit components ---------- */
div[data-testid="stPlotlyChart"] {background:#fff; border:1px solid __LINE__; border-radius:14px; padding:8px 8px 2px 8px;
                                  box-shadow:0 1px 4px rgba(20,33,61,.06); margin-bottom:10px;}
div[data-testid="stDataFrame"] {border:1px solid __LINE__; border-radius:12px; overflow:hidden;}
div[data-testid="stImage"] img {border-radius:12px; border:1px solid __LINE__;}
div[data-testid="stVerticalBlockBorderWrapper"] {border-radius:14px;}
.stTabs [data-baseweb="tab-list"] {gap:4px; border-bottom:1px solid __LINE__;}
.stTabs [data-baseweb="tab"] {padding:9px 16px; border-radius:10px 10px 0 0; font-weight:600;}
.stTabs [aria-selected="true"] {color:__TEAL__;}
.st-key-nav_next {display:flex; justify-content:flex-end;}
.stButton > button, .stDownloadButton > button, div[data-testid="stFormSubmitButton"] > button {border-radius:10px; font-weight:600;}
/* ---------- sidebar ---------- */
section[data-testid="stSidebar"] {border-right:1px solid __LINE__;}
.brand {display:flex; gap:11px; align-items:center; padding:2px 2px 10px 2px;}
.brand-logo {flex:none; width:40px; height:40px; border-radius:11px; background:linear-gradient(135deg,__NAVY__,__TEAL__); display:flex;
             align-items:center; justify-content:center; font-size:1.25rem;}
.brand-name {font-weight:750; line-height:1.15; font-size:1.02rem;}
.brand-sub {font-size:.72rem; opacity:.7; margin-top:2px;}
.side-label {font-size:.68rem; letter-spacing:.14em; text-transform:uppercase; opacity:.65; font-weight:700; margin:16px 0 6px 2px;}
.side-count {background:#F6F8FB; border:1px solid __LINE__; border-radius:12px; padding:9px 12px; color:__INK__; font-size:.82rem; margin:8px 0 6px 0;}
.side-count b {color:__NAVY__; font-size:1rem;}
.side-count .bar {height:6px; background:#DCE4EE; border-radius:6px; margin-top:7px; overflow:hidden;}
.side-count .bar span {display:block; height:100%; background:linear-gradient(90deg,__TEAL__,__SKY__); border-radius:6px;}
.side-note {background:#F6F8FB; border:1px solid __LINE__; border-radius:12px; padding:9px 12px; color:__INK__; font-size:.8rem; margin:8px 0;}
@supports selector(:has(*)) {
  section[data-testid="stSidebar"] div[role="radiogroup"] {gap:3px;}
  section[data-testid="stSidebar"] div[role="radiogroup"] > label {padding:9px 12px; border-radius:11px; width:100%; margin:0; cursor:pointer;
                                                                    border:1px solid transparent;}
  section[data-testid="stSidebar"] div[role="radiogroup"] > label > div:first-child {display:none;}
  section[data-testid="stSidebar"] div[role="radiogroup"] > label:hover {background:rgba(58,134,255,.10);}
  section[data-testid="stSidebar"] div[role="radiogroup"] > label:has(input:checked) {background:linear-gradient(90deg,__NAVY__,#1d3557);
                                                                                      box-shadow:0 3px 10px rgba(20,33,61,.25);}
  section[data-testid="stSidebar"] div[role="radiogroup"] > label:has(input:checked) * {color:#fff !important; font-weight:650;}
}
@media (max-width: 900px) {
  .block-container {padding-left:1rem; padding-right:1rem;}
  .hero {flex-direction:column; padding:18px 18px;} .hero-title {font-size:1.3rem;}
  .kpi-v {font-size:1.4rem;} .explain {grid-template-columns:1fr;}
}
</style>
"""


def inject_css():
    css = CSS
    for k, v in {"NAVY": NAVY, "TEAL": TEAL, "SKY": SKY, "AMBER": AMBER, "RED": RED, "SLATE": SLATE, "INK": INK, "LINE": LINE}.items():
        css = css.replace(f"__{k}__", v)
    st.markdown(css, unsafe_allow_html=True)  # static, developer-defined CSS only


def esc(x) -> str:
    return html.escape(str(x))


def rich(text) -> str:
    """Escape text, then turn **bold** markers into <b> (inline only)."""
    return re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", esc(text))


def html_block(s: str):
    st.markdown(s.replace("\n", ""), unsafe_allow_html=True)  # single-line HTML avoids Markdown code-block parsing


def hero(title: str, subtitle: str = "", chips: list | None = None):
    chip_html = "".join(f'<span class="hero-chip">{c}</span>' for c in (chips or []))
    chip_html = f'<div class="hero-chips">{chip_html}</div>' if chip_html else ""
    sub = f'<div class="hero-sub">{esc(subtitle)}</div>' if subtitle else ""
    html_block(f'<div class="hero"><div class="hero-main"><div class="hero-kicker">Fashion Quality Intelligence</div>'
               f'<div class="hero-title">{esc(title)}</div>{sub}{chip_html}</div>'
               f'<div class="hero-badge"><div class="n">{_PAGE[0] + 1}<span>/{len(PAGES)}</span></div><div class="l">Page</div></div></div>')


def section(title: str, sub: str = ""):
    s = f"<span>{esc(sub)}</span>" if sub else ""
    html_block(f'<div class="sec"><h3>{esc(title)}</h3>{s}</div>')


def ctrl_label(text: str):
    html_block(f'<div class="ctrl-label">⚙ {esc(text)}</div>')


def note(text: str, kind: str = "note"):
    html_block(f'<div class="callout {"caution" if kind == "caution" else ""}">{text}</div>')


def explain(what: str, why: str, use: str):
    html_block(f'<div class="explain"><div><h6>What it shows</h6><p>{esc(what)}</p></div>'
               f'<div><h6>Why it matters</h6><p>{esc(why)}</p></div><div><h6>Management use</h6><p>{esc(use)}</p></div></div>')


def panel(title: str, items: list):
    lis = "".join(f"<li>{rich(i)}</li>" for i in items)
    html_block(f'<div class="panel"><h4>{esc(title)}</h4><ul>{lis}</ul></div>')


def flow(steps: list):
    parts = [f'<span class="st"><i>{n}</i>{esc(s)}</span>' for n, s in enumerate(steps, 1)]
    html_block('<div class="flow">' + '<span class="ar">→</span>'.join(parts) + "</div>")


def style_fig(fig, height=380):
    pie = any(t.type == "pie" for t in fig.data)
    heat = any(t.type in ("heatmap", "image") for t in fig.data)
    named = [t for t in fig.data if t.showlegend is not False and getattr(t, "name", None)]
    legend_on = (not pie) and (fig.layout.showlegend is True or (fig.layout.showlegend is not False and len(named) > 1))
    long_legend = legend_on and sum(len(str(t.name)) for t in named) > 70  # many/long names -> legend goes below the plot
    fig.update_layout(height=height, margin=dict(l=12, r=30, t=60 if long_legend else (76 if legend_on else 60),
                                                 b=120 if long_legend else (48 if pie else 14)), template="plotly_white",
                      paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF", colorway=COLORWAY,
                      font=dict(family="Inter, Segoe UI, Arial, sans-serif", size=12, color=INK),
                      title=dict(x=0.01, xanchor="left", font=dict(size=15, color=NAVY)),
                      hoverlabel=dict(bgcolor="#FFFFFF", bordercolor="#C9D3E0", font_size=12, font_color=NAVY),
                      legend=dict(orientation="h", yanchor="bottom", y=1.01, x=0, font=dict(size=11)))
    if long_legend:
        fig.update_layout(legend=dict(orientation="h", yanchor="top", y=-0.32, x=0))
    if pie:
        fig.update_layout(legend=dict(orientation="h", yanchor="top", y=-0.02, x=0.5, xanchor="center"))
    grid = dict(showgrid=not heat, gridcolor="#EEF2F7", zeroline=False, linecolor="#D5DCE6", automargin=True, title_standoff=8)
    fig.update_xaxes(**grid)
    fig.update_yaxes(**grid)
    try:
        fig.update_yaxes(showgrid=False, selector=dict(overlaying="y"))  # secondary axes: no duplicate grid lines
    except Exception:  # noqa: BLE001 - cosmetic only
        pass
    for t in fig.data:
        if t.type == "bar" and t.text is not None and t.textposition in (None, "auto"):
            t.update(textposition="outside", cliponaxis=False, textfont=dict(size=11, color=INK))
    try:
        fig.update_coloraxes(colorbar=dict(thickness=12, len=0.85, outlinewidth=0))
    except Exception:  # noqa: BLE001 - cosmetic only
        pass
    return fig


def show(fig, height=380):
    _KEY[0] += 1
    st.plotly_chart(style_fig(fig, height), width="stretch", key=f"chart_{_KEY[0]}")


def kpi(col, label, value, help_text="", accent=TEAL, sub=""):
    v = str(value)
    size = "" if len(v) <= 16 else ("font-size:1rem;" if len(v) > 34 else "font-size:1.25rem;")
    tip = f' title="{esc(help_text)}"' if help_text else ""
    info = '<span class="kpi-i">ⓘ</span>' if help_text else ""
    with col:
        html_block(f'<div class="kpi" style="--accent:{accent}"{tip}><div><div class="kpi-l">{esc(label)}{info}</div>'
                   f'<div class="kpi-v" style="{size}">{esc(v)}</div></div><div class="kpi-s">{esc(sub) if sub else "&nbsp;"}</div></div>')


LONG_TEXT_COLS = {"Review Text": "large", "Description": "large", "Suggested action": "large", "Suggested Action": "large", "Evidence": "large",
                  "Business implication": "large", "Training parameters": "large", "Finding": "medium", "Input": "medium",
                  "Title": "medium", "Methods used": "large"}


def table(d, pct=(), **kw):
    """st.dataframe with consistent defaults: full width, wide text columns, optional in-cell bars for rate columns."""
    kw.setdefault("width", "stretch")
    if isinstance(d, pd.DataFrame) and "column_config" not in kw:
        cfg = {c: st.column_config.TextColumn(c, width=w) for c, w in LONG_TEXT_COLS.items() if c in d.columns}
        for c in pct:
            if c in d.columns and len(d):
                cfg[c] = st.column_config.ProgressColumn(c, min_value=0.0, max_value=float(max(d[c].max(), 1)), format="%.1f")
        if cfg:
            kw["column_config"] = cfg
    st.dataframe(d, **kw)


def fmt_pct(x, d=1):
    return "n/a" if pd.isna(x) else f"{100 * x:.{d}f}%"


def csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


# ----------------------------------------------------------------------------------------------
# Data loading and preparation (single source of truth lives in core.py)
# ----------------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------
# Cached data tiers (startup = CSV + validation + KPIs only)
# ----------------------------------------------------------------------------------------------
@st.cache_data(show_spinner="Reading CSV…")
def load_raw() -> pd.DataFrame:
    return core.read_csv_file()


@st.cache_resource(show_spinner="Preparing dataset…")
def get_base() -> pd.DataFrame:
    """Tier 1 — fast: validated columns, numeric coercion, flags. No NLP."""
    return core.prepare(load_raw())


@st.cache_resource(show_spinner="Cleaning review text (one-off, cached)…")
def get_text_df() -> pd.DataFrame:
    """Tier 2 — adds the cleaned text column (needed for n-grams, topics, ML)."""
    return core.add_clean(get_base())


@st.cache_resource
def shared_state() -> dict:
    return {"enriched": False}


@st.cache_resource(show_spinner="Scoring VADER sentiment and tagging complaint themes (one-off, cached)…")
def get_enriched() -> pd.DataFrame:
    """Tier 3 — adds VADER sentiment and negation-aware complaint tags."""
    out = add_complaints(add_sentiment(get_text_df()))
    shared_state()["enriched"] = True
    return out


@st.cache_resource(show_spinner="Building n-gram and TF-IDF matrices (one-off, cached)…")
def text_matrices():
    return core.build_text_matrices(get_text_df())


@st.cache_data(show_spinner=False, max_entries=24)
def make_cloud(freq_items: tuple, colormap: str):
    return core.make_wordcloud(freq_items, colormap)


@st.cache_resource(show_spinner=False)
def lda_cached(n_topics: int, max_iter: int, seed: int):
    return core.fit_lda(get_text_df(), n_topics, max_iter, seed)


@st.cache_resource(show_spinner=False)
def classical_cached(sample_n, C: float, seed: int = SEED):
    return core.train_classical(get_text_df(), sample_n, C, seed)


def run_deep_learning(models, sample_n, epochs, batch, maxlen, vocab_size):
    overall = st.progress(0.0)
    status = st.empty()

    def cb(mi, n_models, name, epoch, total, logs):
        overall.progress(min((mi + epoch / total) / n_models, 1.0))
        status.info(f"{name}: epoch {epoch}/{total} — loss {logs.get('loss', 0):.4f}, val_loss {logs.get('val_loss', 0):.4f}")

    res = core.train_deep(get_text_df(), models, sample_n, epochs, batch, maxlen, vocab_size, progress_cb=cb)
    st.session_state.setdefault("dl", {}).update(res)
    overall.progress(1.0)
    status.success("Deep-learning training finished.")


def get_analyzer():
    return core.get_analyzer()


# ----------------------------------------------------------------------------------------------
# Sidebar filters
# ----------------------------------------------------------------------------------------------
def reset_filters():
    for key in [k for k in st.session_state.keys() if isinstance(k, str) and k.startswith("f_")]:
        del st.session_state[key]


def sidebar_filters(df: pd.DataFrame) -> pd.DataFrame:
    with st.sidebar:
        html_block('<div class="side-label">Filters · this page</div>')
        sel_div = st.multiselect("Division", sorted(df["Division Name"].unique()), key="f_div",
                                 help="Leave empty for all divisions.")
        d1 = df[df["Division Name"].isin(sel_div)] if sel_div else df
        sel_dep = st.multiselect("Department", sorted(d1["Department Name"].unique()),
                                 key="f_dep_" + "|".join(sel_div))
        d2 = d1[d1["Department Name"].isin(sel_dep)] if sel_dep else d1
        sel_cls = st.multiselect("Clothing class", sorted(d2["Class Name"].unique()),
                                 key="f_cls_" + "|".join(sel_div) + "#" + "|".join(sel_dep))
        html_block('<div class="side-label">Customer & review</div>')
        amin, amax = int(df["Age"].min()), int(df["Age"].max())
        age = st.slider("Age range", amin, amax, (amin, amax), key="f_age")
        sel_rat = st.multiselect("Rating", [1, 2, 3, 4, 5], key="f_rat")
        rec = st.selectbox("Recommendation status", ["All", "Recommended", "Not recommended"], key="f_rec")
        sel_sent = []
        if "sent_class" in df.columns:
            sel_sent = st.multiselect("Sentiment class (VADER, ±0.05)", ["Positive", "Neutral", "Negative"], key="f_sent",
                                      help="Reviews without text are excluded when a sentiment class is chosen.")
    m = pd.Series(True, index=df.index)
    if sel_div: m &= df["Division Name"].isin(sel_div)
    if sel_dep: m &= df["Department Name"].isin(sel_dep)
    if sel_cls: m &= df["Class Name"].isin(sel_cls)
    if age != (amin, amax): m &= df["Age"].between(*age)
    if sel_rat: m &= df["Rating"].isin(sel_rat)
    if rec == "Recommended": m &= df["Recommended IND"] == 1
    if rec == "Not recommended": m &= df["Recommended IND"] == 0
    if sel_sent: m &= df["sent_class"].isin(sel_sent)
    out = df[m]
    share = 100 * len(out) / max(len(df), 1)
    with st.sidebar:
        html_block(f'<div class="side-count"><b>{len(out):,}</b> of {len(df):,} reviews selected'
                   f'<div class="bar"><span style="width:{share:.1f}%"></span></div></div>')
        st.button("↺ Reset filters", on_click=reset_filters, key="btn_reset_filters")
    return out


# ----------------------------------------------------------------------------------------------
# PAGE 1 — Executive dashboard
# ----------------------------------------------------------------------------------------------
def page_exec(df):
    k = core_kpis(df)
    hero("AI-Driven Product Quality Intelligence and Customer Dissatisfaction Prediction in Fashion E-Commerce",
         "Text Analytics, Deep Learning and Six Sigma–Based Quality Improvement",
         [f"<b>{k['n']:,}</b> reviews", f"<b>{k['products']:,}</b> products", f"avg rating <b>{k['avg']:.2f}</b> / 5",
          f"<b>{fmt_pct(k['rec'])}</b> recommend"])

    r1 = st.columns(4)
    kpi(r1[0], "Total reviews", f"{k['n']:,}", "All rows with a valid rating and recommendation flag.", SKY, "valid rating + recommendation flag")
    kpi(r1[1], "Unique clothing products", f"{k['products']:,}", "Distinct Clothing IDs.", SKY, "distinct Clothing IDs")
    kpi(r1[2], "Average rating", f"{k['avg']:.2f} / 5", "Mean star rating.", AMBER, "mean star rating")
    kpi(r1[3], "Recommendation rate", fmt_pct(k["rec"]), "Share of reviews with Recommended IND = 1.", TEAL, "Recommended IND = 1")
    r2 = st.columns(4)
    kpi(r2[0], "Non-recommendation rate", fmt_pct(k["nonrec"]), "Share of reviews with Recommended IND = 0.", RED, "Recommended IND = 0")
    kpi(r2[1], "Reviews with usable text", f"{k['text_n']:,}", f"{fmt_pct(k['text_n'] / k['n'])} of all reviews.", NAVY,
        f"{fmt_pct(k['text_n'] / k['n'])} of all reviews")
    kpi(r2[2], "Negative sentiment", "—" if pd.isna(k["neg"]) else fmt_pct(k["neg"]),
        "VADER compound ≤ −0.05, among reviews with text. A lexicon estimate, not ground truth. Shows “—” until the sentiment analysis has been run (button below).",
        RED, "VADER, among reviews with text" if not pd.isna(k["neg"]) else "run the analysis below to populate")
    kpi(r2[3], "Review-based DPMO proxy", f"{k['dpmo']:,.0f}",
        "Non-recommended reviews per million review opportunities (1 opportunity per review). A customer-dissatisfaction proxy, NOT a factory defect rate.",
        AMBER, "dissatisfaction proxy, not a defect rate")
    note("<b>Reading the KPIs:</b> the DPMO figure is a <i>review-based customer dissatisfaction proxy</i>. It treats every non-recommendation as one 'defect' "
         "in one opportunity. It does not measure verified manufacturing defects, returns or revenue loss.")

    section("Customer satisfaction snapshot", "How customers rate and recommend")
    c1, c2 = st.columns([3, 2])
    with c1:
        rc = df["Rating"].value_counts().sort_index().reset_index()
        rc.columns = ["Rating", "Reviews"]
        fig = px.bar(rc, x="Rating", y="Reviews", text="Reviews", title="Rating distribution",
                     color=rc["Rating"].map(RATING_COLORS), color_discrete_map="identity")
        fig.update_traces(texttemplate="%{text:,}", hovertemplate="Rating %{x}: %{y:,} reviews<extra></extra>")
        fig.update_layout(showlegend=False, bargap=0.25)
        fig.update_xaxes(dtick=1)
        show(fig, 400)
    with c2:
        sp = df["Recommended IND"].map({1: "Recommended", 0: "Not recommended"}).value_counts().reset_index()
        sp.columns = ["Status", "Reviews"]
        fig = px.pie(sp, names="Status", values="Reviews", hole=0.62, title="Recommendation split",
                     color="Status", color_discrete_map={"Recommended": TEAL, "Not recommended": RED})
        fig.update_traces(textinfo="percent", sort=False, hovertemplate="%{label}: %{value:,} reviews (%{percent})<extra></extra>")
        fig.add_annotation(text=f"<b>{fmt_pct(k['rec'])}</b><br>recommend", x=0.5, y=0.5, showarrow=False, font=dict(size=15, color=NAVY))
        show(fig, 400)

    section("Where dissatisfaction concentrates", "Departments and recurring complaint themes")
    c3, c4 = st.columns(2)
    dep = seg_table(df, "Department Name", min_n=30)
    with c3:
        dsort = dep.sort_values("nonrec_rate")
        fig = px.bar(dsort, x="nonrec_rate", y="Department Name", orientation="h",
                     text=dsort["nonrec_rate"].map(lambda v: f"{100 * v:.1f}%"),
                     title="Department-level dissatisfaction (non-recommendation rate)",
                     custom_data=["n"], labels={"nonrec_rate": "Non-recommendation rate", "Department Name": ""})
        fig.update_traces(marker_color=AMBER, hovertemplate="%{y}: %{x:.1%} (n=%{customdata[0]:,})<extra></extra>")
        fig.add_vline(x=k["nonrec"], line_dash="dot", line_color=NAVY, annotation_text=f"Overall {fmt_pct(k['nonrec'])}",
                      annotation_position="bottom right", annotation_font=dict(size=11, color=NAVY))
        fig.update_xaxes(tickformat=".0%")
        show(fig, 400)
    has_nlp = "any_complaint" in df.columns
    tt = theme_table(df) if has_nlp else pd.DataFrame()
    with c4:
        if not has_nlp:
            with st.container(border=True):
                st.markdown("**Complaint themes & sentiment**")
                st.info("Sentiment and complaint-theme analysis is computed on demand (about 15–30 s the first time, then cached for everyone).")
                if st.button("▶ Run sentiment & complaint analysis", type="primary"):
                    get_enriched()
                    st.rerun()
        else:
            fig = px.bar(tt.sort_values("n_flagged"), x="n_flagged", y="Theme", orientation="h", text="n_flagged",
                         title="Top recurring complaint themes (rule-based, negation-aware)",
                         labels={"n_flagged": "Reviews tagged", "Theme": ""})
            fig.update_traces(marker_color=SKY, texttemplate="%{text:,}")
            show(fig, 400)

    section("Major findings", "Computed from the data at run time")
    top_dep = dep.iloc[0] if len(dep) else None
    cls = seg_table(df, "Class Name", min_n=100)
    top_cls = cls.iloc[0] if len(cls) else None
    lines = [f"**{k['n']:,} reviews** across **{k['products']:,} products**; average rating **{k['avg']:.2f}**, "
             f"**{fmt_pct(k['rec'])}** recommended and **{fmt_pct(k['nonrec'])}** not recommended."]
    if top_dep is not None:
        lines.append(f"Highest non-recommendation rate among departments (n ≥ 30): **{top_dep['Department Name']}** at "
                     f"**{fmt_pct(top_dep['nonrec_rate'])}** (n = {int(top_dep['n']):,}).")
    if top_cls is not None:
        lines.append(f"Highest non-recommendation rate among classes with ≥ 100 reviews: **{top_cls['Class Name']}** at "
                     f"**{fmt_pct(top_cls['nonrec_rate'])}** (n = {int(top_cls['n']):,}).")
    if len(tt) and tt.iloc[0]["n_flagged"] > 0:
        t0 = tt.iloc[0]
        lines.append(f"The complaint theme most associated with extra non-recommendations is **{t0['Theme']}**: "
                     f"{int(t0['n_flagged']):,} tagged reviews, non-recommendation rate {fmt_pct(t0['nonrec_rate_flagged'])} "
                     f"vs {fmt_pct(t0['nonrec_rate_other'])} for other reviews (association, not causation).")
    panel("Key takeaways", lines)

    section("Prioritised management actions", "Top three complaint themes by associated excess non-recommendations")
    top3 = list(tt.head(3).iterrows())
    if not top3:
        note("Prioritised actions appear here once the sentiment & complaint analysis has been run (button above).")
    else:
        cols = st.columns(3)
        for i, (_, r) in enumerate(top3):
            a, owner, kp = RECS[r["Theme"]]
            with cols[i]:
                html_block(f'<div class="rec"><span class="rk">Priority {i + 1}</span><div class="th">{esc(r["Theme"])}</div>'
                           f'<div class="ac">{esc(a)}</div><div class="mt"><b>Owner:</b> {esc(owner)}<br><b>KPI:</b> {esc(kp)}</div></div>')
    st.caption("Actions are proposals derived from association patterns; they have not been tested and no improvement is claimed.")

    section("Analytical workflow")
    flow(["Customer reviews", "Data cleaning", "EDA", "Sentiment", "Topic modelling", "Machine learning",
          "Deep learning", "Six Sigma", "Management decisions"])


# ----------------------------------------------------------------------------------------------
# PAGE 2 — Objectives and data audit
# ----------------------------------------------------------------------------------------------
def page_objectives(df):
    raw = load_raw()
    hero("Business Objectives & Data Audit", "Problem framing, methodology, data dictionary and data-quality evidence")
    t1, t2, t3, t4 = st.tabs(["Objectives & methodology", "Course alignment", "Data audit", "Column explorer"])
    with t1:
        c1, c2 = st.columns(2)
        with c1:
            with st.container(border=True):
                section("Business problem")
                st.write("How can an anonymised online women's-fashion retailer use customer reviews, ratings, product categories and "
                         "recommendation indicators to identify dissatisfaction themes, anticipate recommendation intent, prioritise "
                         "product-quality concerns and support operational improvement?")
        with c2:
            with st.container(border=True):
                section("Technical problem")
                st.write("Convert unstructured review text into structured quality intelligence: clean and explore the data, measure sentiment, "
                         "discover topics, classify `Recommended IND` from written feedback with classical ML and neural networks, "
                         "and translate results into Six Sigma–style priorities.")
        section("Research questions")
        st.markdown("1. Which departments, classes and products show the highest dissatisfaction, given adequate sample sizes?\n"
                    "2. Which complaint themes recur in reviews, and how strongly are they associated with non-recommendation?\n"
                    "3. How well can review text predict recommendation, especially for the minority non-recommended class?\n"
                    "4. Do neural sequence models (LSTM/GRU) add value over TF-IDF linear baselines?\n"
                    "5. Which issues should management prioritise, and what should be monitored?")
        section("Methodology overview")
        flow(["Clean", "EDA", "Text analytics (n-grams, TF-IDF, co-occurrence)", "VADER sentiment", "LDA topics + rule-based complaint tags", "ML/DL classification", "Pareto/DPMO/DMAIC", "Priority matrix and scenario calculator"])
        note("All models predict <code>Recommended IND</code> from review text only. <code>Rating</code> is never a model input (target-leakage control).")
    with t2:
        table(pd.DataFrame([
            ("Text analytics", "Pre-processing, unigrams, bigrams, co-occurrence, sentiment, topic modelling, classification", "Pages 4, 5, 6, 7"),
            ("Neural networks", "ANN, training, dropout, early stopping, LSTM, GRU", "Page 7"),
            ("Operations analytics", "Pareto analysis, review-based DPMO, quality metrics, Six Sigma DMAIC", "Pages 8, 9"),
            ("Evaluation criteria", "Objectives · Data requirements · EDA · Modelling · Results & recommendations", "Pages 2 · 2 · 3 · 4–7 · 8–9")],
            columns=["Syllabus area", "Methods used", "Where in the app"]), width="stretch", hide_index=True)
    with t3:
        k = core_kpis(df)
        c = st.columns(4)
        kpi(c[0], "Rows in CSV", f"{len(raw):,}")
        kpi(c[1], "Rows analysed", f"{len(df):,}", "Rows lacking Rating or Recommended IND are dropped.")
        kpi(c[2], "Columns", f"{raw.shape[1]}")
        kpi(c[3], "Rows without review text", f"{len(df) - k['text_n']:,}")
        section("Data dictionary")
        table(DATA_DICT, width="stretch", hide_index=True)
        miss = raw.isna().sum().reset_index()
        miss.columns = ["Column", "Missing"]
        miss["Missing %"] = (100 * miss["Missing"] / len(raw)).round(2)
        blank = {c: int((raw[c].astype(str).str.strip() == "").sum()) for c in raw.columns}
        miss["Blank strings"] = miss["Column"].map(blank)
        a, b = st.columns([1, 1])
        with a:
            section("Missing values")
            table(miss, width="stretch", hide_index=True)
        with b:
            fig = px.bar(miss.sort_values("Missing %"), x="Missing %", y="Column", orientation="h",
                         title="Missing values by column (%)")
            fig.update_traces(marker_color=AMBER)
            show(fig, 360)
        section("Variable data types (as loaded)")
        table(pd.DataFrame({"Column": raw.columns, "dtype": raw.dtypes.astype(str).values}), width="stretch", hide_index=True)
        section("Duplicate analysis")
        with_text = df[df["has_text"]]
        dup_text = int(with_text.duplicated("full_text").sum())
        dup_rows = int(raw.drop(columns=[c for c in raw.columns if str(c).startswith("Unnamed")]).duplicated().sum())
        d = st.columns(3)
        kpi(d[0], "Exact duplicate rows", f"{dup_rows:,}")
        kpi(d[1], "Repeated review texts", f"{dup_text:,}", "Rows whose title+text exactly repeats another row.")
        kpi(d[2], "Share of text reviews", fmt_pct(dup_text / max(len(with_text), 1), 2))
        ca, lm = st.columns(2)
        with ca:
            section("Data-cleaning approach")
            st.markdown("- Numeric coercion for Age, Rating, Recommended IND, Positive Feedback Count; rows missing the rating or recommendation flag are dropped.\n"
                        "- Missing Division/Department/Class labelled `Unknown`; blank titles/texts treated as empty.\n"
                        "- Text = title + review text. Lower-casing, contraction expansion (n't → not), punctuation/digit removal, stop-word removal "
                        "**while preserving negations** (not, no, never, nothing, too, very).\n"
                        "- Duplicates are kept for descriptive analytics but removed (by exact title+text) before machine-learning splits to avoid train/test leakage.\n"
                        "- Low rating = Rating ≤ 2; non-recommended = Recommended IND = 0.")
        with lm:
            section("Limitations")
            st.markdown("- Reviews are self-selected; they do not represent all customers or purchases.\n"
                        "- No dates, order, return or cost data: no trends, return rates or financial impact can be derived.\n"
                        "- Non-recommendation is a proxy for dissatisfaction, not a verified defect.\n"
                        "- Lexicon sentiment and keyword complaint tags are heuristics and may misclassify sarcasm or mixed reviews.\n"
                        "- Clothing ID-level results can rest on small samples; confidence intervals and review-count thresholds are shown.")
        st.download_button("Download data-quality summary (CSV)", csv_bytes(miss), "data_quality_summary.csv", "text/csv")
        with st.expander("Environment check (installed analytics packages)"):
            if st.button("Run environment check"):
                rows = core.dependency_report() + core.dependency_report(optional=True)
                table(pd.DataFrame([{"module": m, "pip package": p, "status": "OK" if ok else "missing/failed", "version or error": v} for m, p, ok, v in rows]),
                             hide_index=True, width="stretch")
                st.caption("keras/jax are optional: they are only needed to train ANN/LSTM/GRU.")
    with t4:
        col = st.selectbox("Inspect column", [c for c in df.columns if c not in ("clean", "full_text", "row_id")])
        s = df[col]
        if pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s):
            table(s.astype(float).describe().to_frame().T, width="stretch")
            show(px.histogram(s.astype(float), x=col, nbins=40, title=f"Distribution of {col}"), 320)
        else:
            vc = s.astype(str).value_counts().head(25).reset_index()
            vc.columns = [col, "Count"]
            table(vc, width="stretch", hide_index=True)
            show(px.bar(vc, x="Count", y=col, orientation="h", title=f"Top values of {col}"), 420)


# ----------------------------------------------------------------------------------------------
# PAGE 3 — EDA
# ----------------------------------------------------------------------------------------------
def page_eda(dff):
    hero("Customer Satisfaction & Exploratory Data Analysis", "Filters in the sidebar update every chart and summary on this page")
    if dff.empty:
        st.warning("No reviews match the current filters. Relax one or more filters in the sidebar.")
        return
    k = core_kpis(dff)
    s = st.columns(5)
    kpi(s[0], "Reviews in selection", f"{k['n']:,}", accent=SKY)
    kpi(s[1], "Average rating", f"{k['avg']:.2f} / 5", accent=AMBER)
    kpi(s[2], "Recommended", fmt_pct(k["rec"]), accent=TEAL)
    kpi(s[3], "Low rating (≤ 2★)", fmt_pct(k["low"]), accent=RED)
    kpi(s[4], "Products", f"{k['products']:,}", accent=NAVY)
    with st.container(border=True):
        ctrl_label("Display controls")
        c1, c2 = st.columns(2)
        min_seg = c1.slider("Minimum segment size to display percentages", 10, 300, 30, 5,
                            help="Segments smaller than this are hidden to avoid misleading rates.")
        min_prod = c2.slider("Minimum reviews per product (Clothing ID)", 1, 60, 10)
    if len(dff) < 50:
        note("Very small selection — rates may be unstable.", "caution")
    t1, t2, t3, t4 = st.tabs(["Ratings & recommendation", "Category views", "Customers & reviews", "Products"])
    with t1:
        a, b = st.columns(2)
        with a:
            rc = dff["Rating"].value_counts().sort_index().reset_index()
            rc.columns = ["Rating", "Reviews"]
            fig = px.bar(rc, x="Rating", y="Reviews", text="Reviews", title="1 · Rating distribution",
                         color=rc["Rating"].map(RATING_COLORS), color_discrete_map="identity")
            fig.update_layout(showlegend=False)
            show(fig)
        with b:
            ct = pd.crosstab(dff["Rating"], dff["Recommended IND"].map({1: "Recommended", 0: "Not recommended"}), normalize="index")
            fig = px.imshow(ct * 100, text_auto=".1f", aspect="auto", color_continuous_scale=["#F8FAFC", TEAL],
                            title="11 · Rating vs recommendation (row %)", labels=dict(color="%"))
            show(fig)
        dep = pd.crosstab(dff["Department Name"], dff["Rating"], normalize="index").reset_index().melt(id_vars="Department Name", var_name="Rating", value_name="Share")
        dep["Rating"] = dep["Rating"].astype(str)
        fig = px.bar(dep, x="Department Name", y="Share", color="Rating", title="2 · Rating mix by department (share of reviews)",
                     color_discrete_map={str(k): v for k, v in RATING_COLORS.items()})
        fig.update_yaxes(tickformat=".0%")
        show(fig)
        explain("How ratings and the recommendation flag relate, and how rating mix differs by department.",
                "Shows whether low ratings and non-recommendation coincide, validating the proxy.", "Direct investigation toward departments with heavier 1–2 star mass.")
    with t2:
        dt = seg_table(dff, "Department Name", min_seg)
        ct_ = seg_table(dff, "Class Name", min_seg)
        if dt.empty or ct_.empty:
            st.info("No segment reaches the minimum size under the current filters.")
        else:
            a, b = st.columns(2)
            with a:
                fig = px.bar(dt.sort_values("rec_rate"), x="rec_rate", y="Department Name", orientation="h", custom_data=["n"],
                             title="3 · Recommendation rate by department", labels={"rec_rate": "Recommendation rate", "Department Name": ""})
                fig.update_traces(marker_color=TEAL, hovertemplate="%{y}: %{x:.1%} (n=%{customdata[0]:,})<extra></extra>")
                fig.update_xaxes(tickformat=".0%", range=[0, 1])
                show(fig)
            with b:
                sort_by = st.selectbox("Sort classes by", ["Average rating", "Non-recommendation rate", "Low-rating rate", "Reviews"])
                col = {"Average rating": "avg_rating", "Non-recommendation rate": "nonrec_rate", "Low-rating rate": "low_rate", "Reviews": "n"}[sort_by]
                fig = px.bar(ct_.sort_values(col), x="avg_rating", y="Class Name", orientation="h", color="nonrec_rate", custom_data=["n", "nonrec_rate"],
                             color_continuous_scale=["#5FC9B8", AMBER, RED], title="4 · Satisfaction by clothing class (avg rating; colour = non-rec. rate)",
                             labels={"avg_rating": "Average rating", "Class Name": "", "nonrec_rate": "Non-rec. rate"})
                fig.update_traces(hovertemplate="%{y}: rating %{x:.2f}, non-rec %{customdata[1]:.1%} (n=%{customdata[0]:,})<extra></extra>")
                show(fig, 460)
            lr = pd.concat([seg_table(dff, "Department Name", min_seg).assign(Level="Department").rename(columns={"Department Name": "Segment"}),
                            ct_.assign(Level="Class").rename(columns={"Class Name": "Segment"})])
            fig = px.bar(lr.sort_values("low_rate"), x="low_rate", y="Segment", color="Level", orientation="h", custom_data=["n"],
                         title="12 · Low-rating (≤2★) incidence by category", labels={"low_rate": "Low-rating share"})
            fig.update_traces(hovertemplate="%{y}: %{x:.1%} (n=%{customdata[0]:,})<extra></extra>")
            fig.update_xaxes(tickformat=".0%")
            show(fig, 520)
            explain(f"Segment rates for groups with at least {min_seg} reviews.", "Small segments produce unstable percentages.",
                    "Focus quality reviews on large segments with above-average dissatisfaction.")
    with t3:
        a, b = st.columns(2)
        with a:
            show(px.histogram(dff.dropna(subset=["Age"]), x="Age", nbins=40, title="5 · Customer age distribution", color_discrete_sequence=[SKY]))
        with b:
            ag = seg_table(dff.dropna(subset=["age_group"]), "age_group", min_seg)
            if ag.empty:
                st.info("Age groups are below the minimum size.")
            else:
                ag = ag.sort_values("age_group")
                fig = go.Figure()
                fig.add_bar(x=ag["age_group"].astype(str), y=ag["avg_rating"], name="Average rating", marker_color=TEAL)
                fig.add_scatter(x=ag["age_group"].astype(str), y=ag["nonrec_rate"] * 100, name="Non-rec. rate (%)", yaxis="y2", mode="lines+markers", line=dict(color=RED))
                fig.update_layout(title="6 · Age group vs ratings and dissatisfaction", yaxis=dict(title="Average rating", range=[0, 5]),
                                  yaxis2=dict(title="Non-rec. %", overlaying="y", side="right", rangemode="tozero"))
                show(fig)
        a, b = st.columns(2)
        with a:
            tx = dff[dff["has_text"]]
            if len(tx):
                fig = px.box(tx, x="Rating", y="n_words", color=tx["Recommended IND"].map({1: "Recommended", 0: "Not recommended"}),
                             title="7 · Review length (words) by rating", labels={"n_words": "Words", "color": ""},
                             color_discrete_map={"Recommended": TEAL, "Not recommended": RED})
                show(fig)
        with b:
            pf = dff.groupby("Rating")["Positive Feedback Count"].agg(["mean", "median"]).reset_index()
            fig = go.Figure()
            fig.add_bar(x=pf["Rating"], y=pf["mean"], name="Mean", marker_color=SKY)
            fig.add_bar(x=pf["Rating"], y=pf["median"], name="Median", marker_color=AMBER)
            fig.update_layout(barmode="group", title="8 · Positive feedback count by rating", xaxis_title="Rating", yaxis_title="Votes per review")
            show(fig)
        explain("Who reviews and how, and which reviews get the most helpful votes.", "Long, low-rated reviews often carry richer defect information.",
                "Prioritise reading long low-rated reviews with high helpful-vote counts.")
    with t4:
        pt = seg_table(dff, "Clothing ID", min_prod)
        if pt.empty:
            st.info("No product reaches the minimum review count under the current filters.")
        else:
            pt["Clothing ID"] = pt["Clothing ID"].astype(int).astype(str)
            a, b = st.columns(2)
            with a:
                show(px.histogram(pt, x="nonrec_rate", nbins=25, title=f"9 · Non-recommendation rate across {len(pt):,} products (≥{min_prod} reviews)",
                                  labels={"nonrec_rate": "Non-recommendation rate"}, color_discrete_sequence=[AMBER]))
            with b:
                topn = st.slider("Products in ranking", 5, 30, 15)
                rk = st.selectbox("Rank by", ["Highest non-recommendation rate", "Lowest average rating"])
                top = pt.nlargest(topn, "nonrec_rate") if rk.startswith("Highest") else pt.nsmallest(topn, "avg_rating")
                fig = px.bar(top.iloc[::-1], x="nonrec_rate", y="Clothing ID", orientation="h", custom_data=["n", "ci_low", "ci_high", "avg_rating"],
                             title="10 · Product ranking (review-count safeguard applied)", labels={"nonrec_rate": "Non-rec. rate"})
                fig.update_traces(marker_color=RED, hovertemplate="ID %{y}: %{x:.1%} (95%% CI %{customdata[1]:.1%}–%{customdata[2]:.1%}), n=%{customdata[0]}, rating %{customdata[3]:.2f}<extra></extra>")
                fig.update_xaxes(tickformat=".0%")
                show(fig, 460)
            show_cols = pt[["Clothing ID", "n", "avg_rating", "nonrec_rate", "ci_low", "ci_high"]].copy()
            for c in ["nonrec_rate", "ci_low", "ci_high"]:
                show_cols[c] = (100 * show_cols[c]).round(1)
            show_cols["avg_rating"] = show_cols["avg_rating"].round(2)
            show_cols.columns = ["Clothing ID", "Reviews", "Avg rating", "Non-rec %", "CI low %", "CI high %"]
            table(show_cols, pct=("Non-rec %",), width="stretch", hide_index=True)
            st.download_button("Download product table (CSV)", csv_bytes(show_cols), "product_risk_table.csv", "text/csv")


# ----------------------------------------------------------------------------------------------
# PAGE 4 — Text analytics
# ----------------------------------------------------------------------------------------------
def page_text(dff):
    hero("Text Analytics", "Pre-processing, n-grams, TF-IDF, word clouds and co-occurrence on the filtered reviews")
    d = dff[dff["has_text"] & (dff["clean"].str.len() > 0)]
    if len(d) < 30:
        st.warning("Fewer than 30 reviews with text match the current filters; widen the sidebar filters.")
        return
    mats = text_matrices()
    ids = d["row_id"].to_numpy()
    with st.container(border=True):
        ctrl_label("Text-analysis controls")
        c1, c2, c3, c4 = st.columns(4)
        mode = c1.radio("N-gram mode", ["Unigrams", "Bigrams"], horizontal=True)
        topn = c2.slider("Top N terms", 5, 40, 20)
        minf = c3.slider("Minimum term frequency", 1, 100, 5)
        kco = c4.slider("Terms in co-occurrence", 8, 30, 18)
    M = mats["M1"] if mode == "Unigrams" else mats["M2"]
    terms = mats["t1"] if mode == "Unigrams" else mats["t2"]
    sub = M[ids]
    cnt = np.asarray(sub.sum(axis=0)).ravel()
    dfreq = np.asarray((sub > 0).sum(axis=0)).ravel()

    section("Pre-processing summary")
    neg_rx = r"\b(?:not|no|never|nothing)\b|n't"
    neg_share = d["full_text"].str.lower().str.contains(neg_rx, regex=True).mean()
    tok = d["clean"].str.split().str.len()
    s = st.columns(4)
    kpi(s[0], "Reviews analysed", f"{len(d):,}")
    kpi(s[1], "Avg tokens after cleaning", f"{tok.mean():.1f}")
    kpi(s[2], "Vocabulary (≥3 docs, whole corpus)", f"{len(mats['t1']):,}")
    kpi(s[3], "Reviews containing negation", fmt_pct(neg_share), "Negations are preserved in cleaning.")
    note("Steps: title+text merge → lower-case → contraction expansion → punctuation/digit removal → stop-word removal (negations and 'too/very/enough' kept) → n-gram counting. "
         "Page-4 selections (department, class, recommendation) come from the sidebar filters.")

    order = np.argsort(-cnt)
    order = [i for i in order if cnt[i] >= minf][:topn]
    if not order:
        st.info("No terms meet the minimum frequency. Lower the threshold.")
    else:
        top = pd.DataFrame({"Term": terms[order], "Frequency": cnt[order], "Reviews containing": dfreq[order]})
        top["% of reviews"] = (100 * top["Reviews containing"] / len(d)).round(1)
        a, b = st.columns(2)
        with a:
            fig = px.bar(top.iloc[::-1], x="Frequency", y="Term", orientation="h", custom_data=["% of reviews"],
                         title=f"Top {len(top)} {mode.lower()}")
            fig.update_traces(marker_color=SKY, hovertemplate="%{y}: %{x:,} (in %{customdata[0]}% of reviews)<extra></extra>")
            show(fig, 120 + 22 * len(top))
        with b:
            tf = np.asarray(mats["T"][ids].mean(axis=0)).ravel()
            o2 = np.argsort(-tf)[:topn]
            tdf = pd.DataFrame({"Term": mats["tt"][o2], "Mean TF-IDF": tf[o2]})
            fig = px.bar(tdf.iloc[::-1], x="Mean TF-IDF", y="Term", orientation="h", title=f"Top {topn} TF-IDF terms (1–2-grams)")
            fig.update_traces(marker_color=TEAL)
            show(fig, 120 + 22 * topn)
        st.download_button("Download n-gram table (CSV)", csv_bytes(top), "ngrams.csv", "text/csv")

    section("Terms over-represented in non-recommended reviews")
    nr, rc = d[d["non_rec"] == 1], d[d["non_rec"] == 0]
    if len(nr) >= 20 and len(rc) >= 20:
        dn = np.asarray((M[nr["row_id"].to_numpy()] > 0).sum(axis=0)).ravel()
        dr = np.asarray((M[rc["row_id"].to_numpy()] > 0).sum(axis=0)).ravel()
        lift = ((dn + 0.5) / len(nr)) / ((dr + 0.5) / len(rc))
        ok = np.where(dn >= max(minf, 5))[0]
        ok = ok[np.argsort(-lift[ok])][:topn]
        ov = pd.DataFrame({"Term": terms[ok], "In non-rec reviews": dn[ok], "In recommended": dr[ok], "Relative lift": lift[ok].round(2)})
        table(ov, width="stretch", hide_index=True)
        explain("Terms whose document frequency is much higher among non-recommended reviews (smoothed ratio).",
                "Separates distinctive complaint language from words common to every review.", "Use as a vocabulary seed for complaint monitoring.")
    else:
        st.info("Need at least 20 recommended and 20 non-recommended reviews in the selection for this comparison.")

    section("Word clouds (unigram frequencies)")
    wa, wb = st.columns(2)
    for col, label, sub_d, cmap in ((wa, "Positive (rating ≥ 4)", d[d["Rating"] >= 4], "Greens"), (wb, "Negative (rating ≤ 2)", d[d["Rating"] <= 2], "Reds")):
        with col:
            st.markdown(f"**{label}** · n = {len(sub_d):,}")
            if len(sub_d) < 20:
                st.info("Too few reviews for a word cloud.")
                continue
            if importlib.util.find_spec("wordcloud") is None:
                st.warning("`wordcloud` package unavailable.")
                continue
            c_ = np.asarray(mats["M1"][sub_d["row_id"].to_numpy()].sum(axis=0)).ravel()
            ix = np.argsort(-c_)[:100]
            st.image(make_cloud(tuple((str(mats["t1"][i]), int(c_[i])) for i in ix if c_[i] > 0), cmap), width="stretch")

    section("Word co-occurrence")
    uni = np.asarray(M1_counts(mats, ids))
    tops = np.argsort(-uni)[:kco]
    B = (mats["M1"][ids][:, tops] > 0).astype(np.int32)
    C = (B.T @ B).toarray()
    np.fill_diagonal(C, 0)
    labels = mats["t1"][tops]
    fig = px.imshow(C, x=labels, y=labels, color_continuous_scale=["#F8FAFC", SKY, NAVY], aspect="auto",
                    title="Co-occurrence heat-map (reviews containing both words)")
    show(fig, 520)
    n_edges = st.slider("Edges in association network", 10, 120, 45)
    network_plot(C, labels, uni[tops], n_edges)
    explain("Which frequent words appear together in the same review, and the strongest word pairs as a network.",
            "Co-occurring words such as fit/size or fabric/quality words hint at the attribute a complaint is about.",
            "Use dense clusters to define complaint dictionaries and review categories.")


def M1_counts(mats, ids):
    return np.asarray(mats["M1"][ids].sum(axis=0)).ravel()


def network_plot(C, labels, freq, n_edges):
    if importlib.util.find_spec("networkx") is None:
        st.warning("`networkx` package unavailable; network graph skipped.")
        return
    import networkx as nx
    iu = np.triu_indices_from(C, 1)
    w = C[iu]
    order = np.argsort(-w)[:n_edges]
    G = nx.Graph()
    G.add_nodes_from(range(len(labels)))
    for o in order:
        if w[o] > 0:
            G.add_edge(int(iu[0][o]), int(iu[1][o]), weight=float(w[o]))
    if G.number_of_edges() == 0:
        st.info("No co-occurring pairs to display.")
        return
    pos = nx.spring_layout(G, seed=SEED, weight="weight")
    ws = np.array([d["weight"] for _, _, d in G.edges(data=True)])
    q = np.quantile(ws, [0.33, 0.66]) if len(ws) > 2 else [0, 0]
    fig = go.Figure()
    for lo, hi, width, col in ((-1, q[0], 0.8, "#C9D3E0"), (q[0], q[1], 1.8, "#8FA9C8"), (q[1], 1e12, 3.5, NAVY)):
        xs, ys = [], []
        for u, v, d in G.edges(data=True):
            if lo < d["weight"] <= hi:
                xs += [pos[u][0], pos[v][0], None]
                ys += [pos[u][1], pos[v][1], None]
        fig.add_scatter(x=xs, y=ys, mode="lines", line=dict(width=width, color=col), hoverinfo="skip", showlegend=False)
    nodes = [n for n in G.nodes if G.degree(n) > 0]
    fig.add_scatter(x=[pos[n][0] for n in nodes], y=[pos[n][1] for n in nodes], mode="markers+text", text=[labels[n] for n in nodes],
                    textposition="top center", showlegend=False, hovertemplate="%{text}<extra></extra>",
                    marker=dict(size=[10 + 28 * freq[n] / freq.max() for n in nodes], color=TEAL, line=dict(width=1, color="white")))
    fig.update_layout(title="Word association network (node size = frequency; line width = co-occurrence)",
                      xaxis=dict(visible=False), yaxis=dict(visible=False))
    show(fig, 560)


# ----------------------------------------------------------------------------------------------
# PAGE 5 — Sentiment
# ----------------------------------------------------------------------------------------------
def page_sentiment(dff):
    hero("Sentiment Intelligence", "VADER lexicon polarity — a defensible but imperfect estimate, not verified emotional ground truth")
    d = dff[dff["has_text"]].copy()
    if len(d) < 30:
        st.warning("Fewer than 30 reviews with text match the filters.")
        return
    with st.container(border=True):
        ctrl_label("Sentiment controls")
        c1, c2, c3 = st.columns(3)
        pos_t = c1.slider("Positive threshold (compound ≥)", 0.0, 0.8, 0.05, 0.01)
        neg_t = c2.slider("Negative threshold (compound ≤)", -0.8, 0.0, -0.05, 0.01)
        min_seg = c3.slider("Minimum segment size", 10, 300, 30, 5)
    d["cls"] = np.where(d["compound"] >= pos_t, "Positive", np.where(d["compound"] <= neg_t, "Negative", "Neutral"))
    COLS = {"Positive": TEAL, "Neutral": AMBER, "Negative": RED}
    k = st.columns(4)
    kpi(k[0], "Reviews scored", f"{len(d):,}", accent=SKY)
    for i, c in enumerate(["Positive", "Neutral", "Negative"]):
        kpi(k[i + 1], c, fmt_pct((d["cls"] == c).mean()), f"{int((d['cls'] == c).sum()):,} reviews", accent=COLS[c])
    a, b = st.columns(2)
    with a:
        show(px.histogram(d, x="compound", nbins=40, title="Sentiment polarity (VADER compound)", color_discrete_sequence=[SKY]))
    with b:
        vc = d["cls"].value_counts().reset_index()
        vc.columns = ["Class", "Reviews"]
        show(px.pie(vc, names="Class", values="Reviews", hole=0.5, color="Class", color_discrete_map=COLS, title="Sentiment class distribution"))
    for by, ttl in (("Department Name", "department"), ("Class Name", "clothing class")):
        g = d.groupby(by)["cls"].value_counts(normalize=False).unstack(fill_value=0)
        g = g[g.sum(axis=1) >= min_seg]
        if g.empty:
            st.info(f"No {ttl} meets the minimum segment size.")
            continue
        sh = g.div(g.sum(axis=1), axis=0).reset_index().melt(id_vars=by, var_name="Class", value_name="Share")
        order = g.div(g.sum(axis=1), axis=0).get("Negative", pd.Series(0, index=g.index)).sort_values().index.tolist()
        fig = px.bar(sh, x="Share", y=by, color="Class", orientation="h", color_discrete_map=COLS, title=f"Sentiment by {ttl} (share of reviews)",
                     category_orders={by: order})
        fig.update_xaxes(tickformat=".0%")
        show(fig, 300 if by == "Department Name" else 520)
    a, b = st.columns(2)
    with a:
        show(px.box(d, x="Rating", y="compound", title="Sentiment vs rating", color_discrete_sequence=[SKY]))
    with b:
        fig = px.violin(d, x=d["Recommended IND"].map({1: "Recommended", 0: "Not recommended"}), y="compound", box=True,
                        color=d["Recommended IND"].map({1: "Recommended", 0: "Not recommended"}), title="Sentiment vs recommendation",
                        color_discrete_map={"Recommended": TEAL, "Not recommended": RED}, labels={"x": ""})
        show(fig)
    section("Conflicting signals")
    conflict = d[((d["Rating"] <= 2) & (d["compound"] >= max(pos_t, 0.5))) | ((d["Rating"] >= 4) & (d["compound"] <= min(neg_t, -0.3)))]
    st.caption(f"{len(conflict):,} reviews ({fmt_pct(len(conflict) / len(d))}) show low rating with strongly positive text, or high rating with clearly negative text.")
    cols = ["Rating", "Recommended IND", "compound", "Department Name", "Class Name", "Title", "Review Text"]
    if len(conflict):
        table(conflict.sample(min(15, len(conflict)), random_state=SEED)[cols].rename(columns={"compound": "Sentiment"}), width="stretch", hide_index=True)
    section("Representative reviews")
    t1, t2 = st.tabs(["Most positive", "Most negative"])
    with t1:
        table(d.nlargest(8, "compound")[cols], width="stretch", hide_index=True)
    with t2:
        table(d.nsmallest(8, "compound")[cols], width="stretch", hide_index=True)
    explain("VADER polarity distribution and its relationship with ratings and recommendation.",
            "Text sentiment can reveal dissatisfaction hidden behind moderate ratings (and vice-versa).",
            "Use conflicting-signal reviews for manual audit; do not treat lexicon scores as ground truth.")


# ----------------------------------------------------------------------------------------------
# PAGE 6 — Topics and complaints
# ----------------------------------------------------------------------------------------------


def page_topics(dff):
    hero("Topic Modelling & Complaint Explorer", "LDA topics are exploratory; complaint themes are rule-based tags validated against sentence-level evidence")
    note("<b>Important:</b> LDA topics are statistical word groupings — they are <i>not</i> validated complaint categories. "
         "The complaint-theme tags below require a theme keyword <i>and</i> negative evidence in the same sentence (negation-aware heuristic).", "caution")
    t1, t2 = st.tabs(["LDA topic modelling", "Complaint themes (rule-based)"])
    with t1:
        with st.container(border=True):
            ctrl_label("Topic-model settings")
            c1, c2, c3 = st.columns([1, 1, 1])
            k = c1.slider("Number of topics", 3, 12, 6)
            it = c2.slider("Training iterations", 3, 20, 8)
            with c3:
                st.write("")
                run = st.button("▶ Run / re-run topic model", type="primary")
        if run:
            st.session_state["lda_params"] = (k, it)
        if "lda_params" not in st.session_state:
            st.info("Choose settings and press **Run topic model**. Fitting is deliberate (it can take a few seconds) and cached for identical settings.")
        else:
            kk, ii = st.session_state["lda_params"]
            try:
                with st.spinner(f"Fitting LDA with {kk} topics…"):
                    res = lda_cached(kk, ii, SEED)
            except (ValueError, MemoryError) as err:
                st.error(f"Topic modelling failed: {err}")
                return
            show_topics(dff, res, kk)
    with t2:
        complaint_view(dff)


def show_topics(dff, res, k):
    comps = res["comps"] / res["comps"].sum(axis=1, keepdims=True)
    names = [f"T{i + 1}: " + ", ".join(res["terms"][np.argsort(-comps[i])[:3]]) for i in range(k)]
    st.caption(f"LDA fitted on {res['n_docs']:,} reviews with text, {res['n_terms']:,} terms; perplexity {res['perplexity']:.0f}. "
               f"Settings used: {k} topics. Displayed results use the sidebar-filtered reviews.")
    top_n = st.slider("Terms per topic", 5, 15, 10)
    rows = []
    for i in range(k):
        o = np.argsort(-comps[i])[:top_n]
        rows += [dict(Topic=names[i], Term=res["terms"][j], Weight=comps[i, j]) for j in o]
    tdf = pd.DataFrame(rows)
    fig = px.bar(tdf, x="Weight", y="Term", facet_col="Topic", facet_col_wrap=3, facet_row_spacing=0.12, title="Top terms by topic")
    fig.update_yaxes(matches=None, showticklabels=True, autorange="reversed")
    fig.update_xaxes(matches=None)
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    fig.update_traces(marker_color=SKY)
    show(fig, 300 * int(np.ceil(k / 3)))
    s = pd.DataFrame({"row_id": res["row_ids"], "topic": res["theta"].argmax(axis=1), "tp": res["theta"].max(axis=1)})
    d = dff.merge(s, on="row_id", how="inner")
    if len(d) < 30:
        st.warning("Too few filtered reviews carry topic assignments.")
        return
    d["Topic"] = d["topic"].map(lambda i: names[i])
    prev = d["Topic"].value_counts(normalize=True).reset_index()
    prev.columns = ["Topic", "Share"]
    a, b = st.columns(2)
    with a:
        fig = px.bar(prev.iloc[::-1], x="Share", y="Topic", orientation="h", title="Topic prevalence (dominant topic per review)")
        fig.update_traces(marker_color=TEAL)
        fig.update_xaxes(tickformat=".0%")
        show(fig)
    with b:
        ct = pd.crosstab(d["Department Name"], d["Topic"], normalize="index")
        show(px.imshow(ct * 100, text_auto=".0f", aspect="auto", color_continuous_scale=["#F8FAFC", SKY], title="Topic mix by department (row %)"))
    g = d.groupby(["Topic", "Recommended IND"]).size().unstack(fill_value=0)
    if {0, 1} <= set(g.columns) and g[0].sum() > 0:
        gg = pd.DataFrame({"Recommended": g[1] / g[1].sum(), "Not recommended": g[0] / g[0].sum()}).reset_index().melt(id_vars="Topic", var_name="Group", value_name="Share")
        fig = px.bar(gg, x="Share", y="Topic", color="Group", barmode="group", orientation="h",
                     title="Topic share: recommended vs non-recommended reviews",
                     color_discrete_map={"Recommended": TEAL, "Not recommended": RED})
        fig.update_xaxes(tickformat=".0%")
        fig.update_yaxes(autorange="reversed", title="")
        show(fig, max(380, 110 + 52 * k))
        tn = d.groupby("Topic")["non_rec"].agg(["size", "mean"]).reset_index()
        tn.columns = ["Topic", "Reviews", "Non-rec. rate"]
        tn["Non-rec. rate"] = (100 * tn["Non-rec. rate"]).round(1)
        table(tn.sort_values("Non-rec. rate", ascending=False), width="stretch", hide_index=True)
    sel = st.selectbox("Representative reviews for topic", names)
    rep = d[d["Topic"] == sel].nlargest(5, "tp")[["tp", "Rating", "Recommended IND", "Department Name", "Title", "Review Text"]]
    table(rep.rename(columns={"tp": "Topic probability"}), width="stretch", hide_index=True)


def complaint_view(dff):
    d = dff[dff["has_text"]]
    if len(d) < 30:
        st.warning("Fewer than 30 reviews with text match the filters.")
        return
    tt = theme_table(d)
    a, b = st.columns(2)
    with a:
        fig = px.bar(tt.sort_values("n_flagged"), x="n_flagged", y="Theme", orientation="h", text="n_flagged", title="Complaint-theme frequency (reviews tagged)")
        fig.update_traces(marker_color=SKY)
        show(fig)
    with b:
        m = tt.melt(id_vars="Theme", value_vars=["nonrec_rate_flagged", "nonrec_rate_other"], var_name="Group", value_name="Rate")
        m["Group"] = m["Group"].map({"nonrec_rate_flagged": "Reviews with complaint", "nonrec_rate_other": "Other reviews"})
        fig = px.bar(m, x="Rate", y="Theme", color="Group", barmode="group", orientation="h", title="Non-recommendation rate: tagged vs other reviews",
                     color_discrete_map={"Reviews with complaint": RED, "Other reviews": TEAL})
        fig.update_xaxes(tickformat=".0%")
        show(fig)
    min_seg = st.slider("Minimum class size for the heat-map", 30, 300, 80, 10)
    cls_n = d["Class Name"].value_counts()
    keep = cls_n[cls_n >= min_seg].index
    if len(keep):
        g = d[d["Class Name"].isin(keep)].groupby("Class Name")[[f"cmp_{t}" for t in THEMES]].mean()
        g.columns = list(THEMES)
        show(px.imshow(g * 100, text_auto=".1f", aspect="auto", color_continuous_scale=["#F8FAFC", AMBER, RED],
                       title="Complaint-theme distribution by class (% of class reviews tagged)"), 160 + 28 * len(g))
    show_tbl = tt.copy()
    for c in ["share_reviews", "nonrec_rate_flagged", "nonrec_rate_other", "ci_low", "share_of_nonrec"]:
        show_tbl[c] = (100 * show_tbl[c]).round(1)
    show_tbl[["lift", "excess_nonrec"]] = show_tbl[["lift", "excess_nonrec"]].round(2)
    table(show_tbl.rename(columns={"n_flagged": "Reviews tagged", "share_reviews": "% of reviews", "nonrec_rate_flagged": "Non-rec % (tagged)",
                                          "nonrec_rate_other": "Non-rec % (other)", "lift": "Relative risk", "ci_low": "Wilson CI low % (tagged)",
                                          "excess_nonrec": "Associated excess non-rec. reviews", "share_of_nonrec": "% of all non-rec. reviews",
                                          "reliable": "Reliable"}), width="stretch", hide_index=True)
    ex = st.selectbox("Example reviews for theme", list(THEMES))
    exd = d[d[f"cmp_{ex}"]].nsmallest(5, "compound")[["Rating", "Recommended IND", "Class Name", "Title", "Review Text"]]
    table(exd, width="stretch", hide_index=True)
    st.caption("Multi-label: one review can carry several themes. 'Associated excess' = tagged reviews × (non-rec. rate tagged − non-rec. rate other); it is an association, not a causal estimate.")


# ----------------------------------------------------------------------------------------------
# PAGE 7 — ML / DL
# ----------------------------------------------------------------------------------------------
















def metrics_table(entries: dict) -> pd.DataFrame:
    rows = []
    for name, r in entries.items():
        m = r["metrics"]
        rows.append({"Model": name, "Train n": r.get("n_train"), "Test n": r.get("n_test"), "Accuracy": m["accuracy"],
                     "Precision (rec.)": m["precision_rec"], "Recall (rec.)": m["recall_rec"], "F1 (rec.)": m["f1_rec"],
                     "Precision (non-rec.)": m["precision_non"], "Recall (non-rec.)": m["recall_non"], "F1 (non-rec.)": m["f1_non"],
                     "Macro F1": m["macro_f1"], "ROC-AUC": m["roc_auc"], "PR-AUC (rec.)": m["pr_auc_rec"], "PR-AUC (non-rec.)": m["pr_auc_non"]})
    return pd.DataFrame(rows).round(4)


def page_ml(df):
    hero("Machine Learning & Deep Learning", "Predicting Recommended IND from written feedback — five models, genuine evaluation on a held-out test set")
    sig = data_signature(df)
    md = modeling_frame(df)
    d = st.columns(4)
    kpi(d[0], "Modelling reviews", f"{len(md):,}", "Reviews with text, exact duplicates removed.", SKY)
    kpi(d[1], "Recommended", f"{int(md['y'].sum()):,}", accent=TEAL)
    kpi(d[2], "Not recommended (minority)", f"{int((md['y'] == 0).sum()):,}", accent=RED)
    kpi(d[3], "Minority share", fmt_pct((md["y"] == 0).mean()), accent=AMBER)
    note("Target = <code>Recommended IND</code> (1 = recommended). Input = cleaned title+review text only. <code>Rating</code> and sentiment are <b>not</b> features. "
         "Stratified 80/20 train/test split (seed 42); duplicates removed before splitting. Class-weighting handles imbalance; metrics are computed on the untouched test set.")

    section("Level B · Classical models (cached)")
    with st.container(border=True):
        ctrl_label("Classical-model settings")
        c1, c2 = st.columns(2)
        sample_opt = c1.selectbox("Sample size for classical models", ["All available", 15000, 10000, 5000], index=0)
        Cval = c2.select_slider("Regularisation C (Logistic Regression; SVM uses C/10)", [0.1, 0.3, 1.0, 3.0, 10.0], value=1.0)
    sample_n = None if sample_opt == "All available" else int(sample_opt)
    try:
        with st.spinner("Training TF-IDF linear models (cached per setting)…"):
            cl = classical_cached(sample_n, float(Cval))
    except ValueError as err:
        st.error(f"Classical model training failed: {err}")
        return

    section("Level C · Deep-learning models (explicit training)")
    tf_ok = core.keras_available()
    if not tf_ok:
        note("Keras/JAX is not installed in this runtime, so ANN/LSTM/GRU cannot be trained here. The rest of the app works normally. "
             "Check that keras and jax are in requirements.txt, or import previously exported results below. No deep-learning metrics are shown unless genuinely computed.", "caution")
    else:
        note("Training runs on the server CPU with Keras 3 on the JAX backend (TensorFlow has no Python 3.14 build). Large samples, many epochs and several models can be slow or exceed Streamlit Community Cloud memory. "
             "Start with ≤ 8,000 reviews and 3–5 epochs. Results are kept in your session only.", "caution")
        with st.form("dl_form"):
            f1, f2, f3, f4 = st.columns(4)
            sel = f1.multiselect("Models", ["Multilayer ANN", "LSTM", "GRU"], default=["Multilayer ANN"])
            dl_n = f2.slider("Sample size (reviews)", 2000, 20000, 8000, 1000)
            ep = f3.slider("Max epochs", 1, 15, 5)
            bs = f4.select_slider("Batch size", [32, 64, 128, 256], value=64)
            g1, g2 = st.columns(2)
            ml = g1.select_slider("Max sequence length", [50, 80, 100, 150], value=100)
            vs = g2.select_slider("Vocabulary size", [3000, 5000, 8000, 10000], value=8000)
            go_train = st.form_submit_button("🚀 Train Deep Learning Models", type="primary")
        if go_train:
            if not sel:
                st.warning("Select at least one model.")
            else:
                try:
                    run_deep_learning(sel, dl_n, ep, bs, ml, vs)
                except (ValueError, MemoryError, RuntimeError) as err:
                    st.error(f"Deep-learning training failed ({type(err).__name__}): {err}. Try a smaller sample, fewer models or fewer epochs. No results were recorded for the failed run.")
    with st.expander("Optional: import previously exported deep-learning results (JSON)"):
        up = st.file_uploader("Results JSON exported by this app", type=["json"])
        if up is not None:
            try:
                payload = json.load(up)
                if payload.get("sig") != sig:
                    st.error("Incompatible file: it was produced from a different dataset signature. It was not loaded.")
                else:
                    st.session_state["dl_imported"] = payload["models"]
                    st.success("Imported results loaded (metrics and learning curves only; no live prediction).")
            except (json.JSONDecodeError, KeyError, AttributeError) as err:
                st.error(f"Could not read the file: {err}")

    entries = dict(cl["results"])
    dl = {k: v for k, v in st.session_state.get("dl", {}).items() if v.get("sig") == sig}
    entries.update(dl)
    imported = {f"{k} (imported)": v for k, v in st.session_state.get("dl_imported", {}).items()}
    t1, t2, t3, t4, t5 = st.tabs(["Model descriptions", "Comparison & curves", "Confusion matrices", "Learning curves", "Live prediction"])
    with t1:
        desc = {"TF-IDF + Logistic Regression": "Linear probabilistic classifier on sparse TF-IDF features; interpretable coefficients.",
                "TF-IDF + Linear SVM": "Maximum-margin linear classifier; produces a decision score (not a probability).",
                "Multilayer ANN": "Two dense layers (128, 64) with ReLU and dropout (0.4/0.3) on a 2,000-feature TF-IDF vector; sigmoid output.",
                "LSTM": "Embedding(64) → LSTM(64) → dropout → Dense(32) → sigmoid, on padded token sequences.",
                "GRU": "Embedding(64) → GRU(64) → dropout → Dense(32) → sigmoid, on padded token sequences."}
        rows = []
        for n in desc:
            e = entries.get(n)
            rows.append({"Model": n, "Description": desc[n], "Input": e["input"] if e else "—", "Training parameters": e["params"] if e else "—",
                         "Status": "Trained in this session" if e else "Not trained"})
        table(pd.DataFrame(rows), width="stretch", hide_index=True)
        pn = pd.DataFrame({"Words pushing to NOT recommended": cl["top_neg"], "Coefficient": np.round(cl["coef_neg"], 2),
                           "Words pushing to recommended": cl["top_pos"], "Coef. ": np.round(cl["coef_pos"], 2)})
        st.markdown("**Most influential Logistic-Regression terms**")
        table(pn, width="stretch", hide_index=True)
    allm = {**entries, **imported}
    with t2:
        tbl = metrics_table(allm)
        if len(tbl):
            best = st.columns(3)
            for col_, (lbl, met) in zip(best, [("Best macro-F1", "Macro F1"), ("Best ROC-AUC", "ROC-AUC"),
                                               ("Best minority-class F1", "F1 (non-rec.)")]):
                row_ = tbl.loc[tbl[met].idxmax()]
                kpi(col_, lbl, f"{row_[met]:.3f}", f"Highest {met} among the models shown below.", TEAL, str(row_["Model"]))
        table(tbl, width="stretch", hide_index=True)
        st.download_button("Download comparison table (CSV)", csv_bytes(tbl), "model_comparison.csv", "text/csv")
        note("Models trained on different sample sizes have different test sets, so compare with care (see Train n / Test n). "
             "'(rec.)' = recommended class; '(non-rec.)' = minority not-recommended class. PR-AUC (non-rec.) is the most informative score for the minority class.")
        melt = tbl.melt(id_vars="Model", value_vars=["Macro F1", "F1 (non-rec.)", "ROC-AUC", "PR-AUC (non-rec.)"], var_name="Metric", value_name="Score")
        fig = px.bar(melt, x="Model", y="Score", color="Metric", barmode="group", title="Key metrics by model")
        fig.update_xaxes(tickangle=-15)
        fig.update_yaxes(range=[0, 1.05])
        show(fig, 420)
        roc, prc = go.Figure(), go.Figure()
        for n, r in entries.items():
            roc.add_scatter(x=r["roc"][0], y=r["roc"][1], mode="lines", name=n)
            prc.add_scatter(x=r["pr"][0], y=r["pr"][1], mode="lines", name=n)
        roc.add_scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash", color="grey"), name="Chance")
        roc.update_layout(title="ROC curves (recommended = positive class)", xaxis_title="False positive rate", yaxis_title="True positive rate")
        prc.update_layout(title="Precision–recall curves (recommended = positive class)", xaxis_title="Recall", yaxis_title="Precision")
        a, b = st.columns(2)
        with a: show(roc)
        with b: show(prc)
        if dl:
            exp = {"sig": sig, "models": {k: {kk: vv for kk, vv in v.items() if kk in ("metrics", "cm", "history", "params", "input", "n_train", "n_test", "epochs_run", "seconds")} for k, v in dl.items()}}
            st.download_button("Export deep-learning results (JSON)", json.dumps(exp, default=float).encode(), "dl_results.json", "application/json")
    with t3:
        if allm:
            pick = st.selectbox("Model", list(allm), key="cm_model")
            cm = np.array(allm[pick]["cm"])
            fig = px.imshow(cm, text_auto=",", x=["Pred: Not rec.", "Pred: Rec."], y=["Actual: Not rec.", "Actual: Rec."],
                            color_continuous_scale=["#F8FAFC", SKY], title=f"Confusion matrix — {pick}")
            m = allm[pick]["metrics"]
            ca, cb = st.columns([3, 2])
            with ca:
                show(fig, 400)
            kpi(cb, "Minority-class precision", f"{m['precision_non']:.3f}", "Not-recommended class, held-out test set.", RED)
            kpi(cb, "Minority-class recall", f"{m['recall_non']:.3f}", "Not-recommended class, held-out test set.", RED)
            kpi(cb, "Minority-class F1", f"{m['f1_non']:.3f}", "Not-recommended class, held-out test set.", RED, f"Test sample: {allm[pick].get('n_test', 'n/a')} reviews")
    with t4:
        hist_models = {n: r for n, r in allm.items() if r.get("history")}
        if not hist_models:
            st.info("No neural-network learning curves yet. Train a deep-learning model above (or import exported results).")
        else:
            rows = []
            for n, r in hist_models.items():
                for key, vals in r["history"].items():
                    rows += [dict(Model=n, Metric=key, Epoch=i + 1, Value=v) for i, v in enumerate(vals)]
            h = pd.DataFrame(rows)
            a, b = st.columns(2)
            with a:
                show(px.line(h[h["Metric"].isin(["loss", "val_loss"])], x="Epoch", y="Value", color="Model", line_dash="Metric", markers=True, title="Loss (train vs validation)"))
            with b:
                show(px.line(h[h["Metric"].isin(["accuracy", "val_accuracy"])], x="Epoch", y="Value", color="Model", line_dash="Metric", markers=True, title="Accuracy (train vs validation)"))
            st.caption("Early stopping monitors validation loss (patience 2) and restores the best weights; 15% of the training data is held out for validation.")
    with t5:
        live_prediction(cl, dl)
    with st.expander("Evaluation notes and limitations"):
        st.markdown("- Accuracy alone is misleading because recommended reviews dominate; use macro-F1, minority-class recall/F1 and PR-AUC.\n"
                    "- Class weights are used in training, so predicted probabilities are not calibrated frequencies.\n"
                    "- Single stratified split, seed 42: no cross-validation or confidence intervals are reported, so small differences between models should not be over-interpreted.\n"
                    "- Neural models use reduced samples/epochs to fit cloud resources; they may under-perform relative to a fully tuned pipeline.\n"
                    "- Predicting recommendation from text does not establish why customers are dissatisfied.")


def live_prediction(cl, dl):
    st.markdown("Enter a new product review to score it with a model trained in this session.")
    avail = ["TF-IDF + Logistic Regression", "TF-IDF + Linear SVM"] + list(dl.keys())
    pick = st.selectbox("Model", avail, key="live_model")
    text = st.text_area("Review text", "The fabric feels cheap and the dress runs very small. Not worth the price.", height=110)
    if not st.button("Predict"):
        return
    clean = clean_text(text)
    if len(clean.split()) < 3:
        st.warning("Please enter a longer review (at least three meaningful words); predictions on very short text are unreliable.")
        return
    if pick == "TF-IDF + Logistic Regression":
        p = float(cl["lr"].predict_proba(cl["vec"].transform([clean]))[0, 1])
        label, detail = int(p >= 0.5), f"Estimated probability of recommendation: **{p:.1%}** (class-weighted model; not a calibrated frequency)."
    elif pick == "TF-IDF + Linear SVM":
        s = float(cl["svm"].decision_function(cl["vec"].transform([clean]))[0])
        label, detail = int(s >= 0), f"SVM decision score: **{s:+.2f}** (positive → recommended). This is a margin, not a probability."
    else:
        o = dl[pick]["obj"]
        X = o["vec"].transform([clean]).astype("float32").toarray() if pick == "Multilayer ANN" else encode([clean], o["vocab"], o["maxlen"])
        p = float(o["model"].predict(X, verbose=0).ravel()[0])
        label, detail = int(p >= 0.5), f"Estimated probability of recommendation: **{p:.1%}** (neural network; class-weighted)."
    an = get_analyzer()
    comp = an.polarity_scores(text)["compound"]
    sent = "Positive" if comp >= 0.05 else "Negative" if comp <= -0.05 else "Neutral"
    tags = sorted(tag_complaints(text, an))
    c = st.columns(3)
    kpi(c[0], "Predicted class", "Recommended" if label else "Not recommended", accent=TEAL if label else RED)
    kpi(c[1], "VADER sentiment", f"{sent} ({comp:+.2f})", accent=SKY)
    kpi(c[2], "Complaint attributes", ", ".join(tags) if tags else "None detected", accent=AMBER)
    st.write(detail)
    st.caption("Limitations: trained on one retailer's reviews; short, sarcastic or mixed reviews are error-prone; complaint tags are keyword heuristics; outputs are decision support, not verified outcomes.")


# ----------------------------------------------------------------------------------------------
# PAGE 8 — Six Sigma
# ----------------------------------------------------------------------------------------------
def page_sixsigma(dff, df_all):
    hero("Product Quality & Six Sigma", "Customer-perceived quality analytics — review-based, not factory-based")
    if len(dff) < 30:
        st.warning("Fewer than 30 reviews match the filters.")
        return
    k = core_kpis(dff)
    with st.container(border=True):
        ctrl_label("Display controls")
        min_seg = st.slider("Minimum segment size", 10, 300, 50, 10)
    c = st.columns(4)
    kpi(c[0], "Non-recommendation rate", fmt_pct(k["nonrec"]), f"{int(dff['non_rec'].sum()):,} of {len(dff):,} reviews", RED)
    kpi(c[1], "Low-rating rate (≤2★)", fmt_pct(k["low"]), accent=RED)
    kpi(c[2], "Review-based DPMO proxy", f"{k['dpmo']:,.0f}", "Customer-dissatisfaction proxy; not a verified factory defect rate.", AMBER)
    kpi(c[3], "Illustrative sigma level", f"{sigma_level(k['dpmo']):.2f}", "Normal-approximation with the conventional 1.5σ shift; illustrative only.", NAVY)
    note("<b>Definition (review-based customer dissatisfaction proxy):</b> DPMO = non-recommended reviews ÷ (eligible reviews × 1 opportunity per review) × 1,000,000. "
         f"Here: {int(dff['non_rec'].sum()):,} ÷ ({len(dff):,} × 1) × 10⁶ = {k['dpmo']:,.0f}. It is <u>not</u> a manufacturing defect rate, return rate or cost measure.")
    t1, t2, t3, t4 = st.tabs(["Segment quality", "Pareto & products", "DMAIC", "Design of Experiments (proposal)"])
    with t1:
        a, b = st.columns(2)
        dep = seg_table(dff, "Department Name", min_seg)
        cls = seg_table(dff, "Class Name", min_seg)
        with a:
            if len(dep):
                fig = px.bar(dep.sort_values("dpmo"), x="dpmo", y="Department Name", orientation="h", custom_data=["n", "ci_low", "ci_high"],
                             title="DPMO proxy by department", labels={"dpmo": "DPMO proxy", "Department Name": ""})
                fig.update_traces(marker_color=AMBER, hovertemplate="%{y}: %{x:,.0f} (n=%{customdata[0]:,}; 95%% CI %{customdata[1]:.1%}–%{customdata[2]:.1%})<extra></extra>")
                show(fig)
        with b:
            if len(cls):
                fig = px.bar(cls.sort_values("dpmo").tail(15), x="dpmo", y="Class Name", orientation="h", custom_data=["n"],
                             title="DPMO proxy by class (top 15 of eligible)", labels={"dpmo": "DPMO proxy", "Class Name": ""})
                fig.update_traces(marker_color=RED, hovertemplate="%{y}: %{x:,.0f} (n=%{customdata[0]:,})<extra></extra>")
                show(fig, 460)
        tbl = pd.concat([dep.assign(Level="Department").rename(columns={"Department Name": "Segment"}),
                         cls.assign(Level="Class").rename(columns={"Class Name": "Segment"})])
        if len(tbl):
            out = tbl[["Level", "Segment", "n", "nonrec_rate", "low_rate", "dpmo", "ci_low", "ci_high"]].copy()
            for cc in ["nonrec_rate", "low_rate", "ci_low", "ci_high"]:
                out[cc] = (100 * out[cc]).round(1)
            out["dpmo"] = out["dpmo"].round(0)
            out.columns = ["Level", "Segment", "Reviews", "Non-rec %", "Low-rating %", "DPMO proxy", "CI low %", "CI high %"]
            table(out, pct=("Non-rec %",), width="stretch", hide_index=True)
    with t2:
        tt = theme_table(dff, min_seg)
        if tt["n_flagged"].sum() == 0:
            st.info("No complaint tags in the selection.")
        else:
            p = tt.sort_values("n_flagged", ascending=False).copy()
            nr = dff[(dff["non_rec"] == 1) & dff["has_text"]]
            p["nr_count"] = [int(nr[f"cmp_{t}"].sum()) for t in p["Theme"]]
            p = p.sort_values("nr_count", ascending=False)
            p["cum"] = p["nr_count"].cumsum() / max(p["nr_count"].sum(), 1)
            fig = go.Figure()
            fig.add_bar(x=p["Theme"], y=p["nr_count"], name="Non-recommended reviews tagged", marker_color=SKY)
            fig.add_scatter(x=p["Theme"], y=p["cum"] * 100, name="Cumulative % (of tags)", yaxis="y2", mode="lines+markers", line=dict(color=RED))
            fig.update_layout(title="Complaint Pareto — non-recommended reviews by theme (multi-label)", yaxis_title="Reviews",
                              yaxis2=dict(title="Cumulative %", overlaying="y", side="right", range=[0, 105]))
            fig.update_xaxes(tickangle=-25)
            show(fig, 440)
            st.caption(f"Based on {len(nr):,} non-recommended reviews with text; a review can carry several themes, so cumulative % refers to tag counts.")
        pmin = st.slider("Minimum reviews per product", 5, 60, 15)
        pt = seg_table(dff, "Clothing ID", pmin)
        if pt.empty:
            st.info("No product reaches the minimum review count under current filters.")
        else:
            pt["Clothing ID"] = pt["Clothing ID"].astype(int).astype(str)
            top = pt.head(15)
            fig = px.bar(top.iloc[::-1], x="nonrec_rate", y="Clothing ID", orientation="h", custom_data=["n", "ci_low", "ci_high"], title="Product-level risk ranking (highest non-recommendation rate)")
            fig.update_traces(marker_color=RED, hovertemplate="ID %{y}: %{x:.1%} (95%% CI %{customdata[1]:.1%}–%{customdata[2]:.1%}), n=%{customdata[0]}<extra></extra>")
            fig.update_xaxes(tickformat=".0%")
            show(fig, 460)
    with t3:
        tt = theme_table(df_all)
        dep = seg_table(df_all, "Department Name", 50)
        k0 = core_kpis(df_all)
        top_t = ", ".join(tt.head(3)["Theme"]) if len(tt) else "n/a"
        s = st.tabs(["Define", "Measure", "Analyze", "Improve", "Control"])
        with s[0]:
            st.markdown(f"**Problem:** {fmt_pct(k0['nonrec'])} of {k0['n']:,} reviews do not recommend the purchased product. "
                        "**Goal (proposed):** reduce avoidable non-recommendations in the highest-risk segments. **Scope:** review-based customer perception data only.")
        with s[1]:
            m = pd.DataFrame([("Non-recommendation rate", fmt_pct(k0["nonrec"])), ("Low-rating rate", fmt_pct(k0["low"])),
                              ("DPMO proxy", f"{k0['dpmo']:,.0f}"), ("Complaint-tagged reviews (any theme)", fmt_pct(df_all.loc[df_all['has_text'], 'any_complaint'].mean())),
                              ("Negative sentiment share", fmt_pct(k0["neg"]))], columns=["KPI (full dataset)", "Baseline"])
            table(m, width="stretch", hide_index=True)
        with s[2]:
            st.markdown(f"Top themes by associated excess non-recommendations: **{top_t}**.")
            if len(dep): st.markdown(f"Highest-risk department (n ≥ 50): **{dep.iloc[0]['Department Name']}** at {fmt_pct(dep.iloc[0]['nonrec_rate'])}.")
            st.caption("Use the Pareto chart and priority matrix (Page 9). These are associations in review data, not proven root causes — root-cause verification needs inspection data.")
        with s[3]:
            for _, r in tt.head(3).iterrows():
                st.markdown(f"- **{r['Theme']}** → {RECS[r['Theme']][0]} *(Owner: {RECS[r['Theme']][1]})*")
            st.caption("Proposed interventions only; none has been tested and no improvement is claimed.")
        with s[4]:
            n_sub = st.select_slider("Nominal review subgroup size for a p-chart", [50, 100, 200, 500], value=100)
            p0 = k0["nonrec"]
            se = np.sqrt(p0 * (1 - p0) / n_sub)
            st.markdown(f"Baseline p̄ = {fmt_pct(p0)}. **Proposed** control limits for the non-recommendation proportion per subgroup of {n_sub} new reviews: "
                        f"UCL = {fmt_pct(p0 + 3 * se)}, LCL = {fmt_pct(max(p0 - 3 * se, 0))}. Investigate sustained points above UCL or runs of 8 above the centre line.")
            table(pd.DataFrame([
                ("Non-recommendation rate (weekly, by department)", "Quality Assurance"), ("Complaint-theme rate per 100 reviews", "Merchandising / QA"),
                ("Share of reviews with negative sentiment", "Customer Insights"), ("Product-level review volume & CI-adjusted non-rec. rate", "Category Managers")],
                columns=["Suggested monitoring KPI", "Owner"]), width="stretch", hide_index=True)
    with t4:
        st.markdown("**Proposed extension — no experiment has been run and no results are shown.**")
        table(pd.DataFrame([("A: Size-chart guidance", "Standard", "Enhanced with 'runs small/large' note"),
                                   ("B: Fit information", "Standard description", "Model measurements + fit photos"),
                                   ("C: Fabric detail", "Standard", "Weight/transparency/stretch descriptors")],
                                  columns=["Factor", "Level −", "Level +"]), width="stretch", hide_index=True)
        st.markdown("A 2³ randomised factorial (8 page variants, product-matched) with the non-recommendation rate and complaint-theme rate as responses, "
                    "analysed by main effects/interactions with adequate sample-size planning, would test causal impact. This requires live experimentation outside this dataset.")


# ----------------------------------------------------------------------------------------------
# PAGE 9 — Management insights
# ----------------------------------------------------------------------------------------------
def page_mgmt(df):
    hero("Management Insights & Recommendations", "Decision support built from transparent quantitative criteria")
    with st.container(border=True):
        ctrl_label("Reliability setting")
        min_n = st.slider("Reliability threshold (minimum tagged reviews / segment size)", 20, 300, 50, 10)
    tt = theme_table(df, min_n)
    if tt.empty:
        st.warning("No text data available.")
        return
    k = core_kpis(df)
    tt["Tier"] = pd.qcut(tt["excess_nonrec"].rank(method="first", ascending=False), 3, labels=["High", "Medium", "Low"]) if len(tt) >= 3 else "High"
    s = st.columns(4)
    kpi(s[0], "Themes assessed", f"{len(tt)}", "Rule-based complaint themes with at least one tagged review.", SKY)
    kpi(s[1], "High-priority themes", f"{int((tt['Tier'].astype(str) == 'High').sum())}", "Top tertile of associated excess non-recommendations.", RED)
    kpi(s[2], "Reliable themes", f"{int(tt['reliable'].sum())}", f"≥ {min_n} tagged reviews and Wilson lower bound above the untagged rate.", TEAL)
    kpi(s[3], "Top theme (excess non-rec.)", str(tt.iloc[0]["Theme"]), "Largest associated excess of non-recommended reviews.", AMBER,
        f"≈{tt.iloc[0]['excess_nonrec']:.0f} associated excess reviews")
    section("Priority matrix — complaint themes")
    fig = px.scatter(tt, x="share_reviews", y="lift", size="n_flagged", color="Tier", text="Theme", size_max=48,
                     color_discrete_map={"High": RED, "Medium": AMBER, "Low": TEAL}, title="Frequency vs non-recommendation association (bubble = tagged reviews)",
                     labels={"share_reviews": "Frequency (share of text reviews)", "lift": "Relative risk of non-recommendation (tagged ÷ other)"})
    fig.update_traces(textposition="top center")
    fig.update_xaxes(tickformat=".0%")
    show(fig, 520)
    note("<b>Criteria (transparent):</b> Frequency = tagged ÷ text reviews. Association = relative risk and associated excess non-recommendations = tagged × (rate tagged − rate other). "
         f"Tier = tertile of excess non-recommendations. Reliability = ≥ {min_n} tagged reviews and the Wilson lower bound of the tagged non-rec. rate above the untagged rate. "
         "Associations are not causal effects.")
    seg = []
    base = k["nonrec"]
    for by, lvl in (("Department Name", "Department"), ("Class Name", "Class")):
        g = seg_table(df, by, 1)
        g["excess"] = g["n"] * (g["nonrec_rate"] - base)
        g["reliable"] = (g["n"] >= min_n) & (g["ci_low"] > base)
        seg.append(g.assign(Level=lvl).rename(columns={by: "Segment"}))
    seg = pd.concat(seg)
    risk = seg[seg["n"] >= min_n].sort_values("excess", ascending=False)
    a, b = st.columns(2)
    with a:
        st.markdown("**High-risk departments & classes** (excess non-rec. reviews vs overall rate)")
        r = risk.head(10)[["Level", "Segment", "n", "nonrec_rate", "excess", "reliable"]].copy()
        r["nonrec_rate"] = (100 * r["nonrec_rate"]).round(1)
        r["excess"] = r["excess"].round(1)
        table(r.rename(columns={"n": "Reviews", "nonrec_rate": "Non-rec %", "excess": "Excess vs overall", "reliable": "CI above overall"}), width="stretch", hide_index=True)
    with b:
        pt = seg_table(df, "Clothing ID", min_n // 5 if min_n >= 25 else 5)
        pt["excess"] = pt["n"] * (pt["nonrec_rate"] - base)
        pt = pt.sort_values("excess", ascending=False).head(10)
        pt["Clothing ID"] = pt["Clothing ID"].astype(int)
        st.markdown("**Products deserving investigation** (largest excess non-rec. reviews)")
        o = pt[["Clothing ID", "n", "nonrec_rate", "excess", "ci_low"]].copy()
        o["nonrec_rate"] = (100 * o["nonrec_rate"]).round(1)
        o["excess"] = o["excess"].round(1)
        o["ci_low"] = (100 * o["ci_low"]).round(1)
        table(o.rename(columns={"n": "Reviews", "nonrec_rate": "Non-rec %", "excess": "Excess", "ci_low": "CI low %"}), width="stretch", hide_index=True)

    section("Issue-level recommendations")
    rec_rows = []
    for _, r in tt.iterrows():
        act, owner, kp = RECS[r["Theme"]]
        rec_rows.append({"Priority": r["Tier"], "Finding": f"'{r['Theme']}' complaints appear in {fmt_pct(r['share_reviews'])} of text reviews",
                         "Evidence": f"{int(r['n_flagged']):,} tagged reviews; non-rec. {fmt_pct(r['nonrec_rate_flagged'])} vs {fmt_pct(r['nonrec_rate_other'])} others (RR {r['lift']:.2f}); "
                                     f"≈{r['excess_nonrec']:.0f} associated excess non-rec. reviews; {'reliable' if r['reliable'] else 'low reliability'}",
                         "Business implication": "Perceived-quality problem linked to lower recommendation intent; potential effect on conversion and returns is NOT measured in this data.",
                         "Suggested action": act, "Responsible department": owner, "Monitoring KPI": kp})
    rec = pd.DataFrame(rec_rows)
    table(rec, width="stretch", hide_index=True)
    st.download_button("Download recommendation table (CSV)", csv_bytes(rec), "management_recommendations.csv", "text/csv")

    section("Scenario calculator (hypothetical)")
    note("Hypothetical what-if only. It is not an estimate of realised savings, revenue, returns or causal effects.", "caution")
    with st.container(border=True):
        ctrl_label("Scenario inputs")
        chosen = st.multiselect("Complaint themes addressed", list(THEMES), default=list(tt.head(2)["Theme"]))
        conv = st.slider("Assumed % of affected non-recommended reviews that would become recommendations", 0, 50, 10)
    d = df[df["has_text"]]
    affected = d[(d["non_rec"] == 1) & d[[f"cmp_{t}" for t in chosen]].any(axis=1)] if chosen else d.iloc[0:0]
    reduced = len(affected) * conv / 100
    new_nr = df["non_rec"].sum() - reduced
    new_dpmo = new_nr / len(df) * 1e6
    c = st.columns(4)
    kpi(c[0], "Affected non-rec. reviews", f"{len(affected):,}", "Non-recommended reviews tagged with ≥1 chosen theme (no double counting).", AMBER)
    kpi(c[1], "Hypothetical conversions", f"{reduced:,.0f}", accent=TEAL)
    kpi(c[2], "Hypothetical non-rec. rate", fmt_pct(new_nr / len(df)), f"Baseline {fmt_pct(k['nonrec'])}", TEAL)
    kpi(c[3], "Hypothetical DPMO proxy", f"{new_dpmo:,.0f}", f"Baseline {k['dpmo']:,.0f}", NAVY)



# ----------------------------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------------------------
def go_page(i: int):
    st.session_state["nav_page"] = PAGES[i]


def page_footer(i: int):
    html_block('<div class="foot"><b>Methods:</b> VADER (Hutto &amp; Gilbert, 2014); LDA (Blei, Ng &amp; Jordan, 2003); scikit-learn; Keras 3 (JAX backend). '
               "<b>Dataset:</b> anonymised women's e-commerce clothing reviews supplied for the course project (public Kaggle dataset; commercial identifiers removed).</div>")
    c1, _, c3 = st.columns([1, 1, 1])
    if i > 0:
        c1.button(f"← {PAGES[i - 1]}", key="nav_prev", on_click=go_page, args=(i - 1,))
    if i < len(PAGES) - 1:
        c3.button(f"{PAGES[i + 1]} →", key="nav_next", on_click=go_page, args=(i + 1,))


def main():
    inject_css()
    _KEY[0] = 0
    # Tier 1 only at start-up: CSV + validation + KPIs. NLP tiers load lazily when a page needs them.
    try:
        base = get_base()
    except FileNotFoundError as err:
        st.error(f"**Dataset not found.** {err}")
        st.info("Upload the CSV to the repository root next to app.py with the exact name `Womens Clothing E-Commerce Reviews.csv`.")
        st.stop()
    except (ValueError, pd.errors.ParserError, pd.errors.EmptyDataError) as err:
        st.error(f"**The dataset could not be processed:** {err}")
        st.stop()
    except UnicodeDecodeError as err:
        st.error(f"**Encoding problem while reading the CSV:** {err}")
        st.stop()

    st.session_state.setdefault("nav_page", PAGES[0])
    with st.sidebar:
        html_block('<div class="brand"><div class="brand-logo">👗</div><div><div class="brand-name">Fashion Quality Intelligence</div>'
                   '<div class="brand-sub">Review analytics · Text AI · Six Sigma</div></div></div>')
        html_block('<div class="side-label">Navigate</div>')
        page = st.radio("Navigate", PAGES, key="nav_page", label_visibility="collapsed",
                        format_func=lambda p: f"{PAGE_ICONS[PAGES.index(p)]}  {PAGES.index(p) + 1}. {p}")
    page_idx = PAGES.index(page)
    _PAGE[0] = page_idx

    # Route each page to the lightest data tier it needs.
    try:
        if page in (PAGES[0],):
            df = get_enriched() if shared_state()["enriched"] else base
        elif page in (PAGES[1], PAGES[2]):
            df = base
        elif page in (PAGES[3], PAGES[6]):
            df = get_text_df()
        else:  # pages 5, 6, 8, 9 need sentiment + complaint tags
            df = get_enriched()
    except (ImportError, ModuleNotFoundError) as err:
        st.error(f"**A required analytics package is missing or failed to import:** `{type(err).__name__}: {err}`")
        st.dataframe(pd.DataFrame([{"module": m, "pip package": p, "ok": ok, "detail": v} for m, p, ok, v in core.dependency_report()]),
                     hide_index=True, width="stretch")
        st.stop()

    dff = sidebar_filters(df) if page in FILTER_PAGES else df
    with st.sidebar:
        if page not in FILTER_PAGES:
            html_block('<div class="side-note">This page uses the full dataset. Filters apply on pages 3–6 and 8.</div>')
        st.divider()
        st.caption(f"Dataset: {len(base):,} reviews · signature {data_signature(base)}")

    if page == PAGES[0]: page_exec(df)
    elif page == PAGES[1]: page_objectives(df)
    elif page == PAGES[2]: page_eda(dff)
    elif page == PAGES[3]: page_text(dff)
    elif page == PAGES[4]: page_sentiment(dff)
    elif page == PAGES[5]: page_topics(dff)
    elif page == PAGES[6]: page_ml(df)
    elif page == PAGES[7]: page_sixsigma(dff, df)
    elif page == PAGES[8]: page_mgmt(df)
    page_footer(page_idx)


if __name__ == "__main__":
    main()
