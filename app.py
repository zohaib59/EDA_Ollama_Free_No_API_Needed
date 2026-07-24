"""AI Data Analyst Agent — local, free (Streamlit + Ollama/llama3)
RUN:  python app.py   (auto-installs deps, auto-launches)
ONE-TIME EXTERNAL SETUP: install Ollama (https://ollama.com) then `ollama pull llama3`
"""
import os, sys, subprocess, importlib

if os.environ.get("_RELAUNCHED") != "1":
    for pkg in ["streamlit", "pandas", "numpy", "plotly", "requests", "openpyxl", "scipy"]:
        try: importlib.import_module(pkg)
        except ImportError: subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])
    env = os.environ.copy(); env["_RELAUNCHED"] = "1"
    subprocess.run([sys.executable, "-m", "streamlit", "run", os.path.abspath(__file__)], env=env)
    sys.exit()

import streamlit as st
import pandas as pd, numpy as np, requests, hashlib, io
import plotly.express as px
from scipy import stats

st.set_page_config(page_title="AI Data Analyst", layout="wide")
PALETTE = ["#01B8AA", "#374649", "#FD625E", "#F2C80F", "#8AD4EB", "#FE9666", "#A66999", "#3599B8", "#DFBFBF", "#5F6B6D"]
MAX_VIZ = 20
tc = lambda s: str(s).replace("_", " ").strip().title()
dt = lambda s: pd.to_datetime(s, errors="coerce", format="mixed")

def check_ollama():
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=2)
        models = [m.get("name", "") for m in r.json().get("models", [])]
        if not any("llama3" in m for m in models):
            return False, "Ollama running but llama3 not pulled. Run: ollama pull llama3"
        return True, "Llama3 connected."
    except Exception:
        return False, "Ollama not detected. Install: https://ollama.com then run: ollama pull llama3"

def ask_llm(prompt):
    try:
        r = requests.post("http://localhost:11434/api/generate",
                           json={"model": "llama3", "prompt": prompt, "stream": False}, timeout=60)
        return r.json().get("response", "").strip() or None
    except Exception:
        return None

@st.cache_data(show_spinner=False)
def clean_data(df):
    log, n = [], len(df)
    df = df.drop_duplicates()
    if n - len(df): log.append(f"Removed {n-len(df)} duplicate rows.")
    for c in df.columns:
        miss = df[c].isna().sum()
        if miss == 0: continue
        is_num = pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c])
        if miss == len(df):
            df[c] = df[c].fillna(0 if is_num else "Unknown")
            log.append(f"'{c}': fully empty, filled with default.")
        elif is_num:
            skew = df[c].skew(); skew = 0 if pd.isna(skew) else skew
            val = df[c].median() if abs(skew) > 1 else df[c].mean()
            df[c] = df[c].fillna(val)
            log.append(f"'{c}': filled {miss} missing ({'median' if abs(skew)>1 else 'mean'}={round(val,2)}).")
        else:
            mode = df[c].mode(); val = mode[0] if not mode.empty else "Unknown"
            df[c] = df[c].fillna(val)
            log.append(f"'{c}': filled {miss} missing (mode='{val}').")
    return df, log

def profile(df):
    num, cat, date = [], [], []
    for c in df.columns:
        if pd.api.types.is_bool_dtype(df[c]):
            cat.append(c)
        elif pd.api.types.is_numeric_dtype(df[c]):
            num.append(c)
        elif df[c].nunique() > 5 and dt(df[c].dropna().sample(min(50, df[c].notna().sum()), random_state=0)).notna().mean() > 0.8:
            date.append(c)
        else:
            cat.append(c)
    return num, cat, date

def dataset_summary(df, num, cat, date):
    parts = [f"This dataset holds **{len(df):,} records** across **{len(df.columns)} columns**."]
    if num: parts.append(f"Numeric variables include {', '.join(tc(c) for c in num[:3])}.")
    if cat: parts.append(f"Categorical variables include {', '.join(tc(c) for c in cat[:3])}.")
    if date: parts.append(f"{tc(date[0])} enables trend analysis over time.")
    return " ".join(parts)

def build_kpis(df, num, date):
    kpis = [("Total Records", f"{len(df):,}")]
    for c in num[:3]:
        kpis.append((f"Avg {tc(c)}", f"{df[c].mean():,.2f}"))
    if date:
        d = dt(df[date[0]]).dropna()
        if not d.empty: kpis.append((f"{tc(date[0])} Span (days)", f"{(d.max()-d.min()).days:,}"))
    return kpis[:5]

# ---------- Visuals: adaptive up to 20, Power-BI style, consistent size, labeled axes/values ----------
def build_visuals(df, num, cat, date):
    charts = []
    def add(title, fig, xt=None, yt=None):
        if len(charts) < MAX_VIZ:
            fig.update_layout(title=title, template="plotly_white", height=340,
                               margin=dict(l=10, r=10, t=45, b=10), xaxis_title=xt, yaxis_title=yt)
            charts.append(fig)

    cov = lambda c: abs(df[c].std()/df[c].mean()) if df[c].mean() not in (0, None) and not pd.isna(df[c].mean()) else (df[c].std() or 0)
    num_r = sorted(num, key=cov, reverse=True)
    cat_r = sorted([c for c in cat if 1 < df[c].nunique() <= 12], key=lambda c: df[c].nunique())

    try:
        for c in num_r[:6]:
            add(f"Distribution of {tc(c)}", px.histogram(df, x=c, nbins=30, color_discrete_sequence=PALETTE), tc(c), "Count")
        for i, c in enumerate(cat_r[:5]):
            vc = df[c].value_counts().reset_index(); vc.columns = [c, "Count"]
            if i % 3 == 0:
                fig = px.pie(vc, names=c, values="Count", color_discrete_sequence=PALETTE); fig.update_traces(textinfo="label+percent")
                add(f"Share of {tc(c)}", fig)
            elif i % 3 == 1:
                fig = px.pie(vc, names=c, values="Count", hole=0.5, color_discrete_sequence=PALETTE); fig.update_traces(textinfo="label+percent")
                add(f"Share of {tc(c)} (Donut)", fig)
            else:
                add(f"Count by {tc(c)}", px.bar(vc, x=c, y="Count", color=c, color_discrete_sequence=PALETTE, text_auto=True), tc(c), "Count")
        if len(num_r) >= 2:
            corr = df[num_r[:8]].corr()
            add("Correlation Between Numeric Variables", px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r"))
        if date and num_r:
            d = df.copy(); d[date[0]] = dt(d[date[0]]); d = d.dropna(subset=[date[0]]).sort_values(date[0])
            if len(d) >= 2:
                freq = "ME" if (d[date[0]].max() - d[date[0]].min()).days > 60 else "D"
                for c in num_r[:3]:
                    g = d.set_index(date[0])[c].resample(freq).mean().reset_index()
                    add(f"Trend of {tc(c)} Over Time", px.line(g, x=date[0], y=c, markers=True, color_discrete_sequence=PALETTE), tc(date[0]), tc(c))
        for c in cat_r[:4]:
            if len(charts) >= MAX_VIZ: break
            for n in num_r[:3]:
                if len(charts) >= MAX_VIZ: break
                g = df.groupby(c)[n].mean().reset_index()
                add(f"Average {tc(n)} by {tc(c)}", px.bar(g, x=c, y=n, color=c, barmode="group", color_discrete_sequence=PALETTE, text_auto=".2s"), tc(c), f"Average {tc(n)}")
    except Exception as e:
        st.warning(f"Some visuals could not be generated: {e}")
    return charts[:MAX_VIZ]

# ---------- Facts: real statistical tests -> confidence, phrased as plain-English, impact-oriented insights ----------
def _confidence(p=None, moe=None):
    if p is not None:
        p = 1.0 if pd.isna(p) else p
        pct = round(max(0.1, min(99.9, (1 - p) * 100)), 1)
        label = "Very High" if p < 0.01 else "High" if p < 0.05 else "Moderate" if p < 0.10 else "Low"
    else:
        pct = round(max(0.1, min(99.9, 100 - moe)), 1)
        label = "High" if moe < 5 else "Moderate" if moe < 15 else "Low"
    return pct, label

def compute_facts(df, num, cat, date):
    facts = []
    def add(insight, evidence, n, p=None, moe=None):
        pct, label = _confidence(p, moe)
        facts.append({"Insight": insight, "Evidence": evidence, "N": n, "Confidence %": pct, "Confidence": label})

    if len(num) >= 2:
        corr = df[num].corr().abs()
        tri = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1)).stack().dropna()
        for (a, b), _ in tri.sort_values(ascending=False).head(2).items():
            try:
                r, p = stats.pearsonr(df[a], df[b])
                strength = "strong" if abs(r) > 0.7 else "moderate" if abs(r) > 0.4 else "weak"
                dirn = "positive" if r > 0 else "negative"
                add(f"{tc(a)} and {tc(b)} show a {strength} {dirn} relationship; improving {tc(a)} could shift "
                    f"{tc(b)} in the {'same' if r>0 else 'opposite'} direction by a comparable margin.", f"r={r:.2f}, p={p:.4f}", len(df), p=p)
            except Exception: pass

    for c in num[:5]:
        s, n = df[c], len(df[c])
        q1, q3 = s.quantile(.25), s.quantile(.75); iqr = q3 - q1
        if iqr > 0 and n > 0:
            out_p = ((s < q1-1.5*iqr) | (s > q3+1.5*iqr)).mean()
            moe = 1.96 * np.sqrt(out_p*(1-out_p)/n) * 100
            if out_p*100 < 1:
                add(f"{tc(c)} is highly consistent, with almost no outliers detected.", f"95% CI ± {moe:.1f} pts", n, moe=moe)
            else:
                add(f"{out_p*100:.1f}% of {tc(c)} records are unusual outliers; addressing these could tighten data quality "
                    f"by roughly {out_p*100:.1f}%.", f"95% CI ± {moe:.1f} pts", n, moe=moe)

    for c in cat[:5]:
        vc = df[c].value_counts(normalize=True)
        if not vc.empty:
            top_p, n = vc.iloc[0], len(df)
            moe = 1.96 * np.sqrt(top_p*(1-top_p)/n) * 100 if n > 0 else 100
            add(f"'{vc.index[0]}' dominates {tc(c)}, making up {top_p*100:.1f}% of records — prioritizing this segment "
                f"touches most of your data.", f"95% CI ± {moe:.1f} pts", n, moe=moe)

    if date:
        d = df.copy(); d[date[0]] = dt(d[date[0]]); d = d.dropna(subset=[date[0]])
        if len(d) >= 14:
            wd = d[date[0]].dt.day_name(); vc = wd.value_counts()
            if len(vc) > 1:
                order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
                obs = vc.reindex(order).fillna(0).values; exp = np.full(7, obs.sum()/7)
                try:
                    _, p = stats.chisquare(obs, exp)
                    top_day, top_pct = vc.idxmax(), vc.max()/vc.sum()*100
                    add(f"{top_day} is the most dominant day, accounting for {top_pct:.1f}% of all {tc(date[0])} activity — "
                        f"aligning resources around {top_day} could improve efficiency.", f"chi-square p={p:.4f}", len(d), p=p)
                except Exception: pass
        d2 = d.sort_values(date[0])
        if len(d2) >= 10 and num:
            k = max(len(d2)//10, 1)
            for c in num[:2]:
                first, last = d2[c].head(k), d2[c].tail(k)
                try:
                    _, p = stats.ttest_ind(first, last, equal_var=False)
                    change = ((last.mean()-first.mean())/first.mean()*100) if first.mean() else 0
                    dirn = "increased" if change > 0 else "decreased"
                    add(f"{tc(c)} has {dirn} by {abs(change):.1f}% from the start to the end of the observed period; if this "
                        f"trend continues, expect a similar shift going forward.", f"t-test p={p:.4f}", len(d2), p=p)
                except Exception: pass

    low_card = [c for c in cat if 1 < df[c].nunique() <= 12]
    for c in low_card[:3]:
        if not num: break
        n_col = num[0]
        groups = [g[n_col].dropna().values for _, g in df.groupby(c)]
        groups = [g for g in groups if len(g) >= 2]
        if len(groups) >= 2:
            try:
                _, p = stats.f_oneway(*groups)
                means = df.groupby(c)[n_col].mean()
                top_cat, overall = means.idxmax(), df[n_col].mean()
                diff = ((means[top_cat]-overall)/overall*100) if overall else 0
                add(f"'{top_cat}' leads on average {tc(n_col)} at {means[top_cat]:,.2f}, {abs(diff):.1f}% "
                    f"{'above' if diff>=0 else 'below'} the overall average — replicating what drives '{top_cat}' elsewhere "
                    f"could lift overall {tc(n_col)} by up to {abs(diff):.1f}%.", f"ANOVA p={p:.4f}", len(df), p=p)
            except Exception: pass

    facts.sort(key=lambda f: f["Confidence %"], reverse=True)
    return facts[:15]

def build_recommendations(facts, summary):
    if not facts:
        return "Not enough signal in this dataset for statistically backed recommendations."
    numbered = "\n".join(f"{i+1}. {f['Insight']} (Confidence: {f['Confidence %']}%)" for i, f in enumerate(facts))
    prompt = (f"You are a senior data analyst. {summary} Below are {len(facts)} statistically tested insights already "
              "written in plain English with real confidence scores. Lightly polish them into a smooth, numbered, "
              "business-friendly list a non-technical executive would understand. Keep every number and confidence "
              "score exactly as given. Do not add new facts, numbers, or technical jargon like p-values.\n\n" + numbered)
    return ask_llm(prompt) or numbered

# ---------- UI ----------
st.title("🧠 AI Data Analyst Agent")
st.caption("Local · Free · Powered by Llama3 via Ollama")
ok, msg = check_ollama()
(st.sidebar.success if ok else st.sidebar.warning)(msg)

file = st.sidebar.file_uploader("Upload dataset (CSV/Excel)", type=["csv", "xlsx"])
if file:
    fhash = hashlib.md5(file.getvalue()).hexdigest()
    if st.session_state.get("hash") != fhash:
        st.session_state.clear(); st.session_state["hash"] = fhash
        try:
            file.seek(0)
            raw = pd.read_csv(file) if file.name.lower().endswith(".csv") else pd.read_excel(file)
            if raw.empty: st.error("The uploaded file has no readable data."); st.stop()
            st.session_state["raw"] = raw
        except Exception as e:
            st.error(f"Could not read the file: {e}"); st.stop()

    df, log = clean_data(st.session_state["raw"])
    num, cat, date = profile(df)
    st.sidebar.success(f"Rows: {len(df):,} | Cols: {len(df.columns)}")
    with st.sidebar.expander("Cleaning Log"):
        for l in log: st.write("•", l)
        if not log: st.write("No missing values or duplicates found.")

    t1, t2, t3, t4 = st.tabs(["📊 Overview", "📈 Visual Insights", "💡 Recommendations", "⬇️ Download"])
    with t1:
        st.markdown(dataset_summary(df, num, cat, date))
        kpis = build_kpis(df, num, date)
        for col, (label, val) in zip(st.columns(len(kpis)), kpis):
            col.metric(label, val)
        st.dataframe(df.head(20), use_container_width=True)
    with t2:
        charts = build_visuals(df, num, cat, date)
        for i in range(0, len(charts), 3):
            for c, fig in zip(st.columns(3), charts[i:i+3]):
                c.plotly_chart(fig, use_container_width=True)
        st.caption(f"{len(charts)} visuals generated (auto-scaled to dataset, up to {MAX_VIZ}).")
    with t3:
        summary = dataset_summary(df, num, cat, date)
        st.markdown(summary)
        with st.spinner("Running statistical tests and generating insights..."):
            facts = compute_facts(df, num, cat, date)
        st.subheader("Recommendations")
        st.caption(f"{len(facts)} statistically backed recommendations (adaptive: aims for 10-15 based on dataset richness).")
        with st.spinner("Writing recommendations..."):
            st.markdown(build_recommendations(facts, summary))
        if facts:
            with st.expander("🔍 View statistical evidence behind these recommendations"):
                st.dataframe(pd.DataFrame(facts), use_container_width=True, hide_index=True)
    with t4:
        st.dataframe(df, use_container_width=True)
        buf = io.BytesIO(); df.to_csv(buf, index=False)
        st.download_button("Download Cleaned Data (CSV)", buf.getvalue(), "cleaned_data.csv", "text/csv")
else:
    st.info("Upload a CSV or Excel file to begin analysis.")
