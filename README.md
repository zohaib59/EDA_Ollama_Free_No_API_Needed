EDA_Ollama_Free_No_API_Needed

# 🧠 AI Data Analyst Agent
**A fully local, free, AI-powered data analyst — no API keys, no cloud, no cost.**
Upload any CSV or Excel file and get instant, statistically-backed insights: auto-cleaned data, adaptive visualizations, confidence-scored recommendations, and plain-English narratives — all running on your own machine with Llama3 via Ollama.
*Built for analysts, by analysts who don't want to pay for insight.*

## ✨ What It Does

- **Understands your data** — auto-detects numeric, categorical, and date columns; adapts to any dataset shape or size
- **Cleans it like an analyst would** — removes duplicates, imputes missing values using mean/median/mode based on skew,
- not blind defaults
- **Visualizes it (adaptive, up to 20 charts)** — bar, line, pie, donut, clustered column, and correlation heatmaps, styled with a Power BI–style palette, consistent dimensions, and visible axis/value labels
- **5 KPI cards** summarizing the dataset at a glance
- **10–15 statistically-backed recommendations** in plain English — each with a real confidence score derived from actual hypothesis tests (Pearson correlation, t-tests, ANOVA, chi-square), not guesses
- **Every recommendation includes expected impact** — e.g. *"could lift overall Sales by up to 0.9%"* — grounded in the real numbers, never fabricated
- **Nothing persists** — your data lives only for the session; download the cleaned file when you're done

The app works fully even without Ollama running — it falls back to deterministic, statistics-only recommendations instead 
of LLM-polished narration.

## 🛠 Tech Stack

| Layer | Tool |
|---|---|
| UI | Streamlit |
| Data | pandas, numpy |
| Stats | scipy (Pearson, t-test, ANOVA, chi-square) |
| Visuals | Plotly |
| LLM | Llama3 via Ollama (100% local) |


#AI #DataAnalytics #OpenSource #Python #BuildInPublic



