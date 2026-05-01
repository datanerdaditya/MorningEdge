# MorningEdge

> **An AI-powered morning brief for the leveraged loans and private credit market.**

MorningEdge is a multi-stage AI pipeline that ingests financial news from free
sources, enriches it with entity extraction, event classification, and
sentiment scoring, clusters related stories into themes, and delivers a
polished daily brief — built specifically around leveraged finance and private
credit, with macro context layered around it.

It is deliberately built on a zero-cost stack: free news sources, local
open-source models for bulk processing, and the free tier of Google Gemini
for higher-order reasoning.

---

## Architecture

```
RSS + free APIs
       │
       ▼
┌──────────────┐
│  Ingestion   │ Pull, normalise, deduplicate
└──────┬───────┘
       ▼
┌──────────────┐
│  Enrichment  │ Entities · Events · Asset routing · Sentiment
└──────┬───────┘   (FinBERT · GLiNER · embeddings · zero-shot)
       ▼
┌──────────────┐
│ Aggregation  │ Score · Cluster narratives · Summarise themes
└──────┬───────┘   (HDBSCAN · Gemini Flash)
       ▼
┌──────────────┐
│   Storage    │ DuckDB
└──────┬───────┘
       ▼
┌──────────────┐
│   Delivery   │ Streamlit dashboard · RAG chat · daily brief
└──────────────┘   (Gemini Pro)
```

## What's covered

**Hero coverage** — Leveraged loans · Private credit · CLOs · High yield
**Macro context** — Rates · Banks · Risk-on/off equities
**Breadth** — Sectors (tech, energy, healthcare) · Regions (US, Europe, Asia, EM) · FX · Commodities

## Tech stack

Python · FinBERT · GLiNER · sentence-transformers · HDBSCAN · DuckDB ·
Gemini 2.5 (Flash-Lite, Flash, Pro) · Streamlit · Plotly · GitHub Actions

## Status

🚧 Under active development. Module 02 of a personal Finance × AI series.

## Run locally

```bash
# Clone and enter
git clone https://github.com/datanerdaditya/MorningEdge.git
cd MorningEdge

# Set up Python environment
python -m venv .venv
source .venv/bin/activate          # macOS / Linux
# .venv\Scripts\activate           # Windows

# Install dependencies
pip install -r requirements.txt

# Configure secrets
cp .env.example .env
# then edit .env and add your Gemini API key

# Run the dashboard
streamlit run src/morningedge/dashboard/app.py
```

## Inspiration

The original spark for this project came from
[finance-ai-stack-01-news-sentiment](https://github.com/khandelwalharshit1307/finance-ai-stack-01-news-sentiment)
by Harshit Khandelwal. MorningEdge takes the core idea — quantifying news
sentiment across asset classes — and rebuilds it as a deeper, AI-native
pipeline focused on leveraged finance.

## License

MIT — see [LICENSE](LICENSE).

---

*Built by [Aditya](https://github.com/datanerdaditya), MiM @ ESSEC Business School.*
