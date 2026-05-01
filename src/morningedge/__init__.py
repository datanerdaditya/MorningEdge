"""MorningEdge — AI-powered intelligence for leveraged finance.

Module layout
-------------
ingestion   : Pull news from RSS + free APIs; normalise; dedupe.
enrichment  : Entity extraction, event classification, asset routing,
              sentiment scoring (FinBERT + Gemini).
aggregation : Per-asset scoring, narrative clustering, theme summarisation.
storage     : DuckDB persistence layer.
llm         : Gemini client with model routing (Flash-Lite / Flash / Pro).
delivery    : Daily brief generation, RAG chat, dashboard helpers.
dashboard   : Streamlit entry point and UI components.
"""

__version__ = "0.1.0"
