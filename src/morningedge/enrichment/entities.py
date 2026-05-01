"""Entity extraction — companies, tickers, people, places.

Uses GLiNER (or a finance-tuned spaCy model as fallback) to pull structured
entities out of each headline. This is what lets us go from
"financials are bullish" to "JPMorgan is bullish, Goldman is mixed".
"""
