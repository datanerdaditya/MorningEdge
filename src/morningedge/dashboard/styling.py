"""MorningEdge dashboard styling — dark editorial aesthetic.

Inspired by Bloomberg Terminal × Financial Times. Cormorant Garamond for
display headlines, Inter for body copy, JetBrains Mono for data/numbers,
gold accent on near-black background.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Color tokens
# ---------------------------------------------------------------------------

BG_PRIMARY = "#0A0E14"          # near-black, slight blue
BG_SECONDARY = "#11161D"        # cards
BG_TERTIARY = "#1A2029"         # hover/active

ACCENT_GOLD = "#C9A84C"         # the Échelon gold
ACCENT_GOLD_DIM = "#8C7235"     # for subtle accents

TEXT_PRIMARY = "#E8E6E1"        # off-white, warmer than pure white
TEXT_SECONDARY = "#9AA0A8"
TEXT_MUTED = "#5A6068"

POSITIVE = "#4ADE80"            # green for bullish
NEGATIVE = "#F87171"            # red for bearish
NEUTRAL = "#9AA0A8"             # grey for neutral


# ---------------------------------------------------------------------------
# Global CSS — injected at the top of app.py
# ---------------------------------------------------------------------------

CSS = f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@400;500;600;700&family=Inter:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500;600&display=swap');

  /* App background */
  .stApp {{
    background-color: {BG_PRIMARY};
    color: {TEXT_PRIMARY};
  }}

  /* Sidebar */
  [data-testid="stSidebar"] {{
    background-color: {BG_SECONDARY};
  }}

  /* Headings — serif display */
  h1, h2, h3 {{
    font-family: 'Cormorant Garamond', serif !important;
    font-weight: 600 !important;
    color: {TEXT_PRIMARY} !important;
    letter-spacing: -0.01em;
  }}

  h1 {{
    font-size: 3.2rem !important;
    margin-bottom: 0.2rem !important;
  }}

  h2 {{
    font-size: 2.1rem !important;
    color: {ACCENT_GOLD} !important;
    border-bottom: 1px solid {BG_TERTIARY};
    padding-bottom: 0.4rem;
    margin-top: 1.5rem !important;
  }}

  h3 {{
    font-size: 1.3rem !important;
    color: {TEXT_PRIMARY} !important;
  }}

  /* Body */
  body, .stMarkdown, p, li {{
    font-family: 'Inter', sans-serif !important;
    font-weight: 300 !important;
    color: {TEXT_SECONDARY};
    line-height: 1.6;
  }}

  /* Data / metrics — monospace */
  [data-testid="stMetricValue"] {{
    font-family: 'JetBrains Mono', monospace !important;
    font-weight: 500 !important;
    color: {TEXT_PRIMARY} !important;
  }}

  [data-testid="stMetricLabel"] {{
    font-family: 'Inter', sans-serif !important;
    font-size: 0.7rem !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: {TEXT_MUTED} !important;
  }}

  /* Custom utility classes */
  .me-tagline {{
    font-family: 'Cormorant Garamond', serif;
    font-style: italic;
    font-size: 1.1rem;
    color: {ACCENT_GOLD};
    margin-bottom: 2rem;
  }}

  .me-card {{
    background-color: {BG_SECONDARY};
    border: 1px solid {BG_TERTIARY};
    border-radius: 4px;
    padding: 1.25rem;
    margin-bottom: 1rem;
  }}

  .me-card-title {{
    font-family: 'Cormorant Garamond', serif;
    font-size: 1.4rem;
    font-weight: 600;
    color: {TEXT_PRIMARY};
    margin: 0 0 0.4rem 0;
  }}

  .me-card-meta {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem;
    color: {TEXT_MUTED};
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-bottom: 0.6rem;
  }}

  /* Force sidebar always visible — work around Streamlit collapse-arrow bug */
  [data-testid="stSidebar"] {{
    min-width: 280px !important;
    max-width: 280px !important;
  }}

  [data-testid="stSidebarCollapseButton"],
  button[kind="header"][data-testid="stBaseButton-header"] {{
    display: none !important;
  }}

  /* Reclaim the space — main content shouldn't have a button-shaped gap */
  [data-testid="stSidebarCollapsedControl"] {{
    display: none !important;
  }}

  .me-positive {{ color: {POSITIVE} !important; }}
  .me-negative {{ color: {NEGATIVE} !important; }}
  .me-neutral  {{ color: {NEUTRAL} !important; }}

  /* Hide Streamlit chrome */
  #MainMenu, footer, header {{ visibility: hidden; }}

  /* Tighter top padding */
  .block-container {{
    padding-top: 2rem !important;
    max-width: 1200px;
  }}
</style>
"""


def sentiment_color(score: float) -> str:
    """Map a sentiment score to a hex color."""
    if score > 0.1:
        return POSITIVE
    if score < -0.1:
        return NEGATIVE
    return NEUTRAL