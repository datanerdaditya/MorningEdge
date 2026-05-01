"""Streamlit entry point.

Run with: ``streamlit run src/morningedge/dashboard/app.py``

Layout (Week 3 will fill this in):
    - Sidebar: tier filters (Hero / Macro / Breadth), date range
    - Main: regime banner, hero asset-class tiles, narrative cards,
      drill-down per asset class, RAG chat panel
"""

import streamlit as st


def main() -> None:
    st.set_page_config(
        page_title="MorningEdge",
        page_icon="📈",
        layout="wide",
    )
    st.title("MorningEdge")
    st.caption("AI-powered intelligence for leveraged finance.")
    st.info("🚧 Under construction — see roadmap in README.md")


if __name__ == "__main__":
    main()
