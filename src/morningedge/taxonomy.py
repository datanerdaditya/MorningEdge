"""Asset class taxonomy for MorningEdge.

This is the *opinionated* core of the project: a deliberate hierarchy that
puts leveraged loans and private credit at the centre, with macro and
breadth coverage as supporting context.

Each entry has:
    - id          : stable identifier used in the database
    - label       : human-readable name for the dashboard
    - tier        : "hero" | "macro" | "breadth"
    - description : a one-line semantic anchor used for embedding-based
                    routing (replaces brittle keyword queries).
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class AssetClass:
    id: str
    label: str
    tier: str  # "hero" | "macro" | "breadth"
    description: str


TAXONOMY: list[AssetClass] = [
    # --- HERO: leveraged finance & private credit ---
    AssetClass(
        id="lev_loans",
        label="Leveraged Loans",
        tier="hero",
        description=(
            "Senior secured leveraged loans and broadly syndicated loans (BSL). "
            "LSTA loan index movements, primary market issuance, repricings, "
            "refinancings, loan defaults and recovery rates. Term loan B markets "
            "and institutional loan supply."
        ),
    ),
    AssetClass(
        id="private_credit",
        label="Private Credit",
        tier="hero",
        description=(
            "Direct lending and private debt funds. BDCs (business development "
            "companies), unitranche financing, dry powder, fundraising for credit "
            "funds, covenant trends, and non-bank lending to middle-market "
            "and large-cap borrowers. Apollo, Blackstone, Ares, KKR credit arms."
        ),
    ),
    AssetClass(
        id="clo",
        label="CLOs",
        tier="hero",
        description=(
            "Collateralised loan obligations. CLO primary market issuance, "
            "CLO equity arbitrage, warehouse facilities, AAA tranche spreads, "
            "CLO refinancing and resets, middle-market CLOs."
        ),
    ),
    AssetClass(
        id="high_yield",
        label="High Yield Bonds",
        tier="hero",
        description=(
            "High yield corporate bonds and junk debt. HY spreads versus "
            "Treasuries, fallen angels, rising stars, distressed exchanges, "
            "and high yield mutual fund flows."
        ),
    ),
    # --- MACRO: drivers of credit ---
    AssetClass(
        id="rates",
        label="Rates",
        tier="macro",
        description=(
            "Interest rate decisions and central bank monetary policy. "
            "The Fed cuts or hikes rates by basis points. FOMC meetings, "
            "Federal Reserve policy. ECB and Bank of England rate moves. "
            "Yield curve, Treasury yields, government bond markets, "
            "inflation prints CPI and PCE, and central bank communication."
        ),
    ),
    AssetClass(
        id="banks",
        label="Banks & Financials",
        tier="macro",
        description=(
            "Commercial and investment banks. Loan loss provisions, net interest "
            "margins, regional bank stress, deposit flight, bank earnings, "
            "distressed debt activity at lenders, and banking regulation."
        ),
    ),
    AssetClass(
        id="risk_equity",
        label="Equity Risk Signal",
        tier="macro",
        description=(
            "Broad equity market direction as a risk-on/risk-off proxy. "
            "S&P 500 index moves, VIX volatility spikes, major market drawdowns, "
            "and broad sell-offs or rallies that signal macro sentiment."
        ),
    ),
    # --- BREADTH: contextual coverage ---
    AssetClass(
        id="tech",
        label="Tech Sector",
        tier="breadth",
        description=(
            "Semiconductor industry, AI infrastructure capital expenditure, "
            "hyperscaler cloud spending, and large software platform companies. "
            "NVIDIA, TSMC, Microsoft, Google, Apple, Meta, Amazon strategic moves."
        ),
    ),
    AssetClass(
        id="energy",
        label="Energy Sector",
        tier="breadth",
        description=(
            "Oil and gas exploration and production, OPEC supply decisions, "
            "crude oil price movements, refining margins, LNG, and the energy "
            "transition impact on traditional fossil fuel companies."
        ),
    ),
    AssetClass(
        id="healthcare",
        label="Healthcare Sector",
        tier="breadth",
        description=(
            "Pharmaceutical drug approvals, FDA decisions, biotech clinical "
            "trial results, healthcare merger and acquisition activity, "
            "and pharma pipeline announcements."
        ),
    ),
    AssetClass(
        id="us_macro",
        label="US Macro",
        tier="breadth",
        description=(
            "United States economic indicators. Non-farm payrolls, GDP growth, "
            "ISM manufacturing, retail sales, recession indicators, fiscal "
            "policy, debt ceiling, and Treasury issuance."
        ),
    ),
    AssetClass(
        id="europe_macro",
        label="Europe Macro",
        tier="breadth",
        description=(
            "Eurozone economic data, ECB policy stance, sovereign bond spreads "
            "between core and periphery countries, German Bund yields, and "
            "growth indicators for Germany, France, Italy, Spain."
        ),
    ),
    AssetClass(
        id="asia_em",
        label="Asia & Emerging Markets",
        tier="breadth",
        description=(
            "China economic data and property sector, Japan policy and yen, "
            "India growth story, and emerging market sovereign bonds and "
            "corporate credit including default and restructuring stories."
        ),
    ),
    AssetClass(
        id="commodities",
        label="Commodities",
        tier="breadth",
        description=(
            "Crude oil, natural gas, gold and precious metals, industrial "
            "metals like copper and aluminium, and agricultural commodities "
            "wheat corn soybeans."
        ),
    ),
    AssetClass(
        id="fx",
        label="FX",
        tier="breadth",
        description=(
            "Currency markets and foreign exchange. US dollar index DXY, "
            "EUR USD, USD JPY, emerging market currencies, dollar funding "
            "stress and currency intervention."
        ),
    ),
]


def by_tier(tier: str) -> list[AssetClass]:
    """Return all asset classes in a given tier."""
    return [ac for ac in TAXONOMY if ac.tier == tier]


def by_id(asset_id: str) -> AssetClass | None:
    """Look up an asset class by its stable id."""
    return next((ac for ac in TAXONOMY if ac.id == asset_id), None)
