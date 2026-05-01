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
            "Senior secured leveraged loans, broadly syndicated loans (BSL), "
            "LSTA loan index, repricings, refinancings, primary issuance, "
            "loan defaults, and recovery rates."
        ),
    ),
    AssetClass(
        id="private_credit",
        label="Private Credit",
        tier="hero",
        description=(
            "Direct lending, private debt funds, BDCs, dry powder, fundraising, "
            "covenant trends, fund launches, unitranche financing, and "
            "non-bank lending to middle-market and large-cap borrowers."
        ),
    ),
    AssetClass(
        id="clo",
        label="CLOs",
        tier="hero",
        description=(
            "Collateralised loan obligations, CLO issuance, CLO equity, "
            "warehouse facilities, AAA spreads, refinancing/resets, "
            "and middle-market CLOs."
        ),
    ),
    AssetClass(
        id="high_yield",
        label="High Yield Bonds",
        tier="hero",
        description=(
            "High yield corporate bonds, junk debt, fallen angels, rising stars, "
            "HY spreads vs Treasuries, and HY mutual fund flows."
        ),
    ),
    # --- MACRO: drivers of credit ---
    AssetClass(
        id="rates",
        label="Rates",
        tier="macro",
        description=(
            "Federal Reserve, ECB, Bank of England policy rates, yield curve, "
            "Treasury auctions, inflation prints, and central bank communication."
        ),
    ),
    AssetClass(
        id="banks",
        label="Banks & Financials",
        tier="macro",
        description=(
            "Large commercial and investment banks, distressed debt activity, "
            "loan loss provisions, regional banks, and lender earnings."
        ),
    ),
    AssetClass(
        id="risk_equity",
        label="Equity Risk Signal",
        tier="macro",
        description=(
            "Broad equity indices as a risk-on/risk-off proxy — S&P 500, "
            "VIX, large drawdowns, and major earnings surprises."
        ),
    ),
    # --- BREADTH: contextual coverage ---
    AssetClass(
        id="tech",
        label="Tech Sector",
        tier="breadth",
        description="Technology equities, semiconductors, AI capex, mega-cap tech earnings.",
    ),
    AssetClass(
        id="energy",
        label="Energy Sector",
        tier="breadth",
        description="Oil and gas equities, OPEC, crude prices, energy transition.",
    ),
    AssetClass(
        id="healthcare",
        label="Healthcare Sector",
        tier="breadth",
        description="Pharma, biotech, FDA actions, healthcare M&A.",
    ),
    AssetClass(
        id="us_macro",
        label="US Macro",
        tier="breadth",
        description="US economic data, recession indicators, fiscal policy.",
    ),
    AssetClass(
        id="europe_macro",
        label="Europe Macro",
        tier="breadth",
        description="Eurozone data, ECB, sovereign spreads, German and French growth.",
    ),
    AssetClass(
        id="asia_em",
        label="Asia & Emerging Markets",
        tier="breadth",
        description="China, Japan, India, EM sovereign and corporate credit.",
    ),
    AssetClass(
        id="commodities",
        label="Commodities",
        tier="breadth",
        description="Oil, gold, industrial metals, agricultural commodities.",
    ),
    AssetClass(
        id="fx",
        label="FX",
        tier="breadth",
        description="USD, EUR, JPY, EM currencies, dollar funding stress.",
    ),
]


def by_tier(tier: str) -> list[AssetClass]:
    """Return all asset classes in a given tier."""
    return [ac for ac in TAXONOMY if ac.tier == tier]


def by_id(asset_id: str) -> AssetClass | None:
    """Look up an asset class by its stable id."""
    return next((ac for ac in TAXONOMY if ac.id == asset_id), None)
