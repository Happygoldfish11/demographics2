"""
Streamlit UI for the BISG / BIFSG estimator.

Run with::

    streamlit run app.py

This is the user-facing entry point. The interesting code is in the
``bisg_estimator`` package.
"""

from __future__ import annotations

import io
import textwrap
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from bisg_estimator import (
    NATIONAL_SHARES,
    RACE_CATEGORIES,
    RACE_LABELS,
    EstimationResult,
    default_reference_data,
    estimate,
    estimate_batch,
)
from bisg_estimator.employer import _DEMO_EMPLOYER_TABLE


# ---------------------------------------------------------------------------
# Page setup & styling
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="BISG / BIFSG Race Probability Estimator",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Brand palette — deliberately editorial, not "AI dashboard generic". We use
# a single warm accent and let the typography do the work.
PALETTE = {
    "white":    "#4F6D7A",  # slate
    "black":    "#7A4F62",  # mulberry
    "api":      "#7A6B4F",  # ochre
    "native":   "#4F7A57",  # sage
    "multiple": "#6B4F7A",  # plum
    "hispanic": "#B05A3C",  # terracotta (the warm accent)
}

st.markdown(
    """
    <style>
    /* Tighten the default Streamlit chrome and lean into a more editorial feel. */
    .block-container { padding-top: 2rem; padding-bottom: 4rem; max-width: 1200px; }
    h1, h2, h3 { letter-spacing: -0.02em; }
    h1 { font-weight: 700; font-size: 2.4rem; margin-bottom: 0.25rem; }
    .lede { color: #4a4a4a; font-size: 1.05rem; line-height: 1.5; max-width: 60ch; }
    .stat-card { background: #fafafa; border-left: 3px solid #B05A3C; padding: 1rem 1.25rem; margin: 0.5rem 0; }
    .stat-label { font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.08em; color: #666; }
    .stat-value { font-size: 1.5rem; font-weight: 600; color: #222; }
    .footnote { font-size: 0.85rem; color: #666; line-height: 1.5; }
    code { background: #f3f3f3; padding: 0 0.3em; border-radius: 2px; font-size: 0.9em; }
    .pill { display: inline-block; padding: 0.15rem 0.6rem; border-radius: 999px;
            font-size: 0.75rem; font-weight: 600; background: #f0e6df; color: #5c3520;
            margin-right: 0.4rem; }
    .pill.warn { background: #fff3e0; color: #8a4d00; }
    .pill.err  { background: #fde8e8; color: #8a1c1c; }
    .pill.ok   { background: #e8f3eb; color: #245232; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def _ref_data():
    return default_reference_data()


def _race_to_label(c: str) -> str:
    return RACE_LABELS[c]


def _probabilities_df(probs_dict: dict) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "race": [_race_to_label(c) for c in probs_dict],
            "key": list(probs_dict.keys()),
            "probability": list(probs_dict.values()),
        }
    ).sort_values("probability", ascending=False)


def _bar_chart(probs_dict: dict, *, title: str = "", height: int = 320) -> go.Figure:
    df = _probabilities_df(probs_dict)
    fig = px.bar(
        df,
        x="probability",
        y="race",
        orientation="h",
        text=df["probability"].map(lambda p: f"{p*100:.1f}%"),
        color="key",
        color_discrete_map=PALETTE,
    )
    fig.update_traces(textposition="outside", cliponaxis=False)
    fig.update_layout(
        title=title,
        height=height,
        showlegend=False,
        xaxis_title="Probability",
        yaxis_title="",
        xaxis_tickformat=".0%",
        xaxis_range=[0, 1.05],
        margin=dict(l=10, r=20, t=40 if title else 10, b=10),
        plot_bgcolor="white",
        font=dict(family="-apple-system, system-ui, sans-serif", size=13),
    )
    fig.update_yaxes(autorange="reversed", showgrid=False)
    fig.update_xaxes(showgrid=True, gridcolor="#eee")
    return fig


def _stage_comparison_chart(result: EstimationResult) -> go.Figure:
    """Show the four estimates side-by-side so the user can see how each
    successive piece of evidence shifts the distribution."""
    stages = [
        ("National baseline", result.national.as_dict()),
        ("Surname only", result.surname_only.as_dict()),
        ("BISG (+ geography)", result.bisg.as_dict()),
        ("BIFSG (+ first name)", result.bifsg.as_dict()),
    ]
    if result.employer_evidence.found:
        stages.append(("Final (+ employer)", result.final.as_dict()))

    rows = []
    for stage_name, dist in stages:
        for race, p in dist.items():
            rows.append(
                {
                    "stage": stage_name,
                    "race": race,
                    "label": _race_to_label(race),
                    "probability": p,
                }
            )
    df = pd.DataFrame(rows)

    fig = px.bar(
        df,
        x="stage",
        y="probability",
        color="race",
        color_discrete_map=PALETTE,
        category_orders={"stage": [s[0] for s in stages]},
        labels={"probability": "Probability", "stage": ""},
    )
    fig.update_layout(
        height=420,
        barmode="stack",
        yaxis_tickformat=".0%",
        legend_title="",
        plot_bgcolor="white",
        font=dict(family="-apple-system, system-ui, sans-serif", size=13),
        margin=dict(l=10, r=10, t=10, b=40),
    )
    # Replace race keys with friendly labels in the legend.
    fig.for_each_trace(
        lambda tr: tr.update(name=RACE_LABELS.get(tr.name, tr.name))
    )
    return fig


def _input_chip(status: str) -> str:
    cls = {
        "found": "ok",
        "missing": "warn",
        "not_found": "warn",
        "geocode_failed": "warn",
        "zcta_not_in_table": "warn",
    }.get(status, "err")
    pretty = {
        "found": "matched",
        "missing": "not provided",
        "not_found": "not in reference data",
        "geocode_failed": "geocode failed",
        "zcta_not_in_table": "ZCTA missing from table",
    }.get(status, status)
    return f'<span class="pill {cls}">{pretty}</span>'


# ---------------------------------------------------------------------------
# Sidebar — settings, methodology, references
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("### Settings")
    skip_geocoding = st.checkbox(
        "Skip Census Geocoder (use ZIP from address only)",
        value=False,
        help=(
            "When checked, we don't make a network call to the Census "
            "Geocoder API. We just extract a 5-digit ZIP from the address "
            "string and use it as a ZCTA proxy. Faster, but less accurate "
            "for ZIPs that don't map cleanly to a single ZCTA."
        ),
    )
    show_diagnostics = st.checkbox("Show diagnostic detail", value=True)
    st.markdown("---")
    st.markdown("### About")
    st.markdown(
        textwrap.dedent(
            """
            This tool implements the **Bayesian Improved Surname
            Geocoding** (BISG) method developed by RAND Corporation and
            its first-name extension **BIFSG** (Voicu 2018).

            BISG is widely used by the **Consumer Financial Protection
            Bureau** for fair-lending oversight, by health-services
            researchers studying racial disparities in care, and in
            voting-rights litigation where self-reported race is
            unavailable.

            Reference data:
            - **Surnames** — U.S. Census 2010 surname file (~162k surnames)
            - **First names** — Tzioumis (2018), Sci. Data 5:180025
            - **Geography** — Census 2010 ZCTA × race tabulations

            See *docs/METHODOLOGY.md* and *docs/LIMITATIONS.md* in this
            repository for the full picture.
            """
        )
    )

    ref = _ref_data()
    cov = ref.coverage_summary()
    st.markdown("---")
    st.markdown("### Reference data coverage")
    st.markdown(
        f"- Surnames: **{cov['surnames']:,}**  \n"
        f"- First names: **{cov['first_names']:,}**  \n"
        f"- ZCTAs: **{cov['zctas']:,}**"
    )


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.markdown("# Race probability estimator")
st.markdown(
    '<p class="lede">Bayesian Improved First-name Surname Geocoding '
    "(BIFSG), implemented per RAND's published methodology and "
    "validated bit-for-bit against the canonical <code>surgeo</code> "
    'reference. Estimates a probability distribution over six '
    "Census race / ethnicity categories given a person's name and "
    "location.</p>",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab_single, tab_batch, tab_methodology, tab_limitations = st.tabs(
    ["Single estimate", "Batch (CSV)", "Methodology", "Limitations & ethics"]
)


# === TAB 1: Single estimate ================================================

with tab_single:
    col_in, col_out = st.columns([2, 3], gap="large")

    with col_in:
        st.markdown("#### Inputs")
        with st.form("estimate_form"):
            first_name = st.text_input(
                "First name",
                value=st.session_state.get("first_name", "Maria"),
                help="Compared against the Tzioumis (2018) first-name "
                     "table (~4,250 names).",
            )
            surname = st.text_input(
                "Last name (surname)",
                value=st.session_state.get("surname", "Garcia"),
                help="Compared against the Census 2010 surname file "
                     "(~162,000 surnames).",
            )
            address = st.text_input(
                "Address",
                value=st.session_state.get(
                    "address", "350 Fifth Avenue, New York, NY 10118"
                ),
                help="Resolved to a Census ZCTA via the public Census "
                     "Geocoder API. ZIP-only inputs are also accepted.",
            )

            employer_options = ["(none)"] + sorted(_DEMO_EMPLOYER_TABLE.keys())
            employer_input_kind = st.radio(
                "Employer",
                ["From bundled list", "Free text"],
                horizontal=True,
                help="Employer is *not* part of the standard BIFSG "
                     "method. We only apply this evidence when a "
                     "published diversity-report distribution is "
                     "available for the named employer.",
            )
            if employer_input_kind == "From bundled list":
                employer_pick = st.selectbox(
                    "Pick an employer", employer_options, index=0
                )
                employer = "" if employer_pick == "(none)" else employer_pick
            else:
                employer = st.text_input(
                    "Employer (free text)",
                    value=st.session_state.get("employer", ""),
                )

            submitted = st.form_submit_button(
                "Estimate", type="primary", use_container_width=True
            )

        st.markdown(
            '<p class="footnote">Examples to try: '
            '<code>Wei Chen, San Francisco 94110</code>; '
            '<code>LaToya Washington, Atlanta 30303</code>; '
            "<code>David Goldberg, Brooklyn 11201</code>.</p>",
            unsafe_allow_html=True,
        )

    if submitted or "last_result" in st.session_state:
        if submitted:
            with st.spinner("Estimating…"):
                result = estimate(
                    first_name=first_name or None,
                    surname=surname or None,
                    address=address or None,
                    employer=employer or None,
                    skip_geocoding=skip_geocoding,
                )
            st.session_state["last_result"] = result
            st.session_state["first_name"] = first_name
            st.session_state["surname"] = surname
            st.session_state["address"] = address
            st.session_state["employer"] = employer
        else:
            result = st.session_state["last_result"]

        with col_out:
            st.markdown("#### Final estimate")
            top_race, top_p = result.final.most_likely
            second_race, second_p = result.final.top(2)[1]

            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown(
                    f'<div class="stat-card"><div class="stat-label">Most likely</div>'
                    f'<div class="stat-value">{RACE_LABELS[top_race].split(" (")[0]}</div>'
                    f"<div>{top_p*100:.1f}%</div></div>",
                    unsafe_allow_html=True,
                )
            with c2:
                st.markdown(
                    f'<div class="stat-card"><div class="stat-label">2nd most likely</div>'
                    f'<div class="stat-value">{RACE_LABELS[second_race].split(" (")[0]}</div>'
                    f"<div>{second_p*100:.1f}%</div></div>",
                    unsafe_allow_html=True,
                )
            with c3:
                # Confidence proxy: 1 - normalised entropy. 1.0 = certain, 0 = uniform.
                confidence = 1 - result.final.normalised_entropy
                st.markdown(
                    f'<div class="stat-card"><div class="stat-label">Confidence</div>'
                    f'<div class="stat-value">{confidence*100:.0f}%</div>'
                    f"<div>1 − normalised entropy</div></div>",
                    unsafe_allow_html=True,
                )

            st.plotly_chart(
                _bar_chart(result.final.as_dict()),
                use_container_width=True,
                key="final",
            )

            st.markdown("#### How the estimate evolved")
            st.markdown(
                '<p class="footnote">Each column is a successively more '
                "informed estimate. You can see how each piece of "
                "evidence updates the prior.</p>",
                unsafe_allow_html=True,
            )
            st.plotly_chart(
                _stage_comparison_chart(result),
                use_container_width=True,
                key="stages",
            )

        if show_diagnostics:
            st.markdown("---")
            st.markdown("### Diagnostics")

            cdiag1, cdiag2 = st.columns(2)
            with cdiag1:
                st.markdown("#### Inputs used")
                for entry in result.inputs_used:
                    st.markdown(
                        f"**{entry['input'].replace('_', ' ').title()}**: "
                        f"`{entry['value']}` "
                        f"{_input_chip(entry['status'])}",
                        unsafe_allow_html=True,
                    )
                    st.markdown(
                        f'<p class="footnote">{entry["note"]}</p>',
                        unsafe_allow_html=True,
                    )
            with cdiag2:
                st.markdown("#### Geocoding")
                gc = result.geocode
                method_label = {
                    "census": "Census Geocoder API",
                    "zip_fallback": "ZIP fallback (regex)",
                    "none": "Not resolved",
                }[gc.method]
                st.markdown(f"- **Method**: {method_label}")
                if gc.zcta:
                    st.markdown(f"- **ZCTA**: `{gc.zcta}`")
                if gc.matched_address:
                    st.markdown(f"- **Matched address**: {gc.matched_address}")
                if gc.state:
                    st.markdown(f"- **State**: {gc.state}")
                if gc.county:
                    st.markdown(f"- **County**: {gc.county}")
                st.markdown(
                    f'<p class="footnote">{gc.notes}</p>',
                    unsafe_allow_html=True,
                )

            with st.expander("Per-input distributions (the math, exposed)"):
                st.markdown(
                    "These are the three vectors fed into the BIFSG "
                    "calculation. Multiplying them element-wise and "
                    "normalising reproduces the BIFSG row above."
                )
                df_terms = pd.DataFrame(
                    {
                        "race": [_race_to_label(c) for c in RACE_CATEGORIES],
                        "P(race | surname)": result.p_race_given_surname,
                        "P(first_name | race)": result.p_first_name_given_race,
                        "P(geo | race)": result.p_geo_given_race,
                        "BIFSG posterior": [
                            result.bifsg.as_dict()[c] for c in RACE_CATEGORIES
                        ],
                    }
                )
                st.dataframe(df_terms, use_container_width=True)

            with st.expander("Download this estimate as JSON / CSV"):
                df_out = pd.DataFrame(
                    [
                        {
                            "first_name": st.session_state.get("first_name"),
                            "surname": st.session_state.get("surname"),
                            "address": st.session_state.get("address"),
                            "employer": st.session_state.get("employer"),
                            "zcta": result.geocode.zcta,
                            "geocode_method": result.geocode.method,
                            **{
                                f"final_{c}": v
                                for c, v in result.final.as_dict().items()
                            },
                            "most_likely": result.final.most_likely[0],
                            "most_likely_p": result.final.most_likely[1],
                            "confidence": 1 - result.final.normalised_entropy,
                        }
                    ]
                )
                csv_bytes = df_out.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download CSV", csv_bytes, "estimate.csv", "text/csv"
                )
                st.download_button(
                    "Download JSON",
                    df_out.to_json(orient="records", indent=2).encode("utf-8"),
                    "estimate.json",
                    "application/json",
                )


# === TAB 2: Batch ==========================================================

with tab_batch:
    st.markdown("### Batch estimation")
    st.markdown(
        '<p class="lede">Upload a CSV with columns '
        "<code>first_name</code>, <code>surname</code>, "
        "<code>address</code>, and (optionally) <code>employer</code>. "
        "We'll return the same file plus the six BIFSG probabilities, "
        "the most-likely race, the confidence score, and the resolved "
        "ZCTA.</p>",
        unsafe_allow_html=True,
    )

    sample_csv = (
        "first_name,surname,address,employer\n"
        "Maria,Garcia,\"100 Main St, New York, NY 10001\",\n"
        "John,Smith,\"500 California St, San Francisco, CA 94104\",\n"
        "Wei,Chen,\"100 Mission St, San Francisco, CA 94110\",Google\n"
        "LaToya,Washington,\"100 Peachtree St, Atlanta, GA 30303\",\n"
        "David,Goldberg,\"100 Court St, Brooklyn, NY 11201\",\n"
    )
    with st.expander("See example input CSV"):
        st.code(sample_csv, language="csv")
        st.download_button(
            "Download sample CSV",
            sample_csv.encode("utf-8"),
            "sample.csv",
            "text/csv",
        )

    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded is not None:
        df_in = pd.read_csv(uploaded)
        st.markdown(f"**Rows:** {len(df_in)}")
        st.dataframe(df_in.head(20), use_container_width=True)

        required_cols = {"surname"}
        missing_cols = required_cols - set(df_in.columns)
        if missing_cols:
            st.error(f"Missing required columns: {missing_cols}")
        else:
            if st.button("Run batch", type="primary"):
                records = df_in.to_dict(orient="records")
                progress = st.progress(0.0, "Processing…")
                results = []
                for i, rec in enumerate(records):
                    def clean_cell(value):
    if pd.isna(value):
        return None
    return str(value).strip()


results.append(
    estimate(
        first_name=clean_cell(rec.get("first_name")),
        surname=clean_cell(rec.get("surname")),
        address=clean_cell(rec.get("address")),
        employer=clean_cell(rec.get("employer")),
        skip_geocoding=skip_geocoding,
    )
)
                    progress.progress(
                        (i + 1) / len(records),
                        f"Processed {i+1}/{len(records)}",
                    )
                progress.empty()

                out_rows = []
                for rec, r in zip(records, results):
                    row = dict(rec)
                    final = r.final.as_dict()
                    for c in RACE_CATEGORIES:
                        row[f"p_{c}"] = final[c]
                    row["most_likely"] = r.final.most_likely[0]
                    row["most_likely_p"] = r.final.most_likely[1]
                    row["confidence"] = 1 - r.final.normalised_entropy
                    row["zcta"] = r.geocode.zcta
                    row["geocode_method"] = r.geocode.method
                    out_rows.append(row)
                df_out = pd.DataFrame(out_rows)

                st.success(f"Done. {len(df_out)} rows processed.")
                st.dataframe(df_out, use_container_width=True)

                # Distribution chart for the batch
                most_likely_counts = df_out["most_likely"].value_counts()
                st.markdown("#### Distribution of most-likely race in batch")
                fig = px.bar(
                    x=[RACE_LABELS[c] for c in most_likely_counts.index],
                    y=most_likely_counts.values,
                    color=list(most_likely_counts.index),
                    color_discrete_map=PALETTE,
                    labels={"x": "", "y": "Count"},
                )
                fig.update_layout(
                    showlegend=False, height=320, plot_bgcolor="white"
                )
                st.plotly_chart(fig, use_container_width=True)

                csv_bytes = df_out.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download results CSV",
                    csv_bytes,
                    "bisg_results.csv",
                    "text/csv",
                )


# === TAB 3: Methodology ====================================================

with tab_methodology:
    st.markdown(Path("docs/METHODOLOGY.md").read_text() if Path("docs/METHODOLOGY.md").exists()
                else "*(methodology document not found at docs/METHODOLOGY.md)*")


# === TAB 4: Limitations ====================================================

with tab_limitations:
    st.markdown(Path("docs/LIMITATIONS.md").read_text() if Path("docs/LIMITATIONS.md").exists()
                else "*(limitations document not found at docs/LIMITATIONS.md)*")
