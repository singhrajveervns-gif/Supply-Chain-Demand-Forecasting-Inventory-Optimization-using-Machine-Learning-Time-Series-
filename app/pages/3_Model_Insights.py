"""
Model Insights page (Roadmap Item #25) -- baseline vs. ML vs. classical model comparison.

Loads the saved comparison tables from Notebook 07 (Roadmap Item #15) directly -- no metric
is recomputed here, only visualized.
"""

import streamlit as st
import plotly.express as px

from utils import load_comparison_a, load_comparison_b, load_ml_fold_metrics, load_error_analysis, to_csv_download

st.set_page_config(page_title="Model Insights", page_icon="🔬", layout="wide")
st.title("🔬 Model Comparison & Insights")
st.caption("Baseline vs. Machine Learning vs. Classical Time Series models (Roadmap Item #15)")

comparison_a = load_comparison_a()
comparison_b = load_comparison_b()

# ------------------------------------------------------------
# Comparison A: full catalog, 5-fold
# ------------------------------------------------------------
st.markdown("### Comparison A — Full Catalog, 5-Fold Walk-Forward Average")
st.caption("Baseline vs. ML models, evaluated across all ~6,000 series and all 5 folds")

fig1 = px.bar(
    comparison_a.sort_values("avg_wmape_pct"), x="avg_wmape_pct", y="model", orientation="h",
    color="beats_baseline",
    color_discrete_map={True: "#4C72B0", False: "#C44E52"},
    labels={"avg_wmape_pct": "Average WMAPE (%)", "model": "Model", "beats_baseline": "Beats Baseline"},
    title="Average WMAPE by Model — Full Catalog, 5-Fold",
)
fig1.update_layout(height=420)
st.plotly_chart(fig1, use_container_width=True)

st.dataframe(comparison_a.sort_values("avg_wmape_pct"), use_container_width=True)

best_row = comparison_a.sort_values("avg_wmape_pct").iloc[0]
st.success(
    f"**Best full-catalog model: {best_row['model']}** — {best_row['avg_wmape_pct']:.2f}% WMAPE, "
    f"a {best_row['pct_improvement_vs_baseline']:.1f}% relative improvement over the best baseline."
)

st.markdown("---")

# ------------------------------------------------------------
# Comparison B: fair three-way
# ------------------------------------------------------------
st.markdown("### Comparison B — Fair Three-Way Comparison")
st.caption("Baseline vs. ML vs. Classical, same 180-series sample, same final fold (apples-to-apples)")

family_filter = st.multiselect(
    "Filter by model family", options=sorted(comparison_b["family"].unique().tolist()),
    default=sorted(comparison_b["family"].unique().tolist()),
)
filtered_b = comparison_b[comparison_b["family"].isin(family_filter)]

fig2 = px.bar(
    filtered_b.sort_values("pooled_wmape_pct"), x="pooled_wmape_pct", y="model", orientation="h",
    color="family",
    color_discrete_map={"Baseline": "#C44E52", "ML": "#4C72B0", "Classical": "#55A868"},
    labels={"pooled_wmape_pct": "Pooled WMAPE (%)", "model": "Model", "family": "Model Family"},
    title="Pooled WMAPE — Baseline vs. ML vs. Classical (Same Sample, Final Fold)",
)
fig2.update_layout(height=420)
st.plotly_chart(fig2, use_container_width=True)

st.dataframe(filtered_b.sort_values("pooled_wmape_pct"), use_container_width=True)

st.markdown("---")

# ------------------------------------------------------------
# Fold stability
# ------------------------------------------------------------
st.markdown("### Fold-by-Fold Stability (ML Models)")
st.caption("Lower variance across folds indicates a more production-reliable model")

ml_fold_metrics = load_ml_fold_metrics()
pivot = ml_fold_metrics.pivot(index="fold", columns="model", values="wmape") * 100
pivot_long = pivot.reset_index().melt(id_vars="fold", var_name="model", value_name="wmape_pct")

fig3 = px.line(
    pivot_long, x="fold", y="wmape_pct", color="model", markers=True,
    labels={"fold": "Fold", "wmape_pct": "WMAPE (%)", "model": "Model"},
    title="WMAPE by Fold — ML Model Stability",
)
fig3.update_layout(height=400)
st.plotly_chart(fig3, use_container_width=True)

st.markdown("---")

# ------------------------------------------------------------
# Error analysis breakdown
# ------------------------------------------------------------
st.markdown("### Detailed Error Analysis")
st.caption("Where the best full-catalog model struggles most (Roadmap Item #15, Enhancement #3)")

error_df = load_error_analysis()
breakdown_choice = st.selectbox(
    "Breakdown dimension", sorted(error_df["breakdown"].unique().tolist())
)
breakdown_df = error_df[error_df["breakdown"] == breakdown_choice].sort_values("wmape_pct", ascending=False)

fig4 = px.bar(
    breakdown_df, x="wmape_pct", y="group_value", orientation="h",
    labels={"wmape_pct": "WMAPE (%)", "group_value": breakdown_choice.title()},
    title=f"WMAPE by {breakdown_choice.title()}",
)
fig4.update_layout(height=380)
st.plotly_chart(fig4, use_container_width=True)

st.dataframe(breakdown_df, use_container_width=True)

# ------------------------------------------------------------
# Downloadable outputs
# ------------------------------------------------------------
st.markdown("### Download Comparison Data")
dl_col1, dl_col2, dl_col3 = st.columns(3)
dl_col1.download_button(
    "Comparison A (CSV)", data=to_csv_download(comparison_a),
    file_name="model_comparison_full_catalog.csv", mime="text/csv",
)
dl_col2.download_button(
    "Comparison B (CSV)", data=to_csv_download(comparison_b),
    file_name="model_comparison_fair_sample.csv", mime="text/csv",
)
dl_col3.download_button(
    "Error Analysis (CSV)", data=to_csv_download(error_df),
    file_name="error_analysis_summary.csv", mime="text/csv",
)
