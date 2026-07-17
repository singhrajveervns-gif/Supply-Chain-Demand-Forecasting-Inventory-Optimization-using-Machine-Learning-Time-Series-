"""
Demand Forecast page (Roadmap Item #23) -- interactive actual vs. forecast viewer.

Loads the precomputed forecast_vs_actual_final_fold.csv (Random Forest predictions on the
final walk-forward fold, full catalog) -- no model inference happens in this app.
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from utils import load_forecast_vs_actual, get_category_options, get_store_options, to_csv_download

st.set_page_config(page_title="Demand Forecast", page_icon="📈", layout="wide")
st.title("📈 Demand Forecast — Actual vs. Predicted")
st.caption("Random Forest predictions on the final walk-forward fold (28 days), full catalog")

df = load_forecast_vs_actual()

# ------------------------------------------------------------
# Sidebar filters (Roadmap requirement: category/store filters, product search)
# ------------------------------------------------------------
st.sidebar.header("Filters")

categories = get_category_options(df, "cat_id")
selected_categories = st.sidebar.multiselect("Category", categories, default=categories)

stores = get_store_options(df, "store_id")
selected_stores = st.sidebar.multiselect("Store", stores, default=stores)

filtered_df = df[df["cat_id"].isin(selected_categories) & df["store_id"].isin(selected_stores)]

st.sidebar.markdown("---")
st.sidebar.subheader("Product Search")
item_options = sorted(filtered_df["item_id"].unique().tolist())
if len(item_options) == 0:
    st.warning("No products match the current filters. Adjust the category/store selection.")
    st.stop()

selected_item = st.sidebar.selectbox("Search / Select Product (item_id)", item_options)
store_options_for_item = sorted(filtered_df.loc[filtered_df["item_id"] == selected_item, "store_id"].unique().tolist())
selected_store_for_item = st.sidebar.selectbox("Store for Selected Product", store_options_for_item)

# ------------------------------------------------------------
# KPI cards for the filtered selection
# ------------------------------------------------------------
st.markdown("### Filtered Selection Summary")
kpi1, kpi2, kpi3 = st.columns(3)
kpi1.metric("Products in Selection", f"{filtered_df['item_id'].nunique():,}")
kpi2.metric("Total Actual Units (Final Fold)", f"{filtered_df['actual'].sum():,.0f}")
kpi3.metric("Total Forecast Units (Final Fold)", f"{filtered_df['forecast'].sum():,.0f}")

# ------------------------------------------------------------
# Product-level actual vs. forecast chart
# ------------------------------------------------------------
st.markdown(f"### Actual vs. Forecast — Product `{selected_item}` at `{selected_store_for_item}`")

product_df = filtered_df[
    (filtered_df["item_id"] == selected_item) & (filtered_df["store_id"] == selected_store_for_item)
].sort_values("date")

fig = go.Figure()
fig.add_trace(go.Scatter(x=product_df["date"], y=product_df["actual"], mode="lines+markers", name="Actual"))
fig.add_trace(go.Scatter(x=product_df["date"], y=product_df["forecast"], mode="lines+markers", name="Forecast"))
fig.update_layout(
    title=f"Daily Units Sold — {selected_item} @ {selected_store_for_item}",
    xaxis_title="Date", yaxis_title="Units Sold", height=420, hovermode="x unified",
)
st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------------------
# Category-level aggregate view
# ------------------------------------------------------------
st.markdown("### Category-Level Actual vs. Forecast (Filtered Selection)")
category_agg = filtered_df.groupby(["date", "cat_id"], observed=True)[["actual", "forecast"]].sum().reset_index()

fig2 = px.line(
    category_agg, x="date", y="actual", color="cat_id",
    labels={"date": "Date", "actual": "Total Actual Units", "cat_id": "Category"},
    title="Total Actual Demand by Category (Final Fold)",
)
fig2.update_layout(height=400, hovermode="x unified")
st.plotly_chart(fig2, use_container_width=True)

# ------------------------------------------------------------
# Downloadable output
# ------------------------------------------------------------
st.markdown("### Download Filtered Data")
st.download_button(
    label="Download Filtered Forecast Data (CSV)",
    data=to_csv_download(filtered_df),
    file_name="filtered_forecast_vs_actual.csv",
    mime="text/csv",
)
