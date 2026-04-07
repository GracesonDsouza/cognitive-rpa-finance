import json
from io import StringIO

import pandas as pd
import plotly.express as px
import streamlit as st

from processor import (
    process_financial_data,
    remember_category,
    remember_merchant_alias,
    load_learning_memory,
)

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Cognitive RPA for Personal Finance",
    page_icon="💰",
    layout="wide",
)

# =========================================================
# HELPERS
# =========================================================

def safe_dataframe(df):
    if df is None:
        return pd.DataFrame()
    return df.copy()

def metric_card(label, value, help_text=None):
    st.metric(label=label, value=value, help=help_text)

def make_download_csv(df):
    if df is None or df.empty:
        return None
    return df.to_csv(index=False).encode("utf-8")

def normalize_budget_input(text):
    """
    Budget input format:
    Groceries:5000
    Rent:15000
    Food & Dining:3000
    """
    budget_map = {}
    if not text:
        return budget_map

    lines = text.strip().splitlines()
    for line in lines:
        if ":" in line:
            key, value = line.split(":", 1)
            key = key.strip()
            value = value.strip()
            try:
                budget_map[key] = float(value)
            except Exception:
                continue
    return budget_map

def show_empty_state():
    st.info("Upload at least one CSV/PDF or paste transaction text to begin analysis.")

def health_color(score):
    try:
        score = int(score)
    except Exception:
        return "gray"
    if score >= 80:
        return "green"
    elif score >= 65:
        return "blue"
    elif score >= 50:
        return "orange"
    return "red"

def render_health_badge(score, band, confidence_label, confidence_pct):
    color = health_color(score)
    st.markdown(
        f"""
        <div style="
            border-radius:16px;
            padding:18px;
            background-color:#f8f9fa;
            border:1px solid #e6e6e6;
            margin-bottom:10px;">
            <div style="font-size:15px; color:#666;">Financial Health Score</div>
            <div style="font-size:42px; font-weight:700; color:{color}; line-height:1.2;">{score}/100</div>
            <div style="font-size:16px; margin-top:4px;"><b>{band}</b></div>
            <div style="font-size:14px; color:#555; margin-top:6px;">
                Confidence: <b>{confidence_label}</b> ({confidence_pct}%)
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def prepare_transactions_for_display(df):
    df = safe_dataframe(df)
    if df.empty:
        return df
    display_cols = [
        c for c in [
            "date", "description", "normalized_merchant", "category", "amount",
            "direction", "source", "transaction_id", "is_recurring",
            "is_anomaly", "anomaly_reason", "parse_confidence"
        ] if c in df.columns
    ]
    return df[display_cols].copy()

def prepare_rejected_for_display(df):
    df = safe_dataframe(df)
    if df.empty:
        return df
    cols = [c for c in ["source", "raw_text", "validation_reason", "parse_confidence"] if c in df.columns]
    return df[cols].copy()

def prepare_duplicates_for_display(df):
    df = safe_dataframe(df)
    if df.empty:
        return df
    cols = [c for c in ["date", "description", "amount", "source", "duplicate_reason", "transaction_id"] if c in df.columns]
    return df[cols].copy()


# =========================================================
# TITLE / INTRO
# =========================================================

st.title("💰 Cognitive RPA for Personal Finance Data Consolidation")
st.caption(
    "Upload CSVs, bank PDFs, and SMS/email text. The system consolidates, cleans, deduplicates, "
    "categorizes, detects recurring transactions, flags anomalies, tracks budgets, and computes a confidence-aware health score."
)

# =========================================================
# SIDEBAR INPUTS
# =========================================================

st.sidebar.header("Input Sources")

csv_files = st.sidebar.file_uploader(
    "Upload CSV files",
    type=["csv", "xlsx"],
    accept_multiple_files=True
)

pdf_files = st.sidebar.file_uploader(
    "Upload PDF statements",
    type=["pdf"],
    accept_multiple_files=True
)

text_input = st.sidebar.text_area(
    "Paste SMS / email / raw transaction text",
    height=180,
    placeholder="Example:\n02/04/2026 UPI paid to Swiggy 340\n03/04/2026 Salary credited 45000"
)

st.sidebar.header("Budget Setup")

budget_text = st.sidebar.text_area(
    "Enter category budgets",
    height=180,
    placeholder="Groceries:5000\nFood & Dining:3000\nRent:15000\nTransport:2000"
)

budget_map = normalize_budget_input(budget_text)

st.sidebar.header("Learning System")

with st.sidebar.expander("Add merchant alias memory"):
    alias_raw = st.text_input("Raw merchant text", key="alias_raw")
    alias_norm = st.text_input("Normalized merchant name", key="alias_norm")
    if st.button("Save merchant alias"):
        if alias_raw and alias_norm:
            remember_merchant_alias(alias_raw, alias_norm)
            st.success("Merchant alias saved.")
        else:
            st.warning("Enter both raw and normalized merchant values.")

with st.sidebar.expander("Add merchant category memory"):
    mem_merchant = st.text_input("Merchant / normalized merchant", key="mem_merchant")
    mem_category = st.text_input("Category", key="mem_category")
    if st.button("Save merchant category"):
        if mem_merchant and mem_category:
            remember_category(mem_merchant, mem_category)
            st.success("Merchant category saved.")
        else:
            st.warning("Enter both merchant and category.")

run_btn = st.sidebar.button("Run Analysis", type="primary", use_container_width=True)

# =========================================================
# MAIN PIPELINE
# =========================================================

result = None

if run_btn:
    with st.spinner("Processing and analyzing transactions..."):
        result = process_financial_data(
            csv_files=csv_files,
            pdf_files=pdf_files,
            text_inputs=[text_input] if text_input.strip() else [],
            budget_map=budget_map,
        )
    st.session_state["result"] = result

if "result" in st.session_state:
    result = st.session_state["result"]

if not result:
    show_empty_state()
    st.stop()

transactions = safe_dataframe(result.get("transactions"))
budget_df = safe_dataframe(result.get("budget_tracking"))
rejected_df = safe_dataframe(result.get("rejected_rows"))
duplicate_df = safe_dataframe(result.get("duplicate_rows"))
health = result.get("health", {})
insights = result.get("insights", [])
quality = result.get("data_quality_summary", {})

# =========================================================
# TOP SUMMARY
# =========================================================

score = health.get("health_score", 0)
score_band = health.get("score_band", "Unknown")
confidence_label = health.get("confidence_label", "Low")
confidence_pct = health.get("quality_confidence", 0)

col_a, col_b = st.columns([1.1, 2])

with col_a:
    render_health_badge(score, score_band, confidence_label, confidence_pct)

with col_b:
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("Valid Transactions", len(transactions))
    with c2:
        metric_card("Rejected Rows", int(quality.get("rejected_rows", len(rejected_df))))
    with c3:
        metric_card("Duplicates Removed", int(quality.get("duplicates_removed", len(duplicate_df))))
    with c4:
        metric_card("Acceptance Rate", f"{quality.get('acceptance_rate_pct', 0)}%")

    st.markdown("### Key Insights")
    if insights:
        for insight in insights[:5]:
            st.write(f"• {insight}")
    else:
        st.write("No insights generated yet.")

# =========================================================
# TABS
# =========================================================

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Overview",
    "Transactions",
    "Analytics",
    "Budget & Health",
    "Quality & Audit Trail",
    "Learning Memory",
])

# =========================================================
# TAB 1 - OVERVIEW
# =========================================================

with tab1:
    st.subheader("Overview Dashboard")

    if transactions.empty:
        st.warning("No valid transactions available after validation.")
    else:
        tx = transactions.copy()
        tx["date"] = pd.to_datetime(tx["date"], errors="coerce")

        row1_col1, row1_col2 = st.columns(2)

        with row1_col1:
            if "direction" in tx.columns:
                dir_summary = tx.groupby("direction")["amount"].sum().reset_index()
                if not dir_summary.empty:
                    fig = px.pie(
                        dir_summary,
                        names="direction",
                        values="amount",
                        title="Income vs Expense Distribution"
                    )
                    st.plotly_chart(fig, use_container_width=True)

        with row1_col2:
            if "category" in tx.columns:
                cat_summary = tx.groupby("category")["amount"].sum().reset_index().sort_values("amount", ascending=False)
                if not cat_summary.empty:
                    fig = px.bar(
                        cat_summary.head(10),
                        x="category",
                        y="amount",
                        title="Top Spending Categories"
                    )
                    st.plotly_chart(fig, use_container_width=True)

        row2_col1, row2_col2 = st.columns(2)

        with row2_col1:
            if "normalized_merchant" in tx.columns:
                merch_summary = tx.groupby("normalized_merchant")["amount"].sum().reset_index().sort_values("amount", ascending=False)
                if not merch_summary.empty:
                    fig = px.bar(
                        merch_summary.head(10),
                        x="normalized_merchant",
                        y="amount",
                        title="Top Merchants by Amount"
                    )
                    st.plotly_chart(fig, use_container_width=True)

        with row2_col2:
            if "date" in tx.columns:
                trend = tx.groupby("date")["amount"].sum().reset_index().sort_values("date")
                if not trend.empty:
                    fig = px.line(
                        trend,
                        x="date",
                        y="amount",
                        title="Transaction Amount Trend"
                    )
                    st.plotly_chart(fig, use_container_width=True)

# =========================================================
# TAB 2 - TRANSACTIONS
# =========================================================

with tab2:
    st.subheader("Transaction Explorer")

    if transactions.empty:
        st.warning("No valid transactions found.")
    else:
        tx = transactions.copy()

        f1, f2, f3 = st.columns(3)

        with f1:
            source_filter = st.multiselect(
                "Filter by source",
                options=sorted(tx["source"].dropna().astype(str).unique().tolist()) if "source" in tx.columns else [],
                default=sorted(tx["source"].dropna().astype(str).unique().tolist()) if "source" in tx.columns else []
            )

        with f2:
            direction_filter = st.multiselect(
                "Filter by direction",
                options=sorted(tx["direction"].dropna().astype(str).unique().tolist()) if "direction" in tx.columns else [],
                default=sorted(tx["direction"].dropna().astype(str).unique().tolist()) if "direction" in tx.columns else []
            )

        with f3:
            category_filter = st.multiselect(
                "Filter by category",
                options=sorted(tx["category"].dropna().astype(str).unique().tolist()) if "category" in tx.columns else [],
                default=[]
            )

        search_text = st.text_input("Search description / merchant")

        if source_filter and "source" in tx.columns:
            tx = tx[tx["source"].astype(str).isin(source_filter)]

        if direction_filter and "direction" in tx.columns:
            tx = tx[tx["direction"].astype(str).isin(direction_filter)]

        if category_filter and "category" in tx.columns:
            tx = tx[tx["category"].astype(str).isin(category_filter)]

        if search_text:
            search_lower = search_text.lower()
            tx = tx[
                tx["description"].astype(str).str.lower().str.contains(search_lower, na=False) |
                tx["normalized_merchant"].astype(str).str.lower().str.contains(search_lower, na=False)
            ]

        st.dataframe(prepare_transactions_for_display(tx), use_container_width=True, height=450)

        csv_data = make_download_csv(tx)
        if csv_data:
            st.download_button(
                "Download filtered transactions CSV",
                data=csv_data,
                file_name="filtered_transactions.csv",
                mime="text/csv"
            )

# =========================================================
# TAB 3 - ANALYTICS
# =========================================================

with tab3:
    st.subheader("Analytical View")

    if transactions.empty:
        st.warning("No valid transactions available for analysis.")
    else:
        tx = transactions.copy()

        a1, a2 = st.columns(2)

        with a1:
            recurring_df = tx[tx["is_recurring"] == True] if "is_recurring" in tx.columns else pd.DataFrame()
            st.markdown("#### Recurring Transactions")
            if recurring_df.empty:
                st.info("No recurring transaction patterns detected.")
            else:
                rec_summary = recurring_df.groupby(["normalized_merchant", "category"], as_index=False)["amount"].agg(["count", "mean"]).reset_index()
                rec_summary.columns = ["normalized_merchant", "category", "count", "average_amount"]
                st.dataframe(rec_summary, use_container_width=True, height=250)

        with a2:
            anomaly_df = tx[tx["is_anomaly"] == True] if "is_anomaly" in tx.columns else pd.DataFrame()
            st.markdown("#### Anomaly Detection")
            if anomaly_df.empty:
                st.info("No anomalies detected.")
            else:
                show_cols = [c for c in ["date", "description", "normalized_merchant", "category", "amount", "anomaly_reason"] if c in anomaly_df.columns]
                st.dataframe(anomaly_df[show_cols], use_container_width=True, height=250)

        st.markdown("#### Source Mix")
        if "source" in tx.columns:
            source_summary = tx["source"].value_counts().reset_index()
            source_summary.columns = ["source", "count"]
            fig = px.bar(source_summary, x="source", y="count", title="Accepted Transactions by Source")
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### Category Deep Dive")
        if "category" in tx.columns:
            cat_summary = tx.groupby("category", as_index=False)["amount"].sum().sort_values("amount", ascending=False)
            fig = px.treemap(
                cat_summary,
                path=["category"],
                values="amount",
                title="Category Distribution"
            )
            st.plotly_chart(fig, use_container_width=True)

# =========================================================
# TAB 4 - BUDGET & HEALTH
# =========================================================

with tab4:
    st.subheader("Budget and Financial Health")

    h1, h2, h3, h4 = st.columns(4)
    with h1:
        metric_card("Health Score", f"{health.get('health_score', 0)}/100")
    with h2:
        metric_card("Score Band", health.get("score_band", "Unknown"))
    with h3:
        metric_card("Confidence", health.get("confidence_label", "Low"))
    with h4:
        metric_card("Confidence %", f"{health.get('quality_confidence', 0)}%")

    st.markdown("#### Health Score Components")
    components = health.get("components", {})
    if components:
        comp_df = pd.DataFrame({
            "Component": list(components.keys()),
            "Value": list(components.values())
        })
        st.dataframe(comp_df, use_container_width=True, height=300)

    st.markdown("#### Budget Tracking")
    if budget_df.empty:
        st.info("No budget output available. Add budget inputs in the sidebar.")
    else:
        st.dataframe(budget_df, use_container_width=True, height=300)

        if "category" in budget_df.columns and "spent" in budget_df.columns:
            fig = px.bar(
                budget_df.sort_values("spent", ascending=False),
                x="category",
                y="spent",
                title="Budget Spend by Category"
            )
            st.plotly_chart(fig, use_container_width=True)

# =========================================================
# TAB 5 - QUALITY & AUDIT TRAIL
# =========================================================

with tab5:
    st.subheader("Data Quality and Audit Trail")

    q1, q2, q3, q4 = st.columns(4)
    with q1:
        metric_card("Parsed Rows", quality.get("total_parsed_rows", 0))
    with q2:
        metric_card("Accepted", quality.get("valid_rows_accepted", 0))
    with q3:
        metric_card("Rejected", quality.get("rejected_rows", 0))
    with q4:
        metric_card("Avg Confidence", f"{quality.get('average_valid_confidence_pct', 0)}%")

    st.markdown("#### Quality Summary")
    st.json(quality)

    subtab1, subtab2 = st.tabs(["Rejected Rows", "Duplicate Rows"])

    with subtab1:
        if rejected_df.empty:
            st.success("No rejected rows.")
        else:
            st.dataframe(prepare_rejected_for_display(rejected_df), use_container_width=True, height=350)
            rejected_csv = make_download_csv(rejected_df)
            if rejected_csv:
                st.download_button(
                    "Download rejected rows CSV",
                    data=rejected_csv,
                    file_name="rejected_rows.csv",
                    mime="text/csv"
                )

    with subtab2:
        if duplicate_df.empty:
            st.success("No duplicate rows detected.")
        else:
            st.dataframe(prepare_duplicates_for_display(duplicate_df), use_container_width=True, height=350)
            duplicate_csv = make_download_csv(duplicate_df)
            if duplicate_csv:
                st.download_button(
                    "Download duplicate rows CSV",
                    data=duplicate_csv,
                    file_name="duplicate_rows.csv",
                    mime="text/csv"
                )

# =========================================================
# TAB 6 - LEARNING MEMORY
# =========================================================

with tab6:
    st.subheader("Learning Memory")

    memory = load_learning_memory()

    left, right = st.columns(2)

    with left:
        st.markdown("#### Merchant Category Memory")
        merchant_category_map = memory.get("merchant_category_map", {})
        if merchant_category_map:
            mem_df = pd.DataFrame({
                "merchant": list(merchant_category_map.keys()),
                "category": list(merchant_category_map.values())
            })
            st.dataframe(mem_df, use_container_width=True, height=300)
        else:
            st.info("No merchant-category memory saved yet.")

    with right:
        st.markdown("#### Merchant Normalization Memory")
        merchant_normalization_map = memory.get("merchant_normalization_map", {})
        if merchant_normalization_map:
            alias_df = pd.DataFrame({
                "raw_merchant": list(merchant_normalization_map.keys()),
                "normalized_merchant": list(merchant_normalization_map.values())
            })
            st.dataframe(alias_df, use_container_width=True, height=300)
        else:
            st.info("No merchant alias memory saved yet.")

    st.markdown("#### Memory JSON Preview")
    st.code(json.dumps(memory, indent=2), language="json")
