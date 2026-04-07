import json
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
# CONSTANTS
# =========================================================

PREDEFINED_CATEGORIES = [
    "Groceries",
    "Food & Dining",
    "Transport",
    "Shopping",
    "Bills & Utilities",
    "Rent",
    "Salary",
    "Transfer",
    "Entertainment",
    "Healthcare",
    "Investment",
    "Cash Withdrawal",
    "Education",
    "Income",
    "Other",
]

DEFAULT_BUDGET_ROWS = pd.DataFrame({
    "category": ["Groceries", "Food & Dining", "Transport", "Bills & Utilities", "Rent"],
    "budget": [5000, 3000, 2000, 2500, 15000]
})

# =========================================================
# CUSTOM STYLING
# =========================================================

st.markdown("""
<style>
    .stApp {
        background: linear-gradient(180deg, #07111f 0%, #0b1728 100%);
        color: #f4f7fb;
    }

    .block-container {
        padding-top: 1.4rem;
        padding-bottom: 2rem;
        max-width: 1400px;
    }

    h1, h2, h3, h4 {
        color: #f8fbff !important;
        letter-spacing: 0.2px;
    }

    .subtle-text {
        color: #b9c7d8;
        font-size: 0.95rem;
    }

    .hero-box {
        background: linear-gradient(135deg, #0d2038 0%, #123055 100%);
        border: 1px solid rgba(126, 170, 219, 0.20);
        border-radius: 20px;
        padding: 22px 24px;
        margin-bottom: 18px;
        box-shadow: 0 8px 28px rgba(0, 0, 0, 0.20);
    }

    .section-box {
        background: rgba(15, 31, 53, 0.92);
        border: 1px solid rgba(126, 170, 219, 0.18);
        border-radius: 18px;
        padding: 18px;
        margin-bottom: 16px;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.16);
    }

    .metric-box {
        background: linear-gradient(180deg, rgba(16,34,58,0.95) 0%, rgba(12,28,48,0.95) 100%);
        border: 1px solid rgba(126, 170, 219, 0.18);
        border-radius: 18px;
        padding: 16px;
        text-align: left;
        min-height: 105px;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.16);
    }

    .metric-label {
        color: #9eb3cb;
        font-size: 0.9rem;
        margin-bottom: 8px;
    }

    .metric-value {
        color: #f8fbff;
        font-size: 1.9rem;
        font-weight: 700;
        line-height: 1.15;
    }

    .metric-sub {
        color: #c7d6e6;
        font-size: 0.85rem;
        margin-top: 6px;
    }

    .health-box {
        background: linear-gradient(135deg, #0f2743 0%, #17406f 100%);
        border: 1px solid rgba(126, 170, 219, 0.22);
        border-radius: 22px;
        padding: 22px;
        min-height: 220px;
        box-shadow: 0 10px 32px rgba(0, 0, 0, 0.22);
    }

    .health-title {
        color: #bcd1e7;
        font-size: 1rem;
        margin-bottom: 10px;
    }

    .health-score {
        color: #ffffff;
        font-size: 3rem;
        font-weight: 800;
        line-height: 1.1;
    }

    .health-band {
        color: #dbe9f7;
        font-size: 1.1rem;
        margin-top: 6px;
        font-weight: 600;
    }

    .health-conf {
        color: #c7d6e6;
        font-size: 0.95rem;
        margin-top: 12px;
    }

    .insight-chip {
        background: rgba(28, 58, 94, 0.9);
        border: 1px solid rgba(126, 170, 219, 0.16);
        border-radius: 12px;
        padding: 10px 12px;
        margin-bottom: 8px;
        color: #eef5fc;
        font-size: 0.95rem;
    }

    .small-note {
        color: #9eb3cb;
        font-size: 0.85rem;
    }

    div[data-testid="stDataFrame"] {
        border-radius: 14px;
        overflow: hidden;
    }

    div[data-testid="stMetric"] {
        background: transparent;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        margin-bottom: 12px;
    }

    .stTabs [data-baseweb="tab"] {
        background: rgba(16, 33, 56, 0.95);
        border-radius: 12px;
        padding: 10px 16px;
        color: #d9e7f5;
        border: 1px solid rgba(126, 170, 219, 0.12);
    }

    .stTabs [aria-selected="true"] {
        background: #1d4f86 !important;
        color: white !important;
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0a1627 0%, #0c1c31 100%);
        border-right: 1px solid rgba(126, 170, 219, 0.12);
    }

    .stButton>button {
        border-radius: 12px;
        font-weight: 600;
    }

    .stDownloadButton>button {
        border-radius: 12px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# =========================================================
# HELPERS
# =========================================================

def safe_dataframe(df):
    if df is None:
        return pd.DataFrame()
    return df.copy()

def make_download_csv(df):
    if df is None or df.empty:
        return None
    return df.to_csv(index=False).encode("utf-8")

def metric_box(label, value, subtext=""):
    st.markdown(
        f"""
        <div class="metric-box">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-sub">{subtext}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def health_color(score):
    try:
        score = int(score)
    except Exception:
        return "#D0D7E2"
    if score >= 80:
        return "#46d39a"
    elif score >= 65:
        return "#7bb9ff"
    elif score >= 50:
        return "#f2c66d"
    return "#ff8b8b"

def render_health_card(score, band, confidence_label, confidence_pct):
    color = health_color(score)
    st.markdown(
        f"""
        <div class="health-box">
            <div class="health-title">Financial Health Score</div>
            <div class="health-score" style="color:{color};">{score}/100</div>
            <div class="health-band">{band}</div>
            <div class="health-conf">Confidence: <b>{confidence_label}</b> ({confidence_pct}%)</div>
            <div class="small-note" style="margin-top:14px;">
                Confidence-aware scoring reduces overstatement when source quality is imperfect.
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

def build_budget_map(budget_editor_df):
    budget_map = {}
    if budget_editor_df is None or budget_editor_df.empty:
        return budget_map

    for _, row in budget_editor_df.iterrows():
        category = str(row.get("category", "")).strip()
        budget = row.get("budget", 0)
        if category:
            try:
                budget = float(budget)
                if budget > 0:
                    budget_map[category] = budget
            except Exception:
                continue
    return budget_map

def get_merchant_suggestions(memory, transactions_df=None):
    suggestions = set()

    merchant_category_map = memory.get("merchant_category_map", {})
    merchant_normalization_map = memory.get("merchant_normalization_map", {})

    suggestions.update([str(k).strip() for k in merchant_category_map.keys() if str(k).strip()])
    suggestions.update([str(k).strip() for k in merchant_normalization_map.keys() if str(k).strip()])
    suggestions.update([str(v).strip() for v in merchant_normalization_map.values() if str(v).strip()])

    if transactions_df is not None and not transactions_df.empty:
        if "normalized_merchant" in transactions_df.columns:
            suggestions.update(
                transactions_df["normalized_merchant"].dropna().astype(str).str.strip().tolist()
            )
        if "merchant" in transactions_df.columns:
            suggestions.update(
                transactions_df["merchant"].dropna().astype(str).str.strip().tolist()
            )

    suggestions = sorted([s for s in suggestions if s and s.lower() != "unknown"])
    return suggestions

def show_empty_state():
    st.markdown('<div class="section-box">', unsafe_allow_html=True)
    st.info("Upload at least one CSV/PDF or paste transaction text to begin analysis.")
    st.markdown('</div>', unsafe_allow_html=True)

def chart_layout(fig):
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(12,28,48,0.65)",
        font=dict(color="#E8F0F8"),
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig

# =========================================================
# SESSION DEFAULTS
# =========================================================

if "result" not in st.session_state:
    st.session_state["result"] = None

if "budget_editor" not in st.session_state:
    st.session_state["budget_editor"] = DEFAULT_BUDGET_ROWS.copy()

# =========================================================
# HERO
# =========================================================

st.markdown(
    """
    <div class="hero-box">
        <h1 style="margin-bottom:6px;">💰 Cognitive RPA for Personal Finance Data Consolidation</h1>
        <div class="subtle-text">
            Consolidate financial data from CSV, PDF, and SMS/email text into a unified transaction system with
            validation, deduplication, categorization, recurring detection, anomaly detection, learning memory,
            budget tracking, and confidence-aware financial health scoring.
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.markdown("## Data Inputs")

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
    height=160,
    placeholder="Example:\n02/04/2026 UPI paid to Swiggy 340\n03/04/2026 Salary credited 45000"
)

st.sidebar.markdown("---")
st.sidebar.markdown("## Run Pipeline")
run_btn = st.sidebar.button("Run Analysis", type="primary", use_container_width=True)

# =========================================================
# BUDGET EDITOR
# =========================================================

st.markdown('<div class="section-box">', unsafe_allow_html=True)
st.subheader("Budget Setup")
st.caption("Use the editable table below to set category-wise budgets more easily.")

budget_editor = st.data_editor(
    st.session_state["budget_editor"],
    num_rows="dynamic",
    use_container_width=True,
    key="budget_editor_widget",
    column_config={
        "category": st.column_config.SelectboxColumn(
            "Category",
            help="Select a budget category",
            options=PREDEFINED_CATEGORIES,
            required=True,
        ),
        "budget": st.column_config.NumberColumn(
            "Budget Amount",
            help="Enter planned budget for the category",
            min_value=0.0,
            step=100.0,
            format="%.2f",
            required=True,
        ),
    }
)

st.session_state["budget_editor"] = budget_editor.copy()
budget_map = build_budget_map(budget_editor)
st.markdown('</div>', unsafe_allow_html=True)

# =========================================================
# RUN PIPELINE
# =========================================================

if run_btn:
    with st.spinner("Processing and analyzing transactions..."):
        result = process_financial_data(
            csv_files=csv_files,
            pdf_files=pdf_files,
            text_inputs=[text_input] if text_input.strip() else [],
            budget_map=budget_map,
        )
        st.session_state["result"] = result

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
memory = load_learning_memory()

# =========================================================
# TOP SUMMARY
# =========================================================

score = health.get("health_score", 0)
score_band = health.get("score_band", "Unknown")
confidence_label = health.get("confidence_label", "Low")
confidence_pct = health.get("quality_confidence", 0)

left, right = st.columns([1.05, 2.2], gap="large")

with left:
    render_health_card(score, score_band, confidence_label, confidence_pct)

with right:
    c1, c2, c3, c4 = st.columns(4, gap="medium")
    with c1:
        metric_box("Valid Transactions", len(transactions), "Accepted into final pipeline")
    with c2:
        metric_box("Rejected Rows", int(quality.get("rejected_rows", len(rejected_df))), "Failed validation gate")
    with c3:
        metric_box("Duplicates Removed", int(quality.get("duplicates_removed", len(duplicate_df))), "Filtered using ID + fallback")
    with c4:
        metric_box("Acceptance Rate", f"{quality.get('acceptance_rate_pct', 0)}%", "Valid rows / parsed rows")

    st.markdown('<div class="section-box">', unsafe_allow_html=True)
    st.markdown("### Key Insights")
    if insights:
        for insight in insights[:6]:
            st.markdown(f'<div class="insight-chip">{insight}</div>', unsafe_allow_html=True)
    else:
        st.info("No insights generated yet.")
    st.markdown('</div>', unsafe_allow_html=True)

# =========================================================
# MEMORY MANAGEMENT
# =========================================================

st.markdown('<div class="section-box">', unsafe_allow_html=True)
st.subheader("Learning Memory Controls")
st.caption("Store merchant aliases and category mappings so the system improves over time.")

merchant_suggestions = get_merchant_suggestions(memory, transactions)

m1, m2 = st.columns(2, gap="large")

with m1:
    st.markdown("#### Save Merchant Alias")

    selected_existing_alias = st.selectbox(
        "Select known merchant / alias",
        options=[""] + merchant_suggestions,
        index=0,
        help="Choose an existing merchant if available."
    )

    custom_raw_alias = st.text_input(
        "Or type a new raw merchant name",
        placeholder="Example: abc123 grocery store"
    )

    normalized_alias = st.text_input(
        "Normalized merchant name",
        placeholder="Example: ABC Grocery"
    )

    if st.button("Save Merchant Alias", use_container_width=True):
        raw_value = custom_raw_alias.strip() if custom_raw_alias.strip() else selected_existing_alias.strip()
        norm_value = normalized_alias.strip()

        if raw_value and norm_value:
            remember_merchant_alias(raw_value, norm_value)
            st.success("Merchant alias saved successfully.")
            st.rerun()
        else:
            st.warning("Provide a merchant and normalized merchant name.")

with m2:
    st.markdown("#### Save Merchant Category")

    selected_existing_merchant = st.selectbox(
        "Select merchant",
        options=[""] + merchant_suggestions,
        index=0,
        key="merchant_category_select",
        help="Choose a merchant from memory/current data, or type a new one below."
    )

    custom_merchant_for_category = st.text_input(
        "Or type a new merchant name",
        placeholder="Example: ABC Grocery"
    )

    selected_category = st.selectbox(
        "Select category",
        options=PREDEFINED_CATEGORIES,
        index=0
    )

    if st.button("Save Merchant Category", use_container_width=True):
        merchant_value = custom_merchant_for_category.strip() if custom_merchant_for_category.strip() else selected_existing_merchant.strip()

        if merchant_value and selected_category:
            remember_category(merchant_value, selected_category)
            st.success("Merchant category saved successfully.")
            st.rerun()
        else:
            st.warning("Provide a merchant and select a category.")

st.markdown('</div>', unsafe_allow_html=True)

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
    st.markdown('<div class="section-box">', unsafe_allow_html=True)
    st.subheader("Overview Dashboard")

    if transactions.empty:
        st.warning("No valid transactions available after validation.")
    else:
        tx = transactions.copy()
        tx["date"] = pd.to_datetime(tx["date"], errors="coerce")

        row1_col1, row1_col2 = st.columns(2, gap="large")

        with row1_col1:
            if "direction" in tx.columns:
                dir_summary = tx.groupby("direction", as_index=False)["amount"].sum()
                if not dir_summary.empty:
                    fig = px.pie(
                        dir_summary,
                        names="direction",
                        values="amount",
                        title="Income vs Expense Distribution",
                        hole=0.45,
                    )
                    st.plotly_chart(chart_layout(fig), use_container_width=True)

        with row1_col2:
            if "category" in tx.columns:
                cat_summary = tx.groupby("category", as_index=False)["amount"].sum().sort_values("amount", ascending=False)
                if not cat_summary.empty:
                    fig = px.bar(
                        cat_summary.head(10),
                        x="category",
                        y="amount",
                        title="Top Spending Categories"
                    )
                    st.plotly_chart(chart_layout(fig), use_container_width=True)

        row2_col1, row2_col2 = st.columns(2, gap="large")

        with row2_col1:
            if "normalized_merchant" in tx.columns:
                merch_summary = tx.groupby("normalized_merchant", as_index=False)["amount"].sum().sort_values("amount", ascending=False)
                if not merch_summary.empty:
                    fig = px.bar(
                        merch_summary.head(10),
                        x="normalized_merchant",
                        y="amount",
                        title="Top Merchants by Amount"
                    )
                    st.plotly_chart(chart_layout(fig), use_container_width=True)

        with row2_col2:
            trend = tx.groupby("date", as_index=False)["amount"].sum().sort_values("date")
            if not trend.empty:
                fig = px.line(
                    trend,
                    x="date",
                    y="amount",
                    title="Transaction Amount Trend",
                    markers=True
                )
                st.plotly_chart(chart_layout(fig), use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)

# =========================================================
# TAB 2 - TRANSACTIONS
# =========================================================

with tab2:
    st.markdown('<div class="section-box">', unsafe_allow_html=True)
    st.subheader("Transaction Explorer")

    if transactions.empty:
        st.warning("No valid transactions found.")
    else:
        tx = transactions.copy()

        f1, f2, f3 = st.columns(3, gap="medium")

        with f1:
            source_options = sorted(tx["source"].dropna().astype(str).unique().tolist()) if "source" in tx.columns else []
            source_filter = st.multiselect("Filter by source", options=source_options, default=source_options)

        with f2:
            direction_options = sorted(tx["direction"].dropna().astype(str).unique().tolist()) if "direction" in tx.columns else []
            direction_filter = st.multiselect("Filter by direction", options=direction_options, default=direction_options)

        with f3:
            category_options = sorted(tx["category"].dropna().astype(str).unique().tolist()) if "category" in tx.columns else []
            category_filter = st.multiselect("Filter by category", options=category_options, default=[])

        search_text = st.text_input("Search description / merchant")

        if source_filter and "source" in tx.columns:
            tx = tx[tx["source"].astype(str).isin(source_filter)]
        if direction_filter and "direction" in tx.columns:
            tx = tx[tx["direction"].astype(str).isin(direction_filter)]
        if category_filter and "category" in tx.columns:
            tx = tx[tx["category"].astype(str).isin(category_filter)]
        if search_text:
            q = search_text.lower()
            tx = tx[
                tx["description"].astype(str).str.lower().str.contains(q, na=False) |
                tx["normalized_merchant"].astype(str).str.lower().str.contains(q, na=False)
            ]

        st.dataframe(prepare_transactions_for_display(tx), use_container_width=True, height=470)

        tx_csv = make_download_csv(tx)
        if tx_csv:
            st.download_button(
                "Download filtered transactions CSV",
                data=tx_csv,
                file_name="filtered_transactions.csv",
                mime="text/csv"
            )

    st.markdown('</div>', unsafe_allow_html=True)

# =========================================================
# TAB 3 - ANALYTICS
# =========================================================

with tab3:
    st.markdown('<div class="section-box">', unsafe_allow_html=True)
    st.subheader("Analytical View")

    if transactions.empty:
        st.warning("No valid transactions available for analysis.")
    else:
        tx = transactions.copy()

        a1, a2 = st.columns(2, gap="large")

        with a1:
            st.markdown("#### Recurring Transactions")
            recurring_df = tx[tx["is_recurring"] == True] if "is_recurring" in tx.columns else pd.DataFrame()
            if recurring_df.empty:
                st.info("No recurring transaction patterns detected.")
            else:
                rec_summary = recurring_df.groupby(
                    ["normalized_merchant", "category"], as_index=False
                )["amount"].agg(["count", "mean"]).reset_index()
                rec_summary.columns = ["normalized_merchant", "category", "count", "average_amount"]
                st.dataframe(rec_summary, use_container_width=True, height=260)

        with a2:
            st.markdown("#### Anomaly Detection")
            anomaly_df = tx[tx["is_anomaly"] == True] if "is_anomaly" in tx.columns else pd.DataFrame()
            if anomaly_df.empty:
                st.info("No anomalies detected.")
            else:
                cols = [c for c in ["date", "description", "normalized_merchant", "category", "amount", "anomaly_reason"] if c in anomaly_df.columns]
                st.dataframe(anomaly_df[cols], use_container_width=True, height=260)

        b1, b2 = st.columns(2, gap="large")

        with b1:
            st.markdown("#### Accepted Transactions by Source")
            if "source" in tx.columns:
                source_summary = tx["source"].value_counts().reset_index()
                source_summary.columns = ["source", "count"]
                fig = px.bar(source_summary, x="source", y="count", title="")
                st.plotly_chart(chart_layout(fig), use_container_width=True)

        with b2:
            st.markdown("#### Category Distribution")
            if "category" in tx.columns:
                cat_summary = tx.groupby("category", as_index=False)["amount"].sum().sort_values("amount", ascending=False)
                fig = px.treemap(cat_summary, path=["category"], values="amount", title="")
                st.plotly_chart(chart_layout(fig), use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)

# =========================================================
# TAB 4 - BUDGET & HEALTH
# =========================================================

with tab4:
    st.markdown('<div class="section-box">', unsafe_allow_html=True)
    st.subheader("Budget and Financial Health")

    h1, h2, h3, h4 = st.columns(4, gap="medium")
    with h1:
        metric_box("Health Score", f"{health.get('health_score', 0)}/100", "Overall financial health")
    with h2:
        metric_box("Score Band", health.get("score_band", "Unknown"), "Current score classification")
    with h3:
        metric_box("Confidence", health.get("confidence_label", "Low"), "Reliability of current score")
    with h4:
        metric_box("Confidence %", f"{health.get('quality_confidence', 0)}%", "Data quality confidence")

    st.markdown("#### Health Score Components")
    components = health.get("components", {})
    if components:
        comp_df = pd.DataFrame({
            "Component": list(components.keys()),
            "Value": list(components.values())
        })
        st.dataframe(comp_df, use_container_width=True, height=280)

    st.markdown("#### Budget Tracking")
    if budget_df.empty:
        st.info("No budget output available. Add budgets above and run the analysis.")
    else:
        st.dataframe(budget_df, use_container_width=True, height=320)

        if "category" in budget_df.columns and "spent" in budget_df.columns:
            fig = px.bar(
                budget_df.sort_values("spent", ascending=False),
                x="category",
                y="spent",
                title="Budget Spend by Category"
            )
            st.plotly_chart(chart_layout(fig), use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)

# =========================================================
# TAB 5 - QUALITY & AUDIT TRAIL
# =========================================================

with tab5:
    st.markdown('<div class="section-box">', unsafe_allow_html=True)
    st.subheader("Data Quality and Audit Trail")

    q1, q2, q3, q4 = st.columns(4, gap="medium")
    with q1:
        metric_box("Parsed Rows", quality.get("total_parsed_rows", 0), "Before validation")
    with q2:
        metric_box("Accepted", quality.get("valid_rows_accepted", 0), "Moved into final dataset")
    with q3:
        metric_box("Rejected", quality.get("rejected_rows", 0), "Discarded as noisy/invalid")
    with q4:
        metric_box("Avg Confidence", f"{quality.get('average_valid_confidence_pct', 0)}%", "Mean confidence of accepted rows")

    st.markdown("#### Quality Summary")
    st.json(quality)

    sub1, sub2 = st.tabs(["Rejected Rows", "Duplicate Rows"])

    with sub1:
        if rejected_df.empty:
            st.success("No rejected rows.")
        else:
            st.dataframe(prepare_rejected_for_display(rejected_df), use_container_width=True, height=360)
            rejected_csv = make_download_csv(rejected_df)
            if rejected_csv:
                st.download_button(
                    "Download rejected rows CSV",
                    data=rejected_csv,
                    file_name="rejected_rows.csv",
                    mime="text/csv"
                )

    with sub2:
        if duplicate_df.empty:
            st.success("No duplicate rows detected.")
        else:
            st.dataframe(prepare_duplicates_for_display(duplicate_df), use_container_width=True, height=360)
            duplicate_csv = make_download_csv(duplicate_df)
            if duplicate_csv:
                st.download_button(
                    "Download duplicate rows CSV",
                    data=duplicate_csv,
                    file_name="duplicate_rows.csv",
                    mime="text/csv"
                )

    st.markdown('</div>', unsafe_allow_html=True)

# =========================================================
# TAB 6 - LEARNING MEMORY
# =========================================================

with tab6:
    st.markdown('<div class="section-box">', unsafe_allow_html=True)
    st.subheader("Learning Memory")
    st.caption("These mappings are reused in future runs, helping the system improve over time.")

    merchant_category_map = memory.get("merchant_category_map", {})
    merchant_normalization_map = memory.get("merchant_normalization_map", {})

    left, right = st.columns(2, gap="large")

    with left:
        st.markdown("#### Merchant Category Memory")
        if merchant_category_map:
            mem_df = pd.DataFrame({
                "merchant": list(merchant_category_map.keys()),
                "category": list(merchant_category_map.values())
            })
            st.dataframe(mem_df, use_container_width=True, height=320)
        else:
            st.info("No merchant-category memory saved yet.")

    with right:
        st.markdown("#### Merchant Normalization Memory")
        if merchant_normalization_map:
            alias_df = pd.DataFrame({
                "raw_merchant": list(merchant_normalization_map.keys()),
                "normalized_merchant": list(merchant_normalization_map.values())
            })
            st.dataframe(alias_df, use_container_width=True, height=320)
        else:
            st.info("No merchant alias memory saved yet.")

    st.markdown("#### Memory JSON Preview")
    st.code(json.dumps(memory, indent=2), language="json")
    st.markdown('</div>', unsafe_allow_html=True)
