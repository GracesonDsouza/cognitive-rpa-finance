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
    page_icon="🟢",
    layout="wide",
)

# =========================================================
# NVIDIA-STYLE THEME
# =========================================================

st.markdown(
    """
    <style>
    :root {
        --bg: #0a0a0a;
        --panel: #111111;
        --panel-2: #161616;
        --panel-3: #1b1b1b;
        --border: #242424;
        --text: #f5f5f5;
        --muted: #a1a1aa;
        --green: #76b900;
        --green-2: #8bd80a;
        --green-soft: rgba(118, 185, 0, 0.12);
        --white-soft: rgba(255,255,255,0.04);
    }

    .stApp {
        background: linear-gradient(180deg, #070707 0%, #0b0b0b 100%);
        color: var(--text);
    }

    [data-testid="stAppViewContainer"] {
        background: linear-gradient(180deg, #070707 0%, #0b0b0b 100%);
    }

    [data-testid="stHeader"] {
        background: rgba(0,0,0,0);
    }

    [data-testid="stSidebar"] {
        background: #0b0b0b;
        border-right: 1px solid var(--border);
    }

    section[data-testid="stSidebar"] * {
        color: var(--text) !important;
    }

    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: white;
        margin-bottom: 0.2rem;
        letter-spacing: -0.02em;
    }

    .accent-line {
        width: 92px;
        height: 4px;
        background: linear-gradient(90deg, var(--green), var(--green-2));
        border-radius: 999px;
        margin: 0.4rem 0 0.9rem 0;
    }

    .sub-title {
        color: var(--muted);
        font-size: 1rem;
        margin-bottom: 1rem;
        max-width: 1100px;
    }

    .section-card {
        background: linear-gradient(180deg, #101010 0%, #121212 100%);
        border: 1px solid var(--border);
        border-radius: 20px;
        padding: 18px 18px 14px 18px;
        box-shadow: 0 12px 30px rgba(0,0,0,0.22);
        margin-bottom: 16px;
    }

    .section-card-strong {
        background: linear-gradient(180deg, #0f0f0f 0%, #141414 100%);
        border: 1px solid rgba(118,185,0,0.25);
        border-radius: 22px;
        padding: 18px;
        box-shadow: 0 0 0 1px rgba(118,185,0,0.06), 0 18px 36px rgba(0,0,0,0.28);
        margin-bottom: 16px;
    }

    .soft-card {
        background: linear-gradient(180deg, #131313 0%, #171717 100%);
        border: 1px solid var(--border);
        border-radius: 18px;
        padding: 14px 16px;
        margin-bottom: 12px;
    }

    .hero-card {
        background:
            radial-gradient(circle at top right, rgba(118,185,0,0.16), transparent 28%),
            linear-gradient(135deg, #101010 0%, #151515 60%, #111111 100%);
        color: white;
        border-radius: 24px;
        border: 1px solid rgba(118,185,0,0.24);
        padding: 24px;
        box-shadow: 0 18px 40px rgba(0,0,0,0.30);
        min-height: 190px;
    }

    .hero-label {
        font-size: 0.95rem;
        color: #d4d4d8;
    }

    .hero-score {
        font-size: 3.2rem;
        font-weight: 800;
        line-height: 1.05;
        margin-top: 8px;
        color: white;
    }

    .hero-band {
        font-size: 1.05rem;
        font-weight: 700;
        margin-top: 8px;
        color: var(--green-2);
    }

    .hero-conf {
        font-size: 0.95rem;
        margin-top: 10px;
        color: #e4e4e7;
    }

    .box-title {
        font-size: 1.05rem;
        font-weight: 700;
        color: white;
        margin-bottom: 10px;
    }

    .small-note {
        color: var(--muted);
        font-size: 0.92rem;
    }

    .mini-pill {
        display: inline-block;
        background: rgba(118,185,0,0.12);
        color: #b7f34d;
        border: 1px solid rgba(118,185,0,0.28);
        border-radius: 999px;
        padding: 5px 10px;
        font-size: 0.82rem;
        font-weight: 600;
        margin-right: 6px;
        margin-bottom: 6px;
    }

    div[data-testid="stMetric"] {
        background: linear-gradient(180deg, #101010 0%, #141414 100%);
        border: 1px solid var(--border);
        padding: 10px 12px;
        border-radius: 18px;
        box-shadow: 0 8px 20px rgba(0,0,0,0.18);
    }

    div[data-testid="stMetricLabel"] {
        color: #d4d4d8;
    }

    div[data-testid="stMetricValue"] {
        color: white;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }

    .stTabs [data-baseweb="tab"] {
        background: #111111;
        border-radius: 13px;
        border: 1px solid var(--border);
        padding: 10px 16px;
        color: #e4e4e7;
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(180deg, #151515 0%, #191919 100%) !important;
        color: white !important;
        border: 1px solid rgba(118,185,0,0.35) !important;
        box-shadow: inset 0 0 0 1px rgba(118,185,0,0.10);
    }

    .stButton > button {
        border-radius: 12px !important;
        border: 1px solid rgba(118,185,0,0.35) !important;
        background: linear-gradient(180deg, #171717 0%, #121212 100%) !important;
        color: white !important;
        font-weight: 600 !important;
    }

    .stButton > button:hover {
        border-color: rgba(118,185,0,0.60) !important;
        box-shadow: 0 0 0 1px rgba(118,185,0,0.10), 0 8px 20px rgba(0,0,0,0.18);
    }

    .stDownloadButton > button {
        border-radius: 12px !important;
        border: 1px solid rgba(118,185,0,0.35) !important;
        background: linear-gradient(180deg, #171717 0%, #121212 100%) !important;
        color: white !important;
        font-weight: 600 !important;
    }

    .stTextInput input,
    .stTextArea textarea,
    .stNumberInput input,
    .stSelectbox div[data-baseweb="select"] > div,
    .stMultiSelect div[data-baseweb="select"] > div {
        background: #111111 !important;
        color: white !important;
        border: 1px solid var(--border) !important;
        border-radius: 12px !important;
    }

    .stDataFrame, .stTable {
        border-radius: 16px;
        overflow: hidden;
    }

    .stAlert {
        border-radius: 14px !important;
    }

    .stCodeBlock {
        border-radius: 16px !important;
    }

    hr {
        border-color: var(--border);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================================================
# CONSTANTS
# =========================================================

DEFAULT_CATEGORIES = [
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
    "Other",
    "Income",
]

# =========================================================
# HELPERS
# =========================================================

def safe_df(df):
    return df.copy() if isinstance(df, pd.DataFrame) else pd.DataFrame()

def to_csv_bytes(df):
    if df is None or df.empty:
        return None
    return df.to_csv(index=False).encode("utf-8")

def render_header():
    st.markdown('<div class="main-title">Cognitive RPA for Personal Finance Data Consolidation</div>', unsafe_allow_html=True)
    st.markdown('<div class="accent-line"></div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">Upload CSVs, PDFs, and text. Consolidate transactions, validate data quality, remove duplicates, categorize merchants, detect recurring spending, flag anomalies, track budgets, and compute a confidence-aware financial health score.</div>',
        unsafe_allow_html=True,
    )

def render_health_card(score, band, confidence_label, confidence_pct):
    st.markdown(
        f"""
        <div class="hero-card">
            <div class="hero-label">Financial Health Score</div>
            <div class="hero-score">{score}/100</div>
            <div class="hero-band">{band}</div>
            <div class="hero-conf">Confidence: <b>{confidence_label}</b> ({confidence_pct}%)</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def prepare_tx_display(df):
    df = safe_df(df)
    if df.empty:
        return df
    cols = [c for c in [
        "date",
        "description",
        "normalized_merchant",
        "category",
        "amount",
        "direction",
        "source",
        "transaction_id",
        "is_recurring",
        "is_anomaly",
        "anomaly_reason",
        "parse_confidence",
    ] if c in df.columns]
    return df[cols].copy()

def prepare_rejected_display(df):
    df = safe_df(df)
    if df.empty:
        return df
    cols = [c for c in ["source", "raw_text", "validation_reason", "parse_confidence"] if c in df.columns]
    return df[cols].copy()

def prepare_duplicate_display(df):
    df = safe_df(df)
    if df.empty:
        return df
    cols = [c for c in ["date", "description", "amount", "source", "duplicate_reason", "transaction_id"] if c in df.columns]
    return df[cols].copy()

def build_budget_editor():
    st.markdown('<div class="box-title">Budget Setup</div>', unsafe_allow_html=True)
    st.markdown('<div class="small-note">Add and edit budgets using structured controls instead of a raw text box.</div>', unsafe_allow_html=True)

    if "budget_rows" not in st.session_state:
        st.session_state["budget_rows"] = [
            {"category": "Groceries", "budget": 5000.0},
            {"category": "Food & Dining", "budget": 3000.0},
            {"category": "Transport", "budget": 2000.0},
        ]

    col1, col2 = st.columns([1, 1])

    with col1:
        new_cat = st.selectbox("Category", DEFAULT_CATEGORIES, key="new_budget_category")
    with col2:
        new_budget = st.number_input("Budget Amount", min_value=0.0, step=500.0, value=1000.0, key="new_budget_amount")

    add_col, clear_col = st.columns([1, 1])

    with add_col:
        if st.button("Add / Update Budget", use_container_width=True):
            updated = False
            for row in st.session_state["budget_rows"]:
                if row["category"] == new_cat:
                    row["budget"] = float(new_budget)
                    updated = True
                    break
            if not updated:
                st.session_state["budget_rows"].append({
                    "category": new_cat,
                    "budget": float(new_budget)
                })

    with clear_col:
        if st.button("Clear Budgets", use_container_width=True):
            st.session_state["budget_rows"] = []

    budget_df = pd.DataFrame(st.session_state["budget_rows"])

    if not budget_df.empty:
        edited_budget_df = st.data_editor(
            budget_df,
            use_container_width=True,
            num_rows="dynamic",
            column_config={
                "category": st.column_config.SelectboxColumn(
                    "Category",
                    options=DEFAULT_CATEGORIES,
                    required=True,
                ),
                "budget": st.column_config.NumberColumn(
                    "Budget",
                    min_value=0.0,
                    step=100.0,
                    format="%.2f",
                    required=True,
                ),
            },
            key="budget_editor"
        )
        edited_budget_df = edited_budget_df.dropna(subset=["category"])
        budget_map = {
            str(row["category"]): float(row["budget"])
            for _, row in edited_budget_df.iterrows()
            if pd.notna(row["budget"])
        }
        st.session_state["budget_rows"] = edited_budget_df.to_dict("records")
    else:
        edited_budget_df = pd.DataFrame(columns=["category", "budget"])
        budget_map = {}

    return edited_budget_df, budget_map

def get_memory_merchant_suggestions(transactions, memory):
    suggestions = set()

    tx = safe_df(transactions)
    if not tx.empty:
        for col in ["normalized_merchant", "merchant", "description"]:
            if col in tx.columns:
                vals = tx[col].dropna().astype(str).str.strip().tolist()
                suggestions.update([v for v in vals if v])

    memory_alias = memory.get("merchant_normalization_map", {})
    memory_cat = memory.get("merchant_category_map", {})
    suggestions.update(memory_alias.keys())
    suggestions.update(memory_alias.values())
    suggestions.update(memory_cat.keys())

    return sorted([s for s in suggestions if str(s).strip()])

def render_memory_panel(transactions):
    memory = load_learning_memory()
    merchant_suggestions = get_memory_merchant_suggestions(transactions, memory)

    st.markdown('<div class="box-title">Learning System</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="small-note">Store merchant aliases and category corrections once, then reuse them automatically in future runs.</div>',
        unsafe_allow_html=True,
    )

    sub1, sub2 = st.columns(2)

    with sub1:
        st.markdown('<div class="soft-card">', unsafe_allow_html=True)
        st.markdown("**Merchant Alias Memory**")
        if merchant_suggestions:
            alias_raw = st.selectbox(
                "Select existing merchant / text",
                options=[""] + merchant_suggestions,
                key="alias_select_existing"
            )
        else:
            alias_raw = ""

        alias_raw_manual = st.text_input(
            "Or type raw merchant text",
            value=alias_raw,
            key="alias_raw_manual"
        )
        alias_norm = st.text_input("Normalized merchant name", key="alias_norm_manual")

        if st.button("Save Merchant Alias", use_container_width=True):
            if alias_raw_manual.strip() and alias_norm.strip():
                remember_merchant_alias(alias_raw_manual.strip(), alias_norm.strip())
                st.success("Merchant alias saved.")
            else:
                st.warning("Enter both raw merchant text and normalized merchant name.")
        st.markdown("</div>", unsafe_allow_html=True)

    with sub2:
        st.markdown('<div class="soft-card">', unsafe_allow_html=True)
        st.markdown("**Merchant Category Memory**")

        if merchant_suggestions:
            cat_merchant = st.selectbox(
                "Select merchant",
                options=[""] + merchant_suggestions,
                key="cat_merchant_select_existing"
            )
        else:
            cat_merchant = ""

        cat_merchant_manual = st.text_input(
            "Or type merchant name",
            value=cat_merchant,
            key="cat_merchant_manual"
        )

        selected_category = st.selectbox(
            "Category",
            DEFAULT_CATEGORIES,
            key="category_dropdown_memory"
        )

        if st.button("Save Merchant Category", use_container_width=True):
            if cat_merchant_manual.strip():
                remember_category(cat_merchant_manual.strip(), selected_category)
                st.success("Merchant category saved.")
            else:
                st.warning("Select or type a merchant.")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("### Current Memory")
    left, right = st.columns(2)

    with left:
        merchant_category_map = memory.get("merchant_category_map", {})
        if merchant_category_map:
            mem_df = pd.DataFrame({
                "merchant": list(merchant_category_map.keys()),
                "category": list(merchant_category_map.values()),
            })
            st.dataframe(mem_df, use_container_width=True, height=260)
        else:
            st.info("No merchant-category memory saved yet.")

    with right:
        merchant_normalization_map = memory.get("merchant_normalization_map", {})
        if merchant_normalization_map:
            alias_df = pd.DataFrame({
                "raw_merchant": list(merchant_normalization_map.keys()),
                "normalized_merchant": list(merchant_normalization_map.values()),
            })
            st.dataframe(alias_df, use_container_width=True, height=260)
        else:
            st.info("No merchant alias memory saved yet.")

    with st.expander("Memory JSON Preview"):
        st.code(json.dumps(memory, indent=2), language="json")

# =========================================================
# HEADER
# =========================================================

render_header()

# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:
    st.markdown("## Input Sources")

    csv_files = st.file_uploader(
        "Upload CSV / Excel files",
        type=["csv", "xlsx"],
        accept_multiple_files=True,
        key="csv_uploader"
    )

    pdf_files = st.file_uploader(
        "Upload PDF statements",
        type=["pdf"],
        accept_multiple_files=True,
        key="pdf_uploader"
    )

    text_input = st.text_area(
        "Paste SMS / Email / Raw transaction text",
        height=180,
        placeholder="Example:\n02/04/2026 UPI paid to Swiggy 340\n03/04/2026 Salary credited 45000",
        key="text_input_area"
    )

    st.markdown("---")
    st.markdown("### Run")
    run_btn = st.button("Run Analysis", type="primary", use_container_width=True)

# =========================================================
# BUDGET BUILDER
# =========================================================

with st.container():
    st.markdown('<div class="section-card-strong">', unsafe_allow_html=True)
    budget_editor_df, budget_map = build_budget_editor()
    st.markdown("</div>", unsafe_allow_html=True)

# =========================================================
# PROCESS
# =========================================================

result = st.session_state.get("result")

if run_btn:
    with st.spinner("Processing transactions..."):
        result = process_financial_data(
            csv_files=csv_files,
            pdf_files=pdf_files,
            text_inputs=[text_input] if text_input and text_input.strip() else [],
            budget_map=budget_map,
        )
        st.session_state["result"] = result

if result is None:
    st.info("Upload at least one source and click **Run Analysis**.")
    st.stop()

transactions = safe_df(result.get("transactions"))
budget_df = safe_df(result.get("budget_tracking"))
rejected_df = safe_df(result.get("rejected_rows"))
duplicate_df = safe_df(result.get("duplicate_rows"))
health = result.get("health", {})
insights = result.get("insights", [])
quality = result.get("data_quality_summary", {})

score = health.get("health_score", 0)
band = health.get("score_band", "Unknown")
confidence_label = health.get("confidence_label", "Low")
confidence_pct = health.get("quality_confidence", 0)

# =========================================================
# TOP DASHBOARD
# =========================================================

top1, top2 = st.columns([1.12, 2])

with top1:
    render_health_card(score, band, confidence_label, confidence_pct)

with top2:
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Valid Transactions", len(transactions))
    with m2:
        st.metric("Rejected Rows", int(quality.get("rejected_rows", len(rejected_df))))
    with m3:
        st.metric("Duplicates Removed", int(quality.get("duplicates_removed", len(duplicate_df))))
    with m4:
        st.metric("Acceptance Rate", f"{quality.get('acceptance_rate_pct', 0)}%")

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="box-title">Top Insights</div>', unsafe_allow_html=True)
    if insights:
        for insight in insights[:6]:
            st.markdown(f"- {insight}")
    else:
        st.write("No insights generated yet.")
    st.markdown("</div>", unsafe_allow_html=True)

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
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Overview Dashboard")

    if transactions.empty:
        st.warning("No valid transactions available after validation.")
    else:
        tx = transactions.copy()
        if "date" in tx.columns:
            tx["date"] = pd.to_datetime(tx["date"], errors="coerce")

        c1, c2 = st.columns(2)

        with c1:
            if {"direction", "amount"}.issubset(tx.columns):
                dir_summary = tx.groupby("direction", dropna=False)["amount"].sum().reset_index()
                if not dir_summary.empty:
                    fig = px.pie(
                        dir_summary,
                        names="direction",
                        values="amount",
                        title="Income vs Expense Distribution",
                    )
                    fig.update_layout(
                        paper_bgcolor="#111111",
                        plot_bgcolor="#111111",
                        font_color="white",
                    )
                    st.plotly_chart(fig, use_container_width=True)

        with c2:
            if {"category", "amount"}.issubset(tx.columns):
                cat_summary = (
                    tx.groupby("category", dropna=False)["amount"]
                    .sum()
                    .reset_index()
                    .sort_values("amount", ascending=False)
                )
                if not cat_summary.empty:
                    fig = px.bar(
                        cat_summary.head(10),
                        x="category",
                        y="amount",
                        title="Top Spending Categories",
                    )
                    fig.update_layout(
                        paper_bgcolor="#111111",
                        plot_bgcolor="#111111",
                        font_color="white",
                    )
                    st.plotly_chart(fig, use_container_width=True)

        c3, c4 = st.columns(2)

        with c3:
            if {"normalized_merchant", "amount"}.issubset(tx.columns):
                merch_summary = (
                    tx.groupby("normalized_merchant", dropna=False)["amount"]
                    .sum()
                    .reset_index()
                    .sort_values("amount", ascending=False)
                )
                if not merch_summary.empty:
                    fig = px.bar(
                        merch_summary.head(10),
                        x="normalized_merchant",
                        y="amount",
                        title="Top Merchants by Amount",
                    )
                    fig.update_layout(
                        paper_bgcolor="#111111",
                        plot_bgcolor="#111111",
                        font_color="white",
                    )
                    st.plotly_chart(fig, use_container_width=True)

        with c4:
            if {"date", "amount"}.issubset(tx.columns):
                trend = (
                    tx.groupby("date", dropna=False)["amount"]
                    .sum()
                    .reset_index()
                    .sort_values("date")
                )
                if not trend.empty:
                    fig = px.line(
                        trend,
                        x="date",
                        y="amount",
                        title="Transaction Amount Trend",
                    )
                    fig.update_layout(
                        paper_bgcolor="#111111",
                        plot_bgcolor="#111111",
                        font_color="white",
                    )
                    st.plotly_chart(fig, use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

# =========================================================
# TAB 2 - TRANSACTIONS
# =========================================================

with tab2:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Transaction Explorer")

    if transactions.empty:
        st.warning("No valid transactions found.")
    else:
        tx = transactions.copy()

        f1, f2, f3 = st.columns(3)

        with f1:
            source_options = sorted(tx["source"].dropna().astype(str).unique().tolist()) if "source" in tx.columns else []
            source_filter = st.multiselect("Filter by source", source_options, default=source_options)

        with f2:
            direction_options = sorted(tx["direction"].dropna().astype(str).unique().tolist()) if "direction" in tx.columns else []
            direction_filter = st.multiselect("Filter by direction", direction_options, default=direction_options)

        with f3:
            category_options = sorted(tx["category"].dropna().astype(str).unique().tolist()) if "category" in tx.columns else []
            category_filter = st.multiselect("Filter by category", category_options, default=[])

        search_text = st.text_input("Search description / merchant")

        if source_filter and "source" in tx.columns:
            tx = tx[tx["source"].astype(str).isin(source_filter)]

        if direction_filter and "direction" in tx.columns:
            tx = tx[tx["direction"].astype(str).isin(direction_filter)]

        if category_filter and "category" in tx.columns:
            tx = tx[tx["category"].astype(str).isin(category_filter)]

        if search_text:
            s = search_text.lower()
            tx = tx[
                tx["description"].astype(str).str.lower().str.contains(s, na=False) |
                tx["normalized_merchant"].astype(str).str.lower().str.contains(s, na=False)
            ]

        st.dataframe(prepare_tx_display(tx), use_container_width=True, height=480)

        tx_csv = to_csv_bytes(tx)
        if tx_csv:
            st.download_button(
                "Download Filtered Transactions CSV",
                data=tx_csv,
                file_name="filtered_transactions.csv",
                mime="text/csv"
            )

    st.markdown("</div>", unsafe_allow_html=True)

# =========================================================
# TAB 3 - ANALYTICS
# =========================================================

with tab3:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Analytical View")

    if transactions.empty:
        st.warning("No valid transactions available for analysis.")
    else:
        tx = transactions.copy()

        left, right = st.columns(2)

        with left:
            st.markdown("#### Recurring Transactions")
            recurring_df = tx[tx["is_recurring"] == True] if "is_recurring" in tx.columns else pd.DataFrame()

            if recurring_df.empty:
                st.info("No recurring transaction patterns detected.")
            else:
                rec_summary = (
                    recurring_df
                    .groupby(["normalized_merchant", "category"], dropna=False)
                    .agg(
                        count=("amount", "size"),
                        average_amount=("amount", "mean"),
                        total_amount=("amount", "sum"),
                    )
                    .reset_index()
                    .sort_values(["count", "total_amount"], ascending=[False, False])
                )
                rec_summary["average_amount"] = rec_summary["average_amount"].round(2)
                rec_summary["total_amount"] = rec_summary["total_amount"].round(2)
                st.dataframe(rec_summary, use_container_width=True, height=280)

        with right:
            st.markdown("#### Anomaly Detection")
            anomaly_df = tx[tx["is_anomaly"] == True] if "is_anomaly" in tx.columns else pd.DataFrame()

            if anomaly_df.empty:
                st.info("No anomalies detected.")
            else:
                anomaly_cols = [c for c in [
                    "date",
                    "description",
                    "normalized_merchant",
                    "category",
                    "amount",
                    "anomaly_reason",
                ] if c in anomaly_df.columns]
                st.dataframe(anomaly_df[anomaly_cols], use_container_width=True, height=280)

        low1, low2 = st.columns(2)

        with low1:
            st.markdown("#### Accepted Transactions by Source")
            if "source" in tx.columns:
                source_summary = tx["source"].value_counts(dropna=False).reset_index()
                source_summary.columns = ["source", "count"]
                fig = px.bar(source_summary, x="source", y="count", title="Source Mix")
                fig.update_layout(
                    paper_bgcolor="#111111",
                    plot_bgcolor="#111111",
                    font_color="white",
                )
                st.plotly_chart(fig, use_container_width=True)

        with low2:
            st.markdown("#### Category Distribution")
            if {"category", "amount"}.issubset(tx.columns):
                cat_summary = (
                    tx.groupby("category", dropna=False)["amount"]
                    .sum()
                    .reset_index()
                    .sort_values("amount", ascending=False)
                )
                fig = px.treemap(
                    cat_summary,
                    path=["category"],
                    values="amount",
                    title="Category Distribution"
                )
                fig.update_layout(
                    paper_bgcolor="#111111",
                    plot_bgcolor="#111111",
                    font_color="white",
                )
                st.plotly_chart(fig, use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

# =========================================================
# TAB 4 - BUDGET & HEALTH
# =========================================================

with tab4:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Budget and Financial Health")

    h1, h2, h3, h4 = st.columns(4)
    with h1:
        st.metric("Health Score", f"{health.get('health_score', 0)}/100")
    with h2:
        st.metric("Score Band", health.get("score_band", "Unknown"))
    with h3:
        st.metric("Confidence", health.get("confidence_label", "Low"))
    with h4:
        st.metric("Confidence %", f"{health.get('quality_confidence', 0)}%")

    st.markdown("### Health Score Components")
    components = health.get("components", {})
    if components:
        comp_df = pd.DataFrame({
            "Component": list(components.keys()),
            "Value": list(components.values())
        })
        st.dataframe(comp_df, use_container_width=True, height=320)
    else:
        st.info("No health score components available.")

    st.markdown("### Budget Tracking")
    if budget_df.empty:
        st.info("No budget output available.")
    else:
        st.dataframe(budget_df, use_container_width=True, height=300)

        if {"category", "spent"}.issubset(budget_df.columns):
            fig = px.bar(
                budget_df.sort_values("spent", ascending=False),
                x="category",
                y="spent",
                title="Budget Spend by Category"
            )
            fig.update_layout(
                paper_bgcolor="#111111",
                plot_bgcolor="#111111",
                font_color="white",
            )
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

# =========================================================
# TAB 5 - QUALITY & AUDIT TRAIL
# =========================================================

with tab5:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Data Quality and Audit Trail")

    q1, q2, q3, q4 = st.columns(4)
    with q1:
        st.metric("Parsed Rows", quality.get("total_parsed_rows", 0))
    with q2:
        st.metric("Accepted", quality.get("valid_rows_accepted", 0))
    with q3:
        st.metric("Rejected", quality.get("rejected_rows", 0))
    with q4:
        st.metric("Avg Confidence", f"{quality.get('average_valid_confidence_pct', 0)}%")

    st.markdown("### Quality Summary")
    st.json(quality)

    sub_a, sub_b = st.tabs(["Rejected Rows", "Duplicate Rows"])

    with sub_a:
        if rejected_df.empty:
            st.success("No rejected rows.")
        else:
            st.dataframe(prepare_rejected_display(rejected_df), use_container_width=True, height=360)
            rejected_csv = to_csv_bytes(rejected_df)
            if rejected_csv:
                st.download_button(
                    "Download Rejected Rows CSV",
                    data=rejected_csv,
                    file_name="rejected_rows.csv",
                    mime="text/csv"
                )

    with sub_b:
        if duplicate_df.empty:
            st.success("No duplicate rows detected.")
        else:
            st.dataframe(prepare_duplicate_display(duplicate_df), use_container_width=True, height=360)
            duplicate_csv = to_csv_bytes(duplicate_df)
            if duplicate_csv:
                st.download_button(
                    "Download Duplicate Rows CSV",
                    data=duplicate_csv,
                    file_name="duplicate_rows.csv",
                    mime="text/csv"
                )

    st.markdown("</div>", unsafe_allow_html=True)

# =========================================================
# TAB 6 - LEARNING MEMORY
# =========================================================

with tab6:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    render_memory_panel(transactions)
    st.markdown("</div>", unsafe_allow_html=True)
