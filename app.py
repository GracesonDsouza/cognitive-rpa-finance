import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from processor import (
    MEMORY_FILE,
    combine_sources,
    enrich_transactions,
    extract_from_pdf,
    extract_from_text,
    get_uncertain_transactions,
    load_csv,
    load_memory,
    save_memory_entry,
)

st.set_page_config(page_title="Cognitive RPA Finance", layout="wide")

st.markdown("""
<style>
.block-container {
    padding-top: 1.2rem;
    padding-bottom: 1rem;
    max-width: 1400px;
}
.main-title {
    font-size: 2.2rem;
    font-weight: 800;
    color: #0f172a;
    margin-bottom: 0.15rem;
}
.sub-title {
    font-size: 1rem;
    color: #475569;
    margin-bottom: 1rem;
}
.hero-card {
    background: linear-gradient(135deg, #e0f2fe 0%, #eef2ff 55%, #f8fafc 100%);
    border: 1px solid #dbeafe;
    padding: 20px;
    border-radius: 20px;
    margin-bottom: 16px;
}
.badge {
    display: inline-block;
    background: #0ea5e9;
    color: white;
    padding: 6px 12px;
    border-radius: 999px;
    font-size: 0.78rem;
    margin-right: 8px;
    margin-bottom: 8px;
}
.metric-box {
    background: white;
    border: 1px solid #e2e8f0;
    border-radius: 16px;
    padding: 16px;
    box-shadow: 0 4px 14px rgba(15, 23, 42, 0.04);
}
.metric-label {
    font-size: 0.9rem;
    color: #64748b;
}
.metric-value {
    font-size: 1.7rem;
    font-weight: 800;
    color: #0f172a;
    margin-top: 4px;
}
.section-card {
    background: white;
    border: 1px solid #e2e8f0;
    border-radius: 16px;
    padding: 16px;
    margin-bottom: 14px;
}
.small-note {
    color: #64748b;
    font-size: 0.9rem;
}
</style>
""", unsafe_allow_html=True)


CATEGORIES = [
    "Food", "Travel", "Rent", "EMI/Loan", "Subscriptions",
    "Shopping", "Bills/Utilities", "Investment", "Other", "Income"
]


def format_inr(x):
    return f"₹{x:,.2f}"


def metric_card(title, value):
    st.markdown(
        f"""
        <div class="metric-box">
            <div class="metric-label">{title}</div>
            <div class="metric-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def build_monthly_cashflow(df):
    monthly = (
        df.assign(month_label=pd.to_datetime(df["date"], errors="coerce").dt.to_period("M").astype(str))
        .groupby(["month_label", "type"], dropna=False)["amount"]
        .sum()
        .unstack(fill_value=0)
        .reset_index()
    )
    for col in ["income", "expense"]:
        if col not in monthly.columns:
            monthly[col] = 0.0
    monthly["net_cashflow"] = monthly["income"] - monthly["expense"]
    return monthly


def build_insights(df):
    income_total = df.loc[df["type"] == "income", "amount"].sum()
    expense_total = df.loc[df["type"] == "expense", "amount"].sum()
    savings_rate = ((income_total - expense_total) / income_total * 100) if income_total > 0 else 0

    recurring_expense = df.loc[(df["type"] == "expense") & (df["is_recurring"]), "amount"].sum()
    recurring_burden = (recurring_expense / expense_total * 100) if expense_total > 0 else 0

    anomaly_count = int(df["is_anomaly"].sum()) if "is_anomaly" in df.columns else 0
    duplicate_count = int(df.attrs.get("duplicates_removed", 0))

    expense_df = df[df["type"] == "expense"].copy()
    biggest_category = "N/A"
    highest_spend_month = "N/A"
    top_merchants = []

    if not expense_df.empty:
        cat_spend = expense_df.groupby("category")["amount"].sum().sort_values(ascending=False)
        biggest_category = cat_spend.index[0] if not cat_spend.empty else "N/A"

        monthly = expense_df.groupby("month")["amount"].sum().sort_values(ascending=False)
        highest_spend_month = monthly.index[0] if not monthly.empty else "N/A"

        merchant_spend = expense_df.groupby("merchant_norm")["amount"].sum().sort_values(ascending=False)
        top_merchants = merchant_spend.head(5).items()

    return {
        "income_total": income_total,
        "expense_total": expense_total,
        "savings_rate": savings_rate,
        "recurring_burden": recurring_burden,
        "biggest_category": biggest_category,
        "highest_spend_month": highest_spend_month,
        "top_merchants": top_merchants,
        "anomaly_count": anomaly_count,
        "duplicate_count": duplicate_count,
    }


def calculate_financial_health_score(df, budgets):
    if df.empty:
        return 0, "No Data"

    income_total = df.loc[df["type"] == "income", "amount"].sum()
    expense_total = df.loc[df["type"] == "expense", "amount"].sum()
    savings_rate = ((income_total - expense_total) / income_total * 100) if income_total > 0 else 0

    recurring_burden = 0
    if expense_total > 0:
        recurring_burden = (
            df.loc[(df["type"] == "expense") & (df["is_recurring"]), "amount"].sum() / expense_total
        ) * 100

    anomaly_count = int(df["is_anomaly"].sum()) if "is_anomaly" in df.columns else 0
    months = max(df["month"].nunique(), 1)
    anomaly_rate = anomaly_count / months

    budget_penalty = 0
    expense_df = df[df["type"] == "expense"]
    if not expense_df.empty:
        actual_by_cat = expense_df.groupby("category")["amount"].sum().to_dict()
        for cat, budget in budgets.items():
            if budget > 0:
                actual = actual_by_cat.get(cat, 0)
                if actual > budget:
                    overspend_pct = ((actual - budget) / budget) * 100
                    budget_penalty += min(overspend_pct * 0.15, 8)

    score = 100
    score -= max(0, 25 - savings_rate) * 0.8
    score -= max(0, recurring_burden - 35) * 0.5
    score -= anomaly_rate * 8
    score -= budget_penalty

    score = max(0, min(100, round(score)))

    if score >= 80:
        label = "Excellent"
    elif score >= 65:
        label = "Good"
    elif score >= 50:
        label = "Average"
    else:
        label = "Needs Attention"

    return score, label


def get_score_color(score):
    if score >= 80:
        return "#16a34a"
    elif score >= 65:
        return "#0284c7"
    elif score >= 50:
        return "#f59e0b"
    return "#dc2626"


def make_gauge(score, label):
    color = get_score_color(score)
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        number={"suffix": "/100"},
        title={"text": f"Financial Health Score<br><span style='font-size:14px'>{label}</span>"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": color},
            "steps": [
                {"range": [0, 50], "color": "#fee2e2"},
                {"range": [50, 65], "color": "#fef3c7"},
                {"range": [65, 80], "color": "#dbeafe"},
                {"range": [80, 100], "color": "#dcfce7"},
            ],
        }
    ))
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
    return fig


st.markdown('<div class="main-title">Cognitive RPA for Personal Finance Data Consolidation</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Unified pipeline for CSV, PDF, and text-based transaction data with learning, duplicate prevention, recurring detection, and anomaly insights.</div>',
    unsafe_allow_html=True
)

st.markdown("""
<div class="hero-card">
    <span class="badge">Cognitive RPA</span>
    <span class="badge">Unified Pipeline</span>
    <span class="badge">ML Anomaly Detection</span>
    <span class="badge">Ask-Once Learning</span>
    <span class="badge">Duplicate Prevention</span>
    <div style="margin-top:10px;color:#0f172a;font-size:1rem;">
        Upload CSV files, upload PDF statements, paste SMS/email text, and the app consolidates everything into one common transaction layer before analysis.
    </div>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("Settings")
    consented = st.checkbox("I consent to process this financial data.", value=True)

    st.subheader("Budget Setup")
    budgets = {
        "Food": st.number_input("Food Budget", 0.0, value=10000.0, step=500.0),
        "Travel": st.number_input("Travel Budget", 0.0, value=8000.0, step=500.0),
        "Rent": st.number_input("Rent Budget", 0.0, value=20000.0, step=500.0),
        "EMI/Loan": st.number_input("EMI/Loan Budget", 0.0, value=12000.0, step=500.0),
        "Subscriptions": st.number_input("Subscriptions Budget", 0.0, value=2000.0, step=500.0),
        "Shopping": st.number_input("Shopping Budget", 0.0, value=10000.0, step=500.0),
        "Bills/Utilities": st.number_input("Bills/Utilities Budget", 0.0, value=6000.0, step=500.0),
        "Investment": st.number_input("Investment Budget", 0.0, value=10000.0, step=500.0),
        "Other": st.number_input("Other Budget", 0.0, value=5000.0, step=500.0),
    }

if not consented:
    st.warning("Please provide consent to continue.")
    st.stop()

# Input section
with st.expander("Upload / Paste Financial Data", expanded=True):
    col1, col2 = st.columns(2)

    with col1:
        csv_files = st.file_uploader("Upload CSV files", type=["csv"], accept_multiple_files=True)
        pdf_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)

    with col2:
        sample_text = """01/01/2024 Salary credited INR 55000 txn id SALJAN24
03/01/2024 Rent paid INR 18000 txn id RENTJAN24
05/01/2024 Netflix subscription debited INR 649 txn id NFXJAN24
08/01/2024 Swiggy order paid INR 420 txn id SWG080124
15/01/2024 Amazon purchase debited INR 3499 txn id AMZ150124
20/01/2024 Electricity bill paid INR 2200 txn id ELE200124"""
        text_input = st.text_area("Paste SMS / email-style text", value=sample_text, height=220)

dfs = []

if csv_files:
    for file in csv_files:
        try:
            dfs.append(load_csv(file))
        except Exception as e:
            st.error(f"CSV error in {file.name}: {e}")

if pdf_files:
    for file in pdf_files:
        try:
            dfs.append(extract_from_pdf(file))
        except Exception as e:
            st.error(f"PDF error in {file.name}: {e}")

if text_input.strip():
    try:
        text_df = extract_from_text(text_input)
        if not text_df.empty:
            dfs.append(text_df)
    except Exception as e:
        st.error(f"Text parsing error: {e}")

combined_raw_df = combine_sources(dfs)

if combined_raw_df.empty:
    st.info("No valid transactions found yet. Upload or paste data.")
    st.stop()

df = enrich_transactions(combined_raw_df, MEMORY_FILE)
duplicates_removed = int(df.attrs.get("duplicates_removed", 0))
memory_df = load_memory(MEMORY_FILE)
uncertain_df = get_uncertain_transactions(df, memory_df)

# KPIs
insights = build_insights(df)
score, score_label = calculate_financial_health_score(df, budgets)

income_total = insights["income_total"]
expense_total = insights["expense_total"]
net_cashflow = income_total - expense_total
anomaly_count = insights["anomaly_count"]
recurring_count = int(df["is_recurring"].sum()) if "is_recurring" in df.columns else 0

m1, m2, m3, m4, m5, m6 = st.columns(6)
with m1:
    metric_card("Income", format_inr(income_total))
with m2:
    metric_card("Expense", format_inr(expense_total))
with m3:
    metric_card("Net Cashflow", format_inr(net_cashflow))
with m4:
    metric_card("Health Score", f"{score}/100")
with m5:
    metric_card("Anomalies", str(anomaly_count))
with m6:
    metric_card("Duplicates Removed", str(duplicates_removed))

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Overview", "Budget", "Anomalies", "Recurring", "Learning", "Raw Data"
])

expense_df = df[df["type"] == "expense"].copy()
monthly_cashflow = build_monthly_cashflow(df)

with tab1:
    c1, c2 = st.columns(2)

    with c1:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("Spend by Category")
        if not expense_df.empty:
            spend_by_category = (
                expense_df.groupby("category", as_index=False)["amount"]
                .sum()
                .sort_values("amount", ascending=False)
            )
            fig_cat = px.pie(
                spend_by_category,
                names="category",
                values="amount",
                hole=0.55
            )
            fig_cat.update_layout(height=400, margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig_cat, use_container_width=True)
        else:
            st.info("No expense transactions available.")
        st.markdown('</div>', unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.plotly_chart(make_gauge(score, score_label), use_container_width=True)
        st.markdown(
            "<div class='small-note'>Score is based on savings rate, recurring burden, anomaly frequency, and budget discipline.</div>",
            unsafe_allow_html=True
        )
        st.markdown('</div>', unsafe_allow_html=True)

    c3, c4 = st.columns(2)

    with c3:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("Monthly Cashflow")
        fig_month = px.bar(monthly_cashflow, x="month_label", y="net_cashflow")
        fig_month.update_layout(height=360, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig_month, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with c4:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("Key Insights")
        st.markdown(f"- **Savings rate:** {insights['savings_rate']:.1f}%")
        st.markdown(f"- **Recurring burden:** {insights['recurring_burden']:.1f}% of expense")
        st.markdown(f"- **Biggest spend category:** {insights['biggest_category']}")
        st.markdown(f"- **Highest spend month:** {insights['highest_spend_month']}")
        st.markdown(f"- **Sources processed:** {', '.join(sorted(df['source'].dropna().unique().tolist()))}")

        if insights["top_merchants"]:
            merchant_text = ", ".join([f"{m} ({format_inr(v)})" for m, v in insights["top_merchants"]])
            st.markdown(f"- **Top merchants:** {merchant_text}")
        else:
            st.markdown("- **Top merchants:** N/A")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Smart Alerts")

    alerts = []

    if insights["recurring_burden"] > 40:
        alerts.append(f"Recurring expenses are high at {insights['recurring_burden']:.1f}% of total spending.")

    negative_months = monthly_cashflow[monthly_cashflow["net_cashflow"] < 0]["month_label"].tolist()
    if negative_months:
        alerts.append("Negative cashflow months detected: " + ", ".join(negative_months))

    if not expense_df.empty:
        large_threshold = expense_df["amount"].quantile(0.95)
        large_expense_count = int((expense_df["amount"] >= large_threshold).sum())
        if large_expense_count > 0:
            alerts.append(f"{large_expense_count} large expense transaction(s) detected above the 95th percentile.")

    if anomaly_count > 0:
        alerts.append(f"ML anomaly layer flagged {anomaly_count} unusual transaction(s).")

    if duplicates_removed > 0:
        alerts.append(f"Duplicate prevention removed {duplicates_removed} transaction(s) to avoid double counting across sources.")

    if alerts:
        for alert in alerts:
            st.warning(alert)
    else:
        st.success("No major alerts detected.")
    st.markdown('</div>', unsafe_allow_html=True)

with tab2:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Budget Tracking")

    if not expense_df.empty:
        actual_by_cat = expense_df.groupby("category")["amount"].sum().to_dict()
        budget_rows = []

        for cat, budget in budgets.items():
            actual = actual_by_cat.get(cat, 0.0)
            variance = actual - budget
            status = "Over Budget" if variance > 0 else "Within Budget"
            budget_rows.append({
                "category": cat,
                "budget": budget,
                "actual": actual,
                "variance": variance,
                "status": status
            })

        budget_df = pd.DataFrame(budget_rows)
        st.dataframe(budget_df, use_container_width=True)

        fig_budget = px.bar(
            budget_df,
            x="category",
            y=["budget", "actual"],
            barmode="group"
        )
        fig_budget.update_layout(height=420, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig_budget, use_container_width=True)
    else:
        st.info("No expense transactions available for budget tracking.")
    st.markdown('</div>', unsafe_allow_html=True)

with tab3:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Unusual Spending Detected by ML")
    anomaly_view = df[df["is_anomaly"]].copy().sort_values("amount", ascending=False)

    if anomaly_view.empty:
        st.info("No anomalies detected.")
    else:
        anomaly_view["date"] = pd.to_datetime(anomaly_view["date"], errors="coerce").dt.date
        st.dataframe(
            anomaly_view[["date", "merchant_norm", "category", "source", "amount", "anomaly_reason", "description"]],
            use_container_width=True
        )
    st.markdown('</div>', unsafe_allow_html=True)

with tab4:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Recurring Transactions")
    recurring_view = df[df["is_recurring"]].copy().sort_values(["merchant_norm", "date"])

    if recurring_view.empty:
        st.info("No recurring transactions detected.")
    else:
        recurring_view["date"] = pd.to_datetime(recurring_view["date"], errors="coerce").dt.date
        st.dataframe(
            recurring_view[["date", "merchant_norm", "category", "source", "amount", "description"]],
            use_container_width=True
        )
    st.markdown('</div>', unsafe_allow_html=True)

with tab5:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Ask-Once Learning")
    st.caption("If the system is unsure, you correct it once. The app remembers it for future transactions.")

    if uncertain_df.empty:
        st.success("No new merchant learning required.")
    else:
        for _, row in uncertain_df.iterrows():
            merchant = row["merchant_norm"]
            st.write(f"Merchant: {merchant} | Current guess: {row['category']}")

            c1, c2, c3 = st.columns([2, 2, 1])
            with c1:
                selected_category = st.selectbox(
                    f"Category for {merchant}",
                    options=CATEGORIES,
                    index=CATEGORIES.index("Other"),
                    key=f"cat_{merchant}"
                )
            with c2:
                selected_type = st.selectbox(
                    f"Type for {merchant}",
                    options=["expense", "income"],
                    index=0 if row["type"] == "expense" else 1,
                    key=f"type_{merchant}"
                )
            with c3:
                st.write("")
                st.write("")
                if st.button(f"Save {merchant}", key=f"save_{merchant}"):
                    save_memory_entry(merchant, selected_category, selected_type, MEMORY_FILE)
                    st.success(f"Saved learning for {merchant}. Re-run the app.")
    st.markdown('</div>', unsafe_allow_html=True)

with tab6:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Unified Raw Data")
    raw_view = df.copy()
    raw_view["date"] = pd.to_datetime(raw_view["date"], errors="coerce").dt.date
    st.dataframe(raw_view, use_container_width=True)

    csv_data = df.copy()
    csv_data["date"] = pd.to_datetime(csv_data["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    st.download_button(
        "Download Processed CSV",
        data=csv_data.to_csv(index=False).encode("utf-8"),
        file_name="processed_finance_transactions.csv",
        mime="text/csv",
    )
    st.markdown('</div>', unsafe_allow_html=True)
