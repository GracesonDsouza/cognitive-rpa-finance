import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from processor import (
    MEMORY_FILE,
    enrich_transactions,
    extract_from_pdf,
    extract_from_text,
    get_uncertain_transactions,
    load_csv,
    load_memory,
    save_memory_entry,
)

st.set_page_config(
    page_title="Cognitive RPA for Personal Finance Data Consolidation",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
.block-container {
    padding-top: 1.2rem;
    padding-bottom: 1rem;
}
.main-title {
    font-size: 2.2rem;
    font-weight: 800;
    color: #0f172a;
    margin-bottom: 0.2rem;
}
.sub-title {
    font-size: 1rem;
    color: #475569;
    margin-bottom: 1rem;
}
.metric-box {
    background: linear-gradient(135deg, #ffffff 0%, #f8fbff 100%);
    border: 1px solid #dbeafe;
    padding: 18px;
    border-radius: 18px;
    box-shadow: 0 6px 18px rgba(15, 23, 42, 0.05);
}
.metric-label {
    font-size: 0.95rem;
    color: #475569;
}
.metric-value {
    font-size: 1.9rem;
    font-weight: 800;
    color: #0f172a;
    margin-top: 6px;
}
.section-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    padding: 18px;
    border-radius: 18px;
    box-shadow: 0 6px 18px rgba(15, 23, 42, 0.04);
    margin-bottom: 14px;
}
.hero-card {
    background: linear-gradient(135deg, #e0f2fe 0%, #eef2ff 50%, #f8fafc 100%);
    border: 1px solid #cbd5e1;
    padding: 22px;
    border-radius: 22px;
    box-shadow: 0 10px 26px rgba(15, 23, 42, 0.07);
    margin-bottom: 14px;
}
.badge {
    display: inline-block;
    background: #0ea5e9;
    color: white;
    padding: 5px 11px;
    border-radius: 999px;
    font-size: 0.8rem;
    margin-right: 8px;
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
    '<div class="sub-title">A smarter personal finance assistant that extracts, learns, detects duplicates, flags anomalies, and tracks financial health.</div>',
    unsafe_allow_html=True
)

st.markdown("""
<div class="hero-card">
    <span class="badge">Cognitive RPA</span>
    <span class="badge">ML Anomaly Detection</span>
    <span class="badge">Learning System</span>
    <span class="badge">Budget Tracking</span>
    <span class="badge">Duplicate Prevention</span>
    <div style="margin-top:12px; color:#0f172a; font-size:1rem;">
        This prototype reads messy finance data from <b>CSV</b>, <b>SMS/email text</b>, and <b>PDF bank statements</b>,
        then converts it into actionable financial understanding.
    </div>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.header("Input")
    mode = st.radio("Choose source", ["CSV Transactions", "Text (SMS/Email)", "PDF Statement"])

    st.markdown("---")
    st.subheader("Budget Setup")
    default_budgets = {
        "Food": 10000.0,
        "Travel": 8000.0,
        "Rent": 20000.0,
        "EMI/Loan": 12000.0,
        "Subscriptions": 2000.0,
        "Shopping": 10000.0,
        "Bills/Utilities": 6000.0,
        "Investment": 10000.0,
        "Other": 5000.0,
    }

    budgets = {}
    for cat, default_val in default_budgets.items():
        budgets[cat] = st.number_input(f"{cat} Budget", min_value=0.0, value=float(default_val), step=500.0)

    st.markdown("---")
    consented = st.checkbox(
        "I consent to process this data for the academic prototype.",
        value=True
    )

if not consented:
    st.warning("Please provide consent to continue.")
    st.stop()

raw_df = pd.DataFrame()

if mode == "CSV Transactions":
    uploaded_csv = st.file_uploader("Upload CSV file", type=["csv"])
    if uploaded_csv is not None:
        raw_df = load_csv(uploaded_csv)

elif mode == "Text (SMS/Email)":
    sample_text = """01/01/2024 Salary credited INR 55000 txn id SALJAN24
03/01/2024 Rent paid INR 18000 txn id RENTJAN24
05/01/2024 Netflix subscription debited INR 649 txn id NFXJAN24
08/01/2024 Swiggy order paid INR 420 txn id SWG080124
15/01/2024 Amazon purchase debited INR 3499 txn id AMZ150124
20/01/2024 Electricity bill paid INR 2200 txn id ELE200124
01/02/2024 Salary credited INR 55000 txn id SALFEB24
03/02/2024 Rent paid INR 18000 txn id RENTFEB24
05/02/2024 Netflix subscription debited INR 649 txn id NFXFEB24
14/03/2024 Flight ticket purchase debited INR 18450 txn id FLT140324
25/03/2024 Investment SIP paid INR 5000 txn id SIP250324"""
    text_input = st.text_area("Paste SMS / email-style transaction text", value=sample_text, height=250)
    if text_input.strip():
        raw_df = extract_from_text(text_input)

else:
    uploaded_pdf = st.file_uploader("Upload PDF statement", type=["pdf"])
    if uploaded_pdf is not None:
        raw_df = extract_from_pdf(uploaded_pdf)

if raw_df.empty:
    st.info("Upload or paste data to see the dashboard.")
    st.stop()

before_count = len(raw_df)
df = enrich_transactions(raw_df, MEMORY_FILE)
after_count = len(df)
duplicates_removed = before_count - after_count
df.attrs["duplicates_removed"] = duplicates_removed

memory_df = load_memory(MEMORY_FILE)
uncertain_df = get_uncertain_transactions(df, memory_df)

if not uncertain_df.empty:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Learning System: Ask Once")
    st.caption("For uncertain merchants, the system asks once. After you save it, the app remembers for future transactions.")

    for _, row in uncertain_df.iterrows():
        merchant = row["merchant_norm"]
        st.markdown(f"**Merchant:** `{merchant}` | Current category guess: `{row['category']}`")

        col1, col2, col3 = st.columns([2, 2, 1])
        with col1:
            selected_category = st.selectbox(
                f"Select category for {merchant}",
                options=CATEGORIES,
                index=CATEGORIES.index("Other") if "Other" in CATEGORIES else 0,
                key=f"cat_{merchant}"
            )
        with col2:
            selected_type = st.selectbox(
                f"Select type for {merchant}",
                options=["expense", "income"],
                index=0 if row["type"] == "expense" else 1,
                key=f"type_{merchant}"
            )
        with col3:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button(f"Save {merchant}", key=f"save_{merchant}"):
                save_memory_entry(merchant, selected_category, selected_type, MEMORY_FILE)
                st.success(f"Saved learning for {merchant}. Reload the app or re-run input to apply.")
    st.markdown('</div>', unsafe_allow_html=True)

insights = build_insights(df)
score, score_label = calculate_financial_health_score(df, budgets)
recurring_count = int(df["is_recurring"].sum()) if "is_recurring" in df.columns else 0
anomaly_count = int(df["is_anomaly"].sum()) if "is_anomaly" in df.columns else 0
net_cashflow = insights["income_total"] - insights["expense_total"]

m1, m2, m3, m4, m5, m6 = st.columns(6)
with m1:
    metric_card("Total Income", format_inr(insights["income_total"]))
with m2:
    metric_card("Total Expense", format_inr(insights["expense_total"]))
with m3:
    metric_card("Net Cashflow", format_inr(net_cashflow))
with m4:
    metric_card("Recurring", str(recurring_count))
with m5:
    metric_card("Anomalies", str(anomaly_count))
with m6:
    metric_card("Duplicates Removed", str(duplicates_removed))

st.markdown("")

left, right = st.columns([1, 1])
with left:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.plotly_chart(make_gauge(score, score_label), use_container_width=True)
    st.markdown(
        f"<div class='small-note'>This score is based on savings rate, recurring burden, anomaly frequency, and budget discipline.</div>",
        unsafe_allow_html=True
    )
    st.markdown('</div>', unsafe_allow_html=True)

with right:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Key Insights")
    st.markdown(f"- **Savings rate:** {insights['savings_rate']:.1f}%")
    st.markdown(f"- **Recurring burden:** {insights['recurring_burden']:.1f}% of expense")
    st.markdown(f"- **Biggest spend category:** {insights['biggest_category']}")
    st.markdown(f"- **Highest spend month:** {insights['highest_spend_month']}")
    if insights["top_merchants"]:
        merchant_text = ", ".join([f"{m} ({format_inr(v)})" for m, v in insights["top_merchants"]])
        st.markdown(f"- **Top merchants:** {merchant_text}")
    else:
        st.markdown("- **Top merchants:** N/A")
    st.markdown('</div>', unsafe_allow_html=True)

expense_df = df[df["type"] == "expense"].copy()
monthly_cashflow = build_monthly_cashflow(df)

c1, c2 = st.columns(2)
with c1:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Spend by Category")
    if not expense_df.empty:
        spend_by_category = expense_df.groupby("category", as_index=False)["amount"].sum().sort_values("amount", ascending=False)
        fig_cat = px.pie(
            spend_by_category,
            names="category",
            values="amount",
            hole=0.55
        )
        fig_cat.update_layout(height=420, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig_cat, use_container_width=True)
    else:
        st.info("No expense transactions available.")
    st.markdown('</div>', unsafe_allow_html=True)

with c2:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Monthly Cashflow")
    fig_month = px.bar(monthly_cashflow, x="month_label", y="net_cashflow")
    fig_month.update_layout(height=420, margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig_month, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

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
    alerts.append(f"Duplicate prevention removed {duplicates_removed} transaction(s) to avoid double counting.")

budget_df_check = None
if not expense_df.empty:
    actual_by_cat = expense_df.groupby("category")["amount"].sum().to_dict()
    over_budget = []
    for cat, budget in budgets.items():
        actual = actual_by_cat.get(cat, 0.0)
        if budget > 0 and actual > budget:
            over_budget.append(f"{cat}: spent {format_inr(actual)} vs budget {format_inr(budget)}")
    if over_budget:
        alerts.append("Over budget categories: " + "; ".join(over_budget))

if alerts:
    for alert in alerts:
        st.warning(alert)
else:
    st.success("No major alerts detected.")
st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.subheader("Unusual Spending Detected by ML")
anomaly_view = df[df["is_anomaly"]].copy().sort_values("amount", ascending=False)
if anomaly_view.empty:
    st.info("No unusual spending patterns were flagged.")
else:
    anomaly_view["date"] = pd.to_datetime(anomaly_view["date"], errors="coerce").dt.date
    st.dataframe(
        anomaly_view[["date", "merchant_norm", "category", "amount", "anomaly_reason", "description"]],
        use_container_width=True
    )
st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.subheader("Recurring Transactions")
recurring_view = df[df["is_recurring"]].copy().sort_values(["merchant_norm", "date"])
if recurring_view.empty:
    st.info("No recurring transactions detected.")
else:
    recurring_view["date"] = pd.to_datetime(recurring_view["date"], errors="coerce").dt.date
    st.dataframe(
        recurring_view[["date", "merchant_norm", "category", "amount", "description"]],
        use_container_width=True
    )
st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.subheader("All Transactions")
all_view = df.copy()
all_view["date"] = pd.to_datetime(all_view["date"], errors="coerce").dt.date
st.dataframe(all_view, use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)

csv_data = df.copy()
csv_data["date"] = pd.to_datetime(csv_data["date"], errors="coerce").dt.strftime("%Y-%m-%d")
st.download_button(
    "Download Processed CSV",
    data=csv_data.to_csv(index=False).encode("utf-8"),
    file_name="processed_finance_transactions.csv",
    mime="text/csv",
)
