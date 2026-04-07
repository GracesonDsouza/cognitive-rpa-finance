import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from processor import enrich_transactions, extract_from_pdf, extract_from_text, load_csv

st.set_page_config(page_title="Cognitive RPA for Personal Finance Data Consolidation", layout="wide")


def format_inr(x):
    return f"₹{x:,.2f}"


def metric_card(title, value):
    st.markdown(
        f"""
        <div style="padding:14px;border-radius:14px;border:1px solid #d9d9d9;background:#fafafa;">
            <div style="font-size:14px;color:#555;">{title}</div>
            <div style="font-size:28px;font-weight:700;margin-top:4px;">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def profile_card():
    st.markdown(
        """
        <div style="padding:16px;border-radius:16px;background:linear-gradient(135deg,#eff6ff,#f8fafc);border:1px solid #dbeafe;">
            <div style="font-size:26px;font-weight:700;">Cognitive RPA for Personal Finance Data Consolidation</div>
            <div style="margin-top:6px;font-size:15px;">
                A privacy-aware academic prototype that extracts, understands, and consolidates finance data from CSV, SMS/email text, and PDF statements.
            </div>
            <div style="margin-top:10px;font-size:14px;color:#334155;">
                <b>TRL:</b> 4 &nbsp;&nbsp;|&nbsp;&nbsp; <b>Core:</b> Cognitive RPA &nbsp;&nbsp;|&nbsp;&nbsp; <b>ML layer:</b> Spending anomaly detection
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def consent_banner():
    st.info(
        "This academic prototype processes data only after user consent. It does not require banking credentials and is designed for privacy-aware experimentation."
    )
    return st.checkbox("I consent to process the uploaded / pasted financial data for this academic prototype.", value=True)


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
    }


st.title("Cognitive RPA for Personal Finance Data Consolidation")
st.caption("Streamlit prototype with multi-source extraction, recurring detection, smart insights, and ML-based anomaly detection.")
profile_card()
st.markdown("")

with st.sidebar:
    st.header("Input Mode")
    mode = st.radio("Choose source", ["CSV Transactions", "Text (SMS/Email)", "PDF Statement"])
    st.markdown("---")
    st.subheader("About")
    st.write(
        "This prototype combines rule-based Cognitive RPA with a lightweight ML layer to consolidate and interpret scattered personal finance data."
    )

consented = consent_banner()
if not consented:
    st.warning("Please provide consent to continue.")
    st.stop()

raw_df = pd.DataFrame()

if mode == "CSV Transactions":
    uploaded_csv = st.file_uploader("Upload CSV file", type=["csv"])
    if uploaded_csv is not None:
        raw_df = load_csv(uploaded_csv)
elif mode == "Text (SMS/Email)":
    sample_text = """01/01/2024 Salary credited INR 55000 from Employer
03/01/2024 Rent paid INR 18000 to landlord
05/01/2024 Netflix subscription debited INR 649
08/01/2024 Swiggy order paid INR 420
15/01/2024 Amazon purchase debited INR 3499
20/01/2024 Electricity bill paid INR 2200"""
    text_input = st.text_area("Paste SMS / email-style transaction text", value=sample_text, height=220)
    if text_input.strip():
        raw_df = extract_from_text(text_input)
else:
    uploaded_pdf = st.file_uploader("Upload PDF statement", type=["pdf"])
    if uploaded_pdf is not None:
        raw_df = extract_from_pdf(uploaded_pdf)

if raw_df.empty:
    st.warning("Upload or paste data to see the dashboard.")
    st.stop()

df = enrich_transactions(raw_df)
if df.empty:
    st.error("No valid transactions could be extracted. Try a different file or cleaner input format.")
    st.stop()

insights = build_insights(df)
recurring_count = int(df["is_recurring"].sum()) if "is_recurring" in df.columns else 0
anomaly_count = int(df["is_anomaly"].sum()) if "is_anomaly" in df.columns else 0
net_cashflow = insights["income_total"] - insights["expense_total"]

c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    metric_card("Total Income", format_inr(insights["income_total"]))
with c2:
    metric_card("Total Expense", format_inr(insights["expense_total"]))
with c3:
    metric_card("Net Cashflow", format_inr(net_cashflow))
with c4:
    metric_card("Recurring Transactions", str(recurring_count))
with c5:
    metric_card("Anomalies Found", str(anomaly_count))

st.markdown("")

expense_df = df[df["type"] == "expense"].copy()
monthly_cashflow = build_monthly_cashflow(df)

chart_col1, chart_col2 = st.columns(2)
with chart_col1:
    st.subheader("Spend by Category")
    if not expense_df.empty:
        spend_by_category = expense_df.groupby("category", as_index=False)["amount"].sum().sort_values("amount", ascending=False)
        fig_cat = px.pie(spend_by_category, names="category", values="amount", hole=0.45)
        st.plotly_chart(fig_cat, use_container_width=True)
    else:
        st.info("No expense transactions available for category analysis.")

with chart_col2:
    st.subheader("Monthly Cashflow")
    fig_month = px.bar(monthly_cashflow, x="month_label", y="net_cashflow")
    st.plotly_chart(fig_month, use_container_width=True)

st.subheader("Cognitive Insights")
insight_col1, insight_col2 = st.columns(2)
with insight_col1:
    st.markdown(f"- **Savings rate:** {insights['savings_rate']:.1f}%")
    st.markdown(f"- **Recurring burden:** {insights['recurring_burden']:.1f}% of total expense")
    st.markdown(f"- **Biggest spend category:** {insights['biggest_category']}")
with insight_col2:
    st.markdown(f"- **Highest spend month:** {insights['highest_spend_month']}")
    if insights["top_merchants"]:
        merchant_text = ", ".join([f"{m} ({format_inr(v)})" for m, v in insights["top_merchants"]])
        st.markdown(f"- **Top merchants by spend:** {merchant_text}")
    else:
        st.markdown("- **Top merchants by spend:** N/A")
    st.markdown(f"- **ML anomaly alerts:** {anomaly_count}")

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
    alerts.append(f"ML anomaly layer flagged {anomaly_count} unusual expense transaction(s).")

rec_examples = df.loc[df["is_recurring"], "merchant_norm"].dropna().unique().tolist()[:5]
if rec_examples:
    alerts.append("Recurring examples: " + ", ".join(rec_examples))

if alerts:
    for alert in alerts:
        st.warning(alert)
else:
    st.success("No major alerts detected in the current dataset.")

st.subheader("Unusual Spending Detected by ML")
anomaly_view = df[df["is_anomaly"]].copy().sort_values("amount", ascending=False)
if anomaly_view.empty:
    st.info("No unusual spending patterns were flagged in this dataset.")
else:
    show_cols = ["date", "merchant_norm", "category", "amount", "anomaly_reason", "description"]
    anomaly_view["date"] = pd.to_datetime(anomaly_view["date"], errors="coerce").dt.date
    st.dataframe(anomaly_view[show_cols], use_container_width=True)

st.subheader("Recurring Transactions")
recurring_view = df[df["is_recurring"]].copy().sort_values(["merchant_norm", "date"])
if recurring_view.empty:
    st.info("No recurring transactions detected.")
else:
    recurring_view["date"] = pd.to_datetime(recurring_view["date"], errors="coerce").dt.date
    st.dataframe(recurring_view[["date", "merchant_norm", "category", "amount", "description"]], use_container_width=True)

st.subheader("All Transactions")
all_view = df.copy()
all_view["date"] = pd.to_datetime(all_view["date"], errors="coerce").dt.date
st.dataframe(all_view, use_container_width=True)

csv_data = df.copy()
csv_data["date"] = pd.to_datetime(csv_data["date"], errors="coerce").dt.strftime("%Y-%m-%d")
st.download_button(
    "Download Processed CSV",
    data=csv_data.to_csv(index=False).encode("utf-8"),
    file_name="processed_finance_transactions.csv",
    mime="text/csv",
)
