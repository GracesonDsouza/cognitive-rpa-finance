import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from processor import load_csv, extract_from_text

st.set_page_config(page_title="Cognitive RPA Finance Dashboard", layout="wide")

st.title("💳 Cognitive RPA – Personal Finance Consolidation (Level-2 Prototype)")
st.caption("Upload statements/exports → Extract → Categorize → Detect recurring → Dashboard")

with st.sidebar:
    st.header("Upload Data")
    mode = st.radio("Choose input type", ["CSV Transactions", "Text (SMS/Email style)"], index=0)

    uploaded = None
    raw_text = None

    if mode == "CSV Transactions":
        uploaded = st.file_uploader("Upload CSV (date, description, amount)", type=["csv"])
        st.caption("Tip: Use sample_data/transactions.csv for demo.")
    else:
        raw_text = st.text_area(
            "Paste SMS/Email lines",
            height=220,
            placeholder="12/01/2026 - Swiggy - INR 450\n2026-01-10, Netflix, -499\n10-01-2026 Uber Ride - ₹320"
        )

    process = st.button("🚀 Process Data")

def kpi(label, value):
    st.metric(label, value)

if process:
    try:
        if mode == "CSV Transactions":
            if not uploaded:
                st.warning("Please upload a CSV file.")
                st.stop()
            df = load_csv(uploaded)
        else:
            if not raw_text or not raw_text.strip():
                st.warning("Please paste some text lines.")
                st.stop()
            df = extract_from_text(raw_text)

        if df is None or df.empty:
            st.error("No transactions detected. Try a different file or format.")
            st.stop()

        total_income = df.loc[df["amount"] > 0, "amount"].sum()
        total_expense = df.loc[df["amount"] < 0, "amount"].abs().sum()
        net = total_income - total_expense
        recurring_count = int(df["is_recurring"].sum())

        c1, c2, c3, c4 = st.columns(4)
        with c1: kpi("Total Income", f"₹ {total_income:,.0f}")
        with c2: kpi("Total Expense", f"₹ {total_expense:,.0f}")
        with c3: kpi("Net Cashflow", f"₹ {net:,.0f}")
        with c4: kpi("Recurring Transactions", f"{recurring_count}")

        st.divider()

        left, right = st.columns([1.1, 0.9])

        with left:
            st.subheader("📊 Spend by Category")
            cat_exp = df[df["amount"] < 0].copy()
            cat_exp["expense"] = cat_exp["amount"].abs()
            cat_summary = cat_exp.groupby("category")["expense"].sum().sort_values(ascending=False)

            fig = plt.figure()
            plt.bar(cat_summary.index, cat_summary.values)
            plt.xticks(rotation=45, ha="right")
            plt.ylabel("₹ Expense")
            st.pyplot(fig)

        with right:
            st.subheader("📅 Monthly Cashflow")
            df_plot = df.copy()
            df_plot["month"] = pd.to_datetime(df_plot["date"]).dt.to_period("M").astype(str)
            monthly = df_plot.groupby("month")["amount"].sum()

            fig2 = plt.figure()
            plt.plot(monthly.index, monthly.values, marker="o")
            plt.xticks(rotation=45, ha="right")
            plt.ylabel("₹ Net Cashflow")
            st.pyplot(fig2)

        st.subheader("🔁 Recurring Payments (Detected)")
        rec = df[df["is_recurring"] == True].sort_values("date", ascending=False)
        if rec.empty:
            st.info("No recurring payments detected in this dataset.")
        else:
            st.dataframe(rec[["date", "description", "amount", "category"]], use_container_width=True)

        st.subheader("🧾 All Transactions")
        st.dataframe(df.sort_values("date", ascending=False), use_container_width=True)

    except Exception as e:
        st.error(f"Error: {e}")
