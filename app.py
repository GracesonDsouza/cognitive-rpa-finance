import pandas as pd
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

st.title("Cognitive RPA for Personal Finance Data Consolidation")
st.write("Unified pipeline: CSV + PDF + Text → one dataset → dedup → learning → recurring → anomaly detection")

CATEGORIES = [
    "Food", "Travel", "Rent", "EMI/Loan", "Subscriptions",
    "Shopping", "Bills/Utilities", "Investment", "Other", "Income"
]

def format_inr(x):
    return f"₹{x:,.2f}"

# Sidebar
with st.sidebar:
    st.header("Settings")
    consented = st.checkbox("I consent to process this financial data.", value=True)

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

# Inputs
st.subheader("Input Data")

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
income_total = df.loc[df["type"] == "income", "amount"].sum()
expense_total = df.loc[df["type"] == "expense", "amount"].sum()
net_cashflow = income_total - expense_total
anomaly_count = int(df["is_anomaly"].sum()) if "is_anomaly" in df.columns else 0
recurring_count = int(df["is_recurring"].sum()) if "is_recurring" in df.columns else 0

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Income", format_inr(income_total))
k2.metric("Expense", format_inr(expense_total))
k3.metric("Net Cashflow", format_inr(net_cashflow))
k4.metric("Anomalies", anomaly_count)
k5.metric("Duplicates Removed", duplicates_removed)

tab1, tab2, tab3, tab4, tab5 = st.tabs(["Overview", "Anomalies", "Recurring", "Learning", "Raw Data"])

with tab1:
    st.subheader("Overview")
    st.write("Sources processed:", ", ".join(sorted(df["source"].dropna().unique().tolist())))
    st.write("Recurring transactions detected:", recurring_count)

    if not df[df["type"] == "expense"].empty:
        spend_by_category = (
            df[df["type"] == "expense"]
            .groupby("category", as_index=False)["amount"]
            .sum()
            .sort_values("amount", ascending=False)
        )
        st.bar_chart(spend_by_category.set_index("category"))

with tab2:
    st.subheader("Anomalies")
    anomaly_view = df[df["is_anomaly"]].copy().sort_values("amount", ascending=False)
    if anomaly_view.empty:
        st.info("No anomalies detected.")
    else:
        anomaly_view["date"] = pd.to_datetime(anomaly_view["date"], errors="coerce").dt.date
        st.dataframe(
            anomaly_view[["date", "merchant_norm", "category", "source", "amount", "anomaly_reason", "description"]],
            use_container_width=True
        )

with tab3:
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

with tab4:
    st.subheader("Ask-Once Learning")
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

with tab5:
    st.subheader("Raw Data")
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
