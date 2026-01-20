import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from processor import load_csv, extract_from_text, extract_from_pdf

st.set_page_config(page_title="Cognitive RPA Finance Dashboard", layout="wide")

# ---------- Cool header styling ----------
st.markdown(
    """
    <style>
    .big-title {font-size: 34px; font-weight: 800; margin-bottom: 0px;}
    .subtle {opacity: 0.85; margin-top: 6px;}
    .pill {
        display:inline-block; padding:6px 10px; border-radius:999px;
        border:1px solid rgba(255,255,255,0.2); margin-right:8px;
        font-size: 13px; opacity: 0.9;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<div class="big-title">💳 Cognitive RPA – Personal Finance Consolidation</div>', unsafe_allow_html=True)
st.markdown(
    """
    <div class="subtle">
      <span class="pill">RPA Extraction</span>
      <span class="pill">Unstructured Understanding</span>
      <span class="pill">Pattern Recognition</span>
      <span class="pill">Unified Dashboard</span>
    </div>
    """,
    unsafe_allow_html=True
)

st.write("")

# ---------- Sidebar ----------
with st.sidebar:
    st.header("📥 Input")
    st.caption("Upload data → system extracts & understands transactions automatically.")

    mode = st.radio(
        "Choose input type",
        ["CSV Transactions", "Text (SMS/Email)", "PDF Statement"],
        index=0
    )

    uploaded = None
    raw_text = None

    if mode == "CSV Transactions":
        uploaded = st.file_uploader("Upload CSV (date, description, amount)", type=["csv"])
        st.caption("Tip: your CSV must contain columns: date, description, amount")

    elif mode == "Text (SMS/Email)":
        raw_text = st.text_area(
            "Paste SMS/Email lines",
            height=240,
            placeholder=(
                "Your account was debited with INR 499 at Netflix on 05 Jan 2025.\n"
                "Your account was debited with INR 499 at Netflix on 05 Feb 2025.\n"
                "Payment of Rs. 15000 successful to HDFC Bank EMI on 10 Jan 2025.\n"
                "Payment of Rs. 15000 successful to HDFC Bank EMI on 10 Feb 2025.\n"
                "Salary credited INR 50000 on 01-02-2025."
            )
        )
        st.caption("Works with natural language (not fixed format).")

    else:  # PDF
        uploaded = st.file_uploader("Upload PDF statement", type=["pdf"])
        st.caption("Best-effort extraction: tables → text → transaction inference.")

    process = st.button("🚀 Process Data", use_container_width=True)

def kpi(label, value):
    st.metric(label, value)

def make_spend_by_category_chart(df):
    cat_exp = df[df["amount"] < 0].copy()
    cat_exp["expense"] = cat_exp["amount"].abs()
    cat_summary = cat_exp.groupby("category")["expense"].sum().sort_values(ascending=False)

    fig = plt.figure()
    plt.bar(cat_summary.index, cat_summary.values)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("₹ Expense")
    plt.title("Spend by Category")
    return fig

def make_monthly_cashflow_chart(df):
    df_plot = df.copy()
    df_plot["month"] = pd.to_datetime(df_plot["date"]).dt.to_period("M").astype(str)
    monthly = df_plot.groupby("month")["amount"].sum()

    fig = plt.figure()
    plt.plot(monthly.index, monthly.values, marker="o")
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("₹ Net Cashflow")
    plt.title("Monthly Cashflow")
    return fig

def to_csv_download(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")

if process:
    try:
        with st.spinner("Extracting + understanding transactions..."):
            if mode == "CSV Transactions":
                if not uploaded:
                    st.warning("Please upload a CSV file.")
                    st.stop()
                df = load_csv(uploaded)

            elif mode == "Text (SMS/Email)":
                if not raw_text or not raw_text.strip():
                    st.warning("Please paste some SMS/Email text.")
                    st.stop()
                df = extract_from_text(raw_text)

            else:  # PDF
                if not uploaded:
                    st.warning("Please upload a PDF file.")
                    st.stop()
                df = extract_from_pdf(uploaded)

        if df is None or df.empty:
            st.error("No transactions detected. Try a different file/text format.")
            st.stop()

        # KPIs
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

        # Charts
        left, right = st.columns([1.15, 0.85])
        with left:
            st.pyplot(make_spend_by_category_chart(df))
        with right:
            st.pyplot(make_monthly_cashflow_chart(df))

        st.divider()

        # Recurring section
        st.subheader("🔁 Recurring Payments (Detected)")
        rec = df[df["is_recurring"] == True].sort_values("date", ascending=False)
        if rec.empty:
            st.info("No recurring payments detected in this dataset. Try multi-month entries.")
        else:
            st.dataframe(rec[["date", "description", "amount", "category"]], use_container_width=True)

        # All transactions + download
        st.subheader("🧾 All Transactions")
        st.dataframe(df.sort_values("date", ascending=False), use_container_width=True)

        st.download_button(
            "⬇️ Download Processed Transactions (CSV)",
            data=to_csv_download(df),
            file_name="processed_transactions.csv",
            mime="text/csv",
            use_container_width=True
        )

        # Explainability block (makes it feel more “cognitive”)
        st.markdown("### 🧠 How the system ‘understands’ your data")
        st.write(
            "- **RPA Extraction**: reads CSV/text/PDF and pulls possible transaction signals (date, amount, context).\n"
            "- **Cognitive Parsing**: infers merchant/description from surrounding words, even if format varies.\n"
            "- **Pattern Recognition**: flags recurring payments across months when merchant + amount is stable.\n"
            "- **Consolidation**: builds a unified view (KPIs + charts + transaction table)."
        )

    except Exception as e:
        st.error(f"Error: {e}")
