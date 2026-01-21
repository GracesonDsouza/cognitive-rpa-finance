import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from processor import load_csv, extract_from_text, extract_from_pdf

st.set_page_config(page_title="Cognitive RPA Finance Dashboard", layout="wide")

# ---------- Styling ----------
st.markdown(
    """
    <style>
    .big-title {font-size: 34px; font-weight: 800; margin-bottom: 2px;}
    .subtle {opacity: 0.85; margin-top: 6px;}
    .pill {
        display:inline-block; padding:6px 10px; border-radius:999px;
        border:1px solid rgba(255,255,255,0.2); margin-right:8px;
        font-size: 13px; opacity: 0.9;
    }
    .card {
        border: 1px solid rgba(255,255,255,0.15);
        border-radius: 16px;
        padding: 14px;
        background: rgba(255,255,255,0.03);
    }
    .small {font-size: 13px; opacity: 0.85;}
    </style>
    """,
    unsafe_allow_html=True,
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
    unsafe_allow_html=True,
)
st.write("")

# ---------- Sidebar ----------
with st.sidebar:
    st.header("📥 Input")
    st.caption("Upload data → system extracts & understands transactions automatically.")

    mode = st.radio(
        "Choose input type",
        ["CSV Transactions", "Text (SMS/Email)", "PDF Statement"],
        index=0,
    )

    uploaded = None
    raw_text = None

    if mode == "CSV Transactions":
        uploaded = st.file_uploader("Upload CSV (date, description, amount)", type=["csv"])
        st.caption("Columns needed: date, description, amount")

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
            ),
        )
        st.caption("Works with natural language; not fixed format.")

    else:
        uploaded = st.file_uploader("Upload PDF statement", type=["pdf"])
        st.caption("Best-effort extraction: tables → text → inference.")

    process = st.button("🚀 Process Data", use_container_width=True)

# ---------- Helpers ----------
def to_csv_download(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


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


def get_top_merchants(df, n=5):
    d = df.copy()
    d["abs_amt"] = d["amount"].abs()
    # prefer expenses for merchant ranking
    d = d[d["amount"] < 0]
    if d.empty:
        return pd.Series(dtype=float)
    return d.groupby("description")["abs_amt"].sum().sort_values(ascending=False).head(n)


def compute_insights(df):
    d = df.copy()
    d["month"] = pd.to_datetime(d["date"]).dt.to_period("M").astype(str)

    total_income = d.loc[d["amount"] > 0, "amount"].sum()
    total_expense = d.loc[d["amount"] < 0, "amount"].abs().sum()
    net = total_income - total_expense

    recurring_expense = d.loc[(d["amount"] < 0) & (d["is_recurring"] == True), "amount"].abs().sum()
    recurring_ratio = (recurring_expense / total_expense * 100) if total_expense else 0

    savings_rate = (net / total_income * 100) if total_income else 0

    # biggest category
    if total_expense:
        cat = d[d["amount"] < 0].copy()
        cat["expense"] = cat["amount"].abs()
        biggest_category = cat.groupby("category")["expense"].sum().sort_values(ascending=False).head(1)
        biggest_category_name = biggest_category.index[0] if len(biggest_category) else "N/A"
        biggest_category_amt = float(biggest_category.values[0]) if len(biggest_category) else 0.0
    else:
        biggest_category_name, biggest_category_amt = "N/A", 0.0

    # highest spend month
    m = d.copy()
    m["expense"] = np.where(m["amount"] < 0, m["amount"].abs(), 0.0)
    spend_by_month = m.groupby("month")["expense"].sum().sort_values(ascending=False)
    highest_spend_month = spend_by_month.index[0] if len(spend_by_month) else "N/A"
    highest_spend_amt = float(spend_by_month.values[0]) if len(spend_by_month) else 0.0

    negative_months = (d.groupby("month")["amount"].sum() < 0)
    negative_month_list = list(negative_months[negative_months].index)

    return {
        "total_income": total_income,
        "total_expense": total_expense,
        "net": net,
        "recurring_ratio": recurring_ratio,
        "savings_rate": savings_rate,
        "biggest_category_name": biggest_category_name,
        "biggest_category_amt": biggest_category_amt,
        "highest_spend_month": highest_spend_month,
        "highest_spend_amt": highest_spend_amt,
        "negative_months": negative_month_list,
    }


def build_smart_alerts(df, insights):
    alerts = []

    # recurring burden
    if insights["recurring_ratio"] >= 50:
        alerts.append(f"Recurring expenses are high (~{insights['recurring_ratio']:.0f}% of total spending). Consider trimming subscriptions or refinancing EMIs.")
    elif insights["recurring_ratio"] >= 30:
        alerts.append(f"Recurring expenses are moderate (~{insights['recurring_ratio']:.0f}%). Keep an eye on EMIs/subscriptions.")

    # negative cashflow months
    if insights["negative_months"]:
        months = ", ".join(insights["negative_months"][:4])
        alerts.append(f"Net cashflow went negative in: {months}. This indicates spending exceeded income in those months.")

    # large expense alerts (top 3)
    d = df[df["amount"] < 0].copy()
    if not d.empty:
        d["abs_amt"] = d["amount"].abs()
        top_large = d.sort_values("abs_amt", ascending=False).head(3)
        for _, r in top_large.iterrows():
            alerts.append(f"Large expense spotted: ₹{r['abs_amt']:,.0f} on '{r['description']}' ({r['category']}).")

    # subscription/EMI reminders if recurring exists
    rec = df[df["is_recurring"] == True].copy()
    if not rec.empty:
        # pick a couple of examples
        examples = rec.sort_values("date").groupby("category").head(1)
        for _, r in examples.head(2).iterrows():
            alerts.append(f"Recurring detected: '{r['description']}' likely repeats monthly (Category: {r['category']}).")

    # keep alerts concise
    return alerts[:7]

# ---------- Run ----------
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

            else:
                if not uploaded:
                    st.warning("Please upload a PDF file.")
                    st.stop()
                df = extract_from_pdf(uploaded)

        if df is None or df.empty:
            st.error("No transactions detected. Try a different file/text format.")
            st.stop()

        # compute insights
        insights = compute_insights(df)

        # ---------- Top Row: Profile + Consent ----------
        a, b = st.columns([1, 2])

        with a:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.image("https://i.pravatar.cc/140?img=12", width=110)  # demo avatar
            st.markdown("**Rahul Sharma**")
            st.markdown('<div class="small">Young professional • Bengaluru</div>', unsafe_allow_html=True)
            st.markdown('<div class="small">Primary Bank: HDFC (Demo)</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with b:
            st.success(
                "🔐 **Consent & Privacy Notice**: This prototype processes documents **only after user consent**. "
                "No real banking login is used. Files are for academic demo purposes."
            )

        st.write("")

        # ---------- KPIs ----------
        k1, k2, k3, k4 = st.columns(4)
        with k1: st.metric("Total Income", f"₹ {insights['total_income']:,.0f}")
        with k2: st.metric("Total Expense", f"₹ {insights['total_expense']:,.0f}")
        with k3: st.metric("Net Cashflow", f"₹ {insights['net']:,.0f}")
        with k4: st.metric("Recurring Txns", f"{int(df['is_recurring'].sum())}")

        st.divider()

        # ---------- Insights + Charts ----------
        left, right = st.columns([1.15, 0.85])

        with left:
            st.pyplot(make_spend_by_category_chart(df))
            st.pyplot(make_monthly_cashflow_chart(df))

        with right:
            st.markdown("### 🧠 Cognitive Insights")
            st.markdown(
                f"""
                - **Savings rate:** {insights['savings_rate']:.0f}%  
                - **Recurring burden:** {insights['recurring_ratio']:.0f}% of expenses  
                - **Biggest category:** {insights['biggest_category_name']} (₹ {insights['biggest_category_amt']:,.0f})  
                - **Highest spend month:** {insights['highest_spend_month']} (₹ {insights['highest_spend_amt']:,.0f})  
                """.strip()
            )

            topm = get_top_merchants(df, n=5)
            if len(topm):
                st.markdown("**Top merchants (by spend):**")
                for name, amt in topm.items():
                    st.write(f"• {name} — ₹ {amt:,.0f}")

            st.write("")
            st.markdown("### ⚠️ Smart Alerts")
            alerts = build_smart_alerts(df, insights)
            if not alerts:
                st.info("No alerts for this dataset.")
            else:
                for al in alerts:
                    st.warning(al)

        st.divider()

        # ---------- Recurring ----------
        st.subheader("🔁 Recurring Payments (Detected)")
        rec = df[df["is_recurring"] == True].sort_values("date", ascending=False)
        if rec.empty:
            st.info("No recurring payments detected. Try multi-month entries.")
        else:
            st.dataframe(rec[["date", "description", "amount", "category"]], use_container_width=True)

        # ---------- All Transactions ----------
        st.subheader("🧾 All Transactions")
        st.dataframe(df.sort_values("date", ascending=False), use_container_width=True)

        st.download_button(
            "⬇️ Download Processed Transactions (CSV)",
            data=to_csv_download(df),
            file_name="processed_transactions.csv",
            mime="text/csv",
            use_container_width=True,
        )

    except Exception as e:
        st.error(f"Error: {e}")
