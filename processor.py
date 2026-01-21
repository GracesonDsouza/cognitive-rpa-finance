import re
import pandas as pd
import numpy as np
from dateutil import parser

import pdfplumber

CATEGORY_RULES = {
    "Food": ["swiggy", "zomato", "restaurant", "cafe", "dominos", "kfc", "pizza", "starbucks"],
    "Travel": ["uber", "ola", "irctc", "metro", "flight", "makemytrip", "redbus"],
    "Rent": ["rent", "landlord"],
    "EMI/Loan": ["emi", "loan", "bajaj", "hdfc emi", "sbi emi", "icici emi", "axis emi"],
    "Subscriptions": ["netflix", "spotify", "prime", "hotstar", "subscription", "youtube premium"],
    "Shopping": ["amazon", "flipkart", "myntra", "ajio"],
    "Bills/Utilities": ["electric", "water", "gas", "broadband", "recharge", "jio", "airtel", "vi", "bsnl"],
    "Investment": ["sip", "mutual fund", "zerodha", "groww", "upstox", "etf"],
}

DEBIT_WORDS = ["debit", "debited", "paid", "purchase", "spent", "sent", "withdrawn", "dr", "payment", "bill", "emi"]
CREDIT_WORDS = ["credit", "credited", "received", "refund", "salary", "cr", "deposit", "interest"]


def categorize(description: str) -> str:
    if not isinstance(description, str) or not description.strip():
        return "Other"
    d = description.lower()
    for cat, keywords in CATEGORY_RULES.items():
        if any(k in d for k in keywords):
            return cat
    return "Other"


def normalize_amount(x):
    if pd.isna(x):
        return np.nan
    s = str(x).strip()
    s = s.replace(",", "").replace("₹", "")
    s = s.replace("INR", "").replace("Rs.", "").replace("Rs", "").strip()
    s = re.sub(r"[^\d\.\-]", "", s)
    try:
        return float(s)
    except Exception:
        return np.nan


def parse_date(x):
    if pd.isna(x):
        return pd.NaT
    try:
        return parser.parse(str(x), dayfirst=True).date()
    except Exception:
        return pd.NaT


def guess_type_from_text(s: str) -> str:
    s_low = (s or "").lower()
    if any(w in s_low for w in CREDIT_WORDS):
        return "Income"
    if any(w in s_low for w in DEBIT_WORDS):
        return "Expense"
    return "Expense"


def clean_description_context(text: str) -> str:
    if not isinstance(text, str):
        return "Unknown"
    line = text

    line = re.sub(
        r"(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}-\d{2}-\d{2}|\d{1,2}\s[A-Za-z]{3,9}\s\d{4})",
        " ",
        line,
    )
    line = re.sub(r"(₹|INR|Rs\.?)\s?\d[\d,]*\.?\d*", " ", line, flags=re.IGNORECASE)
    line = re.sub(r"\d[\d,]*\.?\d*\s?(INR|Rs\.?)", " ", line, flags=re.IGNORECASE)

    line = re.sub(
        r"\b(on|at|to|from|via|upi|imps|neft|rtgs|txn|txnid|ref|no|a/c|ac|account)\b",
        " ",
        line,
        flags=re.IGNORECASE,
    )
    line = re.sub(r"\s+", " ", line).strip()

    if not line:
        return "Unknown"
    return line[:70]


def detect_recurring(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["merchant_norm"] = (
        df["description"]
        .astype(str)
        .str.lower()
        .str.replace(r"[^a-z0-9\s]", "", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )

    df["month"] = pd.to_datetime(df["date"], errors="coerce").dt.to_period("M").astype(str)
    df["abs_amount"] = df["amount"].abs()
    df["is_recurring"] = False

    for merchant, g in df.groupby("merchant_norm"):
        if len(g) < 2:
            continue

        months = g["month"].nunique()
        mean_amt = g["abs_amount"].mean()
        std_amt = g["abs_amount"].std(ddof=0)

        tol = (mean_amt * 0.10) + 50
        is_rec = (months >= 2) and (std_amt <= tol)

        if is_rec:
            df.loc[g.index, "is_recurring"] = True

    return df


def load_csv(file) -> pd.DataFrame:
    df = pd.read_csv(file)
    df.columns = [c.strip().lower() for c in df.columns]

    required = {"date", "description", "amount"}
    if not required.issubset(set(df.columns)):
        raise ValueError("CSV must contain columns: date, description, amount")

    df["amount"] = df["amount"].apply(normalize_amount)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["category"] = df["description"].apply(categorize)

    if "type" not in df.columns:
        df["type"] = np.where(df["amount"] < 0, "Expense", "Income")

    df = df.dropna(subset=["date", "amount"])
    return detect_recurring(df)


def extract_from_text(text: str) -> pd.DataFrame:
    rows = []
    for line in (text or "").splitlines():
        line = line.strip()
        if not line:
            continue

        date_match = re.search(
            r"(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}-\d{2}-\d{2}|\d{1,2}\s[A-Za-z]{3,9}\s\d{4})",
            line,
        )

        amt_match = re.search(
            r"((₹|INR|Rs\.?)\s?\d[\d,]*\.?\d*|\d[\d,]*\.?\d*\s?(INR|Rs\.?))",
            line,
            re.IGNORECASE,
        )

        merch_match = re.search(r"(?:at|to|from)\s+([A-Za-z0-9 &]+)", line, re.IGNORECASE)

        if date_match and amt_match:
            dt = parse_date(date_match.group(1))
            amt = normalize_amount(amt_match.group(1))

            if merch_match:
                merchant = merch_match.group(1).strip()
                merchant = re.split(r"\bon\b|\bvia\b|[.,]", merchant, flags=re.IGNORECASE)[0].strip()
            else:
                merchant = clean_description_context(line)

            tx_type = guess_type_from_text(line)
            signed_amt = abs(amt) if tx_type == "Income" else -abs(amt)

            if pd.notna(amt):
                rows.append([dt, merchant, signed_amt, tx_type])

    df = pd.DataFrame(rows, columns=["date", "description", "amount", "type"])
    if df.empty:
        return df

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["amount"] = df["amount"].apply(normalize_amount)
    df["amount"] = np.where(df["type"] == "Income", df["amount"].abs(), -df["amount"].abs())
    df["category"] = df["description"].apply(categorize)

    df = df.dropna(subset=["date", "amount"])
    return detect_recurring(df)


def _extract_pdf_text(file) -> str:
    chunks = []
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            t = page.extract_text() or ""
            if t.strip():
                chunks.append(t)
    return "\n".join(chunks)


def _extract_pdf_tables(file) -> pd.DataFrame:
    rows = []
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            tables = page.extract_tables() or []
            for table in tables:
                for r in table:
                    if not r:
                        continue
                    row_str = " ".join([str(x) for x in r if x is not None]).strip()
                    if row_str:
                        rows.append(row_str)

    if not rows:
        return pd.DataFrame()

    return _parse_transactions_from_blob("\n".join(rows))


def _parse_transactions_from_blob(text_blob: str) -> pd.DataFrame:
    """
    Improved PDF parser:
    - Line-by-line parsing (prevents merging multiple transactions into one)
    - Extract date + amount per line
    - Infer income/expense from the same line
    """
    lines = [ln.strip() for ln in (text_blob or "").splitlines() if ln.strip()]

    date_re = re.compile(
        r"(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}-\d{2}-\d{2}|\d{1,2}\s[A-Za-z]{3,9}\s\d{4})"
    )
    amt_re = re.compile(
        r"((₹|INR|Rs\.?)\s?\d[\d,]*\.?\d*|\d[\d,]*\.?\d*\s?(INR|Rs\.?))",
        re.IGNORECASE,
    )

    rows = []
    for ln in lines:
        d = date_re.search(ln)
        a = amt_re.search(ln)
        if not (d and a):
            continue

        dt = parse_date(d.group(1))
        amt = normalize_amount(a.group(1))
        if pd.isna(amt) or dt is pd.NaT:
            continue

        tx_type = guess_type_from_text(ln)
        signed_amt = abs(amt) if tx_type == "Income" else -abs(amt)
        desc = clean_description_context(ln)

        rows.append([dt, desc, signed_amt, tx_type])

    df = pd.DataFrame(rows, columns=["date", "description", "amount", "type"])
    if df.empty:
        return df

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["amount"] = df["amount"].apply(normalize_amount)
    df["amount"] = np.where(df["type"] == "Income", df["amount"].abs(), -df["amount"].abs())
    df["description"] = df["description"].fillna("Unknown")

    df = df.dropna(subset=["date", "amount"])
    df = df.drop_duplicates(subset=["date", "amount", "description"]).reset_index(drop=True)

    df["category"] = df["description"].apply(categorize)
    return detect_recurring(df)


def extract_from_pdf(file) -> pd.DataFrame:
    df_tables = _extract_pdf_tables(file)
    if df_tables is not None and not df_tables.empty:
        return df_tables

    text = _extract_pdf_text(file)
    return _parse_transactions_from_blob(text)
