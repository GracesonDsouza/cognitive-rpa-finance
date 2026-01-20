import re
import pandas as pd
import numpy as np
from dateutil import parser

# ----------------------------
# Simple "cognitive" rules (keyword-based)
# You can expand these later with ML/NLP.
# ----------------------------
CATEGORY_RULES = {
    "Food": ["swiggy", "zomato", "restaurant", "cafe", "dominos", "kfc", "pizza", "starbucks"],
    "Travel": ["uber", "ola", "irctc", "metro", "flight", "makemytrip", "redbus"],
    "Rent": ["rent", "landlord"],
    "EMI/Loan": ["emi", "loan", "bajaj", "hdfc emi", "sbi emi", "icici emi"],
    "Subscriptions": ["netflix", "spotify", "prime", "hotstar", "subscription", "youtube premium"],
    "Shopping": ["amazon", "flipkart", "myntra", "ajio"],
    "Bills/Utilities": ["electric", "water", "gas", "broadband", "recharge", "jio", "airtel", "vi", "bsnl"],
    "Investment": ["sip", "mutual fund", "zerodha", "groww", "upstox", "etf"],
}

def categorize(description: str) -> str:
    """Assign a category from the keyword rules."""
    if not isinstance(description, str) or not description.strip():
        return "Other"
    d = description.lower()
    for cat, keywords in CATEGORY_RULES.items():
        if any(k in d for k in keywords):
            return cat
    return "Other"

def normalize_amount(x):
    """Convert ₹ and commas to float. Handles negatives too."""
    if pd.isna(x):
        return np.nan
    s = str(x).replace("₹", "").replace(",", "").strip()
    # Remove common currency words
    s = s.replace("INR", "").replace("Rs.", "").replace("Rs", "").strip()
    try:
        return float(s)
    except Exception:
        return np.nan

def parse_date(x):
    """Parse common date formats into a python date (or NaT if failed)."""
    if pd.isna(x):
        return pd.NaT
    try:
        return parser.parse(str(x), dayfirst=True).date()
    except Exception:
        return pd.NaT

def detect_recurring(df: pd.DataFrame) -> pd.DataFrame:
    """
    Basic recurring detection:
    - Same normalized merchant appears in 2+ different months
    - Amount is roughly stable (low std dev)
    """
    df = df.copy()
    df["merchant_norm"] = (
        df["description"]
        .astype(str)
        .str.lower()
        .str.replace(r"[^a-z0-9\s]", "", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    df["month"] = pd.to_datetime(df["date"]).dt.to_period("M").astype(str)
    df["abs_amount"] = df["amount"].abs()

    df["is_recurring"] = False
    for m, g in df.groupby("merchant_norm"):
        if len(g) < 2:
            continue
        months = g["month"].nunique()
        mean_amt = g["abs_amount"].mean()
        std_amt = g["abs_amount"].std(ddof=0)

        # tolerance: 10% of mean + ₹50
        tol = (mean_amt * 0.10) + 50
        is_rec = (months >= 2) and (std_amt <= tol)
        if is_rec:
            df.loc[g.index, "is_recurring"] = True

    return df

def load_csv(file) -> pd.DataFrame:
    """
    Load CSV with columns: date, description, amount
    Optional: type (Income/Expense)
    """
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
    """
    Demo parser for SMS/email-like lines.

    Supported line examples:
      12/01/2026 - Swiggy - INR 450
      2026-01-10, Netflix, -499
      10-01-2026 Uber Ride - ₹320
    """
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue

        # date ... description ... amount
        m = re.search(
            r"(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}-\d{2}-\d{2}).*?([A-Za-z0-9 &]+).*?(-?\s?₹?\s?\d[\d,]*\.?\d*)",
            line
        )
        if m:
            dt = parse_date(m.group(1))
            desc = m.group(2).strip()
            amt = normalize_amount(m.group(3))
            if pd.notna(amt):
                rows.append([dt, desc, amt])

    df = pd.DataFrame(rows, columns=["date", "description", "amount"])
    if df.empty:
        return df

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["amount"] = df["amount"].apply(normalize_amount)
    df["category"] = df["description"].apply(categorize)
    df["type"] = np.where(df["amount"] < 0, "Expense", "Income")
    df = df.dropna(subset=["date", "amount"])
    return detect_recurring(df)
