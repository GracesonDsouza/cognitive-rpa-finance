import re
import pandas as pd
import numpy as np
from dateutil import parser
import pdfplumber

# ----------------------------
# 1) Category Rules (keywords)
# ----------------------------
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


# ----------------------------
# 2) Helper functions
# ----------------------------
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
    # default assumption for unknown lines:
    return "Expense"


def clean_description_context(text: str) -> str:
    """
    Removes date/amount/noise and returns a short best-effort description.
    """
    if not isinstance(text, str):
        return "Unknown"

    line = text
    # Remove common date formats
    line = re.sub(
        r"(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}-\d{2}-\d{2}|\d{1,2}\s[A-Za-z]{3,9}\s\d{4})",
        " ",
        line,
    )
    # Remove amounts
    line = re.sub(r"(₹|INR|Rs\.?)\s?\d[\d,]*\.?\d*", " ", line, flags=re.IGNORECASE)
    line = re.sub(r"\d[\d,]*\.?\d*\s?(INR|Rs\.?)", " ", line, flags=re.IGNORECASE)

    # Remove transaction plumbing words
    line = re.sub(
        r"\b(on|at|to|from|via|upi|imps|neft|rtgs|txn|txnid|ref|no|a/c|ac|account)\b",
        " ",
        line,
        flags=re.IGNORECASE,
    )

    line = re.sub(r"\s+", " ", line).strip()
    return line[:70] if line else "Unknown"


# ----------------------------
# 3) Merchant Normalization (KEY FIX)
# ----------------------------
def normalize_merchant(description: str) -> str:
    """
    Convert messy unstructured descriptions into a canonical merchant identity.
    This is what prevents:
      rent paid / rent paid landlord / monthly rent debit -> "rent"
    """
    s = (description or "").lower()

    # Strong canonical rules first
    if "rent" in s:
        return "rent"

    if "emi" in s:
        # common bank EMIs
        if "hdfc" in s:
            return "hdfc bank emi"
        if "sbi" in s:
            return "sbi emi"
        if "icici" in s:
            return "icici emi"
        if "axis" in s:
            return "axis emi"
        return "emi"

    # subscription merchants
    if "netflix" in s:
        return "netflix"
    if "spotify" in s:
        return "spotify"
    if "prime" in s or "amazon prime" in s:
        return "amazon prime"
    if "hotstar" in s:
        return "hotstar"

    # common merchants
    if "swiggy" in s:
        return "swiggy"
    if "zomato" in s:
        return "zomato"
    if "uber" in s:
        return "uber"
    if "ola" in s:
        return "ola"
    if "amazon" in s:
        return "amazon"
    if "flipkart" in s:
        return "flipkart"

    # banks (non-emi)
    if "hdfc" in s:
        return "hdfc bank"
    if "icici" in s:
        return "icici bank"
    if "sbi" in s:
        return "sbi bank"
    if "axis" in s:
        return "axis bank"

    # fallback: keep first 2 meaningful tokens (still better than raw line)
    cleaned = re.sub(r"[^a-z0-9\s]", " ", s)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    toks = cleaned.split()
    if not toks:
        return "unknown"
    return " ".join(toks[:2])


# ----------------------------
# 4) Recurring Detection (uses merchant_norm)
# ----------------------------
def detect_recurring(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Create normalized merchant identity
    df["merchant_norm"] = df["description"].astype(str).apply(normalize_merchant)

    df["month"] = pd.to_datetime(df["date"], errors="coerce").dt.to_period("M").astype(str)
    df["abs_amount"] = df["amount"].abs()
    df["is_recurring"] = False

    for merchant, g in df.groupby("merchant_norm"):
        if len(g) < 2:
            continue

        months = g["month"].nunique()
        mean_amt = g["abs_amount"].mean()
        std_amt = g["abs_amount"].std(ddof=0)

        # tolerance: 10% + ₹50
        tol = (mean_amt * 0.10) + 50
        is_rec = (months >= 2) and (std_amt <= tol)

        if is_rec:
            df.loc[g.index, "is_recurring"] = True

    return df


# ----------------------------
# 5) Input Handlers (CSV / Text / PDF)
# ----------------------------
def load_csv(file) -> pd.DataFrame:
    df = pd.read_csv(file)
    df.columns = [c.strip().lower() for c in df.columns]

    required = {"date", "description", "amount"}
    if not required.issubset(set(df.columns)):
        raise ValueError("CSV must contain columns: date, description, amount")

    df["amount"] = df["amount"].apply(normalize_amount)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    if "type" not in df.columns:
        df["type"] = np.where(df["amount"] < 0, "Expense", "Income")

    df["category"] = df["description"].apply(categorize)
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

        # try capturing merchant after at/to/from
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
    PDF parser:
    - line-by-line extraction to avoid merging multiple transactions
    - date + amount per line
    - infer income/expense from same line
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
    # Prefer table extraction if possible
    df_tables = _extract_pdf_tables(file)
    if df_tables is not None and not df_tables.empty:
        return df_tables

    # Else use text extraction
    text = _extract_pdf_text(file)
    return _parse_transactions_from_blob(text)
