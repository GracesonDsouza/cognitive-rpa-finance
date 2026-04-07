import io
import re
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

try:
    import pdfplumber
except Exception:  # pragma: no cover
    pdfplumber = None


CATEGORY_KEYWORDS = {
    "Food": ["swiggy", "zomato", "restaurant", "cafe", "food", "dine", "pizza", "burger", "coffee", "starbucks"],
    "Travel": ["uber", "ola", "rapido", "irctc", "flight", "metro", "taxi", "travel", "air", "bus", "train"],
    "Rent": ["rent", "landlord", "lease"],
    "EMI/Loan": ["emi", "loan", "installment", "hdfc loan", "credit card bill", "icici loan"],
    "Subscriptions": ["netflix", "spotify", "prime", "amazon prime", "hotstar", "youtube premium", "subscription"],
    "Shopping": ["amazon", "flipkart", "myntra", "shopping", "mall", "purchase", "store"],
    "Bills/Utilities": ["electricity", "water", "gas", "broadband", "wifi", "recharge", "mobile bill", "utility", "bill", "insurance"],
    "Investment": ["sip", "mutual fund", "zerodha", "groww", "investment", "stock", "nps", "ppf"],
}

INCOME_KEYWORDS = [
    "credited", "salary", "received", "income", "refund", "cashback", "deposit",
    "neft cr", "upi cr", "bonus", "interest credit", "transfer from"
]

EXPENSE_KEYWORDS = [
    "debited", "debit", "paid", "payment", "sent", "purchase", "emi", "rent",
    "withdrawal", "bill", "spent", "dr", "upi/", "pos", "card", "subscription"
]

MERCHANT_PATTERNS = [
    (r"rent|landlord|lease", "rent"),
    (r"hdfc.*emi|emi.*hdfc|loan.*hdfc", "hdfc bank emi"),
    (r"netflix", "netflix"),
    (r"spotify", "spotify"),
    (r"amazon(?!\s*pay)", "amazon"),
    (r"amazon pay|amzn pay", "amazon pay"),
    (r"swiggy", "swiggy"),
    (r"zomato", "zomato"),
    (r"uber", "uber"),
    (r"ola", "ola"),
    (r"rapido", "rapido"),
    (r"flipkart", "flipkart"),
    (r"myntra", "myntra"),
    (r"electricity|power|bescom|adani electricity|torrent power", "electricity bill"),
    (r"water bill|waterboard", "water bill"),
    (r"airtel|jio|vi recharge|vodafone", "mobile recharge"),
    (r"broadband|wifi|internet|act fibernet", "internet bill"),
    (r"salary|payroll|company salary|employer", "salary"),
    (r"atm", "atm withdrawal"),
    (r"mutual fund|zerodha|groww|sip", "investment"),
]

STANDARD_COLUMNS = ["date", "amount", "description", "type", "category", "merchant_norm"]


def normalize_amount(x) -> Optional[float]:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return np.nan

    if isinstance(x, (int, float, np.integer, np.floating)):
        return float(x)

    s = str(x).strip().lower()
    if s in {"", "none", "nan", "na", "null"}:
        return np.nan

    negative = False
    if "(" in s and ")" in s:
        negative = True
    if s.startswith("-"):
        negative = True

    s = s.replace(",", "")
    s = re.sub(r"(?i)inr|rs\.?|₹|mrp", "", s)
    s = re.sub(r"(?i)cr|dr", "", s)
    match = re.search(r"-?\d+(?:\.\d+)?", s)
    if not match:
        return np.nan

    value = float(match.group())
    if negative and value > 0:
        value = -value
    return value


def parse_date(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return pd.NaT
    try:
        return pd.to_datetime(x, errors="coerce", dayfirst=True)
    except Exception:
        return pd.NaT


def guess_type_from_text(s: str) -> str:
    text = str(s).lower()
    if any(k in text for k in INCOME_KEYWORDS):
        return "income"
    if any(k in text for k in EXPENSE_KEYWORDS):
        return "expense"
    return "expense"


def clean_description_context(text: str) -> str:
    s = str(text)
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"(?i)inr|rs\.?|₹", "", s)
    s = re.sub(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b", "", s)
    s = re.sub(r"\b\d{4}-\d{2}-\d{2}\b", "", s)
    s = re.sub(r"\b\d+(?:\.\d+)?\b", "", s)
    s = re.sub(r"\s+", " ", s).strip(" -:|,")
    return s or "Transaction"


def normalize_merchant(description: str) -> str:
    text = str(description).lower().strip()
    if not text:
        return "unknown"

    for pattern, canonical in MERCHANT_PATTERNS:
        if re.search(pattern, text, flags=re.IGNORECASE):
            return canonical

    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\b(upi|neft|imps|debit|credit|txn|ref|utr|via|to|from|paid|payment|purchase|sent|received)\b", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return "unknown"

    tokens = [t for t in text.split() if len(t) > 2]
    return " ".join(tokens[:3]) if tokens else "unknown"


def categorize(description: str, tx_type: Optional[str] = None) -> str:
    text = str(description).lower()
    if tx_type == "income" or any(k in text for k in ["salary", "credited", "refund", "bonus", "interest"]):
        return "Income"

    for category, keywords in CATEGORY_KEYWORDS.items():
        if any(k in text for k in keywords):
            return category
    return "Other"


def _find_column(columns, candidates):
    normalized = {c.lower().strip(): c for c in columns}
    for candidate in candidates:
        for col_lower, original in normalized.items():
            if candidate in col_lower:
                return original
    return None


def _standardize_dataframe(raw_df: pd.DataFrame) -> pd.DataFrame:
    df = raw_df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    date_col = _find_column(df.columns, ["date", "txn date", "transaction date", "posted date", "value date"])
    desc_col = _find_column(df.columns, ["description", "narration", "details", "remarks", "merchant", "transaction"])
    amount_col = _find_column(df.columns, ["amount", "amt"])
    type_col = _find_column(df.columns, ["type", "dr/cr", "transaction type", "nature"])
    credit_col = _find_column(df.columns, ["credit", "deposit"])
    debit_col = _find_column(df.columns, ["debit", "withdrawal", "spent"])

    standardized = pd.DataFrame()
    standardized["date"] = df[date_col].apply(parse_date) if date_col else pd.NaT

    if amount_col:
        standardized["amount_raw"] = df[amount_col].apply(normalize_amount)
    elif credit_col or debit_col:
        credit_vals = df[credit_col].apply(normalize_amount) if credit_col else pd.Series(np.nan, index=df.index)
        debit_vals = df[debit_col].apply(normalize_amount) if debit_col else pd.Series(np.nan, index=df.index)
        standardized["amount_raw"] = credit_vals.fillna(0) - debit_vals.fillna(0)
    else:
        numeric_like_cols = [c for c in df.columns if df[c].astype(str).str.contains(r"\d", regex=True, na=False).mean() > 0.8]
        if numeric_like_cols:
            standardized["amount_raw"] = df[numeric_like_cols[-1]].apply(normalize_amount)
        else:
            standardized["amount_raw"] = np.nan

    if desc_col:
        standardized["description"] = df[desc_col].astype(str)
    else:
        text_cols = [c for c in df.columns if c not in {date_col, amount_col, type_col, credit_col, debit_col}]
        standardized["description"] = df[text_cols].astype(str).agg(" | ".join, axis=1) if text_cols else "Transaction"

    if type_col:
        standardized["type"] = df[type_col].astype(str).str.lower().map(
            lambda x: "income" if any(k in x for k in ["cr", "credit", "income"]) else "expense"
        )
    else:
        standardized["type"] = standardized.apply(_infer_type_from_row, axis=1)

    return _finalize_transactions(standardized)


def _infer_type_from_row(row) -> str:
    amount = row.get("amount_raw", np.nan)
    desc = str(row.get("description", ""))

    if pd.notna(amount):
        if amount < 0:
            return "expense"
        if amount > 0 and any(k in desc.lower() for k in INCOME_KEYWORDS):
            return "income"
    return guess_type_from_text(desc)


def _finalize_transactions(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "date" not in out.columns:
        out["date"] = pd.NaT
    out["date"] = out["date"].apply(parse_date)

    if "amount_raw" not in out.columns and "amount" in out.columns:
        out["amount_raw"] = out["amount"].apply(normalize_amount)
    out["amount_raw"] = out["amount_raw"].apply(normalize_amount)

    if "description" not in out.columns:
        out["description"] = "Transaction"
    out["description"] = out["description"].astype(str).apply(lambda x: x.strip() if x.strip() else "Transaction")

    if "type" not in out.columns:
        out["type"] = out.apply(_infer_type_from_row, axis=1)
    out["type"] = out["type"].astype(str).str.lower().replace({"credit": "income", "debit": "expense"})

    out["amount"] = out["amount_raw"].abs()
    out.loc[out["amount"] == 0, "amount"] = np.nan

    out["description"] = out["description"].apply(clean_description_context)
    out["merchant_norm"] = out["description"].apply(normalize_merchant)
    out["category"] = out.apply(lambda r: categorize(r["description"], r["type"]), axis=1)
    out["month"] = out["date"].dt.to_period("M").astype(str)
    out["day"] = out["date"].dt.day

    out = out.dropna(subset=["amount"]).copy()
    out = out[out["amount"] > 0].copy()
    out = out.sort_values("date", ascending=True, na_position="last").reset_index(drop=True)

    return out[[c for c in ["date", "amount", "description", "type", "category", "merchant_norm", "month", "day"] if c in out.columns]]


def load_csv(file) -> pd.DataFrame:
    df = pd.read_csv(file)
    return _standardize_dataframe(df)


DATE_REGEX = re.compile(
    r"(\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b\d{4}-\d{2}-\d{2}\b|\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4}\b)",
    flags=re.IGNORECASE,
)
AMOUNT_REGEX = re.compile(r"(?:₹|INR|Rs\.?\s*)?\s*-?\d[\d,]*(?:\.\d{1,2})?", flags=re.IGNORECASE)


def _parse_line_to_transaction(line: str):
    if not line or not line.strip():
        return None

    date_match = DATE_REGEX.search(line)
    amount_matches = AMOUNT_REGEX.findall(line)
    if not date_match or not amount_matches:
        return None

    date_text = date_match.group(0)
    amount_text = amount_matches[-1]
    amount = normalize_amount(amount_text)
    if pd.isna(amount):
        return None

    tx_type = guess_type_from_text(line)
    description = clean_description_context(line)

    return {
        "date": parse_date(date_text),
        "amount_raw": amount if tx_type == "income" else -abs(amount),
        "description": description,
        "type": tx_type,
    }


def extract_from_text(text: str) -> pd.DataFrame:
    rows = []
    for raw_line in str(text).splitlines():
        line = raw_line.strip()
        if not line:
            continue
        row = _parse_line_to_transaction(line)
        if row:
            rows.append(row)

    return _finalize_transactions(pd.DataFrame(rows)) if rows else pd.DataFrame(columns=STANDARD_COLUMNS)


def _extract_pdf_text(file) -> str:
    if pdfplumber is None:
        return ""

    text_parts = []
    file.seek(0)
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text() or ""
            text_parts.append(page_text)
    file.seek(0)
    return "\n".join(text_parts)


def _extract_pdf_tables(file):
    if pdfplumber is None:
        return []

    tables = []
    file.seek(0)
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            page_tables = page.extract_tables() or []
            for table in page_tables:
                if not table or len(table) < 2:
                    continue
                header = [str(h).strip() if h is not None else "" for h in table[0]]
                body = table[1:]
                try:
                    temp_df = pd.DataFrame(body, columns=header)
                except Exception:
                    continue
                tables.append(temp_df)
    file.seek(0)
    return tables


def _parse_transactions_from_blob(text_blob: str) -> pd.DataFrame:
    rows = []
    for raw_line in str(text_blob).splitlines():
        line = raw_line.strip()
        if not line:
            continue
        row = _parse_line_to_transaction(line)
        if row:
            rows.append(row)

    return _finalize_transactions(pd.DataFrame(rows)) if rows else pd.DataFrame(columns=STANDARD_COLUMNS)


def extract_from_pdf(file) -> pd.DataFrame:
    all_rows = []
    for table_df in _extract_pdf_tables(file):
        try:
            standardized = _standardize_dataframe(table_df)
            if not standardized.empty:
                all_rows.append(standardized)
        except Exception:
            continue

    if all_rows:
        combined = pd.concat(all_rows, ignore_index=True)
        combined = combined.drop_duplicates(subset=["date", "amount", "description", "type"]).reset_index(drop=True)
        return combined

    text_blob = _extract_pdf_text(file)
    return _parse_transactions_from_blob(text_blob)


def detect_recurring(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        empty = df.copy() if isinstance(df, pd.DataFrame) else pd.DataFrame(columns=STANDARD_COLUMNS)
        empty["is_recurring"] = False
        return empty

    out = df.copy()
    out["is_recurring"] = False

    expense_df = out[out["type"] == "expense"].copy()
    if expense_df.empty:
        return out

    stats = (
        expense_df.groupby("merchant_norm")
        .agg(
            month_count=("month", lambda s: s.nunique()),
            tx_count=("merchant_norm", "size"),
            amount_mean=("amount", "mean"),
            amount_std=("amount", "std"),
        )
        .reset_index()
    )
    stats["amount_std"] = stats["amount_std"].fillna(0)
    stats["cv"] = stats["amount_std"] / stats["amount_mean"].replace(0, np.nan)
    recurring_merchants = stats[(stats["month_count"] >= 2) & ((stats["cv"] <= 0.20) | (stats["tx_count"] >= 3))]["merchant_norm"].tolist()

    out.loc[out["merchant_norm"].isin(recurring_merchants) & (out["type"] == "expense"), "is_recurring"] = True
    return out


def _build_anomaly_reason(row, amount_threshold, rare_merchants):
    reasons = []
    if row["amount"] >= amount_threshold:
        reasons.append("high-value expense")
    if row["merchant_norm"] in rare_merchants:
        reasons.append("rare merchant")
    if row.get("is_recurring", False) is False and row["category"] in {"Shopping", "Travel", "Other"}:
        reasons.append("unusual spending pattern")
    return ", ".join(reasons) if reasons else "model flagged unusual expense"


def detect_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        empty = df.copy() if isinstance(df, pd.DataFrame) else pd.DataFrame(columns=STANDARD_COLUMNS)
        empty["is_anomaly"] = False
        empty["anomaly_score"] = np.nan
        empty["anomaly_reason"] = ""
        return empty

    out = df.copy()
    out["is_anomaly"] = False
    out["anomaly_score"] = np.nan
    out["anomaly_reason"] = ""

    expense_idx = out[out["type"] == "expense"].index
    expense_df = out.loc[expense_idx].copy()

    if len(expense_df) < 5:
        if not expense_df.empty:
            threshold = expense_df["amount"].quantile(0.90)
            flagged = expense_df[expense_df["amount"] >= threshold].index
            out.loc[flagged, "is_anomaly"] = True
            out.loc[flagged, "anomaly_score"] = 1.0
            out.loc[flagged, "anomaly_reason"] = "high-value expense"
        return out

    merchant_freq = expense_df["merchant_norm"].value_counts()
    category_freq = expense_df["category"].value_counts()

    features = pd.DataFrame({
        "log_amount": np.log1p(expense_df["amount"]),
        "day": expense_df["day"].fillna(1),
        "month_num": pd.to_datetime(expense_df["date"], errors="coerce").dt.month.fillna(1),
        "merchant_freq": expense_df["merchant_norm"].map(merchant_freq).fillna(1),
        "category_freq": expense_df["category"].map(category_freq).fillna(1),
    }, index=expense_df.index)

    category_dummies = pd.get_dummies(expense_df["category"], prefix="cat")
    X = pd.concat([features, category_dummies], axis=1)

    contamination = min(0.15, max(0.05, 3 / max(len(expense_df), 10)))
    model = IsolationForest(n_estimators=200, contamination=contamination, random_state=42)
    preds = model.fit_predict(X)
    scores = -model.score_samples(X)

    out.loc[expense_df.index, "is_anomaly"] = preds == -1
    out.loc[expense_df.index, "anomaly_score"] = scores

    amount_threshold = expense_df["amount"].quantile(0.95)
    rare_merchants = set(merchant_freq[merchant_freq <= 1].index)
    for idx in expense_df.index:
        if out.loc[idx, "is_anomaly"]:
            out.loc[idx, "anomaly_reason"] = _build_anomaly_reason(out.loc[idx], amount_threshold, rare_merchants)

    return out


def enrich_transactions(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=[
            "date", "amount", "description", "type", "category", "merchant_norm",
            "month", "day", "is_recurring", "is_anomaly", "anomaly_score", "anomaly_reason"
        ])

    out = detect_recurring(df)
    out = detect_anomalies(out)
    return out.sort_values("date", ascending=True, na_position="last").reset_index(drop=True)
