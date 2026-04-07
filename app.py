import os
import re
from typing import Optional, List

import numpy as np
import pandas as pd

try:
    from sklearn.ensemble import IsolationForest
except Exception:
    IsolationForest = None

try:
    import pdfplumber
except Exception:
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

STANDARD_COLUMNS = [
    "transaction_id",
    "date",
    "amount",
    "description",
    "type",
    "category",
    "merchant_norm",
    "source",
    "month",
    "day",
]

MEMORY_FILE = "user_memory.csv"

DATE_REGEX = re.compile(
    r"(\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b\d{4}-\d{2}-\d{2}\b|\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4}\b)",
    flags=re.IGNORECASE,
)

AMOUNT_REGEX = re.compile(r"(?:₹|INR|Rs\.?\s*)?\s*-?\d[\d,]*(?:\.\d{1,2})?", flags=re.IGNORECASE)

TRANSACTION_ID_REGEX = re.compile(
    r"(?i)(?:txn(?:action)?\s*id|txn\s*ref|utr|ref(?:erence)?\s*no|txn\s*no)[:\s\-]*([A-Za-z0-9\-_\/]+)"
)


def ensure_memory_file(memory_file: str = MEMORY_FILE):
    if not os.path.exists(memory_file):
        pd.DataFrame(columns=["merchant_norm", "category", "type"]).to_csv(memory_file, index=False)


def load_memory(memory_file: str = MEMORY_FILE) -> pd.DataFrame:
    ensure_memory_file(memory_file)
    try:
        mem = pd.read_csv(memory_file)
        if mem.empty:
            return pd.DataFrame(columns=["merchant_norm", "category", "type"])
        mem["merchant_norm"] = mem["merchant_norm"].astype(str).str.strip().str.lower()
        mem["category"] = mem["category"].astype(str).str.strip()
        mem["type"] = mem["type"].astype(str).str.strip().str.lower()
        mem = mem.drop_duplicates(subset=["merchant_norm"], keep="last")
        return mem
    except Exception:
        return pd.DataFrame(columns=["merchant_norm", "category", "type"])


def save_memory_entry(merchant_norm: str, category: str, tx_type: str, memory_file: str = MEMORY_FILE):
    ensure_memory_file(memory_file)
    merchant_norm = str(merchant_norm).strip().lower()
    category = str(category).strip()
    tx_type = str(tx_type).strip().lower()

    mem = load_memory(memory_file)
    new_row = pd.DataFrame([{
        "merchant_norm": merchant_norm,
        "category": category,
        "type": tx_type
    }])

    mem = pd.concat([mem, new_row], ignore_index=True)
    mem = mem.drop_duplicates(subset=["merchant_norm"], keep="last")
    mem.to_csv(memory_file, index=False)


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
    text = re.sub(
        r"\b(upi|neft|imps|debit|credit|txn|ref|utr|via|to|from|paid|payment|purchase|sent|received)\b",
        " ",
        text,
    )
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


def _infer_type_from_row(row) -> str:
    amount = row.get("amount_raw", np.nan)
    desc = str(row.get("description", ""))

    if pd.notna(amount):
        if amount < 0:
            return "expense"
        if amount > 0 and any(k in desc.lower() for k in INCOME_KEYWORDS):
            return "income"

    return guess_type_from_text(desc)


def _finalize_transactions(df: pd.DataFrame, source_name: str) -> pd.DataFrame:
    out = df.copy()

    if "transaction_id" not in out.columns:
        out["transaction_id"] = ""
    out["transaction_id"] = out["transaction_id"].fillna("").astype(str).str.strip()

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
    out["source"] = source_name
    out["month"] = out["date"].dt.to_period("M").astype(str)
    out["day"] = out["date"].dt.day

    out = out.dropna(subset=["amount"]).copy()
    out = out[out["amount"] > 0].copy()
    out = out.sort_values("date", ascending=True, na_position="last").reset_index(drop=True)

    return out[STANDARD_COLUMNS]


def _standardize_dataframe(raw_df: pd.DataFrame, source_name: str) -> pd.DataFrame:
    df = raw_df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    date_col = _find_column(df.columns, ["date", "txn date", "transaction date", "posted date", "value date"])
    desc_col = _find_column(df.columns, ["description", "narration", "details", "remarks", "merchant", "transaction"])
    amount_col = _find_column(df.columns, ["amount", "amt"])
    type_col = _find_column(df.columns, ["type", "dr/cr", "transaction type", "nature"])
    credit_col = _find_column(df.columns, ["credit", "deposit"])
    debit_col = _find_column(df.columns, ["debit", "withdrawal", "spent"])
    transaction_id_col = _find_column(
        df.columns,
        ["transaction id", "txn id", "transactionid", "txn_ref", "reference", "utr", "ref no", "txn no", "id"]
    )

    standardized = pd.DataFrame()

    standardized["transaction_id"] = df[transaction_id_col].astype(str).str.strip() if transaction_id_col else ""
    standardized["date"] = df[date_col].apply(parse_date) if date_col else pd.NaT

    if amount_col:
        standardized["amount_raw"] = df[amount_col].apply(normalize_amount)
    elif credit_col or debit_col:
        credit_vals = df[credit_col].apply(normalize_amount) if credit_col else pd.Series(np.nan, index=df.index)
        debit_vals = df[debit_col].apply(normalize_amount) if debit_col else pd.Series(np.nan, index=df.index)
        standardized["amount_raw"] = credit_vals.fillna(0) - debit_vals.fillna(0)
    else:
        numeric_like_cols = [
            c for c in df.columns
            if df[c].astype(str).str.contains(r"\d", regex=True, na=False).mean() > 0.8
        ]
        if numeric_like_cols:
            standardized["amount_raw"] = df[numeric_like_cols[-1]].apply(normalize_amount)
        else:
            standardized["amount_raw"] = np.nan

    if desc_col:
        standardized["description"] = df[desc_col].astype(str)
    else:
        text_cols = [c for c in df.columns if c not in {date_col, amount_col, type_col, credit_col, debit_col, transaction_id_col}]
        standardized["description"] = df[text_cols].astype(str).agg(" | ".join, axis=1) if text_cols else "Transaction"

    if type_col:
        standardized["type"] = df[type_col].astype(str).str.lower().map(
            lambda x: "income" if any(k in x for k in ["cr", "credit", "income"]) else "expense"
        )
    else:
        standardized["type"] = standardized.apply(_infer_type_from_row, axis=1)

    return _finalize_transactions(standardized, source_name=source_name)


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df

    out = df.copy()

    has_txn_id = out["transaction_id"].fillna("").astype(str).str.strip() != ""
    with_id = out[has_txn_id].copy()
    without_id = out[~has_txn_id].copy()

    if not with_id.empty:
        with_id = with_id.drop_duplicates(subset=["transaction_id"], keep="first")

    if not without_id.empty:
        without_id = without_id.drop_duplicates(
            subset=["date", "amount", "merchant_norm", "type"],
            keep="first"
        )

    out = pd.concat([with_id, without_id], ignore_index=True)
    out = out.sort_values("date", ascending=True, na_position="last").reset_index(drop=True)
    return out


def apply_memory_learning(df: pd.DataFrame, memory_df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty or memory_df is None or memory_df.empty:
        if df is not None and not df.empty:
            df["learned_from_memory"] = False
        return df

    out = df.copy()
    out["merchant_norm"] = out["merchant_norm"].astype(str).str.lower().str.strip()
    out["learned_from_memory"] = False

    mem_map = memory_df.set_index("merchant_norm")[["category", "type"]].to_dict("index")

    for idx in out.index:
        merchant = out.at[idx, "merchant_norm"]
        if merchant in mem_map:
            out.at[idx, "category"] = mem_map[merchant]["category"]
            out.at[idx, "type"] = mem_map[merchant]["type"]
            out.at[idx, "learned_from_memory"] = True

    return out


def get_uncertain_transactions(df: pd.DataFrame, memory_df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    memory_merchants = set()
    if memory_df is not None and not memory_df.empty:
        memory_merchants = set(memory_df["merchant_norm"].astype(str).str.lower().str.strip())

    uncertain = df[
        (df["merchant_norm"].str.lower().str.strip().isin(memory_merchants) == False) &
        ((df["category"] == "Other") | (df["merchant_norm"].isin(["unknown", "investment", "electricity bill"])))
    ].copy()

    uncertain = uncertain.drop_duplicates(subset=["merchant_norm"], keep="first")
    return uncertain


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

    recurring_merchants = stats[
        (stats["month_count"] >= 2) & ((stats["cv"] <= 0.20) | (stats["tx_count"] >= 3))
    ]["merchant_norm"].tolist()

    out.loc[
        out["merchant_norm"].isin(recurring_merchants) & (out["type"] == "expense"),
        "is_recurring"
    ] = True

    return out


def _build_anomaly_reason(row, amount_threshold, rare_merchants):
    reasons = []
    if row["amount"] >= amount_threshold:
        reasons.append("high-value expense")
    if row["merchant_norm"] in rare_merchants:
        reasons.append("rare merchant")
    if row.get("is_recurring", False) is False and row["category"] in {"Shopping", "Travel", "Other", "Investment"}:
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

    if expense_df.empty:
        return out

    if IsolationForest is None or len(expense_df) < 5:
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


def load_csv(file) -> pd.DataFrame:
    df = pd.read_csv(file)
    return _standardize_dataframe(df, source_name="csv")


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

    txn_match = TRANSACTION_ID_REGEX.search(line)
    txn_id = txn_match.group(1).strip() if txn_match else ""

    return {
        "transaction_id": txn_id,
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

    if not rows:
        return pd.DataFrame(columns=STANDARD_COLUMNS)

    return _finalize_transactions(pd.DataFrame(rows), source_name="text")


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

    if not rows:
        return pd.DataFrame(columns=STANDARD_COLUMNS)

    return _finalize_transactions(pd.DataFrame(rows), source_name="pdf")


def extract_from_pdf(file) -> pd.DataFrame:
    all_rows = []

    for table_df in _extract_pdf_tables(file):
        try:
            standardized = _standardize_dataframe(table_df, source_name="pdf")
            if not standardized.empty:
                all_rows.append(standardized)
        except Exception:
            continue

    if all_rows:
        combined = pd.concat(all_rows, ignore_index=True)
        combined = combined.drop_duplicates().reset_index(drop=True)
        return combined

    text_blob = _extract_pdf_text(file)
    return _parse_transactions_from_blob(text_blob)


def combine_sources(dfs: List[pd.DataFrame]) -> pd.DataFrame:
    valid_dfs = [df.copy() for df in dfs if df is not None and not df.empty]
    if not valid_dfs:
        return pd.DataFrame(columns=STANDARD_COLUMNS)

    combined = pd.concat(valid_dfs, ignore_index=True)
    combined = combined.sort_values("date", ascending=True, na_position="last").reset_index(drop=True)
    return combined


def enrich_transactions(df: pd.DataFrame, memory_file: str = MEMORY_FILE) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=[
            "transaction_id", "date", "amount", "description", "type", "category",
            "merchant_norm", "source", "month", "day", "learned_from_memory",
            "is_recurring", "is_anomaly", "anomaly_score", "anomaly_reason"
        ])

    out = df.copy()
    before_count = len(out)

    out = remove_duplicates(out)
    after_count = len(out)

    memory_df = load_memory(memory_file)
    out = apply_memory_learning(out, memory_df)
    out = detect_recurring(out)
    out = detect_anomalies(out)

    out = out.sort_values("date", ascending=True, na_position="last").reset_index(drop=True)
    out.attrs["duplicates_removed"] = before_count - after_count
    return out
