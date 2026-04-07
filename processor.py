# processor.py

import os
import re
import json
import math
import hashlib
from datetime import datetime
from collections import defaultdict

import pandas as pd
import numpy as np

try:
    import pdfplumber
except Exception:
    pdfplumber = None

try:
    from sklearn.ensemble import IsolationForest
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False


# =========================================================
# CONFIG
# =========================================================

MEMORY_FILE = "learning_memory.json"

STANDARD_COLUMNS = [
    "date",
    "description",
    "merchant",
    "amount",
    "direction",
    "transaction_id",
    "category",
    "normalized_merchant",
    "source",
    "raw_text",
    "valid_row",
    "validation_reason",
    "parse_confidence",
    "is_duplicate",
    "duplicate_reason",
    "is_recurring",
    "recurring_group",
    "is_anomaly",
    "anomaly_reason",
]

NOISE_PATTERNS = [
    r"opening balance",
    r"closing balance",
    r"available balance",
    r"ledger balance",
    r"statement period",
    r"account summary",
    r"page\s+\d+",
    r"generated on",
    r"customer care",
    r"branch address",
    r"ifsc",
    r"customer id",
    r"account number",
    r"statement of account",
    r"total",
    r"subtotal",
]

CATEGORY_RULES = {
    "Groceries": [
        "grocery", "groceries", "supermarket", "mart", "bigbasket", "dmart",
        "reliance fresh", "kirana", "blinkit", "instamart"
    ],
    "Food & Dining": [
        "restaurant", "cafe", "swiggy", "zomato", "ubereats", "food", "pizza",
        "burger", "starbucks", "mcdonald", "dominos", "kfc"
    ],
    "Transport": [
        "uber", "ola", "rapido", "fuel", "petrol", "diesel", "metro", "bus",
        "taxi", "parking", "toll"
    ],
    "Shopping": [
        "amazon", "flipkart", "myntra", "ajio", "store", "retail", "mall"
    ],
    "Bills & Utilities": [
        "electricity", "water", "gas", "wifi", "internet", "broadband",
        "mobile bill", "recharge", "utility"
    ],
    "Rent": ["rent", "landlord", "lease"],
    "Salary": ["salary", "payroll", "wages", "stipend", "bonus"],
    "Transfer": ["upi", "imps", "neft", "rtgs", "transfer", "to self"],
    "Entertainment": ["netflix", "spotify", "movie", "bookmyshow", "prime", "hotstar"],
    "Healthcare": ["hospital", "clinic", "pharmacy", "medical", "doctor", "medicines"],
    "Investment": ["mutual fund", "sip", "zerodha", "groww", "upstox", "investment", "stocks"],
    "Cash Withdrawal": ["atm", "cash withdrawal"],
    "Education": ["course", "udemy", "coursera", "college", "university", "fees", "tuition"],
}

MERCHANT_CLEAN_PATTERNS = [
    r"\bupi\b",
    r"\bimps\b",
    r"\bneft\b",
    r"\brtgs\b",
    r"\bpos\b",
    r"\batm\b",
    r"\btxn\b",
    r"\btrf\b",
    r"\bref\b",
    r"\bpayment\b",
    r"\bpaid to\b",
    r"\btransferred to\b",
    r"\bvia\b",
    r"\bdebit\b",
    r"\bcredit\b",
    r"\bdr\b",
    r"\bcr\b",
    r"[/|*#:_\-]+",
]

INCOME_HINTS = [
    "salary", "refund", "cashback", "interest", "credited", "credit",
    "received", "deposit", "bonus", "reversal", "income"
]

EXPENSE_HINTS = [
    "debited", "debit", "spent", "purchase", "paid", "withdrawal",
    "dr", "expense", "bill"
]


# =========================================================
# UTILITIES
# =========================================================

def _to_str(x):
    if pd.isna(x):
        return ""
    return str(x).strip()


def _normalize_text(text):
    text = _to_str(text).lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _is_noise_line(text):
    txt = _normalize_text(text)
    if not txt:
        return True
    for pattern in NOISE_PATTERNS:
        if re.search(pattern, txt, flags=re.I):
            return True
    return False


def _safe_float(x):
    try:
        return float(x)
    except Exception:
        return None


def _parse_date(value):
    if value is None:
        return None

    value = _to_str(value)
    if not value:
        return None

    value = re.sub(r"\s+", " ", value).strip()

    try:
        dt = pd.to_datetime(value, dayfirst=True, errors="coerce")
        if pd.notna(dt):
            year = dt.year
            if 2000 <= year <= datetime.now().year + 1:
                return dt.date()
    except Exception:
        pass

    return None


def _extract_first_date(text):
    if not text:
        return None

    patterns = [
        r"\b\d{2}[/-]\d{2}[/-]\d{4}\b",
        r"\b\d{2}[/-]\d{2}[/-]\d{2}\b",
        r"\b\d{4}[/-]\d{2}[/-]\d{2}\b",
        r"\b\d{1,2}\s+[A-Za-z]{3,9}\s+\d{4}\b",
        r"\b[A-Za-z]{3,9}\s+\d{1,2}\s+\d{4}\b",
    ]

    for pat in patterns:
        m = re.search(pat, text)
        if m:
            dt = _parse_date(m.group(0))
            if dt:
                return dt
    return None


def _parse_amount(value):
    if value is None:
        return None

    s = _to_str(value)
    if not s:
        return None

    s = s.replace(",", "")
    s = s.replace("₹", "").replace("$", "").replace("Rs.", "").replace("Rs", "")
    s = s.strip()

    neg = False
    if s.startswith("(") and s.endswith(")"):
        neg = True
        s = s[1:-1]

    s = re.sub(r"[^\d\.\-]", "", s)
    if not s or s in {"-", ".", ""}:
        return None

    try:
        amt = float(s)
        if math.isnan(amt) or math.isinf(amt):
            return None
        return abs(amt)
    except Exception:
        return None


def _extract_amounts_from_text(text):
    if not text:
        return []

    cleaned = text.replace(",", "")
    pattern = r"(?<!\d)(?:₹|\$|Rs\.?\s*)?-?\(?\d{1,9}(?:\.\d{1,2})?\)?(?!\d)"
    amounts = []

    for m in re.finditer(pattern, cleaned, flags=re.I):
        amt = _parse_amount(m.group(0))
        if amt is not None and 0 < amt < 1e9:
            amounts.append(amt)

    return amounts


def _infer_direction(text="", debit=None, credit=None, amount=None):
    txt = f" {_normalize_text(text)} "

    if debit is not None and debit > 0 and (credit is None or credit == 0):
        return "expense"
    if credit is not None and credit > 0 and (debit is None or debit == 0):
        return "income"

    if " credited " in txt or " credit " in txt or " cr " in txt or " deposit " in txt or " received " in txt:
        return "income"
    if " debited " in txt or " debit " in txt or " dr " in txt or " paid " in txt or " purchase " in txt or " withdrawal " in txt:
        return "expense"

    if any(k in txt for k in INCOME_HINTS):
        return "income"
    if any(k in txt for k in EXPENSE_HINTS):
        return "expense"

    return "expense"


def _normalize_merchant(description):
    text = _normalize_text(description)
    text = re.sub(r"\b\d{4,}\b", " ", text)

    for pat in MERCHANT_CLEAN_PATTERNS:
        text = re.sub(pat, " ", text, flags=re.I)

    text = re.sub(r"[^a-zA-Z0-9\s&]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    if not text:
        return "Unknown"

    tokens = [t for t in text.split() if len(t) > 1]
    if not tokens:
        return "Unknown"

    return " ".join(tokens[:4]).title()


def _rule_based_category(description, direction):
    txt = _normalize_text(description)

    for category, keywords in CATEGORY_RULES.items():
        if any(k in txt for k in keywords):
            return category

    if direction == "income":
        return "Income"

    return "Other"


def _make_transaction_hash(date_val, amount, merchant, description):
    base = f"{date_val}|{float(amount):.2f}|{_normalize_merchant(merchant or description)}|{_normalize_text(description)[:40]}"
    return hashlib.md5(base.encode("utf-8")).hexdigest()[:16]


def _ensure_columns(df):
    if df is None or df.empty:
        df = pd.DataFrame(columns=STANDARD_COLUMNS)
    for col in STANDARD_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
    return df[STANDARD_COLUMNS].copy()


# =========================================================
# MEMORY
# =========================================================

def load_learning_memory(path=MEMORY_FILE):
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    return data
        except Exception:
            pass
    return {
        "merchant_category_map": {},
        "merchant_normalization_map": {},
    }


def save_learning_memory(memory, path=MEMORY_FILE):
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(memory, f, indent=2)
    except Exception:
        pass


def remember_category(merchant, category, path=MEMORY_FILE):
    memory = load_learning_memory(path)
    key = _normalize_text(merchant)
    if key:
        memory["merchant_category_map"][key] = category
        save_learning_memory(memory, path)


def remember_merchant_alias(raw_name, normalized_name, path=MEMORY_FILE):
    memory = load_learning_memory(path)
    key = _normalize_text(raw_name)
    if key:
        memory["merchant_normalization_map"][key] = normalized_name
        save_learning_memory(memory, path)


# =========================================================
# CSV PROCESSING
# =========================================================

def process_csv(file_obj):
    try:
        df = pd.read_csv(file_obj)
    except Exception:
        file_obj.seek(0)
        df = pd.read_excel(file_obj)

    lower_map = {c: _normalize_text(c) for c in df.columns}

    col_date = None
    col_desc = None
    col_amount = None
    col_debit = None
    col_credit = None
    col_txn_id = None

    for c, lc in lower_map.items():
        if col_date is None and any(x in lc for x in ["date", "txn date", "transaction date", "value date"]):
            col_date = c
        if col_desc is None and any(x in lc for x in ["description", "narration", "remarks", "details", "merchant"]):
            col_desc = c
        if col_amount is None and lc == "amount":
            col_amount = c
        if col_debit is None and any(x in lc for x in ["debit", "withdrawal", "dr"]):
            col_debit = c
        if col_credit is None and any(x in lc for x in ["credit", "deposit", "cr"]):
            col_credit = c
        if col_txn_id is None and any(x in lc for x in ["transaction id", "txn id", "reference", "ref no", "utr", "id"]):
            col_txn_id = c

    records = []

    for _, row in df.iterrows():
        desc = _to_str(row.get(col_desc, ""))
        raw_joined = " | ".join([_to_str(v) for v in row.values if _to_str(v)])

        debit = _parse_amount(row.get(col_debit)) if col_debit else None
        credit = _parse_amount(row.get(col_credit)) if col_credit else None

        amount = None
        direction = None

        if debit and debit > 0:
            amount = debit
            direction = "expense"
        elif credit and credit > 0:
            amount = credit
            direction = "income"
        else:
            amount = _parse_amount(row.get(col_amount)) if col_amount else None
            direction = _infer_direction(desc, debit=debit, credit=credit, amount=amount) if amount else None

        date_val = _parse_date(row.get(col_date)) if col_date else None
        txn_id = _to_str(row.get(col_txn_id)) if col_txn_id else ""

        records.append({
            "date": date_val,
            "description": desc,
            "merchant": desc,
            "amount": amount,
            "direction": direction,
            "transaction_id": txn_id,
            "source": "csv",
            "raw_text": raw_joined,
            "parse_confidence": 0.98,
        })

    df_out = pd.DataFrame(records)
    return validate_transactions(df_out)


# =========================================================
# PDF PROCESSING
# =========================================================

def _normalize_header_name(x):
    x = _normalize_text(x)
    x = re.sub(r"[^a-z0-9\s]", " ", x)
    x = re.sub(r"\s+", " ", x).strip()
    return x


def _map_pdf_table_columns(header_row):
    header_norm = [_normalize_header_name(h) for h in header_row]

    mapping = {
        "date": None,
        "description": None,
        "amount": None,
        "debit": None,
        "credit": None,
        "transaction_id": None,
    }

    for i, h in enumerate(header_norm):
        if mapping["date"] is None and any(k in h for k in ["date", "txn date", "value date"]):
            mapping["date"] = i
        if mapping["description"] is None and any(k in h for k in ["description", "narration", "remarks", "details", "merchant", "particular"]):
            mapping["description"] = i
        if mapping["amount"] is None and h == "amount":
            mapping["amount"] = i
        if mapping["debit"] is None and any(k in h for k in ["debit", "withdrawal", "dr"]):
            mapping["debit"] = i
        if mapping["credit"] is None and any(k in h for k in ["credit", "deposit", "cr"]):
            mapping["credit"] = i
        if mapping["transaction_id"] is None and any(k in h for k in ["transaction id", "txn id", "reference", "ref no", "utr", "id"]):
            mapping["transaction_id"] = i

    return mapping


def _parse_pdf_table_row(row, col_map):
    values = [_to_str(x) for x in row]
    joined = " | ".join([v for v in values if v]).strip()

    if not joined or _is_noise_line(joined):
        return None

    def pick(idx):
        if idx is None or idx >= len(values):
            return ""
        return values[idx]

    raw_date = pick(col_map["date"])
    desc = pick(col_map["description"])
    raw_amt = pick(col_map["amount"])
    raw_debit = pick(col_map["debit"])
    raw_credit = pick(col_map["credit"])
    raw_txn_id = pick(col_map["transaction_id"])

    debit = _parse_amount(raw_debit)
    credit = _parse_amount(raw_credit)

    amount = None
    direction = None

    if debit and debit > 0:
        amount = debit
        direction = "expense"
    elif credit and credit > 0:
        amount = credit
        direction = "income"
    else:
        amount = _parse_amount(raw_amt)
        if amount is not None:
            direction = _infer_direction(joined, debit=debit, credit=credit, amount=amount)

    date_val = _parse_date(raw_date) or _extract_first_date(joined)

    confidence = 0.65
    if date_val:
        confidence += 0.10
    if amount is not None:
        confidence += 0.10
    if desc:
        confidence += 0.05
    if raw_txn_id:
        confidence += 0.05

    return {
        "date": date_val,
        "description": desc or joined,
        "merchant": desc or joined,
        "amount": amount,
        "direction": direction,
        "transaction_id": raw_txn_id,
        "source": "pdf",
        "raw_text": joined,
        "parse_confidence": min(confidence, 0.95),
    }


def _parse_pdf_text_line(line):
    line = _to_str(line)
    if not line or _is_noise_line(line):
        return None

    date_val = _extract_first_date(line)
    amounts = _extract_amounts_from_text(line)

    if date_val is None or not amounts:
        return None

    amount = amounts[-1]

    desc = line
    desc = re.sub(r"\b\d{2}[/-]\d{2}[/-]\d{4}\b", " ", desc)
    desc = re.sub(r"\b\d{2}[/-]\d{2}[/-]\d{2}\b", " ", desc)
    desc = re.sub(r"\b\d{4}[/-]\d{2}[/-]\d{2}\b", " ", desc)
    desc = re.sub(r"\b\d{1,2}\s+[A-Za-z]{3,9}\s+\d{4}\b", " ", desc)
    desc = re.sub(r"\b[A-Za-z]{3,9}\s+\d{1,2}\s+\d{4}\b", " ", desc)

    amt_match = None
    for m in re.finditer(r"(?<!\d)(?:₹|\$|Rs\.?\s*)?-?\(?\d{1,9}(?:,\d{3})*(?:\.\d{1,2})?\)?(?!\d)", line):
        amt_match = m
    if amt_match:
        desc = (line[:amt_match.start()] + " " + line[amt_match.end():]).strip()

    desc = re.sub(r"\s+", " ", desc).strip()

    if len(desc) < 3:
        return None
    if re.fullmatch(r"[\d\s\W]+", desc):
        return None

    txn_id_match = re.search(r"\b(?:utr|ref|txn|transaction|id)[\s:.-]*([A-Za-z0-9\-]{5,})\b", line, flags=re.I)
    txn_id = txn_id_match.group(1) if txn_id_match else ""

    direction = _infer_direction(line, amount=amount)

    confidence = 0.58
    if date_val:
        confidence += 0.15
    if amount is not None:
        confidence += 0.15
    if len(desc) >= 6:
        confidence += 0.05
    if txn_id:
        confidence += 0.05

    return {
        "date": date_val,
        "description": desc,
        "merchant": desc,
        "amount": amount,
        "direction": direction,
        "transaction_id": txn_id,
        "source": "pdf",
        "raw_text": line,
        "parse_confidence": min(confidence, 0.93),
    }


def process_pdf(file_obj):
    if pdfplumber is None:
        return _ensure_columns(pd.DataFrame())

    records = []

    try:
        with pdfplumber.open(file_obj) as pdf:
            for page in pdf.pages:
                # Pass A: tables
                try:
                    tables = page.extract_tables()
                except Exception:
                    tables = []

                for table in tables or []:
                    if not table or len(table) < 2:
                        continue

                    header = table[0]
                    col_map = _map_pdf_table_columns(header)

                    for row in table[1:]:
                        if not row:
                            continue
                        parsed = _parse_pdf_table_row(row, col_map)
                        if parsed:
                            records.append(parsed)

                # Pass B: text fallback
                try:
                    page_text = page.extract_text() or ""
                except Exception:
                    page_text = ""

                if page_text:
                    for line in page_text.split("\n"):
                        parsed = _parse_pdf_text_line(line)
                        if parsed:
                            records.append(parsed)

    except Exception:
        return _ensure_columns(pd.DataFrame())

    if not records:
        return _ensure_columns(pd.DataFrame())

    df = pd.DataFrame(records)
    df = validate_transactions(df)
    return df


# =========================================================
# TEXT / SMS / EMAIL PROCESSING
# =========================================================

def process_text(text_input):
    if text_input is None:
        return _ensure_columns(pd.DataFrame())

    if isinstance(text_input, list):
        lines = []
        for item in text_input:
            lines.extend(str(item).splitlines())
    else:
        lines = str(text_input).splitlines()

    records = []

    for line in lines:
        line = _to_str(line)
        if not line or _is_noise_line(line):
            continue

        amounts = _extract_amounts_from_text(line)
        if not amounts:
            continue

        amount = amounts[-1]
        date_val = _extract_first_date(line)
        direction = _infer_direction(line, amount=amount)

        desc = line
        desc = re.sub(r"\b\d{2}[/-]\d{2}[/-]\d{4}\b", " ", desc)
        desc = re.sub(r"\b\d{2}[/-]\d{2}[/-]\d{2}\b", " ", desc)
        desc = re.sub(r"\b\d{4}[/-]\d{2}[/-]\d{2}\b", " ", desc)
        desc = re.sub(r"\s+", " ", desc).strip()

        txn_id_match = re.search(r"\b(?:utr|ref|txn|transaction|id)[\s:.-]*([A-Za-z0-9\-]{5,})\b", line, flags=re.I)
        txn_id = txn_id_match.group(1) if txn_id_match else ""

        records.append({
            "date": date_val,
            "description": desc,
            "merchant": desc,
            "amount": amount,
            "direction": direction,
            "transaction_id": txn_id,
            "source": "text",
            "raw_text": line,
            "parse_confidence": 0.88 if date_val else 0.72,
        })

    df = pd.DataFrame(records)
    return validate_transactions(df)


# =========================================================
# VALIDATION
# =========================================================

def validate_transactions(df):
    if df is None or df.empty:
        return _ensure_columns(pd.DataFrame())

    df = df.copy()

    for c in ["date", "description", "merchant", "amount", "direction", "transaction_id", "source", "raw_text", "parse_confidence"]:
        if c not in df.columns:
            df[c] = np.nan

    valid_flags = []
    reasons = []

    for _, row in df.iterrows():
        source = _to_str(row["source"]).lower()

        date_ok = pd.notna(row["date"])
        amount_val = _safe_float(row["amount"])
        amount_ok = amount_val is not None and amount_val > 0 and amount_val < 1e8

        desc = _to_str(row["description"])
        desc_ok = len(desc) >= 3 and not _is_noise_line(desc)

        direction_ok = _to_str(row["direction"]) in {"income", "expense"}
        conf = _safe_float(row["parse_confidence"])
        conf_ok = conf is not None and 0 <= conf <= 1.0

        structure_ok = True

        if source == "pdf":
            structure_ok = (
                date_ok and
                amount_ok and
                desc_ok and
                direction_ok and
                conf is not None and conf >= 0.60
            )
        elif source == "text":
            structure_ok = amount_ok and desc_ok and direction_ok
        elif source == "csv":
            structure_ok = date_ok and amount_ok and desc_ok and direction_ok

        valid = date_ok and amount_ok and desc_ok and direction_ok and conf_ok and structure_ok

        reason_parts = []
        if not date_ok:
            reason_parts.append("invalid_date")
        if not amount_ok:
            reason_parts.append("invalid_amount")
        if not desc_ok:
            reason_parts.append("invalid_description")
        if not direction_ok:
            reason_parts.append("invalid_direction")
        if not conf_ok:
            reason_parts.append("invalid_confidence")
        if not structure_ok:
            reason_parts.append("invalid_structure")

        valid_flags.append(valid)
        reasons.append("" if valid else ",".join(reason_parts))

    df["valid_row"] = valid_flags
    df["validation_reason"] = reasons

    return _ensure_columns(df)


def split_valid_and_rejected(df):
    df = _ensure_columns(df)
    valid_df = df[df["valid_row"] == True].copy()
    rejected_df = df[df["valid_row"] != True].copy()
    return valid_df, rejected_df


# =========================================================
# DEDUPLICATION
# =========================================================

def deduplicate_transactions(df):
    if df is None or df.empty:
        return _ensure_columns(pd.DataFrame())

    df = df.copy()
    df["is_duplicate"] = False
    df["duplicate_reason"] = ""

    seen_txn_ids = set()
    seen_fallback = set()

    for idx, row in df.iterrows():
        txn_id = _to_str(row["transaction_id"]).lower()
        date_val = row["date"]
        amount = _safe_float(row["amount"])
        desc = _to_str(row["description"])
        merchant = _to_str(row["merchant"])

        if txn_id and txn_id not in {"nan", "none"}:
            if txn_id in seen_txn_ids:
                df.at[idx, "is_duplicate"] = True
                df.at[idx, "duplicate_reason"] = "transaction_id"
                continue
            seen_txn_ids.add(txn_id)

        if pd.notna(date_val) and amount is not None:
            fallback_key = (
                str(date_val),
                round(amount, 2),
                _normalize_merchant(merchant or desc).lower(),
                _normalize_text(desc)[:30],
            )
            if fallback_key in seen_fallback:
                df.at[idx, "is_duplicate"] = True
                df.at[idx, "duplicate_reason"] = "fallback_logic"
                continue
            seen_fallback.add(fallback_key)

    deduped = df[df["is_duplicate"] != True].copy()
    duplicates_df = df[df["is_duplicate"] == True].copy()

    return _ensure_columns(deduped), _ensure_columns(duplicates_df)


# =========================================================
# ENRICHMENT
# =========================================================

def enrich_transactions(df, memory_path=MEMORY_FILE):
    if df is None or df.empty:
        return _ensure_columns(pd.DataFrame())

    memory = load_learning_memory(memory_path)
    cat_map = memory.get("merchant_category_map", {})
    merchant_map = memory.get("merchant_normalization_map", {})

    df = df.copy()

    normalized_merchants = []
    categories = []
    txn_ids = []

    for _, row in df.iterrows():
        desc = _to_str(row["description"])
        raw_merchant = _to_str(row["merchant"]) or desc
        merchant_key = _normalize_text(raw_merchant)

        normalized = merchant_map.get(merchant_key)
        if not normalized:
            normalized = _normalize_merchant(raw_merchant)

        category = cat_map.get(_normalize_text(normalized))
        if not category:
            category = _rule_based_category(desc, _to_str(row["direction"]))

        txn_id = _to_str(row["transaction_id"])
        if not txn_id:
            txn_id = _make_transaction_hash(row["date"], row["amount"], normalized, desc)

        normalized_merchants.append(normalized)
        categories.append(category)
        txn_ids.append(txn_id)

    df["normalized_merchant"] = normalized_merchants
    df["category"] = categories
    df["transaction_id"] = txn_ids

    return _ensure_columns(df)


# =========================================================
# RECURRING DETECTION
# =========================================================

def detect_recurring_transactions(df):
    if df is None or df.empty:
        return _ensure_columns(pd.DataFrame())

    df = df.copy()
    df["is_recurring"] = False
    df["recurring_group"] = ""

    work = df[df["direction"] == "expense"].copy()
    if work.empty:
        return _ensure_columns(df)

    work["date_dt"] = pd.to_datetime(work["date"], errors="coerce")
    work["year_month"] = work["date_dt"].dt.to_period("M")

    groups = work.groupby(["normalized_merchant", "category"])

    recurring_keys = set()
    for (merchant, category), group in groups:
        if len(group) < 2:
            continue

        months = group["year_month"].nunique()
        median_amt = group["amount"].median()
        within_band = group["amount"].between(median_amt * 0.80, median_amt * 1.20).sum()

        if months >= 2 and within_band >= 2:
            recurring_keys.add((merchant, category))

    for idx, row in df.iterrows():
        key = (_to_str(row["normalized_merchant"]), _to_str(row["category"]))
        if key in recurring_keys:
            df.at[idx, "is_recurring"] = True
            df.at[idx, "recurring_group"] = f"{key[0]} | {key[1]}"

    return _ensure_columns(df)


# =========================================================
# ANOMALY DETECTION
# =========================================================

def detect_anomalies(df):
    if df is None or df.empty:
        return _ensure_columns(pd.DataFrame())

    df = df.copy()
    df["is_anomaly"] = False
    df["anomaly_reason"] = ""

    work = df[df["direction"] == "expense"].copy()
    if work.empty:
        return _ensure_columns(df)

    work["date_dt"] = pd.to_datetime(work["date"], errors="coerce")
    work["day"] = work["date_dt"].dt.day.fillna(1)
    work["weekday"] = work["date_dt"].dt.weekday.fillna(0)
    work["month"] = work["date_dt"].dt.month.fillna(1)
    work["log_amount"] = np.log1p(work["amount"].astype(float))

    cat_freq = work["category"].value_counts(normalize=True).to_dict()
    merchant_freq = work["normalized_merchant"].value_counts(normalize=True).to_dict()

    work["cat_freq"] = work["category"].map(cat_freq).fillna(0.0)
    work["merchant_freq"] = work["normalized_merchant"].map(merchant_freq).fillna(0.0)

    if SKLEARN_AVAILABLE and len(work) >= 15:
        try:
            feats = work[["log_amount", "day", "weekday", "month", "cat_freq", "merchant_freq"]].fillna(0)
            contamination = min(max(0.03, 3 / len(work)), 0.10)

            model = IsolationForest(
                n_estimators=200,
                contamination=contamination,
                random_state=42
            )
            preds = model.fit_predict(feats)
            work["iforest_outlier"] = preds == -1
        except Exception:
            work["iforest_outlier"] = False
    else:
        work["iforest_outlier"] = False

    global_median = work["amount"].median()
    global_mad = np.median(np.abs(work["amount"] - global_median))
    if global_mad == 0:
        global_mad = max(global_median * 0.10, 1.0)

    flags = []
    reasons = []

    for _, row in work.iterrows():
        amt = float(row["amount"])
        merchant = row["normalized_merchant"]
        category = row["category"]

        cat_group = work[work["category"] == category]
        cat_med = cat_group["amount"].median() if not cat_group.empty else global_median
        cat_mad = np.median(np.abs(cat_group["amount"] - cat_med)) if len(cat_group) >= 3 else global_mad
        if cat_mad == 0:
            cat_mad = max(cat_med * 0.10, 1.0)

        global_z = abs(amt - global_median) / global_mad
        cat_z = abs(amt - cat_med) / cat_mad

        merchant_count = int((work["normalized_merchant"] == merchant).sum())
        first_time_high = merchant_count == 1 and amt >= max(global_median * 2.5, 1000)

        reason_list = []

        if bool(row["iforest_outlier"]):
            reason_list.append("flagged_by_isolation_forest")
        if global_z >= 6:
            reason_list.append("unusually_high_vs_overall_spending")
        if cat_z >= 5:
            reason_list.append("unusually_high_for_category")
        if first_time_high:
            reason_list.append("first_time_high_value_merchant")

        flagged = len(reason_list) > 0
        flags.append(flagged)
        reasons.append(", ".join(reason_list))

    work["is_anomaly"] = flags
    work["anomaly_reason"] = reasons

    df.loc[work.index, "is_anomaly"] = work["is_anomaly"]
    df.loc[work.index, "anomaly_reason"] = work["anomaly_reason"]

    return _ensure_columns(df)


# =========================================================
# BUDGET TRACKING
# =========================================================

def calculate_budget_tracking(df, budget_map=None):
    if budget_map is None:
        budget_map = {}

    if df is None or df.empty:
        return pd.DataFrame(columns=["category", "spent", "budget", "usage_pct", "status"])

    work = df[df["direction"] == "expense"].copy()
    if work.empty:
        return pd.DataFrame(columns=["category", "spent", "budget", "usage_pct", "status"])

    spent = work.groupby("category", dropna=False)["amount"].sum().reset_index()
    spent.columns = ["category", "spent"]

    records = []
    for _, row in spent.iterrows():
        category = row["category"]
        spent_amt = float(row["spent"])
        budget_amt = float(budget_map.get(category, 0)) if category in budget_map else 0.0

        if budget_amt > 0:
            usage_pct = (spent_amt / budget_amt) * 100
            if usage_pct <= 80:
                status = "Healthy"
            elif usage_pct <= 100:
                status = "Near Limit"
            else:
                status = "Over Budget"
        else:
            usage_pct = np.nan
            status = "No Budget Set"

        records.append({
            "category": category,
            "spent": round(spent_amt, 2),
            "budget": round(budget_amt, 2),
            "usage_pct": round(usage_pct, 2) if pd.notna(usage_pct) else np.nan,
            "status": status,
        })

    return pd.DataFrame(records).sort_values("spent", ascending=False)


# =========================================================
# FINANCIAL HEALTH SCORE
# =========================================================

def calculate_financial_health_score(df, budget_df=None):
    """
    Reliable score:
    - source weighted
    - parse confidence weighted
    - income missing does not unfairly destroy score
    - quality confidence exposed for jury/demo use
    """
    if df is None or df.empty:
        return {
            "health_score": 50,
            "score_band": "Insufficient Data",
            "quality_confidence": 0,
            "confidence_label": "Low",
            "components": {}
        }

    df = df.copy()

    total_rows = len(df)
    valid_rows = len(df)

    if valid_rows == 0:
        return {
            "health_score": 45,
            "score_band": "Low Confidence",
            "quality_confidence": 0,
            "confidence_label": "Low",
            "components": {
                "reason": "No valid transactions available"
            }
        }

    source_weights = {
        "csv": 1.00,
        "text": 0.85,
        "pdf": 0.65
    }

    df["source_weight"] = df["source"].astype(str).str.lower().map(source_weights).fillna(0.75)
    df["parse_confidence"] = pd.to_numeric(df["parse_confidence"], errors="coerce").fillna(0.70)
    df["txn_weight"] = df["source_weight"] * df["parse_confidence"]

    score_df = df[df["txn_weight"] >= 0.45].copy()

    if score_df.empty:
        return {
            "health_score": 50,
            "score_band": "Low Confidence",
            "quality_confidence": 0,
            "confidence_label": "Low",
            "components": {
                "reason": "Confidence too weak for reliable scoring"
            }
        }

    income_df = score_df[score_df["direction"] == "income"].copy()
    expense_df = score_df[score_df["direction"] == "expense"].copy()

    weighted_income = (income_df["amount"] * income_df["txn_weight"]).sum() if not income_df.empty else 0
    weighted_expense = (expense_df["amount"] * expense_df["txn_weight"]).sum() if not expense_df.empty else 0

    income_count = len(income_df)
    expense_count = len(expense_df)

    income_coverage_ok = (income_count >= 1 and weighted_income > 0)

    # 1. cashflow score
    if income_coverage_ok:
        savings_rate = max((weighted_income - weighted_expense) / max(weighted_income, 1), -1)
        cashflow_score = max(0, min(100, 50 + savings_rate * 100))
    else:
        cashflow_score = 60 if weighted_expense > 0 else 55

    # 2. budget score
    budget_score = 70
    if budget_df is not None and not budget_df.empty:
        valid_budget_rows = budget_df[pd.to_numeric(budget_df["budget"], errors="coerce").fillna(0) > 0].copy()
        if not valid_budget_rows.empty:
            over_budget_ratio = (valid_budget_rows["status"] == "Over Budget").mean()
            near_limit_ratio = (valid_budget_rows["status"] == "Near Limit").mean()
            budget_score = 100 - (over_budget_ratio * 45 + near_limit_ratio * 20)
            budget_score = max(0, min(100, budget_score))

    # 3. anomaly score
    anomaly_count = int((score_df["is_anomaly"] == True).sum())
    anomaly_ratio = anomaly_count / max(expense_count, 1)
    anomaly_score = max(0, 100 - anomaly_ratio * 250)

    # 4. quality score
    valid_ratio = valid_rows / total_rows if total_rows else 0
    avg_conf = score_df["txn_weight"].mean() if not score_df.empty else 0
    quality_score = ((valid_ratio * 0.5) + (avg_conf * 0.5)) * 100
    quality_score = max(0, min(100, quality_score))

    income_penalty = 0 if income_coverage_ok else 8

    raw_score = (
        0.40 * cashflow_score +
        0.25 * budget_score +
        0.20 * anomaly_score +
        0.15 * quality_score
    ) - income_penalty

    raw_score = max(0, min(100, raw_score))

    confidence_factor = min(max(avg_conf, 0.35), 1.0)
    stabilized_score = (raw_score * confidence_factor) + (60 * (1 - confidence_factor))
    final_score = int(round(max(0, min(100, stabilized_score))))

    if final_score >= 80:
        band = "Strong"
    elif final_score >= 65:
        band = "Stable"
    elif final_score >= 50:
        band = "Moderate"
    else:
        band = "At Risk"

    quality_confidence = round(quality_score, 1)
    if quality_confidence >= 80:
        confidence_label = "High"
    elif quality_confidence >= 60:
        confidence_label = "Medium"
    else:
        confidence_label = "Low"

    return {
        "health_score": final_score,
        "score_band": band,
        "quality_confidence": quality_confidence,
        "confidence_label": confidence_label,
        "components": {
            "cashflow_score": round(cashflow_score, 1),
            "budget_score": round(budget_score, 1),
            "anomaly_score": round(anomaly_score, 1),
            "data_quality_score": round(quality_score, 1),
            "weighted_income": round(float(weighted_income), 2),
            "weighted_expense": round(float(weighted_expense), 2),
            "anomaly_count": anomaly_count,
            "valid_transaction_count": int(valid_rows),
            "score_transaction_count": int(len(score_df)),
            "total_parsed_rows": int(total_rows),
            "income_detected": bool(income_coverage_ok),
            "income_reliability_penalty": income_penalty,
        }
    }


# =========================================================
# INSIGHTS
# =========================================================

def generate_insights(df, budget_df=None, health=None):
    insights = []

    if df is None or df.empty:
        return ["No valid transactions available."]

    expense_df = df[df["direction"] == "expense"].copy()
    income_df = df[df["direction"] == "income"].copy()

    if not expense_df.empty:
        top_cat = expense_df.groupby("category")["amount"].sum().sort_values(ascending=False)
        if not top_cat.empty:
            insights.append(f"Highest spending category: {top_cat.index[0]} ({top_cat.iloc[0]:.2f}).")

        top_merchant = expense_df.groupby("normalized_merchant")["amount"].sum().sort_values(ascending=False)
        if not top_merchant.empty:
            insights.append(f"Top merchant by spend: {top_merchant.index[0]} ({top_merchant.iloc[0]:.2f}).")

    if not income_df.empty:
        total_income = income_df["amount"].sum()
        total_expense = expense_df["amount"].sum() if not expense_df.empty else 0
        savings = total_income - total_expense
        insights.append(f"Estimated savings after expenses: {savings:.2f}.")

    anomalies = df[df["is_anomaly"] == True]
    if not anomalies.empty:
        insights.append(f"{len(anomalies)} potentially anomalous expense(s) detected.")

    recurring = df[df["is_recurring"] == True]
    if not recurring.empty:
        insights.append(f"{recurring['normalized_merchant'].nunique()} recurring spend pattern(s) detected.")

    if budget_df is not None and not budget_df.empty:
        over_budget = budget_df[budget_df["status"] == "Over Budget"]
        if not over_budget.empty:
            cats = ", ".join(over_budget["category"].astype(str).tolist()[:3])
            insights.append(f"Over budget in: {cats}.")

    if health:
        insights.append(
            f"Financial Health Score: {health.get('health_score', 0)}/100 "
            f"({health.get('score_band', 'Unknown')}) "
            f"| Confidence: {health.get('confidence_label', 'Low')} "
            f"({health.get('quality_confidence', 0)}%)."
        )

    return insights


# =========================================================
# DATA QUALITY SUMMARY
# =========================================================

def generate_data_quality_summary(parsed_df, valid_df, rejected_df, duplicates_df):
    parsed_df = _ensure_columns(parsed_df)
    valid_df = _ensure_columns(valid_df)
    rejected_df = _ensure_columns(rejected_df)
    duplicates_df = _ensure_columns(duplicates_df)

    total_parsed = len(parsed_df)
    total_valid = len(valid_df)
    total_rejected = len(rejected_df)
    total_duplicates = len(duplicates_df)

    source_counts = parsed_df["source"].fillna("unknown").value_counts().to_dict() if not parsed_df.empty else {}
    valid_source_counts = valid_df["source"].fillna("unknown").value_counts().to_dict() if not valid_df.empty else {}

    avg_conf = round(pd.to_numeric(valid_df["parse_confidence"], errors="coerce").fillna(0).mean() * 100, 1) if not valid_df.empty else 0

    summary = {
        "total_parsed_rows": int(total_parsed),
        "valid_rows_accepted": int(total_valid),
        "rejected_rows": int(total_rejected),
        "duplicates_removed": int(total_duplicates),
        "acceptance_rate_pct": round((total_valid / total_parsed) * 100, 1) if total_parsed else 0,
        "average_valid_confidence_pct": avg_conf,
        "parsed_source_mix": source_counts,
        "valid_source_mix": valid_source_counts,
    }

    return summary


# =========================================================
# MASTER PIPELINE
# =========================================================

def process_all_inputs(csv_files=None, pdf_files=None, text_inputs=None, budget_map=None, memory_path=MEMORY_FILE):
    csv_files = csv_files or []
    pdf_files = pdf_files or []
    text_inputs = text_inputs or []

    all_parts = []

    # CSV
    for f in csv_files:
        try:
            f.seek(0)
        except Exception:
            pass
        part = process_csv(f)
        if not part.empty:
            all_parts.append(part)

    # PDF
    for f in pdf_files:
        try:
            f.seek(0)
        except Exception:
            pass
        part = process_pdf(f)
        if not part.empty:
            all_parts.append(part)

    # TEXT
    if isinstance(text_inputs, str):
        text_inputs = [text_inputs]

    for txt in text_inputs:
        part = process_text(txt)
        if not part.empty:
            all_parts.append(part)

    if not all_parts:
        empty_df = _ensure_columns(pd.DataFrame())
        return {
            "transactions": empty_df,
            "budget_tracking": pd.DataFrame(),
            "health": calculate_financial_health_score(empty_df, pd.DataFrame()),
            "insights": ["No data was processed."],
            "rejected_rows": empty_df,
            "duplicate_rows": empty_df,
            "data_quality_summary": {
                "total_parsed_rows": 0,
                "valid_rows_accepted": 0,
                "rejected_rows": 0,
                "duplicates_removed": 0,
                "acceptance_rate_pct": 0,
                "average_valid_confidence_pct": 0,
                "parsed_source_mix": {},
                "valid_source_mix": {},
            }
        }

    parsed_df = pd.concat(all_parts, ignore_index=True)
    parsed_df = _ensure_columns(parsed_df)

    valid_df, rejected_df = split_valid_and_rejected(parsed_df)
    valid_df, duplicates_df = deduplicate_transactions(valid_df)
    valid_df = enrich_transactions(valid_df, memory_path=memory_path)
    valid_df = detect_recurring_transactions(valid_df)
    valid_df = detect_anomalies(valid_df)

    if not valid_df.empty:
        valid_df["date"] = pd.to_datetime(valid_df["date"], errors="coerce").dt.date
        valid_df = valid_df.sort_values(["date", "amount"], ascending=[False, False]).reset_index(drop=True)

    budget_df = calculate_budget_tracking(valid_df, budget_map=budget_map or {})
    health = calculate_financial_health_score(valid_df, budget_df)
    insights = generate_insights(valid_df, budget_df, health)
    data_quality_summary = generate_data_quality_summary(parsed_df, valid_df, rejected_df, duplicates_df)

    return {
        "transactions": _ensure_columns(valid_df),
        "budget_tracking": budget_df,
        "health": health,
        "insights": insights,
        "rejected_rows": _ensure_columns(rejected_df),
        "duplicate_rows": _ensure_columns(duplicates_df),
        "data_quality_summary": data_quality_summary,
    }


# =========================================================
# COMPATIBILITY WRAPPERS
# =========================================================

def process_financial_data(csv_files=None, pdf_files=None, text_inputs=None, budget_map=None, memory_path=MEMORY_FILE):
    return process_all_inputs(
        csv_files=csv_files,
        pdf_files=pdf_files,
        text_inputs=text_inputs,
        budget_map=budget_map,
        memory_path=memory_path,
    )


def run_pipeline(csv_files=None, pdf_files=None, text_inputs=None, budget_map=None, memory_path=MEMORY_FILE):
    return process_all_inputs(
        csv_files=csv_files,
        pdf_files=pdf_files,
        text_inputs=text_inputs,
        budget_map=budget_map,
        memory_path=memory_path,
    )


def parse_csv(file_obj):
    return process_csv(file_obj)


def parse_pdf(file_obj):
    return process_pdf(file_obj)


def parse_text(text_input):
    return process_text(text_input)
