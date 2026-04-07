# processor.py

import os
import re
import io
import json
import math
import hashlib
from datetime import datetime
from collections import Counter, defaultdict

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
    "direction",          # income / expense
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
    r"page\s+\d+",
    r"generated on",
    r"customer care",
    r"account summary",
    r"total",
    r"subtotal",
    r"debit",
    r"credit",
    r"withdrawal",
    r"deposit",
    r"transaction details",
]

DATE_PATTERNS = [
    "%d/%m/%Y",
    "%d-%m-%Y",
    "%d.%m.%Y",
    "%Y-%m-%d",
    "%d/%m/%y",
    "%d-%m-%y",
    "%m/%d/%Y",
    "%m-%d-%Y",
    "%d %b %Y",
    "%d %B %Y",
    "%b %d %Y",
    "%B %d %Y",
]

CATEGORY_RULES = {
    "Groceries": [
        "grocery", "groceries", "supermarket", "mart", "bigbasket", "dmart",
        "reliance fresh", "kirana", "blinkit", "instamart"
    ],
    "Food & Dining": [
        "restaurant", "cafe", "swiggy", "zomato", "ubereats", "food", "eat",
        "pizza", "burger", "starbucks", "mcdonald", "dominos", "kfc"
    ],
    "Transport": [
        "uber", "ola", "rapido", "fuel", "petrol", "diesel", "metro", "bus",
        "taxi", "parking", "toll"
    ],
    "Shopping": [
        "amazon", "flipkart", "myntra", "ajio", "mall", "store", "retail"
    ],
    "Bills & Utilities": [
        "electricity", "water", "gas", "wifi", "internet", "mobile bill",
        "recharge", "broadband", "utility"
    ],
    "Rent": [
        "rent", "landlord", "lease"
    ],
    "Salary": [
        "salary", "payroll", "wages", "stipend", "bonus"
    ],
    "Transfer": [
        "upi", "imps", "neft", "rtgs", "transfer", "bank transfer", "to self"
    ],
    "Entertainment": [
        "netflix", "spotify", "movie", "bookmyshow", "prime", "hotstar", "game"
    ],
    "Healthcare": [
        "hospital", "clinic", "pharmacy", "medical", "doctor", "medicines"
    ],
    "Investment": [
        "mutual fund", "sip", "zerodha", "groww", "upstox", "investment", "stocks"
    ],
    "Cash Withdrawal": [
        "atm", "cash withdrawal"
    ],
    "Education": [
        "course", "udemy", "coursera", "college", "university", "fees", "tuition"
    ],
}

INCOME_HINTS = [
    "salary", "refund", "cashback", "interest", "credit", "received", "deposit",
    "bonus", "reversal", "income", "payment received"
]

EXPENSE_HINTS = [
    "debit", "spent", "purchase", "paid", "bill", "withdrawal", "dr", "expense"
]

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


# =========================================================
# BASIC UTILITIES
# =========================================================

def _ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    for col in STANDARD_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
    return df[STANDARD_COLUMNS].copy()


def _to_str(x):
    if pd.isna(x):
        return ""
    return str(x).strip()


def _normalize_text(text: str) -> str:
    text = _to_str(text).lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _is_noise_line(text: str) -> bool:
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

    # Common cleanup
    value = value.replace(".", "/").replace("-", "/")

    # Try pandas first
    try:
        dt = pd.to_datetime(value, dayfirst=True, errors="coerce")
        if pd.notna(dt):
            year = dt.year
            if 2000 <= year <= datetime.now().year + 1:
                return dt.date()
    except Exception:
        pass

    # Manual fallback
    for fmt in DATE_PATTERNS:
        try:
            dt = datetime.strptime(value, fmt)
            if 2000 <= dt.year <= datetime.now().year + 1:
                return dt.date()
        except Exception:
            continue

    return None


def _extract_first_date(text: str):
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
    """
    Conservative finance amount parser.
    Returns positive float magnitude only.
    Direction is inferred separately.
    """
    if value is None:
        return None

    s = _to_str(value)
    if not s:
        return None

    s = s.replace(",", "")
    s = s.replace("₹", "").replace("$", "").replace("Rs.", "").replace("Rs", "")

    # bracket negative style: (123.45)
    s = s.strip()
    s = re.sub(r"[^\d\.\-\(\)]", "", s)

    if not s or s in {"-", ".", "()", "(.)"}:
        return None

    neg = False
    if s.startswith("(") and s.endswith(")"):
        neg = True
        s = s[1:-1]

    try:
        amt = float(s)
        if math.isnan(amt) or math.isinf(amt):
            return None
        return abs(amt)
    except Exception:
        return None


def _extract_amounts_from_text(text: str):
    """
    Returns list of plausible amount tokens as positive floats.
    Conservative extraction.
    """
    if not text:
        return []

    cleaned = text.replace(",", "")
    patterns = [
        r"(?<!\d)(?:₹|\$|Rs\.?\s*)?-?\(?\d{1,9}(?:\.\d{1,2})?\)?(?!\d)"
    ]

    amounts = []
    for pat in patterns:
        for m in re.finditer(pat, cleaned, flags=re.I):
            amt = _parse_amount(m.group(0))
            if amt is not None and 0 < amt < 1e9:
                amounts.append(amt)

    return amounts


def _infer_direction(text="", debit=None, credit=None, amount=None):
    txt = _normalize_text(text)

    if debit is not None and debit > 0 and (credit is None or credit == 0):
        return "expense"
    if credit is not None and credit > 0 and (debit is None or debit == 0):
        return "income"

    if any(word in txt for word in [" cr ", "credit", "credited", "deposit", "received"]):
        return "income"
    if any(word in txt for word in [" dr ", "debit", "debited", "spent", "purchase", "withdrawal", "paid"]):
        return "expense"

    # fallback by text hints
    if any(k in txt for k in INCOME_HINTS):
        return "income"
    if any(k in txt for k in EXPENSE_HINTS):
        return "expense"

    # final fallback: default to expense because most transactions are expense-like
    return "expense"


def _normalize_merchant(description: str):
    text = _normalize_text(description)

    # remove long numeric refs
    text = re.sub(r"\b\d{4,}\b", " ", text)

    for pat in MERCHANT_CLEAN_PATTERNS:
        text = re.sub(pat, " ", text, flags=re.I)

    text = re.sub(r"[^a-zA-Z0-9\s&]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    if not text:
        return "unknown"

    # take first 2-4 useful tokens
    tokens = [t for t in text.split() if len(t) > 1]
    if not tokens:
        return "unknown"

    return " ".join(tokens[:4]).title()


def _rule_based_category(description: str, direction: str):
    txt = _normalize_text(description)

    for category, keywords in CATEGORY_RULES.items():
        if any(k in txt for k in keywords):
            return category

    if direction == "income":
        return "Income"

    return "Other"


def _make_transaction_hash(date_val, amount, merchant, description):
    base = f"{date_val}|{amount:.2f}|{_normalize_merchant(merchant or description)}|{_normalize_text(description)[:40]}"
    return hashlib.md5(base.encode("utf-8")).hexdigest()[:16]


# =========================================================
# LEARNING MEMORY
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
    merchant_key = _normalize_text(merchant)
    if merchant_key:
        memory["merchant_category_map"][merchant_key] = category
        save_learning_memory(memory, path)


def remember_merchant_alias(raw_name, normalized_name, path=MEMORY_FILE):
    memory = load_learning_memory(path)
    raw_key = _normalize_text(raw_name)
    if raw_key:
        memory["merchant_normalization_map"][raw_key] = normalized_name
        save_learning_memory(memory, path)


# =========================================================
# CSV PARSING
# =========================================================

def process_csv(file_obj) -> pd.DataFrame:
    try:
        df = pd.read_csv(file_obj)
    except Exception:
        file_obj.seek(0)
        df = pd.read_excel(file_obj)

    original_cols = list(df.columns)
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
        if col_amount is None and "amount" in lc:
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
        raw_joined = " | ".join([_to_str(row[c]) for c in original_cols if _to_str(row[c])])

        debit = _parse_amount(row.get(col_debit)) if col_debit else None
        credit = _parse_amount(row.get(col_credit)) if col_credit else None

        amt = None
        direction = None

        if debit and debit > 0:
            amt = debit
            direction = "expense"
        elif credit and credit > 0:
            amt = credit
            direction = "income"
        else:
            amt = _parse_amount(row.get(col_amount)) if col_amount else None
            direction = _infer_direction(desc, debit=debit, credit=credit, amount=amt) if amt else None

        dt = _parse_date(row.get(col_date)) if col_date else None
        txn_id = _to_str(row.get(col_txn_id)) if col_txn_id else ""

        records.append({
            "date": dt,
            "description": desc,
            "merchant": desc,
            "amount": amt,
            "direction": direction,
            "transaction_id": txn_id,
            "source": "csv",
            "raw_text": raw_joined,
            "parse_confidence": 0.95,
        })

    out = pd.DataFrame(records)
    return validate_transactions(out)


# =========================================================
# PDF PARSING
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

    dt = _parse_date(raw_date) or _extract_first_date(joined)

    confidence = 0.70
    if dt:
        confidence += 0.10
    if amount is not None:
        confidence += 0.10
    if desc:
        confidence += 0.05
    if raw_txn_id:
        confidence += 0.05

    return {
        "date": dt,
        "description": desc or joined,
        "merchant": desc or joined,
        "amount": amount,
        "direction": direction,
        "transaction_id": raw_txn_id,
        "source": "pdf",
        "raw_text": joined,
        "parse_confidence": min(confidence, 0.98),
    }


def _parse_pdf_text_line(line):
    line = _to_str(line)
    if not line or _is_noise_line(line):
        return None

    # Must contain at least a date and one amount to be considered
    dt = _extract_first_date(line)
    amounts = _extract_amounts_from_text(line)

    if dt is None or not amounts:
        return None

    # pick the last amount as the transaction amount
    amount = amounts[-1]

    # remove date and trailing amount from description
    desc = line
    desc = re.sub(r"\b\d{2}[/-]\d{2}[/-]\d{4}\b", " ", desc)
    desc = re.sub(r"\b\d{2}[/-]\d{2}[/-]\d{2}\b", " ", desc)
    desc = re.sub(r"\b\d{4}[/-]\d{2}[/-]\d{2}\b", " ", desc)
    desc = re.sub(r"\b\d{1,2}\s+[A-Za-z]{3,9}\s+\d{4}\b", " ", desc)

    # remove last amount occurrence only
    amt_match = None
    for m in re.finditer(r"(?<!\d)(?:₹|\$|Rs\.?\s*)?-?\(?\d{1,9}(?:,\d{3})*(?:\.\d{1,2})?\)?(?!\d)", line):
        amt_match = m
    if amt_match:
        desc = (line[:amt_match.start()] + " " + line[amt_match.end():]).strip()

    desc = re.sub(r"\s+", " ", desc).strip()

    # Reject weak lines aggressively
    if len(desc) < 3:
        return None
    if re.fullmatch(r"[\d\s\W]+", desc):
        return None

    direction = _infer_direction(line, amount=amount)

    # crude transaction id extraction
    txn_id_match = re.search(r"\b(?:utr|ref|txn|transaction|id)[\s:.-]*([A-Za-z0-9\-]{5,})\b", line, flags=re.I)
    txn_id = txn_id_match.group(1) if txn_id_match else ""

    confidence = 0.60
    if dt:
        confidence += 0.15
    if amount is not None:
        confidence += 0.15
    if len(desc) >= 6:
        confidence += 0.05
    if txn_id:
        confidence += 0.05

    return {
        "date": dt,
        "description": desc,
        "merchant": desc,
        "amount": amount,
        "direction": direction,
        "transaction_id": txn_id,
        "source": "pdf",
        "raw_text": line,
        "parse_confidence": min(confidence, 0.95),
    }


def process_pdf(file_obj) -> pd.DataFrame:
    if pdfplumber is None:
        return _ensure_columns(pd.DataFrame())

    records = []

    try:
        with pdfplumber.open(file_obj) as pdf:
            for page in pdf.pages:
                # ---------- PASS A: table extraction ----------
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

                # ---------- PASS B: text fallback ----------
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

    # Strong validation gate
    df = validate_transactions(df)

    # Keep only strong PDF rows
    if not df.empty:
        df = df[
            (df["valid_row"] == True) &
            (df["parse_confidence"].fillna(0) >= 0.65)
        ].copy()

    return _ensure_columns(df)


# =========================================================
# TEXT / SMS / EMAIL PARSING
# =========================================================

def process_text(text_input) -> pd.DataFrame:
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

        dt = _extract_first_date(line)
        amounts = _extract_amounts_from_text(line)

        if not amounts:
            continue

        amount = amounts[-1]
        direction = _infer_direction(line, amount=amount)

        desc = line
        if dt:
            desc = re.sub(r"\b\d{2}[/-]\d{2}[/-]\d{4}\b", " ", desc)
            desc = re.sub(r"\b\d{2}[/-]\d{2}[/-]\d{2}\b", " ", desc)
            desc = re.sub(r"\b\d{4}[/-]\d{2}[/-]\d{2}\b", " ", desc)

        amt_match = None
        for m in re.finditer(r"(?<!\d)(?:₹|\$|Rs\.?\s*)?-?\(?\d{1,9}(?:,\d{3})*(?:\.\d{1,2})?\)?(?!\d)", line):
            amt_match = m
        if amt_match:
            desc = (line[:amt_match.start()] + " " + line[amt_match.end():]).strip()

        desc = re.sub(r"\s+", " ", desc).strip()

        txn_id_match = re.search(r"\b(?:utr|ref|txn|transaction|id)[\s:.-]*([A-Za-z0-9\-]{5,})\b", line, flags=re.I)
        txn_id = txn_id_match.group(1) if txn_id_match else ""

        records.append({
            "date": dt,
            "description": desc if desc else line,
            "merchant": desc if desc else line,
            "amount": amount,
            "direction": direction,
            "transaction_id": txn_id,
            "source": "text",
            "raw_text": line,
            "parse_confidence": 0.85 if dt else 0.70,
        })

    df = pd.DataFrame(records)
    return validate_transactions(df)


# =========================================================
# VALIDATION
# =========================================================

def validate_transactions(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return _ensure_columns(pd.DataFrame())

    df = df.copy()

    required_cols = ["date", "description", "amount", "direction", "source", "raw_text", "parse_confidence"]
    for c in required_cols:
        if c not in df.columns:
            df[c] = np.nan

    valid_flags = []
    reasons = []

    for _, row in df.iterrows():
        date_ok = pd.notna(row["date"])
        amt = _safe_float(row["amount"])
        amount_ok = amt is not None and amt > 0 and amt < 1e8

        desc = _to_str(row["description"])
        desc_ok = len(desc) >= 3 and not _is_noise_line(desc)

        direction_ok = _to_str(row["direction"]) in {"income", "expense"}

        structure_ok = True
        source = _to_str(row["source"]).lower()

        # extra strictness for pdf
        if source == "pdf":
            # require higher structure quality for PDF lines
            structure_ok = (
                pd.notna(row["date"]) and
                amount_ok and
                len(desc) >= 4 and
                row.get("parse_confidence", 0) >= 0.60
            )

        valid = date_ok and amount_ok and desc_ok and direction_ok and structure_ok

        reason_parts = []
        if not date_ok:
            reason_parts.append("invalid_date")
        if not amount_ok:
            reason_parts.append("invalid_amount")
        if not desc_ok:
            reason_parts.append("invalid_description")
        if not direction_ok:
            reason_parts.append("invalid_direction")
        if not structure_ok:
            reason_parts.append("invalid_structure")

        valid_flags.append(valid)
        reasons.append("" if valid else ",".join(reason_parts))

    df["valid_row"] = valid_flags
    df["validation_reason"] = reasons

    # keep all rows in frame, but downstream should use valid_row only
    return _ensure_columns(df)


# =========================================================
# DEDUPLICATION
# =========================================================

def deduplicate_transactions(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return _ensure_columns(pd.DataFrame())

    df = df.copy()
    df["is_duplicate"] = False
    df["duplicate_reason"] = ""

    # Work only on valid rows
    valid_idx = df[df["valid_row"] == True].index.tolist()
    seen_txn_ids = set()
    seen_fallback = set()

    for idx in valid_idx:
        row = df.loc[idx]
        txn_id = _to_str(row["transaction_id"]).lower()
        date_val = row["date"]
        amount = float(row["amount"]) if pd.notna(row["amount"]) else None
        merchant = _to_str(row["merchant"])
        desc = _to_str(row["description"])

        if txn_id and txn_id not in {"nan", "none"}:
            if txn_id in seen_txn_ids:
                df.at[idx, "is_duplicate"] = True
                df.at[idx, "duplicate_reason"] = "transaction_id"
                continue
            seen_txn_ids.add(txn_id)

        if date_val and amount is not None:
            fallback_key = (
                str(date_val),
                round(float(amount), 2),
                _normalize_merchant(merchant or desc).lower(),
                _normalize_text(desc)[:30],
            )
            if fallback_key in seen_fallback:
                df.at[idx, "is_duplicate"] = True
                df.at[idx, "duplicate_reason"] = "fallback_logic"
                continue
            seen_fallback.add(fallback_key)

    df = df[df["is_duplicate"] != True].copy()
    return _ensure_columns(df)


# =========================================================
# ENRICHMENT
# =========================================================

def enrich_transactions(df: pd.DataFrame, memory_path=MEMORY_FILE) -> pd.DataFrame:
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
            txn_id = _make_transaction_hash(row["date"], float(row["amount"]), normalized, desc)

        normalized_merchants.append(normalized)
        categories.append(category)
        txn_ids.append(txn_id)

    df["normalized_merchant"] = normalized_merchants
    df["category"] = categories
    df["transaction_id"] = txn_ids

    return _ensure_columns(df)


# =========================================================
# RECURRING TRANSACTIONS
# =========================================================

def detect_recurring_transactions(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return _ensure_columns(pd.DataFrame())

    df = df.copy()
    df["is_recurring"] = False
    df["recurring_group"] = ""

    valid = df[(df["valid_row"] == True) & (df["direction"] == "expense")].copy()
    if valid.empty:
        return _ensure_columns(df)

    valid["year_month"] = pd.to_datetime(valid["date"]).dt.to_period("M")
    valid["amount_round"] = valid["amount"].round(0)

    groups = valid.groupby(["normalized_merchant", "category"])
    recurring_keys = set()

    for (merchant, category), group in groups:
        if len(group) < 2:
            continue

        months = group["year_month"].nunique()
        median_amt = group["amount"].median()

        # amount consistency
        within_band = group["amount"].between(median_amt * 0.8, median_amt * 1.2).sum()

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

def detect_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    """
    Finance-aware anomaly detection:
    1. Isolation Forest on valid expenses (if available)
    2. Robust fallback using category / global amount deviation
    3. First-time high-value merchant flag
    """
    if df is None or df.empty:
        return _ensure_columns(pd.DataFrame())

    df = df.copy()
    df["is_anomaly"] = False
    df["anomaly_reason"] = ""

    work = df[(df["valid_row"] == True) & (df["direction"] == "expense")].copy()
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

    # -------- Isolation Forest --------
    if SKLEARN_AVAILABLE and len(work) >= 15:
        feats = work[["log_amount", "day", "weekday", "month", "cat_freq", "merchant_freq"]].fillna(0)

        try:
            contamination = min(max(0.03, 3 / len(work)), 0.10)
            model = IsolationForest(
                n_estimators=200,
                contamination=contamination,
                random_state=42
            )
            preds = model.fit_predict(feats)
            scores = model.decision_function(feats)

            work["iforest_outlier"] = preds == -1
            work["iforest_score"] = scores
        except Exception:
            work["iforest_outlier"] = False
            work["iforest_score"] = 0.0
    else:
        work["iforest_outlier"] = False
        work["iforest_score"] = 0.0

    # -------- Robust finance fallback --------
    global_median = work["amount"].median()
    global_mad = np.median(np.abs(work["amount"] - global_median))
    if global_mad == 0:
        global_mad = max(global_median * 0.10, 1.0)

    anomaly_flags = []
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

        flagged = False
        reason_list = []

        if bool(row["iforest_outlier"]):
            flagged = True
            reason_list.append("isolation_forest")

        if global_z >= 6:
            flagged = True
            reason_list.append("global_amount_deviation")

        if cat_z >= 5:
            flagged = True
            reason_list.append("category_amount_deviation")

        if first_time_high:
            flagged = True
            reason_list.append("first_time_high_value_merchant")

        anomaly_flags.append(flagged)
        reasons.append(", ".join(reason_list))

    work["is_anomaly"] = anomaly_flags
    work["anomaly_reason"] = reasons

    df.loc[work.index, "is_anomaly"] = work["is_anomaly"]
    df.loc[work.index, "anomaly_reason"] = work["anomaly_reason"]

    return _ensure_columns(df)


# =========================================================
# BUDGETS
# =========================================================

def calculate_budget_tracking(df: pd.DataFrame, budget_map=None):
    if budget_map is None:
        budget_map = {}

    if df is None or df.empty:
        return pd.DataFrame(columns=["category", "spent", "budget", "usage_pct", "status"])

    work = df[(df["valid_row"] == True) & (df["direction"] == "expense")].copy()
    if work.empty:
        return pd.DataFrame(columns=["category", "spent", "budget", "usage_pct", "status"])

    spent = work.groupby("category", dropna=False)["amount"].sum().reset_index()
    spent.columns = ["category", "spent"]

    records = []
    for _, row in spent.iterrows():
        category = row["category"]
        s = float(row["spent"])
        b = float(budget_map.get(category, 0)) if category in budget_map else 0.0

        if b > 0:
            usage_pct = (s / b) * 100
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
            "spent": round(s, 2),
            "budget": round(b, 2),
            "usage_pct": round(usage_pct, 2) if pd.notna(usage_pct) else np.nan,
            "status": status,
        })

    return pd.DataFrame(records).sort_values("spent", ascending=False)


# =========================================================
# FINANCIAL HEALTH SCORE
# =========================================================

def calculate_financial_health_score(df: pd.DataFrame, budget_df=None):
    """
    More reliable health score:
    - weighted by source confidence
    - penalizes weak data quality
    - avoids overreacting when income is missing
    - returns confidence label for jury defensibility
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
    valid_df = df[df["valid_row"] == True].copy()
    valid_rows = len(valid_df)

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

    # -----------------------------
    # Source reliability weights
    # -----------------------------
    source_weights = {
        "csv": 1.00,
        "text": 0.85,
        "pdf": 0.65
    }

    valid_df["source_weight"] = valid_df["source"].astype(str).str.lower().map(source_weights).fillna(0.75)
    valid_df["parse_confidence"] = pd.to_numeric(valid_df["parse_confidence"], errors="coerce").fillna(0.70)

    # final transaction weight
    valid_df["txn_weight"] = valid_df["source_weight"] * valid_df["parse_confidence"]

    # keep only reasonably trusted rows for score
    score_df = valid_df[valid_df["txn_weight"] >= 0.45].copy()

    if score_df.empty:
        return {
            "health_score": 50,
            "score_band": "Low Confidence",
            "quality_confidence": round((valid_rows / total_rows) * 100, 1),
            "confidence_label": "Low",
            "components": {
                "reason": "Transactions parsed, but confidence too weak for reliable scoring"
            }
        }

    # -----------------------------
    # Weighted income / expense
    # -----------------------------
    income_df = score_df[score_df["direction"] == "income"].copy()
    expense_df = score_df[score_df["direction"] == "expense"].copy()

    weighted_income = (income_df["amount"] * income_df["txn_weight"]).sum() if not income_df.empty else 0
    weighted_expense = (expense_df["amount"] * expense_df["txn_weight"]).sum() if not expense_df.empty else 0

    # detect whether income coverage is trustworthy
    income_count = len(income_df)
    expense_count = len(expense_df)

    income_coverage_ok = (income_count >= 1 and weighted_income > 0)

    # -----------------------------
    # 1. Savings / cashflow score
    # -----------------------------
    if income_coverage_ok:
        savings_rate = max((weighted_income - weighted_expense) / max(weighted_income, 1), -1)
        cashflow_score = max(0, min(100, 50 + savings_rate * 100))
    else:
        # neutral fallback if income is weak/missing
        # do not aggressively punish user for parser weakness
        if weighted_expense <= 0:
            cashflow_score = 55
        else:
            cashflow_score = 60

    # -----------------------------
    # 2. Budget score
    # -----------------------------
    budget_score = 70
    if budget_df is not None and not budget_df.empty:
        valid_budget_rows = budget_df[pd.to_numeric(budget_df["budget"], errors="coerce").fillna(0) > 0].copy()
        if not valid_budget_rows.empty:
            over_budget_ratio = (valid_budget_rows["status"] == "Over Budget").mean()
            near_limit_ratio = (valid_budget_rows["status"] == "Near Limit").mean()
            budget_score = 100 - (over_budget_ratio * 45 + near_limit_ratio * 20)
            budget_score = max(0, min(100, budget_score))

    # -----------------------------
    # 3. Anomaly score
    # -----------------------------
    anomaly_count = int((score_df["is_anomaly"] == True).sum())
    anomaly_base = max(expense_count, 1)
    anomaly_ratio = anomaly_count / anomaly_base
    anomaly_score = max(0, 100 - anomaly_ratio * 250)

    # -----------------------------
    # 4. Data quality score
    # -----------------------------
    valid_ratio = valid_rows / total_rows if total_rows else 0
    avg_conf = score_df["txn_weight"].mean() if not score_df.empty else 0
    quality_score = ((valid_ratio * 0.5) + (avg_conf * 0.5)) * 100
    quality_score = max(0, min(100, quality_score))

    # -----------------------------
    # Penalize if income reliability is weak
    # -----------------------------
    income_reliability_penalty = 0
    if not income_coverage_ok:
        income_reliability_penalty = 8

    # -----------------------------
    # Final weighted score
    # -----------------------------
    raw_score = (
        0.40 * cashflow_score +
        0.25 * budget_score +
        0.20 * anomaly_score +
        0.15 * quality_score
    ) - income_reliability_penalty

    raw_score = max(0, min(100, raw_score))

    # Stabilize toward neutral if quality is weak
    confidence_factor = min(max(avg_conf, 0.35), 1.0)
    stabilized_score = (raw_score * confidence_factor) + (60 * (1 - confidence_factor))
    final_score = int(round(max(0, min(100, stabilized_score))))

    # -----------------------------
    # Banding
    # -----------------------------
    if final_score >= 80:
        band = "Strong"
    elif final_score >= 65:
        band = "Stable"
    elif final_score >= 50:
        band = "Moderate"
    else:
        band = "At Risk"

    # confidence label
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
            "income_reliability_penalty": income_reliability_penalty
        }
    }


# =========================================================
# INSIGHTS
# =========================================================

def generate_insights(df: pd.DataFrame, budget_df=None, health=None):
    insights = []

    if df is None or df.empty:
        return ["No transactions available."]

    usable = df[df["valid_row"] == True].copy()
    if usable.empty:
        return ["No valid transactions passed validation."]

    expense_df = usable[usable["direction"] == "expense"]
    income_df = usable[usable["direction"] == "income"]

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
        if total_income > 0:
            savings = total_income - total_expense
            insights.append(f"Estimated savings after expenses: {savings:.2f}.")

    anomalies = usable[usable["is_anomaly"] == True]
    if not anomalies.empty:
        insights.append(f"{len(anomalies)} potentially anomalous expense(s) detected.")

    recurring = usable[usable["is_recurring"] == True]
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
            f"with data confidence {health.get('quality_confidence', 0)}%."
        )

    return insights


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
        health = calculate_financial_health_score(empty_df, pd.DataFrame())
        return {
            "transactions": empty_df,
            "budget_tracking": pd.DataFrame(),
            "health": health,
            "insights": ["No data was processed."],
        }

    combined = pd.concat(all_parts, ignore_index=True)

    # Revalidate after concat just to be safe
    combined = validate_transactions(combined)

    # Use only valid rows going forward, but keep bad rows in a side frame if needed
    combined = combined[combined["valid_row"] == True].copy()

    combined = deduplicate_transactions(combined)
    combined = enrich_transactions(combined, memory_path=memory_path)
    combined = detect_recurring_transactions(combined)
    combined = detect_anomalies(combined)

    # final sort
    if not combined.empty:
        combined["date"] = pd.to_datetime(combined["date"], errors="coerce").dt.date
        combined = combined.sort_values(["date", "amount"], ascending=[False, False]).reset_index(drop=True)

    budget_df = calculate_budget_tracking(combined, budget_map=budget_map or {})
    health = calculate_financial_health_score(combined, budget_df)
    insights = generate_insights(combined, budget_df, health)

    return {
        "transactions": _ensure_columns(combined),
        "budget_tracking": budget_df,
        "health": health,
        "insights": insights,
    }


# =========================================================
# OPTIONAL COMPATIBILITY WRAPPERS
# =========================================================
# Keep these wrappers so your existing app can call whichever
# names it already uses with minimal changes.

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
