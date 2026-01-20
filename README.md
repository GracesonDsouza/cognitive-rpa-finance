# Cognitive RPA for Personal Finance Data Consolidation (Level-2 Prototype)

This project demonstrates a **Cognitive RPA** prototype that consolidates personal finance data from **unstructured sources**
into a single dashboard.

## What this prototype does
- Accepts **CSV transactions** or **SMS/Email-style text**
- Extracts transactions
- Categorizes expenses using keyword-based rules (expandable to NLP/ML)
- Detects recurring payments (subscriptions/EMIs) using simple pattern recognition
- Displays a Streamlit dashboard (KPIs + charts)

## Tech stack
- Python, Pandas
- Streamlit
- Pattern recognition (recurring detection)
- Basic cognitive rules (categorization)

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Demo data
Use `sample_data/transactions.csv` to quickly test recurring detection (Netflix, Rent).
