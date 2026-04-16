# MakingTheDataset - PII Detection Dataset Generator

> **All the code can work using the `summaryForRegexHybrid` folder.**

This project generates a synthetic dataset for training and evaluating PII (Personally Identifiable Information) detection models using a hybrid Regex + Statistical approach.

---

## Quick Start (Using summaryForRegexHybrid)

All functionality works from the `summaryForRegexHybrid/` folder:

```bash
cd summaryForRegexHybrid

# Step 1: Generate finalPII.csv (combined PII data)
python comb.py

# Step 2: Generate context.tsv and val.csv (text with embedded PII)
python create_context.py

# Step 3: Run the hybrid PII detection
python RegexAndStatistical.py
```

---

## Project Structure

```
MakingTheDataset/
├── summaryForRegexHybrid/          # ⭐ All-in-one working directory
│   ├── comb.py                     # Combines PII datasets
│   ├── create_context.py           # Creates text + validation labels
│   ├── RegexAndStatistical.py      # Runs hybrid PII detection
│   ├── context.tsv                 # Generated: text with PII
│   └── val.csv                     # Generated: ground truth labels
│
├── res/data/                       # Source data files
│   ├── PIIdata.csv                 # Various PII records
│   ├── NamesFinal.csv              # Arabic names
│   ├── EngHalfPII.csv              # English PII half
│   ├── ArHalfPII.csv               # Arabic PII half
│   ├── Date.csv                    # Date samples
│   ├── ar_reviews_100k.tsv         # Arabic text reviews (base text)
│   ├── finalPII.csv                # [Generated] Combined PII data
│   ├── context.tsv                 # [Generated] Text with injected PII
│   └── val.csv                     # [Generated] Ground truth labels
│
├── comb.py                         # Combines PII datasets → res/data/finalPII.csv
├── create_context.py               # Creates context + validation → res/data/
├── RegexAndStatistical.py          # Main detection script (root version)
├── run_hybrid.py                   # Entry point for hybrid detection
├── Regex.ipynb                     # Jupyter notebook for regex testing
├── RegexAndStatistical.ipynb        # Jupyter notebook for hybrid method
│
├── extra.py                        # Utility scripts
├── test_v*.py                      # Various test versions
├── analyze_*.py                    # Analysis scripts
├── PII_Detection_WhitePaper.md     # Documentation
└── LICENSE
```

---

## File Descriptions

### Core Pipeline Files

| File | Description |
|------|-------------|
| `comb.py` | Loads 5 PII source datasets (`PIIdata.csv`, `NamesFinal.csv`, `EngHalfPII.csv`, `Date.csv`, `ArHalfPII.csv`), combines them into `finalPII.csv` with 110 rows. |
| `create_context.py` | Takes 110 Arabic reviews, randomly injects PII values from `finalPII.csv` into brackets `[value]`, generates `context.tsv` (text) and `val.csv` (labels). |
| `RegexAndStatistical.py` | The main detection engine. Uses **regex** for structured PII (credit cards, emails, IPs, phones, etc.) and **statistical/keyword dictionaries** for names, job titles, medical history, etc. |

### SummaryForRegexHybrid Folder Files

| File | Description |
|------|-------------|
| `comb.py` | Same as root `comb.py` - generates `finalPII.csv` in `res/data/` |
| `create_context.py` | Same as root `create_context.py` - generates `context.tsv` and `val.csv` in `res/data/` |
| `RegexAndStatistical.py` | **Fixed paths version** - runs hybrid detection on `res/data/context.tsv` and `res/data/val.csv`. Output is identical to root version. |

---

## Data Flow

```
res/data/source files (PIIdata, NamesFinal, EngHalfPII, Date, ArHalfPII)
           │
           ▼
       comb.py
           │
           ▼
   res/data/finalPII.csv (110 rows of combined PII)
           │
           ▼
   create_context.py (injects PII into Arabic text)
           │
           ├──────────────────┐
           ▼                  ▼
  res/data/context.tsv  res/data/val.csv
  (text with PII)        (ground truth labels)
           │                  │
           └────────┬─────────┘
                    ▼
          RegexAndStatistical.py
                    │
                    ▼
           Detection Results + Metrics
```

---

## PII Types Detected (20 categories)

**Regex-Based (Structured):**
- Credit Card Numbers
- Email Addresses
- IP Addresses
- IBAN/Swift Codes
- Passport Numbers
- MAC Addresses
- Phone Numbers
- National ID Numbers
- Postal Codes
- License Plates
- Date of Birth

**Statistical/Keyword-Based (Complex):**
- FirstName (Arabic names)
- FatherName
- GrandFatherName
- FamilyName
- Street Name
- Username
- Medical History
- Biometric Data
- Job Title

---

## Requirements

```
pandas
numpy
```

Install with:
```bash
pip install pandas numpy
```

---

## Output Example

When running `RegexAndStatistical.py`:

```
Loading data...
Context rows: 110
Val columns: 20
Load time: 0.0023s
Regex patterns compiled: [...]

=========================================================================
PII DETECTION RESULTS (Regex + Statistical Hybrid)
=========================================================================

Credit Card Nos:
  Accuracy:   89.09%  |  Precision: 0.917  |  Recall: 0.917  |  F1: 0.917
  TP:  98  FP:   9  FN:   3  |  Total: 110

Email Addresses:
  Accuracy:  100.00%  |  Precision: 1.000  |  Recall: 1.000  |  F1: 1.000
  TP: 110  FP:   0  FN:   0  |  Total: 110
...

=========================================================================
SUMMARY
=========================================================================

CAN HANDLE (>=80% accuracy): 15/20
CANNOT HANDLE (<80% accuracy): 5/20

Overall Accuracy: 84.23%
Total processing time: 0.1234s
```

---

## Notes

- The `summaryForRegexHybrid/` folder contains **copies** of the essential scripts with **corrected paths** for the data files.
- Both `comb.py` and `create_context.py` output to `res/data/` regardless of which folder they're run from.
- The `RegexAndStatistical.py` in `summaryForRegexHybrid/` uses relative paths (`res/data/context.tsv`, `res/data/val.csv`) instead of hardcoded absolute paths.
- Arabic names detection uses both a dictionary of common names and statistical frequency analysis from the training data.
