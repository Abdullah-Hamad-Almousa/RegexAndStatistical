import pandas as pd
import numpy as np
import random

np.random.seed(42)
random.seed(42)

# Load reviews
reviews_df = pd.read_csv('../res/data/ar_reviews_100k.tsv', sep='\t')
reviews = reviews_df['text'].tolist()

# Load finalPII and get columns
piidf = pd.read_csv('../res/data/finalPII.csv')
pii_columns = piidf.columns.tolist()

# Create a lookup dict: column_name -> list of values
pii_lookup = {col: piidf[col].dropna().tolist() for col in pii_columns}

# We only need rows matching finalPII (110 rows)
num_rows = 110

# Prepare outputs
context_texts = []
val_records = []

for i in range(num_rows):
    base_text = str(reviews[i % len(reviews)])

    # Decide which PII types to inject (random subset: 0 to all)
    num_types_to_inject = random.randint(0, len(pii_columns))
    selected_cols = random.sample(pii_columns, num_types_to_inject)

    injected_text = base_text
    val_record = {}

    for col in pii_columns:
        if col in selected_cols:
            # Pick random value from this column
            val = random.choice(pii_lookup[col])
            injected_text += f" [{val}]"
            val_record[col] = 1
        else:
            val_record[col] = 0

    context_texts.append(injected_text)
    val_records.append(val_record)

# Write context.tsv
context_df = pd.DataFrame({'text': context_texts})
context_df.to_csv('../res/data/context.tsv', sep='\t', index=False)

# Write val.csv
val_df = pd.DataFrame(val_records)
val_df.to_csv('../res/data/val.csv', index=False)

print(f"Created context.tsv with {len(context_texts)} rows")
print(f"Created val.csv with {len(val_df.columns)} PII type columns")
print(f"\nSample val.csv head:")
print(val_df.head())