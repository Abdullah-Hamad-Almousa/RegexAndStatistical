import pandas as pd
import numpy as np

np.random.seed(42)

datasets = {
    'PIIdata': '../res/data/PIIdata.csv',
    'NamesFinal': '../res/data/NamesFinal.csv',
    'EngHalfPII': '../res/data/EngHalfPII.csv',
    'Date': '../res/data/Date.csv',
    'ArHalfPII': '../res/data/ArHalfPII.csv',
}

max_rows = 110
dfs = []

for name, path in datasets.items():
    df = pd.read_csv(path)
    actual_rows = len(df)

    if actual_rows < max_rows:
        # Repeat rows randomly to reach 110
        repeats_needed = max_rows - actual_rows
        indices = np.random.choice(actual_rows, repeats_needed, replace=True)
        df = pd.concat([df, df.iloc[indices]], ignore_index=True)

    # Take exactly 110 rows
    df = df.iloc[:max_rows]

    # Add source column for tracking
    df['_source'] = name
    dfs.append(df)

# Combine all
final = pd.concat(dfs, axis=1)

# Drop the source column for final output
final = final.drop(columns=['_source'])

final.to_csv('../res/data/finalPII.csv', index=False)
print(f"Created finalPII.csv with {len(final)} rows and {len(final.columns)} columns")
print(f"Columns saved to file (Arabic names may not display correctly in console)")