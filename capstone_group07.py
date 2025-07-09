"""
Capstone Project Group 07: Property Assessment Analysis - Boston FY2024

This script loads and cleans the Boston property assessment dataset (182k+ records, 66 columns).
We're prepping the data for downstream modeling, like predicting assessed property value.

Core steps:
1. Load the CSV
2. Clean up messy column names
3. Convert currency/price strings to numeric
4. Handle missing values gracefully

Author: Group 07 Members (Sweekruti Narendra Singh & Zaili Gu)
"""

import pandas as pd

# Step 1: Load the data
def load_data(filepath):
    # Quick load with pandas
    df = pd.read_csv(filepath)
    print(f"Loaded dataset with shape: {df.shape}")
    return df

# Step 2: Standardize column names
def clean_column_names(df):
    # lowercase, replace spaces with underscores, strip weird symbols
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace(r"[^\w\s]", "", regex=True)
    )
    return df

# Step 3: Convert currency/price columns to floats
def convert_currency_to_numeric(df, columns):
    for col in columns:
        df[col] = (
            df[col]
            .replace(r'[\$,]', '', regex=True)             # drop $ and commas
            .replace(r'^\s*-\s*$', pd.NA, regex=True)     # turn ' - ' into NaN
            .replace(r'^\s*$', pd.NA, regex=True)         # turn blanks into NaN
        )
        df[col] = pd.to_numeric(df[col], errors='coerce') # convert to float
    return df

# Step 4: Fill or drop missing values
def handle_missing_values(df, strategy='drop', fill_value=None):
    if strategy == 'drop':
        return df.dropna()
    elif strategy == 'fill':
        return df.fillna(fill_value)
    else:
        raise ValueError("Use strategy='drop' or 'fill'")