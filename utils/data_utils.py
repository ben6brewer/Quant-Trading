# utils/data_utils.py

def save_parquet(df, path):
    df.to_parquet(path)

def load_parquet(path):
    import pandas as pd
    return pd.read_parquet(path)
