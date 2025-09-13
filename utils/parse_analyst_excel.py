import pandas as pd
from pathlib import Path
import re

def safe_name(name):
    return re.sub(r'[^A-Za-z0-9_.-]+', '_', str(name)).strip('_') or "sheet"

def preprocess_analyst_data():
    # Path to your Excel workbook
    excel_path = Path("data/Analyst_Reports_Historical_Data.xlsm")


    # Force output into a subdirectory named "analyst_reviews" inside /data
    out_dir = excel_path.parent / "analyst_reviews"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load workbook
    xls = pd.ExcelFile(excel_path, engine="openpyxl")

    # Loop through all sheets and save each to CSV
    for sheet in xls.sheet_names:
        df = pd.read_excel(xls, sheet_name=sheet, dtype=object)  # preserve values as strings
        out_path = out_dir / f"{safe_name(sheet)}.csv"
        df.to_csv(out_path, index=False, na_rep="")
        print(f"Saved: {out_path}")
