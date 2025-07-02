# utils.pretty_print_df.py

from rich.console import Console
from rich.table import Table
import pandas as pd
import numpy as np

def pretty_print_df(df: pd.DataFrame, title: str = None):
    console = Console()
    table = Table(title=title or getattr(df, 'title', 'DataFrame'))

    # Add column headers
    for column in df.columns:
        table.add_column(str(column), overflow='fold', style="cyan", no_wrap=True)

    # Add rows with numeric formatting to 3 decimals
    for _, row in df.iterrows():
        formatted_row = []
        for value in row:
            if isinstance(value, (float, np.floating)):
                formatted_row.append(f"{value:.3f}")
            else:
                formatted_row.append(str(value))
        table.add_row(*formatted_row)

    console.print(table)
