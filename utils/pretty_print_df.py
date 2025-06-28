# utils.pretty_print_df.py

from rich.console import Console
from rich.table import Table
import pandas as pd

def pretty_print_df(df: pd.DataFrame, title: str = None):
    console = Console()
    table = Table(title=title or getattr(df, 'title', 'DataFrame'))

    # Add column headers
    for column in df.columns:
        table.add_column(str(column), overflow='fold', style="cyan", no_wrap=True)

    # Add rows (convert all values to string to avoid errors)
    for _, row in df.iterrows():
        table.add_row(*[str(value) for value in row])

    console.print(table)