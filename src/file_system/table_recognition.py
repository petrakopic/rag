import numpy as np
import pandas as pd
from tabula.io import read_pdf


def extract_tables(file_path: str):
    dfs = read_pdf(file_path, pages="all", lattice=True)
    tables = []
    for idx, df in enumerate(dfs):
        if not df.empty and len(df.columns) > 1 and not "Unnamed: 0" in df.columns:
            tables.append(remove_nan_and_shift(df))
    return tables


def remove_nan_and_shift(df: pd.DataFrame):
    for column in df:
        # Drop NaN values and reset index without adding a new column
        cleaned_column = df[column].dropna().reset_index(drop=True)
        # Reassign cleaned values back to the DataFrame
        df[column] = cleaned_column
        df[column] = df[column].astype(str)
        # Fill the remaining indices with NaN
        for i in range(len(cleaned_column), len(df)):
            df.at[i, column] = np.nan

        df[column] = df[column].astype(str)
    return df
