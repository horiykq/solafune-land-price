import pandas as pd


def standardize(df: pd.DataFrame) -> pd.DataFrame:
    mean = df.mean()
    std = df.std()

    df_std = (df - mean) / std

    return df_std
