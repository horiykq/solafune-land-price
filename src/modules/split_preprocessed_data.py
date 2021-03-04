import pandas as pd
from sklearn.model_selection import train_test_split


def split_preprocessed_data(x: pd.DataFrame, y: pd.DataFrame) -> pd.DataFrame:
    X_train, X_validation, y_train, y_validation = train_test_split(x, y)

    return X_train, X_validation, y_train, y_validation


if __name__ == "__main__":
    split_preprocessed_data()
