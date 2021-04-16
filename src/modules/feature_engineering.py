from typing import Tuple
import pandas as pd

from constants import MEAN_LIGHT, PLACE_ID, SUM_LIGHT, YEAR


def feature_engineering(x: pd.DataFrame, test_data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, list]:

    features = [PLACE_ID, YEAR, MEAN_LIGHT, SUM_LIGHT]

    return x, test_data, features


if __name__ == "__main__":
    feature_engineering()
