import numpy as np
import pandas as pd
from constants import MEAN_LIGHT, PLACE_ID, SUM_LIGHT, TARGET, YEAR


def preprocess(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    y = train[TARGET]
    y = np.log(y + 1)

    place_id = train[PLACE_ID]
    year = train[YEAR]
    mean_light = train[MEAN_LIGHT]
    sum_light = train[SUM_LIGHT]

    x = pd.concat([place_id, year, mean_light, sum_light], axis=1)

    test_place_id = test[PLACE_ID]
    test_year = test[YEAR]
    test_mean_light = test[MEAN_LIGHT]
    test_sum_light = test[SUM_LIGHT]

    test_data = pd.concat(
        [test_place_id, test_year, test_mean_light, test_sum_light],
        axis=1
    )

    return x, y, test_data


if __name__ == "__main__":
    preprocess()
