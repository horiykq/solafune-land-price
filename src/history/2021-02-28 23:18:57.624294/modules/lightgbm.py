from lightgbm import LGBMModel
from lightgbm.basic import Booster
import pandas as pd

from constants import MEAN_LIGHT, SUM_LIGHT, TARGET
from modules.standardize import standardize
from params import DATA_SPRIT_RATE, LGBM_PARAMS


def lgb_preprocess(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    y = train[TARGET]
    y = standardize(y)

    mean_light = train[MEAN_LIGHT]
    sum_light = train[SUM_LIGHT]

    std_mean_light = standardize(mean_light)
    std_sum_light = standardize(sum_light)

    x = pd.concat([std_mean_light, std_sum_light], axis=1)

    test_mean_light = test[MEAN_LIGHT]
    test_sum_light = test[SUM_LIGHT]

    std_test_mean_light = standardize(test_mean_light)
    std_test_sum_light = standardize(test_sum_light)

    test_data = pd.concat([std_test_mean_light, std_test_sum_light], axis=1)

    return x, y, test_data


def lgb_model() -> LGBMModel:

    return LGBMModel


def lgb_predict(model: LGBMModel, test_data: pd.DataFrame):

    preds = model.predict(test_data)
    return preds
