import lightgbm as lgb
import pandas as pd

from constants import MEAN_LIGHT, SUM_LIGHT, TARGET
from modules.standardize import standardize


def lgb_preprocess(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    y = train[TARGET]

    mean_light = train[MEAN_LIGHT]
    sum_light = train[SUM_LIGHT]

    std_mean_light = standardize(mean_light)
    std_sum_light = standardize(sum_light)

    x = pd.concat([std_mean_light, std_sum_light], axis=1)

    return x, y


# def lgb_model():


# def lgb_learn():


# def lgb_save_model():
