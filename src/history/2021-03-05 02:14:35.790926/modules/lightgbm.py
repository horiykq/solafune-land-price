import lightgbm as lgb
from lightgbm.basic import Booster
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


from constants import MEAN_LIGHT, SUM_LIGHT, TARGET
from params import LGBM_PARAMS


def lgb_preprocess(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    y = train[TARGET]
    y = np.log(y + 1)

    mean_light = train[MEAN_LIGHT]
    sum_light = train[SUM_LIGHT]

    x = pd.concat([mean_light, sum_light], axis=1)

    test_mean_light = test[MEAN_LIGHT]
    test_sum_light = test[SUM_LIGHT]

    test_data = pd.concat([test_mean_light, test_sum_light], axis=1)

    X_train, X_validation, y_train, y_validation = train_test_split(x, y)

    return X_train, X_validation, y_train, y_validation, test_data


def lgb_fit(X_train: pd.DataFrame, X_validation: pd.DataFrame, y_train: pd.DataFrame, y_validation: pd.DataFrame) -> Booster:

    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_eval = lgb.Dataset(X_validation, y_validation, reference=lgb_train)

    model = lgb.train(LGBM_PARAMS, lgb_train, valid_sets=lgb_eval)

    return model


def lgb_predict(model: Booster, test_data: pd.DataFrame):

    pred = model.predict(test_data, num_iteration=model.best_iteration)
    pred = np.exp(pred) - 1

    return pred
