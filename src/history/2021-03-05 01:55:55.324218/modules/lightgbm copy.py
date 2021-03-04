from lightgbm import LGBMRegressor
import pandas as pd
from sklearn.model_selection import train_test_split


from constants import MEAN_LIGHT, SUM_LIGHT, TARGET


def lgb_preprocess(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    y = train[TARGET]

    mean_light = train[MEAN_LIGHT]
    sum_light = train[SUM_LIGHT]

    x = pd.concat([mean_light, sum_light], axis=1)

    test_mean_light = test[MEAN_LIGHT]
    test_sum_light = test[SUM_LIGHT]

    test_data = pd.concat([test_mean_light, test_sum_light], axis=1)

    X_train, X_validation, y_train, y_validation = train_test_split(x, y)

    return X_train, X_validation, y_train, y_validation, test_data


def lgb_model() -> LGBMRegressor:

    return LGBMRegressor()


def lgb_fit(model: LGBMRegressor, x: pd.DataFrame, y: pd.DataFrame) -> LGBMRegressor:

    model.fit(x, y)
    return model


def lgb_predict(model: LGBMRegressor, test_data: pd.DataFrame):

    preds = model.predict(test_data)
    return preds
