import numpy as np
import pandas as pd
from params import LGBM_PARAMS

import lightgbm as lgb
from lightgbm.basic import Booster


def lgb_fit(X_train: pd.DataFrame, X_validation: pd.DataFrame, y_train: pd.DataFrame, y_validation: pd.DataFrame) -> Booster:

    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_eval = lgb.Dataset(X_validation, y_validation, reference=lgb_train)

    model = lgb.train(LGBM_PARAMS, lgb_train, valid_sets=lgb_eval)

    return model


def lgb_predict(model: Booster, test_data: pd.DataFrame) -> list:

    pred = model.predict(test_data, num_iteration=model.best_iteration)
    pred = np.exp(pred) - 1

    return pred
