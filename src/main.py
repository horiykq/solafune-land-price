import sys

import pandas as pd

from constants import DATA_DIR
from modules.create_submit_file import create_submit_file
from modules.feature_engineering import feature_engineering
from modules.fix_seed import fix_seed
from modules.lightgbm import lgb_fit, lgb_predict
from modules.preprocess import preprocess
from modules.save_history import save_history
from modules.split_preprocessed_data import split_preprocessed_data
from params import SAVE_HISTORY, SEED


def main():
    success_fix_seed = fix_seed(SEED)
    if success_fix_seed:
        print("FIX SEED SUCCESS")
    else:
        sys.exit()

    if SAVE_HISTORY:
        success_save_history = save_history()
        if success_save_history:
            print("HISTORY BACKUP SUCCESS")
        else:
            sys.exit()

    train = pd.read_csv(f'{DATA_DIR}/TrainDataSet.csv')
    test = pd.read_csv(f'{DATA_DIR}/EvaluationData.csv')

    x, y, test_data = preprocess(train, test)
    x, test_data = feature_engineering(x, test_data)

    X_train, X_validation, y_train, y_validation = split_preprocessed_data(
        x, y
    )

    print(X_train.shape, X_validation.shape,  y_train.shape,
          y_validation.shape,  test_data.shape)

    model = lgb_fit(X_train, X_validation, y_train, y_validation)

    pred = lgb_predict(model, test_data)

    success_create_submit = create_submit_file(pred)
    if success_create_submit:
        print("CREATE SUBMIT SUCCESS")
    else:
        sys.exit()


if __name__ == "__main__":
    main()
