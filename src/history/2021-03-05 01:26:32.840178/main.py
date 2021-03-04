import sys

import numpy as np
import pandas as pd
import seaborn as sns

from constants import DATA_DIR
from modules.create_submit_file import create_submit_file
from modules.lightgbm import lgb_fit, lgb_model, lgb_predict, lgb_preprocess
from modules.save_history import save_history
from params import DATA_SPRIT_RATE, SAVE_HISTORY


def main():
    if SAVE_HISTORY:
        success_save_history = save_history()
        if success_save_history:
            print("HISTORY BACKUP SUCCESS")
        else:
            sys.exit()

    train = pd.read_csv(f'{DATA_DIR}/TrainDataSet.csv')
    test = pd.read_csv(f'{DATA_DIR}/EvaluationData.csv')

    x, y, test_data = lgb_preprocess(train, test)
    print(x.shape, y.shape, test_data.shape)

    model = lgb_model()
    model = lgb_fit(model, x, y)

    preds = lgb_predict(model, test_data)

    success_create_submit = create_submit_file(preds)
    if success_create_submit:
        print("CREATE SUBMIT SUCCESS")
    else:
        sys.exit()


if __name__ == "__main__":
    main()
