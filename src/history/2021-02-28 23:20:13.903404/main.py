import sys

import numpy as np
import pandas as pd
import seaborn as sns

from constants import DATA_DIR
from modules.lightgbm import lgb_model, lgb_predict, lgb_preprocess
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
    model.fit(x, y)

    preds = lgb_predict(model, test_data)

    for pred in preds:
        print(pred)


if __name__ == "__main__":
    main()
