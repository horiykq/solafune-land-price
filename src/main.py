import sys

import numpy as np
import pandas as pd
import seaborn as sns

from constants import DATA_DIR
from modules.lightgbm import lgb_preprocess
from modules.save_history import save_history
from modules.verify_gpu import verify_gpu
from params import GPU_ACTIVE, SAVE_HISTORY


def main():
    if GPU_ACTIVE:
        verify_gpu()
    if SAVE_HISTORY:
        success_save_history = save_history()
        if not success_save_history:
            sys.exit()

    train = pd.read_csv(f'{DATA_DIR}/TrainDataSet.csv')
    test = pd.read_csv(f'{DATA_DIR}/EvaluationData.csv')

    x, y = lgb_preprocess(train, test)

    print(x.shape, y.shape)


if __name__ == "__main__":
    main()
