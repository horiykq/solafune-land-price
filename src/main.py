import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from constants import DATA_DIR
from matplotlib_venn import venn2

from save_history import save_history


def main():
    save_result = save_history()
    if not save_result:
        sys.exit()

    train = pd.read_csv(f'{DATA_DIR}/TrainDataSet.csv')
    test = pd.read_csv(f'{DATA_DIR}/EvaluationData.csv')
    target = train['AverageLandPrice']

    # EDA
    print(train.info())
    print((train['PlaceID'].value_counts() < 22).sum())
    plt.figure(figsize=(8, 6))
    venn2(subsets=[set(test['PlaceID']), set(train['PlaceID'])],
          set_labels=['test', 'train'])


if __name__ == "__main__":
    main()
