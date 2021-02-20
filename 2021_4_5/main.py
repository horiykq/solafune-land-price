import pandas as pd
import numpy as np
from matplotlib_venn import venn2
import seaborn as sns
import matplotlib.pyplot as plt


def main():
    train = pd.read_csv('data/TrainDataSet.csv')
    test = pd.read_csv('data/EvaluationData.csv')
    target = train['AverageLandPrice']

    # EDA
    print(train.info())
    print((train['PlaceID'].value_counts() < 22).sum())
    plt.figure(figsize=(8, 6))
    venn2(subsets=[set(test['PlaceID']), set(train['PlaceID'])],
          set_labels=['test', 'train'])


if __name__ == "__main__":
    main()
