import matplotlib.pyplot as plt
import pandas as pd
from matplotlib_venn import venn2


def eda(train: pd.DataFrame, test: pd.DataFrame):
    print(train.info())
    print((train['PlaceID'].value_counts() < 22).sum())
    # plt.figure(figsize=(8, 6))
    # venn2(subsets=[set(test['PlaceID']), set(train['PlaceID'])],
    #       set_labels=['test', 'train'])


if __name__ == "__main__":
    eda()
