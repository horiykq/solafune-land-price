import matplotlib.pyplot as plt
import pandas as pd
from matplotlib_venn import venn2


def eda(train: pd.DataFrame, test: pd.DataFrame):
    print(train.info())
    print((train['PlaceID'].value_counts() < 22).sum())
    print(test.info())


if __name__ == "__main__":
    eda()
