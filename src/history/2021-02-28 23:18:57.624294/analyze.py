import pandas as pd

from constants import DATA_DIR
from modules.eda import eda


def main():
    train = pd.read_csv(f'{DATA_DIR}/TrainDataSet.csv')
    test = pd.read_csv(f'{DATA_DIR}/EvaluationData.csv')

    eda(train, test)


if __name__ == "__main__":
    main()
