from lightgbm.basic import Booster
import pandas as pd


def feature_importance(model: Booster, features: list) -> None:
    importance = pd.DataFrame(
        model.feature_importance(), index=features, columns=['importance']
    )
    print(importance)


if __name__ == "__main__":
    feature_importance()
