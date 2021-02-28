# environment settings

SAVE_HISTORY = True


# lgbm

DATA_SPRIT_RATE = 0.7

LGBM_PARAMS = {
    "task": "train",
    'boosting_type': 'gbdt',
    'objective': 'multiclass',
    'num_class': 10,
    'verbose': 2,
}
