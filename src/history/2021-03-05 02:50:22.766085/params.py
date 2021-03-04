# environment settings

SAVE_HISTORY = True
SEED = 42


# lgbm

DATA_SPRIT_RATE = 0.7

LGBM_PARAMS = {
    'objective': 'regression',
    'metric': "rmse"
}
