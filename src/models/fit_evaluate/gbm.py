import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn import metrics

from src.config.core import config

PATH_MODEL_ULTIMATE = config.outputs_dir["pipeline_artifacts"] / "xgboost"

PATH_X = config.outputs_dir["data_processed"] / "X_train.pkl"
PATH_Y = config.outputs_dir["data_processed"] / "y_train.csv"

PATH_X_TEST = config.outputs_dir["data_processed"] / "X.pkl"
PATH_Y_TEST = config.outputs_dir["data_processed"] / "y.csv"

X = pd.read_pickle(PATH_X)
y = pd.read_csv(PATH_Y)["y"]

X_test = pd.read_pickle(PATH_X_TEST)
y_test = pd.read_csv(PATH_Y_TEST)["y"]

dtrain = xgb.DMatrix(X.astype("float"), label=y)
dtest = xgb.DMatrix(X_test.astype("float"))


# declare experiments
grid_tune = pd.DataFrame(
    {
        "objective": "reg:squarederror",
        "num_boost_round": [100],
        "eta": [0.01],
        "max_depth": [2],
        "subsample": [0.5],
        "lambda": [0],
        "alpha": [0],
    }
)
grid_tune = grid_tune.iloc[np.repeat(0, 15), :].reset_index(drop=True)
grid_tune["num_boost_round"] = [
    100,
    200,
    300,
    400,
    500,
    600,
    700,
    800,
    900,
    1000,
    1100,
    1200,
    1300,
    1400,
    1500
]
# grid_tune["max_depth"] = [1, 2, 3, 5]

scores = []
for i in range(grid_tune.shape[0]):

    # fit
    param_all = grid_tune.iloc[i, :].to_dict()
    param = param_all.copy()
    del param["num_boost_round"]
    num_boost_round = param_all["num_boost_round"]
    model = xgb.train(param, dtrain, num_boost_round)

    # predict
    y_pred = model.predict(dtest)

    # evaluate (score preds)
    score = {}
    score["error_summary"] = metrics.mean_squared_error(
        y_test, y_pred, squared=False
    )
    scores.append(score)

    print("iteration " + str(i) + " complete.")

scores


# finalize
param = {
    "objective": "reg:squarederror",
    "eta": 0.01,
    "max_depth": 2,
    "subsample": 0.5,
    "lambda": 0,
    "alpha": 0,
}
num_boost_round = 500

model = xgb.train(param, dtrain, num_boost_round)
model.save_model(PATH_MODEL_ULTIMATE)

feature_importance = pd.DataFrame.from_dict(
    model.get_score(importance_type="gain"), orient="index"
).reset_index(drop=False)
feature_importance.columns = ["feature", "score"]
feature_importance = feature_importance.sort_values("score", ascending=False)

model = xgb.Booster()
model.load_model(PATH_MODEL_ULTIMATE)
