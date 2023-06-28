import pandas as pd
import numpy as np
from skranger.ensemble import RangerForestClassifier, RangerForestRegressor
from sklearn import metrics
import math
import joblib

from src.config.core import config

PATH_MODEL_ULTIMATE = (
    config.outputs_dir["pipeline_artifacts"] / "random_forest.pkl"
)
PATH_FEATURE_IMPORTANCE_MODEL_ULTIMATE = (
    config.outputs_dir["pipeline_artifacts"]
    / "feature_importance_random_forest.csv"
)

X = pd.read_pickle(config.outputs_path["X_train"].with_suffix(".pkl"))
y = pd.read_csv(config.outputs_path["y_train"].with_suffix(".csv"))["y"]

X_test = pd.read_pickle(config.outputs_path["X_test"].with_suffix(".pkl"))
y_test = pd.read_csv(config.outputs_path["y_test"].with_suffix(".csv"))["y"]

# inspired by R ranger default
MTRY = math.floor(math.sqrt(X.shape[1]))
# because 'permutation'-type importance has yielded inexplicable nan
IMPORTANCE = "impurity"
QUANTILES = True


step_n_estimators = 100
n_iter = 25
tune_grid = pd.DataFrame(
    {"n_trees": [step_n_estimators * (i + 1) for i in range(n_iter)]}
)
scores = []

model = RangerForestRegressor(
    n_estimators=step_n_estimators,
    mtry=MTRY,
    importance=IMPORTANCE,
    max_depth=0,
    quantiles=QUANTILES,
)
for i in range(tune_grid.shape[0]):

    model.set_params(n_estimators=tune_grid["n_trees"].iloc[i])

    model.fit(X, y.values)

    y_pred = model.predict(X_test)

    # trial-specific score dict may store multiple metrics
    score = {}

    score["error_summary"] = metrics.mean_squared_error(
        y_test, y_pred, squared=False
    )

    scores.append(score)

    print("Experiment " + str(i) + " complete.")

scores = pd.DataFrame(scores)
results = pd.concat([tune_grid, scores], axis=1)
# results.plot(x="n_trees", y="error_summary", kind="line")


n_estimators_ultimate = 1000

model = RangerForestRegressor(
    n_estimators=n_estimators_ultimate,
    mtry=MTRY,
    importance=IMPORTANCE,
    max_depth=0,
    quantiles=QUANTILES,
)
model.fit(X, y.values)
joblib.dump(model, PATH_MODEL_ULTIMATE)

feature_importance = pd.DataFrame(
    {"feature": X.columns, "score": model.feature_importances_}
).sort_values(["score"], ascending=False)

feature_importance.to_csv(PATH_FEATURE_IMPORTANCE_MODEL_ULTIMATE, index=False)
