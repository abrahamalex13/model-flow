import pandas as pd
import numpy as np
import glmnet
from sklearn import metrics
import joblib

from src.config.core import config
from src.config.models import config_ridge

PATH_MODEL_ULTIMATE = (
    config.outputs_dir["pipeline_artifacts"] / "ridge.pkl"
)
PATH_FEATURE_IMPORTANCE_MODEL_ULTIMATE = (
    config.outputs_dir["pipeline_artifacts"]
    / "feature_importance_ridge.csv"
)

X = pd.read_pickle(config.outputs_path["X_train"].with_suffix(".pkl"))
y = pd.read_csv(config.outputs_path["y_train"].with_suffix(".csv"))["y"]


# glmnet intelligently estimates _path_ of linear model solutions
# path points are lambda values
model = glmnet.ElasticNet(
    alpha=0, standardize=False, fit_intercept=True, n_splits=0, random_state=777
    )
model = model.fit(X, y)

tune_grid = pd.DataFrame({'lambda': model.lambda_path_})


if not config_ridge.model_lambda:

    X_test = pd.read_pickle(config.outputs_path["X_test"].with_suffix(".pkl"))
    y_test = pd.read_csv(config.outputs_path["y_test"].with_suffix(".csv"))["y"]

    multiplier_se_addto_min_error = 0.1

    scores = []

    for i in range(tune_grid.shape[0]):

        y_pred = model.predict(X_test, lamb=tune_grid['lambda'][i])

        # trial-specific score dict may store multiple metrics
        score = {}

        score["rmse"] = metrics.mean_squared_error(y_test, y_pred, squared=False)
        score['mae'] = metrics.mean_absolute_error(y_test, y_pred)
        score["error_summary"] = score["rmse"]

        scores.append(score)

        print("Experiment " + str(i) + " complete.")

    scores = pd.DataFrame(scores)
    results = pd.concat([tune_grid, scores], axis=1)

    score_max = (
        results['error_summary'].min() + 
        multiplier_se_addto_min_error * results['error_summary'].std()
        ) 
    
    idx_model_best = np.argwhere(results['error_summary'] <= score_max)[0][0]

    print(results.iloc[idx_model_best])
    
elif config_ridge.model_lambda: 

    idx_model_best = np.argwhere(
        tune_grid['lambda'] <= config_ridge.model_lambda
        )[0][0]


model.lambda_best_ = tune_grid['lambda'][idx_model_best]
# coefficient path data structure:
# one row is one feature
# one column is one lambda value
model.coef_ = model.coef_path_[:, idx_model_best]
model.intercept_ = model.intercept_path_[idx_model_best]
joblib.dump(model, PATH_MODEL_ULTIMATE)

feature_importance = (
    pd.DataFrame({
        "feature": X.columns, 
        "score": np.abs(model.coef_),
        "coef": model.coef_
        })
    .sort_values(["score"], ascending=False)
    )

feature_importance.to_csv(PATH_FEATURE_IMPORTANCE_MODEL_ULTIMATE, index=False)
