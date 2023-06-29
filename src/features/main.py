import pickle
from pathlib import Path
import pandas as pd

from src.config.core import config

from src.data import extractors
from src.data.db import engine
from src.data.integrate_XY import integrate_XY

from src.features.PipelineImputeScrubEnrichBasis import (
    PipelineImputeScrubEnrichBasis,
)


query_x = extractors.QueryX(config.config_data)
X = pd.read_sql_query(query_x.query, engine)

query_y = extractors.QueryY(config.config_data)
Y = pd.read_sql_query(query_y.query, engine)
Y = Y.rename(columns={config.outcome_definition["title"]: "y"})

XY = integrate_XY(X, Y, config.outcome_definition)

X = XY[config.features]
X_attributes = XY[config.dataset_attributes]
y = XY["y"]

if config.is_training_run:

    pipeline = PipelineImputeScrubEnrichBasis(config.transforms_calls)
    pipeline.fit(X, y=y)
    with open(config.outputs_path["feature_transforms_pipeline"], "wb") as f:
        pickle.dump(pipeline, f)

with open(config.outputs_path["feature_transforms_pipeline"], "rb") as f:
    pipeline = pickle.load(f)

X = pipeline.transform(X)

X.to_csv(config.outputs_path["X"].with_suffix(".csv"), index=False)
X.to_pickle(config.outputs_path["X"].with_suffix(".pkl"))
X_attributes.to_csv(
    Path(str(config.outputs_path["X"]) + "_attributes.csv"), index=False
)

y.to_csv(config.outputs_path["y"].with_suffix(".csv"), index=False)
