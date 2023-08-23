import pickle

from src.config.core import config

from src.data.extractors import ExtractorX, ExtractorY
from src.data.integrate_XY import integrate_XY

from src.features.StaggeredPipeline import StaggeredPipeline


extractor_x = ExtractorX(config)
X = extractor_x.extract()

extractor_y = ExtractorY(config.config_data)

if extractor_y.extract:
    
    Y = extractor_y.extract()

    XY = integrate_XY(X, Y, config.outcome_definition)

    X_attributes = XY[config.dataset_attributes]
    X = XY[config.features]
    y = XY["y"]
    y.to_csv(config.outputs_path["y"].with_suffix(".csv"), index=False)

else:
    X_attributes = X[config.dataset_attributes]
    X = X[config.features]

if config.is_training_run:

    pipeline = StaggeredPipeline(config.config_transforms)
    pipeline.fit(X, y=y)
    with open(config.outputs_path["feature_transforms_pipeline"], "wb") as f:
        pickle.dump(pipeline, f)

with open(config.outputs_path["feature_transforms_pipeline"], "rb") as f:
    pipeline = pickle.load(f)
    
X = pipeline.transform(X)

X.to_csv(config.outputs_path["X"].with_suffix(".csv"), index=False)
X.to_pickle(config.outputs_path["X"].with_suffix(".pkl"))
X_attributes.to_csv(config.outputs_path["X_attributes"].with_suffix(".csv"), index=False)