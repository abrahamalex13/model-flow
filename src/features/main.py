import pickle
from src.config import config
from src.data.extractors import ExtractorX, ExtractorY
from src.data.integrate_XY import integrate_XY
from src.features.StaggeredPipeline import StaggeredPipeline


extractor_x = ExtractorX(config)
X = extractor_x.extract()

if config.is_training_run or config.is_evaluation_run:

    extractor_y = ExtractorY(config.source_Y)
    Y = extractor_y.extract()

    XY = integrate_XY(X, Y, config.source_Y)
    X_attributes = XY[config.dataset_attributes]
    X = XY[config.features]
    y = XY["y"]
    y.to_csv(config.y_path.with_suffix(".csv"), index=False)

else:
    X_attributes = X[config.dataset_attributes]
    X = X[config.features]

if config.is_training_run:

    pipeline = StaggeredPipeline(config.config_transforms)
    pipeline.fit(X, y=y)
    with open(config.feature_transforms_pipeline_path, "wb") as f:
        pickle.dump(pipeline, f)

else:

    with open(config.feature_transforms_pipeline_path, "rb") as f:
        pipeline = pickle.load(f)

X = pipeline.transform(X)

X.to_csv(config.X_path.with_suffix(".csv"), index=False)
X.to_pickle(config.X_path.with_suffix(".pkl"))
X_attributes.to_csv(config.X_attributes_path.with_suffix(".csv"), index=False)
