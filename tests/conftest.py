import pytest
import pandas as pd
import pickle

from src.data.ConfigData import ConfigData
from src.features.ConfigFeatures import ConfigFeatures
from src.ConfigFeatureTransformsPipeline import ConfigFeatureTransformsPipeline
from src.features.StaggeredPipeline import StaggeredPipeline


@pytest.fixture()
def config_data():
    cfg = ConfigData("./tests/src/data/config.yml")
    return cfg


@pytest.fixture()
def config_features():
    cfg = ConfigFeatures("./tests/src/features/config.yml")
    return cfg


@pytest.fixture()
def config_feature_transforms_pipeline():
    cfg = ConfigFeatureTransformsPipeline(
        path_config_data="./tests/src/data/config.yml",
        path_config_features="./tests/src/features/config.yml",
        path_config_subdirs="./tests/src/utils/config_pipeline_outputs_subdirectories.yml",
    )
    return cfg


@pytest.fixture()
def X_transformed():

    config = ConfigFeatureTransformsPipeline(
        path_config_data="./tests/src/data/config.yml",
        path_config_features="./tests/src/features/config.yml",
        path_config_subdirs="./tests/src/utils/config_pipeline_outputs_subdirectories.yml",
    )
    # posthoc patch of config, for easier testability
    config.create_artifact_paths(
        models_dir="./tests/models", data_processed_dir="./tests/data/processed"
    )
    config.make_dirs()
    original_keys = config.config_transforms.copy()
    for key in original_keys:
        if key != "onehot_encode":
            del config.config_transforms[key]

    X = pd.read_csv("./tests/data/dummy.csv")
    pipeline = StaggeredPipeline(config.config_transforms)
    pipeline.fit(X, y=None)
    with open(config.feature_transforms_pipeline_path, "wb") as f:
        pickle.dump(pipeline, f)

    X_transformed = pipeline.transform(X)

    return X_transformed
