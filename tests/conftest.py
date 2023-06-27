import pytest
from src.config.data.ConfigData import ConfigData
from src.config.features.ConfigFeatures import ConfigFeatures
from src.config.features.TransformArgsAdapter import (
    TransformArgsAdapter,
)
from src.config.ConfigFeatureTransformsPipeline import (
    ConfigFeatureTransformsPipeline,
)


@pytest.fixture()
def config_data():
    cfg = ConfigData("./tests/src/config/data/config.yml")
    return cfg


@pytest.fixture()
def config_features():
    cfg = ConfigFeatures("./tests/src/config/features/config.yml")
    return cfg


@pytest.fixture()
def config_feature_transforms_pipeline():
    cfg = ConfigFeatureTransformsPipeline(
        path_config_data="./tests/src/config/data/config.yml",
        path_config_features="./tests/src/config/features/config.yml",
        path_config_subdirs="./tests/src/config/utils/config_pipeline_outputs_subdirectories.yml",
    )
    return cfg
