import pytest
from src.data.ConfigData import ConfigData
from src.features.ConfigFeatures import ConfigFeatures
from src.config.ConfigFeatureTransformsPipeline import (
    ConfigFeatureTransformsPipeline,
)


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
