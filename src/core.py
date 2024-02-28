from src.config.ConfigFeatureTransformsPipeline import (
    ConfigFeatureTransformsPipeline,
)

config = ConfigFeatureTransformsPipeline(
    path_config_data="src/data/config.yml",
    path_config_features="src/features/config.yml",
    path_config_subdirs="src/utils/config_pipeline_outputs_subdirectories.yml",
)
config.make_dirs()
