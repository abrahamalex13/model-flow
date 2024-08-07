# same config instance loaded by several disparate modules.
# given that behavior, centralize the config instance in standalone module.

from .ConfigFeatureTransformsPipeline import ConfigFeatureTransformsPipeline

config = ConfigFeatureTransformsPipeline(
    path_config_data="src/data/config.yml",
    path_config_features="src/features/config.yml",
)
config.make_dirs()
