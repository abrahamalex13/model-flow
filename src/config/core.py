import pathlib
from src.config.ConfigFeatureTransformsPipeline import (
    ConfigFeatureTransformsPipeline,
)
from src.config.features.TransformArgsAdapter import (
    TransformArgsAdapter,
)

DIR = pathlib.Path(__file__).parent

config = ConfigFeatureTransformsPipeline(
    path_config_data=DIR / "data/config.yml",
    path_config_features=DIR / "features/config.yml",
    path_config_subdirs=DIR
    / "utils/config_pipeline_outputs_subdirectories.yml",
)
config.make_dirs()

transforms_args_adapters = {
    x: TransformArgsAdapter(config.config_features, x)
    for x in config.transforms
}
