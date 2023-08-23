from pathlib import Path
from src.config.data.ConfigData import ConfigData
from src.config.features.ConfigFeatures import ConfigFeatures

from src.utils import load_yaml


class ConfigFeatureTransformsPipeline:
    """
    A feature transforms pipeline configuration entails:
    - Data configuration
    - Features configuration
    - Subdirectories configuration
        Subdir titles described via (i) data, (ii) features config keys.
        Because transformers, transformed data dirs vary by run.
    """

    def __init__(
        self,
        path_config_data,
        path_config_features,
        path_config_subdirs,
    ):

        self.path_config_data = path_config_data
        self.path_config_features = path_config_features
        self.path_config_subdirs = path_config_subdirs

        self.config_data = ConfigData(self.path_config_data)
        self.is_training_run = self.config_data.is_training_run
        self.source = self.config_data.source
        self.filters = self.config_data.filters
        self.filters_train = self.config_data.filters_train
        self.outcome_title = self.config_data.outcome_definition["title"]
        self.outcome_definition = self.config_data.outcome_definition
        self.dataset_attributes = self.config_data.dataset_attributes

        self.config_features = ConfigFeatures(self.path_config_features)
        self.pipeline_title = self.config_features.title
        self.features = self.config_features.features
        self.transforms = self.config_features.transforms
        self.config_transforms = self.config_features.config_transforms
        self.features_dtypes = self.config_features.features_dtypes

        self._config_subdirs = self.ConfigSubdirs(self.path_config_subdirs)
        self.outputs = self._config_subdirs.outputs
        self.outputs_subdir_levels = {x: [] for x in self.outputs}
        self.map_outputs_subdirectory_levels()
        self.outputs_root_dirs = {
            "data_processed": "./data/processed",
            "pipeline_artifacts": "./models",
        }
        self.join_outputs_directories()
        self.join_outputs_paths()

    class ConfigSubdirs:
        """
        A nested class within ConfigFeatureTransformsPipeline because,
        subdirectories configuration cannot exist without
        configurations of data and features.

        Expect subdirectories are keys
        which map to values in those other config objects.
        """

        def __init__(self, path):

            self._config0 = load_yaml.load_yaml_to_pydict(path)
            self.outputs = list(self._config0.keys())

    def map_outputs_subdirectory_levels(self):
        """
        Per output, map subdirectory level keys to actual names,
        via config objects' data.

        Proceed over subdirectory level keys, grouped by config.
        """

        config_subdirs = self._config_subdirs

        for output, subdir_lvls_by_cfg0 in config_subdirs._config0.items():

            # 'subdirectory_levels_by_config' key helps document yml
            subdir_lvls_by_cfg = subdir_lvls_by_cfg0[
                "subdirectory_levels_by_config"
            ]

            for cfg_ref, keys_to_lvls in subdir_lvls_by_cfg.items():

                cfg_ref_obj = self.__dict__[cfg_ref]

                for key in keys_to_lvls:

                    if key in self.__dict__:
                        if "title" in self.__dict__[key]:
                            subdir_lvl = self.__dict__[key]["title"]

                    elif "title" in cfg_ref_obj.__dict__:
                        subdir_lvl = cfg_ref_obj.__dict__["title"]

                    else:
                        subdir_lvl = key

                    self.outputs_subdir_levels[output] += [subdir_lvl]

    def join_outputs_directories(self):

        self.outputs_dir = {x: [] for x in self._config_subdirs.outputs}

        for o in self.outputs:
            self.outputs_dir[o] = Path(self.outputs_root_dirs[o]).joinpath(
                *self.outputs_subdir_levels[o]
            )

    def make_dirs(self):
        for o in self.outputs_dir:
            Path(self.outputs_dir[o]).mkdir(parents=True, exist_ok=True)

    def join_outputs_paths(self):
        """Filename may vary by train or not-train execution."""

        self.outputs_path = {}

        self.outputs_path["feature_transforms_pipeline"] = (
            self.outputs_dir["pipeline_artifacts"]
            / "feature_transforms_pipeline.pkl"
        )

        self.outputs_path["X_train"] = (
            self.outputs_dir["data_processed"] / "X_train"
        )
        self.outputs_path["X_train_attributes"] = (
            self.outputs_dir["data_processed"] / "X_train_attributes"
        )
        self.outputs_path["y_train"] = (
            self.outputs_dir["data_processed"] / "y_train"
        )

        self.outputs_path["X_test"] = self.outputs_dir["data_processed"] / "X"
        self.outputs_path["X_test_attributes"] = (
            self.outputs_dir["data_processed"] / "X_attributes"
        )
        self.outputs_path["y_test"] = self.outputs_dir["data_processed"] / "y"

        # short aliases for this run's elements
        if self.is_training_run:
            self.outputs_path["X"] = self.outputs_path["X_train"]
            self.outputs_path["X_attributes"] = self.outputs_path["X_train_attributes"]
            self.outputs_path['y'] = self.outputs_path["y_train"]
        else:
            self.outputs_path["X"] = self.outputs_path["X_test"]
            self.outputs_path["X_attributes"] = self.outputs_path["X_test_attributes"]
            self.outputs_path['y'] = self.outputs_path["y_test"]