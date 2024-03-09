from pathlib import Path
from strictyaml import load

from src.data.ConfigData import ConfigData
from src.features.ConfigFeatures import ConfigFeatures


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
        self.is_evaluation_run = self.config_data.is_evaluation_run
        self.sources_X = self.config_data.sources_X
        self.source_Y = self.config_data.source_Y
        self.filter = self.config_data.filter
        self.filter_train = self.config_data.filter_train
        self.dataset_attributes = self.config_data.dataset_attributes

        self.config_features = ConfigFeatures(self.path_config_features)
        self.pipeline_title = self.config_features.title
        self.features = self.config_features.features
        self.transforms = self.config_features.transforms
        self.config_transforms = self.config_features.config_transforms
        self.features_dtypes = self.config_features.features_dtypes

        # under same subdirectory structure project-to-project,
        # yaml config management alternative seems unnecessarily complex.

        subdir_levels = [
            self.source_Y["title"],
            self.filter_train["title"],
            self.pipeline_title,
        ]

        models_subdir = Path("./models").joinpath(*subdir_levels)
        self.models_subdir = models_subdir
        self.feature_transforms_pipeline_path = (
            models_subdir / "feature_transforms_pipeline.pkl"
        )

        data_processed_subdir = Path("./data/processed").joinpath(*subdir_levels)
        self.data_processed_subdir = data_processed_subdir
        # defer suffixes for flexibility: pkl, csv, parquet, ...
        self.X_train_path = data_processed_subdir / "X_train"
        self.X_train_attributes_path = data_processed_subdir / "X_train_attributes"
        self.y_train_path = data_processed_subdir / "y_train"

        # in evaluation run, reference evaluation & train data separately
        if self.is_evaluation_run:
            self.X_evaluate_path = data_processed_subdir / "X_evaluate"
            self.X_evaluate_attributes_path = (
                data_processed_subdir / "X_evaluate_attributes"
            )
            self.y_evaluate_path = data_processed_subdir / "y_evaluate"

        # regardless of feature transform pipeline's run type,
        # prefer a single data reference name.
        # don't want to switch between, "X_train" or "X_evaluate".
        if self.is_training_run:
            self.X_path = self.X_train_path
            self.X_attributes_path = self.X_train_attributes_path
            self.y_path = self.y_train_path
        elif self.is_evaluation_run:
            self.X_path = self.X_evaluate_path
            self.X_attributes_path = self.X_evaluate_attributes_path
            self.y_path = self.y_evaluate_path
        else:
            self.X_path = data_processed_subdir / "X"
            self.X_attributes_path = data_processed_subdir / "X_attributes"
            self.y_path = data_processed_subdir / "y"

    def make_dirs(self):
        for dir in [self.data_processed_subdir, self.models_subdir]:
            dir.mkdir(parents=True, exist_ok=True)
