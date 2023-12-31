from .SchemasConfigData import SchemaConfigData
from src.utils import load_yaml


class ConfigData:
    """A quantitative model's data configuration."""

    def __init__(self, path):

        self._config0 = load_yaml.load_yaml_to_pydict(path)

        self._config = SchemaConfigData(**self._config0).dict()

        self.is_training_run = self._config["is_training_run"]
        self.source = self._config["source"]

        self.filters = self._config["filters"]
        self.filters_train = self._config["filters_train"]
        if self.is_training_run:
            self.filters = self.filters_train

        self.outcome_definition = self._config["outcome_definition"]
        self.dataset_attributes = self._config["dataset_attributes"]
