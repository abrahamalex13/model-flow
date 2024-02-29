from .SchemasConfigData import SchemaConfigData
from strictyaml import load


class ConfigData:
    """A quantitative model's data configuration."""

    def __init__(self, path):

        with open(path, "r") as yaml:
            self._config0 = load(yaml.read()).data

        self._config = SchemaConfigData(**self._config0).dict()

        self.is_training_run = self._config["is_training_run"]

        self.is_evaluation_run = self._config["is_evaluation_run"]

        self.source = self._config["source"]

        self.filter = self._config["filter"]
        self.filter_train = self._config["filter_train"]
        if self.is_training_run:
            self.filter = self.filter_train

        self.outcome_definition = self._config["outcome_definition"]
        self.dataset_attributes = self._config["dataset_attributes"]
