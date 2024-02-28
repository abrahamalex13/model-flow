"""Subfield schema definitions must precede overall Config definition."""
from pydantic import BaseModel


class SchemaSource(BaseModel):

    storage_type: str
    X: str
    Y: str


class SchemaFilters(BaseModel):
    title: str
    time_field: str
    time_min: int
    time_max: int


class SchemaOutcomeDefinition(BaseModel):
    title: str
    do_drop_na: bool
    fillna_value: int


class SchemaConfigData(BaseModel):

    is_training_run: bool
    source: SchemaSource
    filters: SchemaFilters
    filters_train: SchemaFilters
    outcome_definition: SchemaOutcomeDefinition
    dataset_attributes: list
