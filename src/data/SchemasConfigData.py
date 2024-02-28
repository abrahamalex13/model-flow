"""Subfield schema definitions must precede overall Config definition."""
from pydantic import BaseModel
from typing import Literal


class SchemaSource(BaseModel):

    storage_type: Literal["database", "google_sheet"]
    X: str
    Y: str


class SchemaFilter(BaseModel):
    title: str
    field: str
    field_min: int
    field_max: int


class SchemaOutcomeDefinition(BaseModel):
    title: str
    do_drop_na: bool
    fillna_value: int


class SchemaConfigData(BaseModel):

    is_training_run: bool
    source: SchemaSource
    filter: SchemaFilter
    filter_train: SchemaFilter
    outcome_definition: SchemaOutcomeDefinition
    dataset_attributes: list
