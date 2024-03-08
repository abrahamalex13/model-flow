"""Subfield schema definitions must precede overall Config definition."""

from pydantic import BaseModel
from typing import Literal, Dict


class SchemaSource(BaseModel):

    storage_type: Literal["database", "google_sheet"]
    X: str
    Y: str


class SchemaSourceX(BaseModel):

    storage_type: Literal["database", "google_sheet"]
    location: str


class SchemaSourceY(BaseModel):

    storage_type: Literal["database", "google_sheet"]
    location: str
    title: str
    do_drop_na: bool
    fillna_value: float


class SchemaFilter(BaseModel):
    title: str
    field: str
    field_min: int
    field_max: int


class SchemaOutcomeDefinition(BaseModel):
    title: str
    do_drop_na: bool
    fillna_value: float


class SchemaConfigData(BaseModel):

    is_training_run: bool
    is_evaluation_run: bool
    source: SchemaSource
    sources_X: Dict[str, SchemaSourceX]
    source_Y: SchemaSourceY
    filter: SchemaFilter
    filter_train: SchemaFilter
    outcome_definition: SchemaOutcomeDefinition
    dataset_attributes: list
