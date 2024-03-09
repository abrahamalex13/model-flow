from pydantic import BaseModel
from typing import Dict, Literal, Union


class SchemaFeature(BaseModel):
    """
    High-abstraction data model of `feature` configuration.
    Interesting details lie in transforms, unknown until config file read.
    For now, confirm `transforms` structure:
    {transform1: {arg1: value1, ...}, transform2: {arg1: value1}, ...}.
    Defer validation of each transform's args.
    """

    dtype: Literal["float", "int", "string"]
    # a transform's details may be key-value pairs, or blank
    transforms: Dict[str, Union[dict, Literal[""]]]


class SchemaConfigFeatures(BaseModel):
    title: str
    transformers: Dict[str, dict]
    features: Dict[str, SchemaFeature]
