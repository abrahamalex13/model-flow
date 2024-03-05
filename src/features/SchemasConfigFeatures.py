from pydantic import BaseModel
from typing import Dict, Literal


class SchemaFeature(BaseModel):
    """
    High-abstraction data model of `feature` configuration.
    Interesting details lie in transforms, unknown until config file read. 
    For now, confirm `transforms` structure: 
    {transform1: {arg1: value1, ...}, transform2: {arg1: value1}, ...}.
    Defer validation of each transform's args. 
    """
    dtype: Literal["numeric", "categorical"] 
    transforms: Dict[str, dict]


class SchemaConfigFeatures(BaseModel):
    title: str
    transformers: Dict[str, dict]
    features: Dict[str, SchemaFeature]
