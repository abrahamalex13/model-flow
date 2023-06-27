"""
When a nested schema's low-level-of-abstraction elements
possess different type across runs, 
validate in steps, hierarchically. 
This approach seems more readable versus complex looping.

What higher-level elements have known type
in a particular validation step?
"""

from pydantic import BaseModel
from typing import Dict, Literal


class SchemaFeature(BaseModel):
    dtype: Literal["numeric", "categorical"]
    # particular transforms unknown in advance
    # descend hierarchically in next validation step
    transforms: Dict[str, dict]


class SchemaConfigFeatures(BaseModel):
    title: str
    transforms_default_args: Dict[str, dict]
    features: Dict[str, SchemaFeature]
