import pandas as pd
import numpy as np
from sklearn import impute


def compose_transforms_calls(config_transforms):
    """
    Per sklearn ColumnTransformer argument `transformers`--list of tuples--
    one formatted element includes:
    - Transform name
    - Transformer/estimator call
    - Targeted columns
    """

    transformers = []

    for transform, details in config_transforms.items():

        spec = None

        if "impute_numeric" in transform:

            spec = (
                transform,
                impute.SimpleImputer(
                    missing_values=np.nan,
                    **details["args"],
                    copy=False
                ),
                details["features"],
            )

        elif "impute_str" in transform:

            spec = (
                transform,
                impute.SimpleImputer(
                    missing_values=None,
                    **details["args"],
                    copy=False
                ),
                details["features"],
            )

        if spec is not None:
            transformers += [spec]

    return transformers
