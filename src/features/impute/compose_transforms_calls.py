import pandas as pd
import numpy as np
from sklearn import impute


def specify_transformers(config_transforms):
    """
    A transformer specification includes (name, estimator, columns).
    That data model comes from sklearn ColumnTransformer's arg, `transformers`.
    """

    transformers = []

    for transform, details in config_transforms.items():

        spec = None

        if "impute_numeric" in transform:

            spec = (
                transform,
                impute.SimpleImputer(
                    missing_values=np.nan, **details["args"], copy=False
                ),
                details["features"],
            )

        elif "impute_str" in transform:

            spec = (
                transform,
                impute.SimpleImputer(
                    missing_values=None, **details["args"], copy=False
                ),
                details["features"],
            )

        if spec is not None:
            transformers += [spec]

    return transformers
