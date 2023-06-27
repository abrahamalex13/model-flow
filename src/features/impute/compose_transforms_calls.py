import pandas as pd
import numpy as np
from sklearn import impute


def compose_transforms_calls(configs_transformers):
    """
    Per sklearn ColumnTransformer argument `transformers`--list of tuples--
    one formatted element includes:
    - Transform name
    - Transformer/estimator call
    - Targeted columns
    """

    transformers = []

    for transform in configs_transformers.keys():

        spec = None

        if "impute_numeric" in transform:

            cfg_transform = configs_transformers[transform]

            spec = (
                transform,
                impute.SimpleImputer(
                    missing_values=np.nan,
                    **cfg_transform.tune_parameters,
                    copy=False
                ),
                cfg_transform.features,
            )

        elif "impute_str" in transform:

            cfg_transform = configs_transformers[transform]

            spec = (
                transform,
                impute.SimpleImputer(
                    missing_values=None,
                    **cfg_transform.tune_parameters,
                    copy=False
                ),
                cfg_transform.features,
            )

        if spec is not None:
            transformers += [spec]

    return transformers
