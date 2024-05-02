import pandas as pd
import numpy as np
from sklearn import preprocessing


def specify_transformers(config_transforms):
    """
    A transformer specification includes (name, estimator, columns).
    That data model comes from sklearn ColumnTransformer's arg, `transformers`.
    """

    transformers = []

    for transform, details in config_transforms.items():

        spec = None

        if transform == "standard_scale":

            spec = (
                transform,
                preprocessing.StandardScaler(),
                details["features"],
            )

        if spec is not None:
            transformers += [spec]

    return transformers
