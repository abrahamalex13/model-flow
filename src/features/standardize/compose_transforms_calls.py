import pandas as pd
import numpy as np
from sklearn import preprocessing


def compose_transforms_calls(config_transforms):
    """
    Per sklearn ColumnTransformer argument `transformers`--list of tuples--
    one formatted element includes:
    - Transform name
    - Transformer/estimator call
    - Targeted columns
    """

    transformers = []

    for transform in config_transforms.keys():

        cfg_transform = config_transforms[transform]

        spec = None

        if transform == "standard_scale":

            spec = (
                transform,
                preprocessing.StandardScaler(),
                cfg_transform["features"],
            )

        if spec is not None:
            transformers += [spec]

    return transformers
