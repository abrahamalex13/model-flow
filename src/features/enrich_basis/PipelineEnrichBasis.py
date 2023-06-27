from sklearn.compose import ColumnTransformer
from sklearn import preprocessing
import pandas as pd

from targetencode import TargetEncodeTransformer


class PipelineEnrichBasis:
    """
    Encapsulate transforms which enrich data's representation for modeling.
    """

    def __init__(self, transforms_args_adapters):

        TRANSFORMS_MENU = [
            "target_encode_beta_binomial",
            "target_encode_normal",
            "onehot_encode",
            "standard_scale",
        ]
        self.transforms_args_adapters = {}

        for trfm in TRANSFORMS_MENU:
            if trfm in transforms_args_adapters:
                self.transforms_args_adapters[trfm] = transforms_args_adapters[
                    trfm
                ]

        self.transformers = compose_transforms_calls(
            self.transforms_args_adapters
        )

    def fit(self, X, y):

        self.feature_names_in = X.columns

        self.pipeline = ColumnTransformer(
            transformers=self.transformers,
            remainder="passthrough",
            verbose_feature_names_out=False,
        )
        self.pipeline.fit(X, y)

        return self

    def transform(self, X):

        if not all(self.feature_names_in == X.columns):
            raise Exception(
                "X for transform needs same column order as X into fit."
            )

        X = self.pipeline.transform(X)
        self.feature_names_out = self.pipeline.get_feature_names_out()

        return X


def compose_transforms_calls(transforms_args_adapters):
    """
    Where one element includes:
        - Transform name
        - Transformer object (with tuning parameters)
        - Transformed columns

    Element specification follows from
    sklearn ColumnTransformer argument, `transformers`.

    """

    transformers = []

    for transform in transforms_args_adapters.keys():

        cfg_transform = transforms_args_adapters[transform]

        spec = None

        if "target_encode" in transform:

            spec = (
                transform,
                TargetEncodeTransformer(
                    features=cfg_transform.features,
                    **cfg_transform.tune_parameters
                ),
                cfg_transform.features,
            )

        elif transform == "onehot_encode":

            spec = (
                transform,
                preprocessing.OneHotEncoder(
                    categories=cfg_transform.categories,
                    sparse=False,
                    handle_unknown="ignore",
                ),
                cfg_transform.features,
            )

        elif transform == "standard_scale":

            spec = (
                transform,
                preprocessing.StandardScaler(),
                cfg_transform.features,
            )

        if spec is not None:
            transformers += [spec]

    return transformers
