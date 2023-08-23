from sklearn.compose import ColumnTransformer
from sklearn import preprocessing
import pandas as pd

from targetencode import TargetEncodeTransformer


class PipelineEnrichBasis:
    """
    Encapsulate transforms which enrich data's representation for modeling.
    """

    def __init__(self, config_transforms):

        TRANSFORMS_MENU = [
            "target_encode_beta_binomial",
            "target_encode_normal",
            "onehot_encode",
        ]
        self.config_transforms = {}

        for trfm in TRANSFORMS_MENU:
            if trfm in config_transforms:
                self.config_transforms[trfm] = config_transforms[trfm]

        self.transformers = compose_transforms_calls(self.config_transforms)

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


def compose_transforms_calls(config_transforms):
    """
    Where one element includes:
        - Transform name
        - Transformer object (with tuning parameters)
        - Transformed columns

    Element specification follows from
    sklearn ColumnTransformer argument, `transformers`.

    """

    transformers = []

    for transform in config_transforms.keys():

        cfg_transform = config_transforms[transform]

        spec = None

        if "target_encode" in transform:

            spec = (
                transform,
                TargetEncodeTransformer(
                    features=cfg_transform["features"],
                    **cfg_transform["args"]
                ),
                cfg_transform["features"],
            )

        elif transform == "onehot_encode":

            spec = (
                transform,
                preprocessing.OneHotEncoder(
                    categories=cfg_transform["categories"],
                    sparse=False,
                    handle_unknown="ignore",
                ),
                cfg_transform.features,
            )

        if spec is not None:
            transformers += [spec]

    return transformers
