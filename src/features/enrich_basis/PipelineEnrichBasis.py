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

        self.transformers = specify_transformers(self.config_transforms)

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
            raise Exception("X for transform needs same column order as X into fit.")

        X = self.pipeline.transform(X)
        self.feature_names_out = self.pipeline.get_feature_names_out()

        return X


def specify_transformers(config_transforms):
    """
    Where one element includes:
        - Transform name
        - Transformer object (with tuning parameters)
        - Transformed columns

    Element specification follows from
    sklearn ColumnTransformer argument, `transformers`.

    """

    transformers = []

    for transform, details in config_transforms.items():

        spec = None

        if "target_encode" in transform:

            spec = (
                transform,
                TargetEncodeTransformer(
                    features=details["features"], **details["args"]
                ),
                details["features"],
            )

        elif transform == "onehot_encode":

            spec = (
                transform,
                preprocessing.OneHotEncoder(
                    categories=details["args"]["categories"],
                    sparse_output=False,
                    handle_unknown="ignore",
                ),
                details["features"],
            )

        if spec is not None:
            transformers += [spec]

    return transformers
