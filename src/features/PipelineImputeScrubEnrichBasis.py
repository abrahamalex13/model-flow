import pandas as pd
from sklearn.compose import ColumnTransformer
from . import impute
from .scrub.PipelineScrub import PipelineScrub
from .enrich_basis.PipelineEnrichBasis import PipelineEnrichBasis


class PipelineImputeScrubEnrichBasis:
    """
    Encapsulate _sequentially dependent_ sklearn ColumnTransformer steps.
    For this purpose, a ColumnTransformer is insufficient: 
    that abstraction runs _parallel_  transformations.
    Example: impute _before_ scrubbing values, and then enrich column basis.
    """

    def __init__(self, config_transforms):

        self.config_transforms = config_transforms

    def fit(self, X, y):

        self.feature_names_in = X.columns

        self.pipeline_impute = ColumnTransformer(
            transformers=impute.compose_transforms_calls(
                self.config_transforms
            ),
            remainder="passthrough",
            verbose_feature_names_out=False,
        )
        self.pipeline_impute.fit(X)
        X = self.pipeline_impute.transform(X)
        X = pd.DataFrame(
            X, columns=self.pipeline_impute.get_feature_names_out()
        )

        self.pipeline_scrub = PipelineScrub(self.config_transforms)
        self.pipeline_scrub.fit(X)
        X = self.pipeline_scrub.transform(X)

        self.pipeline_enrich_basis = PipelineEnrichBasis(self.config_transforms)
        self.pipeline_enrich_basis.fit(X, y)

        return self

    def transform(self, X):

        if not all(self.feature_names_in == X.columns):
            raise Exception(
                "X for transform needs same column order as X into fit."
            )

        X = self.pipeline_impute.transform(X)
        X = pd.DataFrame(
            X, columns=self.pipeline_impute.get_feature_names_out()
        )
        X = self.pipeline_scrub.transform(X)
        X = self.pipeline_enrich_basis.transform(X)
        X = pd.DataFrame(
            X, columns=self.pipeline_enrich_basis.feature_names_out
        )

        return X
