import pandas as pd
from sklearn.compose import ColumnTransformer

from . import impute
from .scrub.PipelineScrub import PipelineScrub
from .enrich_basis.PipelineEnrichBasis import PipelineEnrichBasis
from . import standardize


class StaggeredPipeline:
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
        config_transforms = self.config_transforms

        impute_transformers = impute.specify_transformers(config_transforms)
        if impute_transformers:
            pipeline_impute = ColumnTransformer(
                transformers=impute_transformers,
                remainder="passthrough",
                verbose_feature_names_out=False,
            )
            pipeline_impute.fit(X)
            X = pipeline_impute.transform(X)
            X = pd.DataFrame(X, columns=pipeline_impute.get_feature_names_out())
        else:
            pipeline_impute = None
        self.pipeline_impute = pipeline_impute

        pipeline_scrub = PipelineScrub(config_transforms)
        pipeline_scrub.fit(X)
        X = pipeline_scrub.transform(X)
        self.pipeline_scrub = pipeline_scrub

        pipeline_enrich_basis = PipelineEnrichBasis(config_transforms)
        pipeline_enrich_basis.fit(X, y)
        X = pipeline_enrich_basis.transform(X)
        X = pd.DataFrame(X, columns=pipeline_enrich_basis.feature_names_out)
        self.pipeline_enrich_basis = pipeline_enrich_basis

        pipeline_standardize = ColumnTransformer(
            transformers=standardize.compose_transforms_calls(config_transforms),
            remainder="passthrough",
            verbose_feature_names_out=False,
        )
        pipeline_standardize.fit(X)
        self.pipeline_standardize = pipeline_standardize

        return self

    def transform(self, X):

        if not all(self.feature_names_in == X.columns):
            raise Exception("New X needs same column order as trained-on X.")

        X = self.pipeline_impute.transform(X)
        X = pd.DataFrame(X, columns=self.pipeline_impute.get_feature_names_out())

        X = self.pipeline_scrub.transform(X)

        X = self.pipeline_enrich_basis.transform(X)
        X = pd.DataFrame(X, columns=self.pipeline_enrich_basis.feature_names_out)

        X = self.pipeline_standardize.transform(X)
        X = pd.DataFrame(X, columns=self.pipeline_standardize.get_feature_names_out())

        return X
