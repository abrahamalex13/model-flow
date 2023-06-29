import pandas as pd

from .ConsolidateRareLevelsTransformer import ConsolidateRareLevelsTransformer


class PipelineScrub:
    """
    Encapsulate operations which cleanse/scrub/overwrite initial data values.
    """

    def __init__(self, transforms_args_adapters):

        TRANSFORMS_MENU = ["consolidate_rare_levels"]
        self.transforms_args_adapters = {}

        for trfm in TRANSFORMS_MENU:
            if trfm in transforms_args_adapters:
                self.transforms_args_adapters[trfm]: transforms_args_adapters[
                    trfm
                ]

        self.transforms = list(self.transforms_args_adapters.keys())

    def fit(self, X, y=None):

        self.feature_names_in = X.columns

        if "consolidate_rare_levels" in self.transforms:

            cfg_trfm = self.transforms_args_adapters["consolidate_rare_levels"]

            self.consolidate_rare_levels_pipeline = (
                ConsolidateRareLevelsTransformer(
                    features=cfg_trfm["features"], **cfg_trfm["args"]
                )
            )
            self.consolidate_rare_levels_pipeline.fit(X)

        return self

    def transform(self, X):

        if not all(self.feature_names_in == X.columns):
            raise Exception(
                "X for transform needs same column order as X into fit."
            )

        if "consolidate_rare_levels" in self.transforms:
            X = self.consolidate_rare_levels_pipeline.transform(X)

        return X
