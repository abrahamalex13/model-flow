import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
import pickle


class ConsolidateRareLevelsTransformer(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        features,
        thresh_nobs,
        overwrite_with,
        dir_save=None,
        name_save=None,
    ):

        self.features = features
        self.thresh_nobs = thresh_nobs
        self.overwrite_with = overwrite_with
        self.features_levels_keep = {x: [] for x in features}
        self.dir_save = dir_save
        self.name_save = name_save

        self.feature_names_in = None
        self.feature_names_out = None

    def fit(self, X, y=None):

        self.feature_names_in = X.columns

        for var in self.features:

            tbl = tabulate_frequency(X[var])

            does_level_freq_suffice = tbl["n"] >= self.thresh_nobs

            self.features_levels_keep[var] = tbl.loc[
                does_level_freq_suffice, :
            ].reset_index(drop=True)

        return self

    def transform(self, X, y=None):

        self.feature_names_in = X.columns
        self.feature_names_out = X.columns

        for var in self.features_levels_keep.keys():

            idx_overwrite = ~X[var].isin(
                self.features_levels_keep[var]["level"]
            )
            X.loc[idx_overwrite, var] = self.overwrite_with

        return X

    def save(self):
        pickle.dump(self, open(self.dir_save + self.name_save + ".pkl", "wb"))

    def get_feature_names_out(self, input_features=None):
        return self.feature_names_out


def tabulate_frequency(x):

    tbl = x.value_counts()

    tbl = pd.DataFrame({"level": tbl.index, "n": tbl}).reset_index(drop=True)

    tbl["fraction"] = tbl["n"] / sum(tbl["n"])
    tbl["fraction_cumul"] = np.cumsum(tbl["fraction"])

    return tbl
