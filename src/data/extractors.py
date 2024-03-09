import pandas as pd
from functools import reduce

from src.data.db import engine
from src.data.gcloud_client import gcloud_client


class ExtractorX:
    """
    Extract unified feature set X, possibly by integrating multiple datasets.
    Ensure features' data types.
    If applicable, create "derived" features: those which yield from a
    transform of source columns, not directly included in source.
    """

    def __init__(self, config, func_derive_features=None):

        # each extractor will need configuration details
        self.sources_X = config.sources_X
        self.filter = config.filter
        self.features_dtypes = config.features_dtypes
        self.features_numeric_types = [
            x for x, dtype in self.features_dtypes.items() if dtype in ["float"]
        ]
        self.func_derive_features = func_derive_features

        for source, details in self.sources_X.items():

            if details["storage_type"] == "database":
                self.sources_X[source]["extractor"] = self.extract_database
            elif details["storage_type"] == "google_sheet":
                self.sources_X[source]["extractor"] = self.extract_google_sheet

    def extract_database(self, details_source):

        query = f"SELECT * FROM {details_source['location']} WHERE is_valid"

        if details_source["do_filter"]:
            query += (
                f" AND {self.filter['field_min']} <= {self.filter['field']} "
                f" AND {self.filter['field']} <= {self.filter['field_max']}"
            )

        return pd.read_sql_query(query, engine)

    def extract_google_sheet(self, details_source):

        X = gcloud_client.open(details_source["location"])
        X = X.worksheet("request_form")
        X = pd.DataFrame(X.get_all_records())

        return X

    def extract(self):
        """Extract each dataset, then integrate into one."""

        self.datasets_X = [
            details["extractor"](details) for source, details in self.sources_X.items()
        ]

        X = reduce(lambda x, y: pd.merge(x, y, how="left"), self.datasets_X)

        # defer typing until datasets' integration because,
        # one feature could come from any of the sources
        X[self.features_numeric_types] = X[self.features_numeric_types].apply(
            pd.to_numeric, errors="coerce"
        )
        X = X.astype(self.features_dtypes)

        if self.func_derive_features:
            X = self.func_derive_features(X)

        return X


class ExtractorY:
    def __init__(self, config_data):

        self.config_data = config_data

        if self.config_data.source["storage_type"] == "database":
            self.extract = self.extract_database
        else:
            self.extract = None

    def extract_database(self):

        query = f"SELECT * FROM {self.config_data.source['Y']} "
        Y = pd.read_sql_query(query, engine)
        varname0 = self.config_data.outcome_definition["title"]
        Y = Y.rename(columns={varname0: "y"})

        return Y
