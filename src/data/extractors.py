import pandas as pd
from src.data.db import engine
from src.data.gcloud_client import gcloud_client

class ExtractorX:
    """
    Data configuration always specifies extraction details.
    Features configuration conditionally specifies,
    when extracting live (not pre-validated) inputs.
    """
    def __init__(self, config):

        self.config_data = config.config_data
        self.features_dtypes = config.features_dtypes
        self.features_numeric_types = [
            x for x, dtype in self.features_dtypes.items()
            if dtype in ['float']
        ]

        if self.config_data.source["storage_type"] == "database":
            self.extract = self.extract_database
        elif self.config_data.source["storage_type"] == "google_sheet":
            self.extract = self.extract_google_sheet

    def extract_database(self):

        config_data = self.config_data

        self.query = (
            "SELECT * "
            f"FROM {config_data.source['X']} "
            "WHERE "
            f"{config_data.filters['time_min']} <= {config_data.filters['field']} "
            f"AND {config_data.filters['field']} <= {config_data.filters['time_max']} "
            "AND is_valid"
            )

        return pd.read_sql_query(self.query, engine)
    
    def extract_google_sheet(self):

        X = gcloud_client.open(self.config_data.source["X"])
        X = X.worksheet("request_form")
        X = pd.DataFrame(X.get_all_records())

        X[self.features_numeric_types] = (
            X[self.features_numeric_types]
            .apply(pd.to_numeric, errors='coerce')
            ) 
        X = X.astype(self.features_dtypes)

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