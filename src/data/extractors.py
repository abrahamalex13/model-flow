class QueryX:
    def __init__(self, config_data):

        self._config_data = config_data

        if self._config_data.source["storage_type"] == "database":
            self.set_query()

    def set_query(self):

        query0 = (
            "SELECT * "
            "FROM {source} "
            "WHERE "
            "{time_min} <= {time_field} "
            "AND {time_field} <= {time_max} "
            "AND NOT is_model_anomaly"
        )

        self.query = query0.format(
            source=self._config_data.source["X"],
            time_field=self._config_data.filters["time_field"],
            time_min=self._config_data.filters["time_min"],
            time_max=self._config_data.filters["time_max"],
        )


class QueryY:
    def __init__(self, config_data):

        self._config_data = config_data

        if self._config_data.source["storage_type"] == "database":
            self.set_query()

    def set_query(self):

        query0 = "SELECT * " "FROM {source} "
        self.query = query0.format(source=self._config_data.source["Y"],)
