def test_config_data_is_training_run(config_data):
    assert not config_data.is_training_run


def test_config_data_source(config_data):
    expected = {
        "storage_type": "database",
        "X": "analysis.draftees",
        "Y": "analysis.mlb_players_seasons_war",
    }

    assert config_data.source == expected


def test_config_data_filters_train(config_data):
    expected = {
        "title": "draftees_2013_thru_2017_train",
        "time_field": "year_season1",
        "time_min": 2013,
        "time_max": 2017,
    }
    assert config_data.filters_train == expected


def test_config_data_filters(config_data):
    expected = {
        "title": "draftees_2018",
        "time_field": "year_season1",
        "time_min": 2018,
        "time_max": 2018,
    }
    assert config_data.filters == expected


def test_config_data_outcome_definition(config_data):
    expected = {
        "title": "war_annualized_geq1_within_3_years_since_draft",
        "do_drop_na": False,
        "fillna_value": 0,
    }
    assert config_data.outcome_definition == expected
