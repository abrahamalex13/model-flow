def test_is_training_run(config_data):
    assert not config_data.is_training_run


def test_is_evaluation_run(config_data):
    assert config_data.is_evaluation_run


def test_sources_X(config_data):
    expected = {
        "ui": {"storage_type": "google_sheet", "location": "ui"},
        "batch_war": {"storage_type": "database", "location": "batch_war"},
    }

    assert config_data.sources_X == expected


def test_source_Y(config_data):
    expected = {
        "storage_type": "database",
        "location": "y",
        "title": "war_annualized_geq1_within_3_years_since_draft",
        "do_drop_na": False,
        "fillna_value": 0,
    }

    assert config_data.source_Y == expected


def test_data_source(config_data):
    expected = {
        "storage_type": "database",
        "X": "analysis.draftees",
        "Y": "analysis.mlb_players_seasons_war",
    }

    assert config_data.source == expected


def test_data_filter_train(config_data):
    expected = {
        "title": "draftees_2013_thru_2017_train",
        "field": "year_season1",
        "field_min": 2013,
        "field_max": 2017,
    }
    assert config_data.filter_train == expected


def test_data_filter(config_data):
    expected = {
        "title": "draftees_2018",
        "field": "year_season1",
        "field_min": 2018,
        "field_max": 2018,
    }
    assert config_data.filter == expected


def test_dataset_attributes(config_data):
    expected = ["year", "name", "years_old"]

    assert config_data.dataset_attributes == expected
