from pathlib import Path


def test_data_processed_subdir(config_feature_transforms_pipeline):

    expected = Path(
        "./data/processed"
        "/war_annualized_geq1_within_3_years_since_draft/"
        "draftees_2013_thru_2017_train/"
        "complete_wide/"
    )

    assert config_feature_transforms_pipeline.data_processed_subdir == expected


def test_models_subdir(config_feature_transforms_pipeline):

    expected = Path(
        "./models"
        "/war_annualized_geq1_within_3_years_since_draft/"
        "draftees_2013_thru_2017_train/"
        "complete_wide/"
    )

    assert config_feature_transforms_pipeline.models_subdir == expected
