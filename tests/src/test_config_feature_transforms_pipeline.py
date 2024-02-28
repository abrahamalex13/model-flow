from pathlib import Path


def test_pipeline_outputs_subdir_levels(config_feature_transforms_pipeline):

    expected = {
        "data_processed": [
            "war_annualized_geq1_within_3_years_since_draft",
            "draftees_2013_thru_2017_train",
            "complete_wide",
        ],
        "pipeline_artifacts": [
            "war_annualized_geq1_within_3_years_since_draft",
            "draftees_2013_thru_2017_train",
            "complete_wide",
        ],
    }

    given = config_feature_transforms_pipeline.outputs_subdir_levels

    assert given == expected


def test_pipeline_outputs_dir(config_feature_transforms_pipeline):

    expected = {
        "data_processed": Path(
            "./data/processed"
            "/war_annualized_geq1_within_3_years_since_draft/"
            "draftees_2013_thru_2017_train/"
            "complete_wide/"
        ),
        "pipeline_artifacts": Path(
            "./models"
            "/war_annualized_geq1_within_3_years_since_draft/"
            "draftees_2013_thru_2017_train/"
            "complete_wide/"
        ),
    }

    given = config_feature_transforms_pipeline.outputs_dir

    assert given == expected
