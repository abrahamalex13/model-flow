def test_features(config_features):
    expected = ["overall_pick", "team"]
    given = config_features.features
    assert given == expected


def test_features_dtypes(config_features):
    expected = {"overall_pick": "numeric", "team": "categorical"}
    given = config_features.features_dtypes
    assert given == expected


def test_features_transforms(config_features):
    expected = {
        "overall_pick": {"standard_scale": {}},
        "team": {
            "consolidate_rare_levels": {},
            "onehot_encode": {"categories": ["BOS", "PIT"]},
            "target_encode_beta_binomial": {},
        },
    }
    given = config_features.features_transforms
    assert given == expected


def test_consolidate_rare_levels_default_args(config_features):
    expected = {"thresh_nobs": 10, "overwrite_with": "OTHER"}
    given = config_features.transformers["consolidate_rare_levels"]
    assert given == expected


def test_transforms(config_features):
    expected = set(
        [
            "standard_scale",
            "consolidate_rare_levels",
            "onehot_encode",
            "target_encode_beta_binomial",
        ]
    )
    given = set(config_features.transforms)
    assert given == expected


def test_transforms_features(config_features):
    expected = {
        "standard_scale": ["overall_pick"],
        "consolidate_rare_levels": ["team"],
        "onehot_encode": ["team"],
        "target_encode_beta_binomial": ["team"],
    }
    given = config_features.transforms_features
    assert given == expected


def test_config_transforms(config_features):

    expected = {
        "standard_scale": {
            "features": ["overall_pick"],
            "args": {"with_mean": True, "with_std": True},
        },
        "consolidate_rare_levels": {
            "features": ["team"],
            "args": {"thresh_nobs": 10, "overwrite_with": "OTHER"},
        },
        "onehot_encode": {
            "features": ["team"],
            "args": {"categories": [["BOS", "PIT"]]},
        },
        "target_encode_beta_binomial": {
            "features": ["team"],
            "args": {
                "n_cv_splits": 3,
                "target_prior_distribution": {"alpha": 1, "beta": 24, "family": "beta"},
            },
        },
    }

    given = config_features.config_transforms

    assert given == expected
