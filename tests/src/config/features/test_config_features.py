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
        "overall_pick": {
            "standard_scale": {"with_mean": True, "with_std": True}
        },
        "team": {
            "consolidate_rare_levels": {
                "overwrite_with": "OTHER",
                "thresh_nobs": 10,
            },
            "onehot_encode": {"categories": ["BOS", "PIT"]},
            "target_encode_beta_binomial": {
                "n_cv_splits": 3,
                "target_prior_distribution": {
                    "alpha": 1,
                    "beta": 24,
                    "family": "beta",
                },
            },
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
