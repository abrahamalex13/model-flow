def test_onehot_columns(X_transformed):

    expected = ["team_BOS", "team_PIT"]
    expected_intersect_columns = set(expected) & set(X_transformed.columns)
    has_expected_in_columns = set(expected) == expected_intersect_columns

    omitted = ["team_HOU"]
    if set(omitted) & set(X_transformed.columns):
        has_no_omitted = False
    else:
        has_no_omitted = True

    assert has_expected_in_columns and has_no_omitted
