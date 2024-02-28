import pandas as pd


def integrate_XY(X, Y, config_outcome_definition):
    """
    Separate step for X-Y integration accommodates reality that,
    X and Y records not often stored together. Wait until integration step
    to fill missing Y because, missing Y may not be apparent until 
    alignment with X.
    """

    XY = pd.merge(X, Y, how="left")

    is_y_null = XY["y"].isnull()

    if not config_outcome_definition["do_drop_na"]:
        XY.loc[is_y_null, "y"] = config_outcome_definition["fillna_value"]

    elif config_outcome_definition["do_drop_na"]:
        XY = XY.loc[~ is_y_null].reset_index(drop=True)

    return XY
