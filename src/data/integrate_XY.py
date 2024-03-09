import pandas as pd


def integrate_XY(X, Y, config_source_Y):
    """
    Separate step for X-Y integration accommodates reality that,
    X and Y records not often stored together. Wait until integration step
    to fill missing Y because, missing Y may not be apparent until
    alignment with X.
    """

    XY = pd.merge(X, Y, how="left")

    if not config_source_Y["do_drop_na"]:
        XY.loc[XY["y"].isnull(), "y"] = config_source_Y["fillna_value"]

    elif config_source_Y["do_drop_na"]:
        XY = XY.loc[XY["y"].notnull()].reset_index(drop=True)

    return XY
