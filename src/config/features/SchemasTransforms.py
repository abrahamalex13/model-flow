from pydantic import BaseModel, confloat


class SchemaStandardScale(BaseModel):
    with_mean: bool = True
    with_std: bool = True


class SchemaImputeNumeric(BaseModel):
    strategy: str = "median"
    add_indicator: bool = True


class SchemaImputeNumericFlag(BaseModel):
    strategy: str = "most_frequent"
    fill_value: int = None
    add_indicator: bool = True


class SchemaImputeString(BaseModel):
    strategy: str = "constant"
    fill_value: str = "NULL"
    add_indicator: bool = False


class SchemaConsolidateRareLevels(BaseModel):
    thresh_nobs: int = 10
    overwrite_with: str = "OTHER"


class SchemaBetaPrior(BaseModel):
    family: str = "beta"
    alpha: int
    beta: int


class SchemaTargetEncodeBetaBinomial(BaseModel):
    n_cv_splits: int
    target_prior_distribution: SchemaBetaPrior


class SchemaNormalPrior(BaseModel):
    family: str = "normal"
    mu: float
    variance_mu: confloat(gt=0)


class SchemaTargetEncodeNormal(BaseModel):
    n_cv_splits: int
    target_prior_distribution: SchemaNormalPrior


class SchemaOnehotEncode(BaseModel):
    categories: list = []


def get_schema_transform(transform):

    if transform == "consolidate_rare_levels":
        return SchemaConsolidateRareLevels
    elif transform == "target_encode_beta_binomial":
        return SchemaTargetEncodeBetaBinomial
    elif transform == "target_encode_normal":
        return SchemaTargetEncodeNormal
    elif "impute_numeric" in transform:
        return SchemaImputeNumeric
    elif transform == "impute_numeric_flag":
        return SchemaImputeNumericFlag
    elif transform == "impute_string":
        return SchemaImputeString
    elif transform == "onehot_encode":
        return SchemaOnehotEncode
    elif transform == "standard_scale":
        return SchemaStandardScale

    else:
        raise Exception(transform + " schema not supported by ConfigFeatures.")
