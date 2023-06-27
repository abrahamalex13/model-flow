from src.utils import load_yaml
from .SchemasConfigFeatures import SchemaConfigFeatures
from .SchemasTransforms import get_schema_transform


class ConfigFeatures:
    """
    Extract, validate an analyst-friendly specification
    of a feature transforms pipeline. The analyst describes:

    - Pre-configured functions available for feature transforms
        - Where possible, arguments already specified. 
        Feature-wise args are the exception (example, one-hot encode levels).
    - Each feature to be transformed
        - Per feature, each transform: a pre-configured function (above).

    """

    def __init__(self, path):

        self._config0 = load_yaml.load_yaml_to_pydict(path)

        self._config = SchemaConfigFeatures(**self._config0).dict()
        # note, map values previously parsed as '' have now become {}

        self.title = self._config["title"]

        self.features = list(self._config["features"].keys())

        self.set_features_dtypes()

        self.set_transforms_default_args()

        self.set_features_transforms()

        self.set_transforms_features()
        self.set_transforms()

    def set_features_dtypes(self):

        self.features_dtypes = {
            x: self._config["features"][x]["dtype"] for x in self.features
        }

    def set_transforms_default_args(self):
        """For easy reuse by the various features. Encapsulate validation."""

        self.transforms_default_args = {}

        for trfm, args0 in self._config["transforms_default_args"].items():

            self.transforms_default_args[trfm] = validate_transform_args(
                {trfm: args0}
            )

    def set_features_transforms(self):
        """
        Analyst-friendly declaration flow: feature, then its transforms.
        Within that nesting structure, validate transform's args.
        If transform specifies _any_ args, no use of default.
        If transform specifies _no_ args, then inject default.
        """

        self.features_transforms = {
            x: self._config["features"][x]["transforms"]
            for x in self.features
            if "transforms" in self._config["features"][x]
        }

        for feature, transforms in self.features_transforms.items():

            for trfm, args0 in transforms.items():

                if is_args_dict_nonnull(args0):
                    trfm_cln = validate_transform_args({trfm: args0})
                else:
                    trfm_cln = self.transforms_default_args[trfm]

                self.features_transforms[feature][trfm] = trfm_cln

    def set_transforms_features(self):
        """
        Code-friendly flow: proceed by transform, revising each feature set.
        Simply invert features_transforms structure. 
        """

        self.transforms_features = {}

        for feature, transforms_feature in self.features_transforms.items():

            for trfm in transforms_feature:

                if trfm not in self.transforms_features:
                    self.transforms_features[trfm] = []

                self.transforms_features[trfm] += [feature]

    def set_transforms(self):
        self.transforms = list(self.transforms_features.keys())


def validate_transform_args(transform_args: dict):
    """
    Because transforms are unknown until runtime, 
    transform's arguments validate during lower-level pass over input data.
    If all transform args omitted, then fall back on Schema defaults.
    """

    trfm = list(transform_args.keys())[0]
    schema = get_schema_transform(trfm)

    if is_args_dict_nonnull(transform_args[trfm]):
        return schema(**transform_args[trfm]).dict()
    else:
        return schema().dict()


def is_args_dict_nonnull(args_dict):
    return type(args_dict) is dict and len(args_dict) > 0