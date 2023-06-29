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

        self.set_transformers()

        self.set_features_transforms()

        self.set_transforms_features()

        self.set_transforms_calls()

        self.set_transforms()

    def set_features_dtypes(self):

        self.features_dtypes = {
            x: self._config["features"][x]["dtype"] for x in self.features
        }

    def set_transformers(self):
        """Pre-configure for reuse among features."""

        self.transformers = {}

        for trfm, args0 in self._config["transformers"].items():

            self.transformers[trfm] = validate_transformer_args({trfm: args0})

    def set_features_transforms(self):
        """
        Analyst-friendly declaration flow: feature, then its transforms.
        Within feature, validate each transform's args, because
        some transformers need feature-wise args not already validated.
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

                if has_args_filled(args0):
                    trfm_cln = validate_transformer_args({trfm: args0})
                else:
                    trfm_cln = self.transformers[trfm]

                self.features_transforms[feature][trfm] = trfm_cln

    def set_transforms_features(self):
        """
        Modeling workflow proceeds by transform (API), revising each feature set.
        Simply invert features_transforms structure. 
        """

        self.transforms_features = {}

        for feature, transforms_feature in self.features_transforms.items():

            for trfm in transforms_feature:

                if trfm not in self.transforms_features:
                    self.transforms_features[trfm] = []

                self.transforms_features[trfm] += [feature]

    def set_transforms_calls(self):
        """Per transform, pre-configure: features, function args."""

        self.transforms_calls = {}

        for trfm, features in self.transforms_features.items():
            
            self.transforms_calls[trfm] = {}
            self.transforms_calls[trfm]['features'] = features

        for trfm, args in self.transformers.items():

            self.transforms_calls[trfm]['args'] = args

            if trfm == "onehot_encode":
                
                categories = []
                
                for feature in self.transforms_features['onehot_encode']:
                    
                    if categories_add := self.features_transforms[feature]['onehot_encode'].get('categories'):
                        categories.append(categories_add) 

                if categories:
                    self.transforms_calls[trfm]['args']['categories'] = categories
                else:
                    self.transforms_calls[trfm]['args']['categories'] = 'auto'
         
    def set_transforms(self):
        self.transforms = list(self.transforms_features.keys())


def validate_transformer_args(transformer_args: dict):
    """
    Because transforms are unknown until runtime, 
    transform's arguments validate during lower-level pass over input data.
    If all transform args omitted, then fall back on Schema defaults.
    """

    trfm = list(transformer_args.keys())[0]
    schema = get_schema_transform(trfm)

    if has_args_filled(transformer_args[trfm]):
        return schema(**transformer_args[trfm]).dict()
    else:
        return schema().dict()


def has_args_filled(args_dict):
    return type(args_dict) is dict and len(args_dict) > 0
