from .SchemasConfigFeatures import SchemaConfigFeatures
from .SchemasTransforms import get_schema_transform
from strictyaml import load


class ConfigFeatures:
    """
    Bundle arguments for pass to transformer API, where a bundle contains:
    - transformer alias
    - features
    - transformer arguments

    "Bundling" action involves transformation of an analyst-friendly 
    configuration file (`features_config`).
    """

    def __init__(self, path):

        with open(path, "r") as yaml:
            self._config0 = load(yaml.read()).data

        self._config = SchemaConfigFeatures(**self._config0).dict()

        self.title = self._config["title"]

        self.features = list(self._config["features"].keys())

        self.features_dtypes = {
            x: self._config["features"][x]["dtype"] for x in self.features
        }

        self.transformers = {}
        for transform, kwargs0 in self._config["transformers"].items():
            schema_model = get_schema_transform(transform)
            self.transformers[transform] = schema_model(**kwargs0).dict() 

        self.set_features_transforms()

        self.set_transforms_features()

        self.set_config_transforms()

        self.transforms = list(self.transforms_features.keys())


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

        for feature, transforms in self.features_transforms.items():

            for trfm in transforms:

                if trfm not in self.transforms_features:
                    self.transforms_features[trfm] = []

                self.transforms_features[trfm] += [feature]

    def set_config_transforms(self):
        """Per transform, pre-configure: features, function args."""

        self.config_transforms = {}

        for trfm, features in self.transforms_features.items():
            
            self.config_transforms[trfm] = {}
            self.config_transforms[trfm]['features'] = features

        for trfm in self.config_transforms:
            
            self.config_transforms[trfm]['args'] = self.transformers[trfm]

            if trfm == "onehot_encode":
                
                categories = []
                
                for feature in self.transforms_features['onehot_encode']:
                    
                    if categories_add := self.features_transforms[feature]['onehot_encode'].get('categories'):
                        categories.append(categories_add) 

                if categories:
                    self.config_transforms[trfm]['args']['categories'] = categories
                else:
                    self.config_transforms[trfm]['args']['categories'] = 'auto'


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
