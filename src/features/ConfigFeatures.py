from collections import defaultdict
from strictyaml import load
from .SchemasConfigFeatures import SchemaConfigFeatures
from .SchemasTransforms import get_schema_transform


class ConfigFeatures:
    """
    Bundle arguments for pass to transformer API, where a bundle contains:
    - transformer alias
    - features
    - transformer arguments

    "Bundling" action involves transformation of an analyst-friendly 
    configuration file (`features_config`). Primary steps:

    - validate transform-argument pairs, from config's `transformers` section
    - to above, integrate features-transforms inputs, from config's `features`
        - Reshape to transform-features structure. Consider transforms
        with alias in `transformers` config, that impact some feature(s).
        - Integrate (transform-features) with (transform-arguments),
        overlaying featurewise arguments where applicable. 
    """

    def __init__(self, path):

        with open(path, "r") as yaml:
            self._config0 = load(yaml.read()).data

        self._config = SchemaConfigFeatures(**self._config0).dict()

        self.title = self._config["title"]

        self.features = list(self._config["features"].keys())

        # flattened structures are easier to work with later
        self.features_dtypes = {
            x: self._config["features"][x]["dtype"] for x in self.features
        }
        self.features_transforms = {
            x: self._config["features"][x]["transforms"] for x in self.features
        }

        # working toward transform-features-arguments bundle,
        # it's easier to create-then-compose pieces.

        self.transformers = {}
        for transform, kwargs0 in self._config["transformers"].items():
            schema_model = get_schema_transform(transform)
            self.transformers[transform] = schema_model(**kwargs0).dict() 

        self.set_transforms_features()

        self.set_config_transforms()

        self.transforms = list(self.config_transforms.keys())
        

    def set_transforms_features(self):

        # defaultdict allows "just-in-time" addition of transform keys.
        # need not specify all `transformer` aliases in advance,
        # then prune transforms that don't appear among `features` 
        self.transforms_features = defaultdict(lambda: [])
        transforms_available = self.transformers.keys()
        
        for feature, transforms in self.features_transforms.items():
            
            for trfm in transforms.keys() & transforms_available:
                self.transforms_features[trfm] += [feature]

        return self
    
    def set_config_transforms(self):

        config_transforms = {}
        for trfm, features in self.transforms_features.items():
            
            config_transforms[trfm] = {}
            config_transforms[trfm]["features"] = features
            config_transforms[trfm]["args"] = self.transformers[trfm]

        # special case overwrites
        has_onehot_featurewise = (
            "onehot_encode" in config_transforms and 
            self.transformers["onehot_encode"]["categories"] == "featurewise"
            )
        if has_onehot_featurewise:

            config_transforms["onehot_encode"]["args"]["categories"] = []

            for feature in self.transforms_features["onehot_encode"]:

                feature_levels = self.features_transforms[feature]["onehot_encode"]["categories"]

                # list of lists is expected input from scikit-learn onehot encoder
                config_transforms["onehot_encode"]["args"]["categories"].append(feature_levels)

        self.config_transforms = config_transforms

        return self