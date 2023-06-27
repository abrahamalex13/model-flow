class TransformArgsAdapter:
    """
    Convert feature-then-transforms data structure (analyst-friendly),
    to transform-which-features structure--for call to _transform_ function.

    **With exception of few transforms, `default_args` set
    completely specifies a transform, over each targeted feature.
    Featurewise args necessary only when feature values are args.**
    """

    def __init__(self, config_features, transform):

        self._config_features = config_features

        self.transform = transform

        self.features = config_features.transforms_features[transform]

        self.tune_parameters = config_features.transforms_default_args[
            transform
        ]

        if transform == "onehot_encode":

            self.extract_featurewise_args()
            self.format_onehot_categories()

    def extract_featurewise_args(self):

        self.featurewise_args = {}

        for ftr in self.features:

            transforms_feature = self._config_features.features_transforms[ftr]

            self.featurewise_args[ftr] = transforms_feature[self.transform]

    def format_onehot_categories(self):

        self.categories = []

        feature_has_categories = [
            len(self.featurewise_args[ftr]["categories"]) > 0
            for ftr in self.features
        ]

        if all(feature_has_categories):
            for ftr in self.features:
                self.categories.append(self.featurewise_args[ftr]["categories"])

        else:
            self.categories = "auto"
