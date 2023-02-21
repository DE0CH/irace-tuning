from ConfigSpace.read_and_write import pcs
import json
from pyrfr import regression
from utils import filter_nan
import numpy as np
from surrogate import convert_params_to_vec


class SurrogateModel:
    def __init__(self, pcs_fp, instance_features_fp, rss_fs) -> None:
        with open(pcs_fp) as f:
            self.cs = pcs.read(f)
        with open(instance_features_fp) as f:
            self.instances_features = json.load(f)
        self.instances = list(self.instances_features.keys())
        self.model = regression.binary_rss_forest()
        self.model.load_from_binary_file(rss_fs)

    def predict_surrogate(self, configuration, instance):
        """
        Predict the performance of a giving configuration on an instance using the surrogate model.

        Args:
            configuration: A dictionary of configuration
            instance: A string of the instance name

        Returns:
            The performance as a floating number by the surrogate model
        """
        instance_feature = self.instances_features[instance]["__ndarray__"]
        # Call the magic function to convert configurations to a vector
        encoded_configurations = convert_params_to_vec(
            filter_nan(configuration), self.cs
        )
        x = np.hstack([encoded_configurations, instance_feature])
        y = self.model.predict(x)
        return 10**y
