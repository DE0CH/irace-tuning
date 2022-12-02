#!/usr/bin/env python
# coding: utf-8


from irace import irace
from irace.compatibility.config_space import convert_from_config_space
from irace.expressions import List
import numpy as np
from surrogate import convert_params_to_vec
import json
from ConfigSpace.read_and_write import pcs
from pyrfr import regression
from rpy2.robjects.packages import importr
from rpy2.robjects import StrVector
from utils import suppress_stdout, suppress_stderr
from multiprocessing import cpu_count

def get_training_experiment(instances):
    return instances[:len(instances) // 3]

def get_training_validation(instances):
    return instances[len(instances) // 3: len(instances) // 3 * 2]

def get_training_meta_validation(instances):
    return instances[len(instances) // 3:]

with open('./target_algorithms/surrogate/cplex_regions200/config_space.cplex_regions200.par10.random.pcs') as f:
    cs = pcs.read(f)

# Load the list of instance features
with open('./target_algorithms/surrogate/cplex_regions200/inst_feat_dict.cplex_regions200.par10.random.json') as f:
    instances_features = json.load(f)


instances = list(instances_features.keys())

# Load the random forest model
model = regression.binary_rss_forest()
model.load_from_binary_file('./target_algorithms/surrogate/cplex_regions200/pyrfr_model.cplex_regions200.par10.random.bin')

def target_irace(experiment, scenario):
    import json
    def surrogate_target_runner(experiment, scenario):
        with open('abc.json', 'w') as f:
            json.dump(experiment, f)
        with open('db.json', 'w') as f:
            json.dump(list(scenario.keys()), f)
        instance = experiment['instance']
        configuration = dict(experiment['configuration'])
        instance_feature = instances_features[instance]['__ndarray__']
        # Call the magic function to convert configurations to a vector
        encoded_configurations = convert_params_to_vec(configuration, cs)
        x = np.hstack([encoded_configurations, instance_feature])
        return dict(cost=model.predict(x))
  

    scenario = dict(
        instances = get_training_experiment(instances),
        maxExperiments = 1008,
        debugLevel = 0,
        parallel = cpu_count(),
        digits = 15,
        capping = 1,
        boundMax = 1000,
        cappingType = "median",
        boundType = "candidate",
        testType = "f-test",
        elitist = 1,
        logFile = ''
    )
    
    scenario.update(dict(experiment['configuration']))

    tuner = irace(scenario, convert_from_config_space(cs), surrogate_target_runner)

    best_configs = tuner.run()
    print(best_configs)


target_irace(dict(configuration = dict(
    capping = 0,
    cappingType = 'mean',
    boundType = 'candidate',
    testType = 'f-test',
    elitist = 0
)), None)