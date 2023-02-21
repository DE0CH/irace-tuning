#!/usr/bin/env python
# coding: utf-8


from irace import irace, Parameters, Param, Categorical, Symbol
from irace.compatibility.config_space import convert_from_config_space
import numpy as np
from surrogate import convert_params_to_vec
import json
from ConfigSpace.read_and_write import pcs
from pyrfr import regression
from multiprocessing import cpu_count
from utils import suppress_stdout
import pandas as pd
from utils import filter_nan
from config import (
    get_instances_training,
    get_instances_validation,
)
from models import SurrogateModel
from config import SurrogateType, default_serializer

threads = cpu_count()

def target_irace(experiment, scenario):
    '''
    The target runner for the irace being tuned
    '''

    instance = experiment['instance']
    model, training_instances, validation_instances, boundMax, maxTimeOrExperiments, surrogateType = instance
    parameters = convert_from_config_space(model.cs)

    def surrogate_target_runner(experiment, scenario):
        instance = experiment['instance']
        bound = experiment['bound']
        configuration = dict(experiment['configuration'])
        cost = model.predict_surrogate(configuration, instance)
        if surrogateType == SurrogateType.RUNTIME:
            return dict(cost=cost, time=min(bound + 1, cost))
        elif surrogateType == SurrogateType.QUALITY:
            return dict(cost=cost)
        else:
            raise NotImplementedError()
  
    scenario = dict(
        instances = training_instances,
        debugLevel = 0,
        parallel = threads,
        digits = 15,
        boundMax = boundMax,
        logFile = '', 
        seed = experiment['seed']
    )

    if surrogateType == SurrogateType.RUNTIME:
        scenario.update(dict(maxTime=maxTimeOrExperiments))
    elif surrogateType == SurrogateType.QUALITY:
        scenario.update(dict(maxExperiments=maxTimeOrExperiments))
    
    scenario.update(dict(experiment['configuration']))

    tuner = irace(scenario, parameters, surrogate_target_runner)

    with suppress_stdout():
        best_configs: pd.DataFrame = tuner.run()

    # Get a single configuration as the best config.
    best_config = best_configs.to_dict(orient='records')[0]

    def validate(config):
        '''
        Get the performance of a configuration on the validation set
        '''
        sum = 0
        for instance in validation_instances:
            sum += model.predict_surrogate(config, instance)
        return sum / len(validation_instances)
    
    return dict(cost=validate(best_config))


params = Parameters()
params.capping = Param(Categorical(('0', '1')), condition=Symbol('elitist') == '1')
params.cappingType = Param(Categorical(('median', 'mean', 'worst', 'best')), condition=Symbol('capping') == '1')
params.boundType = Param(Categorical(('candidate', 'instance')), condition=Symbol('capping') == '1')
params.testType = Param(Categorical(('f-test', 't-test')))
params.elitist = Param(Categorical(('0', '1')))

def default_instance(model):
    return model, get_instances_training(model.instances), get_instances_validation(model.instances)


scenario = dict(
    instances = [
        (
            *default_instance(SurrogateModel(
                '../target_algorithms/surrogate/cplex_regions200/config_space.cplex_regions200.par10.random.pcs',
                '../target_algorithms/surrogate/cplex_regions200/inst_feat_dict.cplex_regions200.par10.random.json',
                '../target_algorithms/surrogate/cplex_regions200/pyrfr_model.cplex_regions200.par10.random.bin'
            )),
            1000,
            3600,
            SurrogateType.RUNTIME,
        ),
    ],
    maxExperiments = 2000,
    debugLevel = 0,
    parallel = 2,
    digits = 15,
    seed = 123,
    logFile = "log.Rdata",
    instanceObjectSerializer = default_serializer
    )

defaults = pd.DataFrame(data=dict(
    capping = [0],
    cappingType = [np.nan], 
    boundType = [np.nan], 
    elitist = [1],
    testType = ['f-test']
))


tuner = irace(scenario, params, target_irace)
tuner.set_initial(defaults)
best_config = tuner.run()


print(best_config)
