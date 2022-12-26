from enum import Enum

def get_instances_training(instances):
    '''Get the training instances for the target irace'''
    return instances[:len(instances) // 3]

def get_instances_validation(instances):
    '''Get the validation instances for validating the performance of target irace'''
    return instances[len(instances) // 3: len(instances) // 3 * 2]

def get_instances_meta_training(instances):
    '''Get the set of instance for the irace with the best configuration to train on. Currently it equals to the training instances'''
    return instances[:len(instances) // 3]

def get_instances_meta_validation(instances):
    '''Get the validation instances for the meta irace'''
    return instances[len(instances) // 3 * 2:]

class SurrogateType(Enum):
    RUNTIME = 'runtime'
    QUALITY = 'quality'

def default_serializer(obj):
    if isinstance(obj, SurrogateType):
        return obj.value
    else:
        return '<not serializable>'