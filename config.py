from pathlib import Path

def get_data_path():
    return Path('/data/eazabrotsky/data_like/')

def label2id():
    return {
        'dislike': 0,
        'skip': 1, 
        'view': 2,
        'like': 3,
    }

def id2w():
    return {
        0: -10,
        1: -0.1,
        2: 0.1,
        3: 0.5,
    }

def get_params(iterations=1000, verbose=0):
    return  {    
    'iterations': iterations, 
    'verbose': verbose,
    'early_stopping_rounds': 200,
    'loss_function': 'MultiClass',
        "bagging_temperature": 0,
        "depth": 9,
        "l2_leaf_reg": 2.996730370305901, 
        "max_ctr_complexity": 6,
        "model_size_reg": 0.4525805948723012, 
        "random_strength": 0.26669190945436927,
    }