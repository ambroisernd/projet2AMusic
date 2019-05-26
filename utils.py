import random
import numpy as np
import pickle


def parse_duration(s):
    """return string duration to float"""
    if "/" in s:
        return float(s.split("/")[0]) / float(s.split("/")[-1])
    else:
        return float(s)


def RepresentsInt(s):
    """helper fct to check if string is int"""
    try:
        int(s)
        return True
    except ValueError:
        return False


"""
#OLD
def predict_and_sample(inference_model, Ty, n_values):
    indices = []
    pred = inference_model.predict(np.zeros((1, 1, n_values)))
    for i in range(Ty):
        indices.append(np.random.choice([k for k in range(n_values)], p=pred[i].ravel()))
    results = to_categorical(indices)
    return results, indices
"""
