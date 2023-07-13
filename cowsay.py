import numpy as np
from sklearn.cluster import KMeans

import matplotlib.pyplot as plt

def min_max_scale(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def prominent_attribute_decider(intensity: np.ndarray = None,
                                normal: np.ndarray = None,
                                curvature: np.ndarray = None,
                                distance_to_dtm: np.ndarray = None):
    # Create a list of tuples containing the attribute name and data
    attributes = [('intensity', min_max_scale(intensity)),
                  ('normal', min_max_scale(normal)),
                  ('curvature', min_max_scale(curvature)),
                  ('distance_to_dtm', min_max_scale(distance_to_dtm))]

    # Compute the mean for each attribute
    means = []
    for name, data in attributes:
        if data is not None:
            mean = np.mean(data)
            means.append((name, mean))
    
    print(attributes)
    print(means)

    # Find the attribute with the highest mean
    if len(means) > 0:
        name, mean = max(means, key=lambda x: x[1])
        return globals()[name]
    else:
        return None

intensity = np.array([1, 1, 1, 1, 2, 1, 1, 1, 1])
normal = np.array([1, 1, 1, 1, 2, 1, 1, 3, 1])
curvature = np.array([1, 1, 1, 1, 2, 1, 1, 1, 4])
distance_to_dtm = np.array([10, 12, 5, 3, 6, 1, 2, 1, 30])

selected = prominent_attribute_decider(intensity, normal, curvature, distance_to_dtm)
print(selected)