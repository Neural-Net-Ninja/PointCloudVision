import numpy as np
from sklearn.cluster import KMeans

def attribute_weighted_sampling(indices: np.ndarray,
                                num_points: int,
                                attributes: np.ndarray,
                                num_clusters: int = 3) -> np.ndarray:
    """
    Samples points based on weights of the attributes. Attributes value of minority class will have higher weights than
    attributes value of majority class, thus increasing the probability of sampling points with minority
    class attributes value.

    :param indices: The indices of the points in the given neighborhood. Must have shape :math:`(N)` where `N = number of points`.
    :type indices: numpy.ndarray
    :param num_points: Number of points to be sampled from the neighborhood.
    :type num_points: integer
    :param attributes: The chosen attributes of the points in the dataset to consider for sampling. Must have shape :math:`(N)` where `N = number of points`.
    :type attributes: numpy.ndarray
    :param num_clusters: Number of clusters to be used for K-means clustering. Defaults to `3`.
    :type num_clusters: integer
    :return: The indices of the selected points.
    :rtype: numpy.ndarray
    """
    # Perform K-means clustering on the attributes
    kmeans = KMeans(n_clusters=num_clusters, random_state=0)
    kmeans.fit(attributes.reshape(-1, 1))

    # Assign weights to the attributes based on the cluster centers
    weights = np.zeros_like(attributes, dtype=np.float32)
    for i in range(num_clusters):
        cluster_indices = np.where(kmeans.labels_ == i)[0]
        cluster_center = kmeans.cluster_centers_[i]
        weights[cluster_indices] = 1 / np.abs(attributes[cluster_indices] - cluster_center)

    # Normalize the weights
    weights = weights / np.sum(weights)

    # Sample indices with replacement according to the weights
    sampled_indices = np.random.choice(indices, size=num_points, replace=True, p=weights)

    return sampled_indices