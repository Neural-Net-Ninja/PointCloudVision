import numpy as np

intensity = np.array([1,2,3,10,10,3,2,1])
normal = np.array([1, 1, 1, 1, 2, 1, 1, 3, 1])
curvature = np.array([1, 1, 1, 1, 2, 1, 1, 1, 4])
distance_to_dtm = np.array([10, 12, 5, 3, 6, 1, 2, 1, 30])

def determine_threshold(intensity):
    """
    Determines the threshold value for separating the road markings from the road surface based on the intensity values.

    :param intensity: The intensity values of the points in the neighborhood. Must have shape :math:`(N)` where `N = number of points`.
    :type intensity: numpy.ndarray
    :return: The threshold value for separating the road markings from the road surface.
    :rtype: float
    """
    # Create a histogram of the intensity values
    hist, bins = np.histogram(intensity, bins=5)

    # Find the peak of the histogram
    peak_index = np.argmax(hist)
    peak_value = bins[peak_index]

    # Find the valley to the left of the peak
    left_hist = hist[:peak_index]
    left_bins = bins[:peak_index]
    left_valley_index = np.argmin(left_hist)
    left_valley_value = left_bins[left_valley_index]

    # Find the valley to the right of the peak
    right_hist = hist[peak_index:]
    right_bins = bins[peak_index:]
    right_valley_index = np.argmin(right_hist)
    right_valley_value = right_bins[right_valley_index]

    # Choose the threshold value as the midpoint between the two valleys
    threshold = (left_valley_value + right_valley_value) / 2

    return threshold


def calculate_threshold(data):
    """
    Calculates a threshold value based on the mean and standard deviation of the data.

    :param data: The data to calculate the threshold value for.
    :type data: numpy.ndarray
    :return: The threshold value.
    :rtype: float
    """
    # Calculate the mean and standard deviation of the data
    mean = np.mean(data)
    std = np.std(data)

    # Calculate the threshold value as the mean plus one standard deviation
    threshold = mean + std

    return threshold

threshold = calculate_threshold(intensity)
print(threshold)

var = np.var(intensity, axis=0)
print(var)