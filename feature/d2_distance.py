#原文如下
#for two pairwise faces, 512 point pairs are randomly selected, and the ratio of the Euclidean
#distance of each point pair to the diagonal length of the whole solid model bounding box is calculated
import numpy as np

def calculate_d2_distance(face1_points, face2_points):
    # Ensure that both faces have the same number of points
    assert face1_points.shape == face2_points.shape, "Faces must have the same number of points"

    num_points = face1_points.shape[0]

    # Randomly select 512 point pairs
    indices = np.random.choice(num_points, 512, replace=True)

    # Calculate the Euclidean distances of each point pair
    distances = np.linalg.norm(face1_points[indices] - face2_points[indices], axis=1)

    # Calculate the bounding box diagonal length of the whole solid model
    min_coords = np.minimum(face1_points.min(axis=0), face2_points.min(axis=0))
    max_coords = np.maximum(face1_points.max(axis=0), face2_points.max(axis=0))
    diagonal_length = np.linalg.norm(max_coords - min_coords)

    # Calculate the ratio of each distance to the diagonal length
    distance_ratios = distances / diagonal_length

    # Divide the [0,1] interval into 64 sub-intervals
    intervals = np.linspace(0, 1, 33)

    # Calculate the frequency of each distance ratio in the sub-intervals
    frequency, _ = np.histogram(distance_ratios, bins=intervals)

    # Normalize the frequency to get the marginal distribution
    marginal_distribution = frequency / frequency.sum()

    return marginal_distribution

