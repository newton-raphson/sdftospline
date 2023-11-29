
import numpy as np
import matplotlib.pyplot as plt
from geomdl import BSpline
from pyDOE2 import lhs
from concurrent.futures import ProcessPoolExecutor
from signed_distance import signed_distance_polygon,curve_point_nurbs
from cv2 import pointPolygonTest
def pointPolygonTest_wrapper(args):
    print(type(args[1]))
    return pointPolygonTest(*args)
def calculate_signed_distances_parallel(points, polygon_vertices):
    if polygon_vertices.size == 0:
        raise ValueError("polygon_vertices is empty")
    args = [(polygon_vertices, np.array(point, dtype=np.float32), True) for point in points]
    with ProcessPoolExecutor() as executor:
        # Use executor.map to parallelize the signed_distance_polygon function
        signed_distances = list(executor.map(pointPolygonTest_wrapper, args))

    return np.array(signed_distances, dtype=np.float32)


    return np.array(signed_distances)
if __name__ == '__main__':

    # Original control points
    original_ctrlpts = np.array([[-0.7, 0.7, 0], [0, 0.7, 0], [0.7, 0.7, 0], [0.7, 0, 0], [0.7, -0.7, 0],
                                [-0.7, 0, 0], [-0.7, -0.7, 0], [-0.7, 0, 0], [-0.7, 0.7, 0]])

    # Number of control points
    num_ctrlpts = len(original_ctrlpts)

    # Number of samples
    num_samples = 2000
    np.random.seed(20)
    # Generate Latin Hypercube Samples for radius and angle for each control point
    samples = np.random.uniform(0, 1, size=(num_samples, 2,9))
    # Make last and first element of every array same
    samples[:, :, -1] = samples[:, :, 0]

    # Scale and shift samples for radius
    radii = 0.01 + samples[:, 0] * (0.7 - 0.01)

    # # Scale and shift samples for angle (convert to degrees)
    angles_deg = -10 + samples[:, 1] * (10 - (-10))
    # # Convert angles to radians
    angles_rad = np.radians(angles_deg)


    angles = np.radians([0,45,90,135,180,225,270,315,360])

    # # Generate control points based on LHS samples
    lhs_ctrlpts = np.zeros((num_samples,9,2))
    # lhs_ctrlpts[:, :] = (radii * np.cos(angles_rad+angles),radii * np.sin(angles_rad+angles))
    # (radii * np.cos(angles_rad+angles),radii * np.sin(angles_rad+angles))
    lhs_ctrlpts[:,:,0]=radii * np.cos(angles_rad+angles)
    lhs_ctrlpts[:,:,1]=radii * np.sin(angles_rad+angles)

    curve_points_lhs = curve_point_nurbs(lhs_ctrlpts[9])

    import time
    polygon_vertices = np.array([(x, y) for x, y in curve_points_lhs],dtype=np.float32)

    # Create a 256x256 grid
    x = np.linspace(-1, 1, 256)
    y = np.linspace(-1, 1, 256)
    xx, yy = np.meshgrid(x, y)
    grid_points = np.column_stack((xx.flatten(), yy.flatten()))

    # Compute signed distances for the grid points
    signed_distances = np.zeros_like(grid_points[:, 0])
    start_time = time.time()
    for i in range(grid_points.shape[0]):
        point = grid_points[i]
        signed_distance = pointPolygonTest(polygon_vertices, point, True)
        signed_distances[i] = signed_distance
    end_time = time.time()
    print(end_time-start_time)
    # signed_distances = pointPolygonTest_wrapper((polygon_vertices,grid_points[0],True))
    end_time = time.time()
    print("Time taken: {}".format(end_time - start_time))