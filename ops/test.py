import numpy as np
import matplotlib.pyplot as plt
from geomdl import BSpline
import time
# from pyDOE2 import lhs
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

    return np.array(signed_distances, dtype=np.float32).reshape(256, 256)
def generate_control_points(num_samples):
     # Original control points
    original_ctrlpts = np.array([[-0.7, 0.7, 0], [0, 0.7, 0], [0.7, 0.7, 0], [0.7, 0, 0], [0.7, -0.7, 0],
                                [-0.7, 0, 0], [-0.7, -0.7, 0], [-0.7, 0, 0], [-0.7, 0.7, 0]])

    # Number of control points
    num_ctrlpts = len(original_ctrlpts)

    # Number of samples
    num_samples = 60000
    np.random.seed(20)
    # Generate Latin Hypercube Samples for radius and angle for each control point
    samples = np.random.uniform(0, 1, size=(num_samples, 2,num_ctrlpts))
    # Make last and first element of every array same
    samples[:, :, -1] = samples[:, :, 0]

    # Scale and shift samples for radius
    radii = 0.01 + samples[:, 0] * (0.7 - 0.01)

    # # Scale and shift samples for angle (convert to degrees)
    angles_deg = -22.5 + samples[:, 1] * (22.5 - (-22.5))
    # # Convert angles to radians
    angles_rad = np.radians(angles_deg)


    angles = np.radians([0,45,90,135,180,225,270,315,360])

    # # Generate control points based on LHS samples
    lhs_ctrlpts = np.zeros((num_samples,9,2))
    # lhs_ctrlpts[:, :] = (radii * np.cos(angles_rad+angles),radii * np.sin(angles_rad+angles))
    # (radii * np.cos(angles_rad+angles),radii * np.sin(angles_rad+angles))
    lhs_ctrlpts[:,:,0]=radii * np.cos(angles_rad+angles)
    lhs_ctrlpts[:,:,1]=radii * np.sin(angles_rad+angles)
    # save this in a file
    np.save('lhs_ctrlpts.npy',lhs_ctrlpts)


if __name__ == '__main__':

    # Load control points from the numpy file if it exists
    lhs_ctrlpts = np.load('lhs_ctrlpts.npy')

    # Load existing signed distances file if it exists, otherwise create an empty array
    sdf_file_path = 'images/signed_distances.npy'
    # try:
    #     sdf_npy = np.load(sdf_file_path)
    #     # total_iterations = len(lhs_ctrlpts)
    #     total_iterations=100
    #     print("Total iterations:", total_iterations)
    #     iteration_number = len(sdf_npy)
    #     print("Current iteration:", iteration_number)
    # except FileNotFoundError:
    #     sdf_npy = np.empty((0, 256, 256))  # Empty array to store signed distances
    #     total_iterations = 10
    #     iteration_number = 0
    iteration_number=0
    total_iterations=60000
    # Iterate through control points and calculate signed distances for each grid

    # Create a 256x256 grid
    x = np.linspace(-1, 1, 128)
    y = np.linspace(-1, 1, 128)
    xx, yy = np.meshgrid(x, y)
    grid_points = np.column_stack((xx.flatten(), yy.flatten()))

    for i in range(iteration_number, total_iterations):
        # Create a NURBS curve for the current iteration
        curve_points_lhs = curve_point_nurbs(lhs_ctrlpts[i])
        polygon_vertices = np.array([(x, y) for x, y in curve_points_lhs], dtype=np.float32)

        # Compute signed distances for the grid points
        signed_distances = np.zeros_like(grid_points[:, 0])
        for j in range(grid_points.shape[0]):
            point = grid_points[j]
            signed_distance = signed_distance_polygon(point, polygon_vertices)
            signed_distances[j] = signed_distance

        signed_distances = signed_distances.reshape(128, 128)
        
        # Append the signed distances to the array
        # sdf_npy = np.append(sdf_npy, signed_distances[np.newaxis, :, :], axis=0)
        print("Iteration:", i)
        # Save the signed distances periodically (adjust the condition as needed)
        np.save(f'images2/signed_distances{i}.npy', signed_distances)
