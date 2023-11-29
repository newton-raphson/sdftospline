import numpy as np
from geomdl import NURBS
import time
# function to compute signed distance from a point to a convex polygon
def signed_distance_polygon(point, polygon_vertices):
    """
    Calculate the signed distance from a point to a convex polygon.

    Parameters:
    - point: NumPy array of shape (2,) representing the point
    - polygon_vertices: NumPy array of shape (num_vertices, 2) representing the vertices of the convex polygon

    Returns:
    - Signed distance from the point to the polygon
    """
    num_vertices = polygon_vertices.shape[0]

    # Initialize variables
    d = np.inf
    s = 1.0

    for i in range(num_vertices):
        j = (i - 1) % num_vertices
        e = polygon_vertices[j] - polygon_vertices[i]
        w = point - polygon_vertices[i]

        # Calculate the perpendicular distance
        b = w - e * np.clip(np.dot(w, e) / np.dot(e, e), 0.0, 1.0)
        d = min(d, np.linalg.norm(b))

        # Determine the sign based on winding number
        if (polygon_vertices[i, 1] <= point[1] < polygon_vertices[j, 1]) or (polygon_vertices[j, 1] <= point[1] < polygon_vertices[i, 1]):
            if polygon_vertices[i, 1] <= point[1] < polygon_vertices[j, 1]:
                if e[0] * w[1] > e[1] * w[0]:
                    s = -s
            elif polygon_vertices[j, 1] <= point[1] < polygon_vertices[i, 1]:
                if e[0] * w[1] < e[1] * w[0]:
                    s = -s

    return s * d


from geomdl import NURBS

# Create a 3-dimensional B-spline Curve
curve = NURBS.Curve()

# Set degree
curve.degree = 2

# Set control points (weights vector will be 1 by default)
# Use curve.ctrlptsw is if you are using homogeneous points as Pw
# let's vary each point in 1000 * 8
# 6 CP
curve.ctrlpts =[[-0.7, 0.7, 0], [0, 0.7, 0], [0.7, 0.7, 0], [0.7, 0, 0], [0.7, -0.7, 0], [-0.7, 0, 0],[-0.7,-0.7,0],[-0.7,0,0],[-0.7, 0.7, 0]]
# 9th control point is same as first control point 
# let's vary the radius between 0.01 and 0.7 in magnitude and angle theta about -10 to 10 degrees and produce 
# 60000 samples 
# Set knot vector
# this is fixed
curve.knotvector = [0, 0,0,0.14285714,0.28571429,0.42857143,0.57142857,0.71428571,0.85714286,1, 1, 1]
# setting this value such that it is not small
# Set evaluation delta (controls the number of curve points)
curve.delta = 0.003

# Get curve points (the curve will be automatically evaluated)
curve_points = curve.evalpts

# this works 
# Convert to 2D points and arrange in the acceptable form
polygon_vertices = np.array([(x, y) for x, y, _ in curve_points])
time1 = time.time()
# Create a 256x256 grid
x = np.linspace(-1, 1, 256)
y = np.linspace(-1, 1, 256)
xx, yy = np.meshgrid(x, y)
grid_points = np.column_stack((xx.flatten(), yy.flatten()))

# Compute signed distances for the grid points
signed_distances = np.zeros_like(grid_points[:, 0])

for i in range(grid_points.shape[0]):
    point = grid_points[i]
    signed_distance = signed_distance_polygon(point, polygon_vertices)
    signed_distances[i] = signed_distance
time2 = time.time()
# Reshape the signed distances into a 256x256 grid
signed_distances_grid = signed_distances.reshape(256, 256)
print(time2-time1)