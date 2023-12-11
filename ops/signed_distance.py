import sys
sys.path.append('/work/mech-ai-scratch/samundra/experiments/splinetofield')
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

def curve_point_nurbs(control_pts=[[-0.7, 0.7, 0], [0, 0.7, 0], [0.7, 0.7, 0], [0.7, 0, 0], [0.7, -0.7, 0], [-0.7, 0, 0],[-0.7,-0.7,0],[-0.7,0,0],[-0.7, 0.7, 0]],knotvec= [0, 0,0,0.14285714,0.28571429,0.42857143,0.57142857,0.71428571,0.85714286,1, 1, 1],
                delta=0.007):
    # Create a 3-dimensional B-spline Curve
    curve = NURBS.Curve()

    # Set degree
    curve.degree = 2

    # Set control points (weights vector will be 1 by default)
    # Use curve.ctrlptsw is if you are using homogeneous points as Pw
    # let's vary each point in 1000 * 8
    # 6 CP
    # curve.ctrlpts =[[-0.7, 0.7, 0], [0, 0.7, 0], [0.7, 0.7, 0], [0.7, 0, 0], [0.7, -0.7, 0], [-0.7, 0, 0],[-0.7,-0.7,0],[-0.7,0,0],[-0.7, 0.7, 0]]
    curve.ctrlpts=control_pts
    # 9th control point is same as first control point 
    # let's vary the radius between 0.01 and 0.7 in magnitude and angle theta about -10 to 10 degrees and produce 
    # 60000 samples 
    # Set knot vector
    # this is fixed
    curve.knotvector = knotvec
    # setting this value such that it is not small
    # Set evaluation delta (controls the number of curve points)
    curve.delta = delta

    # Get curve points (the curve will be automatically evaluated)
    return curve.evalpts
