import numpy as np
import csv
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation as R

ID = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1]
])

def random_unit_vector():
    """Generate a random 3D unit vector

    Returns:
        np.array: a random 3D unit vector
    """
    z = np.random.uniform(-1, 1)
    theta = np.random.uniform(0, 2*np.pi)
    return(np.array([
        np.sqrt(1-z**2)*np.cos(theta),
        np.sqrt(1-z**2)*np.sin(theta),
        z
    ]))

def gen_observation(p, u, a, d, epsilon=1e-6):
    """Generate an observation from a point looking at a plane.

    Generates an observation (distance and observation point) for a sensor at
    location p looking in the direction given by the vector u looking at the
    plane defined by a[0]x + a[1]y + a[2]z + d = 0.

    https://rosettacode.org/wiki/Find_the_intersection_of_a_line_with_a_plane#Python

    Args:
        p (3-tuple of floats): the position of the sensor (x, y, z).
        u (3-tuple of floats): the orientation of the sensor (x, y, z).
          Does not have to be a unit vector.
        a (3-tuple of floats): the equation for the line where a[0]x + a[1]y + a[2]z + d = 0.
        d (float) the c portion of the line equation.

    Returns:
        The distance and intersection point as a tuple, for example, with distance
        5.2 and intersection point (8.1, 0.3, 4):

        (5.2, (8.1, 0.3, 4)) or float('inf') if the sensor does not see the plane.

    Raises:
        ValueError: The line is undefined.
    """
    a = np.array(a)
    p = np.array(p)
    u = np.array(u)

    if(a[0] != 0):
        plane_point = np.array([-d/a[0], 0, 0])
    elif(a[1] != 0):
        plane_point = np.array([0, -d/a[1], 0])
    elif(a[2] != 0):
        plane_point = np.array([0, 0, -d/a[2]])
    else:
        raise ValueError("The plane with normal a=[0,0,0] is undefined")

    ndotu = a.dot(u)
    if abs(ndotu) < epsilon:
        return float('inf')

    w = p - plane_point
    si = -a.dot(w) / ndotu
    Psi = w + si * u + plane_point
    
    dist = np.linalg.norm(Psi - p)

    if(np.allclose((dist * u) + p, Psi)):
        return (dist, Psi)
    else:
        return float('inf')

def angle_between(u1, u2):
    """Get the angle between two unit vectors, in radians
    
    Args:
        u1: unit vector
        u2: unit vector

    Returns:
        (float): angle between u1 and u2, in radians
    """
    u1 = np.array(u1)
    u2 = np.array(u2)
    assert(
        np.abs(np.linalg.norm(u1) - 1 < 0.0001)
        and np.abs(np.linalg.norm(u1) - 1 < 0.0001)
    )
    angle = np.arccos(np.dot(u1, u2) / (np.linalg.norm(u1) * np.linalg.norm(u2)))
    return angle

def generate_motions(p, u, a, d, plane_center, bbox, radius=500, n=32):
    """Generate random robot motions that point sensor at plane

    Generate n motions that keep the sensor at position p and orientation u
    pointing at the plane given by a[0] + a[1] + a[2] + d = 0

    Args:
        p: 3D position of sensor on robot segment
        u: heading unit vector for sensor
        a: a vector for plane in equation ax+d=0
        d: d scalar for plane in equation ax+d=0
        plane_center: where on the plane to center the points you're aiming for
          around
        bbox: bounding box for the sensor, given as a 2D array with like so:
          [
              [xmin, xmax],
              [ymin, ymax],
              [zmin, zmax]
          ]
        radius (default 500): how far from the center intersection points on the
          plane should be
        n (default 32): how many motions to generate

    Returns:
        (n x 4 x 4 array): robot motions as transforms in homogenous coordinates
    """
    p = np.array(p)
    u = np.array(u)
    a = np.array(a)

    # generate points on plane
    xs = scattered_on_plane(a, d, plane_center.reshape(3), radius, n)

    # generate positions for sensor in space
    ps = []
    while (len(ps) < len(xs)):
        pt = np.array([
            np.random.uniform(*bbox[0]),
            np.random.uniform(*bbox[1]),
            np.random.uniform(*bbox[2])
        ])
        # check that pt is on the same side of the plane as the center of the robot
        if (np.sign(np.dot(a,pt)+d) == np.sign(np.dot(a, np.array([0, 0, 0]))+d)):
            # check that pt is at least 10cm away from the plane
            if (np.abs(np.dot(a,pt)+d) > 100):
                ps.append(pt)

    # generate unit vectors that point sensor points to plane points
    us = [(p - s) / np.linalg.norm(p - s) for p, s in zip(xs, ps)]

    # convert list of points and unit vectors to list of transforms
    tfs = points_to_transforms([p, *ps], [u, *us])

    return tfs

def scattered_on_plane(a, d, center, radius, num_points):
    """Generate points scattered on the plane given by a, d

    Args:
        a: a parameter for plane (3D vector)
        d: d parameter for plane
        center: center point from which points are scattered
        radius: radius of scattered points
        num_points: number of scattered points

    Returns:
        (num_points x 3 array): coordinates of points on plane    
    """
    if(np.dot(a, center)+d > 0.00001):
        raise ValueError("center is not on plane given by ax+d=0")

    # generate a random vector and make it orthogonal to a
    # https://stackoverflow.com/questions/33658620/generating-two-orthogonal-vectors-that-are-orthogonal-to-a-particular-direction
    xvec = np.random.randn(3)
    xvec -= xvec.dot(a) * a / np.linalg.norm(a)**2 
    xvec /= np.linalg.norm(xvec)

    yvec = np.cross(a, xvec)

    points = []
    for _ in range(num_points):
        xcomp = np.random.uniform(-radius, radius)
        ycomp = np.random.uniform(-radius, radius)
        points.append(center + (xcomp*xvec + ycomp*yvec))

    return points

def points_to_transforms(points, units):
    """Convert a set of points to a set of transforms

    Arguments:
        points (list of 3-tuples): point positions (first is starting pos)
        units (list of 3-tuples): unit vector directions (first is starting)
    
    Returns:
        (list of 4x4 np.array): transformations leading from starting point to
          each other point (first will be identity)
    """
    return([get_transform(points[0], units[0], pt, u) for pt, u in zip(points[1:], units[1:])])

def to_hom(vec):
    """Takes a numpy array and adds a 1 to the end of it
    """
    return np.append(vec, [1])

def from_hom(vec):
    """Takes a numpy array and removes a 1 from the end of it
    """
    return vec[:-1]

def get_transform(p1, u1, p2, u2):
    """Get the transform from pos. p1, rot. u1 to pos. p2, rot. u2

    Arguments:
        p1 (3-tuple): x, y, z coordinates of starting position
        u1 (3-tuple): x, y, z coordinates of starting unit vector orientation
        p2 (3-tuple): x, y, z coordinates of final position
        u2 (3-tuple): x, y, z coordinates of final unit vector orientation

    Returns:
        (4x4 np.array): the transform from p1, u1 to p2, u2 in homogenous coord.

    https://math.stackexchange.com/questions/180418/calculate-rotation-matrix-to-align-vector-a-to-vector-b-in-3d/476311#476311
    """

    u1 = np.array(u1)
    u2 = np.array(u2)
    p1 = np.array(p1)
    p2 = np.array(p2)

    if(np.allclose(u1, u2)):
        R = np.identity(3)
    else:
        v = np.cross(u1, u2)
        s = np.linalg.norm(v)
        if(s == 0):
            if(u1[0] == u2[1] and u1[1] == u2[0]): #BUG there are other cases like this that aren't covered
                R = np.array([
                    [0, 1, 0],
                    [1, 0, 1],
                    [0, 0, 1]
                ])
        else:
            c = np.dot(u1, u2)
            vx = np.array([
                [0, -v[2], v[1]],
                [v[2], 0, -v[0]],
                [-v[1], v[0], 0]
            ])

            R = np.identity(3) + vx + (vx @ vx) * ((1 - c)/(s*s))

    new_p = R @ p1

    t = p2 - new_p

    tf = np.array([
        [R[0][0], R[0][1], R[0][2], t[0]],
        [R[1][0], R[1][1], R[1][2], t[1]],
        [R[2][0], R[2][1], R[2][2], t[2]],
        [0, 0, 0, 1]
    ])

    # tf * p1 should = p2
    assert(np.allclose(tf @ np.append(p1, 1), np.append(p2, 1)))

    # tf * u1 (with 0 for third coordinate - no translation) should = u2
    assert(np.allclose(tf @ np.append(u1, 0), np.append(u2, 0)))

    return tf

def read_data(data_path):
    """Read real-world trial data from given folder

    Arguments:
      data_path (string): path to folder containing measurements.csv and
        transforms.csv files to be read in

    Returns:
      (tuple of arrays): measurements array and transforms array populated with
        data from data_path/measurements.csv and data_path/transforms.csv
    """
    with open(data_path + "/measurements.csv") as f:
        csvfile = csv.reader(f)

        measurements = []
        for line in csvfile:
            measurements.append(np.average([float(x) for x in line[1:]]))

    measurements = np.array(measurements)

    with open(data_path + "/transforms.csv") as f:
        csvfile = csv.reader(f)

        raw_transforms = []
        for line in csvfile:
            items = []
            for item in line:
                if(item != ' '):
                    items.append(float(item))
            raw_transforms.append(np.reshape(np.array(items), (4,4)))
    
    # change unit of transforms from meters to mm
    transforms = [rescale_transform(tf, 1000) for tf in raw_transforms]

    return(measurements, transforms)

def rescale_transform(tf, scale):
    """Rescale a 4x4 homogenous transform matrix by some factor

    Arguments:
      tf (4x4 np.array): the 4x4 homogenous transform matrix to scale
      scale: how much to scale it by

    Returns:
      (4x4 np.array): scaled 4x4 homogenous transform matrix
    """
    new_tf = np.array([
        [tf[0][0], tf[0][1], tf[0][2], tf[0][3]*scale],
        [tf[1][0], tf[1][1], tf[1][2], tf[1][3]*scale],
        [tf[2][0], tf[2][1], tf[2][2], tf[2][3]*scale],
        [tf[3][0], tf[3][1], tf[3][2], tf[3][3]],
    ])

    return new_tf

def fit_plane(pts):
    """Fit a plane given by ax+d = 0 to a set of points

    Works by minimizing the sum over all points x of ax+d

    Arguments:
      pts: array of points in 3D space

    Returns:
      (3x1 numpy array): a vector for plane equation
      (float): d in plane equation
      (float): sum of residuals for points to plane (orthogonal l2 distance)
    """

    pts = np.array(pts)

    def loss_fn(x, points):
        a = np.array(x[:3])
        d = x[3]

        loss = 0
        for point in points:
            loss += np.abs(np.dot(a, np.array(point)) + d)

        return loss

    def a_constraint(x):
        return np.linalg.norm(x[:3]) - 1

    soln = minimize(
        loss_fn,
        np.array([0, 1, 0, 0]),
        args=(pts),
        method='slsqp',
        constraints=[
            {
                'type': 'eq',
                'fun': a_constraint
            }
        ]
    )

    a = soln.x[:3]
    d = soln.x[3]
    res = soln.fun

    return a, d, res

def perturb_p(p, radius, symmetrical=False):
    """Perturb a point a random amount in each direction

    Arguments:
        (np.array) p: a point to be perturbed
        (float) radius: amount to perturb in each direction (random)
        (boolean) symmetrical: whether to return the symmetrical negative
          version of each perturbation along with the positive version

    Returns:
        np.array: perturbed point
    
    """
    if symmetrical:
        return (
            p + np.random.uniform(-radius, radius, 3),
            p - np.random.uniform(-radius, radius, 3)
        )
    else:
        return p + np.random.uniform(-radius, radius, 3)

def perturb_u(u, angle_range, symmetrical=False):
    """
    Perturb a unit vector along a given angle range

    Arguments:
        (np.array) u: 3D unit vector to perturb
        (float) angle_range: range of angles upon which to randomly perturb
          within (uniform)
        (boolean) symmetrical: whether to return the symmetrical negative
          version of each perturbation along with the positive version

    Returns:
        (np.array): randomly perturbed unit vector
    """

    angle = np.random.uniform(-angle_range, angle_range)
    if symmetrical:
        # generate a random vector and make it orthogonal to x
        # this will be the axis for our axis angle rotation
        # https://stackoverflow.com/questions/33658620/generating-two-orthogonal-vectors-that-are-orthogonal-to-a-particular-direction
        axis = np.random.randn(3)
        axis -= axis.dot(u) * u / np.linalg.norm(u)**2 
        axis /= np.linalg.norm(axis)

        rot1 = R.from_rotvec(np.radians(angle) * axis)
        rot2 = R.from_rotvec(np.radians(-angle) * axis)

        return(
            rot1.apply(u),
            rot2.apply(u)
        )
    else:
        # generate a random vector and make it orthogonal to x
        # this will be the axis for our axis angle rotation
        # https://stackoverflow.com/questions/33658620/generating-two-orthogonal-vectors-that-are-orthogonal-to-a-particular-direction
        axis = np.random.randn(3)
        axis -= axis.dot(u) * u / np.linalg.norm(u)**2 
        axis /= np.linalg.norm(axis)

        rot = R.from_rotvec(np.radians(angle) * axis)

        return(rot.apply(u))