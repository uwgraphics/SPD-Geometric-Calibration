import numpy as np
from scipy.optimize import minimize

def bf_slsqp(transforms, measurements, p_bounds, d_bounds):
    """Solver which performs a search over many starting estimates to recover
    transforms and measurements. Calls slsqp() function once for every set
    of starting estimates, and picks the solutions for free variables which lead
    to lowest loss.

    Arguments:
        (nx4x4 np.array) transforms: list of transforms in homogenous
          coordinates, representing robot motions (R_1...n and t_1...n)
        (array of n floats) measurements: corresponding sensor measurements
          (m_1...m_n)
        (3x2 array of floats) p_bounds: lower and upper bounds on components of
          p, respectively. Not required; can set to float('inf') or some very
          large values with no practical impact on performance
        (array of 2 floats) d_bounds: lower and upper bounds on d, respectively.
          Not required; can set to float('inf') or some very large values with
          no practical impact on performance

    Returns:
        (2-tuple): best solution (array of [p_x, p_y, p_z, u_x, u_y, u_z, a_x,
          a_y, a_z, d]) and the loss that the solution produces
    """

    p_est_list = [(0, 0, 0)]
    u_est_list = [
        [1, 0, 0],
        [-1, 0, 0],
        [0, 1, 0],
        [0, -1, 0],
        [0, 0, 1],
        [0, 0, -1]
    ]
    a_est_list = [[1, 0, 0]]
    d_est_list = [0]

    best_loss = float('inf')

    i = 0
    for p_est in p_est_list:
        for u_est in u_est_list:
            for a_est in a_est_list:
                for d_est in d_est_list:
                    i += 1
                    soln, res = slsqp(transforms, measurements, a_est, d_est, p_est, u_est, p_bounds, d_bounds)
                    if res.fun < best_loss:
                        best_loss = res.fun
                        best_soln = soln

    return best_soln, best_loss

def slsqp(transforms, measurements, a_est, d_est, p_est, u_est, p_bounds, d_bounds):
    """SLSQP solver for calibration problem, which requires some set of initial
    estimates. Returns a solution for p, u, a, and d.

    Arguments:
        (nx4x4 np.array) transforms: list of transforms in homogenous
          coordinates, representing robot motions (R_1...n and t_1...n)
        (array of n floats) measurements: corresponding sensor measurements
          (m_1...m_n)
        (3x1 np.array) a_est: initial estimate for a
        (float) d_est: initial estimate for d
        (3x1 np.array) p_est: initial estimate for p
        (3x1 np.array) u_est: initial estimate for u
        (3x2 array of floats) p_bounds: lower and upper bounds on components of
          p, respectively. Not required; can set to float('inf') or some very
          large values with no practical impact on performance
        (array of 2 floats) d_bounds: lower and upper bounds on d, respectively.
          Not required; can set to float('inf') or some very large values with
          no practical impact on performance

    Returns:
        (2-tuple): best solution (array of [p_x, p_y, p_z, u_x, u_y, u_z, a_x,
          a_y, a_z, d]) and the loss that the solution produces
    """

    def u_constraint(x):
        """Because this is included as in equality constraint, it must equal 0
        to be feasible
        """
        return np.linalg.norm(np.array([x[3], x[4], x[5]])) - 1

    def a_constraint(x):
        """Because this is included as in equality constraint, it must equal 0
        to be feasible
        """
        return np.linalg.norm(np.array([x[6], x[7], x[8]])) - 1

    def p_bound(x):
        """Because this is included as an inequality constraint, it must
        return non-negative to be feasible
        """
        n1, m1 = p_bounds[0]
        n2, m2 = p_bounds[1]
        n3, m3 = p_bounds[2]

        res = ((-(2/(n1-m1)*x[0]-((n1+m1)/(n1-m1)))**2) + 1) * \
              ((-(2/(n2-m2)*x[1]-((n2+m2)/(n2-m2)))**2) + 1) * \
              ((-(2/(n3-m3)*x[2]-((n3+m3)/(n3-m3)))**2) + 1)

        return res


    def d_bound(x):
        """Because this is included as an inequality constraint, it must
        return non-negative to be feasible

        https://www.desmos.com/calculator/st7aevwbyk
        """
        m, n = d_bounds
        return (-(2/(n-m)*x[9]-((n+m)/(n-m)))**2) + 1


    res = minimize(
        loss_fn,
        np.array([*p_est, *u_est, *a_est, d_est]),
        args=(transforms, measurements),
        method='slsqp',
        constraints=[
            {
                'type': 'eq',
                'fun': u_constraint
            },
            {
                'type': 'eq',
                'fun': a_constraint
            },
            {
                'type': 'ineq',
                'fun': p_bound
            },
            {
                'type': 'ineq',
                'fun': d_bound
            }
        ]
    )

    soln = res.x

    return ((soln[0], soln[1], soln[2]), (soln[3], soln[4], soln[5]), (soln[6], soln[7], soln[8]), soln[9]), res

def loss_fn(x, transforms, measurements):
    """ General loss function for calibration procedure, used by slsqp direct
    minimization functions
    """
    p = (x[0], x[1], x[2])
    u = (x[3], x[4], x[5])
    a = (x[6], x[7], x[8])
    d = x[9]

    lhs = [] # left hand side of equation (A in Ax=b)
    rhs = [] # right hand side of equation (b in Ax=b)

    # construct the rhs and lhs row by row
    for tf, dist in zip(transforms, measurements):
        lhs.append([
                a[0]*tf[0][0]+a[1]*tf[1][0]+a[2]*tf[2][0],
                a[0]*tf[0][1]+a[1]*tf[1][1]+a[2]*tf[2][1],
                a[0]*tf[0][2]+a[1]*tf[1][2]+a[2]*tf[2][2],
                dist*(a[0]*tf[0][0]+a[1]*tf[1][0]+a[2]*tf[2][0]),
                dist*(a[0]*tf[0][1]+a[1]*tf[1][1]+a[2]*tf[2][1]),
                dist*(a[0]*tf[0][2]+a[1]*tf[1][2]+a[2]*tf[2][2])
            ])

        rhs.append([-d-a[0]*tf[0][3]-a[1]*tf[1][3]-a[2]*tf[2][3]])
    
    a = np.array(lhs)
    b = np.array(rhs)
    x = np.array([p[0], p[1], p[2], u[0], u[1], u[2]])

    residuals = np.sum((b.flatten() - (a @ x))**2)

    return residuals