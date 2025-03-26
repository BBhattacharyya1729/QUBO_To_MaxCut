import numpy as np
import cvxpy as cp
from scipy.stats import ortho_group
import scipy 

def randm(n):
    """
    Returns a random symmetric matrix drawn uniformly from [-1, 1] (from the continuous interval [-1, 1]).
    
        Parameters:
            n (int): The size of the matrix.
        Returns:
            np.ndarray: A symmetric (n x n) matrix with random values from [-1,1].
    """
    
    Q= np.random.uniform(low=-1.0, high=1, size=(n,n))
    return np.tril(Q)+np.tril(Q,-1).T

def drandm(n):
    """
    Returns a random matrix drawn uniformly from {-1,1} (either -1 or 1).
    
        Parameters:
            n (int): The size of the matrix.
        Returns:
            np.ndarray: A symmetric (n x n) matrix with random values from {-1,1}.
    """
    
    Q= 2*np.random.randint(0,2, size=(n,n))-1
    return np.tril(Q)+np.tril(Q,-1).T

def get_cost(A,w):
    """
    Returns the quadratic cost function. This is agnostic as to whether the problem is 0-1 or pm formulated (hence w).

    Parameters:
        A (np.ndarray): A square matrix representing cost coefficients.
        w (np.ndarray): A vector representing decision variables.
    Returns:
        float: Quadratic cost value.
    """
    
    return w.dot(A).dot(w)

def brute_force_max01(Q):
    """
    Returns a solved 0-1 QUBO problem using brute force, checks all possible binary solutions (of length n) to find the one that maxamizes the quadratic cost function.
    
    Parameters:
        Q (np.ndarray): A square matrix representing the QUBO problem.
    Returns:
        tuple:
            max_x (np.ndarray): The 01 vector (of length n) that maxamizes the cost function.
            max_f (float): The max cost value.
    """
    
    n=len(Q)
    max_x = []
    max_f = -np.inf
    for k in range(2**n):
        x = np.array([i for i in '0'*(n-len(bin(k)[2:]))+bin(k)[2:]],dtype=int)
        f = get_cost(Q,x)
        if(f>max_f):
            max_x = x
            max_f = f
    return max_x,max_f

def brute_force_maxpm(Q):
    """
    Returns a solved pm QUBO problem using brute force, checks all possible binary solutions (of length n) to find the one that maxamizes the quadratic cost function.
    
    Parameters:
        Q (np.ndarray): A square matrix representing the QUBO problem.
    Returns:
        tuple:
            max_x (np.ndarray): The pm vector (of length n) that maxamizes the cost function.
            max_f (float): The max cost value.
    """
    
    n=len(Q)
    max_y = []
    max_f = -np.inf
    for k in range(2**n):
        x = np.array([i for i in '0'*(n-len(bin(k)[2:]))+bin(k)[2:]],dtype=int)
        y = 2*x-1
        f = get_cost(Q,y)
        if(f>max_f):
            max_y = y
            max_f = f
    return max_y,max_f

def dual_graph(Q):
    """
    Returns the dual graph of the QUBO, going from (n + n) to (n+1, n+1). Negates the original (nxn) QUBO, adds last row/column represents the row sums of Q.
    
    Parameters:
        Q (np.ndarray): A square matrix representing the QUBO problem.
        
    Returns:
        A (np.ndarray): The dual graph of the QUBO.
    """
    
    n = len(Q)
    A = np.zeros((n+1,n+1))
    for i in range(n):
        for j in range(n):
            A[i][j] = -Q[i][j]
    for i in range(n):
        A[i][n] = A[n][i] = np.sum(Q[i])
    return A

def map_reduce(y):
    """
    Returns n+1 pm variable (y) to n 01 variable (x).

    Parameters:
        y (np.ndarray): A pm vector of length n + 1.
        
    Returns:
        np.ndarray: A 01 vector of length n.
    """
    
    y = (y+1)//2
    return np.mod(y[:-1] + y[-1],2)

def GW(A):
    """
    Returns the approximation to the max-cut problem using the Goemans-Williamson (GW) semidefinite relaxation.
    
    Parameters:
        A: (np.ndarray): A ymmetric adjacency matrix of the graph.
        
    Returns:
        np.ndarray: A set of column vectors representing the graph in a higher dimensional space.
    """
    
    n=len(A)
    M=cp.Variable((n,n),PSD=True)
    constraints = [M >> 0]
    constraints += [
        M[i,i] == 1 for i in range(n)
    ]
    objective = cp.trace(-1/4 * A @ M)
    prob = cp.Problem(cp.Maximize(objective),constraints)
    prob.solve()
    
    L,d,_ = scipy.linalg.ldl(M.value)
    d = np.diag(d).copy()
    d = d*(d>0)
    
    d = np.sqrt(d)
    d.shape = (-1,1)
    return d * L.T

def BM2_cost(A,theta_list):
    """
    Computes the cost for BM3.
    
    Parameters:
        A: (np.ndarray): A symmetric adjacency matrix of the graph.
        theta_list (np.ndarray) A list of angles.

    Returns:
        float: The calculated cost value for the given angles.
    """
    
    Y = np.array([[np.cos(t),np.sin(t)] for t in theta_list]).T
    return np.trace(-1/4 * A.T.dot(Y.T.dot(Y)))

def BM3_cost(A,theta_list):
    """
    Computes the cost for BM2.
    
    Parameters:
        A: (np.ndarray): A nxn symmetric adjacency matrix of the graph.
        theta_list (np.ndarray) A list of angles.

    Returns:
        float: The calculated cost value for the given angles.
    """
    
    Y = np.array([[np.sin(t[0])*np.cos(t[1]),np.sin(t[0])*np.sin(t[1]),np.cos(t[0])] for t in theta_list]).T
    return np.trace(-1/4 *A.T.dot(Y.T.dot(Y)))

def solve_BM2(A,iters = 100, reps=50, eta = 0.05):
    """
    Returns the approximation to the max-cut problem using the Burer Monteiro 2 (BM2) semidefinite relaxation via stochastic pertubations on adjaceny matrix A.
    
    Parameters:
        A: (np.ndarray): A symmetric adjacency matrix of the graph.
        iters (int): Number of iterations per repitiion, default is 100.
        reps (int): Number of repititions, default is 50.
        eta (float): Step size of pertubation, default is 0.05.
        
    Returns:
        tuple:
            np.ndarray: A set of column vectors representing the graph in a higher dimensional space.
            max_theta (np.ndarray): The best set of angles.
            max (float): The maximum cost achieved.
    """
    
    n=len(A)
    max  = -np.inf
    max_theta_list = None
    for i in range(reps):
        theta_list = np.random.random(n)*2*np.pi
        current_cost = BM2_cost(A,theta_list)
        for k in range(iters):
            temp_theta_list = np.random.random(n)*2*np.pi + np.random.random(n)*2*eta-eta
            temp_cost=BM2_cost(A,temp_theta_list)
            if(abs(temp_cost-current_cost) < 1e-11):
                break
            if(temp_cost  > current_cost):
                theta_list = temp_theta_list
                current_cost = temp_cost
        if(max < current_cost):
            max=current_cost
            max_theta_list=fix(theta_list)
    return np.array([[np.cos(t),np.sin(t)] for t in max_theta_list]).T,max_theta_list,max

def solve_BM3(A,iters = 100, reps=50, eta = 0.05):
    """
    Returns the approximation to the max-cut problem using the Burer Monteiro 3 (BM3) semidefinite relaxation via stochastic pertubations on adjaceny matrix A.
    
    Parameters:
        A: (np.ndarray): A symmetric adjacency matrix of the graph.
        iters (int): Number of iterations per repitiion, default is 100.
        reps (int): Number of repititions, default is 50.
        eta (float): Step size of pertubation, default is 0.05.
        
    Returns:
        tuple:
            np.ndarray: A set of column vectors representing the graph in a higher dimensional space.
            max_theta (np.ndarray): The best set of angles.
            max (float): The maximum cost achieved.
    """
    
    n=len(A)
    max  = -np.inf
    max_theta_list = None
    for i in range(reps):
        theta_list = np.random.random((n,2))*2*np.pi
        theta_list = fix(theta_list)
        current_cost = BM3_cost(A,theta_list)
        for k in range(iters):
            temp_theta_list = np.random.random((n,2))*2*np.pi + np.random.random((n,2))*2*eta-eta
            temp_cost=BM3_cost(A,temp_theta_list)
            if(abs(temp_cost-current_cost) < 1e-11):
                break
            if(temp_cost  > current_cost):
                theta_list = temp_theta_list
                current_cost = temp_cost
        if(max < current_cost):
            max=current_cost
            max_theta_list=fix(theta_list)
    return np.array([[np.sin(t[0])*np.cos(t[1]),np.sin(t[0])*np.sin(t[1]),np.cos(t[0])] for t in max_theta_list]).T,max_theta_list,max

def GW2(A,reps=50,GW_Y=None):
    """
    The projected GW2 relaxation for adjaceny matrix A.
    
    Parameters:
        A (np.ndarray): A symmetric adjacency matrix of the graph.
        reps (int): Number of repititions, default is 50.
        GW_Y (np.ndarray): An initial embedding of the graph, default is GW(A)

    Returns:
        tuple:
            max_Y (np.ndarray): The best 2D embedding of matrix Y.
            np.ndarray: The set of angles representing Y.
            max (float): The maximum computed cost value.    
    """
    
    if(GW_Y is None):
        GW_Y = GW(A)
    max  = -np.inf
    max_Y = None
    for i in range(reps):
        ortho = ortho_group.rvs(len(A))
        basis = ortho.T[:2]
        Y = basis.dot(GW_Y)
        Y=Y/np.linalg.norm(Y,axis=0)
        if(max < np.trace(-1/4 * A @ Y.T.dot(Y))):
            max=np.trace(-1/4 * A @ Y.T.dot(Y))
            max_Y=Y
        max_Y = np.minimum(np.maximum(-1,max_Y),1)
    return max_Y,get_angle(max_Y),max

def GW3(A,reps=50,GW_Y=None):
    """
    The projected GW3 relaxation for adjaceny matrix A.
    
    Parameters:
        A (np.ndarray): A symmetric adjacency matrix of the graph.
        reps (int): Number of repititions, default is 50.
        GW_Y (np.ndarray): An initial embedding of the graph, default is GW(A)

    Returns:
        tuple:
            max_Y (np.ndarray): The best 2D embedding of matrix Y.
            np.ndarray: The set of angles representing Y.
            max (float): The maximum computed cost value.    
    """
    
    if(GW_Y is None):
        GW_Y = GW(A)
    max  = -np.inf
    max_Y = None
    for i in range(reps):
        ortho = ortho_group.rvs(len(A))
        basis = ortho.T[:3]
        Y = basis.dot(GW_Y)
        Y=Y/np.linalg.norm(Y,axis=0)
        if(max < np.trace(-1/4 * A @ Y.T.dot(Y))):
            max=np.trace(-1/4 * A @ Y.T.dot(Y))
            max_Y=Y
        max_Y = np.minimum(np.maximum(-1,max_Y),1)
    return max_Y,get_angle(max_Y), max

def random_round(Y,A,reps=50):
    """
    Generates random hyperplanes and finds the best partitioning.

    Parameters:
        Y (np.ndarray): 
        A (np.ndarray): 
        reps (int): The number of repititions (defaults to 50).

    Returns:
        tuple:
            max_y (np.ndarray):
            max (float): The highest cost achieved by the best hyperplane.
            max_u(np.ndarray):
        
    """
    k = Y.shape[0]
    max_y = None
    max = -np.inf
    max_u = None
    n=len(A)
    for i in range(reps):
        u = 2* np.random.random(k)-1
        y=np.array(np.sign(Y.T @ u),dtype=int)
        l=get_cost(-1/4 * A,y)
        if(l>max):
            max = l
            max_y = y
            max_u=u
    return max_y,max,max_u

def quality(Y,A):
    """
    Compute the quality of the given embedding Y in terms of A.
    
    Parameters:
        Y (np.ndarray): The embedding matrix.
        A (np.ndarray): The adjaceny matrix.

    Returns:
        float: The quality of Y.
    """
    M = Y.T.dot(Y)
    M = np.maximum(-1,np.minimum(1,M))
    return 1/(2*np.pi) * np.trace(A @ np.arccos(M)).real

def fix(theta_list):
    """
    Normalizes the angles to [0, 2pi] for 1D and 2D arrays.

    Paremters:
        theta_list (np.ndarray): A list of angles.

    Returns:
        The normalized/adjusted list of angles.
    """
    if(theta_list.ndim ==1):
        return np.mod(theta_list, 2*np.pi)
    if(theta_list.ndim ==2):
        new_list = []
        for i in range(len(theta_list)):
            theta,phi = theta_list[i]
            theta %= 2*np.pi
            if(theta > np.pi):
                theta = 2*np.pi-theta
                phi = phi + np.pi
            phi %= 2*np.pi
            new_list.append([theta,phi])
        return np.array(new_list)

def get_angle(Y):
    """
    Computes the angles for a 2D or 3D embedding matrix Y.
    
    Parameters:
        Y (np.ndarray): The embedding matrix.

    Returns:
        np.ndarray: A list of angles.
    """
    if(Y.shape[0] == 3):
        theta_list = []
        for i in Y.T:
            theta = np.arccos(i[2])
            phi  = np.arctan2(i[1],i[0]) % (2*np.pi)
            theta_list.append([theta,phi])
        return np.array(theta_list)
    if(Y.shape[0] == 2):
        theta_list = np.array([np.arctan2(*Y.T[i][::-1]) for i in range(len(Y.T))])
        return theta_list % (2*np.pi)

def vertex_on_top(theta_list,rotation = None,z_rot=None):
    """
    

    Parameters:
        theta_list (np.ndarray): A 2D array of angles in polar (2D) or spherical (3D) coordinates.
        rotation (int): The index of the vertex to move to the top, defaults to None.
        z_rot (float): Angle for an z-axis rotation, defaults to None.

    Returns:
        np.ndarray: The rotated vertices in spherical coordinates.

    """
    if(rotation is None):
        return theta_list
    else:
        if(theta_list.ndim == 1):
            return fix(theta_list - theta_list[rotation])
        else:
            theta, phi = theta_list[rotation]
            temp_thetas = fix(theta_list - np.stack([np.zeros(len(theta_list)),np.ones(len(theta_list)) * phi]).T)
            Y = np.array([[np.sin(t[0])*np.cos(t[1]),np.sin(t[0])*np.sin(t[1]),np.cos(t[0])] for t in temp_thetas]).T
            R_y = np.array([[np.cos(theta), 0 , -np.sin(theta)],[0,1,0],[np.sin(theta), 0 , np.cos(theta)]])
            Y = np.dot(R_y,Y)
            if z_rot is None:
                mu=np.random.random() * 2 * np.pi
            else:
                mu = z_rot
            Y = np.minimum(np.maximum(-1,Y),1)
            return fix(get_angle(Y) + np.stack([np.zeros(len(theta_list)),np.ones(len(theta_list)) * mu]).T)
        