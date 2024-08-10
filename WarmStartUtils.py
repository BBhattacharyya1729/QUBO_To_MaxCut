import numpy as np
import cvxpy as cp
from scipy.stats import ortho_group
import scipy 

"""
Generate a random matrix drawn uniformly from [-1,1]
"""
def randm(n):
    Q= np.random.uniform(low=-1.0, high=1, size=(n,n))
    return np.tril(Q)+np.tril(Q,-1).T
"""
Generate a random matrix drawn uniformly from {-1,1}
"""
def drandm(n):
    Q= 2*np.random.randint(0,2, size=(n,n))-1
    return np.tril(Q)+np.tril(Q,-1).T
"""
Get the quadratic cost function. This is agnostic as to whether the problem is 01 or pm formulated (hence w).
"""
def get_cost(A,w):
    return w.dot(A).dot(w)

"""
Brute-force max solve a 01 QUBO. SLOW
"""
def brute_force_max01(Q):
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

"""
Brute-force max solve a pm QUBO. SLOW
"""
def brute_force_maxpm(Q):
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


"""
Dual graph of Qubo A
"""
def dual_graph(Q):
    n = len(Q)
    A = np.zeros((n+1,n+1))
    for i in range(n):
        for j in range(n):
            A[i][j] = -Q[i][j]
    for i in range(n):
        A[i][n] = A[n][i] = np.sum(Q[i])
    return A

"""
Reduce n+1 pm variable (y) to n 01 variable (x)
"""
def map_reduce(y):
    y = (y+1)//2
    return np.mod(y[:-1] + y[-1],2)

"""
GW Warmstart for adjacency matrix A
"""
def GW(A):
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

"""
BM2 Cost
"""
def BM2_cost(A,theta_list):
    Y = np.array([[np.cos(t),np.sin(t)] for t in theta_list]).T
    return np.trace(-1/4 * A.T.dot(Y.T.dot(Y)))

"""
BM3 Cost
"""
def BM3_cost(A,theta_list):
    Y = np.array([[np.sin(t[0])*np.cos(t[1]),np.sin(t[0])*np.sin(t[1]),np.cos(t[0])] for t in theta_list]).T
    return np.trace(-1/4 *A.T.dot(Y.T.dot(Y)))

"""
BM2 via Stochastic perturb on adj. matrix A
"""
def solve_BM2(A,iters = 100, reps=50, eta = 0.05):
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

"""
BM2 via Stochastic perturb on adj. matrix A
"""
def solve_BM3(A,iters = 100, reps=50, eta = 0.05):
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

"""
proj-GW2 for adj. matrix A
"""
def GW2(A,reps=50,GW_Y=None):
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

"""
proj-GW3 for adj. matrix A
"""
def GW3(A,reps=50,GW_Y=None):
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

"""
Random round 
"""
def random_round(Y,A,reps=50):
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

"""
Quality of Y
"""
def quality(Y,A):
    M = Y.T.dot(Y)
    M = np.maximum(-1,np.minimum(1,M))
    return 1/(2*np.pi) * np.trace(A @ np.arccos(M)).real

"""
Fix theta_list
"""
def fix(theta_list):
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

"""
Get angle of Y
"""
def get_angle(Y):
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

"""
Vertex on top rotation
"""
def vertex_on_top(theta_list,rotation = None,z_rot=None):
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
        