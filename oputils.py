import numpy as np
from functools import reduce
import scipy 
import itertools
import networkx as nx

def num_to_bin(n,N):
    """
    Creates a binary operator object of all 0s for vals with length N and coefficent n.
    
    Parameters:
        n (int): The value of the coefficient.
        N (int): The length of the binary number.
        
    Returns:
        binary_op: An empty binary operator object of length N and coefficient n.
    """
    return binary_op([n],[[0] * N])

def eval_op(op,x):
    """
    Evaluates a QUBO function for a given binary input.
    
    Parameters:
        op (binary_op): A binary operator object.
        x (np.ndarray): A binary number.

    Returns:
        int: The computed value of the QUBO polynomial.
    """
    return np.prod((1-np.array(op.vals)) | x,axis=1).dot(op.coeffs)

def arr_to_bin(arr,N):
    """
    Creates a binary object operator each with length n and the coefficent corresponding to the value of the array.
    
    Parameters:
        arr (np.ndarray): A list of nums (or list of lists of nums with the same length).
        N (int): The desired length of the binary object operator.

    Returns:
        np.ndarray: A list of binary operator objects (or a list of lists of binary operator objects).
    """
    if(len(arr.shape)==1):
        return np.array([num_to_bin(x,N) for x in arr])
    else:
        return np.array([[num_to_bin(arr[i][j],N) for j in range(len(arr[i]))] for i in range(len(arr))])


def indexed_bin_op(i,N):
    """
    Creates a binary operator of length N with a 1 at index i (all other values 0).

    Parameters:
        i (int): The index of 1 of the binary operator.
        N (int): The length of the binary operator.

    Returns:
        binary_op: A binary operator with only the selected index at 1.
    """
    return binary_op([1],[[0 if n!=i else 1 for n in range(N)]])
    
class binary_op():
    def __init__(self,coeffs,vals):
        """
        Initializes a binary operation object.
        
        Parameters:
            coeffs (list): Coefficients associated with numbers.
            vals (list): Binary representation of numbers.

        Returns:
            None
        """
        self.N = len(vals[0])
        self.coeffs = coeffs
        self.vals = vals

    def reduce(self):
        """
        Reduces the duplicate binary values and combines their coefficents.
        
        Parameters:
            None.

        Returns:
            binary_op: The binary operator without duplicate values or 0 in binary if all coefficients are reduced to 0.
        """
        new_coeffs = []
        new_vals = []
        for v in np.unique(self.vals,axis=0).tolist():
            new_vals.append(v)
            new_coeffs.append(0)
            for i,v0 in enumerate(self.vals):
                if(v0==v):
                    new_coeffs[-1]+=self.coeffs[i]
        for i,c in enumerate(new_coeffs):
            if(c==0):
                new_coeffs.pop(i)
                new_vals.pop(i)
        if(len(new_vals)>0):
            return binary_op(new_coeffs,new_vals)
        else:
            return num_to_bin(0,self.N)

    def __add__(self,b2):
        """
        Adds two binary operator objects.
        
        Parameters:
            b2 (binary_op): A binary operator object.

        Returns:
            binary_op: The sum of two binary operator objects.
        """
        new_vals = self.vals + b2.vals
        new_coeffs = self.coeffs + b2.coeffs
        return binary_op(new_coeffs,new_vals).reduce()
    
    def __truediv__(self,n):
        """
        Divides a binary operator object by a number (specifically the coefficients).
        
        Parameters:
            n (int/float): The divisor.

        Returns:
            binary_op: The quotient of the binary operator object and the number.
        """
        return binary_op([c/n for c in self.coeffs],self.vals)

    def __sub__(self,b2):
        """
        Subtracts a binary operator object from another binary operator object.
    
        Parameters:
            b2 (binary_op): A binary operator object.

        Returns:
            binary_op: The difference of two binary operator objects 
        """
        return self + binary_op([-c for c in b2.coeffs],b2.vals)
        
    def __mul__(self,b2):
        """
        Multiplies each binary number in each of the binry operator objects using Bitwise OR, coefficients are mutiplied by one another.
        
        Parameters:
            b2 (binary_op): A binary operator object.

        Returns:
            binary_op: The product of two binary operator objects.
        """
        new_vals = []
        new_coeffs = []
        for i,v1 in enumerate(self.vals):
            for j,v2 in enumerate(b2.vals):
                new_coeffs.append(self.coeffs[i] * b2.coeffs[j])
                new_vals.append((np.array(v1) | np.array(v2)).tolist())
        return binary_op(new_coeffs,new_vals).reduce()    
        
    def __pow__(self,n):
        """
        Takes a binary operator object to a power.
        
        Parameters:
            n (int): The number which the binary operator object is being raised to.
        
        Returns:
            Returns the result of mutipling the binary operator object by itself n times.
        """
        return reduce(lambda x,y:x*y, [self]*n)
            
    def to_matrix(self):
        """
        Converts the binary operator object to a matrix.
        
        Parameters:
            None.

        Returns:
            output (scipy.sparse.csr_array): A matrix representing the binary operator.
        """
        output = scipy.sparse.csr_array(np.zeros((2**self.N,2**self.N)))
        I=scipy.sparse.csr_matrix(np.eye(2))
        x=scipy.sparse.csr_array(np.array([[1,0],[0,0]]))
        for i,v in enumerate(self.coeffs):
            output+=v * reduce(lambda x,y:scipy.sparse.kron(x,y),[I if val==0 else x for val in self.vals[i]])
        return output

    def __str__(self):
        return f'Coeffs: {self.coeffs} Values: {self.vals}'

def bin_to_qubo(op):
    """
    Converts a binary operator object to a QUBO matrix and constant.
    
    Parameters:
        op (binary_op): A binary operator object.
        
    Returns:
        tuple:
            const (float): A constant term, sum of the coefficents of the binary operator.
            Q (np.ndarray): The QUBO matrix. 
    """
    const = 0 
    Q = np.zeros((op.N,op.N))
    for i,v in enumerate(op.vals):
        if(np.sum(v)==0):
            const+=op.coeffs[i]
        elif(np.sum(v)==1):
            Q[np.where(np.array(v)==1)[0][0]][np.where(np.array(v)==1)[0][0]]+=op.coeffs[i]
        elif(np.sum(v)==2):
            Q[np.where(np.array(v)==1)[0][1]][np.where(np.array(v)==1)[0][0]]+=op.coeffs[i]/2
            Q[np.where(np.array(v)==1)[0][0]][np.where(np.array(v)==1)[0][1]]+=op.coeffs[i]/2
        else:
            print("error")
    return const,Q

'''''''''''''''''''''''''''''''''''''''''''TSP Methods'''''''''''''''''''''''''''''''''''''''''''''''''''

def TSP_QUBO(n,max_val,eps= 0.1):
    """
    Creates a QUBO problem representing the TSP.

    Parameters:
        n (int): The number of cities.
        max_val (float): The maximum value for the coordinates randomly positioned in 2D space between -max_val and +max_val.
        eps (float): The scale of the penalty term.
        
    Returns:
        tuple:
            points (np.ndarray): A 2D array of shape (n, 2), representing the coordinates of the cities.
            W_mat (np.ndarry): Distance matrix between all pairs of cities (n x n).
            np.ndarray: The QUBO matrix representing the TSP.
    """
    points = np.random.random((n,2)) * max_val * 2 - max_val
    W_mat = np.array([[np.linalg.norm(points[i]-points[j]) for i in range(n)] for j in range(n)])

    A = num_to_bin((1+eps)*np.max(W_mat),(n-1)**2)
    reward = num_to_bin(0,(n-1)**2)
    for i in range(0,n-1):
        for j in range(0,n-1):
            for t in range(0,n-2):
                reward += num_to_bin(W_mat[i+1][j+1],(n-1)**2) * indexed_bin_op((n-1)*t+i,(n-1)**2) *  indexed_bin_op((n-1)*(t+1)+j,(n-1)**2)
        reward += num_to_bin(W_mat[0][i+1],(n-1)**2) * indexed_bin_op(i,(n-1)**2) 
        reward += num_to_bin(W_mat[0][i+1],(n-1)**2) *  indexed_bin_op((n-1)*(n-2)+i,(n-1)**2)
    
    penalty1=num_to_bin(0,(n-1)**2)
    for t in range(n-1):
        temp_op = num_to_bin(1,(n-1)**2)
        for i in range(n-1):
            temp_op -= indexed_bin_op((n-1)*t+i,(n-1)**2)
        penalty1+=temp_op**2
    
    penalty2=num_to_bin(0,(n-1)**2)
    for i in range(n-1):
        temp_op = num_to_bin(1,(n-1)**2)
        for t in range(n-1):
            temp_op -= indexed_bin_op((n-1)*t+i,(n-1)**2)
        penalty2+=temp_op**2
    
    cost = reward+A*(penalty1+penalty2)

    return points,W_mat, bin_to_qubo(cost)

'''''''''''''''''''''''''''''''''''''''''''Portfiolio Opt'''''''''''''''''''''''''''''''''''''''''''''''''''

def geometric_brownian_prices(max_drift=0.05,max_volatility=0.2,T=250,N=4):
    """
    Simulates the prices of assets using geometric brownian motion.

    Parameters:
        max_drift (float): The max drift of the assets (avg rate of return).
        max_volatility (float): The max volatility of the assets (standard deviation).
        T (int): The number of time steps.
        N (int): The number of assets in the portfolio.
        
    Returns:
        np.ndarray: A 2D array of shape (T+1,N) representing simulated price assets.
    """
    mu = np.random.random(N) * max_drift*2 - max_drift
    sigma = np.random.random(N) * max_volatility*2-max_volatility
    W = np.array([np.concatenate(([0],np.cumsum(np.random.normal(size = T))/np.sqrt(T))) for i in range(N)]).T
    init = np.ones(N) 
    return np.array([init * np.exp((mu-sigma**2/2)*k/N+sigma * W[k]) for k in range(T+1)])

def mu_sigma(p_data):
    """
    Computes the mean returns and covariance matrix.

    Parameters:
        p_data (np.ndarray): A 2D array of shape (T+1,N) representing simulated price assets.
        
    Returns:
        tuple:
            mean_returns (np.ndarray): The mean returns for each asset.
            np.ndarray: The covariance matrix of asset returns.
    """
    T = len(p_data)-1
    return_data = (np.roll(p_data,-1,axis=0)[:T]-p_data[:T])/p_data[:T]
    mean_returns = np.mean(return_data,axis=0)
    deviations = return_data-mean_returns
    return mean_returns, (deviations.T @ deviations)/(T-1)

def simple_PO_QUBO(mu,sigma,q=0.5,B=None,c = None):
    """
    Creates a QUBO problem from portfolio optimization.
    
    Parameters:
        mu (np.ndarray): The mean return of the assets.
        sigma (np.ndarray): The covariance matrix of asset returns.
        q (float): The penalty factor for risk.
        B (int or None): The target portfolio size (default of half).
        c(int or None): The penalty scaling term (default of the sum of magnitudes and expected risk of the assets).
        
    Returns:
        np.ndarray: The QUBO matrix representing the portfolio optimization problem.
    """
    if(c is None):
        c = np.sum(np.abs(mu))+np.sum(np.abs(sigma))
    N = len(mu)
    op = num_to_bin(0, N)
    if(B is None):
        B = N//2
    sum_op =num_to_bin(B,N)
    for i in range(N):
        op += num_to_bin(mu[i],N) * indexed_bin_op(i,N)
        sum_op -= indexed_bin_op(i,N)
        for j in range(N):
            op -= num_to_bin(q*sigma[i][j],N) * indexed_bin_op(i,N) * indexed_bin_op(j,N)
    cost = num_to_bin(-c,N)*sum_op**2 + op
    return bin_to_qubo(cost)

'''''''''''''''''''''''''''''''''''''''''''Max Independent Set'''''''''''''''''''''''''''''''''''''''''''''''''''

def sample_graph(n,graph='gnp'):
    """
    Creates a random connected graph.
    
    Parameters:
        n (int): The number of nodes in the graph.
        graph (str): The type of graph, either gnp or nws.
    
    Returns:
        networkx.Graph: The connected graph.
    """
    connected=False
    G = None
    while(not connected):
        if(graph == 'gnp'):
            G=nx.gnp_random_graph(n,0.25)
        elif(graph=='nws'):
            G=nx.newman_watts_strogatz_graph(n,3,0.5)
        else:
            return "?????????"
        connected = nx.is_connected(G)
        
    return G
    
def MIS_QUBO(n,graph='gnp',A=1.1):
    """
    Creates QUBO problem for MIS.

    Parameters:
        n (int): The number of nodes in the graph.
        graph (str): The type of graph, either gnp or nws.
        A (float): Penalty for selecting adjacent nodes.

    Returns:
        np.ndarray: The QUBO matrix representing the MIS problem.
    """
    G = sample_graph(n,graph)
    return  np.diag(np.ones(n))- (A/2) *nx.adjacency_matrix(G).toarray()