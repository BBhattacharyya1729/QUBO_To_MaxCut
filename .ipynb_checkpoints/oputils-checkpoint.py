import numpy as np
from functools import reduce
import scipy 
import itertools
import networkx as nx
# def num_to_spin(n,N):
#         return spin_op([n],[[0] * N])

def num_to_bin(n,N):
        return binary_op([n],[[0] * N])

# def arr_to_spin(arr,N):
#     if(len(arr.shape)==1):
#         return np.array([num_to_spin(x,N) for x in arr])
#     else:
#         return np.array([[num_to_spin(arr[i][j],N) for j in range(len(arr[i]))] for i in range(len(arr))])

def eval_op(op,x):
    return np.prod((1-np.array(op.vals)) | x,axis=1).dot(op.coeffs)

def arr_to_bin(arr,N):
    if(len(arr.shape)==1):
        return np.array([num_to_bin(x,N) for x in arr])
    else:
        return np.array([[num_to_bin(arr[i][j],N) for j in range(len(arr[i]))] for i in range(len(arr))])
def indexed_bin_op(i,N):
    return binary_op([1],[[0 if n!=i else 1 for n in range(N)]])
class binary_op():
    def __init__(self,coeffs,vals):
        self.N = len(vals[0])
        self.coeffs = coeffs
        self.vals = vals

    def reduce(self):
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
        new_vals = self.vals + b2.vals
        new_coeffs = self.coeffs + b2.coeffs
        return binary_op(new_coeffs,new_vals).reduce()

    
    def __truediv__(self,n):
        return binary_op([c/n for c in self.coeffs],self.vals)

    def __sub__(self,b2):
        return self + binary_op([-c for c in b2.coeffs],b2.vals)
        
    def __mul__(self,b2):
        new_vals = []
        new_coeffs = []
        for i,v1 in enumerate(self.vals):
            for j,v2 in enumerate(b2.vals):
                new_coeffs.append(self.coeffs[i] * b2.coeffs[j])
                new_vals.append((np.array(v1) | np.array(v2)).tolist())
        return binary_op(new_coeffs,new_vals).reduce()    

    # def to_spin_op(self):
    #     op = num_to_spin(0,self.N)
    #     for i,v in enumerate(self.coeffs):
    #         new_op = num_to_spin(v,self.N)
    #         for j,val in enumerate(self.vals[i]):
    #             if(val == 1):
    #                 new_op *= (spin_op([0.5],[[0 if k!=j else 1 for k in range(self.N)]]) + num_to_spin(0.5,self.N))
    #         op+=new_op.reduce()
    #     return op.reduce()
        
    def __pow__(self,n):
        return reduce(lambda x,y:x*y, [self]*n)
            
    def to_matrix(self):
        output = scipy.sparse.csr_array(np.zeros((2**self.N,2**self.N)))
        I=scipy.sparse.csr_matrix(np.eye(2))
        x=scipy.sparse.csr_array(np.array([[1,0],[0,0]]))
        for i,v in enumerate(self.coeffs):
            output+=v * reduce(lambda x,y:scipy.sparse.kron(x,y),[I if val==0 else x for val in self.vals[i]])
        return output

    def __str__(self):
        return f'Coeffs: {self.coeffs} Values: {self.vals}'
'''Never Used (I Think)'''
# class spin_op():
#     def __init__(self,coeffs,vals):
#         self.N = len(vals[0])
#         self.coeffs = coeffs
#         self.vals = vals

#     def reduce(self):
#         new_coeffs = []
#         new_vals = []
#         for v in np.unique(self.vals,axis=0).tolist():
#             new_vals.append(v)
#             new_coeffs.append(0)
#             for i,v0 in enumerate(self.vals):
#                 if(v0==v):
#                     new_coeffs[-1]+=self.coeffs[i]
#         for i,c in enumerate(new_coeffs):
#             if(c==0):
#                 new_coeffs.pop(i)
#                 new_vals.pop(i)
#         if(len(new_vals)>0):
#             return spin_op(new_coeffs,new_vals)
#         else:
#             return num_to_spin(0,self.N)
#     def __add__(self,b2):
#         new_vals = self.vals + b2.vals
#         new_coeffs = self.coeffs + b2.coeffs
#         return spin_op(new_coeffs,new_vals).reduce()

#     def __truediv__(self,n):
#         return spin_op([c/n for c in self.coeffs],self.vals)

#     def __sub__(self,b2):
#         return self + spin_op([-c for c in b2.coeffs],b2.vals)
        
#     def __mul__(self,b2):
#         new_vals = []
#         new_coeffs = []
#         for i,v1 in enumerate(self.vals):
#             for j,v2 in enumerate(b2.vals):
#                 new_coeffs.append(self.coeffs[i] * b2.coeffs[j])
#                 new_vals.append((np.array(v1) ^ np.array(v2)).tolist())
#         return spin_op(new_coeffs,new_vals).reduce()

#     def to_bin_op(self):
#         op = num_to_bin(0,self.N)
#         for i,v in enumerate(self.coeffs):
#             new_op = num_to_spin(v,self.N)
#             for j,val in enumerate(self.vals[i]):
#                 if(val == 1):
#                     new_op *= (binary_op([2],[[0 if k!=j else 1 for k in range(self.N)]]) - num_to_bin(1,self.N))
#             op+=new_op.reduce()
#         return op.reduce()

#     def __pow__(self,n):
#         return reduce(lambda x,y:x*y, [self]*n)
            
#     def to_matrix(self):
#         output = scipy.sparse.csr_array(np.zeros((2**self.N,2**self.N)))
#         I=scipy.sparse.csr_array(np.eye(2))
#         x=scipy.sparse.csr_array(np.array([[1,0],[0,-1]]))
#         for i,v in enumerate(self.coeffs):
#             output+=v * reduce(lambda x,y:scipy.sparse.kron(x,y),[I if val==0 else x for val in self.vals[i]])
#         return output
#     def __str__(self):
#         return f'Coeffs: {self.coeffs} Values: {self.vals}'



def bin_to_qubo(op):
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

# '''Efficent Solve for an optimum via searchin cylcic perms.'''
# def brute_force_TSP(W):
#     n=len(W)
#     min_val = np.inf
#     min_p = None
#     for p in list(itertools.permutations(range(1,n))):
#         c = W[0][p[0]]+W[0][p[-1]]
#         for i in range(n-2):
#             c+=W[p[i]][p[i+1]]
#         if(c<min_val):
#             min_val=c
#             min_p=p
#     return np.concatenate(([0],min_p))

'''''''''''''''''''''''''''''''''''''''''''Portfiolio Opt'''''''''''''''''''''''''''''''''''''''''''''''''''

def geometric_brownian_prices(max_drift=0.05,max_volatility=0.2,T=250,N=4):
    mu = np.random.random(N) * max_drift*2 - max_drift
    sigma = np.random.random(N) * max_volatility*2-max_volatility
    W = np.array([np.concatenate(([0],np.cumsum(np.random.normal(size = T))/np.sqrt(T))) for i in range(N)]).T
    init = np.ones(N) 
    return np.array([init * np.exp((mu-sigma**2/2)*k/N+sigma * W[k]) for k in range(T+1)])

def mu_sigma(p_data):
    T = len(p_data)-1
    return_data = (np.roll(p_data,-1,axis=0)[:T]-p_data[:T])/p_data[:T]
    mean_returns = np.mean(return_data,axis=0)
    deviations = return_data-mean_returns
    return mean_returns, (deviations.T @ deviations)/(T-1)

def simple_PO_QUBO(mu,sigma,q=0.5,B=None,c = None):
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
def MIS_QUBO(n,graph='gnp',A=1.1):
    if(graph == 'gnp'):
        G=nx.gnp_random_graph(n,0.25)
        return  np.diag(np.ones(n))- (A/2) *nx.adjacency_matrix(G).toarray()
    elif(graph=='nws'):
        G=nx.newman_watts_strogatz_graph(n,3,0.5)
        return np.diag(np.ones(n))- (A/2) *nx.adjacency_matrix(G).toarray()
    elif(graph=='regular'):
        G=nx.random_regular_graph(3,n)
        return np.diag(np.ones(n))-(A/2) *nx.adjacency_matrix(G).toarray()
    else:
        return "????"

