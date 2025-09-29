from qiskit.quantum_info import SparsePauliOp
from WarmStartUtils import *
import numpy as np
from functools import reduce
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator, FixedLocator, MaxNLocator, FormatStrFormatter
from tqdm import tqdm
from tqdm.contrib import itertools
from tqdm.notebook import tqdm
import pymupdf as fitz
import os


"""
Hamiltonian from adjacency matrix A
"""
def indexedZ(i,n):
    """
    Returns a SparsePauli Op corresponding to a Z operator on a single qubit

    Parameters:
        i (int): qubit index 
        n (int): number of qubits

    Returns:
        SparsePauliOp: SparsePauli for single Z operator
    """
    
    return SparsePauliOp("I" * (n-i-1) + "Z" + "I" * i)

def getHamiltonian(A):
    """
    Gets a Hamiltonian from a max-cut adjacency matrix

    Parameters:
        A (np.ndarray): max-cut adjacency matrix

    Returns:
        SparsePauliOp: Hamiltonian 
    """
    
    n = len(A)
    H = 0 * SparsePauliOp("I" * n)
    for i in range(n):
        for j in range(n):
            H -= 1/4 * A[i][j] * indexedZ(i,n) @ indexedZ(j,n)
    return H.simplify()

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

def SU2_op(x,y,z,t):
    """
    Get the matrix for a SU2 rotation around an axis 

    Parameters:
        x (float): x-coordinate of rotation axis
        y (float): y-coordinate of rotation axis
        z (float): z-coordinate of rotation axis
        t (float): rotation angle

    Returns:
        np.ndarray: matrix for the SU2 rotation
    """
    
    return np.array([[np.cos(t)-1j * np.sin(t)*z, -np.sin(t) * (y+1j * x)],[ -np.sin(t) * (-y+1j * x),np.cos(t)+1j * np.sin(t)*z]])

def apply_single_qubit_op(psi,U,q):
    """
    Efficiently apply a single qubit operator U on qubit q to statevector psi

    Parameters:
        psi (np.ndarray): Statevector
        U (np.ndarray): Operator 
        q (int): qubit index 

    Returns:
        np.ndarray: New Statevector 
    """
    
    n=int(np.log2(len(psi)))
    axes = [q] + [i for i in range(n) if i != q]
    contract_shape = (2, len(psi)// 2)
    tensor = np.transpose(
        np.reshape(psi, tuple(2 for i in range(n))),
        axes,
    )
    tensor_shape = tensor.shape
    tensor = np.reshape(
        np.dot(U, np.reshape(tensor, contract_shape)),
        tensor_shape,
    )
    return np.reshape(np.transpose(tensor, np.argsort(axes)),len(psi))

def pre_compute(A):
    """
    Pre-compute the diagonal elements of Hamiltonian corresponding to max-cut adjacency matrix

    Parameters:
        A (np.ndarray): adjacency matrix

    Returns:
        np.ndarray: Array containing matrix diagonal  
    """
    
    return np.array(scipy.sparse.csr_matrix.diagonal(getHamiltonian(np.flip(A)).to_matrix(sparse=True))).real

def apply_mixer(psi,U_ops):
    """
    Apply mixer layer to state
    
    Parameters:
        psi (np.ndarray): Original state vector
        U_op (list[np.ndarray]): list of mixer operators

    Returns:
        psi (np.ndarray): New statevector  
    """
    
    for n in range(0,len(U_ops)):
        psi = apply_single_qubit_op(psi, U_ops[n], n)
    return psi

def cost_layer(precomp,psi,t):
    """
    Given a precomputed Hamiltonian diagonal, apply the cost layer

    Parameters:
        precomp (np.ndarray): Precompute diagonal
        psi (np.ndarray): Statevector
        t (float): Rotation angle

    Returns:
        np.ndarray: New statevector
    """
    
    return np.exp(-1j * precomp*t) * psi

def QAOA_eval(precomp,params,mixer_ops=None,init=None):
    """"
    Returns statevector after applying QAOA circuit

    Parameters:
        precomp (np.ndarray): Hamiltonian diagonal
        params (np.ndarray): Array of QAOA circuit parameters
        mixer_ops (list[np.ndarray]): list of mixer parameters
        init (np.ndarray): initial state 

    Returns:
        psi (np.ndarray): new statevector
    """
    
    p = len(params)//2
    gammas = params[p:]
    betas = params[:p]
    
    psi = np.zeros(len(precomp),dtype='complex128')
    if(init is None):
        psi = np.ones(len(psi),dtype='complex128') *  1/np.sqrt(len(psi))
    else:
        psi = init

    if(mixer_ops is None):
        mixer = lambda t: [SU2_op(1,0,0,t) for m in range(int(np.log2(len(psi))))]
    else:
        mixer = mixer_ops
    
    for i in range(p):
        psi = cost_layer(precomp,psi,gammas[i])
        psi = apply_mixer(psi,mixer(betas[i]))
    return psi

def expval(precomp,psi):
    """"
    Compute the expectation value of a diagonal hamiltonian on a state

    Parameters:
        precomp (np.ndarray): Diagonal elements of Hamiltonian 
        psi (np.ndarray): Statevector 

    Returns:
        float: expectation value 
    """
    
    return np.sum(psi.conjugate() * precomp * psi).real

def Q2_data(theta_list,rotation=None):
    """"
    Get warmstart data from polar angles

    Parameters:
        theta_list (np.ndarray): A 2D array of angles in polar (2D) coordinates.
        rotation (int): The index of the vertex to move to the top, defaults to None.

    Returns:
        tuple:
            init (np.ndarray): The initial statevector
            mixer_ops (list[np.ndarray]): the mixer operators 
    """
    
    angles = vertex_on_top(theta_list,rotation)
    init = reduce(lambda a,b: np.kron(a,b), [np.array([np.cos(v/2), np.exp(-1j/2 * np.pi)*np.sin(v/2)],dtype='complex128') for v in angles])
    mixer_ops = lambda t: [  SU2_op(0,-np.sin(v),np.cos(v),t) for v in angles]
    return init,mixer_ops

def Q3_data(theta_list,rotation=None,z_rot=None):
    """"
    Get warmstart data from spherical angles

    Parameters:
        theta_list (np.ndarray): A 2D array of angles in spherical (3D) coordinates.
        rotation (int): The index of the vertex to move to the top, defaults to None.

    Returns:
        tuple:
            init (np.ndarray): The initial statevector
            mixer_ops (list[np.ndarray]): the mixer operators 
    """
    angles = vertex_on_top(theta_list,rotation,z_rot=z_rot)
    init = reduce(lambda a,b: np.kron(a,b), [np.array([np.cos(v[0]/2), np.exp(1j * v[1])*np.sin(v[0]/2)],dtype='complex128') for v in angles])
    mixer_ops = lambda t:  [SU2_op(np.sin(v[0])*np.cos(v[1]),np.sin(v[0])*np.sin(v[1]),np.cos(v[0]),t) for v in angles]
    return init,mixer_ops

def single_circuit_optimization_eff(precomp,opt,mixer_ops,init,p,param_guess=None):
    """"
    Optimize a single qaoa circuit

    Parameters:
        precomp (np.ndarray): Diagonal of cost hamiltonian
        opt (Qiskit.algorithms.optimizers): Qiskit optimizers
        mixer_ops (list[np.ndarray]): List of mixer operators
        init (np.ndarray): Initial state
        p (int): circuit depth
        param_guess (np.ndarray): initial optimization parameters. Defaults to None

    Returns:
        tuple:
            (float):  Optimal cost
            (np.ndarray): Parameters for optimal cost
            (dict): history
    """

    history = {"cost": [], "params": []} if param_guess is None else {"cost": [expval(precomp,QAOA_eval(precomp,param_guess,mixer_ops=mixer_ops,init=init))], "params": [param_guess]}
    def compute_expectation(x):
        l = expval(precomp,QAOA_eval(precomp,x,mixer_ops=mixer_ops,init=init))
        history["cost"].append(l)
        history["params"].append(x)
        return -l
    if(param_guess is None):
        init_param = 2*np.pi*np.random.random(2*p)
    else:
        init_param = param_guess
    res = opt.minimize(fun= compute_expectation, x0=init_param)
    return np.max(history["cost"]),history["params"][np.argmax(history["cost"])],history

def circuit_optimization_eff(precomp,opt,mixer_ops,init,p,reps=10,name=None,verbose=False,param_guesses=None):
    """"
    Run repeated circuit optimization

    Parameters:
        precomp (np.ndarray): Diagonal matrix elements
        opt (Qiskit.algorithms.optimizers):
        mixer_ops (list[np.ndarray]): mixer operators
        init (np.ndarray): initial statevector
        p (int): circuit depth
        reps (int): Number of optimizations. Defaults to 10
        name (str): Name for logging. Defaults to None
        verbose (bool): Whether or not to print progress. Defaults to None
        param_gueses (np.ndarray): initial circuit params. Defaults to None
    
    Returns:
        tuple:
            (list[float]):  Optimal costs
            (list[np.ndarray]): Parameters for optimal costs
            (list[dict]): histories
    """
    
    if(verbose):
        if(name is None):
            print(f"------------Beginning Optimization------------")
        else:
            print(f"------------Beginning Optimization: {name}------------")

    if(param_guesses is None):
        init_params = [None for i in range(reps)]
    else:
        init_params = param_guesses
    history_list = []
    param_list = []
    cost_list = []
    for i in range(reps):
        cost,params,history = single_circuit_optimization_eff(precomp,opt,mixer_ops,init,p,param_guess=init_params[i])
        history_list.append(history)
        param_list.append(params)
        cost_list.append(cost)
        if(verbose):
            print(f"Iteration {i} complete")
    return np.array(cost_list),np.array(param_list),history_list

def initialize(A,BM_kwargs={"iters":100, "reps":50, "eta":0.05},GW_kwargs={"reps":50}):
    """"
    Find relevant warmstart info for circuit

    Parameters:
        A (np.ndarray): Max-cut adj matrix
        BM_kwargs (dict): Dictionary of options for BM warmstart 
        GW_kwargs (dict):  Dictionary of options for GW warmstart  

    Returns:
        list:
            (np.ndarray): precomputed Hamiltonian diagonal
            (np.ndarray): BM2 angles
            (np.ndarray): BM3 angles
            (np.ndarray): GW2 angles
            (np.ndarray): GW3 angles
            (np.ndarray): Optimal solution
            (np.ndarray): Optimal cost
    """
    
    precomp = pre_compute(A)
    _,BM2_theta_list,_=solve_BM2(A,**BM_kwargs)
    _,BM3_theta_list,_=solve_BM3(A,**BM_kwargs)
    GW_Y = GW(A)
    _,GW2_theta_list,_=GW2(A,GW_Y=GW_Y,**GW_kwargs)
    _,GW3_theta_list,_=GW3(A,GW_Y=GW_Y,**GW_kwargs)
    v,M=brute_force_maxcut(A)
    return [precomp,BM2_theta_list,BM3_theta_list,GW2_theta_list,GW3_theta_list,v,M]

def warmstart_comp(A,opt,p_max=5,rotation_options=[None,0,-1],BM_kwargs={"iters":100, "reps":50, "eta":0.05},GW_kwargs={"reps":50},reps=10,optimizer_kwargs={'name':None,'verbose':True},ws_list=['BM2','BM3','GW2','GW3',None],initial_data=None,keep_hist=False):
    """"
    Run a comparison of all the warmstarts

    Parameters:
        A (np.ndarray): Max-cut adj matrix
        opt (Qiskit.algorithms.optimizers): Optimizers
        p_max (int): Maximum depth, QAOA is optimized from 1...p. Defaults to 5
        rotation_options (list): List of rotation options. Defaults to [None,0,-1]
        BM_kwargs (dict): Dictionary of BM warmstart options
        GW_kwargs (dict): Dictionary of GW warmstart options
        reps (int): Number of repetitions for each circuit, defaults to 10
        optimizer_kwargs (dict): Options for optimizers
        ws_list (list): List of warmstarts to compare
        initial_data (list): Initial warmstart data. Defaults to None
        keep_hist (bool): Wether to keep the history

    Returns:
        tuple:
            qc_data (dict): Data for circuits (warmstart info) 
            opt_data (dict): Optimization data
    """
    ###initialization
    if(initial_data is None):
        precomp = pre_compute(A)
        _,BM2_theta_list,_=solve_BM2(A,**BM_kwargs)
        _,BM3_theta_list,_=solve_BM3(A,**BM_kwargs)
        GW_Y = GW(A)
        _,GW2_theta_list,_=GW2(A,GW_Y=GW_Y,**GW_kwargs)
        _,GW3_theta_list,_=GW3(A,GW_Y=GW_Y,**GW_kwargs)
        v,_=brute_force_maxcut(A)
    else:
        precomp = initial_data[0]
        BM2_theta_list = initial_data[1]
        BM3_theta_list = initial_data[2]
        GW2_theta_list = initial_data[3]
        GW3_theta_list = initial_data[4]
        v = initial_data[5] 

    qc_data = {'BM2':{r: Q2_data(BM2_theta_list,r) for r in rotation_options},
    'BM3':{r:Q3_data(BM3_theta_list,r) for r in rotation_options},
    'GW2':{r:Q2_data(GW2_theta_list,r) for r in rotation_options},
    'GW3':{r:Q3_data(GW3_theta_list,r) for r in rotation_options}}

    #ws_list = ['BM2','BM3','GW2','GW3',None]
    opt_data =[{w:({r:{} for r in rotation_options} if w is not None else {}) for w in ws_list} for _ in range(p_max+1)]


    opt_params={w:({r:None for r in rotation_options}  if w is not None else None) for w in ws_list} 

    ###Depth 0
    for ws in ws_list:
        if(ws is None):
            opt_data[0][ws]['cost']=expval(precomp,QAOA_eval(precomp,[],mixer_ops=None,init=None))
            opt_data[0][ws]['params']=[]
            opt_data[0][ws]['probs']=opt_sampling_prob(v,precomp,[],mixer_ops=None,init=None)
        else:
            for r in rotation_options:
                opt_data[0][ws][r]['cost']=expval(precomp,QAOA_eval(precomp,[],mixer_ops=qc_data[ws][r][1],init=qc_data[ws][r][0]))
                opt_data[0][ws][r]['params']=[]
                opt_data[0][ws][r]['probs']=opt_sampling_prob(v,precomp,[],mixer_ops=qc_data[ws][r][1],init=qc_data[ws][r][0])

    ###Depth >0
    for p in range(1,p_max+1):
        for ws in ws_list:
            if(ws is None):
                guess  = [opt_params[ws]] + [None] * (reps-1)
                l=circuit_optimization_eff(precomp,opt,None,None,p,reps=reps,param_guesses=guess,**optimizer_kwargs)
                opt_data[p][ws]['cost']=l[0]
                opt_data[p][ws]['params']=l[1]
                opt_data[p][ws]['probs']=np.array([opt_sampling_prob(v,precomp,param,mixer_ops=None,init=None) for param in l[1]])
                opt_params[None] = list(l[1][np.argmax(l[0])])[:p] + [0] + list(l[1][np.argmax(l[0])])[p:] + [0]

                if(keep_hist):
                    opt_data[p][ws]['hist']=l[2]
            else:
                for r in rotation_options:
                    guess=[opt_params[ws][r]]+ [None] * (reps-1)
                    l=circuit_optimization_eff(precomp,opt,qc_data[ws][r][1],qc_data[ws][r][0],p,reps=reps,param_guesses=guess,**optimizer_kwargs)
                    opt_data[p][ws][r]['cost']=l[0]
                    opt_data[p][ws][r]['params']=l[1]
                    opt_data[p][ws][r]['probs']=np.array([opt_sampling_prob(v,precomp,param,qc_data[ws][r][1],qc_data[ws][r][0]) for param in l[1]])
                    opt_params[ws][r] = list(l[1][np.argmax(l[0])])[:p] + [0] + list(l[1][np.argmax(l[0])])[p:] + [0]
                
                    if(keep_hist):
                        opt_data[p][ws][r]['hist']=l[2]
    return qc_data,opt_data


def brute_force_maxcut(A,precomp=None):
    """
    Brute force max-cut for an array by solving the precompute's maximum (must faster than other brute force if we already have precomp)

    Parameters:
        A (np.ndarray): max-cut adj matrix
        precomp (np.ndarray): If available, cost matrix diagonal elements. Defaults to None

    Returns:
        tuple:
            np.ndarray: optimal solution
            float: optimal cost
    """
    
    if(precomp is None):
        precomp  = pre_compute(A)
    b_list = np.argwhere(abs(precomp - np.amax(precomp))<1e-10)
    b_list = np.reshape(b_list,(len(b_list),))
    b_list = [bin(b)[2:] for b in  b_list]
    return np.array([2*np.array([int(i) for i in '0'*(len(A)-len(b))+b])-1 for b in b_list]),np.max(precomp)

def opt_sampling_prob(v,precomp,params,mixer_ops=None,init=None):
    """
    Find the optimal sampling probability 

    Parameters:
        v (np.ndarray): bit-string for optimal value
        precomp (np.ndarray): Diagonal of cost Hamiltonian
        params (np.ndarray): Circuit parameters
        mixer_ops (list[np.ndarray]): mixer operators
        init (np.ndarray): initial statevector

    Returns:
        float: Optimal sampling probability 
    """
    psi = QAOA_eval(precomp,params,mixer_ops,init)
    return np.sum(abs(psi[[np.sum([2**i * v for i,v in enumerate(l[::-1])]) for l in ((v+1)//2)]])**2)

###Depth 0 Test
def depth0_ws_comp(n,A_func,ws_list = ['BM2','BM3','GW2','GW3'],rotation_options = None,count=1000):
    """
    Compare warmstarts at depth 0 (much faster since now we don't have to optimize)

    Parameters:
        n (int): Number of qubits
        A_func (function): Method to generate adj matricies
        ws_list (list): List of warmstart options
        rotation_options (list): List of rotation options to check. Defualts to None, in which case we check all 
        count (int): Number of samples

    Returns:
        tuple:
            comparison_data (dict): Comparing the relative performance of the warmstarts
            best_angle_data (dict): Contains the best angles for cost and for probability for each warmstart 
            ws_data (list): List of all warmstart parameters
            A_list (list): List of adjacency matricies generated
    """
    if(rotation_options is None):
        rotation_options = list(range(n))+[None]
    best_angle_data = {ws: {'max_cost':[],'max_probs':[]}  for ws in ws_list}
    comparison_data = {ws: {-1:{'cost':[],'probs':[]},0:{'cost':[],'probs':[]},None:{'cost':[],'probs':[]}}  for ws in ws_list} | {None:{'cost':[], 'probs':[]}}
    ws_data = []
    A_list = []
    for _ in tqdm(range(count)):

        A = A_func()
        initial_data = initialize(A)
        precomp = initial_data[0]
        BM2_theta_list = initial_data[1]
        BM3_theta_list = initial_data[2]
        GW2_theta_list = initial_data[3]
        GW3_theta_list = initial_data[4]
        v = initial_data[5]
        M = initial_data[6]
        ws_data.append(initial_data)
        A_list.append(A)
        qc_data = {'BM2':{r: Q2_data(BM2_theta_list,r) for r in rotation_options},
        'BM3':{r:Q3_data(BM3_theta_list,r) for r in rotation_options},
        'GW2':{r:Q2_data(GW2_theta_list,r) for r in rotation_options},
        'GW3':{r:Q3_data(GW3_theta_list,r) for r in rotation_options}}
    
        opt_data = {ws:{'cost':[],'probs':[]} for ws in ws_list}
        
        for ws in ws_list:
            for r in rotation_options:
                opt_data[ws]['cost'].append(expval(precomp,QAOA_eval(precomp,[],mixer_ops=qc_data[ws][r][1],init=qc_data[ws][r][0])))
                opt_data[ws]['probs'].append(opt_sampling_prob(v,precomp,[],mixer_ops=qc_data[ws][r][1],init=qc_data[ws][r][0]))
            l = np.argwhere(opt_data[ws]['cost']==np.max(opt_data[ws]['cost']))
            l = np.reshape(l,(len(l),))
            best_angle_data[ws]['max_cost']+=[l.tolist()]
            l = np.argwhere(opt_data[ws]['probs']==np.max(opt_data[ws]['probs']))
            l = np.reshape(l,(len(l),))
            best_angle_data[ws]['max_probs']+=[l.tolist()]

            comparison_data[ws][-1]['cost'].append(opt_data[ws]['cost'][-2])
            comparison_data[ws][-1]['probs'].append(opt_data[ws]['probs'][-2])
            comparison_data[ws][None]['cost'].append(opt_data[ws]['cost'][-1])
            comparison_data[ws][None]['probs'].append(opt_data[ws]['probs'][-1])
            comparison_data[ws][0]['cost'].append(opt_data[ws]['cost'][0])
            comparison_data[ws][0]['probs'].append(opt_data[ws]['probs'][0])
            
            comparison_data[None]['cost'].append(expval(precomp,QAOA_eval(precomp,[])))
            comparison_data[None]['probs'].append(opt_sampling_prob(v,precomp,[]))
    return comparison_data,best_angle_data,ws_data,A_list



'''''''''''''''''''''''''''''''''FINALIZED PLOTS'''''''''''''''''''''''''''''''''''''''''
def get_depth_cost_comp(prob, idx_dict, PSC_DATA, DATA, M_list, A_list, path=None, p_max=5, std=.25):
    fig, axs = plt.subplots(1, 2, figsize=(20, 10), sharey=True)

    
    plt.gca().set_yscale("log")
    def plot_graph(ws_list, ax, title_suffix):
        ax.tick_params(axis='y', which='both', length=5, width=1)
        for ws in ws_list:
            if ws is None:
                data = []
                for idx in range(*idx_dict[prob]):
                    M = M_list[idx]
                    A = A_list[idx]
                    m = -brute_force_maxcut(-A)[-1] 
                    data.append([(np.max(l) -m) / (M - m) 
                                 for l in [DATA[idx][p][ws]['cost'] for p in range(0, 1 + p_max)]])
                mean_data = np.mean(data, axis=0)
                std_dev_data = np.std(data, axis=0)
                ax.plot(range(0, p_max + 1), (mean_data), marker='o', linestyle='--', label=str(ws) + " ")
                ax.fill_between(range(0, p_max + 1), 
                                (mean_data - std * std_dev_data), 
                                (mean_data + std * std_dev_data), 
                                alpha=0.2)
            else:
                for r in [0, -1, None]:
                    data = []
                    for idx in range(*idx_dict[prob]):
                        M = M_list[idx]
                        A = A_list[idx]
                        m = -brute_force_maxcut(-A)[-1]
                        data.append([(np.max(l) -m-np.sum(-A[:-1, :-1]) / 4) / (M - m) 
                                     for l in [DATA[idx][p][ws][r]['cost'] for p in range(0, 1 + p_max)]])
                    mean_data = np.mean(data, axis=0)
                    std_dev_data = np.std(data, axis=0)
                    ax.plot(range(0, p_max + 1), (mean_data), marker='o', label=str(ws) + " " + str(r))
                    ax.fill_between(range(0, p_max + 1), 
                                    (mean_data - std * std_dev_data), 
                                    (mean_data + std * std_dev_data), 
                                    alpha=0.2)
        
        data = []
        for idx in range(*idx_dict[prob]):
            M = M_list[idx]
            A = A_list[idx]
            m 
            data.append([abs(np.max(l)) / (M + np.sum(-A[:-1, :-1]) / 4) 
                         for l in [PSC_DATA[idx][p]['cost'] for p in range(0, 1 + p_max)]])
        mean_data = np.mean(data, axis=0)
        std_dev_data = np.std(data, axis=0)
        ax.plot(range(0, p_max + 1), (mean_data), marker='o', label='PSC')
        ax.fill_between(range(0, p_max + 1), 
                        (mean_data - std * std_dev_data), 
                        (mean_data + std * std_dev_data), 
                        alpha=0.2)
        
        ax.set_title(title_suffix, fontsize = 15)
        ax.set_xlabel('p', fontsize = 12)
        ax.set_ylabel('Approximation Ratio', fontsize = 12)
        ax.legend()

    plot_graph([None, 'GW2'], axs[0], "GW2")
    
    plot_graph([None, 'GW3'], axs[1], "GW3")

    if path is not None:
        plt.savefig(path + "_subplots.pdf", dpi=300)


def get_depth_prob_comp(prob, idx_dict, PSC_DATA, DATA, M_list, A_list, path=None, p_max=5, std=.25):
    fig, axs = plt.subplots(1, 2, figsize=(20, 10), sharey=True)

    plt.gca().set_yscale("log")
    def plot_graph(ws_list, ax, title_suffix):
        ax.tick_params(axis='y', which='both', length=5, width=1)
        ax.minorticks_off()
        for ws in ws_list:
            if ws is None:
                data = []
                for idx in range(*idx_dict[prob]):
                    M = M_list[idx]
                    A = A_list[idx]
                    data.append([abs(np.max(l))
                                 for l in [DATA[idx][p][ws]['probs'] for p in range(0, 1 + p_max)]])
                mean_data = np.mean(data, axis=0)
                std_dev_data = np.std(data, axis=0)
                ax.plot(range(0, p_max + 1), (mean_data), marker='o', linestyle='--', label=str(ws) + " ")
                ax.fill_between(range(0, p_max + 1), 
                                (mean_data - std * std_dev_data), 
                                (mean_data + std * std_dev_data), 
                                alpha=0.2)
            else:
                for r in [0, -1, None]:
                    data = []
                    for idx in range(*idx_dict[prob]):
                        M = M_list[idx]
                        A = A_list[idx]
                        data.append([abs(np.max(l))
                                     for l in [DATA[idx][p][ws][r]['probs'] for p in range(0, 1 + p_max)]])
                    mean_data = np.mean(data, axis=0)
                    std_dev_data = np.std(data, axis=0)
                    ax.plot(range(0, p_max + 1), (mean_data), marker='o', label=str(ws) + " " + str(r))
                    ax.fill_between(range(0, p_max + 1), 
                                    (mean_data - std * std_dev_data), 
                                    (mean_data + std * std_dev_data), 
                                    alpha=0.2)
        
        data = []
        for idx in range(*idx_dict[prob]):
            M = M_list[idx]
            A = A_list[idx]
            data.append([abs(np.max(l))  
                         for l in [PSC_DATA[idx][p]['probs'] for p in range(0, 1 + p_max)]])
        mean_data = np.mean(data, axis=0)
        std_dev_data = np.std(data, axis=0)
        ax.plot(range(0, p_max + 1), (mean_data), marker='o', label='PSC')
        ax.fill_between(range(0, p_max + 1), 
                        (mean_data - std * std_dev_data), 
                        (mean_data + std * std_dev_data), 
                        alpha=0.2)
        
        ax.set_title(title_suffix, fontsize=15)
        ax.set_xlabel('p', fontsize=12)
        ax.set_ylabel('Probability', fontsize=12)
        ax.legend()

    plot_graph([None, 'GW2'], axs[0], "GW2")
    
    plot_graph([None, 'GW3'], axs[1], "GW3")

    if path is not None:
        plt.savefig(path + "_subplots.pdf", dpi = 300)
    plt.show()

def max_cost_hist(prob,p_data,path=None):
    ws_list = ['BM2','BM3','GW2','GW3']
    comparison_data,best_angle_data,ws_data = p_data[prob][:-1]
    n=p_data[prob][3][0].shape[0]
    fig,ax = plt.subplots(len(ws_list),1,figsize=(8,8), constrained_layout=True)
    plt.suptitle(prob)
    for i,ws in enumerate(ws_list):
        ax[i].hist(reduce(lambda x,y:x+y,best_angle_data[ws]['max_cost']),bins=range(0,n+2))
        ax[i].set_title(ws+' max cost data')
        ax[i].set_ylabel('Count')
        ax[i].set_xlabel('Vertex Rotation')
        ax[i].get_xaxis().set_ticks(np.array(range(n+1))+.5)
        ax[i].set_xticklabels([str(i) for i in range(n)]+["None"])
    if(path is not None):
            plt.savefig(path+".pdf",dpi=300)
    plt.show()


def max_prob_hist(prob,p_data,path=None):
    ws_list = ['BM2','BM3','GW2','GW3']
    comparison_data,best_angle_data,ws_data = p_data[prob][:-1]
    n=p_data[prob][3][0].shape[0]
    fig,ax = plt.subplots(len(ws_list),1,figsize=(8,8), constrained_layout=True)
    plt.suptitle(prob)
    for i,ws in enumerate(ws_list):
        ax[i].hist(reduce(lambda x,y:x+y,best_angle_data[ws]['max_probs']),bins=range(0,n+2))
        ax[i].set_title(ws+' max prob data')
        ax[i].set_ylabel('Count')
        ax[i].set_xlabel('Vertex Rotation')
        ax[i].get_xaxis().set_ticks(np.array(range(n+1))+.5)
        ax[i].set_xticklabels([str(i) for i in range(n)]+["None"])
    if(path is not None):
                plt.savefig(path+".pdf",dpi=300)
    plt.show()

def cost_scatter(prob, p_data,path=None):
    ws_list=['BM2', 'BM3', 'GW2', 'GW3']
    comparison_data, best_angle_data, ws_data = p_data[prob][:-1]
    x = np.array([d[6] for d in ws_data])
    fig = plt.figure(figsize=(12, 20))
    gs = GridSpec(3, 8, wspace=2, hspace=0.2)
    order = [[0, slice(0, 4)], [0, slice(4, 8)], [1, slice(0, 4)], [1, slice(4, 8)]]
    
    plt.suptitle(prob + ' Cost Scatter', y=0.92)
    for i, w in enumerate(ws_list):
        ax = fig.add_subplot(gs[order[i][0], order[i][1]])
        ax.scatter(x, comparison_data[w][0]['cost'], label=w + ' 0', alpha=0.25)
        ax.scatter(x, comparison_data[w][None]['cost'], label=w, alpha=0.25)
        ax.scatter(x, comparison_data[w][-1]['cost'], label=w + ' -1', alpha=0.25)
        ax.plot(x, x, ':', color='k', label='ideal')
        ax.legend()
        ax.set_title(w + ' Cost Comparison')
        ax.set_xlabel('Optimal Cost')
        ax.set_ylabel('Obtained Cost')
    
    ax = fig.add_subplot(gs[2, slice(2, 6)])
    for ws in ws_list:
        ax.scatter(x, comparison_data[ws][-1]['cost'], label=ws, alpha=0.25)
    ax.plot(x, x, ':', color='k', label='ideal')
    ax.legend()
    ax.set_title('-1 Rotation Comparison')
    ax.set_xlabel('Optimal Cost')
    ax.set_ylabel('Obtained Cost')
    if(path is not None):
            plt.savefig(path+".pdf",dpi=300)
    plt.show()


def prob_boxplot(prob, p_data,path=None):
    ws_list=['BM2', 'BM3', 'GW2', 'GW3']
    comparison_data, best_angle_data, ws_data = p_data[prob][:-1]
    x = np.array([d[6] for d in ws_data])
    fig = plt.figure(figsize=(12, 20))
    gs = GridSpec(3, 8, wspace=2, hspace=0.2)
    order = [[0, slice(0, 4)], [0, slice(4, 8)], [1, slice(0, 4)], [1, slice(4, 8)]]
    
    plt.suptitle(prob + ' Prob Boxplot', y=0.92)
    for i, w in enumerate(ws_list):
        ax = fig.add_subplot(gs[order[i][0], order[i][1]])
        ax.boxplot([comparison_data[w][0]['probs'],comparison_data[w][None]['probs'],comparison_data[w][-1]['probs']],0,'',label=[w+' 0',w,w+' -1'])

        ax.legend()
        ax.set_title(w + ' Probability Comparison')
        ax.set_ylabel('Probability')
    
    ax = fig.add_subplot(gs[2, slice(2, 6)])
    ax.boxplot([comparison_data[w][-1]['probs'] for w in ws_list],0,'',label=ws_list)
    ax.legend()
    ax.set_title('-1 Probability Comparison')
    ax.set_ylabel('Probability')
    if(path is not None):
            plt.savefig(path+".pdf",dpi=300)
    plt.show()

def final_boxplot(prob, p_data,path=None):
    ws_list=['BM2', 'BM3', 'GW2', 'GW3']
    comparison_data, best_angle_data, ws_data = p_data[prob][:-1]
    A_list = p_data[prob][-1]
    m = np.array([-brute_force_maxcut(-A)[-1] for A in A_list])
    M = np.array([d[-1] for d in ws_data])
    plt.figure(figsize=(10,6))
    plt.suptitle(prob + " -1 Warmstart Comparison")
    plt.boxplot([(np.array(comparison_data[ws][-1]['cost'])-m)/(M-m) for ws in ws_list],0,'',label=ws_list)
    plt.ylabel('Instance Specific Relative Error')
    plt.legend()
    if(path is not None):
            plt.savefig(path+".pdf",dpi=300)
    plt.show()

def get_depth_combined(prob, DATA, PSC_DATA50, M_list, A_list, idx_list, p_max = 5, std=0.25, problemLength=10, axs=None, path=None):
    # Sizes
    title_size = 45
    title_y_spacing = 1.05
    title_x_spacing = 0.51
    y_axis_title_size = 50
    x_axis_title_size = 25
    x_ticker_size = 25
    y_ticker_size = 23
    axs_title_size = 30
    

    # Initializing Figures
    if axs is None:
        fig, axs = plt.subplots(2, 4, figsize=(40, 7), sharey='row')
        fig.suptitle(prob, fontsize=title_size, y=title_y_spacing, x=title_x_spacing)
    
    axs[0, 0].set_ylabel(r'$\mathcal{P}$', fontsize=y_axis_title_size, labelpad=10)
    axs[1, 0].set_ylabel(r'α', fontsize=y_axis_title_size, labelpad=10)
    
    for ax in axs[1, :]:
        ax.set_xlabel('p', fontsize=x_axis_title_size)

    for row in range(2):
        for col in range(4):
            axs[row, col].tick_params(axis='x', labelsize=x_ticker_size)
            axs[row, col].tick_params(axis='y', labelsize=y_ticker_size)
            axs[row, col].grid(True, which='major', axis='y', linestyle='--', alpha=0.5)

    for ax in axs[0, :]:
        ax.set_xticklabels([])
        ax.set_xticks([])

    axs[0, 0].set_title("GW2", fontsize=axs_title_size)
    axs[0, 1].set_title("GW3", fontsize=axs_title_size)
    axs[0, 2].set_title("BM2", fontsize=axs_title_size)
    axs[0, 3].set_title("BM3", fontsize=axs_title_size)

    plt.subplots_adjust(wspace=0, hspace=0.15)

    # Gathering Probabilty Data
    def get_prob_data(data, idx_list, prob_name, ws=None, p_max=5, rotation=None):
        mean_probs = []
        std_probs = []
    
        for p in range(0, 1 + p_max):
            vals_per_idx = []
            for idx in range(*idx_list[prob_name]):
                prob = data[idx][p][ws][rotation]['probs'] if ws is not None else data[idx][p][None]['probs']
                vals_per_idx.append(np.max(prob)) 
            mean_probs.append(np.mean(vals_per_idx))
            std_probs.append(np.std(vals_per_idx))
    
        return mean_probs, std_probs
    
    def compute_instance_min(A_list, idx_list, prob_name):
        m_values = []
        for idx in range(*idx_list[prob_name]):
            A = A_list[idx]
            m_values.append(-brute_force_maxcut(-A)[-1])
        return m_values

    def get_cost_data(data, idx_list, prob_name, M_values, m_values, ws=None, p_max=5, rotation=None):
        all_instances = []

        for i, idx in enumerate(range(*idx_list[prob_name])):
            m = m_values[i]
            M = M_values[i]
            alpha_vals = []   
            for p in range(0, p_max + 1):

                if ws is not None:
                    cost_ws = np.max(data[idx][p][ws][rotation]['cost'])
                else:
                    cost_ws = np.max(data[idx][p][None]['cost'])
    
                alpha = (cost_ws - m) / (M - m)
                alpha_vals.append(alpha)

            all_instances.append(alpha_vals)
    
        mean_cost = np.mean(all_instances, axis=0)
        std_cost = np.std(all_instances, axis=0)
        
        return mean_cost, std_cost
        
    def get_psc_prob(DATA_psc, idx_list, prob):
        all_instances = []
    
        for idx in range(*idx_list[prob]):
            prob_vals = []
            for p in range(0, p_max + 1):
                prob_p = np.max(DATA_psc[idx][p]['probs'])
                prob_vals.append(prob_p)
            all_instances.append(prob_vals)
    
        mean_probs = np.mean(all_instances, axis=0)
        std_probs = np.std(all_instances, axis=0)
    
        return mean_probs, std_probs, "PSC"
    
    def get_psc_cost(DATA_psc, idx_list, prob, M_values, m_values):
        all_instances = []
    
        for i, idx in enumerate(range(*idx_list[prob])):
            m = m_values[i]
            M = M_values[i]
            alpha_vals = []
            A = A_list[idx] 
            for p in range(0, p_max + 1):
                cost_psc = np.max(DATA_psc[idx][p]['cost'] - np.sum(-A[:-1, :-1]) / 4)
                alpha_vals.append((cost_psc - m) / (M - m))
            all_instances.append(alpha_vals)
    
        mean_costs = np.mean(all_instances, axis=0)
        std_costs = np.std(all_instances, axis=0)
    
        return mean_costs, std_costs, "PSC"
    
    def get_data(DATA, idx_list, prob, M_values, m_values, ws=None, version=None):
        if version == "Probability":
            warm_no_rot_mean, warm_no_rot_std = get_prob_data(DATA, idx_list, prob, ws=ws, rotation=None)
            warm_first_rot_mean, warm_first_rot_std = get_prob_data(DATA, idx_list, prob, ws=ws, rotation=0)
            warm_last_rot_mean, warm_last_rot_std = get_prob_data(DATA, idx_list, prob, ws=ws, rotation=-1)
            return (warm_first_rot_mean, warm_last_rot_std, "Warmstart, first rotation"), (warm_last_rot_mean, warm_last_rot_std, "Warmstart, last rotation"), (warm_no_rot_mean, warm_no_rot_std, "Warmstart, no rotation")
        elif version == "Cost":
            warm_no_rot_mean, warm_no_rot_std = get_cost_data(DATA, idx_list, prob, M_values, m_values, ws=ws, rotation=None)
            warm_first_rot_mean, warm_first_rot_std = get_cost_data(DATA, idx_list, prob, M_values, m_values, ws=ws, rotation=0)
            warm_last_rot_mean, warm_last_rot_std = get_cost_data(DATA, idx_list, prob, M_values, m_values, ws=ws, rotation=-1)
            return (warm_first_rot_mean, warm_first_rot_std, "Warmstart, first rotation"), (warm_last_rot_mean, warm_last_rot_std, "Warmstart, last rotation"), (warm_no_rot_mean, warm_no_rot_std, "Warmstart, no rotation")
        return None
         
    def get_warmstart_data(ws, version, M_values, m_values):
        return get_data(DATA, idx_list, prob, M_values, m_values, ws=ws, version=version)

    idx_range = range(*idx_list[prob])
    M_values = [M_list[idx] for idx in idx_range]
    m_values = compute_instance_min(A_list, idx_list, prob)
    
    none_mean_prob, none_std_prob = get_prob_data(DATA, idx_list, prob, ws=None, rotation=0)
    none_mean_cost, none_std_cost = get_cost_data(DATA, idx_list, prob, M_values, m_values, ws=None, rotation=0)
    
    # Grabbing All Data
    no_prob = (none_mean_prob, none_std_prob, "No Warmstart")
    no_cost = (none_mean_cost, none_std_cost, "No Warmstart")
    psc_prob = get_psc_prob(PSC_DATA50, idx_list, prob)
    psc_cost = get_psc_cost(PSC_DATA50, idx_list, prob, M_values, m_values)
    GW2ProbData = get_warmstart_data("GW2", "Probability", M_values, m_values)
    GW3ProbData = get_warmstart_data("GW3", "Probability", M_values, m_values)
    BM2ProbData = get_warmstart_data("BM2", "Probability", M_values, m_values)
    BM3ProbData = get_warmstart_data("BM3", "Probability", M_values, m_values)
    GW2CostData = get_warmstart_data("GW2", "Cost", M_values, m_values)
    GW3CostData = get_warmstart_data("GW3", "Cost", M_values, m_values)
    BM2CostData = get_warmstart_data("BM2", "Cost", M_values, m_values)
    BM3CostData = get_warmstart_data("BM3", "Cost", M_values, m_values)

    def add(ws, data, array):
        array[ws] = []
        for rot in data:
            array[ws].append(rot)
    
    def combine(GW2Data, GW3Data, BM2Data, BM3Data): 
        array = {}
        add("GW2", GW2Data, array)
        add("GW3", GW3Data, array)
        add("BM2", BM2Data, array)
        add("BM3", BM3Data, array)
        return array

    probabilites = combine(GW2ProbData, GW3ProbData, BM2ProbData, BM3ProbData)
    cost = combine(GW2CostData, GW3CostData, BM2CostData, BM3CostData)
        
    p = range(0, p_max + 1)

    # Plotting
    def plot_line(ax, data, dot=False, square=False):
        lineStyle = "-"
        marker = "o"
        if dot:
            lineStyle = "--"
        if square:
            marker = "s"
        mean_data, std_data, label = data
        ax.plot(p, mean_data, marker=marker, linestyle=lineStyle, label=label)
        ax.fill_between(p,
                        np.array(mean_data) - std * np.array(std_data),
                        np.array(mean_data) + std * np.array(std_data),
                        alpha=0.2)
            

    def plot(ax, no, psc, data):
        plot_line(ax, no, dot=True)
        for d in data:
            plot_line(ax, d)
        plot_line(ax, psc, square=True)


    def plot_all(no_prob, no_cost, psc_prob, psc_cost, probabilities, cost):
        plot(axs[0, 0], no_prob, psc_prob, probabilites["GW2"])
        plot(axs[0, 1], no_prob, psc_prob, probabilites["GW3"])
        plot(axs[0, 2], no_prob, psc_prob, probabilites["BM2"])
        plot(axs[0, 3], no_prob, psc_prob, probabilites["BM3"])

        plot(axs[1, 0], no_cost, psc_cost, cost["GW2"])
        plot(axs[1, 1], no_cost, psc_cost, cost["GW3"])
        plot(axs[1, 2], no_cost, psc_cost, cost["BM2"])
        plot(axs[1, 3], no_cost, psc_cost, cost["BM3"])

        for row in range(2):
            for col in range(4):
                ax = axs[row, col]
                ax.yaxis.set_major_locator(MaxNLocator(nbins=5, prune='both'))
                ax.yaxis.set_minor_locator(MaxNLocator(nbins=25, prune='both'))
                ax.minorticks_on()
        
                ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))

        if path is not None:
            plt.savefig(path + ".pdf", dpi=600, bbox_inches='tight')

    plot_all(no_prob, no_cost, psc_prob, psc_cost, probabilites, cost)

def get_depth_combined_gw2(prob, DATA, PSC_DATA50, M_list, A_list, idx_list, p_max = 5, std=0.25, problemLength=10, axs=None, path=None):
    # Sizes
    title_size = 60
    title_y_spacing = 0.95
    title_x_spacing = 0.48
    y_axis_title_size = 50
    x_axis_title_size = 35
    x_ticker_size = 35
    y_ticker_size = 30
    axs_title_size = 25
    

    # Initializing Figures
    if axs is None:
        fig, axs = plt.subplots(2, 1, figsize=(20, 15), sharey='row')
        fig.suptitle(prob, fontsize=title_size, y=title_y_spacing, x=title_x_spacing)
    
    axs[0].set_ylabel(r'$\mathcal{P}$', fontsize=y_axis_title_size, labelpad=10)
    axs[1].set_ylabel(r'α', fontsize=y_axis_title_size, labelpad=10)
    axs[1].set_xlabel('p', fontsize=x_axis_title_size)

    for ax in axs:
        ax.tick_params(axis='x', labelsize=x_ticker_size)
        ax.tick_params(axis='y', labelsize=y_ticker_size)
        ax.grid(True, which='major', axis='y', linestyle='--', alpha=0.5)

    axs[0].set_xticklabels([])
    axs[0].set_xticks([])

    plt.subplots_adjust(wspace=0, hspace=0.15)

    # Gathering Probabilty Data
    def get_prob_data(data, idx_list, prob_name, ws=None, p_max=5, rotation=None):
        mean_probs = []
        std_probs = []
    
        for p in range(0, 1 + p_max):
            vals_per_idx = []
            for idx in range(*idx_list[prob_name]):
                prob = data[idx][p][ws][rotation]['probs'] if ws is not None else data[idx][p][None]['probs']
                vals_per_idx.append(np.max(prob)) 
            mean_probs.append(np.mean(vals_per_idx))
            std_probs.append(np.std(vals_per_idx))
    
        return mean_probs, std_probs
    
    def compute_instance_min(A_list, idx_list, prob_name):
        m_values = []
        for idx in range(*idx_list[prob_name]):
            A = A_list[idx]
            m_values.append(-brute_force_maxcut(-A)[-1])
        return m_values

    def get_cost_data(data, idx_list, prob_name, M_values, m_values, ws=None, p_max=5, rotation=None):
        all_instances = []

        for i, idx in enumerate(range(*idx_list[prob_name])):
            m = m_values[i]
            M = M_values[i]
            alpha_vals = []   
            for p in range(0, p_max + 1):

                if ws is not None:
                    cost_ws = np.max(data[idx][p][ws][rotation]['cost'])
                else:
                    cost_ws = np.max(data[idx][p][None]['cost'])
    
                alpha = (cost_ws - m) / (M - m)
                alpha_vals.append(alpha)

            all_instances.append(alpha_vals)
    
        mean_cost = np.mean(all_instances, axis=0)
        std_cost = np.std(all_instances, axis=0)
        
        return mean_cost, std_cost
        
    def get_psc_prob(DATA_psc, idx_list, prob):
        all_instances = []
    
        for idx in range(*idx_list[prob]):
            prob_vals = []
            for p in range(0, p_max + 1):
                prob_p = np.max(DATA_psc[idx][p]['probs'])
                prob_vals.append(prob_p)
            all_instances.append(prob_vals)
    
        mean_probs = np.mean(all_instances, axis=0)
        std_probs = np.std(all_instances, axis=0)
    
        return mean_probs, std_probs, "PSC"
    
    def get_psc_cost(DATA_psc, idx_list, prob, M_values, m_values):
        all_instances = []
    
        for i, idx in enumerate(range(*idx_list[prob])):
            m = m_values[i]
            M = M_values[i]
            alpha_vals = []
            A = A_list[idx] 
            for p in range(0, p_max + 1):
                cost_psc = np.max(DATA_psc[idx][p]['cost'] - np.sum(-A[:-1, :-1]) / 4)
                alpha_vals.append((cost_psc - m) / (M - m))
            all_instances.append(alpha_vals)
    
        mean_costs = np.mean(all_instances, axis=0)
        std_costs = np.std(all_instances, axis=0)
    
        return mean_costs, std_costs, "PSC"
    
    def get_data(DATA, idx_list, prob, M_values, m_values, ws=None, version=None):
        if version == "Probability":
            warm_no_rot_mean, warm_no_rot_std = get_prob_data(DATA, idx_list, prob, ws=ws, rotation=None)
            warm_first_rot_mean, warm_first_rot_std = get_prob_data(DATA, idx_list, prob, ws=ws, rotation=0)
            warm_last_rot_mean, warm_last_rot_std = get_prob_data(DATA, idx_list, prob, ws=ws, rotation=-1)
            return (warm_first_rot_mean, warm_last_rot_std, "Warmstart, first rotation"), (warm_last_rot_mean, warm_last_rot_std, "Warmstart, last rotation"), (warm_no_rot_mean, warm_no_rot_std, "Warmstart, no rotation")
        elif version == "Cost":
            warm_no_rot_mean, warm_no_rot_std = get_cost_data(DATA, idx_list, prob, M_values, m_values, ws=ws, rotation=None)
            warm_first_rot_mean, warm_first_rot_std = get_cost_data(DATA, idx_list, prob, M_values, m_values, ws=ws, rotation=0)
            warm_last_rot_mean, warm_last_rot_std = get_cost_data(DATA, idx_list, prob, M_values, m_values, ws=ws, rotation=-1)
            return (warm_first_rot_mean, warm_first_rot_std, "Warmstart, first rotation"), (warm_last_rot_mean, warm_last_rot_std, "Warmstart, last rotation"), (warm_no_rot_mean, warm_no_rot_std, "Warmstart, no rotation")
        return None
         
    def get_warmstart_data(ws, version, M_values, m_values):
        return get_data(DATA, idx_list, prob, M_values, m_values, ws=ws, version=version)

    idx_range = range(*idx_list[prob])
    M_values = [M_list[idx] for idx in idx_range]
    m_values = compute_instance_min(A_list, idx_list, prob)
    
    none_mean_prob, none_std_prob = get_prob_data(DATA, idx_list, prob, ws=None, rotation=0)
    none_mean_cost, none_std_cost = get_cost_data(DATA, idx_list, prob, M_values, m_values, ws=None, rotation=0)
    
    # Grabbing All Data
    no_prob = (none_mean_prob, none_std_prob, "No Warmstart")
    no_cost = (none_mean_cost, none_std_cost, "No Warmstart")
    psc_prob = get_psc_prob(PSC_DATA50, idx_list, prob)
    psc_cost = get_psc_cost(PSC_DATA50, idx_list, prob, M_values, m_values)
    GW2ProbData = get_warmstart_data("GW2", "Probability", M_values, m_values)
    GW2CostData = get_warmstart_data("GW2", "Cost", M_values, m_values)

    def add(ws, data, array):
        array[ws] = []
        for rot in data:
            array[ws].append(rot)
    
    def combine(GW2Data): 
        array = {}
        add("GW2", GW2Data, array)
        return array

    probabilites = combine(GW2ProbData)
    cost = combine(GW2CostData)
        
    p = range(0, p_max + 1)

    # Plotting
    def plot_line(ax, data, dot=False, square=False):
        lineStyle = "-"
        marker = "o"
        if dot:
            lineStyle = "--"
        if square:
            marker = "s"
        mean_data, std_data, label = data
        ax.plot(p, mean_data, marker=marker, linestyle=lineStyle, label=label)
        ax.fill_between(p,
                        np.array(mean_data) - std * np.array(std_data),
                        np.array(mean_data) + std * np.array(std_data),
                        alpha=0.2)
            

    def plot(ax, no, psc, data):
        plot_line(ax, no, dot=True)
        for d in data:
            plot_line(ax, d)
        plot_line(ax, psc, square=True)


    def plot_all(no_prob, no_cost, psc_prob, psc_cost, probabilities, cost):
        plot(axs[0], no_prob, psc_prob, probabilites["GW2"])
        plot(axs[1], no_cost, psc_cost, cost["GW2"])

        for row in range(2):
                ax = axs[row]
                ax.yaxis.set_major_locator(MaxNLocator(nbins=5, prune='both'))
                ax.yaxis.set_minor_locator(MaxNLocator(nbins=25, prune='both'))
                ax.minorticks_on()
        
                ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))

        if path is not None:
            plt.savefig(path + ".pdf", dpi=600, bbox_inches='tight')

    plot_all(no_prob, no_cost, psc_prob, psc_cost, probabilites, cost)

def get_depth_combined_gw3(prob, DATA, PSC_DATA50, M_list, A_list, idx_list, p_max = 5, std=0.25, problemLength=10, axs=None, path=None):
    # Sizes
    title_size = 60
    title_y_spacing = 0.95
    title_x_spacing = 0.48
    y_axis_title_size = 50
    x_axis_title_size = 35
    x_ticker_size = 35
    y_ticker_size = 30
    axs_title_size = 25
    

    # Initializing Figures
    if axs is None:
        fig, axs = plt.subplots(2, 1, figsize=(20, 15), sharey='row')
        fig.suptitle(prob, fontsize=title_size, y=title_y_spacing, x=title_x_spacing)
    
    axs[0].set_ylabel(r'$\mathcal{P}$', fontsize=y_axis_title_size, labelpad=10)
    axs[1].set_ylabel(r'α', fontsize=y_axis_title_size, labelpad=10)
    axs[1].set_xlabel('p', fontsize=x_axis_title_size)

    for ax in axs:
        ax.tick_params(axis='x', labelsize=x_ticker_size)
        ax.tick_params(axis='y', labelsize=y_ticker_size)
        ax.grid(True, which='major', axis='y', linestyle='--', alpha=0.5)

    axs[0].set_xticklabels([])
    axs[0].set_xticks([])

    plt.subplots_adjust(wspace=0, hspace=0.15)

    # Gathering Probabilty Data
    def get_prob_data(data, idx_list, prob_name, ws=None, p_max=5, rotation=None):
        mean_probs = []
        std_probs = []
    
        for p in range(0, 1 + p_max):
            vals_per_idx = []
            for idx in range(*idx_list[prob_name]):
                prob = data[idx][p][ws][rotation]['probs'] if ws is not None else data[idx][p][None]['probs']
                vals_per_idx.append(np.max(prob)) 
            mean_probs.append(np.mean(vals_per_idx))
            std_probs.append(np.std(vals_per_idx))
    
        return mean_probs, std_probs
    
    def compute_instance_min(A_list, idx_list, prob_name):
        m_values = []
        for idx in range(*idx_list[prob_name]):
            A = A_list[idx]
            m_values.append(-brute_force_maxcut(-A)[-1])
        return m_values

    def get_cost_data(data, idx_list, prob_name, M_values, m_values, ws=None, p_max=5, rotation=None):
        all_instances = []

        for i, idx in enumerate(range(*idx_list[prob_name])):
            m = m_values[i]
            M = M_values[i]
            alpha_vals = []   
            for p in range(0, p_max + 1):

                if ws is not None:
                    cost_ws = np.max(data[idx][p][ws][rotation]['cost'])
                else:
                    cost_ws = np.max(data[idx][p][None]['cost'])
    
                alpha = (cost_ws - m) / (M - m)
                alpha_vals.append(alpha)

            all_instances.append(alpha_vals)
    
        mean_cost = np.mean(all_instances, axis=0)
        std_cost = np.std(all_instances, axis=0)
        
        return mean_cost, std_cost
        
    def get_psc_prob(DATA_psc, idx_list, prob):
        all_instances = []
    
        for idx in range(*idx_list[prob]):
            prob_vals = []
            for p in range(0, p_max + 1):
                prob_p = np.max(DATA_psc[idx][p]['probs'])
                prob_vals.append(prob_p)
            all_instances.append(prob_vals)
    
        mean_probs = np.mean(all_instances, axis=0)
        std_probs = np.std(all_instances, axis=0)
    
        return mean_probs, std_probs, "PSC"
    
    def get_psc_cost(DATA_psc, idx_list, prob, M_values, m_values):
        all_instances = []
    
        for i, idx in enumerate(range(*idx_list[prob])):
            m = m_values[i]
            M = M_values[i]
            alpha_vals = []
            A = A_list[idx] 
            for p in range(0, p_max + 1):
                cost_psc = np.max(DATA_psc[idx][p]['cost'] - np.sum(-A[:-1, :-1]) / 4)
                alpha_vals.append((cost_psc - m) / (M - m))
            all_instances.append(alpha_vals)
    
        mean_costs = np.mean(all_instances, axis=0)
        std_costs = np.std(all_instances, axis=0)
    
        return mean_costs, std_costs, "PSC"
    
    def get_data(DATA, idx_list, prob, M_values, m_values, ws=None, version=None):
        if version == "Probability":
            warm_no_rot_mean, warm_no_rot_std = get_prob_data(DATA, idx_list, prob, ws=ws, rotation=None)
            warm_first_rot_mean, warm_first_rot_std = get_prob_data(DATA, idx_list, prob, ws=ws, rotation=0)
            warm_last_rot_mean, warm_last_rot_std = get_prob_data(DATA, idx_list, prob, ws=ws, rotation=-1)
            return (warm_first_rot_mean, warm_last_rot_std, "Warmstart, first rotation"), (warm_last_rot_mean, warm_last_rot_std, "Warmstart, last rotation"), (warm_no_rot_mean, warm_no_rot_std, "Warmstart, no rotation")
        elif version == "Cost":
            warm_no_rot_mean, warm_no_rot_std = get_cost_data(DATA, idx_list, prob, M_values, m_values, ws=ws, rotation=None)
            warm_first_rot_mean, warm_first_rot_std = get_cost_data(DATA, idx_list, prob, M_values, m_values, ws=ws, rotation=0)
            warm_last_rot_mean, warm_last_rot_std = get_cost_data(DATA, idx_list, prob, M_values, m_values, ws=ws, rotation=-1)
            return (warm_first_rot_mean, warm_first_rot_std, "Warmstart, first rotation"), (warm_last_rot_mean, warm_last_rot_std, "Warmstart, last rotation"), (warm_no_rot_mean, warm_no_rot_std, "Warmstart, no rotation")
        return None
         
    def get_warmstart_data(ws, version, M_values, m_values):
        return get_data(DATA, idx_list, prob, M_values, m_values, ws=ws, version=version)

    idx_range = range(*idx_list[prob])
    M_values = [M_list[idx] for idx in idx_range]
    m_values = compute_instance_min(A_list, idx_list, prob)
    
    none_mean_prob, none_std_prob = get_prob_data(DATA, idx_list, prob, ws=None, rotation=0)
    none_mean_cost, none_std_cost = get_cost_data(DATA, idx_list, prob, M_values, m_values, ws=None, rotation=0)
    
    # Grabbing All Data
    no_prob = (none_mean_prob, none_std_prob, "No Warmstart")
    no_cost = (none_mean_cost, none_std_cost, "No Warmstart")
    psc_prob = get_psc_prob(PSC_DATA50, idx_list, prob)
    psc_cost = get_psc_cost(PSC_DATA50, idx_list, prob, M_values, m_values)
    GW3ProbData = get_warmstart_data("GW3", "Probability", M_values, m_values)
    GW3CostData = get_warmstart_data("GW3", "Cost", M_values, m_values)

    def add(ws, data, array):
        array[ws] = []
        for rot in data:
            array[ws].append(rot)
    
    def combine(GW3Data): 
        array = {}
        add("GW3", GW3Data, array)
        return array

    probabilites = combine(GW3ProbData)
    cost = combine(GW3CostData)
        
    p = range(0, p_max + 1)

    # Plotting
    def plot_line(ax, data, dot=False, square=False):
        lineStyle = "-"
        marker = "o"
        if dot:
            lineStyle = "--"
        if square:
            marker = "s"
        mean_data, std_data, label = data
        ax.plot(p, mean_data, marker=marker, linestyle=lineStyle, label=label)
        ax.fill_between(p,
                        np.array(mean_data) - std * np.array(std_data),
                        np.array(mean_data) + std * np.array(std_data),
                        alpha=0.2)
            

    def plot(ax, no, psc, data):
        plot_line(ax, no, dot=True)
        for d in data:
            plot_line(ax, d)
        plot_line(ax, psc, square=True)


    def plot_all(no_prob, no_cost, psc_prob, psc_cost, probabilities, cost):
        plot(axs[0], no_prob, psc_prob, probabilites["GW3"])
        plot(axs[1], no_cost, psc_cost, cost["GW3"])

        for row in range(2):
                ax = axs[row]
                ax.yaxis.set_major_locator(MaxNLocator(nbins=5, prune='both'))
                ax.yaxis.set_minor_locator(MaxNLocator(nbins=25, prune='both'))
                ax.minorticks_on()
        
                ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))

        if path is not None:
            plt.savefig(path + ".pdf", dpi=600, bbox_inches='tight')

    plot_all(no_prob, no_cost, psc_prob, psc_cost, probabilites, cost)

def plot_distance(prob, DATA1, DATA2, prob1="10 Initializations", prob2="50 Initializations", path=None):
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle(prob, fontsize=16)
    
    # Stacked histogram
    ax.hist([DATA1, DATA2], edgecolor='black', density=True, stacked=True, label=[prob1, prob2])
    ax.set_xlabel('Distance To Solution', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12, )
    ax.legend(loc='upper left')
    ax.yaxis.set_label_position("left")
    ax.yaxis.set_label_coords(-0.05, 0.5)
    
    # Save the figure if a path is specified
    if path is not None:
        plt.savefig(path + ".pdf", dpi=300)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.999])
    plt.show()

def calculate_depth_points_gw3(prob, idx_dict, PSC_DATA, DATA, PSC_DATA50, M_list, A_list, p_max=5):
    ws_list = ['BM2','BM3','GW2','GW3', None]
    results = {
        "None": {"p": [], "alpha": []},
        "Warmstart No Rotation": {"p": [], "alpha": []},
        "Warmstart First Rotation": {"p": [], "alpha": []},
        "Warmstart Last Rotation": {"p": [], "alpha": []},
        "QUBO Relaxed (10 Initializations)": {"p": [], "alpha": []},
        "QUBO Relaxed (50 Initializations)": {"p": [], "alpha": []},
    }

    for idx in range(*idx_dict[prob]):
        M = M_list[idx]
        A = A_list[idx]
        m = -brute_force_maxcut(-A)[-1]

        # Calculate for None
        for ws in ws_list:
            if ws is None:
                none_p = [np.max(DATA[idx][p][None]['probs']) for p in range(0, 1 + p_max)]
                none_alpha = [(np.max(DATA[idx][p][None]['cost']) - m) / (M - m) for p in range(0, 1 + p_max)]
                results["None"]["p"].append(none_p)
                results["None"]["alpha"].append(none_alpha)

        # Calculate for Warmstart variants
            else:
                for r, label in zip([None, 0, -1], ["Warmstart No Rotation", "Warmstart First Rotation", "Warmstart Last Rotation"]):
                    ws_p = [np.max(DATA[idx][p]['GW3'][r]['probs']) for p in range(0, 1 + p_max)]
                    ws_alpha = [(np.max(DATA[idx][p]['GW3'][r]['cost']) - m) / (M - m) for p in range(0, 1 + p_max)]
                    results[label]["p"].append(ws_p)
                    results[label]["alpha"].append(ws_alpha)

        # Calculate for QUBO Relaxed (10 Initializations)
        qubo10_p = [np.max(PSC_DATA[idx][p]['probs']) for p in range(0, 1 + p_max)]
        qubo10_alpha = [(np.max(PSC_DATA[idx][p]['cost']) - m - np.sum(-A[:-1, :-1]) / 4) / (M - m)
                        for p in range(0, 1 + p_max)]
        results["QUBO Relaxed (10 Initializations)"]["p"].append(qubo10_p)
        results["QUBO Relaxed (10 Initializations)"]["alpha"].append(qubo10_alpha)

        # Calculate for QUBO Relaxed (50 Initializations)
        qubo50_p = [np.max(PSC_DATA50[idx][p]['probs']) for p in range(0, 1 + p_max)]
        qubo50_alpha = [(np.max(PSC_DATA50[idx][p]['cost']) - m - np.sum(-A[:-1, :-1]) / 4) / (M - m)
                        for p in range(0, 1 + p_max)]
        results["QUBO Relaxed (50 Initializations)"]["p"].append(qubo50_p)
        results["QUBO Relaxed (50 Initializations)"]["alpha"].append(qubo50_alpha)

    # Format output: Averaging results for clarity (optional)
    formatted_results = {}
    for key, value in results.items():
        formatted_results[key] = {
            "p": [round(np.mean([run[i] for run in value["p"]]), 4) for i in range(p_max + 1)],
            "alpha": [round(np.mean([run[i] for run in value["alpha"]]), 4) for i in range(p_max + 1)],
        }

    return formatted_results

def calculate_depth_points_gw2(prob, idx_dict, PSC_DATA, DATA, PSC_DATA50, M_list, A_list, p_max=5, custom_std_scale=0.25):
    ws_list = ['BM2', 'BM3', 'GW2', 'GW3', None]
    results = {
        "None": {"p": [], "a": [], "p_std": [], "a_std": []},
        "Warmstart No Rotation": {"p": [], "a": [], "p_std": [], "a_std": []},
        "Warmstart First Rotation": {"p": [], "a": [], "p_std": [], "a_std": []},
        "Warmstart Last Rotation": {"p": [], "a": [], "p_std": [], "a_std": []},
        "QUBO Relaxed (10 Initializations)": {"p": [], "a": [], "p_std": [], "a_std": []},
        "QUBO Relaxed (50 Initializations)": {"p": [], "a": [], "p_std": [], "a_std": []},
    }

    for idx in range(*idx_dict[prob]):
        M = M_list[idx]
        A = A_list[idx]
        m = -brute_force_maxcut(-A)[-1]

        # Calculate for None
        for ws in ws_list:
            if ws is None:
                none_p = [np.max(DATA[idx][p][None]['probs']) for p in range(0, 1 + p_max)]
                none_a = [(np.max(DATA[idx][p][None]['cost']) - m) / (M - m) for p in range(0, 1 + p_max)]
                results["None"]["p"].append(none_p)
                results["None"]["a"].append(none_a)

        # Calculate for Warmstart variants
            else:
                for r, label in zip([None, 0, -1], ["Warmstart No Rotation", "Warmstart First Rotation", "Warmstart Last Rotation"]):
                    ws_p = [np.max(DATA[idx][p]['GW2'][r]['probs']) for p in range(0, 1 + p_max)]
                    ws_a = [(np.max(DATA[idx][p]['GW2'][r]['cost']) - m) / (M - m) for p in range(0, 1 + p_max)]
                    results[label]["p"].append(ws_p)
                    results[label]["a"].append(ws_a)

        # Calculate for QUBO Relaxed (10 Initializations)
        qubo10_p = [np.max(PSC_DATA[idx][p]['probs']) for p in range(0, 1 + p_max)]
        qubo10_a = [(np.max(PSC_DATA[idx][p]['cost']) - m - np.sum(-A[:-1, :-1]) / 4) / (M - m)
                    for p in range(0, 1 + p_max)]
        results["QUBO Relaxed (10 Initializations)"]["p"].append(qubo10_p)
        results["QUBO Relaxed (10 Initializations)"]["a"].append(qubo10_a)

        # Calculate for QUBO Relaxed (50 Initializations)
        qubo50_p = [np.max(PSC_DATA50[idx][p]['probs']) for p in range(0, 1 + p_max)]
        qubo50_a = [(np.max(PSC_DATA50[idx][p]['cost']) - m - np.sum(-A[:-1, :-1]) / 4) / (M - m)
                    for p in range(0, 1 + p_max)]
        results["QUBO Relaxed (50 Initializations)"]["p"].append(qubo50_p)
        results["QUBO Relaxed (50 Initializations)"]["a"].append(qubo50_a)

    # Format output: Averaging results and adding scaled standard deviations
    formatted_results = {}
    for key, value in results.items():
        formatted_results[key] = {
            "p": [round(np.mean([run[i] for run in value["p"]]), 4) for i in range(p_max + 1)],
            "a": [round(np.mean([run[i] for run in value["a"]]), 4) for i in range(p_max + 1)],
            "p_std": [round(custom_std_scale * np.std([run[i] for run in value["p"]]), 4) for i in range(p_max + 1)],
            "a_std": [round(custom_std_scale * np.std([run[i] for run in value["a"]]), 4) for i in range(p_max + 1)],
        }

    return formatted_results

def format_dictionary(data):
    formatted_output = ""
    for key, values in data.items():
        formatted_output += f"{key}:\n"
        for section, section_values in values.items():
            formatted_output += f"  {section}: {section_values}\n"
        formatted_output += "\n"
    return formatted_output

def plot_numbers(prob, dict):
    
    def warmstart_table(prob, dict):
        print(f"                     {prob} ")
        print(f"{'Warmstart Values':<35} {'p':<10} {'a':<10}")
        print("-" * 55)
        
        for warmstart, values in dict.items():
            p = values["p"][-1]
            a = values["a"][-1]
            print(f"{warmstart:<35} {p:<10} {a:<10}")

    print("-" *55)
    warmstart_table(prob, dict)

    def warmstarts_std(dict):
        # Extract the last value and its std for `p` and `a` for each warmstart
        last_values = {}
        for warmstart, values in dict.items():
            last_values[warmstart] = {
                'p': values['p'][-1], 'p_std': values['p_std'][-1],
                'a': values['a'][-1], 'a_std': values['a_std'][-1]
            }
    
        # Sort warmstarts by largest last value of `p` or `a`
        sorted_p = sorted(last_values.items(), key=lambda x: x[1]['p'], reverse=True)
        sorted_a = sorted(last_values.items(), key=lambda x: x[1]['a'], reverse=True)
    
        # Function to find other warmstarts within STD for a specific warmstart
        def find_within_std(sorted_list, key, std_key):
            results = {}
            for current_name, current_dict in sorted_list:
                within_std = []
                for other_name, other_dict in sorted_list:
                    if current_name != other_name:
                        is_within = abs(current_dict[key] - other_dict[key]) <= (
                            current_dict[std_key] + other_dict[std_key]
                        )
                        if is_within:
                            within_std.append(other_name)
                results[current_name] = within_std
            return results
    
        # Create lists of warmstarts within STD for `p` and `a`
        p_within_std = find_within_std(sorted_p, 'p', 'p_std')
        a_within_std = find_within_std(sorted_a, 'a', 'a_std')
    
        print("-" *55)
        print("Greatest-Least, + standard deviatons\n")
    
        # Print results for `a`
        print("Warmstarts for p (greatest to least):")
        for idx, (warmstart, _) in enumerate(sorted_p, start=1):
            within_std_list = ' '.join(p_within_std[warmstart]) or '0'
            print(f"{idx}. {warmstart}: {within_std_list}")
        
        # Print results for `a`
        print("\nWarmstarts for a (greatest to least):")
        for idx, (warmstart, _) in enumerate(sorted_a, start=1):
            within_std_list = ' '.join(a_within_std[warmstart]) or '0'
            print(f"{idx}. {warmstart}: {within_std_list}")

    warmstarts_std(dict)

def ensure_dir_exists(file_path):
    """Ensure that the directory for a given file path exists."""
    directory = os.path.dirname(file_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)

def delete_files(files):
    for file in files:
        if os.path.exists(file):
            os.remove(file)
        
def combine_plots_vertical(pdf_files, output_file, overlap=5):
    ensure_dir_exists(output_file)
    """
    Combine PDFs vertically into one PDF, centering each page horizontally,
    with optional overlap between pages.
    """
    doc = fitz.open()

    total_width = 0
    total_height = 0
    pages = []

    for pdf in pdf_files:
        src = fitz.open(pdf)
        for page in src:
            rect = page.rect
            pages.append((pdf, page.number, rect))
            total_width = max(total_width, rect.width)
            total_height += rect.height - overlap
        src.close()

    total_height += overlap

    new_page = doc.new_page(width=total_width, height=total_height)

    y_offset = 0
    for pdf, page_num, rect in pages:
        src = fitz.open(pdf)
        x_offset = (total_width - rect.width) / 2
        target = fitz.Rect(x_offset, y_offset, x_offset + rect.width, y_offset + rect.height)
        new_page.show_pdf_page(target, src, page_num)
        y_offset += rect.height - overlap
        src.close()

    doc.save(output_file)

def combine_plots_grid(pdf_files, output_file, per_row=2, padding=10, side_padding=20):
    """
    Combine PDFs in a grid layout.

    Parameters:
        pdf_files (list): list of PDF paths
        output_file (str): output PDF path
        per_row (int): number of plots per row
        padding (float): spacing between plots
        side_padding (float): extra space on the right side
    """
    ensure_dir_exists(output_file)
    doc = fitz.open()

    page_sizes = []
    pages = []
    for pdf in pdf_files:
        src = fitz.open(pdf)
        for page in src:
            pages.append((pdf, page.number, page.rect))
            page_sizes.append(page.rect)
        src.close()

    rows = (len(pages) + per_row - 1) // per_row

    max_width = max(rect.width for rect in page_sizes)
    max_height = max(rect.height for rect in page_sizes)

    new_width = per_row * max_width + (per_row - 1) * padding + side_padding
    new_height = rows * max_height + (rows - 1) * padding
    new_page = doc.new_page(width=new_width, height=new_height)

    for idx, (pdf, page_num, rect) in enumerate(pages):
        row = idx // per_row
        col = idx % per_row

        x_offset = col * (max_width + padding)
        y_offset = row * (max_height + padding)

        target = fitz.Rect(x_offset, y_offset,
                           x_offset + rect.width,
                           y_offset + rect.height)
        src = fitz.open(pdf)
        new_page.show_pdf_page(target, src, page_num)
        src.close()

    doc.save(output_file)
    
def create_legend_pdf(path=None, dpi=600):
    ensure_dir_exists(path)
    """
    Creates a standalone horizontal legend PDF with a box around it,
    larger and centered so it matches other stacked plots.
    """
    fig, ax = plt.subplots(figsize=(13, 3))
    colors = list(mcolors.TABLEAU_COLORS.values())

    handles = [
        Line2D([0], [0], linestyle="--", marker="o", color=colors[0], markersize=15, linewidth=5, label="No Warmstart"),
        Line2D([0], [0], linestyle="-", marker="o", color=colors[1], markersize=15, linewidth=5, label="Warmstart, first rotation"),
        Line2D([0], [0], linestyle="-", marker="o", color=colors[2], markersize=15, linewidth=5, label="Warmstart, last rotation"),
        Line2D([0], [0], linestyle="-", marker="o", color=colors[3], markersize=15, linewidth=5, label="Warmstart, no rotation"),
        Line2D([0], [0], linestyle="-", marker="s", color=colors[4], markersize=15, linewidth=5, label="QUBO Relaxed"),
    ]

    ax.legend(handles=handles,
              loc="center",
              fontsize=30,
              frameon=True,
              borderpad=1.5,
              labelspacing=1.2,
              handlelength=3,
              ncol=len(handles),
              handletextpad=1)

    ax.axis("off")

    if path:
        fig.savefig(path, format="pdf", bbox_inches="tight", dpi=dpi)
    plt.close(fig)