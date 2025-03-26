from qiskit.quantum_info import SparsePauliOp
from WarmStartUtils import *
import numpy as np
from functools import reduce
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from tqdm import tqdm
from tqdm.contrib import itertools
from tqdm.notebook import tqdm

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


def get_depth_combined(prob, idx_dict, PSC_DATA, DATA, PSC_DATA50, M_list, A_list, path=None, p_max=5, std=.25):
   
    fig, axs = plt.subplots(2, 4, figsize=(38.4, 7.2), sharey='row')  # 2x4 grid for all plots

    plt.subplots_adjust(wspace=0, hspace=0.15)  # Adjust spacing

    markers = ['o', 'o', 'o', 'o', 's']  # Define a list of markers
    fig.suptitle(prob, fontsize=50, y=0.94)  # Set the title with the name of `prob`

    def plot_depth(ws_list, ax):
        ax.tick_params(axis='y', which='both', length=5, width=1)
        ax.minorticks_off()
        marker_idx = 0  # Initialize marker index
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
                line, = ax.plot(range(0, p_max + 1), mean_data, marker=markers[marker_idx % len(markers)], linestyle='--', label="None")
                ax.fill_between(range(0, p_max + 1), 
                                (mean_data - std * std_dev_data), 
                                (mean_data + std * std_dev_data), 
                                alpha=0.2, color=line.get_color())
                marker_idx += 1  # Increment marker index
            else:
                for r, label in zip([0, -1, None], ['Warmstart First Rotation', 'Warmstart Last Rotation', 'Warmstart No Rotation']):
                    data = []
                    for idx in range(*idx_dict[prob]):
                        M = M_list[idx]
                        A = A_list[idx]
                        data.append([abs(np.max(l))
                                     for l in [DATA[idx][p][ws][r]['probs'] for p in range(0, 1 + p_max)]])
                    mean_data = np.mean(data, axis=0)
                    std_dev_data = np.std(data, axis=0)
                    line, = ax.plot(range(0, p_max + 1), mean_data, marker=markers[marker_idx % len(markers)], label=label)
                    ax.fill_between(range(0, p_max + 1), 
                                    (mean_data - std * std_dev_data), 
                                    (mean_data + std * std_dev_data), 
                                    alpha=0.2, color=line.get_color())
                    marker_idx += 1  # Increment marker index
        #PSC
        data = []
        for idx in range(*idx_dict[prob]):
            M = M_list[idx]
            A = A_list[idx]
            data.append([abs(np.max(l))  
                         for l in [PSC_DATA[idx][p]['probs'] for p in range(0, 1 + p_max)]])
        mean_data = np.mean(data, axis=0)
        std_dev_data = np.std(data, axis=0)
        line, = ax.plot(range(0, p_max + 1), mean_data, marker=markers[marker_idx % len(markers)], label='PSC')
        ax.fill_between(range(0, p_max + 1), 
                        (mean_data - std * std_dev_data), 
                        (mean_data + std * std_dev_data), 
                        alpha=0.2, color=line.get_color())
        #PSC_50
        data_50 = []
        for idx in range(*idx_dict[prob]):
            M = M_list[idx]
            A = A_list[idx]
            data_50.append([abs(np.max(l))  
                            for l in [PSC_DATA50[idx][p]['probs'] for p in range(0, 1 + p_max)]])
        mean_data_50 = np.mean(data_50, axis=0)
        std_dev_data_50 = np.std(data_50, axis=0)
        line, = ax.plot(range(0, p_max + 1), mean_data_50, marker=markers[marker_idx % len(markers)], label='PSC50', color='goldenrod')
        ax.fill_between(range(0, p_max + 1), 
                        (mean_data_50 - std * std_dev_data_50), 
                        (mean_data_50 + std * std_dev_data_50),
                        color=line.get_color(), alpha=0.2)
        ax.legend().remove()  # Remove legend

    def plot_cost(ws_list, ax):
        def compute_and_plot(data, ws, r, ax, label_suffix, line_style='-', fill_color=None, color=None):
            mean_data = np.mean(data, axis=0)
            std_dev_data = np.std(data, axis=0)
            line, = ax.plot(range(0, p_max + 1), mean_data, marker=markers[marker_idx % len(markers)], label=f'{ws} {label_suffix}', linestyle=line_style, color=color)
            ax.fill_between(range(0, p_max + 1), 
                            mean_data - std * std_dev_data, 
                            mean_data + std * std_dev_data, 
                            alpha=0.2, color=line.get_color())
    
        ws_data = {0: [], -1: [], None: []}
        psc_data = []
        psc_data50 = []
        none_data = []
        marker_idx = 0  # Initialize marker index
    
        for idx in range(*idx_dict[prob]):
            M = M_list[idx]
            A = A_list[idx]
            m = -brute_force_maxcut(-A)[-1]
    
            none_data.append([(np.max(l) - m) / (M - m) 
                              for l in [DATA[idx][p][None]['cost'] for p in range(0, 1 + p_max)]])
    
            if ws_list[0] is not None:
                for r, label in zip([0, -1, None], ['Warmstart First Rotation', 'Warmstart Last Rotation', 'Warmstart No Rotation']):
                    ws_data[r].append([(np.max(l) - m) / (M - m) 
                                       for l in [DATA[idx][p][ws_list[0]][r]['cost'] for p in range(0, 1 + p_max)]])
            
            psc_data.append([(np.max(l) - m - np.sum(-A[:-1, :-1]) / 4) / (M - m) 
                             for l in [PSC_DATA[idx][p]['cost'] for p in range(0, 1 + p_max)]])

            psc_data50.append([(np.max(l) - m - np.sum(-A[:-1, :-1]) / 4) / (M - m) 
                               for l in [PSC_DATA50[idx][p]['cost'] for p in range(0, 1 + p_max)]])
    
        compute_and_plot(none_data, 'None', "", ax, label_suffix="", line_style='--')
        marker_idx += 1  # Increment marker index
        if ws_list[0] is not None:
            for r, r_data in ws_data.items():
                compute_and_plot(r_data, ws_list[0], r, ax, label_suffix=f"{r}")
                marker_idx += 1  # Increment marker index
    
        compute_and_plot(psc_data, 'PSC', "", ax, label_suffix="")
        compute_and_plot(psc_data50, 'PSC50', "", ax, label_suffix="", fill_color="goldenrod", color="goldenrod")
        ax.legend().remove()  # Remove legend
    
    # Plot the depth graphs in the first row
    plot_depth([None, 'GW2'], axs[0, 0])
    plot_depth([None, 'GW3'], axs[0, 1])
    plot_depth([None, 'BM2'], axs[0, 2])
    plot_depth([None, 'BM3'], axs[0, 3])
    
    # Plot the cost graphs in the second row
    plot_cost(['GW2'], axs[1, 0])
    plot_cost(['GW3'], axs[1, 1])
    plot_cost(['BM2'], axs[1, 2])
    plot_cost(['BM3'], axs[1, 3])
    
    # Ensure the x-axis line is visible and remove x-axis labels for the first row
    for ax in axs.flat:
        ax.spines['bottom'].set_visible(True)  # Ensure x-axis line is visible
        ax.set_xticks(range(p_max + 1))  # Set x-axis ticks
        ax.set_xticklabels(range(p_max + 1))  # Set x-axis labels
    
    # Set y-axis label only for the leftmost plot in each row
    axs[0, 0].set_ylabel(r'$\mathcal{P}$', fontsize=25, labelpad=10)
    axs[1, 0].set_ylabel(r'α', fontsize=25, labelpad=10)
    
    # Set labels for x-axis
    for ax in axs[1, :]:
        ax.set_xlabel('p', fontsize=25)
    
    # Set titles for the top row only
    axs[0, 0].set_title("GW2", fontsize=20)
    axs[0, 1].set_title("GW3", fontsize=20)
    axs[0, 2].set_title("BM2", fontsize=20)
    axs[0, 3].set_title("BM3", fontsize=20)

    for ax in axs.flat:
        ax.spines['bottom'].set_visible(True)  # Ensure x-axis line is visible
        ax.set_xticks(range(p_max + 1))  # Set x-axis ticks
        ax.set_xticklabels(range(p_max + 1))  # Set x-axis labels
        ax.tick_params(axis='both', labelsize=15)  # Adjust tick label size for both axes

    for ax in axs[0, :]:
        ax.set_xticklabels([])  # Remove x-axis labels
        ax.set_xticks([])

    # Create a legend at the bottom center
    fig.suptitle(f"{prob}", fontsize=30, y=0.97)
    
    if path is not None:
        plt.savefig(path + ".pdf", dpi=600, bbox_inches='tight')
    
    plt.show()

def get_depth_combined_PO(prob, idx_dict, PSC_DATA, DATA, PSC_DATA50, M_list, A_list, path=None, p_max=5, std=.25):
   
    fig, axs = plt.subplots(2, 4, figsize=(38.4, 7.2), sharey='row')  # 2x4 grid for all plots

    plt.subplots_adjust(wspace=0, hspace=0.15)  # Adjust spacing

    markers = ['o', 'o', 'o', 'o', 's']  # Define a list of markers
    fig.suptitle(prob, fontsize=50, y=0.94)  # Set the title with the name of `prob`

    def plot_depth(ws_list, ax):
        ax.tick_params(axis='y', which='both', length=5, width=1)
        ax.minorticks_off()
        marker_idx = 0  # Initialize marker index
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
                line, = ax.plot(range(0, p_max + 1), mean_data, marker=markers[marker_idx % len(markers)], linestyle='--', label="None")
                ax.fill_between(range(0, p_max + 1), 
                                (mean_data - std * std_dev_data), 
                                (mean_data + std * std_dev_data), 
                                alpha=0.2, color=line.get_color())
                marker_idx += 1  # Increment marker index
            else:
                for r, label in zip([0, -1, None], ['Warmstart First Rotation', 'Warmstart Last Rotation', 'Warmstart No Rotation']):
                    data = []
                    for idx in range(*idx_dict[prob]):
                        M = M_list[idx]
                        A = A_list[idx]
                        data.append([abs(np.max(l))
                                     for l in [DATA[idx][p][ws][r]['probs'] for p in range(0, 1 + p_max)]])
                    mean_data = np.mean(data, axis=0)
                    std_dev_data = np.std(data, axis=0)
                    line, = ax.plot(range(0, p_max + 1), mean_data, marker=markers[marker_idx % len(markers)], label=label)
                    ax.fill_between(range(0, p_max + 1), 
                                    (mean_data - std * std_dev_data), 
                                    (mean_data + std * std_dev_data), 
                                    alpha=0.2, color=line.get_color())
                    marker_idx += 1  # Increment marker index
        #PSC
        data = []
        for idx in range(*idx_dict[prob]):
            M = M_list[idx]
            A = A_list[idx]
            data.append([abs(np.max(l))  
                         for l in [PSC_DATA[idx][p]['probs'] for p in range(0, 1 + p_max)]])
        mean_data = np.mean(data, axis=0)
        std_dev_data = np.std(data, axis=0)
        line, = ax.plot(range(0, p_max + 1), mean_data, marker=markers[marker_idx % len(markers)], label='PSC')
        ax.fill_between(range(0, p_max + 1), 
                        (mean_data - std * std_dev_data), 
                        (mean_data + std * std_dev_data), 
                        alpha=0.2, color=line.get_color())
        #PSC_50
        ax.legend().remove()  # Remove legend

    def plot_cost(ws_list, ax):
        def compute_and_plot(data, ws, r, ax, label_suffix, line_style='-', fill_color=None, color=None):
            mean_data = np.mean(data, axis=0)
            std_dev_data = np.std(data, axis=0)
            line, = ax.plot(range(0, p_max + 1), mean_data, marker=markers[marker_idx % len(markers)], label=f'{ws} {label_suffix}', linestyle=line_style, color=color)
            ax.fill_between(range(0, p_max + 1), 
                            mean_data - std * std_dev_data, 
                            mean_data + std * std_dev_data, 
                            alpha=0.2, color=line.get_color())
    
        ws_data = {0: [], -1: [], None: []}
        psc_data = []
        psc_data50 = []
        none_data = []
        marker_idx = 0  # Initialize marker index
    
        for idx in range(*idx_dict[prob]):
            M = M_list[idx]
            A = A_list[idx]
            m = -brute_force_maxcut(-A)[-1]
    
            none_data.append([(np.max(l) - m) / (M - m) 
                              for l in [DATA[idx][p][None]['cost'] for p in range(0, 1 + p_max)]])
    
            if ws_list[0] is not None:
                for r, label in zip([0, -1, None], ['Warmstart First Rotation', 'Warmstart Last Rotation', 'Warmstart No Rotation']):
                    ws_data[r].append([(np.max(l) - m) / (M - m) 
                                       for l in [DATA[idx][p][ws_list[0]][r]['cost'] for p in range(0, 1 + p_max)]])
            
            psc_data.append([(np.max(l) - m - np.sum(-A[:-1, :-1]) / 4) / (M - m) 
                             for l in [PSC_DATA[idx][p]['cost'] for p in range(0, 1 + p_max)]])

            psc_data50.append([(np.max(l) - m - np.sum(-A[:-1, :-1]) / 4) / (M - m) 
                               for l in [PSC_DATA50[idx][p]['cost'] for p in range(0, 1 + p_max)]])
    
        compute_and_plot(none_data, 'None', "", ax, label_suffix="", line_style='--')
        marker_idx += 1  # Increment marker index
        if ws_list[0] is not None:
            for r, r_data in ws_data.items():
                compute_and_plot(r_data, ws_list[0], r, ax, label_suffix=f"{r}")
                marker_idx += 1  # Increment marker index
    
        compute_and_plot(psc_data, 'PSC', "", ax, label_suffix="")
        ax.legend().remove()  # Remove legend
    
    # Plot the depth graphs in the first row
    plot_depth([None, 'GW2'], axs[0, 0])
    plot_depth([None, 'GW3'], axs[0, 1])
    plot_depth([None, 'BM2'], axs[0, 2])
    plot_depth([None, 'BM3'], axs[0, 3])
    
    # Plot the cost graphs in the second row
    plot_cost(['GW2'], axs[1, 0])
    plot_cost(['GW3'], axs[1, 1])
    plot_cost(['BM2'], axs[1, 2])
    plot_cost(['BM3'], axs[1, 3])
    
    # Ensure the x-axis line is visible and remove x-axis labels for the first row
    for ax in axs.flat:
        ax.spines['bottom'].set_visible(True)  # Ensure x-axis line is visible
        ax.set_xticks(range(p_max + 1))  # Set x-axis ticks
        ax.set_xticklabels(range(p_max + 1))  # Set x-axis labels
    
    # Set y-axis label only for the leftmost plot in each row
    axs[0, 0].set_ylabel(r'$\mathcal{P}$', fontsize=25, labelpad=10)
    axs[1, 0].set_ylabel(r'α', fontsize=25, labelpad=10)
    
    # Set labels for x-axis
    for ax in axs[1, :]:
        ax.set_xlabel('p', fontsize=25)
    
    # Set titles for the top row only
    axs[0, 0].set_title("GW2", fontsize=20)
    axs[0, 1].set_title("GW3", fontsize=20)
    axs[0, 2].set_title("BM2", fontsize=20)
    axs[0, 3].set_title("BM3", fontsize=20)

    for ax in axs.flat:
        ax.spines['bottom'].set_visible(True)  # Ensure x-axis line is visible
        ax.set_xticks(range(p_max + 1))  # Set x-axis ticks
        ax.set_xticklabels(range(p_max + 1))  # Set x-axis labels
        ax.tick_params(axis='both', labelsize=15)  # Adjust tick label size for both axes

    for ax in axs[0, :]:
        ax.set_xticklabels([])  # Remove x-axis labels
        ax.set_xticks([])

    fig.suptitle(f"{prob}", fontsize=30, y=0.97)
    
    if path is not None:
        plt.savefig(path + ".pdf", dpi=600, bbox_inches='tight')
    
    plt.show()

def get_depth_combined_gw2(prob, idx_dict, PSC_DATA, DATA, PSC_DATA50, M_list, A_list, path=None, p_max=5, std=.25):
    fig, axs = plt.subplots(2, 1, figsize=(18, 16), sharex=True)  # Create 2 subplots for depth and cost

    plt.subplots_adjust(wspace=0, hspace=0.1, bottom = .5)  # Adjust spacing between plots
    markers = ['o', 'o', 'o', 'o', 's']  # Define a list of markers
    fig.suptitle(prob, fontsize=50, y=0.94)  # Set the title with the name of `prob`

    def plot_depth(ws_list, ax):
        ax.tick_params(axis='y', which='both', length=5, width=1)
        ax.minorticks_off()
        marker_idx = 0  # Initialize marker index

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
                line, = ax.plot(range(0, p_max + 1), mean_data, marker=markers[marker_idx % len(markers)], linestyle='--', label="None")
                ax.fill_between(range(0, p_max + 1), 
                                (mean_data - std * std_dev_data), 
                                (mean_data + std * std_dev_data), 
                                alpha=0.2, color=line.get_color())
                marker_idx += 1  # Increment marker index
            else:
                for r, label in zip([0, -1, None], ['Warmstart First Rotation', 'Warmstart Last Rotation', 'Warmstart No Rotation']):
                    data = []
                    for idx in range(*idx_dict[prob]):
                        M = M_list[idx]
                        A = A_list[idx]
                        data.append([abs(np.max(l))
                                     for l in [DATA[idx][p][ws][r]['probs'] for p in range(0, 1 + p_max)]])
                    mean_data = np.mean(data, axis=0)
                    std_dev_data = np.std(data, axis=0)
                    line, = ax.plot(range(0, p_max + 1), mean_data, marker=markers[marker_idx % len(markers)], label=label)
                    ax.fill_between(range(0, p_max + 1), 
                                    (mean_data - std * std_dev_data), 
                                    (mean_data + std * std_dev_data), 
                                    alpha=0.2, color=line.get_color())
                    marker_idx += 1  # Increment marker index
        
        # PSC
        data = []
        for idx in range(*idx_dict[prob]):
            M = M_list[idx]
            A = A_list[idx]
            data.append([abs(np.max(l))  
                         for l in [PSC_DATA[idx][p]['probs'] for p in range(0, 1 + p_max)]])
        mean_data = np.mean(data, axis=0)
        std_dev_data = np.std(data, axis=0)
        line, = ax.plot(range(0, p_max + 1), mean_data, marker=markers[marker_idx % len(markers)], label='QUBO Relaxed (10 Initializations)')
        ax.fill_between(range(0, p_max + 1), 
                        (mean_data - std * std_dev_data), 
                        (mean_data + std * std_dev_data), 
                        alpha=0.2, color=line.get_color())
        
        # PSC_50
        data_50 = []
        for idx in range(*idx_dict[prob]):
            M = M_list[idx]
            A = A_list[idx]
            data_50.append([abs(np.max(l))  
                            for l in [PSC_DATA50[idx][p]['probs'] for p in range(0, 1 + p_max)]])
        mean_data_50 = np.mean(data_50, axis=0)
        std_dev_data_50 = np.std(data_50, axis=0)
        line, = ax.plot(range(0, p_max + 1), mean_data_50, marker=markers[marker_idx % len(markers)], label='QUBO Relaxed (50 Initializations)', color='goldenrod')
        ax.fill_between(range(0, p_max + 1), 
                        (mean_data_50 - std * std_dev_data_50), 
                        (mean_data_50 + std * std_dev_data_50),
                        color=line.get_color(), alpha=0.2)
        
    def plot_cost(ws_list, ax):
        def compute_and_plot(data, label_suffix, ax, line_style='-', fill_color=None, color=None):
            mean_data = np.mean(data, axis=0)
            std_dev_data = np.std(data, axis=0)
            line, = ax.plot(range(0, p_max + 1), mean_data, marker=markers[marker_idx % len(markers)], label=label_suffix, linestyle=line_style, color=color)
            ax.fill_between(range(0, p_max + 1), 
                            mean_data - std * std_dev_data, 
                            mean_data + std * std_dev_data, 
                            alpha=0.2, color=line.get_color())
    
        ws_data = {0: [], -1: [], None: []}
        psc_data = []
        psc_data50 = []
        none_data = []
        marker_idx = 0  # Initialize marker index
    
        for idx in range(*idx_dict[prob]):
            M = M_list[idx]
            A = A_list[idx]
            m = -brute_force_maxcut(-A)[-1]
    
            none_data.append([(np.max(l) - m) / (M - m) 
                              for l in [DATA[idx][p][None]['cost'] for p in range(0, 1 + p_max)]])
    
            if ws_list[0] is not None:
                for r in ws_data.keys():
                    ws_data[r].append([(np.max(l) - m) / (M - m) 
                                       for l in [DATA[idx][p][ws_list[0]][r]['cost'] for p in range(0, 1 + p_max)]])
            
            psc_data.append([(np.max(l) - m - np.sum(-A[:-1, :-1]) / 4) / (M - m) 
                             for l in [PSC_DATA[idx][p]['cost'] for p in range(0, 1 + p_max)]])

            psc_data50.append([(np.max(l) - m - np.sum(-A[:-1, :-1]) / 4) / (M - m) 
                               for l in [PSC_DATA50[idx][p]['cost'] for p in range(0, 1 + p_max)]])
    
        compute_and_plot(none_data, 'None', axs[1], line_style='--')
        marker_idx += 1  # Increment marker index
        
        if ws_list[0] is not None:
            for r in ws_data.keys():
                compute_and_plot(ws_data[r], f'GW2 {r}', axs[1])  # Cost plot for GW2
                marker_idx += 1  # Increment marker index
    
        compute_and_plot(psc_data, 'Qubo Relaxed (10 Initializations)', axs[1])
        compute_and_plot(psc_data50, 'Qubo Relaxed (50 Initializations)', axs[1], color='goldenrod')
    
    # Plot the depth and cost graphs for 'GW2'
    plot_depth([None, 'GW2'], axs[0])  # Depth plot for GW2
    plot_cost(['GW2'], axs[1])  # Cost plot for GW2
    
    # Ensure the x-axis line is visible and adjust x-axis ticks
    for ax in axs.flat:
        ax.spines['bottom'].set_visible(True)  # Ensure x-axis line is visible
        ax.set_xticks(range(p_max + 1))  # Set x-axis ticks
        ax.set_xticklabels(range(p_max + 1))  # Set x-axis labels
    
    # Set y-axis label only for the leftmost plot in each row
    axs[0].set_ylabel(r'$\mathcal{P}$', fontsize=35, labelpad=10)
    axs[1].set_ylabel(r'$\alpha$', fontsize=35, labelpad=10)  # Change 'a' to Greek alpha (α) in the y-axis label
    
    # Set labels for x-axis
    axs[1].set_xlabel('p', fontsize=35)

    # # Add combined legend below both subplots
    handles, labels = axs[0].get_legend_handles_labels()
    # fig.legend(handles, labels, loc='lower center', ncol=6, bbox_to_anchor=(0.5, .07), fontsize=12, frameon=True)

    # Adjust spacing to prevent overlap with the legend
    plt.subplots_adjust(bottom=0.2)

    for ax in axs.flat:
        ax.tick_params(axis='both', labelsize=15)  # Adjust tick label size for both axes

    if path is not None:
        plt.savefig(path + ".pdf", dpi=600, bbox_inches='tight')

    plt.show()

def get_depth_combined_gw2_PO(prob, idx_dict, PSC_DATA, DATA, PSC_DATA50, M_list, A_list, path=None, p_max=5, std=.25):
    fig, axs = plt.subplots(2, 1, figsize=(18, 16), sharex=True)  # Create 2 subplots for depth and cost

    plt.subplots_adjust(wspace=0, hspace=0.1, bottom = .5)  # Adjust spacing between plots
    markers = ['o', 'o', 'o', 'o', 's']  # Define a list of markers
    fig.suptitle(prob, fontsize=50, y=0.94)  # Set the title with the name of `prob`

    def plot_depth(ws_list, ax):
        ax.tick_params(axis='y', which='both', length=5, width=1)
        ax.minorticks_off()
        marker_idx = 0  # Initialize marker index

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
                line, = ax.plot(range(0, p_max + 1), mean_data, marker=markers[marker_idx % len(markers)], linestyle='--', label="None")
                ax.fill_between(range(0, p_max + 1), 
                                (mean_data - std * std_dev_data), 
                                (mean_data + std * std_dev_data), 
                                alpha=0.2, color=line.get_color())
                marker_idx += 1  # Increment marker index
            else:
                for r, label in zip([0, -1, None], ['Warmstart First Rotation', 'Warmstart Last Rotation', 'Warmstart No Rotation']):
                    data = []
                    for idx in range(*idx_dict[prob]):
                        M = M_list[idx]
                        A = A_list[idx]
                        data.append([abs(np.max(l))
                                     for l in [DATA[idx][p][ws][r]['probs'] for p in range(0, 1 + p_max)]])
                    mean_data = np.mean(data, axis=0)
                    std_dev_data = np.std(data, axis=0)
                    line, = ax.plot(range(0, p_max + 1), mean_data, marker=markers[marker_idx % len(markers)], label=label)
                    ax.fill_between(range(0, p_max + 1), 
                                    (mean_data - std * std_dev_data), 
                                    (mean_data + std * std_dev_data), 
                                    alpha=0.2, color=line.get_color())
                    marker_idx += 1  # Increment marker index
        
        # PSC
        data = []
        for idx in range(*idx_dict[prob]):
            M = M_list[idx]
            A = A_list[idx]
            data.append([abs(np.max(l))  
                         for l in [PSC_DATA[idx][p]['probs'] for p in range(0, 1 + p_max)]])
        mean_data = np.mean(data, axis=0)
        std_dev_data = np.std(data, axis=0)
        line, = ax.plot(range(0, p_max + 1), mean_data, marker=markers[marker_idx % len(markers)], label='QUBO Relaxed (10 Initializations)')
        ax.fill_between(range(0, p_max + 1), 
                        (mean_data - std * std_dev_data), 
                        (mean_data + std * std_dev_data), 
                        alpha=0.2, color=line.get_color())
        
        # PSC_50
        
    def plot_cost(ws_list, ax):
        def compute_and_plot(data, label_suffix, ax, line_style='-', fill_color=None, color=None):
            mean_data = np.mean(data, axis=0)
            std_dev_data = np.std(data, axis=0)
            line, = ax.plot(range(0, p_max + 1), mean_data, marker=markers[marker_idx % len(markers)], label=label_suffix, linestyle=line_style, color=color)
            ax.fill_between(range(0, p_max + 1), 
                            mean_data - std * std_dev_data, 
                            mean_data + std * std_dev_data, 
                            alpha=0.2, color=line.get_color())
    
        ws_data = {0: [], -1: [], None: []}
        psc_data = []
        psc_data50 = []
        none_data = []
        marker_idx = 0  # Initialize marker index
    
        for idx in range(*idx_dict[prob]):
            M = M_list[idx]
            A = A_list[idx]
            m = -brute_force_maxcut(-A)[-1]
    
            none_data.append([(np.max(l) - m) / (M - m) 
                              for l in [DATA[idx][p][None]['cost'] for p in range(0, 1 + p_max)]])
    
            if ws_list[0] is not None:
                for r in ws_data.keys():
                    ws_data[r].append([(np.max(l) - m) / (M - m) 
                                       for l in [DATA[idx][p][ws_list[0]][r]['cost'] for p in range(0, 1 + p_max)]])
            
            psc_data.append([(np.max(l) - m - np.sum(-A[:-1, :-1]) / 4) / (M - m) 
                             for l in [PSC_DATA[idx][p]['cost'] for p in range(0, 1 + p_max)]])

            psc_data50.append([(np.max(l) - m - np.sum(-A[:-1, :-1]) / 4) / (M - m) 
                               for l in [PSC_DATA50[idx][p]['cost'] for p in range(0, 1 + p_max)]])
    
        compute_and_plot(none_data, 'None', axs[1], line_style='--')
        marker_idx += 1  # Increment marker index
        
        if ws_list[0] is not None:
            for r in ws_data.keys():
                compute_and_plot(ws_data[r], f'GW2 {r}', axs[1])  # Cost plot for GW2
                marker_idx += 1  # Increment marker index
    
        compute_and_plot(psc_data, 'Qubo Relaxed (10 Initializations)', axs[1])
    
    # Plot the depth and cost graphs for 'GW2'
    plot_depth([None, 'GW2'], axs[0])  # Depth plot for GW2
    plot_cost(['GW2'], axs[1])  # Cost plot for GW2
    
    # Ensure the x-axis line is visible and adjust x-axis ticks
    for ax in axs.flat:
        ax.spines['bottom'].set_visible(True)  # Ensure x-axis line is visible
        ax.set_xticks(range(p_max + 1))  # Set x-axis ticks
        ax.set_xticklabels(range(p_max + 1))  # Set x-axis labels
    
    # Set y-axis label only for the leftmost plot in each row
    axs[0].set_ylabel(r'$\mathcal{P}$', fontsize=35, labelpad=10)
    axs[1].set_ylabel(r'$\alpha$', fontsize=35, labelpad=10)  # Change 'a' to Greek alpha (α) in the y-axis label
    
    # Set labels for x-axis
    axs[1].set_xlabel('p', fontsize=35)

    # # Add combined legend below both subplots
    handles, labels = axs[0].get_legend_handles_labels()
    # fig.legend(handles, labels, loc='lower center', ncol=6, bbox_to_anchor=(0.5, .07), fontsize=12, frameon=True)

    # Adjust spacing to prevent overlap with the legend
    plt.subplots_adjust(bottom=0.2)

    for ax in axs.flat:
        ax.tick_params(axis='both', labelsize=15)  # Adjust tick label size for both axes

    if path is not None:
        plt.savefig(path + ".pdf", dpi=600, bbox_inches='tight')

    plt.show()





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
        # Add the main type with a line break
        formatted_output += f"{key}:\n"
        for section, section_values in values.items():
            # Add the section name and its values with a line break
            formatted_output += f"  {section}: {section_values}\n"
        # Add an extra line break after each type
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
 