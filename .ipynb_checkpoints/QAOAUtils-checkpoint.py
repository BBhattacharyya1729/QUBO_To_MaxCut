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
    return SparsePauliOp("I" * (n-i-1) + "Z" + "I" * i)

def getHamiltonian(A):
    n = len(A)
    H = 0 * SparsePauliOp("I" * n)
    for i in range(n):
        for j in range(n):
            H -= 1/4 * A[i][j] * indexedZ(i,n) @ indexedZ(j,n)
    return H.simplify()

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

def SU2_op(x,y,z,t):
    return np.array([[np.cos(t)-1j * np.sin(t)*z, -np.sin(t) * (y+1j * x)],[ -np.sin(t) * (-y+1j * x),np.cos(t)+1j * np.sin(t)*z]])

def apply_single_qubit_op(psi,U,q):
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
    return np.array(scipy.sparse.csr_matrix.diagonal(getHamiltonian(np.flip(A)).to_matrix(sparse=True))).real

def apply_mixer(psi,U_ops):
    for n in range(0,len(U_ops)):
        psi = apply_single_qubit_op(psi, U_ops[n], n)
    return psi

def cost_layer(precomp,psi,t):
    return np.exp(-1j * precomp*t) * psi

def QAOA_eval(precomp,params,mixer_ops=None,init=None):
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
    return np.sum(psi.conjugate() * precomp * psi).real

def Q2_data(theta_list,rotation=None):
    angles = vertex_on_top(theta_list,rotation)
    init = reduce(lambda a,b: np.kron(a,b), [np.array([np.cos(v/2), np.exp(-1j/2 * np.pi)*np.sin(v/2)],dtype='complex128') for v in angles])
    mixer_ops = lambda t: [  SU2_op(0,-np.sin(v),np.cos(v),t) for v in angles]
    return init,mixer_ops

def Q3_data(theta_list,rotation=None,z_rot=None):
    angles = vertex_on_top(theta_list,rotation,z_rot=z_rot)
    init = reduce(lambda a,b: np.kron(a,b), [np.array([np.cos(v[0]/2), np.exp(1j * v[1])*np.sin(v[0]/2)],dtype='complex128') for v in angles])
    mixer_ops = lambda t:  [SU2_op(np.sin(v[0])*np.cos(v[1]),np.sin(v[0])*np.sin(v[1]),np.cos(v[0]),t) for v in angles]
    return init,mixer_ops

def single_circuit_optimization_eff(precomp,opt,mixer_ops,init,p,param_guess=None):

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
    res = opt.minimize(fun= compute_expectation, x0=init_param)#x0 = np.zeros(2*p))#
    return np.max(history["cost"]),history["params"][np.argmax(history["cost"])],history

def circuit_optimization_eff(precomp,opt,mixer_ops,init,p,reps=10,name=None,verbose=False,param_guesses=None):
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
    precomp = pre_compute(A)
    _,BM2_theta_list,_=solve_BM2(A,**BM_kwargs)
    _,BM3_theta_list,_=solve_BM3(A,**BM_kwargs)
    GW_Y = GW(A)
    _,GW2_theta_list,_=GW2(A,GW_Y=GW_Y,**GW_kwargs)
    _,GW3_theta_list,_=GW3(A,GW_Y=GW_Y,**GW_kwargs)
    v,M=brute_force_maxcut(A)
    return [precomp,BM2_theta_list,BM3_theta_list,GW2_theta_list,GW3_theta_list,v,M]

def warmstart_comp(A,opt,p_max=5,rotation_options=[None,0,-1],BM_kwargs={"iters":100, "reps":50, "eta":0.05},GW_kwargs={"reps":50},reps=10,optimizer_kwargs={'name':None,'verbose':True},ws_list=['BM2','BM3','GW2','GW3',None],initial_data=None,keep_hist=False):
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
    if(precomp is None):
        precomp  = pre_compute(A)
    b_list = np.argwhere(abs(precomp - np.amax(precomp))<1e-10)
    b_list = np.reshape(b_list,(len(b_list),))
    b_list = [bin(b)[2:] for b in  b_list]
    return np.array([2*np.array([int(i) for i in '0'*(len(A)-len(b))+b])-1 for b in b_list]),np.max(precomp)

def opt_sampling_prob(v,precomp,params,mixer_ops=None,init=None):
    psi = QAOA_eval(precomp,params,mixer_ops,init)
    return np.sum(abs(psi[[np.sum([2**i * v for i,v in enumerate(l[::-1])]) for l in ((v+1)//2)]])**2)

###Depth 0 Test
def depth0_ws_comp(n,A_func,ws_list = ['BM2','BM3','GW2','GW3'],rotation_options = None,count=1000):
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

    markers = ['o', 'o', 'o', 'o', 'o']  # Define a list of markers

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
    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(handles, ['No Warmstart', 'Warmstart First Rotation', 'Warmstart Last Rotation', 'Warmstart No Rotation', 'Qubo Relaxed (10 Initalizations)', 'Qubo Relaxed (50 Initalizations)'], loc='lower center', bbox_to_anchor=(0.5, -0.1), ncol=6, fontsize=20,)
    fig.suptitle(f"{prob}", fontsize=30, y=0.97)
    
    if path is not None:
        plt.savefig(path + ".pdf", dpi=300, bbox_inches='tight')
    
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

