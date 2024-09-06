from QAOAUtils import *
import scipy
def relax_solve(Q,reps=10):
    x = []
    f = []
    for i in range(reps):
        res = scipy.optimize.minimize(lambda x: -get_cost(Q,x), x0=np.random.random(len(Q)),bounds=[[0,1]]*len(Q),options={"maxiter":1000})
        x.append(res.x)
        f.append(-res.fun)
    return x[np.argmax(f)]

def PSC_data(vals,eps=0.1):
    thetas = []
    for v in vals:
        if(v <= eps):
            thetas.append(2*np.arcsin(eps))
        elif(v>= 1-eps):
            thetas.append(2*np.arcsin(1-eps))
        else:
            thetas.append(2*np.arcsin(v))
    init = reduce(lambda a,b: np.kron(a,b), [np.array([np.cos(v/2), np.sin(v/2)],dtype='complex128') for v in thetas])
    mixer_ops = lambda t: [SU2_op(np.sin(v),0,np.cos(v),t) for v in thetas]
    return init,mixer_ops

def PSCHamiltonian(Q):
    n = len(Q)
    H = np.sum(Q) * SparsePauliOp("I" * n)/4
    for i in range(n):
        H -= 1/2 * np.sum(Q[i]) * indexedZ(i,n) 
        for j in range(n):
            H += 1/4 * Q[i][j] * indexedZ(i,n) @ indexedZ(j,n)
    return H.simplify()

def PSC_pre_compute(Q):
    return np.array(scipy.sparse.csr_matrix.diagonal(PSCHamiltonian(np.flip(Q)).to_matrix(sparse=True))).real


def PSC_opt_sampling_prob(v,precomp,params,mixer_ops=None,init=None):
    psi = QAOA_eval(precomp,params,mixer_ops,init)
    return np.sum(abs(psi[[np.sum([2**i * v for i,v in enumerate(l[::-1])]) for l in [map_reduce(_) for _ in v]]])**2)/2


def PSC_Run(A,opt,p_max,optimizer_kwargs={'name':None,'verbose':True},keep_hist=False,eps=0.1,reps=10):
    ###initialization
    Q = -A[:-1,:-1]
    precomp = PSC_pre_compute(Q)
    v,M=brute_force_maxcut(A)
    vals = relax_solve(Q)
    init,mixer_ops=PSC_data(vals,eps=eps)

    opt_data =[{} for _ in range(p_max+1)]


    opt_params=None

    ###Depth 0
    opt_data[0]['cost']=expval(precomp,QAOA_eval(precomp,[],mixer_ops=mixer_ops,init=init))
    opt_data[0]['params']=[]
    opt_data[0]['probs']=PSC_opt_sampling_prob(v,precomp,[],mixer_ops=mixer_ops,init=init)

    for p in range(1,p_max+1):
        guess  = [opt_params] + [None] * (reps-1)
        l=circuit_optimization_eff(precomp,opt,mixer_ops,init,p,reps=reps,param_guesses=guess,**optimizer_kwargs)
        opt_data[p]['cost']=l[0]
        opt_data[p]['params']=l[1]
        opt_data[p]['probs']=np.array([PSC_opt_sampling_prob(v,precomp,param,mixer_ops=mixer_ops,init=init) for param in l[1]])
        opt_params = list(l[1][np.argmax(l[0])])[:p] + [0] + list(l[1][np.argmax(l[0])])[p:] + [0]

        if(keep_hist):
            opt_data[p]['hist']=l[2]
    
    return opt_data


# def get_PSC_depth_cost_comp(prob,idx_dict,DATA,M_list,A_list,path=None,p_max=5):
#     fig = plt.figure(figsize=(10,10))
    
#     mean_data = np.zeros(p_max+1)
#     for idx in range(*idx_dict[prob]):
#         M = M_list[idx]
#         A = A_list[idx]
#         mean_data+=np.array([abs(np.max(l)-(M+np.sum(-A_list[:-1,:-1])/4))/(M+np.sum(-A_list[:-1,:-1])/4) for l in [DATA[idx][p][ws]['cost'] for p in range(0,1+p_max)]])
#     plt.plot(range(0,p_max+1),np.log10(mean_data/(idx_dict[prob][1]-idx_dict[prob][0])),label=str(ws)+" ")
        
#     plt.title(prob)
#     plt.xlabel('p')
#     plt.ylabel('Log Relative Error')
#     plt.legend()
#     if(path is not None):
#         plt.savefig(path+".pdf",dpi=300)
#     plt.show()


# def get_depth_cost_comp(prob,idx_dict,PSC_DATA,DATA,M_list,A_list,ws_list=[None, 'GW2','GW3'],path=None,p_max=5):
#     fig = plt.figure(figsize=(10,10))
#     for ws in ws_list:
#         if(ws is None):
#             mean_data = np.zeros(p_max+1)
#             for idx in range(*idx_dict[prob]):
#                 M = M_list[idx]
#                 A = A_list[idx]
#                 mean_data+=np.array([abs(np.max(l)-M)/(M+np.sum(-A[:-1,:-1])/4) for l in [DATA[idx][p][ws]['cost'] for p in range(0,1+p_max)]])
#             plt.plot(range(0,p_max+1),np.log10(mean_data/(idx_dict[prob][1]-idx_dict[prob][0])),label=str(ws)+" ")
#         else:
#             for r in [0,-1,None]:
#                 mean_data = np.zeros(p_max+1)
#                 for idx in range(*idx_dict[prob]):
#                     M = M_list[idx]
#                     A = A_list[idx]
#                     mean_data+=np.array([abs(np.max(l)-M)/(M+np.sum(-A[:-1,:-1])/4) for l in [DATA[idx][p][ws][r]['cost'] for p in range(0,1+p_max)]])
#                 plt.plot(range(0,p_max+1),np.log10(mean_data/(idx_dict[prob][1]-idx_dict[prob][0])),label=str(ws)+" "+str(r))
#     for idx in range(*idx_dict[prob]):
#         M = M_list[idx]
#         A = A_list[idx]
#         mean_data+=np.array([abs(np.max(l)-(M+np.sum(-A[:-1,:-1])/4))/(M+np.sum(-A[:-1,:-1])/4) for l in [PSC_DATA[idx][p]['cost'] for p in range(0,1+p_max)]])
#     plt.plot(range(0,p_max+1),np.log10(mean_data/(idx_dict[prob][1]-idx_dict[prob][0])), label = 'PSC')
       
#     plt.title(prob)
#     plt.xlabel('p')
#     plt.ylabel('Log Relative Error')
#     plt.legend()
#     if(path is not None):
#         plt.savefig(path+".pdf",dpi=300)
#     plt.show()


# def get_depth_prob_comp(prob,idx_dict,PSC_DATA,DATA,M_list,A_list,ws_list=[None, 'GW2','GW3'],path=None,p_max=5):
#     fig = plt.figure(figsize=(10,10))
#     for ws in ws_list:
#         if(ws is None):
#             mean_data = np.zeros(p_max+1)
#             for idx in range(*idx_dict[prob]):
#                 M = M_list[idx]
#                 mean_data+=np.array([abs(np.max(l)) for l in [DATA[idx][p][ws]['probs'] for p in range(0,1+p_max)]])
#             plt.plot(range(0,p_max+1),np.log10(1-mean_data/(idx_dict[prob][1]-idx_dict[prob][0])),label=str(ws)+" ")
#         else:
#             for r in [0,-1,None]:
#                 mean_data = np.zeros(p_max+1)
#                 for idx in range(*idx_dict[prob]):
#                     M = M_list[idx]
#                     mean_data+=np.array([abs(np.max(l)) for l in [DATA[idx][p][ws][r]['probs'] for p in range(0,1+p_max)]])
#                 plt.plot(range(0,p_max+1),np.log10(1-mean_data/(idx_dict[prob][1]-idx_dict[prob][0])),label=str(ws)+" "+str(r))
#     for idx in range(*idx_dict[prob]):
#         M = M_list[idx]
#         A = A_list[idx]
#         mean_data+=np.array([abs(np.max(l)) for l in [PSC_DATA[idx][p]['probs'] for p in range(0,1+p_max)]])
#     plt.plot(range(0,p_max+1),np.log10(1-mean_data/(idx_dict[prob][1]-idx_dict[prob][0])),label="PSC")
#     plt.title(prob)
#     plt.xlabel('p (Depth)')
#     plt.ylabel('Probabilty')
#     plt.legend()
#     if(path is not None):
#         plt.savefig(path+".pdf",dpi=300)
#     plt.show()



# def get_PSC_depth_cost_comp(prob,idx_dict,DATA,M_list,A_list,path=None,p_max=5):
#     fig = plt.figure(figsize=(10,10))
    
#     mean_data = np.zeros(p_max+1)
     
#     plt.title(prob)
#     plt.xlabel('p')
#     plt.ylabel('Log Relative Error')
#     plt.legend()
#     if(path is not None):
#         plt.savefig(path+".pdf",dpi=300)
#     plt.show()


# def get_PSC_depth_prob_comp(prob,idx_dict,DATA,M_list,A_list,path=None,p_max=5):
#     fig = plt.figure(figsize=(10,10))
    
#     mean_data = np.zeros(p_max+1)
#     for idx in range(*idx_dict[prob]):
#         M = M_list[idx]
#         A = A_list[idx]
#         mean_data+=np.array([abs(np.max(l)) for l in [DATA[idx][p]['probs'] for p in range(0,1+p_max)]])
#     plt.plot(range(0,p_max+1),np.log10(1-mean_data/(idx_dict[prob][1]-idx_dict[prob][0])),label="PSC")
        
#     plt.title(prob)
#     plt.xlabel('p')
#     plt.ylabel('1 - log prob')
#     plt.legend()
#     if(path is not None):
#         plt.savefig(path+".pdf",dpi=300)
#     plt.show()

# def get_PSC_depth_prob_comp(prob,idx_dict,DATA,M_list,A_list,path=None,p_max=5):
#     fig = plt.figure(figsize=(10,10))
    
#     mean_data = np.zeros(p_max+1)
       
#     plt.title(prob)
#     plt.xlabel('p')
#     plt.ylabel('1 - log prob')
#     plt.legend()
#     if(path is not None):
#         plt.savefig(path+".pdf",dpi=300)
#     plt.show()













# def get_depth_cost_comp(prob, idx_dict, PSC_DATA, DATA, M_list, A_list, ws_list=[None, 'GW2', 'GW3'], path=None, p_max=5):
#     fig = plt.figure(figsize=(10,10))
    
#     for ws in ws_list:
#         if ws is None:
#             data = []
#             for idx in range(*idx_dict[prob]):
#                 M = M_list[idx]
#                 A = A_list[idx]
#                 data.append([abs(np.max(l)-M)/(M+np.sum(-A[:-1,:-1])/4) for l in [DATA[idx][p][ws]['cost'] for p in range(0, 1+p_max)]])
#             mean_data = np.mean(data, axis=0)
#             std_dev_data = np.std(data, axis=0)
#             plt.plot(range(0, p_max+1), np.log10(mean_data), marker='o', label=str(ws)+" ")
#             plt.fill_between(range(0, p_max+1), 
#                              np.log10(mean_data - 0.5 * std_dev_data), 
#                              np.log10(mean_data + 0.5 * std_dev_data), 
#                              alpha=0.2)
#         else:
#             for r in [0, -1, None]:
#                 data = []
#                 for idx in range(*idx_dict[prob]):
#                     M = M_list[idx]
#                     A = A_list[idx]
#                     data.append([abs(np.max(l)-M)/(M+np.sum(-A[:-1,:-1])/4) for l in [DATA[idx][p][ws][r]['cost'] for p in range(0, 1+p_max)]])
#                 mean_data = np.mean(data, axis=0)
#                 std_dev_data = np.std(data, axis=0)
#                 plt.plot(range(0, p_max+1), np.log10(mean_data), marker='o', label=str(ws)+" "+str(r))
#                 plt.fill_between(range(0, p_max+1), 
#                                  np.log10(mean_data - 0.5 * std_dev_data), 
#                                  np.log10(mean_data + 0.5 * std_dev_data), 
#                                  alpha=0.2)
    
#     data = []
#     for idx in range(*idx_dict[prob]):
#         M = M_list[idx]
#         A = A_list[idx]
#         data.append([abs(np.max(l)-(M+np.sum(-A[:-1,:-1])/4))/(M+np.sum(-A[:-1,:-1])/4) for l in [PSC_DATA[idx][p]['cost'] for p in range(0, 1+p_max)]])
#     mean_data = np.mean(data, axis=0)
#     std_dev_data = np.std(data, axis=0)
#     plt.plot(range(0, p_max+1), np.log10(mean_data), marker='o', label='PSC')
#     plt.fill_between(range(0, p_max+1), 
#                      np.log10(mean_data - 0.5 * std_dev_data), 
#                      np.log10(mean_data + 0.5 * std_dev_data), 
#                      alpha=0.2)
    
#     plt.title(prob)
#     plt.xlabel('p')
#     plt.ylabel('Log Relative Error')
#     plt.legend()
#     if path is not None:
#         plt.savefig(path+".pdf", dpi=300)
#     plt.show()






















def get_depth_cost_comp(prob,idx_dict,PSC_DATA,DATA,M_list,A_list,ws_list=[None, 'GW2','GW3'],path=None,p_max=5):
    fig = plt.figure(figsize=(10,10))
    for ws in ws_list:
        if(ws is None):
            mean_data = np.zeros(p_max+1)
            for idx in range(*idx_dict[prob]):
                M = M_list[idx]
                A = A_list[idx]
                mean_data+=np.array([abs(np.max(l)-M)/(M+np.sum(-A[:-1,:-1])/4) for l in [DATA[idx][p][ws]['cost'] for p in range(0,1+p_max)]])
            plt.plot(range(0,p_max+1),np.log10(mean_data/(idx_dict[prob][1]-idx_dict[prob][0])),label=str(ws)+" ")
        else:
            for r in [0,-1,None]:
                mean_data = np.zeros(p_max+1)
                for idx in range(*idx_dict[prob]):
                    M = M_list[idx]
                    A = A_list[idx]
                    mean_data+=np.array([abs(np.max(l)-M)/(M+np.sum(-A[:-1,:-1])/4) for l in [DATA[idx][p][ws][r]['cost'] for p in range(0,1+p_max)]])
                plt.plot(range(0,p_max+1),np.log10(mean_data/(idx_dict[prob][1]-idx_dict[prob][0])),label=str(ws)+" "+str(r))
    mean_data = np.zeros(p_max+1)
    for idx in range(*idx_dict[prob]):
        M = M_list[idx]
        A = A_list[idx]
        mean_data+=np.array([abs(np.max(l)-(M+np.sum(-A[:-1,:-1])/4))/(M+np.sum(-A[:-1,:-1])/4) for l in [PSC_DATA[idx][p]['cost'] for p in range(0,1+p_max)]])
    plt.plot(range(0,p_max+1),np.log10(mean_data/(idx_dict[prob][1]-idx_dict[prob][0])), label = 'PSC')
       
    plt.title(prob)
    plt.xlabel('p')
    plt.ylabel('Log Relative Error')
    plt.legend()
    if(path is not None):
        plt.savefig(path+".pdf",dpi=300)
    plt.show()

def get_depth_cost_comp(prob, idx_dict, PSC_DATA, DATA, M_list, A_list, path=None, p_max=5, std=.25):
    fig, axs = plt.subplots(1, 2, figsize=(20, 10), sharey=True)
    
    def plot_graph(ws_list, ax, title_suffix):
        for ws in ws_list:
            if ws is None:
                data = []
                for idx in range(*idx_dict[prob]):
                    M = M_list[idx]
                    A = A_list[idx]
                    data.append([abs(np.max(l) + np.sum(-A[:-1, :-1]) / 4) / (M + np.sum(-A[:-1, :-1]) / 4) 
                                 for l in [DATA[idx][p][ws]['cost'] for p in range(0, 1 + p_max)]])
                mean_data = np.mean(data, axis=0)
                std_dev_data = np.std(data, axis=0)
                ax.plot(range(0, p_max + 1), np.log10(mean_data), marker='o', linestyle='--', label=str(ws) + " ")
                ax.fill_between(range(0, p_max + 1), 
                                np.log10(mean_data - std * std_dev_data), 
                                np.log10(mean_data + std * std_dev_data), 
                                alpha=0.2)
            else:
                for r in [0, -1, None]:
                    data = []
                    for idx in range(*idx_dict[prob]):
                        M = M_list[idx]
                        A = A_list[idx]
                        data.append([abs(np.max(l) + np.sum(-A[:-1, :-1]) / 4) / (M + np.sum(-A[:-1, :-1]) / 4) 
                                     for l in [DATA[idx][p][ws][r]['cost'] for p in range(0, 1 + p_max)]])
                    mean_data = np.mean(data, axis=0)
                    std_dev_data = np.std(data, axis=0)
                    ax.plot(range(0, p_max + 1), np.log10(mean_data), marker='o', label=str(ws) + " " + str(r))
                    ax.fill_between(range(0, p_max + 1), 
                                    np.log10(mean_data - std * std_dev_data), 
                                    np.log10(mean_data + std * std_dev_data), 
                                    alpha=0.2)
        
        data = []
        for idx in range(*idx_dict[prob]):
            M = M_list[idx]
            A = A_list[idx]
            data.append([abs(np.max(l)) / (M + np.sum(-A[:-1, :-1]) / 4) 
                         for l in [PSC_DATA[idx][p]['cost'] for p in range(0, 1 + p_max)]])
        mean_data = np.mean(data, axis=0)
        std_dev_data = np.std(data, axis=0)
        ax.plot(range(0, p_max + 1), np.log10(mean_data), marker='o', label='PSC')
        ax.fill_between(range(0, p_max + 1), 
                        np.log10(mean_data - std * std_dev_data), 
                        np.log10(mean_data + std * std_dev_data), 
                        alpha=0.2)
        
        ax.set_title(title_suffix)
        ax.set_xlabel('p')
        ax.set_ylabel('Log Relative Error')
        ax.legend()

    plot_graph([None, 'GW2'], axs[0], "GW2")
    
    plot_graph([None, 'GW3'], axs[1], "GW3")

    if path is not None:
        plt.savefig(path + "_subplots.pdf", dpi=300)
    plt.show()


def get_depth_prob_comp(prob, idx_dict, PSC_DATA, DATA, M_list, A_list, path=None, p_max=5, std=.25):
    fig, axs = plt.subplots(1, 2, figsize=(20, 10), sharey=True)
    
    def plot_graph(ws_list, ax, title_suffix):
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
                ax.plot(range(0, p_max + 1), np.log10(mean_data), marker='o', linestyle='--', label=str(ws) + " ")
                ax.fill_between(range(0, p_max + 1), 
                                np.log10(mean_data - std * std_dev_data), 
                                np.log10(mean_data + std * std_dev_data), 
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
                    ax.plot(range(0, p_max + 1), np.log10(mean_data), marker='o', label=str(ws) + " " + str(r))
                    ax.fill_between(range(0, p_max + 1), 
                                    np.log10(mean_data - std * std_dev_data), 
                                    np.log10(mean_data + std * std_dev_data), 
                                    alpha=0.2)
        
        data = []
        for idx in range(*idx_dict[prob]):
            M = M_list[idx]
            A = A_list[idx]
            data.append([abs(np.max(l))  
                         for l in [PSC_DATA[idx][p]['probs'] for p in range(0, 1 + p_max)]])
        mean_data = np.mean(data, axis=0)
        std_dev_data = np.std(data, axis=0)
        ax.plot(range(0, p_max + 1), np.log10(mean_data), marker='o', label='PSC')
        ax.fill_between(range(0, p_max + 1), 
                        np.log10(mean_data - std * std_dev_data), 
                        np.log10(mean_data + std * std_dev_data), 
                        alpha=0.2)
        
        ax.set_title(title_suffix)
        ax.set_xlabel('p')
        ax.set_ylabel('Log Relative Error')
        ax.legend()

    plot_graph([None, 'GW2'], axs[0], "GW2")
    
    plot_graph([None, 'GW3'], axs[1], "GW3")

    if path is not None:
        plt.savefig(path + "_subplots.pdf", dpi=300)
    plt.show()