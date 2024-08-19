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
    init = reduce(lambda a,b: np.kron(a,b), [np.array([np.cos(v/2), np.sin(v/2)],dtype='complex128') for v in angles])
    mixer_ops = lambda t: [SU2_op(np.sin(v),0,np.cos(v),t) for v in angles]
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

def PSC_Run(A,opt,p_max,optimizer_kwargs={'name':None,'verbose':True}.keep_hist=False,eps=0.1):
    ###initialization
    Q = -A[:-1,:-1]
    precomp = PSC_pre_compute(Q)

    vals = relax_solve(Q)
    init,mixer_ops=PSC_data(vals,eps=eps)

    opt_data =[{} for _ in range(p_max+1)]


    opt_params=[None]

    ###Depth 0
    opt_data[0]['cost']=expval(precomp,QAOA_eval(precomp,[],mixer_ops=mixer_ops,init=init))
    opt_data[0]['params']=[]
    opt_data[0]['probs']=opt_sampling_prob(v,precomp,[],mixer_ops=mixer_ops,init=init)

    for p in range(1,p_max+1):
        guess  = opt_params + [None] * (reps-1)
        l=circuit_optimization_eff(precomp,A,opt,mixer_ops,init,p,reps=reps,param_guesses=guess,**optimizer_kwargs)
        opt_data[p]['cost']=l[0]
        opt_data[p]['params']=l[1]
        opt_data[p]['probs']=np.array([opt_sampling_prob(v,precomp,param,mixer_ops=None,init=None) for param in l[1]])
        opt_params = list(l[1][np.argmax(l[0])])[:p] + [0] + list(l[1][np.argmax(l[0])])[p:] + [0]

        if(keep_hist):
            opt_data[p]['hist']=l[2]
    
    return opt_data
