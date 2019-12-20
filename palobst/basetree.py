import numpy as np
from numba import njit, prange

@njit(fastmath=True, parallel=True)
def grow(X, Y, 
        t_rules,
        t_vals,
        distribution,
        subsample,
        nu, # learning_rate
        max_depth,
        min_samples_split, 
        min_samples_leaf,
        cache_t,
        cache_g,
        cache_h,
        cache_n,
        cache_l):

    n, m = X.shape
    cache_t[:,:] = 0

    # split the root node
    n_ib = int(n * subsample)
    n_oob = n - n_ib
    X_ib, X_oob = X[:n_ib,:], X[n_ib:,:]
    Y_ib, Y_oob = Y[:n_ib,:], Y[n_ib:,:]
    svar, sval = split(X_ib, Y_ib, min_samples_leaf, 
                        cache_g, cache_h, cache_n, cache_l)
    c_ib = reorder(X_ib, Y_ib, 0, n_ib, svar, sval)
    c_oob = reorder(X_oob, Y_oob, 0, n_oob, svar, sval)

    t_rules[0,0], t_rules[0,1] = svar, sval
    t_vals[0,0], t_vals[0,1] = np.mean(Y_ib[:,2]), np.mean(Y_oob[:,2])
    cache_t[0,1], cache_t[0,3] = n_ib, n_oob
    cache_t[1,1], cache_t[1,3] = c_ib, c_oob # end of left node
    cache_t[2,0], cache_t[2,2] = c_ib, c_oob # start of right node
    cache_t[2,1], cache_t[2,3] = n_ib, n_oob # end of right node

    for depth in range(1, max_depth+1):
        offset = 2**(depth-1) - 1
        for nid in prange(offset, 2*offset+1):
            s_ib, e_ib = cache_t[nid,0], cache_t[nid,1]
            s_oob, e_oob = cache_t[nid,2], cache_t[nid,3]
            if e_ib == s_ib or e_oob == s_oob:
                continue

            t_vals[nid,0] = np.mean(Y_ib[s_ib:e_ib,2])
            t_vals[nid,1] = np.mean(Y_oob[s_oob:e_oob,2])

            if (e_ib - s_ib < min_samples_leaf or
                depth == max_depth):
                t_rules[nid,0], t_rules[nid,1] = -1, 0
                continue

            svar, sval = split(X_ib[s_ib:e_ib,:], Y_ib[s_ib:e_ib,:], 
                                min_samples_leaf, 
                                cache_g, cache_h, cache_n, cache_l)
            c_ib = reorder(X_ib, Y_ib, s_ib, e_ib, svar, sval)
            c_oob = reorder(X_oob, Y_oob, s_oob, e_oob, svar, sval)


            # NOTE: GAP Pre-pruning
            if (s_oob == c_oob or
                c_oob == e_oob or
                check_GAP(Y_oob, s_oob, c_oob, nu, distribution) or
                check_GAP(Y_oob, c_oob, e_oob, nu, distribution)):
                t_rules[nid,0], t_rules[nid,1] = -1, 0
                continue

            t_rules[nid,0], t_rules[nid,1] = svar, sval
            t_vals[nid,0] = np.mean(Y_ib[s_ib:e_ib,2])
            t_vals[nid,1] = np.mean(Y_oob[s_oob:e_oob,2])
            cache_t[nid*2+1,0], cache_t[nid*2+1,2] = s_ib, s_oob 
            cache_t[nid*2+1,1], cache_t[nid*2+1,3] = c_ib, c_oob 
            cache_t[nid*2+2,0], cache_t[nid*2+2,2] = c_ib, c_oob 
            cache_t[nid*2+2,1], cache_t[nid*2+2,3] = e_ib, e_oob 

    # NOTE: Adaptive Learning Rate (ALR)
    for nid in prange(t_rules.shape[0]):
        if t_rules[nid,0] > -1 or cache_t[nid,0] == cache_t[nid,1]:
            # Not a leaf node
            continue
        s_ib, e_ib = cache_t[nid,0], cache_t[nid,1]
        s_oob, e_oob = cache_t[nid,2], cache_t[nid,3]
        gamma = t_vals[nid,0]
        if gamma == 0:
            continue
        if distribution == "bernoulli":
            num = np.sum(Y_oob[s_oob:e_oob,0]) + 0.5
            denom = (np.sum((1 - Y_oob[s_oob:e_oob,0]) * 
                        np.exp(Y_oob[s_oob:e_oob,1])) + 1)
            nu_adj = np.log(num/denom)/gamma
        else:
            nu_adj = (np.mean((Y_oob[s_oob:e_oob,0] - 
                        Y_oob[s_oob:e_oob,1]))/gamma)
        if nu_adj < 0:
            nu_adj = 0
        elif nu_adj > nu:
            nu_adj = nu
        gamma = nu_adj * gamma
        t_vals[nid,0] = gamma
        Y_ib[s_ib:e_ib,1] += gamma
        Y_oob[s_oob:e_oob,1] += gamma


@njit(fastmath=True, parallel=True)
def check_GAP(Y, s, e, nu, distribution):
    y = Y[s:e,0]
    y_hat_old = Y[s:e,1] 
    y_hat_new = y_hat_old + nu * np.mean(Y[s:e,2])
    if loss(y, y_hat_old, distribution) < loss(y, y_hat_new, distribution):
        return True
    else:
        return False

@njit(fastmath=True, parallel=True)
def split(X, Y, min_samples_leaf, cache_g, cache_h, cache_n, cache_l):

    tol = 1e10
    svar_best = -1
    sval_best = 0
    loss_min = tol
    cache_g[:,:] = 0
    cache_h[:,:] = 0
    cache_n[:,:] = 0
    cache_l[:,0] = tol

    for svar in prange(X.shape[1]):
        sval, loss = get_sval(X[:,svar], Y[:,2], Y[:,3], min_samples_leaf,
                            cache_g[:,svar], 
                            cache_h[:,svar], 
                            cache_n[:,svar])
        if sval < 0:
            continue
        cache_l[svar,0] = loss
        cache_l[svar,1] = sval

    svar = np.argmin(cache_l[:,0])
    if cache_l[svar,0] < tol:
        loss_min = cache_l[svar,0]
        svar_best = svar
        sval_best = cache_l[svar,1]

    return svar_best, sval_best

@njit(fastmath=True, parallel=True)
def get_sval(x, grad, hess, min_samples_leaf, cache_g, cache_h, cache_n):

    reg_lambda = 1.0

    n = x.shape[0]
    G = 0
    H = 0
    g_i = 0
    h_i = 0
    n_i = 0
    x_best = -1
    loss_min = np.inf

    for i in prange(n):
        G += grad[i]
        H += hess[i]
        cache_g[x[i]] += grad[i]
        cache_h[x[i]] += hess[i]
        cache_n[x[i]] += 1
  
    for i in range(n-1):
        if cache_n[i] == 0: # TODO: can be optimized further
            continue
        g_i += cache_g[i]
        h_i += cache_h[i]
        n_i += cache_n[i]
        if (n_i < min_samples_leaf or
            n_i > n - min_samples_leaf):
            continue
        loss = -g_i**2 / (h_i + reg_lambda)
        loss = loss - (G - g_i)**2 / (H - h_i + reg_lambda)
        if loss < loss_min:
            loss_min = loss
            x_best = i
       
    return x_best, loss_min


@njit(fastmath=True, parallel=True)
def reorder(X, Y, s, e, svar, sval):
    X_ = X[s:e,:]
    Y_ = Y[s:e,:]
    ln = X_[:,svar] < sval
    idx = np.argsort(~ln)
    X[s:e,:] = X_[idx,:]
    Y[s:e,:] = Y_[idx,:]
    return s + np.sum(ln)

@njit(fastmath=True, parallel=True)
def loss(y, y_hat, distribution):
    if distribution == "bernoulli":
        return np.mean(-2.0*(y*y_hat - np.logaddexp(0.0, y_hat)))
    else:
        return np.mean((y-y_hat)**2)
    

@njit(fastmath=True, parallel=True)
def apply_tree(X, y, t_svar, t_sval, t_vals, n_estimators, t_nodes):
    for i in prange(X.shape[0]):
        for j in prange(n_estimators):
            y[i] += apply_tree0(X[i,:], 
                        t_svar[(t_nodes*j):(t_nodes*(j+1))],
                        t_sval[(t_nodes*j):(t_nodes*(j+1))],
                        t_vals[(t_nodes*j):(t_nodes*(j+1))])
    return y 

@njit(fastmath=True)
def apply_tree0(x, t_svar, t_sval, t_vals):
    nid = 0
    while t_svar[nid] > -1:
        if x[t_svar[nid]] < t_sval[nid]:
            nid = nid * 2 + 1
        else:
            nid = nid * 2 + 2
    return t_vals[nid]

@njit(fastmath=True)
def update_fi(t_rules, t_vals, cache_t, cache_f):

    n = cache_t.shape[0]
    m = cache_f.shape[0]
    score = 0
    norm_factor = cache_t[0,1] - cache_t[0,0]

    for nid in range(n):
        if (t_rules[nid,0] > -1 or 
            cache_t[nid,0] == cache_t[nid,1]):
            # Not a leaf node
            continue
        score = np.abs(t_vals[nid,0]) 
        score = score * (cache_t[nid,1] - cache_t[nid,0])
        score = score / norm_factor
        nid_tmp = nid
        while nid_tmp > 0:
            if nid_tmp % 2 == 1:
                nid_tmp = int((nid_tmp - 1)/2)
                cache_f[t_rules[nid_tmp,0]] += score
            else:
                nid_tmp = int((nid_tmp - 2)/2)
                cache_f[t_rules[nid_tmp,0]] += score




