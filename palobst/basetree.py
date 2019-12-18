from numba import jit
import numpy as np

@jit(nopython=True)
def grow(X, Y, 
        t_rules,
        t_vals,
        t_idx,
        subsample,
        max_depth,
        min_samples_split, 
        min_samples_leaf,
        task="cls"):

    n, m = X.shape
    t_idx[:,:] = 0

    # split the root node
    n_ib = int(n * subsample)
    n_oob = n - n_ib
    X_ib = X[:n_ib,:]
    X_oob = X[n_ib:,:]
    Y_ib = Y[:n_ib,:]
    Y_oob = Y[n_ib:,:]
    svar, sval = split(X_ib, Y_ib, min_samples_leaf)
    le_ib = reorder(X_ib, Y_ib, 0, n_ib, svar, sval)
    le_oob = reorder(X_oob, Y_oob, 0, n_oob, svar, sval)

    t_rules[0,0], t_rules[0,1] = svar, sval
    t_vals[0,0] = np.mean(Y_ib[:,2])
    t_vals[0,1] = np.mean(Y_oob[:,2])
    t_idx[1,1], t_idx[1,3] = le_ib, le_oob # left node
    t_idx[2,0], t_idx[2,2] = le_ib, le_oob # right node
    t_idx[2,1], t_idx[2,3] = n_ib, n_oob

    for depth in range(1, max_depth+1):
        offset = 2**(depth-1) - 1
        for nid in range(offset, 2*offset+1):
            s_ib, e_ib = t_idx[nid,0], t_idx[nid,1]
            s_oob, e_oob = t_idx[nid,2], t_idx[nid,3]

            if e_ib == s_ib or e_oob == s_oob:
                continue

            t_vals[nid,0] = np.mean(Y_ib[s_ib:e_ib,2])
            t_vals[nid,1] = np.mean(Y_oob[s_oob:e_oob,2])

            if (e_ib - s_ib < min_samples_leaf or
                depth == max_depth):
                t_rules[nid,0] = -1
                continue

            svar, sval = split(X_ib[s_ib:e_ib,:], Y_ib[s_ib:e_ib,:], 
                                min_samples_leaf)
            le_ib = reorder(X_ib, Y_ib, s_ib, e_ib, svar, sval)
            le_oob = reorder(X_oob, Y_oob, s_ib, e_oob, svar, sval)

            t_rules[nid,0], t_rules[nid,1] = svar, sval
            t_vals[nid,0] = np.mean(Y_ib[s_ib:e_ib,2])
            t_vals[nid,1] = np.mean(Y_oob[s_oob:e_oob,2])
            t_idx[nid*2+1,1], t_idx[nid*2+1,3] = le_ib, le_oob # left node
            t_idx[nid*2+2,0], t_idx[nid*2+2,2] = le_ib, le_oob # right node
            t_idx[nid*2+2,1], t_idx[nid*2+2,3] = e_ib, e_oob


@jit(nopython=True)
def split(X, Y, min_samples_leaf):

    loss_min = 1e10
    svar_best = -1
    sval_best = -1

    for svar in range(X.shape[1]):
        sval, loss = get_sval(X[:,svar], Y[:,2], Y[:,3], min_samples_leaf)
        if sval < 0:
            continue
        if loss < loss_min:
            loss_min = loss
            svar_best = svar
            sval_best = sval

    return svar_best, sval_best

@jit(nopython=True)
def get_sval(x, grad, hess, min_samples_leaf):

    reg_lambda = 1.0
   
    n = x.shape[0]
    idx = np.argsort(x)
    x = x[idx]
    grad = grad[idx]
    hess = hess[idx]
    G = np.sum(grad)
    H = np.sum(hess)

    g = 0
    h = 0
    x_prev = -1
    x_best = -1
    loss_min = 1e10
    for i in range(n):
        g = grad[i] + g
        h = hess[i] + h
        loss = g**2 / (h + reg_lambda)
        loss += (G - g)**2 / (H - h + reg_lambda)
        if x_prev != x[i]:
            if (loss < loss_min and
                i > min_samples_leaf and
                i < n - min_samples_leaf):
                x_best = x[i]
                loss_min = loss
            x_prev = x[i]

    return x_best, loss_min

@jit(nopython=True)
def reorder(X, Y, s, e, svar, sval):
    X_ = X[s:e,:]
    Y_ = Y[s:e,:]
    ln = X_[:,svar] < sval
    idx = np.argsort(~ln)
    X[s:e,:] = X_[idx,:]
    Y[s:e,:] = Y_[idx,:]
    return np.sum(ln)


