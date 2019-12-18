import numpy as np
import basetree as bt
from pprint import pprint

class PaloBst:

    def __init__(self, 
                distribution="gaussian",
                learning_rate=0.1, 
                subsample=0.7,
                n_estimators=100,
                max_depth=3,
                min_samples_split=2,
                min_samples_leaf=1):
        self.distribution = distribution
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.xmaps = {}

    def fit(self, X, y):

        n, m = X.shape

        # n_nodex_max x [svar, sval, is_leaf, i_start, i_end], [y, z]
        n_nodes_tree = (2**(self.max_depth + 1) - 1)
        n_nodes_trees = n_nodes_tree * self.n_estimators
        trees = np.zeros((n_nodes_trees, 7))

        # integerfy X
        X_ = np.zeros((n, m), dtype=int)
        for i in range(m):
            self.xmaps[i], X_[:,i] = np.unique(X[:,i], return_inverse=True) 

        Y_ = np.ones((n, 4)) # [y, yhat, grad, hess]
        Y_[:,0] = y
        Y_[:,1] = np.mean(y)
        Y_[:,2] = Y_[:,0] - Y_[:,1]
        if self.distribution == "bernoulli":
            p = 1/(1+np.exp(-Y_[:,0]))
            Y_[:,3] = p * (1-p)
        
        t_nodes = 2**self.max_depth - 1
        t_nodes_all = t_nodes * self.n_estimators
        t_rules = np.zeros((t_nodes_all, 4), dtype=int)
        t_vals = np.zeros((t_nodes_all, 2))
        t_idx = np.zeros((t_nodes, 4), dtype=int)

        for i in range(self.n_estimators):
            ridx = np.random.permutation(n)
            X_, Y_ = X_[ridx], Y_[ridx]
            bt.grow(X_, Y_,
                    t_rules[(t_nodes*i):(t_nodes*(i+1)),:],
                    t_vals[(t_nodes*i):(t_nodes*(i+1)),:],
                    t_idx,
                    self.subsample,
                    self.max_depth,
                    self.min_samples_split,
                    self.min_samples_leaf)

    def predict(self, X):

        pass


if __name__=="__main__":

    np.seterr(all="raise")
    from sklearn.datasets import make_hastie_10_2
    X, y = make_hastie_10_2(n_samples=10000)
    y[y<0] = 0
    palobst = PaloBst(n_estimators=5, max_depth=5)
    palobst.fit(X, y)





