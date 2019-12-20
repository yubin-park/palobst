import numpy as np
from palobst import basetree as bt
from scipy.special import expit

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
        self.intercept = 0
        self.t_svar = None
        self.t_sval = None
        self.t_pred = None

    def warmup(self):
        n = max(self.min_samples_split, self.min_samples_leaf)
        n = max(int(n), 3) * 10
        X = np.random.rand(n, 2)
        y = np.random.rand(n)
        if self.distribution == "bernoulli":
            y = y > 0.5
        self.fit(X, y)
        self.predict(X)

    def fit(self, X, y):

        n, m = X.shape
        n_nodes_tree = (2**(self.max_depth + 1) - 1)
        n_nodes_trees = n_nodes_tree * self.n_estimators
        trees = np.zeros((n_nodes_trees, 7))

        # integerfy X
        X_ = np.zeros((n, m), dtype=int)
        xmaps = {}
        xmaps[-1] = [0]
        for i in range(m):
            xmaps[i], X_[:,i] = np.unique(X[:,i], return_inverse=True) 

        y_avg = np.mean(y) 
        Y_ = np.ones((n, 4)) # [y, yhat, gradient, Hessian]
        Y_[:,0] = y
        if self.distribution == "bernoulli":
            mu = np.log(y_avg/(1-y_avg))
            p = expit(mu)
            self.intercept = mu
            Y_[:,1] = self.intercept
            Y_[:,2] = Y_[:,0] - p # gradient
            Y_[:,3] = p * (1-p)   # Hessian
        else:
            self.intercept = y_avg
            Y_[:,1] = self.intercept
            Y_[:,2] = Y_[:,0] - Y_[:,1] # gradient, Hessian is 1

        t_nodes = 2**self.max_depth - 1
        t_nodes_all = t_nodes * self.n_estimators
        t_rules = np.zeros((t_nodes_all, 2), dtype=int)
        t_rules[:,0] = -1
        t_vals = np.zeros((t_nodes_all, 2))

        cache_t = np.zeros((t_nodes, 4), dtype=int) 
        cache_g = np.zeros((n,m)) # Gradient
        cache_h = np.zeros((n,m)) # Hessian
        cache_n = np.zeros((n,m), dtype=int)
        cache_l = np.zeros((m,2)) # Loss
        cache_f = np.zeros(m) # feature importances

        for i in range(self.n_estimators):
            ridx = np.random.permutation(n)
            X_, Y_ = X_[ridx], Y_[ridx]
            bt.grow(X_, Y_,
                    t_rules[(t_nodes*i):(t_nodes*(i+1)),:],
                    t_vals[(t_nodes*i):(t_nodes*(i+1)),:],
                    self.distribution,
                    self.subsample,
                    self.learning_rate,
                    self.max_depth,
                    self.min_samples_split,
                    self.min_samples_leaf, 
                    cache_t,
                    cache_g,
                    cache_h,
                    cache_n,
                    cache_l)

            bt.update_fi(t_rules[(t_nodes*i):(t_nodes*(i+1)),:],
                        t_vals[(t_nodes*i):(t_nodes*(i+1)),:],
                        cache_t,
                        cache_f)

            if self.distribution == "bernoulli":
                p = expit(Y_[:,1])
                Y_[:,2] = Y_[:,0] - p # gradient
                Y_[:,3] = p * (1-p)   # Hessian
            elif self.distribution == "gaussian":
                Y_[:,2] = Y_[:,0] - Y_[:,1] # gradient



        # re-map X values
        self.t_svar = t_rules[:,0]
        self.t_sval = np.array([xmaps[t_rules[i,0]][t_rules[i,1]] 
                        for i in range(t_nodes_all)])
        self.t_vals = t_vals[:,0] 
        self.feature_importances_ = cache_f/self.n_estimators


    def predict_proba(self, X):
        y = np.full(X.shape[0], self.intercept)
        y = bt.apply_tree(X, y, self.t_svar, self.t_sval, self.t_vals,
                        self.n_estimators,
                        2**self.max_depth -1)
        if self.distribution == "bernoulli":
            y_mat = np.zeros((X.shape[0], 2))
            y_mat[:,1] = expit(y)
            y_mat[:,0] = 1 - y_mat[:,1]
            return y_mat
        else:
            return y
    
    def predict(self, X):
        return self.predict_proba(X)

    def get_feature_importances(self):
        return self.feature_importances_


    



