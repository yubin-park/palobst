from palobst import PaloBst
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import roc_auc_score
from sklearn.datasets import make_hastie_10_2
import numpy as np
import unittest
import time

class PaloBstTestCase(unittest.TestCase):

    def test_cls(self):

        np.random.seed(1)
        n_samples = 10000
        test_size = 0.2
        n_est = 100
        max_depth = 10
        lr = 1.0

        X, y = make_hastie_10_2(n_samples)
        poly = PolynomialFeatures()
        X = poly.fit_transform(X)
        y[y<0] = 0
        X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                test_size=test_size)

        model_palo = PaloBst(distribution="bernoulli",
                                n_estimators=n_est, 
                                learning_rate=lr,
                                max_depth=max_depth)
        model_sklr = GradientBoostingClassifier(
                                n_estimators=n_est, 
                                learning_rate=lr,
                                max_depth=max_depth)

        t_start = time.time()
        model_palo.fit(X_train, y_train)
        t_elapsed = time.time() - t_start
        y_hat = model_palo.predict_proba(X_test)[:,1]
        auc_palo = roc_auc_score(y_test, y_hat)
        print(t_elapsed)

        t_start = time.time()
        model_sklr.fit(X_train, y_train)
        t_elapsed = time.time() - t_start
        y_hat = model_sklr.predict_proba(X_test)[:,1]
        auc_sklr = roc_auc_score(y_test, y_hat)

        print(t_elapsed)
        print(auc_palo, auc_sklr)
        #self.assertTrue(auc_palo > auc_sklr)

if __name__=="__main__":

    unittest.main()

