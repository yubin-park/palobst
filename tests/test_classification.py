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

    def test_simple(self):

        X = np.random.rand(1000, 2)
        y = np.array((X[:,0] + X[:,1]) > 1.0, dtype=int)
        X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                test_size=0.2)

        model_palo = PaloBst(distribution="bernoulli",
                                n_estimators=10, 
                                learning_rate=1.0,
                                max_depth=3)
        model_palo.warmup()
        model_palo.fit(X_train, y_train)
        y_hat = model_palo.predict_proba(X_test)[:,1]
        auc_palo = roc_auc_score(y_test, y_hat)
        self.assertTrue(auc_palo > 0.5)

    def test_hastie(self):

        np.random.seed(1)
        n_samples = 200
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
        model_palo.warmup()

        t_start = time.time()
        model_palo.fit(X_train, y_train)
        t_elapsed_palo = time.time() - t_start
        y_hat = model_palo.predict_proba(X_test)[:,1]
        auc_palo = roc_auc_score(y_test, y_hat)

        t_start = time.time()
        model_sklr.fit(X_train, y_train)
        t_elapsed_sklr = time.time() - t_start
        y_hat = model_sklr.predict_proba(X_test)[:,1]
        auc_sklr = roc_auc_score(y_test, y_hat)

        print(f"Runtime(PaloBst): {t_elapsed_palo:.3f} seconds")
        print(f"Runtime(sklearn): {t_elapsed_sklr:.3f} seconds")
        print(f"AUROC(PaloBst): {auc_palo:.3f}")
        print(f"AUROC(sklearn): {auc_sklr:.3f}")


if __name__=="__main__":

    unittest.main()

