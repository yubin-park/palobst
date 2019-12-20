from palobst import PaloBst
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.datasets import make_friedman1
import numpy as np
import unittest
import time

class PaloBstTestCase(unittest.TestCase):

    def test_simple(self):

        X = np.random.rand(1000, 2)
        y = X[:,0] + X[:,1]
        X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                test_size=0.2)

        model_palo = PaloBst(distribution="gaussian",
                                n_estimators=10, 
                                learning_rate=1.0,
                                max_depth=3)
        model_palo.warmup()
        model_palo.fit(X_train, y_train)
        y_hat = model_palo.predict(X_test)
        mse_palo = np.mean((y_test - y_hat)**2)
        mse_base = np.mean((y_test - np.mean(y_train))**2)

        self.assertTrue(mse_palo < mse_base)

    def test_friedman(self):

        np.random.seed(1)
        n_samples = 1000
        test_size = 0.2
        n_est = 200
        max_depth = 10
        lr = 1.0

        X, y = make_friedman1(n_samples)
        poly = PolynomialFeatures()
        X = poly.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                test_size=test_size)

        model_palo = PaloBst(distribution="gaussian",
                                n_estimators=n_est, 
                                learning_rate=lr,
                                max_depth=max_depth)
        model_sklr = GradientBoostingRegressor(
                                n_estimators=n_est, 
                                learning_rate=lr,
                                max_depth=max_depth)
        model_palo.warmup()

        t_start = time.time()
        model_palo.fit(X_train, y_train)
        t_elapsed_palo = time.time() - t_start
        y_hat = model_palo.predict(X_test)
        mse_palo = np.mean((y_test - y_hat)**2)

        t_start = time.time()
        model_sklr.fit(X_train, y_train)
        t_elapsed_sklr = time.time() - t_start
        y_hat = model_sklr.predict(X_test)
        mse_sklr = np.mean((y_test - y_hat)**2)

        print(f"Runtime(PaloBst): {t_elapsed_palo:.3f} seconds")
        print(f"Runtime(sklearn): {t_elapsed_sklr:.3f} seconds")
        print(f"MSE(PaloBst): {mse_palo:.3f}")
        print(f"MSE(sklearn): {mse_sklr:.3f}")

if __name__=="__main__":

    unittest.main()

