from palobst import PaloBst
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.datasets import make_friedman1
import numpy as np
import unittest

class PaloBstTestCase(unittest.TestCase):

    def test_friedman(self):

        np.random.seed(1)
        n_samples = 1000
        n_est = 100
        max_depth = 5
        lr = 1.0

        X, y = make_friedman1(n_samples)

        model = PaloBst(distribution="gaussian",
                                n_estimators=n_est, 
                                learning_rate=lr,
                                max_depth=max_depth)
        model.warmup()
        model.fit(X, y)

        print(model.feature_importances_)


if __name__=="__main__":

    unittest.main()

