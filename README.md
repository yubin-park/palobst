# PaloBst

PaloBst is an over-fitting robust Gradient Boosting Decision Tree algorithm.
The details of the algorithm are illustrated in ["Tackling Overfitting in Boosting for Noisy Healthcare Data, IEEE TKDE"](https://ieeexplore.ieee.org/document/8933485).

To use the package, you need to install [`numba >= 0.46`](http://numba.pydata.org/).

To install the package, clone the repository, and run:
```
# cd palobst
$ python setup.py develop
```

##  Regression Example

```python
from sklearn.datasets import make_friedman1
from sklearn.model_selection import train_test_split

X, y = make_friedman1(100)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

from palobst import PaloBst

model = PaloBst(distribution="gaussian")
model.warmup() # this runs JIT 

model.fit(X_train, y_train)

y_pred = mode.predict(X_test)
```
Please see `test/test_regression.py` for more details.

## Classification Example

```python
from sklearn.datasets import make_hastie_10_2
from sklearn.model_selection import train_test_split

X, y = make_hastie_10_2(100)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

from palobst import PaloBst

model = PaloBst(distribution="bernoulli")
model.warmup() # this runs JIT 

model.fit(X_train, y_train)

y_pred = mode.predict_proba(X)[:,1]
```

Please see `test/test_classification.py` for more details.

## Reference
- Yubin Park and Joyce C. Ho. Tackling Overfitting in Boosting for Noisy Healthcare Data. IEEE Transactions on Knowledge and Data Engineering. https://ieeexplore.ieee.org/document/8933485



