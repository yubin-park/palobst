# PaloBoost

PaloBoost (`palobst`) is an over-fitting robust Gradient Boosting Decision Tree algorithm.

To use the package, you need `numba >= 0.46`.

```python
from palobst import PaloBst

model = PaloBst()
model.warmup() # this runs JIT 

model.fit(X, y, distribution="bernoulli")
model.predict_proba(X)

model.fit(X, y, distribution="gaussian")
mode.predict(X)
```


## Reference
- Y.Park, J. Ho. Tackling Overfitting in Boosting for Noisy Healthcare Data. IEEE Transactions on Knowledge and Data Engineering. https://ieeexplore.ieee.org/document/8933485



