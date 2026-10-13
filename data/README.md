# Dataset folder

Place the PhiUSIIL Phishing URL Dataset CSV here as `dataset.csv`.

Download from the UCI Machine Learning Repository:
https://archive.ics.uci.edu/dataset/967

or via Python:
```python
pip install ucimlrepo
from ucimlrepo import fetch_ucirepo
data = fetch_ucirepo(id=967)
```

(This file is not included in the delivered project package because it is
~55 MB — you already have it from your own upload.)
