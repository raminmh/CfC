

import unittest
from CFC.data_types import GenericData
import CFC.dense_model as dense_model
import numpy as np
import math
import random
import matplotlib.pyplot as plt
import seaborn as sns

X = np.array([[ii*math.sin((ii + i)/ 10) + (random.random()-0.5)*20 for i in range(10)] for ii in range(100)])
y = np.array([int(math.cos((ii + 11)/ 10) >= 0) for ii in range(100)])

ax=sns.lineplot(x=range(len(X[0])), y=X[0])
for i in range(100):
    ax = sns.lineplot(x=range(len(X[i])), y=X[i], ax=ax)
plt.show()