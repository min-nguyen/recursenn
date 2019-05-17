import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize
import math


y = np.array([100*float(line.rstrip('\n')) for line in open("recursenn_output.txt")])
x = np.arange(0, len(y))

def test_func(x, a, b):
    return a * np.exp(b * x )

def test_func_2(x, a, b):
    return a * np.sin(b * x)

def func(x, b, c, d): 
    return (b/(1.0+np.exp(-(c+d*x))))

def ff(L, v, k):
    return L**(-1/v) * k

params, params_covariance = optimize.curve_fit(test_func, x, y)

print(params)
plt.ylim(bottom=0.0000, top=max(y))
plt.xlabel("Samples Trained")
plt.ylabel("Error")
plt.xlim(left=0, right = len(y))
plt.scatter(x, y, s=1, color='b')
# plt.plot(x, ff(x, *params), "b", label = "fit")
# plt.plot(x, test_func(x, params[0], params[1]), 'r')
plt.show()