import numpy as np

def unitStep(v):
    if v>=0:
        return 1
    else:
        return 0
    
def perceptronModel(x, w, b):
    v = np.dot(w, x) + b
    y = unitStep(v)
    return y

def AND_Logic_Function(x):
    w = np.array([1,1])
    b = -1.5
    return perceptronModel(x, w, b)

test1 = np.array([0,1])
test2 = np.array([1,1])
test3 = np.array([0,0])
test4 = np.array([1,0])

print("___________________________AND Function__________________________")

print("AND({}, {}) = {}".format(0, 1, AND_Logic_Function(test1)))
print("AND({}, {}) = {}".format(1, 1, AND_Logic_Function(test2)))
print("AND({}, {}) = {}".format(0, 0, AND_Logic_Function(test3)))
print("AND({}, {}) = {}".format(1, 0, AND_Logic_Function(test4)))