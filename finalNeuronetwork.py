import numpy as np

def sigmoid(x):
    # activation function
    return 1 / (1 + np.exp(-x))
def deriv_sigmoid(x):
    # 求activation function的导数
    fx = sigmoid(x)
    return fx * (1 - fx)
def mes_loss(ytrue, ypred):
    # 两个参数都是numpy数组，方便向量运算
    return ((ytrue - ypred) ** 2).mean()

