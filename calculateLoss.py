import numpy as np
def mes_loss(ytrue, ypred):
    # 两个参数都是numpy数组，方便向量运算
    return ((ytrue - ypred) ** 2).mean()
ytrue = np.array([1,0,0,1])
ypred = np.array([0,0,0,0])
print(mes_loss(ytrue, ypred))