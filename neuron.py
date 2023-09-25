import numpy as np
def sigmoid(x):
    # activation function,used to make putput in a specific range
    # 这一点详情看网站对于sigmoid函数的介绍
    return 1 / (1 + np.exp(-x))
class Neuron:
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias
    def feedforward(self, input):
        # 使用点乘，加上偏移量，最后调用activation函数并返回结果
        total = np.dot(self.weights, input) + self.bias
        return sigmoid(total)
# 注意这个语法，numpy可以使用向量进行计算，避免写循环
weights = np.array([0,1])
bias = 4
# 不需要传递self
n = Neuron(weights, bias)
inputs = np.array([2,3])
print(n.feedforward(inputs))

# 下面加入多个neuron，形成不同layer，但是原理都是一样的
class ourNerualNetwork:
    '''
  A neural network with:
    - 2 inputs
    - a hidden layer with 2 neurons (h1, h2)
    - an output layer with 1 neuron (o1)
  Each neuron has the same weights and bias:
    - w = [0, 1]
    - b = 0
  '''
    def __init__(self) -> None:
        weights = np.array([0,1])
        bias = 0
        # hidden layer
        self.h1 = Neuron(weights, bias)
        self.h2 = Neuron(weights, bias)
        # output
        self.o1 = Neuron(weights, bias)
    def feedForward(self, inputs):
        out_h1 = self.h1.feedforward(input=inputs)
        out_h2 = self.h2.feedforward(input=inputs)
        # inputs of o1 are from h1 and h2 's outputs
        return self.o1.feedforward(np.array([out_h1, out_h2]))
network = ourNerualNetwork()
inputs = np.array([2,3])
print(network.feedForward(inputs))


