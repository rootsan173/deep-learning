import numpy as np
import matplotlib.pylab as plt

#ステップ関数
def step1(x):
  if x > 0:
    return 1
  else:
    return 0

#ステップ関数(配列入力用)
def step2(x):
  y = x > 0
  return y.astype(np.int)

#シグモイド関数
def sigmoid(x):
  return 1 / (1 + np.exp(-x))

#ReLU関数
def relu(x):
  return np.maximum(0, x)

#恒等関数
def identity(x):
  return x

#ソフトマックス関数
def softmax(a):
  c = np.max(a)
  exp_a = np.exp(a - c) #オーバーフロー対策
  sum_exp_a = np.sum(exp_a)
  y = exp_a / sum_exp_a
  return y


#グラフ表示
x = np.arange(-5.0, 5.0, 0.1)
y = relu(x)
fig = plt.figure()
plt.plot(x, y)
plt.ylim(-1, 5)
fig.savefig("relu_function.png")
