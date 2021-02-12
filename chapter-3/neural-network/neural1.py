import numpy as np
import activation

#入力
X = np.array([1.0, 0.5])
#一層の重み
W1 = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
#一層のバイアス
B1 = np.array([0.1, 0.2, 0.3])
#一層のの重み付き和
A1 = np.dot(X, W1) + B1
#活性化関数で変換
Z1 = activation.sigmoid(A1)

#二層の重み
W2 = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
#二層のバイアス
B2 = np.array([0.1, 0.2])
#二層の重み付き和
A2 = np.dot(Z1, W2) + B2
#活性化関数で変換
Z2 = activation.sigmoid(A2)

#三層の重み
W3 = np.array([[0.1, 0.3], [0.2, 0.4]])
#三層のバイアス
B3 = np.array([0.1, 0.2])
#三層の重み付き和
A3 = np.dot(Z2, W3) + B3
#活性化関数で変換
Y = activation.identity(A3)

print(Y)
