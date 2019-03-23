import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.datasets import load_boston
boston = load_boston()

data = boston.data
target = boston.target
target = target.reshape(-1,1)
print(target.shape)

X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.33, random_state=12)


# データの正規化
ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)


# linear regression model
class LinearRegression(nn.Module):
    def __init__(self, input_size, output_size):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        out = self.linear(x)
        return out

model = LinearRegression(input_size=X_train.shape[1], output_size=1)


# loss and optimizer
criterion = nn.MSELoss()

learning_rate = 0.001
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)


def train(X_train, y_train):
    inputs = torch.from_numpy(X_train).float()
    targets = torch.from_numpy(y_train).float()

    optimizer.zero_grad()  # 勾配の初期化
    outputs = model(inputs)  # forward計算

    loss = criterion(outputs, targets)  # loss計算
    loss.backward()  # 勾配の計算
    optimizer.step()  # パラメータ更新

    return loss.item()  # .item()で0次元テンソルから、値を取り出す

def valid(X_test, y_test):
    inputs = torch.from_numpy(X_test).float()
    targets = torch.from_numpy(y_test).float()

    outputs = model(inputs)
    val_loss = criterion(outputs, targets)

    return val_loss.item()


# train the model
loss_list = []
val_loss_list = []
num_epochs = 5000
for epoch in range(num_epochs):
    # data shuffle
    perm = np.arange(X_train.shape[0])
    np.random.shuffle(perm)
    X_train = X_train[perm]
    y_train = y_train[perm]

    loss = train(X_train, y_train)
    val_loss = valid(X_test, y_test)

    if epoch % 100 == 0:
        print('epoch %d, loss: %.4f val_loss: %.4f' % (epoch, loss, val_loss))

    loss_list.append(loss)
    val_loss_list.append(val_loss)

#print('MSE Loss: %.4f'%(criterion(model(torch.FloatTensor(X_test)), torch.FloatTensor(y_test)).item()))
print('MSE Loss: %.4f'%val_loss_list[-1])


# plot learning curve
plt.plot(range(num_epochs), loss_list, 'r-', label='train_loss')
plt.plot(range(num_epochs), val_loss_list, 'b-', label='val_loss')
plt.legend()
plt.show()




########################################################################
# sklearnでの線形回帰
########################################################################

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

lr = LinearRegression()
lr.fit(X_train, y_train)
print('MSE Loss: %.4f'%mean_squared_error(y_test, lr.predict(X_test)))
