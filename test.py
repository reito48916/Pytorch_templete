import torch
import torch.nn as nn
import torch.nn.functional as F

import dataset
import model

import numpy as np


#定数宣言
gpu_number = 0
gpu_available = torch.cuda.is_available()
x_dim = 100
test_num = 100


#numpyでテストデータの作成
x_test_np = np.random.randn(test_num,x_dim)
y_test_np = np.zeros(test_num)
for i in range(test_num):
    y_test_np[i] = np.sum(x_test_np[i])/100.0


#numpy配列をpytorchで扱うtensorに変換
x_test = torch.from_numpy(x_test_np).float()
y_test = torch.from_numpy(y_test_np).float() #ラベルの場合long()にすること


#model.pyに定義したモデルのインスタンスを作成しパラメータのロード
net = model.SimpleMLP()
#net = model.SimpleCNN()
net.load_state_dict(torch.load("learning_result/parameters_epoch9"))
if gpu_available:
    net = net.to("cuda:"+str(gpu_number))
    print("cuda available")


#lossや正答率を自分のコードで計算するパート
data_num = x_test.size()[0]
total_loss = 0.0
criterion = nn.MSELoss()
for i in range(data_num):
    inputs = x_test[i].view(1,-1)
    teacher = y_test[i].view(1,-1)
    if gpu_available:
            inputs = inputs.to("cuda:"+str(gpu_number))
            teacher = teacher.to("cuda:"+str(gpu_number))
    output = net(inputs)
    loss = criterion(output,teacher)
    total_loss += loss.item()
print("total loss was")
print(total_loss)
