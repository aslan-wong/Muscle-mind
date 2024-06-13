import random

import sklearn.metrics
import torch.nn as nn
import scipy.io as sio
from sklearn.metrics import f1_score, confusion_matrix, classification_report, accuracy_score, precision_score, \
    recall_score
from torch.utils.data import TensorDataset, DataLoader
from torch.autograd import Variable
from sklearn.preprocessing import MinMaxScaler
import torch
import numpy as np


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_data(train_path, ele1, lable_path, ele2, isTrain):
    X = sio.loadmat(train_path).get(ele1).astype(np.float32)
    y1 = sio.loadmat(lable_path).get(ele2).astype(np.longlong)
    y1 = np.transpose(y1)
    # y1 = y1.reshape([len(y1)])
    X = torch.tensor(X)
    y1 = torch.tensor(y1)
    X, y = Variable(X), Variable(y1)
    dataset = TensorDataset(X, y)
    if isTrain:
        loader = DataLoader(dataset, batch_size=100, shuffle=True, drop_last=False)
        return X, y, loader
    loader = DataLoader(dataset, batch_size=3000, shuffle=True, drop_last=False)
    return X, y, loader


def calResult(predictions, y_test):
    mae = np.mean(np.absolute(predictions - y_test))
    print("mae: ", mae)
    corr = np.corrcoef(predictions, y_test)[0][1]
    print("corr: ", corr)
    mult = round(sum(np.round(predictions) == np.round(y_test)) / float(len(y_test)), 5)
    print("mult_acc: ", mult)
    f_score = round(f1_score(np.round(predictions), np.round(y_test), average='weighted'), 5)
    print("mult f_score: ", f_score)
    true_label = (y_test > 0)
    predicted_label = (predictions > 0)
    print("Confusion Matrix :")
    print(confusion_matrix(true_label, predicted_label))
    print("Classification Report :")
    print(classification_report(true_label, predicted_label, digits=5))
    print("Accuracy ", accuracy_score(true_label, predicted_label))

# 定义网络结构
class DNN(torch.nn.Module):
    def __init__(self):
        super(DNN, self).__init__()

        self.mlp1 = torch.nn.Sequential(
            torch.nn.Linear(56, 64),
            torch.nn.Dropout(p=0.2),
            torch.nn.ReLU()
        )
        self.mlp2 = torch.nn.Sequential(
            torch.nn.Linear(64, 32),
            torch.nn.Dropout(p=0.1),
            torch.nn.Sigmoid()
        )

        self.mlp3_1 = torch.nn.Sequential(
            torch.nn.Linear(32, 16),
            torch.nn.Sigmoid()
        )
        self.mlp3_2 = torch.nn.Sequential(
            torch.nn.Linear(16, 8),
            torch.nn.Sigmoid()
        )
        self.mlp3_3 = torch.nn.Sequential(
            torch.nn.Linear(8, 3),
            torch.nn.Sigmoid()
        )

        self.last = torch.nn.Softmax()

    def forward(self, x):
        x = self.mlp1(x)
        x = self.mlp2(x)

        x3 = self.mlp3_1(x)
        x3 = self.mlp3_2(x3)
        x3 = self.mlp3_3(x3)
        return x3


def test(user):
    X_test, y_test, test_loader = read_data("ML_Feature/" + user + "/test_data_1d.mat", "test_data_1d",
                                               "ML_Feature/" + user + "/rm_test_label.mat", "rm_test_label",
                                               False)
    # index = []
    # for i, ele in enumerate(y_test):
    #     if ele[2] == 2:
    #         index.append(i)
    # X_test = X_test[index,:]
    # y_test = y_test[index,:]
    model = torch.load("./model/" + user + "_single_rm.pth")
    test_output1 = model(X_test)
    # if torch.cuda.is_available():
    #     model = model.cuda()
    predicted1 = torch.max(test_output1, 1)[1].numpy()

    target1 = y_test[:, 0].numpy()

    acc1 = accuracy_score(target1, predicted1)

    # with open("./DNN_result.txt", "a+") as f:
    #     f.write(str(user) + "\n")
    with open("./DNN1_result.txt", "a+") as f:
        f.write(str(acc1)+"\n")

    # with open("./DNN_result.txt", "a+") as f:
    #     f.write(str(specificity) + ", ")

set_seed(42)
users = ["user1", "user2", "user3", "user4", "user5", "user6", "user7", "user8", "user9", "user10", "user11",
         "user13"]
with open("./DNN1_result.txt", "a+") as f:
    f.write("DNN1 rm" + "\n")
for i in range(len(users)):
    test(users[i])