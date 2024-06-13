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


def read_data(train_path, ele1, lable_path, ele2, lable_path1, ele3, lable_path2, ele4, isTrain):
    X = sio.loadmat(train_path).get(ele1).astype(np.float32)
    y1 = sio.loadmat(lable_path).get(ele2).astype(np.longlong)
    y1 = np.transpose(y1)
    # y1 = y1.reshape([len(y1)])
    y2 = sio.loadmat(lable_path1).get(ele3).astype(np.longlong)
    y2 = np.transpose(y2)
    # y2= y2.reshape([len(y2)])
    y3 = sio.loadmat(lable_path2).get(ele4).astype(np.longlong)
    y3 = np.transpose(y3)
    indexs = []
    for index in range(len(y2)):
        if y2[index] == 1:
            indexs.append(index)
    X = torch.tensor(X[indexs,:])
    y1 = torch.tensor(y1[indexs,:])
    y2 = torch.tensor(y2[indexs,:])
    y3 = torch.tensor(y3[indexs,:])
    y = torch.cat((y1, y2, y3), dim=1)
    X, y = Variable(X), Variable(y)
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
        self.mlp1_1 = torch.nn.Sequential(
            torch.nn.Linear(32, 16),
            torch.nn.Sigmoid()
        )
        self.mlp1_2 = torch.nn.Sequential(
            torch.nn.Linear(16, 5),
            torch.nn.Sigmoid()
        )
        self.mlp2_1 = torch.nn.Sequential(
            torch.nn.Linear(32, 16),
            torch.nn.Sigmoid()
        )
        self.mlp2_2 = torch.nn.Sequential(
            torch.nn.Linear(16, 2),
            torch.nn.Sigmoid()
        )
        self.mlp3_1 = torch.nn.Sequential(
            torch.nn.Linear(32, 16),
            torch.nn.Sigmoid()
        )
        self.mlp3_2 = torch.nn.Sequential(
            torch.nn.Linear(23, 8),
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
        x1 = self.mlp1_1(x)
        x1 = self.mlp1_2(x1)
        x2 = self.mlp2_1(x)
        x2 = self.mlp2_2(x2)
        x3 = self.mlp3_1(x)
        x3 = torch.cat((x1, x2, x3), dim=1)
        x3 = self.mlp3_2(x3)
        x3 = self.mlp3_3(x3)
        return x1, x2, x3
def test(user):
    X_test, y_test, test_loader = read_data("ML_Feature/" + user + "/test_data_1d.mat", "test_data_1d",
                                               "ML_Feature/" + user + "/action_test_label.mat", "action_test_label",
                                               "ML_Feature/" + user + "/mindeffort_test_label.mat",
                                               "mindeffort_test_label",
                                               "ML_Feature/" + user + "/rm_test_label.mat", "rm_test_label",
                                               False)
    # index = []
    # for i, ele in enumerate(y_test):
    #     if ele[2] == 2:
    #         index.append(i)
    # X_test = X_test[index,:]
    # y_test = y_test[index,:]
    model = torch.load("./model/" + user + ".pth")
    test_output1, test_output2, test_output3 = model(X_test)
    # if torch.cuda.is_available():
    #     model = model.cuda()
    predicted1 = torch.max(test_output1, 1)[1].numpy()
    predicted2 = torch.max(test_output2, 1)[1].numpy()
    predicted3 = torch.max(test_output3, 1)[1].numpy()
    data_num = y_test.size(0)
    target1 = y_test[:, 0].numpy()
    target2 = y_test[:, 1].numpy()
    target3 = y_test[:, 2].numpy()
    correct_num1 = (predicted1 == target1).sum().item()
    correct_num2 = (predicted2 == target2).sum().item()
    correct_num3 = (predicted3 == target3).sum().item()
    acc1 = accuracy_score(target1, predicted1)
    acc2 = accuracy_score(target2, predicted2)
    acc3 = accuracy_score(target3, predicted3)
    precision = precision_score(target2, predicted2)
    specificity = precision_score(target2, predicted2, pos_label=0)
    recall = recall_score(target2, predicted2, pos_label=0)
    matrix = confusion_matrix(target3, predicted3)
    with open("./DNN_result.txt", "a+") as f:
         f.write("[" )
    # with open("./DNN_result.txt", "a+") as f:
    #     f.write(str(user) + "\n")
    with open("./DNN_result.txt", "a+") as f:
        for i, element in enumerate(matrix):
            f.write("[")
            for j, ele in enumerate(element):
                if j < 4:
                    f.write(str(ele) + ",")
                else:
                    f.write(str(ele))
            f.write("]")
    with open("./DNN_result.txt", "a+") as f:
        f.write("]"+"\n")
    # with open("./DNN_result.txt", "a+") as f:
    #     f.write(str(specificity) + ", ")

set_seed(42)
users = ["user1", "user2", "user3", "user4", "user5", "user6", "user7", "user8", "user9", "user10", "user11",
         "user13"]
with open("./DNN_result.txt", "a+") as f:
    f.write("RM confusion_matrix attention" + "\n")
for i in range(len(users)):
    test(users[i])