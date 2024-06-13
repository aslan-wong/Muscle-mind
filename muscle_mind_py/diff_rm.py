import random
import torch.nn as nn
import scipy.io as sio
from sklearn.metrics import f1_score, confusion_matrix, classification_report, accuracy_score
from torch.utils.data import TensorDataset, DataLoader
from torch.autograd import Variable
from sklearn.preprocessing import MinMaxScaler
import torch
import numpy as np

# Hyper parameter
EPOCH = 4000
LR = 0.005  # learning rate


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_data(train_path, ele1, lable_path, ele2, lable_path2, ele3, lable_path3, ele4, isTrain):
    X = sio.loadmat(train_path).get(ele1).astype(np.float32)
    y1 = sio.loadmat(lable_path).get(ele2).astype(np.longlong)
    y1 = np.transpose(y1)
    y2 = sio.loadmat(lable_path2).get(ele3).astype(np.longlong)
    y2 = np.transpose(y2)
    y3 = sio.loadmat(lable_path3).get(ele4).astype(np.longlong)
    y3 = np.transpose(y3)
    index = []
    for i in range(len(y1)):
        if y1[i] == 1 and y2[i] == 1:
            index.append(i)
    X = X[:, :, :, index]
    y1 = y1[index]
    y2 = y2[index]
    y3 = y3[index]
    y1 = torch.tensor(y1)
    y2 = torch.tensor(y2)
    y3 = torch.tensor(y3)
    y = torch.cat((y1, y2, y3), dim=1)
    X = torch.tensor(X.transpose(3, 2, 0, 1))
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


def test(title, X, Y):
    test_output1, test_output2, test_output3 = model(X)
    predicted1 = torch.max(test_output1, 1)[1].numpy()
    predicted2 = torch.max(test_output2, 1)[1].numpy()
    predicted3 = torch.max(test_output3, 1)[1].numpy()
    data_num = Y.size(0)
    target1 = Y[:, 0].numpy()
    correct_num = (predicted1 == target1).sum().item()
    acc1 = correct_num / data_num
    target2 = Y[:, 1].numpy()
    correct_num = (predicted2 == target2).sum().item()
    acc2 = correct_num / data_num
    target3 = Y[:, 2].numpy()
    correct_num = (predicted3 == target3).sum().item()
    acc3 = correct_num / data_num
    print('{} epoch {} =====>> acc1: {:.4f}<<=====>> acc2: {:.4f}<<=====>> acc3: {:.4f}<<=====>> '.format(title, epoch,
                                                                                                          acc1, acc2,
                                                                                                          acc3))


class CrossStich(nn.Module):
    def __init__(self):
        super(CrossStich, self).__init__()
        self.params = nn.Parameter(torch.tensor([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]).resize(3, 3), requires_grad=True)

    def forward(self, x1, x2, x3):
        output1 = x1 * self.params[0, 0] + x2 * self.params[1, 0] + x3 * self.params[2, 0]
        output2 = x1 * self.params[0, 1] + x2 * self.params[1, 1] + x3 * self.params[2, 1]
        output3 = x1 * self.params[0, 2] + x2 * self.params[1, 2] + x3 * self.params[2, 2]
        return output1, output2, output3


# 定义网络结构
class CNNnet(torch.nn.Module):
    def __init__(self):
        super(CNNnet, self).__init__()
        self.conv1_1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=8,
                            out_channels=16,
                            kernel_size=4,
                            stride=3,
                            padding=1),
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU()
        )
        self.conv1_2 = torch.nn.Sequential(
            torch.nn.Conv2d(8, 16, 4, 3, 1),
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU()
        )
        self.conv1_3 = torch.nn.Sequential(
            torch.nn.Conv2d(8, 16, 4, 3, 1),
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU()
        )

        self.cross1 = CrossStich()

        self.conv2_1 = torch.nn.Sequential(
            torch.nn.Conv2d(16, 32, 4, 3, 1),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU()
        )
        self.conv2_2 = torch.nn.Sequential(
            torch.nn.Conv2d(16, 32, 4, 3, 1),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU()
        )
        self.conv2_3 = torch.nn.Sequential(
            torch.nn.Conv2d(16, 32, 4, 3, 1),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU()
        )
        self.cross2 = CrossStich()
        self.conv3_1 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, 4, 3, 1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU()
        )
        self.conv3_2 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, 4, 3, 1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU()
        )
        self.conv3_3 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, 4, 3, 1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU()
        )
        self.cross3 = CrossStich()
        self.mlp1_1 = torch.nn.Linear(2 * 3 * 64, 100)
        self.mlp1_2 = torch.nn.Linear(2 * 3 * 64, 100)
        self.mlp1_3 = torch.nn.Linear(2 * 3 * 64, 100)
        self.mlp2_1 = torch.nn.Linear(100, 5)
        self.mlp2_2 = torch.nn.Linear(100, 2)
        self.mlp2_3 = torch.nn.Linear(100, 7)
        self.mlp3_3 = torch.nn.Linear(7, 3)

    def forward(self, x):
        x1, x2, x3 = x, x, x

        x1 = self.conv1_1(x1)
        x2 = self.conv1_2(x2)
        x3 = self.conv1_3(x3)
        # x1, x2, x3 = self.cross1(x1, x2, x3)

        x1 = self.conv2_1(x1)
        x2 = self.conv2_2(x2)
        x3 = self.conv2_3(x3)
        x1, x2, x3 = self.cross2(x1, x2, x3)

        x1 = self.conv3_1(x1)
        x2 = self.conv3_2(x2)
        x3 = self.conv3_3(x3)
        x1, x2, x3 = self.cross3(x1, x2, x3)

        x1 = self.mlp1_1(x1.view(x.size(0), -1))
        x1 = self.mlp2_1(x1)

        x2 = self.mlp1_2(x2.view(x.size(0), -1))
        x2 = self.mlp2_2(x2)

        x3 = self.mlp1_3(x3.view(x.size(0), -1))
        x3 = self.mlp2_3(x3)
        x3 = x3 * torch.cat((x1,x2), dim=1)
        x3 = self.mlp3_3(x3)
        return x1, x2, x3


set_seed(42)
# mindeffort_test_label1 mindeffort_train_label1
X_train, y_train, train_loader = read_data("dataset/train_data_2d_.mat", "train_data_2d",
                                           "dataset/action_train_label1.mat",
                                           "action_train_label",
                                           "dataset/mindeffort_train_label1.mat", "mindeffort_train_label",
                                           "dataset/rm_train_label1.mat", "rm_train_label", True)
X_test1, y_test1, test_loader1 = read_data("dataset/test_data_2d_.mat", "test_data_2d",
                                           "dataset/action_test_label1.mat",
                                           "action_test_label",
                                           "dataset/mindeffort_test_label1.mat", "mindeffort_test_label",
                                           "dataset/rm_test_label1.mat", "rm_test_label", False)

# X_test4, y_test4, test_loader4 = read_data("datasets/big/live_test.mat", "x1_test", "gt_test", False)
model = CNNnet()  # 图片大小28*28，lstm的每个隐藏层64个节点，2层隐藏层
# if torch.cuda.is_available():
#     model = model.cuda()

optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()

# training and testing
for epoch in range(EPOCH):
    model.train()
    for iteration, (train_x, train_y) in enumerate(train_loader):  # train_x's shape (BATCH_SIZE,1,28,28)
        # 第一个28是序列长度，第二个28是序列中每个数据的长度。
        output1, output2, output3 = model(train_x)
        loss = criterion(output1, train_y[:, 0]) + criterion(output2, train_y[:, 1]) + criterion(output3, train_y[:,
                                                                                                          2])  # cross entropy loss
        optimizer.zero_grad()  # clear gradients for this training step
        loss.backward()  # backpropagation, compute gradients
        optimizer.step()  # apply gradients
        params = list(model.parameters())
        for p in model.parameters():
            p_size = np.array(p.shape)
            if len(p_size) == 3 and p_size[0] == 3 and p_size[1] == 3 and p_size[2] == 3:
                p.data.clamp_(0, 1.0)
    model.eval()
    if epoch % 50 == 0:
        test("train                ", X_train, y_train)
        test("test            ", X_test1, y_test1)
        print("\r\n")
