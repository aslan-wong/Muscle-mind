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
EPOCH = 2000
LR = 0.01  # learning rate

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_data(train_path, ele1, test_path, ele2, isTrain):
    X = sio.loadmat(train_path).get(ele1).astype(np.float32)
    y = sio.loadmat(test_path).get(ele2).astype(np.longlong)
    y = np.transpose(y)
    y = torch.tensor(y.reshape([len(y)]))
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
    test_output = model(X)
    predicted = torch.max(test_output, 1)[1].numpy()
    # print(predict_y-y_test.numpy())
    target = Y.numpy()
    data_num = Y.size(0)
    correct_num = (predicted == target).sum().item()
    acc = correct_num / data_num

    print('{} epoch {} =====>> acc: {:.4f}<<=====>> '.format(title, epoch, acc))


# 定义网络结构
class CNNnet(torch.nn.Module):
    def __init__(self):
        super(CNNnet, self).__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=8,
                            out_channels=16,
                            kernel_size=4,
                            stride=3,
                            padding=1),
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU()
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(16, 32, 4, 3, 1),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU()
        )
        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, 4, 3, 1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU()
        )

        self.mlp1 = torch.nn.Linear(64 * 2 * 3, 100)
        self.mlp2 = torch.nn.Linear(100, 3)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.mlp1(x.view(x.size(0), -1))
        x = self.mlp2(x)
        return x

set_seed(42)
X_train, y_train, train_loader = read_data("dataset/user1_cut/train_data_2d.mat", "train_data_2d", "dataset/user1_cut/rm_train_label.mat",
                                           "rm_train_label", True)
X_test1, y_test1, test_loader1 = read_data("dataset/user1_cut/test_data_2d.mat", "test_data_2d", "dataset/user1_cut/rm_test_label.mat",
                                           "rm_test_label", False)

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
        output = model(train_x)
        loss = criterion(output, train_y)  # cross entropy loss
        optimizer.zero_grad()  # clear gradients for this training step
        loss.backward()  # backpropagation, compute gradients
        optimizer.step()  # apply gradients
    model.eval()
    if epoch % 50 == 0:
        test("train                ", X_train, y_train)
        test("test            ", X_test1, y_test1)
        print("\r\n")
