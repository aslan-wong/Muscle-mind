import random
import torch.nn as nn
import scipy.io as sio
from sklearn.metrics import f1_score, confusion_matrix, classification_report, accuracy_score
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
    X = torch.tensor(X)
    y1 = sio.loadmat(lable_path).get(ele2).astype(np.longlong)
    y1 = np.transpose(y1)
    y1 = y1.reshape([len(y1)])
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


def test(model, epoch, title, X, Y, criterion, best_loss, user):
    test_output1= model(X)
    loss = criterion(test_output1, Y)
    if loss < best_loss:
        torch.save(model, "./model/" + user + "_single_attention.pth")
        best_loss = loss
    predicted1 = torch.max(test_output1, 1)[1].numpy()
    data_num = Y.size(0)
    target1 = Y.numpy()

    correct_num1 = (predicted1 == target1).sum().item()

    acc1 = correct_num1 / data_num

    print('{} epoch {} =====>> acc1: {:.4f}<<=====>>acc1:   '.format(title, epoch,acc1))
    return best_loss



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
            torch.nn.Linear(16, 2),
            torch.nn.Sigmoid()
        )
        # self.mlp3_3 = torch.nn.Sequential(
        #     torch.nn.Linear(8, 3),
        #     torch.nn.Sigmoid()
        # )

        self.last = torch.nn.Softmax()

    def forward(self, x):
        x = self.mlp1(x)
        x = self.mlp2(x)

        x3 = self.mlp3_1(x)
        x3 = self.mlp3_2(x3)
        # x3 = self.mlp3_3(x3)
        return x3


def train(user):
    # Hyper parameter
    EPOCH = 10000
    LR = 0.01  # learning rate
    set_seed(42)
    # mindeffort_test_label1 mindeffort_train_label1
    X_train, y_train, train_loader = read_data("ML_Feature/" + user + "/train_data_1d_all.mat", "train_data_1d_all",
                                                                 "ML_Feature/" + user + "/mindeffort_train_label_all.mat", "mindeffort_train_label_all", True)

    X_test1, y_test1, test_loader1 = read_data("ML_Feature/" + user + "/test_data_1d_all.mat", "test_data_1d_all",
                                                       "ML_Feature/" + user + "/mindeffort_test_label_all.mat", "mindeffort_test_label_all",
                                               False)

    # X_test4, y_test4, test_loader4 = read_data("datasets/big/live_test.mat", "x1_test", "gt_test", False)
    model = DNN()  # 图片大小28*28，lstm的每个隐藏层64个节点，2层隐藏层
    # if torch.cuda.is_available():
    #     model = model.cuda()
    # print(get_parameter_number(model))
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()
    best_loss = 10000
    # training and testing
    for epoch in range(EPOCH):
        model.train()
        for iteration, (train_x, train_y) in enumerate(train_loader):  # train_x's shape (BATCH_SIZE,1,28,28)
            # 第一个28是序列长度，第二个28是序列中每个数据的长度。
            output1 = model(train_x)
            loss = criterion(output1, train_y)
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
            # test("train                ", X_train, y_train, criterion)
            best_loss = test(model, epoch, "test            ", X_test1, y_test1, criterion, best_loss, user)

            print("\r\n")
    print("\r\n best_loss \r\n")



users = ["user1", "user2", "user3", "user4", "user5", "user6", "user7", "user8", "user9", "user10", "user11", "user12",
         "user13"]
users = ["all"]
for i in range(len(users)):
    train(users[i])
