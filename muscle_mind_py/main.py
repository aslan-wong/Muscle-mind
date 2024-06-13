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


def read_data(train_path, ele1, test_path, ele2, isTrain):
    X = sio.loadmat(train_path).get(ele1).astype(np.float32)
    y = sio.loadmat(test_path).get(ele2).astype(np.longlong)
    y = np.transpose(y - 1)
    y = torch.tensor(y.reshape([len(y)]))
    X = torch.tensor(X.transpose(2, 0, 1))
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
    FN, FP, N, P = 0, 0, 0, 0
    correct_num, data_num = 0, 0
    test_output = model(X)
    predicted = torch.max(test_output, 1)[1].numpy()
    # print(predict_y-y_test.numpy())
    target = Y.numpy()
    diff = predicted - target
    data_num = Y.size(0)
    correct_num += (predicted == target).sum().item()
    TP = P - FN
    acc = correct_num / data_num

    print('{} epoch {} =====>> acc: {:.4f}<<=====>> '.format(title, epoch, acc))


class LSTMnet(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_layer, n_class):
        super(LSTMnet, self).__init__()
        self.n_layer = n_layer
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(in_dim, hidden_dim, n_layer, batch_first=True)
        self.linear = nn.Linear(hidden_dim, n_class)

    def forward(self, x):  # x's shape (batch_size, 序列长度, 序列中每个数据的长度)
        # x = x.unsqueeze(-1)
        out, _ = self.lstm(x)  # out's shape (batch_size, 序列长度, hidden_dim)
        out = out[:, -1, :]  # 中间的序列长度取-1，表示取序列中的最后一个数据，这个数据长度为hidden_dim，
        # 得到的out的shape为(batch_size, hidden_dim)
        out = self.linear(out)  # 经过线性层后，out的shape为(batch_size, n_class)
        return out


X_train, y_train, train_loader = read_data("dataset/train_data.mat", "train_data", "dataset/action_train_label.mat","action_train_label", True)
X_test1, y_test1, test_loader1 = read_data("dataset/test_data.mat", "test_data","dataset/action_test_label.mat", "action_test_label", False)

# X_test4, y_test4, test_loader4 = read_data("datasets/big/live_test.mat", "x1_test", "gt_test", False)
model = LSTMnet(8, 4, 2, 5)  # 图片大小28*28，lstm的每个隐藏层64个节点，2层隐藏层
# if torch.cuda.is_available():
#     model = model.cuda()

optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()

# training and testing
for epoch in range(EPOCH):
    for iteration, (train_x, train_y) in enumerate(train_loader):  # train_x's shape (BATCH_SIZE,1,28,28)
        # 第一个28是序列长度，第二个28是序列中每个数据的长度。
        output = model(train_x)
        loss = criterion(output, train_y)  # cross entropy loss
        optimizer.zero_grad()  # clear gradients for this training step
        loss.backward()  # backpropagation, compute gradients
        optimizer.step()  # apply gradients

    if epoch % 50 == 0:
        test("train                ", X_train, y_train)
        test("test            ", X_test1, y_test1)
        print("\r\n")
