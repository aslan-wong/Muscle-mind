import random
import scipy.io as sio
import numpy as np
import sklearn.svm as svm
from sklearn.metrics import accuracy_score


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def read_data(train_path, ele1, test_path, ele2, isTrain):
    X = sio.loadmat(train_path).get(ele1).astype(np.float32)
    y = sio.loadmat(test_path).get(ele2).astype(np.longlong)
    y = np.transpose(y)
    y = y.reshape([len(y)])
    return X, y


set_seed(42)


def train(user):
    with open("./svm_result.txt", "a+") as f:
        f.write(user + "\n")
    X_train, y_train = read_data("ML_Feature/"+user+"/train_data_1d_all.mat", "train_data_1d_all",
                                 "ML_Feature/"+user+"/action_train_label_all.mat",
                                 "action_train_label_all", True)

    X_test, y_test = read_data("ML_Feature/"+user+"/test_data_1d_all.mat", "test_data_1d_all",
                               "ML_Feature/"+user+"/action_test_label_all.mat",
                               "action_test_label_all", False)
    min_train = np.expand_dims(X_train.min(-1), axis=1)
    max_train = np.expand_dims(X_train.max(-1), axis=1)
    X_train = (X_train - min_train) / (max_train - min_train)
    min_test = np.expand_dims(X_test.min(-1), axis=1)
    max_test = np.expand_dims(X_test.max(-1), axis=1)
    X_test = (X_test - min_test) / (max_test - min_test)
    model = svm.SVC(kernel="rbf", decision_function_shape="ovo")

    model.fit(X_train, y_train)
    acu_train = model.score(X_train, y_train)
    acu_test = model.score(X_test, y_test)
    y_pred = model.predict(X_test)
    acc1 = accuracy_score(y_test, y_pred)

    print(acu_train, acu_test, acc1)
    with open("./svm_result.txt", "a+") as f:
        f.write(str(acc1) + "\t")

    X_train, y_train = read_data("ML_Feature/"+user+"/train_data_1d_all.mat", "train_data_1d_all",
                                 "ML_Feature/"+user+"/mindeffort_train_label_all.mat",
                                 "mindeffort_train_label_all", True)

    X_test, y_test = read_data("ML_Feature/"+user+"/test_data_1d_all.mat", "test_data_1d_all",
                               "ML_Feature/"+user+"/mindeffort_test_label_all.mat",
                               "mindeffort_test_label_all", False)
    min_train = np.expand_dims(X_train.min(-1), axis=1)
    max_train = np.expand_dims(X_train.max(-1), axis=1)
    X_train = (X_train - min_train) / (max_train - min_train)
    min_test = np.expand_dims(X_test.min(-1), axis=1)
    max_test = np.expand_dims(X_test.max(-1), axis=1)
    X_test = (X_test - min_test) / (max_test - min_test)
    model = svm.SVC(kernel="rbf", decision_function_shape="ovo")

    model.fit(X_train, y_train)
    acu_train = model.score(X_train, y_train)
    acu_test = model.score(X_test, y_test)
    y_pred = model.predict(X_test)
    acc2 = accuracy_score(y_test, y_pred)

    print(acu_train, acu_test, acc2)
    with open("./svm_result.txt", "a+") as f:
        f.write(str(acc2) + "\t")

    X_train, y_train = read_data("ML_Feature/"+user+"/train_data_1d_all.mat", "train_data_1d_all",
                                 "ML_Feature/"+user+"/rm_train_label_all.mat",
                                 "rm_train_label_all", True)

    X_test, y_test = read_data("ML_Feature/"+user+"/test_data_1d_all.mat", "test_data_1d_all",
                               "ML_Feature/"+user+"/rm_test_label_all.mat",
                               "rm_test_label_all", False)
    min_train = np.expand_dims(X_train.min(-1), axis=1)
    max_train = np.expand_dims(X_train.max(-1), axis=1)
    X_train = (X_train - min_train) / (max_train - min_train)
    min_test = np.expand_dims(X_test.min(-1), axis=1)
    max_test = np.expand_dims(X_test.max(-1), axis=1)
    X_test = (X_test - min_test) / (max_test - min_test)
    model = svm.SVC(kernel="rbf", decision_function_shape="ovo")

    model.fit(X_train, y_train)
    acu_train = model.score(X_train, y_train)
    acu_test = model.score(X_test, y_test)
    y_pred = model.predict(X_test)
    acc3 = accuracy_score(y_test, y_pred)

    print(acu_train, acu_test, acc3)
    with open("./svm_result.txt", "a+") as f:
        f.write(str(acc3) + "\n")

users = ["user1", "user2", "user3", "user4", "user5", "user6", "user7", "user8", "user9", "user10", "user11", "user12",
         "user13"]
users = ["all"]
for i in range(len(users)):
    train(users[i])
