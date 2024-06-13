# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter, MultipleLocator
import seaborn as sns


def to_percent(temp, position):
    return '%1.0f' % (100 * temp) + '%'


user1 = np.array([[24, 0, 0, 0, 0], [0, 24, 0, 0, 0], [0, 0, 24, 0, 0], [1, 0, 0, 23, 0], [0, 0, 0, 0, 24]])
user2 = np.array([[24, 0, 0, 0, 0], [0, 24, 0, 0, 0], [1, 1, 22, 0, 0], [0, 0, 0, 24, 0], [0, 0, 0, 0, 24]])
user3 = np.array([[23, 0, 1, 0, 0], [0, 24, 0, 0, 0], [1, 0, 23, 0, 0], [0, 0, 0, 24, 0], [0, 0, 0, 0, 24]])
user4 = np.array([[24, 0, 0, 0, 0], [0, 24, 0, 0, 0], [0, 0, 24, 0, 0], [0, 0, 0, 24, 0], [0, 0, 0, 0, 23]])
user5 = np.array([[23, 0, 0, 0, 1], [0, 24, 0, 0, 0], [0, 0, 24, 0, 0], [0, 0, 0, 24, 0], [0, 0, 0, 0, 24]])
user6 = np.array([[24, 0, 0, 0, 0], [0, 24, 0, 0, 0], [0, 0, 24, 0, 0], [0, 0, 0, 24, 0], [0, 0, 0, 0, 24]])
user7 = np.array([[24, 0, 0, 0, 0], [0, 24, 0, 0, 0], [0, 0, 24, 0, 0], [0, 0, 0, 24, 0], [0, 0, 0, 0, 24]])
user8 = np.array([[24, 0, 0, 0, 0], [0, 24, 0, 0, 0], [0, 1, 22, 1, 0], [0, 0, 0, 24, 0], [0, 0, 0, 0, 23]])
user9 = np.array([[23, 0, 0, 0, 1], [0, 24, 0, 0, 0], [1, 0, 23, 0, 0], [0, 0, 0, 24, 0], [0, 0, 0, 0, 24]])
user10 = np.array([[22, 0, 2, 0, 0], [0, 24, 0, 0, 0], [0, 0, 24, 0, 0], [0, 0, 0, 24, 0], [0, 0, 0, 0, 24]])
user11 = np.array([[24, 0, 0, 0, 0], [0, 24, 0, 0, 0], [0, 0, 24, 0, 0], [0, 0, 0, 23, 1], [0, 0, 0, 0, 24]])
user12 = np.array([[24, 0, 0, 0, 0], [0, 24, 0, 0, 0], [0, 0, 24, 0, 0], [0, 0, 0, 24, 0], [0, 0, 0, 0, 24]])

action_confusion_matrix_num = user1 + user2 + user3 + user4 + user5 + user6 + user7 + user8 + user9 + user10 + user11 + user12
action_confusion_matrix = action_confusion_matrix_num / np.sum(action_confusion_matrix_num, axis=1)
action_confusion_matrix = np.around(action_confusion_matrix, 4)

xtick = ['bench press', 'pull over', 'front raise', 'kick back', 'bicep curl']
ytick = ['bench press', 'pull over', 'front raise', 'kick back', 'bicep curl']
sns.heatmap(action_confusion_matrix, square=True, vmax = 1, vmin = 0, cmap="Blues", center=0.5, annot=True, cbar=True, xticklabels=xtick,
            yticklabels=ytick)

# plt.matshow(action_confusion_matrix, cmap=plt.cm.Blues) # 根据最下面的图按自己需求更改颜色
# # plt.colorbar()
#
# for i in range(len(action_confusion_matrix)):
#     for j in range(len(action_confusion_matrix)):
#         if i == j:
#             plt.annotate(action_confusion_matrix[j, i], xy=(i, j), horizontalalignment='center', verticalalignment='center',color='w')
#         else:
#             plt.annotate(action_confusion_matrix[j, i], xy=(i, j), horizontalalignment='center', verticalalignment='center')
#
# # plt.tick_params(labelsize=15) # 设置左边和上面的label类别如0,1,2,3,4的字体大小。
#
#
plt.ylabel('True label', fontdict={'family': 'Times New Roman', 'size': 15, 'weight': 'bold'})  # 设置字体大小。
plt.xlabel('Predicted label', fontdict={'family': 'Times New Roman', 'size': 15, 'weight': 'bold'})  # 设置字体大小。
# plt.xticks(range(0,5), labels=['bench press','pull over','front raise','kick back','bicep curl']) # 将x轴或y轴坐标，刻度 替换为文字/字符
# plt.yticks(range(0,5), labels=['bench press','pull over','front raise','kick back','bicep curl'])
plt.yticks(fontproperties='Times New Roman', size=12)  # 设置大小及加粗
plt.xticks(fontproperties='Times New Roman', size=12)
# ax = plt.axes()
# plt.axes().spines['top'].set_color('white')
# ax.spines['right'].set_color('white')
# ax.spines['bottom'].set_color('white')
# ax.spines['left'].set_color('white')

plt.savefig("fig/action_confusion_matrix.jpg", dpi=500)

classNum = int(np.sqrt(np.size(action_confusion_matrix_num)))
TPs, FNs, FPs = [], [], []
for i in range(classNum):
    FN = 0
    for row in range(classNum):
        if i == row:
            TPs.append(action_confusion_matrix_num[i][row])
        else:
            FN += action_confusion_matrix_num[i][row]
    FNs.append(FN)
    FP = 0
    for col in range(classNum):
        if i != col:
            FP += action_confusion_matrix_num[col][i]
    FPs.append(FP)
precisions, recalls = [], []
for i in range(classNum):
    precisions.append(TPs[i] / (TPs[i] + FPs[i]))
    recalls.append(TPs[i] / (TPs[i] + FNs[i]))
print(precisions)
print(recalls)
plt.show()
