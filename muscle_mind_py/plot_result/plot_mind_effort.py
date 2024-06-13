# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter, MultipleLocator


def to_percent(temp, position):
    return '%1.0f' % (100 * temp) + '%'


precision = [0.9523809523809523, 0.9642857142857143, 0.9672131147540983, 1.0, 0.8888888888888888, 0.9423076923076923,
             1.0, 0.9482758620689655, 0.9354838709677419, 0.9344262295081968, 0.9666666666666667, 0.875]
specificity = [1.0, 0.90625, 0.9830508474576272, 0.9523809523809523, 0.9298245614035088, 0.8382352941176471,
               0.967741935483871, 0.9344262295081968, 0.9655172413793104, 0.9491525423728814, 0.9666666666666667,
               0.9285714285714286]

precision_67 = [0.8695652173913043, 1.0, 1.0, 1.0, 0.8636363636363636, 1.0, 1.0, 0.9473684210526315, 0.8695652173913043,
                1.0, 0.95, 0.8260869565217391]
specificity_67 = [1.0, 0.9523809523809523, 1.0, 0.9523809523809523, 0.9444444444444444, 0.8333333333333334,
                  0.9523809523809523, 0.9047619047619048, 1.0, 1.0, 0.95, 0.9411764705882353]

precision_85 = [1.0, 0.9047619047619048, 1.0, 1.0, 0.9473684210526315, 1.0, 1.0, 0.95, 1.0, 0.8333333333333334,
                0.9523809523809523, 0.85]
specificity_85 = [1.0, 0.9473684210526315, 1.0, 0.9523809523809523, 0.9047619047619048, 0.8333333333333334, 1.0, 0.95,
                  0.9523809523809523, 1.0, 1.0, 0.85]

total_width, n = 0.5, 2
width = total_width / n
x1 = np.array(range(len(precision)))
x2 = [x + width * 3 / 2 for x in x1]
x1 = x1 + width / 2

plt.figure()
plt.bar(x1, precision, width=width, label='Precision', color=['#2A5ABB'], edgecolor='black')  # darkseagreen
plt.bar(x2, specificity, width=width, label='Specificity', color=['#F2E1A0'], edgecolor='black')  #gold
plt.xticks([x + width for x in range(len(x1))],
           ['U1', 'U2', 'U3', 'U4', 'U5', 'U6', 'U7', 'U8', 'U9', 'U10', 'U11', 'U12'])
y_major_locator = MultipleLocator(0.1)
ax = plt.gca()
ax.yaxis.set_major_locator(y_major_locator)
plt.ylim(0.6, 1)
plt.yticks(fontproperties='Times New Roman', size=16, weight='bold')  # 设置大小及加粗
plt.xticks(fontproperties='Times New Roman', size=16, weight='bold')
plt.gca().yaxis.set_major_formatter(FuncFormatter(to_percent))
num1 = 1
num2 = 1.01
num3 = 4
num4 = 0
font1 = {'family': 'Times New Roman',
         'weight': 'bold',
         'size': 15,
         }

plt.legend(bbox_to_anchor=(num1, num2), prop=font1, loc=num3, borderaxespad=num4, ncol=3, handletextpad=0.1,
           columnspacing=1,
           frameon=False, shadow=False)
plt.savefig("fig/mind_all.jpg", dpi=500)

plt.figure()
plt.bar(x1, precision_67, width=width, label='Precision', color=['#2A5ABB'], edgecolor='black')
plt.bar(x2, specificity_67, width=width, label='Specificity', color=['#F2E1A0'], edgecolor='black')
plt.xticks([x + width for x in range(len(x1))],
           ['U1', 'U2', 'U3', 'U4', 'U5', 'U6', 'U7', 'U8', 'U9', 'U10', 'U11', 'U12'])
y_major_locator = MultipleLocator(0.1)
ax = plt.gca()
ax.yaxis.set_major_locator(y_major_locator)
plt.ylim(0.6, 1)
plt.yticks(fontproperties='Times New Roman', size=16, weight='bold')  # 设置大小及加粗
plt.xticks(fontproperties='Times New Roman', size=16, weight='bold')
plt.gca().yaxis.set_major_formatter(FuncFormatter(to_percent))
num1 = 1
num2 = 1.01
num3 = 4
num4 = 0
font1 = {'family': 'Times New Roman',
         'weight': 'bold',
         'size': 15,
         }

plt.legend(bbox_to_anchor=(num1, num2), prop=font1, loc=num3, borderaxespad=num4, ncol=3, handletextpad=0.1,
           columnspacing=1,
           frameon=False, shadow=False)
plt.savefig("fig/mind_67.jpg", dpi=500)

plt.figure()
plt.bar(x1, precision_85, width=width, label='Precision', color=['#2A5ABB'], edgecolor='black')
plt.bar(x2, specificity_85, width=width, label='Specificity', color=['#F2E1A0'], edgecolor='black')
plt.xticks([x + width for x in range(len(x1))],
           ['U1', 'U2', 'U3', 'U4', 'U5', 'U6', 'U7', 'U8', 'U9', 'U10', 'U11', 'U12'])
y_major_locator = MultipleLocator(0.1)
ax = plt.gca()
ax.yaxis.set_major_locator(y_major_locator)
plt.ylim(0.6, 1)
plt.yticks(fontproperties='Times New Roman', size=16, weight='bold')  # 设置大小及加粗
plt.xticks(fontproperties='Times New Roman', size=16, weight='bold')
plt.gca().yaxis.set_major_formatter(FuncFormatter(to_percent))
num1 = 1
num2 = 1.01
num3 = 4
num4 = 0
font1 = {'family': 'Times New Roman',
         'weight': 'bold',
         'size': 15,
         }

plt.legend(bbox_to_anchor=(num1, num2), prop=font1, loc=num3, borderaxespad=num4, ncol=3, handletextpad=0.1,
           columnspacing=1,
           frameon=False, shadow=False)
plt.savefig("fig/mind_85.jpg", dpi=500)

plt.show()
