# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter, MultipleLocator


def to_percent(temp, position):
    return '%1.0f' % (100 * temp) + '%'


attention = [0.817080999,	0.903343838, 0.944386088]
rm = [0.84842437,	0.84012605,	0.939437442]
action = [0.982615546,	0.978460551,	0.990960551]

total_width, n = 0.5, 2
width = total_width / n
x1 = np.array(range(len(action)))
# x1 = x1 - width/2*3
x2 = [x + width for x in x1]
x3 = [x + width for x in x2]
x1 = list(x1)

plt.figure()
plt.bar(x1, attention, width=width, label='Attention Focus', color=['#C95C45'], edgecolor='black')  # darkseagreen
plt.bar(x2, rm, width=width, label='1RM', color=['#2A5ABB'], edgecolor='black')
plt.bar(x3, action, width=width, label='Exercise', color=['#0CB599'], edgecolor='black')  # darkseagreen

# gold
plt.xticks([x + width for x in range(len(x1))],
           ['SVM', 'Single Task DNN', 'Our Method'])
y_major_locator = MultipleLocator(0.1)
ax = plt.gca()
ax.yaxis.set_major_locator(y_major_locator)
plt.ylim(.6, 1)
# plt.ylabel('RMS',fontproperties='Times New Roman', size=16, weight='bold')
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
plt.savefig("fig/compare2.jpg", dpi=500, bbox_inches="tight")
plt.show()
