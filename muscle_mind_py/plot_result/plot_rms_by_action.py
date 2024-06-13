# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter, MultipleLocator


def to_percent(temp, position):
    return '%1.0f' % (100 * temp) + '%'


attention67 = [2075.397467, 1825.93775, 1770.056542, 1852.186267, 1265.98975]
no_attention67 = [2136.639433, 2049.9119, 2112.412883, 2295.699317, 1507.973467]
attention85 = [1818.692108, 1804.969092, 1858.853267, 2016.612117, 1344.995433]
no_attention85 = [1895.48095, 1874.137683, 2023.510825, 2143.602658, 1391.483583]



total_width, n = 0.4, 2
width = total_width / n
x1 = np.array(range(len(attention67)))
x1 = x1 - width/2
x2 = [x + width for x in x1]
x3 = [x + width for x in x2]
x4 = [x + width for x in x3]
x1 = list(x1)

plt.figure()
plt.barh(x1, attention67, height=width, label='67% w/o attention', color=['#F2E1A0'], edgecolor='black')  # darkseagreen
plt.barh(x2, no_attention67, height=width, label='67% w/ attention', color=['#2A5ABB'], edgecolor='black')
plt.barh(x3, attention85, height=width, label='85% w/o attention', color=['#0CB599'], edgecolor='black')  # darkseagreen
plt.barh(x4, no_attention85, height=width, label='85% w/ attention', color=['#808080'], edgecolor='black')
# gold
plt.yticks([x + width for x in range(len(x1))],
           ['Bench Press', 'Lying Pullovers', 'Front Raise', 'Triceps Kickback', 'Bicep Curl'])
# y_major_locator = MultipleLocator(0.1)
ax = plt.gca()
# ax.yaxis.set_major_locator(y_major_locator)
plt.xlim(0, 3000)
plt.xlabel('RMS',fontproperties='Times New Roman', size=16, weight='bold')
plt.yticks(fontproperties='Times New Roman', size=14, weight='bold')  # 设置大小及加粗
plt.xticks(fontproperties='Times New Roman', size=14, weight='bold')
# plt.gca().yaxis.set_major_formatter(FuncFormatter(to_percent))
num1 = 1
num2 = 1.01
num3 = 4
num4 = 0
font1 = {'family': 'Times New Roman',
         'weight': 'bold',
         'size': 15,
         }

plt.legend(bbox_to_anchor=(num1, num2), prop=font1, loc=num3, borderaxespad=num4, ncol=2, handletextpad=0.1,
           columnspacing=1,
           frameon=False, shadow=False)
plt.savefig("fig/rms_action.jpg", dpi=500, bbox_inches="tight")
plt.show()
