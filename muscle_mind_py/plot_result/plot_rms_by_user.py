# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter, MultipleLocator


def to_percent(temp, position):
    return '%1.0f' % (100 * temp) + '%'


attention67 = [1490.69328, 2032.53846, 1733.4696, 1134.77542, 1795.89426, 1801.0569, 2875.32204, 1762.47974, 1341.71826,
               1216.66768, 1961.02014, 1949.32688]
no_attention67 = [1940.07458, 2633.83462, 2135.73018, 1641.04562, 2150.15036, 1848.16478, 2908.49916, 1854.20724,
                  1626.45768, 1184.04856, 2170.5419, 2153.57412]
attention85 = [1465.65126, 2084.48024, 1314.24494, 1913.65402, 1863.05772, 1690.73514, 2445.04462, 1550.31952,
               1653.45354, 1163.65932, 2018.3009, 2063.29162]
no_attention85 = [1781.2129, 2390.25942, 1680.91888, 1818.18962, 2055.70298, 1827.8304, 2833.04716, 1385.38606,
                  1521.48944, 1311.35352, 1935.28874, 1847.03856]

total_width, n = 0.4, 2
width = total_width / n
x1 = np.array(range(len(attention67)))
x1 = x1 - width/2
x2 = [x + width for x in x1]
x3 = [x + width for x in x2]
x4 = [x + width for x in x3]
x1 = list(x1)

plt.figure()
plt.bar(x1, attention67, width=width, label='67% w/o attention', color=['#F2E1A0'], edgecolor='black')  # darkseagreen
plt.bar(x2, no_attention67, width=width, label='67% w/ attention', color=['#2A5ABB'], edgecolor='black')
plt.bar(x3, attention85, width=width, label='85% w/o attention', color=['#0CB599'], edgecolor='black')  # darkseagreen
plt.bar(x4, no_attention85, width=width, label='85% w/ attention', color=['#808080'], edgecolor='black')
# gold
plt.xticks([x + width for x in range(len(x1))],
           ['U1', 'U2', 'U3', 'U4', 'U5', 'U6', 'U7', 'U8', 'U9', 'U10', 'U11', 'U12'])
y_major_locator = MultipleLocator(0.1)
ax = plt.gca()
# ax.yaxis.set_major_locator(y_major_locator)
plt.ylim(0, 3000)
plt.ylabel('RMS',fontproperties='Times New Roman', size=16, weight='bold')
plt.yticks(fontproperties='Times New Roman', size=14, weight='bold')  # 设置大小及加粗
plt.xticks(fontproperties='Times New Roman', size=16, weight='bold')
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
plt.savefig("fig/rms_user.jpg", dpi=500, bbox_inches="tight")
plt.show()
