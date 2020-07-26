import numpy as np
import math
import matplotlib.pyplot as plt


x = np.arange(0, 40, 0.1)
y1 = []
y2 = []
y3 = []

for t in x:
    t = t/18.0
    y_1 = ((math.exp(t) - math.exp(-t))/(math.exp(t) + math.exp(-t)))*25
    y1.append(y_1)

for t in x:
    t = t/5.0
    y_1 = ((math.exp(t) - math.exp(-t))/(math.exp(t) + math.exp(-t)))*25
    y2.append(y_1)

for t in x:
    y_1 = t
    y3.append(y_1)

plt.plot(x, y1, label="sigmoid")
plt.plot(x, y2, label="sigmoid-2")
plt.plot(x, y3, label="line")
plt.xlabel("x")
plt.ylabel("y")
plt.ylim(0, 32)
plt.legend()
plt.show()
