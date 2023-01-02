import math

import matplotlib.pyplot as plt

x = []
dx = -20
while dx <= 20:
    x.append(dx)
    dx += 0.1


def ReLU(x):
    return max(x, 0)


px = list(x)
py = [ReLU(xv) for xv in x]


plt.plot(px, py)
ax = plt.gca()
ax.spines["right"].set_color("none")
ax.spines["top"].set_color("none")
ax.xaxis.set_ticks_position("bottom")
ax.spines["bottom"].set_position(("data", 0))
ax.yaxis.set_ticks_position("left")
ax.spines["left"].set_position(("data", 0))
plt.show()
