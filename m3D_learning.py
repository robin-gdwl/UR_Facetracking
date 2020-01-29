import math3d as m3d
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

import numpy


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
plt.ylim(-100,100)
plt.xlim(-100,100)
ax.set_zlim(-100,100)

plt.show()