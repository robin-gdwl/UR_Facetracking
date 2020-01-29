from datetime import datetime
from matplotlib import pyplot
from matplotlib.animation import FuncAnimation
from random import randrange
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

x_data, y_data = [], []

figure = pyplot.figure()
ax = figure.add_subplot(111, projection='3d')
pyplot.ylim(-100,100)
pyplot.xlim(-100,100)
ax.set_zlim(-100,100)

#scatter = pyplot.scatter(x_data, y_data, 0, marker="o")

def update(frame):
    x_data.append(randrange(-100, 100))
    y_data.append(randrange(0, 100))
    ax.scatter(x_data, y_data, 0)

    #figure.gca().relim()
    #figure.gca().autoscale_view()
    #return scatter

animation = FuncAnimation(figure, update, interval=200)
pyplot.show()