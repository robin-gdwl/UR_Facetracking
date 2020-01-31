import math3d as m3d
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

import numpy


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
plt.ylim(-100,100)
plt.xlim(-100,100)
ax.set_zlim(-100,100)
ax.scatter(0,0,0, marker="^")

sphere_radius = 50

def convert_angles_to_xyz(t,s):

    x = sphere_radius * math.cos(s) * math.sin(t)
    y = sphere_radius * math.sin(s) * math.sin(t)
    z = sphere_radius * math.cos(t) + 0

    #ax.scatter(x, y, z, marker="o")

    return [x, y, z]

#def rotate(transform, coords):


angles = [-90,-60,-45,-10,0,10,20,30,45,60,90,120,180]
rad_angles = [math.radians(x) for x in angles] #nice
list_of_coords = []
for ang in rad_angles:
    ang_coord= convert_angles_to_xyz(ang,0)
    list_of_coords.append(ang_coord)

rot_orient = m3d.Orientation.new_axis_angle((0, 1, 0), math.pi/2)
rot_vector = m3d.Vector(0,0,20)
rotation_transform = m3d.Transform(rot_orient,rot_vector)

for coord in list_of_coords:
    vec = m3d.Vector(coord)
    new_vec = rotation_transform * vec
    ax.scatter(new_vec[0], new_vec[1], new_vec[2], marker="o")


print(rad_angles)
plt.show()