import URBasic
import math
import sys
import cv2
import time
from imutils.video import VideoStream

"""# imports for visualisation purposes
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
plt.ylim(-100,100)
plt.xlim(-100,100)
ax.set_zlim(-100,100)
"""

# Setup and variables
RASPBERRY_BOOL = False
if sys.platform == "linux":
    import picamera
    from picamera.array import PiRGBArray
    RASPBERRY_BOOL = True

video_resolution = (1080, 720)
video_midpoint = (
    int(video_resolution[0]/2),
    int(video_resolution[1]/2))
print(video_midpoint)
detection_model_path = 'haarcascade_files/haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(detection_model_path)
vs = VideoStream(usePiCamera= RASPBERRY_BOOL,
                              resolution=video_resolution,
                              framerate = 16,
                              meter_mode = "average",
                              exposure_mode ="auto",
                              shutter_speed = 3800,
                              exposure_compensation = -1,
                              rotation = 180).start()
time.sleep(0.2)

angle_multiplier = 0.01

"""host = '192.168.0.113'   #E.g. a Universal Robot offline simulator, please adjust to match your IP
acc = 0.9
vel = 0.9
robotModel = URBasic.robotModel.RobotModel()
robot = URBasic.urScriptExt.UrScriptExt(host=host,robotModel=robotModel)
robot.reset_error()"""


def find_faces_in_frame(frame):
    list_of_facepos = []
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    #TODO: make it so that it is not as important if a face is not recognised in 1 or 2 frames
    for (x, y, w, h) in faces:
        face_center = (int(x+ w*0.5), int(y+ h*0.5))
        position_from_center = [face_center[0] - video_midpoint[0] , face_center[1] - video_midpoint[1]]


        list_of_facepos.append(position_from_center)
        # draw information about the face on the frame
        cv2.putText(frame, str(position_from_center), face_center, 0, 1, (0,200,0))
        cv2.line(frame, video_midpoint, face_center, (0,200,0), 5)
        cv2.circle(frame, face_center, 4, (0,200,0), 3)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 3)
    #print(list_of_facepos)
    return list_of_facepos ,frame

def draw_angle_vis(frame, angles):
    center_1 = (100,100)
    center_2 = (100,300)
    radius = 100
    s = angles[0]
    t = angles[1]
    cv2.circle(frame, center_1, radius, (0, 250, 0), 6)
    cv2.circle(frame, center_2, radius, (0, 0, 250), 6)

    #s-angle
    sx = center_1[0] + (radius * math.cos(s))
    sy = center_1[1] + (radius * math.sin(s))
    cv2.circle(frame, (int(sx),int(sy)), 4, (200, 200, 0), 5)
    cv2.line

    #t-angle
    tx = center_2[0] + (radius * math.cos(t))
    ty = center_2[1] + (radius * math.sin(t))
    cv2.circle(frame, (int(tx), int(ty)), 4, (200, 200, 0), 5)

    return frame

def show_frame(frame):
    cv2.imshow('img', frame)
    k = cv2.waitKey(6) & 0xff

"""def convert_robot_target_to_angles(rob_target):
    x_val = rob_target[0]
    y_val = rob_target[1]
    t = math.asin(x_val / sphere_radius)
    s = math.asin(y_val / sphere_radius)

    return t,s"""
def check_max_angles(angles):
    s_t = [0,0]
    print("angles before conversion: ", angles)

    if -max_s <= angles[0] <= max_s:  # checks if the resulting angle would be outside of max viewangle
        s_t[0] = angles[0]
    elif -max_s > angles[0]:
        print("0 - angle too small")
        s_t[0] = -max_s
    elif max_s < angles[0]:
        print("0 - angle too big")
        s_t[0] = max_s
    else:
        raise Exception(" angle 0 is wrong somehow:", angles[0] , -max_s, max_s)


    if -max_t <= angles[1] <= max_t:  # checks if the resulting angle would be outside of max viewangle
        s_t[1] = angles[1]
    elif -max_t > angles[1]:
        print("1 - angle too small")
        s_t[1] = -max_t
    elif max_t < angles[1]:
        print("1 - angle too big")
        s_t[1] = max_t
    else:
        raise Exception(" angle 1 is wrong somehow", angles[1], max_t)
    print("angles after conversion: ", s_t)
    return s_t

def convert_angles_to_xyz(t,s):

    x = sphere_radius * math.cos(s) * math.sin(t)
    y = sphere_radius * math.sin(s) * math.sin(t)
    z = sphere_radius * math.cos(t)

    return [x, y, z]


# Actual Process
# Move robot to 0 Position
#robot.movej(q=[-3.14,-1.,0.5, -1.,-1.5,0], a=acc, v=vel)
#time.sleep(20)

robot_position = [0,0]  # Robot Position as Angles in Radians
sphere_center = [0,0,0]
sphere_radius = 100
video_asp_ratio  = video_resolution[0] / video_resolution[1]  # Aspect ration of each frame
video_viewangle_hor = math.radians(45)  # Camera FOV (field of fiew) angle in radians in horizontal direction
video_viewangle_vert = video_viewangle_hor / video_asp_ratio  #  Camera FOV (field of fiew) angle in radians in vertical direction

angle_per_pixel = video_viewangle_hor / video_resolution[0]  # how big of an angle is needed to cover a distance of pixels
print(video_viewangle_hor)

max_t = math.radians(180)
max_s = math.radians(180)
print("max t,s  :  ", max_t, max_s)

#ax.scatter(0,0,0, marker="")
i = 0
#def runloop(i,robot_position):


while True:
    frame = vs.read()
    face_positions, new_frame = find_faces_in_frame(frame)
    face = [0,0]  # TODO: make sure this doesnt block a wandering lookaround
    if len(face_positions) > 0:

        face = list(face_positions[0])  # TODO: find way of making the selected face persistent
        print(face)
        face[:] = [x * angle_per_pixel for x in face]
        print(face)

        robot_target_xy = [a+b for a,b in zip(robot_position,face)]
        print(robot_target_xy)

        robot_target_angles = check_max_angles(robot_target_xy)
        print(robot_target_angles)

        robot_position = robot_target_angles

        x,y,z = convert_angles_to_xyz(robot_target_angles[0],robot_target_angles[1])
        #ax.scatter(x, y, z, marker="o")
        i+=1
        #plt.gcf().show()
        print()

    frame_with_vis = draw_angle_vis(new_frame,robot_position)

    show_frame(frame_with_vis)

