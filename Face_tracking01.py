import URBasic
import math
import numpy as np
import sys
import cv2
import time
import imutils
from imutils.video import VideoStream
from UR5Kinematics import Kinematic
import math3d as m3d

"""# imports for visualisation purposes
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import"""

"""fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
plt.ylim(-2,2)
plt.xlim(-2,2)
ax.set_zlim(-2,2)"""


# Setup and variables
RASPBERRY_BOOL = False
if sys.platform == "linux":
    import picamera
    from picamera.array import PiRGBArray
    RASPBERRY_BOOL = True

video_resolution = (400, 400)
video_midpoint = (
    int(video_resolution[0]/2),
    int(video_resolution[1]/2))
print(video_midpoint)
detection_model_path = 'haarcascade_files/haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(detection_model_path)
vs = VideoStream(src= 1 , usePiCamera= RASPBERRY_BOOL,
                              resolution=video_resolution,
                              framerate = 10,
                              meter_mode = "average",
                              exposure_mode ="auto",
                              shutter_speed = 3800,
                              exposure_compensation = -1,
                              rotation = 0).start()
time.sleep(0.2)

net = cv2.dnn.readNetFromCaffe("MODELS/deploy.prototxt.txt", "MODELS/res10_300x300_ssd_iter_140000.caffemodel")

angle_multiplier = 0.01

host = '192.168.178.20'
#host = "10.211.55.5"#E.g. a Universal Robot offline simulator, please adjust to match your IP
acc = 0.8
vel = 0.6
print("initialising robot")
robotModel = URBasic.robotModel.RobotModel()
robot = URBasic.urScriptExt.UrScriptExt(host=host,robotModel=robotModel)
robot.reset_error()
print("robot initialised")
time.sleep(1)

import cProfile, pstats, io


def profile(fnc):
    """A decorator that uses cProfile to profile a function"""

    def inner(*args, **kwargs):
        pr = cProfile.Profile()
        pr.enable()
        retval = fnc(*args, **kwargs)
        pr.disable()
        s = io.StringIO()
        sortby = 'cumulative'
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        ps.print_stats()
        print(s.getvalue())
        return retval

    return inner

#@profile
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


@profile
def find_faces_dnn(image):
    frame = image
    #frame = imutils.resize(frame, width=400)

    # grab the frame dimensions and convert it to a blob
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                 (300, 300), (104.0, 177.0, 123.0))

    # pass the blob through the network and obtain the detections and
    # predictions
    net.setInput(blob)
    detections = net.forward()
    face_centers = []
    # loop over the detections
    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with the
        # prediction
        confidence = detections[0, 0, i, 2]

        # filter out weak detections by ensuring the `confidence` is
        # greater than the minimum confidence
        if confidence < 0.5:
            continue

        # compute the (x, y)-coordinates of the bounding box for the
        # object
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")


        # draw the bounding box of the face along with the associated
        # probability
        text = "{:.2f}%".format(confidence * 100)
        y = startY - 10 if startY - 10 > 10 else startY + 10

        face_center = (int(startX + (endX - startX) / 2), int(startY + (endY - startY) / 2))
        position_from_center = (face_center[0] - video_midpoint[0], face_center[1] - video_midpoint[1])
        face_centers.append(position_from_center)

        cv2.rectangle(frame, (startX, startY), (endX, endY),
                      (0, 0, 255), 2)
        cv2.putText(frame, text, (startX, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
        #cv2.putText(frame, str(position_from_center), face_center, 0, 1, (0, 200, 0))
        cv2.line(frame, video_midpoint, face_center, (0, 200, 0), 5)
        cv2.circle(frame, face_center, 4, (0, 200, 0), 3)



    return face_centers, frame

    # show the output frame
    #cv2.imshow("Frame", frame)
    #key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    """if key == ord("q"):
        return False"""

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

#@profile
def show_frame(frame):
    cv2.imshow('img', frame)
    k = cv2.waitKey(6) & 0xff

"""def convert_robot_target_to_angles(rob_target):
    x_val = rob_target[0]
    y_val = rob_target[1]
    t = math.asin(x_val / sphere_radius)
    s = math.asin(y_val / sphere_radius)

    return t,s"""
"""def check_max_angles(angles):
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
"""
"""def convert_angles_to_xyz(t,s):

    x = sphere_radius * math.cos(s) * math.sin(t)
    y = sphere_radius * math.sin(s) * math.sin(t)
    z = sphere_radius * math.cos(t) + 0

    return [x, y, z]"""


def convert_rpy(angles):

    # This is very stupid:
    # For some reason this doesnt work if exactly  one value = 0
    # the following simply make it a very small value if that happens
    # I do not understand the math behind this well enough to create a better solution
    zeros = 0
    zero_pos = None
    for i,ang in enumerate(angles):
        if ang == 0 :
            zeros += 1
            zero_pos = i
    if zeros == 1:
        #logging.debug("rotation value" + str(zero_pos+1) +"is 0 a small value is added")
        angles[zero_pos] = 1e-6

    roll = angles[0]
    pitch = angles[1]
    yaw = angles[2]

    # print ("roll = ", roll)
    # print ("pitch = ", pitch)
    # print ("yaw = ", yaw)
    # print ("")

    for ang in angles:
        # print(ang % np.pi)
        pass

    if roll == pitch == yaw:

        if roll % np.pi == 0:
            rotation_vec = [0, 0, 0]
            return rotation_vec

    yawMatrix = np.matrix([
    [math.cos(yaw), -math.sin(yaw), 0],
    [math.sin(yaw), math.cos(yaw), 0],
    [0, 0, 1]
    ])
    # print("yawmatrix")
    # print(yawMatrix)

    pitchMatrix = np.matrix([
    [math.cos(pitch), 0, math.sin(pitch)],
    [0, 1, 0],
    [-math.sin(pitch), 0, math.cos(pitch)]
    ])
    # print("pitchmatrix")
    # print(pitchMatrix)

    rollMatrix = np.matrix([
    [1, 0, 0],
    [0, math.cos(roll), -math.sin(roll)],
    [0, math.sin(roll), math.cos(roll)]
    ])
    # print("rollmatrix")
    # print(rollMatrix)

    R = yawMatrix * pitchMatrix * rollMatrix
    # print("R")
    # print(R)

    theta = math.acos(((R[0, 0] + R[1, 1] + R[2, 2]) - 1) / 2)
    # print("theta = ",theta)
    multi = 1 / (2 * math.sin(theta))
    # print("multi = ", multi)


    rx = multi * (R[2, 1] - R[1, 2]) * theta
    ry = multi * (R[0, 2] - R[2, 0]) * theta
    rz = multi * (R[1, 0] - R[0, 1]) * theta

    rotation_vec = [rx,ry,rz]
    # print(rx, ry, rz)
    return rotation_vec

def check_max_xy(xy_coord):
    x_y = [0, 0]
    #print("xy before conversion: ", xy_coord)

    if -max_x <= xy_coord[0] <= max_x:  # checks if the resulting angle would be outside of max viewangle
        x_y[0] = xy_coord[0]
    elif -max_x > xy_coord[0]:
        #print("0 - angle too small")
        x_y[0] = -max_x
    elif max_x < xy_coord[0]:
        #print("0 - angle too big")
        x_y[0] = max_x
    else:
        raise Exception(" x is wrong somehow:", xy_coord[0], -max_x, max_x)

    if -max_y <= xy_coord[1] <= max_y:  # checks if the resulting angle would be outside of max viewangle
        x_y[1] = xy_coord[1]
    elif -max_y > xy_coord[1]:
        #print("y too small")
        x_y[1] = -max_y
    elif max_y < xy_coord[1]:
        #print("y too big")
        x_y[1] = max_y
    else:
        raise Exception(" y is wrong somehow", xy_coord[1], max_y)
    #print("xy after conversion: ", x_y)
    return x_y

def set_lookorigin():
    position = robot.get_actual_tcp_pose()
    orig = m3d.Transform(position)
    return orig

@profile
def move_to_face(list_of_positions,robot_pos):
    #print("moving")
    face_from_center = list(list_of_positions[0])  # TODO: find way of making the selected face persistent
    # print(face_from_center)
    prev_robot_pos = robot_pos
    scaled_face_pos = [c * m_per_pixel for c in face_from_center]

    robot_target_xy = [a + b for a, b in zip(prev_robot_pos, scaled_face_pos)]
    # print("..", robot_target_xy)

    robot_target_xy = check_max_xy(robot_target_xy)
    prev_robot_pos = robot_target_xy

    x = robot_target_xy[0]
    y = robot_target_xy[1]
    z = 0
    xyz_coords = m3d.Vector(x, y, z)

    x_pos_perc = x / max_x
    y_pos_perc = y / max_y

    x_rot = x_pos_perc * hor_rot_max
    y_rot = y_pos_perc * vert_rot_max * -1
    # print(x_rot, y_rot)

    tcp_rotation_rpy = [y_rot, x_rot, 0]
    # tcp_rotation_rvec = convert_rpy(tcp_rotation_rpy)
    tcp_orient = m3d.Orientation.new_euler(tcp_rotation_rpy, encoding='xyz')
    # print(tcp_orient)
    position_vec_coords = m3d.Transform(tcp_orient, xyz_coords)

    oriented_xyz = origin * position_vec_coords
    oriented_xyz_coord = oriented_xyz.get_pose_vector()

    coordinates = oriented_xyz_coord
    # print("coordinates:", coordinates)
    # print("_______"*20)

    qnear = robot.get_actual_joint_positions()
    # print(list(qnear))
    next_pose = coordinates
    # print(next_pose)
    #print("sending to robot")
    robot.set_realtime_pose(next_pose)
    #print("sent")
    return prev_robot_pos

# Actual Process
# Move robot to 0 Position
"""robot.movej(q=[
    math.radians(-86.62),
    math.radians(-102.94),
    math.radians(103),
    math.radians(179.94),
    math.radians(-93.38),
    0], a=acc, v=vel)"""

robot.movej(q=(math.radians(-218),math.radians(-63),math.radians(-93),math.radians(-20),math.radians(88),math.radians(0)), a=acc, v=vel )

robot_position = [0,0]

video_asp_ratio  = video_resolution[0] / video_resolution[1]  # Aspect ration of each frame
video_viewangle_hor = math.radians(25)  # Camera FOV (field of fiew) angle in radians in horizontal direction
video_viewangle_vert = video_viewangle_hor / video_asp_ratio  #  Camera FOV (field of fiew) angle in radians in vertical direction
m_per_pixel = 00.00002

#ax.scatter(10,0,0, marker="^")
i = 0
#def runloop(i,robot_position):
kinematics = Kinematic()
max_x = 0.2
max_y = 0.2
hor_rot_max = math.radians(50)
vert_rot_max = math.radians(25)

origin = set_lookorigin()

robot.init_realtime_control()
time.sleep(1)
try:
    print("starting loop")
    while True:
        #print("looping")
        #timer = time.time()
        frame = vs.read()
        #frame = cv2.rotate(frame,)
        #face_positions, new_frame = find_faces_in_frame(frame)
        face_positions, new_frame = find_faces_dnn(frame)
        #face_from_center = [0,0]  # TODO: make sure this doesnt block a wandering lookaround
        #print("frame")
        #show_frame(new_frame)
        if len(face_positions) > 0:
            robot_position = move_to_face(face_positions,robot_position)


        #frame_with_vis = draw_angle_vis(new_frame,robot_position)

        #show_frame(new_frame)
        #new_time = time.time() -timer
        #print(new_time)

    print("exiting loop")
except KeyboardInterrupt:
    print("closing robot connection")
    robot.close()

except:
    robot.close()