import URBasic
import sys
import cv2
import time
from imutils.video import VideoStream




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

"""host = '192.168.0.113'   #E.g. a Universal Robot offline simulator, please adjust to match your IP
acc = 0.9
vel = 0.9
robotModel = URBasic.robotModel.RobotModel()
robot = URBasic.urScriptExt.UrScriptExt(host=host,robotModel=robotModel)
robot.reset_error()"""



def find_faces_in_frame():
    list_of_facepos = []
    frame = vs.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    for (x, y, w, h) in faces:
        face_center = (int(x+ w*0.5), int(y+ h*0.5))
        position_from_center = [face_center[0] - video_midpoint[0] , face_center[1] - video_midpoint[1]]


        list_of_facepos.append(face_center)
        cv2.line(frame, video_midpoint, face_center, (0,200,0), 5)
        cv2.circle(frame, face_center, 4, (0,200,0), 3)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 3)
    cv2.imshow('img', frame)
    k = cv2.waitKey(30) & 0xff
    print(list_of_facepos)
    return list_of_facepos



# Actual Process
# Move robot to 0 Position
#robot.movej(q=[-3.14,-1.,0.5, -1.,-1.5,0], a=acc, v=vel)
#time.sleep(20)

robot_position = [0,0]

while True:

    face_positions = find_faces_in_frame()
    if len(face_positions) > 0:
        face = face_positions[0]




