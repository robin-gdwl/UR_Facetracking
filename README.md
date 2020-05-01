# UR_Facetracking

Currently under developement as of 2020-02-05

### **Use at your own risk.**
(test with URsim before connecting a real robot)


 Universal Robot Realtime Facetracking with Python3 and OpenCV
 Uses the [UR - RTDE](https://www.universal-robots.com/how-tos-and-faqs/how-to/ur-how-tos/real-time-data-exchange-rtde-guide-22229/)- protocol to send continuous updates to a Robot for smooth continued motion.

**Demonstration:** https://youtu.be/HHb-5dZoPFQ

![Universal Robot Realtime Face Tracking Python](UR_Facetracking_Demo.jpg)


Run Face_tracking01.py to run the face tracking.
Developed for a Universal Robot UR5 CB running Polyscope 3.7

heavily relies on this repository:
https://github.com/Mandelbr0t/UniversalRobot-RealtimeControl
which builds ontop of:
https://bitbucket.org/RopeRobotics/ur-interface/src/master/

License: GPLv3


___
#### Requirements:
Universal Robot or [URsim-software](https://www.universal-robots.com/download/?option=45440#section16597) (tested with UR5cb running polyscope 3.7)

python 3.x

**Python libraries:**  
[opencv-python 4.1.2.30](https://pypi.org/project/opencv-python/)   
[numpy 1.18.1](https://numpy.org/)  
[math3d 3.3.5](https://gitlab.com/morlin/pymath3d)   
[imutils 0.5.3](https://github.com/jrosebr1/imutils)  

when used with a raspberry pi and picam:  
[picamera](https://picamera.readthedocs.io/en/release-1.13/)  
...  
___
#### Notes:
When using a linnux distribution or raspberry pi the software will try to use a picamera. If you want to use a webcam instead change it [here](https://github.com/robin-gdwl/UR_Facetracking/blob/master/Face_tracking01.py#L26).

___

TODO
- [ ] more testing
- [x] demonstration video
- [ ] explanation video
- [ ] License clarification
- [x] cleanup
- [ ] refactoring
- [ ] documentation
- [x] comments
- [x] add links to libraries and other resources
