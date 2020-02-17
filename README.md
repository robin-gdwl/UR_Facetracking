# UR_Facetracking

Currently under developement as of 2020-02-05

### **Use at your own risk.**
(test with URsim before connecting a real robot)


 Universal Robot Realtime Facetracking with Python3 and OpenCV
 Uses the [UR - RTDE](https://www.universal-robots.com/how-tos-and-faqs/how-to/ur-how-tos/real-time-data-exchange-rtde-guide-22229/)- protocol to send continuous updates to a Robot for smooth continued motion.

Demonstration: https://youtu.be/HHb-5dZoPFQ

![Universal Robot Realtime Face Tracking Python](UR_Facetracking_Demo.jpg)


Run Face_tracking01.py to run the face tracking.
Developed for a Universal Robot UR5 CB running Polyscope 3.7

heavily relies on this repository:
https://github.com/Mandelbr0t/UniversalRobot-RealtimeControl
which builds ontop of:
https://bitbucket.org/RopeRobotics/ur-interface/src/master/

License: GPLv3



#### Requirements:
Universal Robot or URsim-software (tested with UR5cb running polyscope 3.7)

python 3.x

**Python libraries:**  
opencv-python 4.1.2.30   
numpy 1.18.1  
math3d 3.3.5  
imutils 0.5.3  

when used with a raspberry pi and picam:  
picamera  
...  


TODO
- [ ] more testing
- [x] demonstration video
- [ ] explanation video
- [ ] License clarification
- [x] cleanup
- [ ] refactoring
- [ ] documentation
- [x] comments
- [ ] add links to libraries and other resources
