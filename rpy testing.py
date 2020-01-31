import URBasic

host = '172.22.4.105'   #E.g. a Universal Robot offline simulator, please adjust to match your IP
acc = 0.9
vel = 0.9
print("initialising robot")
robotModel = URBasic.robotModel.RobotModel()
robot = URBasic.urScriptExt.UrScriptExt(host=host,robotModel=robotModel)
robot.reset_error()
print("robot initialised")


rpy = [3.14, 1.57, 0]
rot_vec = robot.rpy2rotvec(rpy)
print("______ ")

print(rot_vec)