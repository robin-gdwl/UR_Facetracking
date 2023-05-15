from pyModbusTCP.client import ModbusClient
from pyModbusTCP import utils
import time
import random 
from math import radians as rad

TCP_x = 400
TCP_y = 401
TCP_z = 402
#TCP_x = 400

def unsigned(a):
    if a > 32767:
        a = a - 65535
    else:
        a = a
    return a

def create_unsigned(a):
    if a <0: 
        a = a + 65535
    return a 

try:
    #connect via modbus
    robot = ModbusClient(host="192.168.178.45", port=502, auto_open=True, debug=False)
    print("connected",c.open())
except:
    print("Error with host or port params")



i = 0 
while True:
    # Read the value of the specified register from the UR10
    register_value = robot.read_holding_registers(TCP_x)
    reg_var = unsigned(register_value[0])
    print("TCP-X:",reg_var)

    register_value = robot.read_holding_registers(TCP_y)
    reg_var = unsigned(register_value[0])
    print("TCP-Y:",reg_var)

    register_value = robot.read_holding_registers(TCP_z)
    reg_var = unsigned(register_value[0])
    print("TCP-Z:",reg_var)

    if robot.write_single_register(128, 65530):
            print("write ok")

    while True: 
        j1 = random.randrange(70,100)
        j2 = random.randrange(create_unsigned(-90),create_unsigned(-70))
        j3 = random.randrange(70,100)
        j4 = random.randrange(create_unsigned(-185),create_unsigned(-175))
        j5 = random.randrange(create_unsigned(-80),create_unsigned(-70))
        j6 = random.randrange(0,5)


        if robot.write_single_register(128, j1):
            print("write ok")
        if robot.write_single_register(129, j2):
            print("write ok")
        if robot.write_single_register(130, j3):
            print("write ok")
        if robot.write_single_register(131, j4):
            print("write ok")
        if robot.write_single_register(132, j5):
            print("write ok")
        if robot.write_single_register(133, j6):
            print("write ok")




    time.sleep(1)
    i+=1
    print("--------"*2)