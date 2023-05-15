import urx

import urx
import time
from math import radians as rad

rob = urx.Robot("192.168.178.45")
rob.set_tcp((0, 0, 0.1, 0, 0, 0))
rob.set_payload(2, (0, 0, 0.1))
time.sleep(0.2)  #leave some time to robot to process the setup commands
a = 0.5
v = 0.5
#rob.movej((rad(100), rad(-80), rad(90), rad(-180), rad(-80), rad(10)), a, v)
#rob.movej((rad(70), rad(-80), rad(90), rad(-180), rad(-80), rad(10)), a, v)

prog = '''def test():
    textmsg("test")
end'''
rob.send_program(prog)

prog = '''def realtime_control():
    textmsg("waiting")
    textmsg("done waiting")
    while (True):        
        new_pose = [d2r(read_port_register(128)),
                    d2r(read_port_register(129)),
                    d2r(read_port_register(130)),
                    d2r(read_port_register(131)),
                    d2r(read_port_register(132)),
                    d2r(read_port_register(133))]
           
        servoj(new_pose, t=1.5, lookahead_time= 0.1, gain=350)
            
        sync()
    end
end
'''
prog2 = '''def realtime_control():
    textmsg("waiting")
    wait(5)
    textmsg("done waiting")
    while (True):        
        new_pose = p[read_port_register(128),
                    read_port_register(129),
                    read_port_register(130),
                    read_port_register(131),
                    read_port_register(132),
                    read_port_register(133)]
           
        servoj(get_inverse_kin(new_pose), t=0.2, lookahead_time= 0.1, gain=350)
            
        sync()
    end
end
'''

rob.send_program(prog)

