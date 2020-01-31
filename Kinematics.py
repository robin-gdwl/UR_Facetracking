#!/usr/bin/python2

# UR5/UR10 Inverse Kinematics - Ryan Keating Johns Hopkins University


# ***** lib
import numpy as np
from numpy import linalg

import math3d as m3d
import cmath
import math
from math import cos as cos
from math import sin as sin
from math import atan2 as atan2
from math import acos as acos
from math import asin as asin
from math import sqrt as sqrt
from math import pi as pi

class Kinematic:

    def __init__(self, coordinates=[], tool_length=0):
        #self.coordinates = coordinates
        #self.poses = []
        #self.ischecked = False

        self.mat = np.matrix
        # ****** Coefficients ******
        """self.d1 = 0.1273
        self.a2 = -0.612
        self.a3 = -0.5723
        self.a7 = 0.075
        self.d4 = 0.163941
        self.d5 = 0.1157
        self.d6 = 0.0922"""

        self.d1 = 0.089159
        self.a2 = -0.425
        self.a3 = -0.39225
        self.a7 = 0.075
        self.d4 = 0.10915
        self.d5 = 0.09465
        #self.d6 = 0.0823 + tool_length / 1000
        self.d6 = 0.0823

        self.d = self.mat([0.089159, 0, 0, 0.10915, 0.09465, 0.0823])  # ur5
        #self.d = self.mat([0.1273, 0, 0, 0.163941, 0.1157, 0.0922])  # ur10 mm
        self.a =self.mat([0 ,-0.425 ,-0.39225 ,0 ,0 ,0])  # ur5
        #self.a = self.mat([0, -0.612, -0.5723, 0, 0, 0])  # ur10 mm
        self.alph = self.mat([math.pi/2, 0, 0, math.pi/2, -math.pi/2, 0])  # ur5
        #self.alph = self.mat([pi / 2, 0, 0, pi / 2, -pi / 2, 0])  # ur10

    # ************************************************** FORWARD KINEMATICS

    def select(self,q_sols, q_d, w=[10, 9, 8, 7, 1, 1]):
        """Select the optimal solutions among a set of feasible joint value
           solutions.
        Args:
            q_sols: A set of feasible joint value solutions (unit: radian)
            q_d: A list of desired joint value solution (unit: radian)
            w: A list of weight corresponding to robot joints
        Returns:
            A list of optimal joint value solution.
        """

        error = []
        for q in q_sols:
            error.append(sum([w[i] * (q[i] - q_d[i]) ** 2 for i in range(6)]))

        return q_sols[error.index(min(error))]

    def get_closest_solution(self, q_sols, q_d):

        closest_sol_index = 0
        closest_diffs = 0.0
        diffs = []
        for q in q_sols: #qsols is 8 different solutions with 6 values each
            current_diffs = []
            for i, joint_val in enumerate(q):
                current_diff = self.squared_diff(q_d[i],joint_val)
                current_diffs.append(current_diff)
            current = sum(current_diffs)
            diffs.append(current)

        return q_sols[diffs.index(min(diffs))]

    def get_absolute_joint(self, joint):
        abs_joint = abs(joint)
        result = abs_joint - math.floor(abs_joint / (2*pi)) * (2*pi)
        if result > pi:
            result = result - (2*pi)
        result *= np.sign(joint)
        return result

    def get_absolute_joints(self, joints, prev_joints):

        closest_joints = []

        for i in range(len(joints)):
            prev_joint = self.get_absolute_joint(prev_joints[i])
            joint = self.get_absolute_joint(joints[i])
            difference = joint - prev_joint
            abs_difference = abs(difference)
            if abs_difference > pi:
                difference = (abs_difference- (pi*2)) * np.sign(difference)
            closest_joints.append(prev_joints[i] + difference)
        return closest_joints

    def squared_diff(self, a,b):
        difference = abs(a-b)
        if difference > pi:
            difference = pi * 2 -difference

        return difference*difference

    def AH(self, n, th, c):
        T_a = self.mat(np.identity(4), copy=False)
        T_a[0, 3] = self.a[0, n - 1]
        T_d = self.mat(np.identity(4), copy=False)
        T_d[2, 3] = self.d[0, n - 1]

        Rzt = self.mat([[cos(th[n - 1, c]), -sin(th[n - 1, c]), 0, 0],
                   [sin(th[n - 1, c]), cos(th[n - 1, c]), 0, 0],
                   [0, 0, 1, 0],
                   [0, 0, 0, 1]], copy=False)

        Rxa = self.mat([[1, 0, 0, 0],
                   [0, cos(self.alph[0, n - 1]), -sin(self.alph[0, n - 1]), 0],
                   [0, sin(self.alph[0, n - 1]), cos(self.alph[0, n - 1]), 0],
                   [0, 0, 0, 1]], copy=False)

        A_i = T_d * Rzt * T_a * Rxa

        return A_i


    def HTrans(self, th, c):
        A_1 = self.AH(1, th, c)
        A_2 = self.AH(2, th, c)
        A_3 = self.AH(3, th, c)
        A_4 = self.AH(4, th, c)
        A_5 = self.AH(5, th, c)
        A_6 = self.AH(6, th, c)

        T_06 = A_1 * A_2 * A_3 * A_4 * A_5 * A_6

        return T_06


    # ************************************************** INVERSE KINEMATICS

    def invKine(self, desired_pos, start_pos):  # T60
        des_pos = desired_pos
        des_pos = m3d.Transform(des_pos)
        des_pos = des_pos.matrix
        #print("desired pos: ", desired_pos)
        th = self.mat(np.zeros((6, 8)))
        P_05 = (des_pos * self.mat([0, 0, -self.d6, 1]).T - self.mat([0, 0, 0, 1]).T)
        #print(self.d4 / sqrt(P_05[2 - 1, 0] * P_05[2 - 1, 0] + P_05[1 - 1, 0] * P_05[1 - 1, 0]))
        # **** theta1 ****

        R = (P_05[2 - 1, 0] * P_05[2 - 1, 0] + P_05[1 - 1, 0] * P_05[1 - 1, 0])
        psi = atan2(P_05[2 - 1, 0],  P_05[1 - 1, 0])
        phi = acos(self.d4 / sqrt(P_05[2 - 1, 0] * P_05[2 - 1, 0] + P_05[1 - 1, 0] * P_05[1 - 1, 0]))

        # The two solutions for theta1 correspond to the shoulder
        # being either left or right
        # pi / 2 = 90deg
        th[0, 0:4] = pi / 2 + psi + phi
        th[0, 4:8] = pi / 2 + psi - phi
        th = th.real
        #print("th___")
        #print(th)

        # **** theta5 ****

        cl = [0, 4]  # wrist up or down
        for i in range(0, len(cl)):
            c = cl[i]
            T_10 = linalg.inv(self.AH(1, th, c))
            T_16 = T_10 * des_pos
            th[4, c:c + 2] = + acos((T_16[2, 3] - self.d4) / self.d6);
            th[4, c + 2:c + 4] = - acos((T_16[2, 3] - self.d4) / self.d6);

        th = th.real

        # **** theta6 ****
        # theta6 is not well-defined when sin(theta5) = 0 or when T16(1,3), T16(2,3) = 0.

        cl = [0, 2, 4, 6]
        for i in range(0, len(cl)):
            c = cl[i]
            T_10 = linalg.inv(self.AH(1, th, c))
            T_16 = linalg.inv(T_10 * des_pos)
            th[5, c:c + 2] = atan2((-T_16[1, 2] / sin(th[4, c])), (T_16[0, 2] / sin(th[4, c])))

        th = th.real

        # **** theta3 ****
        cl = [0, 2, 4, 6]
        for i in range(0, len(cl)):
            c = cl[i]
            T_10 = linalg.inv(self.AH(1, th, c))
            T_65 = self.AH(6, th, c)
            T_54 = self.AH(5, th, c)
            T_14 = (T_10 * des_pos) * linalg.inv(T_54 * T_65)
            P_13 = T_14 * self.mat([0, -self.d4, 0, 1]).T - self.mat([0, 0, 0, 1]).T
            t3 = cmath.acos((linalg.norm(P_13) ** 2 - self.a2 ** 2 - self.a3 ** 2) / (2 * self.a2 * self.a3))  # norm ?
            th[2, c] = t3.real
            th[2, c + 1] = -t3.real

        # **** theta2 and theta 4 ****

        cl = [0, 1, 2, 3, 4, 5, 6, 7]
        for i in range(0, len(cl)):
            c = cl[i]
            T_10 = linalg.inv(self.AH(1, th, c))
            T_65 = linalg.inv(self.AH(6, th, c))
            T_54 = linalg.inv(self.AH(5, th, c))
            T_14 = (T_10 * des_pos) * T_65 * T_54
            P_13 = T_14 * self.mat([0, -self.d4, 0, 1]).T - self.mat([0, 0, 0, 1]).T

            # theta 2
            th[1, c] = -atan2(P_13[1], -P_13[0]) + asin(self.a3 * sin(th[2, c]) / linalg.norm(P_13))
            # theta 4
            T_32 = linalg.inv(self.AH(3, th, c))
            T_21 = linalg.inv(self.AH(2, th, c))
            T_34 = T_32 * T_21 * T_14
            th[3, c] = atan2(T_34[1, 0], T_34[0, 0])
        th = th.real
        #print("___" * 30)
        #print(th)
        th = np.transpose(th)
        th = th.tolist()
        print("___" * 30)
        print(th)
        print("___"*30)

        best_th = th
        #best_th = self.select(th, start_pos)
        best_th = self.get_closest_solution(th, start_pos)
        best_th = self.get_absolute_joints(best_th, start_pos)


        return best_th

    """# Calculates Rotation Matrix given euler angles.
    def eulerAnglesToRotationMatrix(self, theta):
        #print("euler angle calculation ")
        R_x = np.array([[1, 0, 0],
                        [0, math.cos(theta[0]), -math.sin(theta[0])],
                        [0, math.sin(theta[0]), math.cos(theta[0])]
                        ])

        R_y = np.array([[math.cos(theta[1]), 0, math.sin(theta[1])],
                        [0, 1, 0],
                        [-math.sin(theta[1]), 0, math.cos(theta[1])]
                        ])

        R_z = np.array([[math.cos(theta[2]), -math.sin(theta[2]), 0],
                        [math.sin(theta[2]), math.cos(theta[2]), 0],
                        [0, 0, 1]
                        ])

        R = np.dot(R_z, np.dot(R_y, R_x))

        return R
"""