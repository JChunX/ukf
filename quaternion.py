import math
import numpy as np


class Quaternion:
    
    def __init__(self, scalar=1, vec=[0,0,0]): 
        self.q = np.array([scalar, 0., 0., 0.])
        self.q[1:4] = vec

    def normalize(self):
        self.q = self.q/np.linalg.norm(self.q)
        return self

    def scalar(self):
        return self.q[0]

    def vec(self):
        return self.q[1:4]

    def axis_angle(self, dt=1.0):
        theta = 2*math.acos(self.scalar()) / dt
        vec = self.vec()
        if (np.linalg.norm(vec) == 0):
            return np.zeros(3)
        vec = vec/np.linalg.norm(vec)
        return vec*theta

    def euler_angles(self):
        phi = math.atan2(2*(self.q[0]*self.q[1]+self.q[2]*self.q[3]), \
                1 - 2*(self.q[1]**2 + self.q[2]**2))
        theta = math.asin(2*(self.q[0]*self.q[2] - self.q[3]*self.q[1]))
        psi = math.atan2(2*(self.q[0]*self.q[3]+self.q[1]*self.q[2]), \
                1 - 2*(self.q[2]**2 + self.q[3]**2))
        return np.array([phi, theta, psi])

    def from_axis_angle(self, a, dt=1.0):
        a_normed = np.linalg.norm(a)
        angle = a_normed * dt
        if angle != 0:
            axis = a / a_normed
        else:
            axis = np.array([1,0,0])
        self.q[0] = math.cos(angle/2)
        self.q[1:4] = axis*math.sin(angle/2)
        self.normalize()
        return self

    def from_rotm(self, R):
        theta = math.acos((np.trace(R)-1)/2)
        omega_hat = (R - np.transpose(R))/(2*math.sin(theta))
        omega = np.array([omega_hat[2,1], -omega_hat[2,0], omega_hat[1,0]])
        self.q[0] = math.cos(theta/2)
        self.q[1:4] = omega*math.sin(theta/2)
        self.normalize()
        return self
        
    def from_euler_angles(self, r):
        (yaw, pitch, roll) = (r[0], r[1], r[2])
        qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
        qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
        qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
        qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
        
        self.q[0] = qw
        self.q[1] = qx
        self.q[2] = qy
        self.q[3] = qz
        self.normalize()
        return self

    def inv(self):
        q_inv = Quaternion(self.scalar(), -self.vec())
        q_inv.normalize()
        return q_inv

    #Implement quaternion multiplication
    def __mul__(self, other):
        t0 = self.q[0]*other.q[0] - \
             self.q[1]*other.q[1] - \
             self.q[2]*other.q[2] - \
             self.q[3]*other.q[3]
        t1 = self.q[0]*other.q[1] + \
             self.q[1]*other.q[0] + \
             self.q[2]*other.q[3] - \
             self.q[3]*other.q[2]
        t2 = self.q[0]*other.q[2] - \
             self.q[1]*other.q[3] + \
             self.q[2]*other.q[0] + \
             self.q[3]*other.q[1]
        t3 = self.q[0]*other.q[3] + \
             self.q[1]*other.q[2] - \
             self.q[2]*other.q[1] + \
             self.q[3]*other.q[0]
        retval = Quaternion(t0, [t1, t2, t3]).normalize()
        return retval

    def __str__(self):
        return str(self.scalar()) + ', ' + str(self.vec())
    
    def __eq__(self, other):
        return np.allclose(self.q, other.q)

