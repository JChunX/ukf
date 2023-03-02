import numpy as np
from quaternion import Quaternion

class IMUState:
    
    def __init__(self, q, omega):
        self.q = q # Quaternion
        self.omega = omega # angular velocity in body frame
    
    @classmethod
    def from_vector(cls, x):
        q = Quaternion()
        q.from_axis_angle(x[0:3])
        omega = x[3:6]
        return cls(q, omega)
    
    def to_vector(self):
        return np.concatenate((self.q.axis_angle(), self.omega))
        
    def get_orientation(self):
        return self.q.euler_angles()
        
    def __str__(self):
        return 'eulers: ' + np.rad2deg(self.q.euler_angles()).__str__() + ', q: ' + str(self.q) + ', omega: ' + str(self.omega)
    
    def __repr__(self):
        return self.__str__()
    
    def __eq__(self, other):
        return self.q == other.q and np.allclose(self.omega, other.omega)
    
    def __ne__(self, other):
        return not self.__eq__(other)
    
    def __add__(self, other):
        return IMUState(other.q * self.q, self.omega + other.omega)
    
    def __sub__(self, other):
        return IMUState(other.q.inv() * self.q, self.omega - other.omega)