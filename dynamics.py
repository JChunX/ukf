import numpy as np
from quaternion import Quaternion
from state import IMUState


class IMUModel:
    
    @staticmethod
    def f(x: IMUState, dt: float):
        '''
        state transition function
        '''
        q_delta = Quaternion()
        q_delta.from_euler_angles(x.omega * dt)
        return IMUState(x.q * q_delta, x.omega)
    
    @staticmethod
    def g(x: IMUState, dt: float):
        '''
        measurement function
        '''
        
        y = np.zeros(6)
        y[:3] = x.omega
        
        grav_vector = np.array([0, 0, 1])
        g_quat = Quaternion()
        g_quat.q[1:] = grav_vector
        g_quat.q[0] = 0
        g_quat.normalize()
        g_prime = x.q.inv() * g_quat * x.q
        y[3:] = g_prime.q[1:]
        
        return y