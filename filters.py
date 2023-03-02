import numpy as np
import scipy

from quaternion import Quaternion
from state import IMUState


class UKF:
    '''
    unscented Kalman filter
    '''
    
    def __init__(self, x0, P0, Q, R, f, g):
        '''
        x0: initial state
        P0: initial covariance
        Q: measurement noise covariance
        R: process noise covariance
        f: state transition function
        g: measurement function
        '''
        
        self.mu_x = x0 # state is IMUState object
        self.Sig_xx = P0  # covariance is 6x6 matrix
        
        self.mu_pred = np.zeros_like(x0)
        self.Sig_pred = np.zeros_like(P0)
        
        self.mu_y = np.zeros((6,))
        self.Sig_yy = np.zeros((6,))
        self.Sig_xy = np.zeros((6,6))

        self.Q = Q # Q is 6x6 matrix
        self.R = R # R is 6x6 matrix
        self.f = f
        self.g = g
        # n: 2n is number of sigma points
        self.n = self.Sig_xx.shape[0]
        # error_list is the state errors for the sigma points
        self.error_list = []

    def _sigma_points(self, mu, Sig, dt, dyn_cov):
        '''
        compute sigma points
        
        inputs
        mu - state mean IMUState object
        Sig - state covariance
        dt - time step
        dyn_cov - noise covariance
        '''
        n = self.n
        sqrt_n_sig = scipy.linalg.sqrtm(n * (Sig + dyn_cov * dt))
        sig_pts = np.empty((2*n,), dtype=object)
        for i in range(n):
            sig_pts[i] = mu + IMUState.from_vector(sqrt_n_sig[i,:])
            sig_pts[i+n] = mu - IMUState.from_vector(sqrt_n_sig[i,:])
            
        return sig_pts # sig_pts is 2n x 1 vector of IMUState objects
        
    def _transform(self, fun, mu, Sig, dt, dyn_cov):
        '''
        propagate sigma points through nonlinear function
        
        inputs
        fun - transformation function
        mu - mean state
        Sig - state covariance
        dt - time step
        dyn_cov - noise covariance
        '''
        w = 1/(2*self.n)
        sig_pts = self._sigma_points(mu, Sig, dt, dyn_cov)
        transformed = np.empty((2*self.n,), dtype=object)
        for i in range(2*self.n):
            transformed[i] = fun(sig_pts[i], dt)
            
        return transformed    
    
    def _get_x_error(self, x, E, omega_bar):
        '''
        combines orientation matrix and angular velocity error
        '''
        error_list = []
        
        for i, x_i in enumerate(x):
            x_omega_err = x_i.omega - omega_bar
            x_quat_error = E[:,i]
            # append 6x1 vector to error_list
            error_list.append(
                np.concatenate((x_quat_error, 
                                x_omega_err)).reshape(6,1)) 
            
        return error_list
    
    def _quaternion_error(self, mu_x, sigma_pts, eps=1e-3):
        '''
        finds mean error between sigma points and current quaternion mean
        
        inputs  
        mu_x: current mean state
        sigma_pts: sigma points, list (2n) of IMUState objects
        eps: convergence threshold
        '''
        q_bar = mu_x.q
        # error vector is rotation between sigma points and current mean
        e_bar = np.ones((3,)) * np.inf
        E = np.zeros((3,2*self.n))
        
        while np.linalg.norm(e_bar) > eps:
            for i in range(2*self.n):
                # find error for each sigma point
                e_i = sigma_pts[i].q * q_bar.inv()
                E[:,i] = e_i.axis_angle()
            # find mean of error vectors
            e_bar = np.mean(E, axis=1)
            # update quaternion mean
            q_bar = Quaternion().from_axis_angle(e_bar) * q_bar
            
        return q_bar, E
        
    def _mu_sig_x(self, transformed_x):
        '''
        computes predicted mean and covariance
        
        inputs:
        transformed_x: transformed sigma points, list (2n) of IMUState objects
        '''
        w = 1/(2*self.n)
        
        q_bar, E = self._quaternion_error(self.mu_x, transformed_x)
        omega_bar = np.mean([transformed_x[i].omega for i in range(2*self.n)], axis=0)
        
        self.error_list = self._get_x_error(transformed_x, E, omega_bar)
            
        Sig_new = np.zeros_like(self.Sig_xx)
        # compute new covariance
        Sig_new = w * np.sum(
            np.outer((self.error_list[i]), 
                      (self.error_list[i].T)) for i in range(2*self.n))
        
        return IMUState(q_bar, omega_bar), Sig_new
    
    def _mu_sig_y(self, transformed_x, transformed_y):
        '''
        computes measurement mean and covariance
        
        inputs:
        transformed_x: transformed sigma points, list (2n) of IMUState objects
        transformed_y: transformed sigma points, list (2n) of IMUState objects
        '''
        _, E = self._quaternion_error(self.mu_pred, transformed_x)
        omega_bar = np.mean([transformed_x[i].omega for i in range(2*self.n)], axis=0)
        
        self.error_list = self._get_x_error(transformed_x, E, omega_bar)
        
        w = 1/(2*self.n)
        y_hat = np.mean([transformed_y[i] for i in range(2*self.n)], axis=0)
        Sig_yy = self.Q + w * np.sum(np.outer(transformed_y[i]-y_hat, (transformed_y[i]-y_hat).T) for i in range(2*self.n))
        
        Sig_xy = w * np.sum(np.outer(self.error_list[i], (transformed_y[i]-y_hat).T) for i in range(2*self.n))
        
        return y_hat, Sig_yy, Sig_xy
    
    def predict(self, dt):
        '''
        predict state and covariance
        '''
        # propagate sigma points through dynamics
        transformed_x = self._transform(self.f, self.mu_x, self.Sig_xx, dt, dyn_cov=self.R)
        self.mu_pred, self.Sig_pred = self._mu_sig_x(transformed_x)
        
    def update(self, y_hat, dt):
        '''
        update state and covariance
        
        inputs:
        y_hat: new measurement, 6x1 vector of (gyro, accel)
        dt: time step
        '''
        # propagate sigma points through measurement function
        transformed_x_kp1 = self._transform(self.f, self.mu_pred, self.Sig_pred, dt, dyn_cov=np.zeros_like(self.R))
        transformed_y = self._transform(self.g, self.mu_pred, self.Sig_pred, dt, dyn_cov=np.zeros_like(self.R))
        self.mu_y, self.Sig_yy, self.Sig_xy = self._mu_sig_y(transformed_x_kp1, transformed_y)
        innovation = y_hat.ravel() - self.mu_y
        self.K = self.Sig_xy @ np.linalg.inv(self.Sig_yy)
        
        # update mean and covariance
        self.mu_x = self.mu_pred + IMUState.from_vector(self.K @ innovation)
        self.Sig_xx = self.Sig_pred - self.K @ self.Sig_yy @ self.K.T
        
    def get_estimate(self):
        '''
        return current state estimate
        '''
        return self.mu_x, self.Sig_xx


    