import numpy as np
from scipy import io

np.set_printoptions(precision=3)
from dynamics import IMUModel
from filters import UKF
from quaternion import Quaternion
from state import IMUState


def raw_to_physical_meas(raw, beta, alpha):
    return (raw - beta) * ((3300.0) / (1023.0 * alpha))

def preprocess_imu(accel, gyro):
    '''
    iron out quirks in the imu data
    
    1. negate x, y in accel
    2. move z gyro to last row
    '''
    accel_out = np.zeros_like(accel)
    # 1. negate x, y in accel
    accel_out[0,:] = np.negative(accel[0,:])
    accel_out[1,:] = np.negative(accel[1,:])
    accel_out[2,:] = accel[2,:]
    
    # 2. switch gyro axes
    gyro_out = np.zeros_like(gyro)
    gyro_out[2,:] = gyro[0,:]
    gyro_out[1,:] = gyro[2,:]
    gyro_out[0,:] = gyro[1,:]
    
    return accel_out, gyro_out

def vicon_to_imu_frame_orientation(vicon_rots, vicon_T):
    # vicon rots are rotation matrices
    # create heading vectors from rotation matrices, and plot the x, y, z components
    heading = np.array([0, 0, 9.81])
    vicon_orientations = np.zeros((3, vicon_T))
    vicon_orientations = np.matmul(vicon_rots.T, heading).T
    
    return vicon_orientations

def vicon_to_quat(vicon_rots, vicon_T):
    vicon_quats = []
    for i in range(vicon_T):
        vicon_quat = Quaternion()
        vicon_quat.from_rotm(vicon_rots[:,:,i])
        vicon_quats.append(vicon_quat)
    return vicon_quats    

def vicon_to_euler(vicon_rots, vicon_T):
    vicon_quats = vicon_to_quat(vicon_rots, vicon_T)
    vicon_eulers = np.zeros((3, vicon_T))
    for i in range(vicon_T):
        vicon_eulers[:,i] = vicon_quats[i].euler_angles()
    return vicon_eulers
    
def vicon_to_omega(vicon_rots, vicon_T, vicon_dt):
    vicon_quats = vicon_to_quat(vicon_rots, vicon_T)
    vicon_omegas = np.zeros((3, vicon_T))
    for i in range(1, vicon_T):
        vicon_quat = vicon_quats[i-1].inv() * vicon_quats[i]
        vicon_omegas[:,i] = vicon_quat.axis_angle(dt=vicon_dt)
        
    return vicon_omegas

def plot_results(state_hist, cov_hist, gyro_raw, vicon_quats):
    ukf_quats = np.array([state_hist[i].q.q for i in range(len(state_hist))])
    ukf_angvel = np.array([state_hist[i].omega for i in range(len(state_hist))])
    vicon_quats = np.array([vicon_quats[i].q for i in range(len(vicon_quats))])

    cov_mat = []
    for cov in cov_hist:
        quat_cov = Quaternion().from_axis_angle(a=np.array([cov[0,0], cov[1,1], cov[2,2]])).q
        cov_mat.append(np.concatenate((quat_cov, np.array([cov[3,3], cov[4,4], cov[5,5]]))))
        
    cov_mat = np.array(cov_mat)
    # plot quaternion
    plt.figure()
    plt.plot(ukf_quats[:,0], label="UKF q w", color="orange")
    plt.plot(ukf_quats[:,1], label="UKF q x", color="red")
    plt.plot(ukf_quats[:,2], label="UKF q y", color="green")
    plt.plot(ukf_quats[:,3], label="UKF q z", color="blue")
    plt.plot(vicon_quats[:,0], label="Vicon q w", color="orange", linestyle="--")
    plt.plot(vicon_quats[:,1], label="Vicon q x", color="red", linestyle="--")
    plt.plot(vicon_quats[:,2], label="Vicon q y", color="green", linestyle="--")
    plt.plot(vicon_quats[:,3], label="Vicon q z", color="blue", linestyle="--")
    # plot covariance for quaternion as error around the ukf quaternion
    plt.fill_between(np.arange(len(cov_mat)), ukf_quats[:,0] - cov_mat[:,0], ukf_quats[:,0] + cov_mat[:,0], color="orange", alpha=0.2)
    plt.fill_between(np.arange(len(cov_mat)), ukf_quats[:,1] - cov_mat[:,1], ukf_quats[:,1] + cov_mat[:,1], color="red", alpha=0.2)
    plt.fill_between(np.arange(len(cov_mat)), ukf_quats[:,2] - cov_mat[:,2], ukf_quats[:,2] + cov_mat[:,2], color="green", alpha=0.2)
    plt.fill_between(np.arange(len(cov_mat)), ukf_quats[:,3] - cov_mat[:,3], ukf_quats[:,3] + cov_mat[:,3], color="blue", alpha=0.2)
    plt.legend()
    plt.title("Quaternion")
    plt.xlabel("Timestep")
    plt.ylabel("Value")
    
    # plot angular velocity
    plt.figure()
    plt.plot(ukf_angvel[:,0], label="UKF omega x", color="red")
    plt.plot(ukf_angvel[:,1], label="UKF omega y", color="green")
    plt.plot(ukf_angvel[:,2], label="UKF omega z", color="blue")
    plt.plot(gyro_raw[0,:], label="Gyro x", color="red", linestyle="--")
    plt.plot(gyro_raw[1,:], label="Gyro y", color="green", linestyle="--")
    plt.plot(gyro_raw[2,:], label="Gyro z", color="blue", linestyle="--")
    plt.fill_between(np.arange(len(cov_mat)), ukf_angvel[:,0] - cov_mat[:,4], ukf_angvel[:,0] + cov_mat[:,4], color="red", alpha=0.2)
    plt.fill_between(np.arange(len(cov_mat)), ukf_angvel[:,1] - cov_mat[:,5], ukf_angvel[:,1] + cov_mat[:,5], color="green", alpha=0.2)
    plt.fill_between(np.arange(len(cov_mat)), ukf_angvel[:,2] - cov_mat[:,6], ukf_angvel[:,2] + cov_mat[:,6], color="blue", alpha=0.2)
    
    plt.legend()
    plt.title("Angular Velocity")
    plt.xlabel("Timestep")
    plt.ylabel("Value")
    plt.show()

def main():
    # read raw data
    data_num = 1
    imu = io.loadmat("hw2/p2/imu/imuRaw"+str(data_num)+".mat")
    accel_raw = imu["vals"][0:3,:].astype(float)
    gyro_raw = imu["vals"][3:6,:].astype(float)
    ts = imu["ts"][0,:].astype(float)
    T = len(ts)
    dt = np.mean(np.diff(ts)) # take dt as average of time differences
    vicon = io.loadmat("hw2/p2/vicon/viconRot"+str(data_num)+".mat")
    vicon_rots = vicon["rots"].astype(float)
    vicon_ts = vicon["ts"][0,:].astype(float)
    vicon_T = len(vicon_ts)
    vicon_dt = np.mean(np.diff(vicon_ts))

    # preprocess
    accel, gyro = preprocess_imu(accel_raw, gyro_raw)
    # calibrate accel and gyro
    accel_alphas = np.array([34.5676, 34.2750, 34.4393]).reshape(3,1)
    accel_betas = np.array([-511.1268, -500.4646, 500.8905]).reshape(3,1)
    gyro_alphas = np.array([250, 250, 250]).reshape(3,1)
    gyro_betas = np.array([374.5, 375.2, 373.1]).reshape(3,1)
    
    accel_calib = raw_to_physical_meas(accel, accel_betas, accel_alphas)
    gyro_calib = raw_to_physical_meas(gyro, gyro_betas, gyro_alphas)
    vicon_orientations = vicon_to_imu_frame_orientation(vicon_rots, vicon_T)
    vicon_eulers = vicon_to_euler(vicon_rots, vicon_T)
    vicon_quats = vicon_to_quat(vicon_rots, vicon_T)
    vicon_omegas = vicon_to_omega(vicon_rots, vicon_T, vicon_dt) 
    
    # initial state (orientation, omega) from vicon
    x0 = IMUState(Quaternion(), 
                  np.zeros(3))
    print(np.rad2deg(x0.get_orientation()))
    # initial covariance, 6x6
    P0 = np.diag(np.ones(6)) * 0.0001
    # measurement noise covariance, 6x6
    Q = np.diag(np.ones(6)) * 0.0001
    # process noise covariance, 6x6
    R = np.diag(np.ones(6)) * 0.00001
    # state transition function
    f = IMUModel.f
    # measurement function
    g = IMUModel.g

    ukf = UKF(x0, P0, Q, R, f, g)
    state_history = []
    cov_history = []
    for i in range(1,T):
        print("timestep: ", i , " of ", T)
        dt = ts[i] - ts[i-1]
        y_hat = np.zeros((6,1))
        y_hat[:3] = gyro_calib[:,i].reshape(3,1)
        y_hat[3:] = accel_calib[:,i].reshape(3,1)
        ukf.predict(dt)
        ukf.update(y_hat, dt)
        mu, Sigma = ukf.get_estimate()
        state_history.append(mu)
        cov_history.append(Sigma)
        
    plot_results(state_history, cov_history, gyro_calib, vicon_quats)
    
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    main()