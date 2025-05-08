"""
EEL 4930/5934: Autonomous Robots
University Of Florida
E. Baker Herrin
"""

import numpy as np


class KF_2D(object):
    def __init__(self, dt, u_x, u_y, std_acc, x_std_meas, y_std_meas):
        """
        dt: sampling time (time for 1 cycle)
        u_x: acceleration in x-direction
        u_y: acceleration in y-direction
        std_acc: process noise magnitude
        x_std_meas: standard deviation of the measurement in x-direction
        y_std_meas: standard deviation of the measurement in y-direction
        """
        self.dt = dt  # sampling time
        self.u = np.matrix([[u_x], [u_y]])  # control input variables
        self.x = np.matrix(
            [[0], [0], [0], [0]]
        )  # intial State [position_x, position_y, velocity_x, velocity_y]

        #### State Transition Matrix A: relationship between current and next state ####
        # The System Dynamics/ motion modeling
        # 1's for position modeling, dt's represent velocity, acceleration term is dt^2
        self.A = np.matrix(
            [
                [1, 0, self.dt, 0.5 * self.dt**2],
                [0, 1, 0, self.dt],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]
        )

        #### Control Input Matrix B ####
        # Models effect of control inputs on the state variables
        # Control input: acceleration.
        self.B = np.matrix(
            [
                [(self.dt**2) / 2, 0],
                [0, (self.dt**2) / 2],
                [self.dt, 0],
                [0, self.dt],
            ]
        )

        """
        #### Measurement Mapping Matrix ####
        
        mapping of the state vector to the measurement vector
        1's indicate it maps directly (no difference in what they are)
        EX) would be different if state vector had position while measurement was
        total distance, or something different. H would then show the conversion.
        also shows measurability: which for us is just the x and y position.
        """
        self.H = np.matrix(
            [[1, 0, 0, 0], [0, 1, 0, 0]]
        )  # x, accel_x, y, accel_y : idea is position can be measured directly

        """
        #### Initial Process Noise Covariance ####
        
        represents covariance derived from kinematic equations for constant acceleration
        - assuming x & y are uncorrelated and have same std deviation
        - dependent on the system being defined, can be white guassian noise
        - models uncertainty due to unmodeled dynamics/ failures of the A matrix
        - can also model physical properties such as friction or water resistance
        - used to update the covariance matrix P
        """
        self.Q = (
            np.matrix(
                [
                    [(self.dt**4) / 4, 0, (self.dt**3) / 2, 0],
                    [0, (self.dt**4) / 4, 0, (self.dt**3) / 2],
                    [(self.dt**3) / 2, 0, self.dt**2, 0],
                    [0, (self.dt**3) / 2, 0, self.dt**2],
                ]
            )
            * std_acc**2
        )

        ##### Initial Measurement Noise Covariance ####
        # uncertainty in sensor measurements- variance in measurement noise
        self.R = np.matrix([[x_std_meas**2, 0], [0, y_std_meas**2]])

        # Initial Covariance Matrix
        self.P = np.eye(self.A.shape[1])

    def predict(self):
        """
        Predicts the next state of the system for the state variable vector x, which starts at 0.
        """
        # Update time state (self.x): x_k =Ax_(k-1) + Bu_(k-1)
        self.x = np.dot(self.A, self.x) + np.dot(self.B, self.u)
        # Calculate error covariance (self.P): P= A*P*A' + Q
        self.P = np.dot(np.dot(self.A, self.P), np.transpose(self.A)) + self.Q
        # Update error covariance matrix self.P
        return self.x[0:2]

    def update(self, z):
        """
        Using the latest measurement z, predicted state x, and updated covariance matrix P,
        update function uses the kalman gain to update the covariance matrix and the state vector x
        """
        e = z - np.dot(
            self.H, self.x
        )  # residual, difference between actual & predicted measurement
        # print(np.shape(e))
        # print(np.shape(np.dot(self.H, self.x)))
        S = (
            np.dot(np.dot(self.H, self.P), np.transpose(self.H)) + self.R
        )  # Innovation cov. mat. (uncertainty in the measurement)
        K = np.dot(
            np.dot(self.P, np.transpose(self.H)), np.linalg.inv(S)
        )  # the gain! (weight factor of z vs. Hx (prediction)) for updating the state
        self.P = np.dot(np.dot(K, self.H), self.P)  # updating the covariance matrix
        self.x = self.x + np.dot(
            K, e
        )  # the gain and residual give us the weight of update

        return self.x[0:2]
