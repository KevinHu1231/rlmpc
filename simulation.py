import numpy as np
import pandas as pd
import sympy as sp
from scipy.spatial.transform import Rotation as R
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class quaternion_uav:
    def __init__(self):

        # UAV state
        self.x = 0 
        self.y = 0
        self.z = 0
        self.x_dot = 0
        self.y_dot = 0
        self.z_dot = 0
        self.phi = 0
        self.theta = 0
        self.psi = 0
        self.phi_dot = 0
        self.theta_dot = 0
        self.psi_dot = 0

        # UAV angular velocity
        self.omega = np.array([[0],[0],[0]])
        self.omega_dot = np.array([[0],[0],[0]])

        # UAV orientation
        self.rotation = R.from_matrix([[1,0,0,],[0,1,0,],[0,0,1]]).as_quat()

        # UAV forces, torques
        self.F = 0
        self.tau = 0

        # For individual motors
        self.Fs = [0,0,0,0]
        self.taus = [0,0,0,0]
        self.l = 0.1
        self.prop_diam = 0.1

        # Constants
        self.m = 1
        self.g = 9.81
        self.J = np.eye(3)

        # Time step
        t = 0.001
    
    # def rpy_to_omega(self):
    #     return np.array([[1,0,-np.sin(self.theta)],[0,np.cos(self.phi),np.cos(self.theta)*np.sin(self.phi)],[0,-np.sin(self.phi),np.cos(self.theta)*np.cos(self.phi)]])@np.array([[self.phi_dot],[self.theta_dot],[self.psi_dot]])
    
    # def omega_to_rpy(self):
    #     return np.array([[1,np.sin(self.phi)*np.tan(self.theta),np.cos(self.phi)*np.tan(self.theta)],[0,np.cos(self.phi),-np.sin(self.phi)],[0,np.sin(self.phi)/np.cos(self.theta),np.cos(self.phi)/np.cos(self.theta)]])@self.omega

    def update_state(self):

        # Update Cartesian position and velocity
        e3 = np.array([[0],[0],[1]])
        v_dot = (self.F/self.m)*self.rotation.inv().as_matrix()@e3 - self.g*e3

        self.x_dot = v_dot[0,0]
        self.y_dot = v_dot[1,0]
        self.z_dot = v_dot[2,0]

        self.x += self.x_dot*self.t
        self.y += self.y_dot*self.t
        self.z += self.z_dot*self.t

        # Update angular velocity and RPY

        # Which one is correct?
        # Is this correct?

        # rpy_dot = self.omega_to_rpy()

        # self.phi_dot = rpy_dot[0,0]
        # self.theta_dot = rpy_dot[1,0]
        # self.psi_dot = rpy_dot[2,0]

        # self.phi = self.phi_dot*self.t
        # self.theta = self.theta_dot*self.t
        # self.psi = self.psi_dot*self.t

        self.omega_dot = np.linalg.inv(self.J)@(np.cross(-self.omega,self.J@self.omega) + self.tau)
        self.omega = self.omega + self.omega_dot*self.t

        # Update orientation
        q_omega = R.from_quat(np.append(self.omega.reshape(3,),0))
        
        BGq = self.rotation
        BGq_dot = 0.5*q_omega*BGq
        BGq = BGq + BGq_dot*self.t
        self.rotation = BGq
        
        #Update RPY

        # Which one is correct?
        # Is this correct?

        rpy = self.rotation.as_euler()
        self.phi = rpy[0]
        self.theta = rpy[1]
        self.psi = rpy[2]

    def simulate(self,T):
        # Create a figure and a 3D axis
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Set labels for the axes
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        # Set axis limits
        ax.set_xlim(-5, 5)
        ax.set_ylim(-5, 5)
        ax.set_zlim(0, 10)

        # Show the plot
        plt.show() 

# Example usage:
if __name__ == "__main__":
    # Creating an instance of MyClass
    uav = quaternion_uav()
    uav.simulate(5)