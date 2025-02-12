import numpy as np
import sympy as sp
import casadi as ca
from tqdm import tqdm
from scipy.integrate import solve_ivp
from scipy.optimize import root
import time

import gym
import gymnasium
from stable_baselines3 import TD3
from stable_baselines3.sac import SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import BaseCallback
import os
from torch.utils.tensorboard import SummaryWriter
from stable_baselines3.common.logger import TensorBoardOutputFormat

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

import trajectory_ASV

### Quaternion operations ###
def R_GB(q):
    qw = q[0]
    qx = q[1]
    qy = q[2]
    qz = q[3]

    RGB = sp.Matrix([
    [1-2*(qy**2+qz**2),2*(qx*qy+qw*qz),2*(qx*qz-qw*qy)],
    [2*(qx*qy-qw*qz),1-2*(qx**2+qz**2),2*(qy*qz+qw*qx)],
    [2*(qx*qz+qw*qy), 2*(qy*qz-qw*qx),1-2*(qx**2+qy**2)]
    ])

    return RGB.T

def qprod(p,q):
    pw = p[0]
    px = p[1]
    py = p[2]
    pz = p[3]

    qw = q[0]
    qx = q[1]
    qy = q[2]
    qz = q[3]

    rw = pw*qw - px*qx - py*qy - pz*qz
    rx = pw*qx + px*qw + py*qz - pz*qy
    ry = pw*qy - px*qz + py*qw + pz*qx
    rz = pw*qz + px*qy - py*qx + pz*qw
    
    # Return the resulting quaternion
    return sp.Matrix([rw, rx, ry, rz])

def q_vec(v):
    return sp.Matrix([v[0],v[1],v[2],0])

def q_inv(v):
    return sp.Matrix([-v[0],-v[1],-v[2],v[3]])

def quatProd(q):
    qw = q[0]
    qx = q[1]
    qy = q[2]
    qz = q[3]

    Q = sp.Matrix([
        [ qw,-qx,-qy,-qz],
        [qx,qw,-qz,qy],
        [qy,qz,qw,-qx],
        [qz,-qy,qx,qw]
    ])
    return Q

### Sympy to Casadi ###

def sympy2casadi(sympy_expr,sympy_var,casadi_var):
    assert casadi_var.is_vector()
    if casadi_var.shape[1]>1:
        casadi_var = casadi_var.T
    casadi_var = ca.vertsplit(casadi_var)
    from sympy.utilities.lambdify import lambdify

    mapping = {'ImmutableDenseMatrix': ca.blockcat,
                'MutableDenseMatrix': ca.blockcat,
                'Abs':ca.fabs
                }
    f = lambdify(sympy_var,sympy_expr,modules=[mapping, ca])
    return f(*casadi_var)

class drone_mpc_env(gym.Env):
    def __init__(self):
        super(drone_mpc_env,self).__init__()

        # Observation space includes the current state, reference state and time
        self.observation_space = gymnasium.spaces.box.Box(low=-1, high=1, shape=(21,), dtype=np.float32)
        self.observation_space_lower = np.hstack([-30*np.ones(3,),-30*np.ones(3,),-1*np.ones(4,),-30*np.ones(3,),-30*np.ones(3,),-1*np.ones(4,),np.zeros(1,)]).astype(np.float32)
        self.observation_space_higher = np.hstack([30*np.ones(3,),30*np.ones(3,),1*np.ones(4,),30*np.ones(3,),30*np.ones(3,),1*np.ones(4,),10*np.ones(1,)]).astype(np.float32)

        # Action space includes linearized A (10x10), B (10x4) matrices, 
        # Q (10x10, 10 params), R (4x4, 4 params), Q_bar (10x10, 10 params) matrix diagonals,
        # Linearized controller K matrix (4x10) and constant c vector (4x1)
        # 208 dimensional action space

        # low = np.hstack([-100*np.ones(140,),0.01*np.ones(24,)])
        # high = np.hstack([100*np.ones(140,),100*np.ones(24,)])
        # low = low.astype(np.float64)
        # high = high.astype(np.float64)

        self.action_space = gymnasium.spaces.box.Box(low=-1, high=1, shape=(24,), dtype=np.float32)
        #self.action_space_lower = 0.00001*np.ones(24,).astype(np.float32)
        #self.action_space_higher = np.hstack([12*np.ones(2,),20*np.ones(1,),2*np.ones(7,),0.0002*np.ones(1,),2*np.ones(3,),12*np.ones(2,),20*np.ones(1,),2*np.ones(7,)]).astype(np.float32)

        self.action_space_lower = np.hstack([7*np.ones(2,),7*np.ones(1,),0.5*np.ones(7,),0.00001*np.ones(1,),0.5*np.ones(3,),7*np.ones(2,),7*np.ones(1,),0.5*np.ones(7,)]).astype(np.float32)
        self.action_space_higher = np.hstack([9*np.ones(2,),9*np.ones(1,),1.5*np.ones(7,),0.0002*np.ones(1,),1.5*np.ones(3,),9*np.ones(2,),9*np.ones(1,),1.5*np.ones(7,)]).astype(np.float32)

        self.save_path = "Distance_Reward/"

        # Constants
        self.m = 1.0175
        self.g = 9.8066

        # Time step
        self.t = 0.01
        self.N_update = 10

        # MPC Parameters
        self.N = 10  # Prediction horizon time steps
        self.X_dim = 10
        self.U_dim = 4
        self.epsilon = 0.001

        # Initial Values
        self.Q = np.diag(np.array([8,8,8,1,1,1,1,1,1,1])) # cost for states 
        self.R = np.diag(np.array([0.0001,1,1,1])) # cost for input
        self.Q_bar = np.diag(np.array([8,8,8,1,1,1,1,1,1,1]))
        # self.K = np.vstack([10*np.ones((1,10)),1*np.ones((3,10))])
        # self.c = np.ones((4,1))
        self.episode = -1

        self.f = self.f_syms()
        
        #self.fx = self.feq_syms()
        # self.A, self.B = self.linearize()
        # self.flinx = self.flin_syms()

        self.f_sys = self.f_casadi()
        self.create_casadi_integrator()

        self.create_run_folder()

    #### Symbolic Expression Creation ####

    def fdot(self,x,u):
        v = x[3:6,0]
        q = x[6:,0]
        w = u[1:,0]
        e = sp.Matrix([[0],[0],[1]])
        p_dot = v
        v_dot = (u[0,0]/self.m)*R_GB(q)@e - self.g*e
        q_dot = 0.5*quatProd(q)@w.row_insert(0, sp.Matrix([0]))
        f_dot = sp.Matrix.vstack(p_dot,v_dot,q_dot)
        self.f_dot = f_dot
        return f_dot
    
    def f_syms(self):
        X = sp.Matrix(sp.symbols("x,y,z,vx,vy,vz,qw,qx,qy,qz"))
        U = sp.Matrix(sp.symbols("F,wx,wy,wz"))
        self.X = X
        self.U = U
        f_s = sp.lambdify([X,U],self.fdot(X,U), 'numpy')
        
        return f_s
    
    def feq(self,x):
        v = x[3:6,0]
        q = x[6:10,0]
        w = x[11:,0]
        e = sp.Matrix([[0],[0],[1]])
        p_dot = v
        v_dot = (x[10,0]/self.m)*R_GB(q)@e - self.g*e
        q_dot = 0.5*quatProd(q)@w.row_insert(0, sp.Matrix([0]))
        #quaterion constraint
        quat = sp.Matrix([q[0]**2 + q[1]**2 + q[2]**2 + q[3]**2 - 1])
        f_dot = sp.Matrix.vstack(p_dot,v_dot,q_dot,quat,p_dot)
        self.f_eq = f_dot
        return f_dot
    
    def feq_syms(self):
        XU = sp.Matrix(sp.symbols("x,y,z,vx,vy,vz,qw,qx,qy,qz,F,wx,wy,wz"))
        self.XU = XU
        feq_s = sp.lambdify([XU],self.feq(XU), 'numpy')  
        return feq_s
    
    ### For Linearization ###

    def linearize(self):
        Fx = lambda x: self.fx(x).flatten()
        x0_init = np.array([0,0,1,0,0,0,1,0,0,0,9.98,0,0,0])
        x0 = root(Fx,x0_init).x
        vals = {'x': x0[0], 'y':  x0[1], 'z':  x0[2],'vx':  x0[3], 'vy':  x0[4], 'vz':  x0[5], 'qw':  x0[6], 'qx':  x0[7], 'qy':  x0[8], 'qz':  x0[9], 'F':  x0[10], 'wx':  x0[11], 'wy':  x0[12], 'wz':  x0[13]}

        A = self.f_dot.jacobian(self.X)
        A = A.subs(vals)
        print(A)
        print("A shape: ", A.shape)

        B = self.f_dot.jacobian(self.U)
        B = B.subs(vals)
        print(B)
        print("B shape: ", B.shape)

        return A, B
    
    def flin(self,x,u):
        A = self.A
        B = self.B
        x_vec = sp.Matrix(x[:,0])
        u_vec = sp.Matrix(u[:,0])

        f_linear = A@x_vec + B@u_vec
        self.f_lin = f_linear
        return f_linear
    
    def flin_syms(self):
        X = sp.Matrix(sp.symbols("x,y,z,vx,vy,vz,qw,qx,qy,qz"))
        U = sp.Matrix(sp.symbols("F,wx,wy,wz"))
        self.X = X
        self.U = U
        flin_s = sp.lambdify([X,U],self.flin(X,U), 'numpy')
        return flin_s
    
    #### Create Casadi Expressions ####

    def f_casadi(self):
        X_cas = ca.SX.sym('X', self.X_dim)
        U_cas = ca.SX.sym('U', self.U_dim)
        XU = self.X.col_join(self.U)
        XU_cas = ca.vertcat(X_cas,U_cas)
        f_casadi = sympy2casadi(self.f_dot, XU, XU_cas)
        f_func = ca.Function('f', [X_cas, U_cas], [f_casadi])
        return f_func
    
    def create_casadi_integrator(self):

        x = ca.MX.sym('x', 10)
        u = ca.MX.sym('u', 4)
        x_dot = self.f_sys(x, u)
        dae = {'x': x, 'p': u, 'ode': x_dot}
        opts = {'tf': self.t}  # Integration step size
        self.integrator = ca.integrator('integrator', 'rk', dae, opts)
    
    ### MPC using Casadi ####

    def mpc(self, x0, x_ref):
        # Optimization variables
        opti = ca.Opti()
        X = opti.variable(self.X_dim, self.N+1)
        U = opti.variable(self.U_dim, self.N)
        #K_mat = opti.variable(4,10)

        # Cost function
        cost = 0
        for k in range(self.N):
            cost += ca.mtimes([(X[:, k] - x_ref).T, self.Q, X[:, k] - x_ref]) + ca.mtimes([U[:, k].T, self.R, U[:, k]])

        cost += ca.mtimes([(X[:, self.N-1] - x_ref).T, self.Q_bar, X[:, self.N-1] - x_ref]) # cost for terminal state

    ##### System dynamics constraints

        # Non Integration
        for k in range(self.N):
            opti.subject_to(X[:, k+1] == X[:, k] + self.t * self.f_sys(X[:, k], U[:, k]))

        # Initial condition constraint
        opti.subject_to(X[:, 0] == x0)

        # # Integration
        # for k in range(self.N):
            
        #     # Linear control constraint
        #     #opti.subject_to(U[:, k] == ca.mtimes([K_mat, X[:, k]]) + self.c)

        #     x_next = self.integrator(x0=X[:, k], p=U[:, k])['xf']

        #     # Linear system constraint
        #     opti.subject_to(X[:, k+1] == x_next)
    
    #####

        # Input constraints
        opti.subject_to(opti.bounded(0, U[0,:], 30))
        opti.subject_to(opti.bounded(-10, U[1,:], 10))
        opti.subject_to(opti.bounded(-10, U[2,:], 10))
        opti.subject_to(opti.bounded(-3, U[3,:], 3))
        
        # Unit Quaternion Constraint
        qw = X[6,:]
        qx = X[7,:]
        qy = X[8,:]
        qz = X[9,:]
        opti.subject_to(opti.bounded(-self.epsilon, (qw**2 + qx**2 + qy**2 + qz**2) - 1, self.epsilon))

        # Minimum Height Constraint
        opti.subject_to(opti.bounded(0.5, X[2,:], 15))

        # Solve the optimization problem
        opti.minimize(cost)
        opts = {
                'print_time': False,
                'ipopt': {
                    'print_level': 0,
                    'sb': 'yes',
                }
        }
        opti.solver('ipopt',opts)


        # print(f"Number of variables: {opti.nx}")
        # print(f"Number of constraints: {opti.ng}")

        # Variable to store the last feasible solution
        try:
            # Solve the optimization problem
            sol = opti.solve()
            cost_optimal = sol.value(cost)
            # Get the optimal control input
            u_optimal = sol.value(U[:, 0])
        except Exception as e:
            print(f"Optimization failed: {e}")
            print("Doing no action:")
            cost_optimal = 1000000
            u_optimal = np.array([0,0,0,0])

        return cost_optimal, u_optimal

    ### Reinforcement Learning ###

    def reset(self, seed=None):
        self.norm_state = self.normalize(np.array([0,0,1,0,0,0,1,0,0,0]),self.observation_space_lower[:10],self.observation_space_higher[:10])
        self.norm_reference = self.normalize(np.array([5,5,5,0,0,0,1,0,0,0]),self.observation_space_lower[:10],self.observation_space_higher[:10])
        self.norm_time = self.normalize(0,self.observation_space_lower[20],self.observation_space_higher[20])

        self.state = np.array([0,0,1,0,0,0,1,0,0,0])
        self.reference = np.array([5,5,5,0,0,0,1,0,0,0])
        self.time = 0

        self.Xs = [self.state]
        self.rewards = []
        self.Us = []
        self.ts = [self.time]

        self.costs_opt = []
        self.costs_real = []
        self.Qs = []
        self.Rs = []
        self.Q_bars = []

        return np.concatenate((self.norm_state, self.norm_reference, np.array([self.norm_time]))), {}
    
    def step(self,action):

        #self.A = self.unnormalize(action[:100],self.action_space_lower[:100],self.action_space_higher[:100]).reshape((10,10))
        #self.B = self.unnormalize(action[100:140],self.action_space_lower[100:140],self.action_space_higher[100:140]).reshape((10,4))
        self.Q = np.diag(self.unnormalize(action[:10],self.action_space_lower[:10],self.action_space_higher[:10])) # cost for states 
        self.R = np.diag(self.unnormalize(action[10:14],self.action_space_lower[10:14],self.action_space_higher[10:14])) # cost for input
        self.Q_bar = np.diag(self.unnormalize(action[14:],self.action_space_lower[14:],self.action_space_higher[14:]))

        self.Qs.append(np.diag(self.Q))
        self.Rs.append(np.diag(self.R))
        self.Q_bars.append(np.diag(self.Q_bar))

        # self.K = action[164:204].reshape((4,10))
        # self.c = action[204:].reshape((4,1))

        # print("A: ", self.A)
        # print("B: ", self.B)
        # print("Q: ", self.Q)
        # print("R: ", self.R)
        # print("Q_bar: ", self.Q_bar)

        self.f_sys = self.f_casadi()
        cost, control_input = self.mpc(self.state,self.reference)
        self.costs_opt.append(cost)
        self.Us.append(control_input)
        self.state = self.update_state(self.state,control_input,self.time,self.time + self.t)
        self.Xs.append(self.state)
        self.time = self.time + self.t
        self.ts.append(self.time)
        current_cost = (self.state - self.reference)@self.Q@(self.state - self.reference) + control_input@self.R@control_input
        self.costs_real.append(current_cost)

        if cost == 1000000:
            reward = -100
            terminated = True
            truncated = False
        else:
            reward = self.reward_function(self.state,self.reference) # Change between reward_function (Reward Function 1) and normalized_distance_reward (Reward Function 2)
            terminated = False
            truncated = self.time >= 10

        self.rewards.append(reward)

        if truncated or terminated:
            try:
                self.plot_info()
            except Exception as e:
                print("Error: ", e)
                print("Matplotlib Error")

        self.norm_state = self.normalize(self.state,self.observation_space_lower[:10],self.observation_space_higher[:10])
        self.norm_reference = self.normalize(self.reference,self.observation_space_lower[:10],self.observation_space_higher[:10])
        self.norm_time = self.normalize(self.time,self.observation_space_lower[20],self.observation_space_higher[20])
        
        return np.concatenate([self.norm_state, self.norm_reference, np.array([self.norm_time])]), reward, terminated, truncated, {}
        
    ### State update using quadrotor model ###

    def update_state(self,x0,u0,t_start,t_end):
        F = lambda t,x: self.f(x,u0).flatten()
        t_eval = np.linspace(t_start, t_end, self.N_update)
        sol = solve_ivp(F,[t_start,t_end],x0,t_eval=t_eval)
        x_vals = sol.y[:,-1]
        return x_vals
    
    def normalize(self, value, min_val, max_val):
        return (2 * ((value - min_val) / (max_val - min_val)) - 1)
    
    def unnormalize(self, normalized_value, min_val, max_val):
        return ((1 + normalized_value)*(max_val - min_val)/2 + min_val)
    
    def reward_function(self,state,reference):
        return -(((state[0]-reference[0])/reference[0])**2 + ((state[1]-reference[1])/reference[1])**2 + ((state[2]-reference[2])/reference[2])**2)
    
    def normalized_distance_reward(self,state,reference):
        return -(np.linalg.norm(state[:3]-reference[:3])/np.linalg.norm(reference[:3]-np.array([0,0,1])))**2

    def create_run_folder(self):
        # Initialize the new path with the base path
        base_path = self.save_path + f"run"
        counter = 1
        path = base_path
        # Check if the folder already exists
        while os.path.exists(path):
            # Modify the folder name by appending a counter
            path = base_path + f"_{counter}"
            counter += 1
        # Create the new directory
        os.makedirs(path)
        self.save_info_path = path

    def plot_info(self):
        self.episode += 1
        self.save_info_path_2 = self.save_info_path + f"/{self.episode}/"
        os.makedirs(self.save_info_path_2)

        self.Xs = np.vstack(self.Xs)
        self.Us = np.vstack(self.Us)
        self.ts = np.array(self.ts)
    
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        axes[0, 0].plot(self.ts, self.Xs[:,0])
        axes[0, 0].set_title('X position')
        axes[0, 0].set_xlabel('Time (s)')
        axes[0, 0].set_ylabel('Distance (m)')


        axes[0, 1].plot(self.ts, self.Xs[:,1])
        axes[0, 1].set_title('Y Position')
        axes[0, 1].set_xlabel('Time (s)')
        axes[0, 1].set_ylabel('Distance (m)')


        axes[0, 2].plot(self.ts, self.Xs[:,2])
        axes[0, 2].set_title('Z Position')
        axes[0, 2].set_xlabel('Time (s)')
        axes[0, 2].set_ylabel('Distance (m)')


        axes[1, 0].plot(self.ts, self.Xs[:,3])
        axes[1, 0].set_title('X Velocity')
        axes[1, 0].set_xlabel('Time (s)')
        axes[1, 0].set_ylabel('Velocity (m/s)')


        axes[1, 1].plot(self.ts, self.Xs[:,4])
        axes[1, 1].set_title('Y Velocity')
        axes[1, 1].set_xlabel('Time (s)')
        axes[1, 1].set_ylabel('Velocity (m/s)')


        axes[1, 2].plot(self.ts, self.Xs[:,5])
        axes[1, 2].set_title('Z Velocity')
        axes[1, 2].set_xlabel('Time (s)')
        axes[1, 2].set_ylabel('Velocity (m/s)')

        fig.suptitle('Quadrotor State Over Time', fontsize=16)

        plt.tight_layout()
        
        traj_name = self.save_info_path_2 + f"traj_{self.episode}.png"
        plt.savefig(traj_name)
        #plt.close(fig)

        fig2, axes2 = plt.subplots(2, 2, figsize=(10, 10))

        axes2[0, 0].plot(self.ts[:-1], self.Us[:,0])
        axes2[0, 0].set_title('Thrust')
        axes2[0, 0].set_xlabel('Time (s)')
        axes2[0, 0].set_ylabel('Force (N)')

        axes2[0, 1].plot(self.ts[:-1], self.Us[:,1])
        axes2[0, 1].set_title('X Angular Velocity')
        axes2[0, 1].set_xlabel('Time (s)')
        axes2[0, 1].set_ylabel('Angular Velocity (rad/s)')


        axes2[1, 0].plot(self.ts[:-1], self.Us[:,2])
        axes2[1, 0].set_title('Y Angular Velocity')
        axes2[1, 0].set_xlabel('Time (s)')
        axes2[1, 0].set_ylabel('Angular Velocity (rad/s)')


        axes2[1, 1].plot(self.ts[:-1], self.Us[:,3])
        axes2[1, 1].set_title('Z Angular Velocity')
        axes2[1, 1].set_xlabel('Time (s)')
        axes2[1, 1].set_ylabel('Angular Velocity (rad/s)')

        fig2.suptitle('Quadrotor Input Over Time', fontsize=16)

        plt.tight_layout()
        input_name = self.save_info_path_2 + f"input_{self.episode}.png"
        plt.savefig(input_name)
        #plt.close(fig2)

        fig3 = plt.figure()
        plt.plot(self.ts[:-1],self.costs_opt)
        plt.title("MPC Objective Cost over Time")
        plt.xlabel("Time (s)")
        plt.ylabel("MPC Objective Cost Over Prediction Horizon")

        cost_name = self.save_info_path_2 + f"ph_costs_{self.episode}.png"
        plt.savefig(cost_name)
        #plt.close(fig3)

        fig4 = plt.figure()
        plt.plot(self.ts[:-1],self.costs_real)
        plt.title("Current MPC Objective Cost over Time")
        plt.xlabel("Time (s)")
        plt.ylabel("Current MPC Objective Cost")

        cost_name = self.save_info_path_2 + f"cur_costs_{self.episode}.png"
        plt.savefig(cost_name)

        fig5 = plt.figure()
        plt.plot(self.ts[:-1],self.rewards)
        plt.title("Reward over Time")
        plt.xlabel("Time (s)")
        plt.ylabel("Reward")

        cost_name = self.save_info_path_2 + f"rewards_{self.episode}.png"
        plt.savefig(cost_name)

        self.Qs_array = np.array(self.Qs)
        self.Rs_array = np.array(self.Rs)
        self.Q_bars_array = np.array(self.Q_bars)

        fig6, axes6 = plt.subplots(3, 1, figsize=(20,10))
        
        axes6[0].plot(self.ts[:-1], self.Qs_array[:,0], label='Q0')
        axes6[0].plot(self.ts[:-1], self.Qs_array[:,1], label='Q1')
        axes6[0].plot(self.ts[:-1], self.Qs_array[:,2], label='Q2')
        axes6[0].plot(self.ts[:-1], self.Qs_array[:,3], label='Q3')
        axes6[0].plot(self.ts[:-1], self.Qs_array[:,4], label='Q4')
        axes6[0].plot(self.ts[:-1], self.Qs_array[:,5], label='Q5')
        axes6[0].plot(self.ts[:-1], self.Qs_array[:,6], label='Q6')
        axes6[0].plot(self.ts[:-1], self.Qs_array[:,7], label='Q7')
        axes6[0].plot(self.ts[:-1], self.Qs_array[:,8], label='Q8')
        axes6[0].plot(self.ts[:-1], self.Qs_array[:,9], label='Q9')

        axes6[0].set_title('Q Matrix Values Over Time')
        axes6[0].set_xlabel('Time (s)')
        axes6[0].set_ylabel('Value')
        axes6[0].legend(loc='right')

        axes6[1].plot(self.ts[:-1], self.Q_bars_array[:,0], label='Qbar0')
        axes6[1].plot(self.ts[:-1], self.Q_bars_array[:,1], label='Qbar1')
        axes6[1].plot(self.ts[:-1], self.Q_bars_array[:,2], label='Qbar2')
        axes6[1].plot(self.ts[:-1], self.Q_bars_array[:,3], label='Qbar3')
        axes6[1].plot(self.ts[:-1], self.Q_bars_array[:,4], label='Qbar4')
        axes6[1].plot(self.ts[:-1], self.Q_bars_array[:,5], label='Qbar5')
        axes6[1].plot(self.ts[:-1], self.Q_bars_array[:,6], label='Qbar6')
        axes6[1].plot(self.ts[:-1], self.Q_bars_array[:,7], label='Qbar7')
        axes6[1].plot(self.ts[:-1], self.Q_bars_array[:,8], label='Qbar8')
        axes6[1].plot(self.ts[:-1], self.Q_bars_array[:,9], label='Qbar9')

        axes6[1].set_title('Q Bar Matrix Values Over Time')
        axes6[1].set_xlabel('Time (s)')
        axes6[1].set_ylabel('Value')
        axes6[1].legend(loc='right')

        axes6[2].plot(self.ts[:-1], self.Rs_array[:,0], label='R0')
        axes6[2].plot(self.ts[:-1], self.Rs_array[:,1], label='R1')
        axes6[2].plot(self.ts[:-1], self.Rs_array[:,2], label='R2')
        axes6[2].plot(self.ts[:-1], self.Rs_array[:,3], label='R3')

        axes6[2].set_title('R Matrix Values Over Time')
        axes6[2].set_xlabel('Time (s)')
        axes6[2].set_ylabel('Value')
        axes6[2].legend(loc='right')

        fig6.suptitle('MPC Parameter Matrices Over Time', fontsize=16)

        plt.tight_layout()
        
        matrix_name = self.save_info_path_2 + f"matrices_{self.episode}.png"
        plt.savefig(matrix_name)
        #plt.close(fig4)
        plt.close('all')

def test_model(env,model_path):
    model = SAC.load(model_path)
    obs = env.reset()
    terminated = False
    truncated = False
    while not (terminated or truncated):
        action, _states = model.predict(obs)
        obs, rewards, terminated, truncated = env.step(action)
        terminated = terminated[0]
        truncated = truncated[0]["TimeLimit.truncated"]

    env.close()

def train(env):
    model = SAC('MlpPolicy', env, tensorboard_log='tensorboard_logs', verbose=1, seed=1)
    print(model.policy)
    model.learn(total_timesteps=10000, log_interval=1, progress_bar=True)
    model.save("sac_drone_1")

    for i in range(99):
        model.learn(total_timesteps=10000, log_interval=1, reset_num_timesteps=False, progress_bar=True)
        model.save(f"sac_drone_{i+2}")

# Example usage:
if __name__ == "__main__":
    
    tr = True # True or False depending on training or testing
    model_path = "Distance_Reward/run_reward_function_1_train/sac_drone_66_best.zip" # Only used for testing

    # Reinforcement Learning
    env = drone_mpc_env()
    env = Monitor(env)  # Monitor wrapper
    env = make_vec_env(lambda: env, n_envs=1)  # DummyVecEnv wrapper

    if tr:
        train(env)
    else:
        test_model(env,model_path)
    #env = VecNormalize(env)

    # The noise objects for TD3
    # n_actions = env.action_space.shape[-1]
    # action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.01*np.ones(n_actions))

    #model = TD3('MlpPolicy', env, learning_rate = 1e-4, action_noise=action_noise, tensorboard_log='C:/Users/kevin/Desktop/MEng_Project/tensorboard_logs', verbose=1, seed=1)

    #callback = SummaryWriterCallback()
    #model.learn(total_timesteps=10000, callback=callback, log_interval=10, progress_bar=True)
    