#%%
import numpy as np
import pandas as pd
import sympy as sp
import casadi as ca
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R
from scipy.integrate import solve_ivp
from scipy.optimize import root
from scipy.linalg import solve_continuous_are
from scipy.linalg import solve_discrete_lyapunov

import matplotlib.pyplot as plt
import trajectory_ASV as traj
from mpl_toolkits.mplot3d import Axes3D

# Transform
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

#Rotation matrix to quaternion
def R_to_q(R):
    q0 = sp.sqrt(1 + R[0, 0] + R[1, 1] + R[2, 2]) / 2  # qw
    q1 = (R[2, 1] - R[1, 2]) / (4 * q0)  # qx
    q2 = (R[0, 2] - R[2, 0]) / (4 * q0)  # qy
    q3 = (R[1, 0] - R[0, 1]) / (4 * q0)  # qz

    return sp.simplify(sp.Matrix([q0, q1, q2, q3]))

#Quaternion Product
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

#Quaternion Vector
def q_vec(v):
    return sp.Matrix([v[0],v[1],v[2],0])

#Inverse Quaternion
def q_inv(v):
    return sp.Matrix([-v[0],-v[1],-v[2],v[3]])

#Quaternion Product Matrix
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

#Sympy to Casadi
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

# Function to smooth trajecotry gradients
def smooth(arr,num_smooth):

    arr[0] = arr[1]
    arr[-1] = arr[-2]

    num_side = int(1 + ((num_smooth - 1) / 2))

    diff1 = (arr[500 + num_side] - arr[500 - num_side]) / (num_smooth + 1)
    diff2 = (arr[1000 + num_side] - arr[1000 - num_side]) / (num_smooth + 1)

    for i in range(1,num_smooth+1,1):
        arr[500-num_side+i] = arr[500-num_side] + i*diff1
        arr[1000-num_side+i] = arr[1000-num_side] + i*diff2

    return arr

class quaternion_uav:
    def __init__(self):
        # Constants
        self.m = 1.0175
        self.g = 9.8066

        # Time step
        self.t = 0.01
        self.N_update = 10

        # MPC Parameters
        self.N = 40  # Prediction horizon time steps
        self.X_dim = 10
        self.U_dim = 4

        self.Q = np.diag(np.array([1000,1000,1000,1,1,1,100,100,100,100])) # cost for states 
        self.R = np.diag(np.array([1,1,1,1])) # cost for input


        self.Q_bar = 10*self.Q #comment

        # Use P matrix from LQR
        # Procedure to find terminal state matrix:
        # Linearize quadrotor system
        # Find optimal controller gain K using LQR
        # Solve discrete Lyapunov equation for Q_bar

        self.f = self.f_syms() # Load sympy quaternion model
        self.u_ref = self.f_ref_syms() # Load sympy trajectory reference expression 

        #self.fx = self.feq_syms()
        #self.A, self.B = self.linearize()

        #self.is_controllable(self.A,self.B)

        #self.K = self.LQR()
        #self.Q_bar = self.solve_lyapunov() #cost for terminal state #LQR

        self.f_sys = self.f_casadi() # Convert from sympy to casadi model
        self.create_casadi_integrator() # Create casadi integration for mpc
        self.load_test() # Load csv simulation
    
    # Sympy drone quaternion model
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
    
    # Sympy trajectory reference model/expression
    def fref(self,x,v_dot,q_dot):
        q = x[6:,0]
        e = sp.Matrix([[0],[0],[1]])
        Fe3 = self.m*(R_GB(q).inv())@(v_dot + self.g*e)
        F = sp.Matrix([Fe3[2]])
        q_ws = 2*(quatProd(q).inv())@q_dot
        ws = sp.Matrix(q_ws[1:])
        u_ref = sp.Matrix.vstack(F,ws)

        return u_ref
    
    # Convert reference to sympy function
    def f_ref_syms(self):
        X = sp.Matrix(sp.symbols("x,y,z,vx,vy,vz,qw,qx,qy,qz"))
        V_dot = sp.Matrix(sp.symbols("ax,ay,az"))
        Q_dot = sp.Matrix(sp.symbols("Qw,Qx,Qy,Qz"))
        f_ref_s = sp.lambdify([X,V_dot,Q_dot],self.fref(X,V_dot,Q_dot), 'numpy')
        return f_ref_s
    
    # Convert drone model to sympy function
    def f_syms(self):
        X = sp.Matrix(sp.symbols("x,y,z,vx,vy,vz,qw,qx,qy,qz"))
        U = sp.Matrix(sp.symbols("F,wx,wy,wz"))
        self.X = X
        self.U = U
        f_s = sp.lambdify([X,U],self.fdot(X,U), 'numpy')
        return f_s
    
    # Linearized quaternion model of quadrotor (Currently not used)
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
    
    # Convert linearized quaternion model of quadrotor (Currently not used)
    def feq_syms(self):
        XU = sp.Matrix(sp.symbols("x,y,z,vx,vy,vz,qw,qx,qy,qz,F,wx,wy,wz"))
        self.XU = XU
        feq_s = sp.lambdify([XU],self.feq(XU), 'numpy')  
        return feq_s
    
    # Linearize quaternion model (Currently not used)
    def linearize(self):
        Fx = lambda x: self.fx(x).flatten()
        x0_init = np.array([0,0,1,0,0,0,1,0,0,0,9.98,0,0,0])
        x0 = root(Fx,x0_init).x
        vals = {'x': x0[0], 'y':  x0[1], 'z':  x0[2],'vx':  x0[3], 'vy':  x0[4], 'vz':  x0[5], 'qw':  x0[6], 'qx':  x0[7], 'qy':  x0[8], 'qz':  x0[9], 'F':  x0[10], 'wx':  x0[11], 'wy':  x0[12], 'wz':  x0[13]}

        A = self.f_dot.jacobian(self.X)
        print(A)
        A = A.subs(vals)
        print(A)

        B = self.f_dot.jacobian(self.U)
        print(B)
        B = B.subs(vals)
        print(B)

        return np.array(A).astype(float), np.array(B).astype(float)
    
    # Check controllability (Currently not used)
    def is_controllable(self, A, B):
        n = A.shape[0]
        controllability_matrix = B
        for i in range(1, n):
            controllability_matrix = np.hstack((controllability_matrix, np.linalg.matrix_power(A, i).dot(B)))
        
        rank = np.linalg.matrix_rank(controllability_matrix)

        if rank == n:
            print("Controllable")
        else:
            print("Not Controllable")
        return
    
    # Create LQR (Currently not used)
    def LQR(self):
        P = solve_continuous_are(self.A, self.B, self.Q, self.R)
        print("P:", P)
        K = np.linalg.inv(self.R) @ self.B.T @ P
        print("K:", K)
        return K
    
    # Solve lyapunov equation (Currently not used)
    def solve_lyapunov(self):
        # Define the matrices A and Q
        A = self.A + self.B@self.K
        Q = self.Q + self.K.T@self.R@self.K
        # Solve the discrete-time Lyapunov equation AXA^T - X + Q = 0
        X = solve_discrete_lyapunov(A.T, Q)
        print("X:", X)
        return X
    
    # Convert sympy function to casadi
    def f_casadi(self):
        X_cas = ca.SX.sym('X', self.X_dim)
        U_cas = ca.SX.sym('U', self.U_dim)
        XU = self.X.col_join(self.U)
        XU_cas = ca.vertcat(X_cas,U_cas)
        f_casadi = sympy2casadi(self.f_dot, XU, XU_cas)
        f_func = ca.Function('f', [X_cas, U_cas], [f_casadi])
        return f_func
    
    # Create casadi integrator
    def create_casadi_integrator(self):
        x = ca.MX.sym('x', 10)
        u = ca.MX.sym('u', 4)
        x_dot = self.f_sys(x, u)
        dae = {'x': x, 'p': u, 'ode': x_dot}
        opts = {'tf': self.t}  # Integration step size
        self.integrator = ca.integrator('integrator', 'rk', dae, opts)

    # System dynamics for MPC
    def add_system_dynamics(self):
        for k in range(self.N):
            x_k = self.X[:, k]
            u_k = self.U[:, k]
            x_next = self.integrator(x0=x_k, p=u_k)['xf']
            self.opti.subject_to(self.X[:, k+1] == x_next)
    
    def integrator_rk4(self):
        X = ca.MX.sym('x', 10)
        U = ca.MX.sym('u', 4)
        k1= self.f_sys(X, U)
        k2 = self.f_sys(X + self.t/2 * k1, U)
        k3 = self.f_sys(X + self.t/2 * k2, U)
        k4 = self.f_sys(X + self.t * k3, U)
        Xp=X+self.t/6*(k1 +2*k2 +2*k3 +k4)
        F =  ca.Function('F', [X, U], [Xp],['x0','u'],['xf'])
        return F

    # MPC Function
    def mpc(self, x0, x_ref):
        # Optimization variables
        opti = ca.Opti()
        X = opti.variable(self.X_dim, self.N+1)
        U = opti.variable(self.U_dim, self.N)
        # Cost function
        cost = 0
        for k in range(self.N):
            cost += ca.mtimes([(X[:, k] - x_ref[k,:]).T, self.Q, X[:, k] - x_ref[k,:]]) + ca.mtimes([U[:, k].T, self.R, U[:, k]])

        cost += ca.mtimes([(X[:, self.N] - x_ref[self.N,:]).T, self.Q_bar, X[:, self.N] - x_ref[self.N,:]]) # cost for terminal state

    ##### System dynamics constraints
        F = self.integrator_rk4()

        # Non Integration
        for k in range(self.N):
            x_next = F(X[:, k], U[:, k])
            opti.subject_to(X[:, k+1] == x_next)

        # Integration
        # for k in range(self.N):
        #     x_next = self.integrator(x0=X[:, k], p=U[:, k])['xf']
        #     opti.subject_to(X[:, k+1] == x_next)
    
    #####

        # Input constraints
        opti.subject_to(opti.bounded(0, U[0,:], 30))
        opti.subject_to(opti.bounded(-10, U[1,:], 10))
        opti.subject_to(opti.bounded(-10, U[2,:], 10))
        opti.subject_to(opti.bounded(-3, U[3,:], 3))

        # Unit Quaternion Constraint
        epsilon = 0.001
        qw = X[6,:]
        qx = X[7,:]
        qy = X[8,:]
        qz = X[9,:]
        opti.subject_to(opti.bounded(-epsilon, (qw**2 + qx**2 + qy**2 + qz**2) - 1, epsilon))

        # Minimum Height Constraint
        opti.subject_to(opti.bounded(0.5, X[2,:], 15))

        # Initial condition constraint
        opti.subject_to(X[:, 0] == x0)

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
        sol = opti.solve()

        # Get the optimal control input
        u_optimal = sol.value(U[:, 0])
        #print("Optimal control input:", u_optimal)

        return u_optimal
    
    # Simulation integration function
    def update_state(self,x0,u0,t_start,t_end):
        F = lambda t,x: self.f(x,u0).flatten()
        t_eval = np.linspace(t_start, t_end, self.N_update)
        sol = solve_ivp(F,[t_start,t_end],x0,t_eval=t_eval)
        x_vals = sol.y[:,-1]
        #Normalize quaternion
        q = x_vals[6:10]
        q_norm = np.linalg.norm(q)
        x_vals[6:10] = q/q_norm
        return x_vals
    
    # Load csv file to test model
    def load_test(self):
        file = "race_0323_slits.csv"
        df = pd.read_csv(file,index_col=None,header=0)
        idx_x = ['p_x','p_y','p_z','v_x','v_y','v_z','q_w','q_x','q_y','q_z']
        idx_u = ['thrust','w_x','w_y','w_z']
        U_r = df[idx_u]
        U_r.loc[:,'thrust'] = U_r['thrust']
        self.X_r = df[idx_x].to_numpy()
        self.U_r = U_r.to_numpy()

# Main Function

if __name__ == "__main__":
    
    test = 1 # Set test = 1 to test quaternion model, test = 2 to test MPC
    traj_type = 1 # Trajectory number to test (0 - eight, 1 - line, 2 - circle, 3 - wave, 4 - ellipse, 5 - hover)

    if test == 1:
        # Test model

        # Set quaternion model
        uav = quaternion_uav()

        Ts = uav.t

        cost = 0
        x0 = uav.X_r[0,:].copy()
        x0[0] = -1
        x0[1] = -2
        x = []
        x_ref = []
        err = []
        t = []
        q = []
        q_ref = []

        x_err = []

        x.append(x0[:3])
        x_ref.append(uav.X_r[0, 0:3])
        t.append(0)
        err.append(np.linalg.norm(x0[:3] - x0[:3]))
        x_err.append(x0[:3] - x0[:3])

        q.append(x0[6::])
        q_ref.append(uav.X_r[0, 6::])
        
        # MPC loop

        for i in range(1000):
            print(i)
            x_r_th = uav.X_r[i:i+uav.N+1, :]
            u0 = uav.mpc(x0,x_r_th)
            # u0 = uav.U_r[i,:]
            t0 = Ts*i
            tp = Ts*(i+1)
            t.append(tp)

            # Integrate update state
            xp = uav.update_state(x0,u0,t0,tp)
            
            # Update total cost
            cost = cost + np.linalg.norm(xp - uav.X_r[i+1,:])

            # Update data lists
            x.append(xp[:3])
            x_ref.append(uav.X_r[i+1,:][:3])
            err.append(np.linalg.norm(xp[:3] - uav.X_r[i+1,:][:3]))
            x_err.append(xp[:3] - uav.X_r[i+1,:][:3])

            q.append(xp[6:10])  
            q_ref.append(uav.X_r[i+1,:][6:10])

            x0 = xp
        
        # Convert data for plotting
        Xs = np.vstack(x)
        X_refs = np.vstack(x_ref)
        x_errs = np.vstack(x_err)
        Qs = np.vstack(q)
        Q_refs = np.vstack(q_ref)


        ts = np.array(t)
        errs = np.array(err)

        #Plot error

        fig_err = plt.figure()
        plt.plot(ts,errs,color='blue')
        plt.title('Simulation Position Error Over Time')
        plt.xlabel('Time (s)')
        plt.ylabel('Position Error (m)')

        ###

        # Position Error Plot

        fig1, axes = plt.subplots(1, 3, figsize=(15, 5))

        axes[0].plot(ts, x_errs[:,0], color='red')
        axes[0].set_title('X Position Error')
        axes[0].set_xlabel('Time (s)')
        axes[0].set_ylabel('Distance (m)')

        axes[1].plot(ts, x_errs[:,1], color='red')
        axes[1].set_title('Y Position Error')
        axes[1].set_xlabel('Time (s)')
        axes[1].set_ylabel('Distance (m)')

        axes[2].plot(ts, x_errs[:,2], color='red')
        axes[2].set_title('Z Position Error')
        axes[2].set_xlabel('Time (s)')
        axes[2].set_ylabel('Distance (m)')

        fig1.suptitle('Quadrotor Position Error Over Time', fontsize=16)
        plt.tight_layout()

        ###

        #Quadrotor state plot

        fig2, axes = plt.subplots(1, 3, figsize=(15, 5))

        axes[0].plot(ts, Xs[:,0], label='Simulation', color='blue')
        axes[0].plot(ts, X_refs[:,0], label='Reference', color='red')
        axes[0].set_title('X position')
        axes[0].set_xlabel('Time (s)')
        axes[0].set_ylabel('Distance (m)')


        axes[1].plot(ts, Xs[:,1], label='Simulation', color='blue')
        axes[1].plot(ts, X_refs[:,1], label='Reference', color='red')
        axes[1].set_title('Y Position')
        axes[1].set_xlabel('Time (s)')
        axes[1].set_ylabel('Distance (m)')


        axes[2].plot(ts, Xs[:,2], label='Simulation', color='blue')
        axes[2].plot(ts, X_refs[:,2], label='Reference', color='red')
        axes[2].set_title('Z Position')
        axes[2].set_xlabel('Time (s)')
        axes[2].set_ylabel('Distance (m)')

        axes[0].legend()
        axes[1].legend()
        axes[2].legend()

        fig2.suptitle('Quadrotor State Over Time', fontsize=16)
        plt.tight_layout()

        fig3, axes = plt.subplots(1, 4, figsize=(15, 5))

        axes[0].plot(ts, Qs[:, 0], label='Simulation', color='blue')
        axes[0].plot(ts, Q_refs[:,0], label='Reference', color='red')
        axes[0].set_title('qw')
        axes[0].set_xlabel('Time (s)')
        axes[0].set_ylabel('Distance (m)')


        axes[1].plot(ts, Qs[:,1], label='Simulation', color='blue')
        axes[1].plot(ts, Q_refs[:,1], label='Reference', color='red')
        axes[1].set_title('qx')
        axes[1].set_xlabel('Time (s)')
        axes[1].set_ylabel('Distance (m)')


        axes[2].plot(ts, Qs[:,2], label='Simulation', color='blue')
        axes[2].plot(ts, Q_refs[:,2], label='Reference', color='red')
        axes[2].set_title('qy')
        axes[2].set_xlabel('Time (s)')
        axes[2].set_ylabel('Distance (m)')

        axes[3].plot(ts, Qs[:,3], label='Simulation', color='blue')
        axes[3].plot(ts, Q_refs[:,3], label='Reference', color='red')
        axes[3].set_title('qz')
        axes[3].set_xlabel('Time (s)')
        axes[3].set_ylabel('Distance (m)')


        axes[0].legend()
        axes[1].legend()
        axes[2].legend()
        axes[3].legend()

        fig2.suptitle('Quadrotor Quaternion Over Time', fontsize=16)
        plt.tight_layout()
        ###

        # Plot trajectory

        fig3d = plt.figure()
        ax = fig3d.add_subplot(111, projection='3d')

        ref_traj_x = X_refs[:,0]
        ref_traj_y = X_refs[:,1]
        ref_traj_z = X_refs[:,2]

        real_traj_x = Xs[:,0]
        real_traj_y = Xs[:,1]
        real_traj_z = Xs[:,2]

        # Plot reference trajectory as a solid line
        ax.plot(ref_traj_x, ref_traj_y, ref_traj_z, label='Reference Trajectory', color='blue')

        # Plot real trajectory as a solid line
        ax.plot(real_traj_x, real_traj_y, real_traj_z, label='Real Trajectory', color='red')

        # Add labels
        ax.set_xlabel('X axis')
        ax.set_ylabel('Y axis')
        ax.set_zlabel('Z axis')

        # Add title
        ax.set_title('Reference vs Real Trajectory of Quadrotor using MPC')

        # Add legend
        ax.legend()

        # Display the plot
        plt.show()

    elif test == 2:

        # Test MPC

        # Create quaternion model of quadrotor
        uav = quaternion_uav()

        Ts = uav.t

        Xs = []
        X_refs = []
        X_urefs = []

        Us = []
        ts = []

        # Create trajectory function
        trajectory = traj.trajecotry()
        data = [(0,0),(10,10),0,5,(10,10),(10,10),5,10.4,1,2,{},{'Amplitude': 5}]
        
        # Specify trajectory
        tra = trajectory.genTrajCombined(data)

        # Get trajectory psi angle expression
        psi_exp = trajectory.psi_exp

        ######

        # Get reference trajectory numerical data from expression
        x_ref = tra['XRef_func']
        y_ref = tra['YRef_func']
        psi_ref = tra['psiRef_func']

        x_ref_d = tra['XRefdot_func']
        y_ref_d = tra['YRefdot_func']

        x_ref_dd = tra['XRefddot_func']
        y_ref_dd = tra['YRefddot_func']

        times = np.arange(0,10.01 + uav.N*0.01,0.01)
        a_s = 1
        l_s = 5

        x_s = x_ref(a_s,times,l_s)
        y_s = y_ref(a_s,times,l_s)
        psi_s = psi_ref(a_s,times,l_s)

        dx_s = x_ref_d(a_s,times,l_s) 
        dy_s = y_ref_d(a_s,times,l_s)

        ddx_s = x_ref_dd(a_s,times,l_s)
        ddy_s = y_ref_dd(a_s,times,l_s)

        # Smooth trajectory data

        dx_s = smooth(dx_s,5)
        dy_s = smooth(dy_s,5)
        ddx_s = smooth(ddx_s,5)
        ddy_s = smooth(ddy_s,5)

        ######
        
        # Convert psi angle to rotation matrix
        angle = sp.symbols('psi')
        rot = sp.Matrix([
                    [sp.cos(angle),-sp.sin(angle),0],[sp.sin(angle), sp.cos(angle),0],[0,0,1]                     
                    ])
        
        # Convert rotation matrix to quaternion in sympy
        q_syms = R_to_q(rot)

        # Substitute angle expression in quaternion expression
        q_t = q_syms.subs({angle: psi_exp})
        q_dt = q_t.diff(trajectory.t)
        print(q_dt)
        qw_dt = q_dt[0]
        qx_dt = q_dt[1]
        qy_dt = q_dt[2]
        qz_dt = q_dt[3]

        # Heaviside and dirac delta function

        def heaviside(x, zero_value=0):
            return np.where(x > 0, 1, 0)
        # Custom DiracDelta function for numpy

        def dirac_delta(x, *args):
            # Define a small threshold for delta approximation
            epsilon = 1e-5
            return np.where(np.abs(x) < epsilon, 1/epsilon, 0)
        
        # Add Heaviside and dirac delta function into function expression
        
        qw_dt_func = sp.lambdify((trajectory.a,trajectory.t,trajectory.t_len), qw_dt, {'Heaviside': heaviside, 'DiracDelta': dirac_delta, 'numpy': np})
        qw_dt_func = np.vectorize(qw_dt_func)

        qx_dt_func = sp.lambdify((trajectory.a,trajectory.t,trajectory.t_len), qx_dt, {'Heaviside': heaviside, 'DiracDelta': dirac_delta, 'numpy': np})
        qx_dt_func = np.vectorize(qx_dt_func)

        qy_dt_func = sp.lambdify((trajectory.a,trajectory.t,trajectory.t_len), qy_dt, {'Heaviside': heaviside, 'DiracDelta': dirac_delta, 'numpy': np})
        qy_dt_func = np.vectorize(qy_dt_func)

        qz_dt_func = sp.lambdify((trajectory.a,trajectory.t,trajectory.t_len), qz_dt, {'Heaviside': heaviside, 'DiracDelta': dirac_delta, 'numpy': np})
        qz_dt_func = np.vectorize(qz_dt_func)

        # Convert to sympy rotation matrix expression
        R_func = sp.lambdify([angle], rot, modules='numpy')
        rots = np.array([R_func(psi) for psi in psi_s])

        # Convert rotation matrix expression to quaternion expression
        rs = R.from_matrix(rots)
        qs = rs.as_quat()
        qs = np.hstack((qs[:, -1:],qs[:, :-1]))

        #####

        # Convert to numerical data

        x0 = np.array([0,0,1,0,0,0,1,0,0,0])
        x0_ref = x0
        
        x_r = np.concatenate((np.array([x_s[0],y_s[0],1,dx_s[0],dy_s[0],0]),qs[0,:]))
        x_r_th = np.vstack((x_s[:uav.N],y_s[:uav.N],1*np.ones((uav.N,)),dx_s[:uav.N],dy_s[:uav.N],np.zeros((uav.N,)))).T
        x_r_th = np.hstack((x_r_th,qs[:uav.N,:]))

        #####

        # Figure out reference output based on quaternion model 
        
        # Reference u
        ref_traj_u = uav.u_ref

        # Acceleration
        a_x = ddx_s[:-1]
        a_y = ddy_s[:-1]
        a_z = np.zeros_like(a_x)
        a = np.vstack((a_x,a_y,a_z))

        # Quaternion derivative for reference output function
        qw_dot = qw_dt_func(a_s,times,l_s)
        qx_dot = qx_dt_func(a_s,times,l_s)
        qy_dot = qy_dt_func(a_s,times,l_s)
        qz_dot = qz_dt_func(a_s,times,l_s)

        qw_dot = qw_dot[:-1]
        qx_dot = qx_dot[:-1]
        qy_dot = qy_dot[:-1]
        qz_dot = qz_dot[:-1]

        qw_dot = smooth(qw_dot,5)
        qx_dot = smooth(qx_dot,5)
        qy_dot = smooth(qy_dot,5)
        qz_dot = smooth(qz_dot,5)

        # Quaternion derivatives for reference output function

        q_dot = np.vstack((qw_dot,qx_dot,qy_dot,qz_dot))

        # Reference trajectory
        x_r_full = np.vstack((x_s.reshape(1, -1),y_s.reshape(1, -1),np.ones((1,x_s.shape[0])),
                                        dx_s.reshape(1, -1),dy_s.reshape(1, -1),np.zeros((1,x_s.shape[0])),qs.T))
        
        # Reference output
        ref_u = np.squeeze(ref_traj_u(x_r_full[:,:-1],a,q_dot))
        ref_u = ref_u[:,:(x_s.shape[0]-1)]
        ref_u = ref_u.T
        
        #####

        # Collect data
        Xs.append(x0)
        
        X_refs.append(x_r)
        X_urefs.append(x0_ref)

        ts.append(0)

        t_steps = 625

        ref_u = ref_u[:t_steps,:]

        for i in tqdm(range(t_steps)):

            # Compare mpc with to reference output
            u0 = uav.mpc(x0,x_r_th)
            u0_ref = ref_u[i,:]

            Us.append(u0)
            #print("U_current: ", u0)

            t0 = Ts*i
            tp = Ts*(i+1)
            ts.append(tp)

            # Update state based on mpc or reference trajectory
            xp = uav.update_state(x0,u0,t0,tp)
            xp_ref = uav.update_state(x0_ref,u0_ref,t0,tp)

            x0 = xp
            x0_ref = xp_ref

            # Update data

            x_r = np.concatenate((np.array([x_s[i+1],y_s[i+1],1,dx_s[i+1],dy_s[i+1],0]),qs[i+1,:]))
            x_r_th = np.vstack((x_s[i+1:i+1+uav.N],y_s[i+1:i+1+uav.N],1*np.ones((uav.N,)),dx_s[i+1:i+1+uav.N],dy_s[i+1:i+1+uav.N],np.zeros((uav.N,)))).T
            x_r_th = np.hstack((x_r_th,qs[i+1:i+1+uav.N,:]))

            Xs.append(x0)
            X_urefs.append(x0_ref)
            X_refs.append(x_r)

            #print("X_current: ", x0)
            #print("X_error: ", x_ref - x0)

        # Convert data to format for plotting
        Xs = np.vstack(Xs)
        X_refs = np.vstack(X_refs)
        X_urefs = np.vstack(X_urefs)

        Us = np.vstack(Us)
        ts = np.array(ts)

        #Plot quadrotor state over time
        fig, axes = plt.subplots(3, 3, figsize=(15, 10))

        axes[0, 0].plot(ts, Xs[:,0], color='blue')
        axes[0, 0].plot(ts, X_urefs[:,0], color='red')
        axes[0, 0].set_title('X position')
        axes[0, 0].set_xlabel('Time (s)')
        axes[0, 0].set_ylabel('Distance (m)')


        axes[0, 1].plot(ts, Xs[:,1], color='blue')
        axes[0, 1].plot(ts, X_urefs[:,1], color='red')
        axes[0, 1].set_title('Y Position')
        axes[0, 1].set_xlabel('Time (s)')
        axes[0, 1].set_ylabel('Distance (m)')


        axes[0, 2].plot(ts, Xs[:,2], color='blue')
        axes[0, 2].plot(ts, X_urefs[:,2], color='red')
        axes[0, 2].set_title('Z Position')
        axes[0, 2].set_xlabel('Time (s)')
        axes[0, 2].set_ylabel('Distance (m)')


        axes[1, 0].plot(ts, Xs[:,3], color='blue')
        axes[1, 0].plot(ts, X_urefs[:,3], color='red')
        axes[1, 0].set_title('X Velocity')
        axes[1, 0].set_xlabel('Time (s)')
        axes[1, 0].set_ylabel('Velocity (m/s)')


        axes[1, 1].plot(ts, Xs[:,4], color='blue')
        axes[1, 1].plot(ts, X_urefs[:,4], color='red')
        axes[1, 1].set_title('Y Velocity')
        axes[1, 1].set_xlabel('Time (s)')
        axes[1, 1].set_ylabel('Velocity (m/s)')


        axes[1, 2].plot(ts, Xs[:,5], color='blue')
        axes[1, 2].plot(ts, X_urefs[:,5], color='red')
        axes[1, 2].set_title('Z Velocity')
        axes[1, 2].set_xlabel('Time (s)')
        axes[1, 2].set_ylabel('Velocity (m/s)')

        axes[2, 0].plot(ts, Xs[:,6], color='blue')
        axes[2, 0].plot(ts, X_urefs[:,6], color='red')
        axes[2, 0].set_title('Q1')
        axes[   2, 0].set_xlabel('Time (s)')
        axes[2, 0].set_ylabel('Velocity (m/s)')


        axes[2, 1].plot(ts, Xs[:,7], color='blue')
        axes[2, 1].plot(ts, X_urefs[:,7], color='red')
        axes[2, 1].set_title('Y Velocity')
        axes[2, 1].set_xlabel('Time (s)')
        axes[2, 1].set_ylabel('Velocity (m/s)')


        axes[2, 2].plot(ts, Xs[:,8], color='blue')
        axes[2, 2].plot(ts, X_urefs[:,8], color='red')
        axes[2, 2].set_title('Z Velocity')
        axes[2, 2].set_xlabel('Time (s)')
        axes[2, 2].set_ylabel('Velocity (m/s)')

        fig.suptitle('Quadrotor State Over Time', fontsize=16)

        plt.tight_layout()

        #Plot output over time
        fig2, axes2 = plt.subplots(2, 2, figsize=(10, 10))

        axes2[0, 0].plot(ts[:-1], Us[:,0], color='blue')
        axes2[0, 0].plot(ts[:-1], ref_u[:,0], color='red')
        axes2[0, 0].set_title('Thrust')
        axes2[0, 0].set_xlabel('Time (s)')
        axes2[0, 0].set_ylabel('Force (N)')

        axes2[0, 1].plot(ts[:-1], Us[:,1], color='blue')
        axes2[0, 1].plot(ts[:-1], ref_u[:,1], color='red')
        axes2[0, 1].set_title('X Angular Velocity')
        axes2[0, 1].set_xlabel('Time (s)')
        axes2[0, 1].set_ylabel('Angular Velocity (rad/s)')

        axes2[1, 0].plot(ts[:-1], Us[:,2], color='blue')
        axes2[1, 0].plot(ts[:-1], ref_u[:,2], color='red')
        axes2[1, 0].set_title('Y Angular Velocity')
        axes2[1, 0].set_xlabel('Time (s)')
        axes2[1, 0].set_ylabel('Angular Velocity (rad/s)')

        axes2[1, 1].plot(ts[:-1], Us[:,3], color='blue')
        axes2[1, 1].plot(ts[:-1], ref_u[:,3], color='red')
        axes2[1, 1].set_title('Z Angular Velocity')
        axes2[1, 1].set_xlabel('Time (s)')
        axes2[1, 1].set_ylabel('Angular Velocity (rad/s)')

        fig2.suptitle('Quadrotor Input Over Time', fontsize=16)

        plt.tight_layout()

        # Plot Input Error Over Time
        U_errs = ref_u - Us

        fig2, axes2 = plt.subplots(2, 2, figsize=(10, 10))

        axes2[0, 0].plot(ts[:-1], U_errs[:,0])
        axes2[0, 0].set_title('Thrust')
        axes2[0, 0].set_xlabel('Time (s)')
        axes2[0, 0].set_ylabel('Force (N)')

        axes2[0, 1].plot(ts[:-1], U_errs[:,1])
        axes2[0, 1].set_title('X Angular Velocity')
        axes2[0, 1].set_xlabel('Time (s)')
        axes2[0, 1].set_ylabel('Angular Velocity (rad/s)')

        axes2[1, 0].plot(ts[:-1], U_errs[:,2])
        axes2[1, 0].set_title('Y Angular Velocity')
        axes2[1, 0].set_xlabel('Time (s)')
        axes2[1, 0].set_ylabel('Angular Velocity (rad/s)')

        axes2[1, 1].plot(ts[:-1], U_errs[:,3])
        axes2[1, 1].set_title('Z Angular Velocity')
        axes2[1, 1].set_xlabel('Time (s)')
        axes2[1, 1].set_ylabel('Angular Velocity (rad/s)')

        fig2.suptitle('Quadrotor Input Error Over Time', fontsize=16)

        plt.tight_layout()

        #Plot reference trajectory and trajectory calculated by mpc
        
        # Create a 3D plot
        fig3d = plt.figure()
        ax = fig3d.add_subplot(111, projection='3d')

        ref_traj_x = X_refs[:,0]
        ref_traj_y = X_refs[:,1]
        ref_traj_z = X_refs[:,2]

        real_traj_x = Xs[:,0]
        real_traj_y = Xs[:,1]
        real_traj_z = Xs[:,2]

        uref_traj_x = X_urefs[:,0]
        uref_traj_y = X_urefs[:,1]
        uref_traj_z = X_urefs[:,2]

        print(uref_traj_x)
        print(uref_traj_y)
        print(uref_traj_z)

        # Plot reference trajectory as a solid line
        #ax.plot(ref_traj_x, ref_traj_y, ref_traj_z, label='Reference Trajectory', color='blue')

        # Plot real trajectory as a solid line
        ax.plot(real_traj_x, real_traj_y, real_traj_z, label='Real Trajectory', color='red')

        # Plot real trajectory as a solid line
        ax.plot(uref_traj_x, uref_traj_y, uref_traj_z, label='Reference Input Trajectory', color='green')

        # Add labels
        ax.set_xlabel('X axis')
        ax.set_ylabel('Y axis')
        ax.set_zlabel('Z axis')

        # Add title
        ax.set_title('Reference vs Real Trajectory of Quadrotor using MPC')

        # Add legend
        ax.legend()

        # Display the plot
        plt.show()

        # plt.figure()
        # plt.plot(ts,Xs[:,0])

        # plt.figure()
        # plt.plot(ts,Xs[:,1])

        # plt.figure()
        # plt.plot(ts,Xs[:,2])

        # plt.figure()
        # plt.plot(ts[:-1],Us[:,0])

        # plt.show()
# %%
