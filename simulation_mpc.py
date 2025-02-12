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

        self.Q = np.diag(np.array([8,8,10,1,1,1,1,1,1,1])) # cost for states 
        self.R = np.diag(np.array([0.003,1,1,1])) # cost for input


        self.Q_bar = np.diag(np.array([8,8,10,1,1,1,1,1,1,1])) #comment


        # Use P matrix from LQR
        # Procedure to find terminal state matrix:
        # Linearize quadrotor system
        # Find optimal controller gain K using LQR
        # Solve discrete Lyapunov equation for Q_bar

        self.f = self.f_syms()
        #self.fx = self.feq_syms()
        #self.A, self.B = self.linearize()

        #self.is_controllable(self.A,self.B)

        #self.K = self.LQR()
        #self.Q_bar = self.solve_lyapunov() #cost for terminal state #LQR

        self.f_sys = self.f_casadi()
        self.create_casadi_integrator()
        self.load_test()
    
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
    
    def LQR(self):
        P = solve_continuous_are(self.A, self.B, self.Q, self.R)
        print("P:", P)
        K = np.linalg.inv(self.R) @ self.B.T @ P
        print("K:", K)
        return K
    
    def solve_lyapunov(self):
        # Define the matrices A and Q
        A = self.A + self.B@self.K
        Q = self.Q + self.K.T@self.R@self.K
        # Solve the discrete-time Lyapunov equation AXA^T - X + Q = 0
        X = solve_discrete_lyapunov(A.T, Q)
        print("X:", X)
        return X

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

    def add_system_dynamics(self):
        for k in range(self.N):
            x_k = self.X[:, k]
            u_k = self.U[:, k]
            x_next = self.integrator(x0=x_k, p=u_k)['xf']
            self.opti.subject_to(self.X[:, k+1] == x_next)
    
    def mpc(self, x0, x_ref):
        # Optimization variables
        opti = ca.Opti()
        X = opti.variable(self.X_dim, self.N+1)
        U = opti.variable(self.U_dim, self.N)

        # Cost function
        cost = 0
        for k in range(self.N):
            cost += ca.mtimes([(X[:, k] - x_ref[k,:]).T, self.Q, X[:, k] - x_ref[k,:]]) + ca.mtimes([U[:, k].T, self.R, U[:, k]])

        cost += ca.mtimes([(X[:, self.N-1] - x_ref[k,:]).T, self.Q_bar, X[:, self.N-1] - x_ref[k,:]]) # cost for terminal state

    ##### System dynamics constraints

        # Non Integration
        for k in range(self.N):
            opti.subject_to(X[:, k+1] == X[:, k] + self.t * self.f_sys(X[:, k], U[:, k]))

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
    
    def update_state(self,x0,u0,t_start,t_end):
        F = lambda t,x: self.f(x,u0).flatten()
        t_eval = np.linspace(t_start, t_end, self.N_update)
        sol = solve_ivp(F,[t_start,t_end],x0,t_eval=t_eval)
        x_vals = sol.y[:,-1]
        return x_vals
    
    def load_test(self):
        file = "race_0323_slits.csv"
        df = pd.read_csv(file,index_col=None,header=0)
        idx_x = ['p_x','p_y','p_z','v_x','v_y','v_z','q_w','q_x','q_y','q_z']
        idx_u = ['thrust','w_x','w_y','w_z']
        U_r = df[idx_u]
        U_r.loc[:,'thrust'] = U_r['thrust']
        self.X_r = df[idx_x].to_numpy()
        self.U_r = U_r.to_numpy()

# Example usage:
if __name__ == "__main__":
    
    test = 2 # Test 1 to test quaternion model, Test 2 to test MPC
    traj_type = 0 # Trajectory number to test

    if test == 1:
        # Test model

        uav = quaternion_uav()

        Ts = uav.t

        cost = 0
        x0 = uav.X_r[0,:]

        for i in range(400):
            u0 = uav.U_r[i,:]
            t0 = Ts*i
            tp = Ts*(i+1)
            xp = uav.update_state(x0,u0,t0,tp)
            
            cost = cost + np.linalg.norm(xp - uav.X_r[i+1,:])
            print(np.linalg.norm(xp - uav.X_r[i+1,:]))
            x0 = xp

    elif test == 2:
        # Test MPC
        uav = quaternion_uav()

        Ts = uav.t

        Xs = []
        X_refs = []
        Us = []
        ts = []

        trajectory = traj.trajecotry()
        tra = trajectory.Trajs[traj_type]
        
        x_ref = tra['XRef_func']
        y_ref = tra['YRef_func']
        psi_ref = tra['psiRef_func']

        x_ref_d = tra['XRefdot_func']
        y_ref_d = tra['YRefdot_func']
        psi_ref_d = tra['psiRefdot_func']

        times = np.arange(0,10.01 + uav.N*0.01,0.01)
        a_s = np.ones_like(times)

        x_s = x_ref(a_s,times)
        y_s = y_ref(a_s,times)
        psi_s = psi_ref(a_s,times)
        print(x_s)

        dx_s = x_ref_d(a_s,times) 
        dy_s = y_ref_d(a_s,times)

        angle = sp.symbols('psi')
        rot = sp.Matrix([
                    [sp.cos(angle),-sp.sin(angle),0],[sp.sin(angle), sp.cos(angle),0],[0,0,1]                     
                    ])
        R_func = sp.lambdify([angle], rot, modules='numpy')
        rots = np.array([R_func(psi) for psi in psi_s])
        rs = R.from_matrix(rots)
        qs = rs.as_quat()
        qs = np.hstack((qs[:, -1:],qs[:, :-1]))
        #####

        hover_x0 = np.array([0,0,1,0,0,0,1,0,0,0])
        hover_x_r = np.array([0,0,1,0,0,0,1,0,0,0])

        hover_x_r_th = np.tile(hover_x_r,(uav.N,1))

        hover_time = 100
        hover_steps = hover_time + uav.N
        
        t_traj = 0
        for i in tqdm(range(hover_steps)):

            u0 = uav.mpc(hover_x0,hover_x_r_th)

            #print("U_current: ", u0)

            t0 = Ts*i
            tp = Ts*(i+1)

            xp = uav.update_state(hover_x0,u0,t0,tp)
            hover_x0 = xp

            hover_steps_left = hover_steps-i

            if hover_steps_left >= uav.N:
                hover_x_r_th = np.tile(hover_x_r,(uav.N,1))
            else:
                x_r_th_hover = np.tile(hover_x_r,(hover_steps_left,1))

                traj_steps = uav.N-hover_steps_left
                x_r_th_traj = np.vstack((x_s[:traj_steps],y_s[:traj_steps],1*np.ones((traj_steps,)),dx_s[:traj_steps],dy_s[:traj_steps],np.zeros((traj_steps,)))).T
                x_r_th_traj = np.hstack((x_r_th_traj,qs[:traj_steps,:]))
                hover_x_r_th = np.vstack((x_r_th_hover,x_r_th_traj))

                Xs.append(hover_x0)
                X_refs.append(hover_x_r)
                ts.append(t_traj)
                Us.append(u0)

                t_traj += Ts

            #print("X_current: ", x0)
            #print("X_error: ", x_ref - x0)


        x0 = hover_x0
        x_r_th = hover_x_r_th
        
        #x_r = np.concatenate((np.array([x_s[0],y_s[0],1,dx_s[0],dy_s[0],0]),qs[0,:]))
        # x_r_th = np.vstack((x_s[:uav.N],y_s[:uav.N],1*np.ones((uav.N,)),dx_s[:uav.N],dy_s[:uav.N],np.zeros((uav.N,)))).T
        # x_r_th = np.hstack((x_r_th,qs[:uav.N,:]))

        # Xs.append(x0)
        # X_refs.append(x_r)
        # ts.append(0)

        # print("X_ref: ", x_ref)
        # print("X_current: ", x0)
        # print("X_error: ", x_ref - x0)

        t_steps = 1000
        for i in tqdm(range(t_steps)):

            u0 = uav.mpc(x0,x_r_th)

            Us.append(u0)
            #print("U_current: ", u0)

            t0 = Ts*i + hover_steps*Ts
            tp = Ts*(i+1) + hover_steps*Ts
            ts.append(t_traj)

            xp = uav.update_state(x0,u0,t0,tp)
            x0 = xp
            x_r = np.concatenate((np.array([x_s[i+1],y_s[i+1],1,dx_s[i+1],dy_s[i+1],0]),qs[i+1,:]))

            x_r_th = np.vstack((x_s[i+1:i+1+uav.N],y_s[i+1:i+1+uav.N],1*np.ones((uav.N,)),dx_s[i+1:i+1+uav.N],dy_s[i+1:i+1+uav.N],np.zeros((uav.N,)))).T
            x_r_th = np.hstack((x_r_th,qs[i+1:i+1+uav.N,:]))

            Xs.append(x0)
            X_refs.append(x_r)
            t_traj += Ts


            #print("X_current: ", x0)
            #print("X_error: ", x_ref - x0)

        Xs = np.vstack(Xs)
        X_refs = np.vstack(X_refs)
        Us = np.vstack(Us)
        ts = np.array(ts)

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        axes[0, 0].plot(ts, Xs[:,0])
        axes[0, 0].set_title('X position')
        axes[0, 0].set_xlabel('Time (s)')
        axes[0, 0].set_ylabel('Distance (m)')


        axes[0, 1].plot(ts, Xs[:,1])
        axes[0, 1].set_title('Y Position')
        axes[0, 1].set_xlabel('Time (s)')
        axes[0, 1].set_ylabel('Distance (m)')


        axes[0, 2].plot(ts, Xs[:,2])
        axes[0, 2].set_title('Z Position')
        axes[0, 2].set_xlabel('Time (s)')
        axes[0, 2].set_ylabel('Distance (m)')


        axes[1, 0].plot(ts, Xs[:,3])
        axes[1, 0].set_title('X Velocity')
        axes[1, 0].set_xlabel('Time (s)')
        axes[1, 0].set_ylabel('Velocity (m/s)')


        axes[1, 1].plot(ts, Xs[:,4])
        axes[1, 1].set_title('Y Velocity')
        axes[1, 1].set_xlabel('Time (s)')
        axes[1, 1].set_ylabel('Velocity (m/s)')


        axes[1, 2].plot(ts, Xs[:,5])
        axes[1, 2].set_title('Z Velocity')
        axes[1, 2].set_xlabel('Time (s)')
        axes[1, 2].set_ylabel('Velocity (m/s)')

        fig.suptitle('Quadrotor State Over Time', fontsize=16)

        plt.tight_layout()

        fig2, axes2 = plt.subplots(2, 2, figsize=(10, 10))

        axes2[0, 0].plot(ts, Us[:,0])
        axes2[0, 0].set_title('Thrust')
        axes2[0, 0].set_xlabel('Time (s)')
        axes2[0, 0].set_ylabel('Force (N)')

        axes2[0, 1].plot(ts, Us[:,1])
        axes2[0, 1].set_title('X Angular Velocity')
        axes2[0, 1].set_xlabel('Time (s)')
        axes2[0, 1].set_ylabel('Angular Velocity (rad/s)')


        axes2[1, 0].plot(ts, Us[:,2])
        axes2[1, 0].set_title('Y Angular Velocity')
        axes2[1, 0].set_xlabel('Time (s)')
        axes2[1, 0].set_ylabel('Angular Velocity (rad/s)')


        axes2[1, 1].plot(ts, Us[:,3])
        axes2[1, 1].set_title('Z Angular Velocity')
        axes2[1, 1].set_xlabel('Time (s)')
        axes2[1, 1].set_ylabel('Angular Velocity (rad/s)')

        fig2.suptitle('Quadrotor Input Over Time', fontsize=16)

        plt.tight_layout()

        # Create a 3D plot
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

        # plt.figure()
        # plt.plot(ts,Xs[:,0])

        # plt.figure()
        # plt.plot(ts,Xs[:,1])

        # plt.figure()
        # plt.plot(ts,Xs[:,2])

        # plt.figure()
        # plt.plot(ts[:-1],Us[:,0])

        # plt.show()