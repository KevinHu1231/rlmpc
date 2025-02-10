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

def S(x):
    return sp.Matrix([[0,-x[2],x[1]], [x[2],0,-x[0]], [-x[1], x[0], 0]])

def R_GB(q):
    epsilon = q[:3]
    s = S(epsilon)
    eta = q[3]
    return (sp.eye(3) - 2*eta*s + 2*s@s).T

def qprod(q,p):
    epsilon_q = sp.Matrix(q[:3])
    eta_q = q[3]

    epsilon_p = sp.Matrix(p[:3])
    eta_p = p[3]

    epsilon_prod = eta_q*epsilon_p + eta_p*epsilon_q - epsilon_q.cross(epsilon_p)
    etp = epsilon_q.T@epsilon_p
    eta_prod = eta_q*eta_p - etp[0,0]

    return sp.Matrix([epsilon_prod, eta_prod])

def q_vec(v):
    return sp.Matrix([v[0],v[1],v[2],0])

def q_inv(v):
    return sp.Matrix([-v[0],-v[1],-v[2],v[3]])

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
        self.N = 10  # Prediction horizon time steps
        self.X_dim = 10
        self.U_dim = 4

        self.Q = np.diag(np.array([10,10,10,1,1,1,1,1,1,1])) # cost for states 
        self.R = np.diag(np.array([0.04,1,1,1])) # cost for input



        # Use P matrix from LQR
        # Procedure to find terminal state matrix:
        # Linearize quadrotor system
        # Find optimal controller gain K using LQR
        # Solve discrete Lyapunov equation for Q_bar

        self.f = self.f_syms()
        self.fx = self.feq_syms()
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
        q_dot = 0.5*qprod(q_vec(w),q)
        #q_dot = 0.5*quatProd(q)@w.row_insert(0, sp.Matrix([0]))
        f_dot = sp.Matrix.vstack(p_dot,v_dot,q_dot)
        self.f_dot = f_dot
        return f_dot
    
    def f_syms(self):
        X = sp.Matrix(sp.symbols("x,y,z,vx,vy,vz,qx,qy,qz,qw"))
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
        q_dot = 0.5*qprod(q_vec(w),q)
        #quaterion constraint
        quat = sp.Matrix([q[0]**2 + q[1]**2 + q[2]**2 + q[3]**2 - 1])
        f_dot = sp.Matrix.vstack(p_dot,v_dot,q_dot,quat,p_dot)
        self.f_eq = f_dot
        return f_dot
    
    def feq_syms(self):
        XU = sp.Matrix(sp.symbols("x,y,z,vx,vy,vz,qx,qy,qz,qw,F,wx,wy,wz"))
        self.XU = XU
        feq_s = sp.lambdify([XU],self.feq(XU), 'numpy')  
        return feq_s
    
    def linearize(self):
        Fx = lambda x: self.fx(x).flatten()
        x0_init = np.array([0,0,1,0,0,0,1,0,0,0,9.98,0,0,0])
        x0 = root(Fx,x0_init).x
        vals = {'x': x0[0], 'y':  x0[1], 'z':  x0[2],'vx':  x0[3], 'vy':  x0[4], 'vz':  x0[5],  'qx':  x0[6], 'qy':  x0[7], 'qz':  x0[8], 'qw':  x0[9], 'F':  x0[10], 'wx':  x0[11], 'wy':  x0[12], 'wz':  x0[13]}

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
        """
        Check if the (A, B) pair is controllable.
        """
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
            cost += ca.mtimes([(X[:, k] - x_ref).T, self.Q, X[:, k] - x_ref]) + ca.mtimes([U[:, k].T, self.R, U[:, k]])

        #cost += ca.mtimes([(X[:, self.N-1] - x_ref).T, self.Q_bar, X[:, self.N-1] - x_ref]) # cost for terminal state

    ##### System dynamics constraints

        # Non Integration
        # for k in range(self.N):
        #     opti.subject_to(X[:, k+1] == X[:, k] + self.t * self.f_sys(X[:, k], U[:, k]))

        # Integration
        for k in range(self.N):
            x_next = self.integrator(x0=X[:, k], p=U[:, k])['xf']
            opti.subject_to(X[:, k+1] == x_next)
    
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
        print("Optimal control input:", u_optimal)

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


    # def simulate(self,T):
    #     # Create a figure and a 3D axis
    #     fig = plt.figure()
    #     ax = fig.add_subplot(111, projection='3d')

    #     # Set labels for the axes
    #     ax.set_xlabel('X')
    #     ax.set_ylabel('Y')
    #     ax.set_zlabel('Z')

    #     # Set axis limits
    #     ax.set_xlim(-5, 5)
    #     ax.set_ylim(-5, 5)
    #     ax.set_zlim(0, 10)

    #     # Show the plot
    #     plt.show() 

# Example usage:
if __name__ == "__main__":
    
    test = 2
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
            #print(np.linalg.norm(xp - uav.X_r[i+1,:]))
            x0 = xp

    elif test == 2:
        # Test MPC
        uav = quaternion_uav()

        Ts = uav.t

        Xs = []
        Us = []
        ts = []

        x0 = np.array([0,0,1,0,0,0,1,0,0,0])
        x_ref = np.array([10,10,10,0,0,0,1,0,0,0])

        Xs.append(x0)
        ts.append(0)

        print("X_ref: ", x_ref)
        print("X_current: ", x0)
        print("X_error: ", x_ref - x0)
        t_steps = 600
        for i in tqdm(range(t_steps)):
            u0 = uav.mpc(x0,x_ref)
            Us.append(u0)
            print("U_current: ", u0)
            t0 = Ts*i
            tp = Ts*(i+1)
            ts.append(tp)

            xp = uav.update_state(x0,u0,t0,tp)
            x0 = xp

            Xs.append(x0)

            print("X_current: ", x0)
            print("X_error: ", x_ref - x0)

        Xs = np.vstack(Xs)
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

        axes2[0, 0].plot(ts[:-1], Us[:,0])
        axes2[0, 0].set_title('Thrust')
        axes2[0, 0].set_xlabel('Time (s)')
        axes2[0, 0].set_ylabel('Force (N)')

        axes2[0, 1].plot(ts[:-1], Us[:,1])
        axes2[0, 1].set_title('X Angular Velocity')
        axes2[0, 1].set_xlabel('Time (s)')
        axes2[0, 1].set_ylabel('Angular Velocity (rad/s)')


        axes2[1, 0].plot(ts[:-1], Us[:,2])
        axes2[1, 0].set_title('Y Angular Velocity')
        axes2[1, 0].set_xlabel('Time (s)')
        axes2[1, 0].set_ylabel('Angular Velocity (rad/s)')


        axes2[1, 1].plot(ts[:-1], Us[:,3])
        axes2[1, 1].set_title('Z Angular Velocity')
        axes2[1, 1].set_xlabel('Time (s)')
        axes2[1, 1].set_ylabel('Angular Velocity (rad/s)')

        fig2.suptitle('Quadrotor Input Over Time', fontsize=16)

        plt.tight_layout()
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