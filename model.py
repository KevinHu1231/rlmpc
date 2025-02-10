import numpy as np
import pandas as pd
import sympy as sp
import casadi as ca
from scipy.spatial.transform import Rotation as R
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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
        self.N = 20  # Prediction horizon time steps
        self.X_dim = 10
        self.U_dim = 4

        self.Q = np.diag(np.array([10,10,10,1,1,1,1,1,1,1])) # cost matrices 
        self.R = np.diag(np.array([1,1,1,1]))
        
        self.f = self.f_syms()
        self.f_sys = self.f_casadi()
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
    
    def f_casadi(self):
        X_cas = ca.SX.sym('X', self.X_dim)
        U_cas = ca.SX.sym('U', self.U_dim)
        XU = self.X.col_join(self.U)
        XU_cas = ca.vertcat(X_cas,U_cas)
        f_casadi = sympy2casadi(self.f_dot, XU, XU_cas)
        f_func = ca.Function('f', [X_cas, U_cas], [f_casadi])
        return f_func
    
    def mpc(self, x0, x_ref):
        # Parameters
        #x_ref = np.array([1, 0])

        # Initial state
        #x0 = np.array([0, 0])

        # Optimization variables
        opti = ca.Opti()
        X = opti.variable(self.X_dim, self.N+1)
        U = opti.variable(self.U_dim, self.N)

        # Cost function
        cost = 0
        for k in range(self.N):
            cost += ca.mtimes([(X[:, k] - x_ref).T, self.Q, X[:, k] - x_ref]) + ca.mtimes([U[:, k].T, self.R, U[:, k]])

        # System dynamics constraints
        for k in range(self.N):
            opti.subject_to(X[:, k+1] == X[:, k] + self.t * self.f_sys(X[:, k], U[:, k]))

        # Input constraints
        opti.subject_to(opti.bounded(0, U[0,:], 1000))
        opti.subject_to(opti.bounded(-10, U[1,:], 10))
        opti.subject_to(opti.bounded(-10, U[2,:], 10))
        opti.subject_to(opti.bounded(-3, U[3,:], 3))


        opti.subject_to(opti.bounded(0.5, X[2,:], 15))

        # Initial condition constraint
        opti.subject_to(X[:, 0] == x0)

        # Solve the optimization problem
        opti.minimize(cost)
        opts = {
                'print_time': False,   # Disable printing time statistics
                'ipopt': {
                    'print_level': 0,  # Disable IPOPT solver output
                    'sb': 'yes',       # Disable solver banner
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
        idx_x = ['p_x','p_y','p_z','v_x','v_y','v_z','q_x','q_y','q_z','q_w']
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
    # Creating an instance of MyClass

    # Test model
    # uav = quaternion_uav()

    # Ts = uav.t

    # cost = 0
    # x0 = uav.X_r[0,:]

    # for i in range(400):
    #     u0 = uav.U_r[i,:]
    #     t0 = Ts*i
    #     tp = Ts*(i+1)
    #     xp = uav.update_state(x0,u0,t0,tp)
    #     cost = cost + np.linalg.norm(xp - uav.X_r[i+1,:])
    #     #print(np.linalg.norm(xp - uav.X_r[i+1,:]))
    #     x0 = xp

    # Test MPC
    uav = quaternion_uav()

    Ts = uav.t

    xs = []
    ys = []
    zs = []
    ts = []

    x0 = np.array([0,0,1,0,0,0,0,0,0,1])
    x_ref = np.array([10,10,10,0,0,0,0,0,0,1])

    xs.append(x0[0])
    ys.append(x0[1])
    zs.append(x0[2])
    ts.append(0)

    print("X_ref: ", x_ref)
    print("X_current: ", x0)
    print("X_error: ", x_ref - x0)
    for i in range(400):
        u0 = uav.mpc(x0,x_ref)
        print("U_current: ", u0)
        t0 = Ts*i
        tp = Ts*(i+1)
        ts.append(tp)

        xp = uav.update_state(x0,u0,t0,tp)
        x0 = xp

        xs.append(x0[0])
        ys.append(x0[1])
        zs.append(x0[2])

        print("X_current: ", x0)
        print("X_error: ", x_ref - x0)

    plt.figure()
    plt.plot(ts,xs)

    plt.figure()
    plt.plot(ts,ys)

    plt.figure()
    plt.plot(ts,zs)
    plt.show()