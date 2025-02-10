import numpy as np
import pandas as pd
import sympy as sp
from scipy.integrate import solve_ivp

class quadSysSym:
    def __init__(self,T):
        self.ts = T
        self.g = 9.8066

    def fdot(self,x,u):
        v = x[3:6,0]
        q = x[6::,0]
        omg = u[1::,0]
    
        p_dot = v
        g = sp.Matrix([[0],[0],[-self.g]])
        c = sp.Matrix([[0],[0],[u[0,0]]])
        v_dot = g + self.Rquat(q)@c
        #print("v_dot", v_dot)
        q_dot = 0.5*self.quatProd(q)@omg.row_insert(0, sp.Matrix([0]))

        #print("q_dot: ", q_dot)
        
        fd = sp.Matrix.vstack(p_dot,v_dot,q_dot)
        #print("fd: ", fd)
        return fd

    def getf_func(self):
        X = sp.Matrix(sp.symbols("x,y,z,vx,vy,vz,qw,qx,qy,qz"))
        U = sp.Matrix(sp.symbols("c,wx,wy,wz"))

        f_func = sp.lambdify([X,U],self.fdot(X,U), 'numpy')
        return f_func
    

    @staticmethod   
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

    @staticmethod   
    def Rquat(q):
        qw = q[0]
        qx = q[1]
        qy = q[2]
        qz = q[3]

        Q = sp.Matrix([
            [1-2*(qy**2+qz**2),2*(qx*qy+qw*qz),2*(qx*qz-qw*qy)],
            [2*(qx*qy-qw*qz),1-2*(qx**2+qz**2),2*(qy*qz+qw*qx)],
            [2*(qx*qz+qw*qy), 2*(qy*qz-qw*qx),1-2*(qx**2+qy**2)]
        ])
        return Q.T  
    
    
class quadSys:

    def __init__(self,T):
        self.ts = T
        self.drone = quadSysSym(self.ts)
        self.loadRef()
        self.loadModel()
    
    def loadModel(self):
        self.f = self.drone.getf_func()

    def updateState(self,x0, u0, t_start,t_end):
        F = lambda t,x: quadrotor.f(x,u0).flatten()
        t_eval = np.linspace(t_start, t_end, 10)
        sol = solve_ivp(F,[t_start,t_end],x0,t_eval=t_eval)
        xp = sol.y[:,-1]
        return xp    

    def loadRef(self):
        self.drone.g = 9.8066
        file = "race_0323_slits.csv"
        df = pd.read_csv(file,index_col=None,header=0)
        idx_x = ['p_x','p_y','p_z','v_x','v_y','v_z','q_w','q_x','q_y','q_z']
        idx_u = ['thrust','w_x','w_y','w_z']
        U_r = df[idx_u]
        U_r.loc[:,'thrust'] = U_r['thrust']
        self.X_r = df[idx_x].to_numpy()
        self.U_r = U_r.to_numpy()

drone = quadSysSym(0.01)

Ts = 0.01
quadrotor = quadSys(Ts)
cost = 0
x0 = quadrotor.X_r[0,:]

for i in range(400):
    u0 = quadrotor.U_r[i,:]
    t0 = Ts*i
    tp = Ts*(i+1)
    xp = quadrotor.updateState(x0,u0,t0,tp)
     
    cost = cost + np.linalg.norm(xp - quadrotor.X_r[i+1,:])
    print(np.linalg.norm(xp - quadrotor.X_r[i+1,:]))
    x0 = xp
