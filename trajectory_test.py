#%%
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as Rot

class trajecotry:
    ''' The reference trajectory for ASV model
        6D: The reference trajectory in horizontal plan
    '''
    def __init__(self):
        self.t = sp.symbols('t')
        self.a = sp.symbols('a')
        self.TrajEight = self.genTrajEight()
        self.TrajLine = self.genTrajLine()
        self.TrajCircle = self.genTrajCircle()
        self.TrajWave = self.genTrajWave()
        self.TrajEllip = self.genTrajEllip()
        self.Trajs = [self.TrajEight,self.TrajLine,self.TrajCircle,self.TrajWave,self.TrajEllip]

    def lodDiff(self,XRef,YRef):
         # X ref
        # self.XRef = a*self.t
        
        XRefdot = XRef.diff(self.t)
        XRefddot = XRefdot.diff(self.t)        
        
        # Y ref
        YRefdot = YRef.diff(self.t)
        YRefddot = YRefdot.diff(self.t)

        # Z ref
        ZRef = 0
        ZRefdot = 0
        Zrefddot = 0

        # Psi ref
        psiRef = sp.atan2(YRefdot,XRefdot)
        psiRefdot = psiRef.diff(self.t)

        R = sp.Matrix([
            [sp.cos(psiRef),-sp.sin(psiRef),0],[sp.sin(psiRef), sp.cos(psiRef),0],[0,0,1]                     
            ])
        eta = sp.Matrix([[XRefdot],[YRefdot],[0]])

        eta_local = R.T*eta
        uRef = eta_local[0,0]
        vRef = eta_local[1,0]
        rRef = psiRefdot

        uRefdot = uRef.diff(self.t)
        vRefdot = vRef.diff(self.t)
        rRefdot = rRef.diff(self.t)

        TrajDyna = {
            'XRef':XRef,
            'XRefdot':XRefdot,
            'XRefddot': XRefddot,
            'YRef': YRef,
            'YRefdot': YRefdot,
            'YRefddot':YRefddot,
            'psiRef': psiRef,
            'psiRefdot':psiRefdot,
            'uRef': uRef,
            'uRefdot':uRefdot,
            'vRef': vRef,
            'vRefdot':vRefdot,
            'rRef': rRef,
            'rRefdot':rRefdot,
        }
        return TrajDyna

    def loadFunc(self,TrajDyna):

        TrajFunc = {}
        for ref in TrajDyna:
            dyna = TrajDyna[ref]
            func = sp.lambdify([self.a,self.t],dyna, 'numpy')
            TrajFunc[ref+'_func'] = func
        
        return TrajFunc

    @staticmethod
    def getTraj(TrajFunc,a,t):

        # reference signals in global frame
        xr = TrajFunc['XRef_func'](a,t)
        yr = TrajFunc['YRef_func'](a,t)
        psir = TrajFunc['psiRef_func'](a,t)

        xrd = TrajFunc['XRefdot_func'](a,t)
        yrd = TrajFunc['YRefdot_func'](a,t)
        psird = TrajFunc['psiRefdot_func'](a,t)

        X = np.array([xr, yr, psir]).astype(np.float64).reshape(-1,1)
        Xdot = np.array([xrd, yrd, psird]).astype(np.float64).reshape(-1,1)

        # reference signals in local frame

        ur = TrajFunc['uRef_func'](a,t)
        vr = TrajFunc['vRef_func'](a,t)
        rr = TrajFunc['rRef_func'](a,t)

        urd = TrajFunc['uRefdot_func'](a,t)
        vrd = TrajFunc['vRefdot_func'](a,t)
        rrd = TrajFunc['rRefdot_func'](a,t)

        V = np.array([ur, vr, rr]).astype(np.float64).reshape(-1,1)
        Vdot = np.array([urd, vrd, rrd]).astype(np.float64).reshape(-1,1)
        return X, Xdot, V, Vdot

    def genTrajEight(self):
        XRef = 5*sp.sin(0.5*self.a*self.t/(sp.pi/2))
        YRef = 2*sp.sin(self.a*self.t/(sp.pi/2))
        TrajDyna = self.lodDiff(XRef,YRef)
        TrajFunc = self.loadFunc(TrajDyna)
        return TrajFunc

    def genTrajWave(self):
        XRef = self.a*self.t
        YRef = sp.sin(self.a*self.t/(sp.pi/2))
        TrajDyna = self.lodDiff(XRef,YRef)
        TrajFunc = self.loadFunc(TrajDyna)
        return TrajFunc

    def genTrajCircle(self):
        XRef = sp.cos(self.a*self.t/(sp.pi/2))
        YRef = sp.sin(self.a*self.t/(sp.pi/2))
        TrajDyna = self.lodDiff(XRef,YRef)
        TrajFunc = self.loadFunc(TrajDyna)
        return TrajFunc

    def genTrajEllip(self):
        XRef = 0.8*sp.cos(self.a*self.t)
        YRef = 1.2*sp.sin(self.a*self.t)
        TrajDyna = self.lodDiff(XRef,YRef)
        TrajFunc = self.loadFunc(TrajDyna)
        return TrajFunc

    def genTrajLine(self):
        XRef = self.a*self.t+0.5
        YRef = 0.8*self.a*self.t+0.8
        TrajDyna = self.lodDiff(XRef,YRef)
        TrajFunc = self.loadFunc(TrajDyna)
        return TrajFunc

traj = trajecotry()
eight = traj.Trajs[0]
x_ref = eight['XRef_func']
y_ref = eight['YRef_func']
psi_ref = eight['psiRef_func']
times = np.arange(0,50.01,0.01)
a_s = np.ones_like(times)

x_s = x_ref(a_s,times) 
y_s = y_ref(a_s,times)
psi_s = psi_ref(a_s,times)

angle = sp.symbols('psi')
R = sp.Matrix([
            [sp.cos(angle),-sp.sin(angle),0],[sp.sin(angle), sp.cos(angle),0],[0,0,1]                     
            ])
R_func = sp.lambdify([angle], R, modules='numpy')
Rots = np.array([R_func(psi) for psi in psi_s])
rs = Rot.from_matrix(Rots)
qs = rs.as_quat()

# plt.plot(x_s,y_s)
# plt.show()