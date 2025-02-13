#%%
import numpy as np
import sympy as sp

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

# Class for creating trajectory function
class trajecotry:
    ''' The reference trajectory for ASV model
        6D: The reference trajectory in horizontal plan
    '''
    def __init__(self):
        # variables for generating trajectory
        self.t = sp.symbols('t')
        self.a = sp.symbols('a')
        self.t_len = sp.symbols('l')

        # generating example trajectories
        self.TrajEight, _ , _ = self.genTrajEight()
        self.TrajLine, _ , _ = self.genTrajLine()
        self.TrajCircle, _ , _ = self.genTrajCircle()
        self.TrajWave, _ , _ = self.genTrajWave()
        self.TrajEllip, _ , _ = self.genTrajEllip()
        self.Trajs = [self.TrajEight,self.TrajLine,self.TrajCircle,self.TrajWave,self.TrajEllip]

    def lodDiff(self,XRef,YRef):

        # Calculate Trajectory Derivatives  

        # X ref      
        XRefdot = XRef.diff(self.t)
        XRefddot = XRefdot.diff(self.t)        
        
        # Y ref
        YRefdot = YRef.diff(self.t)
        YRefddot = YRefdot.diff(self.t)

        # Z ref
        ZRef = 1
        ZRefdot = 0
        Zrefddot = 0

        # Calculate trajectory angle psi

        # Psi ref
        psiRef = sp.atan2(YRefdot,XRefdot)
        self.psi_exp = psiRef
        psiRefdot = psiRef.diff(self.t)

        # Calculating rotation matrix
        R = sp.Matrix([
            [sp.cos(psiRef),-sp.sin(psiRef),0],[sp.sin(psiRef), sp.cos(psiRef),0],[0,0,1]                     
            ])
        eta = sp.Matrix([[XRefdot],[YRefdot],[0]])

        # Calculate relative positions
        eta_local = R.T*eta
        uRef = eta_local[0,0]
        vRef = eta_local[1,0]
        rRef = psiRefdot

        uRefdot = uRef.diff(self.t)
        vRefdot = vRef.diff(self.t)
        rRefdot = rRef.diff(self.t)

        # Trajectory data
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
        # Define your custom Heaviside function
        def heaviside(x, zero_value=0):
            return np.where(x > 0, 1, 0)
        # Custom DiracDelta function for numpy

        def dirac_delta(x, *args):
            # Define a small threshold for delta approximation
            epsilon = 1e-5
            return np.where(np.abs(x) < epsilon, 1/epsilon, 0)

        TrajFunc = {}

        # Create trajectory function
        for ref in TrajDyna:
            dyna = TrajDyna[ref]
            func = sp.lambdify((self.a,self.t,self.t_len), dyna, {'Heaviside': heaviside, 'DiracDelta': dirac_delta, 'numpy': np})
            func = np.vectorize(func)
            
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

    # Trajectory generating functions not used 

    def genTrajEight(self):
        # XRef = sp.Heaviside(self.t-self.hover_t)*3*sp.sin(4*sp.pi*self.a*self.t/self.t_len)
        # YRef = sp.Heaviside(self.t-self.hover_t)*3*sp.sin(2*sp.pi*self.a*self.t/self.t_len)

        # Create figure eight trajectory
        XRef = 3*sp.sin(4*sp.pi*self.a*(self.t)/self.t_len)
        YRef = 3*sp.sin(2*sp.pi*self.a*(self.t)/self.t_len)
        TrajDyna = self.lodDiff(XRef,YRef)
        TrajFunc = self.loadFunc(TrajDyna)
        return TrajFunc, XRef, YRef

    def genTrajWave(self):

        # Create wave trajectory
        XRef = self.a*self.t
        YRef = 5*sp.sin(2*sp.pi*self.a*self.t/self.t_len)
        TrajDyna = self.lodDiff(XRef,YRef)
        TrajFunc = self.loadFunc(TrajDyna)
        return TrajFunc, XRef, YRef

    def genTrajCircle(self):

        # Create circle trajectory
        XRef = sp.cos(2*sp.pi*self.a*self.t/self.t_len)
        YRef = sp.sin(2*sp.pi*self.a*self.t/self.t_len)
        TrajDyna = self.lodDiff(XRef,YRef)
        TrajFunc = self.loadFunc(TrajDyna)
        return TrajFunc, XRef, YRef

    def genTrajEllip(self):

        # Create ellipse trajectory
        XRef = 10 + 4*sp.cos(2*sp.pi*self.a*self.t/self.t_len)
        YRef = 8 + 6*sp.sin(2*sp.pi*self.a*self.t/self.t_len)
        TrajDyna = self.lodDiff(XRef,YRef)
        TrajFunc = self.loadFunc(TrajDyna)
        return TrajFunc, XRef, YRef

    def genTrajLine(self):

        # Create line trajectory
        XRef = self.a*self.t
        YRef = 0.8*self.a*self.t
        TrajDyna = self.lodDiff(XRef,YRef)
        TrajFunc = self.loadFunc(TrajDyna)
        return TrajFunc, XRef, YRef
    
    # Function combining two different trajectories
    def genTrajCombined(self,d_traj):

        # Unpack trajectory parameters
        start1,end1,stime1,etime1,traj1,traj1_params = d_traj.traj1.data
        start2,end2,stime2,etime2,traj2,traj2_params = d_traj.traj2.data

        # Calculate first trajectory start time
        traj_time1 = self.t-stime1

        # Function to turn on and off first trajectory
        offon1 = sp.Heaviside(traj_time1)*sp.Heaviside(etime1-self.t)
        
        # Calculate second trajectory start time
        traj_time2 = self.t-stime2

        # Function to turn on and off second trajectory
        offon2 = sp.Heaviside(traj_time2)*sp.Heaviside(etime2-self.t)

        if traj1 == 0: # Eight

            # First trajectory eight
            XRef1 = offon1*(start1[0] + traj1_params['Amplitude']*sp.sin(4*sp.pi*self.a*traj_time1/self.t_len))
            YRef1 = offon1*(start1[1] + traj1_params['Amplitude']*sp.sin(2*sp.pi*self.a*traj_time1/self.t_len))
        
        if traj2 == 0: # Eight

            # Second trajectory eight
            XRef2 = offon2*(start2[0] + traj2_params['Amplitude']*sp.sin(4*sp.pi*self.a*traj_time2/self.t_len))
            YRef2 = offon2*(start2[1] + traj2_params['Amplitude']*sp.sin(2*sp.pi*self.a*traj_time2/self.t_len))

        if traj1 == 1: # Line

            # First trajectory line
            traj_distx = end1[0] - start1[0]
            traj_disty = end1[1] - start1[1]
            XRef1 = offon1*(start1[0] + (traj_distx/self.t_len)*self.a*(traj_time1))
            YRef1 = offon1*(start1[1] + (traj_disty/self.t_len)*self.a*(traj_time1))

        if traj2 == 1: # Line

            # Second trajectory line
            traj_distx = end2[0] - start2[0]
            traj_disty = end2[1] - start2[1]
            XRef2 = offon2*(start2[0] + (traj_distx/self.t_len)*self.a*(traj_time2))
            YRef2 = offon2*(start2[1] + (traj_disty/self.t_len)*self.a*(traj_time2))

        if traj1 == 2: # Circle

            # First trajectory circle
            XRef1 = offon1*(start1[0] - traj1_params['Amplitude'] + traj1_params['Amplitude']*sp.cos(2*sp.pi*self.a*traj_time1/self.t_len))
            YRef1 = offon1*(start1[1] + traj1_params['Amplitude']*sp.sin(2*sp.pi*self.a*traj_time1/self.t_len))
        
        if traj2 == 2: # Circle

            # Second trajectory circle
            XRef2 = offon2*(start2[0] - traj2_params['Amplitude'] + traj2_params['Amplitude']*sp.cos(2*sp.pi*self.a*traj_time2/self.t_len))
            YRef2 = offon2*(start2[1] + traj2_params['Amplitude']*sp.sin(2*sp.pi*self.a*traj_time2/self.t_len))

        if traj1 == 3: # Wave

            # First trajectory wave
            wave_side = traj1_params['Wave Side']
            traj_dist = end1 - start1
            if wave_side == 'X':
                XRef1 = offon1*(start1[0] + traj1_params['Amplitude']*sp.sin(2*sp.pi*self.a*traj_time1/self.t_len))
                YRef1 = offon1*(start1[1] + (traj_dist[1]/self.t_len)*self.a*(traj_time1))
            if wave_side == 'Y':
                XRef1 = offon1*(start1[0] + (traj_dist[0]/self.t_len)*self.a*(traj_time1))
                YRef1 = offon1*(start1[1] + traj1_params['Amplitude']*sp.sin(2*sp.pi*self.a*traj_time1/self.t_len))

        if traj2 == 3: # Wave

            # Second trajectory wave
            wave_side = traj2_params['Wave Side']
            traj_dist = end2 - start2
            if wave_side == 'X':
                XRef2 = offon2*(start2[0] + traj2_params['Amplitude']*sp.sin(2*sp.pi*self.a*traj_time2/self.t_len))
                YRef2 = offon2*(start2[1] + (traj_dist[1]/self.t_len)*self.a*(traj_time2))
            if wave_side == 'Y':
                XRef2 = offon2*(start2[0] + (traj_dist[0]/self.t_len)*self.a*(traj_time2))
                YRef2 = offon2*(start2[1] + traj2_params['Amplitude']*sp.sin(2*sp.pi*self.a*traj_time2/self.t_len))

        if traj1 == 4: # Ellipse

            # First trajectory ellipse
            XRef1 = offon1*(start1[0] - traj1_params['Amplitude'] + traj1_params['Amplitude']*sp.cos(2*sp.pi*self.a*traj_time1/self.t_len))
            YRef1 = offon1*(start1[1] + traj1_params['Amplitude']*sp.sin(2*sp.pi*self.a*traj_time1/self.t_len))
        
        if traj2 == 4: # Ellipse

            # Second trajectory ellipse
            XRef2 = offon2*(start2[0] - traj2_params['Amplitude'] + traj2_params['Amplitude']*sp.cos(2*sp.pi*self.a*traj_time2/self.t_len))
            YRef2 = offon2*(start2[1] + traj2_params['Amplitude']*sp.sin(2*sp.pi*self.a*traj_time2/self.t_len))

        if traj1 == 5: # Hover

            # First trajectory hover in place
            XRef1 = offon1*start1[0]
            YRef1 = offon1*start1[1]
        
        if traj2 == 5: # Hover

            # Second trajectory hover in place
            XRef2 = offon2*start2[0]
            YRef2 = offon2*start2[1]

        # Combine first and second trajectories
        XRef = XRef1 + XRef2
        YRef = YRef1 + YRef2

        # Calculate trajectory derivatives
        TrajDyna = self.lodDiff(XRef,YRef)

        # Create trajectory functions
        TrajFunc = self.loadFunc(TrajDyna)

        # Collect trajectory and reference data
        data = [TrajFunc, XRef, YRef, XRef1, XRef2, YRef1, YRef2]

        # Pack data into double trajectory class
        double_trajectory = double_traj(data)

        return double_trajectory
    
    # Connect two different trajectories (not used)
    def connectTraj(self,xref1,yref1,xref2,yref2):
        
        XRef = xref1 + xref2
        YRef = yref1 + yref2

        TrajDyna = self.lodDiff(XRef,YRef)
        TrajFunc = self.loadFunc(TrajDyna)

        return TrajFunc, XRef, YRef

# Single trajectory data class
class traj_data:
    def __init__(self,data):
        self.start = data[0]
        self.end = data[1]
        self.start_time = data[2]
        self.end_time = data[3]
        self.traj_type = data[4]
        self.traj_params = data[5]
        self.data = data

# Two connected trajectory data class includes two single trajectory data class objects
class double_traj_data:
    def __init__(self,traj1,traj2):
        self.traj1 = traj1
        self.traj2 = traj2

# Two connected trajectory data class including all trajectory functions and their derivatives
class double_traj:
    def __init__(self,data):
        self.traj_function = data[0]
        self.traj_x = data[1]
        self.traj_y = data[2]
        self.traj_x_1 = data[3]
        self.traj_x_2 = data[4]
        self.traj_y_1 = data[5]
        self.traj_y_2 = data[6]
    
    def create_functions(self,traj_data,N,t):        
        self.x_ref = self.traj_function['XRef_func']
        self.y_ref = self.traj_function['YRef_func']
        self.psi_ref = self.traj_function['psiRef_func']
        self.x_ref_d = self.traj_function['XRefdot_func']
        self.y_ref_d = self.traj_function['YRefdot_func']
        self.psi_ref_d = self.traj_function['psiRefdot_func']
        self.x_ref_dd = self.traj_function['XRefddot_func']
        self.y_ref_dd = self.traj_function['YRefddot_func']

        self.times = np.arange(traj_data.traj1.start_time,traj_data.traj2.end_time + N*t,t)
        self.l_s = traj_data.traj2.end_time - traj_data.traj1.start_time

        self.x_s = self.x_ref(1,self.times,self.l_s)
        self.y_s = self.y_ref(1,self.times,self.l_s)
        self.psi_s = self.psi_ref(1,self.times,self.l_s)
        self.dx_s = self.x_ref_d(1,self.times,self.l_s)
        self.dy_s = self.y_ref_d(1,self.times,self.l_s)
        self.dpsi_s = self.psi_ref_d(1,self.times,self.l_s)
        self.ddx_s = self.x_ref_dd(1,self.times,self.l_s)
        self.ddy_s = self.y_ref_dd(1,self.times,self.l_s)

traj = trajecotry()
