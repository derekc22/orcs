"""Existing balance controller with unilateral, friction and sole-moment bounds.

The solved wrenches only allocate native actuator torques. They are NEVER
applied to MuJoCo as external forces, supports, constraints, or object forces.
"""
import mujoco,numpy as np
from scipy.optimize import minimize,LinearConstraint
from compact_gripper_balance import Balance

class BoundedBalance(Balance):
    def __init__(self,m,d):
        super().__init__(m,d)
        self.w=np.tile([0.,0.,m.body_mass.sum()*9.81/2,0.,0.,0.],2)
        C=[]
        for side in range(2):
            def row(**kw):
                r=np.zeros(12)
                for key,value in kw.items():r[6*side+int(key[1:])]=value
                C.append(r)
            row(c2=1.)
            for axis in [0,1]:
                for sign in [-1,1]:row(**{f'c{axis}':sign,'c2':.7})
            # Conservative center of pressure: x [-25,100] mm, y +/-30 mm.
            row(c4=1.,c2=.100);row(c4=-1.,c2=.025)
            row(c3=1.,c2=.030);row(c3=-1.,c2=.030)
            row(c5=1.,c2=.012);row(c5=-1.,c2=.012)
        self.constraint=LinearConstraint(np.array(C),0,np.inf)
        self.last_result=None

    def torques(self,d,qref,vref=None):
        tau,oldw=super().torques(d,qref,vref)
        required=tau+self.J.T@oldw
        T=np.zeros((12,12))
        for i,b in enumerate(self.feet):
            R=d.xmat[b].reshape(3,3)
            T[6*i:6*i+3,6*i:6*i+3]=R.T
            T[6*i+3:6*i+6,6*i+3:6*i+6]=R.T
        J=T@self.J;A=J[:,:6].T;weight=np.array([1.,1.,1.,12.,12.,12.]);Aw=weight[:,None]*A;b=weight*required[:6]
        regular=np.tile([1.,1.,1.,25.,25.,25.],2)*1e-4
        H=Aw.T@Aw+np.diag(regular);g=-Aw.T@b
        fit=minimize(lambda w:.5*w@H@w+g@w,self.w,jac=lambda w:H@w+g,
                     constraints=[self.constraint],method='SLSQP',options=dict(maxiter=60,ftol=1e-7))
        self.w=fit.x;self.last_result=dict(success=bool(fit.success),root_wrench_error=(A@fit.x-required[:6]).tolist(),local_wrenches=fit.x.tolist(),minimum_cone_slack=float(np.min(self.constraint.A@fit.x)))
        return required-J.T@fit.x,fit.x
