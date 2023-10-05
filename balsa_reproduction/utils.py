import numpy as np

class Predict_Model:
    def __init__(self,y_out,var,dt=0.1,result=True):
        self.y_out = y_out
        self.var = var
        self.result = result
        self.dt = dt

class Predict_Runtime:
    def __init__(self,x,obs):
        self.x = x
        self.obs = obs
        
class AddData_Model:
    def __init__(self,x_next, x, mu_model, obs ,dt):
        self.x_next = x_next
        self.x = x
        self.mu_model = mu_model
        self.obs = obs
        self.dt = dt
        
   