#! /usr/bin/env python
import os

# don't use gpu
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import numpy as np

import numpy as np
import matplotlib.pyplot as plt
from scaledgp import ScaledGP
from scipy import signal
from progress.bar import Bar
import random
from utils import *

BASE_PATH = os.path.expanduser('~/Documents')

class ModelService(object):

    def __init__(self,xdim,odim,use_obs, use_service = True):

        self.xdim=xdim
        self.odim=odim
        self.use_obs = use_obs
        self.verbose = True

        self.config = {}


    def reconfigure_cb(self, config):
        self.N_data = config["N_data"]
        self.verbose = config["learning_verbose"]
        self.N_updates = config["N_updates"]
        self.config["meta_batch_size"] = config["meta_batch_size"]
        self.config["data_horizon"] = config["data_horizon"]
        self.config["test_horizon"] = config["test_horizon"]
        self.config["learning_rate"] = config["learning_rate"]
        self.config["min_datapoints"] = config["min_datapoints"]
        self.config["save_data_interval"] = config["save_data_interval"]

    def predict(self,req):
        # overload
        return None

    def train(self,goal):
        # overload
        return None

    def add_data(self,req):
        # overload
        return None

    def scale_data(self,x,xmean,xstd):
        if (xstd == 0).any():
            return (x-xmean)
        else:
            return (x - xmean) / xstd

    def unscale_data(self,x,xmean,xstd):
        if (xstd == 0).any():
            return x + xmean
        else:
            return x * xstd + xmean


class ModelGPService(ModelService):
    def __init__(self,xdim,odim,use_obs=False,use_service=True):
        ModelService.__init__(self,xdim,odim,use_obs,use_service)
        # note:  use use_obs and observations with caution.  model may overfit to this input.
        model_xdim=self.xdim//2
        if self.use_obs:
             model_xdim += self.odim
        model_ydim=self.xdim//2

        self.m = ScaledGP(xdim=model_xdim,ydim=model_ydim)
        self.y = np.zeros((0,model_ydim))
        self.Z = np.zeros((0,model_xdim))
        self.N_data = 400

    def rotate(self,x,theta):
        x_body = np.zeros((2,1))
        x_body[0] = x[0] * np.cos(theta) + x[1] * np.sin(theta)
        x_body[1] = -x[0] * np.sin(theta) + x[1] * np.cos(theta)
        return x_body

    def make_input(self,x,obs):
        # format input vector
        theta = obs[0]
        x_body = self.rotate(x[2:-1,:],theta)
        if self.use_obs:
            Z = np.concatenate((x_body,obs[1:,:])).T
        else:
            Z = np.concatenate((x_body)).T

        #normalize input by mean and variance
        # Z = (Z - self.Zmean) / self.Zvar

        return Z

    def predict(self,req):
        x = np.expand_dims(req.x, axis=0).T
        obs = np.expand_dims(req.obs, axis=0).T

        # format the input and use the model to make a prediction.
        Z = self.make_input(x,obs)
        y, var = self.m.predict(Z)
        # theta = np.arctan2(x[3]*x[4],x[2]*x[4])
        theta=obs[0]
        y_out = self.rotate(y.T,-theta)

        resp = Predict_Model(y_out.flatten(),var.T.flatten())
      
        return resp

    def train(self, goal=None):
        success = True

        if goal is not None:
            # goal was cancelled
            if self._action_service.is_preempt_requested():
                print("Preempt training request")
                self._action_service.set_preempted()
                success = False

        # train model.  this gets called by the training thread on timer_cb() in adaptive_clbf_node.
        if success and self.Z.shape[0] > 0 and self.Z.shape[0] == self.y.shape[0]:
            self.m.optimize(self.Z,self.y)
            if goal is not None:
                self._train_result.model_trained = True
                self._action_service.set_succeeded(self._train_result)
        else:
            if goal is not None:
                self._train_result.model_trained = False
                self._action_service.set_succeeded(self._train_result)
    
    def add_data(self,req):

        x_next = np.expand_dims(req.x_next, axis=0).T
        x = np.expand_dims(req.x, axis=0).T
        mu_model = np.expand_dims(req.mu_model, axis=0).T
        obs = np.expand_dims(req.obs, axis=0).T
        dt = req.dt

        # add a sample to the history of data
        x_dot = (x_next[2:-1,:]-x[2:-1,:])/dt
        ynew = x_dot - mu_model
        Znew = self.make_input(x,obs)
        # theta = np.arctan2(x[3]*x[4],x[2]*x[4])
        theta=obs[0]
        ynew_rotated = self.rotate(ynew,theta)
        self.y = np.concatenate((self.y,ynew_rotated.T))
        self.Z = np.concatenate((self.Z,Znew))

        # throw away old samples if too many samples collected.
        if self.y.shape[0] > self.N_data:
            self.y = self.y[-self.N_data:,:]
            self.Z = self.Z[-self.N_data:,:]
            # self.y = np.delete(self.y,random.randint(0,self.N_data-1),axis=0)
            # self.Z = np.delete(self.Z,random.randint(0,self.N_data-1),axis=0)

        if self.verbose:
            print("obs", obs)
            print("ynew",ynew)
            print("ynew_rotated", ynew_rotated)
            print("Znew",Znew)
            print("x_dot",x_dot)
            print("mu_model",mu_model)
            print("dt",dt)
            print("n data:", self.y.shape[0])
            # print("prediction error:", self.predict_error)
            # print("predict var:", self.predict_var)

