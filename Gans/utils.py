try:
    from visdom import Visdom
except:
    print('Better install visdom')
import numpy as np
import random

import scipy.misc
from scipy.misc import imsave

import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn as nn

def to_variable(x, requires_grad=True,  var=True,volatile=False):
    
    if type(x) is Variable:
        return x
    if type(x) is np.ndarray:
        x = torch.from_numpy(x.astype(np.float32))
    if var:
        x = Variable(x, requires_grad=requires_grad, volatile=volatile)
    x.volatile = volatile 
    
    return x

def to_device(src, ref, var = True, volatile = False, requires_grad=True):
    requires_grad = requires_grad and (not volatile)
    src = to_variable(src, var=var, volatile=volatile,requires_grad=requires_grad)
    return src.cuda(ref.get_device()) if ref.is_cuda else src


class plot_scalar(object):
    def __init__(self, name='default', env='main', rate= 1, handler=None):

        self.__dict__.update(locals())
        self.values = []
        self.steps = []
        if self.handler is None:
            self.handler = Visdom()
        self.count = 0
        
    def plot(self,values, step = None):
        self.count += 1
        if step is None:
            step = self.count
        self.steps.append(step)
        self.values.append(values)
        
        if self.count % self.rate == 0:
            self.flush()
        
    def reset(self):
        self.steps = []
        self.values = []

    def flush(self):
        print('flush the plot. :)')
        assert type(self.values) is list, 'values have to be list'
        if type(self.values[0]) is not list:
            self.values = [self.values]
             
        n_lines = len(self.values)
        repeat_steps = [self.steps]*n_lines
        steps  = np.array(repeat_steps).transpose()
        values = np.array(self.values).transpose()
        
        assert not np.isnan(values).any(), 'nan error in loss!!!'
        res = self.handler.line(
                X = steps,
                Y=  values,
                win= self.name,
                update='append',
                opts=dict(title = self.name, legend=None),
                env = self.env
            )

        if res != self.name:
            self.handler.line(
                X=steps,
                Y=values,
                win=self.name,
                env=self.env,
                opts=dict(title=self.name, legend=None)
            )

        self.reset()


def plot_img(X=None, win= None, env=None, plot=None):
    if plot is None:
        plot = Visdom()
    if X.ndim == 2:
        plot.heatmap(X=np.flipud(X), win=win,
                 opts=dict(title=win), env=env)
    elif X.ndim == 3:
        # X is BWC
        norm_img = normalize_img(X)
        plot.image(norm_img.transpose(2,0,1), win=win,
                   opts=dict(title=win), env=env)

def normalize_img(X):
    min_, max_ = np.min(X), np.max(X)
    X = (X - min_)/ (max_ - min_ + 1e-9)
    X = X*255
    return X.astype(np.uint8)

def save_images(X, save_path=None, save=True, dim_ordering='tf'):
    # [0, 1] -> [0,255]
    #if isinstance(X.flatten()[0], np.floating):
    #    X = (255.99*X).astype('uint8')
    n_samples = X.shape[0]
    rows = int(np.sqrt(n_samples))
    while n_samples % rows != 0:
        rows -= 1

    nh, nw = rows, n_samples//rows

    if X.ndim == 4:
        # BCHW -> BHWC
        if dim_ordering == 'tf':
            pass
        else:           
            X = X.transpose(0,2,3,1)
        h, w, c = X[0].shape[:3]
        hgap, wgap = int(0.1*h), int(0.1*w)
        img = np.zeros(((h+hgap)*nh - hgap, (w+wgap)*nw-wgap,c))
    elif X.ndim == 3:
        h, w = X[0].shape[:2]
        hgap, wgap = int(0.1*h), int(0.1*w)
        img = np.zeros(((h+hgap)*nh - hgap, (w+wgap)*nw - wgap))
    else:
        assert 0, 'you have wrong number of dimension input {}'.format(X.ndim) 
    for n, x in enumerate(X):
        i = n%nw
        j = n // nw
        rs, cs = j*(h+hgap), i*(w+wgap)
        #print(i,j, h,w, x.shape, img.shape, rs,cs, rs+h,cs+w )
        img[rs:rs+h, cs:cs+w] = x
    if c == 1:
        img = img[:,:,0]
    #imshow(img)
    #print(save_path)
    if save:
        writeImg(img.astype(np.uint8), save_path)
    return img
