# ***********
# * Imports *
# ***********

import os
import time
import json
from itertools import product
from collections import namedtuple
from collections import OrderedDict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid

from IPython.display import display, clear_output

# ************************
# * Function definitions *
# ************************

def to_img(x):
    """
    Denormalises Tensor `x` (normalised from -1 to 1) to image format (from 0 to 1).
    """
    
    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    return x

def imgviz(images, save_fname=None, nrow=10):
    """
    Displays `images` in a grid, and saves (optionally) output as an image file with
    filename `save_fname`. Format of input must be a Tensor of size 4 (B, C, H, W).
    """
    
    assert len(images.shape) == 4, "Size of input images must be 4."
    
    grid = make_grid(to_img(images), nrow=nrow)
    plt.figure(figsize=(15,15))
    plt.imshow(np.transpose(grid, (1,2,0)))
    
    # If save_fname is given and is a string, save image to local directory
    if type(save_fname) == str:
        if not os.path.exists('./saved_imgs'):
            os.mkdir('./saved_imgs')
        plt.imsave('./saved_imgs/' + save_fname, np.transpose(grid, (1,2,0)).numpy())
    
    # Show image
    plt.show()


# *********************
# * Class definitions *
# *********************

class Epoch:
    """
    Epoch class, keeps track of progress for training epochs.
    """
    
    def __init__(self):
        self.count = 0
        self.d_loss = 0
        self.gp = 0
        self.g_loss = 0
        self.d_count = 0
        self.g_count = 0
        self.start_time = None
        self.active = False
    
    # Initialises epoch if not active
    def begin(self):
        assert not self.active, "Cannot initialise epoch, already active"
        self.count += 1
        self.d_loss = 0
        self.gp = 0
        self.g_loss = 0
        self.d_count = 0
        self.g_count = 0
        self.start_time = time.time()
        self.active = True
    
    # Finalises epoch if active and returns epoch results
    def end(self):
        assert self.active, "Cannot finalise epoch, no epoch active"
        duration = self.duration()
        d_loss = self.d_loss / self.d_count
        gp = self.gp / self.d_count
        g_loss = self.g_loss / self.g_count
        self.active = False
        return duration, d_loss, gp, g_loss
    
    # Adds batch loss to total loss
    def add_loss(self, d_loss, gp, g_loss=None):
        self.d_loss += d_loss.item()
        self.gp += gp.item()
        self.d_count += 1
        if g_loss is not None:
            self.g_loss += g_loss.item()
            self.g_count += 1

    # Returns duration of epoch processing time
    def duration(self):
        return time.time() - self.start_time


class Run:
    """
    Run class, keeping track of progress for training runs.
    """
    
    def __init__(self):
        self.hparams = None
        self.count = 0
        self.data = []
        self.start_time = None
        self.active = False
    
    # Initialises run if not active
    def begin(self, hparams):
        assert not self.active, "Cannot initialise run, already active"
        self.hparams = hparams
        self.count += 1
        self.start_time = time.time()
        self.active = True
    
    # Finalises run if active
    def end(self, epoch):
        assert self.active, "Cannot finalise run, no run active"
        epoch.count = 0
        self.active = False
    
    # Returns duration of run processing time
    def duration(self):
        return time.time() - self.start_time
    
    # Appends results to all data and displays data
    def append_and_display_data(self, results):
        # Append results to data
        self.data.append(results)
        
        # Create Pandas DataFrame to all data and display data
        df = pd.DataFrame.from_dict(self.data, orient='columns')
        clear_output(wait=True)
        display(df)
    
    # Saves all data as a json file
    def save(self, filename):
        pd.DataFrame.from_dict(self.data, orient='columns').to_csv(f'{filename}.csv')

        with open(f'{filename}.json', 'w', encoding='utf-8') as f:
            json.dump(self.data, f, ensure_ascii=False, indent=4)


class RunManager:
    """
    RunManager class, keeping track of overall training progress.
    """
    
    def __init__(self):
        self.epoch = Epoch()
        self.run = Run()
        self.net = None
        self.tb = None
        self.images = None
        self.latents = None
    
    def begin_run(self, hparams, net, test_images, test_latents):
        # Begin next run with new hyperparameters
        self.run.begin(hparams)
        
        # Setup network, data and SummaryWriter
        self.net = net
        self.tb = SummaryWriter(comment=f'-{hparams}')
        self.latents = test_latents
        
        # Add images and graph to TensorBoard
        self.images = test_images
        grid = make_grid(to_img(self.images), nrow=10)
        self.tb.add_image('original images', grid)
        #self.tb.add_graph(self.net, self.images) # TO-DO: fix this
        
    def end_run(self):
        self.tb.flush()
        self.tb.close()
        self.net = None
        self.tb = None
        self.images = None
        self.latents = None
        self.run.end(self.epoch)
        
    def begin_epoch(self):
        assert self.run.active, "Run is not active, cannot initialise epoch"
        self.epoch.begin()
        
    def end_epoch(self):
        run_duration = self.run.duration()
        epoch_duration, d_loss, gp, g_loss = self.epoch.end()
        
        self.tb.add_scalar('Critic loss', d_loss, self.epoch.count)
        self.tb.add_scalar('Gradient penalty', gp, self.epoch.count)
        self.tb.add_scalar('Generator loss', g_loss, self.epoch.count)
        with torch.no_grad():
            gen_imgs = self.net.generator(self.latents)
            grid = make_grid(to_img(gen_imgs), nrow=10)
            self.tb.add_image('generated images', grid, self.epoch.count)
        
        for name, param in self.net.named_parameters():
            self.tb.add_histogram(name, param, self.epoch.count)
            self.tb.add_histogram(f'{name}.grad', param.grad, self.epoch.count)
        
        if not os.path.exists('./models_CIFAR'):
            os.mkdir('./models_CIFAR')
        torch.save(self.net, './models_CIFAR/best_' + str(self.run.hparams) + '.pth')
        self.save_img(grid, 'epoch{0}.png'.format(self.epoch.count))
        
        results = OrderedDict()
        results['run'] = self.run.count
        results['epoch'] = self.epoch.count
        results['d_loss'] = d_loss
        results['gp'] = gp
        results['g_loss'] = g_loss
        results['epoch duration'] = epoch_duration
        results['run duration'] = run_duration
        
        for k, v in self.run.hparams._asdict().items():
            if k == 'batch_size':
                results[k] = v * 10
            else:
                results[k] = v
        
        self.run.append_and_display_data(results)
        self.net.display(self.images, self.latents)
        
    def track_loss(self, d_loss, gp, g_loss=None):
        self.epoch.add_loss(d_loss, gp, g_loss)
    
    def save(self, filename):
        self.run.save(filename)
    
    # Save image to local directory
    def save_img(self, grid, filename):
        if not os.path.exists('./gif_CIFAR'):
            os.mkdir('./gif_CIFAR')
        
        plt.figure(figsize=(15,15))
        plt.imsave('./gif_CIFAR/' + filename, np.transpose(grid, (1,2,0)).numpy())
    

class RunBuilder:
    """
    RunBuilder class, builds Run tuples that encapsulate hyperparameters for each training run.
    """
    
    @staticmethod
    def get_runs_product(params):
        Run = namedtuple("Run", params.keys())
        
        runs = []
        for v in product(*params.values()):
            runs.append(Run(*v))
    
        return runs
    
    @staticmethod
    def get_runs_from_params(names, params):
        assert all([len(names) == len(p) for p in params.values()]), \
            "Length of names must be equal to length of parameters for each run"
        
        Run = namedtuple("Run", names)
        
        runs = []
        for v in params.values():
            runs.append(Run(*v))
        return runs
