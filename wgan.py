#!/usr/bin/env python3

# Reference 1 : https://github.com/pytorch/examples
# Reference 3 : https://arxiv.org/pdf/1511.06434.pdf
# Reference 4 : https://arxiv.org/pdf/1701.07875.pdf
# Reference 5 : https://github.com/martinarjovsky/WassersteinGAN
# To get TensorBoard output, use the python command: tensorboard --logdir /home/alexia/Output/WGAN

## Parameters

from Gans.utils import plot_img, plot_scalar, save_images, to_device
from Gans.Models import DCGAN_D as Disc
from Gans.Models import DCGAN_G as Gen

import os
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--image_size', type=int, default=64)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--n_colors', type=int, default=3)
parser.add_argument('--z_size', type=int, default=100) # DCGAN paper original value
parser.add_argument('--G_h_size', type=int, default=64, help='Number of hidden nodes in the Generator. Too small leads to bad results, too big blows up the GPU RAM.') # DCGAN paper original value
parser.add_argument('--D_h_size', type=int, default=64, help='Number of hidden nodes in the Discriminator. Too small leads to bad results, too big blows up the GPU RAM.') # DCGAN paper original value
parser.add_argument('--lr_D', type=float, default=.00005, help='Discriminator learning rate') # WGAN original value
parser.add_argument('--lr_G', type=float, default=.00005, help='Generator learning rate')
parser.add_argument('--n_epoch', type=int, default=500000)
parser.add_argument('--n_critic', type=int, default=5, help='Number of training with D before training G') # WGAN original value
parser.add_argument('--clip', type=float, default=.01, help='Clipping value') # WGAN original value
parser.add_argument('--SELU', type=bool, default=False, help='Using scaled exponential linear units (SELU) which are self-normalizing instead of ReLU with BatchNorm. This improves stability.')
parser.add_argument('--seed', type=int)
#parser.add_argument('--input_folder', default='/home/alexia/Datasets/Meow_64x64', help='input folder')
parser.add_argument('--output_folder', default='.Output/WGAN', help='output folder')
parser.add_argument('--G_load', default='', help='Full path to Generator model to load (ex: /home/output_folder/run-5/models/G_epoch_11.pth)')
parser.add_argument('--D_load', default='', help='Full path to Discriminator model to load (ex: /home/output_folder/run-5/models/D_epoch_11.pth)')
parser.add_argument('--cuda', type=bool, default=True, help='enables cuda')
parser.add_argument('--n_gpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--n_workers', type=int, default=2, help='Number of subprocess to use to load the data. Use at least two or the number of cpu cores - 1.')
param = parser.parse_args()

## Imports
os.environ['CUDA_VISIBLE_DEVICES'] = "1"

# Time
import time
start = time.time()

# Check folder run-i for all i=0,1,... until it finds run-j which does not exists, then creates a new folder run-j
import os
run = 0
base_dir = f"{param.output_folder}/run-{run}"
while os.path.exists(base_dir):
    run += 1
    base_dir = f"{param.output_folder}/run-{run}"
os.makedirs(base_dir)
logs_dir = f"{base_dir}/logs"
os.makedirs(logs_dir)
os.makedirs(f"{base_dir}/images")
os.makedirs(f"{base_dir}/models")

# where we save the output
log_output = open(f"{logs_dir}/log.txt", 'w')
print(param)
print(param, file=log_output)

import numpy 
import torch
import torch.autograd as autograd
from torch.autograd import Variable

# For plotting the Loss of D and G using tensorboard
#from tensorboard_logger import configure, log_value
#configure(logs_dir, flush_secs=5)

if param.cuda:
    import torch.backends.cudnn as cudnn
    cudnn.benchmark = True


## Setting seed
import random
if param.seed is None:
    param.seed = random.randint(1, 10000)
print("Random Seed: ", param.seed)
print("Random Seed: ", param.seed, file=log_output)
random.seed(param.seed)
torch.manual_seed(param.seed)
if param.cuda:
    torch.cuda.manual_seed_all(param.seed)

## Transforming images
mode_name = 'good_wgan'
d_loss_plot = plot_scalar(name = "d_loss", env=  mode_name, rate = 100)
g_loss_plot = plot_scalar(name = "g_loss", env=  mode_name, rate = 100)
    
    
class DataIterator:
    def __init__(self, h5data, batch_size):
        '''
        data: a hdf5 opened pointer
        Index: one array index, [3,5,1,9, 100] index of candidates 
        '''
        self.__dict__.update(locals())
        self.images = h5data['images']
        self.total_imgs = self.images.shape[0]
        self.batch = np.zeros((self.batch_size, )+ self.images.shape[1::]).astype(np.float32)
    def next(self):
        return self.__next__()

    def __next__(self):
        random_indexes = np.random.choice(self.total_imgs,  size=self.batch_size, replace=False)
        for idx, index in enumerate(random_indexes):
                self.batch[idx] = self.images[index]
        return self.batch

    def __iter__(self):
        return self


## Models

## Models
# The number of layers is implicitly determined by the image size
# image_size = (4,8,16,32,64, 128, 256, 512, 1024) leads to n_layers = (1, 2, 3, 4, 5, 6, 7, 8, 9)
# The more layers the bigger the neural get so it's best to decrease G_h_size and D_h_size when the image input is bigger

# DCGAN generator
class DCGAN_G(torch.nn.Module):
    def __init__(self):
        super(DCGAN_G, self).__init__()
        main = torch.nn.Sequential()
        self.register_buffer('device_id', torch.zeros(1))
        # We need to know how many layers we will use at the beginning
        mult = param.image_size // 8

        ### Start block
        # Z_size random numbers
        main.add_module('Start-ConvTranspose2d', torch.nn.ConvTranspose2d(param.z_size, param.G_h_size * mult, kernel_size=4, stride=1, padding=0, bias=False))
        if param.SELU:
            main.add_module('Start-SELU', torch.nn.SELU(inplace=True))
        else:
            main.add_module('Start-BatchNorm2d', torch.nn.BatchNorm2d(param.G_h_size * mult))
            main.add_module('Start-ReLU', torch.nn.ReLU())
        # Size = (G_h_size * mult) x 4 x 4

        ### Middle block (Done until we reach ? x image_size/2 x image_size/2)
        i = 1
        while mult > 1:
            main.add_module('Middle-ConvTranspose2d [%d]' % i, torch.nn.ConvTranspose2d(param.G_h_size * mult, param.G_h_size * (mult//2), kernel_size=4, stride=2, padding=1, bias=False))
            if param.SELU:
                main.add_module('Middle-BatchNorm2d [%d]' % i, torch.nn.BatchNorm2d(param.G_h_size * (mult//2)))
                main.add_module('Middle-SELU [%d]' % i, torch.nn.SELU(inplace=True))
            else:
                main.add_module('Middle-BatchNorm2d [%d]' % i, torch.nn.BatchNorm2d(param.G_h_size * (mult//2)))
                main.add_module('Middle-ReLU [%d]' % i, torch.nn.ReLU())
            # Size = (G_h_size * (mult/(2*i))) x 8 x 8
            mult = mult // 2
            i += 1

        ### End block
        # Size = G_h_size x image_size/2 x image_size/2
        main.add_module('End-ConvTranspose2d', torch.nn.ConvTranspose2d(param.G_h_size, param.n_colors, kernel_size=4, stride=2, padding=1, bias=False))
        main.add_module('End-Tanh', torch.nn.Tanh())
        # Size = n_colors x image_size x image_size
        self.main = main
        self.apply(weights_init)
    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and param.n_gpu > 1:
            output = torch.nn.parallel.data_parallel(self.main, input, range(param.n_gpu))
        else:
            output = self.main(input)
        return output

# DCGAN discriminator (using somewhat the reverse of the generator)
class DCGAN_D(torch.nn.Module):
    def __init__(self):
        super(DCGAN_D, self).__init__()
        main = torch.nn.Sequential()
        self.register_buffer('device_id', torch.zeros(1))
        ### Start block
        # Size = n_colors x image_size x image_size
        main.add_module('Start-Conv2d', torch.nn.Conv2d(param.n_colors, param.D_h_size, kernel_size=4, stride=2, padding=1, bias=False))
        if param.SELU:
            main.add_module('Start-SELU', torch.nn.SELU(inplace=True))
        else:
            main.add_module('Start-LeakyReLU', torch.nn.LeakyReLU(0.2, inplace=True))
        image_size_new = param.image_size // 2
        # Size = D_h_size x image_size/2 x image_size/2

        ### Middle block (Done until we reach ? x 4 x 4)
        mult = 1
        i = 0
        while image_size_new > 4:
            main.add_module('Middle-Conv2d [%d]' % i, torch.nn.Conv2d(param.D_h_size * mult, param.D_h_size * (2*mult), kernel_size=4, stride=2, padding=1, bias=False))
            if param.SELU:
                main.add_module('Middle-SELU [%d]' % i, torch.nn.SELU(inplace=True))
            else:
                main.add_module('Middle-BatchNorm2d [%d]' % i, torch.nn.BatchNorm2d(param.D_h_size * (2*mult)))
                main.add_module('Middle-LeakyReLU [%d]' % i, torch.nn.LeakyReLU(0.2, inplace=True))
            # Size = (D_h_size*(2*i)) x image_size/(2*i) x image_size/(2*i)
            image_size_new = image_size_new // 2
            mult = mult*2
            i += 1

        ### End block
        # Size = (D_h_size * mult) x 4 x 4
        main.add_module('End-Conv2d', torch.nn.Conv2d(param.D_h_size * mult, 1, kernel_size=4, stride=1, padding=0, bias=False))
        # Note: No more sigmoid in WGAN, we take the mean now
        # Size = 1 x 1 x 1 (Is a real cat or not?)
        self.main = main
        self.apply(weights_init)
    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and param.n_gpu > 1:
            output = torch.nn.parallel.data_parallel(self.main, input, range(param.n_gpu))
        else:
            output = self.main(input)
        # From batch_size x 1 x 1 to 1 x 1 x 1 by taking the mean (DCGAN used the sigmoid instead before)
        output = output.mean(0)
        # Convert from 1 x 1 x 1 to 1 so that we can compare to given label (cat or not?)
        return output.view(1)

## Weights init function, DCGAN use 0.02 std
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        # Estimated variance, must be around 1
        m.weight.data.normal_(1.0, 0.02)
        # Estimated mean, must be around 0
        m.bias.data.fill_(0)

import numpy as np
import argparse, os
import torch, h5py

home = os.path.expanduser('~')

h5dataFile = os.path.join(home, 'Dataset/cat/CatImg_size_64.h5')
with h5py.File(h5dataFile, 'r') as h5_data:
    data_sampler = DataIterator(h5_data, param.batch_size).next
    ## Initialization
    G = DCGAN_G()
    D = DCGAN_D()
    
    #D = Disc(input_size = 64, num_chan = 3, hid_dim = 64, out_dim = 1,selu=False)
    #G = Gen(input_size  = 64, noise_dim = 100, num_chan=3, hid_dim= 64,selu=False)
    
    #D = Disc(param)
    #G = Gen(param)
    # Initialize weights
    #G.apply(weights_init)
    #D.apply(weights_init)
    
    # Load existing models
    if param.G_load != '':
        G.load_state_dict(torch.load(param.G_load))
    if param.D_load != '':
        D.load_state_dict(torch.load(param.D_load))
    
    print(G)
    print(D)
    
    # Soon to be variables
    x = torch.FloatTensor(param.batch_size, param.n_colors, param.image_size, param.image_size)
    z = torch.FloatTensor(param.batch_size, param.z_size, 1, 1)
    # This is to see during training, size and values won't change
    z_test = torch.FloatTensor(param.batch_size, param.z_size, 1, 1).normal_(0, 1)
    one = torch.FloatTensor([1])
    one_neg = one * -1
    
    # Everything cuda
    if param.cuda:
        G = G.cuda()
        D = D.cuda()
        x = x.cuda()
        z = z.cuda()
        z_test = z_test.cuda()
        one, one_neg = one.cuda(), one_neg.cuda()
    
    # Now Variables
    x = Variable(x)
    z = Variable(z)
    z_test = Variable(z_test)
    
    # Optimizer
    optimizerD = torch.optim.RMSprop(D.parameters(), lr=param.lr_D)
    optimizerG = torch.optim.RMSprop(G.parameters(), lr=param.lr_G)
    
    ## Fitting model
    
    gen_iterations = 0
    for epoch in range(param.n_epoch):
        i = 0
        for p in D.parameters():
            p.requires_grad = True
            
        # "Trick" used in the Wassertein GAN paper for more stable convergence
        if gen_iterations < 25 or gen_iterations % 500 == 0:
            N_critic = 100
        else:
            N_critic = param.n_critic
            
        t = 0
        while t < N_critic :
            ########################
            # (1) Update D network #
            ########################

            D.zero_grad()

            # Clip weights
            for p in D.parameters():
                p.data.clamp_(-param.clip, param.clip)

            # Sample real data
            real_images = to_device(data_sampler(), D.device_id ,False)
            # Mostly necessary for the last one because if N might not be a multiple of batch_size
            current_batch_size = real_images.size(0)
            if param.cuda:
                real_images = real_images.cuda()
            # Transfer batch of images to x
            x.data.resize_as_(real_images).copy_(real_images)
            # Discriminator Loss real
            
            errD_real = D(x)
            
            errD_real.backward(one)

            # Sample fake data
            # Note that z might be bigger than x here, this is done like this in Wassertein paper, but it could probably be changed
            z.data.resize_(current_batch_size, param.z_size, 1, 1).normal_(0, 1)
            # Volatile requires less memory and make things sightly faster than detach(), so wy not use it with DCGAN?
            # Simply because we reuse the same fake images, but in WGAN we generate new fake images after training for a while the Discriminator
            z_volatile = Variable(z.data, volatile = True)
            x_fake = Variable(G(z_volatile).data)
            # Discriminator Loss fake
            errD_fake = D(x_fake)
            errD_fake.backward(one_neg)

            # Optimize
            errD = (errD_real - errD_fake)
            optimizerD.step()
            d_loss_plot.plot(-errD.cpu().data.numpy().mean())

            # Iterate up
            t = t + 1
            i = i + 1

        ########################
        # (2) Update G network #
        ########################
        for p in D.parameters():
            p.requires_grad = False

        G.zero_grad()

        # Sample fake data
        z.data.resize_(param.batch_size, param.z_size, 1, 1).normal_(0, 1)
        x_fake = G(z)
        # Generator Loss
        errG = D(x_fake)
        errG.backward(one)
        optimizerG.step()

        # Log results so we can see them in TensorBoard after
        #log_value('errD', -errD.data[0], gen_iterations)
        #log_value('errG', errG.data[0], gen_iterations)
        g_loss_plot.plot(errG.cpu().data.numpy().mean())
        
        gen_iterations = gen_iterations + 1

        if gen_iterations % 50 == 0:
            end = time.time()
            print('[%d] W_distance: %.4f Loss_G: %.4f time:%.4f' % (gen_iterations, -errD.data[0], errG.data[0], end - start))
            #print('[%d] W_distance: %.4f Loss_G: %.4f time:%.4f' % (gen_iterations, -errD.data[0], errG.data[0], end - start), file=log_output)
            # Fake images saved
            fake_test = G(z_test).cpu().data.numpy()
            imgs = save_images(
                        fake_test,
                        None,save=False,dim_ordering = 'th'
                        )
            
            plot_img(X=imgs, win='sample_img', env=mode_name)
            
            true_imgs = save_images(real_images.cpu().numpy(), save=False,dim_ordering = 'th')
            plot_img(X=true_imgs, win='real_img', env=mode_name)
            
            #vutils.save_image(fake_test.data, '%s/run-%d/images/fake_samples_iter%05d.png' % (param.output_folder, run, gen_iterations/50), normalize=True)
        # Save models
        if gen_iterations % 500 == 0:
            torch.save(G.state_dict(), '%s/run-%d/models/G_%d.pth' % (param.output_folder, run, gen_iterations/50))
            torch.save(D.state_dict(), '%s/run-%d/models/D_%d.pth' % (param.output_folder, run, gen_iterations/50))# -*- coding: utf-8 -*-

