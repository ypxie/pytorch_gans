import numpy as np
import os
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch import autograd
from torch.autograd import Variable

from torch.nn.utils import clip_grad_norm
from .utils import plot_img, plot_scalar, save_images, to_device

def calc_gradient_penalty(netD, real_data, fake_data):
    #print real_data.size()
    batch_size = real_data.size()[0]
    one_list = [1]*len(list(real_data.size()[1:]))
    alpha = fake_data.data.new(batch_size,*one_list)
    alpha.uniform_(0,1)
    alpha = alpha.expand_as(real_data)
    
    interpolates = alpha * real_data.data + (1 - alpha) * fake_data.data
    
    #interpolates =  to_device(interpolates, netD.device_id)
    interpolates = Variable(interpolates, requires_grad=True)
    #interpolates.requires_grad = True
    disc_interpolates = netD(interpolates)
    
    #grad_outputs = to_device(torch.ones(disc_interpolates.size()), netD.device_id, False)
    grad_outputs = interpolates.data.new(*disc_interpolates.size())
    grad_outputs.fill_(1.0)
    
    gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolates, grad_outputs=grad_outputs,
                               create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradients = gradients.view(batch_size, -1)
    gradient_penalty = torch.mean((gradients.norm(2, dim=1) - 1) ** 2)
    return gradient_penalty
    

def train_gans(x_sampler, model_root, mode_name, netG, netD, args):
    ngen = getattr(args, 'ngen', 1)
    if args.gpwgan:
        optimizerD = optim.Adam(netD.parameters(), lr= args.d_lr, betas=(0.5, 0.9), weight_decay=args.weight_decay)
        optimizerG = optim.Adam(netG.parameters(), lr= args.g_lr, betas=(0.5, 0.9), weight_decay=args.weight_decay)
    else:   
        optimizerD = optim.RMSprop(netD.parameters(), lr= args.d_lr,  weight_decay=args.weight_decay)
        #optimizerG = optim.Adam(netG.parameters(), lr= args.g_lr, betas=(0.5, 0.9), weight_decay=args.weight_decay)
        optimizerG = optim.RMSprop(netG.parameters(), lr= args.g_lr,  weight_decay=args.weight_decay)
    
    #optimizerD = optim.SGD(netD.parameters(), lr=args.d_lr, momentum=args.momentum,weight_decay=args.weight_decay,nesterov=True)
    #optimizerG = optim.SGD(netG.parameters(), lr=args.g_lr, momentum=args.momentum,weight_decay=args.weight_decay,nesterov=True)

    
    model_folder = os.path.join(model_root, mode_name)
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)

    D_weightspath = os.path.join(model_folder, 'd_weights.pth')
    G_weightspath = os.path.join(model_folder, 'g_weights.pth')
    if args.reuse_weigths == 1:
        if os.path.exists(D_weightspath):
            weights_dict = torch.load(D_weightspath,map_location=lambda storage, loc: storage)
            netD.load_state_dict(weights_dict)# 12)
            print('reload weights from {}'.format(D_weightspath))
        
        if os.path.exists(G_weightspath):
            weights_dict = torch.load(G_weightspath,map_location=lambda storage, loc: storage)
            netG.load_state_dict(weights_dict)# 12)
            print('reload weights from {}'.format(G_weightspath))
    d_loss_plot = plot_scalar(name = "d_loss", env= mode_name, rate = args.display_freq)
    g_loss_plot = plot_scalar(name = "g_loss", env= mode_name, rate = args.display_freq)
    
    z = torch.FloatTensor(args.batch_size, args.noise_dim).normal_(0, 1)
    z = to_device(z, netG.device_id, requires_grad = False)
    
    z_test = torch.FloatTensor(args.batch_size, args.noise_dim).normal_(0, 1)
    z_test = to_device(z_test, netG.device_id, volatile=True)
    
    one = netD.device_id.new(1).fill_(1)  
    one_neg = one * -1
    
    gen_iterations, disc_iterations = 0, 0
    for batch_count in range(args.maxepoch):
        
        if gen_iterations < 25 or gen_iterations % 100 == 0:
            ncritic = 100
        else:
            ncritic = args.ncritic
            
        for p in netD.parameters():  # reset requires_grad
            p.requires_grad = True
        for _ in range(ncritic):
            # (1) Update D network
            for p in netD.parameters():
                if not args.gpwgan:
                    p.data.clamp_(-0.01, 0.01)
            
            real = to_device(x_sampler(), netD.device_id)
            netD.zero_grad()
            
            #batch_size = real.size()[0]
            #z   = z_sampler(batch_size, args.noise_dim)
            z.data.normal_(0, 1)
            
            #z2   = z_sampler(batch_size, args.noise_dim)
            fake = netG(  Variable(z.data, volatile=True)   )
            fake = Variable(fake.data)
            
            disc_real  = netD(real)
            disc_fake = netD(fake)
            
            err_real = torch.mean(disc_real.view(-1))
            err_fake = torch.mean(disc_fake.view(-1)) 
            
            err_real.backward(one_neg)
            err_fake.backward(one)
            
            if args.gpwgan:
                grad_penalty = calc_gradient_penalty(netD, real, fake) * args.gp_lambda
                grad_penalty.backward(one)
            else:
                grad_penalty = 0
                
            err_d = err_fake - err_real
            #d_loss = err_d +  grad_penalty
            #d_loss.backward()
            d_loss_plot.plot(err_d.cpu().data.numpy().mean())
            optimizerD.step()
            
        # (2) Update G network
        for p in netD.parameters():
            p.requires_grad = False  # to avoid computation
        
        for _ in range(ngen):
            netG.zero_grad()
            
            z.data.normal_(0, 1)
            fake = netG( z  )
            disc_fake = netD(fake)
            g_loss = torch.mean(disc_fake.view(-1) )
            
            g_loss.backward(one_neg)
            optimizerG.step()
            
            g_loss_plot.plot(-g_loss.cpu().data.numpy().mean())
            gen_iterations += 1
        
        # Calculate dev loss and generate samples every 100 iters
        if batch_count % args.display_freq == 0:
            print('save tmp images, :)')
            #z1   = z_sampler(batch_size, args.noise_dim)
            
            
            samples = netG(z_test).cpu().data.numpy()
            imgs = save_images(
                        samples,
                        os.path.join(args.save_folder,'samples_{}.png'.format(batch_count) ),save=False,dim_ordering = 'th'
                        )
            print(samples.shape)
            plot_img(X=imgs, win='sample_img', env=mode_name)
            
            true_imgs = save_images(real.cpu().data.numpy(), save=False,dim_ordering = 'th')
            plot_img(X=true_imgs, win='real_img', env=mode_name)
        
        if batch_count % args.save_freq == 0:
            D_cur_weights = netD.state_dict()
            G_cur_weights = netG.state_dict()
            torch.save(D_cur_weights, D_weightspath)
            torch.save(G_cur_weights, G_weightspath)
            print('save weights to {} and {}'.format(D_weightspath, G_weightspath),batch_count,args.save_freq)

