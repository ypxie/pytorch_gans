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


class ONE_SIDED(nn.Module):
    def __init__(self):
        super(ONE_SIDED, self).__init__()

        main = nn.ReLU()
        self.main = main

    def forward(self, input):
        output = self.main(-input)
        output = -output.mean(0)
        return output.view(1)

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
    
    optimizerD = optim.Adam(netD.parameters(), lr= args.d_lr, betas=(0.5, 0.9), weight_decay=args.weight_decay)
    optimizerG = optim.Adam(netG.parameters(), lr= args.g_lr, betas=(0.5, 0.9), weight_decay=args.weight_decay)

    
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
    
    
    
    # sigma for MMD
    base = 1.0
    sigma_list = [1, 2, 4, 8, 16]
    sigma_list = [sigma / base for sigma in sigma_list]
    
    lambda_MMD = 1.0
    lambda_AE_X = 8.0
    lambda_AE_Y = 8.0
    lambda_rg = 16.0
    
    
    one_sided = ONE_SIDED()
    
    one = netD.device_id.new(1).fill_(1)  
    one_neg = one * -1
    base_loss = nn.SmoothL1Loss(size_average=True)
    
    gen_iterations, disc_iterations = 0, 0
    for batch_count in range(args.maxepoch):
        
        if gen_iterations < 25 or gen_iterations % 500 == 0:
            ncritic = 100
        else:
            ncritic = args.ncritic
        
        for p in netD.parameters():  # reset requires_grad
                p.requires_grad = True
                
        for _ in range(ncritic):
            # (1) Update D network
            
            # do not clamp paramters of NetD decoder!!!
            
            #for p in netD.encoder.parameters():
            #    p.data.clamp_(-0.01, 0.01)
            
            real = to_device(x_sampler(), netD.device_id, requires_grad= False)
            netD.zero_grad()
            
            f_enc_real_D, f_dec_real_D = netD(real)
            
            #batch_size = real.size()[0]
            #z   = z_sampler(batch_size, args.noise_dim)
            z.data.normal_(0, 1)
            
            #z2   = z_sampler(batch_size, args.noise_dim)
            fake = netG(  Variable(z.data, volatile=True))
            fake = Variable(fake.data)
            
            f_enc_Y_D, f_dec_fake_D = netD(fake)
            
            # compute biased MMD2 and use ReLU to prevent negative value
            mmd2_D = mix_rbf_mmd2(f_enc_real_D, f_enc_Y_D, sigma_list)
            mmd2_D = F.relu(mmd2_D)
            
            # compute rank hinge loss
            one_side_errD = one_sided(f_enc_real_D.mean(0) - f_enc_Y_D.mean(0))
            
            # compute L2-loss of AE
            L2_AE_X_D = base_loss(f_dec_real_D, real)
            L2_AE_Y_D = base_loss(f_dec_fake_D, fake)
            
            errD = torch.sqrt(mmd2_D) + lambda_rg * one_side_errD - lambda_AE_X * L2_AE_X_D - lambda_AE_Y * L2_AE_Y_D
            errD.backward(one_neg)
            
            optimizerD.step()
            d_loss_plot.plot(-errD.cpu().data.numpy().mean())
            
            
        # (2) Update G network
        for p in netD.parameters():
            p.requires_grad = False  # to avoid computation
        
        for _ in range(ngen):
            netG.zero_grad()
            real = to_device(x_sampler(), netD.device_id, requires_grad = False)
            f_enc_real, _ = netD(real)
            
            z.data.normal_(0, 1)    
            fake = netG( z  )
            
            f_enc_fake, _ = netD(fake)

            # compute biased MMD2 and use ReLU to prevent negative value
            mmd2_G = mix_rbf_mmd2(f_enc_real, f_enc_fake, sigma_list)
            mmd2_G = F.relu(mmd2_G)

            # compute rank hinge loss
            one_side_errG = one_sided(f_enc_real.mean(0) - f_enc_fake.mean(0))

            errG = torch.sqrt(mmd2_G) + lambda_rg * one_side_errG
            errG.backward(one)
            optimizerG.step()
            
            g_loss_plot.plot(errG.cpu().data.numpy().mean())
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


# The following is for mmd
min_var_est = 1e-8

# Consider linear time MMD with a linear kernel:
# K(f(x), f(y)) = f(x)^Tf(y)
# h(z_i, z_j) = k(x_i, x_j) + k(y_i, y_j) - k(x_i, y_j) - k(x_j, y_i)
#             = [f(x_i) - f(y_i)]^T[f(x_j) - f(y_j)]
#
# f_of_X: batch_size * k
# f_of_Y: batch_size * k
def linear_mmd2(f_of_X, f_of_Y):
    loss = 0.0
    delta = f_of_X - f_of_Y
    loss = torch.mean((delta[:-1] * delta[1:]).sum(1))
    return loss


# Consider linear time MMD with a polynomial kernel:
# K(f(x), f(y)) = (alpha*f(x)^Tf(y) + c)^d
# f_of_X: batch_size * k
# f_of_Y: batch_size * k
def poly_mmd2(f_of_X, f_of_Y, d=2, alpha=1.0, c=2.0):
    K_XX = (alpha * (f_of_X[:-1] * f_of_X[1:]).sum(1) + c)
    K_XX_mean = torch.mean(K_XX.pow(d))

    K_YY = (alpha * (f_of_Y[:-1] * f_of_Y[1:]).sum(1) + c)
    K_YY_mean = torch.mean(K_YY.pow(d))

    K_XY = (alpha * (f_of_X[:-1] * f_of_Y[1:]).sum(1) + c)
    K_XY_mean = torch.mean(K_XY.pow(d))

    K_YX = (alpha * (f_of_Y[:-1] * f_of_X[1:]).sum(1) + c)
    K_YX_mean = torch.mean(K_YX.pow(d))

    return K_XX_mean + K_YY_mean - K_XY_mean - K_YX_mean


def _mix_rbf_kernel(X, Y, sigma_list):
    assert(X.size(0) == Y.size(0))
    m = X.size(0)

    Z = torch.cat((X, Y), 0)
    ZZT = torch.mm(Z, Z.t())
    diag_ZZT = torch.diag(ZZT).unsqueeze(1)
    Z_norm_sqr = diag_ZZT.expand_as(ZZT)
    exponent = Z_norm_sqr - 2 * ZZT + Z_norm_sqr.t()

    K = 0.0
    for sigma in sigma_list:
        gamma = 1.0 / (2 * sigma**2)
        K += torch.exp(-gamma * exponent)

    return K[:m, :m], K[:m, m:], K[m:, m:], len(sigma_list)


def mix_rbf_mmd2(X, Y, sigma_list, biased=True):
    K_XX, K_XY, K_YY, d = _mix_rbf_kernel(X, Y, sigma_list)
    # return _mmd2(K_XX, K_XY, K_YY, const_diagonal=d, biased=biased)
    return _mmd2(K_XX, K_XY, K_YY, const_diagonal=False, biased=biased)


def mix_rbf_mmd2_and_ratio(X, Y, sigma_list, biased=True):
    K_XX, K_XY, K_YY, d = _mix_rbf_kernel(X, Y, sigma_list)
    # return _mmd2_and_ratio(K_XX, K_XY, K_YY, const_diagonal=d, biased=biased)
    return _mmd2_and_ratio(K_XX, K_XY, K_YY, const_diagonal=False, biased=biased)


################################################################################
# Helper functions to compute variances based on kernel matrices
################################################################################


def _mmd2(K_XX, K_XY, K_YY, const_diagonal=False, biased=False):
    m = K_XX.size(0)    # assume X, Y are same shape

    # Get the various sums of kernels that we'll use
    # Kts drop the diagonal, but we don't need to compute them explicitly
    if const_diagonal is not False:
        diag_X = diag_Y = const_diagonal
        sum_diag_X = sum_diag_Y = m * const_diagonal
    else:
        diag_X = torch.diag(K_XX)                       # (m,)
        diag_Y = torch.diag(K_YY)                       # (m,)
        sum_diag_X = torch.sum(diag_X)
        sum_diag_Y = torch.sum(diag_Y)

    Kt_XX_sums = K_XX.sum(dim=1) - diag_X             # \tilde{K}_XX * e = K_XX * e - diag_X
    Kt_YY_sums = K_YY.sum(dim=1) - diag_Y             # \tilde{K}_YY * e = K_YY * e - diag_Y
    K_XY_sums_0 = K_XY.sum(dim=0)                     # K_{XY}^T * e

    Kt_XX_sum = Kt_XX_sums.sum()                       # e^T * \tilde{K}_XX * e
    Kt_YY_sum = Kt_YY_sums.sum()                       # e^T * \tilde{K}_YY * e
    K_XY_sum = K_XY_sums_0.sum()                       # e^T * K_{XY} * e

    if biased:
        mmd2 = ((Kt_XX_sum + sum_diag_X) / (m * m)
            + (Kt_YY_sum + sum_diag_Y) / (m * m)
            - 2.0 * K_XY_sum / (m * m))
    else:
        mmd2 = (Kt_XX_sum / (m * (m - 1))
            + Kt_YY_sum / (m * (m - 1))
            - 2.0 * K_XY_sum / (m * m))

    return mmd2


def _mmd2_and_ratio(K_XX, K_XY, K_YY, const_diagonal=False, biased=False):
    mmd2, var_est = _mmd2_and_variance(K_XX, K_XY, K_YY, const_diagonal=const_diagonal, biased=biased)
    loss = mmd2 / torch.sqrt(torch.clamp(var_est, min=min_var_est))
    return loss, mmd2, var_est


def _mmd2_and_variance(K_XX, K_XY, K_YY, const_diagonal=False, biased=False):
    m = K_XX.size(0)    # assume X, Y are same shape

    # Get the various sums of kernels that we'll use
    # Kts drop the diagonal, but we don't need to compute them explicitly
    if const_diagonal is not False:
        diag_X = diag_Y = const_diagonal
        sum_diag_X = sum_diag_Y = m * const_diagonal
        sum_diag2_X = sum_diag2_Y = m * const_diagonal**2
    else:
        diag_X = torch.diag(K_XX)                       # (m,)
        diag_Y = torch.diag(K_YY)                       # (m,)
        sum_diag_X = torch.sum(diag_X)
        sum_diag_Y = torch.sum(diag_Y)
        sum_diag2_X = diag_X.dot(diag_X)
        sum_diag2_Y = diag_Y.dot(diag_Y)

    Kt_XX_sums = K_XX.sum(dim=1) - diag_X             # \tilde{K}_XX * e = K_XX * e - diag_X
    Kt_YY_sums = K_YY.sum(dim=1) - diag_Y             # \tilde{K}_YY * e = K_YY * e - diag_Y
    K_XY_sums_0 = K_XY.sum(dim=0)                     # K_{XY}^T * e
    K_XY_sums_1 = K_XY.sum(dim=1)                     # K_{XY} * e

    Kt_XX_sum = Kt_XX_sums.sum()                       # e^T * \tilde{K}_XX * e
    Kt_YY_sum = Kt_YY_sums.sum()                       # e^T * \tilde{K}_YY * e
    K_XY_sum = K_XY_sums_0.sum()                       # e^T * K_{XY} * e

    Kt_XX_2_sum = (K_XX ** 2).sum() - sum_diag2_X      # \| \tilde{K}_XX \|_F^2
    Kt_YY_2_sum = (K_YY ** 2).sum() - sum_diag2_Y      # \| \tilde{K}_YY \|_F^2
    K_XY_2_sum  = (K_XY ** 2).sum()                    # \| K_{XY} \|_F^2

    if biased:
        mmd2 = ((Kt_XX_sum + sum_diag_X) / (m * m)
            + (Kt_YY_sum + sum_diag_Y) / (m * m)
            - 2.0 * K_XY_sum / (m * m))
    else:
        mmd2 = (Kt_XX_sum / (m * (m - 1))
            + Kt_YY_sum / (m * (m - 1))
            - 2.0 * K_XY_sum / (m * m))

    var_est = (
        2.0 / (m**2 * (m - 1.0)**2) * (2 * Kt_XX_sums.dot(Kt_XX_sums) - Kt_XX_2_sum + 2 * Kt_YY_sums.dot(Kt_YY_sums) - Kt_YY_2_sum)
        - (4.0*m - 6.0) / (m**3 * (m - 1.0)**3) * (Kt_XX_sum**2 + Kt_YY_sum**2)
        + 4.0*(m - 2.0) / (m**3 * (m - 1.0)**2) * (K_XY_sums_1.dot(K_XY_sums_1) + K_XY_sums_0.dot(K_XY_sums_0))
        - 4.0*(m - 3.0) / (m**3 * (m - 1.0)**2) * (K_XY_2_sum) - (8 * m - 12) / (m**5 * (m - 1)) * K_XY_sum**2
        + 8.0 / (m**3 * (m - 1.0)) * (
            1.0 / m * (Kt_XX_sum + Kt_YY_sum) * K_XY_sum
            - Kt_XX_sums.dot(K_XY_sums_1)
            - Kt_YY_sums.dot(K_XY_sums_0))
        )
    return mmd2, var_est